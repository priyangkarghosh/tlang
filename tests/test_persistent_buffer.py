# -------------------------------------------------------------
# @file          test_persistent_buffer.py
# @description   GL-marked regression tests for persistent-mapped ("pinned") buffers
#                (`tlang.runtime.pinned_buffer.PinnedBuffer`) and `BufferPool.alloc_pinned`:
#                real mapping + zero-copy access, CPU->GPU write visibility (real dispatch),
#                GPU->CPU write visibility after the documented fence (real dispatch +
#                readback, asserting the actual value), binding by name through `kernel.bind()`
#                exactly like a `moderngl.Buffer`, that construction never touches GL's
#                (process-global) indexed SSBO binding table, release/use-after-release, tagged
#                allocation resolving through the pool's Mapping, and the
#                GL_ARB_buffer_storage-absent fallback.
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangError
from tlang.runtime import pinned_buffer as pinned_buffer_module
from tlang.runtime.buffer_pool import BufferPool
from tlang.runtime.pinned_buffer import PinnedBuffer, PinnedBufferFallback, create_pinned_buffer

pytestmark = pytest.mark.gl


def _u32(value: int) -> bytes:
    return struct.pack('I', value)


def _read_u32(data: bytes) -> int:
    (value,) = struct.unpack('I', data)
    return value


# One block CPU code writes into ('Data'), one block a compute shader writes into ('Result'
# for the CPU->GPU direction) or writes back into ('Data' itself, for the GPU->CPU direction).
SRC = '''\
layout(std430) buffer Data { uint data[]; };
layout(std430) buffer Result { uint result[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_double_into_result() {
    result[0] = data[0] * 2u;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_gpu_writes_data() {
    data[0] = 777u;
}
'''


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    return sm, sm.get_shader(name)


def _require_real_pinned(gl_ctx):
    if not pinned_buffer_module.buffer_storage_supported(gl_ctx):
        pytest.skip("GL_ARB_buffer_storage not available on this context")


# ------------------------------------------------------------------
# Allocation, mapping, zero-copy access
# ------------------------------------------------------------------

def test_alloc_pinned_returns_a_real_mapping_when_supported(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(64)
    try:
        assert buf.is_pinned is True
        assert isinstance(buf, PinnedBuffer)
        assert buf.size == 64
        assert isinstance(buf.glo, int) and buf.glo > 0
    finally:
        pool.free_pinned(buf)


def test_mapping_is_zero_copy_in_both_directions(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(16)
    try:
        # write() through the mapping is immediately visible via the raw memoryview
        buf.write(b'ABCDEFGHIJKLMNOP', sync=False)
        assert bytes(buf.mapping) == b'ABCDEFGHIJKLMNOP'

        # writing directly through the raw memoryview is immediately visible via read()
        buf.mapping[0:4] = b'ZZZZ'
        assert buf.read(sync=False)[0:4] == b'ZZZZ'
    finally:
        pool.free_pinned(buf)


def test_construction_never_touches_the_indexed_ssbo_binding_table(gl_ctx):
    """Regression for a prior hand-rolled attempt that called
    `glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id)` in its constructor, hardcoding binding 0
    and stomping whatever tlang had already bound there. `tlang` allocates SSBO bindings itself
    (`Kernel.bind_ssbo`); a pinned buffer must never bind itself anywhere on construction.
    """
    from OpenGL.GL import glBindBufferBase, glGetIntegeri_v, GL_SHADER_STORAGE_BUFFER, GL_SHADER_STORAGE_BUFFER_BINDING

    _require_real_pinned(gl_ctx)
    sentinel = gl_ctx.buffer(reserve=16)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sentinel.glo)
    before = glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0)

    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(64)
    try:
        after = glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0)
        assert after == before, "constructing a PinnedBuffer must not touch binding index 0"
    finally:
        pool.free_pinned(buf)
        sentinel.release()


# ------------------------------------------------------------------
# Real dispatch: both directions of visibility
# ------------------------------------------------------------------

def test_cpu_write_through_mapping_visible_to_compute_shader(gl_ctx, make_shader_dir):
    _require_real_pinned(gl_ctx)
    sm, shader = _build(gl_ctx, make_shader_dir, 'pinned_cpu_write', SRC)
    kernel = shader.get_kernel('cs_double_into_result')

    pool = BufferPool(gl_ctx)
    data = pool.alloc_pinned(4, tag='Data')
    result = pool.persistent_buffer('Result', size=4)
    try:
        # CPU -> GPU: coherent mapping needs no extra sync for THIS direction (see
        # PinnedBuffer's fencing-contract docstring) -- write() with sync=False is enough
        # since no GPU work has touched this buffer yet.
        data.write(_u32(21), sync=False)

        kernel.buffer_source = pool
        kernel.bind()
        kernel.dispatch(1, 1, 1)

        assert _read_u32(result.read()) == 42
    finally:
        pool.free_pinned('Data')


def test_gpu_write_visible_through_mapping_after_documented_sync(gl_ctx, make_shader_dir):
    _require_real_pinned(gl_ctx)
    sm, shader = _build(gl_ctx, make_shader_dir, 'pinned_gpu_write', SRC)
    kernel = shader.get_kernel('cs_gpu_writes_data')

    pool = BufferPool(gl_ctx)
    data = pool.alloc_pinned(4, tag='Data')
    pool.persistent_buffer('Result', size=4)  # declared by the module; unused by this entry point
    try:
        kernel.buffer_source = pool
        kernel.bind()
        kernel.dispatch(1, 1, 1)

        # GPU -> CPU: this is the hazard fence()/wait() exist for. fence() right after the
        # dispatch, then read() with the (default) sync=True blocks until that fence signals.
        data.fence()
        assert _read_u32(data.read()) == 777
    finally:
        pool.free_pinned('Data')


def test_read_sync_true_is_the_default_and_waits_on_a_pending_fence(gl_ctx, make_shader_dir):
    """A plain `.read()` (no explicit `sync=`) must not hand back torn/stale data -- it should
    behave identically to an explicit `sync=True`."""
    _require_real_pinned(gl_ctx)
    sm, shader = _build(gl_ctx, make_shader_dir, 'pinned_default_sync', SRC)
    kernel = shader.get_kernel('cs_gpu_writes_data')

    pool = BufferPool(gl_ctx)
    data = pool.alloc_pinned(4, tag='Data')
    pool.persistent_buffer('Result', size=4)
    try:
        kernel.buffer_source = pool
        kernel.bind()
        kernel.dispatch(1, 1, 1)

        data.fence()
        assert _read_u32(data.read()) == 777  # default sync=True
    finally:
        pool.free_pinned('Data')


# ------------------------------------------------------------------
# kernel.bind() by name, exactly like a moderngl.Buffer
# ------------------------------------------------------------------

def test_binds_by_name_through_kernel_bind_like_a_moderngl_buffer(gl_ctx, make_shader_dir):
    _require_real_pinned(gl_ctx)
    sm, shader = _build(gl_ctx, make_shader_dir, 'pinned_kernel_bind', SRC)
    kernel_write = shader.get_kernel('cs_gpu_writes_data')
    kernel_double = shader.get_kernel('cs_double_into_result')

    pool = BufferPool(gl_ctx)
    data = pool.alloc_pinned(4, tag='Data')
    result = pool.persistent_buffer('Result', size=4)
    try:
        assert 'Data' in pool
        assert pool['Data'] is data

        kernel_write.bind(**pool)
        kernel_write.dispatch(1, 1, 1)
        data.fence()
        assert _read_u32(data.read()) == 777

        kernel_double.bind(**pool)
        kernel_double.dispatch(1, 1, 1)
        assert _read_u32(result.read()) == 1554
    finally:
        pool.free_pinned('Data')


# ------------------------------------------------------------------
# Release / use-after-release
# ------------------------------------------------------------------

def test_release_frees_and_use_after_release_is_caught(gl_ctx):
    _require_real_pinned(gl_ctx)
    buf = create_pinned_buffer(gl_ctx, 32)
    buf.release()

    with pytest.raises(TlangError):
        buf.read()
    with pytest.raises(TlangError):
        buf.write(b'x' * 4)
    with pytest.raises(TlangError):
        _ = buf.mapping
    with pytest.raises(TlangError):
        _ = buf.glo

    buf.release()  # idempotent, like moderngl.Buffer.release() -- must not raise


def test_pool_free_pinned_releases_and_untags(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(16, tag='Scratch')
    assert 'Scratch' in pool

    pool.free_pinned('Scratch')
    assert 'Scratch' not in pool
    with pytest.raises(TlangError):
        buf.read()

    with pytest.raises(TlangError):
        pool.free_pinned('Scratch')  # already gone


def test_pool_clear_releases_pinned_buffers(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    tagged = pool.alloc_pinned(16, tag='Tagged')
    untagged = pool.alloc_pinned(16)

    pool.clear()

    with pytest.raises(TlangError):
        tagged.read()
    with pytest.raises(TlangError):
        untagged.read()


# ------------------------------------------------------------------
# Tagged allocation resolves through the pool's Mapping
# ------------------------------------------------------------------

def test_tagged_pinned_allocation_resolves_through_pool_mapping(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(64, tag='Widgets')
    try:
        assert pool['Widgets'] is buf
        assert 'Widgets' in pool
        assert 'Widgets' in list(pool)
    finally:
        pool.free_pinned('Widgets')


def test_pinned_tag_collides_with_persistent_buffer_name(gl_ctx):
    pool = BufferPool(gl_ctx)
    pool.persistent_buffer('Shared', size=16)
    with pytest.raises(TlangError):
        pool.alloc_pinned(16, tag='Shared')


def test_persistent_buffer_collides_with_pinned_tag(gl_ctx):
    _require_real_pinned(gl_ctx)
    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(16, tag='Shared')
    try:
        with pytest.raises(TlangError):
            pool.persistent_buffer('Shared', size=16)
    finally:
        pool.free_pinned('Shared')


# ------------------------------------------------------------------
# Fallback when GL_ARB_buffer_storage is unavailable
# ------------------------------------------------------------------

def test_fallback_used_when_extension_unavailable(gl_ctx, monkeypatch, caplog):
    monkeypatch.setattr(pinned_buffer_module, 'buffer_storage_supported', lambda ctx: False)

    with caplog.at_level('WARNING'):
        buf = create_pinned_buffer(gl_ctx, 64)
    try:
        assert isinstance(buf, PinnedBufferFallback)
        assert buf.is_pinned is False
        assert any('fall' in rec.message.lower() for rec in caplog.records)

        # read()/write() still work correctly, just not through a real mapping
        buf.write(_u32(9))
        assert _read_u32(buf.read(4)) == 9
        with pytest.raises(TlangError):
            _ = buf.mapping
    finally:
        buf.release()


def test_pool_alloc_pinned_falls_back_and_still_resolves_by_tag(gl_ctx, monkeypatch):
    monkeypatch.setattr(pinned_buffer_module, 'buffer_storage_supported', lambda ctx: False)

    pool = BufferPool(gl_ctx)
    buf = pool.alloc_pinned(16, tag='Fallback')
    try:
        assert buf.is_pinned is False
        assert pool['Fallback'] is buf
        buf.write(_u32(5))
        assert _read_u32(buf.read(4)) == 5
    finally:
        pool.free_pinned('Fallback')
