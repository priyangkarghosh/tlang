# -------------------------------------------------------------
# @file          test_buffer_source.py
# @description   GL-marked regression tests for Kernel/Pipeline/Shader/ShaderManager's buffer
#                source: `bind()` with no arguments resolving required blocks from a settable
#                `Mapping[str, Buffer]` (a `BufferPool` or a plain dict), explicit kwargs
#                overriding the source per call, a missing block raising with a did-you-mean,
#                a tagged temporary freed and reallocated between frames still resolving
#                correctly through the source, source propagation from `Shader`/`ShaderManager`
#                down to their kernels, and the declared-block-names property.
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangBindingError
from tlang.runtime.buffer_pool import BufferPool

pytestmark = pytest.mark.gl


SRC = '''\
layout(std430) buffer BufOne { uint one_data[]; };
layout(std430) buffer BufTwo { uint two_data[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_needs_one() {
    one_data[0] = 111u;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_needs_both() {
    one_data[0] = 111u;
    two_data[0] = 222u;
}
'''


def _read_u32(buf) -> int:
    (value,) = struct.unpack('I', buf.read(4))
    return value


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    return sm, sm.get_shader(name)


def test_bind_no_args_resolves_from_pool_source(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_pool', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    pool = BufferPool(gl_ctx)
    buf = pool.persistent_buffer('BufOne', size=4)

    kernel.buffer_source = pool
    kernel.bind()
    kernel.dispatch(1, 1, 1)
    assert _read_u32(buf) == 111


def test_bind_no_args_resolves_from_plain_dict(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_dict', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    buf = gl_ctx.buffer(reserve=4)
    kernel.buffer_source = {'BufOne': buf}
    kernel.bind()
    kernel.dispatch(1, 1, 1)
    assert _read_u32(buf) == 111


def test_explicit_kwargs_override_source_for_that_call(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_override', SRC)
    kernel = shader.get_kernel('cs_needs_both')

    from_source = gl_ctx.buffer(reserve=4)
    override = gl_ctx.buffer(reserve=4)
    kernel.buffer_source = {'BufOne': from_source, 'BufTwo': from_source}

    # BufTwo explicit -> override; BufOne still comes from the source.
    kernel.bind(BufTwo=override)
    kernel.dispatch(1, 1, 1)
    assert _read_u32(from_source) == 111
    assert _read_u32(override) == 222


def test_missing_block_raises_with_did_you_mean(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_typo', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    kernel.buffer_source = {'BufOnee': gl_ctx.buffer(reserve=4)}  # typo'd key
    with pytest.raises(TlangBindingError) as exc_info:
        kernel.bind()
    msg = str(exc_info.value)
    assert 'BufOne' in msg
    assert 'source' in msg.lower()
    assert 'BufOnee' in msg


def test_bind_with_no_source_and_no_kwargs_raises_original_message(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_none', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    with pytest.raises(TlangBindingError, match='BufOne'):
        kernel.bind()


def test_tagged_temp_freed_and_reallocated_between_frames_resolves_via_source(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_frames', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    pool = BufferPool(gl_ctx)
    kernel.buffer_source = pool

    # "frame" 1
    handle1 = pool.alloc_temp(4, tag='BufOne')
    kernel.bind()
    kernel.dispatch(1, 1, 1)
    assert _read_u32(handle1) == 111
    pool.free_temp(handle1)

    # "frame" 2 -- a brand new handle takes the same tag; bind() must resolve the NEW one.
    handle2 = pool.alloc_temp(4, tag='BufOne')
    kernel.bind()
    kernel.dispatch(1, 1, 1)
    assert _read_u32(handle2) == 111
    pool.free_temp(handle2)


def test_setting_source_on_shader_reaches_its_kernels(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_shader', SRC)
    kernel_one = shader.get_kernel('cs_needs_one')
    kernel_both = shader.get_kernel('cs_needs_both')

    pool = BufferPool(gl_ctx)
    buf_one = pool.persistent_buffer('BufOne', size=4)
    buf_two = pool.persistent_buffer('BufTwo', size=4)

    shader.buffer_source = pool
    assert kernel_one.buffer_source is pool
    assert kernel_both.buffer_source is pool

    kernel_one.bind()
    kernel_one.dispatch(1, 1, 1)
    kernel_both.bind()
    kernel_both.dispatch(1, 1, 1)
    assert _read_u32(buf_one) == 111
    assert _read_u32(buf_two) == 222


def test_setting_source_on_shader_manager_reaches_whole_tree(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_manager', SRC)
    kernel = shader.get_kernel('cs_needs_one')

    pool = BufferPool(gl_ctx)
    buf = pool.persistent_buffer('BufOne', size=4)

    sm.buffer_source = pool
    assert shader.buffer_source is pool
    assert kernel.buffer_source is pool

    kernel.bind()
    kernel.dispatch(1, 1, 1)
    assert _read_u32(buf) == 111


def test_declared_blocks_exposes_every_block_name(gl_ctx, make_shader_dir):
    sm, shader = _build(gl_ctx, make_shader_dir, 'src_declared', SRC)
    assert shader.declared_blocks == {'BufOne', 'BufTwo'}
    assert sm.declared_blocks == {'BufOne', 'BufTwo'}
