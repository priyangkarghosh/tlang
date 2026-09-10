# -------------------------------------------------------------
# @file          test_kernel_binding_state.py
# @description   GL-marked regression tests for Kernel's deferred, per-kernel SSBO binding
#                state: bind_ssbo/bind_ssbos record rather than write GL immediately, every
#                dispatch entry point re-asserts the kernel's full recorded set (skipping the
#                rebind via a shared generation counter when nothing else touched the global
#                table), dispatching with a required block never bound raises, bind() draws
#                a superset dict by name, and a freed/recycled temp buffer is caught rather
#                than silently dispatched against.
#
#                All tests share one module (see `xwire_shader`, module-scoped) declaring two
#                kernels with DIFFERENT SSBO sets so their per-artifact binding indices
#                overlap numerically -- the exact precondition for the T12 cross-wiring bug
#                (binding through kernel A's object, then dispatching kernel B, silently runs
#                B against A's buffer contents at the same GL index) this mechanism exists to
#                close.
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangBindingError

pytestmark = pytest.mark.gl


XWIRE_SRC = '''\
layout(std430) buffer BufOne { uint one_data[]; };
layout(std430) buffer BufTwo { uint two_data[]; };
layout(std430) buffer BufThree { uint three_data[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_one() {
    one_data[0] = 111u;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_two() {
    two_data[0] = 222u;
    three_data[0] = 333u;
}
'''

OPTIONAL_SRC = '''\
layout(std430) buffer KeepData { uint keep_data[]; };
layout(std430) buffer SkipData { uint skip_data[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_optional() {
    keep_data[0] = 7u;
    skip_data[0] = 9u;
}
'''


def _read_u32(buf) -> int:
    (value,) = struct.unpack('I', buf.read(4))
    return value


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader(name)
    assert shader is not None
    return shader


def test_cross_wiring_two_kernels_different_buffer_sets_dispatch_correctly(gl_ctx, make_shader_dir):
    """The T12 scenario end to end: two kernels whose required sets differ (and whose
    per-artifact binding indices therefore overlap numerically) are bound and dispatched in
    an interleaved order. Each must write the buffer it was actually given, not whatever the
    other kernel last left at the same GL binding index -- proven by real readback."""
    shader = _build(gl_ctx, make_shader_dir, 'xwire', XWIRE_SRC)
    kernel_one = shader.get_kernel('cs_one')
    kernel_two = shader.get_kernel('cs_two')

    assert set(kernel_one.bindings) == {'BufOne'}
    assert set(kernel_two.bindings) == {'BufTwo', 'BufThree'}

    buf_one = gl_ctx.buffer(reserve=4)
    buf_two = gl_ctx.buffer(reserve=4)
    buf_three = gl_ctx.buffer(reserve=4)

    kernel_one.bind_ssbo('BufOne', buf_one)
    kernel_one.dispatch(1, 1, 1)

    kernel_two.bind_ssbo('BufTwo', buf_two)
    kernel_two.bind_ssbo('BufThree', buf_three)
    kernel_two.dispatch(1, 1, 1)

    # Re-dispatch kernel_one AFTER kernel_two ran. If the generation-skip logic wrongly
    # skipped the rebind here, kernel_one would run against whatever kernel_two last bound at
    # the same numeric index instead of buf_one.
    kernel_one.dispatch(1, 1, 1)

    assert _read_u32(buf_one) == 111
    assert _read_u32(buf_two) == 222
    assert _read_u32(buf_three) == 333


def test_dispatch_with_missing_required_buffer_raises_named(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'missing', XWIRE_SRC)
    kernel = shader.get_kernel('cs_two')  # requires BufTwo + BufThree

    with pytest.raises(TlangBindingError, match='BufTwo') as exc_info:
        kernel.dispatch(1, 1, 1)
    assert 'BufThree' in str(exc_info.value)

    buf_two = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('BufTwo', buf_two)

    # Only BufThree is still missing now -- must be the only one named.
    with pytest.raises(TlangBindingError) as exc_info:
        kernel.dispatch(1, 1, 1)
    msg = str(exc_info.value)
    assert 'BufThree' in msg
    assert 'BufTwo' not in msg


def test_allow_unbound_suppresses_missing_required_check(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'optional', OPTIONAL_SRC)
    kernel = shader.get_kernel('cs_optional')

    keep_buf = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('KeepData', keep_buf)

    with pytest.raises(TlangBindingError, match='SkipData'):
        kernel.dispatch(1, 1, 1)

    # Deliberate opt-out: no exception, and the block that WAS bound is still correct.
    kernel.dispatch(1, 1, 1, allow_unbound={'SkipData'})
    assert _read_u32(keep_buf) == 7


def test_bind_draws_required_set_from_superset_and_ignores_extras(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'bindall', XWIRE_SRC)
    kernel_one = shader.get_kernel('cs_one')  # requires only BufOne

    buf_one = gl_ctx.buffer(reserve=4)
    buf_two = gl_ctx.buffer(reserve=4)  # not declared by cs_one -- must be ignored, not an error
    superset = {'BufOne': buf_one, 'BufTwo': buf_two, 'NotDeclaredAnywhere': object()}

    kernel_one.bind(**superset)
    kernel_one.dispatch(1, 1, 1)
    assert _read_u32(buf_one) == 111

    kernel_two = shader.get_kernel('cs_two')  # requires BufTwo + BufThree
    with pytest.raises(TlangBindingError, match='BufThree'):
        kernel_two.bind(BufTwo=buf_two)  # BufThree absent from the provided set


def test_generation_counter_skips_redundant_rebind_but_not_after_another_kernel_writes(gl_ctx, make_shader_dir, monkeypatch):
    import tlang.runtime.kernel as kernel_mod

    shader = _build(gl_ctx, make_shader_dir, 'gen', XWIRE_SRC)
    kernel_one = shader.get_kernel('cs_one')
    kernel_two = shader.get_kernel('cs_two')

    calls = []
    original = kernel_mod.bump_ssbo_table_generation

    def spy():
        calls.append(1)
        return original()

    monkeypatch.setattr(kernel_mod, 'bump_ssbo_table_generation', spy)

    buf_one = gl_ctx.buffer(reserve=4)
    kernel_one.bind_ssbo('BufOne', buf_one)
    kernel_one.dispatch(1, 1, 1)
    assert len(calls) == 1  # first dispatch always asserts

    kernel_one.dispatch(1, 1, 1)
    kernel_one.dispatch(1, 1, 1)
    assert len(calls) == 1  # repeat dispatches of the same, already-current kernel cost nothing

    buf_two = gl_ctx.buffer(reserve=4)
    buf_three = gl_ctx.buffer(reserve=4)
    kernel_two.bind_ssbo('BufTwo', buf_two)
    kernel_two.bind_ssbo('BufThree', buf_three)
    kernel_two.dispatch(1, 1, 1)
    assert len(calls) == 2  # another kernel's dispatch bumps the shared counter

    kernel_one.dispatch(1, 1, 1)
    assert len(calls) == 3  # kernel_one must NOT skip -- the table moved since its last assert


def test_freed_temp_handle_raises_at_dispatch_instead_of_silently_running(gl_ctx, make_shader_dir):
    from tlang.runtime.buffer_pool import BufferPool

    shader = _build(gl_ctx, make_shader_dir, 'freed', XWIRE_SRC)
    kernel = shader.get_kernel('cs_one')

    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(4)
    kernel.bind_ssbo('BufOne', handle)
    kernel.dispatch(1, 1, 1)  # fine while alive
    assert _read_u32(handle) == 111

    pool.free_temp(handle)  # buffer_pool may now hand the same GL buffer to someone else
    with pytest.raises(TlangBindingError, match='BufOne'):
        kernel.dispatch(1, 1, 1)


def test_dispatch_indirect_and_dispatch_timed_reassert(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'idt', XWIRE_SRC)
    kernel_one = shader.get_kernel('cs_one')
    kernel_two = shader.get_kernel('cs_two')

    buf_one = gl_ctx.buffer(reserve=4)
    kernel_one.bind_ssbo('BufOne', buf_one)

    indirect_buf = gl_ctx.buffer(struct.pack('III', 1, 1, 1))
    kernel_one.dispatch_indirect(indirect_buf)
    assert _read_u32(buf_one) == 111

    # kernel_two dispatches (and may reuse the same numeric binding index) in between --
    # kernel_one's next dispatch_indirect must re-assert rather than trust stale state.
    buf_two = gl_ctx.buffer(reserve=4)
    buf_three = gl_ctx.buffer(reserve=4)
    kernel_two.bind_ssbo('BufTwo', buf_two)
    kernel_two.bind_ssbo('BufThree', buf_three)
    kernel_two.dispatch(1, 1, 1)

    kernel_one.dispatch_indirect(indirect_buf)
    assert _read_u32(buf_one) == 111

    # dispatch_timed re-asserts too, and still returns a real elapsed time.
    kernel_two.dispatch(1, 1, 1)
    elapsed = kernel_one.dispatch_timed(1, 1, 1)
    assert isinstance(elapsed, float)
    assert _read_u32(buf_one) == 111


def test_missing_required_check_also_applies_to_dispatch_indirect(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'missing_indirect', XWIRE_SRC)
    kernel = shader.get_kernel('cs_two')

    indirect_buf = gl_ctx.buffer(struct.pack('III', 1, 1, 1))
    with pytest.raises(TlangBindingError, match='BufTwo'):
        kernel.dispatch_indirect(indirect_buf)
