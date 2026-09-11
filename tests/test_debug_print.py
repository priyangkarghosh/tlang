# -------------------------------------------------------------
# @file          test_debug_print.py
# @description   Tests for the `print(...)` debug-log built-in.
#
#                GL-free: the GLSL codegen in `tlang.compiler.debug_print` and the host-side
#                decode in `tlang.runtime.debug_log` -- both work without a GL context.
#
#                GL-marked: a real ShaderManager(debug=True) build, a real dispatch, and a real
#                readback -- exact values (including a mixed uint/float call), release-mode
#                inertness, overflow reporting without corruption, clearing between dispatches,
#                and a print-free kernel getting no log buffer even in a debug build.
# -------------------------------------------------------------

import struct

import pytest

from tlang.compiler.debug_print import (
    contains_print_call, encode_header, render_buffer_decl, render_overloads, render_print_module,
)
from tlang.errors import TlangError
from tlang.runtime.debug_log import DATA_FIELD, DEBUG_BUFFER_HANDLE, decode_record

# ---------------------------------------------------------------------------
# GL-free: codegen + decode
# ---------------------------------------------------------------------------


def test_contains_print_call_finds_real_calls():
    assert contains_print_call('void f() { print(gid); }')
    assert contains_print_call('void f() { print(gid, depth, correction.x); }')


def test_contains_print_call_ignores_comments_and_lookalikes():
    assert not contains_print_call('// print(gid);\nvoid f() {}')
    assert not contains_print_call('/* print(gid); */ void f() {}')
    assert not contains_print_call('void f() { myprint(gid); }')  # not a word-boundary match
    assert not contains_print_call('void f() { int print = 1; }')  # no call, no trailing '('


def test_release_used_artifact_gets_empty_bodied_overloads_no_buffer():
    text = render_print_module(debug=False, capacity=64, used=True)
    assert DEBUG_BUFFER_HANDLE not in text
    assert 'void print(' in text
    assert '{}' in text  # empty body


def test_unused_artifact_gets_nothing_at_all_in_either_mode():
    """`used=False` gates identically in both modes -- an artifact that never calls print
    gets no overloads and no buffer, release or debug (see the perf note in
    render_print_module's docstring for why release isn't unconditional)."""
    assert render_print_module(debug=False, capacity=64, used=False) == ''
    assert render_print_module(debug=True, capacity=64, used=False) == ''


def test_debug_used_artifact_gets_buffer_and_real_bodies():
    text = render_print_module(debug=True, capacity=64, used=True)
    assert DEBUG_BUFFER_HANDLE in text
    assert 'atomicAdd' in text
    assert f'{DATA_FIELD}[' in text
    assert '{}' not in text  # no stub bodies mixed in


def test_buffer_decl_declares_cursor_overflow_and_unsized_data_array():
    decl = render_buffer_decl()
    assert 'tlang_debug_cursor' in decl
    assert 'tlang_debug_overflow' in decl
    assert f'{DATA_FIELD}[]' in decl


def test_mixed_uint_float_two_arg_overload_is_generated():
    """print(gid, depth) with a uint and a float must resolve to an EXACT overload, not an
    implicit-conversion one -- so the exact-type signature must be present in the emitted text."""
    overloads = render_overloads(real=True, capacity=64)
    assert 'void print(uint a0, float a1)' in overloads
    assert 'void print(float a0, uint a1)' in overloads  # both orders


def test_three_arg_mixed_overload_is_generated():
    """Matches the brief's own example: print(gid, depth, correction.x) -- uint, float, float."""
    overloads = render_overloads(real=True, capacity=64)
    assert 'void print(uint a0, float a1, float a2)' in overloads


def test_arity_four_is_same_type_only_not_full_cross_product():
    """Documented policy: arities 4-8 are same-type-only, not the full 4**n cross product."""
    overloads = render_overloads(real=True, capacity=64)
    assert 'void print(uint a0, uint a1, uint a2, uint a3)' in overloads
    assert 'void print(uint a0, float a1, uint a2, uint a3)' not in overloads


def test_encode_decode_header_roundtrip():
    header = encode_header(('uint', 'float', 'bool'))
    words = (header, 7, struct.unpack('<I', struct.pack('<f', 2.5))[0], 1)
    values = decode_record(words)
    assert values == (7, 2.5, True)


def test_decode_record_int_is_signed():
    header = encode_header(('int',))
    neg = struct.unpack('<I', struct.pack('<i', -3))[0]
    assert decode_record((header, neg)) == (-3,)


# ---------------------------------------------------------------------------
# GL-marked: real build, real dispatch, real readback
# ---------------------------------------------------------------------------

PRINT_SRC = '''\
layout(std430) buffer Out { uint out_vals[]; };
layout(std430) buffer OutB { uint out_vals_b[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_print_mixed() {
    uint gid = gl_GlobalInvocationID.x;
    float depth = float(gid) * 0.5;
    print(gid, depth);
    out_vals[gid] = gid;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_print_single() {
    uint gid = gl_GlobalInvocationID.x;
    print(gid);
    out_vals_b[gid] = gid;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_no_print() {
    out_vals[gl_GlobalInvocationID.x] = 99u;
}
'''


@pytest.fixture
def debug_shader(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'dbg.tlang': PRINT_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=64)
    shader = sm.get_shader('dbg')
    assert shader is not None
    return shader


@pytest.mark.gl
def test_print_dispatch_reads_back_exact_mixed_values(gl_ctx, debug_shader):
    n = 4
    kernel = debug_shader.get_kernel('cs_print_mixed')
    out = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('Out', out)
    kernel.debug_log()  # must not raise: this artifact declares the log buffer
    kernel.clear_debug_log()
    kernel.dispatch(n, 1, 1)

    records, overflow = kernel.debug_log()
    assert overflow == 0
    assert len(records) == n

    by_gid = {rec[0]: rec[1] for rec in records}
    assert set(by_gid) == set(range(n))
    for gid, depth in by_gid.items():
        assert isinstance(gid, int)
        assert isinstance(depth, float)
        assert depth == pytest.approx(gid * 0.5)

    out_vals = struct.unpack(f'<{n}I', out.read())
    assert list(out_vals) == list(range(n))


@pytest.mark.gl
def test_print_single_uint_arg(gl_ctx, debug_shader):
    n = 3
    kernel = debug_shader.get_kernel('cs_print_single')
    out_b = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('OutB', out_b)
    kernel.clear_debug_log()
    kernel.dispatch(n, 1, 1)

    records, overflow = kernel.debug_log()
    assert overflow == 0
    assert {rec[0] for rec in records} == set(range(n))
    for rec in records:
        assert len(rec) == 1
        assert isinstance(rec[0], int)


@pytest.mark.gl
def test_clear_debug_log_resets_between_dispatches(gl_ctx, debug_shader):
    kernel = debug_shader.get_kernel('cs_print_single')
    out_b = gl_ctx.buffer(reserve=4 * 4)
    kernel.bind_ssbo('OutB', out_b)

    kernel.clear_debug_log()
    kernel.dispatch(2, 1, 1)
    first_records, first_overflow = kernel.debug_log()
    assert first_overflow == 0
    assert len(first_records) == 2

    kernel.clear_debug_log()
    records_after_clear, overflow_after_clear = kernel.debug_log()
    assert records_after_clear == []
    assert overflow_after_clear == 0

    kernel.dispatch(2, 1, 1)
    second_records, second_overflow = kernel.debug_log()
    assert second_overflow == 0
    # NOT accumulated with the first dispatch's records -- exactly this dispatch's own.
    assert len(second_records) == 2


@pytest.mark.gl
def test_kernel_with_no_print_call_gets_no_log_buffer_even_in_debug_mode(gl_ctx, debug_shader):
    kernel = debug_shader.get_kernel('cs_no_print')
    assert DEBUG_BUFFER_HANDLE not in kernel.bindings

    with pytest.raises(TlangError):
        kernel.debug_log()
    with pytest.raises(TlangError):
        kernel.clear_debug_log()

    # the kernel must still run correctly -- print support being absent is not a build failure
    out = gl_ctx.buffer(reserve=4 * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(4, 1, 1)
    assert list(struct.unpack('<4I', out.read())) == [99, 99, 99, 99]


@pytest.mark.gl
def test_overflow_is_reported_and_earlier_records_are_not_corrupted(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    capacity = 4
    n = 10
    d = make_shader_dir({'dbg_overflow.tlang': PRINT_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=capacity)
    shader = sm.get_shader('dbg_overflow')
    assert shader is not None

    kernel = shader.get_kernel('cs_print_single')
    out_b = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('OutB', out_b)
    kernel.clear_debug_log()
    kernel.dispatch(n, 1, 1)

    records, overflow = kernel.debug_log()
    assert overflow == n - capacity
    assert len(records) == capacity
    # every surviving record decodes to a valid, in-range, single-uint call -- nothing torn or
    # corrupted by the dropped writes past capacity.
    seen_gids = set()
    for rec in records:
        assert len(rec) == 1
        gid = rec[0]
        assert isinstance(gid, int)
        assert 0 <= gid < n
        assert gid not in seen_gids  # each surviving gid is distinct
        seen_gids.add(gid)


@pytest.mark.gl
def test_release_mode_has_no_log_block_and_kernel_still_runs(gl_ctx, make_shader_dir):
    """The release-is-inert check: the SAME source, built with debug=False (the default),
    must have no TlangDebugLog block in the kernel's bindings, and the kernel must still
    dispatch and produce correct results -- print(...) compiles away, it doesn't break
    anything."""
    from tlang import ShaderManager

    d = make_shader_dir({'dbg_release.tlang': PRINT_SRC})
    # debug defaults to False; keep_sources=True only so this test can inspect the generated
    # GLSL directly instead of relying on `kernel.bindings` alone for the inertness check.
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), keep_sources=True)
    shader = sm.get_shader('dbg_release')
    assert shader is not None

    kernel = shader.get_kernel('cs_print_mixed')
    assert DEBUG_BUFFER_HANDLE not in kernel.bindings
    assert DEBUG_BUFFER_HANDLE not in shader.get_source('cs_print_mixed')
    with pytest.raises(TlangError):
        kernel.debug_log()

    n = 4
    out = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(n, 1, 1)
    assert list(struct.unpack(f'<{n}I', out.read())) == list(range(n))
