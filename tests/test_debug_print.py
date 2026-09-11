# -------------------------------------------------------------
# @file          test_debug_print.py
# @description   Tests for the `printf(...)` debug-log built-in (v2 -- replaces v1's untyped
#                `print(...)`; see plugins/tlang/skills/tlang/references/runtime.md).
#
#                GL-free: format-literal extraction/rewriting in `tlang.compiler.printf_glsl`
#                (including the four literal-scanning edge cases), build-time specifier/argument
#                validation, GLSL overload generation, and the host-side wire format/decode in
#                `tlang.runtime.printf_log` -- all work without a GL context.
#
#                GL-marked: a real ShaderManager(debug=True) build, a real dispatch, and a real
#                readback -- exact formatted output, call-site (module, line) resolution,
#                release-mode inertness, ring-buffer overflow (ACK'd flow control) without
#                corruption, a torn-record check against a live background poller, dedup, and
#                streaming to a custom sink.
# -------------------------------------------------------------

import re
import struct
import time

import pytest

from tlang.compiler.printf_glsl import (
    render_buffer_decl, render_overloads, render_printf_module, rewrite_printf_calls,
)
from tlang.errors import TlangAttributeError
from tlang.runtime.printf_log import (
    BUFFER_HANDLE, DATA_FIELD, MAX_PRINTF_ARGS, READY_FIELD, RECORD_WORDS,
    PrintfStream, PrintfTable, decode_records_numpy, format_record, to_python_format,
)

# ---------------------------------------------------------------------------
# GL-free: call-site scanning + rewriting
# ---------------------------------------------------------------------------


def test_rewrite_finds_and_rewrites_a_real_call():
    t = PrintfTable()
    out, used = rewrite_printf_calls('void f() { printf("gid %u\\n", gid); }', 'demo', t)
    assert used
    assert 'printf(0u, uint(gid))' in out
    assert '"' not in out  # the format string never reaches the rewritten GLSL


def test_rewrite_ignores_comments_and_lookalikes():
    t = PrintfTable()
    assert rewrite_printf_calls('// printf("x\\n");\nvoid f() {}', 'demo', t) == ('// printf("x\\n");\nvoid f() {}', False)
    assert not rewrite_printf_calls('/* printf("x\\n"); */ void f() {}', 'demo', t)[1]
    assert not rewrite_printf_calls('void f() { myprintf("x\\n"); }', 'demo', t)[1]  # not a word boundary


def test_no_call_leaves_source_untouched_and_used_false():
    src = 'void f() { int printf = 1; }'
    out, used = rewrite_printf_calls(src, 'demo', PrintfTable())
    assert out == src
    assert not used


def test_bare_call_with_no_arguments():
    t = PrintfTable()
    out, used = rewrite_printf_calls('void f() { printf("hit\\n"); }', 'demo', t)
    assert used
    assert 'printf(0u)' in out
    assert t.callsite(0).specifiers == ()


# --- the four format-literal edge cases -- exactly where a naive (regex-over-raw-text)
# scanner breaks, per the brief. Each asserts the EXACT extracted format text. ---

def test_literal_edge_case_escaped_quote():
    t = PrintfTable()
    rewrite_printf_calls(r'void f() { printf("a \" b\n"); }', 'demo', t)
    assert t.callsite(0).format == 'a " b\n'


def test_literal_edge_case_comma_inside_string():
    t = PrintfTable()
    rewrite_printf_calls(r'void f() { printf("x, y: %d\n", n); }', 'demo', t)
    assert t.callsite(0).format == 'x, y: %d\n'
    assert t.callsite(0).specifiers == ('d',)


def test_literal_edge_case_close_paren_inside_string():
    t = PrintfTable()
    rewrite_printf_calls(r'void f() { printf("f(%d)\n", n); }', 'demo', t)
    assert t.callsite(0).format == 'f(%d)\n'


def test_literal_edge_case_percent_percent():
    t = PrintfTable()
    rewrite_printf_calls(r'void f() { printf("100%% done %d\n", n); }', 'demo', t)
    assert t.callsite(0).format == '100%% done %d\n'
    assert t.callsite(0).specifiers == ('d',)  # %% consumes no argument


def test_nested_call_in_argument_is_not_mistaken_for_the_outer_close_paren():
    t = PrintfTable()
    out, used = rewrite_printf_calls('void f() { printf("%d\\n", foo(a, b)); }', 'demo', t)
    assert used
    assert 'printf(0u, uint(foo(a, b)))' in out


# ---------------------------------------------------------------------------
# GL-free: build-time validation
# ---------------------------------------------------------------------------


def test_specifier_count_exceeding_argument_count_fails_the_build():
    with pytest.raises(TlangAttributeError, match=r'has 3 specifier\(s\) but 2 argument\(s\)'):
        rewrite_printf_calls('void f() { printf("a %d b %d c %d", x, y); }', 'demo', PrintfTable())


def test_argument_count_exceeding_specifier_count_fails_the_build():
    with pytest.raises(TlangAttributeError, match=r'has 1 specifier\(s\) but 2 argument\(s\)'):
        rewrite_printf_calls('void f() { printf("%d", x, y); }', 'demo', PrintfTable())


def test_too_many_arguments_fails_the_build():
    fmt = ' '.join(['%d'] * (MAX_PRINTF_ARGS + 1))
    args = ', '.join(f'a{i}' for i in range(MAX_PRINTF_ARGS + 1))
    with pytest.raises(TlangAttributeError, match='supports at most'):
        rewrite_printf_calls(f'void f() {{ printf("{fmt}", {args}); }}', 'demo', PrintfTable())


def test_unsupported_specifier_fails_the_build():
    with pytest.raises(TlangAttributeError, match="unsupported specifier '%s'"):
        rewrite_printf_calls('void f() { printf("%s", x); }', 'demo', PrintfTable())


def test_non_string_first_argument_fails_the_build():
    with pytest.raises(TlangAttributeError, match='requires a string literal'):
        rewrite_printf_calls('void f() { printf(fmt, x); }', 'demo', PrintfTable())


def test_unterminated_string_fails_the_build():
    with pytest.raises(TlangAttributeError, match='unterminated string literal'):
        rewrite_printf_calls('void f() { printf("unterminated); }', 'demo', PrintfTable())


# ---------------------------------------------------------------------------
# GL-free: call-site (module, line) resolution -- against a `#line`-annotated unit the way
# `Shader._build` actually hands text to `rewrite_printf_calls` (see `Shader.build_map`).
# ---------------------------------------------------------------------------


def test_callsite_resolves_module_and_line_from_line_directives():
    # Mirrors what Shader.build_map emits: one #line directive, then contiguous original
    # lines -- the function itself starts at original line 5.
    src = (
        '#line 5 "physics.dynamics"\n'
        'void cs_step() {\n'
        '    uint gid = gl_GlobalInvocationID.x;\n'
        '    printf("ptc %u\\n", gid);\n'
        '}\n'
    )
    t = PrintfTable()
    rewrite_printf_calls(src, 'fallback', t)
    cs = t.callsite(0)
    assert cs.module == 'physics.dynamics'
    assert cs.line == 7  # directive says line 5 is `void cs_step() {`; the printf is 2 lines later


def test_callsite_resolution_follows_a_module_change_mid_unit():
    """A [link(...)]ed helper from a DIFFERENT module is spliced into the same translation
    unit -- the call site must attribute to ITS OWN module, not the caller's."""
    src = (
        '#line 1 "caller"\n'
        'void helper_wrapper() {}\n'
        '#line 10 "helper_module"\n'
        'void helper() {\n'
        '    printf("from helper\\n");\n'
        '}\n'
    )
    t = PrintfTable()
    rewrite_printf_calls(src, 'fallback', t)
    cs = t.callsite(0)
    assert cs.module == 'helper_module'
    assert cs.line == 11


# ---------------------------------------------------------------------------
# GL-free: GLSL codegen
# ---------------------------------------------------------------------------


def test_release_used_artifact_gets_empty_bodied_overloads_no_buffer():
    text = render_printf_module(debug=False, capacity=64, used=True)
    assert BUFFER_HANDLE not in text
    assert 'void printf(' in text
    assert '{}' in text  # empty body


def test_unused_artifact_gets_nothing_at_all_in_either_mode():
    """`used=False` gates identically in both modes -- an artifact that never calls printf
    gets no overloads and no buffer, release or debug (mirrors v1's `print`)."""
    assert render_printf_module(debug=False, capacity=64, used=False) == ''
    assert render_printf_module(debug=True, capacity=64, used=False) == ''


def test_debug_used_artifact_gets_buffer_and_real_bodies():
    text = render_printf_module(debug=True, capacity=64, used=True)
    assert BUFFER_HANDLE in text
    assert 'atomicCompSwap' in text
    assert 'memoryBarrierBuffer' in text
    assert f'{DATA_FIELD}[' in text
    assert '{}' not in text  # no stub bodies mixed in


def test_buffer_decl_declares_reserved_dropped_ready_and_unsized_data_array():
    decl = render_buffer_decl(64)
    assert 'tlang_pf_reserved' in decl
    assert 'tlang_pf_dropped' in decl
    assert f'{READY_FIELD}[64]' in decl
    assert f'{DATA_FIELD}[]' in decl


def test_zero_arg_overload_is_generated():
    overloads = render_overloads(real=True, capacity=64)
    assert 'void printf(uint tlang_pf_id) {' in overloads


def test_exactly_one_all_uint_overload_per_arity_zero_through_eight():
    """No type cross-product: casting happens at the call site (see the regression tests
    below), so every overload takes plain `uint`s -- one signature per arity, period."""
    overloads = render_overloads(real=True, capacity=64)
    for arity in range(0, MAX_PRINTF_ARGS + 1):
        params = ', '.join(f'uint a{i}' for i in range(arity))
        sig = f'void printf(uint tlang_pf_id{", " + params if params else ""})'
        assert overloads.count(sig) == 1, f'expected exactly one overload for arity {arity}'
    # and no cross-product-style typed parameter (float a.., bool a.., a bare "int a..")
    # ever appears -- every value parameter is `uint`, always.
    assert re.search(r'\b(float|bool)\s+a\d', overloads) is None
    assert re.search(r'(?<!u)\bint\s+a\d', overloads) is None


def test_call_site_casts_each_argument_per_specifier_not_via_overload_type():
    """The regression this exists to prevent: an argument's uint() vs floatBitsToUint()
    conversion must come from ITS OWN specifier, applied at the call site, never from
    resolving to a type-matched overload (which cannot even exist past arity 3)."""
    t = PrintfTable()
    out, used = rewrite_printf_calls(
        'void f() { printf("%d %f %d %f\\n", 7, 2.5, 9, 4.5); }', 'demo', t,
    )
    assert used
    assert 'printf(0u, uint(7), floatBitsToUint(2.5), uint(9), floatBitsToUint(4.5))' in out


def test_call_site_casts_u_and_x_specifiers_to_uint_too():
    t = PrintfTable()
    out, _ = rewrite_printf_calls(
        'void f() { printf("%u %f %x %f\\n", a, b, c, d); }', 'demo', t,
    )
    assert 'printf(0u, uint(a), floatBitsToUint(b), uint(c), floatBitsToUint(d))' in out


# ---------------------------------------------------------------------------
# GL-free: host-side decode + formatting
# ---------------------------------------------------------------------------


def test_decode_records_numpy_roundtrip_all_specifiers():
    t = PrintfTable()
    cid = t.register_callsite('demo', 3, 'u=%u d=%d f=%f x=%x\n', ['u', 'd', 'f', 'x'])
    words = [
        cid,
        7,
        struct.unpack('<I', struct.pack('<i', -3))[0],
        struct.unpack('<I', struct.pack('<f', 2.5))[0],
        255,
    ] + [0] * (RECORD_WORDS - 5)
    raw = struct.pack(f'<{RECORD_WORDS}I', *words)
    [(got_cid, values)] = decode_records_numpy(raw, 1, t)
    assert got_cid == cid
    assert values == (7, -3, 2.5, 255)
    assert format_record(t.callsite(cid), values) == 'demo:3  u=7 d=-3 f=2.500000 x=ff'


def test_decode_keeps_int_and_float_columns_distinct_types():
    """A row mixing %d and %f specifiers must not have numpy upcast the int column to
    float (e.g. -3 silently becoming -3.0)."""
    t = PrintfTable()
    cid = t.register_callsite('demo', 1, '%d %f\n', ['d', 'f'])
    words = [cid, struct.unpack('<I', struct.pack('<i', -3))[0],
             struct.unpack('<I', struct.pack('<f', 1.5))[0]] + [0] * (RECORD_WORDS - 3)
    raw = struct.pack(f'<{RECORD_WORDS}I', *words)
    [(_cid, values)] = decode_records_numpy(raw, 1, t)
    assert values == (-3, 1.5)
    assert isinstance(values[0], int)
    assert isinstance(values[1], float)


def test_decode_multiple_call_sites_in_one_batch_preserves_order():
    t = PrintfTable()
    cid_a = t.register_callsite('demo', 1, 'a %u\n', ['u'])
    cid_b = t.register_callsite('demo', 2, 'b\n', [])
    words = []
    for cid, vals in [(cid_a, [1]), (cid_b, []), (cid_a, [2])]:
        row = [cid] + vals + [0] * (RECORD_WORDS - 1 - len(vals))
        words.extend(row)
    raw = struct.pack(f'<{3 * RECORD_WORDS}I', *words)
    records = decode_records_numpy(raw, 3, t)
    assert [cid for cid, _ in records] == [cid_a, cid_b, cid_a]
    assert [v for _, v in records] == [(1,), (), (2,)]


def test_to_python_format_translates_every_specifier():
    assert to_python_format('a %d b %u c %f d %x e %%') == 'a %d b %d c %f d %x e %%'


def test_dedup_collapses_only_consecutive_identical_lines_with_a_count():
    lines = ['a', 'a', 'a', 'b', 'a']
    assert PrintfStream._dedup(lines) == ['a  (x3)', 'b', 'a']


def test_dedup_leaves_all_distinct_lines_alone():
    lines = ['a', 'b', 'c']
    assert PrintfStream._dedup(lines) == lines


# ---------------------------------------------------------------------------
# GL-marked: real build, real dispatch, real readback
# ---------------------------------------------------------------------------

PRINTF_SRC = '''\
layout(std430) buffer Out { uint out_vals[]; };
layout(std430) buffer OutB { uint out_vals_b[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_printf_mixed() {
    uint gid = gl_GlobalInvocationID.x;
    float depth = float(gid) * 0.5;
    printf("ptc %u depth %f\\n", gid, depth);
    out_vals[gid] = gid;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_printf_single() {
    uint gid = gl_GlobalInvocationID.x;
    printf("gid %u\\n", gid);
    out_vals_b[gid] = gid;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_no_printf() {
    out_vals[gl_GlobalInvocationID.x] = 99u;
}
'''

_MIXED_LINE = 9   # the `printf("ptc %u depth %f\n", ...)` line, 1-based, in PRINTF_SRC above
_SINGLE_LINE = 17  # the `printf("gid %u\n", gid)` line


@pytest.fixture
def debug_shader(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'dbg.tlang': PRINTF_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=64)
    shader = sm.get_shader('dbg')
    assert shader is not None
    return sm, shader


@pytest.mark.gl
def test_printf_dispatch_produces_exact_formatted_string(gl_ctx, debug_shader):
    """End-to-end: a real dispatch, a real readback, asserting the EXACT formatted line --
    including the correct call-site module and line number."""
    sm, shader = debug_shader
    kernel = shader.get_kernel('cs_printf_mixed')
    out = gl_ctx.buffer(reserve=1 * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(1, 1, 1)
    gl_ctx.finish()

    lines = sm.stdout.drain()
    assert lines == [f'dbg:{_MIXED_LINE}  ptc 0 depth 0.000000']


@pytest.mark.gl
def test_printf_single_uint_arg_multiple_invocations(gl_ctx, debug_shader):
    sm, shader = debug_shader
    n = 4
    kernel = shader.get_kernel('cs_printf_single')
    out_b = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('OutB', out_b)
    kernel.dispatch(n, 1, 1)
    gl_ctx.finish()

    lines = sm.stdout.drain()
    assert sm.stdout.dropped == 0
    assert len(lines) == n
    seen = set()
    for line in lines:
        m = re.fullmatch(rf'dbg:{_SINGLE_LINE}  gid (\d+)', line)
        assert m is not None, line
        seen.add(int(m.group(1)))
    assert seen == set(range(n))


@pytest.mark.gl
def test_kernel_with_no_printf_call_gets_no_log_buffer_even_in_debug_mode(gl_ctx, debug_shader):
    _sm, shader = debug_shader
    kernel = shader.get_kernel('cs_no_printf')
    assert BUFFER_HANDLE not in kernel.bindings

    # the kernel must still run correctly -- printf support being absent is not a build failure
    out = gl_ctx.buffer(reserve=4 * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(4, 1, 1)
    gl_ctx.finish()
    assert list(struct.unpack('<4I', out.read())) == [99, 99, 99, 99]


@pytest.mark.gl
def test_release_mode_has_no_log_block_and_kernel_still_runs(gl_ctx, make_shader_dir):
    """The release-is-inert check: the SAME source, built with debug=False (the default),
    must have no TlangPrintfLog block in the kernel's bindings -- reported as the actual
    `kernel.bindings` value -- and the kernel must still dispatch and produce correct
    results. `sm.stdout` must be `None`."""
    from tlang import ShaderManager

    d = make_shader_dir({'dbg_release.tlang': PRINTF_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), keep_sources=True)
    shader = sm.get_shader('dbg_release')
    assert shader is not None
    assert sm.stdout is None

    kernel = shader.get_kernel('cs_no_printf')
    assert dict(kernel.bindings) == {'Out': 0}

    mixed_kernel = shader.get_kernel('cs_printf_mixed')
    assert BUFFER_HANDLE not in mixed_kernel.bindings
    assert BUFFER_HANDLE not in shader.get_source('cs_printf_mixed')
    # the call itself is never stripped -- only its body differs by mode
    assert 'printf(' in shader.get_source('cs_printf_mixed')

    n = 4
    out = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(n, 1, 1)
    gl_ctx.finish()
    assert list(struct.unpack(f'<{n}I', out.read())) == [99, 99, 99, 99]


@pytest.mark.gl
def test_overflow_is_reported_and_arrived_records_are_intact(gl_ctx, make_shader_dir):
    """ACK'd flow control: fill a small ring far past capacity in ONE dispatch (the host
    never drains during it, so its read-side never advances) -- every claim past `capacity`
    must be dropped and counted, and every record that DID land must decode intact (no
    corruption, no duplicate/garbage gid)."""
    from tlang import ShaderManager

    capacity = 8
    n = 100
    d = make_shader_dir({'dbg_overflow.tlang': PRINTF_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=capacity)
    shader = sm.get_shader('dbg_overflow')
    assert shader is not None

    kernel = shader.get_kernel('cs_printf_single')
    out_b = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('OutB', out_b)
    kernel.dispatch(n, 1, 1)
    gl_ctx.finish()

    lines = sm.stdout.drain()
    dropped = sm.stdout.dropped
    assert dropped == n - capacity  # EARLIEST-claimed records survive; the rest are dropped, counted
    assert len(lines) == capacity

    gids = []
    for line in lines:
        m = re.fullmatch(rf'dbg_overflow:{_SINGLE_LINE}  gid (\d+)', line)
        assert m is not None, line  # well-formed -- nothing torn or corrupted
        gids.append(int(m.group(1)))
    assert len(set(gids)) == capacity  # every surviving record is distinct
    assert all(0 <= g < n for g in gids)


@pytest.mark.gl
def test_torn_record_never_decoded_under_live_streaming(gl_ctx, make_shader_dir):
    """Many dispatches back-to-back with NO ctx.finish() between them, a background
    streaming poller running the whole time (touching only the pinned mapping -- no GL
    call). Every line the poller ever produces must be well-formed and internally
    consistent (depth == 0.5*gid) -- a torn/partially-written record would violate that
    relationship with overwhelming probability, since it would mix bytes from two
    different (gid, depth) pairs.
    """
    from tlang import ShaderManager

    d = make_shader_dir({'dbg_torn.tlang': PRINTF_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=256)
    shader = sm.get_shader('dbg_torn')
    assert shader is not None
    assert sm.stdout.capacity == 256

    kernel = shader.get_kernel('cs_printf_mixed')
    out = gl_ctx.buffer(reserve=64 * 4)
    kernel.bind_ssbo('Out', out)

    collected: list[str] = []
    sm.stdout.stream(sink=collected.append, poll_interval=0.001, rate_limit=None)
    try:
        for _ in range(40):
            kernel.dispatch(64, 1, 1)  # no ctx.finish() between dispatches
        gl_ctx.finish()
        time.sleep(0.3)  # give the poller time to drain what's left
    finally:
        sm.stdout.stop()

    assert len(collected) > 0
    pattern = re.compile(rf'dbg_torn:{_MIXED_LINE}  ptc (\d+) depth ([\d.]+)')
    for line in collected:
        m = pattern.fullmatch(line)
        assert m is not None, f'malformed/torn line: {line!r}'
        gid, depth = int(m.group(1)), float(m.group(2))
        assert depth == pytest.approx(gid * 0.5), f'torn record: {line!r}'


@pytest.mark.gl
def test_dedup_collapses_identical_records_with_repeat_count(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    src = '''\
layout(std430) buffer Out { uint out_vals[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_same() {
    printf("hit\\n");
    out_vals[gl_GlobalInvocationID.x] = 1u;
}
'''
    d = make_shader_dir({'dbg_dedup.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=64)
    shader = sm.get_shader('dbg_dedup')
    kernel = shader.get_kernel('cs_same')
    out = gl_ctx.buffer(reserve=8 * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(8, 1, 1)
    gl_ctx.finish()

    lines = sm.stdout.drain(dedup=True)
    assert lines == ['dbg_dedup:6  hit  (x8)']

    lines_raw = sm.stdout.drain(dedup=False)  # nothing left -- already consumed by the drain above
    assert lines_raw == []


@pytest.mark.gl
def test_streaming_to_a_custom_sink_and_stopping_cleanly(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'dbg_stream.tlang': PRINTF_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=64)
    shader = sm.get_shader('dbg_stream')
    kernel = shader.get_kernel('cs_printf_single')
    out_b = gl_ctx.buffer(reserve=8 * 4)
    kernel.bind_ssbo('OutB', out_b)

    collected: list[str] = []
    sm.stdout.stream(sink=collected.append, poll_interval=0.01, rate_limit=None)
    kernel.dispatch(8, 1, 1)
    gl_ctx.finish()
    time.sleep(0.2)
    sm.stdout.stop()

    assert sorted(collected) == sorted(f'dbg_stream:{_SINGLE_LINE}  gid {i}' for i in range(8))

    # stopping is clean: no thread left running, and streaming again afterward still works
    assert sm.stdout._thread is None
    collected.clear()
    kernel.dispatch(4, 1, 1)
    gl_ctx.finish()
    sm.stdout.stream(sink=collected.append, poll_interval=0.01, rate_limit=None)
    time.sleep(0.2)
    sm.stdout.stop()
    assert len(collected) == 4


@pytest.mark.gl
def test_rate_limit_suppresses_a_flood_without_crashing(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    src = '''\
layout(std430) buffer Out { uint out_vals[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_flood() {
    uint gid = gl_GlobalInvocationID.x;
    printf("gid %u\\n", gid);
    out_vals[gid] = gid;
}
'''
    d = make_shader_dir({'dbg_rate.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=256)
    shader = sm.get_shader('dbg_rate')
    kernel = shader.get_kernel('cs_flood')
    n = 200
    out = gl_ctx.buffer(reserve=n * 4)
    kernel.bind_ssbo('Out', out)

    collected: list[str] = []
    sm.stdout.stream(sink=collected.append, poll_interval=0.01, rate_limit=5.0)
    kernel.dispatch(n, 1, 1)
    gl_ctx.finish()
    time.sleep(0.5)
    sm.stdout.stop()

    # The rate limit caps the sink to a handful of calls in this window -- far fewer than
    # n, and the sink must never have been called with a raw exception/garbage.
    assert 0 < len(collected) < n
    assert all(isinstance(line, str) for line in collected)


# ---------------------------------------------------------------------------
# Regression: mixed-type calls at every arity, including past arity 3 -- this is exactly
# where overload resolution used to silently implicit-convert every argument to a common
# type (all-float, chosen because that arity's only overload was same-type) and the host
# read the resulting bit pattern back as the wrong type. See the coordinator's repro:
# `printf("%d %f %d %f\n", 7, 2.5, 9, 4.5)` used to decode as
# "1088421888 2.500000 1091567616 4.500000" (the IEEE-754 bits of 7.0 and 9.0, read as
# int) instead of "7 2.500000 9 4.500000". Fixed by casting per-specifier at the call site
# (see `printf_glsl._cast_expr`) instead of relying on a type-matched overload.
# ---------------------------------------------------------------------------

MIXED_ARITY_SRC = '''\
layout(std430) buffer Out { uint out_vals[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_mixed_arities() {
    printf("a1 %d\\n", -5);
    printf("a2 %d %f\\n", 7, 2.5);
    printf("a3 %d %f %u\\n", -1, 1.5, 9u);
    printf("a4 %d %f %d %f\\n", 7, 2.5, 9, 4.5);
    printf("a5 %d %f %d %f %u\\n", 7, 2.5, 9, 4.5, 3u);
    printf("a6 %d %f %d %f %x %d\\n", 7, 2.5, 9, 4.5, 255u, -11);
    printf("a7 %u %f %d %f %x %d %f\\n", 2u, 2.5, -9, 4.5, 16u, 11, 6.5);
    printf("a8 %d %f %u %f %x %d %f %u\\n", -7, 2.5, 9u, 4.5, 8u, -11, 6.5, 3u);
    out_vals[0] = 1u;
}
'''

EXPECTED_MIXED_ARITY_MESSAGES = [
    'a1 -5',
    'a2 7 2.500000',
    'a3 -1 1.500000 9',
    'a4 7 2.500000 9 4.500000',
    'a5 7 2.500000 9 4.500000 3',
    'a6 7 2.500000 9 4.500000 ff -11',
    'a7 2 2.500000 -9 4.500000 10 11 6.500000',
    'a8 -7 2.500000 9 4.500000 8 -11 6.500000 3',
]


@pytest.mark.gl
def test_mixed_type_calls_produce_correct_values_at_every_arity(gl_ctx, make_shader_dir):
    """One invocation, one call per arity 1..8, each mixing %d/%u/%x with %f -- every
    printed value must be exactly right, including negative ints (sign preserved through
    the int->uint round trip) at the highest arities."""
    from tlang import ShaderManager

    d = make_shader_dir({'dbg_mixed.tlang': MIXED_ARITY_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), debug=True, debug_log_capacity=64)
    shader = sm.get_shader('dbg_mixed')
    assert shader is not None

    kernel = shader.get_kernel('cs_mixed_arities')
    out = gl_ctx.buffer(reserve=1 * 4)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(1, 1, 1)
    gl_ctx.finish()

    lines = sm.stdout.drain()
    assert sm.stdout.dropped == 0
    # One invocation issuing these sequentially claims strictly increasing ring positions,
    # so they come back in program order.
    messages = [line.split('  ', 1)[1] for line in lines]
    assert messages == EXPECTED_MIXED_ARITY_MESSAGES
