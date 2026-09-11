# -------------------------------------------------------------
# @file          test_precompile_extern.py
# @description   Tests for `[extern(precompile=[...])]`: a variant-axis form
#                of the host-supplied constant declared by `[extern]` (see
#                test_extern_constants.py). Unlike a plain `[extern]`, it has
#                NO default/plain artifact at all -- it never consults
#                `constants={...}`, and there is no uniform fallback. Instead
#                `ShaderManager` builds one fully independent, `const`-
#                specialised `Shader` per listed value (and, for a module
#                declaring more than one axis, the full cross product of
#                every axis's values -- there is no partially-resolved
#                module text to fall back to) -- all of this happens EAGERLY
#                at ordinary build time; there is no on-demand/lazy compile
#                step and nothing extra is retained afterward.
#                `Shader.get_kernel(name, X=value)` returns one of those
#                precompiled variants; selecting a value for EVERY axis a
#                module declares is mandatory for every kernel in it, even
#                one that doesn't reference that particular constant --
#                `get_kernel(name)` alone, or a value outside the declared
#                list, is a build-time-shaped error naming the constant, the
#                module, and the permitted values, never a silent fallback.
#
#                GL-free except the `@pytest.mark.gl` tests, which are the
#                ones that actually matter for this feature: each precompiled
#                variant must produce the value its own source literally
#                says via a real dispatch with readback (a variant that
#                computes something different is the failure mode this
#                guards against, more important than raw speed); and every
#                variant must keep the same buffers/bindings (verified:
#                `kernel.bindings` equal across every precompiled value).
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangAttributeError
from tlang.compiler.shader_processor import ShaderProcessor


def _processor(name: str, src: str, strict: bool = True) -> ShaderProcessor:
    return ShaderProcessor(name, src, strict=strict)


def _text(proc: ShaderProcessor) -> str:
    return ''.join(line.data for line in proc.module.values())


# ---------------------------------------------------------------------------
# GL-free: parsing and resolution mechanics
# ---------------------------------------------------------------------------

def test_precompile_extern_emits_nothing_by_default():
    """No `uniform`, no `const` -- a precompile-axis extern has no default
    artifact at all; `resolve_externs` leaves its placeholder line blank."""
    proc = _processor('demo', "[extern(precompile=[1, 2, 4])] int mode;\n")
    proc.resolve_externs({})
    text = _text(proc)
    assert 'uniform' not in text
    assert 'mode' not in text


def test_precompile_extern_ignores_the_constants_dict_entirely():
    """`constants={...}` resolves a plain [extern]; it must never resolve a
    precompile-axis one -- there's nothing for it to resolve to here."""
    proc = _processor('demo', "[extern(precompile=[1, 2, 4])] int mode;\n")
    proc.resolve_externs({'mode': 99})
    assert '99' not in _text(proc)


def test_externs_reflects_the_precompile_list_before_any_variant_is_built():
    proc = _processor('demo', "[extern(precompile=[1, 2, 4])] int mode;\n")
    decl = proc.externs['mode']
    assert decl.precompile == (1, 2, 4)
    assert decl.resolved is False
    assert decl.value is None


def test_extern_without_precompile_has_an_empty_precompile_list():
    proc = _processor('demo', "[extern] int BLOCK_SIZE;\n")
    assert proc.externs['BLOCK_SIZE'].precompile == ()


def test_precompile_with_a_default_is_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(precompile=[1, 2])] int mode = 3;\n")
    msg = str(exc_info.value)
    assert 'mode' in msg
    assert 'default' in msg


def test_unknown_extern_positional_argument_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(bogus)] int X;\n")
    assert 'bogus' in str(exc_info.value)


def test_unknown_extern_keyword_argument_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(oops=1)] int X;\n")
    assert 'oops' in str(exc_info.value)


def test_malformed_precompile_list_names_the_constant():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(precompile=[1, x, 3])] int X;\n")
    assert 'X' in str(exc_info.value)


def test_empty_precompile_list_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(precompile=[])] int X;\n")
    assert 'X' in str(exc_info.value)


def test_wrong_type_in_precompile_list_names_the_constant_and_type():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern(precompile=[1, 2])] bool flag;\n")
    msg = str(exc_info.value)
    assert 'flag' in msg
    assert 'bool' in msg


# ---------------------------------------------------------------------------
# GL: the mechanism that actually matters -- real dispatch, real readback,
# and the "selection is mandatory, no fallback" contract
# ---------------------------------------------------------------------------

# A single axis -- the common case, and the one every identity/binding/error
# test below exercises. `stabilizing` picks which buffer's element 0 lands
# in `Out`; each precompiled variant must produce exactly what its own
# baked-in value says, which is the real correctness question for this
# feature (tlang's DCE keeps both arms of the select in every variant, since
# it's textual, not constant-folding).
ONE_AXIS_SRC = """\
layout(std430) buffer Positions { vec4 ptcPositions[]; };
layout(std430) buffer Predicted { vec4 ptcPredictedPositions[]; };
layout(std430) buffer Out { vec4 outPositions[]; };

[extern(precompile=[false, true])] bool stabilizing;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_solve() {
    outPositions[0] = stabilizing ? ptcPositions[0] : ptcPredictedPositions[0];
}
"""

# A second, independent axis with a restricted numeric set -- used only for
# the "value outside the list"/"undeclared name" errors, where a bool's
# full value space (precompile=[false, true]) can't demonstrate "outside
# the list" at all.
MODE_AXIS_SRC = """\
layout(std430) buffer Out { uint data[]; };

[extern(precompile=[1, 2, 4])] int mode;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_mode() {
    data[0] = uint(mode);
}
"""

# Two axes in one module -- proves the cross-product requirement: a module
# declaring more than one precompile axis has no way to resolve just one of
# them (there's no fallback for the other), so EVERY combination of every
# axis's values must be selected together.
TWO_AXIS_SRC = """\
layout(std430) buffer Out { uint data[]; };

[extern(precompile=[1, 2])] int mode;
[extern(precompile=[false, true])] bool flag;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_two() {
    data[0] = flag ? uint(mode) : 0u;
}
"""


def _read_vec4(buf) -> tuple:
    return struct.unpack('4f', buf.read(16))


def _read_u32(buf) -> int:
    (value,) = struct.unpack('I', buf.read(4))
    return value


@pytest.fixture(scope='module')
def one_axis_shader(gl_ctx, tmp_path_factory):
    from tlang import ShaderManager

    d = tmp_path_factory.mktemp('one_axis')
    (d / 'demo.tlang').write_text(ONE_AXIS_SRC, encoding='utf-8')
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('demo')
    assert shader is not None
    return shader


@pytest.fixture(scope='module')
def mode_axis_shader(gl_ctx, tmp_path_factory):
    from tlang import ShaderManager

    d = tmp_path_factory.mktemp('mode_axis')
    (d / 'demo.tlang').write_text(MODE_AXIS_SRC, encoding='utf-8')
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('demo')
    assert shader is not None
    return shader


@pytest.fixture(scope='module')
def two_axis_shader(gl_ctx, tmp_path_factory):
    from tlang import ShaderManager

    d = tmp_path_factory.mktemp('two_axis')
    (d / 'demo.tlang').write_text(TWO_AXIS_SRC, encoding='utf-8')
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('demo')
    assert shader is not None
    return shader


def _make_buffers(gl_ctx):
    pos = gl_ctx.buffer(struct.pack('4f', 1.0, 2.0, 3.0, 4.0))
    pred = gl_ctx.buffer(struct.pack('4f', 10.0, 20.0, 30.0, 40.0))
    out = gl_ctx.buffer(reserve=16)
    return pos, pred, out


@pytest.mark.gl
def test_module_with_no_precompile_axis_is_unaffected(gl_ctx, tmp_path_factory):
    """The stated compatibility guarantee: a module declaring no precompile
    axis at all works exactly as it always has -- get_kernel(name) with no
    extra arguments, no errors, no change."""
    from tlang import ShaderManager

    src = (
        "layout(std430) buffer Out { uint data[]; };\n"
        "[shader('compute')]\n[numthreads(1, 1, 1)]\n"
        "void cs_plain() { data[0] = 7u; }\n"
    )
    d = tmp_path_factory.mktemp('no_axis')
    (d / 'demo.tlang').write_text(src, encoding='utf-8')
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('demo')
    assert shader is not None and shader.ok

    kernel = shader.get_kernel('cs_plain')
    buf = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('Out', buf)
    kernel.dispatch(1, 1, 1)
    assert _read_u32(buf) == 7


@pytest.mark.gl
def test_get_kernel_with_no_value_raises_naming_the_constant_and_permitted_values(gl_ctx, one_axis_shader):
    """A module declaring a precompile axis has no default artifact -- every
    kernel in it (even calling get_kernel with no extra arguments at all)
    must fail loudly, naming the constant and what it will accept."""
    with pytest.raises(TlangAttributeError) as exc_info:
        one_axis_shader.get_kernel('cs_solve')
    msg = str(exc_info.value)
    assert 'stabilizing' in msg
    assert 'False' in msg and 'True' in msg


@pytest.mark.gl
def test_each_precompiled_variant_computes_its_own_baked_in_value(gl_ctx, one_axis_shader):
    """The correctness question that matters most: a variant computing
    something OTHER than what its own baked-in value says. Both variants
    are checked against real dispatch + readback."""
    pos, pred, out_true = _make_buffers(gl_ctx)
    _, _, out_false = _make_buffers(gl_ctx)

    true_kernel = one_axis_shader.get_kernel('cs_solve', stabilizing=True)
    true_kernel.bind_ssbo('Positions', pos)
    true_kernel.bind_ssbo('Predicted', pred)
    true_kernel.bind_ssbo('Out', out_true)
    true_kernel.dispatch(1, 1, 1)
    assert _read_vec4(out_true) == pytest.approx((1.0, 2.0, 3.0, 4.0))

    false_kernel = one_axis_shader.get_kernel('cs_solve', stabilizing=False)
    false_kernel.bind_ssbo('Positions', pos)
    false_kernel.bind_ssbo('Predicted', pred)
    false_kernel.bind_ssbo('Out', out_false)
    false_kernel.dispatch(1, 1, 1)
    assert _read_vec4(out_false) == pytest.approx((10.0, 20.0, 30.0, 40.0))


@pytest.mark.gl
def test_get_kernel_returns_the_identical_precompiled_kernel_every_time(gl_ctx, one_axis_shader):
    """Every precompiled variant already exists by build time (see
    `ShaderManager._build_precompiled_variants`) -- repeat `get_kernel`
    calls for the same value are a plain lookup, never a fresh compile."""
    k1 = one_axis_shader.get_kernel('cs_solve', stabilizing=True)
    k2 = one_axis_shader.get_kernel('cs_solve', stabilizing=True)
    assert k1 is k2


@pytest.mark.gl
def test_value_not_in_the_precompiled_list_raises_naming_the_set(gl_ctx, mode_axis_shader):
    """No on-demand fallback: `mode=3` was never precompiled (only 1, 2, 4
    were), so this must raise -- not silently compile a new variant."""
    with pytest.raises(TlangAttributeError) as exc_info:
        mode_axis_shader.get_kernel('cs_mode', mode=3)
    msg = str(exc_info.value)
    assert 'mode' in msg
    assert '3' in msg
    assert '1' in msg and '2' in msg and '4' in msg  # names the precompiled set


@pytest.mark.gl
def test_undeclared_extern_name_in_get_kernel_raises_naming_it(gl_ctx, mode_axis_shader):
    with pytest.raises(TlangAttributeError) as exc_info:
        mode_axis_shader.get_kernel('cs_mode', bogus=1, mode=1)
    assert 'bogus' in str(exc_info.value)


@pytest.mark.gl
def test_variants_are_binding_compatible(gl_ctx, one_axis_shader):
    """Both arms of `stabilizing ? ... : ...` stay in the artifact regardless
    of the precompiled value (tlang's DCE is textual, not constant-folding),
    so every variant declares the same buffers -- `bind()` needs no
    per-variant re-keying."""
    true_kernel = one_axis_shader.get_kernel('cs_solve', stabilizing=True)
    false_kernel = one_axis_shader.get_kernel('cs_solve', stabilizing=False)

    assert set(true_kernel.bindings) == {'Positions', 'Predicted', 'Out'}
    assert true_kernel.bindings == false_kernel.bindings


@pytest.mark.gl
def test_two_axis_module_requires_both_values_together(gl_ctx, two_axis_shader):
    """A module declaring TWO precompile axes has no way to resolve just
    one -- every combination of every axis's declared values is what gets
    built (the cross product), so a caller must supply both at once."""
    # missing one of the two axes -> error naming the missing one
    with pytest.raises(TlangAttributeError) as exc_info:
        two_axis_shader.get_kernel('cs_two', mode=1)
    msg = str(exc_info.value)
    assert 'flag' in msg

    # both supplied together -> a real, correct precompiled kernel
    kernel = two_axis_shader.get_kernel('cs_two', mode=2, flag=True)
    buf = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('Out', buf)
    kernel.dispatch(1, 1, 1)
    assert _read_u32(buf) == 2

    kernel_off = two_axis_shader.get_kernel('cs_two', mode=2, flag=False)
    buf_off = gl_ctx.buffer(reserve=4)
    kernel_off.bind_ssbo('Out', buf_off)
    kernel_off.dispatch(1, 1, 1)
    assert _read_u32(buf_off) == 0
