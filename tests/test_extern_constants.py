# -------------------------------------------------------------
# @file          test_extern_constants.py
# @description   Tests for [extern]: a declarative form for host-supplied
#                constants (`ShaderManager(constants={...})`), emitted as a
#                GLSL `const`, replacing the `#define X {{ X }}` idiom where
#                a typed, reflectable, checked-before-the-driver constant is
#                wanted. `{{ CONSTANT }}` keeps working unchanged alongside
#                it -- that path is untouched by this feature.
#
#                GL-free except the two @pytest.mark.gl tests that prove the
#                declaration-before-use ordering a real GLSL build requires
#                (`[numthreads(BS, 1, 1)]` resolving a declared-above
#                `[extern] int BS;`) and that an [extern]-sized array
#                actually compiles.
#
#                GLSL_VERSION_FOR_CONST_LAYOUT: verified for real on this
#                machine (RTX 3090, driver 616.64) that a `const` used
#                inside `layout(local_size_x = ...)` is rejected by NVIDIA's
#                compiler at `#version 430 core` ("non constant expression
#                in layout value") but accepted unchanged at 440/450/460 --
#                GLSL only relaxed layout-qualifier constant expressions to
#                allow a named `const` (not just a literal) from 4.40 on.
#                This suite's usual default ('430 core', see conftest.py's
#                `build_manager`) predates that, so the numthreads-ordering
#                test below pins a newer version explicitly. `[extern]`
#                sizing an ordinary array (not a layout qualifier) has no
#                such restriction -- see the array test, which passes at 430.
# -------------------------------------------------------------

import pytest

from tlang.errors import TlangAttributeError
from tlang.compiler.shader_processor import ShaderProcessor

GLSL_VERSION_FOR_CONST_LAYOUT = '440 core'


def _processor(name: str, src: str, strict: bool = True) -> ShaderProcessor:
    return ShaderProcessor(name, src, strict=strict)


def _text(proc: ShaderProcessor) -> str:
    return ''.join(line.data for line in proc.module.values())


# ---------------------------------------------------------------------------
# each supported type emits a GLSL literal the compiler accepts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('type_name, value, literal', [
    ('int', 256, '256'),
    ('int', -4, '-4'),
    ('uint', 4, '4u'),
    ('float', 0.5, '0.5'),
    ('float', 1, '1.0'),           # a Python int widens harmlessly into a float constant
    ('bool', True, 'true'),
    ('bool', False, 'false'),
])
def test_each_supported_type_emits_a_valid_literal(type_name, value, literal):
    proc = _processor('demo', f"[extern] {type_name} X;\n")
    proc.resolve_externs({'X': value})
    assert f'const {type_name} X = {literal};' in _text(proc)


def test_float_literal_always_carries_a_decimal_point_or_exponent():
    """So `1` never reads as an int literal in a float context -- GLSL requires the dot."""
    proc = _processor('demo', "[extern] float X;\n")
    proc.resolve_externs({'X': 3})
    text = _text(proc)
    assert 'const float X = 3.0;' in text
    assert 'const float X = 3;' not in text


def test_uint_literal_carries_the_u_suffix():
    proc = _processor('demo', "[extern] uint X;\n")
    proc.resolve_externs({'X': 0})
    assert 'const uint X = 0u;' in _text(proc)


# ---------------------------------------------------------------------------
# same-line and next-line forms both work, and emit identically
# ---------------------------------------------------------------------------

def test_same_line_and_next_line_forms_emit_the_same_declaration():
    same_line = _processor('demo', "[extern] int BLOCK_SIZE;\n")
    same_line.resolve_externs({'BLOCK_SIZE': 256})

    next_line = _processor('demo', "[extern]\nint BLOCK_SIZE;\n")
    next_line.resolve_externs({'BLOCK_SIZE': 256})

    assert 'const int BLOCK_SIZE = 256;' in _text(same_line)
    assert 'const int BLOCK_SIZE = 256;' in _text(next_line)


def test_same_line_form_preserves_trailing_text_on_the_line():
    proc = _processor('demo', "[extern] int A; // note\nuniform float pad;\n")
    proc.resolve_externs({'A': 5})
    text = _text(proc)
    assert 'const int A = 5; // note' in text
    assert 'uniform float pad;' in text


def test_next_line_form_leaves_the_usual_attribute_marker_on_its_own_line():
    proc = _processor('demo', "[extern]\nint BLOCK_SIZE;\n")
    proc.resolve_externs({'BLOCK_SIZE': 256})
    text = _text(proc)
    assert "//<<ATTR 'extern'>>//" in text
    assert 'const int BLOCK_SIZE = 256;' in text


# ---------------------------------------------------------------------------
# missing constant(s) -> a tlang error naming the constant, its type, and
# the module; several missing constants in one module are reported together
# ---------------------------------------------------------------------------

def test_missing_constant_names_it_type_and_module():
    proc = _processor('demo', "[extern] int BLOCK_SIZE;\n")
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_externs({})
    msg = str(exc_info.value)
    assert 'demo' in msg
    assert 'BLOCK_SIZE' in msg
    assert 'int' in msg
    assert 'constants=' in msg


def test_several_missing_constants_are_reported_together():
    proc = _processor('demo', "[extern] int A;\n[extern] float B;\n[extern] bool C;\n")
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_externs({})
    msg = str(exc_info.value)
    assert 'A' in msg and 'B' in msg and 'C' in msg


def test_missing_with_a_default_falls_back_instead_of_erroring():
    proc = _processor('demo', "[extern] int WARP_SIZE = 32;\n")
    proc.resolve_externs({})  # must not raise
    assert 'const int WARP_SIZE = 32;' in _text(proc)


# ---------------------------------------------------------------------------
# wrong type -> a tlang error naming the constant, its declared type, and
# what was actually supplied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('type_name, bad_value', [
    ('int', 'sixteen'),
    ('int', 1.5),
    ('int', True),
    ('uint', -1),
    ('uint', 'four'),
    ('float', 'half'),
    ('bool', 1),
    ('bool', 0.0),
])
def test_wrong_type_names_constant_declared_type_and_supplied_value(type_name, bad_value):
    proc = _processor('demo', f"[extern] {type_name} X;\n")
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_externs({'X': bad_value})
    msg = str(exc_info.value)
    assert 'X' in msg
    assert type_name in msg
    assert type(bad_value).__name__ in msg


# ---------------------------------------------------------------------------
# a default makes a constant optional; a supplied value overrides it
# ---------------------------------------------------------------------------

def test_default_makes_the_constant_optional():
    proc = _processor('demo', "[extern] int WARP_SIZE = 32;\n")
    proc.resolve_externs({})
    assert 'const int WARP_SIZE = 32;' in _text(proc)


def test_supplied_value_overrides_the_default():
    proc = _processor('demo', "[extern] int WARP_SIZE = 32;\n")
    proc.resolve_externs({'WARP_SIZE': 64})
    text = _text(proc)
    assert 'const int WARP_SIZE = 64;' in text
    assert '32' not in text


def test_default_value_type_checked_the_same_as_a_supplied_one():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] bool FLAG = 5;\n")
    assert 'FLAG' in str(exc_info.value)


# ---------------------------------------------------------------------------
# reflection: a caller can ask what a module requires
# ---------------------------------------------------------------------------

def test_externs_reflects_declared_requirements_before_resolution():
    proc = _processor('demo', "[extern] int BLOCK_SIZE;\n[extern] float SCALE = 1.0;\n")
    externs = proc.externs
    assert set(externs) == {'BLOCK_SIZE', 'SCALE'}
    assert externs['BLOCK_SIZE'].type_name == 'int'
    assert externs['BLOCK_SIZE'].has_default is False
    assert externs['SCALE'].has_default is True
    assert externs['SCALE'].default_value == 1.0
    assert externs['BLOCK_SIZE'].resolved is False


def test_externs_reflects_resolved_value_and_literal_after_resolve():
    proc = _processor('demo', "[extern] int BLOCK_SIZE;\n")
    proc.resolve_externs({'BLOCK_SIZE': 256})
    decl = proc.externs['BLOCK_SIZE']
    assert decl.resolved is True
    assert decl.value == 256
    assert decl.literal == '256'


def test_shader_manager_exposes_resolved_externs(make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': "[extern] int BLOCK_SIZE;\n"})
    sm = ShaderManager(ctx=None, version='430 core', dir=str(d), constants={'BLOCK_SIZE': 256})
    shader = sm.get_shader('demo')
    assert shader is not None
    assert shader.externs['BLOCK_SIZE'].value == 256
    assert shader.externs['BLOCK_SIZE'].literal == '256'


# ---------------------------------------------------------------------------
# duplicates and malformed declarations
# ---------------------------------------------------------------------------

def test_duplicate_extern_name_in_one_module_raises():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] int A;\n[extern] float A;\n")
    msg = str(exc_info.value)
    assert 'A' in msg
    assert 'demo:1' in msg


def test_extern_name_colliding_with_a_hand_written_const_raises():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "const int A = 1;\n[extern] int A;\n")
    assert 'A' in str(exc_info.value)


def test_unsupported_type_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] vec2 X;\n")
    msg = str(exc_info.value)
    assert 'vec2' in msg
    assert 'int' in msg and 'float' in msg  # names the supported set


def test_array_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] int X[4];\n")
    assert 'array' in str(exc_info.value).lower()


def test_malformed_bool_default_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] bool X = maybe;\n")
    assert 'X' in str(exc_info.value)


def test_multiple_declarators_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[extern] int A, B;\n")
    msg = str(exc_info.value)
    assert 'A' in msg and 'B' in msg


# ---------------------------------------------------------------------------
# module scope only
# ---------------------------------------------------------------------------

def test_extern_inside_a_function_body_is_rejected():
    src = "void helper() {\n    [extern] int A;\n}\n"
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'extern' in msg
    assert 'cannot be used here' in msg or 'global' in msg


# ---------------------------------------------------------------------------
# {{ CONSTANT }} templating keeps working unchanged, alongside [extern] in
# the very same module
# ---------------------------------------------------------------------------

def test_jinja_templating_still_works_alongside_extern(make_shader_dir):
    from tlang import ShaderManager

    src = (
        "[extern] int BLOCK_SIZE;\n"
        "#define DOUBLED {{ DOUBLE }}\n"
        "uniform float pad;\n"
    )
    d = make_shader_dir({'demo.tlang': src})
    sm = ShaderManager(
        ctx=None, version='430 core', dir=str(d),
        constants={'BLOCK_SIZE': 256, 'DOUBLE': 8},
    )
    shader = sm.get_shader('demo')
    assert shader is not None
    assert shader.externs['BLOCK_SIZE'].value == 256


# ---------------------------------------------------------------------------
# GL: declaration-before-use ordering, for real -- [numthreads(BS, 1, 1)]
# resolving a module-scope [extern] int BS, queried back from the linked
# kernel's actual work-group size (not a text-level inspection).
# ---------------------------------------------------------------------------

EXTERN_NUMTHREADS_SRC = """\
layout(std430) buffer Data { uint data[]; };

[extern] int BS;

[shader('compute')]
[numthreads(BS, 1, 1)]
void cs_extern() {
    data[0] = 0u;
}
"""


@pytest.mark.gl
def test_gl_extern_constant_resolves_ordering_for_numthreads(gl_ctx, make_shader_dir):
    """version='440 core', not this suite's usual '430 core' -- see the module docstring's
    note on GLSL_VERSION_FOR_CONST_LAYOUT: a `const` inside `layout(local_size_x = ...)` is
    only a constant-foldable integer expression from GLSL 4.40 on. NVIDIA's compiler (verified
    for real on this machine, RTX 3090 / driver 616.64) enforces that at #version 430 -- the
    identical text with only the `#version` line changed to 440/450/460 compiles clean.
    """
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': EXTERN_NUMTHREADS_SRC})
    sm = ShaderManager(ctx=gl_ctx, version=GLSL_VERSION_FOR_CONST_LAYOUT, dir=str(d), constants={'BS': 256})
    shader = sm.get_shader('demo')
    assert shader is not None
    kernel = shader.get_kernel('cs_extern')
    assert kernel.local_size == (256, 1, 1)


# ---------------------------------------------------------------------------
# GL: an array sized by an [extern] int actually compiles
# ---------------------------------------------------------------------------

EXTERN_ARRAY_SRC = """\
layout(std430) buffer Data { uint data[]; };

[extern] int N;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_array() {
    uint local_scratch[N];
    for (int i = 0; i < N; i++) local_scratch[i] = uint(i);
    data[0] = local_scratch[N - 1];
}
"""


@pytest.mark.gl
def test_gl_array_sized_by_extern_constant_compiles(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': EXTERN_ARRAY_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), constants={'N': 64})
    shader = sm.get_shader('demo')
    assert shader is not None
    assert shader.ok
    assert 'cs_array' in shader.kernels
