# -------------------------------------------------------------
# @file          test_buffer_shorthand.py
# @description   Tests for the [buffer] single-declarator shorthand:
#                `[buffer] Type name[];` desugars to the equivalent
#                `[buffer(std430)] struct Name { Type name[]; };`, with
#                the block name derived from the member (or overridden
#                via name='...'). GL-free except the one @pytest.mark.gl
#                test that proves the derived name is what Python binds by.
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangAttributeError
from tlang.frontend.interface_registry import InterfaceKind, parse_declarator_at
from tlang.compiler.shader_processor import ShaderProcessor


def _processor(name: str, src: str, strict: bool = True) -> ShaderProcessor:
    return ShaderProcessor(name, src, strict=strict)


def _text(proc: ShaderProcessor) -> str:
    return ''.join(line.data for line in proc.module.values())


# ---------------------------------------------------------------------------
# the shorthand emits the same GLSL as the equivalent struct form -- on its
# own line (`[buffer]\nvec2 x[];`) AND on the attribute's own line
# (`[buffer] vec2 x[];`), which is the form the feature was actually
# requested in. Both must produce identical output.
# ---------------------------------------------------------------------------

def test_shorthand_emits_same_glsl_as_struct_form():
    shorthand = _processor('demo', "[buffer]\nvec2 ptcPositions[];\n")
    struct_form = _processor('demo', "[buffer(std430)]\nstruct PtcPositions { vec2 ptcPositions[]; };\n")
    assert _text(shorthand) == _text(struct_form)
    assert 'layout(std430) buffer PtcPositions {' in _text(shorthand)
    assert 'vec2 ptcPositions[];' in _text(shorthand)


def test_same_line_shorthand_emits_same_glsl_as_two_line_shorthand():
    """Same desugared block either way -- the two-line form additionally
    leaves behind its usual '//<<ATTR ...>>//' marker comment on the
    attribute's own (now-separate) line, which the same-line form has no
    equivalent slot for, so this compares the meaningful GLSL, not the
    full text verbatim."""
    same_line = _text(_processor('demo', "[buffer] vec2 ptcPositions[];\n"))
    two_line = _text(_processor('demo', "[buffer]\nvec2 ptcPositions[];\n"))
    block = 'layout(std430) buffer PtcPositions {\n    vec2 ptcPositions[];\n};'
    assert block in same_line
    assert block in two_line


def test_same_line_name_derivation():
    proc = _processor('demo', "[buffer] vec4 grid[];\n")
    decl = proc.interfaces.resolve('Grid', None, proc.diagnostics)
    assert decl.members[0].name == 'grid'
    assert decl.source_member == 'grid'


def test_same_line_name_override():
    proc = _processor('demo', "[buffer(name='ElementCount')] uint numElements;\n")
    decl = proc.interfaces.resolve('ElementCount', None, proc.diagnostics)
    assert decl.members[0].name == 'numElements'


def test_same_line_std140():
    proc = _processor('demo', "[buffer(std140)] ComputeDispatch dispatch[];\n")
    text = _text(proc)
    assert 'layout(std140) buffer Dispatch {' in text
    assert 'ComputeDispatch dispatch[];' in text


def test_same_line_sized_array():
    proc = _processor('demo', "[buffer] vec4 x[16];\n")
    text = _text(proc)
    assert 'layout(std430) buffer X {' in text
    assert 'vec4 x[16];' in text


def test_same_line_scalar():
    proc = _processor('demo', "[buffer] uint numElements;\n")
    text = _text(proc)
    assert 'layout(std430) buffer NumElements {' in text
    assert 'uint numElements;' in text


def test_two_consecutive_same_line_declarations_do_not_bleed_into_each_other():
    """Regression: attribute matching used to scan the WHOLE accumulated
    line text for more '[...]' blocks, so a `[]` inside the first
    declarator's array suffix could be mistaken for a second attribute
    block, corrupting the line and letting the parse run on into the next
    physical line's attribute text."""
    proc = _processor('demo', "[buffer] uint data[];\n[buffer] vec2 pos[];\n")
    assert {'Data', 'Pos'} == {d.name for d in proc.interfaces}
    data = proc.interfaces.resolve('Data', None, proc.diagnostics)
    pos = proc.interfaces.resolve('Pos', None, proc.diagnostics)
    assert data.members[0].name == 'data' and data.members[0].array == '[]'
    assert pos.members[0].name == 'pos' and pos.members[0].array == '[]'
    text = _text(proc)
    assert 'layout(std430) buffer Data {' in text
    assert 'layout(std430) buffer Pos {' in text
    assert 'uint data[];' in text
    assert 'vec2 pos[];' in text


def test_same_line_duplicate_block_name_error_has_correct_line_and_names_member():
    """The same-line declarator's own errors must point at the physical
    line it's actually written on, not line 1 of some internal buffer."""
    src = (
        "uniform float pad1;\n"
        "uniform float pad2;\n"
        "[buffer] vec2 ptcPositions[];\n"
        "[buffer(std430)]\n"
        "struct PtcPositions { vec4 other[]; };\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'demo:3' in msg  # the same-line shorthand's real line
    assert 'PtcPositions' in msg
    assert 'ptcPositions' in msg
    assert 'derived from' in msg


def test_same_line_multiple_declarators_rejected_with_correct_line():
    src = "uniform float pad;\n[buffer] vec2 a[], b[];\n"
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'demo:2' in msg
    assert 'a[]' in msg and 'b[]' in msg


# ---------------------------------------------------------------------------
# a struct sharing the attribute's own line is deliberately rejected -- for
# every declaration attribute, not just [buffer] -- with a message telling
# the author to put it on its own line, instead of silently mis-scanning
# or silently starting to support a new, undocumented same-line struct form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]', '[buffer]'])
def test_same_line_struct_is_rejected_for_every_declaration_attribute(attr):
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', f"{attr} struct X {{ vec3 a; }};\n")
    msg = str(exc_info.value)
    assert 'own line' in msg
    assert 'struct X' in msg


@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]'])
def test_same_line_non_struct_content_is_rejected_for_varyings_and_uniforms(attr):
    """Only [buffer] gets the single-declarator shorthand; a same-line
    declarator after [varyings]/[uniforms] is rejected, not silently
    mis-parsed as a struct-form attempt."""
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', f"{attr} vec2 x;\n")
    msg = str(exc_info.value)
    assert 'same line' in msg


# ---------------------------------------------------------------------------
# the two-line struct form -- including a struct spanning several physical
# lines -- must still work exactly as before; the same-line handling above
# must never intercept it
# ---------------------------------------------------------------------------

def test_multiline_struct_after_buffer_attribute_still_works():
    src = (
        "[buffer(std430)]\n"          # line 1
        "struct Particles {\n"         # line 2
        "    vec4 pos;\n"              # line 3
        "    vec4 vel;\n"              # line 4
        "};\n"                         # line 5
        "\n"                           # line 6
        "uniform float k;\n"           # line 7
    )
    proc = _processor('demo', src)
    decl = proc.interfaces.resolve('Particles', None, proc.diagnostics)
    assert [m.name for m in decl.members] == ['pos', 'vel']
    text = _text(proc)
    assert 'buffer Particles {' in text
    assert 'vec4 pos;' in text and 'vec4 vel;' in text
    assert 'uniform float k;' in text


# ---------------------------------------------------------------------------
# name derivation: upper-case the member's first character
# ---------------------------------------------------------------------------

def test_name_derivation_lowercase_member():
    proc = _processor('demo', "[buffer]\nvec4 grid[];\n")
    assert 'Grid' in proc.interfaces
    decl = proc.interfaces.resolve('Grid', None, proc.diagnostics)
    assert decl.kind is InterfaceKind.BUFFER
    assert decl.members[0].name == 'grid'
    assert decl.source_member == 'grid'


def test_name_derivation_camel_case_member():
    proc = _processor('demo', "[buffer]\nvec2 ptcPositions[];\n")
    assert 'PtcPositions' in proc.interfaces
    decl = proc.interfaces.resolve('PtcPositions', None, proc.diagnostics)
    assert decl.members[0].name == 'ptcPositions'


# ---------------------------------------------------------------------------
# name= override
# ---------------------------------------------------------------------------

def test_name_override_via_name_kwarg():
    proc = _processor('demo', "[buffer(name='ElementCount')]\nuint numElements;\n")
    assert 'ElementCount' in proc.interfaces
    assert 'NumElements' not in proc.interfaces
    decl = proc.interfaces.resolve('ElementCount', None, proc.diagnostics)
    assert decl.members[0].name == 'numElements'
    assert decl.source_member == 'numElements'


def test_invalid_name_override_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer(name='123bad')]\nuint n;\n")
    assert '123bad' in str(exc_info.value)


def test_name_override_with_a_space_is_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer(name='not valid')]\nuint n;\n")
    assert 'not valid' in str(exc_info.value)


def test_empty_name_override_falls_back_to_derivation():
    """`name=''` is indistinguishable from omitting the argument -- the
    attribute-string parser drops empty values before they ever reach a
    Param -- so it must fall back to the derived name rather than silently
    binding an empty block name."""
    proc = _processor('demo', "[buffer(name='')]\nuint numElements;\n")
    assert 'NumElements' in proc.interfaces
    assert '' not in proc.interfaces


def test_registry_level_guard_rejects_an_explicit_empty_block_name():
    """Defense in depth at the `interface_registry` layer itself, for any
    caller that supplies `block_name=''` directly (not reachable through
    `[buffer(name='')]`, which the attribute parser normalises away above)."""
    with pytest.raises(TlangAttributeError) as exc_info:
        parse_declarator_at("uint n;", 0, 'demo', InterfaceKind.BUFFER, layout='std430', block_name='')
    assert 'not a valid block name' in str(exc_info.value)


# ---------------------------------------------------------------------------
# [buffer(std140)] shorthand -- the layout arg still works positionally
# ---------------------------------------------------------------------------

def test_std140_shorthand():
    proc = _processor('demo', "[buffer(std140)]\nComputeDispatch dispatch[];\n")
    text = _text(proc)
    assert 'layout(std140) buffer Dispatch {' in text
    assert 'ComputeDispatch dispatch[];' in text


# ---------------------------------------------------------------------------
# a sized array and a scalar -- not just the overwhelmingly common
# unsized-array case
# ---------------------------------------------------------------------------

def test_sized_array_shorthand():
    proc = _processor('demo', "[buffer]\nvec4 x[16];\n")
    text = _text(proc)
    assert 'layout(std430) buffer X {' in text
    assert 'vec4 x[16];' in text


def test_scalar_shorthand():
    proc = _processor('demo', "[buffer]\nuint numElements;\n")
    text = _text(proc)
    assert 'layout(std430) buffer NumElements {' in text
    assert 'uint numElements;' in text


# ---------------------------------------------------------------------------
# multiple declarators are ambiguous -- rejected with a helpful message
# ---------------------------------------------------------------------------

def test_multiple_declarators_rejected_with_helpful_message():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer]\nvec2 a[], b[];\n")
    msg = str(exc_info.value)
    assert 'a[]' in msg and 'b[]' in msg
    assert 'one name' in msg or 'ambiguous' in msg
    assert 'struct' in msg


# ---------------------------------------------------------------------------
# a declaration that yields no member is rejected too
# ---------------------------------------------------------------------------

def test_empty_shorthand_declares_no_member_raises():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer]\n;\n")
    assert 'empty' in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# a struct on the next line still takes the old, unchanged path
# ---------------------------------------------------------------------------

def test_struct_on_next_line_still_takes_old_path():
    proc = _processor('demo', "[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n")
    decl = proc.interfaces.resolve('Particles', None, proc.diagnostics)
    assert len(decl.members) == 1
    assert decl.members[0].name == 'pos'
    assert decl.source_member == ''  # only the shorthand path sets this


def test_struct_form_with_multiple_members_still_works_unchanged():
    proc = _processor('demo', "[buffer(std430)]\nstruct Contacts { uint a; uint b; };\n")
    decl = proc.interfaces.resolve('Contacts', None, proc.diagnostics)
    assert [m.name for m in decl.members] == ['a', 'b']


# ---------------------------------------------------------------------------
# a {{ CONSTANT }}-sized array still works (plain text at this stage --
# substitution happens later, in ShaderManager)
# ---------------------------------------------------------------------------

def test_constant_sized_array_shorthand():
    proc = _processor('demo', "[buffer]\nvec4 x[{{ N }}];\n")
    text = _text(proc)
    assert 'vec4 x[{{ N }}];' in text
    assert 'layout(std430) buffer X {' in text


# ---------------------------------------------------------------------------
# duplicate-block-name diagnostic: must name both the derived block name
# and the member it came from, since the block name never appears
# literally in the shorthand's own source line
# ---------------------------------------------------------------------------

def test_duplicate_block_name_error_names_derived_name_and_member():
    src = (
        "[buffer]\nvec2 ptcPositions[];\n\n"
        "[buffer(std430)]\nstruct PtcPositions { vec4 other[]; };\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'PtcPositions' in msg
    assert 'ptcPositions' in msg
    assert 'derived from' in msg


# ---------------------------------------------------------------------------
# GL: proves the derived name is what Python actually binds by
# ---------------------------------------------------------------------------

BUFFER_SHORTHAND_SRC = """\
[buffer]
float values[];

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_go() {{
    values[0] = values[0] * 2.0;
}}
"""


@pytest.mark.gl
def test_gl_shorthand_buffer_binds_by_derived_name_and_dispatches(gl_ctx, make_shader_dir):
    d = make_shader_dir({'demo.tlang': BUFFER_SHORTHAND_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    kernel = sh.get_kernel('cs_go')

    assert 'Values' in kernel.bindings

    data = gl_ctx.buffer(struct.pack('f', 21.0))
    kernel.bind_ssbo('Values', data)
    kernel.dispatch(1, 1, 1)

    (result,) = struct.unpack('f', data.read())
    assert result == pytest.approx(42.0)

    data.release()


SAME_LINE_BUFFER_SHORTHAND_SRC = """\
[buffer] float values[];

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_go() {{
    values[0] = values[0] * 2.0;
}}
"""


@pytest.mark.gl
def test_gl_same_line_shorthand_buffer_binds_by_derived_name_and_dispatches(gl_ctx, make_shader_dir):
    """The syntax the feature was actually requested in --
    `[buffer] float values[];` on one line -- must also build, bind by the
    derived name, dispatch, and read back correctly."""
    d = make_shader_dir({'demo.tlang': SAME_LINE_BUFFER_SHORTHAND_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    kernel = sh.get_kernel('cs_go')

    assert 'Values' in kernel.bindings

    data = gl_ctx.buffer(struct.pack('f', 21.0))
    kernel.bind_ssbo('Values', data)
    kernel.dispatch(1, 1, 1)

    (result,) = struct.unpack('f', data.read())
    assert result == pytest.approx(42.0)

    data.release()
