# -------------------------------------------------------------
# @file          test_buffer_shorthand.py
# @description   Tests for the [buffer] single-declarator shorthand:
#                `[buffer] Type name[];` desugars to a block whose HANDLE
#                (what Python binds by -- kernel.bindings, bind(),
#                bind_ssbo, BufferPool tags) is the member's own name (or
#                a name='...' override), while the emitted GLSL block name
#                is synthesised, distinct from the handle, and never binds
#                to anything -- see interface_registry._synthesize_block_name.
#                GL-free except the two @pytest.mark.gl tests that prove the
#                handle is what Python actually binds by.
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
# the handle is the member's own name -- never derived, never capitalised.
# The emitted GLSL block name is a different, synthesised identifier: GLSL
# requires a block's own name to differ from its member's, so the shorthand
# can't legally emit `buffer ptcPositions { vec2 ptcPositions[]; }`.
# ---------------------------------------------------------------------------

def test_shorthand_handle_is_the_member_name():
    proc = _processor('demo', "[buffer]\nvec2 ptcPositions[];\n")
    decl = proc.interfaces.resolve('ptcPositions', None, proc.diagnostics)
    assert decl.name == 'ptcPositions'
    assert decl.members[0].name == 'ptcPositions'
    assert 'PtcPositions' not in proc.interfaces


def test_shorthand_emitted_block_name_is_not_the_member_name():
    proc = _processor('demo', "[buffer]\nvec2 ptcPositions[];\n")
    text = _text(proc)
    decl = proc.interfaces.resolve('ptcPositions', None, proc.diagnostics)
    # the illegal, member-shadowing form must never be emitted
    assert 'buffer ptcPositions {' not in text
    # the synthesised block name is distinct, and the member line is untouched
    assert decl.emitted_name != decl.name
    assert f'buffer {decl.emitted_name} {{' in text
    assert 'vec2 ptcPositions[];' in text


def test_shorthand_emitted_block_name_is_deterministic_across_two_builds():
    src = "[buffer]\nvec2 ptcPositions[];\n"
    first_proc = _processor('demo', src)
    second_proc = _processor('demo', src)
    first = first_proc.interfaces.resolve('ptcPositions', None, first_proc.diagnostics)
    second = second_proc.interfaces.resolve('ptcPositions', None, second_proc.diagnostics)
    assert first.emitted_name == second.emitted_name == 'ptcPositions__blk'


def test_struct_form_emitted_name_equals_its_handle():
    """The struct form is unchanged: the author names the block, so that name is both
    the handle and the emitted GLSL identifier -- no synthesis happens here."""
    proc = _processor('demo', "[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n")
    decl = proc.interfaces.resolve('Particles', None, proc.diagnostics)
    assert decl.emitted_name == decl.name == 'Particles'


def test_raw_glsl_buffer_block_is_untouched_by_the_interface_registry():
    """Raw GLSL (`layout(std430) buffer X { ... };`, written directly, with no
    [buffer(...)] attribute at all) never enters the interface table -- it passes
    through verbatim, so its written name is unconditionally both what the driver
    sees and what Python binds by. See test_gl_shader_build.py's BufA/Pinned/
    AutoAssigned for the GL-level proof that `kernel.bindings` keys by it unchanged."""
    proc = _processor('demo', "layout(std430) buffer RawBlock { uint data[]; };\nuniform float pad;\n")
    assert 'RawBlock' not in proc.interfaces
    assert 'layout(std430) buffer RawBlock {' in _text(proc)


# ---------------------------------------------------------------------------
# the shorthand emits its member exactly like the struct form would -- on its
# own line (`[buffer]\nvec2 x[];`) AND on the attribute's own line
# (`[buffer] vec2 x[];`), which is the form the feature was actually
# requested in. Both must produce an identical member line and handle.
# ---------------------------------------------------------------------------

def test_same_line_shorthand_emits_same_member_line_as_two_line_shorthand():
    """Same desugared block either way -- the two-line form additionally
    leaves behind its usual '//<<ATTR ...>>//' marker comment on the
    attribute's own (now-separate) line, which the same-line form has no
    equivalent slot for, so this compares the meaningful GLSL, not the
    full text verbatim."""
    same_line = _text(_processor('demo', "[buffer] vec2 ptcPositions[];\n"))
    two_line = _text(_processor('demo', "[buffer]\nvec2 ptcPositions[];\n"))
    block = 'layout(std430) buffer ptcPositions__blk {\n    vec2 ptcPositions[];\n};'
    assert block in same_line
    assert block in two_line


def test_same_line_handle_is_member_name():
    proc = _processor('demo', "[buffer] vec4 grid[];\n")
    decl = proc.interfaces.resolve('grid', None, proc.diagnostics)
    assert decl.members[0].name == 'grid'
    assert decl.source_member == 'grid'
    assert 'Grid' not in proc.interfaces


def test_same_line_name_override():
    proc = _processor('demo', "[buffer(name='ElementCount')] uint numElements;\n")
    decl = proc.interfaces.resolve('ElementCount', None, proc.diagnostics)
    assert decl.members[0].name == 'numElements'
    assert 'numElements' not in proc.interfaces


def test_same_line_std140():
    proc = _processor('demo', "[buffer(std140)] ComputeDispatch dispatch[];\n")
    text = _text(proc)
    assert 'layout(std140) buffer dispatch__blk {' in text
    assert 'ComputeDispatch dispatch[];' in text
    assert 'dispatch' in proc.interfaces


def test_same_line_sized_array():
    proc = _processor('demo', "[buffer] vec4 x[16];\n")
    text = _text(proc)
    assert 'layout(std430) buffer x__blk {' in text
    assert 'vec4 x[16];' in text
    assert 'x' in proc.interfaces


def test_same_line_scalar():
    proc = _processor('demo', "[buffer] uint numElements;\n")
    text = _text(proc)
    assert 'layout(std430) buffer numElements__blk {' in text
    assert 'uint numElements;' in text
    assert 'numElements' in proc.interfaces


def test_two_consecutive_same_line_declarations_do_not_bleed_into_each_other():
    """Regression: attribute matching used to scan the WHOLE accumulated
    line text for more '[...]' blocks, so a `[]` inside the first
    declarator's array suffix could be mistaken for a second attribute
    block, corrupting the line and letting the parse run on into the next
    physical line's attribute text."""
    proc = _processor('demo', "[buffer] uint data[];\n[buffer] vec2 pos[];\n")
    assert {'data', 'pos'} == {d.name for d in proc.interfaces}
    data = proc.interfaces.resolve('data', None, proc.diagnostics)
    pos = proc.interfaces.resolve('pos', None, proc.diagnostics)
    assert data.members[0].name == 'data' and data.members[0].array == '[]'
    assert pos.members[0].name == 'pos' and pos.members[0].array == '[]'
    text = _text(proc)
    assert 'layout(std430) buffer data__blk {' in text
    assert 'layout(std430) buffer pos__blk {' in text
    assert 'uint data[];' in text
    assert 'vec2 pos[];' in text


def test_same_line_duplicate_handle_error_has_correct_line_and_names_member():
    """The same-line declarator's own errors must point at the physical
    line it's actually written on, not line 1 of some internal buffer.
    The collision here is between the shorthand's handle ('ptcPositions',
    the member's own name) and a struct form explicitly named the same."""
    src = (
        "uniform float pad1;\n"
        "uniform float pad2;\n"
        "[buffer] vec2 ptcPositions[];\n"
        "[buffer(std430)]\n"
        "struct ptcPositions { vec4 other[]; };\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'demo:3' in msg  # the same-line shorthand's real line
    assert "'ptcPositions'" in msg
    assert 'shorthand' in msg


def test_same_line_multiple_declarators_rejected_with_correct_line():
    src = "uniform float pad;\n[buffer] vec2 a[], b[];\n"
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'demo:2' in msg
    assert 'a[]' in msg and 'b[]' in msg


# ---------------------------------------------------------------------------
# a struct may share the attribute's own line, for every declaration attribute
# -- `struct` is a reserved keyword, so it can never be mistaken for the
# single-declarator shorthand and there is nothing to disambiguate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]', '[buffer]'])
def test_same_line_struct_is_accepted_for_every_declaration_attribute(attr):
    processor = _processor('demo', f"{attr} struct X {{ vec3 a; }};\n")
    decl = processor.interfaces._decls['X']
    assert decl.name == 'X'
    assert [m.name for m in decl.members] == ['a']
    assert decl.line == 1


@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]', '[buffer]'])
def test_same_line_struct_body_may_span_following_lines(attr):
    processor = _processor('demo', f"{attr} struct X {{\n    vec3 a;\n    vec2 b;\n}};\n")
    decl = processor.interfaces._decls['X']
    assert [m.name for m in decl.members] == ['a', 'b']


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
# the handle is the member's own name, unchanged -- no case transformation,
# no derivation step at all
# ---------------------------------------------------------------------------

def test_handle_is_member_name_lowercase():
    proc = _processor('demo', "[buffer]\nvec4 grid[];\n")
    assert 'grid' in proc.interfaces
    assert 'Grid' not in proc.interfaces
    decl = proc.interfaces.resolve('grid', None, proc.diagnostics)
    assert decl.kind is InterfaceKind.BUFFER
    assert decl.members[0].name == 'grid'
    assert decl.source_member == 'grid'
    assert decl.emitted_name == 'grid__blk'


def test_handle_is_member_name_camel_case():
    proc = _processor('demo', "[buffer]\nvec2 ptcPositions[];\n")
    assert 'ptcPositions' in proc.interfaces
    assert 'PtcPositions' not in proc.interfaces
    decl = proc.interfaces.resolve('ptcPositions', None, proc.diagnostics)
    assert decl.members[0].name == 'ptcPositions'


# ---------------------------------------------------------------------------
# name= override: makes the given name the HANDLE. No longer mandatory to
# defeat a capitalisation collision (there is none any more) -- it's purely
# for when a different Python-side name than the member's own is wanted.
# ---------------------------------------------------------------------------

def test_name_override_via_name_kwarg():
    proc = _processor('demo', "[buffer(name='ElementCount')]\nuint numElements;\n")
    assert 'ElementCount' in proc.interfaces
    assert 'numElements' not in proc.interfaces
    decl = proc.interfaces.resolve('ElementCount', None, proc.diagnostics)
    assert decl.members[0].name == 'numElements'
    assert decl.source_member == 'numElements'


def test_name_override_block_name_is_still_synthesized_from_the_member():
    """The emitted GLSL block name is always synthesised from the member, regardless
    of a name='...' override -- the override only ever changes the Python-side handle."""
    proc = _processor('demo', "[buffer(name='ElementCount')]\nuint numElements;\n")
    decl = proc.interfaces.resolve('ElementCount', None, proc.diagnostics)
    assert decl.emitted_name == 'numElements__blk'
    text = _text(proc)
    assert 'buffer numElements__blk {' in text
    assert 'ElementCount' not in text  # the handle never appears in emitted GLSL


def test_invalid_name_override_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer(name='123bad')]\nuint n;\n")
    assert '123bad' in str(exc_info.value)


def test_name_override_with_a_space_is_rejected():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer(name='not valid')]\nuint n;\n")
    assert 'not valid' in str(exc_info.value)


def test_empty_name_override_falls_back_to_member_name():
    """`name=''` is indistinguishable from omitting the argument -- the
    attribute-string parser drops empty values before they ever reach a
    Param -- so it must fall back to the member's own name rather than
    silently binding an empty handle."""
    proc = _processor('demo', "[buffer(name='')]\nuint numElements;\n")
    assert 'numElements' in proc.interfaces
    assert '' not in proc.interfaces


def test_registry_level_guard_rejects_an_explicit_empty_handle_override():
    """Defense in depth at the `interface_registry` layer itself, for any
    caller that supplies `block_name=''` directly (not reachable through
    `[buffer(name='')]`, which the attribute parser normalises away above)."""
    with pytest.raises(TlangAttributeError) as exc_info:
        parse_declarator_at("uint n;", 0, 'demo', InterfaceKind.BUFFER, layout='std430', block_name='')
    assert 'not a valid handle' in str(exc_info.value)


# ---------------------------------------------------------------------------
# [buffer(std140)] shorthand -- the layout arg still works positionally
# ---------------------------------------------------------------------------

def test_std140_shorthand():
    proc = _processor('demo', "[buffer(std140)]\nComputeDispatch dispatch[];\n")
    text = _text(proc)
    assert 'layout(std140) buffer dispatch__blk {' in text
    assert 'ComputeDispatch dispatch[];' in text


# ---------------------------------------------------------------------------
# a sized array and a scalar -- not just the overwhelmingly common
# unsized-array case
# ---------------------------------------------------------------------------

def test_sized_array_shorthand():
    proc = _processor('demo', "[buffer]\nvec4 x[16];\n")
    text = _text(proc)
    assert 'layout(std430) buffer x__blk {' in text
    assert 'vec4 x[16];' in text


def test_scalar_shorthand():
    proc = _processor('demo', "[buffer]\nuint numElements;\n")
    text = _text(proc)
    assert 'layout(std430) buffer numElements__blk {' in text
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
    assert 'layout(std430) buffer x__blk {' in text


# ---------------------------------------------------------------------------
# a shorthand's handle colliding with another module's handle across
# [include(...)] must still report a usable error naming both locations --
# reflection-driven binding would otherwise just silently misassign
# ---------------------------------------------------------------------------

def test_shorthand_handle_colliding_with_an_included_modules_handle_reports_a_usable_error(make_shader_dir):
    d = make_shader_dir({
        'base.tlang': "[buffer(std430)]\nstruct ptcPositions { vec4 other[]; };\n",
        'demo.tlang': "[include(base)]\n[buffer] vec2 ptcPositions[];\n",
    })
    from tlang import ShaderManager

    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=None, version='430 core', dir=str(d), strict=True)
    msg = str(exc_info.value)
    assert 'ptcPositions' in msg
    assert 'base' in msg
    assert 'demo' in msg


# ---------------------------------------------------------------------------
# GL: proves the handle -- not any derived/synthesised name -- is what
# Python actually binds by, end to end through a real dispatch
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
def test_gl_shorthand_buffer_binds_by_handle_and_dispatches(gl_ctx, make_shader_dir):
    d = make_shader_dir({'demo.tlang': BUFFER_SHORTHAND_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    kernel = sh.get_kernel('cs_go')

    assert 'values' in kernel.bindings
    assert 'Values' not in kernel.bindings

    data = gl_ctx.buffer(struct.pack('f', 21.0))
    kernel.bind_ssbo('values', data)
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
def test_gl_same_line_shorthand_buffer_binds_by_handle_and_dispatches(gl_ctx, make_shader_dir):
    """The syntax the feature was actually requested in --
    `[buffer] float values[];` on one line -- must also build, bind by the
    handle, dispatch, and read back correctly."""
    d = make_shader_dir({'demo.tlang': SAME_LINE_BUFFER_SHORTHAND_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    kernel = sh.get_kernel('cs_go')

    assert 'values' in kernel.bindings
    assert 'Values' not in kernel.bindings

    data = gl_ctx.buffer(struct.pack('f', 21.0))
    kernel.bind_ssbo('values', data)
    kernel.dispatch(1, 1, 1)

    (result,) = struct.unpack('f', data.read())
    assert result == pytest.approx(42.0)

    data.release()


# ---------------------------------------------------------------------------
# a brace block with no name in front of it: GLSL needs the block named, and
# with several members there is no member name to stand in for one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('src', [
    "[buffer] { vec2 a[]; vec2 b[]; };\n",
    "[buffer]\n{ vec2 a[]; vec2 b[]; };\n",
    "[buffer] { vec2 a[]; };\n",
    "[buffer] { vec2 a[]; vec2 b[]; }\n",
])
def test_unnamed_block_names_both_valid_forms(src):
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', src)
    msg = str(exc_info.value)
    assert 'has no name' in msg
    assert 'struct Name' in msg          # the multi-member fix
    assert 'Type name[]' in msg          # the single-member fix
    assert 'buffer declarator' not in msg  # never leak the internal placeholder


def test_unnamed_block_error_points_at_the_block_line():
    with pytest.raises(TlangAttributeError) as exc_info:
        _processor('demo', "[buffer]\n{ vec2 a[]; vec2 b[]; };\n")
    assert 'demo:2' in str(exc_info.value)
