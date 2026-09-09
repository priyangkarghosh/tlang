# -------------------------------------------------------------
# @file          test_interface_registry.py
# @description   GL-free tests for interface_registry: struct parsing,
#                location spans, GLSL emission, and InterfaceTable.
# -------------------------------------------------------------

import pytest

from tlang.frontend.attribute_registry import Diagnostics
from tlang.errors import SourceLocation, TlangAttributeError, TlangSyntaxError
from tlang.frontend.interface_registry import (
    InterfaceDecl,
    InterfaceKind,
    InterfaceMember,
    InterfaceTable,
    emit_glsl,
    is_arrayed,
    location_span,
    parse_struct_at,
)
from tlang.shader_stages import ShaderStage


# ---------------------------------------------------------------------------
# parse_struct_at
# ---------------------------------------------------------------------------

def test_no_struct_at_all_returns_none():
    assert parse_struct_at('void foo() {}\n', 0, 'demo', InterfaceKind.VARYINGS) is None


def test_basic_struct_parses_members_in_order():
    src = "struct VertexOut { vec3 color; vec2 uv; };\n"
    decl, end = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert decl.name == 'VertexOut'
    assert decl.kind is InterfaceKind.VARYINGS
    assert [m.name for m in decl.members] == ['color', 'uv']
    assert [m.type_name for m in decl.members] == ['vec3', 'vec2']
    assert src[end:] == '\n'


def test_struct_line_is_the_struct_keyword_line():
    src = "\n\nstruct Frame { mat4 view; };\n"
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.UNIFORMS)
    assert decl.line == 3
    assert decl.location == SourceLocation('demo', 3)


def test_multi_declarator_statement_yields_multiple_members():
    src = "struct S { vec3 a, b; };"
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert [m.name for m in decl.members] == ['a', 'b']
    assert all(m.type_name == 'vec3' for m in decl.members)


def test_qualifier_is_captured_separately_from_type():
    src = "struct S { flat int id; };"
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    m = decl.members[0]
    assert m.qualifiers == ('flat',)
    assert m.type_name == 'int'
    assert m.name == 'id'


def test_array_suffix_binds_to_its_own_declarator():
    src = "struct S { vec4 pos[4], q; };"
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    pos, q = decl.members
    assert pos.name == 'pos' and pos.array == '[4]'
    assert q.name == 'q' and q.array == ''


def test_unsized_array_suffix_preserved_exactly():
    src = "struct S { vec4 pos[]; };"
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.BUFFER)
    assert decl.members[0].array == '[]'


def test_member_line_numbers_track_multiline_body():
    src = (
        "struct VertexOut {\n"
        "    vec3 color;\n"
        "    vec2 uv;\n"
        "};\n"
    )
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    color, uv = decl.members
    assert color.line == 2
    assert uv.line == 3


def test_comment_inside_body_is_ignored_and_does_not_confuse_braces():
    src = (
        "struct S {\n"
        "    // a comment with { braces } inside it\n"
        "    vec3 color; /* another { comment */\n"
        "    float x;\n"
        "};\n"
    )
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert [m.name for m in decl.members] == ['color', 'x']


def test_line_numbers_unaffected_by_comment_shrinkage():
    """Masking never removes characters (only blanks them), so line numbers
    computed against it match the original source exactly."""
    src = (
        "struct S {\n"
        "    vec3 color; // trailing comment that would shift offsets if stripped\n"
        "    float x;\n"
        "};\n"
    )
    decl, _ = parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert decl.members[1].line == 3


def test_start_offset_skips_earlier_text():
    src = "int unrelated;\nstruct S { float x; };"
    offset = src.index('struct')
    decl, _ = parse_struct_at(src, offset, 'demo', InterfaceKind.VARYINGS)
    assert decl.name == 'S'


def test_opts_are_carried_onto_the_decl():
    src = "struct Lights { vec4 positions[16]; };"
    decl, _ = parse_struct_at(
        src, 0, 'demo', InterfaceKind.UNIFORMS, layout='std140', block=True, locations=False,
    )
    assert decl.layout == 'std140'
    assert decl.block is True
    assert decl.locations is False


# --- malformed structs: TlangSyntaxError, never a silent None ---

def test_unterminated_struct_raises_syntax_error():
    src = "struct S { vec3 a;"
    with pytest.raises(TlangSyntaxError) as exc:
        parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert exc.value.location == SourceLocation('demo', 1)


def test_missing_semicolon_after_close_brace_raises():
    src = "struct S { vec3 a; } extra;"
    with pytest.raises(TlangSyntaxError):
        parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)


def test_malformed_declarator_raises_located_error():
    src = "struct S { vec3 a[bad; };"
    with pytest.raises(TlangSyntaxError) as exc:
        parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)
    assert exc.value.location.module == 'demo'


def test_empty_struct_body_raises():
    src = "struct S { };"
    with pytest.raises(TlangSyntaxError):
        parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)


def test_struct_keyword_without_name_is_malformed_not_absent():
    """A `struct` token that doesn't form a valid header must raise, not be
    skipped in favour of some later, unrelated struct in the same file."""
    src = "struct { vec3 a; };\nstruct Real { vec3 b; };"
    with pytest.raises(TlangSyntaxError):
        parse_struct_at(src, 0, 'demo', InterfaceKind.VARYINGS)


# ---------------------------------------------------------------------------
# location_span
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('type_name', [
    'bool', 'float', 'int', 'uint', 'vec2', 'vec3', 'vec4',
    'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4',
    'bvec2', 'bvec3', 'bvec4', 'double', 'dvec2',
])
def test_single_location_types(type_name):
    assert location_span(type_name, '') == 1


@pytest.mark.parametrize('type_name', ['dvec3', 'dvec4'])
def test_double_location_types(type_name):
    assert location_span(type_name, '') == 2


@pytest.mark.parametrize('type_name,expected', [
    ('mat2', 2), ('mat3', 3), ('mat4', 4),
    ('mat2x3', 2), ('mat3x2', 3), ('mat4x2', 4),
])
def test_matrix_span_is_column_count(type_name, expected):
    assert location_span(type_name, '') == expected


@pytest.mark.parametrize('type_name,expected', [
    ('dmat2', 2), ('dmat3', 6), ('dmat4', 8),
    ('dmat2x2', 2), ('dmat4x2', 4), ('dmat2x4', 4),
])
def test_double_matrix_span(type_name, expected):
    assert location_span(type_name, '') == expected


def test_sized_array_multiplies_span():
    assert location_span('vec4', '[4]') == 4
    assert location_span('mat4', '[3]') == 12


def test_unsized_array_is_unmeasurable():
    assert location_span('vec3', '[]') is None


def test_unknown_type_is_unmeasurable():
    assert location_span('MyStruct', '') is None


def test_array_with_non_integer_literal_is_unmeasurable():
    assert location_span('float', '[SIZE]') is None
    assert location_span('float', '[{{ N }}]') is None


# ---------------------------------------------------------------------------
# emit_glsl
# ---------------------------------------------------------------------------

def _varyings_decl(locations=True):
    return InterfaceDecl(
        name='VertexOut', kind=InterfaceKind.VARYINGS, module='demo', line=1, locations=locations,
        members=(
            InterfaceMember('vec3', 'color', line=1),
            InterfaceMember('vec2', 'uv', line=1),
        ),
    )


def test_emit_varyings_with_locations():
    lines = emit_glsl(_varyings_decl(), direction='out')
    assert lines == [
        'layout(location = 0) out vec3 color;',
        'layout(location = 1) out vec2 uv;',
    ]


def test_emit_varyings_location_offset_by_span():
    decl = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='demo', line=1,
        members=(InterfaceMember('mat4', 'view', line=1), InterfaceMember('vec3', 'color', line=2)),
    )
    lines = emit_glsl(decl, direction='in', base_location=2)
    assert lines[0] == 'layout(location = 2) in mat4 view;'
    assert lines[1] == 'layout(location = 6) in vec3 color;'


def test_emit_varyings_without_locations():
    lines = emit_glsl(_varyings_decl(locations=False), direction='out')
    assert lines == ['out vec3 color;', 'out vec2 uv;']


def test_emit_varyings_preserves_qualifiers_before_direction():
    decl = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='demo', line=1,
        members=(InterfaceMember('int', 'id', qualifiers=('flat',), line=1),),
    )
    lines = emit_glsl(decl, direction='out', base_location=3)
    assert lines == ['layout(location = 3) flat out int id;']


def test_emit_varyings_requires_direction():
    with pytest.raises(TlangAttributeError):
        emit_glsl(_varyings_decl())


def test_emit_loose_uniforms():
    decl = InterfaceDecl(
        name='Frame', kind=InterfaceKind.UNIFORMS, module='demo', line=1,
        members=(InterfaceMember('mat4', 'view', line=1), InterfaceMember('float', 'time', line=2)),
    )
    assert emit_glsl(decl) == ['uniform mat4 view;', 'uniform float time;']


def test_emit_ubo_block():
    decl = InterfaceDecl(
        name='Frame', kind=InterfaceKind.UNIFORMS, module='demo', line=1, layout='std140', block=True,
        members=(InterfaceMember('mat4', 'view', line=1), InterfaceMember('float', 'time', line=2)),
    )
    assert emit_glsl(decl) == [
        'layout(std140) uniform Frame {',
        '    mat4 view;',
        '    float time;',
        '};',
    ]


def test_emit_buffer_block():
    decl = InterfaceDecl(
        name='Particles', kind=InterfaceKind.BUFFER, module='demo', line=1, layout='std430',
        members=(InterfaceMember('vec4', 'pos', array='[]', line=1),),
    )
    assert emit_glsl(decl) == [
        'layout(std430) buffer Particles {',
        '    vec4 pos[];',
        '};',
    ]


# --- arrayed per-vertex interfaces ---

@pytest.mark.parametrize('stage,direction,expected', [
    (ShaderStage.VERT, 'in', False), (ShaderStage.VERT, 'out', False),
    (ShaderStage.FRAG, 'in', False), (ShaderStage.FRAG, 'out', False),
    (ShaderStage.GEOM, 'in', True), (ShaderStage.GEOM, 'out', False),
    (ShaderStage.TESC, 'in', True), (ShaderStage.TESC, 'out', True),
    (ShaderStage.TESE, 'in', True), (ShaderStage.TESE, 'out', False),
])
def test_is_arrayed_table(stage, direction, expected):
    assert is_arrayed(stage, direction) is expected


def test_emit_varyings_arrayed_appends_array_suffix_without_changing_span():
    decl = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='demo', line=1,
        members=(InterfaceMember('vec3', 'color', line=1), InterfaceMember('mat4', 'm', line=2)),
    )
    lines = emit_glsl(decl, direction='in', arrayed=True)
    assert lines[0] == 'layout(location = 0) in vec3 color[];'
    assert lines[1] == 'layout(location = 1) in mat4 m[];'  # span of 4, unaffected by arraying


def test_emit_varyings_arrayed_member_that_already_has_array_raises():
    decl = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='demo', line=5,
        members=(InterfaceMember('float', 'weights', array='[3]', line=7),),
    )
    with pytest.raises(TlangAttributeError) as exc:
        emit_glsl(decl, direction='in', arrayed=True)
    assert exc.value.location == SourceLocation('demo', 7)


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------

def test_signature_ignores_whitespace_differences():
    a = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='a', line=1,
        members=(InterfaceMember('vec3', 'color', line=1),),
    )
    src = "struct V { vec3  color; };"
    b, _ = parse_struct_at(src, 0, 'b', InterfaceKind.VARYINGS)
    assert a.signature == b.signature == 'vec3 color'


def test_signature_is_order_sensitive():
    a = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='a', line=1,
        members=(InterfaceMember('vec3', 'color', line=1), InterfaceMember('vec2', 'uv', line=2)),
    )
    b = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='b', line=1,
        members=(InterfaceMember('vec2', 'uv', line=1), InterfaceMember('vec3', 'color', line=2)),
    )
    assert a.signature != b.signature


def test_signature_differs_on_type_change():
    src_a = "struct V { vec3 color; };"
    src_b = "struct V { vec4 color; };"
    a, _ = parse_struct_at(src_a, 0, 'a', InterfaceKind.VARYINGS)
    b, _ = parse_struct_at(src_b, 0, 'b', InterfaceKind.VARYINGS)
    assert a.signature != b.signature


def test_signature_includes_qualifiers():
    a = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='a', line=1,
        members=(InterfaceMember('int', 'id', line=1),),
    )
    b = InterfaceDecl(
        name='V', kind=InterfaceKind.VARYINGS, module='a', line=1,
        members=(InterfaceMember('int', 'id', qualifiers=('flat',), line=1),),
    )
    assert a.signature != b.signature
    assert b.signature == 'flat int id'


# ---------------------------------------------------------------------------
# InterfaceTable
# ---------------------------------------------------------------------------

def _decl(name, module, line=1, members=None):
    return InterfaceDecl(
        name=name, kind=InterfaceKind.VARYINGS, module=module, line=line,
        members=members or (InterfaceMember('vec3', 'color', line=line),),
    )


def test_table_add_and_lookup():
    table = InterfaceTable()
    table.add(_decl('VertexOut', 'demo', line=4))
    assert 'VertexOut' in table
    assert len(table) == 1
    assert list(table) == [table.resolve('VertexOut', None, Diagnostics())]


def test_table_add_duplicate_name_raises_naming_both_lines():
    table = InterfaceTable()
    table.add(_decl('VertexOut', 'demo', line=4))
    with pytest.raises(TlangAttributeError) as exc:
        table.add(_decl('VertexOut', 'demo', line=12))
    msg = str(exc.value)
    assert 'demo:4' in msg
    assert 'demo:12' in msg


def test_table_resolve_unknown_name_fails_with_did_you_mean():
    table = InterfaceTable()
    table.add(_decl('VertexOut', 'demo', line=4))
    diagnostics = Diagnostics(strict=True)
    with pytest.raises(TlangAttributeError) as exc:
        table.resolve('VertexOu', SourceLocation('demo', 31), diagnostics)
    msg = str(exc.value)
    assert "no interface named 'VertexOu'" in msg
    assert "Did you mean 'VertexOut'?" in msg
    assert 'demo:31' in msg


def test_table_resolve_unknown_name_non_strict_logs_and_returns_none():
    table = InterfaceTable()
    diagnostics = Diagnostics(strict=False)
    assert table.resolve('Missing', None, diagnostics) is None


def test_table_resolve_known_name_returns_decl():
    table = InterfaceTable()
    decl = _decl('VertexOut', 'demo')
    table.add(decl)
    assert table.resolve('VertexOut', None, Diagnostics()) is decl


def test_table_merged_with_combines_both():
    a = InterfaceTable()
    a.add(_decl('A', 'demo'))
    b = InterfaceTable()
    b.add(_decl('B', 'other'))
    merged = a.merged_with(b)
    assert {d.name for d in merged} == {'A', 'B'}
    assert len(merged) == 2


def test_table_merged_with_prefers_own_entry_on_name_clash():
    a = InterfaceTable()
    a.add(_decl('Shared', 'demo', line=1))
    b = InterfaceTable()
    b.add(_decl('Shared', 'other', line=99))
    merged = a.merged_with(b)
    assert merged.resolve('Shared', None, Diagnostics()).module == 'demo'


def test_table_merge_does_not_mutate_originals():
    a = InterfaceTable()
    a.add(_decl('A', 'demo'))
    b = InterfaceTable()
    b.add(_decl('B', 'other'))
    a.merged_with(b)
    assert 'B' not in a
    assert 'A' not in b
