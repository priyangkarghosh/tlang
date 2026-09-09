# -------------------------------------------------------------
# @file          test_interface_attributes.py
# @description   GL-free tests for the [varyings]/[uniforms]/[buffer]/
#                [uses(...)] attributes: desugaring, line-slot
#                discipline, and the V1-V6 cross-stage validation rules
#                in ShaderProcessor.resolve_interfaces.
# -------------------------------------------------------------

import pytest

from tlang.compiler.dependency_manager import DependencyManager
from tlang.errors import TlangAttributeError
from tlang.frontend.interface_registry import InterfaceKind, InterfaceTable
from tlang.compiler.shader_manager import ShaderManager
from tlang.compiler.shader_processor import ShaderProcessor


def _processor(name: str, src: str, strict: bool = True) -> ShaderProcessor:
    return ShaderProcessor(name, src, strict=strict)


# ---------------------------------------------------------------------------
# desugaring: [buffer] / [uniforms] write module text; [varyings] blanks
# ---------------------------------------------------------------------------

def test_buffer_desugars_to_ssbo_block_in_module_text():
    src = "[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n"
    proc = _processor('demo', src)
    text = ''.join(line.data for line in proc.module.values())
    assert 'buffer Particles' in text
    assert 'vec4 pos[];' in text
    assert '[buffer' not in text


def test_uniforms_desugars_to_loose_uniform():
    src = "[uniforms]\nstruct Frame { mat4 view; float time; };\n"
    proc = _processor('demo', src)
    text = ''.join(line.data for line in proc.module.values())
    assert 'uniform mat4 view;' in text
    assert 'uniform float time;' in text
    assert 'buffer' not in text


def test_uniforms_std140_desugars_to_ubo_block():
    src = "[uniforms(std140)]\nstruct Frame { mat4 view; };\n"
    proc = _processor('demo', src)
    text = ''.join(line.data for line in proc.module.values())
    assert 'layout(std140) uniform Frame {' in text
    assert 'mat4 view;' in text


def test_varyings_emits_no_module_text():
    src = "[varyings]\nstruct VertexOut { vec3 color; };\n\nuniform float k;\n"
    proc = _processor('demo', src)
    text = ''.join(line.data for line in proc.module.values())
    assert 'color' not in text
    assert 'VertexOut' not in text
    assert 'uniform float k;' in text


def test_declarations_are_recorded_on_the_interface_table():
    src = (
        "[varyings]\nstruct VertexOut { vec3 color; };\n\n"
        "[uniforms]\nstruct Frame { mat4 view; };\n\n"
        "[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n"
    )
    proc = _processor('demo', src)
    assert len(proc.interfaces) == 3
    assert proc.interfaces.resolve('VertexOut', None, proc.diagnostics).kind is InterfaceKind.VARYINGS
    assert proc.interfaces.resolve('Frame', None, proc.diagnostics).kind is InterfaceKind.UNIFORMS
    assert proc.interfaces.resolve('Particles', None, proc.diagnostics).kind is InterfaceKind.BUFFER


# ---------------------------------------------------------------------------
# line-slot discipline: content after the struct keeps its exact line index
# ---------------------------------------------------------------------------

def test_lines_after_a_multiline_struct_are_untouched_and_unshifted():
    src = (
        "[buffer(std430)]\n"          # line 1
        "struct Particles {\n"         # line 2
        "    vec4 pos;\n"              # line 3
        "    vec4 vel;\n"              # line 4
        "};\n"                         # line 5
        "\n"                           # line 6
        "// sentinel\n"                # line 7
        "uniform float k;\n"           # line 8
    )
    proc = ShaderProcessor('demo', src)
    assert proc.src_map[6].data == '\n'
    assert proc.src_map[7].data == '// sentinel\n'
    assert proc.src_map[8].data == 'uniform float k;\n'
    # every slot the struct consumed after its first is blanked
    assert proc.src_map[3].data == '\n'
    assert proc.src_map[4].data == '\n'
    assert proc.src_map[5].data == '\n'
    # the first slot carries the whole desugared (multi-line) block
    assert proc.src_map[2].data.count('\n') > 1


def test_single_line_struct_collapses_into_one_slot():
    src = "[uniforms]\nstruct Frame { mat4 view; float time; };\n\nuniform float k;\n"
    proc = ShaderProcessor('demo', src)
    assert 'uniform mat4 view;' in proc.src_map[2].data
    assert 'uniform float time;' in proc.src_map[2].data
    assert proc.src_map[4].data == 'uniform float k;\n'


# ---------------------------------------------------------------------------
# correctness trap: a declaration attribute must never bind a struct past
# an intervening function -- must raise naming what was actually found
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]', '[buffer]'])
def test_declaration_attribute_does_not_skip_past_a_function(attr):
    src = (
        f"{attr}\n"
        "void vs_main() {\n"
        "    int x = 0;\n"
        "}\n"
        "\n"
        "struct SomethingElse { vec3 a; };\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('demo', src)
    msg = str(exc_info.value)
    assert 'void vs_main() {' in msg
    assert 'SomethingElse' not in msg


@pytest.mark.parametrize('attr', ['[varyings]', '[uniforms]', '[buffer]'])
def test_declaration_attribute_with_nothing_after_it_raises(attr):
    src = f"{attr}\n"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('demo', src)
    assert 'end of file' in str(exc_info.value)


def test_blank_lines_and_comments_before_struct_are_skipped():
    src = "[varyings]\n\n// a comment\n\nstruct VertexOut { vec3 color; };\n"
    proc = ShaderProcessor('demo', src)
    decl = proc.interfaces.resolve('VertexOut', None, proc.diagnostics)
    assert decl.line == 5


# ---------------------------------------------------------------------------
# [uses(...)] recording
# ---------------------------------------------------------------------------

def test_uses_records_a_reference_without_emitting_anything_yet():
    src = (
        "[varyings]\nstruct VertexOut { vec3 color; };\n\n"
        "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"
        "void vs_main() { color = vec3(0.0); }\n"
    )
    proc = ShaderProcessor('demo', src)
    vs = proc.funcs.keyed_items['vs_main'][0]
    assert len(vs.iface_refs) == 1
    assert vs.iface_refs[0].name == 'VertexOut'
    assert vs.iface_refs[0].direction == 'out'
    assert not any('layout' in c for c in vs.config)


# ---------------------------------------------------------------------------
# V1-V6: cross-stage validation in ShaderProcessor.resolve_interfaces
# ---------------------------------------------------------------------------

FUNCS_SRC = (
    "[varyings]\nstruct VertexOut {{ vec3 color; }};\n\n"
    "[shader('vertex')]\n{vert_attrs}\n"
    "void vs_main() {{ color = vec3(0.0); }}\n\n"
    "[shader('fragment')]\n{frag_attrs}\n"
    "void fs_main() {{ }}\n"
)


def _build_and_resolve(vert_attrs: str, frag_attrs: str, strict: bool = True) -> ShaderProcessor:
    src = FUNCS_SRC.format(vert_attrs=vert_attrs, frag_attrs=frag_attrs)
    proc = ShaderProcessor('demo', src, strict=strict)
    proc.resolve_interfaces(proc.interfaces)
    return proc


def test_v1_unknown_uses_name_raises_with_suggestion_and_line():
    with pytest.raises(TlangAttributeError) as exc_info:
        _build_and_resolve("[uses(VertexOut, dir='out')]", "[uses('VertexOu', dir='in')]")
    msg = str(exc_info.value)
    assert 'VertexOu' in msg
    assert 'VertexOut' in msg
    assert 'demo:9' in msg  # fs_main's [uses(...)] line


def test_v2_uses_on_a_non_varyings_interface_raises():
    src = (
        "[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n\n"
        "[shader('vertex')]\n[uses(Particles, dir='out')]\n"
        "void vs_main() { }\n"
    )
    proc = ShaderProcessor('demo', src)
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_interfaces(proc.interfaces)
    msg = str(exc_info.value)
    assert 'Particles' in msg
    assert 'buffer' in msg


def test_v3_two_uses_same_direction_on_one_function_raises():
    src = (
        "[varyings]\nstruct A { vec3 a; };\n"
        "[varyings]\nstruct B { vec3 b; };\n\n"
        "[shader('vertex')]\n[uses(A, dir='out')]\n[uses(B, dir='out')]\n"
        "void vs_main() { }\n"
    )
    proc = ShaderProcessor('demo', src)
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_interfaces(proc.interfaces)
    msg = str(exc_info.value)
    assert 'A' in msg and 'B' in msg
    assert 'one interface per direction' in msg


def test_v5_unmeasurable_location_span_raises():
    src = (
        "[varyings]\nstruct VertexOut { vec4 pos[]; };\n\n"
        "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"
        "void vs_main() { }\n"
    )
    proc = ShaderProcessor('demo', src)
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_interfaces(proc.interfaces)
    assert 'measurable location span' in str(exc_info.value)


def test_v5_locations_false_bypasses_measurement():
    src = (
        "[varyings(locations=false)]\nstruct VertexOut { vec4 pos[]; };\n\n"
        "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"
        "void vs_main() { }\n"
    )
    proc = ShaderProcessor('demo', src)
    proc.resolve_interfaces(proc.interfaces)  # must not raise
    vs = proc.funcs.keyed_items['vs_main'][0]
    assert any('out vec4 pos[];' in c for c in vs.config)
    assert not any('layout' in c for c in vs.config)


def test_v6_uses_on_compute_stage_raises():
    src = (
        "[varyings]\nstruct VertexOut { vec3 color; };\n\n"
        "[shader('compute')]\n[numthreads(1,1,1)]\n[uses(VertexOut, dir='in')]\n"
        "void cs_main() { }\n"
    )
    proc = ShaderProcessor('demo', src)
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_interfaces(proc.interfaces)
    assert 'compute' in str(exc_info.value)


def test_v4_headline_diagnostic_names_both_sides_and_both_lines():
    """The whole point of this feature: a typo'd [uses(...)] name that
    breaks a program's link must produce ONE build-time error naming both
    entry points, both interface names, and both source lines -- not the
    driver's unlocated 'GLSL Linker failed'."""
    src = (
        "[varyings]\nstruct VertexOut { vec3 color; };\n\n"        # line 2
        "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"        # line 5
        "void vs_main() { color = vec3(0.0); }\n\n"
        "[shader('fragment')]\n[uses('VertexOu', dir='in')]\n"      # line 9
        "void fs_main() { }\n\n"
        "[program('default', vert='vs_main', frag='fs_main')]\n"
    )
    proc = ShaderProcessor('demo', src)
    with pytest.raises(TlangAttributeError) as exc_info:
        proc.resolve_interfaces(proc.interfaces)
    msg = str(exc_info.value)
    assert 'vs_main' in msg
    assert 'fs_main' in msg
    assert 'VertexOut' in msg
    assert 'VertexOu' in msg
    assert 'demo:2' in msg  # declaration line
    assert 'demo:9' in msg  # the bad [uses(...)] line
    assert 'default' in msg


def test_a_stage_declaring_no_interface_is_not_an_error():
    """The raw-GLSL escape hatch: a function that never [uses(...)]
    anything must not be flagged by V4 even inside a [program(...)]."""
    src = (
        "[shader('vertex')]\nvoid vs_main() { }\n\n"
        "[shader('fragment')]\nvoid fs_main() { }\n\n"
        "[program('default', vert='vs_main', frag='fs_main')]\n"
    )
    proc = ShaderProcessor('demo', src)
    proc.resolve_interfaces(proc.interfaces)  # must not raise


def test_strict_false_downgrades_uses_problems_to_warnings():
    proc = _build_and_resolve(
        "[uses(VertexOut, dir='out')]", "[uses('VertexOu', dir='in')]", strict=False,
    )  # must not raise
    fs = proc.funcs.keyed_items['fs_main'][0]
    assert fs.config == []  # unresolved reference never got emitted


# ---------------------------------------------------------------------------
# cross-module signature conflict, detected by ShaderManager at merge time
# (InterfaceTable.merged_with can hold only one decl per name, so this has
# to be caught while both are still visible -- see its own docstring)
# ---------------------------------------------------------------------------

def test_cross_module_same_name_different_signature_raises(make_shader_dir):
    d = make_shader_dir({
        'a.tlang': "[varyings]\nstruct VertexOut { vec3 color; };\n",
        'b.tlang': "[varyings]\nstruct VertexOut { vec4 color; };\n",
        'demo.tlang': (
            "[include(a)]\n[include(b)]\n\n"
            "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"
            "void vs_main() { }\n"
        ),
    })
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=None, version='430 core', dir=str(d), strict=True)
    msg = str(exc_info.value)
    assert 'VertexOut' in msg
    assert 'a:2' in msg and 'b:2' in msg
    assert 'vec3 color' in msg and 'vec4 color' in msg


def test_cross_module_identical_signature_is_not_a_conflict(make_shader_dir):
    d = make_shader_dir({
        'a.tlang': "[varyings]\nstruct VertexOut { vec3 color; };\n",
        'b.tlang': "[varyings]\nstruct VertexOut { vec3 color; };\n",
        'demo.tlang': (
            "[include(a)]\n[include(b)]\n\n"
            "[shader('vertex')]\n[uses(VertexOut, dir='out')]\n"
            "void vs_main() { }\n"
        ),
    })
    ShaderManager(ctx=None, version='430 core', dir=str(d), strict=True)  # must not raise
