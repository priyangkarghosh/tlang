# -------------------------------------------------------------
# @file          test_attribute_parsing.py
# @description   GL-free regression tests for AttributeManager /
#                attribute_registry / attribute_handlers: attribute
#                block splitting, typed-arg coercion, stage validation,
#                and the existing attribute syntaxes.
#
#                Every test in this file constructs a `ShaderProcessor`
#                directly -- no GL context involved anywhere.
# -------------------------------------------------------------

import logging

import pytest

from tlang.frontend.attribute_manager import AttributeManager
from tlang.errors import SourceLocation, TlangAttributeError, TlangSyntaxError
from tlang.frontend.attribute_registry import Diagnostics
from tlang.compiler.shader_processor import ShaderProcessor
from tlang.shader_stages import ShaderStage


# ---------------------------------------------------------------------------
# AttributeManager.split_attr_block -- low-level splitting
# ---------------------------------------------------------------------------

def test_parameterless_attribute_parses():
    """Regression: the attribute regex used to REQUIRE parentheses, so a
    bare attribute like `export` (no args at all) silently no-oped --
    split_attr_block('export') returned [] instead of one Attribute. This
    dropped ~20 real attributes ([export], [unroll], [triangles], ...)."""
    attrs = AttributeManager.split_attr_block('export')
    assert len(attrs) == 1
    assert attrs[0].name == 'export'
    assert attrs[0].args == []
    assert attrs[0].kwargs == {}


def test_split_attr_block_multiple_attrs_with_args():
    attrs = AttributeManager.split_attr_block("shader('compute'), numthreads(256, 1, 1)")
    assert [a.name for a in attrs] == ['shader', 'numthreads']
    assert attrs[0].args == ['compute']
    assert attrs[1].args == ['256', '1', '1']


# ---------------------------------------------------------------------------
# Typed-arg coercion: the 'false' / '0' truthy-string class of bug
# ---------------------------------------------------------------------------

def test_frag_early_tests_false_emits_no_qualifier():
    """Regression: the string 'false' was truthy, so [frag(early_tests=false)]
    silently turned the feature ON instead of off."""
    src = (
        "[shader('fragment')]\n"
        "[frag(early_tests=false)]\n"
        "void fs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert not any('early_fragment_tests' in c for c in fn.config)


def test_frag_early_tests_true_emits_qualifier():
    src = (
        "[shader('fragment')]\n"
        "[frag(early_tests=true)]\n"
        "void fs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert any('early_fragment_tests' in c for c in fn.config)


def test_tesc_vertices_zero_emits_nothing():
    """Regression: the string '0' was truthy, so [tesc(vertices=0)] emitted
    `layout(vertices = 0) out;` -- invalid GLSL (must be > 0 if present)."""
    src = (
        "[shader('tess_control')]\n"
        "[tesc(vertices=0)]\n"
        "void tc_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert not any('vertices' in c for c in fn.config)


def test_tesc_vertices_three_emits_layout():
    src = (
        "[shader('tess_control')]\n"
        "[tesc(vertices=3)]\n"
        "void tc_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert any('layout(vertices = 3) out;' == c for c in fn.config)


def test_triangles_adjacency_on_geometry_emits_triangles_adjacency():
    """Regression: [triangles_adjacency] must emit the literal token
    'triangles_adjacency', not be conflated with plain [triangles]."""
    src = (
        "[shader('geometry')]\n"
        "[triangles_adjacency]\n"
        "void gs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    joined = '\n'.join(fn.config)
    assert 'triangles_adjacency' in joined
    # must not be the plain 'triangles' token on its own
    assert 'layout(triangles) in;' not in joined


# ---------------------------------------------------------------------------
# Stage validation / unknown-name diagnostics
# ---------------------------------------------------------------------------

def test_wrong_stage_use_raises_naming_valid_stages():
    """[quads] is a tess-eval-only marker; using it on a geometry function
    must raise, and the message must name the valid stage(s)."""
    src = (
        "[shader('geometry')]\n"
        "[quads]\n"
        "void gs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'quads' in msg
    assert 'tese' in msg
    assert 'geom' in msg


def test_unknown_attribute_suggests_close_match():
    """Regression target: an unknown attribute name should raise with a
    did-you-mean suggestion, not an opaque failure."""
    src = (
        "[shader('compute')]\n"
        "[numthredz(64, 1, 1)]\n"
        "void cs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'numthredz' in msg
    assert 'numthreads' in msg


def test_duplicate_numthreads_raises_conflict():
    """Two conflicting [numthreads(...)] on the same function must raise
    instead of the second one silently overwriting the first's config."""
    src = (
        "[shader('compute')]\n"
        "[numthreads(64, 1, 1)]\n"
        "[numthreads(32, 1, 1)]\n"
        "void cs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'local_size_x' in msg
    assert 'twice' in msg or 'onflict' in msg


def test_quads_then_triangles_last_writer_wins():
    """[quads][triangles] on the SAME tess-eval function: the same settings
    slot is overwritten in source order, so the last marker wins."""
    src = (
        "[shader('tess_eval')]\n"
        "[quads]\n"
        "[triangles]\n"
        "void te_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    joined = '\n'.join(fn.config)
    # tess-eval also emits its spacing/order defaults, so assert on the
    # domain token: `triangles` (written last) must win over `quads`
    assert 'triangles' in joined and 'quads' not in joined
    assert 'quads' not in joined


# ---------------------------------------------------------------------------
# Existing syntaxes must keep working
# ---------------------------------------------------------------------------

def test_shader_attribute_sets_stage():
    src = "[shader('vertex')]\nvoid vs_main() {\n    int x = 0;\n}\n"
    proc = ShaderProcessor('t', src)
    assert proc.funcs.items[0].stage == ShaderStage.VERT


def test_shader_and_numthreads_combo_on_one_line():
    src = "[shader('compute'), numthreads(256, 1, 1)]\nvoid cs_main() {\n    int x = 0;\n}\n"
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert fn.stage == ShaderStage.COMP
    assert any('local_size_x = 256' in c for c in fn.config)


def test_multiline_resourceblock():
    src = (
        "[shader('fragment')]\n"
        "[resourceblock(\n"
        "    out vec4 fragColor;\n"
        "    uniform float time;\n"
        ")]\n"
        "void fs_main() {\n"
        "    fragColor = vec4(time);\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    joined = '\n'.join(fn.config)
    assert 'out vec4 fragColor;' in joined
    assert 'uniform float time;' in joined


def test_alt_attr_syntax_numthreads():
    """The `#name<args>` alternate syntax must behave identically to the
    bracket form."""
    src = "[shader('compute')]\n#numthreads<64, 1, 1>\nvoid cs_main() {\n    int x = 0;\n}\n"
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    assert any('local_size_x = 64' in c for c in fn.config)


def test_funcbody_pragma_unroll_emits_pragma():
    src = (
        "[shader('compute')]\n[numthreads(1, 1, 1)]\n"
        "void cs_main() {\n"
        "    [unroll]\n"
        "    for (int i = 0; i < 4; i++) {}\n"
        "}\n"
    )
    proc = ShaderProcessor('t', src)
    fn = proc.funcs.items[0]
    body_text = ''.join(line.data for _, line in sorted(fn.line_body.items()))
    assert '#pragma unroll' in body_text
    assert '[unroll]' not in body_text


def test_define_with_brackets_untouched_by_attribute_scanner():
    """Regression: `#define IDX(i) data[i]` used to be eaten by the
    bracket-attribute scanner, which treated the `[i]` inside it as an
    attribute block. Dispatch is now keyed off each line's first
    non-whitespace character ('[' vs '#'), so a '#'-led preprocessor line
    never reaches the '['-block scanner at all, and -- since it also
    doesn't match the single-line `#name<args>` alt-attribute syntax -- is
    passed through completely unchanged."""
    src = "#define IDX(i) data[i]\nvoid main_fn() {\n    int x = 0;\n}\n"
    proc = ShaderProcessor('t', src)
    out = ''.join(line.data for _, line in sorted(proc.module.items()))
    assert '#define IDX(i) data[i]' in out


# --- malformed attribute blocks are diagnosed, not silently dropped ---

@pytest.mark.parametrize("block,fragment", [
    ("shader('compute'",                     "missing ')'"),
    ("shader('compute'))",                   "unexpected ')'"),
    ("shader('compute', numthreads(1,1,1)",  "missing ')'"),
])
def test_unbalanced_attribute_block_raises(block, fragment):
    """A malformed block used to yield [] silently, so the build 'succeeded'
    with no kernels even under strict=True."""
    with pytest.raises(TlangSyntaxError) as exc:
        AttributeManager.split_attr_block(block)
    assert fragment in str(exc.value)
    assert block.strip() in str(exc.value)


@pytest.mark.parametrize("block", ["!!!", "shader('a') junk", "foo bar"])
def test_unparseable_attribute_raises(block):
    with pytest.raises(TlangSyntaxError):
        AttributeManager.split_attr_block(block)


def test_digit_prefixed_name_parses_then_fails_as_unknown():
    """`1shader` is a syntactically valid name, so it parses here and is caught
    downstream by the registry, which offers a did-you-mean."""
    attrs = AttributeManager.split_attr_block("1shader('compute')")
    assert [a.name for a in attrs] == ["1shader"]


@pytest.mark.parametrize("block,count", [
    ("export", 1),
    ("shader('compute')", 1),
    ("shader('compute'), numthreads(64, 1, 1)", 2),
    ("numthreads(BLOCK_SIZE, 1, 1)", 1),
    ("program('d', vert='vs', frag='fs')", 1),
    ("resourceblock(\n  out vec4 c;\n)", 1),
    ("extend!(int64)", 1),
    ("", 0),
])
def test_valid_attribute_blocks_still_parse(block, count):
    assert len(AttributeManager.split_attr_block(block)) == count


def test_malformed_block_reports_location_and_honours_strict(caplog):
    """With a Diagnostics collector, strict=False logs instead of raising."""
    loc = SourceLocation("demo", 7)
    with pytest.raises(TlangSyntaxError) as exc:
        AttributeManager.split_attr_block("shader('x'", loc, Diagnostics(strict=True))
    assert "demo:7" in str(exc.value)

    with caplog.at_level(logging.ERROR):
        out = AttributeManager.split_attr_block("shader('x'", loc, Diagnostics(strict=False))
    assert out == []
    assert "demo:7" in caplog.text
