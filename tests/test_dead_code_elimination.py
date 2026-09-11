# -------------------------------------------------------------
# @file          test_dead_code_elimination.py
# @description   GL-free regression tests for
#                dead_code.remove_dead_blocks -- this is the
#                dangerous pass: it used to DELETE LIVE SSBO blocks.
#                Every KEPT case below pins a false-positive-removal bug;
#                every REMOVED case pins that genuinely dead code still
#                gets cleaned up.
# -------------------------------------------------------------

from tlang.compiler.dead_code import remove_dead_blocks


def _dce(src: str) -> str:
    return remove_dead_blocks(src)


# ---------------------------------------------------------------------------
# KEPT -- these must survive
# ---------------------------------------------------------------------------

def test_live_block_with_highp_uint_field_is_kept():
    src = (
        "layout(std430) buffer Live {\n"
        "    highp uint counts[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    counts[0] = 1u;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'buffer Live' in out
    assert 'counts' in out


def test_live_block_with_coherent_field_is_kept():
    src = (
        "layout(std430) buffer LiveCoherent {\n"
        "    coherent uint hits[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    hits[0] = 1u;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'buffer LiveCoherent' in out


def test_block_kept_when_only_one_of_several_declarators_is_used():
    """`uint a, b[4], c;` with only `c` referenced elsewhere -- the whole
    block is one unit; any one live field name keeps it all."""
    src = (
        "layout(std430) buffer Live3 {\n"
        "    uint a, b[4], c;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    c = 1u;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'buffer Live3' in out
    assert 'a, b[4], c' in out


def test_instance_named_block_referenced_through_instance_is_kept():
    """`buffer Blk { uint x[]; } inst;` is referenced in code as `inst.x`,
    not by the bare field name `x` -- DCE must search for the INSTANCE
    name, not the field name, or it will wrongly conclude the block is
    unused."""
    src = (
        "layout(std430) buffer Blk {\n"
        "    uint x[];\n"
        "} inst;\n"
        "\n"
        "void main() {\n"
        "    inst.x[0] = 1u;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'buffer Blk' in out
    assert 'inst' in out


# ---------------------------------------------------------------------------
# REMOVED -- genuinely dead blocks must still be cleaned up
# ---------------------------------------------------------------------------

def test_block_mentioned_only_in_a_comment_is_removed():
    src = (
        "layout(std430) buffer Dead {\n"
        "    uint deadfield[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    // deadfield is unused elsewhere -- this comment shouldn't count as a use\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dce(src)
    # the block declaration must be gone; `deadfield` itself legitimately
    # survives inside the unrelated comment in main(), which DCE must not touch
    assert 'buffer Dead' not in out
    assert 'uint deadfield[]' not in out
    # line count must be preserved (removed span -> blank lines, not deletion)
    assert out.count('\n') == src.count('\n')


def test_block_whose_field_name_collides_with_line_directive_is_removed():
    """A `#line N "quoted.module"` directive the pipeline itself injects
    must not poison this pass -- a field name that happens to appear only
    inside such a directive's quoted argument must not count as a use."""
    src = (
        "layout(std430) buffer Dead2 {\n"
        "    uint particles[];\n"
        "};\n"
        "#line 1 \"vfx.particles\"\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'Dead2' not in out
    # the block goes; the `#line 1 "vfx.particles"` directive must stay
    assert 'buffer' not in out
    assert '#line 1 "vfx.particles"' in out
    assert out.count('\n') == src.count('\n')


def test_unused_readonly_buffer_is_removed():
    src = (
        "layout(std430) readonly buffer DeadRO {\n"
        "    uint ro_field[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'DeadRO' not in out
    assert 'ro_field' not in out


def test_unused_block_with_nested_struct_removed_cleanly():
    """A nested-struct block, unused, must be removed with no dangling
    `};` fragment left behind (the whole span -- including the trailing
    instance declarator -- is replaced by blank lines, never split)."""
    src = (
        "layout(std430) buffer DeadStruct {\n"
        "    struct Inner { float a; float b; } items[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dce(src)
    assert 'DeadStruct' not in out
    assert 'Inner' not in out
    assert 'items' not in out
    # no dangling fragment of the removed block's closing syntax
    assert '};' not in out
    assert out.count('\n') == src.count('\n')
