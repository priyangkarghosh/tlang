# -------------------------------------------------------------
# @file          test_attribute_pathological.py
# @description   Regression tests for T14: ATTR_PATTERN's old `(?R)`-based
#                recursion was catastrophically ambiguous, so a `[...]`
#                attribute block whose argument text contained a comment
#                with a parenthesis could hang the preprocessor at 100% CPU
#                forever. Covers the linear replacement pattern, comment
#                masking in split_attr_block / parse_args, and the timeout
#                backstop -- plus the existing attribute syntaxes, which
#                must keep parsing exactly as before.
# -------------------------------------------------------------

import time

import pytest

from tlang.frontend.attribute_manager import AttributeManager
from tlang.errors import TlangSyntaxError


# ---------------------------------------------------------------------------
# The exact T14 repro
# ---------------------------------------------------------------------------

T14_BODY = (
    "resourceblock(uniform bool stabilizing;\n"
    "// FU-3: the substep length (Sec4.3 Eq. 10), used for the Eq. 10 clamp\n"
    "uniform float deltaTime;\n"
    ")"
)


def test_t14_body_parses_fast_and_keeps_comment_text():
    """The real T14 body used to hang forever (catastrophic backtracking on
    the balanced '(Sec4.3 Eq. 10)' inside the comment). It must now parse
    quickly, and the comment text must survive verbatim in raw_args since
    resourceblock's handler reads attr.raw_args directly and comments must
    not be stripped from emitted GLSL."""
    t0 = time.perf_counter()
    attrs = AttributeManager.split_attr_block(T14_BODY)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.name == 'resourceblock'
    assert 'uniform bool stabilizing;' in attr.raw_args
    assert 'uniform float deltaTime;' in attr.raw_args
    # the comment, parenthesised clause included, must still be there verbatim
    assert '// FU-3: the substep length (Sec4.3 Eq. 10), used for the Eq. 10 clamp' in attr.raw_args


def test_t14_body_survives_through_shader_processor():
    """Full pipeline: the T14 body reaches ShaderProcessor without hanging,
    and its comment text (parenthesised clause included) survives into the
    emitted resourceblock declaration -- comments must not be stripped from
    what gets emitted."""
    from tlang.compiler.shader_processor import ShaderProcessor

    src = (
        "[shader('compute')]\n"
        "[numthreads(1, 1, 1)]\n"
        f"[{T14_BODY}]\n"
        "void cs_main() {\n"
        "    int x = 0;\n"
        "}\n"
    )
    t0 = time.perf_counter()
    proc = ShaderProcessor('t', src)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0

    fn = proc.funcs.items[0]
    joined = '\n'.join(fn.config)
    assert 'uniform bool stabilizing;' in joined
    assert 'uniform float deltaTime;' in joined
    assert 'Sec4.3 Eq. 10' in joined


# ---------------------------------------------------------------------------
# Real time-bound regression: a real clock, not just a correctness check
# ---------------------------------------------------------------------------

def test_unbalanced_paren_after_prose_is_fast_not_exponential():
    """On the old `(?R)`-based pattern, ~40 chars of prose before a lone,
    never-closed '(' is roughly 2^40 units of backtracking work and never
    returns in practice. The fix makes this linear: it must complete in well
    under a second even on a loaded machine.

    This calls match_attr (ATTR_PATTERN) directly rather than going through
    split_attr_block: split_attr_block's own char-counting pre-check would
    reject an overall-unbalanced string in O(n) before ever reaching the
    regex, which would make this test measure the wrong thing."""
    prose = "the quick brown fox jumps over lazy dg"  # ~40 chars
    assert len(prose) >= 38
    body = f"resourceblock({prose}("

    t0 = time.perf_counter()
    result = AttributeManager.match_attr(body)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"unbalanced-paren parse took {elapsed:.3f}s -- looks exponential again"
    assert result is None  # unbalanced -- no match, not a hang


def test_unbalanced_paren_raises_clear_syntax_error_quickly():
    """An unbalanced paren must fail loudly and fast, not hang."""
    t0 = time.perf_counter()
    with pytest.raises(TlangSyntaxError) as exc:
        AttributeManager.split_attr_block("resourceblock(uniform float x;")
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0
    assert "missing ')'" in str(exc.value)


# ---------------------------------------------------------------------------
# Comment masking: commas/parens inside comments must not steer the parse
# ---------------------------------------------------------------------------

def test_comment_with_comma_inside_attribute_args_does_not_mis_split():
    """A comma inside a comment that lives inside an attribute's argument
    list must not be treated as an argument separator."""
    args, kwargs = AttributeManager.parse_args("64, /* uses, a comma */ 1, 1")
    assert args == ['64', '1', '1']
    assert kwargs == {}


def test_comment_with_comma_between_attributes_does_not_split_block():
    """A comma inside a comment that lives between two top-level attributes
    in the same [...] block must not be mistaken for the attribute
    separator (it sits inside resourceblock's own parens, so the real
    top-level split must still land after resourceblock's closing paren)."""
    block = (
        "resourceblock(\n"
        "    // note: uses a comma, right here\n"
        "    uniform float x;\n"
        "), export"
    )
    attrs = AttributeManager.split_attr_block(block)
    assert [a.name for a in attrs] == ['resourceblock', 'export']
    assert 'uses a comma, right here' in attrs[0].raw_args


def test_balanced_parens_in_resourceblock_prose_now_match():
    """Regression: balanced parens in prose inside [resourceblock(...)] used
    to fail to match at all (on top of hanging) -- (?R) could not close a
    parenthetical that didn't start with an identifier immediately before
    it. A balanced paren in a comment is now legal."""
    body = "resourceblock(uniform float x; // see (the notes) for details\n)"
    attrs = AttributeManager.split_attr_block(body)
    assert len(attrs) == 1
    assert '(the notes)' in attrs[0].raw_args


# ---------------------------------------------------------------------------
# Existing syntaxes must still parse exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block,count", [
    ("export", 1),
    ("shader('compute')", 1),
    ("numthreads(64,1,1)", 1),
    ("program('x', vert='a', frag='b')", 1),
])
def test_existing_forms_still_parse(block, count):
    attrs = AttributeManager.split_attr_block(block)
    assert len(attrs) == count


def test_numthreads_with_nested_call_still_parses():
    attrs = AttributeManager.split_attr_block("numthreads(BLOCK(1), 1, 1)")
    assert len(attrs) == 1
    assert attrs[0].name == 'numthreads'
    assert attrs[0].args == ['BLOCK(1)', '1', '1']


def test_multiline_attribute_block_still_parses():
    block = (
        "resourceblock(\n"
        "    out vec4 fragColor;\n"
        "    uniform float time;\n"
        ")"
    )
    attrs = AttributeManager.split_attr_block(block)
    assert len(attrs) == 1
    assert 'out vec4 fragColor;' in attrs[0].raw_args
    assert 'uniform float time;' in attrs[0].raw_args


def test_program_named_args_with_quotes_still_parse():
    attrs = AttributeManager.split_attr_block("program('x', vert='a', frag='b')")
    assert len(attrs) == 1
    attr = attrs[0]
    assert attr.args == ['x']
    assert attr.kwargs == {'vert': 'a', 'frag': 'b'}
