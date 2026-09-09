# -------------------------------------------------------------
# @file          test_function_extraction.py
# @description   GL-free regression tests for FunctionManager.extract_funcs:
#                comment-blindness, control-flow rejection, GLSL overloads.
# -------------------------------------------------------------

from tlang.frontend.function_manager import FunctionManager
from tlang.shader_source_line import ShaderSourceLine


def _extract(src: str):
    lines = src.splitlines(keepends=True)
    src_map = {i: ShaderSourceLine('t', line) for i, line in enumerate(lines, start=1)}
    return FunctionManager.extract_funcs('t', src, src_map)


def test_comment_blind_fake_function_not_extracted_and_real_one_is_byte_correct():
    """Regression: a `/* fake func foo(int x) { */` comment, and a
    `// ... } {` line, used to be able to desync brace-depth counting (or
    be matched as a header outright) if extraction wasn't comment-aware.
    A real function whose body itself contains braces inside comments must
    still extract with byte-correct body text and line numbers."""
    src = (
        "/* void fake(int x) {\n"
        "   this is not a real function */\n"
        "// stray closer } {\n"
        "void real(int x) {\n"
        "    /* a brace in a comment { */\n"
        "    int y = x + 1; // trailing comment with a brace }\n"
        "}\n"
    )
    funcs = _extract(src)

    names = [f.name for f in funcs.items]
    assert names == ['real'], f"expected only 'real' extracted, got {names}"

    real = funcs.items[0]
    assert real.line_start == 4
    assert real.line_end == 7

    # body is sliced from the ORIGINAL (unmasked) source, so comment text
    # -- braces and all -- must appear in it verbatim
    func_start = src.index('void real')
    close_brace_pos = src.index('\n}\n', func_start) + 1  # index of the '}' char itself
    expected_body = src[func_start:close_brace_pos + 1]
    assert real.body == expected_body
    assert '/* a brace in a comment { */' in real.body
    assert '// trailing comment with a brace }' in real.body


def test_control_flow_not_extracted_as_function():
    """`if (...) {` / `for (...) {` / `while (...) {` must never be
    extracted as functions. Bare single-keyword forms don't even match
    FUNC_PATTERN's two-identifier-token header shape, but `else if (cond) {`
    reads as a two-word header (ret_type='else', name='if') -- that's
    exactly what CONTROL_KEYWORDS exists to reject."""
    src = (
        "if (x > 0) {\n"
        "    doThing();\n"
        "}\n"
        "\n"
        "for (int i = 0; i < 10; i++) {\n"
        "    doThing();\n"
        "}\n"
        "\n"
        "while (running) {\n"
        "    doThing();\n"
        "}\n"
        "\n"
        "if (a) {\n"
        "    doThing();\n"
        "} else if (b) {\n"
        "    doOther();\n"
        "}\n"
        "\n"
        "void real_fn(int x) {\n"
        "    doThing();\n"
        "}\n"
    )
    funcs = _extract(src)
    names = [f.name for f in funcs.items]
    assert names == ['real_fn']


def test_glsl_overloads_both_survive():
    src = (
        "uint add(uint x, uint y) {\n"
        "    return x + y;\n"
        "}\n"
        "\n"
        "float add(float x, float y) {\n"
        "    return x + y;\n"
        "}\n"
    )
    funcs = _extract(src)
    assert len(funcs.items) == 2
    assert len(funcs.keyed_items['add']) == 2
    assert {f.return_type for f in funcs.keyed_items['add']} == {'uint', 'float'}
