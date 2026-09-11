# -------------------------------------------------------------
# @file          glsl_text.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Text-scanning helpers shared by binding_registry.py and dead_code.py: masking,
#                brace matching, and statement/declarator splitting over generated GLSL.
#
#                NOT the same masking as `shader_utils.mask_comments_and_strings`, and the two
#                must not be merged: `mask` here additionally blanks preprocessor lines, because
#                it runs on GENERATED GLSL full of `#line`/`#version`/`#extension` directives.
#                The frontend's version deliberately leaves preprocessor lines alone, because
#                source `.tlang` uses `#name<args>` as attribute syntax.
# @license       MIT
# -------------------------------------------------------------

import regex as re

# Comments, strings and preprocessor lines, blanked before scanning so none of
# them count as a use. Blanked character-for-character to preserve offsets.
MASK_PATTERN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|^[ \t]*#[^\n]*',
    re.DOTALL | re.MULTILINE,
)

# Qualifier words that prefix a field declarator without naming it.
FIELD_QUALIFIER_WORDS = frozenset({
    "highp", "mediump", "lowp",
    "readonly", "writeonly", "coherent", "volatile", "restrict", "const",
})


def mask(src: str) -> str:
    """Blank comments, string literals, and preprocessor lines in `src`.

    Same length, same newline positions as `src` -- offsets computed
    against the result stay valid against the original.
    """
    blank = lambda m: ''.join(c if c == '\n' else ' ' for c in m.group(0))
    return MASK_PATTERN.sub(blank, src)


def match_brace(text: str, open_pos: int) -> int | None:
    """Index of the '}' matching the '{' at `open_pos` in `text`, or None if unbalanced."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0: return i
    return None


def split_top_level(body: str) -> list[str]:
    """Split a block body into field statements on top-level ';' only.

    A nested struct's own members (inside its `{ }`) sit at brace
    depth > 0, so e.g. `struct S { vec3 a; float b; } s[];` comes back
    as one statement, not three -- the inner ';'s don't split it.
    """
    stmts, depth, start = [], 0, 0
    for i, c in enumerate(body):
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        elif c == ';' and depth == 0:
            stmts.append(body[start:i])
            start = i + 1
    return stmts


def declarator_names(decl_list: str) -> list[str]:
    """Trailing identifier of each comma-separated declarator in `decl_list`.

    Strips array suffixes, initializers, and leading qualifier/precision words -- so
    `highp uint counts[]` yields `counts`, and `uint a, b[4], c` yields `a`, `b`, `c`.
    """
    names = []
    for segment in decl_list.split(','):
        segment = re.sub(r'\[[^\]]*\]', '', segment).split('=')[0]
        words = [w for w in segment.split() if w not in FIELD_QUALIFIER_WORDS]
        if (ids := re.findall(r'[A-Za-z_]\w*', ' '.join(words))):
            names.append(ids[-1])
    return names


def statement_field_names(stmt: str) -> list[str]:
    """Field identifier(s) declared by one top-level block statement.

    A plain statement is just a declarator list. A nested-struct statement
    (`struct S { ... } s[]`) skips the type body -- only the declarator list after its
    closing brace names an actual block field.
    """
    stmt = stmt.strip()
    if not stmt: return []
    if (brace := stmt.find('{')) == -1:
        return declarator_names(stmt)
    close = match_brace(stmt, brace)
    return declarator_names(stmt[close + 1:]) if close is not None else []
