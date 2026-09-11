# -------------------------------------------------------------
# @file          dead_code.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Dead-code elimination for generated GLSL: strips SSBO/uniform blocks and whole
#                functions unreachable from `main`, plus the diagnostic for a call that resolves
#                to no definition at all. All three share one reachability analysis, walked over
#                masked text via `glsl_scan`.
# @license       MIT
# -------------------------------------------------------------

import logging
from collections import defaultdict
import regex as re

from tlang.compiler.binding_registry import BLOCK_PATTERN
from tlang.compiler.glsl_scan import mask, match_brace, split_top_level, statement_field_names
from tlang.frontend.function_manager import CONTROL_KEYWORDS, FUNC_PATTERN

logger = logging.getLogger(__name__)

# Optional instance name (and array suffix) between a block's '}' and its ';'.
INSTANCE_TAIL_PATTERN = re.compile(r"\s*(\w+)?(?:\s*\[[^\]]*\])?\s*;")

# An identifier immediately followed by '(' -- a call site, a function header, OR a builtin/
# type-constructor invocation (`vec4(...)`, `atomicAdd(...)`). Builtins are never keys in the
# function map this module builds, so they fall out of consideration on their own -- nothing
# here special-cases them.
CALL_SITE_PATTERN = re.compile(r'\b([A-Za-z_]\w*)\s*\(')


def remove_dead_blocks(src: str, keywords: tuple[str, ...] = ('buffer', 'uniform')) -> str:
    """Strip blocks of each kind in `keywords` whose fields are never referenced elsewhere in `src`.

    A best-effort identifier scan with no notion of scope, biased toward keeping when
    uncertain (a wrongly-kept block costs a binding slot; a wrongly-removed one is a
    compile error). An instance-named block is searched for by its instance name. Must
    run before `allocate_artifact` so bindings are only spent on survivors.
    """
    masked = mask(src)
    removable: list[tuple[int, int]] = []

    for keyword in keywords:
        pattern = BLOCK_PATTERN[keyword]
        for lm in pattern.finditer(masked):
            block_name = lm.group(3)
            open_brace = lm.end() - 1

            if (close_brace := match_brace(masked, open_brace)) is None:
                logger.warning("DCE: unbalanced braces in %s block '%s' -- keeping.", keyword, block_name)
                continue

            if not (tail := INSTANCE_TAIL_PATTERN.match(masked, close_brace + 1)):
                logger.warning("DCE: %s block '%s' has no terminating ';' -- keeping.", keyword, block_name)
                continue

            instance_name, block_end = tail.group(1), tail.end()
            body = masked[open_brace + 1:close_brace]
            field_names = [n for stmt in split_top_level(body)
                           for n in statement_field_names(stmt)]

            # an instance-named block is only ever referenced through the
            # instance (`inst.field`), never the bare field names
            search_names = [instance_name] if instance_name else field_names
            if not search_names:
                logger.warning("DCE: %s block '%s' has no field/instance name to check -- keeping.", keyword, block_name)
                continue

            rest = masked[:lm.start()] + masked[block_end:]
            if any(re.search(r'\b' + re.escape(n) + r'\b', rest) for n in search_names):
                continue  # at least one name is referenced elsewhere -- keep

            removable.append((lm.start(), block_end))

    if not removable: return src
    removable.sort()

    # Replace each removed span with blank lines (not a deletion) so #line-based
    # diagnostics further down the module stay line-accurate.
    out, cursor = [], 0
    for start, end in removable:
        out.append(src[cursor:start])
        out.append('\n' * src.count('\n', start, end))
        cursor = end
    out.append(src[cursor:])
    return ''.join(out)


def _function_spans(masked: str) -> list[tuple[str, int, int, int, int]] | None:
    """Every top-level function definition in `masked`, in source order.

    Each entry is `(name, def_start, def_end, body_start, body_end)` where
    `[def_start, def_end)` spans the whole definition (header + body, for
    blanking) and `[body_start, body_end)` spans just the body (for a callee
    scan that must not mistake a function's own header for a call to itself).

    Reuses `FunctionManager.FUNC_PATTERN`/`CONTROL_KEYWORDS` so a multi-line
    parameter list (its `[^\\)]*` spans newlines) and an `else if (` false
    match are handled identically to real function extraction. Returns
    `None` -- "do nothing, bias toward keeping" -- on an unbalanced brace,
    the same policy `remove_dead_blocks` applies per-block.
    """
    spans: list[tuple[str, int, int, int, int]] = []
    pos = 0
    while (m := FUNC_PATTERN.search(masked, pos)):
        def_start, header_end = m.span()
        name = m.group('name')
        ret_type = m.group('ret_type').strip()
        ret_last_word = ret_type.rsplit(None, 1)[-1] if ret_type else ''

        if ret_last_word in CONTROL_KEYWORDS or name in CONTROL_KEYWORDS:
            pos = def_start + 1
            continue

        open_brace = header_end - 1
        if (close_brace := match_brace(masked, open_brace)) is None:
            logger.warning("DFE: unbalanced braces in function '%s' -- keeping everything.", name)
            return None

        spans.append((name, def_start, close_brace + 1, open_brace + 1, close_brace))
        pos = close_brace + 1

    return spans


def _analyze_reachability(
    src: str,
) -> tuple[list[tuple[str, int, int, int, int]], set[str], set[str]] | None:
    """Walks the call graph of `src` from `main`, on masked text.

    Returns `(spans, reachable_names, unresolved_calls)`:
      - `spans`: every top-level function definition, as `_function_spans` returns.
      - `reachable_names`: function names reachable from `main`, keyed by NAME (an
        overload group is one node -- if any overload is called, the name is reachable
        and every body sharing it is kept; a textual walk cannot resolve overloads, and
        over-inclusion is the safe direction).
      - `unresolved_calls`: names called from `main`, from a name in `reachable_names`,
        or from module-scope code (outside any function -- a global initializer counts
        as a root, same as `main`), that resolve to no definition anywhere in `src`.
        Builtins/type-constructors end up here and are simply never looked up further.

    Returns `None` -- do nothing -- when the text can't be safely analyzed (unbalanced
    braces) or has no `main` to root the walk at.
    """
    masked = mask(src)
    if (spans := _function_spans(masked)) is None: return None

    by_name: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for name, def_start, def_end, body_start, body_end in spans:
        by_name[name].append((def_start, def_end, body_start, body_end))

    if 'main' not in by_name:
        logger.warning("DFE: no 'main' function found in generated unit -- keeping everything.")
        return None

    graph: dict[str, set[str]] = {}
    unresolved: dict[str, set[str]] = {}
    for name, occurrences in by_name.items():
        callees: set[str] = set()
        stray: set[str] = set()
        for _, _, body_start, body_end in occurrences:
            for call in CALL_SITE_PATTERN.finditer(masked, body_start, body_end):
                target = call.group(1)
                (callees if target in by_name else stray).add(target)
        graph[name] = callees
        unresolved[name] = stray

    module_scope_text, cursor = [], 0
    for _, def_start, def_end, _, _ in spans:
        module_scope_text.append(masked[cursor:def_start])
        cursor = def_end
    module_scope_text.append(masked[cursor:])
    module_calls = {m.group(1) for m in CALL_SITE_PATTERN.finditer(''.join(module_scope_text))}

    roots = {'main'} | (module_calls & by_name.keys())
    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n in reachable: continue
        reachable.add(n)
        stack.extend(graph.get(n, ()))

    reachable_unresolved = set(module_calls - by_name.keys())
    for n in reachable:
        reachable_unresolved |= unresolved.get(n, set())

    return spans, reachable, reachable_unresolved


def remove_dead_functions(src: str) -> str:
    """Blanks every top-level function definition in `src` unreachable from `main`.

    Must run before `remove_dead_blocks` (and before `allocate_artifact`) so the block
    DCE sees only text `main` can actually reach -- otherwise an `[export()]`ed helper
    that this particular entry point never calls still keeps whatever SSBO/UBO blocks
    it touches alive. Same bias as `remove_dead_blocks`: keep when uncertain, since a
    wrongly-removed function is a loud compile error, never silent wrongness.

    Blanks with `'\\n' * newline-count`, exactly like `remove_dead_blocks`, so `#line`
    directives further down the unit stay line-accurate.
    """
    if (analysis := _analyze_reachability(src)) is None: return src
    spans, reachable, _ = analysis

    removable = [(def_start, def_end) for name, def_start, def_end, _, _ in spans if name not in reachable]
    if not removable: return src
    removable.sort()

    out, cursor = [], 0
    for start, end in removable:
        out.append(src[cursor:start])
        out.append('\n' * src.count('\n', start, end))
        cursor = end
    out.append(src[cursor:])
    return ''.join(out)


def find_missing_export_calls(src: str) -> set[str]:
    """Names called from `main`-reachable code (or module scope) in `src` that resolve
    to no function definition anywhere in `src` itself.

    `Shader._build` must call this before `remove_dead_functions` touches the same
    `src`: this diagnostic exists to name the fix for a helper that was never emitted
    at all, and once DFE runs the reachable call sites it needs are the only evidence
    that a call happened -- checking after would mean re-deriving exactly what DFE just
    threw away, for no benefit.

    A name that comes back here is either a builtin/type-constructor (not this
    diagnostic's concern) or a real problem for the caller to classify against whatever
    it knows about the project's module-scope functions (same file, `[include]` closure).
    """
    if (analysis := _analyze_reachability(src)) is None: return set()
    _, _, reachable_unresolved = analysis
    return reachable_unresolved
