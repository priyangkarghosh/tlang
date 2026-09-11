# -------------------------------------------------------------
# @file          function_manager.py
# @author        Priyangkar Ghosh
# @created       2025-06-13
# @description   Extracts all functions (including kernels) from the shader source.
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import bisect
from tlang.frontend.attribute import Attribute
from tlang.errors import SourceLocation, TlangSyntaxError
from tlang.frontend.interface_registry import InterfaceTable
from tlang.shader_stages import ShaderStage
from tlang.shader_utils import mask_comments_and_strings
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple
import regex as re

from tlang.shader_source_line import ShaderSourceLine


class InterfaceRef(NamedTuple):
    """One [uses(...)] reference, resolved later against the merged table."""
    name: str
    direction: str
    location: SourceLocation


@dataclass(frozen=True, slots=True)
class TopLevelDecl:
    """One raw (non-`[buffer]`/`[uniforms]`-attribute) top-level declaration --
    a `const` or a `buffer`/`uniform { ... }` block -- found outside every
    function body. `ShaderManager` cross-references these by name across a
    module's include closure the same way it already does for functions."""
    kind: str    # 'const' | 'buffer' | 'uniform'
    name: str
    line: int    # 1-based


# matches function declarations with opening brace
FUNC_PATTERN = re.compile(r'''
    ^[ \t]*                          # leading indent only -- \s* would span into a
                                     # masked-out comment block above the function
    (?P<ret_type>\w[\w\s\*]*)\s+     # return type
    (?P<name>\w+)\s*                 # function name
    \((?P<params>[^\)]*)\)\s*        # parameter list
    \{                               # opening brace
''', re.MULTILINE | re.VERBOSE)

# GLSL control-flow keywords, so `else if (cond) {` isn't mistaken for a function header.
CONTROL_KEYWORDS: frozenset[str] = frozenset({'if', 'for', 'while', 'switch', 'else', 'do'})

# raw `buffer Name { ... }` / `uniform Name { ... }` blocks -- anchored on the required
# keyword literal so this never regresses build time the way an unanchored pattern did.
# `[buffer(std430)]`/`[uniforms(...)]` attribute markers never match: those are followed
# by '(', not a name then '{'.
_BLOCK_RE = re.compile(r'\b(buffer|uniform)\s+([A-Za-z_]\w*)\s*\{')

# start of a top-level `const` statement; the statement's end is found separately by
# scanning forward for a bracket-depth-zero ';' (`_stmt_end`), since a const initializer
# can itself contain '(', '[' or '{' (e.g. `float[](0.0, 1.0)`).
_CONST_RE = re.compile(r'\bconst\b')

# qualifiers recognised on a function parameter; anything else in the leading word run is
# treated as (part of) the parameter's type, not a qualifier.
_PARAM_QUALIFIER_WORDS = frozenset({
    'in', 'out', 'inout', 'const', 'highp', 'mediump', 'lowp',
    'precise', 'coherent', 'volatile', 'restrict', 'readonly', 'writeonly', 'patch',
})


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep` at bracket-depth zero only, so a nested `(...)`/`[...]`/`{...}`
    -- a default-array initializer, a function call inside a const expression -- never
    gets split on its own internal separators."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c in '([{': depth += 1
        elif c in ')]}': depth -= 1
        elif c == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def _stmt_end(mask: str, start: int) -> int | None:
    """Index of the bracket-depth-zero ';' at/after `start`, or None if there isn't one."""
    depth = 0
    for i in range(start, len(mask)):
        c = mask[i]
        if c in '([{': depth += 1
        elif c in ')]}': depth -= 1
        elif c == ';' and depth <= 0: return i
    return None


def param_type_signature(params: str) -> tuple[str, ...] | None:
    """Normalised parameter *type* list for one function's parameter text -- names and
    whitespace dropped, so two overloads (different types, legal GLSL) read as different
    keys while a genuine duplicate (same types, maybe different parameter names) collides.

    tlang has no type system: this is textual, not resolved. Returns None when a
    parameter's shape can't be read with confidence, so a duplicate-declaration check
    built on top of this can stay silent rather than guess -- see utils.tlang's three
    `hash` overloads, the false-positive class this guards against.
    """
    params = params.strip()
    if not params: return ()

    sigs: list[str] = []
    for part in _split_top_level(params, ','):
        part = part.strip()
        if not part: return None

        consumed = 0
        while (qm := re.match(r'(\w+)\s+', part[consumed:])):
            if qm.group(1) not in _PARAM_QUALIFIER_WORDS: break
            consumed += qm.end()

        if not (tm := re.match(r'([A-Za-z_]\w*)', part[consumed:])):
            return None
        type_name = tm.group(1)
        tail = part[consumed + tm.end():].strip()

        array = ''
        if tail:
            if (nm := re.match(r'[A-Za-z_]\w*\s*(\[[^\]]*\])?$', tail)):
                array = nm.group(1) or ''
            elif (am := re.match(r'(\[[^\]]*\])$', tail)):
                array = am.group(1)
            else:
                return None  # unrecognised shape -- stay quiet rather than guess

        sigs.append(re.sub(r'\s+', '', type_name + array))
    return tuple(sigs)

@dataclass(eq=False)
class FunctionDef:
    name: str
    return_type: str
    params: str
    stage: ShaderStage | None

    body: str
    line_start: int
    line_end: int
    line_body: dict[int, ShaderSourceLine]

    exported: bool = False

    # FunctionDefs this function depends on via [link(...)]; their bodies are emitted ahead of
    # this function's own in its stage source.
    links: list['FunctionDef'] = field(
        default_factory=list
    )
    attrs: list[Attribute] = field(
        default_factory=list
    )
    config: list[str] = field(
        default_factory=list
    )
    helpers: list[str] = field(
        default_factory=list
    )

    # [uses(...)] references recorded at attach time, resolved later
    iface_refs: list[InterfaceRef] = field(
        default_factory=list
    )

@dataclass
class FunctionList:
    items: list[FunctionDef]
    # GLSL permits overloads, so a name may map to more than one FunctionDef; a lookup that
    # needs exactly one must check the list length and diagnose ambiguity itself.
    keyed_items: dict[str, list[FunctionDef]]

    # [varyings]/[uniforms]/[buffer] declarations for this module only (ShaderManager merges
    # in transitive dependencies).
    interfaces: InterfaceTable = field(default_factory=InterfaceTable)

    # raw (non-attribute) top-level `const`/`buffer`/`uniform` block declarations found
    # outside every function body, for this module only -- see TopLevelDecl.
    decls: list[TopLevelDecl] = field(default_factory=list)

    def __post_init__(self):
        self.starts: list[int] = [fn.line_start for fn in self.items]

    def find_next(self, line: int) -> FunctionDef | None:
        i = bisect.bisect_right(self.starts, line)
        return self.items[i] if i < len(self.items) else None

    def find_within(self, line: int) -> FunctionDef | None:
        i = bisect.bisect_right(self.starts, line) - 1
        if 0 <= i < len(self.items):
            if (fn := self.items[i]).line_start <= line <= fn.line_end:
                return fn
        return None
    
    def is_within(self, line: int) -> bool:
        return 0 <= bisect.bisect_right(self.starts, line) - 1 < len(self.items)

class FunctionManager:
    @staticmethod
    def extract_funcs(shader_name: str, src: str, src_map: dict[int, ShaderSourceLine]) -> FunctionList:
        funcs: list[FunctionDef] = []
        keyed_funcs: dict[str, list[FunctionDef]] = {}

        # Mask used only to locate headers and match braces, so comments/strings/`if (x) {`
        # can't be mistaken for a function. Bodies are always sliced from the original `src`.
        mask = mask_comments_and_strings(src)

        search_pos: int = 0
        while match := FUNC_PATTERN.search(mask, search_pos):
            func_start, header_end = match.span()  # span of the function header only

            name = match.group("name")
            ret_type = match.group("ret_type").strip()
            params = match.group("params").strip()

            # `else if (cond) {` reads as a two-word header ("else" + "if") -- reject it.
            ret_last_word = ret_type.rsplit(None, 1)[-1] if ret_type else ''
            if ret_last_word in CONTROL_KEYWORDS or name in CONTROL_KEYWORDS:
                search_pos = func_start + 1
                continue

            line_start = src[:func_start].count('\n') + 1

            # Brace matching runs on the mask, not the original source.
            def match_brace():
                brace_depth = 1
                for i, chr in enumerate(mask[header_end:], start=header_end):
                    if chr == '{': brace_depth += 1
                    elif chr == '}': brace_depth -= 1
                    if not brace_depth: return i
                raise TlangSyntaxError("Unmatched brace in function", SourceLocation(shader_name, line_start))
            func_end = match_brace() + 1

            line_end = src[:func_end].count('\n') + 1
            line_body = {
                index: src_map.pop(index)
                for index in range(line_start, line_end + 1) if index in src_map
            }

            logger.info("Found function '%s' in %s", name, shader_name)

            fdef = FunctionDef(
                name=name,
                return_type=ret_type,
                params=params,
                stage=None,

                body=src[func_start:func_end],
                line_start=line_start,
                line_end=line_end,
                line_body=line_body,
            )
            funcs.append(fdef)

            # GLSL allows overloads -- keep every definition instead of clobbering earlier ones.
            overloads = keyed_funcs.setdefault(name, [])
            if overloads: logger.info("Function '%s' has %d overloads in %s", name, len(overloads) + 1, shader_name)
            overloads.append(fdef)

            search_pos = func_end

        result = FunctionList(funcs, keyed_funcs)
        result.decls = FunctionManager._extract_top_level_decls(src, mask, result)
        return result

    @staticmethod
    def _extract_top_level_decls(src: str, mask: str, funcs: FunctionList) -> list[TopLevelDecl]:
        """Raw `const`/`buffer`/`uniform` block declarations outside every function body.
        Runs on the same mask already used for header matching, so a lookalike inside a
        comment or string never surfaces here either. `funcs.is_within` (built from the
        function spans just extracted) is what keeps a local inside a function body --
        e.g. a `const` loop bound, or a parameter's own `const` qualifier -- from being
        mistaken for a module-scope declaration.
        """
        decls: list[TopLevelDecl] = []

        for m in _BLOCK_RE.finditer(mask):
            line = src[:m.start()].count('\n') + 1
            if funcs.is_within(line): continue
            decls.append(TopLevelDecl(kind=m.group(1), name=m.group(2), line=line))

        pos = 0
        while (m := _CONST_RE.search(mask, pos)):
            pos = m.end()
            line = src[:m.start()].count('\n') + 1
            if funcs.is_within(line): continue

            if (end := _stmt_end(mask, m.end())) is None: continue
            stmt = mask[m.end():end]

            if not (tm := re.match(r'\s*([A-Za-z_]\w*)\s+', stmt)): continue
            decl_list_offset = m.end() + tm.end()

            running = 0
            for part in _split_top_level(stmt[tm.end():], ','):
                if (nm := re.match(r'\s*([A-Za-z_]\w*)', part)):
                    name_offset = decl_list_offset + running + nm.start(1)
                    decl_line = src[:name_offset].count('\n') + 1
                    decls.append(TopLevelDecl(kind='const', name=nm.group(1), line=decl_line))
                running += len(part) + 1  # +1 accounts for the comma `_split_top_level` consumed

        return decls
