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

        return FunctionList(funcs, keyed_funcs)
