# -------------------------------------------------------------
# @file          function_manager.py
# @author        Priyangkar Ghosh
# @created       2025-06-13
# @description   Extracts all functions (including kernels) from
#                the shader src
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import bisect
from tlang.attribute import Attribute
from tlang.shader_stages import ShaderStage
from dataclasses import dataclass, field
from enum import Enum
import regex as re

from tlang.shader_source_line import ShaderSourceLine

# matches function declarations with opening brace
FUNC_PATTERN = re.compile(r'''
    ^\s*
    (?P<ret_type>\w[\w\s\*]*)\s+     # return type
    (?P<name>\w+)\s*                 # function name
    \((?P<params>[^\)]*)\)\s*        # parameter list
    \{                               # opening brace
''', re.MULTILINE | re.VERBOSE)

@dataclass
class FunctionDef:
    name: str
    return_type: str
    params: str
    stage: ShaderStage | None

    line_start: int
    line_end: int
    line_body: dict[int, ShaderSourceLine]

    attrs: list[Attribute] = field(
        default_factory=list
    )
    config: list[str] = field(
        default_factory=list
    )

@dataclass
class FunctionList:
    items: list[FunctionDef]

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

        # search for function declarations
        # -> loop while there are matches
        search_pos: int = 0
        while match := FUNC_PATTERN.search(src, search_pos):
            # this is only the span of the function HEADER
            # -> again, this pattern ONLY matches the HEADER
            func_start, header_end = match.span()

            # get function parameters from the match
            name = match.group("name")
            ret_type = match.group("ret_type").strip()
            params = match.group("params").strip()
            line_start = src[:func_start].count('\n') + 1

            # find the full func body using brace matching
            def match_brace():
                brace_depth = 1
                for i, chr in enumerate(src[header_end:], start=header_end):
                    if chr == '{': brace_depth += 1
                    elif chr == '}': brace_depth -= 1
                    if not brace_depth: return i
                raise SyntaxError("Unmatched brace in function")
            func_end = match_brace() + 1
            
            # get the full function body
            line_end = src[:func_end].count('\n') + 1
            line_body = {
                index: src_map.pop(index) 
                for index in range(line_start, line_end + 1) if index in src_map
            }

            # add to func list
            logger.info("Found function '%s' in %s", name, shader_name)
            funcs.append(FunctionDef(
                name=name,
                return_type=ret_type,
                params=params,
                stage=None,

                line_start=line_start,
                line_end=line_end,
                line_body=line_body,
            ))

            # move past the current function
            search_pos = func_end

        # return the stripped src and a list of the functions
        return FunctionList(funcs)