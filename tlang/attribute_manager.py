import logging
logger = logging.getLogger(__name__)

import re
import bisect
from dataclasses import dataclass
from typing import Callable

from tlang.attribute import Attribute
from tlang.attribute_handlers import *
from tlang.function_manager import FunctionList
from tlang.shader_source_line import ShaderSourceLine


BLOCK_PATTERN = re.compile(r'\[([^\]]+)\]')
ATTR_PATTERN = re.compile(r'(?P<name>\w+!?)(?:\((?P<args>[^)]*)\))?')
ALT_ATTR_PATTERN = re.compile(r'#(?P<name>\w+!?)\s*<\s*(?P<args>.*?)\s*>')
# ALT_ATTR_PATTERN ONLY SUPPORTS 1 LINE PER PATTERN AND SINGLE LINE DECLARATIONS

ARG_PATTERN = re.compile(r"""
    \s*
    (?:@?(?P<key>\w+)\s*=\s*)?
    (?P<value>
        '[^']*'
        |"[^"]*"
        |[^,]+
    )
    \s*(?:,|$)
""", re.VERBOSE)

# class to extract anything in the form []
class AttributeManager:
    @classmethod
    def match_attr(cls, attr_str: str) -> Attribute | None:
        if (m := ATTR_PATTERN.fullmatch(attr_str)):
            return Attribute(
                m.group('name'), 
                *cls.parse_args(m.group('args') or '')
            )
        return None

    @staticmethod
    def parse_args(arg_str: str) -> tuple[list[str], dict[str, str]]:
        args: list[str] = []
        kwargs: dict[str, str] = {}
        for m in ARG_PATTERN.finditer(arg_str):
            val = m.group("value").strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1].strip()

            if len(val):
                key = m.group("key")
                if key: kwargs[key] = val
                else: args.append(val)
        return args, kwargs

    # takes an attr line in the form "[shader('compute'), numthreads(64, 1, 1)]"
    # -> splits it into its individual attribute components
    # -> i.e. ["shader('compute')", "numthreads(64, 1, 1)"]
    @classmethod
    def split_attr_block(cls, block_str: str) -> list[Attribute]:
        buffer: list[str] = []
        attrs: list[Attribute] = []
        stack_depth: int = 0

        def flush_buffer():
            attr = cls.match_attr(''.join(buffer).strip())
            if attr: attrs.append(attr)
            buffer.clear()      

        for c in block_str:
            # update the stack depth
            if c == '(': stack_depth += 1
            elif c == ')': stack_depth -= 1

            # check if this is the end of an attr declaration
            if c == ',' and not stack_depth: flush_buffer()
            else: buffer.append(c)
        
        # flush any remaining characters
        if buffer: flush_buffer()

        # if there is an unclosed attr at the end, throw an error
        if stack_depth != 0: logger.error("Error stacking attributes with block: %s", block_str)
        return attrs

    @classmethod
    def process_attrs(cls, shader_name: str, src_map: dict[int, ShaderSourceLine], funcs: FunctionList) -> list[Attribute]:
        cls._attach_func_ctx_attrs(shader_name, funcs)
        return cls._attach_glob_ctx_attrs(shader_name, src_map, funcs)
    
    @classmethod
    def _attach_func_ctx_attrs(cls, shader_name: str, funcs: FunctionList):
        for func in funcs.items:
            line_indices, i = sorted(func.line_body), 0
            while i < len(line_indices):
                idx_out, repl = cls._process_attr_line(
                    shader_name, line_indices[i], func.line_body, "function", FUNC_CTX_ATTR_MAP
                )
                func.line_body[idx_out].data = repl
                i = bisect.bisect_right(line_indices, idx_out)

    @classmethod
    def _attach_glob_ctx_attrs(cls, shader_name: str, src_map: dict[int, ShaderSourceLine], funcs: FunctionList) -> list[Attribute]:
        glob_attachments: list[Attribute] = []
        line_indices, i = sorted(src_map), 0
        while i < len(line_indices):
            idx_out, repl = cls._process_attr_line(
                shader_name, line_indices[i], src_map, "global", GLOB_CTX_ATTR_MAP,
                funcs=funcs, glob_attachments=glob_attachments
            )
            src_map[idx_out].data = repl
            i = bisect.bisect_right(line_indices, idx_out)
        return glob_attachments

    @classmethod
    def _process_attr_line(
        cls,
        shader_name: str,
        index: int,
        map: dict[int, ShaderSourceLine],
        line_type: str,
        attr_map: dict[str, Callable[..., str]],
        **kwargs
    ) -> tuple[int, str]:
        # early exit to first check whether this is an actual attribute
        # -> i.e. the line must actually start with a [ or a #
        if not re.match(r'^\s*(?:\[|#)', init := map[index].data):
            return index, init

        # first try processing the line using the block pattern
        # -> this is attributes in the style [attr(val)]
        # -> have to check if it spans multiple lines
        # -> thus, use brace match to find the entire attribute
        start_index = index
        line_str, depth = '', 0
        while True:
            line_str += (line := map[index].data)
            map[index].data = '\n'
            depth += line.count('[') - line.count(']')
            if depth <= 0 or (index + 1) not in map: break
            index += 1

        # handle an attribute
        def handle_attr(attr: Attribute, attr_type: str) -> str:
            handler = attr_map.get(name := attr.name or '')
            if handler: return handler(attr, index=start_index, **kwargs)
            else: logger.warning(f"Unhandled {line_type} {attr_type} attribute '%s' in %s", name, shader_name)
            return ''
        
        out_line, last_match = '', 0
        for match in BLOCK_PATTERN.finditer(line_str):
            # handle the attribute(s)
            for attr in cls.split_attr_block(match.group(1).strip()): 
                out_line += handle_attr(attr, 'block')
            last_match = match.end()
        
        # try processing using the alt attr pattern
        if match := ALT_ATTR_PATTERN.search(line_str, last_match):
            attr = Attribute(match.group('name'), *cls.parse_args(match.group('args')))
            out_line += handle_attr(attr, 'direct')
            last_match = match.end()
        
        # add any code left over at the end of all the attributes
        # -> this allows for comments at the end, but no other code (also includes the \n)
        # -> replace the line
        out_line += line_str[last_match:]
        return index, out_line