from dataclasses import dataclass
import re
import logging
from typing import Callable

from attribute import Attribute
from function_manager import FunctionList
from shader_source_line import ShaderSourceLine


logger = logging.getLogger(__name__)

BLOCK_PATTERN = re.compile(r'\[([^\]]+)\]')
ATTR_PATTERN = re.compile(r'(?P<name>\w+)(?:\((?P<args>[^)]*)\))?') # i.e. [shader('test')]
ALT_ATTR_PATTERN = re.compile(r'#(?P<name>\w+)\s*<\s*(?P<args>.*?)\s*>') # i.e. #shader('test')
# ALT_ATTR_PATTERN ONLY SUPPORTS 1 LINE PER PATTERN

ARG_PATTERN = re.compile(
    r"""
    \s*
    (?:@?(?P<key>\w+)\s*=\s*)?
    (?P<value>        
        '[^']*'
        |"[^"]*"
        |[^'",\s\]]+
    )
    \s*(?:,|$)
    """, 
    re.VERBOSE
)

class AttributeHandlers:
    @staticmethod
    def unroll(attr: Attribute, **kwargs) -> str:
        return '#pragma unroll'
    
    @staticmethod
    def program(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return f"//<<PROGRAM DEC '{attr.name}'>>//\n"
    
    @staticmethod
    def entry_point(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(funcs := kwargs.get('funcs'), FunctionList):
            raise TypeError("Expected 'funcs' to be a FunctionList")
        if not isinstance(index := kwargs.get('index'), int):
            raise TypeError("Expected 'index' to be an int")

        # add attribute to the function definition
        if fn := funcs.find_next(index): fn.attrs.append(attr)
        return f"//<<ATTR '{attr.name}'>>//\n"
    
    @staticmethod
    def dependency(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return '\n'.join([f"{{% include '{sn}' %}}" for sn in attr.args])
    
    @staticmethod
    def extension(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return f"//<<EXTENSION '{attr.name}', ARGS: {attr.args}>>//\n"

GLOB_CTX_ATTR_MAP: dict[str, Callable[..., str]] = {
    'shader': AttributeHandlers.entry_point,
    'numthreads': AttributeHandlers.entry_point,
    'program': AttributeHandlers.program,
    'include': AttributeHandlers.dependency,
    'extend': AttributeHandlers.extension,
    'extend!': AttributeHandlers.extension,
    'require': AttributeHandlers.extension,
}

FUNC_CTX_ATTR_MAP: dict[str, Callable[..., str]] = {
    'unroll': AttributeHandlers.unroll
}
    
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
                val = val[1:-1]

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
    def process_attrs(cls, src_map: dict[int, ShaderSourceLine], funcs: FunctionList) -> list[Attribute]:
        cls._attach_func_ctx_attrs(funcs)
        return cls._attach_glob_ctx_attrs(src_map, funcs)
    
    @classmethod
    def _attach_func_ctx_attrs(cls, funcs: FunctionList):
        for func in funcs.items:
            for index, ssl in func.line_body.items():
                func.line_body[index].data = cls._process_attr_line(
                    ssl.data, 'function', FUNC_CTX_ATTR_MAP
                )
    
    @classmethod
    def _attach_glob_ctx_attrs(cls, src_map: dict[int, ShaderSourceLine], funcs: FunctionList) -> list[Attribute]:
        glob_attachments: list[Attribute] = []
        
        # if a match exists in src it MUST also exist in stripped_src or in a func definition
        # -> this automatically skips any code that is within a function
        for index, ssl in src_map.items():
            # restore line number if there's been a change
            src_map[index].data = cls._process_attr_line(
                ssl.data, 'global', GLOB_CTX_ATTR_MAP, index=index,
                line=ssl.data, funcs=funcs, glob_attachments=glob_attachments
            )

        # return stripped src code and attachments
        return glob_attachments

    @classmethod
    def _process_attr_line(
        cls,
        line_str: str,
        line_type: str,
        attr_map: dict[str, Callable[..., str]],
        **kwargs
    ) -> str:
        # first try processing the line using the block pattern
        # -> this is attributes in the style [attr(val)]
        out_line: str = ''
        last_match: int = 0

        # handle an attribute
        def handle_attr(attr: Attribute, attr_type: str) -> str:
            handler = attr_map.get(name := attr.name or '')
            if handler: return handler(attr, **kwargs)
            else: logger.warning(f"Unhandled {line_type} {attr_type} attribute: %s", name)
            return ''

        for match in BLOCK_PATTERN.finditer(line_str):
            # handle the attribute(s)
            for attr in cls.split_attr_block(match.group(1).strip()): 
                out_line += handle_attr(attr, 'block')
            last_match = match.end()
        
        # now try processing using the alt attr pattern
        if match := ALT_ATTR_PATTERN.search(line_str, last_match):
            attr = Attribute(match.group('name'), *cls.parse_args(match.group('args')))
            out_line += handle_attr(attr, 'direct')
            last_match = match.end()
        
        # add any code left over at the end of all the attributes
        # -> this allows for comments at the end, but no other code (also includes the \n)
        # -> replace the line
        out_line += line_str[last_match:]
        return out_line