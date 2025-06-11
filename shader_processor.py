from dataclasses import dataclass
import re

from attribute_manager import AttributeManager
from function_manager import FunctionManager
from shader_source_line import ShaderSourceLine

EXTENSION_GROUPS = {
    'subgroup': [
        'GL_KHR_shader_subgroup_basic',
        'GL_KHR_shader_subgroup_vote',
        'GL_KHR_shader_subgroup_ballot',
        'GL_KHR_shader_subgroup_arithmetic',
    ],

    'subgroup_all': [
        'GL_KHR_shader_subgroup_basic',
        'GL_KHR_shader_subgroup_vote',
        'GL_KHR_shader_subgroup_ballot',
        'GL_KHR_shader_subgroup_arithmetic',
        'GL_KHR_shader_subgroup_shuffle',
        'GL_KHR_shader_subgroup_shuffle_relative',
        'GL_KHR_shader_subgroup_clustered',
        'GL_KHR_shader_subgroup_quad'
    ],
}

ARG_PATTERN = re.compile(
    r"""
    (?:
        @(?P<key>\w+)\s*=\s*   # @key =
        (?:
            '(?P<val1>[^']*)'  # '@key'='value'
            |"(?P<val2>[^"]*)"
            |(?P<val3>[^'",\s\]]+)
        )
    )
    |
    (?:
        '(?P<pos1>[^']*)'
        |"(?P<pos2>[^"]*)"
        |(?P<pos3>[^'",\s\]]+)
    )
    """,
    re.VERBOSE
)

class ShaderProcessor:
    def __init__(self, name: str, src: str) -> None:
        self.name, self.src = name, src
        self.src_map: dict[int, ShaderSourceLine] = {
            index: ShaderSourceLine(name, line) # store line info and virtual context (file)
            for index, line in enumerate(src.splitlines(keepends=True), start=1)
        }
        
        self.funcs = FunctionManager.extract_funcs(self.src, self.src_map)
        self.glob_attrs = AttributeManager.process_attrs(self.src_map, self.funcs)

        self.ext: set[str] = set()
        self.dps: set[str] = set()
        self.programs: dict[str, dict[str, str | None]] = {}
        self.entries = []
        self.module = {}
    
    @staticmethod
    def parse_args(arg_str: str) -> tuple[list[str], dict[str, str]]:
        def strip(s: str | None):
            s = (s or '').strip()
            return (s[1:-1] if s.startswith(("'", '"')) and s.endswith(("'", '"')) 
                    else s)

        args = []
        kwargs = {}
        for match in ARG_PATTERN.finditer(arg_str):
            if (key := strip(match.group("key"))):
                kwargs[key] = strip(match.group("value"))
            elif (val := strip(match.group("positional"))):
                args.append(val)
        return args, kwargs
    
    # is basically whats exported to other files in the project
    # -> this includes all common code (excluding kernels/shader defs)
    # -> also includes buffers and shared memory and stuff
    def _create_module(self):
        pass

    def _process_global_attrs(self):
        for attr in self.glob_attrs:
            # split the args into tokens
            args, kwargs = self.parse_args(attr.get('args', ''))

            match attr.get('name', ''):
                case 'program':
                    if not args: raise ValueError("Missing program name in [program(...)]")
                    if (name := args[0]) in self.programs: raise ValueError(f"Program '{name}' is already defined")

                    # take the first arg to be the name
                    self.programs[args[0]] = {
                        'vert': kwargs.get('vert') or kwargs.get('vertex'),
                        'frag': kwargs.get('frag') or kwargs.get('fragment'),
                        'geom': kwargs.get('geom') or kwargs.get('geometry'),
                        'tesc': kwargs.get('tesc') or kwargs.get('tess_control'),
                        'tese': kwargs.get('tese') or kwargs.get('tess_eval')
                    }

                case 'include':
                    self.dps.update(args)

                case 'extend':
                    # resolve groups and flatten
                    for token in args: 
                        mapped = EXTENSION_GROUPS.get(token)
                        if mapped: self.ext.update(mapped)
                        else: self.ext.add(token)
                case _:
                    break

    def _process_function_attrs(self):
        for func in self.funcs.items:
            for attr in func.attrs:
                match attr.get('name', ''):
                    case 'shader':
                        break
                    case 'numthreads':
                        break
                    case _:
                        break
        