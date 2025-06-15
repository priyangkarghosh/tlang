from dataclasses import dataclass
import re

from attribute_manager import AttributeManager
from function_manager import FunctionDef, FunctionManager
from shader_source_line import ShaderSourceLine
from shader_stages import ShaderStage

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

class ShaderProcessor:
    def __init__(self, name: str, src: str) -> None:
        self.name, self.src = name, src
        self.src_map: dict[int, ShaderSourceLine] = {
            index: ShaderSourceLine(name, line) # store line info and virtual context (file)
            for index, line in enumerate(src.splitlines(keepends=True), start=1)
        }

        self.dps: set[str] = set()
        self.ext: set[str] = set(['GL_KHR_vulkan_glsl : require'])
        self.programs: dict[str, dict[ShaderStage, str | None]] = {}
        
        # processing steps
        self.funcs = FunctionManager.extract_funcs(self.src, self.src_map)
        self.glob_attrs = AttributeManager.process_attrs(self.src_map, self.funcs)
        
        # create module
        self.module: dict[int, ShaderSourceLine] = {}
        self._process_global_attrs()
        self._process_function_attrs()
        self._create_module()
    
    # is basically whats exported to other files in the project
    # -> this includes all common code (excluding kernels/shader defs)
    # -> also includes buffers and shared memory and stuff
    def _create_module(self) -> None:
        self.module = self.src_map.copy()
        for func in self.funcs.items:
            if not func.stage: self.module.update(func.line_body)

    def _process_global_attrs(self) -> None:
        for attr in self.glob_attrs:
            match attr.name:
                case 'program':
                    if not attr.args: raise ValueError("Missing program name in [program(...)]")
                    if (name := attr.args[0]) in self.programs: raise ValueError(f"Program '{name}' is already defined")
                    self.programs[attr.args[0]] = ShaderStage.gather_stages(attr.kwargs)

                case 'include':
                    self.dps.update(attr.args)

                case 'extend':
                    # resolve groups and flatten
                    for token in attr.args: 
                        mapped = EXTENSION_GROUPS.get(token)
                        if mapped: self.ext.update([ext + ' : enable' for ext in mapped])
                        else: self.ext.add(token + ' : enable')
                
                case 'extend!' | 'require':
                    # resolve groups and flatten
                    for token in attr.args: 
                        mapped = EXTENSION_GROUPS.get(token)
                        if mapped: self.ext.update([ext + ' : require' for ext in mapped])
                        else: self.ext.add(token + ' : require')

                case _:
                    break

    def _process_function_attrs(self) -> None:
        for func in self.funcs.items:
            for attr in func.attrs:
                match attr.name:
                    case 'shader':
                        func.stage = ShaderStage.from_token(attr.args[0])

                    case 'numthreads':
                        # get num threads
                        if attr.args:
                            numthreads = [1, 1, 1]
                            for i in range(min(len(attr.args), 3)):
                                numthreads[i] = int(attr.args[i])
                        else:
                            numthreads = (
                                int(attr.kwargs.get('x', 1)),
                                int(attr.kwargs.get('y', 1)),
                                int(attr.kwargs.get('z', 1)),
                            )
                        
                        # add layout str to config
                        func.config.append(
                            f'layout(local_size_x={numthreads[0]}, local_size_y={numthreads[1]}, local_size_z={numthreads[2]}) in;'
                        )

                    case _:
                        break
        