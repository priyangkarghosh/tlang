# -------------------------------------------------------------
# @file          shader_processor.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Creates all Kernel/Program objects
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import regex as re
from dataclasses import dataclass
from typing import Any

from tlang.attribute_manager import AttributeManager
from tlang.function_manager import FunctionDef, FunctionManager
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_stages import ShaderStage
from tlang.shader_utils import *

# '\n'.join([f"{{ % extends '{sn}' % }}" for sn in attr.args])

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
        self.funcs = FunctionManager.extract_funcs(self.name, self.src, self.src_map)
        self.glob_attrs = AttributeManager.process_attrs(self.name, self.src_map, self.funcs)
        
        # create module
        self.module: dict[int, ShaderSourceLine] = {}
        self._process_function_attrs()
        self._process_global_attrs()
        self._create_module()
    
    # is basically whats exported to other files in the project
    # -> this includes all common code (excluding kernels/shader defs)
    # -> also includes buffers and shared memory and stuff
    def _create_module(self) -> None:
        # create the module
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
                    continue

    def _process_function_attrs(self) -> None:
        for func in self.funcs.items:
            # settings-dict lookup
            stage_settings: dict[str, Any] | None = None
            match func.stage:
                case ShaderStage.FRAG: stage_settings = FRAG_DEFAULTS.copy()
                case ShaderStage.GEOM: stage_settings = GEOM_DEFAULTS.copy()
                case ShaderStage.TESC: stage_settings = TESC_DEFAULTS.copy()
                case ShaderStage.TESE: stage_settings = TESE_DEFAULTS.copy()
            
            # process the passed functions
            for attr in func.attrs:
                # stage settings specific handling
                if stage_settings:
                    if attr.name == (func.stage.value if func.stage else ''):
                        for k, v in attr.kwargs.items():   # iterate over key-value pairs
                            if (k := ALIAS.get(k, k)) not in stage_settings:
                                raise ValueError(
                                    f"Unknown modifier '{k}' for stage '{func.stage}' "
                                    f"in [{attr.name}] on function '{func.name}'"
                                )
                            stage_settings[k] = ALIAS.get(v, v)
                    
                    elif (simp_attr := SIMPLE_ATTR.get(attr.name)):
                        stage_key, setting_key, kind, *fixed = simp_attr
                        # skip if this attr is for a different stage
                        if (func.stage and func.stage.value) != stage_key:
                            continue

                        match kind:
                            case 'flag': stage_settings[setting_key] = True
                            case 'alias': stage_settings[setting_key] = fixed[0]
                            case 'value':
                                val = attr.args[0] if attr.args else next(iter(attr.kwargs.values()))
                                stage_settings[setting_key] = ALIAS.get(val, val)
                        continue
                
                # other processes
                match attr.name:
                    case _: continue
            
            # emit layout for the func
            if stage_settings: self._emit_layout(func, stage_settings)

    def _emit_layout(self, func: FunctionDef, settings: dict[str, str]) -> None:
        def _tokens(table):
            parts = []
            for key, fn in table:
                tok = fn(settings[key])
                if tok is None: continue
                if isinstance(tok, tuple): parts.extend(tok)
                else: parts.append(tok)
            return parts

        match func.stage:
            case ShaderStage.FRAG:
                if in_parts := _tokens(FRAG_EMIT_IN): 
                    func.config.append(f"layout({', '.join(in_parts)}) in;")
            case ShaderStage.GEOM:
                if in_parts := _tokens(GEOM_EMIT_IN): 
                    func.config.append(f"layout({', '.join(in_parts)}) in;")
                if out_parts := _tokens(GEOM_EMIT_OUT): 
                    func.config.append(f"layout({', '.join(out_parts)}) out;")
            case ShaderStage.TESC:
                if out_parts := _tokens(TESC_EMIT_OUT):
                    func.config.append(f"layout({', '.join(out_parts)}) out;")
            case ShaderStage.TESE:
                if in_parts := _tokens(TESE_EMIT_IN): 
                    func.config.append(f"layout({', '.join(in_parts)}) in;")
