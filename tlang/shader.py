import logging
logger = logging.getLogger(__name__)

import re
from moderngl import ComputeShader, Context, Program
from tlang.kernel import Kernel
from tlang.shader_processor import ShaderProcessor
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_stages import ShaderStage

class Shader:
    def __init__(self, ctx: Context, name: str, version: str, module: str, processor: ShaderProcessor) -> None:
        self._ctx = ctx
        self._name = name
        self._version = version

        self._kernels: dict[str, Kernel] = {}
        self._programs: dict[str, Program] = {}
        self._build(module, processor)
    
    @property
    def kernels(self) -> dict[str, Kernel]:
        return self._kernels

    @property
    def programs(self) -> dict[str, Program]:
        return self._programs

    def get_kernel(self, name: str) -> Kernel:
        return self._kernels[name]

    def get_program(self, name: str) -> Program:
        return self._programs[name]
    
    @staticmethod
    def _func_header_regex(ret_type: str, name: str) -> re.Pattern:
        ret = r'\s+'.join([re.escape(w) for w in ret_type.split()])
        return re.compile(rf'\b{ret}\s+{re.escape(name)}\s*\(', re.MULTILINE)
    
    def _build(self, module: str, process: ShaderProcessor):
        logger.info("Starting build for %s...", self._name)

        # create a str combining all the extensions
        # -> then create the base src string using the extensions and common module
        ext_str = '\n'.join(f"#extension {ext}" for ext in process.ext)
        base = '\n'.join(['#line 1 "VCTX_EXTENSION_LIST"', ext_str, module]) + '\n'

        # go through each func and build if necessary
        stages: dict[str, str] = {}
        for func in process.funcs.items:
            # # create the replacement pattern
            pattern = Shader._func_header_regex(func.return_type, func.name)

            # create the src str
            src = f'#version {self._version}\n' + base
            src += f'#line 1 "FUNC_CONFIG({func.name})"\n' + '\n'.join(func.config) + '\n'
            src += pattern.sub('void main(', Shader.build_map(func.line_body), count=1)

            # behaviour based off stage
            match func.stage:
                case ShaderStage.COMP:
                    logger.info("-> Created new kernel: %s", func.name)
                    self._kernels[func.name] = Kernel(self._ctx, func.name, self._ctx.compute_shader(src))
                case ShaderStage.VERT | ShaderStage.FRAG | ShaderStage.GEOM | ShaderStage.TESC | ShaderStage.TESE:
                    stages[func.name] = src
                case _:
                    continue

        # create each program
        for prog_name, prog_dec in process.programs.items():
            logger.info("-> Created new program: %s", prog_name)

            # get all stage names
            frag = prog_dec.get(ShaderStage.FRAG)
            geom = prog_dec.get(ShaderStage.GEOM)
            tesc = prog_dec.get(ShaderStage.TESC)
            tese = prog_dec.get(ShaderStage.TESE)
            
            # create the program
            self._programs[prog_name] = self._ctx.program(
                vertex_shader = stages[str(prog_dec[ShaderStage.VERT])],
                fragment_shader = stages.get(frag) if frag else None,
                geometry_shader = stages.get(geom) if geom else None,
                tess_control_shader = stages.get(tesc) if tesc else None,
                tess_evaluation_shader = stages.get(tese) if tese else None,
            )
        
        #
        logger.info('Shader built successfully')

    
    @staticmethod
    def build_map(map: dict[int, ShaderSourceLine]) -> str:
        parts = []
        prev_index, prev_vctx = 0, None
        for index, ssl in sorted(map.items()):
            span = ssl.data.count('\n') > 1
            incl = 'include' in ssl.data
            emit = ssl.vctx != prev_vctx or (index - prev_index) != 1
            
            # check if this next ssl requires a head-side line directive
            if emit or span or incl: parts.append(f'#line {index} "{ssl.vctx}"\n')

            # add line data
            parts.append(ssl.data)

            # check if this ssl requires a tail-side line directtive
            if span or incl: parts.append(f'#line {index + 1} "{ssl.vctx}"\n')

            # set index and vctx
            prev_index, prev_vctx = index, ssl.vctx
        return "".join(parts) + '\n'
    