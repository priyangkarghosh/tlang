import re
from moderngl import ComputeShader, Context, Program

from shader_processor import ShaderProcessor
from shader_source_line import ShaderSourceLine
from shader_stages import ShaderStage

FUNC_REG_EXP = r'\b{ret_type}\s+{name}\s*\('

class Shader:
    def __init__(self, ctx: Context, version: str, module: str, processor: ShaderProcessor) -> None:
        self._ctx = ctx
        self._version = version

        self._kernels: dict[str, ComputeShader] = {}
        self._programs: dict[str, Program] = {}
        self._build(module, processor)
    
    def _build(self, module: str, process: ShaderProcessor):
        # create a str combining all the extensions
        # -> then create the base src string using the extensions and common module
        ext_str = '\n'.join(f"#extension {ext}" for ext in process.ext)
        base = '\n'.join(['#line 1 "VCTX_EXTENSION_LIST"', ext_str, module]) + '\n'

        # go through each func and build if necessary
        stages: dict[str, str] = {}
        for func in process.funcs.items:
            # create the replacement pattern
            pattern = re.compile(
                FUNC_REG_EXP.format(
                    ret_type=func.return_type, name=func.name
                )
            )

            # create the src str
            src = f'#version {self._version}\n' + base
            src += f'#line 1 "FUNC_CONFIG({func.name})"\n' + '\n'.join(func.config) + '\n'
            src += Shader.build_map(func.line_body)
            src = re.sub(pattern, 'void main(', src, count=1)

            # behaviour based off stage
            match func.stage:
                case ShaderStage.COMP:
                    self._kernels[func.name] = self._ctx.compute_shader(src)
                    continue
                
                case ShaderStage.VERT | ShaderStage.FRAG | ShaderStage.GEOM | ShaderStage.TESC | ShaderStage.TESE:
                    stages[func.name] = src
                    continue
                
                case _:
                    continue

        # create each program
        for prog_name, prog_dec in process.programs.items():
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

                # --> LOOK INTO VARYINGS AND OUTPUTS
            )

    
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
    