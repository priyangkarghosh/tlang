import logging

from bitarray import bitarray

from tlang.binding_registry import BindingRegistry
logger = logging.getLogger(__name__)

import time
import regex as re
from moderngl import Context, Program
from tlang.kernel import Kernel
from tlang.shader_processor import ShaderProcessor
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_stages import ShaderStage

SSBO_PATTERN = re.compile(
    r'layout\s*\(\s*([^)]*?)\s*\)\s*(readonly\s+|coherent\s+|volatile\s+|restrict\s+|)*buffer\s+(\w+)\s*{',
    re.MULTILINE
)

class Shader:
    def __init__(self, ctx: Context, name: str, version: str, module: str, processor: ShaderProcessor) -> None:
        self._ctx = ctx
        self._name = name
        self._version = version

        self._kernels: dict[str, Kernel] = {}
        self._programs: dict[str, Program] = {}
        self._sources: dict[str, str] = {}
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

    @staticmethod
    def _inject_bindings(ctx: Context, src: str):
        # create a free bindings bit array and set all of them to available
        free_bindings = bitarray(max_bindings := ctx.info['GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS'])
        free_bindings.setall(True)

        # mark manually assigned bindings as unavailable
        explicit_pattern = re.compile(
            r'layout\s*\(\s*([^)]*?)\s*\)\s*(?:readonly\s+|coherent\s+|volatile\s+|restrict\s+|)*buffer\s+(\w+)\s*{',
            re.MULTILINE,
        )

        binding_regex = re.compile(r'\bbinding\s*=\s*(\d+)\b')

        for match in explicit_pattern.finditer(src):
            layout_args, block_name = match.group(1), match.group(2)
            if (m := binding_regex.search(layout_args)):
                binding = int(m.group(1))
                if binding < max_bindings:
                    free_bindings[binding] = False

        # inject bindings into blocks in 'used' set with no explicit binding
        def replacer(match):
            layout_args, qualifier, block_name = match.group(1), match.group(2) or '', match.group(3)
            if binding_regex.search(layout_args): return match.group(0)

            # find first available binding
            try: binding = free_bindings.index(True)
            except ValueError:
                raise RuntimeError("Out of SSBO bindings!")

            free_bindings[binding] = False
            new_args = f"binding = {binding}, {layout_args}".strip().strip(',')
            return f"layout({new_args}) {qualifier}buffer {block_name} {{"

        inject_pattern = re.compile(
            r'layout\s*\(\s*([^)]*?)\s*\)\s*(readonly\s+|coherent\s+|volatile\s+|restrict\s+|)*buffer\s+(\w+)\s*{',
            re.MULTILINE,
        )

        src = inject_pattern.sub(replacer, src)
        return src
        
    def _build(self, module: str, process: ShaderProcessor):
        # start time for shader manager
        t0 = time.perf_counter()
        logger.info("Starting build for %s...", self._name)

        # create a str combining all the extensions
        # -> then create the base src string using the extensions and common module
        ext_str = '\n'.join(f"#extension {ext}" for ext in process.ext)
        base = '\n'.join(['#line 1 "VCTX_EXTENSION_LIST"', ext_str, module]) + '\n'

        # go through each func and build if necessary
        stages: dict[str, str] = {}
        for func in process.funcs.items:
            # create the replacement pattern
            pattern = Shader._func_header_regex(func.return_type, func.name)

            # create the src str
            src = f'#version {self._version}\n' + base
            src += f'#line 1 "FUNC_CONFIG({func.name})"\n' + '\n'.join(func.config) + '\n'
            src += pattern.sub('void main(', Shader.build_map(func.line_body), count=1)
            
            # inject ssbo bindings into the src
            src = BindingRegistry.remove_unused_buffers(src)

            # behaviour based off stage
            match func.stage:
                case ShaderStage.COMP:
                    logger.info("-> Compiling kernel: %s", func.name)
                    try: self._kernels[func.name] = Kernel(self._ctx, func.name, self._ctx.compute_shader(src))
                    except Exception as e: logger.error("Failed to compile %s shader '%s': %s", func.stage, func.name, e)
                    self._sources[func.name] = src
                case ShaderStage.VERT | ShaderStage.FRAG | ShaderStage.GEOM | ShaderStage.TESC | ShaderStage.TESE:
                    stages[func.name] = self._sources[func.name] = src
                case _:
                    continue                

        for prog_name, prog_dec in process.programs.items():
            logger.info("-> Linking program: %s", prog_name)

            try:
                frag = prog_dec.get(ShaderStage.FRAG)
                geom = prog_dec.get(ShaderStage.GEOM)
                tesc = prog_dec.get(ShaderStage.TESC)
                tese = prog_dec.get(ShaderStage.TESE)

                self._programs[prog_name] = self._ctx.program(
                    vertex_shader=stages[str(prog_dec[ShaderStage.VERT])],
                    fragment_shader=stages.get(frag) if frag else None,
                    geometry_shader=stages.get(geom) if geom else None,
                    tess_control_shader=stages.get(tesc) if tesc else None,
                    tess_evaluation_shader=stages.get(tese) if tese else None,
                )

            except Exception as e:
                logger.error("Failed to link program '%s': %s", prog_name, e)
        
        # log build
        t1 = time.perf_counter()
        logger.info(
            'Shader built in %.2f seconds with %d kernels and %d programs',
            t1 - t0, len(self._kernels), len(self._programs)
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
    