# -------------------------------------------------------------
# @file          shader.py
# @author        Priyangkar Ghosh
# @created       2025-06-13
# @description   Compiles a module's stage functions into kernels and programs.
# @license       MIT
# -------------------------------------------------------------

import logging

from tlang.compiler.binding_registry import BindingRegistry
from tlang.errors import SourceLocation, TlangBindingError, TlangCompileError, TlangLinkError
logger = logging.getLogger(__name__)

import time
from typing import Mapping
import regex as re
from moderngl import Context, Program
from tlang.frontend.function_manager import FunctionDef
from tlang.runtime.kernel import Kernel
from tlang.runtime.pipeline import Pipeline
from tlang.frontend.interface_registry import InterfaceDecl
from tlang.compiler.shader_processor import ShaderProcessor
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_stages import ShaderStage

# Best-effort match of a driver GLSL error's source location, so a compile failure can point
# back at a `#line N "module"` directive. Handles NVIDIA Cg-style `module(line)`/`0(line)` and
# glslang/ANGLE-style `"module":line`; falls back to just the module name if neither matches.
DRIVER_LOCATION_PATTERN = re.compile(
    r'(?:"(?P<paren_qmodule>[^"]+)"|(?P<paren_module>[\w./-]+))\(\s*(?P<paren_line>\d+)\s*\)'
    r'|"(?P<colon_module>[^"]+)"\s*:\s*(?P<colon_line>\d+)'
)

class Shader:
    def __init__(
        self, ctx: Context, name: str, version: str, module: str, processor: ShaderProcessor,
        pref_rank: dict[str, int] | None = None, strict: bool = True,
    ) -> None:
        self._ctx = ctx
        self._name = name
        self._version = version
        self._strict = strict  # raise TlangCompileError/TlangLinkError instead of logging and continuing

        self._kernels: dict[str, Kernel] = {}
        self._programs: dict[str, Program] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._sources: dict[str, str] = {}
        self._interfaces: dict[str, InterfaceDecl] = {d.name: d for d in processor.resolved_interfaces}
        self._build(module, processor, pref_rank or {})

    @property
    def kernels(self) -> dict[str, Kernel]:
        return self._kernels

    @property
    def programs(self) -> dict[str, Program]:
        return self._programs

    @property
    def pipelines(self) -> dict[str, Pipeline]:
        """Name-keyed `Pipeline` wrapper for every linked `[program(...)]`, the graphics-side
        counterpart of `kernels`; adds the same bind_ssbo/set_uniform ergonomics as `Kernel`."""
        return self._pipelines

    @property
    def interfaces(self) -> Mapping[str, InterfaceDecl]:
        """Every [varyings]/[uniforms]/[buffer] interface visible to this module,
        its transitive includes included. Use `member_locations` for assignments."""
        return self._interfaces

    @property
    def sources(self) -> dict[str, str]:
        """Read-only map of entry-point name -> the exact generated GLSL handed to the driver."""
        return self._sources

    def get_kernel(self, name: str) -> Kernel:
        return self._kernels[name]

    def get_program(self, name: str) -> Program:
        return self._programs[name]

    def get_pipeline(self, name: str) -> Pipeline:
        return self._pipelines[name]

    def get_source(self, name: str) -> str:
        """The generated GLSL for entry point `name` (compiled successfully or not)."""
        return self._sources[name]

    @staticmethod
    def _in_shared_module(f: FunctionDef) -> bool:
        """A plain (no-stage) [export]ed helper is already folded into the shared module text,
        so it must not also be re-emitted via [link] (that would redefine it in GLSL)."""
        return f.stage is None and f.exported

    @staticmethod
    def _func_header_regex(ret_type: str, name: str) -> re.Pattern:
        ret = r'\s+'.join([re.escape(w) for w in ret_type.split()])
        return re.compile(rf'\b{ret}\s+{re.escape(name)}\s*\(', re.MULTILINE)

    @staticmethod
    def _parse_error_location(message: str, fallback_module: str) -> SourceLocation:
        if (m := DRIVER_LOCATION_PATTERN.search(message)):
            if m.group('paren_line'):
                module = m.group('paren_qmodule') or m.group('paren_module')
                return SourceLocation(fallback_module if module == '0' else module, int(m.group('paren_line')))
            if m.group('colon_line'):
                return SourceLocation(m.group('colon_module'), int(m.group('colon_line')))
        return SourceLocation(fallback_module)

    def _build(self, module: str, process: ShaderProcessor, pref_rank: dict[str, int]):
        t0 = time.perf_counter()
        logger.info("Starting build for %s...", self._name)

        ext_lines = [f"#extension {ext}" for ext in process.ext]
        base = '\n'.join(['#line 1 "VCTX_EXTENSION_LIST"', *ext_lines, module]) + '\n'

        # Pass 1: assemble + dead-code-eliminate every stage entry point's source. Bindings are
        # assigned later, per artifact, so a binding is only ever spent on a block that survives
        # DCE into the exact text handed to the driver.
        stage_of: dict[str, ShaderStage] = {}
        dced: dict[str, str] = {}
        for func in process.funcs.items:
            if func.stage is None: continue  # plain helper, already folded into `module`/links

            pattern = Shader._func_header_regex(func.return_type, func.name)

            src = f'#version {self._version}\n' + base
            src += f'#line 1 "FUNC_CONFIG({func.name})"\n' + '\n'.join(func.config) + '\n'

            # Linked helpers are emitted from `f.line_body` (already processed: attributes
            # rewritten, constants substituted), not raw `f.body`. Skip helpers already folded
            # into the shared module so each appears exactly once.
            links = [f for f in func.links if not Shader._in_shared_module(f)]
            src += ''.join(Shader.build_map(f.line_body) for f in links)

            src += pattern.sub('void main(', Shader.build_map(func.line_body), count=1)

            # strip SSBO blocks this entry point never references
            src = BindingRegistry.remove_unused_buffers(src)

            stage_of[func.name] = func.stage
            dced[func.name] = src

            # Provisional, no bindings assigned; overwritten below once an artifact's bindings
            # are known. A stage used by neither a kernel nor a program keeps this, so
            # get_source/.sources always has something for every declared entry point.
            self._sources[func.name] = src

        # Pass 2: compute kernels -- each is its own artifact.
        for name, stage in stage_of.items():
            if stage != ShaderStage.COMP: continue
            logger.info("-> Compiling kernel: %s", name)

            try:
                patched, canon, uniform_canon = BindingRegistry.allocate_artifact(
                    self._ctx, name, {ShaderStage.COMP: dced[name]}, pref_rank
                )
                src = self._sources[name] = patched[ShaderStage.COMP]

                shader = self._ctx.compute_shader(src)
                BindingRegistry.verify_link(shader, canon, name, uniform_canon)
                self._kernels[name] = Kernel(self._ctx, name, shader, bindings=canon)

            except TlangBindingError as e:
                logger.error(str(e))
                if self._strict: raise

            except Exception as e:
                logger.error("Failed to compile %s shader '%s': %s", stage, name, e)
                if self._strict:
                    raise TlangCompileError(
                        f"Failed to compile {stage} shader '{name}': {e}",
                        Shader._parse_error_location(str(e), self._name),
                        stage=str(stage), entry_point=name, source=dced[name],
                    ) from e

        # Pass 3: programs -- every stage of one [program(...)] is one
        # artifact, so a block shared across e.g. vertex + fragment keeps
        # exactly one binding everywhere it's declared.
        for prog_name, pdef in process.programs.items():
            logger.info("-> Linking program: %s", prog_name)

            try:
                # `pdef.stages` was fully validated in ShaderProcessor, so `dced[fn.name]` is
                # guaranteed to exist here.
                stage_srcs: dict[ShaderStage, str] = {stage: dced[fn.name] for stage, fn in pdef.stages.items()}
                entries: dict[ShaderStage, str] = {stage: fn.name for stage, fn in pdef.stages.items()}

                patched, canon, uniform_canon = BindingRegistry.allocate_artifact(self._ctx, prog_name, stage_srcs, pref_rank)
                for stage, entry in entries.items(): self._sources[entry] = patched[stage]

                program = self._ctx.program(
                    vertex_shader=patched.get(ShaderStage.VERT),
                    fragment_shader=patched.get(ShaderStage.FRAG),
                    geometry_shader=patched.get(ShaderStage.GEOM),
                    tess_control_shader=patched.get(ShaderStage.TESC),
                    tess_evaluation_shader=patched.get(ShaderStage.TESE),
                )
                BindingRegistry.verify_link(program, canon, prog_name, uniform_canon)
                self._programs[prog_name] = program
                self._pipelines[prog_name] = Pipeline(self._ctx, prog_name, program, bindings=canon)

            except (TlangLinkError, TlangBindingError) as e:
                logger.error(str(e))
                if self._strict: raise

            except Exception as e:
                logger.error("Failed to link program '%s': %s", prog_name, e)
                if self._strict:
                    raise TlangLinkError(
                        f"Failed to link program '{prog_name}': {e}",
                        Shader._parse_error_location(str(e), self._name),
                    ) from e

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

            if emit or span or incl: parts.append(f'#line {index} "{ssl.vctx}"\n')
            parts.append(ssl.data)
            if span or incl: parts.append(f'#line {index + 1} "{ssl.vctx}"\n')

            prev_index, prev_vctx = index, ssl.vctx
        return "".join(parts) + '\n'
    