# -------------------------------------------------------------
# @file          shader_manager.py
# @author        Priyangkar Ghosh
# @created       2025-06-04
# @description   Main entry point: compiles all shaders in a directory and gives access to them.
# @license       MIT
# -------------------------------------------------------------

import logging

from tlang.compiler.binding_registry import BindingRegistry
logger = logging.getLogger(__name__)

import sys
import time
from pathlib import Path

from moderngl import Context
from tlang.compiler.dependency_manager import DependencyManager
from tlang.errors import SourceLocation, TlangAttributeError, TlangDependencyError
from tlang.frontend.interface_registry import InterfaceDecl, InterfaceTable
from tlang.compiler.shader import Shader
from tlang.compiler.shader_processor import ShaderProcessor


def _lookup(table: InterfaceTable, name: str) -> InterfaceDecl | None:
    for decl in table:
        if decl.name == name: return decl
    return None


FILE_EXT = '.tlang'
class ShaderManager:
    def __init__(
        self, ctx: Context, version: str, dir: str, constants: dict | None = None, strict: bool = True
    ) -> None:
        # strict: forwarded to every built `Shader`. True raises TlangCompileError/TlangLinkError
        # on a failed compile/link; False logs and continues, leaving that kernel/program missing.
        self._ctx = ctx
        self._strict = strict
        constants = constants if constants is not None else {}

        t0 = time.perf_counter()

        # A relative `dir` resolves against the immediate caller's file (sys._getframe(1), cheaper
        # than inspect.stack()); pass an absolute `dir` if constructing ShaderManager indirectly.
        path = Path(dir)
        if not path.is_absolute():
            path = (Path(sys._getframe(1).f_code.co_filename).resolve().parent / dir).resolve()

        if not path.is_dir():
            raise TlangDependencyError(f"Shader directory does not exist: '{path}'", SourceLocation(module=dir))

        filepaths = list(path.rglob(f'*{FILE_EXT}'))
        if not filepaths:
            raise TlangDependencyError(f"No '{FILE_EXT}' files found under '{path}'", SourceLocation(module=dir))

        dm = DependencyManager(constants)
        processors: dict[str, ShaderProcessor] = {}
        for fp in filepaths:
            name = '.'.join(fp.relative_to(path).with_suffix('').parts)
            if name in processors:
                raise TlangDependencyError(
                    f"Duplicate module name '{name}' (from '{fp}')", SourceLocation(module=name)
                )

            process = ShaderProcessor(name, fp.read_text(encoding='utf-8'), strict=self._strict)
            processors[name] = process
            dm.register(process)

        # Resolve [uses(...)] before the {{ CONSTANT }} pass so emitted array suffixes are
        # substituted too. A same-name/different-signature conflict is caught here, while both
        # declarations are still visible.
        iface_problems: list[str] = []
        for name, process in processors.items():
            merged = InterfaceTable()
            for dep in dm.resolve_dependencies(name):
                for decl in processors[dep].interfaces:
                    if (existing := _lookup(merged, decl.name)) is None: continue
                    if existing.signature == decl.signature: continue
                    a_txt, b_txt = ShaderProcessor._first_signature_diff(existing, decl)
                    iface_problems.append(
                        f"{name}: '{decl.name}' is declared differently in two modules: "
                        f"{existing.location} has '{a_txt}' but {decl.location} has '{b_txt}'"
                    )
                merged = processors[dep].interfaces.merged_with(merged)
            process.resolve_interfaces(merged)

        if iface_problems:
            message = '\n'.join(iface_problems)
            if self._strict: raise TlangAttributeError(message)
            for line in iface_problems: logger.warning("%s (continuing: strict=False)", line)

        # `dm.build_all()` only renders module-scope text; function bodies were already popped
        # out into `FunctionDef.line_body`/`.config` and never see that substitution, so patch
        # them here directly before any `Shader` is built.
        for process in processors.values():
            for func in process.funcs.items:
                for index, line in func.line_body.items():
                    line.data = dm.render(line.data, process.name, line=index)
                func.config = [dm.render(cfg, process.name, line=func.line_start) for cfg in func.config]

        # `commons` is rendered GLSL text; each `Shader` runs its own DCE-then-allocate binding
        # pass per artifact. `usage`/`pref_rank` stay global: a project-wide popularity ranking
        # of block names, used by every Shader as a tie-break so a shared block (e.g. `Globals`)
        # gets the same binding across every kernel/program that uses it.
        commons = dm.build_all()
        usage = BindingRegistry.compute_usage(commons)
        pref_rank = BindingRegistry.preference_rank(usage)

        self._shaders: dict[str, Shader] = {}
        for name, common in commons.items():
            process = processors[name]

            # Consolidate extensions from transitive dependencies too, not just direct
            # [include(...)] targets.
            for dep in dm.resolve_dependencies(name):
                if dep != name: process.ext.update(processors[dep].ext)

            self._shaders[name] = Shader(
                ctx, name, version, common, process, pref_rank, strict=self._strict
            )

        t1 = time.perf_counter()
        logger.info(
            'Built and compiled %d shaders in %.2f seconds',
            len(self._shaders), t1 - t0
        )

    @property
    def ctx(self) -> Context: return self._ctx

    def __getitem__(self, name: str) -> Shader | None:
        return self.get_shader(name)

    def __contains__(self, value: str) -> bool:
        return value in self._shaders

    def get_shader(self, name: str) -> Shader | None:
        """Look up a built shader by name, returning `None` if it wasn't built (unlike
        `Shader.get_kernel`/`get_program`, which raise `KeyError`)."""
        return self._shaders.get(name)

