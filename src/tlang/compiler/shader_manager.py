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
from typing import Mapping

from moderngl import Buffer, Context
from tlang.compiler.dependency_manager import DependencyManager
from tlang.errors import SourceLocation, TlangAttributeError, TlangBuildError, TlangDependencyError
from tlang.frontend.function_manager import param_type_signature
from tlang.frontend.interface_registry import InterfaceDecl, InterfaceTable
from tlang.compiler.shader import Shader
from tlang.compiler.shader_processor import ShaderProcessor
from tlang.runtime.printf_log import DEFAULT_LOG_CAPACITY, PrintfLog, PrintfStream, PrintfTable


def _lookup(table: InterfaceTable, name: str) -> InterfaceDecl | None:
    for decl in table:
        if decl.name == name: return decl
    return None


def _check_duplicate_declarations(
    processors: dict[str, ShaderProcessor], dm: DependencyManager, strict: bool,
) -> None:
    """Same-named top-level declarations -- plain exported functions, `const`s, raw
    `buffer`/`uniform` blocks -- collide the instant two included modules' text lands in
    the same translation unit; the driver reports that as a redefinition at a
    generated-GLSL line number, not in tlang's own terms. This runs over each module's
    MERGED include closure, before any of that text is rendered or handed to a driver.

    Only a plain (no [shader(...)] stage) *exported* function can actually appear in a
    dependent's merged module text (see `ShaderProcessor._create_module` / T15) -- a
    non-exported helper never leaves its own file, so it can't collide with anything
    outside it. A `const`/raw block, by contrast, is always part of that shared text.

    GLSL overloading (same name, different parameter types) is legal and used for real
    (utils.tlang's three `hash` overloads) -- see `param_type_signature`'s own docstring
    for why an unreadable signature is skipped rather than guessed at.

    `dm.resolve_dependencies` already dedupes a module's transitive closure with a
    visited set, so a diamond include graph lists each shared dependency once; nothing
    here needs to guard against comparing a module's declarations against themselves.

    Collects every duplicate across the whole tree before raising/logging once, so one
    collision never hides the rest -- consistent with `ShaderProcessor._process_global_attrs`.
    """
    problems: list[str] = []
    reported: set[tuple] = set()

    def _report(kind: str, ident: str, loc_a: SourceLocation, loc_b: SourceLocation) -> None:
        a, b = sorted((loc_a, loc_b), key=lambda l: (l.module or '', l.line or 0))
        key = (kind, ident, a.module, a.line, b.module, b.line)
        if key in reported: return
        reported.add(key)
        problems.append(f"{kind} '{ident}' is declared twice across the include closure (first at {a}, again at {b})")

    for name in processors:
        funcs_seen: dict[tuple[str, tuple[str, ...]], SourceLocation] = {}
        consts_seen: dict[str, SourceLocation] = {}
        blocks_seen: dict[str, SourceLocation] = {}

        for dep in dm.resolve_dependencies(name):
            process = processors[dep]

            for fn in process.funcs.items:
                if fn.stage is not None or not fn.exported: continue
                if (sig := param_type_signature(fn.params)) is None: continue  # unreadable -- stay quiet
                key = (fn.name, sig)
                loc = SourceLocation(dep, fn.line_start)
                if (prev := funcs_seen.get(key)) is not None:
                    _report('function', f"{fn.name}({', '.join(sig)})", prev, loc)
                else:
                    funcs_seen[key] = loc

            for decl in process.funcs.decls:
                seen = consts_seen if decl.kind == 'const' else blocks_seen
                label = 'const' if decl.kind == 'const' else f'{decl.kind} block'
                loc = SourceLocation(dep, decl.line)
                if (prev := seen.get(decl.name)) is not None:
                    _report(label, decl.name, prev, loc)
                else:
                    seen[decl.name] = loc

    if problems:
        message = '\n'.join(problems)
        if strict: raise TlangAttributeError(message)
        for line in problems: logger.warning("%s (continuing: strict=False)", line)


FILE_EXT = '.tlang'
class ShaderManager:
    def __init__(
        self, ctx: Context, version: str, dir: str, constants: dict | None = None, strict: bool = True,
        keep_sources: bool = False, debug: bool = False, debug_log_capacity: int = DEFAULT_LOG_CAPACITY,
    ) -> None:
        # strict: forwarded to every built `Shader`. True raises TlangCompileError/TlangLinkError
        # on a failed compile/link; False logs and continues, leaving that kernel/program missing.
        # keep_sources: forwarded to every built `Shader`. False (default) drops a successfully
        # compiled/linked entry point's generated GLSL once its artifact exists; a failed entry
        # point's source is always kept. True keeps every entry point's source, always.
        # debug: turns on `printf(...)` inside shader code (see references/runtime.md). False
        # (default) is completely inert: every `printf(...)` overload compiles to an empty body
        # with no backing buffer anywhere in the tree -- the driver eliminates it for free. True
        # allocates ONE pinned `PrintfLog` ring buffer (sized for `debug_log_capacity` records,
        # shared by every kernel/pipeline this manager builds) and gives a real body to
        # `printf(...)` in whichever artifacts actually call it. `self.stdout` is the shader's
        # stdout either way -- `None` when `debug=False`.
        self._ctx = ctx
        self._strict = strict
        # Always built, regardless of `debug` -- every printf(...) call site is scanned and its
        # specifier/argument count validated at build time even in a release build; only the
        # GPU-side ring buffer (`_printf_log` below) is debug-only.
        self._printf_table = PrintfTable()
        self._printf_log: PrintfLog | None = PrintfLog(ctx, self._printf_table, debug_log_capacity) if debug else None
        self._stdout: PrintfStream | None = PrintfStream(self._printf_log) if self._printf_log is not None else None
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
            # Must run before `dm.register` below: that call snapshots this module's text for
            # the {{ CONSTANT }}/[include] pass, so an [extern] constant has to already be
            # resolved to its final `const ...;` line by then, or the snapshot never sees it.
            process.resolve_externs(constants)
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

        # Same-named top-level functions/consts/raw buffer-uniform blocks across a module's
        # merged include closure -- a plain GLSL redefinition, not an interface conflict --
        # also has to be caught before this text is rendered or handed to a driver.
        _check_duplicate_declarations(processors, dm, self._strict)

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

        # Each module is built in isolation: one module's failure (a broken kernel, a bad link)
        # must never prevent the rest of the tree from building, and must never be reported only
        # as "the first thing that went wrong" -- under strict=True every module is still
        # attempted, and every failure collected, before a single error is raised at the end.
        self._shaders: dict[str, Shader] = {}
        self._failures: dict[str, list[Exception]] = {}
        for name, common in commons.items():
            process = processors[name]

            # Consolidate extensions from transitive dependencies too, not just direct
            # [include(...)] targets. Also collect every module-scope function name declared
            # anywhere in the dependency closure (module it's defined in, first one wins on a
            # name clash) -- `Shader` uses this only to name the module in its T15 diagnostic
            # when a call resolves to a real function that was simply never exported.
            dep_plain_funcs: dict[str, str] = {}
            for dep in dm.resolve_dependencies(name):
                if dep == name: continue
                process.ext.update(processors[dep].ext)
                for fn in processors[dep].funcs.items:
                    if fn.stage is None: dep_plain_funcs.setdefault(fn.name, dep)

            try:
                shader = Shader(
                    ctx, name, version, common, process, pref_rank, strict=self._strict,
                    dep_plain_funcs=dep_plain_funcs, keep_sources=keep_sources,
                    debug=debug, printf_table=self._printf_table, printf_log=self._printf_log,
                )
            except Exception as e:
                self._failures[name] = [e]
                continue

            self._shaders[name] = shader
            if not shader.ok: self._failures[name] = shader.failures

        if self._strict and self._failures:
            named = [(name, err) for name, errs in self._failures.items() for err in errs]
            if len(named) == 1:
                raise named[0][1]  # sole failure: preserve its exact type/attributes

            lines = '\n'.join(f'{name}: {err}' for name, err in named)
            raise TlangBuildError(
                f"{len(self._failures)} shader module(s) failed to build:\n{lines}",
                failures=dict(self._failures),
            )

        t1 = time.perf_counter()
        logger.info(
            'Built and compiled %d shaders in %.2f seconds',
            len(self._shaders), t1 - t0
        )

        # buffer source shared by every Shader (and, transitively, every kernel/pipeline) in the
        # tree -- see the `source` property.
        self._buffer_source: Mapping[str, Buffer] | None = None

    @property
    def ctx(self) -> Context: return self._ctx

    def __getitem__(self, name: str) -> Shader | None:
        return self.get_shader(name)

    def __contains__(self, value: str) -> bool:
        return value in self._shaders

    @property
    def failures(self) -> dict[str, list[Exception]]:
        """Module name -> the errors it hit while building (only failed modules are keys)."""
        return self._failures

    @property
    def declared_blocks(self) -> frozenset[str]:
        """Every SSBO block name declared anywhere in the whole shader tree -- the union of
        every built `Shader.declared_blocks`. Lets a caller validate a `BufferPool` tag (or any
        other buffer name) against reality before it becomes an orphan nothing ever binds."""
        names: set[str] = set()
        for shader in self._shaders.values(): names.update(shader.declared_blocks)
        return frozenset(names)

    @property
    def buffer_source(self) -> Mapping[str, Buffer] | None:
        """The buffer source shared by every `Shader` in the tree. Setting it here is the one
        call that reaches every kernel and pipeline this manager built -- equivalent to setting
        `.buffer_source` on each `Shader` individually."""
        return self._buffer_source

    @buffer_source.setter
    def buffer_source(self, value: Mapping[str, Buffer] | None) -> None:
        self._buffer_source = value
        for shader in self._shaders.values(): shader.buffer_source = value

    @property
    def stdout(self) -> PrintfStream | None:
        """The shader's stdout -- every `printf(...)` call from every kernel/pipeline this
        manager built streams (or drains) through this ONE object, or `None` when this
        manager was built with `debug=False` (the default):

            sm.stdout.stream(sink=print)   # background thread -> sink(line) per record
            sm.stdout.stop()
            sm.stdout.drain()              # -> formatted lines available right now

        See `references/runtime.md` for the full contract (dedup, rate limiting, the ring
        buffer's ACK'd flow control)."""
        return self._stdout

    def get_shader(self, name: str, *, allow_failed: bool = False) -> Shader | None:
        """Look up a built shader by name, returning `None` if it wasn't built (unlike
        `Shader.get_kernel`/`get_program`, which raise `KeyError`) OR if it built but didn't
        fully compile (`strict=False`) -- `shader.ok` is False. Pass `allow_failed=True` to get
        the `Shader` back anyway, e.g. to inspect `.failures` for diagnosis."""
        shader = self._shaders.get(name)
        if shader is None: return None
        if not allow_failed and not shader.ok: return None
        return shader

