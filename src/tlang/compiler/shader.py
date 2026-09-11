# -------------------------------------------------------------
# @file          shader.py
# @author        Priyangkar Ghosh
# @created       2025-06-13
# @description   Compiles a module's stage functions into kernels and programs.
# @license       MIT
# -------------------------------------------------------------

import logging

from tlang.compiler.binding_registry import BindingRegistry
from tlang.errors import SourceLocation, TlangAttributeError, TlangBindingError, TlangCompileError, TlangError, TlangLinkError
logger = logging.getLogger(__name__)

import difflib
import time
from dataclasses import replace
from typing import Any, Mapping
import regex as re
from moderngl import Buffer, Context, Program
from tlang.compiler.printf_codegen import render_printf_module, rewrite_printf_calls
from tlang.frontend.function_manager import FunctionDef
from tlang.runtime.printf_log import BUFFER_HANDLE as PRINTF_BUFFER_HANDLE, DEFAULT_LOG_CAPACITY, PrintfLog, PrintfTable
from tlang.runtime.kernel import Kernel
from tlang.runtime.pipeline import Pipeline
from tlang.frontend.interface_registry import ExternConst, InterfaceDecl, InterfaceKind
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
        dep_plain_funcs: dict[str, str] | None = None, keep_sources: bool = False,
        debug: bool = False, printf_table: PrintfTable | None = None, printf_log: PrintfLog | None = None,
        variants: Mapping[tuple[tuple[str, Any], ...], 'Shader'] | None = None,
    ) -> None:
        self._ctx = ctx
        self._name = name
        self._version = version
        self._strict = strict  # raise TlangCompileError/TlangLinkError instead of logging and continuing
        # debug: emit real printf(...) bodies (writing into printf_log's ring buffer) instead
        # of empty stubs, but only in an artifact whose own source actually calls printf --
        # see `tlang.compiler.printf_codegen.render_printf_module`. `printf_table` is the
        # shared call-site/format registry (see `tlang.runtime.printf_log.PrintfTable`),
        # always present regardless of `debug` -- the build-time specifier/argument-count
        # check it backs is not a debug-only nicety. `printf_log` is the ONE `PrintfLog`
        # (the actual GPU ring buffer) shared by every Shader/Kernel/Pipeline `ShaderManager`
        # builds; always non-None when `debug` is True (ShaderManager's job to guarantee that).
        self._debug = debug
        self._printf_table = printf_table if printf_table is not None else PrintfTable()
        self._printf_log = printf_log
        # False (default): a successfully compiled/linked entry point's generated GLSL is
        # dropped once its artifact is built and its bindings verified -- nothing downstream
        # needs the text once the driver has compiled it. A failed entry point's source is
        # always kept regardless of this flag (see `_build`), since that's exactly when
        # someone needs to read it.
        self._keep_sources = keep_sources

        self._kernels: dict[str, Kernel] = {}
        self._programs: dict[str, Program] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._sources: dict[str, str] = {}
        self._interfaces: dict[str, InterfaceDecl] = {d.name: d for d in processor.resolved_interfaces}
        # [extern] constants this module declares, resolved (see ExternConst.value/.literal)
        # by the time `processor` reached here -- ShaderManager resolves them right after
        # constructing each ShaderProcessor, before this Shader is ever built. Snapshotted
        # with `replace(e)` (a shallow copy of each small dataclass), not just re-keyed into a
        # new dict: `processor.funcs.externs`'s own `ExternConst` objects are mutated in place,
        # once per declared `[extern(precompile=[...])]` value, while `ShaderManager` builds
        # each precompiled variant of this same module (see `_build_precompiled_variants`) --
        # without this copy, this `Shader`'s `.externs` would go stale the moment the NEXT
        # variant (or the module's own default reset) overwrites the same `ExternConst` object.
        self._externs: dict[str, ExternConst] = {n: replace(e) for n, e in processor.externs.items()}
        # emitted GLSL block name -> HANDLE (InterfaceDecl.name), for every [buffer] interface.
        # Identity for a struct-form block or raw GLSL (never in this map at all, so a lookup
        # miss just falls back to the emitted name itself) -- only the [buffer] single-
        # declarator shorthand's synthesised block name differs from its handle. `BindingRegistry`
        # scans emitted GLSL, so its canon comes back keyed by emitted name; every canon handed
        # to a `Kernel`/`Pipeline` below is re-keyed through `_to_handles` before construction,
        # so `kernel.bindings`/`bind()`/`bind_ssbo`/`declared_blocks` only ever see handles.
        self._handle_of: dict[str, str] = {
            d.emitted_name: d.name for d in processor.resolved_interfaces if d.kind is InterfaceKind.BUFFER
        }
        self._failures: list[Exception] = []
        self._declared_entries: set[str] = set()
        self._ok = True
        # buffer source shared by every kernel/pipeline in this module -- see the `source`
        # property. Kept here (rather than only pushed out) so setting it after construction
        # still reaches kernels/pipelines built by `_build` below.
        self._buffer_source: Mapping[str, Buffer] | None = None

        # Every precompiled variant of this module (see `get_kernel`), already fully built by
        # `ShaderManager._build_precompiled_variants` -- one entry per combination of every
        # declared `[extern(precompile=[...])]` axis in this module, keyed
        # `tuple(sorted({name: value, ...}.items()))`. Nothing here is lazy: every variant that
        # will ever exist for this `Shader` exists by the time this constructor returns, so
        # `get_kernel` is a plain lookup, never a compile.
        self._variants: dict[tuple[tuple[str, Any], ...], 'Shader'] = dict(variants) if variants else {}

        if self._variants:
            # A module declaring a precompile axis has NO default artifact: `module` here
            # still has a blank, unresolved line for every such constant (see
            # `resolve_externs`), so compiling it directly would fail on every kernel that
            # references one (an undefined identifier) -- there is nothing to `_build` here.
            # Every kernel this module declares only exists inside `self._variants`; aggregate
            # this container's `ok`/`failures`/declared-entries from them instead so `get_kernel`,
            # `.ok`, and `.failures` all still mean the same thing they always have.
            self._ok = all(v.ok for v in self._variants.values())
            self._failures = [f for v in self._variants.values() for f in v.failures]
            self._declared_entries = set().union(*(v._declared_entries for v in self._variants.values()))
        else:
            self._build(module, processor, pref_rank or {}, dep_plain_funcs or {})

    @property
    def ok(self) -> bool:
        """True when every declared entry point of this module produced a kernel (compute) or
        was part of a successfully linked program (vert/frag/geom/tesc/tese). False means at
        least one entry point failed -- the natural `get_shader(name) is not None` check on
        `ShaderManager` reflects this, so a module that didn't fully compile is never mistaken
        for one that did."""
        return self._ok

    @property
    def failures(self) -> list[Exception]:
        """Errors this module hit while compiling/linking (empty when `ok`). Populated even
        under `strict=False`, where the module otherwise builds silently around the gap."""
        return self._failures

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
    def externs(self) -> Mapping[str, ExternConst]:
        """Every [extern] constant this module declares -- what it requires from
        `ShaderManager(constants={...})`, and what it actually resolved to."""
        return self._externs

    @property
    def declared_blocks(self) -> frozenset[str]:
        """Every SSBO block HANDLE declared anywhere in this module -- the union of every
        kernel's and pipeline's `bindings`. A tag a caller is about to hand to
        `BufferPool.alloc_temp`/`persistent_buffer` can be checked against this to catch a typo
        that would otherwise just create a buffer nothing ever binds."""
        names: set[str] = set()
        for kernel in self._kernels.values(): names.update(kernel.bindings)
        for pipeline in self._pipelines.values(): names.update(pipeline.bindings)
        return frozenset(names)

    def _to_handles(self, canon: Mapping[str, int]) -> dict[str, int]:
        """Re-key an SSBO canon from `BindingRegistry` (keyed by the emitted GLSL block name)
        onto tlang's own handles, via `self._handle_of` -- see its docstring. A `Kernel`/
        `Pipeline` must never see the emitted name for a [buffer] shorthand block, only the
        handle the author actually wrote."""
        return {self._handle_of.get(emitted, emitted): binding for emitted, binding in canon.items()}

    def _attach_printf_log(self, artifact: Kernel | Pipeline) -> None:
        """Bind the shared printf ring-buffer to `artifact`, but only when `artifact`
        actually declared the block (i.e. its own source called `printf(...)` -- see
        `render_printf_module`). Never runs at all outside a `debug=True` build. The buffer
        itself is a `PinnedBuffer` (see `pinned_buffer.py`), which duck-types
        `moderngl.Buffer` for everything `bind_ssbo` touches, so no special-casing is
        needed here. `Pipeline.bind_ssbo` binds immediately (no dispatch-time hook to defer
        to); `Kernel.bind_ssbo` records and is re-asserted on the next dispatch, same as any
        other required block."""
        if self._printf_log is None or PRINTF_BUFFER_HANDLE not in artifact.bindings: return
        artifact.bind_ssbo(PRINTF_BUFFER_HANDLE, self._printf_log.buffer)

    @property
    def buffer_source(self) -> Mapping[str, Buffer] | None:
        """The buffer source (`Kernel.buffer_source`/`Pipeline.buffer_source`) shared by every kernel and
        pipeline in this module. Setting it here reaches all of them in one call, rather than
        setting `.buffer_source` on each individually."""
        return self._buffer_source

    @buffer_source.setter
    def buffer_source(self, value: Mapping[str, Buffer] | None) -> None:
        self._buffer_source = value
        for kernel in self._kernels.values(): kernel.buffer_source = value
        for pipeline in self._pipelines.values(): pipeline.buffer_source = value

    @property
    def sources(self) -> dict[str, str]:
        """Read-only map of entry-point name -> the exact generated GLSL handed to the driver,
        for whichever entry points actually retained their source: every one of them under
        `keep_sources=True`, only the failed ones otherwise."""
        return self._sources

    def get_kernel(self, name: str, **variant: Any) -> Kernel:
        """The kernel named `name`.

        A module declaring no `[extern(precompile=[...])]` axis at all (`self._variants` is
        empty) works exactly as it always has -- `variant` is expected to be empty, and `name`
        is looked up in this `Shader`'s own compiled kernels.

        A module declaring one or more precompile axes has NO default artifact -- selecting a
        value for EVERY axis is mandatory for every kernel in it, `name` included, regardless
        of whether `name` itself references that particular constant. `variant` must supply
        exactly the axes this module declares (`tuple(sorted(variant.items()))` is the same key
        `ShaderManager._build_precompiled_variants` built), each with one of its precompiled
        values; the lookup is then a plain dict hit -- every combination that will ever exist
        was already compiled at ordinary `ShaderManager` build time, so this never compiles
        anything itself.

        Raises `TlangAttributeError` -- naming the constant(s), the module, and the permitted
        values -- if `variant` is missing a required axis, names something that isn't a
        precompile axis in this module, or gives a value outside that axis's declared list.
        """
        if not self._variants:
            if variant:
                raise TlangAttributeError(
                    self._describe_variant_miss(name, variant), SourceLocation(module=self._name),
                )
            return self._kernels[name]

        key = tuple(sorted(variant.items()))
        if (found := self._variants.get(key)) is not None:
            return found.get_kernel(name)
        raise TlangAttributeError(self._describe_variant_miss(name, variant), SourceLocation(module=self._name))

    def _describe_variant_miss(self, name: str, variant: Mapping[str, Any]) -> str:
        axis_decls = sorted((e for e in self._externs.values() if e.precompile), key=lambda e: e.name)
        axis_names = [d.name for d in axis_decls]
        problems: list[str] = []

        for key in variant:
            if key in axis_names: continue
            suggestion = ''
            if (matches := difflib.get_close_matches(key, axis_names, n=1)):
                suggestion = f" Did you mean '{matches[0]}'?"
            problems.append(
                f"'{key}' is not declared with [extern(precompile=[...])] in module "
                f"'{self._name}' (precompiled axes: {', '.join(axis_names) or '(none)'}).{suggestion}"
            )

        for decl in axis_decls:
            shown = ', '.join(repr(v) for v in decl.precompile)
            if decl.name not in variant:
                problems.append(
                    f"'{decl.name}' needs a value -- module '{self._name}' has no default "
                    f"artifact once it declares [extern(precompile=[...])]; call "
                    f"get_kernel('{name}', {decl.name}=<one of {{{shown}}}>{', ...' if len(axis_decls) > 1 else ''})"
                )
            elif variant[decl.name] not in decl.precompile:
                problems.append(
                    f"'{decl.name}={variant[decl.name]!r}' was not precompiled -- module "
                    f"'{self._name}' only precompiled {decl.name} in {{{shown}}}"
                )

        if not problems:
            # Every individual name/value pair is valid on its own, but this exact combination
            # isn't in `self._variants` -- shouldn't happen (every combination of every axis's
            # declared values is built, see `_build_precompiled_variants`), kept only as a
            # never-silent fallback rather than a confusing KeyError.
            shown = ', '.join(f'{k}={v!r}' for k, v in sorted(variant.items()))
            problems.append(f"get_kernel('{name}', {shown}): this combination was not found")

        return '\n'.join(problems)

    def get_program(self, name: str) -> Program:
        return self._programs[name]

    def get_pipeline(self, name: str) -> Pipeline:
        return self._pipelines[name]

    def get_source(self, name: str) -> str:
        """The generated GLSL for entry point `name`. Always available for a failed entry
        point, or for any entry point when this `Shader` was built with `keep_sources=True`;
        otherwise raises `TlangError` naming `name` -- a successfully compiled entry point's
        source is dropped by default once nothing needs it anymore."""
        if name not in self._sources:
            if name in self._declared_entries:
                raise TlangError(
                    f"'{name}': source was not retained (built with keep_sources=False) -- "
                    f"rebuild the ShaderManager/Shader with keep_sources=True to inspect it.",
                    SourceLocation(module=self._name),
                )
            raise TlangError(f"'{name}' is not a declared entry point of '{self._name}'", SourceLocation(module=self._name))
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

    def _raise_missing_export(
        self, func: FunctionDef, missing: set[str], process: ShaderProcessor, dep_plain_funcs: dict[str, str],
    ) -> None:
        """Raises for the first `missing` name (sorted for determinism) that resolves to a
        real module-scope function -- same file or an [include]d one -- so T15's raw driver
        error ("undefined variable") gets a tlang diagnostic naming the fix instead. A name
        matching neither is a genuine unknown (builtin, typo, macro) and is left alone; the
        driver's own message will name it.
        """
        same_file = {f.name: f for f in process.funcs.items if f.stage is None}

        for name in sorted(missing):
            if (helper := same_file.get(name)) is not None:
                note = ''
                if helper.links:
                    # T10: [link(...)] was attached to the helper instead of the caller, so
                    # the helper gained a link of its own instead of being pulled into `func`.
                    note = (
                        f" Note: '{name}' itself carries a [link(...)] -- if that was meant to pull "
                        f"'{name}' into '{func.name}', [link(...)] belongs on '{func.name}', not on '{name}'."
                    )
                raise TlangAttributeError(
                    f"'{func.name}' calls '{name}', a module-scope function defined in '{self._name}' "
                    f"but never emitted into this translation unit.{note} Fix: add [export()] to "
                    f"'{name}', or [link('{name}')] on '{func.name}'.",
                    SourceLocation(self._name, helper.line_start),
                )

            if (owner := dep_plain_funcs.get(name)) is not None:
                raise TlangAttributeError(
                    f"'{func.name}' calls '{name}', a module-scope function defined in included "
                    f"module '{owner}' but never exported, so it was never emitted into this "
                    f"translation unit. [link(...)] can't reach across modules -- fix: add "
                    f"[export()] to '{name}' in '{owner}'.",
                    SourceLocation(self._name, func.line_start),
                )

        # None of `missing` matched a known module-scope function -- an ordinary undefined
        # identifier (builtin typo, etc), not this diagnostic's concern.

    @staticmethod
    def _parse_error_location(message: str, fallback_module: str) -> SourceLocation:
        if (m := DRIVER_LOCATION_PATTERN.search(message)):
            if m.group('paren_line'):
                module = m.group('paren_qmodule') or m.group('paren_module')
                return SourceLocation(fallback_module if module == '0' else module, int(m.group('paren_line')))
            if m.group('colon_line'):
                return SourceLocation(m.group('colon_module'), int(m.group('colon_line')))
        return SourceLocation(fallback_module)

    def _build(
        self, module: str, process: ShaderProcessor, pref_rank: dict[str, int],
        dep_plain_funcs: dict[str, str],
    ):
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
            body_text = ''.join(Shader.build_map(f.line_body) for f in links)
            body_text += pattern.sub('void main(', Shader.build_map(func.line_body), count=1)

            # printf(...): the format string never reaches GLSL -- `rewrite_printf_calls`
            # replaces each call's format-string argument with a call-site id resolving
            # host-side (see `tlang.runtime.printf_log.PrintfTable`) BEFORE this text is
            # assembled any further, so every downstream pass only ever sees plain GLSL.
            #
            # GLSL, like C, has no forward declaration across a call site -- the overloads
            # (and, in debug, the ring buffer they write into) must appear in `src` BEFORE
            # `body_text`, not after, or a real call reads as an undefined identifier despite
            # `printf` being defined later in the same unit. Also must land before the
            # missing-export check below, for the same reason. `used` gates injection in BOTH
            # modes identically -- an artifact whose own source never calls printf(...) gets
            # nothing at all, release or debug (see render_printf_module's docstring: this
            # mirrors v1's `print`, where unconditionally injecting every artifact in release
            # measurably cost real driver-side parse time across a many-kernel project).
            capacity = self._printf_log.capacity if self._printf_log is not None else DEFAULT_LOG_CAPACITY
            body_text, used = rewrite_printf_calls(body_text, self._name, self._printf_table)
            if (printf_text := render_printf_module(debug=self._debug, capacity=capacity, used=used)):
                src += printf_text
            src += body_text

            # T15: before DFE removes anything, check whether reachable code calls a name
            # that exists as a module-scope function (this file, or an [include]d one) but
            # was never emitted into this unit -- an `[export()]`/`[link(...)]` omission,
            # not a real "undefined variable". Once DFE runs, an unreachable caller's call
            # sites are gone; a reachable caller's aren't, but the check is cheap enough
            # to just always run here rather than depend on that distinction.
            if (missing := BindingRegistry.find_missing_export_calls(src)):
                self._raise_missing_export(func, missing, process, dep_plain_funcs)

            # strip functions unreachable from `main` -- must run before the block DCE below
            # so it only sees blocks text `main` can actually reach.
            src = BindingRegistry.remove_dead_functions(src)

            # strip SSBO blocks this entry point never references
            src = BindingRegistry.remove_unused_buffers(src)

            stage_of[func.name] = func.stage
            dced[func.name] = src
            self._declared_entries.add(func.name)

            # Provisional, no bindings assigned; overwritten below once an artifact's bindings
            # are known (or retained as-is, for an entry point neither a kernel nor a program
            # ever attempts). Pruned at the end of `_build` for anything that isn't a failure,
            # unless `keep_sources` was requested.
            self._sources[func.name] = src

        # Every entry point that hit an exception below -- its source stays in `self._sources`
        # regardless of `keep_sources`, since a failure is exactly when the text is needed.
        failed_entries: set[str] = set()

        # Pass 2: compute kernels -- each is its own artifact.
        for name, stage in stage_of.items():
            if stage != ShaderStage.COMP: continue
            logger.info("-> Compiling kernel: %s", name)

            try:
                patched, canon, uniform_canon = BindingRegistry.allocate_artifact(
                    self._ctx, name, {ShaderStage.COMP: dced[name]}, pref_rank
                )
                src = patched[ShaderStage.COMP]
                texture_canon, image_canon = BindingRegistry.allocate_opaque_units(
                    self._ctx, name, {ShaderStage.COMP: src}
                )
                counter_patched, counter_canon = BindingRegistry.allocate_atomic_counters(
                    self._ctx, name, {ShaderStage.COMP: src}
                )
                src = self._sources[name] = counter_patched[ShaderStage.COMP]

                shader = self._ctx.compute_shader(src)
                BindingRegistry.verify_link(shader, canon, name, uniform_canon)
                BindingRegistry.assign_opaque_units(shader, texture_canon, image_canon)
                active_bindings = BindingRegistry.active_atomic_counter_bindings(shader)
                counter_canon = {n: pos for n, pos in counter_canon.items() if pos[0] in active_bindings}
                kernel = self._kernels[name] = Kernel(
                    self._ctx, name, shader, bindings=self._to_handles(canon),
                    texture_units=texture_canon, image_units=image_canon,
                    atomic_counters=counter_canon,
                )
                self._attach_printf_log(kernel)

            except TlangBindingError as e:
                logger.error(str(e))
                self._failures.append(e)
                failed_entries.add(name)
                if self._strict: raise

            except Exception as e:
                logger.error("Failed to compile %s shader '%s': %s", stage, name, e)
                err = TlangCompileError(
                    f"Failed to compile {stage} shader '{name}': {e}",
                    Shader._parse_error_location(str(e), self._name),
                    stage=str(stage), entry_point=name, source=dced[name],
                )
                self._failures.append(err)
                failed_entries.add(name)
                if self._strict: raise err from e

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
                texture_canon, image_canon = BindingRegistry.allocate_opaque_units(self._ctx, prog_name, patched)
                patched, counter_canon = BindingRegistry.allocate_atomic_counters(self._ctx, prog_name, patched)
                for stage, entry in entries.items(): self._sources[entry] = patched[stage]

                program = self._ctx.program(
                    vertex_shader=patched.get(ShaderStage.VERT),
                    fragment_shader=patched.get(ShaderStage.FRAG),
                    geometry_shader=patched.get(ShaderStage.GEOM),
                    tess_control_shader=patched.get(ShaderStage.TESC),
                    tess_evaluation_shader=patched.get(ShaderStage.TESE),
                )
                BindingRegistry.verify_link(program, canon, prog_name, uniform_canon)
                BindingRegistry.assign_opaque_units(program, texture_canon, image_canon)
                active_bindings = BindingRegistry.active_atomic_counter_bindings(program)
                counter_canon = {n: pos for n, pos in counter_canon.items() if pos[0] in active_bindings}
                self._programs[prog_name] = program
                pipeline = self._pipelines[prog_name] = Pipeline(
                    self._ctx, prog_name, program, bindings=self._to_handles(canon),
                    texture_units=texture_canon, image_units=image_canon,
                    atomic_counters=counter_canon,
                )
                self._attach_printf_log(pipeline)

            except (TlangLinkError, TlangBindingError) as e:
                logger.error(str(e))
                self._failures.append(e)
                failed_entries.update(fn.name for fn in pdef.stages.values())
                if self._strict: raise

            except Exception as e:
                logger.error("Failed to link program '%s': %s", prog_name, e)
                err = TlangLinkError(
                    f"Failed to link program '{prog_name}': {e}",
                    Shader._parse_error_location(str(e), self._name),
                )
                self._failures.append(err)
                failed_entries.update(fn.name for fn in pdef.stages.values())
                if self._strict: raise err from e

        # A declared entry point is "ok" when it produced a kernel (compute) or belongs to a
        # program that linked (raster). A raster entry orphaned from every [program(...)] was
        # never going to produce anything either way -- `_warn_orphan_stage_functions` already
        # flags that (non-fatally) at process time, so it's excluded here rather than counted
        # as a build failure.
        entry_programs: dict[str, list[str]] = {}
        for prog_name, pdef in process.programs.items():
            for fn in pdef.stages.values():
                entry_programs.setdefault(fn.name, []).append(prog_name)

        def _entry_ok(name: str, stage: ShaderStage) -> bool:
            if stage == ShaderStage.COMP: return name in self._kernels
            progs = entry_programs.get(name)
            return progs is None or any(p in self._programs for p in progs)

        self._ok = all(_entry_ok(name, stage) for name, stage in stage_of.items())

        # Drop every entry point's source that isn't a failure -- a compiled kernel or a
        # linked program's stage no longer needs its generated GLSL text once the artifact
        # exists and its bindings are verified. `keep_sources=True` opts out entirely; a
        # failed entry point (in `failed_entries`) is never dropped either way.
        if not self._keep_sources:
            for name in list(self._sources):
                if name not in failed_entries: del self._sources[name]

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
    