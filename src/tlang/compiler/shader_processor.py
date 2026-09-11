# -------------------------------------------------------------
# @file          shader_processor.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Creates all Kernel/Program objects
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import difflib
from dataclasses import dataclass
from typing import Any

from tlang.frontend.attribute import Attribute
from tlang.frontend.attribute_handlers import REGISTRY
from tlang.frontend.attribute_manager import AttributeManager
from tlang.frontend.attribute_registry import AttrCtx, AttrSpec, Diagnostics, StageConfig, USE_ARG, bind_params, bind_update
from tlang.errors import SourceLocation, TlangAttributeError, TlangSyntaxError
from tlang.frontend.function_manager import FunctionDef, FunctionManager, InterfaceRef
from tlang.frontend.interface_registry import (
    ExternConst,
    InterfaceDecl,
    InterfaceKind,
    InterfaceTable,
    check_extern_value,
    emit_glsl,
    extern_literal,
    is_arrayed,
)
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_stages import ShaderStage, _SHADER_STAGE_ALIASES
from tlang.shader_utils import EXTENSION_GROUPS


@dataclass(slots=True, frozen=True)
class PipelineDef:
    """One validated `[program(...)]`: a name plus its fully-resolved stage -> FunctionDef mapping.

    Every kwarg has already been checked (known alias, no collision, entry point exists, stage
    matches its slot) and the stage combination is legal, so `Shader._build` can index `stages`
    directly with no defensive `.get()`.
    """
    name: str
    stages: dict[ShaderStage, FunctionDef]
    location: SourceLocation | None = None


# Stage tokens a raster [program(...)] may carry (compute is dispatched via a kernel, never
# linked into a program).
_RASTER_STAGES: frozenset[ShaderStage] = frozenset({
    ShaderStage.VERT, ShaderStage.FRAG, ShaderStage.GEOM, ShaderStage.TESC, ShaderStage.TESE,
})


def _did_you_mean(name: str, candidates: list[str]) -> str:
    matches = difflib.get_close_matches(name, candidates, n=1)
    return f" Did you mean '{matches[0]}'?" if matches else ""


class ShaderProcessor:
    def __init__(self, name: str, src: str, strict: bool = True) -> None:
        self.name, self.src, self.strict = name, src, strict
        self.src_map: dict[int, ShaderSourceLine] = {
            index: ShaderSourceLine(name, line)
            for index, line in enumerate(src.splitlines(keepends=True), start=1)
        }

        self.dps: set[str] = set()
        # Extensions are opt-in via [require(...)]/[extend(...)] (see EXTENSION_GROUPS).
        self.ext: set[str] = set()
        self.programs: dict[str, PipelineDef] = {}
        self.resolved_interfaces = InterfaceTable()
        self.diagnostics = Diagnostics(strict=strict)

        self.funcs = FunctionManager.extract_funcs(self.name, self.src, self.src_map)
        self.glob_attrs = AttributeManager.process_attrs(self.name, self.src_map, self.funcs, strict=strict)

        self.module: dict[int, ShaderSourceLine] = {}
        self._process_function_attrs()
        self._process_global_attrs()
        self._create_module()

    @property
    def interfaces(self) -> InterfaceTable:
        """This module's own interface declarations, excluding dependencies."""
        return self.funcs.interfaces

    @property
    def externs(self) -> dict[str, ExternConst]:
        """Every [extern] constant this module declares, name -> ExternConst. Populated at
        attach time (parsing); `.resolved`/`.value`/`.literal` are only meaningful after
        `resolve_externs` has run -- reflect on `.type_name`/`.has_default` beforehand to ask
        what a module requires without needing `constants={...}` yet."""
        return {e.name: e for e in self.funcs.externs}

    def resolve_externs(self, constants: dict[str, Any]) -> None:
        """Resolves every [extern] declaration, emitting the final text in place of the
        declaration's placeholder line. Mirrors `resolve_interfaces`: every problem in this
        module is collected and raised together, not one at a time -- a module needing five
        constants shouldn't make an author fix them one build at a time.

        A **plain** `[extern]` resolves against `constants` (a required one falls back to its
        own `= default` when absent), emitting `const <type> <name> = <literal>;`.

        An `[extern(precompile=[...])]` -- a non-empty `decl.precompile` -- is left completely
        untouched here: it never consults `constants`, and gets no default/plain artifact of
        any kind (the placeholder blank line `AttributeHandlers.extern` wrote stays blank).
        Only `ShaderManager._build_precompiled_variants` ever gives it text -- one
        `const <type> <name> = <literal>;` per listed value, in a fully separate `Shader`
        build per value, never in this module's own default build (there isn't one for a
        module carrying a precompile axis -- see `ShaderManager.__init__`).

        Must run before this processor's module text is registered with `DependencyManager`
        (`ShaderManager` does so immediately after construction) -- once that snapshot is
        taken, a later mutation here would never reach the text that's actually built.
        """
        problems: list[str] = []
        for decl in self.funcs.externs:
            if decl.precompile:
                continue  # resolved only per-variant, in ShaderManager._build_precompiled_variants

            if decl.name in constants:
                value = constants[decl.name]
            elif decl.has_default:
                value = decl.default_value
            else:
                problems.append(
                    f"{decl.location}: [extern] {decl.type_name} {decl.name} is required but "
                    f"'{decl.name}' was not supplied -- add constants={{'{decl.name}': <{decl.type_name}>, "
                    f"...}} to ShaderManager(...), or give it a default: "
                    f"'[extern] {decl.type_name} {decl.name} = ...;'"
                )
                continue

            if (bad := check_extern_value(decl.type_name, value)) is not None:
                problems.append(
                    f"{decl.location}: [extern] {decl.type_name} {decl.name} expects {bad}"
                )
                continue

            decl.value = value
            decl.literal = extern_literal(decl.type_name, value)
            decl.resolved = True
            # `decl.trailing` always carries the line's terminating '\n' (plus, for the
            # same-line form, anything written after the ';' on that line) -- mirrors
            # `_declare_buffer_same_line`, which relies on `line_tail[end_offset:]` the same way.
            self.src_map[decl.line].data = f'const {decl.type_name} {decl.name} = {decl.literal};' + decl.trailing

        if problems:
            message = '\n'.join(problems)
            if self.strict: raise TlangAttributeError(message)
            for line in problems: logger.warning("%s (continuing: strict=False)", line)

    # Builds the module text exported to dependents: common code plus any [export]ed helpers.
    def _create_module(self) -> None:
        # Shallow copy: self.module and self.src_map share the same ShaderSourceLine instances.
        # Safe since neither is mutated after this point; treat both as read-only from here on.
        self.module = self.src_map.copy()
        for func in self.funcs.items:
            if not func.stage and func.exported:
                self.module.update(func.line_body)

    def _process_global_attrs(self) -> None:
        # Every [program(...)] is fully checked before anything is raised, so all problems are
        # reported in one shot. strict=False logs warnings instead and drops the bad program(s).
        problems: list[str] = []
        program_locs: dict[str, SourceLocation | None] = {}
        referenced: set[str] = set()

        for attr in self.glob_attrs:
            match attr.name:
                case 'program':
                    self._process_program_attr(attr, program_locs, referenced, problems)

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

        self._warn_orphan_stage_functions(referenced)

        if problems:
            message = '\n'.join(problems)
            if self.strict: raise TlangAttributeError(message)
            for line in problems: logger.warning("%s (continuing: strict=False)", line)

    def _process_program_attr(
        self, attr: Attribute, program_locs: dict[str, SourceLocation | None],
        referenced: set[str], problems: list[str],
    ) -> None:
        """Validate one [program(...)]: alias/collision checks before entry-point lookup, then
        stage-combination checks. On success records a `PipelineDef`; on any failure the program
        is left out entirely and every problem is appended to `problems` rather than raised.
        """
        loc = attr.location
        if not attr.args:
            problems.append(f"{loc}: Missing program name in [program(...)]")
            return
        name = attr.args[0]

        # rule 8: duplicate program name (already enforced -- now with a
        # source location for both the original and the repeat definition)
        if (prev := program_locs.get(name)) is not None:
            problems.append(f"{loc}: Program '{name}' is already defined (first defined at {prev})")
            return
        program_locs[name] = loc

        ok = True
        # rules 1 + 2: every kwarg must be a known stage alias, and no two
        # aliases may target the same stage (grouping first is what lets
        # both be diagnosed instead of one silently overwriting the other)
        by_stage: dict[ShaderStage, list[tuple[str, str]]] = {}
        for token, entry_name in attr.kwargs.items():
            if (stage := _SHADER_STAGE_ALIASES.get(token)) is None:
                valid = ', '.join(sorted(_SHADER_STAGE_ALIASES))
                suggestion = _did_you_mean(token, sorted(_SHADER_STAGE_ALIASES))
                problems.append(f"{loc}: Program '{name}': unknown stage kwarg '{token}' (valid: {valid}).{suggestion}")
                ok = False
                continue
            by_stage.setdefault(stage, []).append((token, entry_name))

        stages: dict[ShaderStage, FunctionDef] = {}
        for stage, entries in by_stage.items():
            if len(entries) > 1:
                shown = ', '.join(f"{k}='{v}'" for k, v in entries)
                problems.append(f"{loc}: Program '{name}': multiple aliases target the {stage.value} stage ({shown}); use only one")
                ok = False
                continue
            (alias, entry_name), = entries

            # rule 6b: compute is dispatched via a kernel, never linked into
            # a program -- reject the alias outright, before even looking
            # the entry point up (it's illegal regardless of what it names)
            if stage not in _RASTER_STAGES:
                problems.append(f"{loc}: Program '{name}': {alias}='{entry_name}' -- compute shaders are dispatched as kernels, not linked into a [program(...)]")
                ok = False
                continue

            # rule 3: named entry point must exist
            if not (candidates := self.funcs.keyed_items.get(entry_name)):
                suggestion = _did_you_mean(entry_name, sorted(self.funcs.keyed_items))
                problems.append(f"{loc}: Program '{name}': {alias}='{entry_name}' -- no such function.{suggestion}")
                ok = False
                continue
            if len(candidates) > 1:
                problems.append(f"{loc}: Program '{name}': {alias}='{entry_name}' is ambiguous ({len(candidates)} overloads of '{entry_name}' exist)")
                ok = False
                continue
            fn = candidates[0]

            # rule 5: the entry point must have a stage at all (not a plain helper)
            if fn.stage is None:
                problems.append(f"{loc}: Program '{name}': {alias}='{entry_name}' names a plain helper function with no [shader(...)] stage")
                ok = False
                continue

            # rule 4: its declared stage must match the slot it was named in
            if fn.stage != stage:
                problems.append(
                    f"{loc}: Program '{name}': {alias}='{entry_name}' is in the {stage.value} slot, "
                    f"but '{entry_name}' is declared [shader('{fn.stage.value}')], not {stage.value}"
                )
                ok = False
                continue

            stages[stage] = fn

        # rule 6: stage-combination legality
        if ShaderStage.VERT not in by_stage:
            problems.append(f"{loc}: Program '{name}' requires a vertex stage entry point (vert=...); none was given")
            ok = False
        if ShaderStage.TESC in stages and ShaderStage.TESE not in stages:
            problems.append(f"{loc}: Program '{name}' declares a tesc stage without a tese stage; GL requires both or neither (tese alone is fine, tesc alone is a link error)")
            ok = False
        # NOTE: tese without tesc is deliberately legal in GL 4.x (patch size
        # comes from glPatchParameteri) -- no check rejects that combination.

        if not ok: return
        referenced.update(fn.name for fn in stages.values())
        self.programs[name] = PipelineDef(name, stages, loc)

    def _warn_orphan_stage_functions(self, referenced: set[str]) -> None:
        """Rule 7: warn (never error) on a raster-stage function that no
        validated [program(...)] references -- almost always a typo'd entry
        point or a stale one left behind after a rename. Compute functions
        are exempt: they're reached via a kernel, not a [program(...)]."""
        for func in self.funcs.items:
            if func.stage is None or func.stage not in _RASTER_STAGES or func.name in referenced: continue
            logger.warning(
                "%s: function '%s' has stage '%s' but is not referenced by any [program(...)] -- likely a typo or a stale entry point",
                SourceLocation(self.name, func.line_start), func.name, func.stage.value,
            )

    # ----- per-function attribute resolution -----
    #
    # Stage-dependent attributes (markers, frag/geom/tesc/tese, numthreads, glsl)
    # can't resolve until a function's stage is known. That resolution happens here, once per
    # function, building a StageConfig that flattens into `func.config` (see StageConfig.emit()).

    # AttrSpec rows for the stage-settings "meta" attributes are named after the stage token
    # itself (e.g. the row for `[geom(...)]` is named 'geom').
    _META_NAMES: dict[ShaderStage, str] = {
        ShaderStage.FRAG: 'frag', ShaderStage.GEOM: 'geom',
        ShaderStage.TESC: 'tesc', ShaderStage.TESE: 'tese',
    }

    def _meta_spec(self, stage: ShaderStage | None) -> AttrSpec | None:
        if stage is None or (name := self._META_NAMES.get(stage)) is None: return None
        for spec in REGISTRY.rows(name):
            if spec.stages and stage in spec.stages: return spec
        return None

    def _process_function_attrs(self) -> None:
        for func in self.funcs.items:
            self._resolve_func(func)

    def _resolve_func(self, func: FunctionDef) -> None:
        stage_config = StageConfig()
        meta_spec = self._meta_spec(func.stage)
        settings: dict[str, Any] = {p.name: p.default for p in meta_spec.params} if meta_spec else {}

        for attr in func.attrs:
            canonical = REGISTRY.canonical(attr.name)

            # direct use of the stage-settings meta attribute itself, e.g.
            # `[geom(in='points', max_verts=6)]` on a geometry function --
            # only the keys actually supplied are updated (unsupplied keys
            # keep whatever default/earlier value `settings` already holds)
            if meta_spec and canonical == meta_spec.name:
                settings.update(bind_update(meta_spec.params, attr, REGISTRY.canonical))
                continue

            spec = REGISTRY.resolve_stage(canonical, func.stage, attr.location, self.diagnostics)
            if spec is None: continue  # unknown / wrong-stage -- diagnosed already (dropped only when strict=False)

            if spec.handler is not None:
                # numthreads / glsl -- imperative, but deferred
                # until a StageConfig exists to receive their contribution
                bound = bind_params(spec.params, attr, REGISTRY.canonical, spec.variadic)
                ctx = AttrCtx(shader_name=self.name, diagnostics=self.diagnostics, attr=attr, func=func, stage_config=stage_config)
                spec.handler(ctx, bound)
                continue

            if spec.sets is not None:
                # bare/value marker -- assign one settings-slot
                slot, value = spec.sets
                if value is USE_ARG:
                    bound = bind_params(spec.params, attr, REGISTRY.canonical)
                    value = bound['value']
                settings[slot] = value
                continue

            logger.warning("Attribute '%s' resolved but has neither a handler nor `sets` -- ignoring", attr.name)

        # Render settings into layout qualifiers once per direction, so e.g. `[quads][triangles]`
        # on the same function lets the last marker win instead of emitting two contradictory lines.
        if meta_spec:
            by_direction: dict[str, dict[str, str | None]] = {}
            for p in meta_spec.params:
                if p.direction is None or p.render is None: continue
                if (tok := p.render(settings[p.name])) is None: continue
                by_direction.setdefault(p.direction, {})[tok[0]] = tok[1]
            for direction, tokens in by_direction.items():
                stage_config.set_layout(
                    direction, tokens, origin=f"stage defaults for '{func.stage.value}'" if func.stage else 'stage defaults',
                )

        func.config = stage_config.emit()

    # -- second pass: resolves every [uses(...)] reference against the
    #    merged interface table, emits GLSL into func.config, validates V1-V6 --

    _PIPELINE_ORDER: tuple[ShaderStage, ...] = (
        ShaderStage.VERT, ShaderStage.TESC, ShaderStage.TESE, ShaderStage.GEOM, ShaderStage.FRAG,
    )

    def resolve_interfaces(self, table: InterfaceTable) -> None:
        """Resolves func.iface_refs against `table`, emits GLSL, validates
        V1-V6. Collects every problem and raises once, joined, under strict."""
        self.resolved_interfaces = table
        problems: list[str] = []
        quiet = Diagnostics(strict=False)

        for func in self.funcs.items:
            self._resolve_func_iface_refs(func, table, quiet, problems)
        for pdef in self.programs.values():
            self._validate_program_interfaces(pdef, table, quiet, problems)

        if problems:
            message = '\n'.join(problems)
            if self.strict: raise TlangAttributeError(message)
            for line in problems: logger.warning("%s (continuing: strict=False)", line)

    def _resolve_func_iface_refs(
        self, func: FunctionDef, table: InterfaceTable, quiet: Diagnostics, problems: list[str],
    ) -> None:
        seen: dict[str, InterfaceRef] = {}
        for ref in func.iface_refs:
            # V3: a stage may reference at most one interface per direction
            if (existing := seen.get(ref.direction)) is not None:
                problems.append(
                    f"{ref.location}: [uses('{ref.name}', dir='{ref.direction}')]: function '{func.name}' "
                    f"already declares an '{ref.direction}' interface ('{existing.name}' at {existing.location}); "
                    f"a stage may reference one interface per direction"
                )
                continue
            seen[ref.direction] = ref

            # V6: compute has no stage in/out to attach an interface to
            if func.stage is ShaderStage.COMP:
                problems.append(
                    f"{ref.location}: [uses('{ref.name}')]: function '{func.name}' is a compute stage; "
                    f"compute has no stage interface direction to attach '{ref.name}' to"
                )
                continue

            # V1: name must resolve against the merged table
            if (decl := table.resolve(ref.name, ref.location, quiet)) is None:
                suggestion = _did_you_mean(ref.name, sorted(d.name for d in table))
                problems.append(
                    f"{ref.location}: [uses('{ref.name}')]: no interface named '{ref.name}' is declared "
                    f"or included.{suggestion}"
                )
                continue

            # V2: only a varyings interface may be referenced this way
            if decl.kind is not InterfaceKind.VARYINGS:
                problems.append(
                    f"{ref.location}: [uses('{ref.name}')]: '{ref.name}' is a {decl.kind.value} interface "
                    f"(declared {decl.location}), not varyings; only a [varyings] declaration can be "
                    f"referenced with [uses(...)]"
                )
                continue

            arrayed = func.stage is not None and is_arrayed(func.stage, ref.direction)
            try:
                lines = emit_glsl(decl, direction=ref.direction, arrayed=arrayed)
            except (TlangSyntaxError, TlangAttributeError) as e:
                problems.append(str(e))  # V5: emit_glsl already located the message
                continue

            func.config.extend(lines)

    def _validate_program_interfaces(
        self, pdef: PipelineDef, table: InterfaceTable, quiet: Diagnostics, problems: list[str],
    ) -> None:
        """V4: pairs consecutive raster stages and compares interfaces.
        A stage referencing none at all is the raw-GLSL escape hatch, not an error."""
        present = [s for s in self._PIPELINE_ORDER if s in pdef.stages]
        for stage_a, stage_b in zip(present, present[1:]):
            fn_a, fn_b = pdef.stages[stage_a], pdef.stages[stage_b]
            out_ref = next((r for r in fn_a.iface_refs if r.direction == 'out'), None)
            in_ref = next((r for r in fn_b.iface_refs if r.direction == 'in'), None)
            if out_ref is None or in_ref is None: continue

            if out_ref.name != in_ref.name:
                decl_a = table.resolve(out_ref.name, out_ref.location, quiet)
                declared_at = decl_a.location if decl_a else out_ref.location
                suggestion = _did_you_mean(in_ref.name, sorted(d.name for d in table))
                problems.append(
                    f"{self.name}: program '{pdef.name}': {stage_a.value} stage '{fn_a.name}' writes "
                    f"interface '{out_ref.name}' (declared {declared_at}) but {stage_b.value} stage "
                    f"'{fn_b.name}' reads '{in_ref.name}' (used at {in_ref.location}).{suggestion}"
                )
                continue

            decl_a = table.resolve(out_ref.name, out_ref.location, quiet)
            decl_b = table.resolve(in_ref.name, in_ref.location, quiet)
            if decl_a is None or decl_b is None or decl_a.signature == decl_b.signature: continue

            a_txt, b_txt = self._first_signature_diff(decl_a, decl_b)
            problems.append(
                f"{self.name}: program '{pdef.name}': '{out_ref.name}' is declared differently in two "
                f"modules: {decl_a.location} has '{a_txt}' but {decl_b.location} has '{b_txt}'"
            )

    @staticmethod
    def _first_signature_diff(decl_a: InterfaceDecl, decl_b: InterfaceDecl) -> tuple[str, str]:
        def render(m) -> str:
            raw = f"{' '.join(m.qualifiers)} {m.type_name} {m.name}{m.array}"
            return ' '.join(raw.split())

        a_members, b_members = decl_a.members, decl_b.members
        for i in range(max(len(a_members), len(b_members))):
            a_txt = render(a_members[i]) if i < len(a_members) else '<nothing>'
            b_txt = render(b_members[i]) if i < len(b_members) else '<nothing>'
            if a_txt != b_txt: return a_txt, b_txt
        return '', ''
