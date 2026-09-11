# -------------------------------------------------------------
# @file          attribute_registry.py
# @author        Priyangkar Ghosh
# @created       2026-09-08
# @description   Declarative registry infrastructure for attributes: one AttrSpec row per
#                (name, stage) combination.
# @license       MIT
# -------------------------------------------------------------

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import difflib
import regex as re
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable, Literal, cast

from tlang.frontend.attribute import Attribute
from tlang.errors import SourceLocation, TlangAttributeError, TlangError
from tlang.shader_stages import ShaderStage


class Scope(Flag):
    """Where in a .tlang file an attribute may legally appear."""
    GLOBAL = auto()     # a '[attr(...)]' block attached to the next function, or a project-level directive
    FUNCBODY = auto()   # a pragma written *inside* a function body
    ANY = GLOBAL | FUNCBODY


# sentinel distinguishing "no default supplied" (required argument) from a
# real default value of None/False/0/'' -- all of which are falsy but valid
_MISSING = object()

# sentinel used by a value-marker's `sets` (e.g. `[max_verts(3)]`) to mean
# "use this attribute's own single bound argument", as opposed to a marker
# like `[triangles]` whose `sets` value is the fixed literal it always writes
USE_ARG = object()

Direction = Literal['in', 'out']


# ---------------------------------------------------------------------------
# Typed parameter coercion: values are parsed into real bool/int before any handler or emit
# sees them, so e.g. `early_tests='false'` isn't a truthy string.
# ---------------------------------------------------------------------------

_BOOL_TRUE = {'true', '1', 'on', 'yes'}
_BOOL_FALSE = {'false', '0', 'off', 'no'}

def _coerce_bool(raw: str, ctx: str, loc: SourceLocation | None) -> bool:
    if (low := raw.strip().lower()) in _BOOL_TRUE: return True
    if low in _BOOL_FALSE: return False
    raise TlangAttributeError(f"{ctx}: expected a boolean (true/false/1/0/on/off), got '{raw}'", loc)

def _coerce_int(raw: str, ctx: str, loc: SourceLocation | None) -> int:
    try: return int(raw.strip())
    except ValueError:
        raise TlangAttributeError(f"{ctx}: expected an integer, got '{raw}'", loc) from None

def _coerce_str(raw: str, ctx: str, loc: SourceLocation | None) -> str: return raw

_COERCERS: dict[type, Callable[[str, str, SourceLocation | None], Any]] = {
    bool: _coerce_bool,
    int: _coerce_int,
    str: _coerce_str,
}


@dataclass(frozen=True, slots=True)
class Param:
    """One argument of an attribute.

    `choices` is checked after alias resolution, so `spacing='even'` and `spacing='equal_spacing'`
    both pass. `direction`/`render` (meaningful only on stage-settings meta rows) say which layout
    qualifier direction a slot contributes to, and render its value to a `(token, value)` pair,
    or `None` to suppress the qualifier entirely.
    """
    name: str
    type: type
    choices: tuple[str, ...] = ()
    default: Any = _MISSING
    positional: int | None = None
    direction: Direction | None = None
    render: Callable[[Any], tuple[str, str | None] | None] | None = None


@dataclass(frozen=True, slots=True)
class AttrSpec:
    """One user-visible attribute, for one stage (or stage-independent).

    The registry is a list, not a dict keyed on `name`: a name like 'triangles' can legitimately
    own two rows (one for GEOM, one for TESE), so `[quads][triangles]` resolves correctly per stage.
    """
    name: str
    scope: Scope
    stages: frozenset[ShaderStage] | None = None   # None = valid on any/no stage
    params: tuple[Param, ...] = ()
    aliases: tuple[str, ...] = ()
    handler: Callable[['AttrCtx', dict[str, Any]], None] | None = None
    deferred: bool = False       # must wait until the target function's stage/StageConfig are known
    literal: str | None = None   # bare pragmas that just emit fixed text (unroll, flatten, ...)
    sets: tuple[str, Any] | None = None  # (settings-slot, fixed value | USE_ARG) for markers
    variadic: bool = False       # accepts arbitrary/dynamic kwargs beyond `params` (e.g. [program(vert=..., frag=...)])
    summary: str = ""
    example: str = ""


@dataclass(slots=True)
class LayoutQualifier:
    direction: Direction
    tokens: dict[str, str | None] = field(default_factory=dict)
    origin: str = ''
    exclusive: bool = False                       # True once a verbatim [resourceblock] layout line claims this direction
    _origins: dict[str, str] = field(default_factory=dict)  # per-token origin, for conflict diagnostics


@dataclass(slots=True)
class RawDecl:
    text: str
    origin: str


@dataclass(slots=True)
class StageConfig:
    """Builder for one function's `config`.

    Declarations and layout qualifiers are tracked separately and merged (not appended), so a
    conflict is raised where it occurs instead of reaching the GLSL compiler as two contradictory
    lines. `emit()` flattens this into the `list[str]` that `func.config` holds.
    """
    layouts: dict[str, LayoutQualifier] = field(default_factory=dict)
    decls: list[RawDecl] = field(default_factory=list)

    def set_layout(
        self, direction: Direction, tokens: dict[str, str | None], origin: str,
        *, exclusive: bool = False, loc: SourceLocation | None = None,
    ) -> None:
        if not tokens: return
        existing = self.layouts.get(direction)
        if existing and existing.origin != origin and (exclusive or existing.exclusive):
            raise TlangAttributeError(
                f"Conflicting layout({direction}) declarations for this function: "
                f"'{origin}' collides with '{existing.origin}', which already claimed layout({direction})",
                loc,
            )
        q = existing or self.layouts.setdefault(direction, LayoutQualifier(direction))
        for key, val in tokens.items():
            if key in q._origins and q._origins[key] != origin and q.tokens.get(key) != val:
                raise TlangAttributeError(
                    f"Layout qualifier '{key}' for '{direction}' is set twice: "
                    f"first by '{q._origins[key]}', again by '{origin}'",
                    loc,
                )
            q.tokens[key] = val
            q._origins[key] = origin
        q.origin = origin
        if exclusive: q.exclusive = True

    def add_decl(self, text: str, origin: str) -> None:
        if text.strip(): self.decls.append(RawDecl(text, origin))

    def emit(self) -> list[str]:
        """Render to the flat `list[str]` that `FunctionDef.config` holds.
        Declarations first, then layout qualifiers, deterministically."""
        lines = [d.text for d in self.decls]
        for direction in ('in', 'out'):
            if not (q := self.layouts.get(direction)) or not q.tokens: continue
            parts = [k if v is None else f'{k} = {v}' for k, v in q.tokens.items()]
            lines.append(f"layout({', '.join(parts)}) {direction};")
        return lines


@dataclass(slots=True)
class AttrCtx:
    """What a handler gets instead of re-deriving its contract with
    isinstance guards. `func`/`stage_config` are only populated once a
    deferred attribute is being resolved against a known function/stage."""
    shader_name: str
    diagnostics: 'Diagnostics'
    attr: Attribute
    funcs: Any = None                 # FunctionList | None (typed Any to avoid a hard import cycle)
    index: int | None = None
    glob_attachments: list[Attribute] | None = None
    func: Any = None                  # FunctionDef | None
    stage_config: StageConfig | None = None

    # Declaration attributes ([varyings]/[uniforms]/[buffer]) consume the
    # `struct` following them, so they need the map and the last line the
    # attribute block itself occupied. Only set for global-scope dispatch.
    src_map: Any = None               # dict[int, ShaderSourceLine] | None
    end_index: int | None = None

    # Same-line shorthand support ([buffer]'s single-declarator form): the raw text
    # following this attribute's own closing ']', when it's the last attribute in a
    # contiguous run at the start of its line -- '' otherwise (global-scope block-attr
    # dispatch only; never set for '#name<args>' or function-body attributes).
    line_tail: str = ''
    # A handler sets `result` to override the default '//<<ATTR name>>//' marker text
    # emitted in its place, and `tail_consumed = True` to tell the dispatcher that
    # `line_tail` was folded into `result` itself and must not also be appended verbatim.
    result: str | None = None
    tail_consumed: bool = False


@dataclass(slots=True)
class Diagnostics:
    """Routes an attribute problem to either a hard failure or a logged
    warning, depending on `strict` -- the same flag `ShaderManager`/`Shader`
    already expose. Every message carries the offending `SourceLocation`."""
    strict: bool = True

    def fail(
        self, message: str, loc: SourceLocation | None,
        error_type: type[TlangError] = TlangAttributeError,
    ) -> None:
        err = error_type(message, loc)
        if self.strict: raise err
        logger.error("%s (continuing: strict=False)", err)


# ---------------------------------------------------------------------------
# parameter binding
# ---------------------------------------------------------------------------

def _label(attr: Attribute) -> str: return f"[{attr.name}]"

def _coerce_param(p: Param, raw: str, resolve_alias: Callable[[str], str], label: str, loc: SourceLocation | None) -> Any:
    if p.choices:
        raw = resolve_alias(raw)
        if raw not in p.choices:
            raise TlangAttributeError(f"{label}: '{p.name}' must be one of {{{', '.join(p.choices)}}} (got '{raw}')", loc)
    return _COERCERS[p.type](raw, f"{label}({p.name}=...)", loc)

def bind_params(
    params: tuple[Param, ...], attr: Attribute, resolve_alias: Callable[[str], str], variadic: bool = False,
) -> dict[str, Any]:
    """Full bind: every param gets a value (its own, or its default).

    `variadic` skips the "unknown kwarg" check, for attributes like [program(...)] whose kwargs
    are dynamic rather than a fixed, declarable set."""
    label, loc = _label(attr), attr.location
    by_name = {p.name: p for p in params}
    bound: dict[str, Any] = {}
    for p in params:
        if p.positional is not None and p.positional < len(attr.args):
            bound[p.name] = _coerce_param(p, attr.args[p.positional], resolve_alias, label, loc)
        elif p.name in attr.kwargs:
            bound[p.name] = _coerce_param(p, attr.kwargs[p.name], resolve_alias, label, loc)
        elif p.default is not _MISSING:
            bound[p.name] = p.default
        else:
            raise TlangAttributeError(f"{label}: missing required argument '{p.name}'", loc)
    if not variadic:
        for k in attr.kwargs:
            if k not in by_name:
                raise TlangAttributeError(f"{label}: unknown argument '{k}' (expected: {', '.join(by_name) or '(none)'})", loc)
    return bound

def bind_update(
    params: tuple[Param, ...], attr: Attribute, resolve_alias: Callable[[str], str],
) -> dict[str, Any]:
    """Partial bind: only params the attribute actually supplied are returned, for updating a
    settings dict that already holds defaults for everything else."""
    label, loc = _label(attr), attr.location
    by_name = {p.name: p for p in params}
    out: dict[str, Any] = {}
    for k, raw in attr.kwargs.items():
        if (p := by_name.get(k)) is None:
            raise TlangAttributeError(f"{label}: unknown argument '{k}' (expected: {', '.join(by_name) or '(none)'})", loc)
        out[p.name] = _coerce_param(p, raw, resolve_alias, label, loc)
    for p in params:
        if p.positional is not None and p.name not in out and p.positional < len(attr.args):
            out[p.name] = _coerce_param(p, attr.args[p.positional], resolve_alias, label, loc)
    return out


# ---------------------------------------------------------------------------
# Splits verbatim `layout(...) in|out;` lines (merged/conflict-checked against stage defaults)
# out of a resourceblock, from everything else (plain declarations, appended verbatim).
# ---------------------------------------------------------------------------

_LAYOUT_LINE_RE = re.compile(r'^[ \t]*layout\s*\(([^)]*)\)\s*(in|out)\s*;[ \t]*$', re.MULTILINE)

def parse_resourceblock(raw: str) -> tuple[dict[Direction, dict[str, str | None]], str]:
    claims: dict[Direction, dict[str, str | None]] = {}

    def repl(m: re.Match) -> str:
        tokens: dict[str, str | None] = {}
        for part in m.group(1).split(','):
            if not (part := part.strip()): continue
            if '=' in part:
                key, val = part.split('=', 1)
                tokens[key.strip()] = val.strip()
            else:
                tokens[part] = None
        claims.setdefault(cast(Direction, m.group(2)), {}).update(tokens)
        return ''

    remainder = _LAYOUT_LINE_RE.sub(repl, raw)
    return claims, remainder.strip('\n')


# ---------------------------------------------------------------------------
# the registry itself
# ---------------------------------------------------------------------------

class AttrRegistry:
    def __init__(self, specs: list[AttrSpec]) -> None:
        self._by_name: dict[str, list[AttrSpec]] = {}
        self._alias_to_name: dict[str, str] = {}
        for spec in specs:
            self._by_name.setdefault(spec.name, []).append(spec)
            for alias in spec.aliases:
                if (existing := self._alias_to_name.get(alias)) and existing != spec.name:
                    raise ValueError(f"attribute alias '{alias}' is claimed by both '{existing}' and '{spec.name}'")
                self._alias_to_name[alias] = spec.name

        # Every row sharing a name must agree on scope/deferred-ness: attach-time dispatch has
        # to decide "run now or push to fn.attrs" before the stage is known.
        for name, rows in self._by_name.items():
            if len({r.scope for r in rows}) > 1:
                raise ValueError(f"attribute '{name}' has rows disagreeing on scope")
            if len({r.deferred for r in rows}) > 1:
                raise ValueError(f"attribute '{name}' has rows disagreeing on deferred-ness")

        self.all_names: list[str] = sorted(self._by_name)

    def canonical(self, raw_name: str) -> str:
        return self._alias_to_name.get(raw_name, raw_name)

    def rows(self, name: str) -> list[AttrSpec]:
        return self._by_name.get(self.canonical(name), [])

    def _suggest(self, name: str) -> str:
        matches = difflib.get_close_matches(name, self.all_names, n=1)
        return f" Did you mean '{matches[0]}'?" if matches else ""

    def resolve_scope(self, raw_name: str, scope: Scope, loc: SourceLocation | None, diagnostics: Diagnostics) -> list[AttrSpec] | None:
        """Attach-time check (name known + usable in this scope). Returns the
        matching rows, or `None` if a (non-strict) diagnostic was logged and
        the caller should just drop the attribute."""
        if not (rows := self.rows(raw_name)):
            diagnostics.fail(f"Unknown attribute '{raw_name}'.{self._suggest(raw_name)}", loc)
            return None
        if not (matching := [r for r in rows if r.scope & scope]):
            other = 'a function-body pragma' if rows[0].scope == Scope.FUNCBODY else 'a global/file-level attribute'
            diagnostics.fail(f"'{raw_name}' is {other}; it cannot be used here", loc)
            return None
        return matching

    def resolve_stage(self, raw_name: str, stage: ShaderStage | None, loc: SourceLocation | None, diagnostics: Diagnostics) -> AttrSpec | None:
        """Resolve-time check (shader_processor): name known + valid for the
        target function's `stage`. Returns the one matching row, or `None`."""
        if not (rows := self.rows(raw_name)):
            diagnostics.fail(f"Unknown attribute '{raw_name}'.{self._suggest(raw_name)}", loc)
            return None
        candidates = [r for r in rows if r.stages is None or stage in r.stages]
        if not candidates:
            valid = sorted({s.value for r in rows for s in (r.stages or ())})
            where = stage.value if stage else '<unassigned stage>'
            diagnostics.fail(
                f"'{raw_name}' is not valid on a {where} function (valid on: {', '.join(valid) or 'nothing'})", loc,
            )
            return None
        return candidates[0]


# ---------------------------------------------------------------------------
# documentation generation
# ---------------------------------------------------------------------------

def _cell(text: str) -> str:
    """Flatten text for a Markdown table cell: no raw newlines, no bare pipes."""
    return re.sub(r'\s*\n\s*', ' ', text).replace('|', r'\|')


def generate_docs(registry: AttrRegistry) -> str:
    """Render the registry to a Markdown reference table."""
    lines = [
        "| Name | Scope | Stages | Params | Aliases | Summary | Example |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in registry.all_names:
        for spec in registry.rows(name):
            stages = ', '.join(s.value for s in sorted(spec.stages, key=lambda s: s.value)) if spec.stages else 'any'
            params = '; '.join(
                f"{p.name}: {p.type.__name__}"
                + (f" ∈ {{{','.join(p.choices)}}}" if p.choices else "")
                + (f" = {p.default!r}" if p.default is not _MISSING else "")
                for p in spec.params
            ) or '—'
            aliases = ', '.join(spec.aliases) or '—'
            scope = 'global' if spec.scope == Scope.GLOBAL else 'function-body'
            summary = _cell(spec.summary) or '—'
            example = '<br>'.join(f"`{line}`" for line in spec.example.splitlines()) or '—'
            lines.append(f"| `{spec.name}` | {scope} | {stages} | {params} | {aliases} | {summary} | {example} |")
    return '\n'.join(lines)


__all__ = [
    'Scope', 'Direction', 'Param', 'AttrSpec', 'LayoutQualifier', 'RawDecl', 'StageConfig',
    'AttrCtx', 'Diagnostics', 'AttrRegistry', 'USE_ARG',
    'bind_params', 'bind_update', 'parse_resourceblock', 'generate_docs',
]
