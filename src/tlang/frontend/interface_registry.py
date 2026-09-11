# -------------------------------------------------------------
# @file          interface_registry.py
# @author        Priyangkar Ghosh
# @created       2026-09-08
# @description   Parses `[varyings]`/`[uniforms]`/`[buffer]` struct
#                declarations, measures GLSL location spans, and
#                emits the equivalent flat GLSL. GL-free: pure text.
# @license       MIT
# -------------------------------------------------------------

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import difflib
import regex as re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tlang.frontend.attribute_registry import Diagnostics
from tlang.errors import SourceLocation, TlangAttributeError, TlangSyntaxError
from tlang.shader_stages import ShaderStage
from tlang.shader_utils import mask_comments_and_strings


class InterfaceKind(str, Enum):
    VARYINGS = 'varyings'
    UNIFORMS = 'uniforms'
    BUFFER = 'buffer'
    EXTERN = 'extern'   # not an interface block -- see ExternConst below. Shares this enum
                         # only so `parse_declarator_at` can serve both with one label scheme.


@dataclass(frozen=True, slots=True)
class InterfaceMember:
    type_name: str
    name: str
    array: str = ''                    # '' | '[]' | '[16]', exactly as written
    qualifiers: tuple[str, ...] = ()
    line: int = 0                      # 1-based source line the declarator sits on
    default: str = ''                  # raw '= <expr>' text, unparsed -- '' when absent.
                                        # Only ever populated when the caller opted in via
                                        # `allow_default=True` (see parse_declarator_at); every
                                        # other caller rejects a declarator carrying one.


@dataclass(frozen=True, slots=True)
class InterfaceDecl:
    name: str                          # the HANDLE -- what Python binds by (kernel.bindings,
                                        # bind(), BufferPool lookups, Shader.declared_blocks)
    kind: InterfaceKind
    members: tuple[InterfaceMember, ...]
    module: str
    line: int                          # 1-based line of the `struct` keyword
    layout: str = ''                   # 'std430' | 'std140' | ''
    locations: bool = True             # False => emit without layout(location=N)
    block: bool = False                # uniforms only: True => UBO block form
    emit_name: str = ''                # the GLSL block identifier actually emitted, when it
                                        # differs from `name` (the [buffer] single-declarator
                                        # shorthand only -- see `_synthesize_block_name`).
                                        # '' means "same as `name`"; use `emitted_name` below.
    source_member: str = ''            # set only by the [buffer] single-declarator shorthand:
                                        # the member `name` was derived (or overridden) from,
                                        # so a duplicate-handle diagnostic can name it

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.module, self.line)

    @property
    def emitted_name(self) -> str:
        """The identifier actually written into the generated GLSL for this block --
        `name` itself, except for the [buffer] single-declarator shorthand, which emits a
        synthesised name distinct from the handle (see `_synthesize_block_name`)."""
        return self.emit_name or self.name

    @property
    def signature(self) -> str:
        """Canonical comparable text -- member order and content, whitespace-normalised.

        tlang has no type system: matching compares this text, not resolved types.
        """
        parts = []
        for m in self.members:
            raw = f"{' '.join(m.qualifiers)} {m.type_name} {m.name}{m.array}"
            parts.append(re.sub(r'\s+', ' ', raw).strip())
        return '; '.join(parts)


# ---------------------------------------------------------------------------
# [extern]: a host-supplied constant, resolved from `ShaderManager(constants={...})` and
# emitted as a GLSL `const`. Never enters `InterfaceTable`/`resolved_interfaces` -- unlike
# varyings/uniforms/buffer, there is no cross-module signature to compare: each module just
# states what it needs, independently, against the one project-wide `constants` mapping.
# ---------------------------------------------------------------------------

EXTERN_TYPES = frozenset({'int', 'uint', 'float', 'bool'})


@dataclass
class ExternConst:
    """One `[extern] TYPE NAME [= default];` declaration. Mutable: attach time (attribute
    processing) fills in everything up to `default_value`; `ShaderProcessor.resolve_externs`
    fills in `value`/`literal`/`resolved` once `constants={...}` is known.

    `precompile` backs `[extern(precompile=[...])]` (see the module docstring below the
    plain-`[extern]` section): its presence alone is what makes this constant a variant axis,
    and it has NO default/plain artifact at all -- `constants={...}` never resolves it, and
    `resolve_externs` never emits a `uniform` (or anything else) for it. Instead `ShaderManager`
    builds one fully independent `const`-specialised `Shader` per listed value, all at
    ordinary build time -- nothing about it is on-demand or lazy. `Shader.get_kernel(name,
    THIS_NAME=value)` returns one of those precompiled variants; calling `get_kernel` without a
    value for this constant at all, or with a `value` not in `precompile`, is a build-time-
    shaped error (see `Shader.get_kernel`) -- selecting one is mandatory, not optional.
    """
    name: str
    type_name: str      # one of EXTERN_TYPES
    module: str
    line: int
    has_default: bool = False
    default_value: Any = None   # parsed Python value of the declared default, if any
    trailing: str = ''          # same-line form only: raw text after the ';' to preserve

    # [extern(precompile=[...])] -- non-empty makes this a variant axis with NO default
    # artifact: ShaderManager builds one const-specialised Shader per value, all at ordinary
    # build time (see ShaderManager._build_precompiled_variants); selecting a value is
    # mandatory for every kernel in a module declaring one.
    precompile: tuple[Any, ...] = ()

    resolved: bool = False
    value: Any = None            # the Python value actually used (constants[name], else the default)
    literal: str = ''            # the GLSL literal text emitted for `value`

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.module, self.line)


def check_extern_value(type_name: str, value: Any) -> str | None:
    """`None` if `value` is an acceptable Python value for an `[extern]` constant declared
    `type_name`, else a short description of what's required -- for a "wrong type" message.

    `bool` is a subclass of `int` in Python, so it's checked first in every branch: a `bool`
    must never silently pass as a valid `int`/`uint`/`float` value.
    """
    is_bool = isinstance(value, bool)
    if type_name == 'bool':
        return None if is_bool else f"a bool, got {type(value).__name__} ({value!r})"
    if is_bool:
        return f"{'an' if type_name == 'int' else 'a'} {type_name}, got bool ({value!r})"
    if type_name == 'int':
        return None if isinstance(value, int) else f"an int, got {type(value).__name__} ({value!r})"
    if type_name == 'uint':
        if isinstance(value, int) and value >= 0: return None
        if isinstance(value, int): return f"a non-negative int (uint), got {value!r}"
        return f"a non-negative int (uint), got {type(value).__name__} ({value!r})"
    if type_name == 'float':
        # a Python int widens harmlessly into a float constant
        return None if isinstance(value, (int, float)) else f"a float (int also accepted), got {type(value).__name__} ({value!r})"
    raise AssertionError(f"unreachable: unknown [extern] type {type_name!r}")


def extern_literal(type_name: str, value: Any) -> str:
    """`value` (already validated by `check_extern_value`) -> the GLSL literal `tlang` emits
    for it. `float` always carries a decimal point or exponent (Python's `repr` guarantees
    this for any finite float), so it can never read as an int literal in a float context."""
    if type_name == 'bool': return 'true' if value else 'false'
    if type_name == 'uint': return f'{int(value)}u'
    if type_name == 'int': return str(int(value))
    if type_name == 'float': return repr(float(value))
    raise AssertionError(f"unreachable: unknown [extern] type {type_name!r}")


def parse_extern_default(type_name: str, raw: str) -> Any:
    """Declared `= <expr>` text -> a Python value of the right Python type for `type_name`.
    Raises plain `ValueError` (the caller attaches location/attribute context) -- this never
    touches `constants={...}`, so a malformed default is a syntax problem, not a "missing/
    wrong type" one.
    """
    text = raw.strip()
    if type_name == 'bool':
        low = text.lower()
        if low in ('true', '1'): return True
        if low in ('false', '0'): return False
        raise ValueError(f"'{text}' is not a valid bool default (use true/false)")
    if type_name in ('int', 'uint'):
        try:
            n = int(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a valid {type_name} default") from None
        if type_name == 'uint' and n < 0:
            raise ValueError(f"'{text}' is not a valid uint default (must be non-negative)")
        return n
    if type_name == 'float':
        try:
            return float(text.rstrip('fF'))  # tolerate a GLSL-style float suffix, e.g. '0.5f'
        except ValueError:
            raise ValueError(f"'{text}' is not a valid float default") from None
    raise AssertionError(f"unreachable: unknown [extern] type {type_name!r}")


def parse_extern_precompile_list(type_name: str, raw: str) -> tuple[Any, ...]:
    """`raw` is the bracketed list text from `[extern(precompile=[...])]`, e.g. `'[1, 2, 4]'`
    -- each token is parsed exactly like a declared `= default` (`parse_extern_default`), so
    the list is validated against `type_name` the same way, at attach time. Raises plain
    `ValueError` on a malformed token; the caller attaches location/attribute context, same
    convention as `parse_extern_default`.

    Every value here gets its own fully compiled `Shader` variant, built eagerly at
    `ShaderManager` build time (see `ShaderManager._build_precompiled_variants`) -- there is
    no on-demand compile, so a value NOT in this list is a `Shader.get_kernel` error, not a
    fallback.
    """
    text = raw.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    tokens = [t.strip() for t in text.split(',') if t.strip()]
    if not tokens:
        raise ValueError("'precompile=[...]' is empty -- list at least one value, or omit it")
    return tuple(parse_extern_default(type_name, t) for t in tokens)


# ---------------------------------------------------------------------------
# struct parsing
# ---------------------------------------------------------------------------

_BARE_STRUCT_RE = re.compile(r'\bstruct\b')
_STRUCT_HEADER_RE = re.compile(r'\bstruct\s+(\w+)\s*\{')
_TAIL_RE = re.compile(r'\s*;')
_DECLARATOR_RE = re.compile(r'^\s*(\w+)\s*(\[[^\]]*\])?\s*(?:=\s*(.+))?$')

# qualifiers recognised on an interface member; anything else in the leading
# word run is treated as (part of) the type, not a qualifier
_QUALIFIER_WORDS = frozenset({
    'flat', 'noperspective', 'smooth', 'centroid', 'sample', 'patch',
    'highp', 'mediump', 'lowp', 'readonly', 'writeonly', 'coherent',
    'volatile', 'restrict', 'precise', 'invariant',
})


def _match_brace(text: str, open_pos: int) -> int | None:
    """Index of the '}' matching the '{' at `open_pos`, or None if unbalanced."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0: return i
    return None


def _split_top_level(body: str) -> list[tuple[int, int]]:
    """Spans of each statement, split on top-level ';' only so a nested `{}` -- a
    struct member list -- never splits."""
    spans, depth, start = [], 0, 0
    for i, c in enumerate(body):
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        elif c == ';' and depth == 0:
            spans.append((start, i))
            start = i + 1
    return spans


def _line_at(src: str, offset: int) -> int:
    return src.count('\n', 0, offset) + 1


def _parse_member_statement(
    masked_stmt: str, stmt_offset: int, src: str, module: str, struct_name: str,
    *, allow_default: bool = False,
) -> list[InterfaceMember]:
    """One struct body statement -> one or more members.

    Runs on masked text so an embedded comment cannot be read as part of the
    declaration; names are sliced from `src` at the same offsets.

    `allow_default`, when False (every caller except `[extern]`), rejects a declarator
    carrying a '= <expr>' the same way an unparseable declarator always has -- a struct
    member or a `[buffer]` shorthand has nowhere in GLSL to put a default, so silently
    dropping one would be a worse outcome than the syntax error it already was.
    """
    consumed = 0
    qualifiers: list[str] = []
    while (qm := re.match(r'\s*(\w+)\s+', masked_stmt[consumed:])):
        word = qm.group(1)
        if word not in _QUALIFIER_WORDS: break
        qualifiers.append(word)
        consumed += qm.end()

    if not (type_m := re.match(r'\s*(\w+)', masked_stmt[consumed:])):
        line = _line_at(src, stmt_offset)
        raise TlangSyntaxError(
            f"struct '{struct_name}': malformed member declaration '{masked_stmt.strip()}'",
            SourceLocation(module, line),
        )
    type_name = type_m.group(1)
    consumed += type_m.end()

    decl_list = masked_stmt[consumed:]
    decl_list_offset = stmt_offset + consumed

    members: list[InterfaceMember] = []
    pos = 0
    for part in decl_list.split(','):
        part_offset = decl_list_offset + pos
        pos += len(part) + 1  # +1 accounts for the comma `split` consumed

        if not (dm := _DECLARATOR_RE.match(part)) or (dm.group(3) and not allow_default):
            line = _line_at(src, part_offset)
            raise TlangSyntaxError(
                f"struct '{struct_name}': malformed member declarator '{part.strip()}'",
                SourceLocation(module, line),
            )

        name_offset = part_offset + dm.start(1)
        members.append(InterfaceMember(
            type_name=type_name,
            name=src[name_offset:name_offset + len(dm.group(1))],
            array=dm.group(2) or '',
            qualifiers=tuple(qualifiers),
            line=_line_at(src, name_offset),
            default=(dm.group(3) or '').strip(),
        ))
    return members


def parse_struct_at(
    src: str, start_offset: int, module: str, kind: InterfaceKind, **opts: Any,
) -> tuple[InterfaceDecl, int] | None:
    """Locate `struct Name { ... };` at/after `start_offset`.

    Returns `None` only when no `struct` follows at all; a malformed one raises
    `TlangSyntaxError`. `opts` supplies `layout`/`locations`/`block`.
    """
    mask = mask_comments_and_strings(src)

    if not (bare := _BARE_STRUCT_RE.search(mask, start_offset)):
        return None
    line = _line_at(src, bare.start())
    loc = SourceLocation(module, line)

    if not (header := _STRUCT_HEADER_RE.match(mask, bare.start())):
        raise TlangSyntaxError("expected 'struct Name { ... };' here", loc)
    name = header.group(1)
    open_brace = header.end() - 1

    if (close_brace := _match_brace(mask, open_brace)) is None:
        raise TlangSyntaxError(f"struct '{name}': unterminated -- no matching '}}'", loc)

    if not (tail := _TAIL_RE.match(mask, close_brace + 1)):
        found = mask[close_brace + 1:close_brace + 21].strip().split()
        found_desc = found[0] if found else '<end of file>'
        raise TlangSyntaxError(
            f"struct '{name}': expected ';' immediately after the closing '}}' "
            f"(a flat interface struct takes no instance name), found '{found_desc}'",
            loc,
        )
    end_offset = tail.end()

    body = mask[open_brace + 1:close_brace]
    members: list[InterfaceMember] = []
    for start, end in _split_top_level(body):
        stmt = body[start:end]
        if not stmt.strip(): continue
        members.extend(_parse_member_statement(stmt, open_brace + 1 + start, src, module, name))

    if not members:
        raise TlangSyntaxError(f"struct '{name}' declares no members", loc)

    decl = InterfaceDecl(
        name=name, kind=kind, members=tuple(members), module=module, line=line,
        layout=opts.get('layout', ''), locations=opts.get('locations', True),
        block=opts.get('block', False),
    )
    return decl, end_offset


def _synthesize_block_name(member_name: str) -> str:
    """Deterministic GLSL block identifier for the [buffer] single-declarator shorthand,
    e.g. `ptcPositions` -> `ptcPositions__blk`.

    GLSL requires a block's own name to differ from its member's, so the shorthand can't
    just emit `buffer ptcPositions { vec2 ptcPositions[]; }` -- that shadows the member and
    fails to compile (verified: "undefined variable"). This name is purely a GLSL-legality
    artifact: nothing in Python ever binds by it (see `InterfaceDecl.name`, the handle).
    Deriving it from the member name keeps builds reproducible and driver errors greppable,
    and the `__` marker makes it unmistakably tlang-synthesised rather than user-written.
    """
    return f'{member_name}__blk'


def _find_top_level_semicolon(mask: str, start: int) -> int | None:
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == '{': depth += 1
        elif mask[i] == '}': depth -= 1
        elif mask[i] == ';' and depth == 0: return i
    return None


def parse_declarator_at(
    src: str, start_offset: int, module: str, kind: InterfaceKind,
    *, block_name: str | None = None, allow_default: bool = False, **opts: Any,
) -> tuple[InterfaceDecl, int] | None:
    """Locate a single `Type name[...];` (or, with `allow_default=True`, `Type name = expr;`)
    member statement at/after `start_offset`.

    Two callers share this: the `[buffer]` single-declarator shorthand, which desugars e.g.
    `[buffer] vec2 ptcPositions[];` to a block whose HANDLE (what Python binds by) is
    `ptcPositions` -- the member's own name -- and whose emitted GLSL block name is a
    synthesised one, distinct from the member so the block never shadows it (see
    `_synthesize_block_name`); and `[extern]`, which never becomes a block at all -- its
    caller reads `.members[0]` straight off the returned `InterfaceDecl` and discards the
    rest (see `ExternConst`). Messages below are generic across every `kind` except where a
    `[buffer]`-specific fix (a block, `name=...`) is actually being suggested.

    `block_name`, when given, overrides the handle (`[buffer(name='ElementCount')]`
    makes `ElementCount` what Python binds by); otherwise the handle is the
    member's own name, unchanged. Either way the emitted block name is always
    synthesised from the member -- `name=...` no longer needs to defeat a
    capitalisation collision, since there is no longer a capitalisation step.

    Returns `None` only when no statement-terminating ';' follows at all,
    mirroring `parse_struct_at`'s "nothing here" contract; a malformed or
    multi-declarator statement raises.
    """
    mask = mask_comments_and_strings(src)
    if (end := _find_top_level_semicolon(mask, start_offset)) is None:
        return None

    stmt = mask[start_offset:end]
    line = _line_at(src, start_offset)
    loc = SourceLocation(module, line)
    label = f'[{kind.value}]'
    if not stmt.strip():
        raise TlangAttributeError(
            f"{label}: shorthand declaration is empty -- expected a single "
            f"'Type name[...];' member before the ';'",
            loc,
        )

    members = _parse_member_statement(
        stmt, start_offset, src, module, '<buffer declarator>', allow_default=allow_default,
    )
    if len(members) != 1:
        found = ', '.join(f'{m.type_name} {m.name}{m.array}' for m in members)
        if kind is InterfaceKind.BUFFER:
            raise TlangAttributeError(
                f"[buffer]: shorthand declares {len(members)} members ({found}) -- a buffer block "
                f"has exactly one name, so multiple declarators here are ambiguous; give each its "
                f"own block, or use the struct form: '[buffer(...)]\\nstruct Name {{ ... }};'",
                loc,
            )
        raise TlangAttributeError(
            f"{label}: declares {len(members)} members ({found}) -- exactly one name is allowed "
            f"here; declare each on its own '{label}' line",
            loc,
        )
    member = members[0]

    handle = block_name if block_name is not None else member.name
    if not handle or not handle.isidentifier():
        if kind is InterfaceKind.BUFFER:
            raise TlangAttributeError(
                f"[buffer]: '{handle!r}' is not a valid handle for member '{member.name}' -- "
                f"give a valid identifier with [buffer(name='...')]",
                loc,
            )
        raise TlangAttributeError(f"{label}: '{handle!r}' is not a valid name for '{member.name}'", loc)

    decl = InterfaceDecl(
        name=handle, kind=kind, members=(member,), module=module, line=line,
        layout=opts.get('layout', ''), locations=opts.get('locations', True),
        block=opts.get('block', False), source_member=member.name,
        emit_name=_synthesize_block_name(member.name) if kind is InterfaceKind.BUFFER else '',
    )
    return decl, end + 1


# ---------------------------------------------------------------------------
# location spans -- a lookup table over GLSL's builtin types, not a type system
# ---------------------------------------------------------------------------

_SINGLE_LOCATION_TYPES = frozenset({
    'bool', 'float', 'int', 'uint',
    'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
    'uvec2', 'uvec3', 'uvec4', 'bvec2', 'bvec3', 'bvec4',
    'double', 'dvec2',
})
_DOUBLE_LOCATION_TYPES = frozenset({'dvec3', 'dvec4'})
_MAT_RE = re.compile(r'^mat(\d)(?:x(\d))?$')
_DMAT_RE = re.compile(r'^dmat(\d)(?:x(\d))?$')
_SIZED_ARRAY_RE = re.compile(r'^\[(\d+)\]$')


def _base_span(type_name: str) -> int | None:
    if type_name in _SINGLE_LOCATION_TYPES: return 1
    if type_name in _DOUBLE_LOCATION_TYPES: return 2
    if (m := _MAT_RE.match(type_name)): return int(m.group(1))
    if (m := _DMAT_RE.match(type_name)):
        cols = int(m.group(1))
        rows = int(m.group(2)) if m.group(2) else cols
        return cols * (2 if rows > 2 else 1)
    return None


def location_span(type_name: str, array: str) -> int | None:
    """Number of `layout(location=)` slots this member consumes, or `None`
    if it can't be measured (unsized array, unknown/user type, ...)."""
    if (base := _base_span(type_name)) is None: return None
    if not array: return base
    if array == '[]': return None
    if not (m := _SIZED_ARRAY_RE.match(array)): return None
    return base * int(m.group(1))


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

# per-vertex interfaces are arrayed (one array slot per adjacent vertex) on
# exactly these (stage, direction) pairs -- emitting the non-arrayed form
# there is GLSL that fails to compile, not merely a stylistic mismatch
_ARRAYED_IN = frozenset({ShaderStage.GEOM, ShaderStage.TESC, ShaderStage.TESE})
_ARRAYED_OUT = frozenset({ShaderStage.TESC})


def is_arrayed(stage: ShaderStage, direction: str) -> bool:
    if direction == 'in': return stage in _ARRAYED_IN
    if direction == 'out': return stage in _ARRAYED_OUT
    return False


def member_locations(decl: InterfaceDecl, base_location: int = 0) -> dict[str, int]:
    """Member name -> assigned location, mirroring what `emit_glsl` writes.

    Empty when the interface opts out of locations or a member has no
    measurable span.
    """
    if not decl.locations: return {}
    out, loc = {}, base_location
    for m in decl.members:
        if (span := location_span(m.type_name, m.array)) is None: return {}
        out[m.name] = loc
        loc += span
    return out


def _emit_varyings(decl: InterfaceDecl, direction: str, base_location: int, arrayed: bool) -> list[str]:
    lines = []
    loc = base_location
    for m in decl.members:
        if arrayed and m.array:
            raise TlangAttributeError(
                f"interface '{decl.name}' member '{m.type_name} {m.name}{m.array}' already "
                f"declares an array; a per-vertex arrayed interface can't add another array "
                f"dimension here -- use [glsl(...)] for an array-of-arrays interface",
                SourceLocation(decl.module, m.line),
            )
        prefix = ''
        if decl.locations:
            if (span := location_span(m.type_name, m.array)) is None:
                raise TlangSyntaxError(
                    f"interface '{decl.name}' member '{m.type_name} {m.name}{m.array}' has no "
                    f"measurable location span; add [varyings(locations=false)] or give the "
                    f"array a literal size",
                    SourceLocation(decl.module, m.line),
                )
            prefix = f'layout(location = {loc}) '
            loc += span
        quals = ''.join(f'{q} ' for q in m.qualifiers)
        suffix = '[]' if arrayed else ''
        lines.append(f'{prefix}{quals}{direction} {m.type_name} {m.name}{m.array}{suffix};')
    return lines


def _emit_loose(decl: InterfaceDecl, keyword: str) -> list[str]:
    lines = []
    for m in decl.members:
        quals = ''.join(f'{q} ' for q in m.qualifiers)
        lines.append(f'{quals}{keyword} {m.type_name} {m.name}{m.array};')
    return lines


def _emit_block(decl: InterfaceDecl, keyword: str) -> list[str]:
    header = f'layout({decl.layout}) ' if decl.layout else ''
    lines = [f'{header}{keyword} {decl.emitted_name} {{']
    for m in decl.members:
        quals = ''.join(f'{q} ' for q in m.qualifiers)
        lines.append(f'    {quals}{m.type_name} {m.name}{m.array};')
    lines.append('};')
    return lines


def emit_glsl(
    decl: InterfaceDecl, *, direction: str | None = None, base_location: int = 0,
    arrayed: bool = False,
) -> list[str]:
    """Render `decl` to flat GLSL lines. `direction` and `arrayed` only
    apply to `VARYINGS`; both are ignored for the other kinds."""
    if decl.kind is InterfaceKind.VARYINGS:
        if direction not in ('in', 'out'):
            raise TlangAttributeError(
                f"emitting varyings interface '{decl.name}' requires direction='in' or 'out'",
                decl.location,
            )
        return _emit_varyings(decl, direction, base_location, arrayed)
    if decl.kind is InterfaceKind.UNIFORMS:
        return _emit_block(decl, 'uniform') if decl.block else _emit_loose(decl, 'uniform')
    return _emit_block(decl, 'buffer')  # InterfaceKind.BUFFER


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

class InterfaceTable:
    """Name -> InterfaceDecl, with duplicate detection and did-you-mean resolution."""

    def __init__(self) -> None:
        self._decls: dict[str, InterfaceDecl] = {}

    def add(self, decl: InterfaceDecl) -> None:
        if (existing := self._decls.get(decl.name)) is not None:
            def hint(d: InterfaceDecl) -> str:
                # this handle is a [buffer] shorthand's member name (or its name='...'
                # override), not a block declaration the author can grep the emitted
                # GLSL for -- name the member so the duplicate is easy to find
                return f" (the [buffer] shorthand handle for member '{d.source_member}')" if d.source_member else ''
            raise TlangAttributeError(
                f"interface '{decl.name}' is declared twice in this module "
                f"(first at {existing.location}{hint(existing)}, again at {decl.location}{hint(decl)})",
                decl.location,
            )
        self._decls[decl.name] = decl

    def merged_with(self, other: 'InterfaceTable') -> 'InterfaceTable':
        """Union of both tables; a module's own declarations shadow included ones."""
        merged = InterfaceTable()
        merged._decls.update(other._decls)
        merged._decls.update(self._decls)
        return merged

    def resolve(self, name: str, loc: SourceLocation | None, diagnostics: Diagnostics) -> InterfaceDecl | None:
        if (decl := self._decls.get(name)) is not None:
            return decl
        diagnostics.fail(f"no interface named '{name}' is declared or included.{self._suggest(name)}", loc)
        return None

    def _suggest(self, name: str) -> str:
        matches = difflib.get_close_matches(name, sorted(self._decls), n=1)
        return f" Did you mean '{matches[0]}'?" if matches else ''

    def __iter__(self):
        return iter(self._decls.values())

    def __len__(self) -> int:
        return len(self._decls)

    def __contains__(self, name: str) -> bool:
        return name in self._decls


__all__ = [
    'InterfaceKind', 'InterfaceMember', 'InterfaceDecl',
    'parse_struct_at', 'parse_declarator_at', 'location_span', 'member_locations', 'is_arrayed',
    'emit_glsl', 'InterfaceTable',
    'ExternConst', 'EXTERN_TYPES', 'check_extern_value', 'extern_literal', 'parse_extern_default',
    'parse_extern_precompile_list',
]
