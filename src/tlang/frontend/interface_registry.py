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


@dataclass(frozen=True, slots=True)
class InterfaceMember:
    type_name: str
    name: str
    array: str = ''                    # '' | '[]' | '[16]', exactly as written
    qualifiers: tuple[str, ...] = ()
    line: int = 0                      # 1-based source line the declarator sits on


@dataclass(frozen=True, slots=True)
class InterfaceDecl:
    name: str
    kind: InterfaceKind
    members: tuple[InterfaceMember, ...]
    module: str
    line: int                          # 1-based line of the `struct` keyword
    layout: str = ''                   # 'std430' | 'std140' | ''
    locations: bool = True             # False => emit without layout(location=N)
    block: bool = False                # uniforms only: True => UBO block form
    source_member: str = ''            # set only by the [buffer] single-declarator shorthand:
                                        # the member `name` was derived (or overridden) from,
                                        # so a duplicate-name diagnostic can name it

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.module, self.line)

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
# struct parsing
# ---------------------------------------------------------------------------

_BARE_STRUCT_RE = re.compile(r'\bstruct\b')
_STRUCT_HEADER_RE = re.compile(r'\bstruct\s+(\w+)\s*\{')
_TAIL_RE = re.compile(r'\s*;')
_DECLARATOR_RE = re.compile(r'^\s*(\w+)\s*(\[[^\]]*\])?\s*$')

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
) -> list[InterfaceMember]:
    """One struct body statement -> one or more members.

    Runs on masked text so an embedded comment cannot be read as part of the
    declaration; names are sliced from `src` at the same offsets.
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

        if not (dm := _DECLARATOR_RE.match(part)):
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


def _derive_block_name(member_name: str) -> str:
    """`ptcPositions` -> `PtcPositions`: upper-case the first character only."""
    return member_name[:1].upper() + member_name[1:]


def _find_top_level_semicolon(mask: str, start: int) -> int | None:
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == '{': depth += 1
        elif mask[i] == '}': depth -= 1
        elif mask[i] == ';' and depth == 0: return i
    return None


def parse_declarator_at(
    src: str, start_offset: int, module: str, kind: InterfaceKind,
    *, block_name: str | None = None, **opts: Any,
) -> tuple[InterfaceDecl, int] | None:
    """Locate a single `Type name[...];` member statement at/after
    `start_offset` -- the `[buffer]` single-declarator shorthand, which
    desugars e.g. `[buffer] vec2 ptcPositions[];` to the equivalent
    `[buffer(std430)] struct PtcPositions { vec2 ptcPositions[]; };`.

    `block_name`, when given, overrides the name derived from the member
    (`[buffer(name='ElementCount')]`); otherwise it's `_derive_block_name`
    of the member's own name.

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
    if not stmt.strip():
        raise TlangAttributeError(
            "[buffer]: shorthand declaration is empty -- expected a single "
            "'Type name[...];' member before the ';'",
            loc,
        )

    members = _parse_member_statement(stmt, start_offset, src, module, '<buffer declarator>')
    if len(members) != 1:
        found = ', '.join(f'{m.type_name} {m.name}{m.array}' for m in members)
        raise TlangAttributeError(
            f"[buffer]: shorthand declares {len(members)} members ({found}) -- a buffer block "
            f"has exactly one name, so multiple declarators here are ambiguous; give each its "
            f"own block, or use the struct form: '[buffer(...)]\\nstruct Name {{ ... }};'",
            loc,
        )
    member = members[0]

    name = block_name if block_name is not None else _derive_block_name(member.name)
    if not name or not name.isidentifier():
        raise TlangAttributeError(
            f"[buffer]: '{name!r}' is not a valid block name for member '{member.name}' -- "
            f"give a valid identifier with [buffer(name='...')]",
            loc,
        )

    decl = InterfaceDecl(
        name=name, kind=kind, members=(member,), module=module, line=line,
        layout=opts.get('layout', ''), locations=opts.get('locations', True),
        block=opts.get('block', False), source_member=member.name,
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
    lines = [f'{header}{keyword} {decl.name} {{']
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
                # a shorthand-derived name never appears literally in its own
                # source line, so name the member it came from or the error
                # points at a symbol the author can't find
                return f" (derived from [buffer] member '{d.source_member}')" if d.source_member else ''
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
]
