# -------------------------------------------------------------
# @file          debug_print.py
# @author        Priyangkar Ghosh
# @created       2026-09-10
# @description   Codegen for the `print(...)` debug-log built-in: GLSL overloads (1..8 args,
#                uint/int/float/bool) plus the SSBO block they append records to. Release builds
#                get empty-bodied overloads (the driver strips them for free); debug builds get
#                real bodies, but only in an artifact that actually calls `print(`.
# @license       MIT
# -------------------------------------------------------------

import itertools

import regex as re

from tlang.runtime.debug_log import (
    CURSOR_FIELD, DATA_FIELD, DEBUG_BUFFER_HANDLE, OVERFLOW_FIELD, RECORD_WORDS,
    TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_UINT,
)
from tlang.shader_utils import mask_comments_and_strings

# Wire format (record layout, type tags, buffer field names) lives in
# `tlang.runtime.debug_log`, imported above -- that module is the single source of
# truth both this GLSL-side encoder and the host-side decoder read from.
_TYPE_TAG: dict[str, int] = {'uint': TAG_UINT, 'int': TAG_INT, 'float': TAG_FLOAT, 'bool': TAG_BOOL}
_TYPES: tuple[str, ...] = ('uint', 'int', 'float', 'bool')

MAX_PRINT_ARGS = RECORD_WORDS - 1  # one header word + one word per argument

# Arities where EVERY combination of the 4 argument types gets its own overload, so a
# mixed-type call (`print(gid, depth)` -- uint, float) resolves to an exact-type overload
# instead of leaning on GLSL's implicit int/uint->float conversions, which are one-way,
# lossy about the argument's real type, and would silently reinterpret rather than
# bit-cast the value. Arity 1 never needs the cross product (nothing to mix). Arities
# 4-8 fall back to same-type-only: the full cross product there is 4**4..4**8 overloads
# for a shape of call real shaders essentially never make. A mixed call of 4+ arguments
# should cast its arguments to one common type explicitly (`float(x)`, `int(x)`, ...) to
# resolve to that arity's uniform overload.
_FULL_CROSS_ARITIES = (2, 3)


def _type_tuples(arity: int) -> list[tuple[str, ...]]:
    if arity in _FULL_CROSS_ARITIES:
        return list(itertools.product(_TYPES, repeat=arity))
    return [(t,) * arity for t in _TYPES]


def encode_header(types: tuple[str, ...]) -> int:
    """The compile-time-constant header value for a call site with these argument types,
    in call order. Decoded by `tlang.runtime.debug_log.decode_record`."""
    header = len(types)
    for i, t in enumerate(types):
        header |= _TYPE_TAG[t] << (4 + 2 * i)
    return header


def _store_expr(t: str, arg: str) -> str:
    if t == 'uint': return arg
    if t == 'int': return f'uint({arg})'
    if t == 'float': return f'floatBitsToUint({arg})'
    if t == 'bool': return f'({arg} ? 1u : 0u)'
    raise ValueError(f"unhandled print argument type: {t!r}")  # pragma: no cover -- exhaustive above


def _signature(types: tuple[str, ...]) -> str:
    return ', '.join(f'{t} a{i}' for i, t in enumerate(types))


def _overload(types: tuple[str, ...], *, real: bool, capacity: int) -> str:
    sig = _signature(types)
    if not real:
        return f'void print({sig}) {{}}\n'

    lines = [f'void print({sig}) {{']
    lines.append(f'    uint tlang_dbg_slot = atomicAdd({CURSOR_FIELD}, 1u);')
    lines.append(f'    if (tlang_dbg_slot >= {capacity}u) {{')
    lines.append(f'        atomicAdd({OVERFLOW_FIELD}, 1u);')
    lines.append('        return;')
    lines.append('    }')
    lines.append(f'    uint tlang_dbg_base = tlang_dbg_slot * {RECORD_WORDS}u;')
    lines.append(f'    {DATA_FIELD}[tlang_dbg_base] = {encode_header(types)}u;')
    for i, t in enumerate(types):
        lines.append(f'    {DATA_FIELD}[tlang_dbg_base + {i + 1}u] = {_store_expr(t, f"a{i}")};')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def render_overloads(*, real: bool, capacity: int) -> str:
    """Every `print(...)` overload (arity 1..MAX_PRINT_ARGS), one signature per line group.

    `real=False` (release, or a debug artifact that never calls `print`): every body is
    `{}` -- a no-op the driver is free to strip entirely. `real=True` (debug + this
    artifact actually calls `print`): every body appends a record to the log buffer.
    """
    return '\n'.join(
        _overload(types, real=real, capacity=capacity)
        for arity in range(1, MAX_PRINT_ARGS + 1)
        for types in _type_tuples(arity)
    )


def render_buffer_decl() -> str:
    """The SSBO block `print`'s real body writes into. Single unnamed-instance block, so
    `BindingRegistry.remove_dead_blocks`'s reachability scan (which searches for a bare
    field name when there's no instance name) sees exactly the identifiers the real
    overload bodies reference."""
    return (
        f'layout(std430) buffer {DEBUG_BUFFER_HANDLE} {{\n'
        f'    uint {CURSOR_FIELD};\n'
        f'    uint {OVERFLOW_FIELD};\n'
        f'    uint {DATA_FIELD}[];\n'
        '};\n'
    )


# A call site -- `print(` -- never a declaration, since this module is what defines
# `print` in the first place and the scan always runs on text from BEFORE that
# injection. Comments/strings are masked first so neither counts as a "call".
_PRINT_CALL_PATTERN = re.compile(r'\bprint\s*\(')


def contains_print_call(src: str) -> bool:
    """Whether `src` calls `print(...)` anywhere (reachable or not -- this is a cheap,
    conservative pre-DCE scan, not a reachability analysis; see `render_print_module`)."""
    return _PRINT_CALL_PATTERN.search(mask_comments_and_strings(src)) is not None


def render_print_module(*, debug: bool, capacity: int, used: bool) -> str:
    """GLSL text to append to one stage artifact's source for the `print(...)` feature.

    Must be appended BEFORE `BindingRegistry.find_missing_export_calls`/
    `remove_dead_functions`/`remove_unused_buffers` run on that source: those need
    `print` to already resolve to a real definition (or the missing-export check raises
    a false positive), and the DCE passes are what actually prunes an unused overload
    set (and, in turn, the now-unreferenced log buffer) back out per artifact.

    `used` (from `contains_print_call` on that artifact's own pre-injection source) gates
    BOTH modes identically: an artifact that never calls `print` gets nothing at all,
    release or debug -- measured on a 13-module/many-kernel real project, unconditionally
    injecting ~100 empty stub overloads into every artifact (the "release never needs to
    scan" alternative) cost real, driver-side parse time across that many compile units,
    which release-mode-must-cost-nothing rules out. The only edge `used` can miss is a
    call reaching `print` through a macro tlang never expands (`#define LOG(x)
    print(x)`) -- rare, and the fix is the same in both modes: call `print` directly, or
    write the macro so its own body contains the literal token.

    `debug=False` (release), `used=True`: every overload, empty-bodied -- the driver is
    free to eliminate a called-but-empty function entirely.

    `debug=True`, `used=True`: the real (buffer-writing) overloads plus the log buffer's
    own declaration.
    """
    if not used:
        return ''
    if not debug:
        return '\n#line 1 "TLANG_DEBUG_PRINT"\n' + render_overloads(real=False, capacity=capacity)
    return (
        '\n#line 1 "TLANG_DEBUG_PRINT"\n'
        + render_buffer_decl()
        + render_overloads(real=True, capacity=capacity)
    )


__all__ = [
    'MAX_PRINT_ARGS',
    'encode_header', 'contains_print_call', 'render_print_module',
    'render_overloads', 'render_buffer_decl',
]
