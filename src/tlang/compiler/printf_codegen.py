# -------------------------------------------------------------
# @file          printf_codegen.py
# @author        Priyangkar Ghosh
# @created       2026-09-10
# @description   Codegen for the `printf(...)` debug-log built-in: locating and rewriting
#                each call site (the format string never reaches GLSL, and every argument
#                is cast to uint AT THE CALL SITE per its own specifier -- see
#                `rewrite_printf_calls`), the GLSL overloads (one per arity 0..8, all-uint)
#                plus the ring-buffer SSBO block they append records to. Release builds get
#                empty-bodied overloads (the driver strips them for free); debug builds get
#                real bodies, but only in an artifact that actually calls `printf(`.
# @license       MIT
# -------------------------------------------------------------

import regex as re

from tlang.errors import SourceLocation, TlangAttributeError
from tlang.runtime.printf_log import (
    BUFFER_HANDLE, CAS_SPIN_LIMIT, DATA_FIELD, DROPPED_FIELD, MAX_PRINTF_ARGS, READY_FIELD,
    RECORD_WORDS, RESERVED_FIELD, VALUE_SPECIFIERS, PrintfTable,
)
from tlang.shader_utils import mask_comments_and_strings

# Wire format (record layout, field names) lives in `tlang.runtime.printf_log`, imported
# above -- that module is the single source of truth both this GLSL-side encoder and the
# host-side decoder read from.
#
# ONE overload per arity, every value parameter `uint` -- deliberately NOT a type
# cross-product. An earlier version emitted a full cross product of uint/int/float/bool
# only for arities 2-3 (matching v1's `print`) and fell back to same-type-only overloads
# for arities 4-8, on the theory that a mixed-type call that long is rare. That was a real
# bug, not just an inconvenience: `printf("%d %f %d %f\n", 7, 2.5, 9, 4.5)` has no
# same-type overload to resolve to, so GLSL silently implicit-converted the two ints to
# float and the host decoded the resulting float bit patterns as if they were ints --
# wrong values, no error, no warning. Casting each argument at the CALL SITE instead (see
# `rewrite_printf_calls`), driven by the format specifier rather than the argument's own
# GLSL type, sidesteps overload resolution (and its implicit-conversion hazard) entirely:
# every overload takes exactly `uint`s, so there is only one signature per arity, period.
def _signature(arity: int) -> str:
    params = ['uint tlang_pf_id'] + [f'uint a{i}' for i in range(arity)]
    return ', '.join(params)


def _overload(arity: int, *, real: bool, capacity: int) -> str:
    sig = _signature(arity)
    if not real:
        return f'void printf({sig}) {{}}\n'

    lines = [f'void printf({sig}) {{']
    # Bounded MPSC ring claim (Vyukov's bounded queue -- see the long comment in
    # `tlang.runtime.printf_log`'s module docstring for why a plain
    # `atomicAdd(reserved, 1u)` is NOT enough here): `tlang_pf_reserved` only ever advances
    # for a position that is ACTUALLY going to be written. A naive unconditional atomicAdd
    # would "spend" a position number on every attempt, including dropped ones -- once the
    # host has consumed past that position, nothing ever retries it, so a single overflow
    # event would strand every later (successfully written) record behind a permanent gap
    # the read side can never cross. Claiming via compare-and-swap against the slot's own
    # `tlang_pf_ready` sequence number means a dropped attempt never touches `reserved` at
    # all -- every position that DOES get claimed is guaranteed to get written.
    lines.append('    uint tlang_pf_pos = ' + RESERVED_FIELD + ';')
    lines.append('    bool tlang_pf_claimed = false;')
    lines.append('    uint tlang_pf_slot = 0u;')
    lines.append(f'    for (int tlang_pf_spin = 0; tlang_pf_spin < {CAS_SPIN_LIMIT}; ++tlang_pf_spin) {{')
    lines.append(f'        tlang_pf_slot = tlang_pf_pos % {capacity}u;')
    lines.append(f'        int tlang_pf_dif = int({READY_FIELD}[tlang_pf_slot]) - int(tlang_pf_pos);')
    lines.append('        if (tlang_pf_dif == 0) {')
    lines.append(f'            uint tlang_pf_prev = atomicCompSwap({RESERVED_FIELD}, tlang_pf_pos, tlang_pf_pos + 1u);')
    lines.append('            if (tlang_pf_prev == tlang_pf_pos) { tlang_pf_claimed = true; break; }')
    lines.append('            tlang_pf_pos = tlang_pf_prev;')
    lines.append('        } else if (tlang_pf_dif < 0) {')
    lines.append(f'            atomicAdd({DROPPED_FIELD}, 1u);')
    lines.append('            return;')
    lines.append('        } else {')
    lines.append(f'            tlang_pf_pos = {RESERVED_FIELD};')
    lines.append('        }')
    lines.append('    }')
    lines.append('    if (!tlang_pf_claimed) {')  # spin budget exhausted -- pathological contention, treat as drop
    lines.append(f'        atomicAdd({DROPPED_FIELD}, 1u);')
    lines.append('        return;')
    lines.append('    }')
    lines.append(f'    uint tlang_pf_base = tlang_pf_slot * {RECORD_WORDS}u;')
    lines.append(f'    {DATA_FIELD}[tlang_pf_base] = tlang_pf_id;')
    for i in range(arity):
        # Already converted to its final uint bit pattern at the CALL SITE (see
        # `rewrite_printf_calls`) -- stored verbatim, no further conversion here.
        lines.append(f'    {DATA_FIELD}[tlang_pf_base + {i + 1}u] = a{i};')
    # Publish ordering (see the brief): the barrier must land AFTER every data word is
    # written and BEFORE the ready flag becomes visible, or a reader could observe
    # `tlang_pf_ready[slot] == pos+1` while the body is still landing -- a torn record.
    # `memoryBarrierBuffer()` is per-invocation, not the device-wide `glMemoryBarrier`, so
    # it costs nothing close to a real device barrier's price.
    lines.append('    memoryBarrierBuffer();')
    lines.append(f'    {READY_FIELD}[tlang_pf_slot] = tlang_pf_pos + 1u;')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def render_overloads(*, real: bool, capacity: int) -> str:
    """Every `printf(...)` overload -- exactly ONE per arity, 0..MAX_PRINTF_ARGS, all-`uint`
    value parameters (see the module docstring for why: casting happens at the call site,
    not via type-matched overloads).

    `real=False` (release, or a debug artifact that never calls `printf`): every body is
    `{}` -- a no-op the driver is free to strip entirely. `real=True` (debug + this
    artifact actually calls `printf`): every body reserves a ring slot, writes a record
    (dropping and counting it instead if the ring is full -- see the module docstring in
    `tlang.runtime.printf_log`), and publishes it.
    """
    return '\n'.join(
        _overload(arity, real=real, capacity=capacity)
        for arity in range(0, MAX_PRINTF_ARGS + 1)
    )


def render_buffer_decl(capacity: int) -> str:
    """The SSBO block `printf`'s real body writes into. Single unnamed-instance block, so
    `dead_code.remove_dead_blocks`'s reachability scan (which searches for a bare
    field name when there's no instance name) sees exactly the identifiers the real
    overload bodies reference. `tlang_pf_ready` is a FIXED-size array (`capacity` is a
    compile-time constant baked in as a literal) so it can precede the block's one
    unsized trailing array, `tlang_pf_data` -- GLSL permits any number of ordinary members
    before the single unsized array that must be last.

    `tlang_pf_ready[slot]` doubles as BOTH the publish-ordering flag (section 6: a reader
    only trusts a slot once this equals `pos+1`, set after `memoryBarrierBuffer()`) AND the
    ring's flow-control handshake (section 7: the HOST writes `pos+capacity` back into it
    after consuming, which is what lets the CAS claim loop in `_overload` re-use that slot
    for the next lap) -- there is no separate host-writable "read cursor" field; see the
    module docstring in `tlang.runtime.printf_log` for why merging the two into one
    per-slot sequence number, rather than a single scalar cursor, is what keeps the ring
    gap-free.
    """
    return (
        f'layout(std430) buffer {BUFFER_HANDLE} {{\n'
        f'    uint {RESERVED_FIELD};\n'
        f'    uint {DROPPED_FIELD};\n'
        f'    uint {READY_FIELD}[{capacity}];\n'
        f'    uint {DATA_FIELD}[];\n'
        '};\n'
    )


def render_printf_module(*, debug: bool, capacity: int, used: bool) -> str:
    """GLSL text to append to one stage artifact's source for the `printf(...)` feature.

    Must be appended BEFORE `dead_code.find_missing_export_calls`/
    `remove_dead_functions`/`remove_dead_blocks` run on that source, and AFTER
    `rewrite_printf_calls` has already replaced every call site's `printf("...", ...)`
    text with `printf(<id>u, ...)` in `used`'s own source -- see `Shader._build`.

    `used` (whether `rewrite_printf_calls` found at least one call in that artifact's own
    pre-injection source) gates BOTH modes identically: an artifact that never calls
    `printf` gets nothing at all, release or debug -- same reasoning as v1's `print` (see
    git history): unconditionally injecting the full overload set into every artifact
    measurably cost real driver-side parse time across a many-kernel project.

    `debug=False` (release), `used=True`: every overload, empty-bodied -- the driver is
    free to eliminate a called-but-empty function entirely. The CALL ITSELF is never
    stripped in either mode -- only the callee's body differs.

    `debug=True`, `used=True`: the real (ring-buffer-writing) overloads plus the log
    buffer's own declaration.
    """
    if not used:
        return ''
    if not debug:
        return '\n#line 1 "TLANG_PRINTF"\n' + render_overloads(real=False, capacity=capacity)
    return (
        '\n#line 1 "TLANG_PRINTF"\n'
        + render_buffer_decl(capacity)
        + render_overloads(real=True, capacity=capacity)
    )


# ---------------------------------------------------------------------------
# Call-site scanning + rewriting -- the format string never reaches GLSL.
# ---------------------------------------------------------------------------

# A call site -- `printf(` -- never a declaration, since this module is what defines
# `printf` in the first place and the scan always runs on text from BEFORE that injection.
_PRINTF_CALL_PATTERN = re.compile(r'\bprintf\s*\(')
_SPEC_PATTERN = re.compile(r'%(.)')

_ESCAPES: dict[str, str] = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', '0': '\0'}


def _unescape(raw: str) -> str:
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == '\\' and i + 1 < n:
            out.append(_ESCAPES.get(raw[i + 1], raw[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


# `Shader.build_map` (compiler/shader.py) has already annotated `src` with `#line N
# "module"` directives by the time `rewrite_printf_calls` runs on it -- the SAME
# directives a driver compile error is matched against (see `Shader._parse_error_location`).
# A printf call site's real (module, line) has to be resolved the same way: `src` is one
# assembled translation unit that can splice in text from several original modules (a
# `[link(...)]`ed helper from elsewhere), so a naive newline count over `src` itself would
# report the wrong module entirely, not just the wrong line, for a call inside such a helper.
_LINE_DIRECTIVE = re.compile(r'^#line\s+(\d+)\s+"([^"]*)"\s*$', re.MULTILINE)


class _LineIndex:
    """One `#line`-directive index built once per `rewrite_printf_calls` call, then queried
    for every match found in the same `src` -- O(directives) to build, O(log directives) . 1
    per lookup would be overkill for a source file's worth of call sites, so this just keeps
    the (text_line, original_line, module) triples sorted and does a linear scan; `src` is
    one shader entry point's assembled unit, not a whole project."""

    __slots__ = ('_entries', '_fallback_module')

    def __init__(self, src: str, fallback_module: str) -> None:
        self._fallback_module = fallback_module
        self._entries: list[tuple[int, int, str]] = []
        for m in _LINE_DIRECTIVE.finditer(src):
            text_line = src.count('\n', 0, m.start()) + 1
            self._entries.append((text_line, int(m.group(1)), m.group(2)))
        self._entries.sort()

    def resolve(self, src: str, offset: int) -> tuple[str, int]:
        target_line = src.count('\n', 0, offset) + 1
        module, original = self._fallback_module, target_line
        for text_line, orig_line, mod in self._entries:
            if text_line >= target_line: break
            module, original = mod, orig_line + (target_line - text_line - 1)
        return module, original


def _scan_string_literal(src: str, start: int, loc: SourceLocation) -> tuple[str, int]:
    """`src[start] == '"'`. Returns (raw content between the quotes, index just past the
    closing quote). Handles an escaped quote (`\\"`) by skipping two characters at a time --
    a comma or a close-paren inside the string is just more content, never a delimiter,
    since this walks character-by-character rather than regex-matching up to the next
    special character."""
    i, n = start + 1, len(src)
    while i < n:
        c = src[i]
        if c == '\\':
            i += 2
            continue
        if c == '"':
            return src[start + 1:i], i + 1
        i += 1
    raise TlangAttributeError(
        "printf(...): unterminated string literal (missing closing '\"')",
        loc,
    )


def _parse_specifiers(fmt: str, loc: SourceLocation) -> list[str]:
    """Every format directive in `fmt` that consumes an argument, in order. `%%` is
    recognised but never appended -- it consumes no argument."""
    specs: list[str] = []
    for m in _SPEC_PATTERN.finditer(fmt):
        c = m.group(1)
        if c == '%': continue
        if c not in VALUE_SPECIFIERS:
            raise TlangAttributeError(
                f"printf format {fmt!r} uses unsupported specifier '%{c}' -- supported: "
                f"%d, %u, %f, %x, %%",
                loc,
            )
        specs.append(c)
    return specs


def _cast_expr(spec: str, expr: str) -> str:
    """Convert one call argument to its final stored `uint` bit pattern, driven by the
    FORMAT SPECIFIER rather than the argument's own GLSL type -- this is what lets every
    `printf` overload take plain `uint`s (see the module docstring): `%d`/`%u`/`%x` all
    want the argument's bits reinterpreted as `uint` (GLSL's int->uint conversion is
    bit-preserving, and `%u`/`%x` are already unsigned), `%f` wants the IEEE-754 bit
    pattern of a float. Applied at the call site, so there is no overload-resolution step
    (and therefore no implicit-conversion hazard) between the caller's real argument type
    and what gets stored.
    """
    if spec in ('d', 'u', 'x'): return f'uint({expr})'
    if spec == 'f': return f'floatBitsToUint({expr})'
    raise ValueError(f"unhandled printf specifier: {spec!r}")  # pragma: no cover -- exhaustive above


def _split_call_args(masked: str, src: str, start: int, loc: SourceLocation) -> tuple[list[str], int]:
    """`masked[start]` is the character right after the format string's closing quote (in
    the comment+string-masked copy of `src` -- same length/positions as `src`, so an index
    into one is valid in the other). Returns (raw argument expression texts, index just
    past the call's closing ')'). Bracket-depth-aware, so a nested call in an argument
    (`printf("%d", foo(a, b))`) never mistakes its inner comma/close-paren for the outer
    call's own."""
    i, n = start, len(masked)
    while i < n and masked[i] in ' \t\r\n': i += 1
    if i < n and masked[i] == ')':
        return [], i + 1
    if i >= n or masked[i] != ',':
        found = src[i:i + 1] or '<end of file>'
        raise TlangAttributeError(
            f"printf(...): expected ',' or ')' after the format string, found {found!r}",
            loc,
        )
    i += 1  # skip the comma after the format string

    args: list[str] = []
    depth = 0
    arg_start = i
    while i < n:
        c = masked[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if c == ')' and depth == 0:
                args.append(src[arg_start:i].strip())
                return args, i + 1
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(src[arg_start:i].strip())
            arg_start = i + 1
        i += 1
    raise TlangAttributeError("printf(...): unterminated call (missing closing ')')", loc)


def rewrite_printf_calls(src: str, module: str, table: PrintfTable) -> tuple[str, bool]:
    """Find and rewrite every `printf("...", ...)` call in `src`, replacing the format
    string (which never reaches GLSL) with a call-site id `table` resolves host-side to
    `(module, line, format, specifiers)`:

        printf("ptc %d depth %f\\n", gid, depth);   ->   printf(7u, gid, depth);

    Returns `(rewritten_src, used)`; `used` is whether at least one call was found (a
    cheap, conservative pre-DCE signal -- see `render_printf_module`'s docstring for why
    that's the right check, mirroring v1's `contains_print_call`).

    `module` is only the FALLBACK module name for a location that precedes any `#line`
    directive in `src` (shouldn't normally happen -- `Shader._build` always emits one
    before handing this function any real body text). The real (module, line) attributed
    to each call site is resolved from `src`'s own embedded `#line N "module"` directives
    (see `_LineIndex`), the same ones a driver compile error is matched against -- not from
    a naive newline count over `src`, which would misattribute any call inside a
    `[link(...)]`ed helper spliced in from a different module.

    Validates at BUILD TIME (raising `TlangAttributeError` naming the format and the
    mismatch) that the format string's specifier count matches the call's argument count,
    and that the argument count doesn't exceed `MAX_PRINTF_ARGS`. Preserves line numbers:
    the replacement text is padded with the same number of newlines the original call span
    contained, so every `#line` directive further down the assembled unit stays accurate.
    """
    masked = mask_comments_and_strings(src, mask_strings=False)  # strings visible, comments blanked
    line_index = _LineIndex(src, module)
    out: list[str] = []
    cursor = 0
    used = False

    for m in _PRINTF_CALL_PATTERN.finditer(masked):
        call_start = m.start()
        if call_start < cursor: continue  # inside an already-rewritten call's original span
        paren_open = m.end() - 1
        call_module, call_line = line_index.resolve(src, call_start)
        loc = SourceLocation(call_module, call_line)

        i = paren_open + 1
        while i < len(src) and src[i] in ' \t\r\n': i += 1
        if i >= len(src) or src[i] != '"':
            raise TlangAttributeError(
                "printf(...) requires a string literal as its first argument -- GLSL has no "
                "strings, so tlang needs the format text at build time, not at runtime",
                loc,
            )

        raw_fmt, after_fmt = _scan_string_literal(src, i, loc)
        fmt = _unescape(raw_fmt)
        specifiers = _parse_specifiers(fmt, loc)

        args, call_end = _split_call_args(masked, src, after_fmt, loc)

        if len(args) != len(specifiers):
            raise TlangAttributeError(
                f"printf format {fmt!r} has {len(specifiers)} specifier(s) but {len(args)} "
                f"argument(s) were passed",
                loc,
            )
        if len(args) > MAX_PRINTF_ARGS:
            raise TlangAttributeError(
                f"printf(...) supports at most {MAX_PRINTF_ARGS} arguments, got {len(args)}",
                loc,
            )

        callsite_id = table.register_callsite(call_module, call_line, fmt, specifiers)
        # Cast HERE, per specifier, not via a type-matched overload -- see `_cast_expr`'s
        # docstring for why: overload resolution cannot express this past arity 3 (that was
        # the actual bug this replaced -- a mixed-type call at arity 4+ silently
        # implicit-converted every argument to the arity's single same-type overload).
        cast_args = [_cast_expr(spec, arg) for spec, arg in zip(specifiers, args)]
        new_call = f'printf({callsite_id}u' + (', ' + ', '.join(cast_args) if cast_args else '') + ')'
        new_call += '\n' * src.count('\n', call_start, call_end)  # keep #line accuracy downstream

        out.append(src[cursor:call_start])
        out.append(new_call)
        cursor = call_end
        used = True

    out.append(src[cursor:])
    return ''.join(out), used


__all__ = [
    'MAX_PRINTF_ARGS',
    'render_overloads', 'render_buffer_decl', 'render_printf_module', 'rewrite_printf_calls',
]
