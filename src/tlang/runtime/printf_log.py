# -------------------------------------------------------------
# @file          printf_log.py
# @author        Priyangkar Ghosh
# @created       2026-09-10
# @description   Host side of the `printf(...)` debug-log built-in: the pinned ring buffer
#                tlang owns (never the user), numpy-vectorised decoding of its records, and
#                the `sm.stdout` surface (dedup, rate-limited streaming, on-demand drain).
#                See `tlang.compiler.printf_codegen` for the GLSL side (the emitted `printf`
#                overloads, the call-site rewrite, and the SSBO block they write into) --
#                this module is the single source of truth for the wire format both sides
#                share, plus the `PrintfTable` build-time registry both sides populate/read.
# @license       MIT
# -------------------------------------------------------------

import logging
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import regex as re
from moderngl import Context

from tlang.errors import SourceLocation, TlangError
from tlang.runtime.pinned_buffer import PinnedBuffer, PinnedBufferFallback, create_pinned_buffer

logger = logging.getLogger(__name__)

# ----- wire format (see `tlang.compiler.printf_codegen` for the GLSL-side encoder) -----
#
# layout(std430) buffer TlangPrintfLog {
#     uint tlang_pf_reserved;        // next slot POSITION to claim -- see the CAS note below
#     uint tlang_pf_dropped;         // count of records dropped because the ring was full
#     uint tlang_pf_ready[CAPACITY]; // per-slot sequence number -- see below
#     uint tlang_pf_data[];          // CAPACITY fixed-size RECORD_WORDS-uint slots
# };
#
# One record per `printf(...)` call: word 0 is the CALL-SITE id (not a type/count header --
# the call site's own format string, resolved host-side via `PrintfTable`, already says how
# many arguments there are and what type each one is; see the module docstring). Words
# 1..len(specifiers) are the arguments, each bit-cast to uint exactly as v1's `print` did.
# Fixed slot width (`RECORD_WORDS`) means a slot index converts to a byte offset with no
# per-record length bookkeeping.
#
# ## Ring buffer + ACK'd flow control -- why this is NOT a plain atomicAdd cursor
#
# The obvious first design -- `pos = atomicAdd(reserved, 1u)`, drop if `pos - read_cursor >=
# CAPACITY` -- has a real bug: `reserved` advances on EVERY attempt, including dropped ones,
# so a single overflow burst permanently "spends" position numbers nothing ever retries.
# Once the host's sequential read cursor reaches a position that was dropped, it can never
# advance past it (there is no data there, ever) -- and since the read side must consume
# positions in order to know a slot is safe to read, every record written AFTER that gap,
# even if it landed successfully, becomes permanently stranded. Reproduced empirically while
# building this feature (see the ACK test) before landing on the fix below.
#
# The fix is Vyukov's bounded MPSC queue: `tlang_pf_ready[slot]` is a per-slot SEQUENCE
# NUMBER, initialised to `slot` itself (not zero), with two meanings depending on who's
# looking at it:
#   - to a PRODUCER: `ready[slot] == pos` means slot is free for position `pos` -- claim it
#     via `atomicCompSwap(reserved, pos, pos+1)`; a CAS failure means another invocation got
#     there first, so retry with the fresh value; `ready[slot] < pos` means the slot hasn't
#     been vacated by the host yet -- the ring is genuinely full, so drop and count, WITHOUT
#     ever touching `reserved`. Because `reserved` only advances on a WINNING claim, there
#     are no gaps: every position that gets claimed is guaranteed to get written.
#   - to the CONSUMER (host): `ready[slot] == pos+1` (set by the producer, AFTER
#     `memoryBarrierBuffer()` -- see "publish ordering" below) means that slot's record for
#     position `pos` is safe to read. After decoding it, the host writes `ready[slot] =
#     pos+CAPACITY` back into the SAME buffer (a plain memory write to a coherent persistent
#     mapping -- no GL call), which is exactly what lets a producer's CAS claim that slot
#     again for the NEXT lap (`pos+CAPACITY`). This per-slot write, not a single scalar
#     cursor, IS the "host writes back a read cursor the shader can see" the brief asks for
#     -- merged with the same field the publish-ordering flag already needed, rather than a
#     second one, specifically because a single scalar cursor is what has the gap bug above.
#
# Silently losing the EARLIEST records is the failure mode this design refuses: what gets
# dropped is always the newest claim attempt, and `dropped` reports exactly how many.
MAX_PRINTF_ARGS = 8  # see the module docstring's "maximum argument count" note below
RECORD_WORDS = 1 + MAX_PRINTF_ARGS  # call-site id + up to 8 values
RECORD_BYTES = RECORD_WORDS * 4

HEADER_WORDS = 2  # tlang_pf_reserved, tlang_pf_dropped
HEADER_BYTES = HEADER_WORDS * 4

DEFAULT_LOG_CAPACITY = 4096  # records

# Retry budget for a producer's CAS claim loop (see above) before giving up and counting the
# attempt as dropped instead. Only exhausted under pathological contention (many invocations
# racing the exact same slot in the exact same instant); a normal claim succeeds in 1-2
# iterations. Generous enough to never fire in practice, bounded so a claim can never spin
# forever.
CAS_SPIN_LIMIT = 64

# The raw GLSL names `printf`'s real body touches -- see `tlang.compiler.printf_codegen`,
# the only other place these are spelled out.
BUFFER_HANDLE = 'TlangPrintfLog'
RESERVED_FIELD = 'tlang_pf_reserved'
DROPPED_FIELD = 'tlang_pf_dropped'
READY_FIELD = 'tlang_pf_ready'
DATA_FIELD = 'tlang_pf_data'

# Supported printf specifiers -- decode side. `%%` is handled separately (it consumes no
# argument at all, see `tlang.compiler.printf_codegen._parse_specifiers`).
VALUE_SPECIFIERS = ('d', 'u', 'f', 'x')

# A sensible ceiling, not an arbitrary one: RECORD_WORDS sizes every slot in the ring
# regardless of how many arguments any individual call actually used, so this is a direct
# per-record memory cost (36 bytes/record at 8) -- matches v1 `print`'s own limit, which was
# already tuned against "how many values does one debug line realistically carry" rather
# than any GLSL/driver ceiling.


def buffer_size_for_capacity(capacity: int) -> int:
    return HEADER_BYTES + capacity * 4 + capacity * RECORD_BYTES  # header + ready[] + data[]


@dataclass(slots=True, frozen=True)
class PrintfCallSite:
    """One `printf(...)` call site, as `PrintfTable.register_callsite` recorded it at build
    time. `format_id` indexes `PrintfTable.format` -- interned separately from the call site
    itself, since two call sites can share the exact same format text."""
    module: str
    line: int
    format_id: int
    format: str
    specifiers: tuple[str, ...]

    @property
    def location(self) -> SourceLocation:
        return SourceLocation(self.module, self.line)


class PrintfTable:
    """Build-time registry shared by every `Shader` a `ShaderManager` builds, so a
    call-site/format id assigned while compiling one module stays valid (and decodable)
    against the ONE shared `PrintfLog` buffer the whole tree writes into.

    Exists, and is always populated, whether or not `debug=True` -- the specifier-vs-
    argument-count validation `tlang.compiler.printf_codegen.rewrite_printf_calls` performs
    while populating this table is a real build-time bug check (see the brief), not a
    debug-only nicety. It costs nothing in release beyond the one-time scan already required
    to find and rewrite each call site.
    """

    __slots__ = ('_formats', '_format_list', '_callsites')

    def __init__(self) -> None:
        self._formats: dict[str, int] = {}
        self._format_list: list[str] = []
        self._callsites: list[PrintfCallSite] = []

    def intern_format(self, text: str) -> int:
        """Return `text`'s format id, assigning a fresh one the first time this exact
        string is seen anywhere in the tree -- two call sites with identical format text
        share one entry."""
        if (fid := self._formats.get(text)) is not None: return fid
        fid = len(self._format_list)
        self._formats[text] = fid
        self._format_list.append(text)
        return fid

    def register_callsite(self, module: str, line: int, format_text: str, specifiers: list[str]) -> int:
        """Assign a fresh call-site id (never dedup'd -- two call sites are two places in
        the source, even if their format text happens to match) and return it."""
        format_id = self.intern_format(format_text)
        callsite_id = len(self._callsites)
        self._callsites.append(PrintfCallSite(module, line, format_id, format_text, tuple(specifiers)))
        return callsite_id

    def callsite(self, callsite_id: int) -> PrintfCallSite:
        return self._callsites[callsite_id]

    def format(self, format_id: int) -> str:
        return self._format_list[format_id]

    def __len__(self) -> int:
        return len(self._callsites)


# ---------------------------------------------------------------------------
# Formatting: a call site's raw format text (with tlang's %d/%u/%f/%x/%%) translated once
# per call into a Python `%`-format string, so decoded values substitute in with a single
# `%` application instead of tlang hand-rolling printf-style formatting itself.
# ---------------------------------------------------------------------------

_SPEC_PATTERN = re.compile(r'%(.)')
_PY_SPEC = {'d': '%d', 'u': '%d', 'f': '%f', 'x': '%x', '%': '%%'}


def to_python_format(fmt: str) -> str:
    """`fmt` has already been validated (`printf_codegen._parse_specifiers`) to contain only
    `%d`/`%u`/`%f`/`%x`/`%%` -- every `%` is the start of one of those five sequences, so a
    single substitution pass is enough."""
    return _SPEC_PATTERN.sub(lambda m: _PY_SPEC[m.group(1)], fmt)


def format_record(callsite: PrintfCallSite, values: tuple) -> str:
    """One decoded record -> the line a caller sees: `module:line  message`, trailing
    newline(s) from the format string stripped (the sink -- `print` by default -- adds its
    own)."""
    py_fmt = to_python_format(callsite.format)
    msg = (py_fmt % tuple(values)) if values else py_fmt
    return f'{callsite.location}  {msg.rstrip(chr(10))}'


# ---------------------------------------------------------------------------
# numpy decode -- vectorised per distinct call site present in a batch, not per record.
# Real printf usage is dominated by a handful of static call sites hit many times (the
# volume-control section's whole point), so grouping by call-site id keeps the expensive
# per-column reinterpret-cast vectorised over N while the (cheap, call-site-count-bounded)
# Python loop only runs once per DISTINCT site, not once per record.
# ---------------------------------------------------------------------------


def decode_records_numpy(raw: bytes, count: int, table: PrintfTable) -> list[tuple[int, tuple]]:
    """`raw` is `count` contiguous `RECORD_WORDS`-uint32 slots. Returns `(callsite_id,
    values)` pairs in the SAME order as `raw` (ring/call order) -- `values` is a plain
    Python tuple of `int`/`float`, decoded per that call site's own specifiers. A slot's
    unused tail words (past its own specifier count) are never touched."""
    if count == 0: return []
    arr = np.frombuffer(raw, dtype='<u4', count=count * RECORD_WORDS).reshape(count, RECORD_WORDS)
    callsite_ids = arr[:, 0]
    out: list[tuple | None] = [None] * count

    for cid in np.unique(callsite_ids).tolist():
        cid = int(cid)
        idxs = np.nonzero(callsite_ids == cid)[0]
        specifiers = table.callsite(cid).specifiers
        n = len(specifiers)
        cols = arr[idxs, 1:1 + n]

        decoded_cols = []
        for k, spec in enumerate(specifiers):
            col = cols[:, k]
            if spec in ('u', 'x'):
                decoded_cols.append(col.astype(np.int64))
            elif spec == 'd':
                decoded_cols.append(col.view(np.int32).astype(np.int64))
            elif spec == 'f':
                decoded_cols.append(col.view(np.float32).astype(np.float64))
            else:  # pragma: no cover -- validated exhaustively at build time
                raise TlangError(f"printf: unknown specifier '{spec}' in decoded call site {cid}")

        # Columns can mix int64 and float64 specifiers (`%d depth %f`) -- `np.stack` would
        # upcast the whole row to float and silently turn an int argument into e.g. `-3.0`.
        # Keep each column's own dtype (so `.tolist()` gives back the right Python type per
        # column) and recombine into per-row tuples with `zip` (C-level, not a Python loop
        # indexing each column per row -- that alone was the difference between numpy
        # tracking struct's throughput and actually beating it at 1M records).
        rows = list(zip(*(col.tolist() for col in decoded_cols))) if decoded_cols else [()] * len(idxs)
        for pos, idx in enumerate(idxs.tolist()):
            out[idx] = (cid, rows[pos])

    return out  # type: ignore[return-value]


class PrintfLog:
    """Owns the one pinned GPU buffer backing every `printf(...)` call in a debug build --
    see the module docstring for the wire format. Created (at most once) by
    `ShaderManager(debug=True)` and shared by every `Shader`/`Kernel`/`Pipeline` it builds;
    tlang allocates, sizes, binds and frees this, never user code.

    Allocated as a PINNED (persistent-mapped) buffer specifically so a background thread
    with no GL context can poll it (see `poll(use_mapping=True)`): reading `.mapping` is a
    plain memory access, not a GL call. Falls back to a plain buffer (`.is_pinned is False`)
    when `GL_ARB_buffer_storage` is unavailable -- `poll(use_mapping=False)` (via `.read()`)
    still works, just not from a context-less thread; see `PrintfStream.stream`.
    """

    __slots__ = ('_ctx', '_capacity', '_table', '_buffer', '_read_cursor')

    def __init__(self, ctx: Context, table: PrintfTable, capacity: int = DEFAULT_LOG_CAPACITY) -> None:
        if capacity <= 0:
            raise TlangError(f"printf log capacity must be positive, got {capacity}")
        self._ctx = ctx
        self._capacity = capacity
        self._table = table
        self._buffer = create_pinned_buffer(ctx, buffer_size_for_capacity(capacity))
        self._read_cursor = 0
        self.clear()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def buffer(self):
        """The raw pinned buffer -- bound automatically by tlang; duck-types
        `moderngl.Buffer` for everything `Kernel`/`Pipeline` touch (see `pinned_buffer.py`)."""
        return self._buffer

    @property
    def table(self) -> PrintfTable:
        return self._table

    @property
    def is_pinned(self) -> bool:
        return self._buffer.is_pinned

    def clear(self) -> None:
        """Reset `reserved`/`dropped` to zero and every `ready[]` slot back to its Vyukov
        initial value (`ready[slot] == slot`, meaning "free for position == slot"), so a
        stale ready flag from before the clear can never be mistaken for a fresh record."""
        header = struct.pack('<II', 0, 0)
        ready_init = np.arange(self._capacity, dtype='<u4').tobytes()
        if self._buffer.is_pinned:
            self._buffer.mapping[0:HEADER_BYTES] = header
            self._buffer.mapping[HEADER_BYTES:HEADER_BYTES + len(ready_init)] = ready_init
        else:
            self._buffer.write(header, offset=0)
            self._buffer.write(ready_init, offset=HEADER_BYTES)
        self._read_cursor = 0

    def _fetch(self, offset: int, size: int, *, use_mapping: bool) -> bytes:
        """Byte fetch, routed to avoid any GL call when `use_mapping` (the background
        streaming thread has no GL context to make one with): a raw `.mapping` slice is
        plain memory access. `use_mapping=False` (a caller on the GL thread, e.g. `drain()`)
        goes through `.read()`, which is fence-synced and therefore torn-data-safe even
        against an in-flight dispatch."""
        if use_mapping:
            if not self._buffer.is_pinned:
                raise TlangError("printf log has no real mapping (GL_ARB_buffer_storage fallback) -- poll with use_mapping=False")
            return bytes(self._buffer.mapping[offset:offset + size])
        return self._buffer.read(size, offset=offset)

    def _mark_slot_free(self, slot: int, pos: int, *, use_mapping: bool) -> None:
        """After consuming the record at `slot` (originally written for position `pos`),
        tell the GPU that slot is free again for the NEXT lap (`pos + capacity`) -- see the
        module docstring's Vyukov explanation. This IS the ring's flow-control ack; there is
        no separate scalar read cursor."""
        packed = struct.pack('<I', (pos + self._capacity) & 0xFFFFFFFF)
        offset = HEADER_BYTES + slot * 4
        if use_mapping:
            # Plain memory write to a coherent persistent mapping -- no GL call, and none
            # needed: CPU->GPU visibility needs no explicit sync under GL_MAP_COHERENT_BIT
            # (see PinnedBuffer's docstring). Only the other direction (GPU-write-then-
            # CPU-read) needs a fence, which the `ready[]` publish check already covers.
            self._buffer.mapping[offset:offset + 4] = packed
        else:
            self._buffer.write(packed, offset=offset, sync=False)

    def poll(self, *, use_mapping: bool) -> list[tuple[int, tuple]]:
        """One non-blocking scan: decode every record from the current read cursor up to
        the first not-yet-published slot (per `tlang_pf_ready`), free each consumed slot for
        reuse, and return `(callsite_id, values)` pairs in call order. Never blocks and
        never partially decodes a record -- `tlang_pf_ready[slot]` only reads back `pos+1`
        after that slot's `memoryBarrierBuffer()`-ordered write has landed, and `reserved`
        (see the module docstring) never has gaps, so "the next `capacity` slots are
        contiguous positions starting at the read cursor" always holds.
        """
        r = self._read_cursor
        ready_raw = self._fetch(HEADER_BYTES, self._capacity * 4, use_mapping=use_mapping)
        ready = np.frombuffer(ready_raw, dtype='<u4')

        n_ready = 0
        for i in range(self._capacity):
            slot = (r + i) % self._capacity
            if int(ready[slot]) != (r + i + 1) & 0xFFFFFFFF: break
            n_ready += 1
        if n_ready == 0: return []

        data_offset = HEADER_BYTES + self._capacity * 4
        start_slot = r % self._capacity
        first_chunk = min(n_ready, self._capacity - start_slot)
        chunks = [(start_slot, first_chunk)]
        if n_ready > first_chunk:
            chunks.append((0, n_ready - first_chunk))

        raw = b''.join(
            self._fetch(data_offset + slot0 * RECORD_BYTES, cnt * RECORD_BYTES, use_mapping=use_mapping)
            for slot0, cnt in chunks
        )
        results = decode_records_numpy(raw, n_ready, self._table)

        for i in range(n_ready):
            self._mark_slot_free((r + i) % self._capacity, r + i, use_mapping=use_mapping)
        self._read_cursor = r + n_ready
        return results

    @property
    def dropped(self) -> int:
        _reserved, dropped = struct.unpack('<II', self._fetch(0, HEADER_BYTES, use_mapping=False))
        return int(dropped)

    def release(self) -> None:
        self._buffer.release()


class PrintfStream:
    """`ShaderManager(debug=True).stdout` -- the shader's stdout (see the brief: "Host
    surface is the shader's stdout"). Wraps the shared `PrintfLog` with the volume-control
    machinery a real printf feature needs to stay usable: consecutive-record dedup with a
    repeat count, and a token-bucket rate limit on whatever sink `stream()` is given.

    ```python
    sm.stdout.stream(sink=print)   # background thread -> sink(line) per formatted record
    sm.stdout.stop()               # stop it cleanly
    sm.stdout.drain()              # -> formatted lines available right now (synchronous)
    sm.stdout.dropped              # records dropped so far because the ring was full
    sm.stdout.clear()              # reset cursors/counters (does between-dispatch what
                                    # v1's clear_debug_log() did)
    ```

    **This is closer to an assert than a log.** One unguarded `printf(...)` in a
    million-invocation dispatch is a million records; `dedup`/`rate_limit` blunt the
    console-flooding symptom, but the fix is `if (gid == 0) printf(...)` -- see
    `references/runtime.md`.
    """

    __slots__ = ('_log', '_thread', '_stop_event')

    def __init__(self, log: PrintfLog) -> None:
        self._log = log
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def dropped(self) -> int:
        """Records dropped because the ring was full when a `printf(...)` call tried to
        reserve a slot -- 0 means nothing was lost. Always check this before trusting that
        `drain()`/`stream()` saw the complete log for a dispatch."""
        return self._log.dropped

    @property
    def capacity(self) -> int:
        return self._log.capacity

    def clear(self) -> None:
        """Reset the shared ring's cursors/counters to zero -- call this between dispatches
        the same way v1's `clear_debug_log()` was called."""
        self._log.clear()

    @staticmethod
    def _dedup(lines: list[str]) -> list[str]:
        """Collapse a run of CONSECUTIVE identical lines into one, suffixed with a repeat
        count -- not a whole-batch/global dedup (two identical lines separated by a
        different one are both kept), which is the useful behaviour for the tight-loop
        flood this exists to blunt."""
        out: list[str] = []
        last: str | None = None
        run = 0

        def flush():
            if last is None: return
            out.append(last if run == 1 else f'{last}  (x{run})')

        for line in lines:
            if line == last:
                run += 1
                continue
            flush()
            last, run = line, 1
        flush()
        return out

    def _formatted(self, use_mapping: bool, *, dedup: bool) -> list[str]:
        records = self._log.poll(use_mapping=use_mapping)
        lines = [format_record(self._log.table.callsite(cid), values) for cid, values in records]
        return self._dedup(lines) if dedup else lines

    def drain(self, *, dedup: bool = True) -> list[str]:
        """Decode and format every record published since the last `drain()`/`stream()`
        poll, right now, synchronously. Safe to call from the GL thread at any time (goes
        through `.read()`, which is fence-synced -- see `PrintfLog._fetch`)."""
        return self._formatted(use_mapping=False, dedup=dedup)

    def stream(
        self, sink: Callable[[str], None] = print, *,
        poll_interval: float = 0.02, rate_limit: float | None = 200.0, dedup: bool = True,
    ) -> None:
        """Start a background daemon thread that polls the shared ring and calls
        `sink(line)` for each formatted record, until `stop()`.

        Touches only `PrintfLog.poll(use_mapping=True)` when the buffer is really pinned --
        i.e. never a GL call, so this thread never needs (and must never be given) a current
        GL context; see the class/module docstrings. Falls back to `use_mapping=False`
        (`.read()`) when `GL_ARB_buffer_storage` was unavailable at buffer creation --
        correctness there depends on your GL binding tolerating a cross-thread call, which
        moderngl/PyOpenGL do not guarantee, so prefer draining manually from the render
        thread in that fallback case.

        `rate_limit` (lines/second, a token bucket seeded full so a burst up to
        `rate_limit` lines is never delayed) caps how fast `sink` is called -- `None`
        disables the cap entirely. Exceeding it does not drop the data (deduped, still
        counted); it defers the sink call and reports how many lines were withheld with a
        single summary line. `dedup=True` (default) collapses consecutive identical lines
        (see `_dedup`) before the rate limiter ever sees them.
        """
        if self._thread is not None and self._thread.is_alive():
            raise TlangError("printf stream is already running -- call stop() first")

        use_mapping = self._log.is_pinned
        if not use_mapping:
            logger.warning(
                "printf stream: GL_ARB_buffer_storage unavailable -- streaming thread will "
                "call .read() across threads, which most GL bindings do not guarantee is "
                "safe; prefer drain() from the render thread instead"
            )

        self._stop_event.clear()

        def _run() -> None:
            bucket = rate_limit if rate_limit is not None else 0.0
            last_refill = time.monotonic()
            suppressed = 0
            while not self._stop_event.is_set():
                for line in self._formatted(use_mapping, dedup=dedup):
                    if rate_limit is not None:
                        now = time.monotonic()
                        bucket = min(rate_limit, bucket + (now - last_refill) * rate_limit)
                        last_refill = now
                        if bucket < 1.0:
                            suppressed += 1
                            continue
                        bucket -= 1.0
                    if suppressed:
                        sink(f'... {suppressed} line(s) suppressed by rate limit ...')
                        suppressed = 0
                    sink(line)
                self._stop_event.wait(poll_interval)

        self._thread = threading.Thread(target=_run, name='tlang-printf-stream', daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the background thread started by `stream()`, joining it (bounded by
        `timeout`). A no-op if `stream()` was never called, or was already stopped."""
        if self._thread is None: return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def release(self) -> None:
        self.stop()
        self._log.release()


__all__ = [
    'MAX_PRINTF_ARGS', 'RECORD_WORDS', 'RECORD_BYTES', 'HEADER_WORDS', 'HEADER_BYTES',
    'DEFAULT_LOG_CAPACITY', 'VALUE_SPECIFIERS',
    'BUFFER_HANDLE', 'RESERVED_FIELD', 'DROPPED_FIELD', 'READY_FIELD', 'DATA_FIELD', 'CAS_SPIN_LIMIT',
    'buffer_size_for_capacity', 'to_python_format', 'format_record', 'decode_records_numpy',
    'PrintfCallSite', 'PrintfTable', 'PrintfLog', 'PrintfStream',
]
