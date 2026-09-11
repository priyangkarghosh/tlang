# -------------------------------------------------------------
# @file          debug_log.py
# @author        Priyangkar Ghosh
# @created       2026-09-10
# @description   Host side of the `print(...)` debug-log feature: the GPU buffer tlang owns
#                (never the user), and decoding its fixed-slot records back into Python
#                int/float/bool tuples. See `tlang.compiler.debug_print` for the GLSL side
#                (the emitted `print` overloads and the SSBO block they write into) -- this
#                module is the single source of truth for the wire format both sides share.
# @license       MIT
# -------------------------------------------------------------

import struct
from typing import NamedTuple

from moderngl import Context

from tlang.errors import TlangError

# ----- wire format (see `tlang.compiler.debug_print` for the GLSL-side encoder) -----
#
# layout(std430) buffer TlangDebugLog {
#     uint tlang_debug_cursor;    // next record slot to try -- atomicAdd'd, unconditionally
#     uint tlang_debug_overflow;  // count of records dropped because the log was full
#     uint tlang_debug_data[];    // CAPACITY fixed-size RECORD_WORDS-uint slots
# };
#
# Each slot is RECORD_WORDS uints regardless of how many arguments that call actually
# used: word 0 is the header (arg count in bits [0:4), a 2-bit type tag per argument in
# bits [4+2*i : 6+2*i)), words 1..count are the arguments, each bit-cast to uint. Fixed
# slot width means a slot index converts to a byte offset with no per-record length
# bookkeeping, and capacity is enforced by simply refusing to write past it -- a slot at
# or past `cursor >= capacity` increments `tlang_debug_overflow` and touches no data.
TAG_UINT, TAG_INT, TAG_FLOAT, TAG_BOOL = 0, 1, 2, 3
_TAG_TYPE: dict[int, str] = {TAG_UINT: 'uint', TAG_INT: 'int', TAG_FLOAT: 'float', TAG_BOOL: 'bool'}

MAX_PRINT_ARGS = 8
RECORD_WORDS = 1 + MAX_PRINT_ARGS  # header + up to 8 values
RECORD_BYTES = RECORD_WORDS * 4
HEADER_BYTES = 8  # tlang_debug_cursor (4) + tlang_debug_overflow (4), at offset 0

DEFAULT_LOG_CAPACITY = 4096  # records

# The raw GLSL names `print`'s real body touches -- see `tlang.compiler.debug_print`,
# which is the only other place these are spelled out.
DEBUG_BUFFER_HANDLE = 'TlangDebugLog'
CURSOR_FIELD = 'tlang_debug_cursor'
OVERFLOW_FIELD = 'tlang_debug_overflow'
DATA_FIELD = 'tlang_debug_data'


def decode_value(tag: int, word: int) -> int | float | bool:
    """One raw uint32 word back to the Python value its type tag says it is."""
    if tag == TAG_UINT: return word
    if tag == TAG_INT: return struct.unpack('<i', struct.pack('<I', word))[0]
    if tag == TAG_FLOAT: return struct.unpack('<f', struct.pack('<I', word))[0]
    if tag == TAG_BOOL: return word != 0
    raise TlangError(f"debug log: unknown print argument type tag {tag} -- corrupt record?")


def decode_record(words: tuple[int, ...]) -> tuple[int | float | bool, ...]:
    """`words` is one `RECORD_WORDS`-long slot (header first). Returns just the decoded
    argument values, in call order -- padding words past the header's own count are
    ignored (a call with fewer than MAX_PRINT_ARGS arguments leaves them unspecified)."""
    header = words[0]
    count = header & 0xF
    values = []
    for i in range(count):
        tag = (header >> (4 + 2 * i)) & 0x3
        values.append(decode_value(tag, words[1 + i]))
    return tuple(values)


def buffer_size_for_capacity(capacity: int) -> int:
    return HEADER_BYTES + capacity * RECORD_BYTES


class DebugLogResult(NamedTuple):
    """`DebugLog.read()`'s return value. Destructures like a plain `(records, overflow)`
    tuple, but also names both fields for a caller that wants `.records`/`.overflow`."""
    records: list[tuple]
    overflow: int


class DebugLog:
    """Owns the one GL buffer backing every `print(...)` call in a debug build.

    Created (at most once) by `ShaderManager(debug=True)` and shared by every `Shader`,
    `Kernel` and `Pipeline` it builds -- tlang allocates, sizes, binds and frees this;
    user code never touches the buffer directly. `capacity` is fixed for the buffer's
    life (records past it are dropped and counted in `overflow`, never overwritten or
    corrupted -- see `read()`).
    """

    __slots__ = ('_ctx', '_capacity', '_buffer')

    def __init__(self, ctx: Context, capacity: int = DEFAULT_LOG_CAPACITY) -> None:
        if capacity <= 0:
            raise TlangError(f"debug log capacity must be positive, got {capacity}")
        self._ctx = ctx
        self._capacity = capacity
        self._buffer = ctx.buffer(reserve=buffer_size_for_capacity(capacity), dynamic=True)
        self.clear()

    @property
    def capacity(self) -> int:
        """Max records the buffer holds before writes start being dropped/counted."""
        return self._capacity

    @property
    def buffer(self):
        """The raw `moderngl.Buffer` -- bound automatically by tlang; not for user code
        to bind itself (there is nothing further to configure, and no name to bind it
        under -- `Kernel`/`Pipeline` already did that at build time)."""
        return self._buffer

    def clear(self) -> None:
        """Reset the cursor and overflow counter to zero. Does not touch the record
        bytes themselves -- `read()` only ever decodes `min(cursor, capacity)` records,
        so anything past a freshly-zeroed cursor is simply never reached."""
        self._buffer.write(struct.pack('<II', 0, 0), offset=0)

    def read(self) -> DebugLogResult:
        """Decode every live record. Returns a `DebugLogResult(records, overflow)`:

        - `records`: one tuple of decoded Python values per `print(...)` call, in the
          order they were written (`cursor` order), each `int`/`float`/`bool` exactly as
          the type tag baked into that call site's overload says.
        - `overflow`: how many calls were dropped because the log was already at
          `capacity` when they ran -- 0 means nothing was lost. Always check this before
          trusting that `records` is the complete log for a dispatch.
        """
        cursor, overflow = struct.unpack('<II', self._buffer.read(HEADER_BYTES))
        count = min(cursor, self._capacity)
        if count == 0: return DebugLogResult([], overflow)

        raw = self._buffer.read(count * RECORD_BYTES, offset=HEADER_BYTES)
        records = []
        for i in range(count):
            words = struct.unpack_from(f'<{RECORD_WORDS}I', raw, i * RECORD_BYTES)
            records.append(decode_record(words))
        return DebugLogResult(records, overflow)

    def release(self) -> None:
        """Release the underlying GL buffer. Idempotent-if-unused: calling this twice
        raises only because `moderngl.Buffer.release()` itself does."""
        self._buffer.release()


__all__ = [
    'TAG_UINT', 'TAG_INT', 'TAG_FLOAT', 'TAG_BOOL',
    'MAX_PRINT_ARGS', 'RECORD_WORDS', 'RECORD_BYTES', 'HEADER_BYTES', 'DEFAULT_LOG_CAPACITY',
    'DEBUG_BUFFER_HANDLE', 'CURSOR_FIELD', 'OVERFLOW_FIELD', 'DATA_FIELD',
    'decode_value', 'decode_record', 'buffer_size_for_capacity', 'DebugLogResult', 'DebugLog',
]
