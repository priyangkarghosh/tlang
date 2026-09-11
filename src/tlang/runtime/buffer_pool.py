# -------------------------------------------------------------
# @file          buffer_pool.py
# @author        Priyangkar Ghosh
# @created       2025-09-03
# @description   Segregated-free-list pool of reusable transient GL buffers, plus a
#                named persistent buffer registry. Both namespaces share one tag space,
#                so the pool itself doubles as a name -> Buffer map a Kernel/Pipeline can
#                bind straight from.
# @license       MIT
# -------------------------------------------------------------

import logging
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from moderngl import Buffer, Context

from tlang.errors import TlangError
from tlang.runtime.pinned_buffer import PinnedBuffer, PinnedBufferFallback, create_pinned_buffer

logger = logging.getLogger(__name__)

PinnedBufferLike = PinnedBuffer | PinnedBufferFallback

_POISON_BYTE = b'\xCD'


def clear_buffer(buffer: Buffer, pattern: bytes | None = None, *, size: int = -1, offset: int = 0) -> None:
    """Zero-fill (or repeat `pattern` across) a GL buffer's contents. `pattern=None` clears to zero.

    Deliberately does not `buffer.orphan()` first: this never maps the buffer, so the clear is
    just another command in the GL stream and orphaning would only force a needless reallocation.
    """
    buffer.clear(size=size, offset=offset, chunk=pattern)


@dataclass(slots=True, frozen=True)
class BufferPoolMetrics:
    """Point-in-time snapshot of `BufferPool` usage."""
    bytes_pooled: int          # idle bytes sitting in transient free lists
    bytes_checked_out: int     # bytes currently handed out via alloc_temp
    high_water_mark: int       # peak (bytes_pooled + bytes_checked_out) ever observed
    hits: int                  # alloc_temp calls satisfied from a free list
    misses: int                # alloc_temp calls that allocated a new GL buffer
    persistent_bytes: int      # bytes held by the persistent registry
    persistent_count: int      # number of named persistent buffers


class TempHandle:
    """Opaque handle returned by `alloc_temp`.

    Transparently proxies attribute access to the underlying `moderngl.Buffer`. Tracks its own
    liveness (not the GL object name, which GL can recycle after release), so any access after
    `free_temp` raises `TlangError` instead of silently touching a buffer handed to someone else.
    """
    __slots__ = ('_buffer', '_size_class', '_requested_size', '_alive', '_tag')

    def __init__(self, buffer: Buffer, size_class: int, requested_size: int, tag: str | None = None) -> None:
        self._buffer = buffer
        self._size_class = size_class
        self._requested_size = requested_size
        self._alive = True
        self._tag = tag

    def _kill(self) -> Buffer:
        if not self._alive: raise TlangError("Buffer handle already freed (double free)")
        self._alive = False
        return self._buffer

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def size_class(self) -> int:
        return self._size_class

    @property
    def requested_size(self) -> int:
        return self._requested_size

    @property
    def tag(self) -> str | None:
        return self._tag

    def __getattr__(self, item: str):
        if not self._alive:
            raise TlangError(f"Use of buffer handle after free: '{item}' accessed on a freed temp buffer")
        return getattr(self._buffer, item)

    def __repr__(self) -> str:
        state = 'alive' if self._alive else 'freed'
        tagged = f" tag='{self._tag}'" if self._tag else ''
        return f'<TempHandle size_class={self._size_class} requested={self._requested_size}{tagged} {state}>'


class BufferPool(Mapping[str, Buffer]):
    """Three independent responsibilities in one object, sharing only the `Context` and `clear()`.

    1. **Persistent registry** (`persistent_buffer`) -- buffers named once, created lazily, and
       never recycled.
    2. **Transient pool** (`alloc_temp` / `free_temp` / `temp` / `frame`) -- size-classed, freely
       recyclable scratch buffers with no identity beyond their `TempHandle`.
    3. **Pinned registry** (`alloc_pinned` / `free_pinned`) -- persistent-mapped buffers backed by
       immutable `glBufferStorage` (see `pinned_buffer.PinnedBuffer`). Immutable storage cannot be
       resized or orphaned, so these never enter the transient free lists either -- each is its
       own GL allocation with its own lifetime, exactly like a persistent buffer.

    No buffer ever moves between pools. Transient buffers are kept in per-power-of-two-size
    free lists, so acquire/release are O(1) and a request is only ever satisfied by a buffer of
    its own size class -- never a larger one. `trim()` bounds the resulting per-class memory cost.

    Every persistent buffer's name, every pinned buffer's tag, and every temporary's `tag` (see
    `alloc_temp`), lives in one shared namespace: the pool itself implements `Mapping[str, Buffer]`,
    so `pool['SomeTag']` resolves whichever kind of buffer registered that name, and a
    `Kernel`/`Pipeline` can be given the pool directly as a buffer source (see `Kernel.source`).
    A tag freed via `free_temp`/`free_pinned` stops resolving; a name can't be reused while its
    current buffer is still live.
    """

    def __init__(self, ctx: Context, min_size: int = 256, *, debug_poison: bool = False) -> None:
        self._ctx = ctx
        self._min_size = min_size
        self.debug_poison = debug_poison

        # --- persistent registry ---
        self._persistent: dict[str, Buffer] = {}

        # --- pinned registry ---
        self._pinned: dict[str, PinnedBufferLike] = {}          # tag -> pinned buffer (shared namespace)
        self._pinned_untagged: list[PinnedBufferLike] = []      # anonymous pinned buffers, tracked for teardown only

        # --- transient pool ---
        self._free: dict[int, list[Buffer]] = {}          # size class -> idle buffers
        self._checked_out: dict[int, TempHandle] = {}      # id(buffer) -> live handle
        self._frame_stack: list[list[TempHandle]] = []     # open `frame()` scopes, innermost last
        self._tags: dict[str, TempHandle] = {}             # tag -> live tagged temporary

        # --- metrics ---
        self._bytes_pooled = 0
        self._bytes_checked_out = 0
        self._high_water_mark = 0
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Mapping[str, Buffer] -- persistent names and live temp tags share one namespace
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Buffer:
        if (buf := self._persistent.get(name)) is not None: return buf
        if (buf := self._pinned.get(name)) is not None: return buf
        if (handle := self._tags.get(name)) is not None: return handle
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return name in self._persistent or name in self._pinned or name in self._tags

    def __iter__(self) -> Iterator[str]:
        yield from self._persistent
        yield from self._pinned
        yield from self._tags

    def __len__(self) -> int:
        return len(self._persistent) + len(self._pinned) + len(self._tags)

    def _claim_name(self, name: str, kind: str) -> None:
        """Raise if `name` is already live anywhere else in the shared namespace. `kind` is what
        the CALLER is about to register ('persistent buffer', 'pinned buffer', or 'tagged temp'),
        used to name both the request and whatever already holds the name in the error.
        """
        if name in self._persistent:
            raise TlangError(f"Cannot register {kind} '{name}': a persistent buffer '{name}' already exists")
        if name in self._pinned:
            raise TlangError(f"Cannot register {kind} '{name}': a pinned buffer '{name}' already exists")
        if name in self._tags:
            raise TlangError(f"Cannot register {kind} '{name}': a temporary is already tagged '{name}'")

    def _align(self, size: int) -> int:
        # Round up to the next power of two, floored at `_min_size` (requires `_min_size >= 1`).
        if size < 0: raise TlangError(f"Buffer size must be non-negative, got {size}")
        return 1 << (max(size, self._min_size) - 1).bit_length()

    # ------------------------------------------------------------------
    # Persistent registry
    # ------------------------------------------------------------------

    def persistent_buffer(self, name: str, contents: bytes | None = None, size: int | None = None, dynamic: bool = True) -> Buffer:
        """Return the named persistent buffer, creating it on first call. Only `clear()` releases it.

        `name` doubles as this buffer's entry in the pool's `Mapping` -- `pool[name]` resolves it
        once created. Calling again with the same `name` is idempotent (returns the existing
        buffer); registering it while a live temporary already holds that tag raises `TlangError`
        naming both.
        """
        if (ret := self._persistent.get(name)) is not None: return ret
        self._claim_name(name, 'persistent buffer')
        if contents is not None: buf = self._ctx.buffer(data=contents, dynamic=dynamic)
        elif size is not None: buf = self._ctx.buffer(reserve=size, dynamic=dynamic)
        else: raise TlangError("No buffer size or content was provided")

        self._persistent[name] = buf
        logger.debug(f"Created persistent buffer '{name}' of size {buf.size}")
        return buf

    # ------------------------------------------------------------------
    # Pinned registry
    # ------------------------------------------------------------------

    def alloc_pinned(self, size: int, *, tag: str | None = None, read: bool = True, write: bool = True) -> PinnedBufferLike:
        """Allocate a persistent-mapped ("pinned") buffer: immutable `glBufferStorage` mapped
        once for its whole life, so the CPU touches it through a memoryview instead of
        `glBufferSubData`/`glGetBufferSubData` (see `pinned_buffer.PinnedBuffer` for the fencing
        contract that makes that safe by default).

        Immutable storage cannot be resized or orphaned, so pinned buffers never enter the
        transient free lists `alloc_temp` recycles from -- each is its own GL allocation, live
        until `free_pinned`/`clear()` releases it. Unlike `persistent_buffer`, calling this
        again does not return an existing buffer: every call allocates a new one.

        `tag`, if given, registers the buffer in this pool's shared name -> buffer `Mapping` --
        `pool[tag]` / `kernel.bind(**pool)` then resolve it exactly like a persistent buffer or a
        tagged temporary. Raises `TlangError` if `tag` is already live anywhere in that
        namespace. Without a `tag`, the buffer is still tracked by this pool (so `clear()` frees
        it) but does not resolve by name.

        Falls back to a plain `moderngl.Buffer` behind the identical API when
        `GL_ARB_buffer_storage` is unavailable -- check `.is_pinned` on the result if that
        distinction matters to the caller; a warning is logged either way.
        """
        if tag is not None: self._claim_name(tag, 'pinned buffer')
        buf = create_pinned_buffer(self._ctx, size, read=read, write=write)
        if tag is not None: self._pinned[tag] = buf
        else: self._pinned_untagged.append(buf)
        logger.debug(f"Allocated pinned buffer of size {size} (tag={tag!r}, is_pinned={buf.is_pinned})")
        return buf

    def free_pinned(self, tag_or_buffer: 'str | PinnedBufferLike') -> None:
        """Release one pinned buffer immediately: accepts either the tag it was allocated with,
        or the buffer object itself (the only way to address an untagged allocation). Untags it
        from the shared namespace first, then calls its `release()`. Raises `TlangError` for an
        unknown tag or a buffer this pool never allocated.
        """
        if isinstance(tag_or_buffer, str):
            buf = self._pinned.pop(tag_or_buffer, None)
            if buf is None: raise TlangError(f"No pinned buffer tagged '{tag_or_buffer}'")
        else:
            buf = tag_or_buffer
            if buf in self._pinned_untagged:
                self._pinned_untagged.remove(buf)
            elif (tag := next((k for k, v in self._pinned.items() if v is buf), None)) is not None:
                del self._pinned[tag]
            else:
                raise TlangError("free_pinned was given a buffer this pool did not allocate")
        buf.release()

    # ------------------------------------------------------------------
    # Transient pool
    # ------------------------------------------------------------------

    def alloc_temp(self, size: int, *, zero: bool = False, tag: str | None = None) -> TempHandle:
        """Check out a scratch buffer of at least `size` bytes.

        Recycled memory is undefined unless `zero=True` is passed. With `debug_poison=True`,
        buffers are stamped with `0xCD` when returned to the pool (see `free_temp`), so stale
        reads are an obvious sentinel rather than plausible garbage. Requests are only ever
        satisfied from their own power-of-two size class, never a larger one.

        `tag`, if given, registers this handle under that name in the pool's `Mapping` -- some
        other stage can then bind straight from the pool by name instead of being handed the
        handle directly. Raises `TlangError` if `tag` is already live (persistent or tagged
        temp); `free_temp` unregisters it, after which the name resolves to nothing until
        re-tagged.
        """
        aligned = self._align(size)
        bucket = self._free.get(aligned)
        if bucket:
            buf = bucket.pop()
            self._bytes_pooled -= aligned
            self._hits += 1
            logger.debug(f"Reusing pooled buffer of size {aligned} for request {size}")
        else:
            buf = self._ctx.buffer(reserve=aligned, dynamic=True)
            self._misses += 1
            logger.info(f"Allocating new buffer of size {aligned} for request {size}")

        if zero: clear_buffer(buf)

        if tag is not None: self._claim_name(tag, 'tagged temp')

        handle = TempHandle(buf, aligned, size, tag)
        self._checked_out[id(buf)] = handle
        self._bytes_checked_out += aligned
        self._high_water_mark = max(self._high_water_mark, self._bytes_pooled + self._bytes_checked_out)
        if self._frame_stack: self._frame_stack[-1].append(handle)
        if tag is not None: self._tags[tag] = handle
        return handle

    def free_temp(self, handle: TempHandle) -> None:
        """Return a checked-out buffer to its size class's free list.

        `handle` must be a live `TempHandle` from this pool's `alloc_temp`; freeing it twice, or
        touching it afterwards, raises `TlangError`. If `handle` was tagged, its tag is
        unregistered here, before the buffer is even returned to the free list -- the name stops
        resolving immediately, rather than continuing to point at memory someone else can now
        reuse.
        """
        if not isinstance(handle, TempHandle): raise TlangError(f"free_temp expects a TempHandle, got {type(handle).__name__}")
        tag = handle.tag
        buf = handle._kill()
        if self._checked_out.pop(id(buf), None) is None:
            raise TlangError("Buffer handle was not tracked as checked out by this pool")

        if tag is not None: self._tags.pop(tag, None)

        self._bytes_checked_out -= handle.size_class
        if self.debug_poison: clear_buffer(buf, _POISON_BYTE)
        self._free.setdefault(handle.size_class, []).append(buf)
        self._bytes_pooled += handle.size_class
        logger.debug(f"Returned buffer of size {handle.size_class} to pool")

    @contextmanager
    def temp(self, size: int, *, zero: bool = False, tag: str | None = None) -> Generator[TempHandle]:
        """`with pool.temp(size) as buf: ...` -- always freed (and untagged), even on exception."""
        handle = self.alloc_temp(size, zero=zero, tag=tag)
        try:
            yield handle
        finally:
            if handle.alive: self.free_temp(handle)

    @contextmanager
    def frame(self) -> Generator[None]:
        """Reclaim every `alloc_temp` made inside this scope on exit, including on exception.

        Handles already freed manually inside the block are left alone. Frames nest: an inner
        frame reclaims its own leftovers at its own `with` exit.
        """
        self._frame_stack.append([])
        try:
            yield
        finally:
            pending = self._frame_stack.pop()
            for handle in pending:
                if handle.alive: self.free_temp(handle)

    # ------------------------------------------------------------------
    # Metrics + trimming
    # ------------------------------------------------------------------

    def metrics(self) -> BufferPoolMetrics:
        """Snapshot of current pool usage (see `BufferPoolMetrics`)."""
        persistent_bytes = sum(buf.size for buf in self._persistent.values())
        return BufferPoolMetrics(
            bytes_pooled=self._bytes_pooled,
            bytes_checked_out=self._bytes_checked_out,
            high_water_mark=self._high_water_mark,
            hits=self._hits,
            misses=self._misses,
            persistent_bytes=persistent_bytes,
            persistent_count=len(self._persistent),
        )

    def trim(self) -> int:
        """Release every currently-idle transient buffer; return bytes freed.

        Checked-out buffers and the persistent registry are untouched; `hits`/`misses`/
        `high_water_mark` survive the trim.
        """
        released = self._bytes_pooled
        for bucket in self._free.values():
            for buf in bucket: buf.release()
        self._free.clear()
        self._bytes_pooled = 0
        if released: logger.info(f"Trimmed {released} bytes of idle pooled buffers")
        return released

    def clear(self) -> None:
        """Release every buffer this pool owns: persistent, pinned, pooled, and checked-out.

        Any handle a caller still holds becomes invalid. This is a safety net against leaking
        GL objects, not the intended flow -- callers should free everything first.
        """
        if self._checked_out:
            logger.warning(f"Clearing BufferPool with {len(self._checked_out)} buffer(s) still checked out")

        for buf in self._persistent.values(): buf.release()
        for buf in self._pinned.values(): buf.release()
        for buf in self._pinned_untagged: buf.release()
        for bucket in self._free.values():
            for buf in bucket: buf.release()
        for handle in list(self._checked_out.values()):
            handle._kill().release()

        self._persistent.clear()
        self._pinned.clear()
        self._pinned_untagged.clear()
        self._free.clear()
        self._checked_out.clear()
        self._frame_stack.clear()
        self._tags.clear()
        self._bytes_pooled = 0
        self._bytes_checked_out = 0
        # high_water_mark / hits / misses intentionally survive `clear()` -- historical.

        logger.info("Cleared all buffers from pool and persistent storage")
