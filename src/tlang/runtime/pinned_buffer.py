# -------------------------------------------------------------
# @file          pinned_buffer.py
# @author        Priyangkar Ghosh
# @created       2026-09-10
# @description   Persistent-mapped ("pinned") GL buffers: immutable storage mapped ONCE for the
#                life of the buffer, so the CPU reads/writes it through a memoryview instead of
#                glBufferSubData/glGetBufferSubData. Duck-types moderngl.Buffer for everything
#                Kernel/Pipeline touch, with a plain-moderngl.Buffer fallback when
#                GL_ARB_buffer_storage is unavailable.
# @license       MIT
# -------------------------------------------------------------

import ctypes
import logging

from moderngl import Context

from OpenGL.GL import (
    glGenBuffers, glBindBuffer, glDeleteBuffers, glMapBufferRange, glUnmapBuffer,
    glBindBufferRange, GL_SHADER_STORAGE_BUFFER,
    GL_MAP_READ_BIT, GL_MAP_WRITE_BIT, GL_MAP_PERSISTENT_BIT, GL_MAP_COHERENT_BIT,
    glFenceSync, glClientWaitSync, glDeleteSync,
    GL_SYNC_GPU_COMMANDS_COMPLETE, GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_TIMEOUT_EXPIRED, GL_WAIT_FAILED,
)
from OpenGL.GL.ARB.buffer_storage import glBufferStorage

from tlang.errors import TlangError

logger = logging.getLogger(__name__)

# Generous but finite: a fence that never signals is a real bug (a dispatch that never
# completed, or a context lost) worth surfacing as an error, not hanging the process forever.
_DEFAULT_WAIT_TIMEOUT_NS = 5_000_000_000  # 5s


def buffer_storage_supported(ctx: Context) -> bool:
    """Real feature-detection for `GL_ARB_buffer_storage` (core in GL 4.4; tlang's own test
    context is created with `require=430`, so this cannot be assumed). `create_pinned_buffer`
    uses this to decide whether to allocate a real `PinnedBuffer` or fall back to a plain
    `moderngl.Buffer`; tests force the fallback path by monkeypatching this function.
    """
    try:
        return 'GL_ARB_buffer_storage' in ctx.extensions
    except Exception:  # pragma: no cover -- defensive, `extensions` is a plain property
        return False


class PinnedBuffer:
    """A GL buffer allocated with immutable storage (`glBufferStorage`) and mapped ONCE,
    persistently, for its entire life, so CPU code reads/writes it through a `memoryview`
    instead of round-tripping every access through `glBufferSubData`/`glGetBufferSubData`.

    Duck-types the slice of `moderngl.Buffer` tlang's binding layer actually touches --
    `.glo`, `.size`, `.bind_to_storage_buffer(binding, offset=0, size=-1)`, `.read`, `.write`,
    `.release` -- so `Kernel.bind_ssbo`/`bind` work on this exactly as they do on a real
    `moderngl.Buffer`, with no special-casing anywhere: `kernel.bind(**pool)` just works.

    Never binds itself into GL's (process-global) indexed SSBO binding table on construction --
    that table is tlang's to allocate (see `Kernel.bind_ssbo`). The `glBindBuffer` calls in here
    only touch the unindexed `GL_SHADER_STORAGE_BUFFER` target slot, which `glBufferStorage`/
    `glMapBufferRange` require and which is unrelated to that indexed table.

    ## Fencing contract

    A coherent persistent mapping provides ZERO ordering between the CPU and the GPU on its own:
    the CPU can read bytes mid-write from an in-flight compute dispatch, or overwrite bytes a
    dispatch has not finished reading, and neither ever raises -- it silently produces torn or
    stale data. This class closes that hole with an explicit fence, and makes the safe path the
    default:

      - `fence()` -- call this immediately after submitting GL work that reads or writes this
        buffer (right after a `kernel.dispatch(...)`). Records a `glFenceSync` covering every GL
        command issued so far, replacing (and deleting) whatever fence was recorded before.
      - `read()`/`write()` default to `sync=True`: before touching the mapping they block on the
        most recent fence via `glClientWaitSync` (flushing the queue), so every GL command up to
        that fence has retired before CPU bytes are touched. No `fence()` call yet means nothing
        recorded to wait on, so the very first access proceeds immediately. This is what makes a
        plain `.read()` torn-data-free by default.
      - `sync=False` skips that wait, for a caller who has already synchronised some other way
        (e.g. `ctx.finish()`, or its own fence) and wants to avoid a redundant stall.
      - `.mapping` (the raw `memoryview`) is the explicit, opt-in, UNSYNCED escape hatch --
        slicing or assigning through it never waits on anything. Only touch it after your own
        sync bookkeeping, or when you know there is no concurrent GPU access.

    CPU -> GPU is already ordered by `GL_MAP_COHERENT_BIT` alone: a client write is guaranteed
    visible to any GL command issued afterwards, no extra call needed. The two hazards `fence`/
    `wait` guard against are the other directions: GPU-write-then-CPU-read, and
    CPU-write-racing-a-still-in-flight GPU-read.

    `.is_pinned` is always `True` here (see `PinnedBufferFallback` for the `False` case).
    """

    __slots__ = ('_glo', '_size', '_mv', '_fence', '_released', '_read', '_write')

    def __init__(self, ctx: Context, size: int, *, read: bool = True, write: bool = True) -> None:
        if size <= 0: raise TlangError(f"Pinned buffer size must be positive, got {size}")
        if not (read or write): raise TlangError("Pinned buffer needs at least one of read=True/write=True")

        self._size = size
        self._fence = None
        self._released = False
        self._read = read
        self._write = write

        flags = GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT
        if read: flags |= GL_MAP_READ_BIT
        if write: flags |= GL_MAP_WRITE_BIT

        glo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, glo)
        try:
            glBufferStorage(GL_SHADER_STORAGE_BUFFER, size, None, flags)
            ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, flags)
        finally:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)  # never leave the target slot occupied

        if not ptr:
            glDeleteBuffers(1, [glo])
            raise TlangError("glMapBufferRange returned NULL while mapping a persistent buffer")

        self._glo = glo
        addr = ctypes.cast(ptr, ctypes.c_void_p).value
        # Exactly this shape: a plain c_ubyte array (or an uncast c_char view) raises
        # `NotImplementedError: memoryview: unsupported format` on slice assignment.
        self._mv = memoryview((ctypes.c_char * size).from_address(addr)).cast('B')
        logger.debug(f"Mapped persistent buffer glo={glo} size={size} read={read} write={write}")

    def _check_alive(self) -> None:
        if self._released: raise TlangError("Use of pinned buffer after release()")

    @property
    def is_pinned(self) -> bool:
        """True: this is a real `glBufferStorage`-backed persistent mapping."""
        return True

    @property
    def glo(self) -> int:
        self._check_alive()
        return self._glo

    @property
    def size(self) -> int:
        return self._size

    @property
    def mapping(self) -> memoryview:
        """The raw persistent mapping. Zero-copy, UNSYNCED -- see the class docstring's fencing
        contract before reading/writing through this directly instead of `read()`/`write()`."""
        self._check_alive()
        return self._mv

    def bind_to_storage_buffer(self, binding: int, offset: int = 0, size: int = -1) -> None:
        """Bind this buffer into GL's indexed SSBO binding table at `binding` -- exactly the
        `moderngl.Buffer` method `Kernel.bind_ssbo` calls at dispatch time. Never called by this
        class itself; only ever invoked by tlang's own binding layer or an explicit caller.
        """
        self._check_alive()
        sz = self._size - offset if size == -1 else size
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding, self._glo, offset, sz)

    def fence(self) -> None:
        """Record a fence covering every GL command submitted so far. Call this immediately
        after issuing GPU work that reads or writes this buffer. Replaces (and deletes) any
        fence recorded by a previous call."""
        self._check_alive()
        if self._fence is not None: glDeleteSync(self._fence)
        self._fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    def wait(self, timeout_ns: int = _DEFAULT_WAIT_TIMEOUT_NS) -> None:
        """Block until the most recently recorded `fence()` has signalled. No-op if `fence()`
        was never called (or already waited-out) -- nothing outstanding to wait on. Used
        internally by `read()`/`write()` under `sync=True`; exposed directly for a caller that
        wants to wait without immediately touching the mapping.
        """
        self._check_alive()
        if self._fence is None: return
        status = glClientWaitSync(self._fence, GL_SYNC_FLUSH_COMMANDS_BIT, timeout_ns)
        if status == GL_WAIT_FAILED:
            raise TlangError("glClientWaitSync failed while waiting on a pinned buffer's fence")
        if status == GL_TIMEOUT_EXPIRED:
            raise TlangError(f"Timed out after {timeout_ns}ns waiting for the GPU to finish with a pinned buffer")
        glDeleteSync(self._fence)
        self._fence = None

    def read(self, size: int = -1, offset: int = 0, *, sync: bool = True) -> bytes:
        """Copy `size` bytes (default: the rest of the buffer) out of the mapping, starting at
        `offset`. `sync=True` (default) waits on the pending fence first -- see the class
        docstring's fencing contract; pass `sync=False` only when you have already synchronised.
        """
        self._check_alive()
        if not self._read: raise TlangError("Pinned buffer was allocated with read=False")
        sz = self._size - offset if size == -1 else size
        if offset < 0 or sz < 0 or offset + sz > self._size:
            raise TlangError(f"read(size={size}, offset={offset}) out of bounds for buffer of size {self._size}")
        if sync: self.wait()
        return bytes(self._mv[offset:offset + sz])

    def write(self, data: bytes, offset: int = 0, *, sync: bool = True) -> None:
        """Copy `data` into the mapping at `offset`. `sync=True` (default) waits on the pending
        fence first -- see the class docstring's fencing contract; pass `sync=False` only when
        you have already synchronised.
        """
        self._check_alive()
        if not self._write: raise TlangError("Pinned buffer was allocated with write=False")
        n = len(data)
        if offset < 0 or offset + n > self._size:
            raise TlangError(f"write of {n} bytes at offset {offset} overflows buffer of size {self._size}")
        if sync: self.wait()
        self._mv[offset:offset + n] = data

    def release(self) -> None:
        """Unmap and delete the underlying GL buffer. Idempotent like `moderngl.Buffer.release()`
        -- calling it again is a silent no-op. Any OTHER method call after release raises
        `TlangError` instead of touching freed GPU memory.
        """
        if self._released: return
        self._released = True
        if self._fence is not None:
            glDeleteSync(self._fence)
            self._fence = None
        self._mv.release()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._glo)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glDeleteBuffers(1, [self._glo])
        logger.debug(f"Released pinned buffer glo={self._glo}")

    def __del__(self) -> None:
        try:
            if not getattr(self, '_released', True): self.release()
        except Exception:  # pragma: no cover -- best-effort cleanup during interpreter teardown
            pass

    def __repr__(self) -> str:
        state = 'released' if self._released else 'alive'
        return f'<PinnedBuffer glo={getattr(self, "_glo", "?")} size={self._size} {state}>'


class PinnedBufferFallback:
    """Fallback used when `GL_ARB_buffer_storage` is unavailable: a plain `moderngl.Buffer`
    behind the exact same API `PinnedBuffer` exposes, so calling code is portable across both
    without special-casing which one it got.

    There is no real mapping here. `read()`/`write()` go through
    `glGetBufferSubData`/`glBufferSubData` -- already correctly ordered against other GL commands
    by the driver, which is precisely the synchronisation `PinnedBuffer.fence()`/`wait()` exist to
    replace, so `fence()`/`wait()` are no-ops and `sync=` is accepted but has nothing to do.
    `.mapping` raises `TlangError`: there is no zero-copy view to hand back. `.is_pinned` is
    `False`, the one thing this class exists to make observable to a caller that cares.
    """

    __slots__ = ('_buf', '_read', '_write')

    def __init__(self, ctx: Context, size: int, *, read: bool = True, write: bool = True) -> None:
        if not (read or write): raise TlangError("Pinned buffer needs at least one of read=True/write=True")
        self._buf = ctx.buffer(reserve=size, dynamic=True)
        self._read = read
        self._write = write

    @property
    def is_pinned(self) -> bool:
        """False: no real persistent mapping backs this buffer (GL_ARB_buffer_storage was
        unavailable) -- it is a plain moderngl.Buffer behind the same API."""
        return False

    @property
    def glo(self) -> int:
        return self._buf.glo

    @property
    def size(self) -> int:
        return self._buf.size

    @property
    def mapping(self):
        raise TlangError(
            "No real persistent mapping is available (GL_ARB_buffer_storage fallback); "
            "use read()/write() instead of .mapping"
        )

    def bind_to_storage_buffer(self, binding: int, offset: int = 0, size: int = -1) -> None:
        self._buf.bind_to_storage_buffer(binding, offset=offset, size=size)

    def fence(self) -> None:
        pass  # glBufferSubData/glGetBufferSubData are already correctly ordered by the driver

    def wait(self, timeout_ns: int = _DEFAULT_WAIT_TIMEOUT_NS) -> None:
        pass

    def read(self, size: int = -1, offset: int = 0, *, sync: bool = True) -> bytes:
        if not self._read: raise TlangError("Pinned buffer was allocated with read=False")
        return self._buf.read(size, offset)

    def write(self, data: bytes, offset: int = 0, *, sync: bool = True) -> None:
        if not self._write: raise TlangError("Pinned buffer was allocated with write=False")
        self._buf.write(data, offset)

    def release(self) -> None:
        self._buf.release()

    def __repr__(self) -> str:
        return f'<PinnedBufferFallback size={self.size}>'


def create_pinned_buffer(ctx: Context, size: int, *, read: bool = True, write: bool = True) -> 'PinnedBuffer | PinnedBufferFallback':
    """Feature-detect `GL_ARB_buffer_storage` and allocate the best backing available: a real
    `PinnedBuffer` when the extension is present, else a `PinnedBufferFallback` wrapping a plain
    `moderngl.Buffer` behind the identical API. Never raises for a missing extension -- logs a
    warning and returns the fallback instead, so a caller with no special-casing degrades
    gracefully rather than crashing on an older/other driver.
    """
    if buffer_storage_supported(ctx):
        return PinnedBuffer(ctx, size, read=read, write=write)
    logger.warning(
        "GL_ARB_buffer_storage not available on this context -- falling back to a plain "
        "moderngl.Buffer for a pinned-buffer allocation (no real persistent mapping; "
        ".is_pinned is False on the result)"
    )
    return PinnedBufferFallback(ctx, size, read=read, write=write)


__all__ = [
    'PinnedBuffer',
    'PinnedBufferFallback',
    'buffer_storage_supported',
    'create_pinned_buffer',
]
