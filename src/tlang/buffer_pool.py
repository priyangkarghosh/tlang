import ctypes
import logging
from bisect import bisect_left
from moderngl import Context, Buffer
from OpenGL.GL import glClearNamedBufferData, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT

logger = logging.getLogger(__name__)

class BufferPool:
    def __init__(self, ctx: Context, min_size: int = 256):
        self._ctx = ctx
        self._min_size = min_size
        
        self._persistent: dict[str, Buffer] = {}
        self._pool: list[Buffer] = []
        self._untagged: set[int] = set()

        self._tagged: dict[str, Buffer] = {}
        self._tag_map: dict[int, str] = {}

    def _align(self, size: int) -> int:
        return 1 << (max(size, self._min_size) - 1).bit_length()

    def persistent_buffer(self, name: str, contents: bytes | None = None, size: int | None = None, dynamic: bool = True) -> Buffer:
        if (ret := self._persistent.get(name, None)) is not None: return ret
        if contents is not None: buf = self._ctx.buffer(data=contents, dynamic=dynamic)
        elif size is not None: buf = self._ctx.buffer(reserve=size, dynamic=dynamic)
        else: raise ValueError("No buffer size or content was provided")

        self._persistent[name] = buf
        logger.debug(f"Created persistent buffer '{name}' of size {buf.size}")
        return buf

    def alloc_temp(self, size: int, tag: str | None = None) -> Buffer:
        # sort the pool by buffer sizes (ascending)
        aligned_size = self._align(size)
        self._pool.sort(key=lambda b: b.size)
        i = bisect_left(self._pool, aligned_size, key=lambda b: b.size)
        if i < len(self._pool):
            buf = self._pool.pop(i)
            logger.debug(f"Reusing buffer of size {buf.size} for request {aligned_size}")
        elif len(self._pool) > 0:
            buf = self._pool.pop()
            if aligned_size > buf.size: 
                logger.info(f"Orphaning buffer from {buf.size} to {aligned_size}")
                buf.orphan(aligned_size)
        else:
            buf = self._ctx.buffer(reserve=aligned_size, dynamic=True)
            logger.info(f"Creating new buffer of size: {aligned_size}")

        # track usage
        if tag:
            assert tag not in self._tagged, f"Buffer with tag '{tag}' already exists"
            self._tagged[tag] = buf
            self._tag_map[buf.glo] = tag
        else:
            assert buf.glo not in self._untagged
            self._untagged.add(buf.glo)
        return buf

    def free_temp(self, buffer: Buffer):
        glo = buffer.glo
        if glo in self._untagged:
            self._untagged.remove(glo)
        elif (tag := self._tag_map.get(glo)) is not None:
            self._tagged.pop(tag, None)
            self._tag_map.pop(glo, None)
        else:
            raise ValueError("Buffer was not tracked as in-use")

        # insert buffer back into the pool in sorted order
        self._pool.append(buffer)
        logger.debug(f"Returned buffer of size {buffer.size} to pool")

    def grab_tagged(self, tag: str) -> Buffer:
        if (buf := self._tagged.get(tag)) is None:
            raise ValueError(f"Buffer with tag '{tag}' not found")
        return buf
    
    @staticmethod
    def clear_buffer(buffer: Buffer, value: int = 0):
        buffer.orphan()
        ptr = ctypes.byref(ctypes.c_uint(value))
        glClearNamedBufferData(
            buffer.glo,
            GL_R32UI,
            GL_RED_INTEGER,
            GL_UNSIGNED_INT,
            ptr
        )

    def clear(self):
        for buf in self._persistent.values(): buf.release()
        for buf in self._pool: buf.release()

        self._pool.clear()
        self._persistent.clear()
        self._untagged.clear()
        self._tagged.clear()
        self._tag_map.clear()

        logger.info("Cleared all buffers from pool and persistent storage")