# -------------------------------------------------------------
# @file          kernel.py
# @author        Priyangkar Ghosh
# @created       2025-06-08
# @description   Kernel/Compute Shader wrapper with helper functions
# @license       MIT
# -------------------------------------------------------------

import logging
import time
from collections.abc import Mapping
from typing import Any
from moderngl import (
    SHADER_STORAGE_BARRIER_BIT, Buffer, ComputeShader,
    Context, StorageBlock, Uniform, UniformBlock
)
from OpenGL.GL import (
    glBindBufferRange, GL_ATOMIC_COUNTER_BUFFER,
    glDispatchComputeIndirect, GL_DISPATCH_INDIRECT_BUFFER,
    glBindBuffer, glUseProgram
)

from tlang.errors import SourceLocation, TlangBindingError

logger = logging.getLogger(__name__)

# sentinel distinguishing "never cached" from a cached value of `None`
_NOT_CACHED = object()

class Kernel:
    __slots__ = ('_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache')

    def __init__(self, ctx: Context, name: str, shader: ComputeShader, bindings: Mapping[str, int] | None = None):
        self._ctx = ctx
        self._name = name
        self._mglo = shader
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # the name -> binding map `BindingRegistry.allocate_artifact` decided
        # for this kernel, exposed for debugging/diffing across builds --
        # NOT consulted by `bind_ssbo` below, which independently reflects
        # (and caches) each binding from the linked `shader` by name. The two
        # should always agree (see `BindingRegistry.verify_link`, run before
        # this object is constructed), but `bind_ssbo`'s cache is the one
        # actually used to bind buffers.
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}

    @property
    def ctx(self): return self._ctx # mgl context

    @property
    def name(self): return self._name # shader name

    @property
    def mglo(self): return self._mglo # moderngl object

    @property
    def glo(self) -> int: return self._mglo.glo # gl object

    @property
    def bindings(self) -> Mapping[str, int]: return self._bindings # name -> assigned SSBO binding

    @property
    def uniform_blocks(self) -> Mapping[str, int]:
        """Uniform-block name -> binding, reflected from the linked program."""
        return {n: b.binding for n in self._mglo
                if isinstance(b := self._mglo[n], UniformBlock)}

    def __getitem__(self, uniform: str) -> Any:
        return self._mglo[uniform]

    def __setitem__(self, uniform: str, value: Any):
        self.set_uniform(uniform, value)

    def __contains__(self, value: str) -> bool:
        return value in self._mglo

    def dispatch(
        self, 
        group_x: int = 1, 
        group_y: int = 1, 
        group_z: int = 1, 
        barrier: bool = True, 
        barrier_bits: int = SHADER_STORAGE_BARRIER_BIT
    ) -> None:
        self._mglo.run(group_x, group_y, group_z)
        if barrier: self._ctx.memory_barrier(barrier_bits)
    
    def dispatch_indirect(
        self, 
        buffer: Buffer, 
        offset: int = 0,
        barrier: bool = True, 
        barrier_bits: int = SHADER_STORAGE_BARRIER_BIT
    ) -> None:
        glUseProgram(self.glo)
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer.glo)
        glDispatchComputeIndirect(offset)
        if barrier: self._ctx.memory_barrier(barrier_bits)

    def dispatch_timed(
        self, 
        group_x: int = 1, 
        group_y: int = 1, 
        group_z: int = 1
    ) -> float:
        t0 = time.perf_counter()
        self.dispatch(group_x, group_y, group_z)
        self._ctx.memory_barrier()
        self._ctx.finish()
        t1 = time.perf_counter()
        logger.info('Time for kernel "%s": %.3f ms', self.name, (t := (t1 - t0) * 1000))
        return t
   
    def set_uniforms(self, **uniforms: Any) -> None:
        loc = self.set_uniform
        for k, v in uniforms.items():
            loc(k, v)

    # set uniform for a kernel
    def set_uniform(self, uniform: str, value: Any) -> None:
        cached = self._uniform_cache.get(uniform, _NOT_CACHED)
        if cached is not _NOT_CACHED and self._values_equal(cached, value): return

        block = self._mglo.get(uniform, None)
        if not isinstance(block, Uniform):
            raise TlangBindingError(f"'{uniform}' is not a uniform", SourceLocation(module=self._name))
        self._uniform_cache[uniform] = value
        block.value = value

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        # `==` on numpy arrays (and some other array-likes) returns an array
        # rather than a bool, which raises on the truthiness check below for
        # anything but a single-element array. Treat anything that can't
        # cleanly resolve to a bool as "not equal" so we never skip a
        # genuinely-changed uniform update.
        try:
            return bool(a == b)
        except (ValueError, TypeError):
            return False

    def bind_ssbos(self, **buffers: Buffer | tuple[Buffer, int, int]) -> None:
        loc = self.bind_ssbo
        for k, v in buffers.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    # bind buffer, size is -1 by 0 to not specify a fixed size
    def bind_ssbo(
        self, buffer_name: str, buffer: Buffer, offset: int = 0, size: int = -1
    ) -> None:  
        if (binding := self._binding_cache.get(buffer_name, None)) is None:
            block = self._mglo.get(buffer_name, None)
            if not isinstance(block, StorageBlock):
                raise TlangBindingError(f"'{buffer_name}' is not a valid buffer block (Missing binding)", SourceLocation(module=self._name))
            self._binding_cache[buffer_name] = binding = block.binding
        # offset/size are keyword-only on moderngl <= 5.8.x; passing them
        # positionally raises TypeError there. Keyword form works on every
        # supported version -- see https://moderngl.readthedocs.io/en/5.8.2
        buffer.bind_to_storage_buffer(binding, offset=offset, size=size)

    def bind_ubos(self, **buffers: Buffer | tuple[Buffer, int, int]) -> None:
        loc = self.bind_ubo
        for k, v in buffers.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_ubo(
        self, block_name: str, buffer: Buffer, offset: int = 0, size: int = -1
    ) -> None:
        """Bind `buffer` to the uniform block `block_name` declared by [uniforms(std140)]."""
        if (binding := self._ubo_cache.get(block_name, None)) is None:
            block = self._mglo.get(block_name, None)
            if not isinstance(block, UniformBlock):
                raise TlangBindingError(
                    f"'{block_name}' is not a valid uniform block (Missing binding)",
                    SourceLocation(module=self._name),
                )
            self._ubo_cache[block_name] = binding = block.binding
        buffer.bind_to_uniform_block(binding, offset=offset, size=size)
    
    def bind_atomic_counter(
        self, binding: int, buffer: Buffer, offset: int = 0
    ) -> None:
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, 4)

    def bind_atomic_counters(
        self, *buffers: tuple[int, Buffer] | tuple[int, Buffer, int]
    ) -> None:
        loc = self.bind_atomic_counter
        for v in buffers: loc(*v) if len(v) == 3 else loc(v[0], v[1])