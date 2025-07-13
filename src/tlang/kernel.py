import time
from typing import Any
from moderngl import (
    SHADER_STORAGE_BARRIER_BIT, Buffer, ComputeShader,
    Context, StorageBlock, Uniform
)
from OpenGL.GL import (
    glBindBufferRange, GL_ATOMIC_COUNTER_BUFFER,
    glDispatchComputeIndirect, GL_DISPATCH_INDIRECT_BUFFER,
    glBindBuffer, glMemoryBarrier, GL_COMMAND_BARRIER_BIT,
    glUseProgram, GL_SHADER_STORAGE_BARRIER_BIT
)

__slots__ = ('_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache')
class Kernel:
    __slots__ = ('_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache')

    def __init__(self, ctx: Context, name: str, shader: ComputeShader):
        self._ctx = ctx
        self._name = name
        self._mglo = shader
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}

    @property
    def ctx(self): return self._ctx # mgl context

    @property
    def name(self): return self._name # shader name

    @property
    def mglo(self): return self._mglo # moderngl object

    @property
    def glo(self) -> int: return self._mglo.glo # gl object

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
        #t0 = time.perf_counter()
        glUseProgram(self.glo)
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer.glo)
        glDispatchComputeIndirect(offset)
        if barrier: self._ctx.memory_barrier(barrier_bits)
        # self._ctx.finish()
        # t1 = time.perf_counter()
        # print(f"Time for kernel \"{self.name}\": {(t := (t1 - t0) * 1000)} ms")

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
        print(f"Time for kernel \"{self.name}\": {(t := (t1 - t0) * 1000)} ms")
        return t
   
    def set_uniforms(self, **uniforms: Any) -> None:
        loc = self.set_uniform
        for k, v in uniforms.items():
            loc(k, v)

    # set uniform for a kernel
    def set_uniform(self, uniform: str, value: Any) -> None:
        if self._uniform_cache.get(uniform) == value: return
        else:
            block = self._mglo.get(uniform, None)
            if not isinstance(block, Uniform): raise ValueError(f"'{uniform}' is not a uniform")
            self._uniform_cache[uniform] = value
            block.value = value

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
                raise TypeError(f"'{buffer_name}' is not a valid buffer block (Missing binding)")
            self._binding_cache[buffer_name] = binding = block.binding
        buffer.bind_to_storage_buffer(binding, offset, size)
    
    def bind_atomic_counter(
        self, binding: int, buffer: Buffer, offset: int = 0
    ) -> None:
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, 4)

    def bind_atomic_counters(
        self, *buffers: tuple[int, Buffer] | tuple[int, Buffer, int]
    ) -> None:
        loc = self.bind_atomic_counter
        for v in buffers: loc(*v) if len(v) == 3 else loc(v[0], v[1])