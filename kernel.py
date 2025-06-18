import time
from typing import Any
from moderngl import Buffer, ComputeShader, Context


class Kernel:
    def __init__(self, ctx: Context, name: str, shader: ComputeShader):
        self._ctx = ctx
        self._name = name
        self._mglo = shader
        self._uniform_cache: dict[str, Any] = {}

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

    def dispatch(self, group_x: int = 1, group_y: int = 1, group_z: int = 1):
        self._mglo.run(group_x, group_y, group_z)

    def dispatch_timed(self, group_x: int = 1, group_y: int = 1, group_z: int = 1):
        t0 = time.perf_counter()
        self.dispatch(group_x, group_y, group_z)
        self._ctx.memory_barrier()
        self._ctx.finish()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000
    
    def has_uniform(self, name: str) -> bool:
        return name in self._mglo

    # set uniform for a kernel
    def set_uniform(self, uniform: str, value: Any, cache=True):
        # make sure that the uniform actually exists
        if uniform not in self._mglo: raise ValueError('Uniform does not exist')

        # check the uniform cache and don't set it if it's already the same value
        # -> only set it if it hasn't
        if cache and self._uniform_cache.get(uniform) == value: return
        self._mglo[uniform] = self._uniform_cache[uniform] = value

    # bind buffer, size is -1 by 0 to not specify a fixed size
    def bind_buffer(self, buffer_name: str, buffer: Buffer, offset: int = 0, size: int = -1):
        if not hasattr(block := self._mglo[buffer_name], 'binding'):
            raise TypeError(f"'{buffer_name}' is not a valid buffer block (Missing binding!)")
        buffer.bind_to_storage_buffer(int(block.binding), offset, size) # type: ignore

