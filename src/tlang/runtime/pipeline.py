# -------------------------------------------------------------
# @file          pipeline.py
# @author        Priyangkar Ghosh
# @created       2026-09-07
# @description   Graphics program wrapper mirroring Kernel's ergonomics: name-keyed SSBO/uniform
#                binding for [program(...)]s instead of raw moderngl calls.
# @license       MIT
# -------------------------------------------------------------

import logging
from collections.abc import Mapping
from typing import Any
from moderngl import Buffer, Context, Program, StorageBlock, UniformBlock, Uniform

from tlang.errors import SourceLocation, TlangBindingError

logger = logging.getLogger(__name__)

# sentinel distinguishing "never cached" from a cached value of `None`
_NOT_CACHED = object()

class Pipeline:
    """Wraps a linked `moderngl.Program` for one `[program(...)]` with name-keyed SSBO/uniform
    binding, so callers never hardcode binding numbers. Get one via `Shader.get_pipeline(name)`."""
    __slots__ = ('_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache')

    def __init__(self, ctx: Context, name: str, program: Program, bindings: Mapping[str, int] | None = None):
        self._ctx = ctx
        self._name = name
        self._mglo = program
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # The binding map BindingRegistry.allocate_artifact decided for the program as a whole,
        # kept for debugging/diffing; bind_ssbo independently reflects each binding by name.
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}

    @property
    def ctx(self): return self._ctx

    @property
    def name(self): return self._name

    @property
    def mglo(self): return self._mglo

    @property
    def glo(self) -> int: return self._mglo.glo

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

    def set_uniforms(self, **uniforms: Any) -> None:
        loc = self.set_uniform
        for k, v in uniforms.items():
            loc(k, v)

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
        # `==` on numpy arrays can return an array, not a bool, which raises on truthiness
        # checks; treat anything that can't cleanly resolve to a bool as "not equal".
        try:
            return bool(a == b)
        except (ValueError, TypeError):
            return False

    def bind_ssbos(self, **buffers: Buffer | tuple[Buffer, int, int]) -> None:
        loc = self.bind_ssbo
        for k, v in buffers.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_ssbo(
        self, buffer_name: str, buffer: Buffer, offset: int = 0, size: int = -1
    ) -> None:
        if (binding := self._binding_cache.get(buffer_name, None)) is None:
            block = self._mglo.get(buffer_name, None)
            if not isinstance(block, StorageBlock):
                raise TlangBindingError(f"'{buffer_name}' is not a valid buffer block (Missing binding)", SourceLocation(module=self._name))
            self._binding_cache[buffer_name] = binding = block.binding
        # offset/size are keyword-only on moderngl <= 5.8.x; positional args raise TypeError there.
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
