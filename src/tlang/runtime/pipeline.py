# -------------------------------------------------------------
# @file          pipeline.py
# @author        Priyangkar Ghosh
# @created       2026-09-07
# @description   Graphics program wrapper mirroring Kernel's ergonomics: name-keyed SSBO/uniform
#                binding for [program(...)]s instead of raw moderngl calls.
# @license       MIT
# -------------------------------------------------------------

import difflib
import logging
from collections.abc import Mapping
from typing import Any
from moderngl import Buffer, Context, Program, StorageBlock, Texture, UniformBlock, Uniform

from tlang.errors import SourceLocation, TlangBindingError
from tlang.runtime.kernel import (
    bump_image_table_generation, bump_ssbo_table_generation, bump_texture_table_generation,
)

logger = logging.getLogger(__name__)

# sentinel distinguishing "never cached" from a cached value of `None`
_NOT_CACHED = object()

class Pipeline:
    """Wraps a linked `moderngl.Program` for one `[program(...)]` with name-keyed SSBO/uniform
    binding, so callers never hardcode binding numbers. Get one via `Shader.get_pipeline(name)`."""
    __slots__ = (
        '_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache', '_buffer_source',
        '_texture_units', '_image_units',
    )

    def __init__(
        self, ctx: Context, name: str, program: Program, bindings: Mapping[str, int] | None = None,
        texture_units: Mapping[str, int] | None = None, image_units: Mapping[str, int] | None = None,
    ):
        self._ctx = ctx
        self._name = name
        self._mglo = program
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # The binding map BindingRegistry.allocate_artifact decided for the program as a whole,
        # kept for debugging/diffing; bind_ssbo independently reflects each binding by name.
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}
        # buffer source consulted by `bind()` for any required name not passed explicitly --
        # see the `source` property. `None` means bind() can only resolve explicit kwargs.
        self._buffer_source: Mapping[str, Buffer] | None = None
        # name -> assigned texture/image unit, the sampler/image counterpart of `_bindings` --
        # see `Kernel.texture_units`/`Kernel.image_units`, which this mirrors.
        self._texture_units: dict[str, int] = dict(texture_units) if texture_units is not None else {}
        self._image_units: dict[str, int] = dict(image_units) if image_units is not None else {}

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
    def texture_units(self) -> Mapping[str, int]: return self._texture_units # name -> assigned texture unit

    @property
    def image_units(self) -> Mapping[str, int]: return self._image_units # name -> assigned image unit

    @property
    def buffer_source(self) -> Mapping[str, Buffer] | None:
        """The `Mapping[str, Buffer]` that `bind()` draws unnamed required blocks from -- see
        `Kernel.buffer_source`, which this mirrors. Read fresh on every `bind()` call, never cached."""
        return self._buffer_source

    @buffer_source.setter
    def buffer_source(self, value: Mapping[str, Buffer] | None) -> None:
        self._buffer_source = value

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
        """Bind `buffer` to `buffer_name` IMMEDIATELY, unlike `Kernel.bind_ssbo`.

        `Kernel` can defer its binds because every dispatch entry point re-asserts the kernel's
        full recorded set right before running -- there's a hook to do it at. A `Pipeline` has no
        such hook: drawing happens in moderngl's `VAO.render`, entirely outside tlang, so there is
        no "just before this pipeline runs" moment to re-assert at. Binding immediately is
        therefore the only option here.

        This does still bump the same process-global generation counter `Kernel` uses (see
        `tlang.runtime.kernel.bump_ssbo_table_generation`), so any `Kernel` that dispatches after
        this call correctly notices the table has moved and re-asserts its own bindings. The
        asymmetry this leaves: a compute dispatch issued BETWEEN a `pipeline.bind_ssbo(s)` call
        and the eventual `VAO.render` draw can still silently rewire this pipeline's bindings out
        from under it, and tlang currently has no mechanism to detect or prevent that -- only to
        keep kernels honest about it.
        """
        if (binding := self._binding_cache.get(buffer_name, None)) is None:
            block = self._mglo.get(buffer_name, None)
            if not isinstance(block, StorageBlock):
                raise TlangBindingError(f"'{buffer_name}' is not a valid buffer block (Missing binding)", SourceLocation(module=self._name))
            self._binding_cache[buffer_name] = binding = block.binding
        # offset/size are keyword-only on moderngl <= 5.8.x; positional args raise TypeError there.
        buffer.bind_to_storage_buffer(binding, offset=offset, size=size)
        bump_ssbo_table_generation()

    def bind(self, **explicit: Buffer | tuple[Buffer, int, int]) -> None:
        """Bind exactly this program's REQUIRED set (`self.bindings`) by name, drawn from
        `explicit` first and then from `self.buffer_source` -- mirrors `Kernel.bind`, but (like
        `bind_ssbo`) binds immediately rather than deferring, since a `Pipeline` has no
        dispatch-time hook to re-assert at."""
        for buffer_name in self._bindings:
            if buffer_name in explicit:
                value = explicit[buffer_name]
            elif self._buffer_source is not None:
                if buffer_name not in self._buffer_source:
                    raise TlangBindingError(
                        f"Pipeline '{self._name}' requires buffer '{buffer_name}' but it was not "
                        f"found in the bound buffer source.{self._suggest_from_source(buffer_name)}",
                        SourceLocation(module=self._name),
                    )
                value = self._buffer_source[buffer_name]
            else:
                raise TlangBindingError(
                    f"Pipeline '{self._name}' requires buffer '{buffer_name}' but it was not in "
                    f"the provided buffers",
                    SourceLocation(module=self._name),
                )
            if isinstance(value, tuple): self.bind_ssbo(buffer_name, *value)
            else: self.bind_ssbo(buffer_name, value)

    def _suggest_from_source(self, name: str) -> str:
        if self._buffer_source is None: return ''
        matches = difflib.get_close_matches(name, sorted(self._buffer_source), n=1)
        return f" Did you mean '{matches[0]}'?" if matches else ''

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

    def bind_textures(self, **textures: Texture) -> None:
        loc = self.bind_texture
        for k, v in textures.items(): loc(k, v)

    def bind_texture(self, name: str, texture: Texture) -> None:
        """Bind `texture` to `name`'s texture unit IMMEDIATELY, unlike `Kernel.bind_texture`.

        Same asymmetry as `bind_ssbo` above, for the same reason: a `Pipeline` has no
        dispatch-time hook to defer to (drawing happens in moderngl's `VAO.render`, entirely
        outside tlang), so binding immediately is the only option here. Also bumps the shared
        texture-unit generation counter (see `tlang.runtime.kernel.bump_texture_table_generation`)
        so any `Kernel` that dispatches after this call notices the table moved and re-asserts.
        """
        if (unit := self._texture_units.get(name)) is None:
            raise TlangBindingError(f"'{name}' is not a declared texture (sampler) uniform", SourceLocation(module=self._name))
        texture.use(unit)
        bump_texture_table_generation()

    def bind_images(self, **images: Texture | tuple[Texture, bool, bool, int, int]) -> None:
        loc = self.bind_image
        for k, v in images.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_image(
        self, name: str, image: Texture, read: bool = True, write: bool = True,
        level: int = 0, format: int = 0,
    ) -> None:
        """Bind `image` to `name`'s image unit IMMEDIATELY, via `Texture.bind_to_image(...)` --
        the image counterpart of `bind_texture` above, same asymmetry and same rationale."""
        if (unit := self._image_units.get(name)) is None:
            raise TlangBindingError(f"'{name}' is not a declared image uniform", SourceLocation(module=self._name))
        image.bind_to_image(unit, read=read, write=write, level=level, format=format)
        bump_image_table_generation()
