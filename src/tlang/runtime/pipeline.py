# -------------------------------------------------------------
# @file          pipeline.py
# @author        Priyangkar Ghosh
# @created       2026-09-07
# @description   Graphics program wrapper mirroring Kernel's ergonomics: name-keyed SSBO/uniform
#                binding for [program(...)]s instead of raw moderngl calls.
# @license       MIT
# -------------------------------------------------------------

import ctypes
import difflib
import logging
from collections.abc import Mapping
from typing import Any
from moderngl import Buffer, Context, Program, StorageBlock, Texture, UniformBlock, Uniform, VertexArray
from OpenGL.GL import (
    glBindBufferRange, GL_ATOMIC_COUNTER_BUFFER,
    glGetIntegeri_v, GL_ATOMIC_COUNTER_BUFFER_BINDING,
)

from tlang.errors import SourceLocation, TlangBindingError
from tlang.runtime.kernel import (
    bump_counter_table_generation, bump_image_table_generation, bump_ssbo_table_generation,
    bump_texture_table_generation, counter_table_generation, image_table_generation,
    ssbo_table_generation, texture_table_generation,
)

logger = logging.getLogger(__name__)

# sentinel distinguishing "never cached" from a cached value of `None`
_NOT_CACHED = object()

class Pipeline:
    """Wraps a linked `moderngl.Program` for one `[program(...)]` with name-keyed SSBO/uniform
    binding, so callers never hardcode binding numbers. Get one via `Shader.get_pipeline(name)`."""
    __slots__ = (
        '_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache', '_buffer_source',
        '_texture_units', '_image_units', '_atomic_counters',
        '_ssbo_bindings', '_bound_generation',
        '_texture_bindings', '_bound_texture_generation',
        '_image_bindings', '_bound_image_generation',
        '_counter_bindings', '_bound_counter_generation',
    )

    def __init__(
        self, ctx: Context, name: str, program: Program, bindings: Mapping[str, int] | None = None,
        texture_units: Mapping[str, int] | None = None, image_units: Mapping[str, int] | None = None,
        atomic_counters: Mapping[str, tuple[int, int]] | None = None,
    ):
        self._ctx = ctx
        self._name = name
        self._mglo = program
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # HANDLE -> binding: the canon BindingRegistry.allocate_artifact decided for the
        # program as a whole, re-keyed from the emitted GLSL name onto tlang's own handle by
        # `Shader._to_handles` before this constructor ever runs. `bind_ssbo` resolves against
        # this first -- see `_resolve_ssbo_binding`.
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}
        # buffer source consulted by `bind()` for any required name not passed explicitly --
        # see the `source` property. `None` means bind() can only resolve explicit kwargs.
        self._buffer_source: Mapping[str, Buffer] | None = None
        # name -> assigned texture/image unit, the sampler/image counterpart of `_bindings` --
        # see `Kernel.texture_units`/`Kernel.image_units`, which this mirrors.
        self._texture_units: dict[str, int] = dict(texture_units) if texture_units is not None else {}
        self._image_units: dict[str, int] = dict(image_units) if image_units is not None else {}
        # name -> (binding, offset-within-binding), pruned to counters this program's linked
        # GL_ATOMIC_COUNTER_BUFFER interface actually reports active -- see
        # `Kernel.atomic_counters`, which this mirrors.
        self._atomic_counters: dict[str, tuple[int, int]] = dict(atomic_counters) if atomic_counters is not None else {}

        # name -> (buffer, offset, size) most recently passed to `bind_ssbo`, kept so `render()`
        # has a full recorded set to re-assert -- the `Kernel._ssbo_bindings` counterpart, except
        # `bind_ssbo` here ALSO writes it to GL immediately (see that method's docstring).
        self._ssbo_bindings: dict[str, tuple[Buffer, int, int]] = {}
        # `ssbo_table_generation()` as of this pipeline's last full re-assert; -1 means "never
        # asserted", same convention as `Kernel._bound_generation`.
        self._bound_generation: int = -1

        # texture/image/counter counterparts of the pair above, one per independent GL table --
        # see `Kernel`'s equivalents for the discipline each shares.
        self._texture_bindings: dict[str, Texture] = {}
        self._bound_texture_generation: int = -1
        self._image_bindings: dict[str, tuple[Texture, bool, bool, int, int]] = {}
        self._bound_image_generation: int = -1
        self._counter_bindings: dict[str, tuple[Buffer, int]] = {}
        self._bound_counter_generation: int = -1

    @property
    def ctx(self): return self._ctx

    @property
    def name(self): return self._name

    @property
    def mglo(self): return self._mglo

    @property
    def glo(self) -> int: return self._mglo.glo

    @property
    def bindings(self) -> Mapping[str, int]: return self._bindings # handle -> assigned SSBO binding

    @property
    def texture_units(self) -> Mapping[str, int]: return self._texture_units # name -> assigned texture unit

    @property
    def image_units(self) -> Mapping[str, int]: return self._image_units # name -> assigned image unit

    @property
    def atomic_counters(self) -> Mapping[str, tuple[int, int]]:
        """Read-only: atomic counter name -> (binding, offset-within-binding). See
        `Kernel.atomic_counters`, which this mirrors."""
        return self._atomic_counters

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

    def _resolve_ssbo_binding(self, buffer_name: str) -> int:
        """Resolve `buffer_name`'s GL binding index, preferring the static canon
        (`self._bindings`, HANDLE-keyed -- see `Kernel._resolve_ssbo_binding`, which this
        mirrors) over driver reflection. Reflection alone is not enough here: for a [buffer]
        single-declarator shorthand block, the handle is never the name the driver reflects
        under (that's the synthesised emitted block name -- see `InterfaceDecl.emitted_name`),
        so a name absent from the canon falls back to reflection only for a block tlang's own
        canon doesn't know about at all. Raises `TlangBindingError` if `buffer_name` isn't a
        real storage block by either source -- that's a typo.
        """
        if (binding := self._bindings.get(buffer_name)) is not None:
            return binding
        if (binding := self._binding_cache.get(buffer_name)) is not None:
            return binding
        block = self._mglo.get(buffer_name, None)
        if not isinstance(block, StorageBlock):
            raise TlangBindingError(f"'{buffer_name}' is not a valid buffer block (Missing binding)", SourceLocation(module=self._name))
        self._binding_cache[buffer_name] = binding = block.binding
        return binding

    def bind_ssbo(
        self, buffer_name: str, buffer: Buffer, offset: int = 0, size: int = -1
    ) -> None:
        """Bind `buffer` to `buffer_name` (a HANDLE) IMMEDIATELY, unlike `Kernel.bind_ssbo` --
        and record it, so `render()` (see its docstring) can re-assert it later.

        Binding immediately, rather than only recording the way `Kernel` does, is what keeps a
        caller who dispatches `vao.render()` directly working exactly as before: drawing happens
        in moderngl's `VertexArray.render()`, entirely outside tlang, so that caller has no
        dispatch-like hook to defer to. `render()` is such a hook, though -- routing through it
        instead re-asserts this pipeline's full recorded set right before `vao.render()` runs, so
        a compute dispatch issued between this call and `render()` can no longer leave this
        pipeline reading someone else's buffers. Only a caller who bypasses `render()` and calls
        `vao.render()` directly is still exposed to that race.

        This also still bumps the same process-global generation counter `Kernel` uses (see
        `tlang.runtime.kernel.bump_ssbo_table_generation`), so any `Kernel` that dispatches after
        this call correctly notices the table has moved and re-asserts its own bindings. The
        recorded generation is fast-forwarded to the post-bump value rather than forced stale --
        unlike `Kernel.bind_ssbo`, which always forces a re-assert since it never writes GL
        itself -- because this entry was just physically written: `render()` must not immediately
        rewrite it again. Fast-forwarding only when this pipeline's whole recorded set was
        already current keeps that safe: if some other binder moved the table since this
        pipeline's last full assert, the miss is preserved so `render()` still catches it.
        """
        binding = self._resolve_ssbo_binding(buffer_name)
        # offset/size are keyword-only on moderngl <= 5.8.x; positional args raise TypeError there.
        buffer.bind_to_storage_buffer(binding, offset=offset, size=size)
        self._ssbo_bindings[buffer_name] = (buffer, offset, size)
        was_current = self._bound_generation == ssbo_table_generation()
        generation = bump_ssbo_table_generation()
        self._bound_generation = generation if was_current else -1

    def bind(self, **explicit: Buffer | tuple[Buffer, int, int]) -> None:
        """Bind exactly this program's REQUIRED set (`self.bindings`) by name, drawn from
        `explicit` first and then from `self.buffer_source` -- mirrors `Kernel.bind`, going
        through `bind_ssbo` (see its docstring) for each name, so every bind is both immediate
        and recorded for `render()` to re-assert."""
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
        """Bind `texture` to `name`'s texture unit IMMEDIATELY, unlike `Kernel.bind_texture` --
        and record it, so `render()` can re-assert it later. Same reasoning as `bind_ssbo` above:
        immediate binding keeps a direct `vao.render()` caller working unchanged, `render()`
        closes the cross-wiring hole for anyone who calls it instead, and the recorded generation
        is fast-forwarded only when this pipeline's whole texture set was already current.

        Also bumps the shared texture-unit generation counter (see
        `tlang.runtime.kernel.bump_texture_table_generation`) so any `Kernel` that dispatches
        after this call notices the table moved and re-asserts.
        """
        if (unit := self._texture_units.get(name)) is None:
            raise TlangBindingError(f"'{name}' is not a declared texture (sampler) uniform", SourceLocation(module=self._name))
        texture.use(unit)
        self._texture_bindings[name] = texture
        was_current = self._bound_texture_generation == texture_table_generation()
        generation = bump_texture_table_generation()
        self._bound_texture_generation = generation if was_current else -1

    def bind_images(self, **images: Texture | tuple[Texture, bool, bool, int, int]) -> None:
        loc = self.bind_image
        for k, v in images.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_image(
        self, name: str, image: Texture, read: bool = True, write: bool = True,
        level: int = 0, format: int = 0,
    ) -> None:
        """Bind `image` to `name`'s image unit IMMEDIATELY, via `Texture.bind_to_image(...)` --
        the image counterpart of `bind_texture` above: same immediate-plus-recorded discipline,
        same rationale, same `render()`-closes-the-hole reasoning."""
        if (unit := self._image_units.get(name)) is None:
            raise TlangBindingError(f"'{name}' is not a declared image uniform", SourceLocation(module=self._name))
        image.bind_to_image(unit, read=read, write=write, level=level, format=format)
        self._image_bindings[name] = (image, read, write, level, format)
        was_current = self._bound_image_generation == image_table_generation()
        generation = bump_image_table_generation()
        self._bound_image_generation = generation if was_current else -1

    def bind_counters(self, **counters: Buffer | tuple[Buffer, int]) -> None:
        loc = self.bind_counter
        for k, v in counters.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_counter(self, name: str, buffer: Buffer, offset: int = 0) -> None:
        """Bind `buffer` to atomic counter `name`'s binding IMMEDIATELY, unlike
        `Kernel.bind_counter` -- and record it, so `render()` can re-assert it later. Same
        immediate-plus-recorded discipline as `bind_ssbo`/`bind_texture` above, for the same
        reason.

        `offset` is the byte offset in `buffer` where GL's bound range begins -- see
        `Kernel.bind_counter`'s docstring for the full composition. The range's size is derived
        from every counter `self.atomic_counters` says shares `name`'s binding (not just `name`
        itself), computed fresh from the static canon each call -- so binding either of two
        counters sharing one binding, in either order, produces the same correctly-sized range,
        with no cross-call bookkeeping needed the way `Kernel`'s deferred assert requires.
        """
        if (pos := self._atomic_counters.get(name)) is None:
            raise TlangBindingError(f"'{name}' is not a declared atomic counter uniform", SourceLocation(module=self._name))
        binding, _counter_offset = pos
        size = max(o + 4 for b, o in self._atomic_counters.values() if b == binding)
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, size)
        self._counter_bindings[name] = (buffer, offset)
        was_current = self._bound_counter_generation == counter_table_generation()
        generation = bump_counter_table_generation()
        self._bound_counter_generation = generation if was_current else -1

    def render(self, vao: VertexArray, mode: int | None = None, vertices: int = -1, first: int = 0, instances: int = -1) -> None:
        """Re-assert this pipeline's full recorded SSBO/texture/image/counter binding set, then
        delegate to `vao.render(mode, vertices, first, instances)` -- mirrors
        `moderngl.VertexArray.render`'s signature exactly.

        This is the hook `bind_ssbo`'s docstring says a `Pipeline` never had: call `render()`
        instead of `vao.render()` directly and a compute dispatch issued between binding this
        pipeline and now can no longer leave it drawing with someone else's buffers -- the same
        guarantee `Kernel.dispatch` gives a kernel, applied here right before the draw call
        instead of right before a dispatch call. A caller who still calls `vao.render()` directly
        bypasses this and keeps the old exposure.
        """
        self._assert_bindings()
        vao.render(mode, vertices, first, instances)

    def render_indirect(self, vao: VertexArray, buffer: Buffer, mode: int | None = None, count: int = -1, first: int = 0) -> None:
        """`render()`'s counterpart for `moderngl.VertexArray.render_indirect` -- same re-assert,
        same rationale, mirrors that method's signature exactly."""
        self._assert_bindings()
        vao.render_indirect(buffer, mode, count, first)

    def transform(
        self, vao: VertexArray, buffer: Buffer, mode: int | None = None, vertices: int = -1,
        first: int = 0, instances: int = -1, buffer_offset: int = 0,
    ) -> None:
        """`render()`'s counterpart for `moderngl.VertexArray.transform` -- same re-assert, same
        rationale, mirrors that method's signature exactly."""
        self._assert_bindings()
        vao.transform(buffer, mode, vertices, first, instances, buffer_offset)

    def _assert_bindings(self) -> None:
        """Everything `render`/`render_indirect`/`transform` need before delegating to the real
        `moderngl.VertexArray` call -- the render-time counterpart of the four asserts
        `Kernel.dispatch` runs at the top of every dispatch entry point."""
        self._assert_ssbo_bindings()
        self._assert_texture_bindings()
        self._assert_image_bindings()
        self._assert_counter_bindings()

    def _assert_ssbo_bindings(self) -> None:
        """Pipeline counterpart of `Kernel._assert_ssbo_bindings`: fail on a required block
        never bound, catch a freed buffer, then re-assert the full recorded set into GL's global
        binding table unless the shared generation counter shows nothing has touched it since
        this pipeline's last full assert."""
        missing = [name for name in self._bindings if name not in self._ssbo_bindings]
        if missing:
            raise TlangBindingError(
                f"Pipeline '{self._name}' rendered with required buffer(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first)",
                SourceLocation(module=self._name),
            )

        for name, (buffer, _offset, _size) in self._ssbo_bindings.items():
            if not getattr(buffer, 'alive', True):
                raise TlangBindingError(
                    f"Pipeline '{self._name}': buffer bound to '{name}' has been freed "
                    f"(recycled temp buffer) -- re-bind before rendering",
                    SourceLocation(module=self._name),
                )

        if self._bound_generation == ssbo_table_generation():
            return  # nothing else has touched the global table since our last full assert

        for name, (buffer, offset, size) in self._ssbo_bindings.items():
            binding = self._resolve_ssbo_binding(name)
            buffer.bind_to_storage_buffer(binding, offset=offset, size=size)

        self._bound_generation = bump_ssbo_table_generation() if self._ssbo_bindings else ssbo_table_generation()

    def _assert_texture_bindings(self) -> None:
        """Texture-unit counterpart of `_assert_ssbo_bindings` -- same jobs, same
        generation-counter fast path, against the separate texture-unit table/counter."""
        missing = [name for name in self._texture_units if name not in self._texture_bindings]
        if missing:
            raise TlangBindingError(
                f"Pipeline '{self._name}' rendered with required texture(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first)",
                SourceLocation(module=self._name),
            )

        for name, texture in self._texture_bindings.items():
            if not getattr(texture, 'alive', True):
                raise TlangBindingError(
                    f"Pipeline '{self._name}': texture bound to '{name}' has been freed -- "
                    f"re-bind before rendering",
                    SourceLocation(module=self._name),
                )

        if self._bound_texture_generation == texture_table_generation():
            return  # nothing else has touched the global texture-unit table since our last assert

        for name, texture in self._texture_bindings.items():
            texture.use(self._texture_units[name])

        self._bound_texture_generation = (
            bump_texture_table_generation() if self._texture_bindings else texture_table_generation()
        )

    def _assert_image_bindings(self) -> None:
        """Image-unit counterpart of `_assert_ssbo_bindings` -- same jobs, same
        generation-counter fast path, against the separate image-unit table/counter."""
        missing = [name for name in self._image_units if name not in self._image_bindings]
        if missing:
            raise TlangBindingError(
                f"Pipeline '{self._name}' rendered with required image(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first)",
                SourceLocation(module=self._name),
            )

        for name, (image, *_rest) in self._image_bindings.items():
            if not getattr(image, 'alive', True):
                raise TlangBindingError(
                    f"Pipeline '{self._name}': image bound to '{name}' has been freed -- "
                    f"re-bind before rendering",
                    SourceLocation(module=self._name),
                )

        if self._bound_image_generation == image_table_generation():
            return  # nothing else has touched the global image-unit table since our last assert

        for name, (image, read, write, level, format) in self._image_bindings.items():
            image.bind_to_image(self._image_units[name], read=read, write=write, level=level, format=format)

        self._bound_image_generation = (
            bump_image_table_generation() if self._image_bindings else image_table_generation()
        )

    def _assert_counter_bindings(self) -> None:
        """Atomic-counter counterpart of `_assert_ssbo_bindings`, with the same deliberate
        divergence `Kernel._assert_counter_bindings` documents: a counter never bound through
        this pipeline is checked against GL's own binding state, not this pipeline's records,
        since binding one once outside any `Pipeline`/`Kernel` and never rebinding is legitimate
        usage. See that method's docstring for the full reasoning -- this mirrors it exactly."""
        for name, (buffer, _offset) in self._counter_bindings.items():
            if not getattr(buffer, 'alive', True):
                raise TlangBindingError(
                    f"Pipeline '{self._name}': buffer bound to atomic counter '{name}' has been "
                    f"freed (recycled temp buffer) -- re-bind before rendering",
                    SourceLocation(module=self._name),
                )

        unbound = {n for n in self._atomic_counters if n not in self._counter_bindings}
        if unbound:
            slot = (ctypes.c_int * 1)()
            for name in sorted(unbound):
                binding, _off = self._atomic_counters[name]
                glGetIntegeri_v(GL_ATOMIC_COUNTER_BUFFER_BINDING, binding, slot)
                if slot[0]: continue
                raise TlangBindingError(
                    f"Pipeline '{self._name}' rendered with atomic counter '{name}' unbound: "
                    f"nothing is bound at atomic-counter binding {binding}, so it would read no "
                    f"buffer at all. Bind it with bind_counter('{name}', ...)",
                    SourceLocation(module=self._name),
                )

        if self._bound_counter_generation == counter_table_generation():
            return  # nothing else has touched the global counter-buffer table since our last assert

        by_binding: dict[int, tuple[Buffer, int]] = {}
        for name, (buffer, offset) in self._counter_bindings.items():
            binding, _counter_offset = self._atomic_counters[name]
            if binding in by_binding:
                existing_buffer, existing_offset = by_binding[binding]
                if existing_buffer is not buffer or existing_offset != offset:
                    raise TlangBindingError(
                        f"Pipeline '{self._name}': atomic counters sharing binding {binding} "
                        f"were bound to different buffers/offsets -- bind every counter that "
                        f"shares one binding to the same buffer and the same range offset",
                        SourceLocation(module=self._name),
                    )
                continue
            by_binding[binding] = (buffer, offset)

        for binding, (buffer, offset) in by_binding.items():
            size = max(o + 4 for b, o in self._atomic_counters.values() if b == binding)
            glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, size)

        self._bound_counter_generation = (
            bump_counter_table_generation() if self._counter_bindings else counter_table_generation()
        )
