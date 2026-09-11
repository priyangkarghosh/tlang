# -------------------------------------------------------------
# @file          kernel.py
# @author        Priyangkar Ghosh
# @created       2025-06-08
# @description   Kernel/Compute Shader wrapper with helper functions
# @license       MIT
# -------------------------------------------------------------

import ctypes
import difflib
import logging
import time
from collections.abc import Mapping
from typing import Any
from moderngl import (
    ATOMIC_COUNTER_BARRIER_BIT, SHADER_STORAGE_BARRIER_BIT, Buffer, ComputeShader,
    Context, StorageBlock, Texture, Uniform, UniformBlock
)
from OpenGL.GL import (
    glBindBufferRange, GL_ATOMIC_COUNTER_BUFFER,
    glDispatchComputeIndirect, GL_DISPATCH_INDIRECT_BUFFER,
    glBindBuffer, glUseProgram,
    glGetProgramiv, GL_COMPUTE_WORK_GROUP_SIZE,
    glGetIntegeri_v, GL_ATOMIC_COUNTER_BUFFER_BINDING,
)

from tlang.errors import SourceLocation, TlangBindingError

logger = logging.getLogger(__name__)

# sentinel distinguishing "never cached" from a cached value of `None`
_NOT_CACHED = object()

# Process-global generation counter for GL's shader-storage binding table. GL has exactly ONE
# such table (it's a driver-side array indexed by binding number, not something owned by any one
# program), so every `Kernel` and `Pipeline` in the process shares this single counter. It is
# bumped every time anything actually WRITES to that table -- a real `bind_to_storage_buffer`
# call -- never on a mere `bind_ssbo` recording (see `Kernel.bind_ssbo`). A `Kernel` remembers the
# generation at which it last asserted its own full recorded set (`_bound_generation`); if the
# counter hasn't moved since, nothing else has touched the table and the kernel can skip
# re-binding entirely. This is what makes repeat dispatches of the same kernel free.
_ssbo_table_generation = 0


def bump_ssbo_table_generation() -> int:
    """Record one write to GL's global SSBO binding table and return the new generation.

    Called from `Kernel`'s own dispatch-time (re)assert and from `Pipeline.bind_ssbo`, which
    binds immediately rather than deferring (see that method's docstring for why). Any other
    code that ever writes `glBindBufferRange`/`bind_to_storage_buffer` directly against this
    table must call this too, or kernels can go stale without noticing.
    """
    global _ssbo_table_generation
    _ssbo_table_generation += 1
    return _ssbo_table_generation


# Same generation-counter discipline as `_ssbo_table_generation` above, one counter per GL
# binding table: texture units (`Texture.use`) and image units (`Texture.bind_to_image`) are
# separate process-global tables from each other AND from the SSBO one, so each gets its own
# counter -- sharing one across unrelated tables would make a write to one silently mask a stale
# read from another.
_texture_table_generation = 0
_image_table_generation = 0


def bump_texture_table_generation() -> int:
    """Record one write to GL's global texture-unit binding table (`Texture.use`) and return the
    new generation. See `bump_ssbo_table_generation` -- same discipline, separate table."""
    global _texture_table_generation
    _texture_table_generation += 1
    return _texture_table_generation


def bump_image_table_generation() -> int:
    """Record one write to GL's global image-unit binding table (`Texture.bind_to_image`) and
    return the new generation. See `bump_ssbo_table_generation` -- same discipline, separate
    table."""
    global _image_table_generation
    _image_table_generation += 1
    return _image_table_generation


# Same discipline again, for GL's atomic counter buffer binding table (`glBindBufferRange` on
# `GL_ATOMIC_COUNTER_BUFFER`) -- a fourth, independent process-global table.
_counter_table_generation = 0


def bump_counter_table_generation() -> int:
    """Record one write to GL's global atomic-counter-buffer binding table and return the new
    generation. See `bump_ssbo_table_generation` -- same discipline, separate table. Also called
    from `Pipeline.bind_counter`, which binds immediately rather than deferring, exactly like
    `Pipeline.bind_ssbo`."""
    global _counter_table_generation
    _counter_table_generation += 1
    return _counter_table_generation


class Kernel:
    __slots__ = (
        '_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache',
        '_ssbo_bindings', '_bound_generation', '_buffer_source',
        '_texture_units', '_image_units', '_texture_bindings', '_image_bindings',
        '_bound_texture_generation', '_bound_image_generation',
        '_atomic_counters', '_counter_bindings', '_bound_counter_generation',
        '_local_size',
    )

    def __init__(
        self, ctx: Context, name: str, shader: ComputeShader, bindings: Mapping[str, int] | None = None,
        texture_units: Mapping[str, int] | None = None, image_units: Mapping[str, int] | None = None,
        atomic_counters: Mapping[str, tuple[int, int]] | None = None,
    ):
        self._ctx = ctx
        self._name = name
        self._mglo = shader
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # buffer source consulted by `bind()` for any required name not passed explicitly --
        # see the `source` property. `None` means bind() can only resolve explicit kwargs.
        self._buffer_source: Mapping[str, Buffer] | None = None
        # HANDLE -> binding map `BindingRegistry.allocate_artifact` decided for this kernel (the
        # artifact's declared/required SSBO blocks, patched into the GLSL text as `binding = N`),
        # re-keyed from the emitted GLSL block name onto tlang's own handle by
        # `Shader._to_handles` before this constructor ever runs -- see `InterfaceDecl.name` vs
        # `InterfaceDecl.emitted_name`. This is the canon: static, driver-independent, and known
        # before the program ever links. `_resolve_ssbo_binding` below prefers it over
        # reflection, and `dispatch` treats its keys as the kernel's REQUIRED set (see
        # `_assert_ssbo_bindings`).
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}

        # name -> (buffer, offset, size) recorded by `bind_ssbo`/`bind_ssbos`/`bind`, NOT yet
        # written to GL. Asserted into the real binding table at the top of every dispatch entry
        # point -- see `_assert_ssbo_bindings`.
        self._ssbo_bindings: dict[str, tuple[Buffer, int, int]] = {}
        # `_ssbo_table_generation` as of this kernel's last full assert; -1 means "never
        # asserted", which always forces a rebind on the first dispatch.
        self._bound_generation: int = -1

        # name -> assigned texture/image unit, the sampler/image counterpart of `_bindings` --
        # every sampler/image uniform this artifact declares, decided by
        # `BindingRegistry.allocate_opaque_units` before the program ever links. Two separate
        # pools, exactly like SSBO vs UBO above.
        self._texture_units: dict[str, int] = dict(texture_units) if texture_units is not None else {}
        self._image_units: dict[str, int] = dict(image_units) if image_units is not None else {}

        # name -> Texture / (Texture, read, write, level, format) recorded by `bind_texture`/
        # `bind_image`, NOT yet written to GL -- same deferred-until-dispatch discipline as
        # `_ssbo_bindings`, asserted by `_assert_texture_bindings`/`_assert_image_bindings`.
        self._texture_bindings: dict[str, Texture] = {}
        self._image_bindings: dict[str, tuple[Texture, bool, bool, int, int]] = {}
        # generation counters as of this kernel's last full assert of each table; -1 means
        # "never asserted", same convention as `_bound_generation`.
        self._bound_texture_generation: int = -1
        self._bound_image_generation: int = -1

        # name -> (binding, offset-within-binding) decided by
        # `BindingRegistry.allocate_atomic_counters` and patched into the GLSL as
        # `layout(binding = N, offset = M)`, PRUNED to counters the linked program's own
        # GL_ATOMIC_COUNTER_BUFFER program interface actually reports active (see
        # `BindingRegistry.active_atomic_counter_bindings` -- reflection cannot see atomic
        # counters at all, so this raw post-link query is the only source of truth for which
        # declared counters this artifact genuinely uses). This is the kernel's REQUIRED set,
        # exactly like `_bindings`/`_texture_units`/`_image_units` above.
        self._atomic_counters: dict[str, tuple[int, int]] = dict(atomic_counters) if atomic_counters is not None else {}
        # name -> (buffer, range_offset) recorded by `bind_counter`/`bind_counters`, NOT yet
        # written to GL -- same deferred-until-dispatch discipline as `_ssbo_bindings`.
        # `range_offset` is the byte offset in `buffer` where GLSL's own `offset = 0` would
        # land -- see `bind_counter`'s docstring for how this composes with the counter's own
        # canon offset.
        self._counter_bindings: dict[str, tuple[Buffer, int]] = {}
        self._bound_counter_generation: int = -1

        # lazily-queried, then cached forever -- a linked program's work-group size cannot
        # change. See `local_size`.
        self._local_size: tuple[int, int, int] | None = None

    @property
    def ctx(self): return self._ctx # mgl context

    @property
    def name(self): return self._name # shader name

    @property
    def mglo(self): return self._mglo # moderngl object

    @property
    def glo(self) -> int: return self._mglo.glo # gl object

    @property
    def bindings(self) -> Mapping[str, int]: return self._bindings # handle -> assigned SSBO binding

    @property
    def texture_units(self) -> Mapping[str, int]: return self._texture_units # name -> assigned texture unit

    @property
    def image_units(self) -> Mapping[str, int]: return self._image_units # name -> assigned image unit

    @property
    def atomic_counters(self) -> Mapping[str, tuple[int, int]]:
        """Read-only: atomic counter name -> (binding, offset-within-binding), pruned to
        counters this artifact's linked program actually uses (see the constructor's
        docstring). tlang's own textual canon is the only source of truth here -- reflection
        cannot see atomic counters at all."""
        return self._atomic_counters

    @property
    def buffer_source(self) -> Mapping[str, Buffer] | None:
        """The `Mapping[str, Buffer]` -- a `BufferPool`, a plain dict, anything -- that `bind()`
        draws unnamed required blocks from. Read fresh on every `bind()` call, never cached, so a
        live mapping (temporaries appearing and disappearing between frames) works naturally."""
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

    @property
    def default_barrier_bits(self) -> int:
        """`SHADER_STORAGE_BARRIER_BIT`, plus `ATOMIC_COUNTER_BARRIER_BIT` when this artifact
        declares a genuinely-used (per `atomic_counters`) counter. Consumers used to hand-write
        that OR themselves, with a comment explaining why -- tlang knows its own artifact's
        counter set, so it derives this instead. This is what `dispatch`/`dispatch_indirect` use
        whenever `barrier_bits` is not passed explicitly; an explicit value always wins
        unchanged. Exposed read-only so a caller can inspect what a bare `dispatch()` will use
        without needing to trigger one."""
        bits = SHADER_STORAGE_BARRIER_BIT
        if self._atomic_counters: bits |= ATOMIC_COUNTER_BARRIER_BIT
        return bits

    @property
    def local_size(self) -> tuple[int, int, int]:
        """The linked kernel's actual work-group size (`local_size_x/y/z`), queried once from the
        driver and cached -- it cannot change for an already-linked program.

        This queries GL rather than reading back `[numthreads(...)]`'s arguments, deliberately:
        tlang does not coerce those arguments to `int` (see the note in
        `AttributeHandlers.numthreads`) since `numthreads(BLOCK_SIZE, 1, 1)` is a legitimate GLSL
        preprocessor macro name resolved by the driver, not a Python integer. The linked program
        is therefore the only authoritative source for the real size --
        `glGetProgramiv(glo, GL_COMPUTE_WORK_GROUP_SIZE, ...)` resolves any such macro the same
        way the driver did at link time.
        """
        if self._local_size is None:
            sizes = (ctypes.c_int * 3)()
            glGetProgramiv(self.glo, GL_COMPUTE_WORK_GROUP_SIZE, sizes)
            self._local_size = (sizes[0], sizes[1], sizes[2])
        return self._local_size

    def dispatch(
        self,
        group_x: int = 1,
        group_y: int = 1,
        group_z: int = 1,
        barrier: bool = True,
        barrier_bits: int | None = None,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self._assert_ssbo_bindings(allow_unbound)
        self._assert_texture_bindings(allow_unbound)
        self._assert_image_bindings(allow_unbound)
        self._assert_counter_bindings(allow_unbound)
        self._mglo.run(group_x, group_y, group_z)
        if barrier: self._ctx.memory_barrier(barrier_bits if barrier_bits is not None else self.default_barrier_bits)

    def dispatch_for(
        self,
        x: int,
        y: int = 1,
        z: int = 1,
        elems_per_thread: int = 1,
        barrier: bool = True,
        barrier_bits: int | None = None,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        """Dispatch enough work-groups to cover `x` (and, for a 2-D/3-D kernel, `y`/`z`)
        invocations, deriving the group counts from `local_size` -- the linked program's actual
        work-group size -- instead of making the caller redeclare `[numthreads(...)]`'s value in
        Python and recompute the group count by hand:

            kernel.dispatch_for(num_particles)              # 1-D, the common case
            kernel.dispatch_for(width, height)               # 2-D
            kernel.dispatch_for(width, height, depth)        # 3-D
            kernel.dispatch_for(n, elems_per_thread=4)       # scan-style, 4 elements/thread

        Ceiling division throughout: a kernel is never under-dispatched, even when `x`/`y`/`z`
        isn't an exact multiple of `local_size`. `elems_per_thread` divides `x` before the
        ceiling division -- for a kernel whose each invocation processes more than one element
        along its (typically only) axis. It applies to `x` alone, matching the 1-D idiom it's
        modeled on; a kernel needing per-thread multiplicity on more than one axis should call
        `dispatch` directly with hand-computed group counts.

        Forwards `barrier`/`barrier_bits`/`allow_unbound` to `dispatch` unchanged, so this goes
        through the exact same binding re-assert path (`_assert_ssbo_bindings` and its
        texture/image/counter counterparts) -- it is a drop-in replacement for
        `dispatch((n + local_size[0] - 1) // local_size[0])`, not a separate binding path.
        """
        lsx, lsy, lsz = self.local_size
        gx = -(-x // (lsx * elems_per_thread))
        gy = -(-y // lsy)
        gz = -(-z // lsz)
        self.dispatch(gx, gy, gz, barrier=barrier, barrier_bits=barrier_bits, allow_unbound=allow_unbound)

    def dispatch_indirect(
        self,
        buffer: Buffer,
        offset: int = 0,
        barrier: bool = True,
        barrier_bits: int | None = None,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self._assert_ssbo_bindings(allow_unbound)
        self._assert_texture_bindings(allow_unbound)
        self._assert_image_bindings(allow_unbound)
        self._assert_counter_bindings(allow_unbound)
        glUseProgram(self.glo)
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer.glo)
        glDispatchComputeIndirect(offset)
        if barrier: self._ctx.memory_barrier(barrier_bits if barrier_bits is not None else self.default_barrier_bits)

    def dispatch_timed(
        self,
        group_x: int = 1,
        group_y: int = 1,
        group_z: int = 1,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> float:
        t0 = time.perf_counter()
        self.dispatch(group_x, group_y, group_z, allow_unbound=allow_unbound)
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
        """Record that `buffer_name` should be bound to `buffer` (at `offset`/`size`) on this
        kernel. This does NOT touch GL's binding table -- the actual `bind_to_storage_buffer`
        call is deferred until the next `dispatch`/`dispatch_indirect`/`dispatch_timed`, which
        re-asserts this kernel's ENTIRE recorded set immediately before running.

        This is a real semantic change from the old immediate-bind behaviour: GL's SSBO binding
        table is process-global, so a bind made "now" could previously be silently overwritten by
        an unrelated kernel's dispatch before this kernel ever ran, with no error either way
        (see the module using this for the T12 cross-wiring bug this exists to close). Deferring
        the write to dispatch time means what's actually in the table always matches what the
        about-to-run kernel asked for.

        `buffer_name` is still validated immediately: a name this artifact does not declare at
        all is a typo, and raises `TlangBindingError` right here, same as before.
        """
        self._resolve_ssbo_binding(buffer_name)  # validate now; raises TlangBindingError on typo
        self._ssbo_bindings[buffer_name] = (buffer, offset, size)
        # Recorded set changed -- force a re-assert on this kernel's next dispatch even if no
        # other kernel/pipeline has touched the global table in the meantime.
        self._bound_generation = -1

    def bind(self, **explicit: Buffer | tuple[Buffer, int, int]) -> None:
        """Bind exactly this kernel's REQUIRED set (`self.bindings`, the artifact's declared SSBO
        blocks) by name, drawn from `explicit` first and then from `self.buffer_source` (see the
        `source` property).

        Called with no arguments, every required block is resolved from `self.buffer_source` alone:

            kernel.buffer_source = pool          # once, e.g. when the module is set up
            kernel.bind()                 # every dispatch
            kernel.dispatch(groups)

        Called with kwargs (the original form), those override `self.buffer_source` for this call only;
        names that this artifact does not declare are silently ignored -- this is what lets one
        caller pass a single superset dict of buffers across every kernel in a pipeline:

            kernel.bind(**self._buffers)
            kernel.dispatch(groups)

        A required name resolved from neither raises `TlangBindingError`. The message names
        whether the miss is "not in the provided buffers" (an incomplete `explicit` and no
        source) or "not found in the bound buffer source" (fell through to `self.buffer_source` and it
        doesn't have it either) -- the two mean different fixes. Accepts the same value shapes as
        `bind_ssbos`: a bare `Buffer`, or `(Buffer, offset, size)`.
        """
        for buffer_name in self._bindings:
            if buffer_name in explicit:
                value = explicit[buffer_name]
            elif self._buffer_source is not None:
                if buffer_name not in self._buffer_source:
                    raise TlangBindingError(
                        f"Kernel '{self._name}' requires buffer '{buffer_name}' but it was not "
                        f"found in the bound buffer source.{self._suggest_from_source(buffer_name)}",
                        SourceLocation(module=self._name),
                    )
                value = self._buffer_source[buffer_name]
            else:
                raise TlangBindingError(
                    f"Kernel '{self._name}' requires buffer '{buffer_name}' but it was not in the "
                    f"provided buffers",
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
        """Record that `name` should be bound to `texture`'s texture unit on this kernel. Like
        `bind_ssbo`, this does NOT touch GL's (also process-global) texture-unit table
        immediately -- the actual `Texture.use(unit)` call is deferred to the next
        `dispatch`/`dispatch_indirect`/`dispatch_timed`, which re-asserts this kernel's entire
        recorded texture set right before running. See `bind_ssbo`'s docstring for why deferring
        matters: an immediate bind here could otherwise be silently overwritten by an unrelated
        kernel's dispatch before this one ever runs.

        `name` is validated immediately against `self.texture_units` -- a name this artifact
        does not declare at all is a typo, and raises `TlangBindingError` right here.
        """
        self._resolve_texture_unit(name)  # validate now; raises TlangBindingError on typo
        self._texture_bindings[name] = texture
        self._bound_texture_generation = -1  # force a re-assert on this kernel's next dispatch

    def bind_images(self, **images: Texture | tuple[Texture, bool, bool, int, int]) -> None:
        loc = self.bind_image
        for k, v in images.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_image(
        self, name: str, image: Texture, read: bool = True, write: bool = True,
        level: int = 0, format: int = 0,
    ) -> None:
        """Record that `name` should be bound to `image`'s image unit on this kernel, via
        `Texture.bind_to_image(unit, read=read, write=write, level=level, format=format)` at the
        next dispatch. Same deferred-binding discipline as `bind_texture`/`bind_ssbo` -- see
        `bind_ssbo`'s docstring -- and the same immediate typo check on `name`.
        """
        self._resolve_image_unit(name)  # validate now; raises TlangBindingError on typo
        self._image_bindings[name] = (image, read, write, level, format)
        self._bound_image_generation = -1  # force a re-assert on this kernel's next dispatch

    def bind_counters(self, **counters: Buffer | tuple[Buffer, int]) -> None:
        loc = self.bind_counter
        for k, v in counters.items(): loc(k, *v) if isinstance(v, tuple) else loc(k, v)

    def bind_counter(self, name: str, buffer: Buffer, offset: int = 0) -> None:
        """Record that atomic counter `name` should be bound to `buffer` on this kernel,
        resolving its (binding, offset-within-binding) from `self.atomic_counters` -- the canon
        `BindingRegistry.allocate_atomic_counters` assigned and patched into the GLSL as
        `layout(binding = N, offset = M)`. Deferred until the next dispatch, same discipline as
        `bind_ssbo`/`bind_texture` -- see `bind_ssbo`'s docstring for why.

        **Offset composition.** `offset` is the byte offset WITHIN `buffer` where GL's bound
        RANGE begins -- i.e. where the shader's own `layout(offset = 0)` would land -- NOT
        `name`'s own 4 bytes. `name`'s own canon offset `M` is baked into the compiled GLSL, so
        its actual address is `offset + M` in `buffer`. This is the natural composition for the
        idiomatic packed form: two counters sharing one `binding` (`layout(binding=0, offset=0)`
        / `layout(binding=0, offset=4)`) are bound by calling `bind_counter` for EACH name
        against the SAME `buffer` and the SAME `offset` (typically 0, the start of your packed
        counter storage) -- tlang works out the range size from the canon so both land
        correctly. Binding counters that share a `binding` to different buffers or different
        `offset`s is a caller bug and raises at the next dispatch (`_assert_counter_bindings`),
        not here, since the conflict is only visible once every name sharing that binding has
        been recorded.

        `name` is validated immediately against `self.atomic_counters` -- a name this artifact
        does not require (never declared, or declared but pruned as genuinely unused -- see the
        constructor's docstring) is a typo, and raises `TlangBindingError` right here.
        """
        self._resolve_counter_binding(name)  # validate now; raises TlangBindingError on typo
        self._counter_bindings[name] = (buffer, offset)
        self._bound_counter_generation = -1  # force a re-assert on this kernel's next dispatch

    def _resolve_counter_binding(self, name: str) -> tuple[int, int]:
        """Resolve `name`'s (binding, offset) from the static canon (`self._atomic_counters`).
        Raises `TlangBindingError` if `name` isn't a required atomic counter -- that's a typo,
        or a counter this artifact declared but never actually uses (pruned from the required
        set, see the constructor's docstring)."""
        if (pos := self._atomic_counters.get(name)) is not None:
            return pos
        raise TlangBindingError(f"'{name}' is not a declared atomic counter uniform", SourceLocation(module=self._name))

    def _assert_counter_bindings(self, allow_unbound: frozenset[str] | set[str] = frozenset()) -> None:
        """Atomic-counter counterpart of `_assert_ssbo_bindings`/the texture equivalent, with one
        deliberate divergence: a counter not bound through this kernel is checked against GL's
        own binding state rather than against this kernel's records. Everything else mirrors them exactly -- liveness checked every
        dispatch, full recorded set re-asserted with the same generation-counter fast path
        against this table's own counter (`_counter_table_generation`), one `glBindBufferRange`
        call per distinct GL binding (not per name, since several names can share one binding --
        see `bind_counter`), sized wide enough to cover every live counter the canon says shares
        it.

        Why records alone are the wrong test here, unlike every other pool: a counter is commonly
        bound ONCE with a raw `glBindBufferRange` outside any `Kernel` and never rebound -- correct
        usage for a resource that, unlike an SSBO, is not swapped between kernels or frames. A
        "bound through this kernel or raise" rule rejects that, and an over-broad check is worse
        than none because it teaches people to suppress it. Asking GL instead keeps the error that
        matters (nothing bound at that index at all, so the kernel reads no buffer) and drops the
        false positive. `allow_unbound` opts a name out.
        """
        for name, (buffer, _offset) in self._counter_bindings.items():
            if not getattr(buffer, 'alive', True):
                raise TlangBindingError(
                    f"Kernel '{self._name}': buffer bound to atomic counter '{name}' has been "
                    f"freed (recycled temp buffer) -- re-bind before dispatching",
                    SourceLocation(module=self._name),
                )

        # A counter this kernel never bound may still be legitimately bound from outside tlang.
        # Ask GL rather than our own records: `GL_ATOMIC_COUNTER_BUFFER_BINDING` reports whatever
        # is bound at an index, including a raw `glBindBufferRange` we never saw, and reports 0
        # when nothing is. So this fires only on the case that is always a bug -- the kernel will
        # read a counter backed by no buffer at all -- and never on a caller who binds once,
        # globally, outside any Kernel.
        unbound = {n for n in self._atomic_counters if n not in self._counter_bindings and n not in allow_unbound}
        if unbound:
            slot = (ctypes.c_int * 1)()
            for name in sorted(unbound):
                binding, _off = self._atomic_counters[name]
                glGetIntegeri_v(GL_ATOMIC_COUNTER_BUFFER_BINDING, binding, slot)
                if slot[0]: continue
                raise TlangBindingError(
                    f"Kernel '{self._name}' dispatched with atomic counter '{name}' unbound: "
                    f"nothing is bound at atomic-counter binding {binding}, so it would read no "
                    f"buffer at all. Bind it with bind_counter('{name}', ...), or pass "
                    f"allow_unbound={{'{name}'}}",
                    SourceLocation(module=self._name),
                )

        if self._bound_counter_generation == _counter_table_generation:
            return  # nothing else has touched the global counter-buffer table since our last assert

        by_binding: dict[int, tuple[Buffer, int]] = {}
        for name, (buffer, offset) in self._counter_bindings.items():
            binding, _counter_offset = self._atomic_counters[name]
            if binding in by_binding:
                existing_buffer, existing_offset = by_binding[binding]
                if existing_buffer is not buffer or existing_offset != offset:
                    raise TlangBindingError(
                        f"Kernel '{self._name}': atomic counters sharing binding {binding} were "
                        f"bound to different buffers/offsets -- bind every counter that shares "
                        f"one binding to the same buffer and the same range offset",
                        SourceLocation(module=self._name),
                    )
                continue
            by_binding[binding] = (buffer, offset)

        for binding, (buffer, offset) in by_binding.items():
            size = max(o + 4 for b, o in self._atomic_counters.values() if b == binding)
            glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, size)

        generation = bump_counter_table_generation() if self._counter_bindings else _counter_table_generation
        self._bound_counter_generation = generation

    def bind_atomic_counter(
        self, binding: int, buffer: Buffer, offset: int = 0
    ) -> None:
        """Documented escape hatch, kept working exactly as before `bind_counter` existed: binds
        `buffer` to the RAW `binding` index immediately (no deferral, no name validation, no
        generation-counter tracking). Prefer `bind_counter`/`bind_counters` for anything tlang
        itself assigned a binding to -- this one takes the number you hand it, on faith."""
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer.glo, offset, 4)

    def bind_atomic_counters(
        self, *buffers: tuple[int, Buffer] | tuple[int, Buffer, int]
    ) -> None:
        loc = self.bind_atomic_counter
        for v in buffers: loc(*v) if len(v) == 3 else loc(v[0], v[1])

    def _resolve_ssbo_binding(self, buffer_name: str) -> int:
        """Resolve `buffer_name`'s GL binding index, preferring the static canon
        (`self._bindings`, the index tlang itself assigned and patched into the GLSL as
        `binding = N`) over driver reflection. The canon is known before the program even links
        and never depends on whether the driver kept the block active, so it's authoritative;
        reflection is only a fallback for a name absent from it. Binding a buffer to an index the
        program doesn't actually use is legal and harmless. Raises `TlangBindingError` if
        `buffer_name` isn't a real storage block by either source -- that's a typo.
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

    def _assert_ssbo_bindings(self, allow_unbound: frozenset[str] | set[str] = frozenset()) -> None:
        """Called at the top of every dispatch entry point. Three jobs, in order:

        1. Fail loudly if a required block (`self._bindings`, the artifact's declared set) was
           never bound on this kernel -- silently reading whatever another kernel left at that
           index is exactly the bug class this whole mechanism exists to kill. `allow_unbound`
           opts specific names out for the deliberate case.
        2. Check every recorded buffer is still alive -- a recycled `TempHandle` (freed, then
           handed back out by `buffer_pool` to someone else) must never be dispatched against
           silently just because the generation counter looks unchanged.
        3. Re-assert this kernel's full recorded set into GL's global binding table, unless the
           table hasn't moved since this kernel last asserted it (the generation-counter
           fast path) -- repeat dispatches of the same kernel then do zero GL work.
        """
        missing = [
            name for name in self._bindings
            if name not in self._ssbo_bindings and name not in allow_unbound
        ]
        if missing:
            raise TlangBindingError(
                f"Kernel '{self._name}' dispatched with required buffer(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first, or pass allow_unbound={{...}})",
                SourceLocation(module=self._name),
            )

        # Liveness must be checked EVERY dispatch, including the generation-skip path below --
        # a freed TempHandle recorded here and never re-bound would otherwise dispatch silently
        # against a buffer buffer_pool has since handed to someone else.
        for name, (buffer, _offset, _size) in self._ssbo_bindings.items():
            if not getattr(buffer, 'alive', True):
                raise TlangBindingError(
                    f"Kernel '{self._name}': buffer bound to '{name}' has been freed "
                    f"(recycled temp buffer) -- re-bind before dispatching",
                    SourceLocation(module=self._name),
                )

        if self._bound_generation == _ssbo_table_generation:
            return  # nothing else has touched the global table since our last full assert

        for name, (buffer, offset, size) in self._ssbo_bindings.items():
            binding = self._resolve_ssbo_binding(name)
            # offset/size are keyword-only on moderngl <= 5.8.x; passing them positionally raises
            # TypeError there. Keyword form works on every supported version -- see
            # https://moderngl.readthedocs.io/en/5.8.2
            buffer.bind_to_storage_buffer(binding, offset=offset, size=size)

        generation = bump_ssbo_table_generation() if self._ssbo_bindings else _ssbo_table_generation
        self._bound_generation = generation

    def _resolve_texture_unit(self, name: str) -> int:
        """Resolve `name`'s GL texture unit from the static canon (`self._texture_units`,
        decided by `BindingRegistry.allocate_opaque_units` before the program ever links).
        Raises `TlangBindingError` if `name` isn't a declared sampler uniform -- that's a typo.
        """
        if (unit := self._texture_units.get(name)) is not None:
            return unit
        raise TlangBindingError(f"'{name}' is not a declared texture (sampler) uniform", SourceLocation(module=self._name))

    def _resolve_image_unit(self, name: str) -> int:
        """Image counterpart of `_resolve_texture_unit`, resolved from `self._image_units`."""
        if (unit := self._image_units.get(name)) is not None:
            return unit
        raise TlangBindingError(f"'{name}' is not a declared image uniform", SourceLocation(module=self._name))

    def _assert_texture_bindings(self, allow_unbound: frozenset[str] | set[str] = frozenset()) -> None:
        """Texture-unit counterpart of `_assert_ssbo_bindings` -- same three jobs, same
        generation-counter fast path, against the separate texture-unit table/counter."""
        missing = [
            name for name in self._texture_units
            if name not in self._texture_bindings and name not in allow_unbound
        ]
        if missing:
            raise TlangBindingError(
                f"Kernel '{self._name}' dispatched with required texture(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first, or pass allow_unbound={{...}})",
                SourceLocation(module=self._name),
            )

        for name, texture in self._texture_bindings.items():
            if not getattr(texture, 'alive', True):
                raise TlangBindingError(
                    f"Kernel '{self._name}': texture bound to '{name}' has been freed -- "
                    f"re-bind before dispatching",
                    SourceLocation(module=self._name),
                )

        if self._bound_texture_generation == _texture_table_generation:
            return  # nothing else has touched the global texture-unit table since our last assert

        for name, texture in self._texture_bindings.items():
            texture.use(self._resolve_texture_unit(name))

        generation = bump_texture_table_generation() if self._texture_bindings else _texture_table_generation
        self._bound_texture_generation = generation

    def _assert_image_bindings(self, allow_unbound: frozenset[str] | set[str] = frozenset()) -> None:
        """Image-unit counterpart of `_assert_ssbo_bindings` -- same three jobs, same
        generation-counter fast path, against the separate image-unit table/counter."""
        missing = [
            name for name in self._image_units
            if name not in self._image_bindings and name not in allow_unbound
        ]
        if missing:
            raise TlangBindingError(
                f"Kernel '{self._name}' dispatched with required image(s) never bound: "
                f"{', '.join(sorted(missing))} (bind them first, or pass allow_unbound={{...}})",
                SourceLocation(module=self._name),
            )

        for name, (image, *_rest) in self._image_bindings.items():
            if not getattr(image, 'alive', True):
                raise TlangBindingError(
                    f"Kernel '{self._name}': image bound to '{name}' has been freed -- "
                    f"re-bind before dispatching",
                    SourceLocation(module=self._name),
                )

        if self._bound_image_generation == _image_table_generation:
            return  # nothing else has touched the global image-unit table since our last assert

        for name, (image, read, write, level, format) in self._image_bindings.items():
            image.bind_to_image(self._resolve_image_unit(name), read=read, write=write, level=level, format=format)

        generation = bump_image_table_generation() if self._image_bindings else _image_table_generation
        self._bound_image_generation = generation
