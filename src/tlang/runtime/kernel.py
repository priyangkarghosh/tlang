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


class Kernel:
    __slots__ = (
        '_ctx', '_name', '_mglo', '_uniform_cache', '_binding_cache', '_bindings', '_ubo_cache',
        '_ssbo_bindings', '_bound_generation',
    )

    def __init__(self, ctx: Context, name: str, shader: ComputeShader, bindings: Mapping[str, int] | None = None):
        self._ctx = ctx
        self._name = name
        self._mglo = shader
        self._uniform_cache: dict[str, Any] = {}
        self._binding_cache: dict[str, int] = {}
        self._ubo_cache: dict[str, int] = {}
        # the name -> binding map `BindingRegistry.allocate_artifact` decided for this kernel
        # (the artifact's declared/required SSBO blocks, patched into the GLSL text as
        # `binding = N`). This is the canon: static, driver-independent, and known before the
        # program ever links. `_resolve_ssbo_binding` below prefers it over reflection, and
        # `dispatch` treats its keys as the kernel's REQUIRED set (see `_assert_ssbo_bindings`).
        self._bindings: dict[str, int] = dict(bindings) if bindings is not None else {}

        # name -> (buffer, offset, size) recorded by `bind_ssbo`/`bind_ssbos`/`bind`, NOT yet
        # written to GL. Asserted into the real binding table at the top of every dispatch entry
        # point -- see `_assert_ssbo_bindings`.
        self._ssbo_bindings: dict[str, tuple[Buffer, int, int]] = {}
        # `_ssbo_table_generation` as of this kernel's last full assert; -1 means "never
        # asserted", which always forces a rebind on the first dispatch.
        self._bound_generation: int = -1

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
        barrier_bits: int = SHADER_STORAGE_BARRIER_BIT,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self._assert_ssbo_bindings(allow_unbound)
        self._mglo.run(group_x, group_y, group_z)
        if barrier: self._ctx.memory_barrier(barrier_bits)

    def dispatch_indirect(
        self,
        buffer: Buffer,
        offset: int = 0,
        barrier: bool = True,
        barrier_bits: int = SHADER_STORAGE_BARRIER_BIT,
        allow_unbound: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self._assert_ssbo_bindings(allow_unbound)
        glUseProgram(self.glo)
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer.glo)
        glDispatchComputeIndirect(offset)
        if barrier: self._ctx.memory_barrier(barrier_bits)

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

    def bind(self, **available: Buffer | tuple[Buffer, int, int]) -> None:
        """Bind exactly this kernel's REQUIRED set (`self.bindings`, the artifact's declared SSBO
        blocks) by name out of `available`.

        Names in `available` that this artifact does not declare are silently ignored -- this is
        what lets one caller pass a single superset dict of buffers across every kernel in a
        pipeline:

            kernel.bind(**self._buffers)
            kernel.dispatch(groups)

        A required name absent from `available` raises `TlangBindingError` naming it. Accepts the
        same value shapes as `bind_ssbos`: a bare `Buffer`, or `(Buffer, offset, size)`.
        """
        for buffer_name in self._bindings:
            if buffer_name not in available:
                raise TlangBindingError(
                    f"Kernel '{self._name}' requires buffer '{buffer_name}' but it was not in the "
                    f"provided buffers",
                    SourceLocation(module=self._name),
                )
            value = available[buffer_name]
            if isinstance(value, tuple): self.bind_ssbo(buffer_name, *value)
            else: self.bind_ssbo(buffer_name, value)

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
