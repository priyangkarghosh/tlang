# Runtime: ShaderManager, Shader, Kernel, Pipeline, BufferPool

## ShaderManager

```python
sm = ShaderManager(
    ctx=ctx,               # moderngl.Context -- must already exist
    version='460 core',    # emitted as `#version 460 core`
    dir='shaders',         # see path resolution below
    constants={'BLOCK_SIZE': 256},
    strict=True,           # default: raise on any build failure
)
```

**Path resolution.** A relative `dir` resolves against the directory of the *file that
constructed* `ShaderManager`, not the process working directory (via
`sys._getframe(1)`, so an indirect construction through a factory/wrapper resolves
against that wrapper's file — pass an absolute `dir` if that's not what you want).
This lets a library ship shaders next to its own source regardless of where the app
launched from. Absolute paths are used as-is.

**Module names** are the file's path relative to `dir`, dotted, without the
extension: `shaders/fx/blur.tlang` -> `'fx.blur'`. Two files mapping to the same
module name (`a/b.tlang` and `a.b.tlang`) are rejected.

**`strict`.** `True` (default): any compile/link/attribute/pipeline error raises
immediately. `False`: downgrades to a logged warning and continues with a partial
build — a missing kernel/program then surfaces later as a plain `KeyError`, so this
trades an early loud failure for a late confusing one. Prefer `True` except while
actively iterating.

```python
sm.ctx                      # the moderngl.Context
sm['name']                  # same as get_shader(name)
'name' in sm                # was this module built
sm.get_shader('demo')       # -> Shader | None (None if absent -- does NOT raise)
```

## Shader

One `.tlang` module's build result.

```python
shader.kernels              # dict[str, Kernel] -- compute entry points
shader.programs             # dict[str, moderngl.Program] -- raw linked programs
shader.pipelines            # dict[str, Pipeline] -- name-keyed wrapper, prefer this
shader.interfaces           # Mapping[str, InterfaceDecl] -- this module + its includes
shader.sources              # dict[str, str] -- entry point name -> generated GLSL

shader.get_kernel('cs_go')        # -> Kernel        (KeyError if absent)
shader.get_program('default')     # -> moderngl.Program (KeyError if absent)
shader.get_pipeline('default')    # -> Pipeline      (KeyError if absent)
shader.get_source('cs_go')        # generated GLSL for any entry point, success or not
```

`get_kernel`/`get_program`/`get_pipeline` raise `KeyError` for an unknown name
(unlike `ShaderManager.get_shader`, which returns `None`) — once a `Shader` exists,
a missing kernel/program signals a real bug, not an expected absence.

## Kernel and Pipeline

Both expose the identical name-keyed API — `Kernel` for compute entry points,
`Pipeline` for `[program(...)]`s. Graphics code never needs a hardcoded binding
number any more than compute code does.

```python
k.bind_ssbo('Particles', buf)                            # by name, never a literal int
k.bind_ssbos(Particles=buf, Fixed=(buf2, 0, 1024))        # tuple = (buffer, offset, size)
k.bind_ubo('Params', ubo_buf)
k.bind_ubos(Params=ubo_buf, Lights=(lights_buf, 0, 256))

k.set_uniform('threshold', 0.5)
k.set_uniforms(threshold=0.5, count=n)
k['threshold'] = 0.5                                      # __setitem__ == set_uniform
k['threshold']                                             # __getitem__ -> raw moderngl.Uniform

k.bindings          # Mapping[str, int] -- SSBO block name -> binding, for debugging
k.uniform_blocks    # Mapping[str, int] -- UBO block name -> binding, for debugging
```

`bindings` is SSBO-only on both classes; `uniform_blocks` is the UBO equivalent — the
two are separate GL pools (see `references/resources.md`). Binding numbers are an
implementation detail that can be different every build; code that hardcodes
`buffer.bind_to_storage_buffer(3)` will break the moment the artifact's block set
changes, `bind_ssbo('Name', ...)` will not.

`Kernel` additionally has dispatch:

```python
kernel.dispatch(groups_x, groups_y=1, groups_z=1)         # issues a memory barrier by default
kernel.dispatch(n, barrier=False)                         # skip when chaining dependent passes
kernel.dispatch_indirect(indirect_buffer, offset=0)
elapsed_ms = kernel.dispatch_timed(n)                     # blocks on finish() -- profiling only

kernel.bind_atomic_counter(binding, buf, offset=0)
kernel.bind_atomic_counters((0, buf_a), (1, buf_b, 4))
```

`dispatch` takes **workgroup counts, not thread counts**. With `[numthreads(256,1,1)]`,
covering `n` elements is `kernel.dispatch((n + 255) // 256)`. Passing `n` directly
launches 256x too many threads — nothing errors, you just get wrong results or an
out-of-bounds write.

## Unused SSBO/UBO blocks are stripped per artifact

tlang deletes any buffer/uniform block an entry point's reachable code doesn't
reference, because the per-stage block limit is far lower than the binding-index
ceiling — `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` is often 96 while
`GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS` is often only 16, and exceeding the latter is
a hard link failure. The analysis is textual, not a real GLSL parse, and is
deliberately biased toward keeping a block when uncertain.

A stripped block is gone from reflection too — `bind_ssbo`/`bind_ubo` on one raises
`TlangBindingError`:

```
cs_noop: 'Unused' is not a valid buffer block (Missing binding)
```

The fix is never on the Python side: something in that entry point's reachable code
must actually read or write the block's fields.

Inspect what an artifact actually got:

```python
print(dict(kernel.bindings))         # {'Particles': 0, 'Fixed': 3}
print(dict(kernel.uniform_blocks))   # {'Lights': 0}
```

## BufferPool

Recycles scratch GL buffers. Prefer the scope guards — they return the buffer even
when the body raises, which manual `free_temp` cannot promise.

```python
from tlang import BufferPool

pool = BufferPool(ctx, min_size=256, debug_poison=False)

with pool.temp(size_bytes, zero=True) as scratch:   # one buffer, auto-returned
    kernel.bind_ssbo('Scratch', scratch)
    kernel.dispatch(groups)

with pool.frame():                  # everything alloc_temp'd inside is reclaimed at exit
    a = pool.alloc_temp(1024)
    b = pool.alloc_temp(4096)
    # no matching free_temp needed -- both returned when the `with` exits, even on raise

buf = pool.persistent_buffer('globals', size=256)   # named, created once, never recycled

print(pool.metrics())   # BufferPoolMetrics: bytes pooled/checked out, high-water, hits/misses
pool.trim()             # release all currently-idle transient buffers; returns bytes freed
pool.clear()            # release EVERYTHING (persistent + pooled + checked-out) -- safety net
```

**Recycled memory is undefined by default.** `alloc_temp`/`temp(...)` hand back
whatever a previous consumer wrote unless you pass `zero=True` (costs one GL clear).
Skipping it when your shader assumes clean scratch works by accident until pool
occupancy changes, then silently reads stale data. `BufferPool(ctx, debug_poison=True)`
stamps every buffer with `0xCD` on return-to-pool so a build with poisoning enabled
fails loudly instead of quietly on a forgotten `zero=True`.

Size classes are power-of-two and segregated: a small request can never consume an
oversized pooled buffer.

**Handles track their own liveness**, independent of the underlying GL object name
(which GL can recycle after release). Use-after-free and double-free both raise
`TlangError` rather than silently touching a buffer that now belongs to someone else:

```
Use of buffer handle after free: 'read' accessed on a freed temp buffer
Buffer handle already freed (double free)
```

## Verifying a change

There is no CLI — build the shaders in-process and inspect the result:

```python
sm = ShaderManager(ctx=ctx, version='460 core', dir='shaders', constants={...})
sh = sm.get_shader('demo')
print(sorted(sh.kernels), sorted(sh.programs))
```

With `strict=True` a failure raises immediately with a location and message that
names the fix. Getting a `Shader` back with the kernels/programs you expected means
the whole pipeline compiled and linked — trust that over reading GLSL and reasoning
about whether it should work.

tlang logs through the standard `logging` module under `tlang.*`:

```python
import logging
logging.getLogger('tlang').setLevel(logging.INFO)   # build timings, per-stage progress
```
