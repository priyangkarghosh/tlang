# Runtime: ShaderManager, Shader, Kernel, Pipeline, BufferPool

## ShaderManager

```python
sm = ShaderManager(
    ctx=ctx,               # moderngl.Context -- must already exist
    version='460 core',    # emitted as `#version 460 core`
    dir='shaders',         # see path resolution below
    constants={'BLOCK_SIZE': 256},
    strict=True,           # default: raise on any build failure
    keep_sources=False,    # default: drop a successful entry point's generated GLSL
    debug=False,           # default: printf(...) compiles to nothing -- see "printf(...)" below
    debug_log_capacity=4096,   # records the printf ring buffer holds before it starts overflowing
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

**`strict`.** `True` (default): the whole tree is still built — every module is
attempted, in isolation, so one broken file can never stop another from compiling —
and then a single error is raised at the end. If exactly one module failed, its own
exception is re-raised unchanged; if several did, a `TlangBuildError` names every one
of them, with `.failures` mapping module -> its errors. `False`: failures are logged
and collected, and the build continues.

**`keep_sources`.** `False` (default): once an entry point's kernel has compiled (or its
program has linked) and its bindings are verified, its generated GLSL text is dropped --
nothing needs it in memory once the driver has it. A **failed** entry point's source is
always kept regardless of this flag, since that's exactly when someone needs to read it
(and `TlangCompileError.source`/`TlangLinkError` already carry it independently). `True`
retains every entry point's source, unconditionally -- today's behaviour before this flag
existed; use it when you actually intend to inspect `shader.sources`/`get_source(...)`
for successful builds (e.g. dumping generated GLSL for debugging).

**A failed module is not silently usable.** `get_shader(name)` returns `None` for a
module that built but did not fully compile, so the cheap check is also the correct
one:

```python
assert sm.get_shader('physics.dynamics') is not None   # correct: fails if it failed
```

```python
sm.ctx                       # the moderngl.Context
sm['name']                   # same as get_shader(name)
'name' in sm                 # was this module built
sm.get_shader('demo')        # -> Shader | None (None if absent OR not fully compiled)
sm.get_shader('demo', allow_failed=True)   # -> the Shader anyway, to inspect .failures
sm.failures                  # {module: [errors]} for every module that failed
shader.ok                    # did every declared entry point of this module build
shader.failures              # this module's errors (empty when ok)
```

## Shader

One `.tlang` module's build result.

```python
shader.kernels              # dict[str, Kernel] -- compute entry points
shader.programs             # dict[str, moderngl.Program] -- raw linked programs
shader.pipelines            # dict[str, Pipeline] -- name-keyed wrapper, prefer this
shader.interfaces           # Mapping[str, InterfaceDecl] -- this module + its includes
shader.sources              # dict[str, str] -- entry point name -> generated GLSL,
                             # for only what was actually retained (see keep_sources)

shader.get_kernel('cs_go')        # -> Kernel        (KeyError if absent)
shader.get_program('default')     # -> moderngl.Program (KeyError if absent)
shader.get_pipeline('default')    # -> Pipeline      (KeyError if absent)
shader.get_source('cs_go')        # generated GLSL -- always for a failed entry point,
                                   # or any entry point when built with keep_sources=True
```

`get_kernel`/`get_program`/`get_pipeline` raise `KeyError` for an unknown name
(unlike `ShaderManager.get_shader`, which returns `None`) — once a `Shader` exists,
a missing kernel/program signals a real bug, not an expected absence.

`get_source(name)` raises `TlangError` (not `KeyError`, not `None`) for a *successfully
compiled* entry point whose source was dropped under the default `keep_sources=False` --
the message names the entry point and tells you to rebuild with `keep_sources=True`. A
failed entry point's source is always there to read, regardless of the flag.

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

kernel.bind_counter('slotCounter', buf)                  # by name, like everything else above
kernel.bind_counters(slotCounter=buf, other=(buf2, 4))    # tuple = (buffer, range offset)

kernel.bind_atomic_counter(binding, buf, offset=0)        # raw escape hatch -- still works, see below
kernel.bind_atomic_counters((0, buf_a), (1, buf_b, 4))

kernel.atomic_counters     # Mapping[str, tuple[int, int]] -- name -> (binding, offset), for debugging
kernel.default_barrier_bits   # what a bare dispatch(...) will use -- see "Atomic counters" below
```

`dispatch` takes **workgroup counts, not thread counts**. With `[numthreads(256,1,1)]`,
covering `n` elements is `kernel.dispatch((n + 255) // 256)`. Passing `n` directly
launches 256x too many threads — nothing errors, you just get wrong results or an
out-of-bounds write. Prefer `kernel.dispatch_for(n)` below instead of writing that
ceiling division by hand.

## `local_size` and `dispatch_for` — covering N invocations without redeclaring the block size

`[numthreads(...)]`'s arguments are never coerced to `int` by tlang (see the note in
`AttributeHandlers.numthreads`) — `[numthreads(BLOCK_SIZE, 1, 1)]` is a legitimate GLSL
preprocessor macro name, resolved by the driver at compile time, not a Python integer.
That means the generated GLSL text is not a reliable place to read the work-group size
back from. The **linked program** is:

```python
kernel.local_size   # -> (x, y, z), queried once from the driver and cached forever
                     # (a linked program's work-group size can't change)
```

This is a real GL query (`glGetProgramiv(glo, GL_COMPUTE_WORK_GROUP_SIZE, ...)`), so it
resolves any `{{ CONSTANT }}`-templated macro exactly the way the driver did at link
time — a build with `constants={'BS': 128}` and `[numthreads(BS, 1, 1)]` reports
`local_size == (128, 1, 1)`, not the literal text `'BS'`.

`dispatch_for` derives its group counts from `local_size`, so the caller never redeclares
the block size in Python (and can never let that redeclaration drift from the shader's):

```python
kernel.dispatch_for(num_particles)                  # 1-D, the common case
kernel.dispatch_for(width, height)                   # 2-D
kernel.dispatch_for(width, height, depth)            # 3-D
kernel.dispatch_for(n, elems_per_thread=4)           # scan-style: 4 elements per thread
```

Ceiling division throughout — a kernel is never under-dispatched, even when `x`/`y`/`z`
isn't an exact multiple of `local_size`. `elems_per_thread` divides `x` before the
ceiling division (it applies to `x` alone, matching the 1-D idiom it's modeled on — a
kernel needing per-thread multiplicity on more than one axis should call `dispatch`
directly with hand-computed group counts). `dispatch_for` forwards `barrier`/
`barrier_bits`/`allow_unbound` to `dispatch` unchanged, so it goes through the exact same
`_assert_ssbo_bindings` (and texture/image/counter) re-assert path — it is a drop-in
replacement for `dispatch((n + local_size[0] - 1) // local_size[0])`, not a separate
binding path. `dispatch(groups_x, ...)` remains the explicit, raw-group-count form.

`Pipeline` has no equivalent — work-group size is a compute-only concept (raster stages
have no `[numthreads(...)]`, and `VAO.render`'s vertex/instance counts are unrelated).

## Blocks are stripped per artifact, by reachability from `main()`

Each entry point gets its own translation unit, and tlang removes from it, in order:
every top-level function not reachable from `main()`, then every buffer/uniform block
none of the surviving code references. So `kernel.bindings` means **"blocks the code
reachable from this kernel's `main()` touches"** — not "blocks declared anywhere in
the file or its `[include]`s".

That distinction matters: an `[export()]`ed helper is emitted into every entry point
of the including file, so before reachability analysis a one-line kernel could declare
a dozen blocks it never touches. It no longer does.

Stripping exists because the per-stage block limit is far lower than the binding-index
ceiling — `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` is often 96 while
`GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS` is often only 16, and exceeding the latter is a
hard link failure. Both passes are textual, not a real GLSL parse, and are deliberately
biased toward keeping when uncertain.

**You no longer need to match a bind list to the stripped set by hand.** `kernel.bind()`
takes the superset you own and binds only what this artifact declares:

```python
kernel.bind(**self._buffers)      # extra names ignored; a REQUIRED name missing raises
```

`bind_ssbo`/`bind_ssbos` remain the explicit form and still raise `TlangBindingError`
for a name the artifact does not declare — that is a typo, and worth catching.

Inspect what an artifact actually got:

```python
print(dict(kernel.bindings))         # {'Particles': 0, 'Fixed': 3}
print(dict(kernel.uniform_blocks))   # {'Lights': 0}
```

## Bindings are per-kernel, and re-asserted at dispatch

GL's SSBO binding table is process-global, and each artifact numbers its blocks
independently. Binding through kernel A and then dispatching kernel B therefore used to
write A's name->index mapping and run B against its own, differently-numbered indices —
silently, with no exception and plausible-looking output.

That is now structurally impossible:

* `bind_ssbo`/`bind_ssbos`/`bind` **record** on the kernel; they do not write GL. The
  name is validated immediately, so a typo still raises at the call.
* `dispatch`, `dispatch_indirect` and `dispatch_timed` each re-assert that kernel's
  entire recorded set immediately before running. What is in the table always matches
  what the kernel about to run asked for.
* A module-level generation counter makes a repeat dispatch of an already-current
  kernel free; any other kernel's or pipeline's bind invalidates it.

```python
a.bind_ssbos(X=buf1); b.bind_ssbos(X=buf2)
a.dispatch(n)    # a sees buf1
b.dispatch(n)    # b sees buf2 -- no interleaving hazard
```

**Dispatching with a required block never bound raises** `TlangBindingError` naming it,
rather than reading whatever another kernel left at that index:

```python
kernel.dispatch(n, allow_unbound={'Debug'})   # opt out deliberately
```

A `TempHandle` freed back to the `BufferPool` while still recorded on a kernel also
raises at the next dispatch, instead of dispatching against a buffer the pool has since
handed to someone else.

**`Pipeline` (raster) is the exception.** Drawing happens in moderngl's `VAO.render`,
outside tlang, so `Pipeline.bind_ssbo` binds immediately and cannot re-assert. A compute
dispatch between a pipeline's bind and its draw can still rewire it.

## Atomic counters

`uniform atomic_uint x;` is bound by name, like every other resource -- `bind_counter`/
`bind_counters` resolve `(binding, offset)` from tlang's own textual canon and call
`glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER, binding, buffer, offset, size)` for you. Discovery,
allocation and (unlike SSBOs/UBOs) textual patching all happen automatically:

```python
kernel.bind_counter('slotCounter', buf)     # buf's own byte 0 is where offset=0 in GLSL lands
kernel.dispatch(n)                          # ATOMIC_COUNTER_BARRIER_BIT included automatically
```

**Allocation is 2-D: `(binding, offset)`, not a single index.** GL is designed to pack several
counters into one binding at successive 4-byte offsets --
`layout(binding=0, offset=0) uniform atomic_uint a;` /
`layout(binding=0, offset=4) uniform atomic_uint b;` compile and work together, and that is the
idiomatic form, not a conflict. Unpinned counters of one artifact are packed into a single binding
this way, spending only one of the driver's (typically 8, per stage) counter-buffer slots no
matter how many counters share it. An explicit `layout(binding=..., offset=...)` pin is honoured
and never moved; two counters pinned to the same `(binding, offset)` is a build-time
`TlangBindingError`.

**Atomic counters are invisible to moderngl's own reflection entirely** -- `program.get('x')` is
always `None`, verified against a live GL 4.6 context, even for a counter the shader actively
uses. So unlike samplers/images (assigned post-link, through a writable `.value`), an unpinned
counter's binding/offset is patched directly into the generated GLSL text, the same way SSBO/UBO
blocks are. Reflection cannot even tell tlang whether a *pinned* declaration compiled correctly,
so verification for this pool goes through raw pyOpenGL's `GL_ATOMIC_COUNTER_BUFFER`
program-interface query instead, post-link -- the only source that can see which of the
textually-discovered counters the driver actually kept active. That query is also what prunes
`kernel.atomic_counters`/`pipeline.atomic_counters` down to counters the artifact genuinely uses,
exactly like the sampler/image pools already do: a declared-but-unused counter never demands a
binding.

**`bind_counter`'s `offset` is the start of the bound RANGE in your buffer, not the counter's own
4 bytes.** `name`'s own canon offset `M` is baked into the compiled GLSL, so its actual address is
`offset + M` in `buffer`. Two counters sharing one `binding` are bound by calling `bind_counter`
for EACH name against the SAME `buffer` and the SAME `offset` (typically 0, the start of your
packed counter storage) -- tlang works out the correct range size from the canon so every live
counter at that binding lands right, regardless of which name you bind first. Binding counters
that share a binding to different buffers/offsets raises at the next dispatch, once every name
sharing it has been recorded.

**Dispatching with a declared-and-required counter never bound through `bind_counter` does NOT
raise** -- the one deliberate divergence from every other pool's `bind_ssbo`/`bind_texture`/
`bind_image`-style "never bound" error. A real consumer of this feature binds its one global
atomic counter buffer exactly once, via a raw `glBindBufferRange` call made entirely outside any
`Kernel`, and never rebinds it for the life of the process -- correct, idiomatic usage for a
resource that (unlike an SSBO) is never swapped to a different buffer between kernels or frames.
Enforcing "bound through this kernel or dispatch raises" would break that pattern. `bind_counter`
and its deferred re-assert (liveness check, generation-counter fast path, same discipline as
`bind_ssbo`) still exist and still protect a caller who opts in by calling it at all -- a typo in
the name still raises immediately, at the `bind_counter` call itself.

**The `bind_atomic_counter(binding: int, buffer, offset=0)` raw escape hatch keeps working
exactly as before** `bind_counter` existed: it binds the raw index you hand it, immediately, with
no name resolution and no tracking of any kind. Prefer `bind_counter` for anything tlang itself
assigned a binding to.

**The default `barrier_bits` includes `ATOMIC_COUNTER_BARRIER_BIT` automatically** when the
kernel's artifact declares a genuinely-used counter (`SHADER_STORAGE_BARRIER_BIT` is always
included regardless, unchanged from before this feature). Consumers used to hand-write that OR
themselves with a comment explaining why -- `kernel.default_barrier_bits` shows what a bare
`dispatch()` will use, and an explicitly passed `barrier_bits` still wins exactly as before:

```python
kernel.dispatch(n)                                    # counter bit added automatically if declared
kernel.dispatch(n, barrier_bits=SHADER_STORAGE_BARRIER_BIT)   # explicit value always wins
```

`Pipeline.bind_counter`/`bind_counters` mirror the `Kernel` API but bind immediately, exactly like
`Pipeline.bind_ssbo`/`bind_texture` -- see "`Pipeline` (raster) is the exception" above.

## `printf(...)` -- a debug log callable from inside shader code

OpenGL has no `debugPrintfEXT` (that's Vulkan-only). `printf(...)` is tlang's replacement, and the
**leading, normal way to use it is guarded by thread id**:

```glsl
if (gid == 0) printf("frame start, dt=%f\n", dt);   // the common case: one line per dispatch
```

Be honest with yourself about the alternative before reaching for it: **per-thread printing at
full occupancy is not workable.** One unguarded `printf(...)` in a million-invocation dispatch is a
million records, most of which will be dropped (see "Bounded ring, ACK'd" below) and the rest of
which will flood any console you point at them. This feature is closer to an **assert** you can
leave compiled in than a general-purpose log -- reach for `if (gid == 0)`, `if (gid == targetId)`,
or a rare-condition guard (`if (isnan(x))`) before printing from every invocation.

```glsl
printf("ptc %d depth %f\n", gid, depth);   // %d, %u, %f, %x, %% -- see "Format specifiers" below
printf("hit\n");                            // no arguments is fine
```

GLSL has no strings, so the format string never reaches it: tlang lifts it out of your source at
build time, assigns it (and the call site itself) an id, and rewrites the call to carry only the id
and the values:

```
printf("ptc %d depth %f\n", gid, depth);   ->   printf(7u, gid, depth);
```

-- and reads the decoded, formatted lines back on the host, from the shader's stdout:

```python
sm = ShaderManager(ctx, '460 core', 'shaders', debug=True)   # off by default
...
kernel.dispatch(n)
for line in sm.stdout.drain(): print(line)   # 'physics.dynamics:42  ptc 3 depth 0.5'
if sm.stdout.dropped: print(f'{sm.stdout.dropped} printf(...) calls were dropped -- ring was full')
```

Every line leads with WHERE it came from (`module:line`, resolved from the call site, not baked
into the format text) followed by the formatted message -- the format string says what the values
mean, the call site says where they came from.

**`debug=False` (the default) is completely inert.** `printf(...)` calls are never rewritten or
stripped from your source depending on mode -- tlang has been burned by textual call-stripping
before, and doing that to `printf(...)` specifically would be the same mistake. The format-string
extraction and call-site id rewrite happen in BOTH modes (so a specifier/argument mismatch is a
build error in a release build too, not only when `debug=True`); what differs is only whether
`printf`'s emitted GLSL *definition* has a real body or an empty one. **An artifact that never calls
`printf` gets no overloads, no ring buffer, and no binding slot, in either build** -- proven by
benchmark, not assumed: unconditionally injecting the full overload set into every artifact
regardless of use measurably slowed a many-kernel real project's build (this was v1's `print`'s own
finding, and still holds), so both modes gate injection on the same cheap
`printf(`-in-this-artifact's-own-source check. The only build that pays anything for printf support
is one that actually calls it; an empty release body costs nothing further, since the driver is
free to eliminate a called-but-empty function.

**Format specifiers.** `%d` (signed int), `%u` (unsigned), `%f` (float), `%x` (unsigned hex), `%%`
(a literal `%`, consumes no argument). The specifier count must match the argument count exactly --
checked at BUILD TIME, naming the format and the mismatch:

```
demo:4: printf format 'a %d b %d c %d' has 3 specifier(s) but 2 argument(s) were passed
```

**Up to 8 arguments per call** (`MAX_PRINTF_ARGS` in `tlang.runtime.printf_log`) -- a direct
per-record memory cost (each ring slot is sized for the worst case regardless of how many
arguments any individual call actually used), tuned the same way v1 `print`'s limit was: against
how many values one debug line realistically carries, not any GLSL/driver ceiling.

**Locating the format string.** A naive regex over raw text breaks on an escaped quote, a comma or
a close-paren inside the string, and `%%` -- tlang scans character-by-character (respecting `\"`
escapes) to find the literal's real span, so all four are handled correctly:

```glsl
printf("a \" b\n");        // escaped quote -- not the end of the string
printf("x, y: %d\n", n);   // comma inside the string -- not an argument separator
printf("f(%d)\n", n);      // close-paren inside the string -- not the call's own
printf("100%% done\n");    // %% -- a literal percent, not a specifier
```

**Overloads.** GLSL has no varargs, so tlang emits one concrete overload per arity (0-8), and every
value parameter is `uint` -- there is no type-matched overload set to resolve at all. Each argument
is cast to its stored `uint` bit pattern AT THE CALL SITE, driven by that argument's OWN format
specifier (`%d`/`%u`/`%x` -> `uint(expr)`, `%f` -> `floatBitsToUint(expr)`), not by leaning on GLSL
overload resolution:

```
printf("%d %f %d %f\n", 7, 2.5, 9, 4.5);
  ->  printf(3u, uint(7), floatBitsToUint(2.5), uint(9), floatBitsToUint(4.5));
```

This is deliberate, not incidental: an earlier version tried to make mixed-type calls resolve to an
exact-type overload the way v1's `print` did, via a full type cross product -- but GLSL's overload
rules only make that tractable for arities 2-3 (a 4-type cross product is `4**n` overloads), so
arities 4-8 fell back to same-type-only overloads. A mixed-type call at arity 4+ (e.g. `printf("%d
%f %d %f\n", 7, 2.5, 9, 4.5)`) then had no matching overload, so GLSL silently implicit-converted
every integer argument to `float` to match the all-float overload -- and the host, decoding per the
format's `%d`, read the resulting float bit pattern back as if it were an integer. Wrong values,
no error. Casting per-specifier at the call site removes the overload-resolution step (and its
implicit-conversion hazard) for every arity uniformly, and is smaller: 9 overloads total instead of
a type cross product.

**Publish ordering.** Each record is: reserve a ring slot, write the body, call
`memoryBarrierBuffer()`, THEN publish (a per-slot ready word) -- a reader can never observe a record
before its body has landed. `memoryBarrierBuffer()` is per-invocation, not the device-wide
`glMemoryBarrier`, so it costs nothing close to a real device barrier's price.

**Bounded ring, ACK'd.** The ring holds `debug_log_capacity` records (default 4096, set on
`ShaderManager(...)`). A claim past capacity is dropped, not wrapped or corrupted, and counted --
`sm.stdout.dropped` reports exactly how many. What gets dropped is always the NEWEST claim attempt;
the earliest records that fit are the ones that survive, since silently losing those would be the
worst version of this failure. The host acknowledges a record by writing back into the SAME buffer
after decoding it (a plain memory write through a coherent persistent mapping -- no GL call), which
is what lets the ring reuse that slot; see `tlang.runtime.printf_log`'s module docstring for exactly
how (a bounded lock-free MPSC queue, not a plain atomic counter -- the plain-counter version has a
real bug where a single overflow can permanently strand every later record, found and fixed while
building this feature).

**Volume control.** Three things make this usable instead of a console-flooding liability:
- **Guard by thread id** (see the top of this section) -- the intended default usage.
- **Dedup**: `sm.stdout.drain()`/`stream()` collapse a run of CONSECUTIVE identical lines into one,
  suffixed with a repeat count (`'dbg:6  hit  (x8)'`) -- pass `dedup=False` to see every line raw.
- **A rate limit on `stream()`'s sink** (a token bucket, `rate_limit` lines/second, default 200) so
  a flood can never overwhelm whatever `sink` does -- excess lines are withheld and reported as one
  summary line (`'... N line(s) suppressed by rate limit ...'`), not silently discarded data (the
  ring's own `dropped` count is unaffected either way).

**tlang owns the ring buffer.** You never declare, bind, size, or free it --
`ShaderManager(debug=True)` allocates ONE pinned (persistent-mapped) buffer shared by every
kernel/pipeline it builds, and automatically binds it (as an ordinary SSBO, no atomic-counter
machinery involved) to whichever artifacts actually declare it.

## `sm.stdout` -- the shader's stdout

```python
sm.stdout.stream(sink=print)   # background thread -> sink(line) per formatted record, until stop()
sm.stdout.stop()               # stop it cleanly
sm.stdout.drain()              # -> formatted lines available right now, synchronously
sm.stdout.dropped              # records dropped so far because the ring was full
sm.stdout.clear()              # reset the ring's cursors/counters
sm.stdout.capacity             # the ring's capacity, in records
```

`None` when `debug=False` (the default). Shared by the WHOLE `ShaderManager` tree, not one kernel:
`clear()`/`dropped` and every drained/streamed line reflect every participating kernel/pipeline's
`printf(...)` calls since the last clear, not just one artifact's own.

`stream()` runs its poll loop on a background daemon thread that touches only the ring's pinned
memory mapping directly -- never a GL call, so it needs no GL context and never stalls the render
thread the way `buffer.read()` would. (Verified while building this: a thread with no GL context
polling the raw mapping saw every record live and in order while the GL thread kept dispatching.)
Without `GL_ARB_buffer_storage` (rare on anything from the last decade), the ring falls back to a
plain buffer with no real mapping -- `stream()` still runs, but correctness there depends on your GL
binding tolerating a cross-thread `.read()` call, which is not guaranteed; prefer `drain()` from the
render thread in that fallback case. `drain()` always works correctly either way, mapped or not.

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

### Pinned buffers -- persistent-mapped GL storage

`alloc_pinned` gives you a buffer allocated with immutable `glBufferStorage` and mapped ONCE for
its entire life, so CPU code reads/writes it through a `memoryview` instead of round-tripping
every access through `glBufferSubData`/`glGetBufferSubData`. It duck-types exactly the slice of
`moderngl.Buffer` tlang's binding layer touches (`.glo`, `.size`, `.bind_to_storage_buffer`,
`.read`, `.write`, `.release`), so it drops straight into `kernel.bind(**pool)` with zero
special-casing, same as a `persistent_buffer` or a tagged `alloc_temp`:

```python
buf = pool.alloc_pinned(size_bytes, tag='Particles', read=True, write=True)

kernel.buffer_source = pool
kernel.bind()                 # resolves 'Particles' from the pool exactly like any other buffer
kernel.dispatch(groups)

pool.free_pinned('Particles')          # or pool.free_pinned(buf) for an untagged allocation
```

**Immutable storage means no recycling.** `glBufferStorage` allocates memory that cannot be
resized or orphaned, so pinned buffers never enter `alloc_temp`'s size-classed free lists --
each is its own GL allocation with its own lifetime. They still live in the pool's shared
name -> buffer `Mapping` when tagged (`pool[tag]` resolves them like anything else), and
`pool.clear()` releases every pinned buffer (tagged or not) along with everything else.

**The fencing contract -- read this before passing `sync=False` anywhere.** A coherent
persistent mapping gives the CPU and GPU zero ordering on their own: the CPU can read bytes an
in-flight compute dispatch hasn't finished writing, or overwrite bytes a dispatch is still
reading, and neither ever raises -- it silently produces torn or stale data. `PinnedBuffer`
closes that hole and makes the safe path the default:

- `buf.fence()` -- call this immediately after GL work that reads or writes the buffer (right
  after `kernel.dispatch(...)`). Records a fence covering every GL command issued so far.
- `buf.read(...)` / `buf.write(...)` default to `sync=True`: they block on the most recent
  fence before touching the mapping, so a plain `.read()` can never hand back torn data. No
  `fence()` yet means nothing to wait on, so the very first access proceeds immediately.
- `sync=False` skips that wait -- only for a caller who has already synchronised some other way
  (e.g. `ctx.finish()`) and wants to avoid a redundant stall.
- `buf.mapping` is the raw `memoryview` -- zero-copy, and deliberately UNSYNCED. It is the
  explicit opt-in escape hatch for a caller doing its own sync bookkeeping; `read()`/`write()`
  are the safe default entry points.

CPU -> GPU writes need no extra sync call: `GL_MAP_COHERENT_BIT` alone guarantees a client write
is visible to any GL command issued afterwards. `fence`/`wait` exist for the other two hazards --
GPU-write-then-CPU-read, and CPU-write-racing-a-still-in-flight GPU-read.

**Feature detection, not a hard requirement.** Pinned buffers need `GL_ARB_buffer_storage` (core
in GL 4.4). When it's absent, `alloc_pinned` does not raise -- it transparently falls back to a
plain `moderngl.Buffer` behind the identical API and logs a warning. Check `buf.is_pinned` if the
distinction matters to your code (`True` = real persistent mapping; `False` = fallback, and
`.mapping` raises `TlangError` since there's nothing zero-copy to hand back -- use `read()`/
`write()`, which work correctly either way).

**Measured on an RTX 3090 (GL 4.6, moderngl 5.12):** 200 x 1 MiB writes, ~9 ms through a pinned
mapping vs ~45 ms through `moderngl.Buffer.write` (roughly 5x); 200 x 4 KiB reads, ~0.1 ms
through a pinned mapping vs ~0.6 ms through `moderngl.Buffer.read` with no GPU work in flight
(the read gap widens sharply once a real fence wait is involved on the `moderngl.Buffer` side,
since that path has no cheaper way to synchronise than a full round trip).

## Which tlang am I actually running?

A copied (non-editable) install and an editable checkout are indistinguishable by version
number — and the version has gone *backwards* across a refactor before, so comparing
versions cannot answer "is my install current?". The package directory can:

```python
import tlang
tlang.__version__          # '1.3.26'
tlang.build_info()         # {'version': ..., 'package_dir': PosixPath(...), 'editable': True}
```

`package_dir` is the load-bearing field: if it points into your source checkout, your edits
are live; if it points into `site-packages`, you are running a copy and edits to the source
tree do nothing.

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
