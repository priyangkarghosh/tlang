# Errors

Every exception derives from `TlangError` and (usually) carries a source location
that renders as `module:line`. Every message below was triggered for real against a
live GL context (`moderngl.create_context(require=460, standalone=True)`, RTX 3090)
and is pasted verbatim, not paraphrased.

```python
from tlang.errors import (
    TlangError, TlangSyntaxError, TlangAttributeError, TlangDependencyError,
    TlangBindingError, TlangCompileError, TlangLinkError, TlangBuildError,
)
```

| Exception | Raised when |
|---|---|
| `TlangSyntaxError` | malformed attribute, unbalanced brackets, unterminated body, malformed `[varyings]`/`[uniforms]`/`[buffer]` struct |
| `TlangAttributeError` | unknown/misapplied attribute, bad arguments, invalid `[program(...)]`, unresolved/mismatched `[uses(...)]` |
| `TlangDependencyError` | missing module, circular `[include]`, duplicate module name, missing shader dir, missing template constant, `{{ }}` template syntax error |
| `TlangBindingError` | binding conflict/exhaustion, unknown block name at runtime, stage over its block limit |
| `TlangCompileError` | a stage failed to compile — carries `.stage`, `.entry_point`, `.source` |
| `TlangLinkError` | a `[program(...)]` failed to link |
| `TlangBuildError` | more than one module in the tree failed — `.failures` maps module -> its errors |
| `TlangError` (base) | `BufferPool`/`TempHandle` misuse (use-after-free, double-free, bad size), `Shader.get_source(...)` on a source that was dropped (`keep_sources=False`) |

`e.source` (on `TlangCompileError`) always has the exact GLSL handed to the driver, with
tlang's `#line` directives intact, so a driver message's line number points back at your
`.tlang` file. `Shader.get_source(...)` has it too for a *failed* entry point, always --
but for a *successfully* compiled one only when the `ShaderManager`/`Shader` was built
with `keep_sources=True` (default `False`: a successful entry point's source is dropped
once nothing needs it anymore). Calling `get_source(...)` on a dropped entry point raises
`TlangError` naming it, not `KeyError` and not `None`:

```
demo: 'cs_go': source was not retained (built with keep_sources=False) -- rebuild the ShaderManager/Shader with keep_sources=True to inspect it.
```

## Attribute and syntax errors

**Unknown attribute — did-you-mean:**
```
demo:2: Unknown attribute 'shadr'. Did you mean 'shader'?
```

**Stage-specific attribute on the wrong stage** — names the stages it's valid on:
```
demo:3: 'early_fragment_tests' is not valid on a comp function (valid on: frag)
```

**Conflicting duplicate attribute** (two `[numthreads(...)]` on one function):
```
demo:4: Layout qualifier 'local_size_x' for 'in' is set twice: first by '[numthreads(64,1,1)] at demo:3', again by '[numthreads(128,1,1)] at demo:4'
```

**Malformed interface struct** (an instance name after the closing brace — a flat
interface struct doesn't take one):
```
demo:1: struct 'Bad': expected ';' immediately after the closing '}' (a flat interface struct takes no instance name), found 'named_instance;'
```

**An unbalanced parenthesis inside an attribute's argument list** — including one inside
a `//` comment, since the comment is part of the argument text:
```
demo:3: Malformed attribute 'glsl(
    uniform float dt;
    // the substep length, used for the clamp (see Sec4.3 Eq. 10
)' in [glsl(
    uniform float dt;
    // the substep length, used for the clamp (see Sec4.3 Eq. 10
)]. Expected 'name' or 'name(args)'.
```
Comments inside an attribute's parentheses are otherwise fine, including ones containing
commas and *balanced* parentheses. Attribute parsing is also time-bounded, so no input
can hang the preprocessor; exceeding the bound raises `TlangSyntaxError` naming the
attribute rather than spinning.

## `[varyings]`/`[uses(...)]` errors

**Typo in `[uses(...)]` — reported twice: once at the attribute, once at the
program's cross-stage check, both with did-you-mean:**
```
demo:12: [uses('VertexOu')]: no interface named 'VertexOu' is declared or included. Did you mean 'VertexOut'?
demo: program 'default': vert stage 'vs_main' writes interface 'VertexOut' (declared demo:3) but frag stage 'fs_main' reads 'VertexOu' (used at demo:12). Did you mean 'VertexOut'?
```

**Two modules declare the same interface name with different members:**
```
a: 'VertexOut' is declared differently in two modules: b:3 has 'vec4 color' but a:4 has 'vec3 color'
```

**Second `[uses(..., dir=...)]` for a direction already claimed on this function:**
```
demo:9: [uses('B', dir='out')]: function 'vs_main' already declares an 'out' interface ('A' at demo:8); a stage may reference one interface per direction
```

**`[uses(...)]` on a compute stage** (no in/out to attach to):
```
demo:6: [uses('A')]: function 'cs_bad' is a compute stage; compute has no stage interface direction to attach 'A' to
```

**`[uses(...)]` naming a `[uniforms]`/`[buffer]` struct instead of `[varyings]`:**
```
demo:6: [uses('Frame')]: 'Frame' is a uniforms interface (declared demo:3), not varyings; only a [varyings] declaration can be referenced with [uses(...)]
```

**Missing `[varyings(locations=false)]` on a member with no measurable location
span** (raised once per stage that uses the interface, so it can appear twice for
one declaration):
```
demo:3: interface 'Batch' member 'vec4 items[{{ N }}]' has no measurable location span; add [varyings(locations=false)] or give the array a literal size
```

## `[buffer]` single-declarator shorthand errors

**Two declarators in one shorthand statement** (a block has exactly one name, so this is
ambiguous — works the same whether the declarator is on its own line or shares
`[buffer]`'s own line):
```
demo:1: [buffer]: shorthand declares 2 members (vec2 a[], vec2 b[]) -- a buffer block has exactly one name, so multiple declarators here are ambiguous; give each its own block, or use the struct form: '[buffer(...)]
struct Name { ... };'
```

**A struct sharing the attribute's own line** — rejected for every declaration attribute,
not just `[buffer]`, so a same-line struct never gets silently mis-scanned:
```
demo:1: [buffer]: the struct must be on its own line after this attribute, not on the same line as [buffer] (found 'struct X { vec3 a; };')
```

**Duplicate block name where one side was derived by the shorthand** — names both the
derived block name and the member it came from, since the block name never appears
literally in the shorthand's own source line:
```
demo:5: interface 'PtcPositions' is declared twice in this module (first at demo:3 (derived from [buffer] member 'ptcPositions'), again at demo:5)
```

**Invalid `name=` override:**
```
demo:1: [buffer(name=...)]: '123bad' is not a valid GLSL block name -- give a valid identifier
```

## `[program(...)]` errors

**Compute entry point named as a raster stage:**
```
demo:2: Program 'default': vert='cs_x' is in the vert slot, but 'cs_x' is declared [shader('comp')], not vert
```

**`tesc` without `tese`** (GL requires both or neither — `tese` alone is fine):
```
demo:2: Program 'default' declares a tesc stage without a tese stage; GL requires both or neither (tese alone is fine, tesc alone is a link error)
```

**A raster stage function no `[program(...)]` references** — a warning, not a raised
error, even under `strict=True`; the function is compiled then discarded:
```
demo:7: function 'vs_main' has stage 'vert' but is not referenced by any [program(...)] -- likely a typo or a stale entry point
```

## Binding errors

**`bind_ssbo`/`bind_ubo` on a name that was stripped as unreferenced, or never
existed:**
```
cs_noop: 'Unused' is not a valid buffer block (Missing binding)
cs_x: 'Data' is not a valid uniform block (Missing binding)
```

**Dispatching with a required storage block never bound on this kernel.** The block is
in `kernel.bindings`, so the code reachable from `main()` touches it, but nothing bound
it — the kernel would otherwise read whatever another kernel left at that index:
```
cs_go: Kernel 'cs_go' dispatched with required buffer(s) never bound: B (bind them first, or pass allow_unbound={...})
```
Fix by binding it — usually `kernel.bind(**buffers)`, which derives the set for you.
`dispatch(..., allow_unbound={'Name'})` opts out deliberately.

**A `TempHandle` recorded on a kernel was freed back to the `BufferPool`** before the
dispatch that would have used it. The pool recycles the underlying buffer, so this would
otherwise dispatch against memory someone else now owns:
```
cs_go: Kernel 'cs_go': buffer bound to 'A' has been freed (recycled temp buffer) -- re-bind before dispatching
```

**`kernel.bind(...)` missing a block the kernel requires** (extras are ignored; a
required name is not):
```
cs_go: Kernel 'cs_go' requires buffer 'B' but it was not in the provided buffers
```

**`set_uniform` on a name that isn't a uniform** (e.g. it's actually a buffer block
member, or misspelled):
```
cs_x: 'Data' is not a uniform
```

**A stage over its per-kind block limit** (`GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS`
etc., read from the driver, not hardcoded — this GPU reports 16) — names the
artifact, the count, and every block name:
```
cs_overflow: Artifact 'cs_overflow': comp stage references 17 SSBO blocks ['Blk0', 'Blk1', 'Blk10', 'Blk11', 'Blk12', 'Blk13', 'Blk14', 'Blk15', 'Blk16', 'Blk2', 'Blk3', 'Blk4', 'Blk5', 'Blk6', 'Blk7', 'Blk8', 'Blk9'] but the driver allows only 16 (GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)
```

## Atomic counter binding errors

Atomic counters (`uniform atomic_uint x;`) are a 2-D pool -- `(binding, offset)`, not a single
index -- because GL is designed to pack several counters into one binding at successive 4-byte
offsets (`layout(binding=0, offset=0)` / `layout(binding=0, offset=4)` is the idiomatic, intended
form, not a conflict). Unlike SSBO/UBO blocks or sampler/image uniforms, an atomic counter is
completely invisible to moderngl's own reflection, so an unpinned declaration is patched into the
generated GLSL as `layout(binding = N, offset = M)` -- there is no post-link `.value` to assign
the way there is for a sampler.

**Two counters explicitly pinned to the same `(binding, offset)`** (sharing one binding at
*different* offsets, the idiomatic form above, is never an error):
```
cs_conflict: Artifact 'cs_conflict': atomic counters 'a' and 'b' are both explicitly bound to binding=1, offset=0
```

**An explicit `binding=` pin past the driver's binding-index ceiling** (`GL_MAX_ATOMIC_COUNTER_
BUFFER_BINDINGS` is not reported at all on this GPU/driver, so this is the fallback of 8):
```
cs_ceil: Artifact 'cs_ceil': binding 10 on atomic counter 'a' exceeds the driver's binding-index ceiling (8, from GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (not reported, using fallback))
```

**The binding pool is exhausted** while packing an unpinned counter (every binding up to the
ceiling is already claimed, by a pin or by an earlier unpinned counter's own packed binding):
```
cs_exhaust: Artifact 'cs_exhaust': out of atomic counter bindings while assigning 'overflow' (binding-index ceiling 8, from GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (not reported, using fallback))
```

**`bind_counter`/`bind_counters` on a name that was never declared, or was pruned as
declared-but-genuinely-unused** (see `kernel.atomic_counters` -- pruning is scoped to the raw
`GL_ATOMIC_COUNTER_BUFFER` bindings the linked program's own program-interface query reports
active, since reflection can't see individual counter names at all):
```
cs_typo: 'cc' is not a declared atomic counter uniform
```

**Two counters sharing one binding were bound (via `bind_counter`) to different buffers or
different range offsets** -- GL has exactly one bound range per binding, so every name sharing a
binding must agree on where that range starts:
```
cs_share: Kernel 'cs_share': atomic counters sharing binding 0 were bound to different buffers/offsets -- bind every counter that shares one binding to the same buffer and the same range offset
```

**Unlike every other binding pool, dispatching with a declared-and-required counter never bound
through `bind_counter` does NOT raise.** This is a deliberate divergence from `bind_ssbo`/
`bind_texture`/`bind_image`'s "never bound" error: a real, driver-verified consumer of this
feature binds its one global atomic counter buffer exactly once, via a raw `glBindBufferRange`
call made entirely outside any `Kernel`, and never rebinds it again for the life of the process --
correct, idiomatic usage for a resource that (unlike an SSBO) is not swapped to a different buffer
between kernels or frames. `bind_counter` and its re-assert-at-dispatch discipline still exist and
still protect a caller who opts in by calling it at all.

## Missing `[export()]` / misattached `[link]`

A module-scope helper is emitted into an entry point's translation unit only if it is
`[export()]`ed (module-wide) or named by `[link('helper')]` **on the entry point**. A
helper with neither is emitted nowhere, and calling it used to fail as a raw driver
`error C1503: undefined variable "boundaryFriction"` at a generated-GLSL line number.
tlang now names it:

```
demo:3: 'cs_go' calls 'helper', a module-scope function defined in 'demo' but never emitted into this translation unit. Fix: add [export()] to 'helper', or [link('helper')] on 'cs_go'.
```

Across modules, `[link]` cannot help — only `[export()]` can:
```
demo:4: 'cs_go' calls 'scale', a module-scope function defined in included module 'lib' but never exported, so it was never emitted into this translation unit. [link(...)] can't reach across modules -- fix: add [export()] to 'scale' in 'lib'.
```

**`[link]` decorates the caller and names the helper.** Attached the other way round it
is silently inert — the helper has no stage, so nothing ever visits its link. tlang
detects that shape and says so:
```
demo:4: 'cs_go' calls 'helper', a module-scope function defined in 'demo' but never emitted into this translation unit. Note: 'helper' itself carries a [link(...)] -- if that was meant to pull 'helper' into 'cs_go', [link(...)] belongs on 'cs_go', not on 'helper'. Fix: add [export()] to 'helper', or [link('helper')] on 'cs_go'.
```

## Extension groups — what `[extend(...)]` actually enables

`[extend(name)]` / `[require(name)]` take either a raw `GL_*` extension or one of tlang's
group aliases. The groups expand to exactly this, and nothing else — so "I asked for the
`int64` group and int64 atomics still don't resolve" is a checkable statement rather than
a dead end:

| group | expands to |
|---|---|
| `int64` | `GL_ARB_gpu_shader_int64`, `GL_EXT_shader_atomic_int64`, `GL_KHR_shader_atomic_int64`, `GL_NV_shader_atomic_int64`, `GL_NV_gpu_shader5` |
| `subgroup` | `GL_KHR_shader_subgroup_basic`, `GL_KHR_shader_subgroup_vote`, `GL_KHR_shader_subgroup_ballot`, `GL_KHR_shader_subgroup_arithmetic` |
| `subgroup_all` | `GL_KHR_shader_subgroup_basic`, `GL_KHR_shader_subgroup_vote`, `GL_KHR_shader_subgroup_ballot`, `GL_KHR_shader_subgroup_arithmetic`, `GL_KHR_shader_subgroup_shuffle`, `GL_KHR_shader_subgroup_shuffle_relative`, `GL_KHR_shader_subgroup_clustered`, `GL_KHR_shader_subgroup_quad` |
| `vulkan_glsl` | `GL_KHR_vulkan_glsl` |

**`GL_NV_gpu_shader5` is in the `int64` group deliberately, and is load-bearing.** On
NVIDIA the 64-bit atomic *builtin overloads* are gated behind it — NV's int64 **type**
extension — not behind the extension whose name says "atomic_int64". Without it,
`atomicCompSwap` on a `uint64_t` fails with a message that blames the hardware rather
than the extension list:

```
error C1115: unable to find compatible overloaded function "atomicCompSwap(uint64_t, u64vec2)"
```

Verified on RTX 3090 / NVIDIA 616.64 / GL 4.6: the group without `GL_NV_gpu_shader5`
fails that shader, and with it compiles. If you see C1115 on a 64-bit atomic, check the
emitted `#extension` lines before concluding the GPU can't do it.

## Dependency/template errors

**Circular `[include]`:**
```
a: Circular dependency detected at 'a'
```

**Missing `[include]` target:**
```
a: Missing dependency: 'nope'
```

**Duplicate module name** (two files map to the same dotted name):
```
a.b: Duplicate module name 'a.b' (from '<path>\dup_module\a\b.tlang')
```

**Shader directory does not exist:**
```
<path>: Shader directory does not exist: '<path>'
```

**Shader directory has no `.tlang` files:**
```
<path>: No '.tlang' files found under '<path>'
```

**Missing template constant** — every `{{ NAME }}` needs a matching key in
`constants`:
```
demo:3: Undefined template constant in 'demo': 'MISSING' is undefined
```

**`{{ }}` collides with a GLSL brace initializer** (`mat2({{1.0, 0.0}, {0.0, 1.0}})`
parses as Jinja, not GLSL) — fix with a space (`{ {1.0, 0.0}, ... }`) or constructor
form:
```
demo:4: Template syntax error while rendering 'demo': unexpected '}'
```

## Compile / link errors

**A genuine GLSL compile error** — carries `.stage`, `.entry_point`, `.source` (the
exact GLSL handed to the driver, with `#line` directives):
```
demo:4: Failed to compile comp shader 'cs_bad': GLSL Compiler failed

compute_shader
==============
demo(4) : error C0000: syntax error, unexpected ';', expecting "::" at token ";"
```

**A `[program(...)]` link failure** (e.g. a raw `[glsl(...)]` on each stage
declaring the same varying name with mismatched types — the struct/`[uses(...)]`
form catches this before it ever reaches the driver; raw declarations don't get that
check):
```
demo: Failed to link program 'default': GLSL Linker failed

Program
=======
Link info
---------
error: Type mismatch between variables of same name "vcolor"
```

Under `strict=False`, every one of the above becomes a logged warning instead of a
raised exception, and the build continues with that kernel/program simply absent —
which then surfaces later as a `KeyError` on `get_kernel`/`get_program`/`get_pipeline`.

## BufferPool errors

**Use of a `TempHandle` after it was freed:**
```
Use of buffer handle after free: 'read' accessed on a freed temp buffer
```

**Freeing the same handle twice:**
```
Buffer handle already freed (double free)
```
