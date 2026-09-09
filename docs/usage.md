# Using tlang

A practical guide to building and running shaders with tlang. For the exhaustive
attribute list see [attribute-reference.md](attribute-reference.md); for how the
compiler works internally see [architecture.md](architecture.md).

## Contents

- [Setting up](#setting-up)
- [Writing a .tlang file](#writing-a-tlang-file)
- [Modules: include and export](#modules-include-and-export)
- [Programs and kernels](#programs-and-kernels)
- [Buffers and bindings](#buffers-and-bindings)
- [Declaring resources as structs](#declaring-resources-as-structs)
- [Dispatching compute](#dispatching-compute)
- [Templating with constants](#templating-with-constants)
- [Transient buffers](#transient-buffers)
- [Errors and debugging](#errors-and-debugging)
- [Known limitations](#known-limitations)

## Setting up

tlang needs a live ModernGL context before anything else — it compiles and links
through the driver at build time.

```python
import moderngl as mgl
from tlang import ShaderManager

ctx = mgl.create_context(require=460, standalone=True)   # or your window's context

sm = ShaderManager(
    ctx=ctx,                       # the parameter is `ctx`, not `context`
    version='460 core',            # emitted as `#version 460 core`
    dir='shaders',                 # see path resolution below
    constants={'BLOCK_SIZE': 256},
    strict=True,                   # raise on any build failure (default)
)
```

**Path resolution.** A relative `dir` resolves against **the directory of the file that
constructed the `ShaderManager`**, not the process working directory. This lets a
library ship shaders next to its own source and keep working regardless of where the
application was launched from. Absolute paths are used as-is. A missing directory, or
one containing no `.tlang` files, raises `TlangDependencyError` rather than silently
building nothing.

**Module names** are the file's path relative to `dir`, dotted, without the extension:

| File | Module name |
|---|---|
| `shaders/demo.tlang` | `demo` |
| `shaders/fx/blur.tlang` | `fx.blur` |

That dotted name is what `[include(...)]` and `get_shader()` take. Two files that map to
the same module name (`a/b.tlang` and `a.b.tlang`) are rejected.

**`strict`.** With `strict=True` (the default) any compile, link, attribute or pipeline
error raises. `strict=False` downgrades them to log messages and continues with a
partial build — useful when iterating, risky in production, since a missing kernel then
surfaces much later as a `KeyError`.

## Writing a .tlang file

A `.tlang` file is GLSL. Attributes in square brackets add structure on top:

```glsl
[shader('compute'), numthreads(256, 1, 1)]
void cs_accumulate() {
    uint gid = gl_GlobalInvocationID.x;
    totals[gid] += 1u;
}
```

Attributes are **positional**: a global-scope attribute attaches to the *next* function
in the file. Blank lines and comments between them are fine; another function in between
is not.

Several attributes can share one bracket, comma-separated, or sit on their own lines —
`[shader('compute'), numthreads(256,1,1)]` and

```glsl
[shader('compute')]
[numthreads(256, 1, 1)]
```

are equivalent.

Each entry point becomes its own GLSL translation unit: tlang emits the shared module
code, then that one function renamed to `main`. Functions belonging to other stages are
excluded, so stages don't collide.

## Modules: include and export

Sharing code across files needs two attributes, and you need both:

```glsl
// shaders/math.tlang
[export]
uint add(uint x, uint y) { return x + y; }
```

```glsl
// shaders/demo.tlang
[include(math)]

[shader('compute'), numthreads(64, 1, 1)]
void cs_go() { data[0] = add(1u, 2u); }
```

- `[include(module)]` goes in the **consumer** and makes the module available.
- `[export]` goes on each **helper function** in the provider.

A function without `[export]` is private to its file. Global-scope declarations —
buffers, constants, `#define`s — cross module boundaries with `[include]` alone;
`[export]` is only needed for functions.

Includes resolve **transitively and topologically**. If `a` includes `b` and `b`
includes `c`, then `a` sees `c`'s exports *and* `c`'s `[extend(...)]` extensions.
Circular includes raise `TlangDependencyError`.

There is also `[link(fn)]`, which inlines a specific helper's body into the stage that
names it — useful for a helper you want in one entry point without exporting it
module-wide. A function may be both `[export]`ed and `[link]`ed; it is still emitted
exactly once.

## Programs and kernels

Two different things come out of a build:

**Kernels** are compute entry points. They need no declaration beyond
`[shader('compute')]` — they are dispatched directly.

**Programs** are raster pipelines and must be declared:

```glsl
[program('default', vert='vs_main', frag='fs_main')]
```

Stage keywords accept aliases (`vert`/`vertex`, `frag`/`fragment`, `geom`/`geometry`,
`tesc`/`tess_control`, `tese`/`tess_eval`). Program declarations are validated at build
time — an unknown keyword, two aliases for the same stage, a missing entry point, a
compute function named as a raster stage, or `tesc` without `tese` are all reported with
the specific problem rather than an opaque driver error. All problems across all
programs in a file are reported together.

A raster stage function that no `[program(...)]` references is compiled and discarded,
with a warning — usually a typo or a stale entry point.

```python
shader   = sm.get_shader('demo')       # -> Shader | None
kernel   = shader.get_kernel('cs_go')  # -> Kernel        (KeyError if absent)
program  = shader.get_program('default')   # -> moderngl.Program
pipeline = shader.get_pipeline('default')  # -> Pipeline (name-keyed binding helpers)
```

`Pipeline` wraps a program with the same ergonomics as `Kernel` — `bind_ssbo`,
`bind_ssbos`, `set_uniform`, `set_uniforms`, `bindings` — so graphics code never needs
a hardcoded binding number either.

## Buffers and bindings

Declare an SSBO normally; tlang assigns the binding point:

```glsl
layout(std430) buffer Particles { vec4 pos[]; };             // auto-assigned
layout(binding = 3, std430) buffer Fixed { uint counts[]; };  // pinned by you
```

Bindings are allocated **per compiled artifact** — each compute kernel, and each
`[program(...)]` as a unit across its stages — from only the blocks that artifact
actually references. Explicit pins are reserved first and never moved.

Always bind **by block name**:

```python
kernel.bind_ssbo('Particles', buf)
kernel.bind_ssbos(Particles=buf, Fixed=(buf2, 0, 1024))   # (buffer, offset, size)
print(dict(kernel.bindings))   # the name -> binding map, for debugging
```

Binding numbers are an implementation detail that can change between builds. Code that
calls `buffer.bind_to_storage_buffer(3)` with a literal will break; `bind_ssbo('Name', ...)`
will not.

**Unused blocks are removed.** tlang strips buffer blocks an entry point does not
reference. This is not an optimisation but a necessity: the per-stage block limit is far
lower than the binding-index limit. On a typical card
`GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` is 96 while
`GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS` is **16**, and exceeding the latter is a hard link
failure. Include-heavy projects depend on this stripping to fit.

If `bind_ssbo('Foo', ...)` raises `TlangBindingError: 'Foo' is not a valid buffer block`,
the usual cause is that no reachable code in that entry point reads or writes `Foo`, so
it was stripped. Exceeding the per-stage limit raises a `TlangBindingError` naming the
artifact and its blocks.

## Declaring resources as structs

Varyings, uniforms and SSBOs can be declared once, as a struct, and referenced by name
instead of hand-written and duplicated across stages:

```glsl
[varyings]
struct VertexOut { vec3 color; vec2 uv; };

[shader('vertex')]
[uses(VertexOut, dir='out')]
void vs_main() {
    color = vec3(1.0, 0.0, 0.0);
    uv = vec2(0.0, 0.0);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

[shader('fragment')]
[uses(VertexOut, dir='in')]
[resourceblock(out vec4 fragColor;)]
void fs_main() {
    fragColor = vec4(color, 1.0);
}

[program('default', vert='vs_main', frag='fs_main')]
```

Both stages reference the same declaration, so `layout(location = N)` is assigned once
and matches on both sides by construction — a name mismatch is a build-time error
instead of a driver link failure (see below).

This is an alternative syntax, not a replacement. Raw GLSL declarations keep working
exactly as before, and the two can be mixed freely — migrate one stage at a time, or
never.

**The four declaration attributes:**

| Attribute | Declares | Emits |
|---|---|---|
| `[varyings]` | a stage-to-stage interface | nothing at module scope — only through `[uses(...)]`, per stage (see below) |
| `[uniforms]` | loose uniforms (default) | `uniform <type> <name>;` per member |
| `[uniforms(std140)]` | a UBO block | `layout(std140) uniform Name { ... };` |
| `[buffer(std430)]` (or `std140`) | an SSBO block | `layout(std430) buffer Name { ... };` |

`[uniforms]`, `[uniforms(std140)]` and `[buffer(...)]` desugar to ordinary module-scope
GLSL as soon as the attribute is attached — everything downstream (binding assignment,
dead-block stripping, `#line` accounting) sees exactly the text it would have seen had
you written the raw form. An unreferenced interface of any of these three kinds is
simply stripped, the same as an unreferenced raw `buffer`/`uniform` block.

`[varyings]` is different: it has no single correct module-scope form (the same
declaration is `out` in the vertex stage and `in` in the fragment stage), so it emits
nothing until a stage brings it in with `[uses(...)]`.

**`[uses(Name, dir='in'|'out')]`** ties one stage function to one declared `[varyings]`
interface, in one direction:

```glsl
[shader('vertex')]
[uses(MeshVertex, dir='in')]     // vertex attributes
[uses(VertexOut,  dir='out')]    // varyings out
void vs_main() { ... }
```

A stage may reference at most one interface per direction — a second `[uses(..., dir='out')]`
on the same function is a build error naming both interfaces and both lines. `[uses(...)]`
only accepts a `[varyings]` declaration; naming a `[uniforms]` or `[buffer]` struct is
rejected, naming the kind it actually is. Compute stages have no stage in/out to attach
to, so `[uses(...)]` on a `[shader('compute')]` function is also an error.

**Locations come from member order, not an allocator.** The first member gets location
0; each subsequent member's location is the previous one plus how many location slots
the previous member's type consumes. A `vec3` or `mat2` each take a different number of
slots, so inserting or reordering a member shifts every location after it:

```glsl
[varyings]
struct Shifted { mat4 xform; vec3 color; };   // mat4 -> 4 locations
```

```
>>> member_locations(shader.interfaces['Shifted'])
{'xform': 0, 'color': 4}
```

`shader.interfaces['Shifted']` returns the `InterfaceDecl`; `member_locations(decl)`
(also importable from `tlang`) reports the assignment tlang actually used, rather than
requiring you to work it out by hand. It returns `{}` when the interface can't be
located (see next).

**`[varyings(locations=false)]`** opts a declaration out of `layout(location = N)`
entirely, emitting plain `out`/`in` members that GL matches by name instead. This is
required, not optional, when a member's type has no measurable location span — an
unsized array, or a member whose array size is a `{{ CONSTANT }}` rather than a literal
digit at parse time:

```glsl
[varyings(locations=false)]
struct Batch { vec4 items[{{ N }}]; };
```

Leaving `locations=true` (the default) on a declaration like this is a build-time error
naming the offending member and its line, and pointing at `[varyings(locations=false)]`
as the fix.

**Geometry and tessellation stages array their per-vertex interfaces automatically.**
GLSL requires `in vec3 color[];` (not `in vec3 color;`) for a per-vertex input on these
stages — `[uses(..., dir='in')]` emits the arrayed form for you:

| stage | `in` arrayed? | `out` arrayed? |
|---|---|---|
| vertex | no | no |
| fragment | no | no |
| geometry | yes | no |
| tess control | yes | yes |
| tess evaluation | yes | no |

```glsl
[shader('geometry')]
[geom(in='triangles', out='triangle_strip', max_verts=3)]
[uses(VOut, dir='in')]
void gs_main() {
    for (int i = 0; i < 3; i++) {
        gColor = color[i];              // `color` is `in vec3 color[];` here
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
```

Arraying does not add a location — a per-vertex array of `vec3` still occupies one
location, the array dimension is separate from the location count.

**`[resourceblock(...)]` still works, unchanged.** It stays the escape hatch for
anything the struct form doesn't cover — an interface block, a resource whose members
mix directions, or GLSL the struct syntax has no way to express:

```glsl
[shader('fragment')]
[resourceblock(out vec4 fragColor;)]
void fs_main() { fragColor = vec4(1.0); }
```

**Cross-stage validation.** Because both stages of a `[program(...)]` reference the
declaration by name, a typo or a divergent struct is caught before the driver ever sees
the GLSL. A one-character typo in one stage's `[uses(...)]`:

```
demo:13: [uses('VertexOu')]: no interface named 'VertexOu' is declared or included. Did you mean 'VertexOut'?
demo: program 'default': vert stage 'vs_main' writes interface 'VertexOut' (declared demo:2) but frag stage 'fs_main' reads 'VertexOu' (used at demo:13). Did you mean 'VertexOut'?
```

Two modules declaring the same interface name with different members is also caught,
at whichever point the conflicting declarations are merged into one file's dependency
set — naming both modules, both lines, and the first differing member:

```
demo: 'VertexOut' is declared differently in two modules: b:2 has 'vec4 color' but a:2 has 'vec3 color'
```

Either way, `strict=False` downgrades these to logged warnings, like every other tlang
diagnostic — the message names the fix either way.

**`[uniforms(std140)]` bindings work like SSBO bindings, but are a separate pool** —
`GL_MAX_UNIFORM_BUFFER_BINDINGS` and the per-stage `GL_MAX_*_UNIFORM_BLOCKS` are
distinct from the SSBO limits described above. tlang allocates and strips dead UBO
blocks the same way it does SSBOs. Bind them by name, like buffers:

```python
kernel.bind_ubo('Params', ubo_buffer)
kernel.bind_ubos(Params=ubo_buffer, Lights=(lights_buf, 0, 256))   # (buffer, offset, size)
print(dict(kernel.uniform_blocks))   # block name -> binding, for debugging
```

`kernel.bindings` reports SSBO blocks only, unchanged; `kernel.uniform_blocks` is the
uniform-block equivalent. Both `Kernel` and `Pipeline` carry the same four methods.

A `std140` block needs its contents laid out to the std140 rules, which is why loose
`[uniforms]` (settable with `set_uniforms(...)`) is the default and the block form is
opt-in.

**Reflection.** Every module's declared interfaces are on the `Shader`:

```python
shader.interfaces               # Mapping[str, InterfaceDecl] -- this module + its includes
shader.interfaces['VertexOut']  # -> InterfaceDecl(name='VertexOut', kind=VARYINGS, members=(...), ...)
```

`InterfaceDecl`, `InterfaceKind`, `InterfaceMember`, `location_span` and
`member_locations` are all importable directly from `tlang`.

**tlang has no type system.** Whether two interfaces "match" is a comparison of
declared text (member order, qualifiers, type name, array suffix), not resolved GLSL
types — a `float` member and an `int` member of the same name are a mismatch by this
comparison the same as any other difference, and there is no coercion or aliasing
check beyond that.

## Dispatching compute

```python
kernel.dispatch(groups_x, groups_y, groups_z)   # issues a memory barrier by default
kernel.dispatch(n, barrier=False)               # skip when chaining dependent passes
kernel.dispatch_indirect(indirect_buffer, offset=0)
elapsed_ms = kernel.dispatch_timed(n)           # blocks on finish(); profiling only

kernel['threshold'] = 0.5
kernel.set_uniforms(threshold=0.5, count=n)
```

`dispatch` takes **workgroup counts, not thread counts**. With `[numthreads(256,1,1)]`,
covering `n` elements is:

```python
kernel.dispatch((n + 255) // 256)
```

Passing `n` directly launches 256× too many threads. Nothing errors — you just get
wrong results or an out-of-bounds write, so it is worth double-checking.

## Templating with constants

`{{ NAME }}` placeholders are substituted from the `constants` dict at build time,
everywhere in the file — global scope, function bodies, and attribute arguments:

```glsl
#define TILE {{ BLOCK_SIZE }}

[shader('compute'), numthreads({{ BLOCK_SIZE }}, 1, 1)]
void cs_go() {
    uint n = {{ BLOCK_SIZE }}u;
}
```

A `{{ NAME }}` with no matching key raises `TlangDependencyError` naming the constant
and its location, rather than emitting a bogus identifier.

Substitution is Jinja-based, which has one sharp edge — see the limitation below.

## Transient buffers

`BufferPool` recycles scratch GPU buffers. Prefer the scope guards: they return the
buffer even when the body raises, which a manual `free_temp` cannot promise.

```python
from tlang import BufferPool

pool = BufferPool(ctx)

with pool.temp(size_bytes, zero=True) as scratch:
    kernel.bind_ssbo('Scratch', scratch)
    kernel.dispatch(groups)

with pool.frame():                  # everything allocated inside is reclaimed at exit
    a = pool.alloc_temp(1024)
    b = pool.alloc_temp(4096)

globals_buf = pool.persistent_buffer('globals', size=256)   # named, never recycled

print(pool.metrics())   # bytes pooled/checked out, high-water mark, hit/miss counts
pool.trim()             # release idle capacity
```

**Recycled memory is undefined by default.** Pass `zero=True` when your shader assumes
zeroed scratch, or you will silently read the previous pass's data — and it will appear
to work until pool occupancy changes. `BufferPool(ctx, debug_poison=True)` fills recycled
buffers with `0xCD` so stale reads fail loudly during development.

Buffers are pooled in power-of-two size classes, so a small request can never consume a
much larger pooled buffer. Using a handle after it is freed raises `TlangError` rather
than corrupting a buffer that now belongs to something else.

## Errors and debugging

Every exception derives from `TlangError` and carries a `module:line` location:

| Exception | Raised when |
|---|---|
| `TlangSyntaxError` | malformed attribute, unbalanced brackets, unterminated function body, malformed `[varyings]`/`[uniforms]`/`[buffer]` struct |
| `TlangAttributeError` | unknown or misapplied attribute, bad arguments, invalid `[program(...)]`, unresolved or mismatched `[uses(...)]` interface |
| `TlangDependencyError` | missing module, circular `[include]`, duplicate module name, missing constant |
| `TlangBindingError` | binding conflict or exhaustion, unknown block name at runtime |
| `TlangCompileError` | a stage failed to compile — carries `.stage`, `.entry_point`, `.source` |
| `TlangLinkError` | a `[program(...)]` failed to link |

The generated GLSL is always reachable:

```python
from tlang import TlangCompileError

try:
    sm = ShaderManager(ctx=ctx, version='460 core', dir='shaders')
except TlangCompileError as e:
    print(e)          # demo:26: Failed to compile comp shader 'cs_go': ...
    print(e.source)   # the exact GLSL handed to the driver

print(shader.get_source('cs_go'))   # any successfully generated stage
```

Because tlang emits `#line` directives, driver messages reference your `.tlang` file and
line rather than generated output. When something is genuinely puzzling, dumping
`get_source(...)` is the fastest way through — it shows exactly what the driver saw,
including injected bindings and layout qualifiers.

tlang logs through the standard `logging` module under the `tlang.*` hierarchy:

```python
import logging
logging.getLogger('tlang').setLevel(logging.INFO)   # build timings, per-stage progress
```

## Known limitations

**`{{ }}` collides with GLSL brace initializers.** Templating is Jinja-based, so legal
GLSL can break the build:

```glsl
mat2 m = mat2({{1.0, 0.0}, {0.0, 1.0}});   // TemplateSyntaxError
```

Add a space (`{ {1.0, 0.0}, ... }`) or use constructor form. Nested brace initializers
are the only common construct affected. Moving to `${NAME}` delimiters — `$` is not a
valid GLSL character — is on the roadmap.

**Buffer liveness analysis is textual, not a GLSL parse.** It is deliberately biased
toward keeping blocks (a wrongly-kept block costs one binding slot; a wrongly-removed one
is a confusing compile error), but an unrelated identifier of the same name elsewhere in
the file will keep a block alive.

**One `constants` set per `ShaderManager`.** Building the same shaders with different
constants means constructing a second manager, which rebuilds everything. Shader variants
are on the roadmap.

**A stage references one `[varyings]` interface per direction.** A vertex stage can
`[uses(..., dir='in')]` one interface and `[uses(..., dir='out')]` one interface, no
more — there's no way to compose two structs into one stage's inputs short of putting
every member in a single struct.

**No interface-block form for `[varyings]` yet.** `[varyings]` always desugars to flat,
individually-located members, never a GLSL interface block (`out Block { ... } name;`).
Flat covers all three directions with one mechanism and lets you migrate one stage at a
time; block emission is future work, not a current option. `[uniforms(std140)]` and
`[buffer(...)]` do emit block form, since blocks are legal (and required) there.

**Only GLSL's builtin scalar/vector/matrix types can be auto-located.** `location_span`
is a fixed lookup table over `bool`/`float`/`int`/`uint`/`vec*`/`ivec*`/`uvec*`/`bvec*`/
`double`/`dvec*`/`mat*`/`dmat*` and their literal-sized arrays. A member of a user
struct type, or an array sized by anything other than an integer literal (including a
`{{ CONSTANT }}`), has no measurable span — declare that interface with
`[varyings(locations=false)]`, or it's a build-time error.
