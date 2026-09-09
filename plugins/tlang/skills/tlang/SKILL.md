---
name: tlang
description: Write, debug, and build .tlang shader files — a GLSL preprocessor for Python/ModernGL that keeps every pipeline stage in one file using [shader(...)] attributes, struct-declared varyings/uniforms/buffers matched across stages by name, cross-file [include]/[export], and {{ CONSTANT }} templating. Use this whenever you encounter a .tlang file, the tlang package, ShaderManager/Shader/Kernel/Pipeline/BufferPool, or are asked to write, fix, or compile GLSL compute/vertex/fragment/geometry/tessellation shaders in a project that has tlang as a dependency — including when the user just says "the shader" or "the compute kernel" and the repo contains .tlang files.
---

# tlang

tlang is a GLSL preprocessor, not a new language. A `.tlang` file is ordinary GLSL
plus attributes in `[...]`. One file holds every stage of a pipeline; tlang compiles
it into ModernGL `Program`/`ComputeShader` objects at runtime, in-process. **No CLI,
no file output** — everything happens through `ShaderManager`.

Each `[shader(...)]` function becomes its own GLSL translation unit: tlang emits the
shared module code, then that function renamed to `main`. Other stages' functions are
excluded, so stages never collide.

## Build a project

```python
import moderngl as mgl
from tlang import ShaderManager

ctx = mgl.create_context(require=460, standalone=True)   # a GL context must exist first

sm = ShaderManager(
    ctx=ctx,                  # NOTE: param is `ctx`, not `context`
    version='460 core',
    dir='shaders',            # resolves against the CALLING FILE's dir, not cwd
    constants={'BLOCK_SIZE': 256},
    strict=True,              # default: raise on any compile/link/attribute failure
)

shader   = sm.get_shader('demo')           # -> Shader | None
kernel   = shader.get_kernel('cs_go')      # -> Kernel        (KeyError if absent)
pipeline = shader.get_pipeline('default')  # -> Pipeline      (prefer over get_program)
```

Module names are the file's path relative to `dir`, dotted, no extension:
`shaders/fx/blur.tlang` -> `'fx.blur'`. That's what `[include(...)]` and `get_shader()` take.

## Complete example (vertex + fragment, structured interfaces)

Verified to build. This is the pattern to reach for by default — declare each
resource once as a struct, reference it by name from each stage:

```glsl
// Vertex attributes from the vertex buffer. No measurable location span issue
// here since every member is a plain vec3 — locations=false is NOT needed.
[varyings]
struct MeshVertex { vec3 position; vec3 normal; };

// Varyings passed vertex -> fragment. Declared once; both [uses(...)] below
// reference it, so layout(location=N) matches by construction.
[varyings]
struct VertexOut { vec3 color; vec2 uv; };

// Loose uniforms (default form). Set with pipeline.set_uniforms(time=...).
[uniforms]
struct Frame { float time; };

[program('default', vert='vs_main', frag='fs_main')]   // required to link a raster stage

[shader('vertex')]
[uses(MeshVertex, dir='in')]     // vertex attributes in
[uses(VertexOut,  dir='out')]    // varyings out
void vs_main() {
    color = normal * 0.5 + 0.5;
    uv = position.xy;
    gl_Position = vec4(position + vec3(0.0, 0.0, sin(time)), 1.0);
}

[shader('fragment')]
[uses(VertexOut, dir='in')]      // SAME declaration -> matches by construction
[resourceblock(                  // escape hatch for anything structs can't express
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = vec4(color, uv.x);
}
```

A compute kernel needs no `[program(...)]` — it's dispatched directly:

```glsl
layout(std430) buffer Data { uint data[]; };   // binding auto-assigned

[shader('compute'), numthreads(256, 1, 1)]
void cs_go() {
    uint gid = gl_GlobalInvocationID.x;
    data[gid] = data[gid] + 1u;
}
```

```python
kernel = shader.get_kernel('cs_go')
kernel.bind_ssbo('Data', buf)                 # bind BY NAME, never a hardcoded number
kernel.dispatch((n + 255) // 256)             # workgroup counts, not thread counts
```

## Rules that prevent the frequent mistakes

- **No type system.** Whether two stages' interfaces "match" is a text comparison
  (member order, qualifiers, type name, array suffix) — never describe or reason about
  it as type checking.
- **Locations come from member order.** `mat4` consumes 4 locations, `vec3` consumes 1.
  Reordering or inserting a member shifts every location after it. Use
  `member_locations(decl)` (importable from `tlang`) instead of computing by hand.
- **`[uses(name, dir=...)]`** only names a `[varyings]` declaration — never
  `[uniforms]`/`[buffer]` — and takes exactly one `dir='in'` and one `dir='out'` per
  stage function. A second `dir='out'` on the same function is a build error.
- **The struct must sit immediately after its declaration attribute** — nothing between
  `[varyings]`/`[uniforms(...)]`/`[buffer(...)]` and `struct Name { ... };`.
- **`[varyings(locations=false)]`** is required, not optional, for a member with no
  measurable location span: an unsized array, or an array sized by `{{ CONSTANT }}`
  rather than an integer literal. Leaving it out is a build-time error naming the
  member.
- **Geometry/tess stages array their per-vertex `in` (and tess-control `out`)
  interfaces automatically** — `[uses(..., dir='in')]` emits `in vec3 color[];`, you
  index `color[i]`. Do not add your own `[]`.
- **`[resourceblock(...)]` still works and is not deprecated** — it's the escape hatch
  for interface blocks, mixed-direction resources, and anything the struct form can't
  express. Don't "migrate" existing `[resourceblock(...)]` usage unasked.
- **Raw GLSL declarations (`layout(std430) buffer X {...};`, plain `in`/`out`) still
  work exactly as before.** The struct form is an alternative, never required — mix
  freely, migrate one stage at a time or never.
- **Unused SSBO/UBO blocks are stripped per artifact.** An entry point that doesn't
  reference a block loses it entirely — `bind_ssbo`/`bind_ubo` on a stripped name
  raises `TlangBindingError` at runtime, not at build time.
- **`{{ }}` collides with GLSL brace initializers** (`mat2({{1.0,0.0},{0.0,1.0}})`)
  — add a space or use constructor form.
- **A raster stage function with no `[program(...)]` reference is compiled and
  discarded with a warning** — usually a typo. Compute entry points never go in
  `[program(...)]`.

## References

| Question | Read |
|---|---|
| Every attribute, its args, valid stages | `references/attributes.md` (generated — do not hand-edit) |
| Declaring varyings/uniforms/buffers as structs, locations, `[uses]`, `[resourceblock]`, reflection | `references/resources.md` |
| `ShaderManager`, `Shader`, `Kernel`, `Pipeline`, `BufferPool`, dispatch, binding by name | `references/runtime.md` |
| An exact error message and what to do about it | `references/errors.md` |
| Full working shaders to copy: compute kernel, vert+frag with structured varyings, cross-module include/export, geometry stage | `references/patterns.md` |

Requires `moderngl>=5.8`. A GL context must exist before any tlang call. There is no
CLI — verify a change by building it: `sm = ShaderManager(...)`; a successful
construction with `strict=True` (the default) means the whole pipeline compiled and
linked. Prefer this over reading GLSL and reasoning about whether it should work.
