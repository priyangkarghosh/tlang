# Declaring resources: varyings, uniforms, buffers

Three attributes declare a resource once, as a struct, for reference by name from any
stage. All three desugar to plain GLSL text before anything downstream (binding
assignment, dead-block stripping, `#line` accounting) runs — an unreferenced one is
stripped exactly like a hand-written `buffer`/`uniform` block.

| Attribute | Declares | Emits |
|---|---|---|
| `[varyings]` | stage-to-stage interface (also used for vertex attributes) | nothing at module scope — only through `[uses(...)]`, per stage |
| `[uniforms]` | loose uniforms (default) | `uniform <type> <name>;` per member |
| `[uniforms(std140)]` | a UBO block | `layout(std140) uniform Name { ... };` |
| `[buffer(std430)]` (or `std140`) | an SSBO block | `layout(std430) buffer Name { ... };` |

```glsl
[varyings]
struct VertexOut { vec3 color; vec2 uv; };

[uniforms]
struct Frame { mat4 view; float time; };

[uniforms(std140)]
struct Lights { vec4 positions[16]; };

[buffer(std430)]
struct Particles { vec4 pos[]; };
```

The struct must be on the line(s) **immediately after** the declaration attribute —
nothing between them. It takes no instance name (`struct Name { ... };`, not
`struct Name { ... } inst;`) — that's a `TlangSyntaxError` if violated.

`[uniforms]`/`[uniforms(std140)]`/`[buffer(...)]` are unconditional, unambiguous GLSL —
they mean the same thing regardless of which stage reads them, so they need no
direction and are visible at module scope to any stage that references a member.
`[varyings]` is different: the same declaration is `out` in the vertex stage and `in`
in the fragment stage, so it has no single correct module-scope form and emits nothing
until a stage brings it in with `[uses(...)]`.

## `[uses(Name, dir='in'|'out')]`

Ties one stage function to one declared `[varyings]` interface, in one direction:

```glsl
[shader('vertex')]
[uses(MeshVertex, dir='in')]     // vertex attributes
[uses(VertexOut,  dir='out')]    // varyings out
void vs_main() { ... }
```

- One interface per direction per stage. A second `[uses(..., dir='out')]` on the same
  function is a build error naming both interfaces and both lines — not a silent
  overwrite.
- Only accepts a `[varyings]` declaration. Naming a `[uniforms]`/`[buffer]` struct is
  rejected, and the message names the kind it actually is.
- Compute stages have no in/out to attach to — `[uses(...)]` on a `[shader('compute')]`
  function is a build error.
- `[varyings]` used for vertex attributes (`dir='in'` on the vertex stage) works the
  same as any other varyings interface; there's no separate "attributes" attribute.

## Locations

The first member gets location 0; each subsequent member's location is the previous
one plus however many location slots the previous member's type consumes. `vec3` and
`mat2` consume different numbers of slots (a `matN` consumes `N`), so **inserting or
reordering a member shifts every location after it**:

```glsl
[varyings]
struct Shifted { mat4 xform; vec3 color; };   // mat4 -> 4 locations
```

```python
>>> member_locations(shader.interfaces['Shifted'])
{'xform': 0, 'color': 4}
```

`shader.interfaces['Shifted']` is the `InterfaceDecl`; `member_locations(decl)`
(importable from `tlang`) reports the assignment tlang actually used — use it instead
of computing by hand. Returns `{}` when the interface can't be located or opts out of
locations (see next).

**`location_span` only knows GLSL's builtin scalar/vector/matrix types** —
`bool`/`float`/`int`/`uint`/`vec*`/`ivec*`/`uvec*`/`bvec*`/`double`/`dvec*`/`mat*`/
`dmat*` and their literal-sized arrays. Anything else (a user struct member, an unsized
array, an array sized by `{{ CONSTANT }}` rather than a literal digit) has no
measurable span.

**`[varyings(locations=false)]`** is the required opt-out for that case — it emits
plain `out`/`in` members with no `layout(location=N)`, matched by name instead:

```glsl
[varyings(locations=false)]
struct Batch { vec4 items[{{ N }}]; };
```

Leaving `locations=true` (the default) on a declaration like this is a build-time
error naming the offending member and its line:

```
demo:3: interface 'Batch' member 'vec4 items[{{ N }}]' has no measurable location span; add [varyings(locations=false)] or give the array a literal size
```

Note: interface parsing runs *before* the `{{ CONSTANT }}` substitution pass, so a
`{{ N }}`-sized array is literally unmeasurable text at the point tlang checks it —
this is true even though `N` will resolve to a concrete integer later. Always use
`locations=false` for a `{{ CONSTANT }}`-sized array member.

## Geometry and tessellation: automatic arraying

GLSL requires `in vec3 color[];` (not `in vec3 color;`) for a per-vertex input on
these stages. `[uses(..., dir=...)]` emits the arrayed form automatically — you index
the member, you never write your own `[]`:

| stage | `in` arrayed? | `out` arrayed? |
|---|---|---|
| vertex | no | no |
| fragment | no | no |
| geometry | yes | no |
| tess control | yes | yes |
| tess evaluation | yes | no |

```glsl
[varyings]
struct VOut { vec3 color; };

[varyings]
struct GOut { vec3 gcolor; };

[program('default', vert='vs_main', geom='gs_main', frag='fs_main')]

[shader('vertex')]
[uses(VOut, dir='out')]
void vs_main() {
    color = vec3(1.0, 0.0, 0.0);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

[shader('geometry')]
[geom(in='triangles', out='triangle_strip', max_verts=3)]
[uses(VOut, dir='in')]
[uses(GOut, dir='out')]
void gs_main() {
    for (int i = 0; i < 3; i++) {
        gcolor = color[i];              // `color` is `in vec3 color[];` here
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}

[shader('fragment')]
[uses(GOut, dir='in')]
[resourceblock( out vec4 fragColor; )]
void fs_main() { fragColor = vec4(gcolor, 1.0); }
```

The geometry stage emits `layout(location = 0) in vec3 color[];` and
`layout(location = 0) out vec3 gcolor;` — the input is arrayed, the output is not.

Arraying does not add a location — a per-vertex array of `vec3` still occupies one
location; the array dimension and the location count are independent. A member that
*already* declares its own array (`vec3 color[4]`) cannot also be arrayed by the
stage — that's a build error pointing at `[resourceblock(...)]` as the escape hatch
for an array-of-arrays interface.

## `[resourceblock(...)]` — the escape hatch

Injects verbatim GLSL immediately before a function's body. Still fully supported,
not deprecated. Use it for anything the struct form can't express: an interface
block (`out Block { ... } name;`), a resource whose members mix directions, or GLSL
syntax the struct parser doesn't accept.

```glsl
[shader('fragment')]
[resourceblock(
    out vec4 fragColor;
)]
void fs_main() { fragColor = vec4(1.0); }
```

Its `layout(...) in;`/`out;` lines conflict-check against the stage's generated
layout, so you can't silently emit two contradictory qualifiers. Prefer the struct
form for new code; do not "fix" existing `[resourceblock(...)]` usage unasked.

## Cross-stage and cross-module validation

Because both stages of a `[program(...)]` reference an interface by name, a mismatch
is caught before the driver ever sees GLSL. A typo in one stage's `[uses(...)]`:

```
demo:12: [uses('VertexOu')]: no interface named 'VertexOu' is declared or included. Did you mean 'VertexOut'?
demo: program 'default': vert stage 'vs_main' writes interface 'VertexOut' (declared demo:3) but frag stage 'fs_main' reads 'VertexOu' (used at demo:12). Did you mean 'VertexOut'?
```

Two modules declaring the same interface name with different members is caught at
whichever point the conflicting declarations are merged into one file's dependency
set — naming both modules, both lines, and the first differing member:

```
a: 'VertexOut' is declared differently in two modules: b:3 has 'vec4 color' but a:4 has 'vec3 color'
```

Fix by renaming one or making them identical — never by `[export]`ing around it (that
attribute has nothing to do with interface conflicts). `strict=False` downgrades both
to logged warnings, same as every other tlang diagnostic.

**tlang has no type system.** "Matching" compares declared text (member order,
qualifiers, type name, array suffix) — not resolved GLSL types. A `float` member and
an `int` member of the same name are a mismatch the same as any other difference;
there is no coercion or aliasing check.

## Reflection

```python
shader.interfaces               # Mapping[str, InterfaceDecl] -- this module + includes
shader.interfaces['VertexOut']  # InterfaceDecl(name=..., kind=VARYINGS, members=(...), ...)
member_locations(decl)          # {member_name: location}, or {} if unlocated/opted out
```

`InterfaceDecl`, `InterfaceKind`, `InterfaceMember`, `location_span`, and
`member_locations` are all importable directly from `tlang`.

## Bindings for `[uniforms(std140)]` and `[buffer(...)]`

UBOs and SSBOs are separate GL binding pools (`GL_MAX_UNIFORM_BUFFER_BINDINGS` /
per-stage `GL_MAX_*_UNIFORM_BLOCKS` vs. `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` /
per-stage `GL_MAX_*_SHADER_STORAGE_BLOCKS`). tlang allocates and strips dead blocks
in each pool independently, per compiled artifact (one compute kernel, or one
`[program(...)]`'s stages together). "Dead" means unreachable from that entry point's
`main()`: functions the entry point cannot reach are removed first, so a block reaching
the file only through an `[export()]`ed helper nobody calls does not survive. See
`references/runtime.md` for `kernel.bind(...)`, the per-kernel binding model, and the
exact stripping/limit-exceeded errors.

A `std140` block must be laid out to the std140 rules, which is why loose
`[uniforms]` (set with `set_uniforms(...)`, no layout rules to get right) is the
default and the block form is opt-in.
