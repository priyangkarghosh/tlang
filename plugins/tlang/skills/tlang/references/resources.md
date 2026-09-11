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
nothing between them, and never sharing the attribute's own line (that's rejected with a
message naming what was actually written — see below). It takes no instance name
(`struct Name { ... };`, not `struct Name { ... } inst;`) — that's a `TlangSyntaxError` if
violated.

`[uniforms]`/`[uniforms(std140)]`/`[buffer(...)]` are unconditional, unambiguous GLSL —
they mean the same thing regardless of which stage reads them, so they need no
direction and are visible at module scope to any stage that references a member.
`[varyings]` is different: the same declaration is `out` in the vertex stage and `in`
in the fragment stage, so it has no single correct module-scope form and emits nothing
until a stage brings it in with `[uses(...)]`.

## `[buffer]` single-declarator shorthand

The overwhelming majority of real SSBO blocks are one member — usually a single unsized
array. GLSL requires a block's own name to differ from its member's (`buffer ptcPositions
{ vec2 ptcPositions[]; }` fails to compile: `undefined variable "ptcPositions"`, the block
name shadows the member), so writing the block out by hand means inventing a second name
purely to satisfy the compiler. `[buffer]` closes that gap: it also accepts a single
declarator directly, in place of the struct, and the block name that GLSL requires becomes
tlang's business, not yours:

```glsl
[buffer]              vec2 ptcPositions[];   // -> kernel.bind(ptcPositions=buf)
[buffer(std140)]      ComputeDispatch dispatch[];
[buffer(name='ElementCount')] uint numElements;
```

Both forms work — the declarator on its own line right after `[buffer]` (mirroring the
struct form), or sharing the attribute's own line, `[buffer] vec2 ptcPositions[];`, which
is the natural way to write it.

**The name you write is the name you bind by — nothing else.** Every declaration form has
exactly one name that is both the Python-side handle (what `kernel.bindings`/`bind()`/
`bind_ssbo()`/`BufferPool` tags key by) and the emitted GLSL identifier, *except* this one
shorthand, where the member's name can't also be the block's:

| form | handle (what Python binds by) | emitted block name |
|---|---|---|
| `[buffer] vec2 ptcPositions[];` | `ptcPositions` (the member) | synthesised |
| `[buffer] struct Config { uint a; uint b; };` | `Config` (the struct name) | `Config` |
| `layout(std430) buffer X { ... };` (raw GLSL) | `X` | `X` |

For the shorthand, the handle is the member's own name, unchanged — `ptcPositions` stays
`ptcPositions`, never `PtcPositions`. The block GLSL actually needs is synthesised
deterministically from the member name (`ptcPositions` -> `ptcPositions__blk`) so builds
are reproducible, generated-GLSL diffs stay clean, and a driver error mentioning the block
is still greppable back to its source. That synthesised name is purely a GLSL-legality
artifact: it never appears on the Python side, and nothing ever binds by it.

`[buffer(name='...')]` overrides the *handle* — as a keyword, since positional arg 0 is
`layout` (`[buffer(std430)]`):

```glsl
[buffer(name='ElementCount')] uint numElements;   // handle: 'ElementCount'
```

Since the block name is always synthesised now, `name=...` is no longer needed to defeat a
would-be shadow collision — `ElementCount { uint numElements; }` and `Dispatch {
ComputeDispatch computeDispatch[]; }` need no override at all, `numElements`/
`computeDispatch` bind by their own names. Reach for `name=...` only when you actually want
a different Python-side handle than the member's name.

**Only `[buffer]`** gets this shorthand — `[varyings]`/`[uniforms]` always take the struct
form, since a bare varying/uniform declarator has no single obviously-correct desugaring
(direction, block-vs-loose) the way a buffer's `layout(std430) buffer Name { ... };` does.

**One declarator only:** a block has exactly one name, so `vec2 a[], b[];` (two
declarators in one statement) is rejected — give each its own `[buffer]`, or use the
struct form for a genuinely multi-member block:

```glsl
[buffer(std430)]
struct ContactCounts { uint contactPairs; uint pairOverflowCount; };
```

## `[extern]` — host-supplied constants

Declares a constant the *host* (Python) supplies, instead of the GLSL preprocessor:

```glsl
[extern] int BLOCK_SIZE;          // required -- value comes from ShaderManager(constants={...})
[extern] float PTC_RADIUS;
[extern] int WARP_SIZE = 32;      // optional: a default used when constants= doesn't supply one
```

Each becomes a plain GLSL `const`:

```glsl
const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
```

This replaces the `{{ CONSTANT }}` + `#define X {{ X }}` idiom for the common case of a single
named number: `{{ }}` is pure text substitution, so a missing constant is either a Jinja failure
with no tlang context, or (outside a `StrictUndefined` render) silently-wrong GLSL, and nothing
about it is typed or reflectable. `[extern]` fixes all three — the shader states what it needs, a
missing or wrong-typed value is a build error naming the constant, and `shader.externs`/
`proc.externs` let a caller ask what a module requires before ever building it.

**`{{ CONSTANT }}` still works, completely unchanged, including in the same module as
`[extern]`** — this is additive, not a replacement for `{{ }}`'s actual strength: substituting into
arbitrary text (`#define`, a value embedded in a comment, anything a `const` declaration can't sit
inside). Reach for `[extern]` when the shader just needs a typed constant; reach for `{{ }}` when
text needs to be substituted into something that isn't a standalone declaration.

Supported types: `int`, `uint`, `float`, `bool`. A Python `int` widens harmlessly into a `float`
declaration; anything else that doesn't match the declared type (a `str` for an `int`, a `float`
for an `int`, ...) is a build error naming the constant, its declared type, and what was actually
supplied. Several missing/wrong-typed constants in one module are collected and reported together,
not one build attempt per constant.

Like `[buffer]`'s single-declarator shorthand, both forms work — the declarator on `[extern]`'s
own line, or on the line right after it — and both reuse the same declarator parser
(`parse_declarator_at`) rather than a second one. Unlike every other declaration attribute,
`[extern]` never takes a struct: it declares exactly one constant per attribute. It's module scope
only — writing `[extern]` inside a function body is a build error naming the correct scope, an
array (`[extern] int SIZES[4];`) is rejected (declare one constant per `[extern]`), and a name
already used by another `[extern]` or a hand-written `const` in the same module is a duplicate-
declaration error.

**The critical use case — sizing a compute dispatch:**

```glsl
[extern] int BLOCK_SIZE;

[shader('compute')]
[numthreads(BLOCK_SIZE, 1, 1)]
void cs_main() { ... }
```

`[extern]`'s `const` lands in the module text, which `Shader._build` always assembles ahead of
every function's `FUNC_CONFIG` (where `[numthreads(...)]`'s `layout(local_size_x = ...)` lands) --
so GLSL's declare-before-use rule is satisfied regardless of where in the file `[extern]` is
written relative to the function that uses it.

One real driver restriction to know about: a `const` used *inside a layout qualifier* (as
`local_size_x` above does) is only a legal constant-foldable expression from **GLSL 4.40 on** --
verified for real on an RTX 3090 (driver 616.64), NVIDIA's compiler rejects it at `#version 430
core` ("non constant expression in layout value") and accepts the identical text unchanged at
440/450/460. This restriction is specific to layout qualifiers -- an ordinary array sized by an
`[extern]` constant (`uint scratch[BLOCK_SIZE];`) has no such version floor. Pass
`ShaderManager(version=...)` >= `'440 core'` if `[numthreads(...)]`/other layout qualifiers need to
reference an `[extern]` constant.

## `[extern(precompile=[...])]` — compiled variants of a host-supplied constant

Adding `precompile=[...]` to an `[extern]` declares a **variant axis**: `NAME` compiles to
`const NAME = v;` in a fully independent `Shader` per listed value, built eagerly by
`ShaderManager` at ordinary build time. Nothing about it is lazy or on-demand — every value is
compiled the moment `ShaderManager(...)` runs, so nothing extra is retained afterward and there
is no first-use compile hitch to warm away.

```glsl
[extern] int BLOCK_SIZE;                    // unchanged: one baked const from constants=
[extern(precompile=[1, 2, 4])] int mode;    // three compiled variants, nothing else
[extern(precompile=[false, true])] bool stabilizing;
```

**There is no default artifact.** Unlike a plain `[extern]`, a precompile axis is not a uniform
you can optionally specialise — it is *only* the listed compile-time values. `constants={...}`
never resolves it, and `get_kernel` with no value for it is an error, not a fallback:

```python
kernel = shader.get_kernel('solve')  # ERROR: 'mode' needs a value -- module 'demo' has no
                                      # default artifact once it declares [extern(precompile=[...])]
kernel = shader.get_kernel('solve', mode=1)  # OK -- tlang emitted 'const int mode = 1;'
```

**This is a breaking change for an existing uniform.** Adding `precompile=[...]` to an `[extern]`
that used to be a plain uniform stops `set_uniform(...)` from working at all, for every kernel in
that module — not just the one that reads it — and stops any value outside the declared list from
being usable. That trade is intentional: once every value is declared, there is nothing left for a
runtime uniform to do that a compile-time constant doesn't already do better, and no attribute
should mean "a uniform, except when it isn't."

**A module with more than one precompile axis needs every axis's value together, every time** —
there is no partially-resolved text to fall back to for the axis you didn't mention:

```glsl
[extern(precompile=[1, 2])] int mode;
[extern(precompile=[false, true])] bool flag;
```
```python
kernel = shader.get_kernel('solve', mode=1)             # ERROR: 'flag' needs a value too
kernel = shader.get_kernel('solve', mode=1, flag=True)  # OK
```
`ShaderManager` builds the full cross product of every axis's declared values — a module with a
2-value axis and a 3-value axis gets `2 * 3 = 6` compiled variants, not `2 + 3`.

The generated shader SOURCE is otherwise identical across every variant of the same combination
shape — same buffers, same bindings (`kernel.bindings`) — only the declaration(s) differ. tlang's
dead-code elimination is textual, not constant-folding, so both arms of a
`stabilizing ? a : b` stay in the artifact regardless of which value was baked in; `bind()` behaves
identically across every variant, with no per-variant re-keying needed.

**A value not in `precompile=[...]` is a build-time-shaped error, not a fallback:**

```
demo: 'mode=3' was not precompiled -- module 'demo' only precompiled mode in {1, 2, 4}
```

This is deliberate: a silent recompile-on-demand would be a silent performance cliff the first
time a caller passes an unexpected value; an explicit error, naming the permitted set, is not.

**Why it's worth it, measured:** a `stabilizing ? ptcPositions[i] : ptcPredictedPositions[i]`
select (5+ read sites in a real solver's hottest loop), compared against the equivalent
uniform-driven branch, benchmarked at **~0.10 ms/dispatch as a uniform vs ~0.06 ms precompiled**
on this machine — roughly **1.5-1.6x** — because the driver can fold the select and drop a buffer
load once the branch condition is a compile-time constant, which it cannot do while the value is
a runtime uniform.

**Cost.** Every precompiled combination is a full rebuild of the ENTIRE module, not just the one
kernel using the constant(s) — a module with 5 kernels and a single 2-value `precompile=[...]`
axis compiles all 5 kernels twice over, not once, and there is no default build to fall back to
in between. A module that declares no `precompile=[...]` axis at all is completely unaffected:
build time and retained memory are identical to a build predating this feature.

`precompile=[...]` cannot be combined with `= default` — `precompile=[...]` already supplies
every value this constant will ever take, so a fallback default has nothing left to fall back to.

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
[glsl( out vec4 fragColor; )]
void fs_main() { fragColor = vec4(gcolor, 1.0); }
```

The geometry stage emits `layout(location = 0) in vec3 color[];` and
`layout(location = 0) out vec3 gcolor;` — the input is arrayed, the output is not.

Arraying does not add a location — a per-vertex array of `vec3` still occupies one
location; the array dimension and the location count are independent. A member that
*already* declares its own array (`vec3 color[4]`) cannot also be arrayed by the
stage — that's a build error pointing at `[glsl(...)]` as the escape hatch
for an array-of-arrays interface.

## `[glsl(...)]` — the escape hatch

Named `[glsl]` because that is what it means: stop preprocessing, the text inside is
raw GLSL and reaches the driver untouched. `[resourceblock(...)]` is the former name and
still works as an alias, so existing shaders need no edit.

Injects verbatim GLSL immediately before a function's body. Still fully supported,
not deprecated. Use it for anything the struct form can't express: an interface
block (`out Block { ... } name;`), a resource whose members mix directions, or GLSL
syntax the struct parser doesn't accept.

```glsl
[shader('fragment')]
[glsl(
    out vec4 fragColor;
)]
void fs_main() { fragColor = vec4(1.0); }
```

Its `layout(...) in;`/`out;` lines conflict-check against the stage's generated
layout, so you can't silently emit two contradictory qualifiers. Prefer the struct
form for new code; do not "fix" existing `[glsl(...)]` usage unasked.

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

`[extern]` constants are reflected separately, since they aren't interfaces (see above):

```python
shader.externs                     # Mapping[str, ExternConst] -- this module's own [extern]s
shader.externs['BLOCK_SIZE']       # ExternConst(name='BLOCK_SIZE', type_name='int', ...)
shader.externs['BLOCK_SIZE'].value    # 256 -- the Python value actually used
shader.externs['BLOCK_SIZE'].literal  # '256' -- the GLSL literal text emitted
```

`proc.externs` (on a `ShaderProcessor`, before a `Shader` is built) reflects the same
declarations, `.has_default`/`.default_value` included, before `constants={...}` is even known --
`.resolved`/`.value`/`.literal` only populate once `resolve_externs` has run. `ExternConst` is
importable directly from `tlang`.

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
