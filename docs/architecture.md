# tlang architecture

How the compiler works, for contributors. For using tlang see
[usage.md](usage.md).

## Source layout

```
src/tlang/
├── errors.py, shader_stages.py,     shared leaves -- no internal dependencies
│   shader_source_line.py, shader_utils.py
├── frontend/   .tlang text -> IR
│     attribute, attribute_registry, attribute_handlers, attribute_manager,
│     interface_registry, function_manager
├── compiler/   IR -> GLSL -> linked GL objects
│     shader_processor, shader, shader_manager, binding_registry, dependency_manager
└── runtime/    GL wrappers used at run time, not build time
      kernel, pipeline, buffer_pool
```

Dependencies run one way: `compiler → frontend → leaves`. `runtime` imports nothing but
`errors`, which is what makes the compile path separable from GL (see the constraints at
the end of this document). The public API is re-exported from `tlang/__init__.py`, so
`from tlang import ShaderManager` is unaffected by where a module lives.

## The pipeline

```
.tlang files
     │
     ├─ ShaderProcessor        per file: extract functions, resolve attributes
     │      ├─ FunctionManager     find function definitions, split out their bodies
     │      ├─ AttributeManager    parse [attrs], dispatch through the registry
     │      ├─ InterfaceRegistry   parse [varyings]/[uniforms]/[buffer] structs,
     │      │                      desugar them to GLSL, record the declarations
     │      └─ (module text)       everything not owned by a stage function,
     │                             plus [export]ed helper bodies
     │
     ├─ DependencyManager      resolve [include] graph, render {{ constants }}
     │
     ├─ resolve_interfaces     per module: merge own + dependency interfaces,
     │                         resolve [uses(...)], emit per-stage GLSL, validate
     │
     ├─ BindingRegistry        project-wide block popularity → preference ranks
     │
     └─ Shader._build          per entry point: assemble → DCE → allocate
            │                  bindings → compile → verify → wrap
            ├─ Kernel          compute entry points
            └─ Pipeline        [program(...)] declarations
```

`ShaderManager.__init__` drives all of it. Everything happens at construction time.

## The core data structure

The intermediate representation is `dict[int, ShaderSourceLine]` — a sparse map from
1-based source line number to `(vctx, data)`, where `vctx` is the "virtual context" (the
module the line came from) and `data` is the line's text.

Sparse, because `FunctionManager.extract_funcs` **pops** each function's lines out of the
map into `FunctionDef.line_body`. What remains in `src_map` is the file's module-level
code. This is why:

- a stage function's body is not part of the module text other files can include, and
- `Shader.build_map` can detect gaps (`index - prev_index != 1`) and emit a `#line`
  directive to resynchronise the driver's view.

`#line` accounting is the reason for the whole structure. Every generated stage carries
directives mapping back to the original `.tlang` file and line, so driver errors are
reported against your source rather than against generated output.

Note the ordering constraint this creates: because function bodies leave `src_map`
before the module text is assembled, anything that must apply to *all* emitted text —
constant substitution, most notably — has to be applied to `src_map`, every
`func.line_body`, and every `func.config` separately. `ShaderManager` does exactly that,
and the comment there explains why.

## Attributes: the registry

`attribute_registry.py` provides the machinery; `attribute_handlers.py` holds the data.

An attribute is one `AttrSpec` row:

```python
AttrSpec(
    name='geom',
    scope=Scope.GLOBAL,
    stages=frozenset({ShaderStage.GEOM}),
    params=(Param('in', str, choices=(...), default='triangles'), ...),
    summary='Geometry-stage settings.',
    example="[geom(in='points', out='triangle_strip', max_verts=6)]",
)
```

The table is a **list**, and lookup is keyed on `(name, stage)`, not on name alone. That
matters: `[triangles]` means "input primitive" for a geometry shader and "domain" for a
tess-eval shader, so it has two rows. A flat name-keyed dict cannot express that, and the
previous design worked around it with a `triangles_tese` entry no user could ever type.

Everything else derives from the table:

- **Dispatch** — one function resolves `(name, scope)`, filters by stage, coerces
  arguments against `params`, then either calls the spec's handler or applies its
  declarative `emits`. Only genuinely imperative attributes (`shader`, `program`,
  `include`, `extend`, `link`, `export`, `numthreads`, `glsl`) need handler
  code; the ~20 bare markers and the function-body pragmas are pure data.
- **Validation** — `Param.type` coercion means `[frag(early_tests=false)]` yields a real
  `False`, not the truthy string `'false'`. `Param.choices` rejects invalid values with
  the legal set listed.
- **Diagnostics** — unknown names get a `difflib` did-you-mean; wrong-stage use names the
  stages that *are* valid; wrong-scope use says whether the attribute belongs at file
  level or inside a function body.
- **Documentation** — `scripts/gen_attribute_docs.py` renders the table to Markdown, so
  the reference cannot drift from the implementation.

Some attributes resolve immediately; stage-dependent ones are deferred onto
`FunctionDef.attrs` and resolved by `ShaderProcessor` once the owning function's stage is
known.

`StageConfig` collects a function's layout qualifiers and raw declarations separately,
merging qualifiers by direction so conflicts (two `numthreads`, or a `[glsl]`
line contradicting a stage default) are detected rather than emitted twice. It is
flattened to the `list[str]` that `func.config` exposes.

## Dependencies and templating

`DependencyManager` owns the include graph and the Jinja environment.

`resolve_dependencies(name)` returns a module's full transitive dependency list in
topological order, memoised, raising `TlangDependencyError` on cycles and missing
modules. It is the **single source of truth** for the dependency relation — extension
propagation consults it rather than walking direct includes, which previously made
`[extend(...)]` inheritance depend on filesystem ordering.

`render(text, module, line)` is the single choke point for constant substitution.
`StrictUndefined` means a missing constant is a located error, not a silently emitted
identifier.

Templating renders Jinja directly over GLSL, which is why `mat2({{1.0, 0.0}, ...})`
breaks. Migrating to `${NAME}` delimiters (`$` is not a valid GLSL character) is the
planned fix.

## Resource interfaces

`[varyings]`, `[uniforms]` and `[buffer]` declare a resource once as a struct.
`interface_registry.py` parses the struct, computes location spans, and emits flat GLSL.

Each declaration does two things: it **desugars to ordinary GLSL** written back into the
same `src_map` slots it occupied, and it **records an `InterfaceDecl`** that drives
validation and reflection. Because the desugared text is what a raw declaration would
have produced, DCE, `BindingRegistry` and `#line` accounting needed no changes.

The line-slot discipline keeps `#line` exact: all generated text goes into the first
consumed slot and the remaining slots become `'
'`, so `build_map` brackets the blob
with `#line N` / `#line N+1` and everything after it stays line-accurate.

`[buffer]` and `[uniforms]` desugar into module text. **`[varyings]` does not** — the
same declaration is `out` in one stage and `in` in the next, so there is no correct
module-scope form. Varyings reach a stage only through a `[uses(...)]` reference, which
emits into `func.config`.

Emission is **flat**, not GLSL interface blocks: blocks are illegal for vertex inputs and
fragment outputs, so flat is the only form covering all three directions, and it lets one
stage migrate while the other keeps its raw declarations.

Locations need no allocator. Both stages reference the same declaration, so member index
determines the location and the sides agree by construction. `location_span` maps a GLSL
builtin type to the number of slots it consumes — `mat4` is 4, `dvec3` is 2 — which is
what stops `mat4 m; vec3 c;` from silently aliasing. An unmeasurable type is a located
error, with `[varyings(locations=false)]` as the opt-out.

Resolution is a second pass (`ShaderProcessor.resolve_interfaces`), driven by
`ShaderManager` after every module is registered — an interface may be declared in a
dependency, so the include graph must be known first — and before constant substitution,
so a `{{ N }}` in an emitted array suffix is rendered rather than passed to the driver.

## Bindings

Allocation happens **per compiled artifact**, after dead-code elimination:

1. `ShaderManager` scans all rendered modules once for block *popularity* and reduces it
   to a `preference_rank` map. This is advisory, not authoritative.
2. `Shader._build` assembles an entry point's source, then runs
   `remove_unused_buffers` on it.
3. `BindingRegistry.allocate_artifact` reserves explicit `layout(binding = N)` pins
   first, then assigns remaining live blocks to the lowest free index in preference-rank
   order.
4. After linking, `verify_link` reflects the program's storage blocks and asserts the
   map matches what tlang intended.

A compute kernel is one artifact. A `[program(...)]` is one artifact spanning its stages,
so a block shared between vertex and fragment gets one binding in both.

SSBO blocks and uniform blocks (`[uniforms(std140)]`) are separate GL namespaces, so each
allocates from its own pool through the same code path, and `verify_link` reflects both.
`Kernel.bindings` stays SSBO-only; `Kernel.uniform_blocks` is the uniform counterpart.

Two limits are checked, and they are different: the binding *index* ceiling
(`GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS`, typically 96) and the per-stage *block count*
(`GL_MAX_<STAGE>_SHADER_STORAGE_BLOCKS`, often 16). The latter is the one that actually
bites, and exceeding it is a hard link failure.

The preference rank preserves a useful property: a widely-shared block like `Globals`
lands on the same low binding in every artifact that uses it, without spending the global
budget on blocks that never coexist.

`verify_link` matters more than it looks. GL silently accepts two blocks aliased onto one
binding — no error, just wrong data. Reflecting and asserting turns any future gap in the
textual analysis into a build failure instead of silent corruption.

## Dead-code elimination

`remove_dead_blocks` strips buffer *and* uniform blocks an entry point does not
reference (`remove_unused_buffers` is the older name, kept as a delegating alias). It is
load-bearing rather than an optimisation: the whole module is concatenated into every
stage, so an include-heavy project would blow the per-stage block limit without it — and
the uniform-block limit (~14) is lower than the SSBO one (16).

It is textual, not a GLSL parse. It masks comments, string literals and preprocessor
lines before scanning; brace-matches block bodies; takes the trailing identifier of each
declarator after stripping array suffixes, qualifiers and precision; and follows the
instance name for instance-named blocks.

The analysis is deliberately biased toward **keeping** a block when uncertain: a wrongly
kept block costs one binding slot, while a wrongly removed one is a compile error whose
message points at a use whose declaration has vanished. That asymmetry drives the design.

Its known limit: an unrelated identifier of the same name elsewhere in the file keeps a
block alive. Restricting the search to reachable code from the entry point would fix it,
and needs a call graph.

Structured declarations do **not** currently feed this pass — a `[buffer]` desugars to
text and is then rediscovered textually like any other block. Liveness could instead work
from the recorded `InterfaceDecl`, which would tighten the analysis considerably. That is
deliberately not done here: it belongs with the entry-point-granular DCE described in the
roadmap, which needs the same call graph.

## Errors

`errors.py` defines a single hierarchy rooted at `TlangError`, each carrying an optional
`SourceLocation(module, line)` that renders as `demo:26`. `TlangCompileError` also carries
the stage, entry point, and the exact GLSL handed to the driver.

The design principle is that a preprocessor's job is to fail where the mistake is. Every
diagnostic should name the file, the line, and — where a set of valid options exists —
what would have been correct.

`strict=False` downgrades build failures to logs. It exists for iteration; it leaves a
`Shader` missing kernels, which surfaces later as a `KeyError`.

## Runtime layer

`kernel.py`, `pipeline.py` and `buffer_pool.py` are the only modules that touch GL at
run time rather than build time, and they have no dependency on the compiler.

`Kernel` and `Pipeline` reflect block bindings from the linked program *by name* and
cache them, so they are agnostic to how allocation decided the numbers. `BufferPool`
recycles buffers in power-of-two size classes with scope guards and use-after-free
detection.

## Design constraints worth preserving

- **The compiler needs GL for one integer plus the driver.** Everything from parsing
  through GLSL assembly is pure text manipulation; only the binding limits
  (`ctx.info`) and the final `compute_shader`/`program` calls need a context. Separating
  those would make the preprocessor unit-testable without a GPU — the largest remaining
  structural improvement.
- **No GLSL parser.** tlang needs to locate attributes, function headers and buffer
  blocks. It does not need to understand types, expressions or scope. Every design here
  stays inside that boundary, and that boundary is what keeps the codebase small.
- **The `.tlang` surface syntax is the product.** Internal restructuring should not
  require users to edit shaders.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the current roadmap.
