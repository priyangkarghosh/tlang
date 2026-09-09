# Errors

Every exception derives from `TlangError` and (usually) carries a source location
that renders as `module:line`. Every message below was triggered for real against a
live GL context (`moderngl.create_context(require=460, standalone=True)`, RTX 3090)
and is pasted verbatim, not paraphrased.

```python
from tlang.errors import (
    TlangError, TlangSyntaxError, TlangAttributeError, TlangDependencyError,
    TlangBindingError, TlangCompileError, TlangLinkError,
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
| `TlangError` (base) | `BufferPool`/`TempHandle` misuse (use-after-free, double-free, bad size) |

`get_source(...)` / `e.source` always has the exact GLSL handed to the driver, with
tlang's `#line` directives intact, so a driver message's line number points back at
your `.tlang` file.

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

**A `[program(...)]` link failure** (e.g. a raw `[resourceblock(...)]` on each stage
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
