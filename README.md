# tlang

tlang is a GLSL preprocessor for Python and ModernGL. A `.tlang` file is ordinary GLSL
plus C#-style attributes, so one file can hold every stage of a pipeline — vertex,
fragment, geometry, tessellation and compute — and tlang compiles it straight into
ModernGL `Program` and `ComputeShader` objects.

```glsl
[include(math)]

#define TILE {{ BLOCK_SIZE }}

layout(std430) buffer Data { uint data[]; };

[program('default', vert='vs_main', frag='fs_main')]

[shader('vertex')]
void vs_main() {
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

[shader('fragment')]
[resourceblock(
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}

[shader('compute'), numthreads(256, 1, 1)]
void cs_go() {
    uint gid = gl_GlobalInvocationID.x;
    data[gid] = add(gid, 1u);   // `add` comes from math.tlang
}
```

```python
import moderngl as mgl
from tlang import ShaderManager

ctx = mgl.create_context(require=460, standalone=True)

sm = ShaderManager(
    ctx=ctx,
    version='460 core',
    dir='shaders',
    constants={'BLOCK_SIZE': 256},
)

shader = sm.get_shader('demo')
program = shader.get_program('default')   # vertex + fragment pipeline
kernel  = shader.get_kernel('cs_go')      # compute kernel

kernel.bind_ssbo('Data', buf)
kernel.dispatch(n_groups)
```

## Features

- **Every stage in one file.** `[shader('vertex')]`, `[shader('fragment')]`,
  `[shader('compute'), numthreads(...)]` and the rest live side by side. Each entry
  point is compiled as its own translation unit, so stages stay independent while the
  code they share stays in one place.
- **Pipelines declared in-source.** `[program('name', vert=..., frag=...)]` links stages
  into a GL program, validated at build time — a missing entry point or an illegal stage
  combination is a clear error, not a driver message.
- **Cross-file modules.** `[include(other)]` pulls in another `.tlang` file;
  `[export]` marks a helper as part of a module's public surface. Includes resolve
  transitively, with circular-import detection.
- **Automatic SSBO bindings.** Buffer blocks are assigned binding points per compiled
  artifact, from only the blocks that artifact actually references. Pin one yourself
  with `layout(binding = N)` and tlang works around it.
- **Templating.** `{{ CONSTANT }}` placeholders are substituted at build time from the
  `constants` dict — everywhere, including inside kernel bodies and `numthreads(...)`.
- **Errors that point at your source.** tlang emits `#line` directives, so driver
  messages reference your `.tlang` file and line, and every exception carries a
  `module:line` location.
- **Extension groups.** `[extend('int64')]` expands to the relevant
  `#extension` lines and propagates to every module that includes it.

## Install

```bash
pip install git+https://github.com/priyangkarghosh/tlang.git
```

Requires Python 3.10+ and `moderngl>=5.8`. A GL context must exist before any tlang call.

### Editor and agent tooling

**VS Code syntax highlighting** for `.tlang` files — attributes, `#name<...>` directives,
`{{ CONSTANT }}` templating and embedded GLSL:

```bash
cd editors/vscode-tlang && npx @vscode/vsce package
code --install-extension tlang-*.vsix
```

Remove any `"files.associations": {"*.tlang": "glsl"}` override first, or it wins over the
extension's own language contribution.

**Claude Code plugin** — installs the bundled skill so coding agents write correct tlang:

```bash
/plugin install tlang@priyangkarghosh/tlang
```

Or from a local checkout, `claude plugin install tlang@./path/to/tlang`. The skill is also
usable without the plugin: point any agent at [plugins/tlang/skills/tlang/](plugins/tlang/skills/tlang/).

## Documentation

| Document | Contents |
|---|---|
| [docs/usage.md](docs/usage.md) | Building shaders, programs vs. kernels, buffers, dispatch, debugging |
| [docs/attribute-reference.md](docs/attribute-reference.md) | Every attribute, its arguments and applicable stages (generated from the registry) |
| [docs/architecture.md](docs/architecture.md) | How the compiler works, for contributors |
| [docs/slang-alignment.md](docs/slang-alignment.md) | What tlang borrows from Slang, and the roadmap that follows from it |
| [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) | Known limitations and the roadmap |
| [plugins/tlang/skills/tlang/](plugins/tlang/skills/tlang/) | A portable skill so coding agents use tlang correctly (installable as a Claude Code plugin) |
| [editors/vscode-tlang/](editors/vscode-tlang/) | VS Code syntax highlighting for `.tlang` files |

## Scope

tlang is **a library, not a compiler CLI**. It does not write `.vert`/`.frag`/`.comp`
files to disk — shaders are built in-process and handed to ModernGL. You can inspect
any generated stage with `shader.get_source(entry_point)`, and a `TlangCompileError`
carries the exact GLSL that failed.

## Development

```bash
pip install -e .[dev]
python -m pytest -m "not gl"   # preprocessor tests, no GPU required
python -m pytest               # adds integration tests needing a GL context
```

The attribute reference is generated — after changing any `AttrSpec`, run
`python scripts/gen_attribute_docs.py` rather than editing the tables by hand.

## License

MIT — see [LICENSE](LICENSE).
