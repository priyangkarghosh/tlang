# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Struct-based resource declarations: `[varyings]`, `[uniforms]`/`[uniforms(std140)]`,
  and `[buffer(std430|std140)]` declare a resource once, as a
  `struct Name { ... };`, instead of hand-written raw GLSL duplicated across stages.
  `[uses(Name, dir='in'|'out')]` ties a stage function to a declared `[varyings]`
  interface; because both stages reference the same declaration, `layout(location = N)`
  is derived from member order and matches on both sides by construction. `[buffer]`
  and `[uniforms]` desugar to ordinary module-scope GLSL at attach time, so DCE and
  binding allocation see exactly the text a raw declaration would have produced;
  `[varyings]` emits nothing until a `[uses(...)]` reference brings it into a stage.
  Geometry and tessellation stages array their per-vertex `[uses(dir='in')]` interface
  (and tess-control's `dir='out'`) automatically. `[varyings(locations=false)]` opts a
  declaration out of explicit locations, for member types `location_span` can't
  measure (a user struct, or an array sized by anything but an integer literal).
  `[resourceblock(...)]` is unchanged and remains the escape hatch for anything the
  struct form doesn't cover; raw GLSL declarations keep working without modification.
- `tlang.frontend.interface_registry`: GL-free struct parsing, GLSL emission, and
  location-span computation for the declarations above (`InterfaceDecl`,
  `InterfaceMember`, `InterfaceKind`, `InterfaceTable`, `location_span`,
  `member_locations`, `emit_glsl`), all importable from `tlang`.
- Cross-stage interface validation: an unresolved `[uses(...)]` name, a `[uses(...)]`
  naming a non-`varyings` interface, two `[uses(...)]` for the same direction on one
  function, a compute stage referencing an interface, a location-span that can't be
  measured, and two program stages whose interfaces differ by name or by declared
  members are all located, named diagnostics raised before the driver is asked to
  link — replacing an unlocated `GLSL Linker failed` for the mismatch case.
- `BindingRegistry` now allocates and tracks UBO bindings (`[uniforms(std140)]`) in a
  pool separate from SSBO bindings, respecting `GL_MAX_UNIFORM_BUFFER_BINDINGS` and the
  per-stage `GL_MAX_*_UNIFORM_BLOCKS` limit; dead UBO blocks are stripped by the same
  liveness pass as SSBOs. `kernel.bindings`/`pipeline.bindings` continue to report SSBO
  blocks only, unchanged.
- `Shader.interfaces` (`Mapping[str, InterfaceDecl]`): every `[varyings]`/`[uniforms]`/
  `[buffer]` interface visible to a module, keyed by name, for reflection.
- `Kernel`/`Pipeline` gain `bind_ubo`, `bind_ubos` and `uniform_blocks`, the
  uniform-block counterparts of `bind_ssbo`/`bind_ssbos`/`bindings`, so a
  `[uniforms(std140)]` block is bound by name rather than through moderngl's raw
  reflection. An unknown block name raises a located `TlangBindingError`.
- `src/tlang/py.typed` marker (PEP 561) so downstream type checkers pick up
  tlang's type annotations; verified it is included in a built wheel via
  `[tool.setuptools.package-data]`.
- `license`, `classifiers`, `keywords`, and `[project.urls]`
  (Homepage/Repository/Issues) in `pyproject.toml`.
- `[project.optional-dependencies]` `dev` extra (`pytest`, `build`, `twine`)
  for contributor tooling, separate from runtime dependencies.
- `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and a registered
  `gl` marker for tests that require a live OpenGL context.
- `plugins/tlang/skills/tlang/`: the bundled agent skill, restructured into an entry point plus
  `references/` for resources, runtime, errors and complete worked patterns. Every
  shader snippet in it is built against a real GL context; every quoted diagnostic is
  a message actually produced by the compiler.
- Claude Code plugin packaging: `.claude-plugin/marketplace.json` at the repo root
  (the repo is its own marketplace) and the plugin itself under `plugins/tlang/`,
  matching the layout Anthropic's own marketplace uses. Install with
  `/plugin install tlang@priyangkarghosh/tlang`. Both manifests pass
  `claude plugin validate --strict`. Nothing plugin-related reaches the PyPI wheel,
  since package discovery is scoped to `src/`.
- `editors/vscode-tlang/`: a VS Code extension giving `.tlang` files real syntax
  highlighting — attributes and their arguments, `#name<...>` directives, `{{ ... }}`
  templating, and embedded GLSL types/builtins. The GLSL grammar is embedded rather
  than included from another extension, so highlighting cannot silently degrade to
  nothing when that extension is absent.

### Changed
- Normalized the package version from the PEP 440-invalid `1.03.26` (which
  silently normalizes to `1.3.26`, causing the declared version and the
  built artifact to disagree) to the explicit, valid `1.3.26`.
- Corrected `requires-python` from `>=3.9` to `>=3.10`: the source uses
  `match`/`case` statements (`shader.py`, `shader_processor.py`) and
  `@dataclass(slots=True)` (`attribute.py`, `shader_source_line.py`,
  `errors.py`), both introduced in Python 3.10.
- Rewrote `requirements.txt` from a full `pip freeze` of a dev environment
  (mixing runtime deps with publishing/build tooling such as twine,
  keyring, rich, docutils) into a thin, commented file that installs the
  package in editable mode with the `dev` extra; the actual runtime
  dependency list continues to live in `pyproject.toml`.
- `license` now declared via the repository's MIT `LICENSE` file instead
  of being unset.
- Split `src/tlang/` into subpackages: `frontend/` (.tlang text to IR), `compiler/`
  (IR to GLSL to linked GL objects) and `runtime/` (GL wrappers used at run time),
  with `errors`/`shader_stages`/`shader_source_line`/`shader_utils` remaining as
  shared leaves. Dependencies run one way, `compiler -> frontend -> leaves`, and
  `runtime` imports only `errors`. The public API is unchanged: everything is still
  re-exported from `tlang/__init__.py`, so `from tlang import ShaderManager` and every
  other documented import keeps working. Code importing private module paths directly
  (`tlang.binding_registry`) must now use `tlang.compiler.binding_registry`.

### Fixed
- `requirements.txt` was UTF-16 encoded with a BOM, which broke or
  misparsed `pip install -r requirements.txt` on many setups; it is now
  plain UTF-8 with LF line endings.
- Shader directory resolution in `ShaderManager.__init__` now uses the
  path computed relative to the calling file instead of discarding it.
- Per-module SSBO binding reservations in `BindingRegistry.inject_bindings`
  are now preserved across modules, preventing binding collisions.

### Removed
- Deleted a runaway, self-nested `build/lib/build/lib/...` directory chain
  and the stale `src/tlang.egg-info/`, both untracked build artifacts left
  over from repeated local builds. `dist/` (published artifacts) was left
  untouched.

### Known Issues
- Interface matching compares declared text, not types — tlang has no type system.
  A member type outside GLSL's builtin set cannot be auto-located; use
  `[varyings(locations=false)]`.
- One interface per direction per stage. There is no GLSL interface-block emission
  form yet; declarations emit flat `in`/`out`.
- Dead-code elimination does not yet consult structured declarations; liveness is
  still a textual identifier scan.
