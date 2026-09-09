# tlang for VS Code

Syntax highlighting for `.tlang` files -- GLSL plus tlang's three extra syntaxes.

## What it highlights

- **Plain GLSL**, embedded directly in this grammar (not via a dependency on
  another extension -- see "Why embedded GLSL" below): control keywords
  (`if`/`for`/`return`/`discard`/...), storage qualifiers (`in`/`out`/`uniform`/
  `buffer`/`layout`/`highp`/...), the full built-in type set (`vec3`, `mat4`,
  `sampler2D`, `uvec4`, image types, ...), built-in functions (`texture`, `dot`,
  `mix`, `imageLoad`, `barrier`, ...), built-in variables (`gl_Position`,
  `gl_GlobalInvocationID`, `gl_TessCoord`, ...), numbers (int/float/hex, with
  `u`/`f` suffixes), strings, and `//` / `/* */` comments.
- **Attribute blocks** -- `[shader('compute'), numthreads(64, 1, 1)]` at file
  scope, function-body pragmas like `[unroll]`/`[branch]`, and bare (no-parens)
  forms like `[export]`. The attribute name is highlighted distinctly
  (`entity.other.attribute-name`) from its arguments; string and numeric
  arguments get real string/number scopes; a block may span multiple lines
  (e.g. `[resourceblock(...)]`) and the raw GLSL inside a `resourceblock`
  payload is tokenized with the same GLSL rules as everywhere else. Also
  covers the newer `varyings`, `uniforms`, `buffer`, and `uses` attributes.
  A `[...]` is only ever treated as an attribute block when it is the first
  thing on its line, so an array subscript like `data[gid]` is never mistaken
  for one.
- **Alternate directive syntax** -- `#name<args>` (e.g. `#shader<'compute'>`),
  single line only. A real preprocessor line (`#version`, `#define`,
  `#extension`, `#ifdef`, ...) is never misread as this syntax, because the
  alt-directive pattern only matches when a `<...>` immediately follows the
  name; ordinary directives still get standard preprocessor highlighting.
- **Jinja templating** -- `{{ NAME }}` is highlighted as a template expression
  (`meta.template-expression`, with the name scoped as
  `variable.other.constant`) wherever it appears -- global scope, function
  bodies, attribute arguments, or inside a `#define`. The known collision
  where `{{` inside a nested GLSL brace initializer (e.g.
  `mat2({ {1,0}, {0,1} })`) can confuse the Jinja-based build step is a
  build-time ambiguity, not a highlighting bug: this grammar just highlights
  literal `{{ ... }}` runs and does not try to resolve that ambiguity.

## Why embedded GLSL (not `include: "source.glsl"`)

This grammar does not depend on another extension providing `source.glsl`.
A TextMate `include` of a scope nobody has registered silently produces *no*
highlighting for that region -- worse than the current `files.associations`
workaround, since at least plain GLSL keywords light up under the generic
`glsl` language today.

Two ways to avoid that dependency: embed the GLSL patterns directly, or
`include: "source.c"` (ships built into every VS Code install, so it's always
present). This extension embeds its own GLSL patterns because `source.c`
knows nothing about GLSL-specific vocabulary -- `vec3`, `mat4`, `sampler2D`,
`gl_Position`, `gl_GlobalInvocationID`, etc. -- and would either leave them
unscoped or (worse) mis-scope `vec3`/`mat4`/etc. as plain identifiers while a
theme still tries to color `int`/`float` via C's grammar. Embedding is more
maintenance surface, but it is the only option that lights up GLSL types and
builtins correctly and never depends on an extension that may or may not be
installed.

## Install (local use)

Pick one:

### Option A: copy/symlink into the extensions folder

1. Copy (or symlink) this entire `vscode-tlang` folder into your VS Code
   extensions directory:

   ```
   %USERPROFILE%\.vscode\extensions\vscode-tlang
   ```

   PowerShell, from the repo root:

   ```powershell
   Copy-Item -Recurse editors\vscode-tlang "$env:USERPROFILE\.vscode\extensions\vscode-tlang"
   ```

   or, to keep it live-editable, a symlink instead of a copy:

   ```powershell
   New-Item -ItemType Junction -Path "$env:USERPROFILE\.vscode\extensions\vscode-tlang" -Target "$(Resolve-Path editors\vscode-tlang)"
   ```

2. Restart VS Code (or run "Developer: Reload Window").

### Option B: package with `vsce` and install the `.vsix`

```powershell
cd editors\vscode-tlang
npx @vscode/vsce package
code --install-extension tlang-0.1.0.vsix
```

(Use `@vscode/vsce`, the currently maintained package, not the deprecated
`vsce` name -- the latter pulls in a broken old dependency chain on recent
Node versions.)

### Remove the old workaround

Whichever option you use, remove the global override that currently maps
`.tlang` to plain GLSL, or it will win over this extension's own language
contribution and you'll see GLSL highlighting instead of tlang's:

Open your user `settings.json` (Ctrl+Shift+P -> "Preferences: Open User
Settings (JSON)") and delete:

```json
"files.associations": {
    "*.tlang": "glsl"
}
```

(or just the `"*.tlang": "glsl"` entry if `files.associations` has other
entries you want to keep).

## Files

- `package.json` -- language + grammar contribution
- `language-configuration.json` -- comments, brackets, auto-closing/surrounding pairs
- `syntaxes/tlang.tmLanguage.json` -- the TextMate grammar (`source.tlang`)
- `samples/showcase.tlang` -- a sample file exercising every construct above,
  for eyeballing highlighting after install
- `tests/unit.test.tlang` -- a `vscode-tmgrammar-test` unit test asserting
  real token scopes (see "Testing" below)

## Testing

Grammar correctness was checked with
[`vscode-tmgrammar-test`](https://github.com/PanAeon/vscode-tmgrammar-test),
run offline via `npx`:

```powershell
cd editors\vscode-tlang
npx vscode-tmgrammar-test -g syntaxes/tlang.tmLanguage.json "tests/*.test.tlang"
```

This asserts, against the real grammar (not by inspection): attribute names
vs. arguments tokenize separately with string/number scopes; `#shader<...>`
tokenizes as the alt-directive form while `#version ...` does not; `{{ }}`
tokenizes as a template expression; and a multi-line `[resourceblock(...)]`
closes correctly (the `void` on the line after it is plain GLSL, not still
inside the attribute).
