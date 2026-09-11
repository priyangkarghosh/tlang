# Patterns

Every shader on this page was built for real against a live GL context
(`moderngl.create_context(require=460, standalone=True)`, RTX 3090) via
`ShaderManager(..., strict=True)` and, where noted, dispatched/linked and its output
read back. Copy and adapt rather than starting from scratch.

## Compute kernel

```glsl
layout(std430) buffer Data { uint data[]; };

[shader('compute'), numthreads(256, 1, 1)]
void cs_add_one() {
    uint gid = gl_GlobalInvocationID.x;
    data[gid] = data[gid] + 1u;
}
```

```python
sm = ShaderManager(ctx=ctx, version='460 core', dir='shaders', constants={})
shader = sm.get_shader('kernel')          # module name = filename without .tlang
k = shader.get_kernel('cs_add_one')

n = 64
buf = ctx.buffer(array.array('I', range(n)).tobytes())
k.bind_ssbo('Data', buf)
k.dispatch((n + 255) // 256)              # workgroup counts, not thread counts

out = array.array('I'); out.frombytes(buf.read())
# out == [1, 2, 3, ..., 64] -- verified
```

## Vertex + fragment with structured varyings

Vertex attributes, inter-stage varyings, and a loose uniform, each declared once:

```glsl
// Vertex attributes from the vertex buffer.
[varyings]
struct MeshVertex { vec3 position; vec3 normal; };

// Varyings passed vertex -> fragment.
[varyings]
struct VertexOut { vec3 color; vec2 uv; };

// Loose uniforms -- set with pipeline.set_uniforms(time=...).
[uniforms]
struct Frame { float time; };

[program('default', vert='vs_main', frag='fs_main')]

[shader('vertex')]
[uses(MeshVertex, dir='in')]
[uses(VertexOut, dir='out')]
void vs_main() {
    color = normal * 0.5 + 0.5;
    uv = position.xy;
    gl_Position = vec4(position + vec3(0.0, 0.0, sin(time)), 1.0);
}

[shader('fragment')]
[uses(VertexOut, dir='in')]
[glsl(
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = vec4(color, uv.x);
}
```

```python
shader = sm.get_shader('pipe')
p = shader.get_pipeline('default')
p.set_uniforms(time=1.5)                  # verified: p['time'].value == 1.5
```

`member_locations(...)` for each interface after this build: `MeshVertex` ->
`{'position': 0, 'normal': 1}`, `VertexOut` -> `{'color': 0, 'uv': 1}`, `Frame` ->
`{}` (loose uniforms never have locations).

## Cross-module `[include]`/`[export]`

```glsl
// shaders/mathlib.tlang
[export]
float square(float x) { return x * x; }

[export]
float cube(float x) { return x * x * x; }
```

```glsl
// shaders/user.tlang
[include(mathlib)]

layout(std430) buffer Out { float results[]; };

[shader('compute'), numthreads(64, 1, 1)]
void cs_eval() {
    uint gid = gl_GlobalInvocationID.x;
    results[gid] = square(float(gid)) + cube(float(gid));
}
```

`[include(mathlib)]` in the consumer, `[export]` on each helper in the provider —
you need both. A function without `[export]` stays private to its file. Global-scope
declarations (buffers, constants, `#define`s) cross module boundaries with
`[include]` alone. Built and dispatched: `cs_eval` links and runs, `sm.get_shader
('user').get_kernel('cs_eval')` exists.

## Geometry stage: automatic input arraying

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
        gcolor = color[i];              // `color` is `in vec3 color[];` here -- arrayed automatically
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}

[shader('fragment')]
[uses(GOut, dir='in')]
[glsl(
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = vec4(gcolor, 1.0);
}
```

Note `gs_main` does not add `[]` itself — `[uses(VOut, dir='in')]` on a geometry
stage already emits `in vec3 color[];`; the shader indexes `color[i]`.

## UBO (`[uniforms(std140)]`) bound by name

```glsl
[uniforms(std140)]
struct Lights { vec4 positions[16]; int count; };

layout(std430) buffer Out { vec4 results[]; };

[shader('compute'), numthreads(16, 1, 1)]
void cs_sum() {
    uint gid = gl_GlobalInvocationID.x;
    vec4 acc = vec4(0.0);
    for (int i = 0; i < count; i++) acc += positions[i];
    results[gid] = acc;
}
```

```python
k = sm.get_shader('ubo').get_kernel('cs_sum')
k.bind_ubo('Lights', lights_buf)          # UBO pool -- separate from bind_ssbo
k.bind_ssbo('Out', out_buf)
k.dispatch(1)
# verified: 3 light positions of (1,2,3,4) summed -> [3, 6, 9, 12] read back correctly
print(dict(k.uniform_blocks))             # {'Lights': 0}
print(dict(k.bindings))                   # {'Out': 0} -- SSBO pool, unaffected by the UBO
```

## `{{ CONSTANT }}`-sized varying (requires `locations=false`)

```glsl
[varyings(locations=false)]
struct Batch { vec4 items[{{ N }}]; };

[program('default', vert='vs_main', frag='fs_main')]

[shader('vertex')]
[uses(Batch, dir='out')]
void vs_main() {
    for (int i = 0; i < {{ N }}; i++) items[i] = vec4(float(i));
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

[shader('fragment')]
[uses(Batch, dir='in')]
[glsl(
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = items[0];
}
```

Built with `constants={'N': 8}`. Omitting `[varyings(locations=false)]` here is a
build-time error — see `references/errors.md`.
