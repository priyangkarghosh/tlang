# tlang
tlang is a shader preprocessing language that extends GLSL with attributes and templating, making it easier to organize, reuse, and maintain GPU shaders across all stages. Designed specifically for Python and ModernGL, tlang lets you define vertex, fragment, compute, and geometry shaders in a single file while keeping your code clean and maintainable.

##  Features
-  **Multi-stage Shaders in One File**
Define **vertex, fragment, compute (and more)** shaders in the same file using `[shader(...)]` attributes.

- **Attribute-based Metadata**
  - `[shader('vertex')]`, `[shader('fragment')]`, `[shader('compute'), numthreads(...)]`
  - `[resourceblock(...)]` for declaring uniforms + shared memory compactly.
  - `[program(...)]` to define **pipeline programs** linking shader stages together
  - `[extend('name')]` for importing shared utilities (e.g. subgroup helpers).

-  **Templating**
Replace placeholders like `{{ BLOCK_SIZE }}` with values at build time.

- **Automatic Buffer Binding**
  -   SSBOs are automatically assigned binding points.
  -   You can override by explicitly adding `layout(binding = X)` yourself.
  `layout(binding=3, std430) buffer MyBuffer { // my buffer... }`

-  **Readable, Modular, Reusable**
No more splitting code into dozens of `.vert` / `.frag` / `.comp` files.
Keep **all shader stages in sync** inside one `.tlang` file.

- **Debugging Friendly**  
  - Automatic `#line` directives map generated GLSL back to your `.tlang` file, making driver error messages meaningful.

- **Includes / Modular Code**
  - `tlang` supports `#include`-style directives using `[include(...)]`, letting you split reusable functionality into separate files and keep shaders organized:
  - TLang does NOT currently support importing 'shader' functions between files, but all buffers, constants, and functions are imported
  -   Example usage: ```[include(utils)]```
---

## Example
`math.tlang`
```glsl
uint add(uint x, uint y) {
  return x + y;
}

// geometry shader
[shader('geom')]
[resourceblock(
    layout(triangles) in;
    layout(triangle_strip, max_vertices=3) out;
)]
void gs_main() {
    // emit the triangle with slight offset for demonstration
    for (int i = 0; i < 3; i++) {
        vec4 pos = gl_in[i].gl_Position;
        pos.xy += vec2(0.1 * i, 0.1 * i); // small offset
        gl_Position = pos;
        EmitVertex();
    }
    EndPrimitive();
}
```

`shader.tlang`
```glsl
[include(math)]
[extend('int64')]

// compiler constants
#define BLOCK_SIZE {{ BLOCK_SIZE }}

// programs
[program('default', vert='vs_main', frag='fs_main')]

// vertex shader
[shader('vertex')]
void vs_main() {
	gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

// fragment shader
[shader('fragment')]
[resourceblock(
	out vec4 fragColor;
)]
void fs_main() {
	fragColor = vec4(1.0, 0.0, 0.0, 1.0); // red
}

// compute shader
[shader('compute'), numthreads(256, 1, 1)]
void cs_randomize() {
	uint gid = gl_GlobalInvocationID.x;
	// ...
}

```

  

### Compilation Example

  

**Input (`shader.tlang`):**

Contains vertex, fragment, and compute shaders in one file (above).
*Note: outputs may not actually reflect what is processed, this is simply a readable example

  

**Output:**

  

-  `shader.vert`

```glsl
#version 450

#line 1 "VCTX_EXTENSION_LIST"
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_EXT_shader_atomic_int64: enable
#extension GL_KHR_shader_atomic_int64: enable
#extension GL_NV_shader_atomic_int64: enable

#line 1 "math"
uint add(uint x, uint y) {
  return x + y;
}

#define BLOCK_SIZE 256

#line 11 "shader"
void main() {
	gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
```


-  `shader.frag`

```glsl
#version 450

#define BLOCK_SIZE 256

#line 1 "VCTX_EXTENSION_LIST"
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_EXT_shader_atomic_int64: enable
#extension GL_KHR_shader_atomic_int64: enable
#extension GL_NV_shader_atomic_int64: enable

#line 1 "math"
uint add(uint x, uint y) {
  return x + y;
}

#line 17 "RESOURCE_BLOCK_DECL"
out vec4 fragColor;

#line 20 "shader"
void main() {
	fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
```

  

-  `shader.comp`

```glsl
#version 450

#line 1 "VCTX_EXTENSION_LIST"
#extension GL_ARB_gpu_shader_int64: enable
#extension GL_EXT_shader_atomic_int64: enable
#extension GL_KHR_shader_atomic_int64: enable
#extension GL_NV_shader_atomic_int64: enable

#line 1 "math"
uint add(uint x, uint y) {
  return x + y;
}

#define BLOCK_SIZE 256

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
#line 26 "shader"
void main() {
	uint gid = gl_GlobalInvocationID.x;
	// ...
}
```


## Usage

  

1.  **Install from source**
```bash
git clone https://github.com/priyangkarghosh/tlang.git
cd tlang
pip install -e .
```

2.  **Write your shaders in `.tlang` format**, combining multiple stages as needed.

  

4.  **Compile with tlang**

  
This generates stage-specific GLSL files (e.g. `shader.vert`, `shader.frag`, `shader.comp`).
```python
import  moderngl  as  mgl
from tlang import ShaderManager

ctx = mgl.create_context(require=460, standalone=True)
shader_manager = ShaderManager(
	context=ctx,           # moderngl context to use
	version='460 core',    # version string
	dir='shaders',         # relative or absolute path to the shaders to be built
	constants={            # dict of templates/constants
		'BLOCK_SIZE': 256,
	}
)

shader = shader_manager.get_shader('shader')
prog = shader.get_program('default') # vertex + fragment pipeline
kernel = shader.get_kernel('cs_randomize') # compute kernel
```

---
