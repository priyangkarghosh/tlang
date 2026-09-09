# -------------------------------------------------------------
# @file          test_gl_interfaces.py
# @description   GL-marked integration tests for [varyings]/[uniforms]/
#                [buffer]/[uses(...)]: builds and links a real vert+frag
#                program, dispatches a compute kernel, and confirms the
#                V4 headline diagnostic replaces the driver's unlocated
#                link failure on a real typo.
# -------------------------------------------------------------

import struct

import pytest
from moderngl import UniformBlock

from tlang.errors import TlangAttributeError

pytestmark = pytest.mark.gl


VARYINGS_SRC = """\
[varyings]
struct VertexOut {{ vec3 color; vec2 uv; }};

[shader('vertex')]
[uses(VertexOut, dir='out')]
void vs_main() {{
    color = vec3(1.0, 0.0, 0.0);
    uv = vec2(0.0, 0.0);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}}

[shader('fragment')]
[uses(VertexOut, dir='in')]
[resourceblock(out vec4 fragColor;)]
void fs_main() {{
    fragColor = vec4(color, 1.0);
}}

[program('default', vert='vs_main', frag='fs_main')]
"""


def test_varyings_and_uses_link_a_real_program_with_matching_locations(gl_ctx, make_shader_dir):
    d = make_shader_dir({'demo.tlang': VARYINGS_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    assert sh.get_program('default') is not None

    # both sides agree on locations by construction (member order) -- the
    # generated GLSL itself is the source of truth, since moderngl doesn't
    # reflect inter-stage varyings as program members
    vs_source = sh.get_source('vs_main')
    fs_source = sh.get_source('fs_main')
    assert 'layout(location = 0) out vec3 color;' in vs_source
    assert 'layout(location = 1) out vec2 uv;' in vs_source
    assert 'layout(location = 0) in vec3 color;' in fs_source
    assert 'layout(location = 1) in vec2 uv;' in fs_source


def test_varying_name_typo_raises_located_error_instead_of_driver_link_failure(gl_ctx, make_shader_dir):
    """The headline scenario: today a one-character typo produces the
    driver's bare 'GLSL Linker failed' with no line number. This must
    instead raise TlangAttributeError naming both entry points, both interface
    names, and both source lines -- before the driver is ever asked to link."""
    bad_src = VARYINGS_SRC.format().replace(
        "[uses(VertexOut, dir='in')]", "[uses('VertexOu', dir='in')]",
    )
    d = make_shader_dir({'demo.tlang': bad_src})
    from tlang import ShaderManager

    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'vs_main' in msg
    assert 'fs_main' in msg
    assert 'VertexOut' in msg
    assert 'VertexOu' in msg
    assert 'demo:2' in msg
    assert 'default' in msg
    assert 'GLSL Linker failed' not in msg


BUFFER_SRC = """\
[buffer(std430)]
struct Data {{ float values[]; }};

[uniforms(std140)]
struct Params {{ float scale; }};

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_go() {{
    values[0] = scale * 2.0;
}}
"""


def test_buffer_and_uniforms_dispatch_a_real_compute_kernel(gl_ctx, make_shader_dir):
    d = make_shader_dir({'demo.tlang': BUFFER_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)
    sh = sm.get_shader('demo')
    kernel = sh.get_kernel('cs_go')

    assert 'Data' in kernel.bindings
    params_block = kernel.mglo['Params']
    assert isinstance(params_block, UniformBlock)

    data = gl_ctx.buffer(struct.pack('f', 0.0))
    params = gl_ctx.buffer(struct.pack('f', 21.0) + b'\x00' * 12)

    kernel.bind_ssbo('Data', data)
    params.bind_to_uniform_block(params_block.binding)
    kernel.dispatch(1, 1, 1)

    (result,) = struct.unpack('f', data.read())
    assert result == pytest.approx(42.0)

    data.release()
    params.release()


CONST_ARRAY_SRC = """\
[varyings(locations=false)]
struct Batch {{ vec4 items[{{{{ N }}}}]; }};

[shader('vertex')]
[uses(Batch, dir='out')]
void vs_main() {{
    items[0] = vec4(1.0);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}}

[shader('fragment')]
[uses(Batch, dir='in')]
[resourceblock(out vec4 fragColor;)]
void fs_main() {{
    fragColor = items[0];
}}

[program('default', vert='vs_main', frag='fs_main')]
"""


def test_constant_in_emitted_member_array_suffix_is_rendered(gl_ctx, make_shader_dir):
    """A {{ CONSTANT }} inside a member's array suffix, emitted into
    func.config by resolve_interfaces, must be substituted before reaching
    the driver -- silent if the ordering in ShaderManager is wrong."""
    d = make_shader_dir({'demo.tlang': CONST_ARRAY_SRC.format()})
    from tlang import ShaderManager

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True, constants={'N': 4})
    sh = sm.get_shader('demo')

    vs_source = sh.get_source('vs_main')
    assert '{{' not in vs_source
    assert 'items[4]' in vs_source
    assert sh.get_program('default') is not None
