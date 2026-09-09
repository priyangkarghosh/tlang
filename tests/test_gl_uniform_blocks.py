# -------------------------------------------------------------
# @file          test_gl_uniform_blocks.py
# @description   GL-marked integration test for uniform-block
#                bindings: builds a real compute shader combining a
#                `layout(std140) uniform` block with a `layout(std430)
#                buffer` block, allocates bindings through
#                BindingRegistry, links against the driver, and reads
#                back the reflected UniformBlock/StorageBlock binding
#                to prove the two pools are independent and correct --
#                not just that the source compiles.
# -------------------------------------------------------------

import struct

import pytest
from moderngl import StorageBlock, UniformBlock

from tlang.compiler.binding_registry import BindingRegistry
from tlang.shader_stages import ShaderStage

pytestmark = pytest.mark.gl


COMPUTE_SRC = '''\
#version 430 core
layout(local_size_x = 1) in;

layout(std140) uniform Params {
    float scale;
};

layout(std430) buffer Data {
    float values[];
};

void main() {
    values[0] = scale * 2.0;
}
'''


def test_uniform_and_ssbo_bindings_are_independent_and_verified_against_the_driver(gl_ctx):
    stage_sources = {ShaderStage.COMP: COMPUTE_SRC}
    patched, ssbo_canon, uniform_canon = BindingRegistry.allocate_artifact(
        gl_ctx, 'cs_ubo_test', stage_sources, {}
    )

    # separate pools: both land at index 0 in their own namespace
    assert ssbo_canon == {'Data': 0}
    assert uniform_canon == {'Params': 0}

    shader = gl_ctx.compute_shader(patched[ShaderStage.COMP])
    BindingRegistry.verify_link(shader, ssbo_canon, 'cs_ubo_test', uniform_canon)

    # the driver's own reflection agrees with what tlang assigned
    data_block = shader['Data']
    params_block = shader['Params']
    assert isinstance(data_block, StorageBlock)
    assert isinstance(params_block, UniformBlock)
    assert data_block.binding == ssbo_canon['Data']
    assert params_block.binding == uniform_canon['Params']

    # end-to-end: actually dispatch and read the result back, proving the
    # uniform block binding isn't just reflected correctly but functionally
    # wired -- a value written into the UBO at its bound index reaches the
    # shader and the computed result comes back out through the SSBO.
    ubo = gl_ctx.buffer(struct.pack('f', 21.0) + b'\x00' * 12)  # std140 padding to 16 bytes
    ssbo = gl_ctx.buffer(struct.pack('f', 0.0))

    ubo.bind_to_uniform_block(uniform_canon['Params'])
    ssbo.bind_to_storage_buffer(ssbo_canon['Data'])
    shader.run(1, 1, 1)

    (result,) = struct.unpack('f', ssbo.read())
    assert result == pytest.approx(42.0)

    ubo.release()
    ssbo.release()
    shader.release()


PROGRAM_VERT_SRC = '''\
#version 430 core
layout(std140) uniform Frame {
    float offset;
};

void main() {
    gl_Position = vec4(offset, 0.0, 0.0, 1.0);
}
'''

PROGRAM_FRAG_SRC = '''\
#version 430 core
layout(std140) uniform Frame {
    float offset;
};
out vec4 frag_color;

void main() {
    frag_color = vec4(offset, 0.0, 0.0, 1.0);
}
'''


def test_uniform_block_shared_across_vertex_and_fragment_gets_one_binding(gl_ctx):
    """A `Frame` UBO declared identically in both stages of one artifact
    must resolve to exactly one binding, shared everywhere it's declared
    -- the same guarantee `allocate_artifact` already gives SSBOs."""
    stage_sources = {ShaderStage.VERT: PROGRAM_VERT_SRC, ShaderStage.FRAG: PROGRAM_FRAG_SRC}
    patched, ssbo_canon, uniform_canon = BindingRegistry.allocate_artifact(
        gl_ctx, 'prog_ubo_test', stage_sources, {}
    )
    assert ssbo_canon == {}
    assert uniform_canon == {'Frame': 0}

    program = gl_ctx.program(
        vertex_shader=patched[ShaderStage.VERT],
        fragment_shader=patched[ShaderStage.FRAG],
    )
    BindingRegistry.verify_link(program, ssbo_canon, 'prog_ubo_test', uniform_canon)

    frame_block = program['Frame']
    assert isinstance(frame_block, UniformBlock)
    assert frame_block.binding == uniform_canon['Frame']

    program.release()
