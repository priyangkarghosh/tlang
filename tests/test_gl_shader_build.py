# -------------------------------------------------------------
# @file          test_gl_shader_build.py
# @description   GL-marked integration tests for ShaderManager/Shader:
#                constant substitution reaching kernel bodies at
#                runtime, cross-module [include]/[export], [link]ed
#                helpers, per-artifact SSBO binding allocation, and
#                explicit layout(binding=N) pins.
#
#                All tests in this module build ONE shared project
#                (see `main_shader`, module-scoped) so the (slow-ish)
#                compile+link work happens once and every regression
#                gets its own focused assertion against the result.
# -------------------------------------------------------------

import struct

import pytest

pytestmark = pytest.mark.gl


LIB_SRC = '''\
// Shared library: six SSBO blocks. Kept deliberately small for test
// speed, but enough to prove that per-artifact DCE + binding allocation
// narrows what each kernel actually gets -- a much larger shared library
// (100 blocks) used to blow the binding budget for every single kernel
// that included it ("RuntimeError: Out of SSBO bindings!"), even though
// any one kernel only ever touched a couple of these blocks.
layout(std430) buffer BufA { uint a_data[]; };
layout(std430) buffer BufB { uint b_data[]; };
layout(std430) buffer BufC { uint c_data[]; };
layout(std430) buffer BufD { uint d_data[]; };
layout(std430) buffer BufE { uint e_data[]; };
layout(std430) buffer BufF { uint f_data[]; };

[export]
uint helper_add(uint x, uint y) {
    return x + y;
}
'''

MAIN_SRC = '''\
[include(lib)]

#define BLOCK_SIZE {{ BLOCK_SIZE }}

uint scaled(uint x) {
    uint result = x;
    [unroll]
    for (int i = 0; i < 1; i++) {
        result = result * BLOCK_SIZE;
    }
    return result;
}

[export]
uint doubled(uint x) {
    return x * 2u;
}

layout(std430, binding = 5) buffer Pinned { uint pinned_data[]; };
layout(std430) buffer AutoAssigned { uint auto_data[]; };

[shader('compute')]
[link('scaled')]
[numthreads(BLOCK_SIZE, 1, 1)]
void cs_const() {
    if (gl_LocalInvocationID.x == 0u) {
        a_data[0] = scaled(7u);
    }
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_kernel_b() {
    b_data[0] = helper_add(1u, 2u);
    c_data[0] = 0u;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_kernel_c() {
    d_data[0] = 0u;
    e_data[0] = 0u;
}

[shader('compute')]
[link('doubled')]
[numthreads(1, 1, 1)]
void cs_export_link() {
    f_data[0] = doubled(4u);
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_pins() {
    pinned_data[0] = 1u;
    auto_data[0] = 2u;
}
'''


@pytest.fixture(scope='module')
def main_shader(gl_ctx, tmp_path_factory):
    from tlang import ShaderManager

    d = tmp_path_factory.mktemp('main_project')
    (d / 'lib.tlang').write_text(LIB_SRC, encoding='utf-8')
    (d / 'main.tlang').write_text(MAIN_SRC, encoding='utf-8')

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), constants={'BLOCK_SIZE': 4})
    shader = sm.get_shader('main')
    assert shader is not None
    return shader


def _read_u32(gl_ctx, buf) -> int:
    (value,) = struct.unpack('I', buf.read(4))
    return value


def test_constants_reach_kernel_bodies_at_runtime(gl_ctx, main_shader):
    """`{{ BLOCK_SIZE }}` must substitute in text (no literal '{{' left in
    the generated source) AND the value must be real at runtime -- proven
    by dispatching the kernel and reading back the result, not just by
    inspecting text."""
    src = main_shader.get_source('cs_const')
    assert '{{' not in src
    assert '}}' not in src

    buf = gl_ctx.buffer(reserve=4)
    kernel = main_shader.get_kernel('cs_const')
    kernel.bind_ssbo('BufA', buf)
    kernel.dispatch(1, 1, 1)

    assert _read_u32(gl_ctx, buf) == 7 * 4  # BLOCK_SIZE == 4


def test_linked_helper_emits_pragma_and_substituted_constant(gl_ctx, main_shader):
    """[link('scaled')] must inline 'scaled's PROCESSED text: the
    [unroll] pragma emitted as '#pragma unroll' (not the literal
    attribute text), and its `{{ }}`-templated body already substituted."""
    src = main_shader.get_source('cs_const')
    assert '#pragma unroll' in src
    assert '[unroll' not in src
    assert 'uint scaled(uint x)' in src


def test_include_export_cross_module_call_dispatches_correctly(gl_ctx, main_shader):
    """A function [export]ed from one module and reached via [include] in
    another must actually run correctly on the GPU, not just compile."""
    buf_b = gl_ctx.buffer(reserve=4)
    buf_c = gl_ctx.buffer(reserve=4)
    kernel = main_shader.get_kernel('cs_kernel_b')
    kernel.bind_ssbo('BufB', buf_b)
    kernel.bind_ssbo('BufC', buf_c)
    kernel.dispatch(1, 1, 1)

    assert _read_u32(gl_ctx, buf_b) == 1 + 2  # helper_add(1, 2), from lib.tlang


def test_export_and_link_on_same_helper_yields_exactly_one_definition(gl_ctx, main_shader):
    """'doubled' is both [export]ed (folded into the shared module) AND
    targeted by [link('doubled')] from cs_export_link in the SAME module.
    It must appear exactly once in the generated source (a duplicate
    would be a GLSL redefinition error), and the kernel must actually
    compile and run."""
    src = main_shader.get_source('cs_export_link')
    assert src.count('uint doubled(uint x)') == 1

    buf_f = gl_ctx.buffer(reserve=4)
    kernel = main_shader.get_kernel('cs_export_link')
    kernel.bind_ssbo('BufF', buf_f)
    kernel.dispatch(1, 1, 1)
    assert _read_u32(gl_ctx, buf_f) == 4 * 2


def test_per_artifact_bindings_stay_small_despite_shared_library(gl_ctx, main_shader):
    """Regression: a shared library declaring many SSBOs used to blow the
    binding budget for every kernel that included it, even though any one
    kernel only ever references a couple of those blocks after DCE. Each
    kernel's `.bindings` must reflect only what THAT kernel actually uses."""
    assert set(main_shader.get_kernel('cs_const').bindings) == {'BufA'}
    assert set(main_shader.get_kernel('cs_kernel_b').bindings) == {'BufB', 'BufC'}
    assert set(main_shader.get_kernel('cs_kernel_c').bindings) == {'BufD', 'BufE'}
    assert set(main_shader.get_kernel('cs_export_link').bindings) == {'BufF'}


def test_explicit_binding_pin_is_honoured_and_does_not_collide(gl_ctx, main_shader):
    """An explicit `layout(binding = 5)` pin must be kept exactly at 5,
    and the auto-assigned neighbour block must never be handed the same
    index."""
    bindings = main_shader.get_kernel('cs_pins').bindings
    assert bindings['Pinned'] == 5
    assert bindings['AutoAssigned'] != 5

    pinned_buf = gl_ctx.buffer(reserve=4)
    auto_buf = gl_ctx.buffer(reserve=4)
    kernel = main_shader.get_kernel('cs_pins')
    kernel.bind_ssbo('Pinned', pinned_buf)
    kernel.bind_ssbo('AutoAssigned', auto_buf)
    kernel.dispatch(1, 1, 1)

    assert _read_u32(gl_ctx, pinned_buf) == 1
    assert _read_u32(gl_ctx, auto_buf) == 2


# ---------------------------------------------------------------------------
# Graphics [program(...)] + Pipeline.bind_ssbo
# ---------------------------------------------------------------------------

GFX_SRC = '''\
layout(std430) buffer FragData { uint frag_vals[]; };

[program('gfx', vert='vs_m', frag='fs_m')]

[shader('vertex')]
void vs_m() {
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}

[shader('fragment')]
[resourceblock(
    out vec4 fragColor;
)]
void fs_m() {
    fragColor = vec4(float(frag_vals[0]));
}
'''


def test_graphics_program_links_and_binds_ssbo(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'gfx.tlang': GFX_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('gfx')
    assert shader is not None

    pipeline = shader.get_pipeline('gfx')
    buf = gl_ctx.buffer(reserve=4)
    pipeline.bind_ssbo('FragData', buf)  # must not raise
