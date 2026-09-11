# -------------------------------------------------------------
# @file          test_pipeline_binding_state.py
# @description   GL-marked regression tests for Pipeline.render's re-assert: closes the
#                cross-wiring hole Pipeline.bind_ssbo's immediate bind alone leaves open (a
#                compute dispatch issued between binding a pipeline and drawing it can rewire
#                GL's process-global SSBO table out from under the pipeline -- see
#                Pipeline.bind_ssbo's docstring). render()/render_indirect()/transform() give
#                Pipeline the dispatch-like hook it never had to re-assert its own recorded set
#                at, mirroring Kernel's dispatch-time asserts exactly.
#
#                XWIRE_SRC declares a [program(...)] and an unrelated compute kernel whose
#                single SSBO blocks have different names but, since binding allocation restarts
#                at 0 for every artifact, land at the SAME numeric GL binding -- the exact
#                precondition test_kernel_binding_state.py uses for the Kernel side of this bug.
# -------------------------------------------------------------

import struct

import moderngl
import pytest

from tlang.errors import TlangBindingError

pytestmark = pytest.mark.gl


XWIRE_SRC = '''\
layout(std430) buffer PipelineColor { vec4 color; };

[program('draw', vert='vs_main', frag='fs_main')]

[shader('vertex')]
[resourceblock(
    in vec2 vert;
)]
void vs_main() {
    gl_Position = vec4(vert, 0.0, 1.0);
}

[shader('fragment')]
[resourceblock(
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = color;
}

layout(std430) buffer KernelJunk { vec4 junk; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_clobber() {
    junk = vec4(10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0, 40.0 / 255.0);
}
'''

FLAT_SRC = '''\
[program('flat', vert='vs_flat', frag='fs_flat')]

[shader('vertex')]
[resourceblock(
    in vec2 vert;
)]
void vs_flat() {
    gl_Position = vec4(vert, 0.0, 1.0);
}

[shader('fragment')]
[resourceblock(
    out vec4 fragColor;
)]
void fs_flat() {
    fragColor = vec4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 255.0 / 255.0);
}
'''

QUAD = struct.pack('8f', -1, -1, 1, -1, -1, 1, 1, 1)


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    return sm.get_shader(name)


def test_pipeline_render_reasserts_after_intervening_kernel_dispatch(gl_ctx, make_shader_dir):
    """The headline reproduction. PipelineColor (draw's only SSBO) and KernelJunk (cs_clobber's
    only SSBO) are different names that both land at GL binding 0, since each artifact's
    BindingRegistry starts numbering at 0 independently. Binding the pipeline, dispatching the
    kernel, then calling pipeline.render(...) must still draw with the PIPELINE's buffer -- if
    render() skipped the re-assert, this would read cs_clobber's last-written contents instead."""
    shader = _build(gl_ctx, make_shader_dir, 'xwire_pipeline', XWIRE_SRC)
    pipeline = shader.get_pipeline('draw')
    kernel = shader.get_kernel('cs_clobber')

    assert pipeline.bindings == {'PipelineColor': 0}
    assert kernel.bindings == {'KernelJunk': 0}

    pipeline_buf = gl_ctx.buffer(struct.pack('4f', 50 / 255, 60 / 255, 70 / 255, 255 / 255))
    kernel_buf = gl_ctx.buffer(reserve=16)

    vbo = gl_ctx.buffer(QUAD)
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((1, 1), 4)])
    fbo.use()

    pipeline.bind_ssbo('PipelineColor', pipeline_buf)

    # Dispatched IN BETWEEN binding the pipeline and drawing it -- cs_clobber's dispatch writes
    # kernel_buf to the SAME GL binding index (0) PipelineColor is bound at.
    kernel.bind_ssbo('KernelJunk', kernel_buf)
    kernel.dispatch(1, 1, 1)

    pipeline.render(vao, moderngl.TRIANGLE_STRIP)

    assert list(fbo.color_attachments[0].read()) == [50, 60, 70, 255]

    vbo.release()
    vao.release()
    fbo.release()
    pipeline_buf.release()
    kernel_buf.release()


def test_pipeline_render_with_no_bindings_does_not_raise(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'flat', FLAT_SRC)
    pipeline = shader.get_pipeline('flat')
    assert pipeline.bindings == {}

    vbo = gl_ctx.buffer(QUAD)
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((1, 1), 4)])
    fbo.use()

    pipeline.render(vao, moderngl.TRIANGLE_STRIP)  # nothing required at all -- must not raise

    assert list(fbo.color_attachments[0].read()) == [64, 128, 192, 255]

    vbo.release()
    vao.release()
    fbo.release()


def test_pipeline_render_missing_required_buffer_raises_named(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'xwire_missing', XWIRE_SRC)
    pipeline = shader.get_pipeline('draw')

    vbo = gl_ctx.buffer(QUAD)
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((1, 1), 4)])
    fbo.use()

    with pytest.raises(TlangBindingError, match='PipelineColor'):
        pipeline.render(vao, moderngl.TRIANGLE_STRIP)

    vbo.release()
    vao.release()
    fbo.release()


def test_pipeline_bind_ssbo_immediate_bind_still_works_with_direct_vao_render(gl_ctx, make_shader_dir):
    """A caller who never adopts render() and keeps calling vao.render() directly must see no
    change: bind_ssbo's immediate bind alone is still enough when nothing else touches the
    table in between."""
    shader = _build(gl_ctx, make_shader_dir, 'xwire_direct', XWIRE_SRC)
    pipeline = shader.get_pipeline('draw')

    pipeline_buf = gl_ctx.buffer(struct.pack('4f', 50 / 255, 60 / 255, 70 / 255, 255 / 255))
    vbo = gl_ctx.buffer(QUAD)
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((1, 1), 4)])
    fbo.use()

    pipeline.bind_ssbo('PipelineColor', pipeline_buf)
    vao.render(moderngl.TRIANGLE_STRIP)

    assert list(fbo.color_attachments[0].read()) == [50, 60, 70, 255]

    vbo.release()
    vao.release()
    fbo.release()
    pipeline_buf.release()
