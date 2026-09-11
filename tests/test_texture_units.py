# -------------------------------------------------------------
# @file          test_texture_units.py
# @description   Sampler/image binding allocation: discovery, per-pool allocation (mirroring
#                BindingRegistry's buffer/uniform pools), reflection-based unit assignment (NOT
#                textual patching -- samplers/images reflect as a plain `Uniform` with a
#                writable `.value`), and the Kernel/Pipeline `bind_texture`/`bind_image`
#                ergonomics this all exists to support.
#
#                Split like `test_uniform_block_bindings.py`/`test_gl_uniform_blocks.py`: a
#                GL-free tier exercising BindingRegistry directly against a fake context, and a
#                GL-marked tier building real kernels/pipelines and proving the right texture/
#                image data actually arrives via readback -- not a mock.
# -------------------------------------------------------------

import struct

import moderngl
import pytest

import tlang.compiler.binding_registry as br
from tlang.compiler.binding_registry import BindingRegistry
from tlang.errors import TlangBindingError
from tlang.shader_stages import ShaderStage


class _FakeCtx:
    def __init__(self, info: dict):
        self.info = info


def _ctx(**overrides) -> _FakeCtx:
    info = {'GL_MAX_TEXTURE_IMAGE_UNITS': 32}
    info.update(overrides)
    return _FakeCtx(info)


def _comp_src(*decls: str) -> dict[ShaderStage, str]:
    return {ShaderStage.COMP: '\n'.join(decls) + '\nvoid main() {}\n'}


# ---------------------------------------------------------------------------
# GL-free: discovery + allocation
# ---------------------------------------------------------------------------

def test_two_samplers_get_distinct_units():
    """The headline collision case this feature exists to close: without tlang, both `a` and
    `b` would default to unit 0 and silently alias -- see the module docstring's measured
    facts. Allocation must give them two different units."""
    ctx = _ctx()
    stage_sources = _comp_src("uniform sampler2D a;", "uniform sampler2D b;")
    texture_canon, image_canon = BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)
    assert image_canon == {}
    assert len(texture_canon) == 2
    assert texture_canon['a'] != texture_canon['b']


def test_explicit_binding_pin_is_honoured_and_not_moved():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(binding = 5) uniform sampler2D pinned;",
        "uniform sampler2D other;",
    )
    texture_canon, _ = BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)
    assert texture_canon['pinned'] == 5
    assert texture_canon['other'] != 5


def test_two_names_pinned_to_same_unit_raises():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(binding = 2) uniform sampler2D a;",
        "layout(binding = 2) uniform sampler2D b;",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)


def test_texture_and_image_pools_are_independent():
    """Both pools start numbering from 0 -- a sampler and an image in the same stage must each
    land at 0 in their own pool, proving the two are never merged."""
    ctx = _ctx()
    stage_sources = _comp_src(
        "uniform sampler2D tex;",
        "layout(rgba8) uniform image2D img;",
    )
    texture_canon, image_canon = BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)
    assert texture_canon == {'tex': 0}
    assert image_canon == {'img': 0}


def test_texture_unit_pool_exhaustion_raises():
    ctx = _ctx(GL_MAX_TEXTURE_IMAGE_UNITS=1)
    stage_sources = _comp_src("uniform sampler2D a;", "uniform sampler2D b;")
    with pytest.raises(TlangBindingError):
        BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)


def test_image_unit_ceiling_falls_back_when_driver_omits_it(caplog):
    """GL_MAX_IMAGE_UNITS is not reported at all on the reference machine this feature was
    measured against -- `_max_pool`'s existing fallback path must cover it, loudly."""
    import logging

    ctx = _ctx()  # no GL_MAX_IMAGE_UNITS key
    stage_sources = _comp_src("layout(rgba8) uniform image2D img;")
    with caplog.at_level(logging.WARNING):
        _, image_canon = BindingRegistry.allocate_opaque_units(ctx, 'art', stage_sources)
    assert image_canon == {'img': 0}
    assert any('GL_MAX_IMAGE_UNITS' in rec.message for rec in caplog.records)


def test_commented_out_sampler_declaration_is_not_picked_up():
    """Discovery reuses `_mask` -- a declaration inside a comment must never be scanned."""
    ctx = _ctx()
    src = (
        "/* uniform sampler2D ghost; */\n"
        "uniform sampler2D live;\n"
    )
    texture_canon, _ = BindingRegistry.allocate_opaque_units(ctx, 'art', {ShaderStage.COMP: src + 'void main(){}\n'})
    assert texture_canon == {'live': 0}
    assert 'ghost' not in texture_canon


class _FakeUniform:
    def __init__(self):
        self.value = None


class _FakeLinked:
    def __init__(self, members):
        self._members = members

    def get(self, name, default=None):
        return self._members.get(name, default)


def test_assign_opaque_units_sets_reflected_uniform_value(monkeypatch):
    monkeypatch.setattr(br, 'Uniform', _FakeUniform)
    tex_uniform = _FakeUniform()
    linked = _FakeLinked({'tex': tex_uniform})
    BindingRegistry.assign_opaque_units(linked, {'tex': 3}, {})
    assert tex_uniform.value == 3


def test_assign_opaque_units_skips_a_name_reflection_cannot_find(monkeypatch):
    """GL strips inactive (declared-but-unreferenced) uniforms -- assignment must not raise for
    a canon name the linked program no longer reflects."""
    monkeypatch.setattr(br, 'Uniform', _FakeUniform)
    linked = _FakeLinked({})  # 'stripped' never reflects
    BindingRegistry.assign_opaque_units(linked, {'stripped': 0}, {})  # must not raise


# ---------------------------------------------------------------------------
# GL-marked: real dispatch/render proving the right unit AND the right data arrive
# ---------------------------------------------------------------------------


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    return sm.get_shader(name)


TWO_SAMPLER_SRC = '''\
uniform sampler2D texA;
uniform sampler2D texB;
layout(std430) buffer Result { uint out_a; uint out_b; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_sample() {
    out_a = uint(texelFetch(texA, ivec2(0, 0), 0).r * 255.0 + 0.5);
    out_b = uint(texelFetch(texB, ivec2(0, 0), 0).r * 255.0 + 0.5);
}
'''


@pytest.mark.gl
def test_two_samplers_in_one_artifact_get_distinct_units_and_dispatch_correctly(gl_ctx, make_shader_dir):
    """End to end: two sampler uniforms with no explicit binding get distinct units, and
    `bind_texture` routes each by NAME to the driver-verified correct texture -- proven by real
    texel readback, not just reflection."""
    shader = _build(gl_ctx, make_shader_dir, 'two_samplers', TWO_SAMPLER_SRC)
    kernel = shader.get_kernel('cs_sample')

    assert set(kernel.texture_units) == {'texA', 'texB'}
    assert kernel.texture_units['texA'] != kernel.texture_units['texB']

    tex_a = gl_ctx.texture((1, 1), 1, data=bytes([100]))
    tex_b = gl_ctx.texture((1, 1), 1, data=bytes([200]))
    result = gl_ctx.buffer(reserve=8)

    kernel.bind_texture('texA', tex_a)
    kernel.bind_texture('texB', tex_b)
    kernel.bind_ssbo('Result', result)
    kernel.dispatch(1, 1, 1)

    out_a, out_b = struct.unpack('II', result.read())
    assert out_a == 100
    assert out_b == 200

    # swap which texture is bound to which name -- the readback must swap too, proving this is
    # genuinely routed by name and not by declaration order or GL's own default (unit 0 for both).
    kernel.bind_texture('texA', tex_b)
    kernel.bind_texture('texB', tex_a)
    kernel.dispatch(1, 1, 1)
    out_a, out_b = struct.unpack('II', result.read())
    assert out_a == 200
    assert out_b == 100

    tex_a.release()
    tex_b.release()
    result.release()


PINNED_SRC = '''\
layout(binding = 3) uniform sampler2D pinned;
uniform sampler2D other;
layout(std430) buffer Out { float x; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_pinned() {
    x = texelFetch(pinned, ivec2(0, 0), 0).r + texelFetch(other, ivec2(0, 0), 0).r;
}
'''


@pytest.mark.gl
def test_explicit_layout_binding_pin_is_honoured_in_real_build(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'pinned', PINNED_SRC)
    kernel = shader.get_kernel('cs_pinned')

    assert kernel.texture_units['pinned'] == 3
    assert kernel.texture_units['other'] != 3
    # the driver's own reflection agrees with what tlang assigned via .value
    assert kernel.mglo['pinned'].value == 3
    assert kernel.mglo['other'].value == kernel.texture_units['other']


CONFLICT_SRC = '''\
layout(binding = 1) uniform sampler2D a;
layout(binding = 1) uniform sampler2D b;
layout(std430) buffer Out { float x; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_conflict() {
    x = texelFetch(a, ivec2(0, 0), 0).r + texelFetch(b, ivec2(0, 0), 0).r;
}
'''


@pytest.mark.gl
def test_two_names_pinned_to_same_unit_raises_at_build_time(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'conflict.tlang': CONFLICT_SRC})
    with pytest.raises(TlangBindingError):
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))


MIXED_SRC = '''\
layout(std430) buffer Out { uint result; };
layout(rgba8) uniform image2D img;
uniform sampler2D tex;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_mixed() {
    vec4 v = texelFetch(tex, ivec2(0, 0), 0);
    imageStore(img, ivec2(0, 0), v);
    result = 1u;
}
'''


@pytest.mark.gl
def test_image_and_sampler_do_not_contend_for_same_pool(gl_ctx, make_shader_dir):
    """`tex` and `img` land at unit 0 in their OWN pools -- proven not just by the canon (see
    the GL-free version above) but by a real imageStore/texelFetch round trip: if the pools
    were merged, one of the two units would be stolen by the other and this would read back
    zeros instead of the source texture's data."""
    shader = _build(gl_ctx, make_shader_dir, 'mixed', MIXED_SRC)
    kernel = shader.get_kernel('cs_mixed')

    assert kernel.texture_units == {'tex': 0}
    assert kernel.image_units == {'img': 0}

    src_tex = gl_ctx.texture((1, 1), 4, data=bytes([10, 20, 30, 255]))
    dst_tex = gl_ctx.texture((1, 1), 4, dtype='f1')
    out = gl_ctx.buffer(reserve=4)

    kernel.bind_texture('tex', src_tex)
    kernel.bind_image('img', dst_tex, read=False, write=True)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(1, 1, 1)

    assert list(dst_tex.read()) == [10, 20, 30, 255]
    (flag,) = struct.unpack('I', out.read())
    assert flag == 1

    src_tex.release()
    dst_tex.release()
    out.release()


UNUSED_SAMPLER_SRC = '''\
uniform sampler2D used;
uniform sampler2D unused;
layout(std430) buffer Out { float x; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_unused() {
    x = texelFetch(used, ivec2(0, 0), 0).r;
}
'''


@pytest.mark.gl
def test_declared_but_unused_sampler_does_not_break_build(gl_ctx, make_shader_dir):
    """GL strips an inactive uniform from reflection, so `unused` never reflects as a real
    `Uniform` -- assignment (and the build as a whole) must not raise over that."""
    shader = _build(gl_ctx, make_shader_dir, 'unused_sampler', UNUSED_SAMPLER_SRC)
    assert shader.ok
    kernel = shader.get_kernel('cs_unused')
    assert 'used' in kernel.texture_units


@pytest.mark.gl
def test_set_uniform_still_works_on_sampler_name(gl_ctx, make_shader_dir):
    """The escape hatch `set_uniform`/`set_uniforms` must keep working on a sampler name --
    tlang's own unit assignment must not have claimed exclusive ownership of `.value`."""
    shader = _build(gl_ctx, make_shader_dir, 'set_uniform_sampler', TWO_SAMPLER_SRC)
    kernel = shader.get_kernel('cs_sample')

    kernel.set_uniform('texA', 7)
    assert kernel.mglo['texA'].value == 7

    kernel.set_uniforms(texB=6)
    assert kernel.mglo['texB'].value == 6


RASTER_SRC = '''\
[program('draw', vert='vs_main', frag='fs_main')]

[shader('vertex')]
[resourceblock(
    in vec2 vert;
    out vec2 uv;
)]
void vs_main() {
    uv = vert * 0.5 + 0.5;
    gl_Position = vec4(vert, 0.0, 1.0);
}

[shader('fragment')]
[resourceblock(
    in vec2 uv;
    uniform sampler2D tex;
    out vec4 fragColor;
)]
void fs_main() {
    fragColor = texture(tex, uv);
}
'''


@pytest.mark.gl
def test_pipeline_bind_texture_routes_correct_texture_via_render(gl_ctx, make_shader_dir):
    """The raster path matters at least as much as compute here -- samplers are overwhelmingly
    a fragment-stage thing (see the radiance-cascade reference this feature was scoped from).
    `Pipeline.bind_texture` binds immediately (no dispatch hook to defer to, same asymmetry as
    `Pipeline.bind_ssbo`); prove it actually wires the right texture with a real render +
    framebuffer readback."""
    from tlang import ShaderManager

    d = make_shader_dir({'draw.tlang': RASTER_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('draw')
    pipeline = shader.get_pipeline('draw')

    assert set(pipeline.texture_units) == {'tex'}

    vbo = gl_ctx.buffer(struct.pack('8f', -1, -1, 1, -1, -1, 1, 1, 1))
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])

    tex = gl_ctx.texture((1, 1), 4, data=bytes([50, 60, 70, 255]))
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((1, 1), 4)])
    fbo.use()

    pipeline.bind_texture('tex', tex)
    vao.render(moderngl.TRIANGLE_STRIP)

    assert list(fbo.color_attachments[0].read()) == [50, 60, 70, 255]

    vbo.release()
    vao.release()
    tex.release()
    fbo.release()
