# -------------------------------------------------------------
# @file          test_atomic_counters.py
# @description   Atomic counter binding allocation: discovery, 2-D (binding, offset) allocation
#                (mirroring BindingRegistry's other pools but packing unpinned counters into one
#                binding at successive 4-byte offsets, since two counters sharing a binding at
#                different offsets is the idiomatic GLSL form), textual patching (atomic counters
#                are invisible to moderngl reflection, unlike samplers/images, so an unpinned
#                declaration's binding/offset MUST be injected into the GLSL text), post-link
#                pruning via raw pyOpenGL (the only way to learn which declared counters a linked
#                program actually kept active), and the Kernel/Pipeline `bind_counter` ergonomics.
#
#                Split like `test_texture_units.py`: a GL-free tier exercising BindingRegistry
#                directly against a fake context, and a GL-marked tier building real
#                kernels/pipelines and proving the right counter values actually arrive via
#                readback -- not a mock.
# -------------------------------------------------------------

import logging
import struct

import pytest

import tlang.compiler.binding_registry as br
from tlang.compiler.binding_registry import BindingRegistry
from tlang.errors import TlangBindingError
from tlang.shader_stages import ShaderStage


class _FakeCtx:
    def __init__(self, info: dict):
        self.info = info


def _ctx(**overrides) -> _FakeCtx:
    info = {'GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS': 8}
    info.update(overrides)
    return _FakeCtx(info)


def _comp_src(*decls: str) -> dict[ShaderStage, str]:
    return {ShaderStage.COMP: '\n'.join(decls) + '\nvoid main() {}\n'}


# ---------------------------------------------------------------------------
# GL-free: discovery + allocation
# ---------------------------------------------------------------------------

def test_two_counters_in_one_artifact_get_distinct_binding_offset():
    """Two unpinned counters must be packed into ONE binding (that's what GL is designed for --
    see the module docstring) at two distinct 4-byte offsets, never sharing a (binding, offset)
    pair."""
    ctx = _ctx()
    stage_sources = _comp_src("uniform atomic_uint a;", "uniform atomic_uint b;")
    canon = BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)
    assert set(canon) == {'a', 'b'}
    assert canon['a'][0] == canon['b'][0]  # packed into the same binding
    assert canon['a'][1] != canon['b'][1]  # at distinct offsets
    assert {canon['a'][1], canon['b'][1]} == {0, 4}


def test_explicit_pin_is_honoured_and_not_moved():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(binding = 3, offset = 0) uniform atomic_uint pinned;",
        "uniform atomic_uint other;",
    )
    canon = BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)
    assert canon['pinned'] == (3, 0)
    assert canon['other'][0] != 3


def test_two_counters_sharing_one_binding_at_different_offsets_is_fine():
    """The idiomatic packed form, pinned explicitly: no conflict."""
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(binding = 0, offset = 0) uniform atomic_uint a;",
        "layout(binding = 0, offset = 4) uniform atomic_uint b;",
    )
    canon = BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)
    assert canon == {'a': (0, 0), 'b': (0, 4)}


def test_two_counters_pinned_to_same_binding_and_offset_raises():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(binding = 0, offset = 0) uniform atomic_uint a;",
        "layout(binding = 0, offset = 0) uniform atomic_uint b;",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)


def test_explicit_pin_past_ceiling_raises():
    ctx = _ctx(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS=2)
    stage_sources = _comp_src("layout(binding = 5, offset = 0) uniform atomic_uint a;")
    with pytest.raises(TlangBindingError):
        BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)


def test_counter_binding_pool_exhaustion_raises():
    """One binding pinned, ceiling of 1 total -- the unpinned counter has nowhere left to pack
    into."""
    ctx = _ctx(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS=1)
    stage_sources = _comp_src(
        "layout(binding = 0, offset = 0) uniform atomic_uint pinned;",
        "uniform atomic_uint unpinned;",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)


def test_counter_buffer_ceiling_falls_back_when_driver_omits_it(caplog):
    """GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS is not reported at all on the reference machine this
    feature was measured against -- `_max_pool`'s existing fallback path must cover it, loudly."""
    ctx = _ctx()  # no GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS key
    stage_sources = _comp_src("uniform atomic_uint a;")
    with caplog.at_level(logging.WARNING):
        canon = BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)
    assert canon == {'a': (0, 0)}
    assert any('GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS' in rec.message for rec in caplog.records)


def test_stage_over_counter_buffer_limit_raises():
    """Two counters explicitly pinned to two different bindings in one stage, but the driver
    (per this fake ctx) allows only one distinct counter-buffer binding per stage."""
    ctx = _ctx(GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS=1)
    stage_sources = _comp_src(
        "layout(binding = 0, offset = 0) uniform atomic_uint a;",
        "layout(binding = 1, offset = 0) uniform atomic_uint b;",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry._allocate_counter_canon(ctx, 'art', stage_sources)


def test_commented_out_counter_declaration_is_not_picked_up():
    """Discovery reuses `_mask` -- a declaration inside a comment must never be scanned."""
    ctx = _ctx()
    src = (
        "/* uniform atomic_uint ghost; */\n"
        "uniform atomic_uint live;\n"
    )
    canon = BindingRegistry._allocate_counter_canon(ctx, 'art', {ShaderStage.COMP: src + 'void main(){}\n'})
    assert canon == {'live': (0, 0)}
    assert 'ghost' not in canon


# ---------------------------------------------------------------------------
# GL-free: textual patching
# ---------------------------------------------------------------------------

def test_patch_injects_layout_on_unpinned_declaration():
    src = "uniform atomic_uint a;\nvoid main() {}\n"
    patched = BindingRegistry._patch_counter_bindings(src, {'a': (2, 4)})
    assert "layout(binding = 2, offset = 4) uniform atomic_uint a;" in patched


def test_patch_leaves_an_existing_explicit_pin_untouched():
    src = "layout(binding = 1, offset = 0) uniform atomic_uint a;\nvoid main() {}\n"
    patched = BindingRegistry._patch_counter_bindings(src, {'a': (1, 0)})
    assert patched == src


def test_patch_leaves_a_name_absent_from_canon_untouched():
    """A declaration inside a comment never enters the canon (see the discovery test above), so
    patching must not touch it either -- it's masked-scan-then-raw-patch, exactly like
    `_patch_bindings`."""
    src = "/* uniform atomic_uint ghost; */\nuniform atomic_uint live;\nvoid main() {}\n"
    patched = BindingRegistry._patch_counter_bindings(src, {'live': (0, 0)})
    assert "/* uniform atomic_uint ghost; */" in patched
    assert "layout(binding = 0, offset = 0) uniform atomic_uint live;" in patched


# ---------------------------------------------------------------------------
# GL-free: post-link pruning via raw pyOpenGL (monkeypatched)
# ---------------------------------------------------------------------------

def test_active_atomic_counter_bindings_queries_raw_opengl(monkeypatch):
    calls = []

    def fake_interface_query(prog, interface, pname):
        calls.append((prog, interface, pname))
        return 2

    def fake_resource_query(prog, interface, index, prop_count, props, buf_size, length, params):
        params[0] = index * 3  # arbitrary distinct binding per resource index

    monkeypatch.setattr(br, 'glGetProgramInterfaceiv', fake_interface_query)
    monkeypatch.setattr(br, 'glGetProgramResourceiv', fake_resource_query)

    class _FakeLinked:
        glo = 42

    bindings = BindingRegistry.active_atomic_counter_bindings(_FakeLinked())
    assert bindings == {0, 3}
    assert calls  # the interface query was actually made


# ---------------------------------------------------------------------------
# GL-marked: real dispatch/render proving the right data actually arrives
# ---------------------------------------------------------------------------


def _build(gl_ctx, make_shader_dir, name, src):
    from tlang import ShaderManager

    d = make_shader_dir({f'{name}.tlang': src})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    return sm.get_shader(name)


TWO_COUNTER_SRC = '''\
layout(std430) buffer Out { uint a_val; uint b_val; };
uniform atomic_uint a;
uniform atomic_uint b;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_count() {
    atomicCounterIncrement(a);
    atomicCounterIncrement(a);
    atomicCounterIncrement(b);
    a_val = atomicCounter(a);
    b_val = atomicCounter(b);
}
'''


@pytest.mark.gl
def test_two_counters_in_one_artifact_get_distinct_binding_offset_and_dispatch_correctly(gl_ctx, make_shader_dir):
    """End to end: two unpinned counters are packed into one binding at distinct offsets, and
    `bind_counter` routes each by NAME to the right 4 bytes of the SAME buffer -- proven by real
    readback, not just the canon."""
    shader = _build(gl_ctx, make_shader_dir, 'two_counters', TWO_COUNTER_SRC)
    kernel = shader.get_kernel('cs_count')

    assert set(kernel.atomic_counters) == {'a', 'b'}
    binding_a, offset_a = kernel.atomic_counters['a']
    binding_b, offset_b = kernel.atomic_counters['b']
    assert binding_a == binding_b
    assert offset_a != offset_b

    counters = gl_ctx.buffer(reserve=8)
    out = gl_ctx.buffer(reserve=8)

    kernel.bind_counters(a=counters, b=counters)
    kernel.bind_ssbo('Out', out)
    kernel.dispatch(1, 1, 1)

    a_val, b_val = struct.unpack('II', out.read())
    assert a_val == 2  # atomicCounterIncrement returns the PRE-increment value; called twice: 0, then 1 -> final read is 2
    assert b_val == 1

    counters.release()
    out.release()


PINNED_SRC = '''\
layout(std430) buffer Out { uint x; };
layout(binding = 2, offset = 0) uniform atomic_uint pinned;
uniform atomic_uint other;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_pinned() {
    x = atomicCounterIncrement(pinned) + atomicCounterIncrement(other);
}
'''


@pytest.mark.gl
def test_explicit_pin_is_honoured_in_real_build(gl_ctx, make_shader_dir):
    shader = _build(gl_ctx, make_shader_dir, 'pinned_counter', PINNED_SRC)
    kernel = shader.get_kernel('cs_pinned')

    assert kernel.atomic_counters['pinned'] == (2, 0)
    assert kernel.atomic_counters['other'][0] != 2


CONFLICT_SRC = '''\
layout(std430) buffer Out { uint x; };
layout(binding = 1, offset = 0) uniform atomic_uint a;
layout(binding = 1, offset = 0) uniform atomic_uint b;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_conflict() {
    x = atomicCounterIncrement(a) + atomicCounterIncrement(b);
}
'''


@pytest.mark.gl
def test_conflicting_pin_raises_at_build_time(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'conflict_counter.tlang': CONFLICT_SRC})
    with pytest.raises(TlangBindingError):
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))


UNUSED_COUNTER_SRC = '''\
layout(std430) buffer Out { uint x; };
layout(binding = 0, offset = 0) uniform atomic_uint used;
uniform atomic_uint unused;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_unused_counter() {
    x = atomicCounterIncrement(used);
}
'''


@pytest.mark.gl
def test_declared_but_unused_counter_does_not_break_dispatch(gl_ctx, make_shader_dir):
    """`unused` is pinned to a DIFFERENT binding than `used` (so packing can't accidentally keep
    it alive by association -- see the module docstring's pruning-is-per-binding note), and GL
    drops that whole binding from its ATOMIC_COUNTER_BUFFER interface once linked. It must be
    pruned from `kernel.atomic_counters` entirely, and a real dispatch binding only `used` must
    not raise."""
    shader = _build(gl_ctx, make_shader_dir, 'unused_counter', UNUSED_COUNTER_SRC)
    assert shader.ok
    kernel = shader.get_kernel('cs_unused_counter')

    assert 'used' in kernel.atomic_counters
    assert 'unused' not in kernel.atomic_counters

    buf = gl_ctx.buffer(reserve=4)
    out = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('Out', out)
    kernel.bind_counter('used', buf)
    kernel.dispatch(1, 1, 1)  # must not raise about 'used' or 'unused'

    (val,) = struct.unpack('I', out.read())
    assert val == 0  # atomicCounterIncrement's pre-increment return value, first call

    buf.release()
    out.release()


NO_COUNTER_SRC = '''\
layout(std430) buffer Out { uint x; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_no_counter() {
    x = 1u;
}
'''


@pytest.mark.gl
def test_barrier_bit_is_added_automatically_only_when_a_counter_is_declared(gl_ctx, make_shader_dir):
    """U6: a kernel whose artifact declares a genuinely-used atomic counter must default to
    including ATOMIC_COUNTER_BARRIER_BIT; one that declares none must not."""
    from moderngl import ATOMIC_COUNTER_BARRIER_BIT, SHADER_STORAGE_BARRIER_BIT

    with_counter = _build(gl_ctx, make_shader_dir, 'with_counter', PINNED_SRC).get_kernel('cs_pinned')
    without_counter = _build(gl_ctx, make_shader_dir, 'without_counter', NO_COUNTER_SRC).get_kernel('cs_no_counter')

    assert with_counter.default_barrier_bits & ATOMIC_COUNTER_BARRIER_BIT
    assert not (without_counter.default_barrier_bits & ATOMIC_COUNTER_BARRIER_BIT)
    # SHADER_STORAGE_BARRIER_BIT is always present regardless -- unchanged default behaviour
    assert with_counter.default_barrier_bits & SHADER_STORAGE_BARRIER_BIT
    assert without_counter.default_barrier_bits & SHADER_STORAGE_BARRIER_BIT

    # an explicitly passed barrier_bits must still win exactly as before
    out = gl_ctx.buffer(reserve=4)
    without_counter.bind_ssbo('Out', out)
    without_counter.dispatch(1, 1, 1, barrier_bits=ATOMIC_COUNTER_BARRIER_BIT)  # must not raise
    out.release()


LEGACY_SRC = '''\
layout(std430) buffer Out { uint x; };
layout(binding = 0, offset = 0) uniform atomic_uint c;

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_legacy() {
    x = atomicCounterIncrement(c);
    x = atomicCounterIncrement(c);
}
'''


@pytest.mark.gl
def test_legacy_bind_atomic_counter_still_works(gl_ctx, make_shader_dir):
    """The raw, pre-existing `bind_atomic_counter(binding: int, ...)` escape hatch must keep
    working exactly as before -- immediate bind, no name resolution, no required-set check."""
    shader = _build(gl_ctx, make_shader_dir, 'legacy_counter', LEGACY_SRC)
    kernel = shader.get_kernel('cs_legacy')

    buf = gl_ctx.buffer(reserve=4)
    out = gl_ctx.buffer(reserve=4)
    kernel.bind_ssbo('Out', out)
    kernel.bind_atomic_counter(0, buf)
    kernel.dispatch(1, 1, 1)

    (val,) = struct.unpack('I', out.read())
    assert val == 1  # second call's pre-increment return value (0, then 1)

    buf.release()
    out.release()


RASTER_SRC = '''\
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
    uniform atomic_uint hits;
    out vec4 fragColor;
)]
void fs_main() {
    atomicCounterIncrement(hits);
    fragColor = vec4(1.0);
}
'''


@pytest.mark.gl
def test_pipeline_bind_counter_binds_immediately_and_dispatch_free_kernel_reads_it_back(gl_ctx, make_shader_dir):
    """`Pipeline.bind_counter` has no dispatch hook to defer to (drawing happens in moderngl's
    `VAO.render`, outside tlang -- same asymmetry as `Pipeline.bind_ssbo`), so it must bind
    immediately. Proven end to end: render a full-screen triangle strip counting fragment
    invocations, then read the count back through an unrelated compute kernel bound to the same
    binding via the legacy escape hatch (Pipeline has no dispatch of its own to read through)."""
    import moderngl
    from tlang import ShaderManager

    READBACK_SRC = '''\
    layout(std430) buffer Out { uint val; };
    layout(binding = 0, offset = 0) uniform atomic_uint hits;

    [shader('compute')]
    [numthreads(1, 1, 1)]
    void cs_read() {
        val = atomicCounter(hits);
    }
    '''

    d = make_shader_dir({'draw_counter.tlang': RASTER_SRC, 'readback.tlang': READBACK_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    pipeline = sm.get_shader('draw_counter').get_pipeline('draw')
    reader = sm.get_shader('readback').get_kernel('cs_read')

    assert set(pipeline.atomic_counters) == {'hits'}
    binding, _offset = pipeline.atomic_counters['hits']
    assert binding == 0  # matches READBACK_SRC's pinned binding

    counter_buf = gl_ctx.buffer(reserve=4)
    out = gl_ctx.buffer(reserve=4)

    vbo = gl_ctx.buffer(struct.pack('8f', -1, -1, 1, -1, -1, 1, 1, 1))
    vao = gl_ctx.vertex_array(pipeline.mglo, [(vbo, '2f', 'vert')])
    fbo = gl_ctx.framebuffer(color_attachments=[gl_ctx.texture((4, 4), 4)])
    fbo.use()

    pipeline.bind_counter('hits', counter_buf)
    vao.render(moderngl.TRIANGLE_STRIP)
    gl_ctx.finish()
    gl_ctx.memory_barrier()

    reader.bind_ssbo('Out', out)
    reader.bind_atomic_counter(0, counter_buf)
    reader.dispatch(1, 1, 1)

    (count,) = struct.unpack('I', out.read())
    assert count == 16  # one increment per covered fragment of a 4x4 framebuffer

    vbo.release()
    vao.release()
    fbo.release()
    counter_buf.release()
    out.release()
