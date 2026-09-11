# -------------------------------------------------------------
# @file          test_dispatch_for.py
# @description   GL-marked tests for Kernel.local_size and Kernel.dispatch_for: querying the
#                linked program's actual work-group size (including when [numthreads(...)]'s
#                argument is a {{ CONSTANT }}-templated macro name rather than a literal, the
#                case that proves querying the driver beats parsing the attribute), deriving
#                dispatch_for's group counts from it with ceiling division, elems_per_thread,
#                2-D coverage, and the same binding re-assert path dispatch() already has.
#
#                Group counts are verified two ways: reading gl_NumWorkGroups back from the GPU
#                (the group-count tests), and reading back which buffer slots were actually
#                touched by a guarded per-index write (the coverage test) -- both real GPU
#                readback, never an inspection of Python-side arithmetic alone.
# -------------------------------------------------------------

import struct

import pytest

from tlang.errors import TlangBindingError

pytestmark = pytest.mark.gl


MAIN_SRC = '''\
layout(std430) buffer Data { uint data[]; };

[uniforms]
struct Params { uint n; };

[shader('compute')]
[numthreads(64, 2, 1)]
void cs_local_size_literal() {
    data[0] = 0u;
}

[shader('compute')]
[numthreads(256, 1, 1)]
void cs_touch() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid < n) {
        data[gid] += 1u;
    }
}

[shader('compute')]
[numthreads(128, 1, 1)]
void cs_group_count_1d() {
    if (gl_GlobalInvocationID.x == 0u) {
        data[0] = gl_NumWorkGroups.x;
    }
}

[shader('compute')]
[numthreads(8, 4, 1)]
void cs_group_count_2d() {
    if (gl_GlobalInvocationID.x == 0u && gl_GlobalInvocationID.y == 0u) {
        data[0] = gl_NumWorkGroups.x;
        data[1] = gl_NumWorkGroups.y;
        data[2] = gl_NumWorkGroups.z;
    }
}
'''

TEMPLATED_SRC = '''\
layout(std430) buffer Data { uint data[]; };

#define BS {{ BS }}

[shader('compute')]
[numthreads(BS, 1, 1)]
void cs_templated() {
    data[0] = 0u;
}
'''


@pytest.fixture(scope='module')
def main_shader(gl_ctx, tmp_path_factory):
    from tlang import ShaderManager

    d = tmp_path_factory.mktemp('dispatch_for_project')
    (d / 'main.tlang').write_text(MAIN_SRC, encoding='utf-8')

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('main')
    assert shader is not None
    return shader


def _read_u32s(buf, count) -> tuple:
    return struct.unpack(f'{count}I', buf.read(4 * count))


def test_local_size_reports_declared_literal_size(main_shader):
    kernel = main_shader.get_kernel('cs_local_size_literal')
    assert kernel.local_size == (64, 2, 1)


def test_local_size_is_cached(main_shader):
    kernel = main_shader.get_kernel('cs_local_size_literal')
    first = kernel.local_size
    assert kernel.local_size is first  # same tuple object -- queried once, then cached


def test_local_size_resolves_templated_constant(gl_ctx, make_shader_dir):
    """The case that proves querying the linked program beats parsing [numthreads(...)]'s
    argument text: `BS` is a GLSL macro name (via `{{ BS }}` -> `#define BS 128`), never
    coerced to a Python int by tlang itself (see AttributeHandlers.numthreads) -- only the
    driver-resolved, linked program can say what it actually compiled to."""
    from tlang import ShaderManager

    d = make_shader_dir({'templated.tlang': TEMPLATED_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), constants={'BS': 128})
    shader = sm.get_shader('templated')
    assert shader is not None

    kernel = shader.get_kernel('cs_templated')
    assert kernel.local_size == (128, 1, 1)


def test_dispatch_for_covers_non_multiple_count_exactly(gl_ctx, main_shader):
    """1000 elements against a local_size-256 kernel is not an exact multiple -- dispatch_for
    must launch enough groups (4, covering 1024 lanes) that the shader's own `gid < n` guard
    still lets exactly the first 1000 slots be touched, and none beyond it."""
    kernel = main_shader.get_kernel('cs_touch')
    n = 1000
    guard_capacity = 1024  # 4 groups * local_size 256 -- every lane dispatch_for(1000) launches

    buf = gl_ctx.buffer(data=bytes(4 * guard_capacity))
    kernel.bind_ssbo('Data', buf)
    kernel.set_uniform('n', n)
    kernel.dispatch_for(n)

    values = _read_u32s(buf, guard_capacity)
    assert all(v == 1 for v in values[:n])
    assert all(v == 0 for v in values[n:])


def test_elems_per_thread_halves_group_count(gl_ctx, main_shader):
    kernel = main_shader.get_kernel('cs_group_count_1d')  # local_size (128, 1, 1)
    buf = gl_ctx.buffer(data=bytes(4 * 3))
    kernel.bind_ssbo('Data', buf)

    kernel.dispatch_for(1024)  # ceil(1024 / 128) = 8
    assert _read_u32s(buf, 1)[0] == 8

    kernel.dispatch_for(1024, elems_per_thread=2)  # ceil(1024 / 256) = 4 -- half of 8
    assert _read_u32s(buf, 1)[0] == 4


def test_elems_per_thread_quarters_group_count(gl_ctx, main_shader):
    kernel = main_shader.get_kernel('cs_group_count_1d')  # local_size (128, 1, 1)
    buf = gl_ctx.buffer(data=bytes(4 * 3))
    kernel.bind_ssbo('Data', buf)

    kernel.dispatch_for(1024, elems_per_thread=4)  # ceil(1024 / 512) = 2 -- a quarter of 8
    assert _read_u32s(buf, 1)[0] == 2


def test_dispatch_for_covers_2d(gl_ctx, main_shader):
    kernel = main_shader.get_kernel('cs_group_count_2d')  # local_size (8, 4, 1)
    buf = gl_ctx.buffer(data=bytes(4 * 3))
    kernel.bind_ssbo('Data', buf)

    kernel.dispatch_for(20, 10)  # ceil(20/8)=3, ceil(10/4)=3, z stays 1
    gx, gy, gz = _read_u32s(buf, 3)
    assert (gx, gy, gz) == (3, 3, 1)


def test_dispatch_for_raises_on_unbound_required_buffer(gl_ctx, make_shader_dir):
    """dispatch_for must go through the same _assert_ssbo_bindings path as dispatch -- a
    required block never bound raises TlangBindingError, not a silent dispatch."""
    from tlang import ShaderManager

    # Fresh, unbound build so this test doesn't depend on bind order against the shared
    # module-scoped kernels used above.
    d = make_shader_dir({'unbound.tlang': MAIN_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('unbound')
    assert shader is not None
    kernel = shader.get_kernel('cs_touch')

    with pytest.raises(TlangBindingError, match='Data'):
        kernel.dispatch_for(10)
