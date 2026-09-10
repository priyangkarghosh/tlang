# -------------------------------------------------------------
# @file          test_extension_groups.py
# @description   Regression tests for EXTENSION_GROUPS['int64'] (U8): NVIDIA
#                gates the 64-bit atomic builtin overloads behind
#                GL_NV_gpu_shader5, not behind any extension whose name says
#                "atomic_int64", so the group must include it.
#
#                Split into a GL-free tier (group membership, and expansion
#                through [extend(...)] -- pure text/regex processing, no GL
#                context needed) and a GL-marked tier (the real compile-time
#                regression: a compute shader actually using the atomic on a
#                live driver).
# -------------------------------------------------------------

import pytest

from tlang.shader_utils import EXTENSION_GROUPS
from tlang.compiler.shader_processor import ShaderProcessor


def test_int64_group_includes_nv_gpu_shader5():
    """Regression for U8: without GL_NV_gpu_shader5, atomicCompSwap on a
    uint64_t in an SSBO fails to compile on NVIDIA with a misleading
    "unable to find compatible overloaded function" error that reads like
    a hardware limitation. See the comment above the group in
    shader_utils.py for the verified repro."""
    assert 'GL_NV_gpu_shader5' in EXTENSION_GROUPS['int64']


def test_int64_group_still_has_the_originally_named_extensions():
    """Guard against U8's fix accidentally dropping members instead of
    adding one."""
    for ext in (
        'GL_ARB_gpu_shader_int64',
        'GL_EXT_shader_atomic_int64',
        'GL_KHR_shader_atomic_int64',
        'GL_NV_shader_atomic_int64',
    ):
        assert ext in EXTENSION_GROUPS['int64']


def test_extend_expands_int64_group_into_enable_lines():
    """[extend(int64)] must resolve the group alias into every one of its
    member extensions, each tagged ' : enable'."""
    proc = ShaderProcessor('m', "[extend(int64)]\n")
    expected = {ext + ' : enable' for ext in EXTENSION_GROUPS['int64']}
    assert proc.ext == expected


def test_require_expands_int64_group_into_require_lines():
    """[extend!(int64)] (a.k.a. [require(int64)]) must resolve the same
    group but tag each member ' : require' instead of ' : enable'."""
    proc = ShaderProcessor('m', "[extend!(int64)]\n")
    expected = {ext + ' : require' for ext in EXTENSION_GROUPS['int64']}
    assert proc.ext == expected


def test_unknown_token_passes_through_unmapped():
    """A token that isn't a known group name is treated as a literal
    extension name rather than silently dropped."""
    proc = ShaderProcessor('m', "[extend(GL_SOME_MADE_UP_EXTENSION)]\n")
    assert proc.ext == {'GL_SOME_MADE_UP_EXTENSION : enable'}


# ---------------------------------------------------------------------------
# GL-marked: the real regression -- this must FAIL if GL_NV_gpu_shader5 is
# ever removed from the int64 group again.
# ---------------------------------------------------------------------------

ATOMIC_INT64_SRC = '''\
[extend(int64)]

layout(std430) buffer CounterBuf { uint64_t counter[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_atomic_swap() {
    atomicCompSwap(counter[0], 0ul, 42ul);
}
'''


@pytest.mark.gl
def test_int64_atomic_compswap_compiles(gl_ctx, make_shader_dir):
    """The actual U8 regression: a compute shader using [extend(int64)] and
    atomicCompSwap on a uint64_t in an SSBO must compile. This fails with
    driver error C1115 ("unable to find compatible overloaded function")
    if GL_NV_gpu_shader5 is missing from the int64 group -- skip (not fail)
    on a driver that genuinely lacks the needed extensions, so this stays
    CI-safe on non-NVIDIA hardware."""
    needed = {'GL_NV_gpu_shader5', 'GL_ARB_gpu_shader_int64'}
    missing = needed - gl_ctx.extensions
    if missing:
        pytest.skip(f"GPU/driver lacks required extension(s): {sorted(missing)}")

    from tlang import ShaderManager

    d = make_shader_dir({'atomic64.tlang': ATOMIC_INT64_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    shader = sm.get_shader('atomic64')
    assert shader is not None

    kernel = shader.get_kernel('cs_atomic_swap')
    buf = gl_ctx.buffer(reserve=8)
    kernel.bind_ssbo('CounterBuf', buf)
    kernel.dispatch(1, 1, 1)  # must not raise -- proves it actually compiled and ran
