# -------------------------------------------------------------
# @file          test_compile_isolation.py
# @description   GL-marked regression tests for module-level build isolation
#                (T1: one broken module must never abort the rest of the tree)
#                and for "did this compile?" being answerable without a false
#                positive (T2: get_shader/.ok/.failures on a partially-failed
#                module).
# -------------------------------------------------------------

import pytest

pytestmark = pytest.mark.gl

GOOD_A_SRC = """\
layout(std430) buffer BufGoodA { uint a_data[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_good_a() {
    a_data[0] = 1u;
}
"""

GOOD_B_SRC = """\
layout(std430) buffer BufGoodB { uint b_data[]; };

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_good_b() {
    b_data[0] = 2u;
}
"""

BROKEN_SRC = "[shader('compute')]\n[numthreads(1, 1, 1)]\nvoid cs_broken() {\n    int x = ;\n}\n"
BROKEN2_SRC = "[shader('compute')]\n[numthreads(1, 1, 1)]\nvoid cs_broken2() {\n    int y = ;\n}\n"


def test_one_broken_module_leaves_others_usable_under_strict_false(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({
        'good_a.tlang': GOOD_A_SRC,
        'good_b.tlang': GOOD_B_SRC,
        'broken.tlang': BROKEN_SRC,
    })
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=False)

    good_a = sm.get_shader('good_a')
    good_b = sm.get_shader('good_b')
    assert good_a is not None and good_a.ok
    assert good_b is not None and good_b.ok
    assert 'cs_good_a' in good_a.kernels
    assert 'cs_good_b' in good_b.kernels

    buf = gl_ctx.buffer(reserve=4)
    good_a.get_kernel('cs_good_a').bind_ssbo('BufGoodA', buf)
    good_a.get_kernel('cs_good_a').dispatch(1, 1, 1)  # must not raise -- fully usable

    assert sm.get_shader('broken') is None

    broken = sm.get_shader('broken', allow_failed=True)
    assert broken is not None
    assert broken.ok is False
    assert broken.failures


def test_strict_true_names_every_broken_module(gl_ctx, make_shader_dir):
    """Core T1 assertion: two independently broken modules must BOTH be named in the
    single error raised at the end of construction, not just the first one hit."""
    from tlang import ShaderManager
    from tlang.errors import TlangError

    d = make_shader_dir({
        'good_a.tlang': GOOD_A_SRC,
        'broken.tlang': BROKEN_SRC,
        'broken2.tlang': BROKEN2_SRC,
    })

    with pytest.raises(TlangError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'broken' in msg
    assert 'broken2' in msg


def test_shader_ok_true_for_healthy_false_for_broken_kernel(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({
        'good_a.tlang': GOOD_A_SRC,
        'broken.tlang': BROKEN_SRC,
    })
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=False)

    good_a = sm.get_shader('good_a')
    assert good_a is not None
    assert good_a.ok is True
    assert good_a.failures == []

    broken = sm.get_shader('broken', allow_failed=True)
    assert broken.ok is False
    assert len(broken.failures) >= 1


def test_fully_healthy_tree_unchanged(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({
        'good_a.tlang': GOOD_A_SRC,
        'good_b.tlang': GOOD_B_SRC,
    })
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    for name in ('good_a', 'good_b'):
        shader = sm.get_shader(name)
        assert shader is not None
        assert shader.ok is True
        assert shader.failures == []

    assert sm.failures == {}
