# -------------------------------------------------------------
# @file          test_source_retention.py
# @description   GL-marked integration tests for `ShaderManager(keep_sources=...)`:
#                a successfully compiled entry point's generated GLSL must not
#                survive the build by default, a failed entry point's must always
#                survive, and `keep_sources=True` restores today's retain-everything
#                behaviour exactly.
# -------------------------------------------------------------

import pytest

from tlang.errors import TlangCompileError, TlangError

pytestmark = pytest.mark.gl


HEALTHY_SRC = """\
[shader('compute')]
[numthreads(1, 1, 1)]
void cs_healthy() {
    float unused = 1.0;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_healthy_2() {
    float unused = 2.0;
}
"""

# `broken_ident` is a plain (non-call) undefined identifier, not a `name(...)` call --
# it must fall straight through tlang's T15 "missing [export]" diagnostic and reach the
# driver as a genuine GLSL compile error.
FAILING_SRC = """\
[shader('compute')]
[numthreads(1, 1, 1)]
void cs_healthy() {
    float unused = 1.0;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_bad() {
    float unused = broken_ident;
}
"""


def test_default_build_drops_healthy_sources_and_get_source_explains_why(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': HEALTHY_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    sh = sm.get_shader('demo')
    assert sh is not None and sh.ok

    # nothing failed, so nothing should be sitting in memory
    assert sh.sources == {}

    with pytest.raises(TlangError) as exc_info:
        sh.get_source('cs_healthy')

    msg = str(exc_info.value)
    assert 'cs_healthy' in msg
    assert 'keep_sources=True' in msg
    assert not isinstance(exc_info.value, KeyError)


def test_keep_sources_true_retains_everything_with_line_directives(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': HEALTHY_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), keep_sources=True)
    sh = sm.get_shader('demo')
    assert sh is not None and sh.ok

    src = sh.get_source('cs_healthy')
    assert '#line' in src
    assert 'void main(' in src  # entry point rewritten from `cs_healthy` to `main`

    assert set(sh.sources.keys()) == {'cs_healthy', 'cs_healthy_2'}


def test_failed_entry_point_source_is_retained_under_default(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'demo.tlang': FAILING_SRC})
    # strict=False: the tree still builds, with the broken kernel simply absent from
    # `sh.kernels` and its error collected in `sh.failures` / `sm.failures` instead of
    # raised -- must remain fully diagnosable with keep_sources left at its default.
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=False)
    sh = sm.get_shader('demo', allow_failed=True)
    assert sh is not None and not sh.ok

    errs = [e for e in sh.failures if isinstance(e, TlangCompileError)]
    assert len(errs) == 1
    err = errs[0]
    assert err.entry_point == 'cs_bad'
    assert err.source is not None
    assert 'broken_ident' in err.source

    # the failed entry point's source is retained on the Shader itself too, by default
    assert 'cs_bad' in sh.sources
    assert 'broken_ident' in sh.sources['cs_bad']
    assert sh.get_source('cs_bad') == sh.sources['cs_bad']

    # the healthy sibling entry point in the same module still had its source dropped
    assert 'cs_healthy' not in sh.sources
    with pytest.raises(TlangError):
        sh.get_source('cs_healthy')


def test_default_retains_substantially_less_than_keep_sources_true(gl_ctx, make_shader_dir):
    """Regression guard: build the same healthy tree both ways and compare the total
    retained source bytes. A future change that silently starts retaining successful
    sources again must fail this loudly, not just look fine in isolated unit checks."""
    from tlang import ShaderManager

    big_src = HEALTHY_SRC + ''.join(
        f"""
[shader('compute')]
[numthreads(1, 1, 1)]
void cs_extra_{i}() {{
    float unused = {i}.0;
}}
"""
        for i in range(10)
    )

    d = make_shader_dir({'demo.tlang': big_src})
    sm_default = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d))
    sm_keep = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), keep_sources=True)

    def retained_bytes(sm) -> int:
        return sum(
            len(v)
            for name in sm._shaders
            for v in sm.get_shader(name, allow_failed=True).sources.values()
        )

    default_bytes = retained_bytes(sm_default)
    keep_bytes = retained_bytes(sm_keep)

    assert default_bytes == 0
    assert keep_bytes > 0
    assert default_bytes < keep_bytes * 0.1
