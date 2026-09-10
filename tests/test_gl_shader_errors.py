# -------------------------------------------------------------
# @file          test_gl_shader_errors.py
# @description   GL-marked regression tests for the failure paths in
#                Shader._build / BindingRegistry.allocate_artifact:
#                exceeding the per-stage SSBO block limit, and a
#                genuine GLSL compile error under strict=True/False.
# -------------------------------------------------------------

import pytest

pytestmark = pytest.mark.gl


def test_exceeding_stage_block_limit_raises_naming_artifact(gl_ctx, make_shader_dir):
    """One compute kernel referencing more distinct SSBO blocks than the
    driver's GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS must raise
    TlangBindingError naming the offending artifact (the kernel), not
    silently link something broken. The limit is read from ctx.info, not
    hardcoded -- this card reports 16, but the test must not assume that."""
    from tlang import ShaderManager
    from tlang.errors import TlangBindingError

    limit = gl_ctx.info.get('GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS') or 16
    n = limit + 1

    decls = '\n'.join(f'layout(std430) buffer Blk{i} {{ uint v{i}[]; }};' for i in range(n))
    body = '\n    '.join(f'v{i}[0] = 0u;' for i in range(n))
    src = (
        f"{decls}\n\n"
        "[shader('compute')]\n[numthreads(1, 1, 1)]\n"
        f"void cs_overflow() {{\n    {body}\n}}\n"
    )
    d = make_shader_dir({'overflow.tlang': src})

    with pytest.raises(TlangBindingError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'cs_overflow' in msg


def test_compile_error_raises_with_stage_and_entry_point(gl_ctx, make_shader_dir):
    from tlang import ShaderManager
    from tlang.errors import TlangCompileError

    src = "[shader('compute')]\n[numthreads(1, 1, 1)]\nvoid cs_bad() {\n    int x = ;\n}\n"
    d = make_shader_dir({'bad.tlang': src})

    with pytest.raises(TlangCompileError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    err = exc_info.value
    assert err.entry_point == 'cs_bad'
    assert 'comp' in str(err.stage).lower()
    assert err.source is not None


def test_compile_error_strict_false_logs_and_continues(gl_ctx, make_shader_dir):
    """A module whose only kernel failed to compile is not "built" in any useful sense: plain
    `get_shader` must report that (None), matching `.ok`/`.failures` -- not hand back a Shader
    that merely looks empty. `allow_failed=True` still reaches it for diagnosis."""
    from tlang import ShaderManager

    src = "[shader('compute')]\n[numthreads(1, 1, 1)]\nvoid cs_bad() {\n    int x = ;\n}\n"
    d = make_shader_dir({'bad2.tlang': src})

    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=False)
    assert sm.get_shader('bad2') is None

    shader = sm.get_shader('bad2', allow_failed=True)
    assert shader is not None
    assert shader.ok is False
    assert shader.failures
    assert 'cs_bad' not in shader.kernels
