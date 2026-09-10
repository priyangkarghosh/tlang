# -------------------------------------------------------------
# @file          test_missing_export_diagnostic.py
# @description   GL-marked regression tests for T15: a module-scope helper
#                that reachable code calls but that was never emitted into
#                the translation unit (missing [export()]/[link(...)],
#                same file or an [include]d one) must raise a tlang error
#                naming the helper, the caller, and the fix -- never a raw
#                driver "undefined variable" at a generated line number.
# -------------------------------------------------------------

import pytest

from tlang.errors import TlangAttributeError, TlangError

pytestmark = pytest.mark.gl


SAME_FILE_SRC = """\
float boundaryFriction(float v) {
    return v * 0.5;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_dynamics() {
    float f = boundaryFriction(1.0);
}
"""

# T10 shape: [link(...)] attached to the helper instead of the caller. This
# does not pull `boundaryFriction` into `cs_dynamics` (that would require
# `[link('boundaryFriction')]` on `cs_dynamics` itself) -- it just gives the
# helper an (unhelpful) link of its own, so the symptom is identical to the
# plain missing-export case above.
BACKWARDS_LINK_SRC = """\
float other() {
    return 1.0;
}

[link('other')]
float boundaryFriction(float v) {
    return v * 0.5;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_dynamics() {
    float f = boundaryFriction(1.0);
}
"""

EXPORTED_SRC = """\
[export]
float boundaryFriction(float v) {
    return v * 0.5;
}

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_dynamics() {
    float f = boundaryFriction(1.0);
}
"""

INCLUDED_SRC = """\
float sharedHelper(float v) {
    return v * 2.0;
}
"""

INCLUDED_EXPORTED_SRC = """\
[export]
float sharedHelper(float v) {
    return v * 2.0;
}
"""

CROSS_FILE_MAIN_SRC = """\
[include(included)]

[shader('compute')]
[numthreads(1, 1, 1)]
void cs_main() {
    float f = sharedHelper(1.0);
}
"""


def test_same_file_missing_export_raises_naming_helper_and_fix(gl_ctx, make_shader_dir):
    from tlang import ShaderManager

    d = make_shader_dir({'dynamics.tlang': SAME_FILE_SRC})
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'boundaryFriction' in msg
    assert 'cs_dynamics' in msg
    assert '[export' in msg


def test_backwards_link_on_helper_also_raises_and_mentions_link(gl_ctx, make_shader_dir):
    """T10: [link(...)] on the helper instead of the caller leaves the helper
    unemitted too -- same symptom, and the message should point out the
    backwards attachment since it's cheap to detect."""
    from tlang import ShaderManager

    d = make_shader_dir({'dynamics.tlang': BACKWARDS_LINK_SRC})
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'boundaryFriction' in msg
    assert 'cs_dynamics' in msg
    assert '[link' in msg


def test_exported_helper_does_not_raise(gl_ctx, make_shader_dir):
    """Sanity/regression guard: once the helper is [export]ed it's emitted
    into the unit and this diagnostic must not fire."""
    from tlang import ShaderManager

    d = make_shader_dir({'dynamics.tlang': EXPORTED_SRC})
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    shader = sm.get_shader('dynamics')
    assert shader is not None and shader.ok
    assert 'cs_dynamics' in shader.kernels


def test_cross_file_missing_export_raises_naming_defining_module(gl_ctx, make_shader_dir):
    """The helper is defined in an [include]d module, not the caller's own
    file -- the diagnostic must still fire and name the module it needs
    [export()] added in, not just "this file"."""
    from tlang import ShaderManager

    d = make_shader_dir({
        'included.tlang': INCLUDED_SRC,
        'main.tlang': CROSS_FILE_MAIN_SRC,
    })
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    msg = str(exc_info.value)
    assert 'sharedHelper' in msg
    assert 'included' in msg
    assert '[export' in msg


def test_cross_file_exported_helper_does_not_raise(gl_ctx, make_shader_dir):
    """Regression guard for the cross-file path: once the [include]d helper
    is [export]ed, calling it from the including module must build cleanly."""
    from tlang import ShaderManager

    d = make_shader_dir({
        'included.tlang': INCLUDED_EXPORTED_SRC,
        'main.tlang': CROSS_FILE_MAIN_SRC,
    })
    sm = ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    shader = sm.get_shader('main')
    assert shader is not None and shader.ok
    assert 'cs_main' in shader.kernels


def test_raised_error_is_a_tlang_error_not_a_raw_driver_message(gl_ctx, make_shader_dir):
    """The whole point: the failure must be attributed by tlang before it
    ever reaches the driver, not surface as a bare GLSL compile error."""
    from tlang import ShaderManager

    d = make_shader_dir({'dynamics.tlang': SAME_FILE_SRC})
    with pytest.raises(TlangError) as exc_info:
        ShaderManager(ctx=gl_ctx, version='430 core', dir=str(d), strict=True)

    assert 'C1503' not in str(exc_info.value)
