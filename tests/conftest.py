# -------------------------------------------------------------
# @file          conftest.py
# @description   Shared fixtures for the tlang test suite.
#
#                The suite is split into two tiers:
#                  - GL-free tests (the majority): ShaderProcessor,
#                    AttributeManager, FunctionManager,
#                    dead_code.remove_dead_blocks, and
#                    DependencyManager.resolve_dependencies all work
#                    without a GL context, so these must run on any
#                    machine (no GPU needed) -- that's what makes CI
#                    possible.
#                  - GL tests (`@pytest.mark.gl`) build a real
#                    ShaderManager against a session-scoped standalone
#                    context and dispatch real compute kernels. If
#                    context creation fails (headless box, no GPU), the
#                    whole GL tier is skipped cleanly via `pytest.skip`
#                    so the GL-free tier still runs.
#
#                IMPORTANT: nothing at module import time here touches
#                moderngl/GL -- the `gl_ctx` fixture only creates a
#                context when a test that depends on it (directly, or
#                transitively via `build_manager`) actually runs. This
#                is what lets `pytest -m "not gl"` do zero GPU work.
# -------------------------------------------------------------

import pytest


@pytest.fixture(scope="session")
def gl_ctx():
    """Session-scoped standalone GL context -- created once (context
    creation is slow). Skips the whole `gl` tier cleanly if a context
    can't be created, so the GL-free tier still runs on a headless box.

    `require=430` because compute shaders (GL_ARB_compute_shader) and
    SSBOs need GL 4.3+; a context created without `require` handed back
    only GL 3.3 on this machine (see tests/README-equivalent notes in
    the test-suite report), which is too old for anything tlang does
    with kernels.
    """
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_context(require=430, standalone=True)
    except Exception as exc:  # pragma: no cover -- environment dependent
        pytest.skip(f"Could not create a standalone GL context: {exc}")
        return  # unreachable, keeps type-checkers happy

    yield ctx
    ctx.release()


@pytest.fixture
def make_shader_dir(tmp_path):
    """Factory fixture: write {relative_path: source} into a fresh temp
    directory and return its path. Lets GL tests inline their .tlang
    fixtures right next to the assertion instead of maintaining separate
    fixture files on disk."""

    def _make(files: dict) -> "object":
        for rel_path, content in files.items():
            p = tmp_path / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return tmp_path

    return _make


@pytest.fixture
def build_manager(gl_ctx):
    """Factory fixture: construct a `ShaderManager` against the shared
    session GL context. Only pulls in `gl_ctx` (and therefore only ever
    creates a context) when a gl-marked test actually requests this."""

    def _build(dir_path, constants=None, strict=True, version="430 core", keep_sources=False):
        from tlang import ShaderManager

        return ShaderManager(
            ctx=gl_ctx,
            version=version,
            dir=str(dir_path),
            constants=constants or {},
            strict=strict,
            keep_sources=keep_sources,
        )

    return _build
