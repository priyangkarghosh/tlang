# -------------------------------------------------------------
# @file          test_uniform_block_bindings.py
# @description   GL-free regression tests for BindingRegistry's
#                generalisation to uniform blocks: the shared
#                buffer/uniform regex template, DCE over both
#                keywords, two-pool allocation, and verify_link's
#                UniformBlock branch (moderngl types monkeypatched
#                so this needs no GPU).
# -------------------------------------------------------------

import logging

import pytest

import tlang.compiler.binding_registry as br
from tlang.compiler.binding_registry import BindingRegistry, BLOCK_PATTERN
from tlang.errors import TlangBindingError
from tlang.shader_stages import ShaderStage


class _FakeCtx:
    """Duck-typed stand-in for `moderngl.Context` -- `_stage_limit`/`_max_pool`
    only ever touch `.info.get(key)`, so no real GL is needed to exercise them."""

    def __init__(self, info: dict):
        self.info = info


def _ctx(**overrides) -> _FakeCtx:
    info = {
        'GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS': 16,
        'GL_MAX_UNIFORM_BUFFER_BINDINGS': 16,
        'GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS': 16,
        'GL_MAX_COMPUTE_UNIFORM_BLOCKS': 14,
        'GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS': 16,
        'GL_MAX_VERTEX_UNIFORM_BLOCKS': 14,
        'GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS': 16,
        'GL_MAX_FRAGMENT_UNIFORM_BLOCKS': 14,
    }
    info.update(overrides)
    return _FakeCtx(info)


# ---------------------------------------------------------------------------
# regex template
# ---------------------------------------------------------------------------

def test_buffer_and_uniform_patterns_are_built_from_one_template():
    src = "layout(std430) buffer B { uint x[]; };\nlayout(std140) uniform U { float y; };\n"
    buf_hits = BLOCK_PATTERN['buffer'].findall(src)
    uni_hits = BLOCK_PATTERN['uniform'].findall(src)
    assert [h[2] for h in buf_hits] == ['B']
    assert [h[2] for h in uni_hits] == ['U']


def test_uniform_pattern_ignores_buffer_blocks_and_vice_versa():
    src = "layout(std430) buffer OnlyBuffer { uint x[]; };\n"
    assert BLOCK_PATTERN['uniform'].findall(src) == []
    src2 = "layout(std140) uniform OnlyUniform { float y; };\n"
    assert BLOCK_PATTERN['buffer'].findall(src2) == []


# ---------------------------------------------------------------------------
# DCE: remove_unused_buffers now strips dead uniform blocks too
# ---------------------------------------------------------------------------

def test_dead_uniform_block_is_removed_under_the_old_public_name():
    src = (
        "layout(std140) uniform DeadFrame {\n"
        "    float time;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'DeadFrame' not in out
    assert out.count('\n') == src.count('\n')


def test_live_uniform_block_is_kept():
    src = (
        "layout(std140) uniform LiveFrame {\n"
        "    float time;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    float t = time;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'uniform LiveFrame' in out


def test_instance_named_uniform_block_referenced_through_instance_is_kept():
    src = (
        "layout(std140) uniform Blk {\n"
        "    float x;\n"
        "} inst;\n"
        "\n"
        "void main() {\n"
        "    float t = inst.x;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'uniform Blk' in out


def test_mixed_dead_uniform_and_live_buffer_block_only_the_dead_one_goes():
    """A dead `uniform` block sits ahead of a live `buffer` block in the
    same file -- removal must sort by source position across keywords and
    leave the live block, and the rest of the file, untouched."""
    src = (
        "layout(std140) uniform Dead {\n"
        "    float unused_field;\n"
        "};\n"
        "\n"
        "layout(std430) buffer Live {\n"
        "    uint used_field[];\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    used_field[0] = 1u;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'Dead' not in out
    assert 'buffer Live' in out
    assert 'used_field' in out
    assert out.count('\n') == src.count('\n')


def test_mixed_dead_buffer_and_live_uniform_block_only_the_dead_one_goes():
    src = (
        "layout(std430) buffer Dead {\n"
        "    uint unused_field[];\n"
        "};\n"
        "\n"
        "layout(std140) uniform Live {\n"
        "    float used_field;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    float t = used_field;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'Dead' not in out
    assert 'uniform Live' in out
    assert out.count('\n') == src.count('\n')


def test_unused_uniform_block_with_nested_struct_removed_cleanly():
    src = (
        "layout(std140) uniform DeadStruct {\n"
        "    struct Inner { float a; float b; } items;\n"
        "};\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = BindingRegistry.remove_unused_buffers(src)
    assert 'DeadStruct' not in out
    assert 'Inner' not in out
    assert 'items' not in out
    assert '};' not in out
    assert out.count('\n') == src.count('\n')


# ---------------------------------------------------------------------------
# allocation: separate pools, not merged
# ---------------------------------------------------------------------------

def _comp_src(*blocks: str) -> dict[ShaderStage, str]:
    return {ShaderStage.COMP: '\n'.join(blocks) + '\nvoid main() {}\n'}


def test_allocate_artifact_returns_three_tuple():
    ctx = _ctx()
    stage_sources = _comp_src("layout(std430) buffer Data { uint x[]; };")
    result = BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})
    assert len(result) == 3
    patched, ssbo_canon, uniform_canon = result
    assert ssbo_canon == {'Data': 0}
    assert uniform_canon == {}


def test_ssbo_and_uniform_blocks_allocate_from_independent_pools():
    """Both pools start numbering from 0 -- a shared `Data` SSBO and a
    `Frame` uniform block in the same stage must each land at 0 in their
    own pool, proving the two are never merged into one binding space."""
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(std430) buffer Data { uint x[]; };",
        "layout(std140) uniform Frame { float time; };",
    )
    patched, ssbo_canon, uniform_canon = BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})
    assert ssbo_canon == {'Data': 0}
    assert uniform_canon == {'Frame': 0}
    assert 'Frame' not in ssbo_canon
    assert 'Data' not in uniform_canon


def test_explicit_pins_on_same_index_in_different_pools_do_not_conflict():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(std430, binding = 0) buffer Data { uint x[]; };",
        "layout(std140, binding = 0) uniform Frame { float time; };",
    )
    patched, ssbo_canon, uniform_canon = BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})
    assert ssbo_canon == {'Data': 0}
    assert uniform_canon == {'Frame': 0}


def test_explicit_uniform_binding_pin_is_patched_into_source():
    ctx = _ctx()
    stage_sources = _comp_src("layout(std140) uniform Frame { float time; };")
    patched, ssbo_canon, uniform_canon = BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})
    assert uniform_canon == {'Frame': 0}
    assert 'layout(binding = 0, std140) uniform Frame' in patched[ShaderStage.COMP]


def test_two_uniform_blocks_explicitly_pinned_to_same_binding_raises():
    ctx = _ctx()
    stage_sources = _comp_src(
        "layout(std140, binding = 2) uniform A { float a; };",
        "layout(std140, binding = 2) uniform B { float b; };",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})


def test_per_stage_uniform_block_limit_is_enforced():
    ctx = _ctx(GL_MAX_COMPUTE_UNIFORM_BLOCKS=1)
    stage_sources = _comp_src(
        "layout(std140) uniform A { float a; };",
        "layout(std140) uniform B { float b; };",
    )
    with pytest.raises(TlangBindingError, match="uniform"):
        BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})


def test_uniform_pool_exhaustion_raises():
    ctx = _ctx(GL_MAX_UNIFORM_BUFFER_BINDINGS=1, GL_MAX_COMPUTE_UNIFORM_BLOCKS=8)
    stage_sources = _comp_src(
        "layout(std140) uniform A { float a; };",
        "layout(std140) uniform B { float b; };",
    )
    with pytest.raises(TlangBindingError):
        BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})


def test_uniform_limit_fallback_logs_loudly_when_driver_omits_it(caplog):
    info = {
        'GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS': 16,
        'GL_MAX_UNIFORM_BUFFER_BINDINGS': 16,
        'GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS': 16,
        # GL_MAX_COMPUTE_UNIFORM_BLOCKS deliberately absent
    }
    ctx = _FakeCtx(info)
    stage_sources = _comp_src("layout(std140) uniform A { float a; };")
    with caplog.at_level(logging.WARNING):
        BindingRegistry.allocate_artifact(ctx, 'art', stage_sources, {})
    assert any('GL_MAX_COMPUTE_UNIFORM_BLOCKS' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# verify_link: UniformBlock branch (moderngl types monkeypatched, no GPU)
# ---------------------------------------------------------------------------

class _FakeStorageBlock:
    def __init__(self, binding): self.binding = binding


class _FakeUniformBlock:
    def __init__(self, binding): self.binding = binding


class _FakeLinked:
    def __init__(self, members): self._members = members
    def get(self, name, default=None): return self._members.get(name, default)


@pytest.fixture
def fake_block_types(monkeypatch):
    monkeypatch.setattr(br, 'StorageBlock', _FakeStorageBlock)
    monkeypatch.setattr(br, 'UniformBlock', _FakeUniformBlock)


def test_verify_link_passes_when_reflected_uniform_binding_matches(fake_block_types):
    linked = _FakeLinked({'Data': _FakeStorageBlock(0), 'Frame': _FakeUniformBlock(3)})
    BindingRegistry.verify_link(linked, {'Data': 0}, 'art', {'Frame': 3})  # must not raise


def test_verify_link_raises_on_uniform_binding_mismatch(fake_block_types):
    linked = _FakeLinked({'Frame': _FakeUniformBlock(5)})
    with pytest.raises(TlangBindingError):
        BindingRegistry.verify_link(linked, {}, 'art', {'Frame': 2})


def test_verify_link_ignores_missing_uniform_canon():
    """`uniform_canon` is optional -- omitting it must not affect the SSBO check."""
    linked = _FakeLinked({})
    BindingRegistry.verify_link(linked, {}, 'art')  # must not raise


# --- regression: commented-out block declarations must not reach the patcher ---

@pytest.mark.parametrize("keyword", ["buffer", "uniform"])
def test_patch_bindings_ignores_commented_out_declaration(keyword):
    """Allocation scans masked source, patching scans raw, so a declaration inside a
    comment has no binding assigned. It must be left alone, not raise KeyError."""
    src = (
        f"/* leftover:\nlayout(std430) {keyword} Ghost {{ uint x[]; }};\n*/\n"
        f"layout(std430) {keyword} Live {{ uint y[]; }};\n"
    )
    out = br.BindingRegistry._patch_bindings(src, keyword, {"Live": 0})
    assert "binding = 0" in out
    assert f"{keyword} Ghost" in out          # comment survives verbatim
    assert out.count("binding =") == 1        # ghost got no binding


@pytest.mark.parametrize("comment", ["// layout(std430) buffer Ghost { uint x[]; };",
                                     "/* layout(std430) buffer Ghost { uint x[]; }; */"])
def test_patch_bindings_line_and_block_comments(comment):
    src = f"{comment}\nlayout(std430) buffer Live {{ uint y[]; }};\n"
    out = br.BindingRegistry._patch_bindings(src, "buffer", {"Live": 0})
    assert out.count("binding =") == 1
