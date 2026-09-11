# -------------------------------------------------------------
# @file          test_duplicate_declarations.py
# @description   GL-free tests for ShaderManager's cross-include
#                duplicate-declaration check: same-named top-level
#                functions/consts/raw buffer-uniform blocks colliding
#                across a module's merged include closure. Covers the
#                overload false-positive class this must not fire on
#                (utils.tlang's three `hash` overloads), the diamond
#                include graph, and comment-lookalike immunity.
# -------------------------------------------------------------

import logging

import pytest

from tlang.errors import TlangAttributeError
from tlang.compiler.shader_manager import ShaderManager


def _build(make_shader_dir, files: dict, strict: bool = True):
    d = make_shader_dir(files)
    return ShaderManager(ctx=None, version='430 core', dir=str(d), strict=strict)


# ---------------------------------------------------------------------------
# functions: same name + same parameter types is a real redefinition
# ---------------------------------------------------------------------------

def test_duplicate_function_same_signature_raises_naming_both_modules_and_lines(make_shader_dir):
    files = {
        'a.tlang': "[export()]\nfloat scale(float x) {\n    return x * 2.0;\n}\n",
        'b.tlang': "[export()]\nfloat scale(float x) {\n    return x * 3.0;\n}\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    with pytest.raises(TlangAttributeError) as exc_info:
        _build(make_shader_dir, files)
    msg = str(exc_info.value)
    assert 'scale' in msg
    assert 'a:2' in msg and 'b:2' in msg


def test_duplicate_const_raises_naming_both_modules_and_lines(make_shader_dir):
    files = {
        'a.tlang': "const float EPSILON = 1e-6;\n",
        'b.tlang': "const float EPSILON = 2e-6;\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    with pytest.raises(TlangAttributeError) as exc_info:
        _build(make_shader_dir, files)
    msg = str(exc_info.value)
    assert 'EPSILON' in msg
    assert 'a:1' in msg and 'b:1' in msg


def test_duplicate_raw_buffer_block_raises(make_shader_dir):
    files = {
        'a.tlang': "layout(std430) buffer Config { uint x; };\n",
        'b.tlang': "layout(std430) buffer Config { uint y; };\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    with pytest.raises(TlangAttributeError) as exc_info:
        _build(make_shader_dir, files)
    msg = str(exc_info.value)
    assert 'Config' in msg
    assert 'a:1' in msg and 'b:1' in msg


def test_duplicate_raw_uniform_block_raises(make_shader_dir):
    files = {
        'a.tlang': "uniform Globals { float time; };\n",
        'b.tlang': "uniform Globals { float dt; };\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    with pytest.raises(TlangAttributeError) as exc_info:
        _build(make_shader_dir, files)
    msg = str(exc_info.value)
    assert 'Globals' in msg
    assert 'a:1' in msg and 'b:1' in msg


# ---------------------------------------------------------------------------
# the trap: GLSL overloading is legal and must never be flagged
# ---------------------------------------------------------------------------

def test_same_name_different_parameter_types_is_not_a_duplicate(make_shader_dir):
    """Models utils.tlang's three `hash` overloads (hash(uint64_t), hash(uvec2),
    hash(uint)) -- exactly the false-positive class already hit twice before
    (kernel.bindings, the texture pool)."""
    files = {
        'utils.tlang': (
            "[export()]\nuint hash(uint64_t v) {\n    return uint(v);\n}\n\n"
            "[export()]\nuint hash(uvec2 v) {\n    return v.x ^ v.y;\n}\n\n"
            "[export()]\nuint hash(uint v) {\n    return v;\n}\n"
        ),
        'demo.tlang': "[include(utils)]\n",
    }
    _build(make_shader_dir, files)  # must not raise


def test_non_exported_same_signature_function_is_not_flagged(make_shader_dir):
    """A plain helper that never leaves its own file (no [export()]) can't collide
    with anything in a dependent's merged text -- only exported helpers can."""
    files = {
        'a.tlang': "float scale(float x) {\n    return x * 2.0;\n}\n",
        'b.tlang': "float scale(float x) {\n    return x * 3.0;\n}\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    _build(make_shader_dir, files)  # must not raise


# ---------------------------------------------------------------------------
# diamond include graph: a shared dependency is visited once, never against itself
# ---------------------------------------------------------------------------

def test_diamond_include_graph_does_not_false_positive(make_shader_dir):
    files = {
        'common.tlang': "const float EPSILON = 1e-6;\n",
        'a.tlang': "[include(common)]\n",
        'b.tlang': "[include(common)]\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    _build(make_shader_dir, files)  # must not raise


# ---------------------------------------------------------------------------
# comments/strings are masked out, same as every other scan in this codebase
# ---------------------------------------------------------------------------

def test_lookalike_declaration_inside_a_comment_does_not_trigger(make_shader_dir):
    files = {
        'a.tlang': "const float EPSILON = 1e-6;\n",
        'b.tlang': "// const float EPSILON = 2e-6;\n/* const float EPSILON = 3e-6; */\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    _build(make_shader_dir, files)  # must not raise


# ---------------------------------------------------------------------------
# severity routing: strict=True hard-errors, strict=False warns and keeps building
# ---------------------------------------------------------------------------

def test_strict_false_warns_and_still_builds(make_shader_dir, caplog):
    files = {
        'a.tlang': "[export()]\nfloat scale(float x) {\n    return x * 2.0;\n}\n",
        'b.tlang': "[export()]\nfloat scale(float x) {\n    return x * 3.0;\n}\n",
        'demo.tlang': "[include(a)]\n[include(b)]\n",
    }
    with caplog.at_level(logging.WARNING):
        sm = _build(make_shader_dir, files, strict=False)  # must not raise

    assert sm.get_shader('demo') is not None
    assert sm.get_shader('a') is not None
    assert any('scale' in rec.message for rec in caplog.records)
