# -------------------------------------------------------------
# @file          test_dead_function_elimination.py
# @description   GL-free regression tests for
#                BindingRegistry.remove_dead_functions -- dead-function
#                elimination (DFE), which must run before the block DCE
#                (test_dead_code_elimination.py) so an [export()]ed helper
#                a given entry point never calls doesn't keep that helper's
#                SSBO/UBO blocks alive. Same KEPT/REMOVED shape as that
#                file's tests.
# -------------------------------------------------------------

from tlang.compiler.binding_registry import BindingRegistry


def _dfe(src: str) -> str:
    return BindingRegistry.remove_dead_functions(src)


def _dfe_then_dce(src: str) -> str:
    return BindingRegistry.remove_unused_buffers(BindingRegistry.remove_dead_functions(src))


# ---------------------------------------------------------------------------
# KEPT -- these functions must survive
# ---------------------------------------------------------------------------

def test_directly_called_helper_is_kept():
    src = (
        "float liveHelper() {\n"
        "    return 1.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    float v = liveHelper();\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'liveHelper' in out
    assert 'return 1.0;' in out


def test_transitively_called_helper_is_kept_uncalled_sibling_removed():
    """main -> a -> b: b is only ever called by a, never by main directly --
    it must still be kept. `c` is never called by anything and must go."""
    src = (
        "float b() {\n"
        "    return 2.0;\n"
        "}\n"
        "\n"
        "float a() {\n"
        "    return b();\n"
        "}\n"
        "\n"
        "float c() {\n"
        "    return 3.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    float v = a();\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'float b()' in out
    assert 'return 2.0;' in out
    assert 'float a()' in out
    assert 'return b();' in out
    assert 'float c()' not in out
    assert 'return 3.0;' not in out


def test_all_overloads_kept_when_any_is_called():
    """A textual walk can't do overload resolution -- a call to `hash(` must
    keep EVERY body named `hash`, not just the one whose signature would
    really be selected."""
    src = (
        "uint hash(uint x) {\n"
        "    return x * 2654435761u;\n"
        "}\n"
        "\n"
        "uvec2 hash(uvec2 x) {\n"
        "    return uvec2(hash(x.x), hash(x.y));\n"
        "}\n"
        "\n"
        "uint hash(uint64_t x) {\n"
        "    return uint(x);\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    uint h = hash(1u);\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'return x * 2654435761u;' in out
    assert 'return uvec2(hash(x.x), hash(x.y));' in out
    assert 'return uint(x);' in out


def test_multiline_signature_helper_is_kept():
    """contacts.tlang's `solvePair` has its parameter list split across two
    lines -- a header regex restricted to single-line whitespace between
    tokens would silently fail to recognize this as a function at all and
    its body would be dropped as unreachable, breaking the kernel."""
    src = (
        "void solvePair(\n"
        "    inout vec3 a,\n"
        "    inout vec3 b\n"
        ") {\n"
        "    a += b;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    vec3 p = vec3(0.0);\n"
        "    vec3 q = vec3(0.0);\n"
        "    solvePair(p, q);\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'solvePair' in out
    assert 'a += b;' in out


def test_linked_helper_called_by_main_is_kept():
    """A [link(...)]ed helper is inlined ahead of its caller's body before
    this pass ever runs -- from here it's just an ordinary top-level
    definition that main's call graph reaches."""
    src = (
        "float linkedHelper() {\n"
        "    return 4.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    float v = linkedHelper();\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'linkedHelper' in out
    assert 'return 4.0;' in out


def test_module_scope_call_site_is_a_root():
    """A global initializer that calls a helper keeps it, even though
    nothing inside any function body ever calls it."""
    src = (
        "float initValue() {\n"
        "    return 5.0;\n"
        "}\n"
        "\n"
        "float LUT = initValue();\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'initValue' in out
    assert 'return 5.0;' in out


# ---------------------------------------------------------------------------
# REMOVED -- genuinely unreachable functions must be cleaned up
# ---------------------------------------------------------------------------

def test_uncalled_helper_is_removed():
    src = (
        "float deadHelper() {\n"
        "    return 9.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'deadHelper' not in out
    assert 'return 9.0;' not in out
    assert out.count('\n') == src.count('\n')


def test_uncalled_helpers_buffer_block_drops_out_after_dfe_then_dce():
    """The actual bug this pass fixes: an uncalled helper's SSBO block must
    disappear once DFE removes the helper body that block DCE was
    previously (wrongly) kept alive by."""
    src = (
        "layout(std430) buffer DeadBlock {\n"
        "    uint deadfield[];\n"
        "};\n"
        "\n"
        "float deadHelper() {\n"
        "    return float(deadfield[0]);\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    # block DCE alone (no DFE) wrongly keeps the block -- `deadfield` is
    # still referenced from deadHelper's still-present body
    assert 'DeadBlock' in BindingRegistry.remove_unused_buffers(src)

    out = _dfe_then_dce(src)
    assert 'deadHelper' not in out
    assert 'DeadBlock' not in out
    assert 'deadfield' not in out


def test_called_helpers_buffer_block_is_kept_after_dfe_then_dce():
    src = (
        "layout(std430) buffer LiveBlock {\n"
        "    uint livefield[];\n"
        "};\n"
        "\n"
        "float liveHelper() {\n"
        "    return float(livefield[0]);\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    float v = liveHelper();\n"
        "}\n"
    )
    out = _dfe_then_dce(src)
    assert 'liveHelper' in out
    assert 'LiveBlock' in out
    assert 'livefield' in out


def test_newline_count_preserved_across_several_removed_functions():
    src = (
        "float dead1() {\n"
        "    return 1.0;\n"
        "}\n"
        "\n"
        "float dead2() {\n"
        "    return 2.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    int x = 1;\n"
        "}\n"
    )
    out = _dfe(src)
    assert 'dead1' not in out
    assert 'dead2' not in out
    assert out.count('\n') == src.count('\n')


def test_builtins_and_type_constructors_are_not_treated_as_user_functions():
    """`vec4(...)`/`atomicAdd(...)` are never keys in the function map --
    they must not be mistaken for a removable (or keepable) user function,
    and must not spuriously "reach" anything."""
    src = (
        "float deadHelper() {\n"
        "    return 1.0;\n"
        "}\n"
        "\n"
        "void main() {\n"
        "    vec4 v = vec4(1.0, 2.0, 3.0, atomicAdd(deadHelper, 1));\n"
        "}\n"
    )
    # `atomicAdd(deadHelper, 1)` passes the bare identifier `deadHelper`, not a
    # call `deadHelper(...)` -- it must not count as a call site. The identifier
    # legitimately survives inside main() either way; what must go is the
    # function DEFINITION itself.
    out = _dfe(src)
    assert 'float deadHelper()' not in out
    assert 'return 1.0;' not in out
