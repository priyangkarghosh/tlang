# -------------------------------------------------------------
# @file          test_pipeline_validation.py
# @description   GL-free regression tests for
#                ShaderProcessor._process_program_attr -- every
#                [program(...)] misuse must produce a specific,
#                readable diagnostic, never an opaque KeyError.
# -------------------------------------------------------------

import pytest

from tlang.errors import TlangAttributeError
from tlang.compiler.shader_processor import ShaderProcessor


FUNCS_SRC = (
    "[shader('vertex')]\n"
    "void vs() {\n    int x = 0;\n}\n"
    "\n"
    "[shader('fragment')]\n"
    "void fs() {\n    int x = 0;\n}\n"
    "\n"
    "[shader('compute')]\n[numthreads(1,1,1)]\n"
    "void cs() {\n    int x = 0;\n}\n"
    "\n"
    "[shader('tess_control')]\n[tesc(vertices=3)]\n"
    "void tc() {\n    int x = 0;\n}\n"
    "\n"
    "[shader('tess_eval')]\n"
    "void te() {\n    int x = 0;\n}\n"
)


def test_program_missing_vertex_stage_raises():
    src = f"[program('p', frag='fs')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    assert 'vertex' in str(exc_info.value)


def test_program_nonexistent_entry_point_raises():
    src = f"[program('p', vert='nope', frag='fs')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    assert 'nope' in str(exc_info.value)


def test_program_wrong_stage_for_slot_raises():
    """'cs' is a compute function; naming it in the vert= slot is a
    stage mismatch, not merely an unknown/missing function."""
    src = f"[program('p', vert='cs', frag='fs')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'cs' in msg
    assert 'vert' in msg


def test_program_unknown_kwarg_raises():
    src = f"[program('p', vert='vs', frag='fs', foo='bar')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'foo' in msg
    assert 'unknown' in msg.lower()


def test_program_two_aliases_for_same_stage_raises():
    """'vert' and 'vertex' are both aliases for the VERT stage slot --
    supplying both must be rejected, not silently let one clobber the
    other."""
    src = f"[program('p', vert='vs', vertex='cs', frag='fs')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'vert' in msg
    assert 'multiple aliases' in msg.lower() or 'target the' in msg.lower()


def test_program_comp_kwarg_rejected_as_dispatched_not_linked():
    """Compute shaders are dispatched via a kernel, never linked into a
    [program(...)] -- naming one in comp= must be rejected outright."""
    src = f"[program('p', vert='vs', frag='fs', comp='cs')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'compute' in msg.lower()
    assert 'kernel' in msg.lower() or 'dispatch' in msg.lower()


def test_program_tesc_without_tese_rejected():
    src = f"[program('p', vert='vs', frag='fs', tesc='tc')]\n{FUNCS_SRC}"
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'tesc' in msg
    assert 'tese' in msg


def test_program_tese_without_tesc_is_legal():
    """tese-alone is legal in GL 4.x (patch size comes from
    glPatchParameteri) -- this combination must NOT raise."""
    src = f"[program('p', vert='vs', frag='fs', tese='te')]\n{FUNCS_SRC}"
    proc = ShaderProcessor('t', src)
    assert 'p' in proc.programs


def test_multiple_broken_programs_report_all_problems():
    """strict=True must raise ONE error naming every broken program, not
    stop at the first."""
    src = (
        "[program('p1', frag='fs')]\n"          # missing vertex
        "[program('p2', vert='vs', frag='fs', bogus='x')]\n"  # unknown kwarg
        f"{FUNCS_SRC}"
    )
    with pytest.raises(TlangAttributeError) as exc_info:
        ShaderProcessor('t', src)
    msg = str(exc_info.value)
    assert 'p1' in msg and 'vertex' in msg
    assert 'p2' in msg and 'bogus' in msg


def test_strict_false_downgrades_broken_program_to_warning():
    src = f"[program('p', frag='fs')]\n{FUNCS_SRC}"  # missing vertex
    proc = ShaderProcessor('t', src, strict=False)  # must not raise
    assert 'p' not in proc.programs
