# -------------------------------------------------------------
# @file          test_build_info.py
# @description   GL-free regression tests for `tlang.__version__` and
#                `tlang.build_info()` -- the "what am I actually running?"
#                helper added because a stale, non-editable copy of the
#                package (missing exports, older layout) resolved imports
#                fine and reported its own version, with no way short of
#                manually inspecting `tlang.__file__` to tell.
# -------------------------------------------------------------

from pathlib import Path

import tlang


def test_version_is_a_non_empty_string():
    assert isinstance(tlang.__version__, str)
    assert tlang.__version__ != ''


def test_version_is_exported():
    assert '__version__' in tlang.__all__
    assert 'build_info' in tlang.__all__


def test_build_info_package_dir_matches_tlang_file():
    """The load-bearing fact `build_info()` exists to surface: the resolved
    package directory must be the same directory `tlang.__file__` actually
    lives in, not a hardcoded or cached guess."""
    info = tlang.build_info()
    assert info['package_dir'] == Path(tlang.__file__).parent.resolve()


def test_build_info_version_matches_dunder_version():
    info = tlang.build_info()
    assert info['version'] == tlang.__version__


def test_build_info_editable_is_a_bool():
    info = tlang.build_info()
    assert isinstance(info['editable'], bool)
