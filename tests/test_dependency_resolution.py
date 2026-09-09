# -------------------------------------------------------------
# @file          test_dependency_resolution.py
# @description   GL-free regression tests for
#                DependencyManager.resolve_dependencies -- transitive
#                include resolution, order-independence, and
#                circular/missing dependency detection.
#
#                ShaderProcessor does not require a GL context (it only
#                does text/regex processing), so these are fully GL-free.
# -------------------------------------------------------------

import pytest

from tlang.compiler.dependency_manager import DependencyManager
from tlang.errors import TlangDependencyError
from tlang.compiler.shader_processor import ShaderProcessor


def _processor(name: str, src: str) -> ShaderProcessor:
    return ShaderProcessor(name, src)


def _register_chain(order):
    """a -> b -> c include chain; only 'c' declares [extend(int64)].
    `order` controls the sequence modules are registered with the
    DependencyManager, independent of the dependency structure itself."""
    procs = {
        'a': _processor('a', "[include(b)]\n"),
        'b': _processor('b', "[include(c)]\n"),
        'c': _processor('c', "[extend(int64)]\n"),
    }
    dm = DependencyManager({})
    for name in order:
        dm.register(procs[name])
    return dm, procs


@pytest.mark.parametrize('order', [['a', 'b', 'c'], ['c', 'b', 'a'], ['b', 'a', 'c']])
def test_transitive_extension_propagation_is_order_independent(order):
    """Regression: extension propagation used to depend on filesystem glob
    discovery order. With an a -> b -> c include chain where only 'c'
    declares [extend('int64')], 'a' must end up seeing that extension
    regardless of the order modules were registered in."""
    dm, procs = _register_chain(order)

    deps = dm.resolve_dependencies('a')
    assert deps == ['c', 'b', 'a'], f"expected dependency-first topological order, got {deps}"

    # mirror ShaderManager's own transitive extension-propagation loop
    for dep in dm.resolve_dependencies('a'):
        if dep != 'a':
            procs['a'].ext.update(procs[dep].ext)

    assert procs['c'].ext, "sanity: 'c' should have picked up the int64 extension group"
    assert procs['a'].ext == procs['c'].ext, (
        f"'a' did not transitively see c's extension in order {order}: {procs['a'].ext}"
    )


def test_circular_include_raises():
    procs = {
        'a': _processor('a', "[include(b)]\n"),
        'b': _processor('b', "[include(a)]\n"),
    }
    dm = DependencyManager({})
    for p in procs.values():
        dm.register(p)

    with pytest.raises(TlangDependencyError):
        dm.resolve_dependencies('a')


def test_missing_include_raises():
    proc = _processor('a', "[include(does_not_exist)]\n")
    dm = DependencyManager({})
    dm.register(proc)

    with pytest.raises(TlangDependencyError):
        dm.resolve_dependencies('a')


def test_resolve_dependencies_is_memoized():
    dm, _procs = _register_chain(['a', 'b', 'c'])
    first = dm.resolve_dependencies('a')
    second = dm.resolve_dependencies('a')
    assert first == second
    assert first is second  # memoized result object, not just equal
