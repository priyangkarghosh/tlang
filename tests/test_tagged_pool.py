# -------------------------------------------------------------
# @file          test_tagged_pool.py
# @description   GL-marked regression tests for BufferPool's tagged allocation: alloc_temp(...,
#                tag=...) registers under that name, free_temp unregisters it, a live tag can't
#                be reused (persistent or temp, either direction), untagged alloc_temp behaves
#                exactly as before, and the pool satisfies collections.abc.Mapping[str, Buffer]
#                over both its persistent and tagged-temp names.
# -------------------------------------------------------------

from collections.abc import Mapping

import pytest

from tlang.runtime.buffer_pool import BufferPool, TempHandle
from tlang.errors import TlangError

pytestmark = pytest.mark.gl


def test_tagged_alloc_registers_and_looks_up_by_name(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(256, tag='Scratch')

    assert 'Scratch' in pool
    assert pool['Scratch'] is handle
    assert handle.tag == 'Scratch'

    pool.free_temp(handle)


def test_free_temp_unregisters_the_tag(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(256, tag='Scratch')
    pool.free_temp(handle)

    assert 'Scratch' not in pool
    with pytest.raises(KeyError):
        pool['Scratch']


def test_freed_and_reallocated_tag_resolves_to_the_new_handle(gl_ctx):
    pool = BufferPool(gl_ctx)
    first = pool.alloc_temp(256, tag='Scratch')
    pool.free_temp(first)

    second = pool.alloc_temp(256, tag='Scratch')
    assert pool['Scratch'] is second
    assert pool['Scratch'] is not first
    pool.free_temp(second)


def test_duplicate_live_tag_raises_naming_both(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(256, tag='Scratch')

    with pytest.raises(TlangError) as exc_info:
        pool.alloc_temp(64, tag='Scratch')
    msg = str(exc_info.value)
    assert 'Scratch' in msg

    pool.free_temp(handle)
    # freed -- the tag is available again, no error this time.
    other = pool.alloc_temp(64, tag='Scratch')
    pool.free_temp(other)


def test_tag_colliding_with_a_persistent_name_raises_either_direction(gl_ctx):
    pool = BufferPool(gl_ctx)
    pool.persistent_buffer('Named', size=64)

    with pytest.raises(TlangError, match='Named'):
        pool.alloc_temp(64, tag='Named')

    handle = pool.alloc_temp(64, tag='OtherTag')
    with pytest.raises(TlangError, match='OtherTag'):
        pool.persistent_buffer('OtherTag', size=64)
    pool.free_temp(handle)


def test_untagged_alloc_temp_behaviour_unchanged(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(256)
    assert handle.tag is None
    assert len(pool) == 0  # untagged temporaries never appear in the Mapping

    pool.free_temp(handle)
    assert len(pool) == 0


def test_pool_satisfies_mapping_protocol(gl_ctx):
    pool = BufferPool(gl_ctx)
    assert isinstance(pool, Mapping)

    persistent = pool.persistent_buffer('Persist', size=64)
    tagged = pool.alloc_temp(128, tag='Temp')

    assert len(pool) == 2
    assert set(iter(pool)) == {'Persist', 'Temp'}
    assert dict(pool.items()) == {'Persist': persistent, 'Temp': tagged}
    assert pool.get('NoSuchTag') is None
    assert pool.get('Persist') is persistent

    pool.free_temp(tagged)
    assert len(pool) == 1
    assert list(pool) == ['Persist']


def test_persistent_buffer_is_idempotent_and_visible_in_mapping(gl_ctx):
    pool = BufferPool(gl_ctx)
    first = pool.persistent_buffer('Named', size=64)
    second = pool.persistent_buffer('Named', size=64)
    assert first is second
    assert pool['Named'] is first


def test_alloc_temp_tag_is_a_temp_handle(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(64, tag='Scratch')
    assert isinstance(pool['Scratch'], TempHandle)
    pool.free_temp(handle)
