# -------------------------------------------------------------
# @file          test_gl_buffer_pool.py
# @description   GL-marked regression tests for BufferPool: zeroed
#                temp allocation, segregated free lists (no small
#                request ever consumes an oversized pooled buffer),
#                exception-safe `with pool.temp(...)`, and
#                use-after-free detection.
# -------------------------------------------------------------

import pytest

from tlang.runtime.buffer_pool import BufferPool
from tlang.errors import TlangError

pytestmark = pytest.mark.gl


def test_alloc_temp_zero_true_returns_zeroed_memory_after_prior_write(gl_ctx):
    pool = BufferPool(gl_ctx)
    h1 = pool.alloc_temp(256)
    h1.write(b'\xff' * 256)
    pool.free_temp(h1)

    # same size class -> very likely the exact same GL buffer, recycled
    h2 = pool.alloc_temp(256, zero=True)
    assert h2.read() == b'\x00' * 256
    pool.free_temp(h2)


def test_small_request_never_consumes_an_oversized_pooled_buffer(gl_ctx):
    """Segregated free lists: each power-of-two size class is its own
    free list, so a 512-byte request can never be satisfied by (and
    thereby consume) a pooled 16 MB buffer."""
    pool = BufferPool(gl_ctx)
    big = pool.alloc_temp(16 * 1024 * 1024)
    big_size_class = big.size_class
    pool.free_temp(big)

    metrics_before = pool.metrics()
    assert metrics_before.bytes_pooled == big_size_class

    small = pool.alloc_temp(512)
    assert small.size_class != big_size_class

    metrics_after = pool.metrics()
    # the big buffer must still be sitting idle, untouched by the small request
    assert metrics_after.bytes_pooled == big_size_class
    pool.free_temp(small)


def test_temp_context_manager_frees_buffer_even_when_body_raises(gl_ctx):
    pool = BufferPool(gl_ctx)
    captured = {}

    with pytest.raises(RuntimeError):
        with pool.temp(256) as buf:
            captured['buf'] = buf
            raise RuntimeError("boom")

    assert captured['buf'].alive is False
    assert pool.metrics().bytes_checked_out == 0


def test_use_after_free_raises_tlang_error(gl_ctx):
    pool = BufferPool(gl_ctx)
    handle = pool.alloc_temp(256)
    pool.free_temp(handle)

    with pytest.raises(TlangError):
        handle.write(b'x' * 4)

    with pytest.raises(TlangError):
        pool.free_temp(handle)  # double free
