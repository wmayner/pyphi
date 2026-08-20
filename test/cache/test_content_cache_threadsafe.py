"""Concurrency safety of ContentCache under the thread scheduler.

These tests run on any interpreter; their assertions hold under the GIL and
free-threaded. The eviction crash they guard reproduces reliably free-threaded
(worker threads run Python concurrently) and is timing-dependent under the GIL,
so the tests apply heavy concurrent insert-versus-evict pressure rather than
relying on a guaranteed crash under the GIL.
"""

from __future__ import annotations

import gc
import itertools
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from pyphi.cache.content import ContentCache
from pyphi.conf import config


def test_concurrent_evict_and_insert_does_not_crash() -> None:
    """evict() running concurrently with get_or_compute() inserts must not
    raise (today: RuntimeError: dictionary changed size during iteration)."""
    cache = ContentCache("test.concurrent_evict")
    fingerprints = [bytes([i]) * 32 for i in range(8)]

    # Seed a sizable cache so evict's key scan spans many entries.
    for fp in fingerprints:
        for j in range(200):
            cache.get_or_compute(fp, (j,), lambda j=j: j)

    errors: list[BaseException] = []
    stop = threading.Event()

    def inserter(fp: bytes) -> None:
        j = 0
        while not stop.is_set():
            try:
                cache.get_or_compute(fp, ("ins", j), lambda j=j: j)
            except BaseException as exc:
                errors.append(exc)
                return
            j += 1

    def evictor(fp: bytes) -> None:
        while not stop.is_set():
            try:
                cache.evict(fp)
            except BaseException as exc:
                errors.append(exc)
                return

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for fp in fingerprints:
            futures.append(pool.submit(inserter, fp))
            futures.append(pool.submit(evictor, fp))
        # Let them race, then stop.
        for _ in range(2000):
            if errors:
                break
        stop.set()
        for fut in futures:
            fut.result()

    assert not errors, f"concurrent evict/insert raised: {errors[:3]}"


def test_hits_concurrent_with_bounded_admission_do_not_raise() -> None:
    """Guards defect: ``ByteBoundedStore.admit``'s eviction loop iterated the
    live dict (``next(iter(self.data.items()))``), so a concurrent lock-free
    hit (pop + reinsert) raised ``RuntimeError: dictionary changed size during
    iteration`` — violating the module contract that no operation raises under
    concurrent access. Pre-fix this crashed within 0.1s; the loop below is
    bounded at 1.5s with early exit on error."""
    cache = ContentCache("test.bounded_admission_race")

    # Warm while memory is ample: the store grows freely, budget stays unset.
    with config.override(memory_ceiling_bytes=10**15):
        for i in range(40):
            cache.get_or_compute(b"fp", ("warm", i), lambda i=i: i)

    errors: list[BaseException] = []
    stop = threading.Event()
    counter = itertools.count()

    def hitter() -> None:
        """Lock-free pop + reinsert on existing keys."""
        try:
            while not stop.is_set():
                for i in range(40):
                    cache.get_or_compute(b"fp", ("warm", i), lambda i=i: i)
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    def misser() -> None:
        """Fresh keys: the locked admit runs the eviction loop every time."""
        try:
            while not stop.is_set():
                cache.get_or_compute(b"fp", ("miss", next(counter)), lambda: 0)
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        # Ceiling of 1 byte: the budget latches at the current weight, so
        # every admission runs the eviction loop while hitters churn the dict.
        with config.override(memory_ceiling_bytes=1):
            cache.get_or_compute(b"fp", ("latch",), lambda: 0)
            assert cache._store._budget is not None  # bounded regime reached
            threads = [
                threading.Thread(target=hitter, daemon=True) for _ in range(6)
            ] + [threading.Thread(target=misser, daemon=True) for _ in range(6)]
            for t in threads:
                t.start()
            deadline = time.time() + 1.5
            while time.time() < deadline and not errors:
                time.sleep(0.05)
            stop.set()
            for t in threads:
                t.join(2)
    finally:
        sys.setswitchinterval(old_interval)

    assert not errors, f"hit/admit race raised: {errors[:3]}"


def test_concurrent_get_returns_correct_values_and_no_leak() -> None:
    """Concurrent observe()+get_or_compute() always returns the value the
    compute callable would produce, and eviction leaves no leak once all
    carriers are released."""

    class _Carrier:
        """A weakref-able stand-in for a System/Substrate."""

    cache = ContentCache("test.concurrent_values")
    errors: list[BaseException] = []

    def expected(fp: bytes, arg: int) -> int:
        return int.from_bytes(fp[:2], "big") * 1000 + arg

    def worker(seed: int) -> None:
        try:
            for k in range(50):
                fp = bytes([seed % 4, k % 4]) + b"\x00" * 30
                carrier = _Carrier()
                cache.observe(carrier, fp)
                for arg in range(4):
                    got = cache.get_or_compute(
                        fp, (arg,), lambda fp=fp, arg=arg: expected(fp, arg)
                    )
                    if got != expected(fp, arg):
                        raise AssertionError(f"wrong value {got} for {fp!r},{arg}")
                # carrier dies here -> finalizer -> _on_death (concurrent)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(worker, s) for s in range(32)]:
            fut.result()

    assert not errors, f"concurrent access failed: {errors[:3]}"

    # All carriers are gone; force collection and assert sound eviction.
    gc.collect()
    assert cache.size == 0, f"cache leaked {cache.size} entries"
    assert not cache._live, f"_live leaked {cache._live}"
