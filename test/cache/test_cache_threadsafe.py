"""Concurrency safety of the module-level ``@cache()`` decorator.

The thread scheduler shares these caches across worker threads, so the hit
path must not assume single-threaded access.
"""

from __future__ import annotations

import sys
import threading
import time

from pyphi.cache import cache


def test_cache_decorator_concurrent_hits_do_not_raise() -> None:
    """Guards defect: the ``@cache()`` hit path did an unsynchronized get
    followed by ``del entries[key]``, so concurrent hits on the same key
    raised ``KeyError``. Pre-fix this crashed 15/16 threads within 0.1s;
    the loop below is bounded at 1s with early exit on error."""

    @cache()
    def f(a: int, b: int) -> int:
        return a + b

    keys = [(i, i) for i in range(4)]
    for k in keys:
        f(*k)  # warm: every subsequent call is a hit

    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer() -> None:
        try:
            while not stop.is_set():
                for k in keys:
                    assert f(*k) == k[0] + k[1]
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(16)]
        for t in threads:
            t.start()
        deadline = time.time() + 1.0
        while time.time() < deadline and not errors:
            time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join(2)
    finally:
        sys.setswitchinterval(old_interval)

    assert not errors, f"concurrent hits raised: {errors[:3]}"
