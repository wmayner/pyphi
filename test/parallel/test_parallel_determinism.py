"""Determinism guards for the parallel layer.

Parallel evaluation must yield the same results as sequential evaluation:
short-circuit truncation must happen at the same submission-order prefix on
every backend, tie selection must not depend on worker completion order, and
a worker exception must not leave orphaned work running in the executors.
"""

from __future__ import annotations

from pyphi.parallel import false
from pyphi.parallel.scheduler import ShortcircuitPolicy


def test_shortcircuit_policy_active():
    assert not ShortcircuitPolicy().active
    assert not ShortcircuitPolicy(func=false).active
    assert ShortcircuitPolicy(func=lambda r: r == 0).active
