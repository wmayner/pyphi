"""The thread-backed parallel SIA must equal the sequential SIA.

Companion to test_parallel_equals_sequential.py (which covers the loky process
scheduler). The thread scheduler shares the parent's caches across worker
threads; this guards that the sharing produces identical results. Runs on any
interpreter — it is the standard-lane guard for the free-threaded code path,
since the thread scheduler is what ``parallel_backend="auto"`` selects on a
free-threaded runtime.
"""

from __future__ import annotations

import pytest

from pyphi import System
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets

_SUBSTRATES = {
    "basic": (examples.basic_substrate, (1, 0, 0)),
    "xor": (examples.xor_substrate, (0, 0, 0)),
}


def _thread_override(threshold: int = 2) -> dict:
    """Force the thread scheduler on the outer SIA/CES evaluation levels at a
    low sequential threshold so dispatch actually parallelizes (map_reduce
    parallelizes only when a level produces more than one chunk)."""
    c = config.infrastructure
    forced = {"parallel": True, "sequential_threshold": threshold}
    keys = (
        "parallel_partition_evaluation",
        "parallel_concept_evaluation",
        "parallel_purview_evaluation",
    )
    return {
        "parallel": True,
        "parallel_backend": "thread",
        **{k: {**getattr(c, k), **forced} for k in keys},
    }


@pytest.mark.parametrize("name", list(_SUBSTRATES))
def test_iit4_sia_thread_backend_equals_sequential(name: str) -> None:
    """IIT 4.0 (2023, GID): the thread-backed SIA equals the sequential SIA.

    The thread scheduler runs the cut, concept, and purview levels across
    worker threads that share the parent's content caches; the result must be
    identical to sequential evaluation.
    """
    factory, state = _SUBSTRATES[name]
    with config.override(**presets.iit4_2023, parallel=False):
        seq = System(factory(), state).sia()
    with config.override(**presets.iit4_2023, **_thread_override()):
        par = System(factory(), state).sia()

    assert seq == par, (
        f"{name}: IIT 4.0 SIA diverged under the thread backend — sequential "
        f"φ {seq.phi} vs thread φ {par.phi}"
    )
