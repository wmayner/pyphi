"""The dask-backed parallel SIA must equal the sequential SIA.

Companion to test_parallel_equals_sequential.py (loky process scheduler) and
test_thread_backend_equals_sequential.py. The dask scheduler ships each chunk
to distributed worker processes with the caller's configuration snapshot;
this guards that distribution preserves results exactly. Runs against a
local two-worker cluster; nested parallel levels run in-task by design, so
the outermost level exercises distribution.
"""

from __future__ import annotations

import pytest

from pyphi import System
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets

distributed = pytest.importorskip("distributed")

_SUBSTRATES = {
    "basic": (examples.basic_substrate, (1, 0, 0)),
    "xor": (examples.xor_substrate, (0, 0, 0)),
}


def _dask_override(threshold: int = 2) -> dict:
    """Force the dask scheduler on the outer SIA/CES evaluation levels at a
    low sequential threshold so dispatch actually parallelizes (map_reduce
    parallelizes only when a level produces more than one chunk)."""
    c = config.infrastructure
    forced = {"parallel": True, "sequential_threshold": threshold}
    keys = (
        "parallel_partition_evaluation",
        "parallel_distinction_evaluation",
        "parallel_purview_evaluation",
    )
    return {
        "parallel": True,
        "parallel_backend": "dask",
        **{k: {**getattr(c, k), **forced} for k in keys},
    }


@pytest.mark.parametrize("name", list(_SUBSTRATES))
def test_iit4_sia_dask_backend_equals_sequential(name: str, dask_client) -> None:
    """IIT 4.0 (2023, GID): the dask-backed SIA equals the sequential SIA."""
    factory, state = _SUBSTRATES[name]
    with config.override(**presets.iit4_2023, parallel=False):
        seq = System(factory(), state).sia()
    with config.override(**presets.iit4_2023, **_dask_override()):
        par = System(factory(), state).sia()

    assert seq == par, (
        f"{name}: IIT 4.0 SIA diverged under the dask backend — sequential "
        f"φ {seq.phi} vs dask φ {par.phi}"
    )
