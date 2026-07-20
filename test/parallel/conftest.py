"""Fixtures shared by the parallel-backend tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def dask_client():
    """A local two-worker Dask cluster, registered as the current client.

    Workers are single-threaded separate processes, matching the
    deployment recommendation for CPU-bound work.
    """
    distributed = pytest.importorskip("distributed")

    with (
        distributed.LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client
