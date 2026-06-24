"""Integration test: after running an IIT 4.0 SIA computation, confirm
that the kernel + combinatorial caches all show up in the registry."""

from __future__ import annotations

import pytest

from pyphi import cache
from pyphi import examples
from pyphi.formalism.queries import sia
from pyphi.system import System


@pytest.fixture(autouse=True)
def _clear_caches():
    cache.clear_all()
    yield
    cache.clear_all()


def test_sia_run_populates_kernel_and_combinatorial_caches():
    """End-to-end: a SIA run touches kernel and combinatorial caches."""
    substrate = examples.basic_substrate()
    cs = System(
        substrate=substrate,
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )
    _ = sia(cs)

    info = cache.info()

    kernel_keys = [k for k in info if k.startswith("kernel.")]
    combinatorial_keys = [
        k
        for k in info
        if k.startswith("pyphi.partition.")
        or k.startswith("pyphi.combinatorics.")
        or k.startswith("pyphi.distribution.")
    ]

    assert kernel_keys, f"no kernel cache entries; got: {sorted(info)}"
    assert combinatorial_keys, f"no combinatorial cache entries; got: {sorted(info)}"

    # At least one kernel cache has non-zero size after a SIA run.
    assert any(info[k].currsize > 0 for k in kernel_keys)


def test_clear_all_empties_every_registered_cache():
    """After clear_all, every cache has currsize == 0."""
    substrate = examples.basic_substrate()
    cs = System(
        substrate=substrate,
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )
    _ = sia(cs)
    assert any(v.currsize > 0 for v in cache.info().values())

    cache.clear_all()

    for name, ci in cache.info().items():
        assert ci.currsize == 0, f"cache {name!r} still has {ci.currsize} entries"
