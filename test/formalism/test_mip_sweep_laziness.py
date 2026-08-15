"""The mechanism-MIP sweep consumes its partitions lazily.

The sweep stops at the first reducible partition, so the partitions it never
reaches must never be constructed either. Materializing them costs the full
partition count in time and memory regardless of where the sweep stops — for a
fully connected 6-unit system, 31.9 million partitions built to evaluate 4.4
million. Call counts of PyPhi operations do not move when that happens, so the
property is pinned here directly.
"""

from __future__ import annotations

import pytest

from pyphi import System
from pyphi import config
from pyphi import examples
from pyphi.conf import presets
from pyphi.cost import partition_sweep_count
from pyphi.direction import Direction
from pyphi.formalism import queries


@pytest.fixture
def _iit4():
    with config.override(**presets.iit4_2023, progress_bars=False):
        yield


def _counting_partitions(monkeypatch):
    """Patch the sweep's partition source to count what it actually consumes."""
    consumed = []
    original = queries.mechanism_partitions

    def counting(mechanism, purview, node_labels=None):
        for partition in original(mechanism, purview, node_labels):
            consumed.append(1)
            yield partition

    monkeypatch.setattr(queries, "mechanism_partitions", counting)
    return consumed


def _system():
    return System(examples.basic_substrate(), (1, 0, 0), node_indices=(0, 1, 2))


def test_a_truncated_sweep_builds_only_what_it_evaluates(_iit4, monkeypatch):
    """Short-circuiting stops construction, not just evaluation.

    Mechanism AB over purview ABC on the cause side is reducible, so the sweep
    meets a partition with zero φ partway through and stops there.
    """
    consumed = _counting_partitions(monkeypatch)
    system = _system()
    mechanism = (0, 1)
    purview = (0, 1, 2)

    with config.override(shortcircuit_sia=True):
        mip = system.find_mip(Direction.CAUSE, mechanism, purview)

    assert float(mip.phi) == 0.0, "the fixture must be one the sweep truncates"
    total = partition_sweep_count(len(mechanism), len(purview))
    assert sum(consumed) < total, (
        f"the sweep built all {total} partitions; short-circuiting should stop "
        "construction as well as evaluation"
    )


def test_a_complete_sweep_builds_every_partition(_iit4, monkeypatch):
    """With no short-circuit every partition is built, and none more than once.

    The search runs one sweep per specified-state pin, so an exhaustive search
    consumes a whole multiple of the partition count — never a fraction of it
    (partitions skipped) and never an odd surplus (partitions rebuilt).
    """
    consumed = _counting_partitions(monkeypatch)
    system = _system()
    mechanism = (0, 1)
    purview = (0, 1, 2)

    with config.override(shortcircuit_sia=False):
        system.find_mip(Direction.CAUSE, mechanism, purview)

    total = partition_sweep_count(len(mechanism), len(purview))
    assert sum(consumed) >= total
    assert sum(consumed) % total == 0


def test_the_margin_is_reported_for_a_complete_sweep(_iit4):
    """A lazily consumed sweep still knows whether it saw every partition.

    The margin is meaningful only against the full partition set, so it is
    withheld when the sweep stopped early. Recognising a complete sweep uses
    the memoized partition count rather than the length of a materialized
    list.
    """
    system = _system()
    with config.override(shortcircuit_sia=False):
        mip = system.find_mip(Direction.EFFECT, (0, 1, 2), (0, 1, 2))
    assert mip.partition_margin is not None


def test_the_partition_count_matches_the_scheme(_iit4):
    """The count standing in for ``len(partitions)`` is the enumeration's own.

    It depends on the mechanism and purview sizes alone, which is what lets a
    lazily consumed sweep recognise completeness without materializing.
    """
    for mechanism, purview in (((0,), (0, 1)), ((0, 1), (0, 1, 2)), ((0, 1, 2), (2,))):
        enumerated = sum(1 for _ in queries.mechanism_partitions(mechanism, purview))
        assert enumerated == partition_sweep_count(len(mechanism), len(purview))
