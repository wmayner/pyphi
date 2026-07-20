"""Tests for pyphi.cost: the single-system analysis workload pre-flight."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.conf import presets
from pyphi.cost import _MECHANISM_PARTITION_COUNT_MEMO
from pyphi.cost import _PARTITION_COUNT_MEMO
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partitions


@pytest.fixture(autouse=True)
def _pin_formalism():
    with pyphi.config.override(**presets.iit4_2026, progress_bars=False):
        yield


def _dense3():
    return examples.basic_substrate(cm=np.ones((3, 3)))


class TestSeeds:
    def test_system_partition_seeds_match_enumeration(self):
        for m in range(1, 7):
            direct = sum(1 for _ in system_partitions(tuple(range(m))))
            assert _PARTITION_COUNT_MEMO[("DIRECTED_SET_PARTITION", m)] == direct

    @pytest.mark.slow
    def test_system_partition_seeds_match_enumeration_large(self):
        # m = 9 (240 s to enumerate) is excluded; its seed was verified by
        # one direct enumeration of the same generator.
        for m in (7, 8):
            direct = sum(1 for _ in system_partitions(tuple(range(m))))
            assert _PARTITION_COUNT_MEMO[("DIRECTED_SET_PARTITION", m)] == direct

    def test_mechanism_partition_seeds_match_enumeration(self):
        for a in range(1, 6):
            for b in range(1, 6):
                direct = sum(
                    1
                    for _ in mechanism_partitions(
                        tuple(range(a)), tuple(range(a, a + b))
                    )
                )
                assert (
                    _MECHANISM_PARTITION_COUNT_MEMO[("JOINT_PARTITION_ALL", a, b)]
                    == direct
                )

    @pytest.mark.slow
    def test_mechanism_partition_seeds_match_enumeration_large(self):
        # Pairs (7, 7) (218 s), (7, 6) (36 s), and (6, 7) (24 s) are
        # excluded; those seeds were verified by one direct enumeration of
        # the same generator.
        pairs = [
            (a, b)
            for a in range(1, 8)
            for b in range(1, 8)
            if (a >= 6 or b >= 6) and (a, b) not in {(6, 7), (7, 6), (7, 7)}
        ]
        for a, b in pairs:
            direct = sum(
                1 for _ in mechanism_partitions(tuple(range(a)), tuple(range(a, a + b)))
            )
            assert (
                _MECHANISM_PARTITION_COUNT_MEMO[("JOINT_PARTITION_ALL", a, b)] == direct
            )
