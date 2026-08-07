"""Tests for pyphi.cost: the single-system analysis workload pre-flight."""

from math import comb

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.conf import presets
from pyphi.cost import _MECHANISM_PARTITION_COUNT_MEMO
from pyphi.cost import _PARTITION_COUNT_MEMO
from pyphi.cost import estimate_analysis
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partitions
from test.conftest import IIT_3_CONFIG


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


class TestCounts:
    def test_counts_match_direct_enumeration(self):
        est = estimate_analysis(_dense3())
        assert est.n_units == 3
        assert est.compute == "full"
        assert est.state_space_size == 8
        assert est.mechanisms == 7
        assert est.system_partitions == sum(1 for _ in system_partitions((0, 1, 2)))
        # Fully connected: every nonempty purview is a candidate for every
        # mechanism in both directions.
        assert est.purview_evaluations == 7 * 2 * 7
        expected = sum(
            comb(3, a)
            * comb(3, b)
            * 2
            * sum(
                1 for _ in mechanism_partitions(tuple(range(a)), tuple(range(a, a + b)))
            )
            for a in (1, 2, 3)
            for b in (1, 2, 3)
        )
        assert est.mechanism_partition_sweeps == expected
        assert est.relations_closed_form is True
        assert est.possible_distinctions == 7
        assert est.possible_relations == 2**7 - 1
        assert est.capped is False

    def test_dense_3unit_reference_values(self):
        est = estimate_analysis(_dense3())
        assert est.system_partitions == 22
        assert est.purview_evaluations == 98
        assert est.mechanism_partition_sweeps == 1102

    def test_sparse_connectivity_prunes_purviews(self):
        sparse = estimate_analysis(examples.basic_substrate())
        dense = estimate_analysis(_dense3())
        assert sparse.purview_evaluations == 30
        assert sparse.mechanism_partition_sweeps == 526
        assert sparse.purview_evaluations < dense.purview_evaluations
        assert sparse.mechanism_partition_sweeps < dense.mechanism_partition_sweeps

    def test_subset_restricts_the_walk(self):
        est = estimate_analysis(_dense3(), subset=(0, 1))
        assert est.n_units == 2
        assert est.state_space_size == 4
        assert est.mechanisms == 3
        assert est.purview_evaluations == 3 * 2 * 3
        assert est.system_partitions == sum(1 for _ in system_partitions((0, 1)))


class TestScope:
    def test_sia_scope(self):
        est = estimate_analysis(_dense3(), compute="sia")
        assert est.compute == "sia"
        assert est.system_partitions == 22
        assert est.mechanisms is None
        assert est.purview_evaluations is None
        assert est.mechanism_partition_sweeps is None
        assert est.relations_closed_form is None
        assert est.possible_distinctions is None
        assert est.possible_relations is None

    def test_ces_scope_charges_the_iit4_system_partition_axis(self):
        # Under IIT 4.0 unfolding a cause-effect structure computes a system
        # irreducibility analysis first (Eq. 57), so a 'ces' estimate that
        # left this axis out would undercount by the entire partition sweep —
        # unboundedly, for a sparse substrate whose distinction axis is small.
        est = estimate_analysis(_dense3(), compute="ces")
        assert est.compute == "ces"
        assert est.system_partitions == 22
        assert est.mechanism_partition_sweeps == 1102

    def test_ces_scope_omits_the_axis_under_iit3(self):
        with IIT_3_CONFIG:
            est = estimate_analysis(_dense3(), compute="ces")
        assert est.system_partitions is None
        assert est.mechanism_partition_sweeps is not None

    def test_distinctions_scope(self):
        est = estimate_analysis(_dense3(), compute="distinctions")
        assert est.compute == "distinctions"
        assert est.system_partitions is None
        assert est.mechanism_partition_sweeps == 1102
        # Distinctions stop before relations.
        assert est.relations_closed_form is None
        assert est.possible_relations is None
        assert est.possible_distinctions == 2**3 - 1

    def test_unknown_compute_raises(self):
        with pytest.raises(ValueError, match="compute"):
            estimate_analysis(_dense3(), compute="everything")


class TestConfigSensitivity:
    def test_system_partition_scheme_changes_the_count(self):
        with pyphi.config.override(system_partition_scheme="DIRECTED_BIPARTITION"):
            est = estimate_analysis(_dense3(), compute="sia")
            direct = sum(1 for _ in system_partitions((0, 1, 2)))
            assert est.system_partitions == direct
        default = estimate_analysis(_dense3(), compute="sia")
        assert est.system_partitions != default.system_partitions

    def test_concrete_relations_backend_reports_enumeration(self):
        with pyphi.config.override(relation_computation="CONCRETE"):
            est = estimate_analysis(_dense3())
        assert est.relations_closed_form is False
        assert est.possible_relations == 2**7 - 1

    def test_iit3_counts_without_iit4_context(self):
        with IIT_3_CONFIG:
            est = estimate_analysis(_dense3())
            direct = sum(1 for _ in system_partitions((0, 1, 2)))
            assert est.system_partitions == direct
            assert est.mechanism_partition_sweeps is not None
            assert est.relations_closed_form is None
            assert est.possible_distinctions is None
            assert est.possible_relations is None

    def test_kary_work_axes_without_binary_context(self):
        rng = np.random.default_rng(2026)
        f0 = rng.uniform(size=(3, 3, 3))
        f0 = f0 / f0.sum(axis=-1, keepdims=True)
        f1 = rng.uniform(size=(3, 3, 3))
        f1 = f1 / f1.sum(axis=-1, keepdims=True)
        sub = pyphi.Substrate(marginals=[f0, f1], state_space=("LOW", "MID", "HIGH"))
        est = estimate_analysis(sub)
        assert est.state_space_size == 9
        assert est.mechanisms == 3
        assert est.purview_evaluations is not None
        assert est.relations_closed_form is not None
        assert est.possible_distinctions is None
        assert est.possible_relations is None


class TestBudget:
    def test_limit_truncates_the_walk(self):
        est = estimate_analysis(_dense3(), limit=10)
        assert est.capped is True
        assert est.purview_evaluations is not None
        assert est.purview_evaluations < 98
        assert est.mechanism_partition_sweeps < 1102

    def test_memoized_counts_do_not_consume_budget(self):
        # Seeded system-partition counts resolve even under a unit budget.
        est = estimate_analysis(_dense3(), compute="sia", limit=1)
        assert est.system_partitions == 22
        assert est.capped is False


class TestPresentation:
    def test_pandas_record(self):
        record = estimate_analysis(_dense3()).to_pandas()
        assert record["n_units"] == 3
        assert record["mechanisms"] == 7
        assert record["capped"] is False

    def test_card_renders(self):
        est = estimate_analysis(_dense3())
        assert "AnalysisEstimate" in str(est)

    def test_capped_card_uses_lower_bound_qualifier(self):
        est = estimate_analysis(_dense3(), limit=10)
        assert "≥" in str(est)


class TestScopedEstimation:
    def test_scope_narrows_counts(self):
        from pyphi.campaign.scope import AxisScope
        from pyphi.campaign.scope import CESScope

        substrate = examples.basic_substrate()
        full = estimate_analysis(substrate, compute="ces")
        scoped = estimate_analysis(
            substrate,
            compute="ces",
            scope=CESScope(mechanisms=AxisScope(max_order=1)),
        )
        assert scoped.mechanisms == 3  # singletons only
        assert scoped.purview_evaluations < full.purview_evaluations
        assert scoped.mechanism_partition_sweeps < full.mechanism_partition_sweeps

    def test_purview_scope_narrows_purview_axis(self):
        from pyphi.campaign.scope import AxisScope
        from pyphi.campaign.scope import CESScope

        substrate = examples.basic_substrate()
        scope = CESScope(
            cause_purviews=AxisScope(max_order=1),
            effect_purviews=AxisScope(max_order=1),
        )
        full = estimate_analysis(substrate, compute="ces")
        scoped = estimate_analysis(substrate, compute="ces", scope=scope)
        assert scoped.mechanisms == full.mechanisms
        assert scoped.purview_evaluations < full.purview_evaluations

    def test_mechanism_workloads_sum_matches_estimate(self):
        from pyphi.campaign.scope import AxisScope
        from pyphi.campaign.scope import CESScope
        from pyphi.cost import PURVIEW_EVALUATION_UNITS
        from pyphi.cost import mechanism_workloads

        substrate = examples.basic_substrate()
        scope = CESScope(mechanisms=AxisScope(containing=(0,)))
        workloads = mechanism_workloads(substrate, scope=scope)
        scoped = estimate_analysis(substrate, compute="ces", scope=scope)
        assert set(workloads) == {(0,), (0, 1), (0, 2), (0, 1, 2)}
        assert sum(w.units for w in workloads.values()) == (
            PURVIEW_EVALUATION_UNITS * scoped.purview_evaluations
            + scoped.mechanism_partition_sweeps
        )

    def test_partition_sweep_count_matches_enumeration(self):
        from pyphi.cost import partition_sweep_count

        count = partition_sweep_count(2, 2)
        enumerated = len(list(mechanism_partitions((0, 1), (0, 2))))
        assert count == enumerated


def test_estimate_analysis_respects_order_caps():
    from pyphi import examples
    from pyphi.campaign.scope import CESScope
    from pyphi.cost import estimate_analysis

    substrate = examples.basic_substrate()
    base = estimate_analysis(substrate, compute="ces", scope=CESScope())
    capped = estimate_analysis(
        substrate,
        compute="ces",
        scope=CESScope(max_purview_order_by_mechanism_order=((1, 1),)),
    )
    assert capped.purview_evaluations < base.purview_evaluations


def test_mechanism_workloads_records_max_repertoire_cells():
    from pyphi import examples
    from pyphi.cost import mechanism_workloads

    substrate = examples.basic_substrate()  # 3 binary units
    workloads = mechanism_workloads(substrate)
    # connectivity restricts (0,) to cause purviews of size <= 2
    assert workloads[(0,)].max_repertoire_cells == 2**2
    assert workloads[(0, 1, 2)].max_repertoire_cells == 2**3
    total = sum(w.units for w in workloads.values())
    assert total > 0


def test_shard_memory_bytes_and_rounding():
    from pyphi.cost import BASE_MEMORY_BYTES
    from pyphi.cost import CACHE_HEADROOM_BYTES
    from pyphi.cost import REPERTOIRE_FACTOR
    from pyphi.cost import round_memory_bytes
    from pyphi.cost import shard_memory_bytes

    floor = BASE_MEMORY_BYTES + CACHE_HEADROOM_BYTES
    assert shard_memory_bytes(0) == floor
    assert shard_memory_bytes(100) == REPERTOIRE_FACTOR * 8 * 100 + floor
    half_gb = 512 * 1024**2
    assert round_memory_bytes(1) == half_gb
    assert round_memory_bytes(half_gb) == half_gb
    assert round_memory_bytes(half_gb + 1) == 2 * half_gb


def _spy_purview_calls(monkeypatch):
    """Record the (direction, max_order) of every purview enumeration."""
    from pyphi.substrate import Substrate

    calls: list = []
    orig = Substrate.potential_purviews

    def spy(self, direction, mechanism, max_order=None):
        calls.append((direction, max_order))
        return orig(self, direction, mechanism, max_order=max_order)

    monkeypatch.setattr(Substrate, "potential_purviews", spy)
    return calls


def test_scoped_walks_bound_the_purview_enumeration(monkeypatch):
    """A scoped purview cap reaches the enumeration itself, not just the
    post-filter — otherwise large substrates pay the full powerset."""
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope
    from pyphi.cost import mechanism_workloads
    from pyphi.direction import Direction

    scope = CESScope(
        cause_purviews=AxisScope(max_order=2),
        effect_purviews=AxisScope(max_order=1),
    )
    substrate = _dense3()
    calls = _spy_purview_calls(monkeypatch)

    mechanism_workloads(substrate, scope=scope)
    assert calls
    assert all(mo == (2 if d == Direction.CAUSE else 1) for d, mo in calls)

    calls.clear()
    estimate_analysis(substrate, compute="ces", scope=scope)
    assert calls
    assert all(mo == (2 if d == Direction.CAUSE else 1) for d, mo in calls)
