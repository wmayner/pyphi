import numpy as np

from pyphi import examples
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.campaign.shards import bottleneck_order
from pyphi.campaign.shards import cut_present_edges
from pyphi.campaign.shards import enumerate_partition_stride
from pyphi.campaign.shards import plan_ces_shards
from pyphi.campaign.shards import plan_sia_shards
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.partition import mechanism_partitions
from pyphi.system import System

PIN = {"parallel": False, "progress_bars": False}


def _system():
    return System(examples.basic_substrate(), (1, 0, 0))


def test_generous_budget_yields_mechanism_shards():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), CESScope(), units_per_job=1e9)
    assert all(s.payload_kind == "mechanisms" for s in specs)
    covered = sorted(m for s in specs for m in s.mechanisms)
    assert covered == sorted([(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])


def test_tiny_budget_descends_the_ladder():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), CESScope(), units_per_job=2.0)
    kinds = {s.payload_kind for s in specs}
    assert "partition_stride" in kinds
    strides = [
        s for s in specs if s.payload_kind == "partition_stride" and s.stride is not None
    ]
    assert strides, "expected stride shards under a tiny budget"
    i, k = strides[0].stride
    assert 0 <= i < k


def test_plan_is_deterministic():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        a = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
        b = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
    assert a == b


def test_scope_restricts_the_plan():
    scope = CESScope(mechanisms=AxisScope(explicit=((0,),)))
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), scope, units_per_job=1e9)
    covered = [m for s in specs for m in s.mechanisms]
    assert covered == [(0,)]


def test_stride_enumeration_partitions_the_enumeration():
    system = _system()
    mechanism, purview = (0, 1), (0, 2)
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        full = list(mechanism_partitions(mechanism, purview, system.node_labels))
        k = 3
        seen_indices = []
        seen_parts = []
        for i in range(k):
            parts, indices = enumerate_partition_stride(
                mechanism, purview, system.node_labels, i, k
            )
            assert indices == list(range(i, len(full), k))
            seen_indices.extend(indices)
            seen_parts.extend(str(p) for p in parts)
    assert sorted(seen_indices) == list(range(len(full)))
    assert sorted(seen_parts) == sorted(str(p) for p in full)


def test_bottleneck_order_finds_zero_cut_first():
    # Sparse chain 0 -> 1 -> 2 -> 3 (with self-loops): far-apart mechanism
    # and purview admit partitions that cut no present connection.
    cm = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        parts = list(mechanism_partitions((0, 3), (1, 2)))
        ordered, indices = bottleneck_order(
            parts, list(range(len(parts))), cm, Direction.EFFECT
        )
    counts = [cut_present_edges(p, cm, Direction.EFFECT) for p in ordered]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert len(indices) == len(parts)


def test_sia_shards_cover_system_partitions():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_sia_shards(_system(), units_per_job=5.0)
    assert all(s.payload_kind == "partition_stride" for s in specs)
    assert all(s.mechanism is None for s in specs)
    ks = {s.stride[1] for s in specs}
    assert len(ks) == 1
    assert sorted(s.stride[0] for s in specs) == list(range(ks.pop()))


def test_precomputed_workloads_match_internal_walk():
    from pyphi.cost import mechanism_workloads

    system = _system()
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        workloads = mechanism_workloads(
            system.substrate, subset=system.node_indices, scope=CESScope()
        )
        a = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
        b = plan_ces_shards(
            _system(), CESScope(), units_per_job=5.0, workloads=workloads
        )
    assert a == b


def test_order_cap_restricts_planned_purviews():
    capped = CESScope(max_purview_order_by_mechanism_order=((1, 1),))
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        base = plan_ces_shards(_system(), CESScope(), units_per_job=1e9)
        capped_specs = plan_ces_shards(_system(), capped, units_per_job=1e9)
    total = sum(s.units for s in base)
    capped_total = sum(s.units for s in capped_specs)
    assert capped_total < total
