"""Tests for pyphi.models.diff (B15 result.diff())."""

import pyphi
from pyphi.conf import presets


def test_resultdiff_describe_and_pandas():
    from pyphi.models.diff import Change
    from pyphi.models.diff import ResultDiff

    rd = ResultDiff(
        subject="ΔΦ_s = +0.10",
        level="system",
        delta_phi=0.1,
        mip_changed=True,
        binding_direction_changed=False,
        changes=(
            Change(kind="distinction_gained", key=(0,), a_value=None, b_value=0.25),
        ),
        config_diff={"numerics.precision": (13, 6)},
    )
    assert "ΔΦ_s = +0.10" in repr(rd)
    assert "distinction_gained" in repr(rd)
    assert "<" in rd._repr_html_()  # HTML backend rendered markup

    df = rd.to_pandas()
    assert list(df.columns) == ["category", "key", "a", "b"]
    # one row per change + one per config-diff entry + scalar rows
    assert (df["category"] == "distinction_gained").any()
    assert (df["category"] == "config").any()


def test_mip_reshuffle_not_flagged_but_real_change_is(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import _diff_common

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    # Same analysis recomputed: identical phi, MIP is a co-optimal member of a.ties.
    b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    common = _diff_common(a, b)
    assert common["mip_changed"] is False  # identical / tie-equivalent MIP
    assert float(common["delta_phi"]) == 0.0
    assert common["config_diff"] == {}


def test_config_diff_surfaces_precision_change(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import _diff_common

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    with pyphi.config.override(precision=6):
        b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    common = _diff_common(a, b)
    assert "numerics.precision" in common["config_diff"]


def test_iit4_sia_diff(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import ResultDiff

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    assert float(rd.delta_phi) == 0.0
    assert rd.mip_changed is False


def test_diff_type_mismatch_raises(s):
    import pytest

    from pyphi.formalism import FORMALISM_REGISTRY

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)

    with pytest.raises(TypeError):
        a.diff("not a result")


def test_ces_diff_distinctions_and_relations(s):
    from pyphi.formalism import iit3
    from pyphi.models.diff import ResultDiff

    with pyphi.config.override(**presets.iit3):
        a = iit3.ces(s)
        b = iit3.ces(s)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    # identical CESs: no element changes
    assert rd.changes == ()
    assert float(rd.delta_phi) == 0.0


def test_mechanism_diff(s):
    from pyphi.formalism import iit3
    from pyphi.models.diff import ResultDiff

    with pyphi.config.override(**presets.iit3):
        da = iit3.concept(s, (1,))
        db = iit3.concept(s, (1,))
    rd = da.diff(db)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "mechanism"
    assert float(rd.delta_phi) == 0.0
    # MICE delegates
    assert isinstance(da.cause.diff(db.cause), ResultDiff)


def test_ac_diff():
    from pyphi import actual
    from pyphi import examples
    from pyphi.direction import Direction
    from pyphi.models.diff import ResultDiff

    t = examples.prevention_transition()
    a = actual.sia(t, Direction.BIDIRECTIONAL)
    b = actual.sia(t, Direction.BIDIRECTIONAL)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    assert float(rd.delta_phi) == 0.0

    acc_a = actual.account(t, Direction.BIDIRECTIONAL)
    acc_b = actual.account(t, Direction.BIDIRECTIONAL)
    rd_acc = acc_a.diff(acc_b)
    assert isinstance(rd_acc, ResultDiff)
    assert rd_acc.changes == ()  # identical accounts


def test_diff_is_total(s):
    """Every top-level result type diffs against another of its kind into a
    valid, renderable, exportable ResultDiff (the B15 coverage invariant)."""
    from pyphi import actual
    from pyphi import examples
    from pyphi.direction import Direction
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3
    from pyphi.models.diff import ResultDiff

    pairs = []
    pairs.append(
        (
            FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s),
            FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s),
        )
    )
    with pyphi.config.override(**presets.iit3):
        pairs.append((iit3.sia(s), iit3.sia(s)))
        pairs.append((iit3.ces(s), iit3.ces(s)))
        da, db = iit3.concept(s, (1,)), iit3.concept(s, (1,))
    pairs.append((da, db))  # Distinction
    pairs.append((da.cause, db.cause))  # MICE
    pairs.append((da.cause.ria, db.cause.ria))  # RIA
    t = examples.prevention_transition()
    pairs.append(
        (
            actual.sia(t, Direction.BIDIRECTIONAL),
            actual.sia(t, Direction.BIDIRECTIONAL),
        )
    )
    pairs.append(
        (
            actual.account(t, Direction.BIDIRECTIONAL),
            actual.account(t, Direction.BIDIRECTIONAL),
        )
    )

    for a, b in pairs:
        rd = a.diff(b)
        name = type(a).__name__
        assert isinstance(rd, ResultDiff), name
        assert rd.level in {"system", "mechanism"}, name
        assert repr(rd)
        rd.to_pandas()


def test_ces_diff_with_differing_relation_sets():
    """Relation gained/lost changes on CESs whose relation sets differ."""
    from pyphi import examples
    from pyphi.models.diff import ResultDiff

    substrate = examples.basic_substrate()
    state_a = examples.basic_state()
    state_b = tuple(1 - s if i == 0 else s for i, s in enumerate(state_a))
    with pyphi.config.override(**presets.iit4_2023):
        a = substrate.ces(state_a)
        b = substrate.ces(state_b)
    n_a = a.relations.num_relations()
    n_b = b.relations.num_relations()
    assert n_a != n_b  # precondition: the relation sets genuinely differ

    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    relation_changes = [
        c for c in rd.changes if c.kind in ("relation_gained", "relation_lost")
    ]
    assert len(relation_changes) >= abs(n_b - n_a)
    for change in relation_changes:
        # Keys are deterministic: mechanisms in sorted order.
        assert change.key == tuple(sorted(change.key))


def test_ces_diff_relation_statistic_deltas_on_analytical():
    """Analytical relations produce relation statistic deltas (Σφ_r, count,
    per-degree) even though they cannot be enumerated for gained/lost rows."""
    from pyphi import examples
    from pyphi.relations import AnalyticalRelations

    substrate = examples.basic_substrate()
    state_a = examples.basic_state()
    state_b = tuple(1 - v if i == 0 else v for i, v in enumerate(state_a))
    with pyphi.config.override(**presets.iit4_2023, relation_computation="ANALYTICAL"):
        a = substrate.ces(state_a)
        b = substrate.ces(state_b)
    assert isinstance(a.relations, AnalyticalRelations)  # precondition
    rd = a.diff(b)
    kinds = {c.kind for c in rd.changes}
    # At least one statistic delta is present, and no per-relation rows (the
    # analytical backend is not enumerable).
    assert kinds & {"relation_sum_phi", "relation_count", "relation_degree"}
    assert not (kinds & {"relation_gained", "relation_lost"})


def test_mip_reshuffle_and_genuine_change_branches():
    """Exercise both _mip_changed branches with genuinely different partitions.

    A tie-reshuffle (b picks a co-optimal member of a's tie set) is not a
    change; a partition outside a's tie set is. The no-ties fallback flags a
    change only when the partition AND phi both differ.
    """
    from types import SimpleNamespace

    from pyphi.direction import Direction
    from pyphi.models import DirectedBipartition
    from pyphi.models.diff import _mip_changed

    p1 = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    p2 = DirectedBipartition(Direction.EFFECT, (0, 2), (1,))
    p3 = DirectedBipartition(Direction.CAUSE, (1, 2), (0,))
    assert len({p.lex_key() for p in (p1, p2, p3)}) == 3  # genuinely distinct

    def tie(part):
        return SimpleNamespace(partition=part)

    a = SimpleNamespace(partition=p1, ties=[tie(p1), tie(p2)], phi=1.0)
    # Reshuffle: b selected the co-optimal p2.
    assert (
        _mip_changed(a, SimpleNamespace(partition=p2, ties=[tie(p2)], phi=1.0)) is False
    )
    # Genuine change: p3 is outside a's tie set.
    assert (
        _mip_changed(a, SimpleNamespace(partition=p3, ties=[tie(p3)], phi=0.5)) is True
    )

    # Fallback path (no tie set on a): flag only partition + phi both changed.
    a_no_ties = SimpleNamespace(partition=p1, ties=None, phi=1.0)
    assert (
        _mip_changed(a_no_ties, SimpleNamespace(partition=p3, ties=None, phi=0.5))
        is True
    )
    assert (
        _mip_changed(a_no_ties, SimpleNamespace(partition=p3, ties=None, phi=1.0))
        is False
    )
