"""Tests for pyphi.models.diff (B15 result.diff())."""

import pyphi


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
