"""Tests for pyphi.models.explanation (B8 result.explain())."""

from pyphi.models.explanation import NullResultReason


def test_every_reason_has_a_structural_level():
    for reason in NullResultReason:
        assert reason.level in {"system", "mechanism"}


def test_level_partition_is_correct():
    system = {
        NullResultReason.NO_SYSTEM,
        NullResultReason.NO_STRONG_CONNECTIVITY,
        NullResultReason.MONAD_WITH_NO_SELFLOOP,
        NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI,
        NullResultReason.NO_VALID_PARTITIONS,
        NullResultReason.NO_CAUSE,
        NullResultReason.NO_EFFECT,
        NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE,
    }
    mechanism = {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
    }
    assert {r for r in NullResultReason if r.level == "system"} == system
    assert {r for r in NullResultReason if r.level == "mechanism"} == mechanism
    assert system | mechanism == set(NullResultReason)


def test_explanation_describe_and_pandas():
    from pyphi.models.explanation import Explanation
    from pyphi.models.explanation import Finding

    expl = Explanation(
        subject="Φ_s = 0.0",
        level="system",
        findings=(
            Finding(
                kind="null_result",
                label="Reason",
                value=NullResultReason.NO_STRONG_CONNECTIVITY,
            ),
            Finding(kind="binding_direction", label="Binding direction", value="CAUSE"),
        ),
    )
    # repr/HTML render without error and mention the subject + reason.
    assert "Φ_s = 0.0" in repr(expl)
    assert "NO_STRONG_CONNECTIVITY" in repr(expl)
    assert "<" in expl._repr_html_()  # HTML backend produced markup

    df = expl.to_pandas()
    assert list(df.columns) == ["level", "kind", "label", "value"]
    assert len(df) == 2
    assert df.iloc[0]["kind"] == "null_result"


def test_iit3_null_sia_carries_reason(s_empty):
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with pyphi.config.override(**presets.iit3):
        analysis = iit3.sia(s_empty)
    assert analysis.phi == 0
    assert NullResultReason.NO_SYSTEM in (analysis.reasons or [])
