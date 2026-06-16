"""Tests for pyphi.models.explanation (B8 result.explain())."""

import pytest

from pyphi.models.explanation import NullResultReason


def test_every_reason_has_a_structural_level():
    for reason in NullResultReason:
        assert reason.level in {"system", "mechanism"}


def test_level_partition_is_correct():
    system = {
        NullResultReason.NO_SYSTEM,
        NullResultReason.NO_STRONG_CONNECTIVITY,
        NullResultReason.NO_WEAK_CONNECTIVITY,
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


def test_ac_null_sia_carries_reason():
    from pyphi import actual
    from pyphi import examples
    from pyphi.direction import Direction

    substrate = examples.actual_causation_substrate()
    # Over the OR-AND substrate this transition has an empty unpartitioned
    # account in the cause direction, so the AC SIA short-circuits to alpha = 0.
    transition = actual.Transition(substrate, (1, 1), (0, 0), (0,), (1,))
    sia = actual.sia(transition, Direction.CAUSE)
    assert float(sia.alpha) == 0
    assert NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE in (sia.reasons or [])
    assert all(isinstance(r, NullResultReason) for r in sia.reasons)


def test_runner_up_retained_on_phi_positive_system(s):
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3

    # IIT 4.0: a phi>0 system has a partition whose phi exceeds the MIP's.
    sia4 = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    assert sia4.phi > 0
    assert sia4.runner_up is not None
    assert float(sia4.runner_up.phi) > float(sia4.phi)

    # IIT 3.0: same property along the distribution-distance path.
    with pyphi.config.override(**presets.iit3):
        sia3 = iit3.sia(s)
    assert sia3.phi > 0
    assert sia3.runner_up is not None
    assert float(sia3.runner_up.phi) > float(sia3.phi)


def test_iit4_sia_explain_short_circuit_and_positive(s, s_empty):
    from pyphi.formalism import FORMALISM_REGISTRY

    # Short-circuit: an empty system → a NO_SYSTEM null-result finding.
    null_sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s_empty)
    expl = null_sia.explain()
    assert expl.level == "system"
    assert any(f.kind == "null_result" for f in expl.findings)
    assert any(f.value is NullResultReason.NO_SYSTEM for f in expl.findings)

    # phi>0: winning partition + binding direction + runner-up/gap findings.
    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    expl = sia.explain()
    kinds = {f.kind for f in expl.findings}
    assert {"winning_partition", "binding_direction"} <= kinds
    assert sia.runner_up is not None
    gap = next(f for f in expl.findings if f.kind == "gap")
    assert float(gap.value) == pytest.approx(float(sia.runner_up.phi) - float(sia.phi))
