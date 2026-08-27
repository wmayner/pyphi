"""Tests for pyphi.models.explanation (B8 result.explain())."""

import pytest

from pyphi.models.explanation import NullResultReason
from pyphi.models.explanation import binding_direction_finding
from pyphi.models.explanation import runner_up_from_candidates
from test.conftest import IIT_4_CONFIG


@pytest.fixture(autouse=True)
def _pin_iit4_2023():
    """Pin the 2023/GID formalism for this module, so φ assertions do not
    depend on the ambient default. Tests that need another formalism override
    it locally with a ``with`` block, which nests inside this pin."""
    with IIT_4_CONFIG:
        yield


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
        NullResultReason.NONUNIQUE_SYSTEM_STATE,
        NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE,
    }
    mechanism = {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_POSITIVE_ALPHA,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
        NullResultReason.OTHER_DIRECTION_REDUCIBLE,
    }
    assert {r for r in NullResultReason if r.level == "system"} == system
    assert {r for r in NullResultReason if r.level == "mechanism"} == mechanism
    assert system | mechanism == set(NullResultReason)


def test_other_direction_reducible_is_mechanism_level():
    assert NullResultReason.OTHER_DIRECTION_REDUCIBLE.level == "mechanism"


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
    transition = actual.Transition(substrate, (1, 0), (1, 0), (0,), (1,))
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
    # The runner-up is ranked by normalized φ (the default selection
    # quantity), so the gap is reported in normalized φ as well.
    assert float(gap.value) == pytest.approx(
        float(sia.runner_up.normalized_phi) - float(sia.normalized_phi)
    )


def test_iit3_sia_explain(s, s_empty):
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with pyphi.config.override(**presets.iit3):
        sia = iit3.sia(s)
        null = iit3.sia(s_empty)

    expl = sia.explain()
    assert expl.level == "system"
    assert expl.subject.startswith("Φ")
    assert any(f.kind == "winning_partition" for f in expl.findings)
    assert any(f.kind == "gap" for f in expl.findings)

    null_expl = null.explain()
    assert any(f.value is NullResultReason.NO_SYSTEM for f in null_expl.findings)


def test_mechanism_explain(s):
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with pyphi.config.override(**presets.iit3):
        distinction = iit3.concept(s, (1,))

    expl = distinction.explain()
    assert expl.level == "mechanism"
    # A distinction reports which direction (cause/effect) binds its phi.
    assert any(f.kind == "binding_direction" for f in expl.findings)

    # A MICE delegates to its RIA.
    mice_expl = distinction.cause.explain()
    assert mice_expl.level == "mechanism"
    assert any(f.kind == "winning_partition" for f in mice_expl.findings)


def test_ac_explain():
    from pyphi import actual
    from pyphi import examples
    from pyphi.direction import Direction

    # A null AC SIA explains its short-circuit reason.
    substrate = examples.actual_causation_substrate()
    null_t = actual.Transition(substrate, (1, 0), (1, 0), (0,), (1,))
    null_expl = actual.sia(null_t, Direction.CAUSE).explain()
    assert null_expl.level == "system"
    assert null_expl.subject.startswith("α")
    assert any(f.kind == "null_result" for f in null_expl.findings)

    # An account explains its causal links.
    t = examples.prevention_transition()
    account = actual.account(t, Direction.BIDIRECTIONAL)
    acc_expl = account.explain()
    assert acc_expl.level == "system"
    assert len(acc_expl.findings) == len(account)
    assert all(f.kind == "link" for f in acc_expl.findings)


def test_explain_is_total(s):
    """Every top-level result type returns a valid Explanation that renders
    and exports without error (the B8 coverage invariant)."""
    import pyphi
    from pyphi import actual
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.direction import Direction
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3
    from pyphi.models.explanation import Explanation

    results = [FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)]
    with pyphi.config.override(**presets.iit3):
        results.append(iit3.sia(s))  # IIT3SystemIrreducibilityAnalysis
        distinction = iit3.concept(s, (1,))  # Distinction
    results.append(distinction)
    results.append(distinction.cause)  # MICE
    results.append(distinction.cause.ria)  # RIA

    transition = examples.prevention_transition()
    account = actual.account(transition, Direction.BIDIRECTIONAL)
    results.append(account)  # Account / DirectedAccount
    results.append(actual.sia(transition, Direction.BIDIRECTIONAL))  # AcSIA
    link = next(iter(account))
    results.append(link)  # CausalLink
    results.append(link.ria)  # AcRIA

    for result in results:
        name = type(result).__name__
        expl = result.explain()
        assert isinstance(expl, Explanation), name
        assert expl.level in {"system", "mechanism"}, name
        assert repr(expl)  # renders without error
        expl.to_pandas()  # exports without error


NOISE = 5.6e-16


class TestBindingDirectionTies:
    def test_finding_reports_tie(self):
        finding = binding_direction_finding(0.3, 0.3 + NOISE)
        assert finding.value == "TIED"

    def test_finding_reports_cause_when_strictly_smaller(self):
        finding = binding_direction_finding(0.2, 0.3)
        assert finding.value == "CAUSE"

    def test_finding_reports_effect_when_strictly_smaller(self):
        finding = binding_direction_finding(0.3, 0.2)
        assert finding.value == "EFFECT"


class TestRunnerUpTieBreak:
    def test_equal_runner_ups_pick_lex_smallest_partition(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class P:
            key: bytes

            def lex_key(self):
                return self.key

        @dataclass(frozen=True)
        class C:
            phi: float
            partition: P

        mip = C(0.1, P(b"\x00"))
        r1 = C(0.5 + NOISE, P(b"\x02"))
        r2 = C(0.5, P(b"\x01"))
        out_a = runner_up_from_candidates([mip, r1, r2], mip.phi)
        out_b = runner_up_from_candidates([mip, r2, r1], mip.phi)
        assert out_a.partition == out_b.partition == P(b"\x01")


class TestRunnerUpRankingKey:
    """The runner-up must be ranked by the same quantity that selects the
    MIP (``sia_tie_resolution``'s primary φ-valued component), not always
    by raw φ."""

    def _candidates(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class P:
            key: bytes

            def lex_key(self):
                return self.key

        @dataclass(frozen=True)
        class C:
            phi: float
            normalized_phi: float
            partition: P

        mip = C(phi=0.3, normalized_phi=0.1, partition=P(b"\x00"))
        # Nearest competitor by raw φ, but farthest by normalized φ.
        raw_nearest = C(phi=0.4, normalized_phi=0.9, partition=P(b"\x01"))
        # Nearest competitor by normalized φ, but farthest by raw φ.
        norm_nearest = C(phi=0.8, normalized_phi=0.2, partition=P(b"\x02"))
        return mip, raw_nearest, norm_nearest

    def test_default_key_ranks_by_raw_phi(self):
        mip, raw_nearest, norm_nearest = self._candidates()
        out = runner_up_from_candidates([mip, raw_nearest, norm_nearest], mip.phi)
        assert out.partition == raw_nearest.partition
        assert out.normalized_phi is None

    def test_normalized_key_ranks_by_normalized_phi(self):
        from pyphi.models.explanation import sia_runner_up_key

        key, normalized = sia_runner_up_key(("NORMALIZED_PHI", "NEGATIVE_PHI"))
        assert normalized
        mip, raw_nearest, norm_nearest = self._candidates()
        out = runner_up_from_candidates(
            [mip, raw_nearest, norm_nearest],
            key(mip),
            key=key,
            normalized=normalized,
        )
        assert out.partition == norm_nearest.partition
        assert out.phi == norm_nearest.phi
        assert out.normalized_phi == norm_nearest.normalized_phi

    def test_non_phi_strategy_falls_back_to_raw_phi(self):
        from pyphi.models.explanation import sia_runner_up_key

        key, normalized = sia_runner_up_key(("PARTITION_LEX",))
        assert not normalized
        mip, raw_nearest, norm_nearest = self._candidates()
        out = runner_up_from_candidates(
            [mip, raw_nearest, norm_nearest],
            key(mip),
            key=key,
            normalized=normalized,
        )
        assert out.partition == raw_nearest.partition
        assert out.normalized_phi is None

    def test_string_strategy_accepted(self):
        from pyphi.models.explanation import sia_runner_up_key

        key, normalized = sia_runner_up_key("PHI")
        assert not normalized
        _mip, raw_nearest, _ = self._candidates()
        assert float(key(raw_nearest)) == raw_nearest.phi
