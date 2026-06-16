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
