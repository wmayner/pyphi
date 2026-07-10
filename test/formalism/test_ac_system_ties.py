"""System-level AC tie handling: MIP selection and causal nexus."""

from dataclasses import dataclass

import pyphi
from pyphi import resolve_ties
from pyphi.conf import presets
from pyphi.examples import actual_causation_substrate

NOISE = 5.6e-16


@dataclass(frozen=True)
class FakePartition:
    key: bytes

    def lex_key(self):
        return self.key


@dataclass(frozen=True)
class FakeAcSIA:
    alpha: float
    size: int
    partition: FakePartition
    cause_indices: tuple = (0,)
    effect_indices: tuple = (0,)


class TestResolveAcSiaTie:
    def test_noise_tied_partitions_escalate_to_lex(self):
        a = FakeAcSIA(0.2 + NOISE, 2, FakePartition(b"\x02"))
        b = FakeAcSIA(0.2, 2, FakePartition(b"\x01"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_sia_tie([a, b], context=ctx)
        assert outcome.resolved is b  # lex-smallest partition
        assert set(outcome.tied_set) == {a, b}

    def test_genuine_minimum_wins(self):
        a = FakeAcSIA(0.5, 2, FakePartition(b"\x01"))
        b = FakeAcSIA(0.2, 2, FakePartition(b"\x02"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_sia_tie([a, b], context=ctx)
        assert outcome.resolved is b
        # A genuine minimum resolves at the Integration level, without
        # escalating to the lexicographic tie-break.
        assert outcome.cascade_level == "Integration"
        assert outcome.outcome == "RESOLVED"


class TestResolveAcNexusTie:
    def test_noise_tied_transitions_escalate_to_size(self):
        # Equal alpha (within noise); larger transition wins.
        a = FakeAcSIA(0.3, 1, FakePartition(b"\x01"))
        b = FakeAcSIA(0.3 + NOISE, 2, FakePartition(b"\x02"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_nexus_tie([a, b], context=ctx)
        assert outcome.resolved is b  # larger size
        assert set(outcome.tied_set) == {a, b}

    def test_genuine_maximum_wins(self):
        a = FakeAcSIA(0.5, 1, FakePartition(b"\x01"))
        b = FakeAcSIA(0.2, 2, FakePartition(b"\x02"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_nexus_tie([a, b], context=ctx)
        assert outcome.resolved is a
        # A genuine maximum resolves at the Integration level, without
        # escalating to the size or lexicographic tie-breaks.
        assert outcome.cascade_level == "Integration"
        assert outcome.outcome == "RESOLVED"


class TestAcSiaEndToEnd:
    def test_sia_populates_ties(self):
        # The canonical OR-AND example; ties may or may not exist, but the
        # attribute must be populated and consistent.
        substrate = actual_causation_substrate()
        with pyphi.config.override(**presets.iit3):
            transition = pyphi.actual.Transition(
                substrate, (1, 0), (1, 0), (0, 1), (0, 1)
            )
            sia = pyphi.actual.sia(transition)
        assert isinstance(sia.ties, tuple)
        assert sia in sia.ties or sia.ties == (sia,)

    def test_causal_nexus_deterministic(self):
        substrate = actual_causation_substrate()
        with pyphi.config.override(**presets.iit3):
            a = pyphi.actual.causal_nexus(substrate, (1, 0), (1, 0))
            b = pyphi.actual.causal_nexus(substrate, (1, 0), (1, 0))
        # The same transition (cause/effect index sets) must win both times.
        assert (a.cause_indices, a.effect_indices) == (
            b.cause_indices,
            b.effect_indices,
        )
        assert a.alpha == b.alpha
        assert a == b
