"""System-level AC tie handling: MIP selection and causal nexus."""

from dataclasses import dataclass

import pyphi
from pyphi import numerics
from pyphi import resolve_ties
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.examples import actual_causation_substrate
from pyphi.formalism.actual_causation import compute
from pyphi.models import AcSystemIrreducibilityAnalysis

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


def _or_and_transition():
    substrate = actual_causation_substrate()
    return pyphi.actual.Transition(substrate, (1, 0), (1, 0), (0, 1), (0, 1))


def _sia_with_alphas(monkeypatch, make_alphas):
    """Run the real ``_sia`` with per-partition α values injected.

    ``make_alphas(n_cuts)`` returns the α assigned to each evaluated
    partition, in evaluation order.
    """
    with pyphi.config.override(**presets.iit3):
        transition = _or_and_transition()
        cuts = list(compute._get_partitions(transition, Direction.BIDIRECTIONAL))
        alphas = make_alphas(len(cuts))
        assert len(alphas) == len(cuts)
        it = iter(alphas)

        def fake_evaluate(partition, **kwargs):
            return AcSystemIrreducibilityAnalysis(
                alpha=next(it),
                partition=partition,
                size=len(transition),
                cause_indices=transition.cause_indices,
                effect_indices=transition.effect_indices,
            )

        monkeypatch.setattr(compute, "_evaluate_partition", fake_evaluate)
        return compute._sia(transition), alphas


class TestSiaTiesAreAlphaCluster:
    def test_unique_minimum_yields_singleton_ties(self, monkeypatch):
        sia, _ = _sia_with_alphas(
            monkeypatch, lambda n: [0.2] + [0.3 + 0.1 * i for i in range(n - 1)]
        )
        assert sia.alpha == 0.2
        # No false tie: every non-winning candidate had a distinct α.
        assert sia.ties == (sia,)

    def test_tied_minimum_yields_exact_alpha_cluster(self, monkeypatch):
        sia, alphas = _sia_with_alphas(
            monkeypatch,
            lambda n: [0.2, 0.2 + NOISE] + [0.9 + 0.1 * i for i in range(n - 2)],
        )
        assert len(alphas) > 2  # cluster must be a strict subset of candidates
        assert len(sia.ties) == 2
        assert sorted(t.alpha for t in sia.ties) == [0.2, 0.2 + NOISE]
        assert sia in sia.ties


class _StubAlphaMeasure:
    """DistributionMeasure stub yielding preset α values per partition."""

    name = "stub"
    asymmetric = True

    def __init__(self, alphas):
        self._it = iter(alphas)

    def __call__(self, p, q):  # noqa: ARG002
        return next(self._it)


def _find_mip_with_alphas(make_alphas):
    """Run the real ``_find_mip`` with per-partition α values injected."""
    with pyphi.config.override(**presets.iit3):
        transition = _or_and_transition()
        mechanism = (0, 1)
        purview = (0, 1)
        n = len(
            list(
                compute.mechanism_partitions(mechanism, purview, transition.node_labels)
            )
        )
        alphas = make_alphas(n)
        assert len(alphas) == n
        ria = compute._find_mip(
            transition,
            Direction.CAUSE,
            mechanism,
            purview,
            alpha_measure=_StubAlphaMeasure(alphas),
        )
        return ria, alphas


class TestFindMipPartitionTies:
    def test_unique_minimum_yields_no_partition_ties(self):
        ria, _ = _find_mip_with_alphas(
            lambda n: [0.2] + [0.3 + 0.1 * i for i in range(n - 1)]
        )
        assert ria.alpha == 0.2
        assert ria.partition_ties is None

    def test_tied_minimum_yields_exact_alpha_cluster(self):
        ria, alphas = _find_mip_with_alphas(
            lambda n: [0.2, 0.2] + [0.9 + 0.1 * i for i in range(n - 2)]
        )
        assert len(alphas) > 2  # cluster must be a strict subset of candidates
        assert ria.partition_ties is not None
        assert len(ria.partition_ties) == 2
        assert all(t.alpha == 0.2 for t in ria.partition_ties)
        assert ria in ria.partition_ties


class TestAcSiaEndToEnd:
    def test_sia_populates_ties(self):
        # The canonical OR-AND example; ties may or may not exist, but the
        # attribute must be populated and consistent.
        with pyphi.config.override(**presets.iit3):
            sia = pyphi.actual.sia(_or_and_transition())
        assert isinstance(sia.ties, tuple)
        assert sia in sia.ties or sia.ties == (sia,)
        # Every recorded tie is at the winning α, within tolerance.
        assert all(numerics.eq(t.alpha, sia.alpha) for t in sia.ties)

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
        # Every recorded tie is at the winning α, within tolerance.
        assert all(numerics.eq(t.alpha, a.alpha) for t in a.ties)
