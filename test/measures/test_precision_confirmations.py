"""Confirmation experiment for the exact zero-guard in the PMI measures.

``pointwise_mutual_information`` guards its ``log2(p / q)`` with the exact
comparison ``p == 0.0 or q == 0.0`` (and ``weighted_pointwise_mutual_information``
inherits that guard by delegating to it). The guard is safe only if the actual
causation probability plumbing never delivers a mathematically-zero probability
as a small nonzero floating-point residue: a value such as ``1e-16`` would slip
past ``== 0.0`` and produce a spurious large ``log2``.

This module drives the real plumbing exhaustively over the canonical AC example
and pins the finding that every probability it returns is either exactly ``0.0``
or bounded well away from zero, so the exact guard cannot misfire.
"""

import itertools

import numpy as np

import pyphi
from pyphi import actual
from pyphi import utils
from pyphi.examples import actual_causation_substrate
from pyphi.partition import mechanism_partitions

TOLERANCE = 10 ** (-13)


def _iter_probabilities():
    """Yield every transition probability reachable through the AC plumbing.

    Covers the canonical two-node AC substrate (an ``OR`` and an ``AND`` gate
    with self-loops), across all sixteen before/after state pairs, both temporal
    directions, every nonempty mechanism/purview pair, and — for each valid pair
    — every mechanism partition. Both the unpartitioned
    (:meth:`~pyphi.actual.Transition.probability`) and partitioned
    (:meth:`~pyphi.actual.Transition.partitioned_probability`) values are
    yielded.

    Combinations whose purview state is unreachable in the requested direction
    raise :class:`~pyphi.exceptions.StateUnreachableError` inside the plumbing
    and are skipped: for those the plumbing raises rather than returning a
    probability, so no value reaches the PMI guard at all.
    """
    substrate = actual_causation_substrate()
    n = substrate.size
    states = list(itertools.product((0, 1), repeat=n))
    for before, after in itertools.product(states, states):
        transition = actual.Transition(
            substrate, before, after, tuple(range(n)), tuple(range(n))
        )
        for direction in (pyphi.Direction.CAUSE, pyphi.Direction.EFFECT):
            for mechanism in utils.powerset(range(n), nonempty=True):
                for purview in utils.powerset(range(n), nonempty=True):
                    try:
                        yield float(
                            transition.probability(direction, mechanism, purview)
                        )
                    except pyphi.exceptions.StateUnreachableError:
                        continue
                    for partition in mechanism_partitions(
                        mechanism, purview, transition.node_labels
                    ):
                        try:
                            yield float(
                                transition.partitioned_probability(direction, partition)
                            )
                        except pyphi.exceptions.StateUnreachableError:
                            continue


def test_no_subprecision_probability_residues():
    """Every AC probability is either exactly ``0.0`` or above the tolerance.

    Confirms the exact guard ``p == 0.0`` in
    :func:`~pyphi.measures.distribution.pointwise_mutual_information` is safe:
    no returned probability lands in the open interval ``(0, 1e-13)`` where the
    exact guard would misfire.

    Result over the canonical AC substrate: 792 probability values sampled
    (216 unpartitioned, 576 partitioned); 160 are exactly ``0.0`` and the
    smallest nonzero value is ``0.0625`` (:math:`2^{-4}`), far above the
    tolerance. Should this ever fail, both PMI guards must switch to
    ``numerics.is_zero(p) or numerics.is_zero(q)``.
    """
    values = np.array(list(_iter_probabilities()))
    assert values.size > 0
    nonzero = values[values != 0.0]
    assert nonzero.size == 0 or nonzero.min() > TOLERANCE
