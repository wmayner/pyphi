"""Confirmation experiments for the exact float comparisons in the measures.

Two comparisons in :mod:`pyphi.measures.distribution` are written exactly rather
than through the tolerant :mod:`pyphi.numerics` layer. Each is safe only if the
values that reach it never land in the narrow floating-point band where an exact
operator disagrees with the intended mathematical one. This module drives the
real plumbing over the standard examples and pins that finding for both.

1. ``pointwise_mutual_information`` guards its ``log2(p / q)`` with the exact
   comparison ``p == 0.0 or q == 0.0`` (and
   ``weighted_pointwise_mutual_information`` inherits that guard by delegating to
   it). The guard is safe only if the actual causation probability plumbing
   never delivers a mathematically-zero probability as a small nonzero
   floating-point residue: a value such as ``1e-16`` would slip past ``== 0.0``
   and produce a spurious large ``log2``. See
   :func:`test_no_subprecision_probability_residues`.

2. ``approximate_specified_state`` picks each non-fixed node's state with the
   exact comparison ``discriminant < tmp_inform``. A sub-precision gap
   ``0 < |discriminant - tmp_inform| < 10⁻¹³`` would make the greedy choice
   sensitive to floating-point noise. See
   :func:`test_approximate_specified_state_discriminant_gap`.
"""

import functools
import itertools

import numpy as np

import pyphi
from pyphi import Direction
from pyphi import actual
from pyphi import examples
from pyphi import utils
from pyphi.examples import actual_causation_substrate
from pyphi.measures.distribution import approximate_specified_state
from pyphi.partition import mechanism_partitions

TOLERANCE = 10 ** (-13)


def _iter_probabilities():
    """Yield every transition probability reachable through the AC plumbing.

    Covers the canonical two-node AC substrate (an ``OR`` and an ``AND`` gate
    with self-loops), across all realizable before/after state pairs, both
    temporal directions, every nonempty mechanism/purview pair, and — for each
    valid pair — every mechanism partition. Both the unpartitioned
    (:meth:`~pyphi.actual.Transition.probability`) and partitioned
    (:meth:`~pyphi.actual.Transition.partitioned_probability`) values are
    yielded.

    State pairs violating the Realization requirement cannot be constructed as
    :class:`~pyphi.actual.Transition` objects, and combinations whose purview
    state is unreachable in the requested direction raise
    :class:`~pyphi.exceptions.StateUnreachableError` inside the plumbing; both
    are skipped, since for those no value reaches the PMI guard at all.
    """
    substrate = actual_causation_substrate()
    n = substrate.size
    states = list(itertools.product((0, 1), repeat=n))
    for before, after in itertools.product(states, states):
        try:
            transition = actual.Transition(
                substrate, before, after, tuple(range(n)), tuple(range(n))
            )
        except pyphi.exceptions.StateUnreachableError:
            # Transitions violating the Realization requirement cannot be
            # constructed, so no probability from them can reach the guard.
            continue
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

    Result over the canonical AC substrate: 198 probability values sampled
    (54 unpartitioned, 144 partitioned); none are exactly ``0.0`` — zero
    probabilities arose only from occurrences within unrealizable transitions,
    which cannot be constructed — and the smallest value is ``1/9``, far above
    the tolerance. Should this ever fail, both PMI guards must switch to
    ``numerics.is_zero(p) or numerics.is_zero(q)``.
    """
    values = np.array(list(_iter_probabilities()))
    assert values.size > 0
    nonzero = values[values != 0.0]
    assert nonzero.size == 0 or nonzero.min() > TOLERANCE


# ---------------------------------------------------------------------------
# approximate_specified_state discriminant comparison
# ---------------------------------------------------------------------------

EXAMPLE_SYSTEMS = (
    examples.basic_system,
    examples.xor_system,
    examples.residue_system,
    examples.grid3_system,
    examples.rule110_system,
    examples.rule154_system,
)

# Cap on mechanism partitions sampled per (mechanism, purview) pair. Bounds the
# example-substrate sweep to a few thousand repertoire pairs while still drawing
# genuine partitioned repertoires from every system and direction.
MAX_PARTITIONS_PER_PURVIEW = 8


def _discriminant_gaps(repertoire, partitioned_repertoire):
    """Yield ``|discriminant - tmp_inform|`` for each non-fixed node step.

    This mirrors the greedy non-fixed-node loop of
    :func:`~pyphi.measures.distribution.approximate_specified_state`
    (``pyphi/measures/distribution.py`` lines 1018-1091, this revision): the
    ``joint_to_marginals`` reduction, the fixed-node selection, and the
    ``discriminant``/``tmp_inform`` recurrence are reproduced verbatim so the
    sequence of comparisons seen here matches the production sequence exactly.

    The mirroring is the honest cost of instrumenting a single ``<`` deep inside
    the function without a return-value hook: the production code is not
    refactored, so this copy must be kept in step with it. Inputs that the
    production function itself rejects (a purview that squeezes to a scalar, or a
    zero-probability marginal that makes an informativeness ratio undefined)
    raise here exactly as they do there; the caller skips them.

    Yields
    ------
    float
        The absolute gap ``|discriminant - tmp_inform|`` at one non-fixed node
        step, in the order the production loop evaluates them.
    """

    def joint_to_marginals(rep):
        rep = np.asarray(rep).squeeze()
        node_indices = set(range(rep.ndim))
        complements = [node_indices - {n} for n in tuple(node_indices)]
        return np.vstack([rep.sum(tuple(c)) for c in complements])

    with np.errstate(divide="ignore", invalid="ignore"):
        P = joint_to_marginals(repertoire)
        Q = joint_to_marginals(partitioned_repertoire)
        purview_size = P.shape[0]
        specified_P = np.ones((purview_size, 1))
        specified_Q = np.ones((purview_size, 1))

        is_selective = P >= (1 / 2)
        informativeness = P / Q
        is_informative = informativeness >= 1
        fixed_nodes = np.where(np.sum(is_selective * is_informative, axis=1))[0]

        def informative_state(node):
            return np.where(informativeness[node, :] == informativeness[node, :].max())[
                0
            ]

        for fixed_node in fixed_nodes:
            state = np.where(
                is_selective[fixed_node, :] * is_informative[fixed_node, :]
            )[0][0]
            specified_P[fixed_node] = P[fixed_node, state]
            specified_Q[fixed_node] = Q[fixed_node, informative_state(fixed_node)[0]]

        if fixed_nodes.size == purview_size:
            return

        nonfixed_nodes = np.setdiff1d(np.arange(purview_size), fixed_nodes)
        p = np.array([P[n, informative_state(n)] for n in nonfixed_nodes]).flatten()
        q = np.array([Q[n, informative_state(n)] for n in nonfixed_nodes]).flatten()
        discriminants = (p * np.log2(p / q) - (1 - p) * np.log2((1 - p) / (1 - q))) / (
            1 - 2 * p
        )
        discriminant_indices = np.argsort(discriminants)
        discriminants = np.sort(discriminants)

        for index, discriminant in zip(
            discriminant_indices, discriminants, strict=False
        ):
            tmp_inform = np.log2(specified_P.prod()) - np.log2(specified_Q.prod())
            yield abs(float(discriminant) - float(tmp_inform))

            nonfixed_node = nonfixed_nodes[index]
            if discriminant < tmp_inform:
                state = int(not informative_state(nonfixed_node)[0])
            else:
                state = informative_state(nonfixed_node)[0]
            specified_P[nonfixed_node] = P[nonfixed_node, state]
            specified_Q[nonfixed_node] = Q[nonfixed_node, state]


def _collect_gaps(repertoire, partitioned_repertoire):
    """Return the finite discriminant gaps for one repertoire pair, or ``[]``.

    Inputs the production function rejects raise inside
    :func:`_discriminant_gaps` and are reported as no evaluated steps.
    """
    try:
        gaps = list(_discriminant_gaps(repertoire, partitioned_repertoire))
    except (IndexError, ValueError):
        return []
    return [g for g in gaps if np.isfinite(g)]


def _example_substrate_gaps():
    """Collect discriminant gaps over the standard example substrates.

    For every example system, temporal direction, mechanism, and purview of at
    least two nodes, the unpartitioned repertoire is paired with up to
    ``MAX_PARTITIONS_PER_PURVIEW`` mechanism partitions' repertoires (each the
    product of the per-part repertoires, exactly the distribution a non-composite
    mechanism measure forms). Every resulting pair that
    :func:`~pyphi.measures.distribution.approximate_specified_state` accepts
    contributes its non-fixed node steps.
    """
    gaps = []
    for make_system in EXAMPLE_SYSTEMS:
        system = make_system()
        n = system.size

        @functools.cache
        def repertoire(direction, mechanism, purview, _system=system):
            return np.asarray(_system.repertoire(direction, mechanism, purview))

        for mechanism in utils.powerset(range(n), nonempty=True):
            for purview in utils.powerset(range(n), nonempty=True):
                if len(purview) < 2:
                    continue
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    try:
                        unpartitioned = repertoire(direction, mechanism, purview)
                    except Exception:
                        continue
                    partitions = itertools.islice(
                        mechanism_partitions(mechanism, purview),
                        MAX_PARTITIONS_PER_PURVIEW,
                    )
                    for partition in partitions:
                        try:
                            parts = [
                                repertoire(direction, part.mechanism, part.purview)
                                for part in partition
                            ]
                        except Exception:
                            continue
                        partitioned = parts[0]
                        for extra in parts[1:]:
                            partitioned = partitioned * extra
                        gaps.extend(_collect_gaps(unpartitioned, partitioned))
    return gaps


def _random_repertoire(rng, n_nodes):
    """Return a normalized random binary-node joint repertoire of ``n_nodes``."""
    repertoire = rng.random((2,) * n_nodes)
    return repertoire / repertoire.sum()


def _random_gaps(seed=20260710, n_pairs=200):
    """Collect discriminant gaps over a seeded sweep of random repertoire pairs.

    Each pair draws an independent size (2-4 nodes) and two normalized random
    joint repertoires from an isolated generator seeded with ``seed``.
    """
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_pairs):
        n_nodes = int(rng.integers(2, 5))
        repertoire = _random_repertoire(rng, n_nodes)
        partitioned = _random_repertoire(rng, n_nodes)
        gaps.extend(_collect_gaps(repertoire, partitioned))
    return gaps


def test_approximate_specified_state_discriminant_gap():
    """The ``discriminant < tmp_inform`` comparison never sits sub-precision.

    Confirms the exact comparison at
    ``pyphi/measures/distribution.py`` line 1085 is safe: no non-fixed node step
    produces a gap ``|discriminant - tmp_inform|`` in the open band
    ``(0, 1e-13]`` where floating-point noise could flip the greedy state choice.

    :func:`~pyphi.measures.distribution.approximate_specified_state` is a
    documented linear-time approximation, so a sub-precision tie would be an
    arbitrary resolution rather than a correctness bug. Were one found, the
    remedy is to pin the witnessing repertoire pair here (not to change the
    production comparison). None is found.

    Result over the two sources: the example-substrate sweep contributes about
    855 non-fixed node steps and the seeded random sweep another ~145; every gap
    is strictly positive and the smallest is about ``3.1e-3``, more than ten
    orders of magnitude above the tolerance. (An uncapped example-substrate sweep
    over all mechanism partitions reaches ~1.6 × 10⁵ steps with the same
    conclusion and a smallest gap near ``2.9e-4``.) The comparison is mirrored
    from source (see :func:`_discriminant_gaps`), so this test must be updated if
    that loop changes.
    """
    # Ground the mirror against the production function: it accepts a
    # representative example pair and returns one state per purview node.
    system = examples.basic_system()
    purview = (0, 1, 2)
    real_state = approximate_specified_state(
        system.repertoire(Direction.CAUSE, (0, 1, 2), purview),
        system.unconstrained_repertoire(Direction.CAUSE, purview),
    )
    assert real_state.shape == (1, len(purview))

    gaps = np.array(_example_substrate_gaps() + _random_gaps())
    assert gaps.size > 0
    # No gap in the open band (0, 1e-13]; exactly-zero gaps are honest ties.
    subprecision = gaps[(gaps > 0.0) & (gaps <= TOLERANCE)]
    assert subprecision.size == 0, (
        f"sub-precision discriminant gaps found (min {gaps[gaps > 0].min():.3e}); "
        "pin the witnessing repertoire pair rather than changing production code"
    )
