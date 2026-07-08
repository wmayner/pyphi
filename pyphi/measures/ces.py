# metrics/ces.py
"""Functions for computing distances between cause-effect structures."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyphi import utils
from pyphi.conf import config
from pyphi.registry import Registry
from pyphi.types import Repertoire

from . import distribution

if TYPE_CHECKING:
    from pyphi.models.distinction import Distinction as Concept
    from pyphi.models.distinctions import Distinctions
    from pyphi.system import System


class CESMeasureRegistry(Registry):
    """Storage for distance functions between cause-effect structures.

    Users can define custom measures. The third positional or keyword
    argument is the :class:`~pyphi.system.System` context; measures that
    do not need it can accept it as ``system=None`` and ignore it.
    Register a measure, then select it by setting
    ``config.ces_distance = 'ALWAYS_ZERO'``.

    Examples
    --------
    >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
    ... def always_zero(a, b, system=None):
    ...    return 0
    """

    # pylint: disable=arguments-differ

    desc = "distance functions between cause-effect structures"

    def __init__(self) -> None:
        super().__init__()
        self._asymmetric: list[str] = []

    def register(  # type: ignore[override]  # Intentionally extends base signature with asymmetric parameter
        self, name: str, asymmetric: bool = False
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator for registering a CES measure with PyPhi.

        Parameters
        ----------
        name : str
            The name of the measure.
        asymmetric : bool
            ``True`` if the measure is asymmetric.
        """

        def register_func(func: Callable[..., float]) -> Callable[..., float]:
            if asymmetric:
                self._asymmetric.append(name)
            self.store[name] = func  # type: ignore[assignment]  # Registry[T] specialized to float
            return func

        return register_func

    def asymmetric(self) -> list[str]:
        """Return a list of asymmetric measures."""
        return self._asymmetric


measures = CESMeasureRegistry()


def emd_ground_distance(r1: Repertoire, r2: Repertoire) -> float:
    """Compute the distance between two repertoires of a system.

    The measure named by ``config.formalism.iit.mechanism_phi_measure`` is
    used as the ground distance. An asymmetric measure cannot serve this
    role, because the system-level EMD requires a symmetric ground metric.

    Parameters
    ----------
    r1 : np.ndarray
        The first repertoire.
    r2 : np.ndarray
        The second repertoire.

    Returns
    -------
    float
        The distance between ``r1`` and ``r2``.

    Raises
    ------
    ValueError
        If the configured mechanism measure is asymmetric.
    """
    measure_name = config.formalism.iit.mechanism_phi_measure
    asymmetric_metrics = (
        {
            name
            for name, metric in distribution.distribution_measures.items()
            if getattr(metric, "asymmetric", False)
        }
        | {
            name
            for name, metric in distribution.stateful_distribution_measures.items()
            if getattr(metric, "asymmetric", False)
        }
        | {
            name
            for name, metric in distribution.composite_measures.items()
            if getattr(metric, "asymmetric", False)
        }
    )
    if measure_name in asymmetric_metrics:
        raise ValueError(
            f"The repertoire measure {measure_name} is "
            "asymmetric and cannot be used as the ground distance for "
            "the system-level EMD"
        )
    measure = distribution.resolve_distribution_measure(measure_name)
    return distribution.repertoire_distance(
        r1, r2, direction=None, repertoire_distance=measure
    )


def emd_concept_distance(c1: Concept, c2: Concept, system: System) -> float:
    """Return the EMD distance between two concepts in concept space.

    The distance is the sum of the cause-side and effect-side ground
    distances. Each concept's cause and effect repertoires are first
    expanded to the combined purview of the two concepts, so that the two
    EMD signatures have the same size.

    Parameters
    ----------
    c1 : Concept
        The first concept.
    c2 : Concept
        The second concept.
    system : System
        The system whose repertoire algebra expands each concept's cause
        and effect repertoires to the combined purview before the ground
        distance is taken.

    Returns
    -------
    float
        The distance between the two concepts in concept space.
    """
    # Calculate the sum of the cause and effect EMDs, expanding the repertoires
    # to the combined purview of the two concepts, so that the EMD signatures
    # are the same size.
    assert c1.cause is not None
    assert c1.effect is not None
    assert c2.cause is not None
    assert c2.effect is not None
    cause_purview = tuple(set(c1.cause.purview + c2.cause.purview))
    effect_purview = tuple(set(c1.effect.purview + c2.effect.purview))
    return emd_ground_distance(
        system.expand_cause_repertoire(c1.cause.repertoire, new_purview=cause_purview),
        system.expand_cause_repertoire(c2.cause.repertoire, new_purview=cause_purview),
    ) + emd_ground_distance(
        system.expand_effect_repertoire(
            c1.effect.repertoire, new_purview=effect_purview
        ),
        system.expand_effect_repertoire(
            c2.effect.repertoire, new_purview=effect_purview
        ),
    )


def _emd_simple(C1: Sequence[Concept], C2: Sequence[Concept], system: System) -> float:
    """Return the distance between two cause-effect structures.

    Assumes the only difference between them is that some concepts have
    disappeared.
    """
    # Make C1 refer to the bigger CES.
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    null_concept = system.null_concept
    return sum(c.phi * emd_concept_distance(c, null_concept, system) for c in destroyed)


def _emd(
    unique_C1: Sequence[Concept], unique_C2: Sequence[Concept], system: System
) -> float:
    """Return the distance between two cause-effect structures.

    Uses the generalized EMD.
    """
    # Get the pairwise distances between the concepts in the unpartitioned and
    # partitioned CESs.
    distances = np.array(
        [[emd_concept_distance(i, j, system) for j in unique_C2] for i in unique_C1]
    )
    # We need distances from all concepts---in both the unpartitioned and
    # partitioned CESs---to the null concept, because:
    # - often a concept in the unpartitioned CES is destroyed by a
    #   cut (and needs to be moved to the null concept); and
    # - in certain cases, the partitioned system will have *greater* sum of
    #   small-phi, even though it has less big-phi, which means that some
    #   partitioned-CES concepts will be moved to the null concept.
    null_concept = system.null_concept
    distances_to_null = np.array(
        [
            emd_concept_distance(c, null_concept, system)
            for ces in (unique_C1, unique_C2)
            for c in ces
        ]
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now we make the distance matrix, which will look like this:
    #
    #        C1       C2     0
    #    +~~~~~~~~+~~~~~~~~+~~~+
    #    |        |        |   |
    # C1 |   X    |    D   |   |
    #    |        |        |   |
    #    +~~~~~~~~+~~~~~~~~+ D |
    #    |        |        | n |
    # C2 |   D'   |    X   |   |
    #    |        |        |   |
    #    +~~~~~~~~+~~~~~~~~+~~~|
    #  0 |        Dn'      | X |
    #    +~~~~~~~~~~~~~~~~~~~~~+
    #
    # The diagonal blocks marked with an X are set to a value larger than any
    # pairwise distance between concepts. The transport marginals already
    # forbid within-CES moves (each CES's mass sits on its own rows/columns),
    # so the large value is a numerical safeguard rather than the mechanism.
    # The D block is filled with the pairwise distances between the two CESs,
    # and Dn is filled with the distances from each concept to the null
    # concept.
    N, M = len(unique_C1), len(unique_C2)
    # Add one to the side length for the null concept distances.
    distance_matrix: NDArray[np.float64] = np.empty([N + M + 1] * 2)
    # Ensure that concepts are never moved within their own CES.
    distance_matrix[:] = np.max(distances) + 1
    # Set the top-right block to the pairwise CES distances.
    distance_matrix[:N, N:-1] = distances
    # Set the bottom-left block to the same, but transposed.
    distance_matrix[N:-1, :N] = distances.T
    # Do the same for the distances to the null concept.
    distance_matrix[-1, :-1] = distances_to_null
    distance_matrix[:-1, -1] = distances_to_null.T
    distance_matrix[-1, -1] = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct the two phi distributions, with an entry at the end for the
    # null concept.
    d1 = [c.phi for c in unique_C1] + [0] * M + [0]
    d2 = [0] * N + [c.phi for c in unique_C2] + [0]
    # Balance the two phi signatures onto a common total mass by assigning each
    # side's deficit to its own null concept. Both signatures stay non-negative
    # (a valid optimal-transport problem) even when the partitioned constellation
    # carries more phi than the unpartitioned one, and the construction is
    # symmetric in the two constellations.
    sum1, sum2 = sum(d1), sum(d2)
    total = max(sum1, sum2)
    d1[-1] = total - sum1
    d2[-1] = total - sum2
    # The sum of the two signatures should be the same.
    assert utils.eq(sum(d1), sum(d2))
    # Calculate!
    return distribution.EMD.compute(np.array(d1), np.array(d2), distance_matrix)


@measures.register("EMD")
def emd(C1: Distinctions, C2: Distinctions, system: System | None = None) -> float:
    """Return the generalized EMD between two cause-effect structures.

    When the two structures differ only in that some concepts have
    disappeared, the simpler :func:`_emd_simple` is used; otherwise the full
    generalized EMD in :func:`_emd` is taken. The result is rounded to
    ``config.numerics.precision``.

    Parameters
    ----------
    C1 : Distinctions
        The first :class:`~pyphi.models.distinctions.Distinctions`.
    C2 : Distinctions
        The second :class:`~pyphi.models.distinctions.Distinctions`.
    system : System
        The system the CESs were computed over. Required for the EMD
        measure: repertoire expansion and the null concept both come from
        the system.

    Returns
    -------
    float
        The generalized EMD between the two structures.

    Raises
    ------
    ValueError
        If ``system`` is ``None``.
    """
    if system is None:
        raise ValueError(
            "The EMD CES measure requires a ``system`` argument; "
            "pass ``system=`` through ``ces_distance``."
        )
    concepts_only_in_C1 = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    concepts_only_in_C2 = [c2 for c2 in C2 if not any(c2.emd_eq(c1) for c1 in C1)]
    # If the only difference in the CESs is that some concepts
    # disappeared, then we don't need to use the EMD.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        dist = _emd_simple(C1, C2, system)
    else:
        dist = _emd(concepts_only_in_C1, concepts_only_in_C2, system)
    return round(dist, config.numerics.precision)  # type: ignore[arg-type]


@measures.register("SUM_SMALL_PHI")
def sum_small_phi(
    C1: Distinctions,
    C2: Distinctions,
    system: System | None = None,  # noqa: ARG001
) -> float:
    """Return the difference in summed φ between two structures.

    The value is ``sum(C1.phis) - sum(C2.phis)``. This is a signed
    difference of the total small-φ of each
    :class:`~pyphi.models.distinctions.Distinctions`, not a symmetric
    distance, and may be negative. The ``system`` argument is accepted for
    a uniform measure signature and is ignored.
    """
    return sum(C1.phis) - sum(C2.phis)


def ces_distance(
    C1: Distinctions,
    C2: Distinctions,
    measure: str | None = None,
    *,
    system: System | None = None,
) -> float:
    """Return the distance between two cause-effect structures.

    Dispatches to the registered CES measure named by ``measure`` and rounds
    the result to ``config.numerics.precision``.

    Parameters
    ----------
    C1 : Distinctions
        The first :class:`~pyphi.models.distinctions.Distinctions`.
    C2 : Distinctions
        The second :class:`~pyphi.models.distinctions.Distinctions`.
    measure : str, optional
        Which registered CES measure to use. If ``None``, defaults to
        ``config.formalism.iit.ces_measure``.
    system : System, optional
        The system the CESs were computed over. Required by measures that
        operate on full repertoires (e.g. ``EMD``); ignored by purely
        phi-summing measures (e.g. ``SUM_SMALL_PHI``).

    Returns
    -------
    float
        The distance between the two cause-effect structures.
    """
    measure_name: str = config.formalism.iit.ces_measure if measure is None else measure  # type: ignore[assignment]
    dist: float = measures[measure_name](C1, C2, system=system)
    return round(dist, config.numerics.precision)  # type: ignore[arg-type]
