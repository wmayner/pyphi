# metrics/ces.py
"""Functions for computing distances between cause-effect structures."""

import numpy as np

from .. import utils
from ..conf import config
from ..registry import Registry
from . import distribution


class CESMeasureRegistry(Registry):
    """Storage for distance functions between cause-effect structures.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.CES_DISTANCE = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions between cause-effect structures"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering a CES measure with PyPhi.

        Args:
            name (string): The name of the measure.

        Keyword Args:
            asymmetric (boolean): ``True`` if the measure is asymmetric.
        """

        def register_func(func):
            if asymmetric:
                self._asymmetric.append(name)
            self.store[name] = func
            return func

        return register_func

    def asymmetric(self):
        """Return a list of asymmetric measures."""
        return self._asymmetric


measures = CESMeasureRegistry()


def emd_ground_distance(r1, r2):
    """Compute the distance between two repertoires of a system.

    Args:
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.

    Returns:
        float: The distance between ``r1`` and ``r2``.
    """
    if config.REPERTOIRE_DISTANCE in distribution.measures.asymmetric():
        raise ValueError(
            "The repertoire-distance {} is asymmetric and cannot be used as the "
            "ground distance for the system-level EMD".format(
                config.REPERTOIRE_DISTANCE
            )
        )
    return distribution.repertoire_distance(r1, r2, direction=None)


def emd_concept_distance(c1, c2):
    """Return the EMD distance between two concepts in concept space.

    Args:
        c1 (Concept): The first concept.
        c2 (Concept): The second concept.

    Returns:
        float: The distance between the two concepts in concept space.
    """
    # Calculate the sum of the cause and effect EMDs, expanding the repertoires
    # to the combined purview of the two concepts, so that the EMD signatures
    # are the same size.
    cause_purview = tuple(set(c1.cause.purview + c2.cause.purview))
    effect_purview = tuple(set(c1.effect.purview + c2.effect.purview))
    # Take the sum
    return emd_ground_distance(
        c1.expand_cause_repertoire(cause_purview),
        c2.expand_cause_repertoire(cause_purview),
    ) + emd_ground_distance(
        c1.expand_effect_repertoire(effect_purview),
        c2.expand_effect_repertoire(effect_purview),
    )


def _emd_simple(C1, C2):
    """Return the distance between two cause-effect structures.

    Assumes the only difference between them is that some concepts have
    disappeared.
    """
    # Make C1 refer to the bigger CES.
    if len(C2) > len(C1):
        C1, C2 = C2, C1
    destroyed = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    return sum(
        c.phi * emd_concept_distance(c, c.subsystem.null_concept) for c in destroyed
    )


def _emd(unique_C1, unique_C2):
    """Return the distance between two cause-effect structures.

    Uses the generalized EMD.
    """
    # Get the pairwise distances between the concepts in the unpartitioned and
    # partitioned CESs.
    distances = np.array(
        [[emd_concept_distance(i, j) for j in unique_C2] for i in unique_C1]
    )
    # We need distances from all concepts---in both the unpartitioned and
    # partitioned CESs---to the null concept, because:
    # - often a concept in the unpartitioned CES is destroyed by a
    #   cut (and needs to be moved to the null concept); and
    # - in certain cases, the partitioned system will have *greater* sum of
    #   small-phi, even though it has less big-phi, which means that some
    #   partitioned-CES concepts will be moved to the null concept.
    distances_to_null = np.array(
        [
            emd_concept_distance(c, c.subsystem.null_concept)
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
    # pairwise distance between concepts. This ensures that concepts are never
    # moved to another concept within their own CES; they must always go either
    # from one CES to another, or to the null concept N. The D block is filled
    # with the pairwise distances between the two CESs, and Dn is filled with
    # the distances from each concept to the null concept.
    N, M = len(unique_C1), len(unique_C2)
    # Add one to the side length for the null concept distances.
    distance_matrix = np.empty([N + M + 1] * 2)
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
    # Calculate how much phi disappeared and assign it to the null concept.
    d2[-1] = sum(d1) - sum(d2)
    # The sum of the two signatures should be the same.
    assert utils.eq(sum(d1), sum(d2))
    # Calculate!
    return distribution.EMD.compute(np.array(d1), np.array(d2), distance_matrix)


@measures.register("EMD")
def emd(C1, C2):
    """Return the generalized EMD between two cause-effect structures.

    Args:
        C1 (CauseEffectStructure): The first |CauseEffectStructure|.
        C2 (CauseEffectStructure): The second |CauseEffectStructure|.

    Returns:
        float
    """
    concepts_only_in_C1 = [c1 for c1 in C1 if not any(c1.emd_eq(c2) for c2 in C2)]
    concepts_only_in_C2 = [c2 for c2 in C2 if not any(c2.emd_eq(c1) for c1 in C1)]
    # If the only difference in the CESs is that some concepts
    # disappeared, then we don't need to use the EMD.
    if not concepts_only_in_C1 or not concepts_only_in_C2:
        dist = _emd_simple(C1, C2)
    else:
        dist = distribution.EMD.compute(concepts_only_in_C1, concepts_only_in_C2)
    return round(dist, config.PRECISION)


@measures.register("SUM_SMALL_PHI")
def sum_small_phi(C1, C2):
    """Return the difference in |small_phi| between |CauseEffectStructure|."""
    return sum(C1.phis) - sum(C2.phis)


def ces_distance(C1, C2, measure=None):
    """Return the distance between two cause-effect structures.

    Args:
        C1 (CauseEffectStructure): The first |CauseEffectStructure|.
        C2 (CauseEffectStructure): The second |CauseEffectStructure|.

    Returns:
        float: The distance between the two cause-effect structures.
    """
    measure = config.CES_DISTANCE if measure is None else measure
    dist = measures[measure](C1, C2)
    return round(dist, config.PRECISION)
