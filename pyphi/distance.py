#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distance.py

"""
Functions for measuring distances.
"""

from contextlib import ContextDecorator

import numpy as np
from pyemd import emd
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from . import Direction, config, constants, utils, validate
from .distribution import flatten, marginal_zero
from .registry import Registry

# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = utils.load_data('hamming_matrices',
                                    _NUM_PRECOMPUTED_HAMMING_MATRICES)


class MeasureRegistry(Registry):
    """Storage for measures registered with PyPhi.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting ``config.MEASURE = 'ALWAYS_ZERO'``.
    """
    # pylint: disable=arguments-differ

    desc = 'measures'

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering a measure with PyPhi.

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


measures = MeasureRegistry()


class np_suppress(np.errstate, ContextDecorator):
    """Decorator to suppress NumPy warnings about divide-by-zero and
    multiplication of ``NaN``.

    .. note::
        This should only be used in cases where you are *sure* that these
        warnings are not indicative of deeper issues in your code.
    """
    def __init__(self):
        super().__init__(divide='ignore', invalid='ignore')


# TODO extend to nonbinary nodes
def _hamming_matrix(N):
    """Return a matrix of Hamming distances for the possible states of |N|
    binary nodes.

    Args:
        N (int): The number of nodes under consideration

    Returns:
        np.ndarray: A |2^N x 2^N| matrix where the |ith| element is the Hamming
        distance between state |i| and state |j|.

    Example:
        >>> _hamming_matrix(2)
        array([[0., 1., 1., 2.],
               [1., 0., 2., 1.],
               [1., 2., 0., 1.],
               [2., 1., 1., 0.]])
    """
    if N < _NUM_PRECOMPUTED_HAMMING_MATRICES:
        return _hamming_matrices[N]
    return _compute_hamming_matrix(N)


@constants.joblib_memory.cache
def _compute_hamming_matrix(N):
    """Compute and store a Hamming matrix for |N| nodes.

    Hamming matrices have the following sizes::

        N   MBs
        ==  ===
        9   2
        10  8
        11  32
        12  128
        13  512

    Given these sizes and the fact that large matrices are needed infrequently,
    we store computed matrices using the Joblib filesystem cache instead of
    adding computed matrices to the ``_hamming_matrices`` global and clogging
    up memory.

    This function is only called when |N| >
    ``_NUM_PRECOMPUTED_HAMMING_MATRICES``. Don't call this function directly;
    use |_hamming_matrix| instead.
    """
    possible_states = np.array(list(utils.all_states((N))))
    return cdist(possible_states, possible_states, 'hamming') * N


# TODO extend to binary nodes
@measures.register('EMD')
def hamming_emd(d1, d2):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node) using the Hamming distance between states
    as the transportation cost function.

    Singleton dimensions are sqeezed out.
    """
    N = d1.squeeze().ndim
    d1, d2 = flatten(d1), flatten(d2)
    return emd(d1, d2, _hamming_matrix(N))


def effect_emd(d1, d2):
    """Compute the EMD between two effect repertoires.

    Because the nodes are independent, the EMD between effect repertoires is
    equal to the sum of the EMDs between the marginal distributions of each
    node, and the EMD between marginal distribution for a node is the absolute
    difference in the probabilities that the node is OFF.

    Args:
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``.
    """
    return sum(abs(marginal_zero(d1, i) - marginal_zero(d2, i))
               for i in range(d1.ndim))


@measures.register('L1')
def l1(d1, d2):
    """Return the L1 distance between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The sum of absolute differences of ``d1`` and ``d2``.
    """
    return np.absolute(d1 - d2).sum()


@measures.register('KLD', asymmetric=True)
def kld(d1, d2):
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.

    Returns:
        float: The KLD of ``d1`` from ``d2``.
    """
    d1, d2 = flatten(d1), flatten(d2)
    return entropy(d1, d2, 2.0)


@measures.register('ENTROPY_DIFFERENCE')
def entropy_difference(d1, d2):
    """Return the difference in entropy between two distributions."""
    d1, d2 = flatten(d1), flatten(d2)
    return abs(entropy(d1, base=2.0) - entropy(d2, base=2.0))


@measures.register('PSQ2')
@np_suppress()
def psq2(d1, d2):
    """Compute the PSQ2 measure.

    Args:
        d1 (np.ndarray): The first distribution.
        d2 (np.ndarray): The second distribution.
    """
    d1, d2 = flatten(d1), flatten(d2)

    def f(p):
        return sum((p ** 2) * np.nan_to_num(np.log2(p * len(p))))

    return abs(f(d1) - f(d2))


@measures.register('MP2Q', asymmetric=True)
@np_suppress()
def mp2q(p, q):
    """Compute the MP2Q measure.

    Args:
        p (np.ndarray): The unpartitioned repertoire
        q (np.ndarray): The partitioned repertoire
    """
    p, q = flatten(p), flatten(q)
    entropy_dist = 1 / len(p)
    return sum(entropy_dist * np.nan_to_num((p ** 2) / q * np.log2(p / q)))


@measures.register('KLM', asymmetric=True)
@measures.register('BLD', asymmetric=True)  # Backwards-compatible alias
@np_suppress()
def klm(p, q):
    """Compute the KLM divergence."""
    p, q = flatten(p), flatten(q)
    return max(abs(p * np.nan_to_num(np.log2(p / q))))


def directional_emd(direction, d1, d2):
    """Compute the EMD between two repertoires for a given direction.

    The full EMD computation is used for cause repertoires. A fast analytic
    solution is used for effect repertoires.

    Args:
        direction (Direction): |CAUSE| or |EFFECT|.
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``, rounded to |PRECISION|.

    Raises:
        ValueError: If ``direction`` is invalid.
    """
    if direction == Direction.CAUSE:
        func = hamming_emd
    elif direction == Direction.EFFECT:
        func = effect_emd
    else:
        # TODO: test that ValueError is raised
        validate.direction(direction)

    return round(func(d1, d2), config.PRECISION)


def repertoire_distance(direction, r1, r2):
    """Compute the distance between two repertoires for the given direction.

    Args:
        direction (Direction): |CAUSE| or |EFFECT|.
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.

    Returns:
        float: The distance between ``d1`` and ``d2``, rounded to |PRECISION|.
    """
    if config.MEASURE == 'EMD':
        dist = directional_emd(direction, r1, r2)
    else:
        dist = measures[config.MEASURE](r1, r2)

    return round(dist, config.PRECISION)


def system_repertoire_distance(r1, r2):
    """Compute the distance between two repertoires of a system.

    Args:
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.

    Returns:
        float: The distance between ``r1`` and ``r2``.
    """
    if config.MEASURE in measures.asymmetric():
        raise ValueError(
            '{} is asymmetric and cannot be used as a system-level '
            'irreducibility measure.'.format(config.MEASURE))

    return measures[config.MEASURE](r1, r2)
