#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# metrics/distribution.py

"""Metrics on probability distributions."""

from contextlib import ContextDecorator
from itertools import product
from math import log2

import numpy as np
from pyemd import emd as _emd
from scipy.spatial.distance import cdist
from scipy.special import entr, rel_entr

from .. import config, utils, validate
from ..cache import joblib_memory
from ..conf import fallback
from ..direction import Direction
from ..distribution import flatten, marginal_zero
from ..registry import Registry

_LN_OF_2 = np.log(2)


class DistributionMeasureRegistry(Registry):
    """Storage for distance functions between probability distributions.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions between probability distributions"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering a distribution measure with PyPhi.

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


measures = DistributionMeasureRegistry()


class ActualCausationMeasureRegistry(Registry):
    """Storage for distance functions used in :mod:`pyphi.actual`.

    Users can define custom measures:

    Examples:
        >>> @measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.REPERTOIRE_DISTANCE = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions for use in actual causation calculations"

    def __init__(self):
        super().__init__()
        self._asymmetric = []

    def register(self, name, asymmetric=False):
        """Decorator for registering an actual causation measure with PyPhi.

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


actual_causation_measures = ActualCausationMeasureRegistry()


class np_suppress(np.errstate, ContextDecorator):
    """Decorator to suppress NumPy warnings about divide-by-zero and
    multiplication of ``NaN``.

    .. note::
        This should only be used in cases where you are *sure* that these
        warnings are not indicative of deeper issues in your code.
    """

    def __init__(self):
        super().__init__(divide="ignore", invalid="ignore")


# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = utils.load_data(
    "hamming_matrices", _NUM_PRECOMPUTED_HAMMING_MATRICES
)


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


@joblib_memory.cache
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
    we store computed matrices using the joblib filesystem cache instead of
    adding computed matrices to the ``_hamming_matrices`` global and clogging
    up memory.

    This function is only called when |N| >
    ``_NUM_PRECOMPUTED_HAMMING_MATRICES``. Don't call this function directly;
    use |_hamming_matrix| instead.
    """
    possible_states = np.array(list(utils.all_states((N))))
    return cdist(possible_states, possible_states, "hamming") * N


# TODO extend to nonbinary nodes
def hamming_emd(p, q):
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node) using the Hamming distance between states
    as the transportation cost function.

    Singleton dimensions are sqeezed out.
    """
    N = p.squeeze().ndim
    p, q = flatten(p), flatten(q)
    return _emd(p, q, _hamming_matrix(N))


def effect_emd(p, q):
    """Compute the EMD between two effect repertoires.

    Because the nodes are independent, the EMD between effect repertoires is
    equal to the sum of the EMDs between the marginal distributions of each
    node, and the EMD between marginal distribution for a node is the absolute
    difference in the probabilities that the node is OFF.

    Args:
        p (np.ndarray): The first repertoire.
        q (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``p`` and ``q``.
    """
    return sum(abs(marginal_zero(p, i) - marginal_zero(q, i)) for i in range(p.ndim))


@measures.register("EMD")
def emd(p, q, direction=None):
    """Compute the EMD between two repertoires for a given direction.

    The full EMD computation is used for cause repertoires. A fast analytic
    solution is used for effect repertoires.

    Args:
        p (np.ndarray): The first repertoire.
        q (np.ndarray): The second repertoire.
        direction (Direction | None): |CAUSE| or |EFFECT|. If |EFFECT|, then the
            special-case ``effect_emd`` is used (optimized for this case). Otherwise
            the ``hamming_emd`` is used. Defaults to |CAUSE|.

    Returns:
        float: The EMD between ``p`` and ``q``, rounded to |PRECISION|.

    Raises:
        ValueError: If ``direction`` is invalid.
    """
    if (direction == Direction.CAUSE) or (direction is None):
        func = hamming_emd
    elif direction == Direction.EFFECT:
        func = effect_emd
    else:
        # TODO: test that ValueError is raised
        validate.direction(direction)

    return round(func(p, q), config.PRECISION)


@measures.register("L1")
def l1(p, q):
    """Return the L1 distance between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The sum of absolute differences of ``p`` and ``q``.
    """
    return np.abs(p - q).sum()


@measures.register("ENTROPY_DIFFERENCE")
def entropy_difference(p, q):
    """Return the difference in entropy between two distributions."""
    hp = entr(p).sum() / _LN_OF_2
    hq = entr(q).sum() / _LN_OF_2
    return abs(hp - hq)


@measures.register("PSQ2")
def psq2(p, q):
    r"""Compute the PSQ2 measure.

    This is defined as :math:`\mid f(p) - f(q) \mid`, where

    .. math::
        f(x) = \sum_{i=0}^{N-1} p_i^2 \log_2 (p_i N)

    Args:
        p (np.ndarray): The first distribution.
        q (np.ndarray): The second distribution.
    """
    fp = (p * (-1.0 * entr(p))).sum() / _LN_OF_2 + (p**2 * log2(len(p))).sum()
    fq = (q * (-1.0 * entr(q))).sum() / _LN_OF_2 + (q**2 * log2(len(q))).sum()
    return abs(fp - fq)


@measures.register("MP2Q", asymmetric=True)
@np_suppress()
def mp2q(p, q):
    r"""Compute the MP2Q measure.

    This is defined as

    .. math::
        \frac{1}{N}
        \sum_{i=0}^{N-1} \frac{p_i^2}{q_i} \log_2\left(\frac{p_i}{q_i}\right)

    Args:
        p (np.ndarray): The first distribution.
        q (np.ndarray): The second distribution.

    Returns:
        float: The distance.
    """
    # There is already a factor of p in the `information_density`, so we only
    # multiply by p, not p**2
    return np.sum(p / q * information_density(p, q) / len(p))


def information_density(p, q):
    """Return the information density of p relative to q, in base 2.

    This is also known as the element-wise relative entropy; see
    :func:`scipy.special.rel_entr`.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        np.ndarray: The information density of ``p`` relative to ``q``.
    """
    return rel_entr(p, q) / _LN_OF_2


@measures.register("KLD", asymmetric=True)
def kld(p, q):
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The KLD of ``p`` from ``q``.
    """
    return information_density(p, q).sum()


def absolute_information_density(p, q):
    """Return the absolute information density function of two distributions.

    The information density is also known as the element-wise relative
    entropy; see :func:`scipy.special.rel_entr`.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        np.ndarray: The absolute information density of ``p`` relative to ``q``.
    """
    return np.abs(information_density(p, q))


def specified_index(repertoire, partitioned_repertoire):
    """Return the indices of the state(s) with the maximal AID between the repertoires.

    The index is relative to the entire network (i.e., suitable for indexing
    into a repertoire).

    Note that there can be ties.

    Returns:
        np.ndarray: A 2D array where each row is a maximal state.
    """
    # TODO(4.0) this is unnecessarily recomputed; should make a
    # DistanceResult class that can carry auxilliary data, e.g. the maximal
    # states
    # TODO(4.0) make configurable
    density = absolute_information_density(repertoire, partitioned_repertoire)
    return (density == density.max()).nonzero()


def specified_state(repertoire, partitioned_repertoire):
    """Return the state(s) with the maximal AID between the repertoires.

    This returns only the state of the purview nodes (i.e., there is one element
    in the state vector for each purview node, not for each node in the
    network).

    Note that there can be ties.

    Returns:
        np.ndarray: A 2D array where each row is a maximal state.
    """
    # TODO(relations)
    density = absolute_information_density(
        repertoire.squeeze(), partitioned_repertoire.squeeze()
    )
    return np.transpose(np.where(density == density.max()))


def approximate_specified_state(p, q, abs_condition="without_abs"):
    """
    Estimate the purview state that maximizes intrinsic information.

    Args
        p (np.ndarray): The unpartitioned repertoire.
        q (np.ndarray): The partitioned repertoire.
        abs_condition (str): 'with_abs' or 'without_abs'.

    Returns:
        np.ndarray: A 2D array where each row is a maximal state.
    """

    def joint_to_marginals(repertoire):
        """Converts a joint repertoire in multidimensional form to a 2D array of
        single-node marginal repertoires.

        Args:
            repertoire (np.ndarray): The joint repertoire of a purview in
                multidimensional form, e.g., as obtained from
                :mod:`pyphi.subsystem`. Note that `repertoire` is assumed to be
                a well-formed probability distribution whose sum over all states
                equals one.
        Returns:
            np.ndarray: A 2D array with one row per node in the purview (the
                marginalized repertoires) and one column per state, in the same
                order as the argument.
        """
        # Remove singleton dimensions.
        repertoire = repertoire.squeeze()
        # Map each dimension in the squeezed repertoire to a local node index.
        node_indices = set(range(repertoire.ndim))
        # All the sets of indices of size n - 1 (i.e. combinations(n, n - 1)).
        complements = [node_indices - set((n,)) for n in tuple(node_indices)]
        # Marginalize out all the complementary dimensions for each
        # node in the repertoire.
        marginals = [repertoire.sum(tuple(c)) for c in complements]
        return np.vstack(marginals)

    p = joint_to_marginals(p)
    q = joint_to_marginals(q)

    log2 = np.log2
    max = np.max
    argmax = np.argmax
    abs = np.abs

    MAX_SEARCH_NODE = 0
    # If the number of undetermined nodes (i.e. not having p>0.5 && p>q and
    # listed in "non_fixed") is
    # - less than this number, then we compute all the possible states of
    #   "non_fixed" nodes.
    # - more than this number, considering all possible states of "non_fixed"
    #   nodes gets harder, so determine the state based on the following
    #   inequality:

    # Consider there are k-nodes already determined (the unpartitioned &
    # partitioned probabilities are p_k and q_k) and adding a node "z".
    # Let the node "z" have probablity p_z for the state "x". For the
    # complementary state "y", the probability is (1 - p_z). We can assume
    # p_z > q_z without losing generality. We want to know which state of the
    # node "z" gives higher intrinsic information when it is added to the
    # k-nodes. In other words, we want to compare I_x and I_y
    #
    #       I_x = (p_k * p_z) * log( p_k*p_z / q_k*q_z )
    #       I_y = (p_k * (1-p_z)) * log( p_k*(1-p_z) / q_k*(1-q_z) )
    #
    # If state "y" gives higher intrinsic information, i.e., I_y > I_x,
    # p_z and q_z need to satisfy the following 2 equations:
    #       (1) p_z < 1/2
    #       (2) log(p_k/q_k) < (1/(1-2*p_z)) * { p_z*log(p_z/q_z) - (1-p_z)*log((1-p_z)/(1-q_z)) }
    # Otherwise, we should pick the state "x" as the state of node "z".  Note
    # that the result of "z" depends on the already chosen k-nodes and there is
    # no guarantee to get the optimal state.


    # the size of the candidate purview
    n = p.shape[0]
    # If r > 1, the probability of the state goes up because of the
    # connection from the mechanism
    r = p / q

    # Get the node indices whose state we can determine without searching.
    # --- If the state gives the highest probability && the state is the one the
    #     mechanism specifies (i.e., p > q),
    # --- then the state will be chosen as the optimal state for intrinsic
    #     information no matter what nodes are selected.
    p_state = 1 * (p >= 1/2)
    r_state = 1 * (r >= 1)

    fixed_node_lid = np.where(np.sum(p_state * r_state, axis=1) > 0)[0]

    optimal_state = np.zeros((n,1)) * np.nan
    optimal_p = np.ones((n,1))
    optimal_q = np.ones((n,1))

    for tmp_lid in fixed_node_lid:
        opt_state_col = np.where(p_state[tmp_lid, :] * r_state[tmp_lid, :])[0]

        # If "opt_state_col" has 2 elements, that means (p_on=p_off=0.5 &&
        # q_on=q_off=0.5).  In this case, both ON and OFF are equally good in
        # terms of the intrinsic information.  Since it is impossible to
        # determine the state based on the intrinsic infromation, we simply set
        # the state with an arbitrary state.
        if opt_state_col.size > 1:
            opt_state_col = 0

        optimal_state[tmp_lid] = opt_state_col
        optimal_p[tmp_lid] = p[tmp_lid, opt_state_col]

        tmp_r = np.where(r_state[tmp_lid, :] == 1)[0]

        if tmp_r.size > 1:
            tmp_r = tmp_r[0]

        optimal_q[tmp_lid] = q[tmp_lid, tmp_r]


    # Find the node whose states can be determined by knowing
    # "temporary-informativeness".
    # --- For a node, about the state that acheives p>q,
    # --- if the following equation is satisfied, we can select a state that
    #     does not give p>q.

    non_fixed = np.setdiff1d(np.arange(n), fixed_node_lid)
    if non_fixed.size == 0:
        return optimal_state.astype(int).flatten()

    # Approximate method
    if non_fixed.size > MAX_SEARCH_NODE:

        val_p = np.zeros((non_fixed.size, 1))
        val_q = np.zeros((non_fixed.size, 1))

        for i in range(non_fixed.size):
            tmp_lid = non_fixed[i]
            tmp_state = np.where(r_state[tmp_lid, :] == 1)[0]
            val_p[i, 0] = p[tmp_lid, tmp_state]
            val_q[i, 0] = q[tmp_lid, tmp_state]

        val = (1/(1-2*val_p)) * (val_p*log2(val_p/val_q) - (1-val_p)*log2((1-val_p)/(1-val_q)))


        # For the nodes whose states could not be determined ("non_fixed")
        # --- compute "val" and compare it to the temporary informativeness
        #     "tmp_inform".
        # --- if val < tmp_inform, its state most probably will be the one that
        #     gives p<q.
        # --- "tmp_inform" can decrease slightly by adding more nodes in
        #     "non_fixed"
        # --- A node that has small "val" is more likely to be in "p<q" state
        #     than higher "val" nodes.
        # --- Thus, we sort nodes based on "val" and determine its state from
        #     the smallest "val" node.
        val_sort = np.sort(val)  # ascending
        idx_sort = np.argsort(val)
        for i in range(non_fixed.size):

            # Compute the "temporary-informativeness"
            tmp_inform = log2(optimal_p.prod()) - log2(optimal_q.prod())

            tmp_val = val_sort[i]
            tmp_lid = non_fixed[idx_sort[i]]
            tmp_state_col = np.where(r_state[tmp_lid, :] == 1)[0]

            if tmp_val < tmp_inform:
                opt_state = np.array(tmp_state_col == 0, dtype='int')
                optimal_state[tmp_lid] = opt_state
                optimal_p[tmp_lid] = p[tmp_lid, opt_state]
                optimal_q[tmp_lid] = q[tmp_lid, opt_state]
            else:
                opt_state = tmp_state_col
                optimal_state[tmp_lid] = opt_state
                optimal_p[tmp_lid] = p[tmp_lid, opt_state]
                optimal_q[tmp_lid] = q[tmp_lid, opt_state]

    # non_fixed.size <= MAX_SEARCH_NODE
    else:

        num_non_fixed = non_fixed.size
        non_fixed_state_all = np.array([
            s[::-1] for s in product((0, 1), repeat=num_non_fixed)
        ])
        tmp_non_fixed_p = np.zeros(non_fixed_state_all.shape)
        tmp_non_fixed_q = np.zeros(non_fixed_state_all.shape)

        for l in range(num_non_fixed):
            idx = non_fixed[l]
            tmp_p = p[idx, :].T
            tmp_q = q[idx, :].T
            tmp_non_fixed_p[:, l] = tmp_p[non_fixed_state_all[:, l]]
            tmp_non_fixed_q[:, l] = tmp_q[non_fixed_state_all[:, l]]

        fixed_p = np.prod(optimal_p)
        fixed_q = np.prod(optimal_q)
        non_fixed_p = np.prod(tmp_non_fixed_p, axis=1)
        non_fixed_q = np.prod(tmp_non_fixed_q, axis=1)

        if abs_condition == 'without_abs':
            optimal_ii = max(
                (fixed_p * non_fixed_p) * (log2(fixed_p * non_fixed_p) - log2(fixed_q * non_fixed_q))
            )
            opt_idx = argmax(
                (fixed_p * non_fixed_p) * (log2(fixed_p * non_fixed_p) - log2(fixed_q * non_fixed_q))
            )
        elif abs_condition == 'with_abs':
            optimal_ii = max(
                (fixed_p * non_fixed_p) * abs(log2(fixed_p * non_fixed_p) - log2(fixed_q * non_fixed_q))
            )
            opt_idx = max(
                (fixed_p * non_fixed_p) * abs(log2(fixed_p * non_fixed_p) - log2(fixed_q * non_fixed_q))
            )
        else:
            print('Specify "with" or "without" for the 4th input argument'
                  ' "abs_condition"')

        non_fixed_state = non_fixed_state_all[opt_idx, :]

        for l in range(num_non_fixed):
            idx = non_fixed[l]
            tmp_state = non_fixed_state[l]
            optimal_state[idx] = tmp_state
            optimal_p[idx] = p[idx, tmp_state]
            optimal_q[idx] = q[idx, tmp_state]

    return optimal_state.astype(int).flatten()


@measures.register("ID", asymmetric=True)
def intrinsic_difference(p, q):
    r"""Compute the intrinsic difference (ID) between two distributions.

    This is defined as

    .. math::
        \max_i \left\{
            p_i \log_2 \left( \frac{p_i}{q_i} \right)
        \right\}

    where we define :math:`p_i \log_2 \left( \frac{p_i}{q_i} \right)` to be
    :math:`0` when :math:`p_i = 0` or :math:`q_i = 0`.

    See the following paper:

        Barbosa LS, Marshall W, Streipert S, Albantakis L, Tononi G (2020).
        A measure for intrinsic information.
        *Sci Rep*, 10, 18803. https://doi.org/10.1038/s41598-020-75943-4

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The intrinsic difference.
    """
    return np.max(information_density(p, q))


@measures.register("AID", asymmetric=True)
@measures.register("KLM", asymmetric=True)  # Backwards-compatible alias
@measures.register("BLD", asymmetric=True)  # Backwards-compatible alias
def absolute_intrinsic_difference(p, q):
    """Compute the absolute intrinsic difference (AID) between two
    distributions.

    This is the same as the ID, but with the absolute value taken before the
    maximum is taken.

    See documentation for :func:`intrinsic_difference` for further details
    and references.

    Args:
        p (float): The first probability distribution.
        q (float): The second probability distribution.

    Returns:
        float: The absolute intrinsic difference.
    """
    return np.max(absolute_information_density(p, q))


@measures.register("IIT_4.0_SMALL_PHI", asymmetric=True)
def iit_4_small_phi(p, q, state):
    # TODO docstring
    return absolute_information_density(p, q).squeeze()[state]


@measures.register("IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE", asymmetric=True)
def iit_4_small_phi_no_absolute_value(p, q, state):
    # TODO docstring
    return information_density(p, q).squeeze()[state]


@measures.register("APMI", asymmetric=True)
@np_suppress()
def absolute_pointwise_mutual_information(p, q, state):
    """Computes the state-specific absolute pointwise mutual information between
    two distributions.

    This is the same as the MI, but with the absolute value.

    Args:
        p (np.ndarray[float]): The first probability distribution.
        q (np.ndarray[float]): The second probability distribution.

    Returns:
        float: The maximum absolute pointwise mutual information.
    """
    return np.abs(np.nan_to_num(np.log2(p / q), nan=0.0)).squeeze()[state]


@np_suppress()
def pointwise_mutual_information_vector(p, q):
    return np.nan_to_num(np.log2(p / q), nan=0.0)


@actual_causation_measures.register("PMI", asymmetric=True)
def pointwise_mutual_information(p, q):
    """Compute the pointwise mutual information (PMI).

    This is defined as

    .. math::
        \\log_2\\left(\\frac{p}{q}\\right)

    when :math:`p \\neq 0` and :math:`q \\neq 0`, and :math:`0` otherwise.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Returns:
        float: the pointwise mutual information.
    """
    if p == 0.0 or q == 0.0:
        return 0.0
    return log2(p / q)


@actual_causation_measures.register("WPMI", asymmetric=True)
def weighted_pointwise_mutual_information(p, q):
    """Compute the weighted pointwise mutual information (WPMI).

    This is defined as

    .. math::
        p \\log_2\\left(\\frac{p}{q}\\right)

    when :math:`p \\neq 0` and :math:`q \\neq 0`, and :math:`0` otherwise.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Returns:
        float: The weighted pointwise mutual information.
    """
    return p * pointwise_mutual_information(p, q)


def repertoire_distance(r1, r2, direction=None, repertoire_distance=None, **kwargs):
    """Compute the distance between two repertoires for the given direction.

    Args:
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.
        direction (Direction): |CAUSE| or |EFFECT|.

    Returns:
        float: The distance between ``r1`` and ``r2``, rounded to |PRECISION|.
    """
    func_key = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
    func = measures[func_key]
    try:
        distance = func(r1, r2, direction=direction, **kwargs)
    except TypeError:
        distance = func(r1, r2, **kwargs)
    return round(distance, config.PRECISION)
