# metrics/distribution.py
"""Metrics on probability distributions."""

from contextlib import ContextDecorator
from math import log2

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import entr, rel_entr

from .. import utils, validate
from ..cache import joblib_memory
from ..conf import config, fallback
from ..direction import Direction
from ..distribution import flatten, marginal_zero
from ..exceptions import MissingOptionalDependenciesError
from ..registry import Registry

_LN_OF_2 = np.log(2)


class OptionalEMD:
    """Class to handle EMD computations.

    Allows deferring import of `pyemd` in case it is not needed.
    """

    def __init__(self):
        self._pyemd = None

    @property
    def pyemd(self):
        if self._pyemd is None:
            try:
                import pyemd

                self._pyemd = pyemd
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    MissingOptionalDependenciesError.MSG.format(dependencies="pyemd")
                ) from exc
        return self._pyemd

    def compute(self, *args, **kwargs):
        return self.pyemd.emd(*args, **kwargs)


# Usage
EMD = OptionalEMD()


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
    return EMD.compute(p, q, _hamming_matrix(N))


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


def approximate_specified_state(repertoire, partitioned_repertoire):
    """Estimate the purview state that maximizes the AID between the repertoires.

    This returns only the state of the purview nodes (i.e., there is one element
    in the state vector for each purview node, not for each node in the
    network).

    .. note::
        Although deterministic, *results are only a good guess*. This function
        should only be used in cases where running :func:`specified_state`
        becomes unfeasible.

    This algorithm runs in linear time as a function of purview size, as opposed
    to the exponential (on average) exhaustive exact search. Single-node
    (i.e. marginal) repertoires are considered one by one, and their state is
    determined according to the following heuristics:

    If the most probable state in the unpartitioned repertoire (:math:`p > 1/2`)
    becomes less probable in the partitioned one (:math:`p > q`), we should pick
    that state for that node. Note that there can be ties. In that case, the
    state with the lowest index is arbitrarily chosen.

    Now suppose that was enough to specify the state of only :math:`k` nodes,
    with joint point unpartitioned probability :math:`p_k` and partitioned
    probability :math:`q_k`, and suppose we add node :math:`z`.  Let the node
    :math:`z` have probability :math:`p_z` for the state ``0``. For the
    complementary state ``1``, the probability is :math:`1 - p_z`.  We want to
    know which state of :math:`z` gives higher intrinsic information when it is
    added to the :math:`k` nodes. In other words, we want to compare :math:`I_x`
    and :math:`I_y`:

    .. math::
        I_x = \\left( p_k p_z \\right) \\log_2 \\left( \\frac{p_k p_z}{q_k q_z} \\right)

    .. math::
        I_y = \\left( p_k (1-p_z) \\right) \\log_2 \\left( \\frac{p_k (1-p_z)}{q_k(1-q_z)} \\right)

    For state ``1`` to give higher intrinsic information (i.e., :math:`I_y >
    I_x`), :math:`p_z` and :math:`q_z` must satisfy two equations:

    .. math::
        p_z < 1/2

    .. math::
        \\log_2 \\left( \\frac{p_k}{q_k} \\right) < \\left( \\frac{1}{1-2p_z}
        \\right) \\left( p_z \\log_2 \\left( \\frac{p_z}{q_z} \\right) - (1-p_z)
        \\log_2 \\left( \\frac{1-p_z}{1-q_z} \\right) \\right)

    Otherwise, we should pick the state ``0`` as the state of node :math:`z`.

    Args:
        repertoire (np.ndarray): The first probability distribution.
        partitioned_repertoire (np.ndarray): The second probability distribution.

    Returns:
        np.ndarray: A 2D array where the single row is the approximate :func:`specified_state`.

    """

    # TODO: All the marginalization defeats the whole purpose. Config option
    # must prevent calculating outer product at `subsystem`, and pass node
    # marginal repertoires instead.
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

    P = joint_to_marginals(repertoire)
    Q = joint_to_marginals(partitioned_repertoire)

    # Preallocate arrays for the specified states and their corresponding point
    # probabilities in P and Q.
    purview_size = P.shape[0]
    specified_states = np.zeros((purview_size, 1)) * np.nan
    specified_P = np.ones((purview_size, 1))
    specified_Q = np.ones((purview_size, 1))

    # Find "fixed" nodes. A fixed node is defined as one for which its most
    # selective state according to the unpartitioned repertoire (p > 0.5) is
    # also informative for the node itself, as revealed by partitioning the
    # mechanism (i.e., p > q).
    is_selective = P >= (1 / 2)
    informativeness = P / Q
    is_informative = informativeness >= 1
    fixed_nodes = np.where(np.sum(is_selective * is_informative, axis=1))[0]

    def informative_state(node):
        return np.where(informativeness[node, :] == informativeness[node, :].max())[0]

    for fixed_node in fixed_nodes:
        specified_state = np.where(
            is_selective[fixed_node, :] * is_informative[fixed_node, :]
        )[0]

        # TODO: state ties.
        # If P[ON] == P[OFF] == Q[ON] == Q[OFF] then |specified_state| > 1.
        # Arbitrarily pick the first state.
        specified_state = specified_state[0]

        specified_states[fixed_node] = specified_state
        specified_P[fixed_node] = P[fixed_node, specified_state]
        specified_Q[fixed_node] = Q[fixed_node, informative_state(fixed_node)[0]]

    if fixed_nodes.size == purview_size:
        return specified_states.astype(int).T

    # Estimate the state of the remaining (i.e. non-fixed) nodes, one by one,
    # based on a greedy search on their impact on "temporary informativeness".

    nonfixed_nodes = np.setdiff1d(np.arange(purview_size), fixed_nodes)

    # First, compute discriminant values for every non-fixed node. This
    # discriminant will be compared to the temporary informativeness.
    p = np.array([P[n, informative_state(n)] for n in nonfixed_nodes]).flatten()
    q = np.array([Q[n, informative_state(n)] for n in nonfixed_nodes]).flatten()
    discriminants = (p * np.log2(p / q) - (1 - p) * np.log2((1 - p) / (1 - q))) / (
        1 - 2 * p
    )

    # The smaller the discriminant of a purview node, the more likely its true
    # specified state is to violate p > q. Thus we consider nodes in that order.
    discriminant_indices = np.argsort(discriminants)
    discriminants = np.sort(discriminants)  # ascending

    for index, discriminant in zip(discriminant_indices, discriminants):
        # The temporary-informativeness, updated as new nodes are included.
        tmp_inform = np.log2(specified_P.prod()) - np.log2(specified_Q.prod())

        nonfixed_node = nonfixed_nodes[index]

        # TODO: nonbinary states.
        # If discriminant < tmp_inform, select the state that gives p < q.
        if discriminant < tmp_inform:
            specified_state = int(not informative_state(nonfixed_node)[0])
        else:
            specified_state = informative_state(nonfixed_node)[0]

        specified_states[nonfixed_node] = specified_state
        specified_P[nonfixed_node] = P[nonfixed_node, specified_state]
        specified_Q[nonfixed_node] = Q[nonfixed_node, specified_state]

    return specified_states.astype(int).T


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


@measures.register("GENERALIZED_INTRINSIC_DIFFERENCE", asymmetric=True)
def generalized_intrinsic_difference(
    forward_repertoire,
    partitioned_forward_repertoire,
    selectivity_repertoire,
    state=None,
):
    informativeness = pointwise_mutual_information_vector(
        forward_repertoire, partitioned_forward_repertoire
    )
    gid = selectivity_repertoire * informativeness
    if state is None:
        return gid
    return gid[state]


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
        try:
            distance = func(r1, r2, direction=direction, **kwargs)
        except TypeError:
            distance = func(r1, r2, **kwargs)
    except TypeError:
        distance = func(r1, r2, direction=direction)
    return round(distance, config.PRECISION)
