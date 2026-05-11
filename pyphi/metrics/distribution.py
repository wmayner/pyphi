# metrics/distribution.py
"""Metrics on probability distributions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from collections.abc import Iterable
from contextlib import ContextDecorator
from math import log2
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.special import entr
from scipy.special import rel_entr

from pyphi import utils
from pyphi import validate
from pyphi.cache import joblib_memory
from pyphi.conf import config
from pyphi.data_structures.pyphi_float import PyPhiFloat
from pyphi.direction import Direction
from pyphi.distribution import flatten
from pyphi.distribution import marginal_zero
from pyphi.exceptions import MissingOptionalDependenciesError
from pyphi.metrics.protocols import CompositeMeasure
from pyphi.metrics.protocols import DistributionMeasure
from pyphi.metrics.protocols import StateAwareMeasure
from pyphi.metrics.protocols import StatefulDistributionMeasure
from pyphi.metrics.protocols import satisfies_composite_measure
from pyphi.metrics.protocols import satisfies_distribution_measure
from pyphi.metrics.protocols import satisfies_state_aware_measure
from pyphi.metrics.protocols import satisfies_stateful_distribution_measure
from pyphi.registry import Registry
from pyphi.types import Repertoire
from pyphi.types import State

_LN_OF_2 = np.log(2)


class DistanceResult(PyPhiFloat):
    """A numeric result that can carry auxiliary data about its computation.

    DistanceResult extends PyPhiFloat to attach arbitrary metadata to phi values,
    enabling introspection of how values were computed. This is particularly useful
    in scientific workflows where understanding the provenance of results is
    important.

    The class behaves like a PyPhiFloat for all mathematical operations (comparisons,
    arithmetic, min/max) while preserving metadata. This allows transparent use in
    existing code while providing rich information for analysis.

    Args:
        value: The numeric value.
        **kwargs: Arbitrary keyword arguments stored as metadata attributes.

    Attributes:
        All attributes from float and PyPhiFloat are available, plus any metadata
        passed as keyword arguments.

    Note:
        **NumPy Array Performance**: When creating NumPy arrays from DistanceResult
        objects, the ``__array__`` protocol automatically extracts float values,
        creating fast float64 arrays. Metadata is not preserved in the array.

        This design provides the best of both worlds:

        - **Batch numerical analysis**: ``DistanceResult.values_array(results)``
          returns the float values as a NumPy array (metadata dropped).
        - **Sophisticated users**: Metadata available on individual results

    Examples:
        Basic usage with metadata:

        >>> from pyphi.metrics.distribution import DistanceResult
        >>> result = DistanceResult(0.5, method='EMD', direction='CAUSE')
        >>> float(result)  # Extract numeric value
        0.5
        >>> result.method  # Access metadata
        'EMD'
        >>> result.direction
        'CAUSE'

        Mathematical operations preserve the numeric value:

        >>> result + 0.3
        0.8
        >>> result * 2
        1.0
        >>> result > 0.3
        True

        Type preservation in min/max with metadata:

        >>> results = [
        ...     DistanceResult(0.5, method='EMD'),
        ...     DistanceResult(0.3, method='L1'),
        ...     DistanceResult(0.7, method='GID')
        ... ]
        >>> min_result = min(results)
        >>> float(min_result)
        0.3
        >>> min_result.method  # Metadata from the minimum value is preserved
        'L1'

        NumPy array creation (explicit float extraction):

        >>> results = [DistanceResult(0.5), DistanceResult(0.3), DistanceResult(0.7)]
        >>> arr = DistanceResult.values_array(results)
        >>> arr.dtype
        dtype('float64')
        >>> arr
        array([0.5, 0.3, 0.7])

        JSON serialization with metadata:

        >>> result = DistanceResult(0.5, method='EMD', direction='CAUSE')
        >>> json_data = result.to_json()
        >>> json_data
        {'value': 0.5, 'method': 'EMD', 'direction': 'CAUSE'}
        >>> restored = DistanceResult.from_json(json_data)
        >>> restored.method
        'EMD'

        Typical scientific workflow:

        >>> # Compute multiple phi values
        >>> phi_values = [
        ...     DistanceResult(0.5, method='EMD', system='ABC'),
        ...     DistanceResult(0.3, method='L1', system='ABC'),
        ...     DistanceResult(0.7, method='GID', system='DEF')
        ... ]  # doctest: +SKIP
        >>> # Find maximum
        >>> max_phi = max(phi_values)  # doctest: +SKIP
        >>> print(f"Max φ = {max_phi:.3f} using {max_phi.method}")  # doctest: +SKIP
        Max φ = 0.700 using GID
        >>> # Statistical analysis
        >>> # Explicitly extract floats for statistics (metadata dropped):
        >>> phi_array = DistanceResult.values_array(phi_values)  # doctest: +SKIP
        >>> np.mean(phi_array)  # doctest: +SKIP
        0.5
    """

    def __new__(cls, value, **kwargs):
        instance = super().__new__(cls, value)
        for key, val in kwargs.items():
            setattr(instance, key, val)
        return instance

    def _public_aux_data(self) -> dict:
        """Auxiliary data the user attached at construction or via setattr.

        Excludes underscore-prefixed names (used internally — e.g., for
        the precision snapshot inherited from :class:`PyPhiFloat`)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self):
        aux_data = self._public_aux_data()
        if aux_data:
            aux_str = ", ".join(f"{k}={v!r}" for k, v in aux_data.items())
            return f"DistanceResult({float(self)}, {aux_str})"
        return f"DistanceResult({float(self)})"

    def __str__(self):
        """Short representation for use in formatted output."""
        from pyphi.models import fmt

        return fmt.fmt_number(float(self))

    def _preserve_aux_data(self, other_result):
        """Copy auxiliary data from another DistanceResult if this one wins a
        comparison."""
        if isinstance(other_result, DistanceResult):
            for key, val in other_result._public_aux_data().items():
                if not hasattr(self, key):
                    setattr(self, key, val)
        return self

    def __copy__(self):
        """Ensure auxiliary data is preserved when copying."""
        return DistanceResult(float(self), **self._public_aux_data())

    def __deepcopy__(self, memo):
        """Ensure auxiliary data is preserved when deep copying."""
        import copy

        aux_data = {
            k: copy.deepcopy(v, memo) for k, v in self._public_aux_data().items()
        }
        return DistanceResult(float(self), **aux_data)

    def to_json(self) -> dict:
        """Serialize to JSON, preserving auxiliary data."""
        result = {"value": float(self)}
        result.update(self._public_aux_data())
        return result

    @classmethod
    def from_json(cls, data: dict) -> DistanceResult:
        """Deserialize from JSON, restoring auxiliary data."""
        value = data["value"]
        # All keys except "value" are auxiliary data
        aux_data = {k: v for k, v in data.items() if k != "value"}
        return cls(value, **aux_data)

    @classmethod
    def values_array(
        cls, results: Iterable[DistanceResult], dtype: Any = None
    ) -> np.ndarray:
        """Return the float values of an iterable of ``DistanceResult``s
        as a NumPy array.

        Auxiliary metadata (``method``, ``state``, etc.) is intentionally
        dropped — callers that need it should iterate the input directly.
        Use this method when you want explicit control over the metadata-loss
        boundary, rather than relying on implicit ``np.array(results)``
        coercion (which was previously implemented via ``__array__`` and
        silently dropped metadata at unpredictable points).

        Args:
            results: Iterable of DistanceResult objects.
            dtype: Optional NumPy dtype for the array (default: float64).

        Returns:
            np.ndarray: 1-D array of the float values.

        Examples:
            >>> results = [DistanceResult(0.5, method='EMD'),
            ...            DistanceResult(0.3, method='L1')]
            >>> arr = DistanceResult.values_array(results)
            >>> arr.dtype
            dtype('float64')
            >>> arr
            array([0.5, 0.3])
        """
        import numpy as np

        if dtype is None:
            dtype = np.float64
        return np.fromiter((float(r) for r in results), dtype=dtype)


class OptionalEMD:
    """Class to handle EMD computations.

    Allows deferring import of `pyemd` in case it is not needed.
    """

    def __init__(self) -> None:
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

    def compute(self, *args, **kwargs) -> float:
        return float(self.pyemd.emd(*args, **kwargs))


# Usage
EMD = OptionalEMD()


class DistributionMeasureRegistry(Registry):
    """Storage for ``(p, q) -> float | DistanceResult`` distance functions.

    Each registered function is validated at registration time to have
    exactly two required positional parameters named ``p`` and ``q``;
    signature drift fails at import rather than deep in a phi
    computation. The ``asymmetric`` flag attaches to the function as an
    attribute so callers can filter without consulting a parallel list.

    Examples:
        >>> @distribution_measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(p, q):
        ...    return 0
    """

    # pylint: disable=arguments-differ

    desc = "distribution-to-distribution distance functions"

    def register(  # type: ignore[override]
        self, name: str, asymmetric: bool = False
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator for registering a :class:`DistributionMeasure`.

        Args:
            name: The name of the measure.

        Keyword Args:
            asymmetric: ``True`` if the measure is asymmetric. Stored as
                an attribute on the function.
        """

        def register_func(func: Callable[..., float]) -> Callable[..., float]:
            if not satisfies_distribution_measure(func):
                raise TypeError(
                    f"Cannot register {func!r} as DistributionMeasure {name!r}: "
                    f"required params must be exactly (p, q); got "
                    f"{list(inspect.signature(func).parameters)}."
                )
            func.name = name  # type: ignore[attr-defined]
            func.asymmetric = asymmetric  # type: ignore[attr-defined]
            self.store[name] = func
            return func

        return register_func


class StateAwareMeasureRegistry(Registry):
    """Storage for ``(p, state) -> float | DistanceResult`` measures.

    The function reads off a single state's value from a single
    distribution.
    """

    # pylint: disable=arguments-differ

    desc = "pointwise state-aware measures"

    def register(  # type: ignore[override]
        self, name: str
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator for registering a :class:`StateAwareMeasure`."""

        def register_func(func: Callable[..., float]) -> Callable[..., float]:
            if not satisfies_state_aware_measure(func):
                raise TypeError(
                    f"Cannot register {func!r} as StateAwareMeasure {name!r}: "
                    f"required params must be exactly (p, state); got "
                    f"{list(inspect.signature(func).parameters)}."
                )
            func.name = name  # type: ignore[attr-defined]
            self.store[name] = func
            return func

        return register_func


class CompositeMeasureRegistry(Registry):
    """Storage for composite measures of shape
    ``(forward, partitioned, selectivity, *, state) -> DistanceResult``.

    Used at the system / mechanism boundary by GID, INTRINSIC_SPECIFICATION,
    and INTRINSIC_INFORMATION.
    """

    # pylint: disable=arguments-differ

    desc = "composite measures"

    def register(  # type: ignore[override]
        self, name: str, asymmetric: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering a :class:`CompositeMeasure`.

        Args:
            name: The name of the measure.

        Keyword Args:
            asymmetric: ``True`` if the measure is asymmetric. Stored as
                an attribute on the function.
        """

        def register_func(func: Callable[..., Any]) -> Callable[..., Any]:
            if not satisfies_composite_measure(func):
                raise TypeError(
                    f"Cannot register {func!r} as CompositeMeasure {name!r}: "
                    f"first three params must include 'forward', "
                    f"'partitioned', and 'selectivity'; got "
                    f"{list(inspect.signature(func).parameters)}."
                )
            func.name = name  # type: ignore[attr-defined]
            func.asymmetric = asymmetric  # type: ignore[attr-defined]
            self.store[name] = func
            return func

        return register_func


class StatefulDistributionMeasureRegistry(Registry):
    """Storage for ``(p, q, state) -> float | DistanceResult`` measures.

    Both distributions are load-bearing; the state selects an element.
    Used by IIT_4.0_SMALL_PHI variants and APMI.
    """

    # pylint: disable=arguments-differ

    desc = "two-distribution state-aware measures"

    def register(  # type: ignore[override]
        self, name: str, asymmetric: bool = False
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator for registering a :class:`StatefulDistributionMeasure`.

        Args:
            name: The name of the measure.

        Keyword Args:
            asymmetric: ``True`` if the measure is asymmetric. Stored as
                an attribute on the function.
        """

        def register_func(func: Callable[..., float]) -> Callable[..., float]:
            if not satisfies_stateful_distribution_measure(func):
                raise TypeError(
                    f"Cannot register {func!r} as StatefulDistributionMeasure "
                    f"{name!r}: required params must be exactly (p, q, state); "
                    f"got {list(inspect.signature(func).parameters)}."
                )
            func.name = name  # type: ignore[attr-defined]
            func.asymmetric = asymmetric  # type: ignore[attr-defined]
            self.store[name] = func
            return func

        return register_func


distribution_measures = DistributionMeasureRegistry()
state_aware_measures = StateAwareMeasureRegistry()
composite_measures = CompositeMeasureRegistry()
stateful_distribution_measures = StatefulDistributionMeasureRegistry()


class ActualCausationMeasureRegistry(Registry):
    """Storage for distance functions used in :mod:`pyphi.actual`.

    Users can define custom measures:

    Examples:
        >>> @actual_causation_measures.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def always_zero(a, b):
        ...    return 0

    And use them by setting, *e.g.*, ``config.repertoire_distance = 'ALWAYS_ZERO'``.
    """

    # pylint: disable=arguments-differ

    desc = "distance functions for use in actual causation calculations"

    def __init__(self) -> None:
        super().__init__()
        self._asymmetric: list[str] = []

    def register(
        self, name: str, asymmetric: bool = False
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Decorator for registering an actual causation measure with PyPhi.

        Args:
            name (string): The name of the measure.

        Keyword Args:
            asymmetric (boolean): ``True`` if the measure is asymmetric.
        """

        def register_func(func: Callable[..., float]) -> Callable[..., float]:
            if asymmetric:
                self._asymmetric.append(name)
            self.store[name] = func
            return func

        return register_func

    def asymmetric(self) -> list[str]:
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

    def __init__(self) -> None:
        super().__init__(divide="ignore", invalid="ignore")


# Load precomputed hamming matrices.
_NUM_PRECOMPUTED_HAMMING_MATRICES = 10
_hamming_matrices = utils.load_data(
    "hamming_matrices", _NUM_PRECOMPUTED_HAMMING_MATRICES
)


# TODO extend to nonbinary nodes
def _hamming_matrix(N: int) -> np.ndarray:
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
def _compute_hamming_matrix(N: int) -> np.ndarray:
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
    possible_states = np.array(list(utils.all_states(N)))
    return cdist(possible_states, possible_states, "hamming") * N


# TODO extend to nonbinary nodes
def hamming_emd(p: ArrayLike, q: ArrayLike) -> float:
    """Return the Earth Mover's Distance between two distributions (indexed
    by state, one dimension per node) using the Hamming distance between states
    as the transportation cost function.

    Singleton dimensions are sqeezed out.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    N = p.squeeze().ndim
    p_flat = flatten(p)
    q_flat = flatten(q)
    assert p_flat is not None
    assert q_flat is not None
    return EMD.compute(p_flat, q_flat, _hamming_matrix(N))


def effect_emd(p: ArrayLike, q: ArrayLike) -> float:
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
    p = np.asarray(p)
    q = np.asarray(q)
    return float(
        sum(abs(marginal_zero(p, i) - marginal_zero(q, i)) for i in range(p.ndim))
    )


@distribution_measures.register("EMD")
def emd(p: ArrayLike, q: ArrayLike, direction: Direction | None = None) -> float:
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
        raise ValueError(f"Invalid direction: {direction}")

    return DistanceResult(func(p, q), method="EMD", direction=direction)


@distribution_measures.register("L1")
def l1(p: ArrayLike, q: ArrayLike) -> float:
    """Return the L1 distance between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The sum of absolute differences of ``p`` and ``q``.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return DistanceResult(np.abs(p - q).sum(), method="L1")


@distribution_measures.register("ENTROPY_DIFFERENCE")
def entropy_difference(p: ArrayLike, q: ArrayLike) -> float:
    """Return the difference in entropy between two distributions."""
    hp = entr(p).sum() / _LN_OF_2
    hq = entr(q).sum() / _LN_OF_2
    return DistanceResult(abs(hp - hq), method="ENTROPY_DIFFERENCE")


@distribution_measures.register("PSQ2")
def psq2(p: ArrayLike, q: ArrayLike) -> float:
    r"""Compute the PSQ2 measure.

    This is defined as :math:`\mid f(p) - f(q) \mid`, where

    .. math::
        f(x) = \sum_{i=0}^{N-1} p_i^2 \log_2 (p_i N)

    Args:
        p (np.ndarray): The first distribution.
        q (np.ndarray): The second distribution.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    fp = (p * (-1.0 * entr(p))).sum() / _LN_OF_2 + (p**2 * log2(len(p))).sum()
    fq = (q * (-1.0 * entr(q))).sum() / _LN_OF_2 + (q**2 * log2(len(q))).sum()
    return DistanceResult(abs(fp - fq), method="PSQ2")


@distribution_measures.register("MP2Q", asymmetric=True)
@np_suppress()
def mp2q(p: ArrayLike, q: ArrayLike) -> float:
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
    p = np.asarray(p)
    q = np.asarray(q)
    # There is already a factor of p in the `information_density`, so we only
    # multiply by p, not p**2
    return DistanceResult(
        np.sum(p / q * information_density(p, q) / len(p)),
        method="MP2Q",
        asymmetric=True,
    )


def information_density(p: ArrayLike, q: ArrayLike) -> np.ndarray:
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


@distribution_measures.register("KLD", asymmetric=True)
def kld(p: ArrayLike, q: ArrayLike) -> float:
    """Return the Kullback-Leibler Divergence (KLD) between two distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The KLD of ``p`` from ``q``.
    """
    return DistanceResult(information_density(p, q).sum(), method="KLD", asymmetric=True)


def absolute_information_density(p: ArrayLike, q: ArrayLike) -> np.ndarray:
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


def approximate_specified_state(
    repertoire: ArrayLike, partitioned_repertoire: ArrayLike
) -> np.ndarray:
    """Estimate the purview state that maximizes the AID between the repertoires.

    This returns only the state of the purview nodes (i.e., there is one element
    in the state vector for each purview node, not for each node in the
    substrate).

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
        I_y = \\left( p_k (1-p_z) \\right) \\log_2 \\left(
        \\frac{p_k (1-p_z)}{q_k(1-q_z)} \\right)

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
        np.ndarray: A 2D array where the single row is the approximate
            :func:`specified_state`.

    """

    # TODO: All the marginalization defeats the whole purpose. Config option
    # must prevent calculating outer product at `system`, and pass node
    # marginal repertoires instead.
    def joint_to_marginals(repertoire: np.ndarray) -> np.ndarray:
        """Converts a joint repertoire in multidimensional form to a 2D array of
        single-node marginal repertoires.

        Args:
            repertoire (np.ndarray): The joint repertoire of a purview in
                multidimensional form, e.g., as obtained from
                :mod:`pyphi.system`. Note that `repertoire` is assumed to be
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
        complements = [node_indices - {n} for n in tuple(node_indices)]
        # Marginalize out all the complementary dimensions for each
        # node in the repertoire.
        marginals = [repertoire.sum(tuple(c)) for c in complements]
        return np.vstack(marginals)

    P = joint_to_marginals(np.asarray(repertoire))
    Q = joint_to_marginals(np.asarray(partitioned_repertoire))

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

    for index, discriminant in zip(discriminant_indices, discriminants, strict=False):
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


@distribution_measures.register("ID", asymmetric=True)
def intrinsic_difference(p: ArrayLike, q: ArrayLike) -> float:
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
    return DistanceResult(
        np.max(information_density(p, q)), method="ID", asymmetric=True
    )


@distribution_measures.register("AID", asymmetric=True)
@distribution_measures.register("KLM", asymmetric=True)  # Backwards-compatible alias
@distribution_measures.register("BLD", asymmetric=True)  # Backwards-compatible alias
def absolute_intrinsic_difference(p: ArrayLike, q: ArrayLike) -> float:
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
    return DistanceResult(
        np.max(absolute_information_density(p, q)), method="AID", asymmetric=True
    )


@stateful_distribution_measures.register("IIT_4.0_SMALL_PHI", asymmetric=True)
def iit_4_small_phi(p: ArrayLike, q: ArrayLike, state: State) -> float:
    # TODO docstring
    return DistanceResult(
        absolute_information_density(p, q).squeeze()[state],
        method="IIT_4.0_SMALL_PHI",
        asymmetric=True,
        state=state,
    )


@stateful_distribution_measures.register(
    "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE", asymmetric=True
)
def iit_4_small_phi_no_absolute_value(p: ArrayLike, q: ArrayLike, state: State) -> float:
    # TODO docstring
    return DistanceResult(
        information_density(p, q).squeeze()[state],
        method="IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
        asymmetric=True,
        state=state,
    )


@composite_measures.register("GENERALIZED_INTRINSIC_DIFFERENCE", asymmetric=True)
@composite_measures.register("INTRINSIC_SPECIFICATION", asymmetric=True)
def generalized_intrinsic_difference(
    forward_repertoire: ArrayLike,
    partitioned_forward_repertoire: ArrayLike,
    selectivity_repertoire: ArrayLike,
    state: State | None = None,
) -> Repertoire | float:
    selectivity_repertoire = np.asarray(selectivity_repertoire)
    informativeness = pointwise_mutual_information_vector(
        forward_repertoire, partitioned_forward_repertoire
    )
    gid = selectivity_repertoire * informativeness
    if state is None:
        return gid
    return DistanceResult(
        gid[state],
        method="GENERALIZED_INTRINSIC_DIFFERENCE",
        asymmetric=True,
        state=state,
    )


intrinsic_specification = generalized_intrinsic_difference  # alias


def pointwise_intrinsic_differentiation(p):
    return -np.log2(p, where=(p > 0))


@state_aware_measures.register("INTRINSIC_DIFFERENTIATION")
def intrinsic_differentiation(p, state):
    p = p.squeeze()[state]
    positive_entries = pointwise_intrinsic_differentiation(p)[
        pointwise_intrinsic_differentiation(p) > 0
    ]
    return DistanceResult(
        np.min(positive_entries) if positive_entries.size > 0 else 0.0,
        method="INTRINSIC_DIFFERENTIATION",
        asymmetric=False,
        state=state,
    )


@composite_measures.register("INTRINSIC_INFORMATION", asymmetric=True)
def intrinsic_information(
    forward_repertoire,
    partitioned_forward_repertoire,
    selectivity_repertoire,
    state=None,
):
    iit_cfg = config.formalism.iit
    specification_func = composite_measures[iit_cfg.specification_measure]
    differentiation_func = state_aware_measures[iit_cfg.differentiation_measure]

    specification = specification_func(
        forward_repertoire,
        partitioned_forward_repertoire,
        selectivity_repertoire,
        state=state,
    )
    differentiation = differentiation_func(forward_repertoire, state=state)
    # Assumes single value at this point; state selection delegated to sub-functions.
    if not np.isscalar(specification) or not np.isscalar(differentiation):
        return np.minimum(specification, differentiation)
    # Single value
    return DistanceResult(
        min(specification, differentiation),  # pyright: ignore[reportArgumentType]
        method="INTRINSIC_INFORMATION",
        asymmetric=True,
        state=state,
        specification=specification,
        differentiation=differentiation,
    )


@stateful_distribution_measures.register("APMI", asymmetric=True)
@np_suppress()
def absolute_pointwise_mutual_information(
    p: ArrayLike, q: ArrayLike, state: int | tuple[int, ...]
) -> float:
    """Computes the state-specific absolute pointwise mutual information between
    two distributions.

    This is the same as the MI, but with the absolute value.

    Args:
        p (np.ndarray[float]): The first probability distribution.
        q (np.ndarray[float]): The second probability distribution.

    Returns:
        float: The maximum absolute pointwise mutual information.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return float(np.abs(np.nan_to_num(np.log2(p / q), nan=0.0)).squeeze()[state])


@np_suppress()
def pointwise_mutual_information_vector(p: ArrayLike, q: ArrayLike) -> np.ndarray:
    p = np.asarray(p)
    q = np.asarray(q)
    return np.nan_to_num(np.log2(p / q), nan=0.0)


@actual_causation_measures.register("PMI", asymmetric=True)
def pointwise_mutual_information(p: float, q: float) -> float:
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
def weighted_pointwise_mutual_information(p: float, q: float) -> float:
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


# ---------------------------------------------------------------------------
# Resolver helpers: name -> typed measure callable
# ---------------------------------------------------------------------------


def resolve_mechanism_measure(
    name: str,
) -> (
    DistributionMeasure
    | StateAwareMeasure
    | StatefulDistributionMeasure
    | CompositeMeasure
):
    """Look up a measure usable at the mechanism level.

    Mechanism-level integration accepts distribution measures (IIT 3.0
    EMD/L1/KLD/...), state-aware pointwise measures, stateful-distribution
    measures (IIT 4.0 small-phi variants), or composite measures (GID at
    the partition layer). Pyright sees the Protocol union; downstream
    parameters typed as a narrower Protocol (e.g.,
    :class:`CompositeMeasure` at the system level) statically reject
    scope-mismatched assignments.
    """
    from typing import cast

    if name in distribution_measures:
        return cast(DistributionMeasure, distribution_measures[name])
    if name in state_aware_measures:
        return cast(StateAwareMeasure, state_aware_measures[name])
    if name in stateful_distribution_measures:
        return cast(StatefulDistributionMeasure, stateful_distribution_measures[name])
    if name in composite_measures:
        return cast(CompositeMeasure, composite_measures[name])
    available = sorted(
        set(distribution_measures)
        | set(state_aware_measures)
        | set(stateful_distribution_measures)
        | set(composite_measures)
    )
    raise ValueError(f"Unknown mechanism measure {name!r}. Available: {available}")


def resolve_system_measure(name: str) -> CompositeMeasure:
    """Look up a measure usable at the system level.

    Only composite measures are valid system-level measures; the return
    type is :class:`CompositeMeasure` so pyright catches scope mismatches.
    """
    from typing import cast

    if name in composite_measures:
        return cast(CompositeMeasure, composite_measures[name])
    raise ValueError(
        f"Unknown system measure {name!r}. Available: {sorted(composite_measures)}"
    )


def resolve_distribution_measure(name: str) -> DistributionMeasure:
    """Look up a distribution measure for EMD ground distance and IIT-side dispatch.

    Only distribution measures (two-distribution distances) are valid
    here; the return type is :class:`DistributionMeasure` so pyright
    catches scope mismatches.
    """
    from typing import cast

    if name in distribution_measures:
        return cast(DistributionMeasure, distribution_measures[name])
    raise ValueError(
        f"Unknown distribution measure {name!r}. "
        f"Available: {sorted(distribution_measures)}"
    )


def resolve_actual_causation_measure(name: str) -> DistributionMeasure:
    """Look up a measure registered in :data:`actual_causation_measures`.

    The actual-causation alpha computation uses ``(p, q) -> float``
    distribution-shape callables from :data:`actual_causation_measures`
    (e.g., ``PMI``, ``WPMI``). The return type is :class:`DistributionMeasure`
    so call-site type checks reject scope-mismatched assignments.
    """
    from typing import cast

    if name in actual_causation_measures:
        return cast(DistributionMeasure, actual_causation_measures[name])
    raise ValueError(
        f"Unknown actual-causation measure {name!r}. "
        f"Available: {sorted(actual_causation_measures)}"
    )


def repertoire_distance(
    r1: ArrayLike,
    r2: ArrayLike,
    direction: Direction | None = None,
    repertoire_distance: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
        | None
    ) = None,
    **kwargs,
) -> float:
    """Compute the distance between two repertoires for the given direction.

    Args:
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.
        direction (Direction): |CAUSE| or |EFFECT|.
        repertoire_distance: A Protocol-typed measure callable
            (:class:`DistributionMeasure`, :class:`StateAwareMeasure`,
            :class:`StatefulDistributionMeasure`, or
            :class:`CompositeMeasure`). Required for callers below the
            formalism-class boundary; public-API callers
            (``System.cause_info``, etc.) resolve from config at their
            method boundary and pass the object through.

    Returns:
        float: The distance between ``r1`` and ``r2``, rounded to |PRECISION|.
    """
    if repertoire_distance is None:
        raise ValueError(
            "repertoire_distance must be provided explicitly; callers below "
            "the formalism boundary thread the measure object as a kwarg."
        )
    func = repertoire_distance
    if satisfies_stateful_distribution_measure(func):
        # (p, q, state) — caller threads ``state`` via kwargs. ``direction``
        # is not part of the stateful measure signature.
        try:
            state = kwargs.pop("state")
        except KeyError as exc:
            raise TypeError(
                f"StatefulDistributionMeasure "
                f"{getattr(func, 'name', repr(func))!r} requires a 'state' "
                f"keyword argument, but none was provided."
            ) from exc
        distance = func(r1, r2, state)  # type: ignore[call-arg]
    elif satisfies_distribution_measure(func):
        # (p, q) — may accept an optional ``direction`` parameter (e.g. EMD)
        # and/or additional keyword arguments. Pass only those that appear
        # in the measure's signature so unrecognized kwargs do not produce
        # signature-mismatch TypeErrors masquerading as bugs.
        sig_params = inspect.signature(func).parameters
        call_kwargs: dict[str, Any] = {
            name: value for name, value in kwargs.items() if name in sig_params
        }
        if "direction" in sig_params:
            call_kwargs["direction"] = direction
        distance = func(r1, r2, **call_kwargs)  # type: ignore[call-arg]
    else:
        observed_params = list(inspect.signature(func).parameters)
        raise TypeError(
            f"Cannot dispatch repertoire_distance with measure "
            f"{getattr(func, 'name', repr(func))!r}: signature parameters "
            f"{observed_params} do not match any registered measure Protocol "
            f"that takes two repertoires (DistributionMeasure or "
            f"StatefulDistributionMeasure)."
        )
    return round(distance, config.numerics.precision)  # type: ignore[arg-type]
