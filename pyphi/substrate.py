# substrate.py
"""Represents the substrate of interest.

This is the primary object of PyPhi and the context of all |small_phi| and
|big_phi| computation.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from . import cache
from . import connectivity
from . import jsonify
from . import utils
from . import validate
from .direction import Direction
from .labels import NodeLabels
from .tpm import ExplicitTPM
from .types import ConnectivityMatrix
from .types import Mechanism
from .types import NodeIndices
from .types import Purview


class Substrate:
    """A substrate of nodes.

    Represents the substrate under analysis and holds auxilary data about it.

    Args:
        tpm (np.ndarray): The transition probability matrix of the substrate.

            The TPM can be provided in any of three forms: **state-by-state**,
            **state-by-node**, or **multidimensional state-by-node** form.
            In the state-by-node forms, row indices must follow the
            little-endian convention (see :ref:`little-endian-convention`). In
            state-by-state form, column indices must also follow the
            little-endian convention.

            If the TPM is given in state-by-node form, it can be either
            2-dimensional, so that ``tpm[i]`` gives the probabilities of each
            node being ON if the previous state is encoded by |i| according to
            the little-endian convention, or in multidimensional form, so that
            ``tpm[(0, 0, 1)]`` gives the probabilities of each node being ON if
            the previous state is |N_0 = 0, N_1 = 0, N_2 = 1|.

            The shape of the 2-dimensional form of a state-by-node TPM must be
            ``(s, n)``, and the shape of the multidimensional form of the TPM
            must be ``[2] * n + [n]``, where ``s`` is the number of states and
            ``n`` is the number of nodes in the substrate.

    Keyword Args:
        cm (np.ndarray): A square binary adjacency matrix indicating the
            connections between nodes in the substrate. ``cm[i][j] == 1`` means
            that node |i| is connected to node |j| (see :ref:`cm-conventions`).
            **If no connectivity matrix is given, PyPhi assumes that every node
            is connected to every node (including itself)**.
        node_labels (tuple[str] or |NodeLabels|): Human-readable labels for
            each node in the substrate.

    Example:
        In a 3-node substrate, ``the_substrate.tpm[(0, 0, 1)]`` gives the
        transition probabilities for each node at |t| given that state at |t-1|
        was |N_0 = 0, N_1 = 0, N_2 = 1|.
    """

    # TODO make tpm also optional when implementing logical substrate definition
    def __init__(
        self,
        tpm: ExplicitTPM | NDArray[np.float64] | dict[str, Any],
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        purview_cache: cache.PurviewCache | None = None,
    ) -> None:
        # Initialize _tpm according to argument type.
        if isinstance(tpm, ExplicitTPM):
            self._tpm = tpm
        elif isinstance(tpm, np.ndarray):
            self._tpm = ExplicitTPM(tpm, validate=True)
        elif isinstance(tpm, dict):
            # From JSON.
            self._tpm = ExplicitTPM(tpm["_tpm"], validate=True)
        else:
            raise TypeError(f"Invalid tpm of type {type(tpm)}.")

        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        validate.substrate(self)

    @property
    def tpm(self) -> ExplicitTPM:
        """pyphi.tpm.ExplicitTPM: The TPM object which contains this
        substrate's transition probability matrix, in multidimensional
        form.
        """
        return self._tpm

    @property
    def cm(self) -> ConnectivityMatrix:
        """np.ndarray: The substrate's connectivity matrix.

        A square binary adjacency matrix indicating the connections between
        nodes in the substrate.
        """
        return self._cm

    def _build_cm(self, cm: ArrayLike | None) -> tuple[ConnectivityMatrix, int]:
        """Convert the passed CM to the proper format, or construct the
        unitary CM if none was provided.
        """
        cm_array: ConnectivityMatrix
        if cm is None:
            # Assume all are connected.
            cm_array = np.ones((self.size, self.size), dtype=int)
        else:
            cm_array = np.array(cm, dtype=int)

        utils.np_immutable(cm_array)

        return (cm_array, utils.np_hash(cm_array))

    @property
    def connectivity_matrix(self) -> ConnectivityMatrix:
        """np.ndarray: Alias for ``cm``."""
        return self._cm

    @property
    def causally_significant_nodes(self) -> NodeIndices:
        """See :func:`pyphi.connectivity.causally_significant_nodes`."""
        return connectivity.causally_significant_nodes(self.cm)

    @property
    def size(self) -> int:
        """int: The number of nodes in the substrate."""
        return len(self)

    # TODO extend to nonbinary nodes
    @property
    def num_states(self) -> int:
        """int: The number of possible states of the substrate."""
        return 2**self.size

    @property
    def node_indices(self) -> NodeIndices:
        """tuple[int]: The indices of nodes in the substrate.

        This is equivalent to ``tuple(range(substrate.size))``.
        """
        return self._node_indices

    @property
    def node_labels(self) -> NodeLabels:
        """tuple[str]: The labels of nodes in the substrate."""
        return self._node_labels

    # TODO: this should really be a System method, but we're
    # interested in caching at the Substrate-level...
    @cache.method("purview_cache")
    def potential_purviews(
        self, direction: Direction, mechanism: Mechanism
    ) -> list[Purview]:
        """All purviews which are not clearly reducible for mechanism.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism which all purviews are
                checked for reducibility over.

        Returns:
            list[tuple[int]]: All purviews which are irreducible over
            ``mechanism``.
        """
        all_purviews = utils.powerset(self._node_indices)
        return irreducible_purviews(self.cm, direction, mechanism, all_purviews)

    # ---- substrate-level analysis ----
    #
    # Thin convenience methods that delegate to the formalism-agnostic
    # module-level functions defined below. ``sia`` and ``ces`` construct
    # a :class:`pyphi.system.System` over the requested node subset; the
    # remaining methods (``all_sias``, ``irreducible_sias``, ``complexes``,
    # ``maximal_complex``) walk the candidate space.

    def sia(
        self,
        state: tuple[int, ...],
        indices: NodeIndices | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the SIA of a single candidate system over this substrate."""
        from pyphi.system import System

        return System.from_substrate(
            self, state, indices if indices is not None else self.node_indices
        ).sia(**kwargs)

    def ces(
        self,
        state: tuple[int, ...],
        indices: NodeIndices | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the cause-effect structure of a single candidate system."""
        from pyphi.system import System

        return System.from_substrate(
            self, state, indices if indices is not None else self.node_indices
        ).ces(**kwargs)

    def all_sias(
        self,
        state: tuple[int, ...],
        candidates: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return SIAs for every candidate system; see :func:`all_sias`."""
        return all_sias(self, state, candidates=candidates, **kwargs)

    def irreducible_sias(
        self,
        state: tuple[int, ...],
        candidates: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return SIAs with |big_phi| > 0; see :func:`irreducible_sias`."""
        return irreducible_sias(self, state, candidates=candidates, **kwargs)

    def complexes(
        self,
        state: tuple[int, ...],
        candidates: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Return the substrate's complexes; see :func:`complexes`."""
        return complexes(self, state, candidates=candidates, **kwargs)

    def maximal_complex(
        self,
        state: tuple[int, ...],
        candidates: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the maximal complex; see :func:`maximal_complex`."""
        return maximal_complex(self, state, candidates=candidates, **kwargs)

    def __len__(self) -> int:
        """int: The number of nodes in the substrate."""
        return self.tpm.shape[-1]

    def __repr__(self) -> str:
        return f"Substrate({self.tpm}, cm={self.cm})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Return whether this substrate equals the other object.

        Substrates are equal if they have the same TPM and CM.
        """
        return (
            isinstance(other, Substrate)
            and self.tpm.array_equal(other.tpm)
            and np.array_equal(self.cm, other.cm)
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((hash(self.tpm), self._cm_hash))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "tpm": self.tpm,
            "cm": self.cm,
            "size": self.size,
            "node_labels": self.node_labels,
        }

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> Substrate:
        """Return a |Substrate| object from a JSON dictionary representation."""
        del json_dict["size"]
        # Older fixtures serialized tpm/cm as raw arrays, which the JSON
        # decoder converts to tuples.  Convert them back to ndarray so that
        # Substrate.__init__ can accept them.
        if isinstance(json_dict.get("tpm"), tuple):
            json_dict["tpm"] = np.asarray(json_dict["tpm"], dtype=float)
        if isinstance(json_dict.get("cm"), tuple):
            json_dict["cm"] = np.asarray(json_dict["cm"])
        return Substrate(**json_dict)


def irreducible_purviews(
    cm: ConnectivityMatrix,
    direction: Direction,
    mechanism: Mechanism,
    purviews: Iterable[Purview],
) -> list[Purview]:
    """Return all purviews which are irreducible for the mechanism.

    Args:
        cm (np.ndarray): An |N x N| connectivity matrix.
        direction (Direction): |CAUSE| or |EFFECT|.
        mechanism (tuple[int]): The mechanism in question.
        purviews (Iterable[tuple[int]]): The purviews to check.

    Returns:
        list[tuple[int]]: All purviews in ``purviews`` which are not reducible
        over ``mechanism``.

    Raises:
        ValueError: If ``direction`` is invalid.
    """

    def reducible(purview: Purview) -> bool:
        """Return ``True`` if purview is trivially reducible."""
        _from, to = direction.order(mechanism, purview)
        return connectivity.block_reducible(cm, _from, to)

    # TODO(4.0) use generator?
    return [purview for purview in purviews if not reducible(purview)]


def from_json(filename: str) -> Substrate:
    """Convert a JSON substrate to a PyPhi substrate.

    Args:
        filename (str): A path to a JSON file representing a substrate.

    Returns:
       Substrate: The corresponding PyPhi substrate object.
    """
    with open(filename, encoding="utf-8") as f:
        result: Substrate = jsonify.load(f)
        return result


# ============================================================================
# Substrate-level system iteration (formalism-agnostic)
# ============================================================================
#
# These helpers walk the powerset of node subsets and yield System (alias
# for System) instances. They don't depend on a specific formalism;
# IIT 3.0's ``all_complexes`` and IIT 4.0's ``all_complexes`` both consume
# them.


def reachable_systems(
    substrate: Substrate,
    indices: tuple[int, ...],
    state: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    """A generator over all systems in a valid state."""
    import contextlib

    from pyphi import exceptions
    from pyphi.system import System

    validate.is_substrate(substrate)

    # Return systems largest to smallest to optimize parallel
    # resource usage.
    for subset in utils.powerset(indices, nonempty=True, reverse=True):
        with contextlib.suppress(exceptions.StateUnreachableError):
            yield System.from_substrate(substrate, state, subset, **kwargs)


def systems(substrate: Substrate, state: tuple[int, ...], **kwargs: Any) -> Any:
    """Return a generator of all **possible** systems of a substrate.

    .. note::
        Does not return systems that are in an impossible state (after
        conditioning the system TPM on the state of the other nodes).
    """
    return reachable_systems(substrate, substrate.node_indices, state, **kwargs)


def possible_complexes(
    substrate: Substrate, state: tuple[int, ...], **kwargs: Any
) -> Any:
    """Return a generator of systems of a substrate that could be a complex.

    The powerset of nodes that have at least one input and one output. Nodes
    with no inputs or no outputs cannot be part of a main complex because
    they have no causal link with the rest of the system.
    """
    return reachable_systems(
        substrate, substrate.causally_significant_nodes, state, **kwargs
    )


# ============================================================================
# Substrate-level analysis (formalism-agnostic)
# ============================================================================
#
# The per-candidate SIA computation is the only formalism-specific step;
# iteration, filtering, and condensation are identical across IIT 3.0 and
# IIT 4.0. These functions resolve the active formalism once at the call
# site (avoiding per-subprocess config-mismatch hazards under parallel
# MapReduce) and then map it over the candidate iterator.


def _resolved_sia(**sia_kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve the formalism's per-system ``sia`` callable and its kwargs.

    Reads the active formalism from ``config.formalism.iit.version`` and,
    under IIT 4.0, fills in ``system_measure`` and ``specification_measure``
    from config when not supplied. Returns a ``(callable, kwargs)`` pair
    safe to hand to :class:`pyphi.parallel.MapReduce`.
    """
    from pyphi.conf import config as _config

    kwargs = dict(sia_kwargs)
    if _config.formalism.iit.version == "IIT_3_0":
        from pyphi.formalism.iit3 import sia as _sia
    else:
        from pyphi.formalism.iit4 import sia as _sia
        from pyphi.metrics.distribution import resolve_mechanism_measure
        from pyphi.metrics.distribution import resolve_system_measure

        kwargs.setdefault(
            "system_measure",
            resolve_system_measure(_config.formalism.iit.system_phi_measure),
        )
        kwargs.setdefault(
            "specification_measure",
            resolve_mechanism_measure(_config.formalism.iit.specification_measure),
        )
    return _sia, kwargs


def all_sias(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    parallel_kwargs: dict[str, Any] | None = None,
    **sia_kwargs: Any,
) -> list[Any]:
    """Return SIAs for every candidate system of the substrate.

    Includes reducible (|big_phi| = 0) candidates. The default candidate
    iterator is :func:`possible_complexes`, which skips subsets containing
    nodes that lack either inputs or outputs in the substrate — a
    mathematically safe optimization under both formalisms, since such
    candidates are not strongly connected and have |big_phi| = 0.
    """
    from pyphi import conf as _conf
    from pyphi.conf import config as _config
    from pyphi.parallel import MapReduce

    iterable = possible_complexes(substrate, state) if candidates is None else candidates

    sia_fn, map_kwargs = _resolved_sia(**sia_kwargs)
    map_kwargs.setdefault("progress", False)

    pkwargs = _conf.parallel_kwargs(
        _config.infrastructure.parallel_complex_evaluation,
        **(parallel_kwargs or {}),
    )
    result = MapReduce(
        sia_fn,
        iterable,
        total=2 ** len(substrate) - 1,
        map_kwargs=map_kwargs,
        desc="Evaluating complexes",
        **pkwargs,
    ).run()
    assert result is not None
    return result


def irreducible_sias(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Return candidate SIAs with |big_phi| > 0.

    These are *not* complexes — overlapping candidates may both appear in
    the returned list. The complexes (a subset satisfying exclusion) are
    obtained from :func:`complexes`.
    """
    return list(filter(None, all_sias(substrate, state, candidates, **kwargs)))


def _sia_node_indices(sia: Any) -> tuple[int, ...] | None:
    """Return the candidate-system node indices of a SIA, across formalisms.

    IIT 3.0 SIAs carry a ``System`` reference under ``.system``; IIT 4.0
    SIAs expose ``.node_indices`` directly.
    """
    system = getattr(sia, "system", None)
    if system is not None:
        return system.node_indices
    return getattr(sia, "node_indices", None)


def complexes(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Return the complexes of the substrate in its current state.

    A complex is a set of units that is a local maximum of |big_phi| — no
    overlapping candidate has higher |big_phi|. The returned list is
    non-overlapping (exclusion), ordered by |big_phi| descending.

    Found by greedy condensation: iterate irreducible candidates in
    descending |big_phi| order; accept each whose units do not overlap
    any already-accepted complex. This is equivalent to the recursive
    search of Albantakis et al. 2023 (Eqs. 24-25), which identifies
    :math:`S^*_k = \\mathrm{argmax}_{S \\subseteq U_k} \\varphi_s(S)` over
    a shrinking universe :math:`U_{k+1} = U_k \\setminus S^*_k`.
    """
    result: list[Any] = []
    covered: set[int] = set()
    for sia in sorted(
        irreducible_sias(substrate, state, candidates, **kwargs), reverse=True
    ):
        indices = _sia_node_indices(sia)
        if indices is not None and not (set(indices) & covered):
            result.append(sia)
            covered.update(indices)
    return result


def maximal_complex(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Return the complex with maximum |big_phi| over the substrate.

    Equivalent to the first element of :func:`complexes`. Returns a null
    SIA over the empty system when no irreducible candidate exists.
    """
    from pyphi.system import System

    found = complexes(substrate, state, candidates, **kwargs)
    if found:
        return found[0]
    # No irreducible candidate; return a null SIA over the empty system.
    empty = System.from_substrate(substrate, state, ())
    from pyphi.conf import config as _config

    if _config.formalism.iit.version == "IIT_3_0":
        from pyphi.formalism.iit3 import _null_sia

        return _null_sia(empty)
    from pyphi.formalism.iit4 import NullCauseEffectStructure

    return NullCauseEffectStructure()
