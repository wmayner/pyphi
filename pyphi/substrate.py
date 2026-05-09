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
