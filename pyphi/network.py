# network.py
"""Represents the network of interest.

This is the primary object of PyPhi and the context of all |small_phi| and
|big_phi| computation.
"""

from typing import Any, Dict, Optional, Sequence, Union
import numpy as np

from . import cache, connectivity, jsonify, utils, validate
from .labels import NodeLabels
from .node import generate_nodes, generate_node
from .tpm import ExplicitTPM, ImplicitTPM


class Network:
    """A network of nodes.

    Represents the network under analysis and holds auxilary data about it.

    Args:
        tpm (np.ndarray or ExplicitTPM or Sequence[np.ndarray] or ImplicitTPM):
            The transition probability matrix of the network.

            If a single numpy.ndarray or |ExplicitTPM| is provided, pyphi
            assumes it is an old-style TPM for the whole network, and it will be
            converted to an |ImplicitTPM|.

            If an |ImplicitTPM| or a list of numpy.ndarray is provided, it must
            contain one TPM per node, and their order should match those of
            ``cm`` and ``node_labels``. For node |j|, the number of dimensions
            of its TPM must be |inputs(j) + 1| and its shape must be
            |(s_1, s_2, ... , s_i, s_j)| (also in order), where |inputs(j)| is
            the number of nodes that are direct inputs of |j| and |s_i| is the
            number of states for node |i|. In other words ``tpm_j[0, 1, 2, 3]``
            stands for |Pr(j_{t+1}=3 | a_{t}=0, j_{t}=1, z_{t}=2)|.

            See :ref:`tpm-conventions:`.

    Keyword Args:
        cm (np.ndarray): A square binary adjacency matrix indicating the
            connections between nodes in the network. ``cm[i][j] == 1`` means
            that node |i| is connected to node |j| (see :ref:`cm-conventions`).
            **If no connectivity matrix is given, PyPhi assumes that every node
            is connected to every node (including itself)**.
        node_labels (Sequence[str] or |NodeLabels|): Human-readable labels for
            each node in the network.

    """

    def __init__(
            self,
            tpm: Union[np.ndarray, ExplicitTPM, Sequence, ImplicitTPM, Dict[str, Any]],
            cm: Optional[np.ndarray] = None,
            node_labels: Optional[Sequence[str]] = None,
            purview_cache: Optional[cache.PurviewCache] = None,
    ):
        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        # Initialize _tpm according to argument type.

        if isinstance(tpm, (np.ndarray, ExplicitTPM)):
            # Old-style TPM: validate and convert to state-by-node format first.
            tpm = ExplicitTPM(tpm, validate=True)
            nodes = generate_nodes(
                tpm,
                self._cm,
                self._node_indices,
                self._node_labels
            )
            self._tpm = ImplicitTPM(nodes)

        elif isinstance(tpm, Sequence):
            # Individual node TPMs were provided, format into an ImplicitTPM.
            invalid = [
                i for i in tpm if not isinstance(i, (np.ndarray, ExplicitTPM))
            ]

            if invalid:
                raise TypeError("Invalid set of nodes containing {}.".format(
                    ', '.join(str(i) for i in invalid)
                ))

            tpm = [ExplicitTPM(node_tpm, validate=False) for node_tpm in tpm]

            nodes = tuple(
                generate_node(node_tpm, self._cm, index, self._node_labels)
                for index, node_tpm in zip(self._node_indices, tpm)
            )
            self._tpm = ImplicitTPM(nodes)

        elif isinstance(tpm, ImplicitTPM):
            self._tpm = tpm

        # FIXME(TPM) initialization from JSON
        elif isinstance(tpm, dict):
            # From JSON.
            self._tpm = ImplicitTPM(tpm["_tpm"])

        else:
            raise TypeError(f"Invalid TPM of type {type(tpm)}.")

        validate.network(self)

    @property
    def tpm(self):
        """ExplicitTPM: The TPM object which contains this
        network's transition probability matrix, in multidimensional
        form.
        """
        return self._tpm

    @property
    def cm(self):
        """np.ndarray: The network's connectivity matrix.

        A square binary adjacency matrix indicating the connections between
        nodes in the network.
        """
        return self._cm

    def _build_cm(self, cm):
        """Convert the passed CM to the proper format, or construct the
        unitary CM if none was provided.
        """
        if cm is None:
            # Assume all are connected.
            cm = np.ones((self.size, self.size))
        else:
            cm = np.array(cm)

        utils.np_immutable(cm)

        return (cm, utils.np_hash(cm))

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for ``cm``."""
        return self._cm

    @property
    def causally_significant_nodes(self):
        """See :func:`pyphi.connectivity.causally_significant_nodes`."""
        return connectivity.causally_significant_nodes(self._cm)

    @property
    def size(self):
        """int: The number of nodes in the network."""
        return len(self)

    @property
    def num_states(self):
        """int: The number of possible states of the network."""
        return np.prod(self._tpm.shape)

    @property
    def node_indices(self):
        """tuple[int]: The indices of nodes in the network.

        This is equivalent to ``tuple(range(network.size))``.
        """
        return self._node_indices

    @property
    def node_labels(self):
        """tuple[str]: The labels of nodes in the network."""
        return self._node_labels

    # TODO: this should really be a Subsystem method, but we're
    # interested in caching at the Network-level...
    @cache.method("purview_cache")
    def potential_purviews(self, direction, mechanism):
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

    def __len__(self):
        """int: The number of nodes in the network."""
        return self._cm.shape[0]

    def __repr__(self):
        cm = str(self.cm).replace('\n', '\n       ')
        return "Network(\n    {},\n    cm={},\n    node_labels={}\n)".format(
            self.tpm, cm, self.node_labels.labels
        )

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Networks are equal if they have the same TPM and CM.
        """
        return (
            isinstance(other, Network)
            and self.tpm.array_equal(other.tpm)
            and np.array_equal(self.cm, other.cm)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.tpm), self._cm_hash))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "tpm": self.tpm,
            "cm": self.cm,
            "size": self.size,
            "node_labels": self.node_labels,
        }

    @classmethod
    def from_json(cls, json_dict):
        """Return a |Network| object from a JSON dictionary representation."""
        del json_dict["size"]
        return Network(**json_dict)


def irreducible_purviews(cm, direction, mechanism, purviews):
    """Return all purviews which are irreducible for the mechanism.

    Args:
        cm (np.ndarray): An |N x N| connectivity matrix.
        direction (Direction): |CAUSE| or |EFFECT|.
        purviews (list[tuple[int]]): The purviews to check.
        mechanism (tuple[int]): The mechanism in question.

    Returns:
        list[tuple[int]]: All purviews in ``purviews`` which are not reducible
        over ``mechanism``.

    Raises:
        ValueError: If ``direction`` is invalid.
    """

    def reducible(purview):
        """Return ``True`` if purview is trivially reducible."""
        _from, to = direction.order(mechanism, purview)
        return connectivity.block_reducible(cm, _from, to)

    # TODO(4.0) use generator?
    return [purview for purview in purviews if not reducible(purview)]


def from_json(filename):
    """Convert a JSON network to a PyPhi network.

    Args:
        filename (str): A path to a JSON file representing a network.

    Returns:
       Network: The corresponding PyPhi network object.
    """
    with open(filename) as f:
        return jsonify.load(f)
