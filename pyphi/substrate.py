# substrate.py
"""Represents the substrate of interest.

This is the primary object of PyPhi and the context of all |small_phi| and
|big_phi| computation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from . import connectivity
from . import utils
from . import validate
from .cache.content import ContentCache
from .core.tpm import _display
from .core.tpm.factored import FactoredTPM
from .core.tpm.factored import StateSpace
from .core.tpm.joint_distribution import JointTPM
from .direction import Direction
from .display import HIGH
from .display import LOW
from .display import Description
from .display import Displayable
from .display import Row
from .display import Section
from .labels import NodeLabels
from .serializable import Serializable
from .types import ConnectivityMatrix
from .types import Mechanism
from .types import NodeIndices
from .types import Purview

# Module-level cache for potential purviews, keyed on the substrate connectivity
# fingerprint. Shared across every substrate with the same ``cm`` (a single
# registered instance, not per-Substrate, so no registry leak).
_PURVIEW_CACHE = ContentCache("substrate.potential_purviews")


def _coerce_state_to_indices(
    state: tuple[Any, ...],
    state_space: tuple[tuple[Any, ...], ...],
) -> tuple[int, ...]:
    """Convert a state tuple to integer indices via ``state_space`` lookup.

    Each element of ``state`` is either an integer index into the
    corresponding alphabet or a label present in ``state_space[i]``.
    Returns a tuple of integer indices in the same order.

    Args:
        state: The state to coerce — may mix integer indices and labels.
        state_space: Per-node label tuples from :attr:`Substrate.state_space`.

    Returns:
        tuple[int]: Integer-indexed state.

    Raises:
        ValueError: If ``len(state) != len(state_space)`` or any element
            is neither a valid label nor a valid index.
    """
    if len(state) != len(state_space):
        raise ValueError(
            f"state length {len(state)} != state_space length {len(state_space)}"
        )
    indices: list[int] = []
    for i, (s, labels) in enumerate(zip(state, state_space, strict=True)):
        if s in labels:
            indices.append(labels.index(s))
        elif isinstance(s, int) and 0 <= s < len(labels):
            indices.append(s)
        else:
            raise ValueError(
                f"state[{i}] = {s!r} is not in state_space[{i}] = {labels!r} "
                f"and is not a valid index for alphabet size {len(labels)}"
            )
    return tuple(indices)


class Substrate(Displayable, Serializable):
    """A substrate of nodes.

    Represents the substrate under analysis and holds auxiliary data about it.

    The TPM is stored canonically as a :class:`~pyphi.core.tpm.factored.FactoredTPM`
    (per-node-factored conditional). ``substrate.tpm`` returns this
    ``FactoredTPM`` directly. The joint conditional ndarray is available on
    demand via :meth:`joint_tpm`.

    Two mutually exclusive forms of TPM input are accepted — exactly one must
    be supplied:

    * **Joint form** (``tpm=``): a standard joint conditional array.  Accepted
      shapes are 2-D state-by-node ``(s, n)``, 2-D state-by-state ``(s, s)``,
      or multidimensional state-by-node ``[2]*n + [n]``.  Row indices follow
      the little-endian convention (see :ref:`little-endian-convention`).
      Passing a :class:`~pyphi.core.tpm.factored.FactoredTPM` via ``tpm=``
      raises ``ValueError``; use ``marginals=`` or
      :meth:`from_factored` instead.

    * **Factored form** (``marginals=``): a sequence of per-node conditional
      arrays, one per node.  Each factor has shape
      ``(*alphabet_sizes, alphabet_size_i)``.

    Args:
        tpm (np.ndarray): The joint transition probability matrix of the
            substrate (joint form only — see above).

    Keyword Args:
        marginals (sequence of np.ndarray): Per-node conditional arrays
            (factored form). Mutually exclusive with ``tpm=``.
        state_space: The state space for the substrate nodes. Accepts a
            uniform-flat integer alphabet size, a tuple of per-node label
            tuples ``((labels_0, ...), (labels_1, ...), ...)``, or a single
            flat tuple of labels applied uniformly to every node. When
            ``None``, defaults to binary ``(0, 1)`` per node.
        alphabet (int): Shortcut for a uniform integer alphabet of the given
            size — equivalent to ``state_space=tuple(range(alphabet))``.
            Mutually exclusive with ``state_space=``.
        cm (np.ndarray): A square binary adjacency matrix indicating the
            connections between nodes in the substrate. ``cm[i][j] == 1`` means
            that node |i| is connected to node |j| (see :ref:`cm-conventions`).
            **If no connectivity matrix is given, PyPhi assumes that every node
            is connected to every node (including itself)**.
        node_labels (tuple[str] or |NodeLabels|): Human-readable labels for
            each node in the substrate.

    See Also:
        :meth:`from_factored`: Build a ``Substrate`` directly from an existing
        :class:`~pyphi.core.tpm.factored.FactoredTPM`.

    Example:
        In a 3-node binary substrate, ``the_substrate.joint_tpm()[(0, 0, 1)]``
        gives, for each node at |t|, the per-alphabet-value distribution
        given that state at |t-1| was |N_0 = 0, N_1 = 0, N_2 = 1|; e.g.
        ``the_substrate.joint_tpm()[(0, 0, 1)][i, 1]`` is the probability
        that node |i| at |t| takes value 1.
    """

    def __init__(
        self,
        tpm: JointTPM | NDArray[np.float64] | dict[str, Any] | None = None,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        *,
        marginals: Sequence[ArrayLike] | None = None,
        state_space: StateSpace = None,
        alphabet: int | None = None,
    ) -> None:
        if tpm is not None and marginals is not None:
            raise ValueError("Pass tpm= or marginals=, not both")
        if tpm is None and marginals is None:
            raise ValueError("Must pass tpm= (joint) or marginals= (factored)")
        if alphabet is not None and state_space is not None:
            raise ValueError("Pass alphabet= or state_space=, not both")

        if tpm is not None and isinstance(tpm, FactoredTPM):
            raise ValueError(
                "pass FactoredTPM instances via marginals=... or use "
                "Substrate.from_factored(...), not tpm="
            )

        # Translate alphabet= shortcut to a uniform integer state_space.
        if alphabet is not None:
            if alphabet < 2:
                raise ValueError(f"alphabet must be >= 2; got {alphabet}")
            state_space = tuple(range(alphabet))  # type: ignore[assignment]

        if marginals is not None:
            self._factored_tpm = FactoredTPM(factors=marginals, state_space=state_space)
        else:
            arr = self._coerce_joint_array(tpm)
            self._factored_tpm = FactoredTPM.from_joint(arr, state_space=state_space)

        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        # Attach display labels to the canonical TPM so its repr shows node names
        # (the bare TPM otherwise has no knowledge of substrate node names).
        self._factored_tpm._node_labels = tuple(self._node_labels)

        validate.substrate(self)

    @staticmethod
    def _coerce_joint_array(
        tpm: JointTPM | NDArray[np.float64] | dict[str, Any] | Any,
    ) -> NDArray[np.float64]:
        """Coerce supported ``tpm=`` argument forms to a joint ndarray.

        Accepts the same input forms as the legacy ``JointTPM`` constructor
        (2-D state-by-node, 2-D state-by-state, multidimensional
        state-by-node) and routes them through the legacy validator so
        callers don't have to pre-reshape. Also accepts the explicit-alphabet
        shape ``(*alphabet_sizes, n_nodes, max_alphabet)`` produced by
        :meth:`joint_tpm`, which is forwarded as-is to ``FactoredTPM.from_joint``.
        """
        if isinstance(tpm, dict):
            key = "_tpm" if "_tpm" in tpm else "tpm"
            data: Any = tpm[key]
        elif isinstance(tpm, JointTPM):
            return np.asarray(tpm)
        elif hasattr(tpm, "to_array"):
            data = tpm.to_array()  # type: ignore[attr-defined]
        else:
            data = tpm

        arr = np.asarray(data, dtype=np.float64)
        # Explicit-alphabet shape: (*alphabet_sizes, n_nodes, max_alphabet).
        # Detected when ndim == n + 2 with a leading per-axis alphabet block.
        # For binary substrates that's (2, ..., 2, n, 2) where ndim == n + 2
        # and the second-to-last axis equals n.
        if arr.ndim >= 3:
            n_candidate = int(arr.shape[-2])
            if (
                arr.ndim == n_candidate + 2
                and all(s >= 2 for s in arr.shape[:n_candidate])
                and arr.shape[-1] == max(arr.shape[:n_candidate])
            ):
                return arr

        # Otherwise route through the legacy joint TPM so 2-D and
        # state-by-state forms get normalized to multidimensional
        # state-by-node form.
        return np.asarray(JointTPM(data, validate=True), dtype=np.float64)

    @property
    def tpm(self) -> FactoredTPM:
        """The per-node-factored conditional TPM of the substrate."""
        return self._factored_tpm

    @property
    def factored_tpm(self) -> FactoredTPM:
        """Alias for :attr:`tpm` — explicit per-node-factored access."""
        return self._factored_tpm

    @property
    def state_space(self) -> tuple[tuple[Any, ...], ...]:
        """Per-node label tuples, delegated from the underlying FactoredTPM."""
        return self._factored_tpm.state_space

    def joint_tpm(self) -> NDArray[np.float64]:
        """Materialize the joint conditional TPM on demand.

        Returns the explicit-alphabet array
        ``[a_1, ..., a_N, N, max_alphabet]`` for both binary and k-ary
        substrates. Per-row, axis ``-1`` holds factor ``i``'s distribution
        in slots ``[:alphabet_sizes[i]]``; trailing slots are zero when
        alphabets are heterogeneous. Recomputes on every call (no cache);
        callers needing it repeatedly should cache locally.
        """
        return self._factored_tpm.to_joint()

    def _legacy_binary_joint(self) -> NDArray[np.float64]:
        """Binary-only SBN-form rendering of the substrate's TPM.

        Returns an array of shape ``[a_1, ..., a_N, N]`` where each entry
        holds ``P(node_i = 1 | s_t)``. Used by ``to_json`` serialization,
        AC's ``TransitionSystem``, and several tests. Raises
        ``ValueError`` for k-ary substrates because
        SBN-form encodes only ``P(node=1|s_t)``, which has no k-ary
        generalization.
        """
        if not all(a == 2 for a in self._factored_tpm.alphabet_sizes):
            raise ValueError(
                "legacy binary joint shape is binary-only; "
                f"alphabet_sizes={self._factored_tpm.alphabet_sizes}"
            )
        n = self._factored_tpm.n_nodes
        return np.stack(
            [self._factored_tpm.factor(i)[..., 1] for i in range(n)],
            axis=-1,
        )

    @classmethod
    def from_factored(
        cls,
        factored: FactoredTPM,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
    ) -> Substrate:
        """Construct a Substrate from an existing FactoredTPM."""
        s = cls.__new__(cls)
        s._factored_tpm = factored
        s._cm, s._cm_hash = s._build_cm(cm)
        s._node_indices = tuple(range(s.size))
        s._node_labels = NodeLabels(node_labels, s._node_indices)
        validate.substrate(s)
        return s

    @property
    def cm(self) -> ConnectivityMatrix:
        """np.ndarray: The substrate's connectivity matrix.

        A square binary adjacency matrix indicating the connections between
        nodes in the substrate.
        """
        return self._cm

    @cached_property
    def _cm_fingerprint(self) -> bytes:
        """blake2b-256 digest of the connectivity matrix (label-free).

        The exact dependency of :meth:`potential_purviews`, which reads only
        ``cm``; substrates that share a topology but differ in TPM weights
        share this fingerprint and so share their potential-purview results.
        """
        cm = np.ascontiguousarray(self._cm).astype(np.int8, copy=False)
        h = hashlib.blake2b(digest_size=32)
        h.update(repr(cm.shape).encode())
        h.update(cm.tobytes())
        return h.digest()

    @cached_property
    def _math_fingerprint(self) -> bytes:
        """blake2b-256 digest of the full label-free substrate math identity.

        Covers exactly what :meth:`__eq__` compares: alphabet sizes, the factor
        array bytes (``+ 0.0`` folds ``-0.0`` like ``FactoredTPM.__hash__``), and
        the connectivity. Excludes ``node_labels`` / ``state_space`` labels.
        """
        ftpm = self._factored_tpm
        h = hashlib.blake2b(digest_size=32)
        h.update(repr(ftpm.alphabet_sizes).encode())
        for i in range(ftpm.n_nodes):
            h.update((ftpm.factor(i) + 0.0).tobytes())
        h.update(self._cm_fingerprint)
        return h.digest()

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

        Depends only on connectivity, so the result is cached on
        ``_cm_fingerprint`` and shared across every substrate with the same
        ``cm`` (a parameter sweep over a fixed topology reuses it).
        """
        from pyphi.conf import config as _config

        def compute() -> list[Purview]:
            all_purviews = utils.powerset(self._node_indices)
            return irreducible_purviews(self.cm, direction, mechanism, all_purviews)

        fp = self._cm_fingerprint
        _PURVIEW_CACHE.observe(self, fp)
        return _PURVIEW_CACHE.get_or_compute(
            fp,
            (direction, mechanism),
            compute,
            store=_config.infrastructure.cache_potential_purviews,
        )

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
    ) -> tuple[Any, ...]:
        """Return the substrate's complexes as |Complex| objects; see
        :func:`complexes`."""
        return complexes(self, state, candidates=candidates, **kwargs)

    def maximal_complex(
        self,
        state: tuple[int, ...],
        candidates: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the maximal |Complex|; see :func:`maximal_complex`."""
        return maximal_complex(self, state, candidates=candidates, **kwargs)

    def __len__(self) -> int:
        """int: The number of nodes in the substrate."""
        return self.tpm.shape[-1]

    def _describe(self, verbosity: int) -> Description:
        compact = f"Substrate({self.tpm._compact_repr()}, cm={self.cm})"
        if verbosity == LOW:
            return Description(title="Substrate", compact=compact)
        labels = [str(label) for label in self.node_labels]
        alphabet = self.tpm.alphabet_sizes
        state_space = (
            "binary" if all(a == 2 for a in alphabet) else f"alphabets {tuple(alphabet)}"
        )
        sections = [
            Section(
                rows=(
                    Row("Units", ", ".join(labels)),
                    Row("State space", state_space),
                )
            ),
            Section(
                label="Connectivity",
                body=(_display.connectivity_grid(labels, self.cm),),
            ),
        ]
        if verbosity >= HIGH:  # the embedded TPM grid is the heavy part
            sections.append(Section(label="TPM", body=self.tpm.grid_section().body))
        return Description(
            title="Substrate",
            subtitle=f"{self.size} units · {', '.join(labels)}",
            sections=tuple(sections),
            compact=compact,
        )

    def __eq__(self, other: object) -> bool:
        """Return whether this substrate equals the other object.

        Substrates are equal if they have the same TPM and CM.
        """
        return (
            isinstance(other, Substrate)
            and self._factored_tpm == other._factored_tpm
            and np.array_equal(self.cm, other.cm)
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((hash(self._factored_tpm), self._cm_hash))

    def to_networkx(self, connectivity: str = "inferred") -> Any:
        """Return a node-labeled :class:`networkx.DiGraph` of the substrate.

        By default edges are the TPM-inferred causal connectivity; pass
        ``connectivity="declared"`` to use the declared ``cm`` verbatim.
        Requires the ``visualize`` extra (networkx).
        """
        from pyphi import graph

        return graph.substrate_to_networkx(self, connectivity)

    @classmethod
    def from_networkx(
        cls, graph: Any, tpm: Any, *, node_labels: Any = None
    ) -> Substrate:
        """Build a :class:`Substrate` from a networkx DiGraph topology and a TPM.

        The graph supplies connectivity and node order; ``tpm`` supplies the
        dynamics (required). Runs the B19 connectivity validator, so a graph
        that omits a real TPM-implied edge is rejected.
        """
        from pyphi import graph as graph_module

        return graph_module.substrate_from_networkx(graph, tpm, node_labels=node_labels)

    def to_graphml(self, path: str, connectivity: str = "inferred") -> None:
        """Write the substrate graph to a GraphML file (see :meth:`to_networkx`)."""
        from pyphi import graph

        graph.to_graphml(self, path, connectivity)

    def to_adjacency(self, connectivity: str = "inferred") -> Any:
        """Return the connectivity matrix as a node-labeled ``pandas.DataFrame``."""
        from pyphi import graph

        return graph.to_adjacency(self, connectivity)


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

    # TODO: use generator?
    return [purview for purview in purviews if not reducible(purview)]


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
# map_reduce) and then map it over the candidate iterator.


def _resolved_sia(**sia_kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve the formalism's per-system ``sia`` callable and its kwargs.

    Reads the active formalism from ``config.formalism.iit.version`` and,
    under IIT 4.0, fills in ``system_measure`` and ``specification_measure``
    from config when not supplied. Returns a ``(callable, kwargs)`` pair
    safe to hand to :func:`pyphi.parallel.map_reduce`.
    """
    from pyphi.conf import config as _config

    kwargs = dict(sia_kwargs)
    if _config.formalism.iit.version == "IIT_3_0":
        from pyphi.formalism.iit3 import sia as _sia
    else:
        from pyphi.formalism.iit4 import sia as _sia
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.measures.distribution import resolve_system_measure

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
    from pyphi.parallel import map_reduce

    iterable = possible_complexes(substrate, state) if candidates is None else candidates

    sia_fn, map_kwargs = _resolved_sia(**sia_kwargs)
    map_kwargs.setdefault("progress", False)

    pkwargs = _conf.parallel_kwargs(
        _config.infrastructure.parallel_complex_evaluation,
        **(parallel_kwargs or {}),
    )
    result = map_reduce(
        sia_fn,
        iterable,
        total=2 ** len(substrate) - 1,
        map_kwargs=map_kwargs,
        desc="Evaluating complexes",
        **pkwargs,
    )
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


def _exclusion_records(
    accepted: list[Any], sorted_sias: list[Any]
) -> dict[tuple[int, ...], tuple[Any, ...]]:
    """Map each accepted complex (by units) to the ExcludedCandidate records
    it excluded: every irreducible candidate that overlaps it and was not
    itself accepted.

    A candidate that overlaps several accepted complexes appears in each of
    their exclusion sets. Reads only values the cascade already computed.
    """
    from pyphi.models.complex import ExcludedCandidate

    accepted_indices = {tuple(_sia_node_indices(s) or ()) for s in accepted}
    records: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for acc in accepted:
        acc_idx = tuple(_sia_node_indices(acc) or ())
        acc_set = set(acc_idx)
        recs = []
        for cand in sorted_sias:
            cand_idx = tuple(_sia_node_indices(cand) or ())
            if cand_idx == acc_idx or cand_idx in accepted_indices:
                continue
            if acc_set & set(cand_idx):
                recs.append(ExcludedCandidate(cand_idx, float(cand.phi)))
        records[acc_idx] = tuple(recs)
    return records


def complexes(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Return the complexes of the substrate in its current state.

    A complex is a set of units that is a local maximum of |big_phi|
    (|small_phi_s|) — no overlapping candidate has higher |small_phi_s|.
    The returned tuple is non-overlapping (exclusion), ordered by
    |small_phi_s| descending.

    Both formalisms walk SIAs in descending |big_phi| tiers and group
    survivors into overlap cliques per tier. Under IIT 4.0, each
    multi-candidate clique escalates to the Composition cascade (max
    |big_phi|), and ties at Composition fail the exclusion postulate.
    Under IIT 3.0, no further escalation exists (IIT 3.0 provides no
    paper-canonical system-level tie-break); multi-candidate cliques
    are skipped as indeterminate, and the tier walk continues to the
    next group.
    """
    from pyphi import validate
    from pyphi.models.complex import Complex

    sorted_sias = sorted(
        irreducible_sias(substrate, state, candidates, **kwargs), reverse=True
    )
    if not sorted_sias:
        return ()

    if _config_iit_version() == "IIT_3_0":
        accepted = _iit3_exclusion_cascade(sorted_sias, substrate, state)
    else:
        accepted = _substrate_exclusion_cascade(sorted_sias, substrate, state)
    if not accepted:
        return ()

    records = _exclusion_records(accepted, sorted_sias)
    result = tuple(
        Complex(
            sia=sia,
            substrate=substrate,
            is_maximal=(i == 0),
            excluded=records[tuple(_sia_node_indices(sia) or ())],
        )
        for i, sia in enumerate(accepted)
    )
    validate.non_overlapping(result)
    return result


def _config_iit_version() -> str:
    from pyphi.conf import config as _config

    return _config.formalism.iit.version


def _accept(sia: Any, result: list[Any], covered: set[int]) -> None:
    """Add a SIA to the accepted-complex result list and mark its units as covered."""
    indices = _sia_node_indices(sia)
    if indices is None:
        return
    result.append(sia)
    covered.update(indices)


def _phi_groups(sorted_sias: list[Any]) -> Iterable[list[Any]]:
    """Yield contiguous groups of SIAs sharing the same |small_phi_s| value
    (precision-aware), assuming the input is sorted by ``.order_by()``
    descending."""
    from pyphi import utils as _utils

    i = 0
    while i < len(sorted_sias):
        tier_phi = float(sorted_sias[i].phi)
        j = i + 1
        while j < len(sorted_sias) and _utils.eq(float(sorted_sias[j].phi), tier_phi):
            j += 1
        yield sorted_sias[i:j]
        i = j


def _find_overlap_cliques(sias: list[Any]) -> list[list[Any]]:
    """Group SIAs into connected components by unit overlap."""
    n = len(sias)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    units = [set(_sia_node_indices(sia) or ()) for sia in sias]
    for i in range(n):
        for j in range(i + 1, n):
            if units[i] & units[j]:
                union(i, j)

    groups: dict[int, list[Any]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(sias[i])
    return list(groups.values())


def _big_phi_of_sia(sia: Any, substrate: Substrate, state: tuple[int, ...]) -> float:
    """Compute the structure-integrated information |big_phi| of the SIA's
    candidate system. Builds the system from substrate + state + the
    SIA's units and invokes the active formalism's cause-effect-structure
    computation.
    """
    from pyphi.system import System

    indices = _sia_node_indices(sia)
    if indices is None:
        return 0.0
    system = System.from_substrate(substrate, state, indices)
    return float(system.ces().big_phi)


def _resolve_clique_by_big_phi(
    clique: list[Any], substrate: Substrate, state: tuple[int, ...]
) -> Any | None:
    """Pick the |big_phi|-maximal candidate in an overlap clique via the
    substrate-exclusion cascade (Composition escalation). Returns ``None``
    when |big_phi| ties — the exclusion postulate is violated for that
    clique and none of its candidates qualify as a complex.
    """
    from dataclasses import dataclass

    from pyphi import resolve_ties

    @dataclass(frozen=True)
    class _CandidateProxy:
        sia: Any
        big_phi: float

    proxies = [
        _CandidateProxy(sia=sia, big_phi=_big_phi_of_sia(sia, substrate, state))
        for sia in clique
    ]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Composition")
    outcome = resolve_ties.resolve_complex_tie(proxies, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        return outcome.resolved.sia
    return None


def _substrate_exclusion_cascade(
    sorted_sias: list[Any],
    substrate: Substrate,
    state: tuple[int, ...],
) -> list[Any]:
    """Walk SIAs in descending |small_phi_s| tiers, applying the S1
    substrate-exclusion cascade within each tier."""
    result: list[Any] = []
    covered: set[int] = set()

    for tier in _phi_groups(sorted_sias):
        # Within this tier, discard candidates whose units overlap any
        # already-accepted complex.
        survivors = [
            sia for sia in tier if not (set(_sia_node_indices(sia) or ()) & covered)
        ]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            if len(clique) == 1:
                _accept(clique[0], result, covered)
                continue
            winner = _resolve_clique_by_big_phi(clique, substrate, state)
            if winner is not None:
                _accept(winner, result, covered)
    return result


def _resolve_clique_iit3(clique: list[Any]) -> Any | None:
    """Return the unique complex from an IIT 3.0 overlap clique, or None
    when the clique is indeterminate.

    Single-candidate cliques resolve trivially; multi-candidate cliques
    always flag ``UNRESOLVED_WITHIN_BUDGET`` because IIT 3.0 has no
    paper-canonical escalation level. The caller treats None as
    exclusion-postulate failure for the clique.
    """
    from pyphi import resolve_ties

    if len(clique) == 1:
        return clique[0]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Exclusion")
    outcome = resolve_ties.resolve_iit3_complex_tie(clique, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        return outcome.resolved
    return None


def _iit3_exclusion_cascade(
    sorted_sias: list[Any],
    substrate: Any,  # noqa: ARG001 — kept for parity with iit4 cascade signature
    state: Any,  # noqa: ARG001 — kept for parity with iit4 cascade signature
) -> list[Any]:
    """Walk SIAs in descending |big_phi| tiers, applying the IIT 3.0
    cross-subsystem cascade within each overlap clique.

    Within a tier, drop candidates whose units overlap an already-
    accepted complex, then group survivors into overlap cliques.
    Each clique with one member is accepted directly; cliques with
    multiple members run through ``_resolve_clique_iit3`` and are
    skipped when indeterminate.
    """
    result: list[Any] = []
    covered: set[int] = set()
    for tier in _phi_groups(sorted_sias):
        survivors = [
            sia for sia in tier if not (set(_sia_node_indices(sia) or ()) & covered)
        ]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            winner = _resolve_clique_iit3(clique)
            if winner is not None:
                _accept(winner, result, covered)
    return result


def maximal_complex(
    substrate: Substrate,
    state: tuple[int, ...],
    candidates: Iterable[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Return the complex with maximum |big_phi| over the substrate.

    Equivalent to the first element of :func:`complexes`. Returns a
    null-object |Complex| (falsy, with empty units) when no irreducible
    candidate exists.
    """
    from pyphi.models.complex import Complex
    from pyphi.system import System

    found = complexes(substrate, state, candidates, **kwargs)
    if found:
        return found[0]
    # No irreducible candidate; return a null-object Complex over the empty
    # system (falsy, with empty units).
    from pyphi.conf import config as _config

    empty = System.from_substrate(substrate, state, ())
    if _config.formalism.iit.version == "IIT_3_0":
        from pyphi.formalism.iit3 import _null_sia

        null_sia = _null_sia(empty)
    else:
        from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

        null_sia = NullSystemIrreducibilityAnalysis()
    return Complex(sia=null_sia, substrate=substrate, is_maximal=True, excluded=())
