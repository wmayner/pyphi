"""System value type — ``(Substrate, state, node_subset, partition)``.

A System is the unit of analysis for IIT: a substrate evaluated in a specific
state over a specific subset of its nodes, optionally with a partition
applied. Immutable, hashable. The partition is a constructor argument
(default :class:`NullCut`, i.e. no edges severed), not a hidden mode.
"""

from __future__ import annotations

import hashlib
from dataclasses import KW_ONLY
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from pyphi import connectivity
from pyphi import utils
from pyphi import validate
from pyphi.conf.formalism import _VALID_BACKGROUND_CONDITIONING
from pyphi.display import HIGH
from pyphi.display import LOW
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.models.pandas import ToPandasMixin
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import concise_partition
from pyphi.serializable import Serializable
from pyphi.substrate import Substrate
from pyphi.substrate import _coerce_state_to_indices

from .core.tpm.factored import FactoredTPM
from .core.tpm.marginalization import CauseMarginals
from .core.tpm.marginalization import _effect_marginal_factored
from .core.tpm.marginalization import cause_conditioned as _condition_cause
from .core.tpm.marginalization import cause_marginal as _marginalize_cause

if TYPE_CHECKING:
    from pyphi.types import NodeIndices
    from pyphi.types import State


def _given(**options: Any) -> dict[str, Any]:
    """Return the options a caller set, dropping those left at ``None``."""
    return {name: value for name, value in options.items() if value is not None}


# The options the mechanism-level MIP queries accept, besides the parallel
# options of :data:`pyphi.conf.PARALLEL_KWARGS`.
_MIP_OPTIONS = ("partitions", "state", "mechanism_measure", "specification_measure")


def _reject_unknown(caller: str, kwargs: Any, *names: str) -> None:
    from pyphi.conf import reject_unknown_kwargs
    from pyphi.conf._helpers import PARALLEL_KWARGS

    reject_unknown_kwargs(f"System.{caller}", kwargs, [*names, *PARALLEL_KWARGS])


def _resolve_measure(argument: str, measure: Any, resolve: Any) -> Any:
    """Return the measure a name or measure argument stands for."""
    if measure is None or (callable(measure) and hasattr(measure, "name")):
        return measure
    if isinstance(measure, str):
        return resolve(measure)
    raise TypeError(
        f"{argument} must be a measure name or a measure returned by "
        f"{resolve.__name__}(); got {type(measure).__name__}"
    )


# Attributes of the 1.x and IIT 4.0-branch ``Subsystem`` that 2.0 renamed on
# ``System``.
_RENAMED_ATTRIBUTES = {
    "concept": "distinction",
    "null_concept": "null_distinction",
    "apply_cut": "apply_partition",
    "cut": "partition",
    "cause_tpm": "cause_marginal",
    "effect_tpm": "effect_marginal",
    "proper_cause_tpm": "proper_cause_marginal",
    "proper_effect_tpm": "proper_effect_marginal",
}


@dataclass(frozen=True, eq=False, repr=False)
class System(Displayable, ToPandasMixin, Serializable):
    """A substrate evaluated in a specific state over a node subset, with partition.

    The keyword-only ``subset`` argument gives the units of the candidate
    system, by index or label; ``None`` uses every unit of the substrate.
    The resolved units are the ``node_indices`` attribute.

    The ``external_indices`` field specifies which substrate units are the
    background when computing repertoires. When ``None`` (the default), it
    resolves in ``__post_init__`` to ``substrate - node_indices``, the
    background units W = U \\ S, which under the default convention are
    causally marginalized conditional on the current state (Albantakis et al.
    2023, Eqs. 3-4). An explicit override (used by
    ``TransitionSystem`` for actual-causation analysis) may overlap with
    ``node_indices``.

    The ``background_conditioning`` field pins this System to one
    cause-side background convention. ``None`` (the default) resolves
    ``config.formalism.iit.background_conditioning`` at compute time.

    The ``background_state`` field supplies the state at which external
    units are conditioned, when that differs from ``state``. ``None``
    (the default) conditions them at ``state``. Used by
    ``TransitionSystem`` for actual causation, where the cause direction
    evaluates mechanisms against the observed after-state while the
    background is fixed at the observed before-state.
    """

    substrate: Substrate
    state: State
    _: KW_ONLY
    subset: InitVar[NodeIndices | None] = None
    partition: DirectedBipartition = field(default=None)  # type: ignore[assignment]
    external_indices: tuple[int, ...] = field(default=None)  # type: ignore[assignment]
    background_conditioning: str | None = field(default=None)
    background_state: tuple[int, ...] | None = field(default=None)
    node_indices: NodeIndices = field(init=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> System:
        # Point a ported ``Subsystem(network, state, nodes)`` call at the
        # keyword-only ``subset`` instead of a bare argument-count error.
        old = [name for name in ("indices", "nodes", "node_indices") if name in kwargs]
        if len(args) > 2 or old:
            given = (
                f"keyword argument {old[0]!r}" if old else "a third positional argument"
            )
            raise TypeError(
                f"{cls.__name__}() got {given}; pass the units of the candidate "
                "system as the keyword-only subset=..."
            )
        return super().__new__(cls)

    def __post_init__(self, subset: NodeIndices | None) -> None:
        substrate = self.substrate
        if not isinstance(substrate, Substrate):
            hint = (
                "to analyze a System, pass its parts: pyphi.analyze("
                "system.substrate, system.state, subset=system.node_indices)"
                if isinstance(substrate, System)
                else "build one with pyphi.Substrate(tpm, cm=cm)"
            )
            raise TypeError(
                f"expected a Substrate, got {type(substrate).__name__}; {hint}"
            )
        # Coerce label-states to integer indices before value validation.
        coerced = _coerce_state_to_indices(self.state, substrate.state_space)
        object.__setattr__(self, "state", coerced)
        validate.node_states(self.state, substrate.factored_tpm.alphabet_sizes)
        if subset is None:
            object.__setattr__(self, "node_indices", tuple(range(substrate.size)))
        else:
            object.__setattr__(
                self,
                "node_indices",
                substrate.node_labels.coerce_to_indices(subset),
            )
        if self.partition is None:
            object.__setattr__(
                self, "partition", NullCut(self.node_indices, substrate.node_labels)
            )
        else:
            cut_idx = getattr(self.partition, "indices", None)
            if cut_idx is not None and set(cut_idx) != set(self.node_indices):
                raise ValueError(
                    f"{self.partition} nodes are not equal to "
                    f"system nodes {self.node_indices}"
                )
        # Resolve external_indices: None resolves to substrate-minus-system
        # (the IIT 4.0 extended-background convention). An explicit override
        # is validated for shape, then accepted as-is.
        if self.external_indices is None:
            all_indices = set(range(substrate.size))
            object.__setattr__(
                self,
                "external_indices",
                tuple(sorted(all_indices - set(self.node_indices))),
            )
        else:
            ext = tuple(self.external_indices)
            for i in ext:
                if not (0 <= i < substrate.size):
                    raise ValueError(
                        f"external_indices contains out of range index {i}; "
                        f"must satisfy 0 <= i < {substrate.size}"
                    )
            if list(ext) != sorted(ext):
                raise ValueError(f"external_indices must be sorted; got {ext}")
            if len(set(ext)) != len(ext):
                raise ValueError(
                    f"external_indices must not contain duplicates; got {ext}"
                )
            object.__setattr__(self, "external_indices", ext)
        if (
            self.background_conditioning is not None
            and self.background_conditioning not in _VALID_BACKGROUND_CONDITIONING
        ):
            raise ValueError(
                f"background_conditioning={self.background_conditioning!r} "
                f"not in {sorted(_VALID_BACKGROUND_CONDITIONING)} (or None)"
            )
        if self.background_state is not None:
            coerced_bg = _coerce_state_to_indices(
                self.background_state, substrate.state_space
            )
            validate.node_states(coerced_bg, substrate.factored_tpm.alphabet_sizes)
            object.__setattr__(self, "background_state", coerced_bg)
        from pyphi.conf import config as _config

        if _config.infrastructure.validate_system_states:
            self._validate_state_reachable()

    def _validate_state_reachable(self) -> None:
        """Raise if the state is unreachable under this system's dynamics."""
        validate.state_reachable(self)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) < len(other.node_indices)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) <= len(other.node_indices)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) > len(other.node_indices)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return len(self.node_indices) >= len(other.node_indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.state == other.state
            and self.node_indices == other.node_indices
            and self.partition == other.partition
            and self.external_indices == other.external_indices
            and self.background_conditioning == other.background_conditioning
            and self.background_state == other.background_state
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.substrate,
                self.state,
                self.node_indices,
                self.partition,
                self.external_indices,
                self.background_conditioning,
                self.background_state,
            )
        )

    @cached_property
    def _fingerprint(self) -> bytes:
        """blake2b-256 digest of the label-free system math identity.

        Serializes exactly the components :meth:`__eq__` compares: the substrate
        math fingerprint, the index-coerced state, the node and external indices,
        the partition's mathematical content (``indices`` + ``removed_edges``),
        and the pinned background convention. Used as the repertoire kernel
        cache key so distinct-but-equivalent systems share entries.
        """
        h = hashlib.blake2b(digest_size=32)
        h.update(self.substrate._fingerprint)
        h.update(repr(tuple(self.state)).encode())
        h.update(repr(tuple(self.node_indices)).encode())
        h.update(repr(tuple(self.external_indices)).encode())
        h.update(repr(tuple(sorted(self.partition.indices))).encode())
        h.update(repr(sorted(self.partition.removed_edges())).encode())
        h.update(repr(self.background_conditioning).encode())
        h.update(repr(self.background_state).encode())
        return h.digest()

    def __len__(self) -> int:
        return len(self.node_indices)

    def _unit_labels(self) -> tuple[str, ...]:
        """The system's per-unit display labels (node names), in node order."""
        return tuple(
            str(label) for label in self.node_labels.coerce_to_labels(self.node_indices)
        )

    def _to_pandas(self):
        import pandas as pd

        labels = self._unit_labels()
        rows = [
            {"node": idx, "label": label, "state": self.state[idx]}
            for label, idx in zip(labels, self.node_indices, strict=True)
        ]
        return pd.DataFrame(rows)

    def _describe(self, verbosity: int) -> Description:
        from pyphi.core.tpm import _display

        labels = list(self._unit_labels())
        label_str = ", ".join(labels)
        compact = f"System({label_str})"
        if verbosity == LOW:
            return Description(title="System", compact=compact)
        state_str = ", ".join(
            f"{label}={self.state[idx]}"
            for label, idx in zip(labels, self.node_indices, strict=True)
        )
        rows = [
            Row("Units", label_str),
            Row("State", state_str),
            Row("Substrate", f"{self.substrate.size} units"),
        ]
        if self.external_indices:
            ext_labels = self.node_labels.coerce_to_labels(self.external_indices)
            rows.append(
                Row(
                    "Background",
                    ", ".join(
                        f"{label}={self.state[idx]}"
                        for label, idx in zip(
                            ext_labels, self.external_indices, strict=True
                        )
                    ),
                )
            )
        if not isinstance(self.partition, NullCut):
            rows.append(Row("Partition", concise_partition(self.partition)))

        subset_cm = self.substrate.cm[np.ix_(self.node_indices, self.node_indices)]
        sections = [
            Section(rows=tuple(rows)),
            Section(
                label="Connectivity",
                body=(_display.connectivity_grid(labels, subset_cm),),
            ),
        ]
        # The conditioned cause/effect marginals are the heavy part — compute
        # them only at HIGH verbosity.
        if verbosity >= HIGH:
            cause = self.proper_cause_marginal.grid_section()
            effect = self.proper_effect_marginal.grid_section()
            sections.append(Section(label="Cause TPM", body=cause.body, tone="cause"))
            sections.append(Section(label="Effect TPM", body=effect.body, tone="effect"))
        return Description(
            title="System",
            subtitle=f"{len(self.node_indices)} units · {label_str}",
            sections=tuple(sections),
            compact=compact,
        )

    def __getattr__(self, name: str) -> Any:
        # Called only when normal attribute lookup fails.
        message = f"{type(self).__name__!r} object has no attribute {name!r}"
        if name in _RENAMED_ATTRIBUTES:
            message += f". PyPhi 2.0 renamed it `{_RENAMED_ATTRIBUTES[name]}`"
        raise AttributeError(message)

    def apply_partition(self, partition: DirectedBipartition) -> System:
        """Return a new System with the given partition applied.

        ``substrate``, ``state``, and ``node_indices`` are unchanged. The
        cause/effect marginals depend only on those inputs — the cut enters
        downstream through the cut connectivity matrix when node TPMs
        marginalize out severed inputs — so the marginals are materialized
        on this instance and shared with the new one rather than re-derived.

        """
        _ = self.cause_marginal
        _ = self.effect_marginal
        new = replace(self, subset=self.node_indices, partition=partition)
        for name in (
            "_typed_tpm",
            "_cause_marginals",
            "effect_marginal",
            "proper_effect_marginal",
        ):
            if name in self.__dict__:
                new.__dict__[name] = self.__dict__[name]
        return new

    # ---- cached cheap derived properties ----

    @cached_property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    def to_networkx(self, connectivity: str = "inferred") -> Any:
        """Return the substrate's :class:`networkx.DiGraph` with node attributes.

        Each node carries ``state`` and ``in_system``. Edges default to the
        TPM-inferred causal connectivity (``connectivity="declared"`` for the
        declared ``cm``).
        """
        from pyphi import graph

        return graph.system_to_networkx(self, connectivity)

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def _typed_tpm(self) -> Any:
        """The canonical FactoredTPM stored on the substrate."""
        return self.substrate.factored_tpm

    def _resolved_background_conditioning(self) -> str:
        """The cause-side background convention in effect for this System:
        the instance pin when set, else the live config value."""
        if self.background_conditioning is not None:
            return self.background_conditioning
        from pyphi.conf import config as _config

        return _config.formalism.iit.background_conditioning

    @cached_property
    def _cause_marginals(self) -> dict[str, CauseMarginals]:
        """Per-convention cause factors, computed on demand."""
        return {}

    @property
    def _background_reference_state(self) -> tuple[int, ...]:
        """The state at which external units are conditioned: the explicit
        ``background_state`` when set, else ``state``."""
        return self.background_state if self.background_state is not None else self.state

    @property
    def cause_marginal(self) -> CauseMarginals:
        """Per-system-unit cause factors under the active background
        convention: the marginalization of Albantakis et al. (2023), Eq. 4, or
        the background conditioned at its observed state
        (``CONDITION_CURRENT_STATE``).
        """
        convention = self._resolved_background_conditioning()
        if convention not in self._cause_marginals:
            if convention == "CONDITION_CURRENT_STATE":
                external_state = utils.state_of(
                    self.external_indices, self._background_reference_state
                )
                background = dict(
                    zip(self.external_indices, external_state, strict=True)
                )
                marginals = _condition_cause(
                    self._typed_tpm, self.node_indices, background
                )
            else:
                marginals = _marginalize_cause(
                    self._typed_tpm,
                    self.state,
                    self.node_indices,
                )
            self._cause_marginals[convention] = marginals
        return self._cause_marginals[convention]

    @cached_property
    def effect_marginal(self) -> FactoredTPM:
        """Forward TPM conditioned on the external units at their observed state."""
        external_state = utils.state_of(
            self.external_indices, self._background_reference_state
        )
        background = dict(zip(self.external_indices, external_state, strict=False))
        return _effect_marginal_factored(self._typed_tpm, background)

    @cached_property
    def proper_effect_marginal(self) -> FactoredTPM:
        """Effect TPM restricted to system units.

        Per system unit ``i`` in ``node_indices``, the returned FactoredTPM
        carries the forward factor of :attr:`effect_marginal` — the external
        units (``external_indices``) conditioned at the background reference
        state — with all non-system input dims dropped, so the returned
        shape is ``(*system_alphabet, k_i)`` per system output unit.
        Substrate units neither in the system nor external are marginalized
        uniformly (the noise-background convention). The effect-side dual of
        :attr:`proper_cause_marginal`.
        """
        factored = self.effect_marginal
        return FactoredTPM(
            factors=[
                self._restrict_to_system_inputs(factored.factor(i))
                for i in self.node_indices
            ],
            node_labels=self._unit_labels(),
        )

    def _restrict_to_system_inputs(self, factor: np.ndarray) -> np.ndarray:
        """Drop a factor's non-system input dims.

        Non-system axes that are size 1 — conditioned at the background
        state, or marginalized under the weighting of Albantakis et al.
        (2023), Eq. 4 — are squeezed. Non-system axes left free (units
        neither in the system nor external, as under the noise-background
        convention) are marginalized uniformly.
        """
        system = set(self.node_indices)
        drop = tuple(j for j in range(self._typed_tpm.n_nodes) if j not in system)
        if not drop:
            return factor
        free = tuple(j for j in drop if factor.shape[j] != 1)
        if free:
            factor = factor.mean(axis=free, keepdims=True)
        return np.squeeze(factor, axis=drop)

    @property
    def proper_cause_marginal(self) -> FactoredTPM:
        """Cause TPM restricted to system units.

        Per system unit ``i`` in ``node_indices``, the returned FactoredTPM
        carries the cause factor of :attr:`cause_marginal` — background
        handled per the active convention (the marginalization of Albantakis
        et al. (2023), Eq. 4, or the external units conditioned at the
        background reference state) — with all non-system input dims dropped,
        so the returned shape is ``(*system_alphabet, k_i)`` per system output
        unit.
        Substrate units neither in the system nor external are marginalized
        uniformly (the noise-background convention).
        """
        marginals = self.cause_marginal
        return FactoredTPM(
            factors=[
                self._restrict_to_system_inputs(marginals.factor(i))
                for i in self.node_indices
            ],
            node_labels=self._unit_labels(),
        )

    @cached_property
    def cm(self) -> Any:
        return self.partition.apply_cut(self.substrate.cm)

    @cached_property
    def proper_cm(self) -> Any:
        return connectivity.subadjacency(self.cm, self.node_indices)

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def partition_indices(self) -> NodeIndices:
        return self.node_indices

    @cached_property
    def partition_node_labels(self) -> Any:
        from pyphi.labels import NodeLabels

        if self.partition_indices == self.node_indices:
            return self.node_labels
        labels = self.node_labels.coerce_to_labels(self.partition_indices)
        return NodeLabels(labels, self.partition_indices)

    @cached_property
    def is_partitioned(self) -> bool:
        return not isinstance(self.partition, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.substrate.size

    @cached_property
    def _nodes_by_convention(self) -> dict[str, Any]:
        return {}

    @property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        convention = self._resolved_background_conditioning()
        if convention not in self._nodes_by_convention:
            self._nodes_by_convention[convention] = generate_nodes(
                self.cause_marginal,
                self.effect_marginal,
                self.cm,
                self.state,
                self.node_indices,
                self.node_labels,
            )
        return self._nodes_by_convention[convention]

    @cached_property
    def partitioned_mechanisms(self) -> Any:
        return list(self.partition.all_cut_mechanisms())

    @cached_property
    def _index2node_by_convention(self) -> dict[str, dict[int, Any]]:
        return {}

    @property
    def _index2node(self) -> dict[int, Any]:
        convention = self._resolved_background_conditioning()
        if convention not in self._index2node_by_convention:
            self._index2node_by_convention[convention] = {
                node.index: node for node in self.nodes
            }
        return self._index2node_by_convention[convention]

    @cached_property
    def null_distinction(self) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.null_distinction(self)

    # ---- repertoire algebra proxies ----

    def cause_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_repertoire(self, mechanism, purview, **kwargs)

    def effect_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_repertoire(self, mechanism, purview, **kwargs)

    def repertoire(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.repertoire(self, direction, mechanism, purview, **kwargs)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction: Any, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(
        self,
        direction: Any,
        partition: Any,
        *,
        mechanism_measure: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        mechanism_measure = _resolve_measure(
            "mechanism_measure", mechanism_measure, resolve_mechanism_measure
        )
        return ra.partitioned_repertoire(
            self, direction, partition, mechanism_measure=mechanism_measure, **kwargs
        )

    def expand_repertoire(
        self,
        direction: Any,
        repertoire_array: Any,
        new_purview: Any | None = None,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_repertoire(
            self, direction, repertoire_array, new_purview=new_purview
        )

    def expand_cause_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_cause_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def expand_effect_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_effect_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def forward_cause_repertoire(
        self, mechanism: Any, purview: Any, purview_state: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(
        self, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_repertoire(self, mechanism, purview, **kwargs)

    def forward_repertoire(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_repertoire(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(self, mechanism, purview)

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(self, mechanism, purview)

    def unconstrained_forward_repertoire(
        self, direction: Any, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(self, direction, mechanism, purview)

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kwargs: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_probability(
            self, direction, mechanism, purview, purview_state, **kwargs
        )

    # ---- info / phi proxies ----

    def cause_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(
                _config.formalism.iit.mechanism_phi_measure,
                self.substrate.factored_tpm.alphabet_sizes,
            ),
        )
        return ra.cause_info(self, mechanism, purview, **kwargs)

    def effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(
                _config.formalism.iit.mechanism_phi_measure,
                self.substrate.factored_tpm.alphabet_sizes,
            ),
        )
        return ra.effect_info(self, mechanism, purview, **kwargs)

    def cause_effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.conf import config as _config
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        kwargs.setdefault(
            "repertoire_distance",
            resolve_mechanism_measure(
                _config.formalism.iit.mechanism_phi_measure,
                self.substrate.factored_tpm.alphabet_sizes,
            ),
        )
        return ra.cause_effect_info(self, mechanism, purview, **kwargs)

    def intrinsic_specification(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        *,
        specification_measure: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the state a mechanism specifies over a purview.

        The specified state is the purview state of maximal intrinsic
        specification (Mayner et al. 2026, Eq. 12).

        Parameters
        ----------
        direction : Direction
            Cause or effect.
        mechanism : tuple[int, ...]
            The mechanism's node indices.
        purview : tuple[int, ...]
            The purview's node indices.
        specification_measure : str or measure
            The measure of intrinsic specification maximized over the purview
            states: a name that ``config.formalism.iit.specification_measure``
            accepts, or the measure
            :func:`~pyphi.measures.distribution.resolve_mechanism_measure`
            returns for one.
        **kwargs
            Passed to
            :func:`pyphi.core.repertoire_algebra.intrinsic_specification`;
            ``states`` restricts the candidate purview states.

        Returns
        -------
        StateSpecification
            The specified state and its intrinsic specification. States that
            tie with it within ``config.numerics.precision`` are on its
            ``ties``.
        """
        from pyphi.core import repertoire_algebra as ra
        from pyphi.measures.distribution import resolve_mechanism_measure

        specification_measure = _resolve_measure(
            "specification_measure", specification_measure, resolve_mechanism_measure
        )
        spec = ra.intrinsic_specification(
            self,
            direction,
            mechanism,
            purview,
            specification_measure=specification_measure,
            **kwargs,
        )
        # Stamp display labels here, at the labeled layer; the kernel that
        # builds the specification stays label-free. Cover the tie family so
        # tied-state cards are labeled too.
        labels = self.node_labels
        if labels is not None and hasattr(spec, "node_labels"):
            for tied in {spec, *getattr(spec, "ties", ())}:
                if tied.node_labels is None:
                    tied.node_labels = labels
        return spec

    # ---- formalism queries ----
    #
    # System-level entry points (``sia``, ``ces``, ``distinctions``) and the
    # mechanism-level queries used to build them (``find_mip``, ``find_mice``,
    # ``distinction``, ``all_distinctions``, ``evaluate_partition``, …) are
    # exposed as thin convenience methods that dispatch via the active
    # formalism. The same operations live as free functions in
    # :mod:`pyphi.formalism` for callers who prefer that grammar.

    def sia(
        self,
        *,
        formalism: Any = None,
        system_state: Any = None,
        partitions: Any = None,
        partition_scheme: str | None = None,
        directions: Any = None,
        system_measure: Any = None,
        specification_measure: Any = None,
        parallel: bool | None = None,
        progress: bool | None = None,
    ) -> Any:
        """Return the system irreducibility analysis of this system.

        Parameters
        ----------
        formalism : str, mapping, or None, optional
            The version of IIT, selected as by :func:`pyphi.analyze`, for
            this call only. ``None`` uses the active configuration.
        system_state : SystemStateSpecification, optional
            The system's specified state, when already computed. It must be
            the specification the analysis would itself compute, since
            nothing checks it. A call that supplies it bypasses the disk
            result cache, so a non-canonical state can never be stored as,
            or served for, the plain ``sia()`` result.
        partitions : iterable, optional
            The system partitions to evaluate, in place of those the
            partition scheme generates.
        partition_scheme : str, optional
            The system partition scheme. ``None`` uses
            ``config.formalism.iit.system_partition_scheme``.
        directions : iterable of Direction, optional
            The directions whose specified states are computed. ``None``
            computes both.
        system_measure : str or measure, optional
            The system φ measure, in place of
            ``config.formalism.iit.system_phi_measure``: a name that option
            accepts, such as ``"GENERALIZED_INTRINSIC_DIFFERENCE"``, or the
            measure :func:`~pyphi.measures.distribution.resolve_system_measure`
            returns for one.
        specification_measure : str or measure, optional
            The measure of intrinsic specification, in place of
            ``config.formalism.iit.specification_measure``: a name that option
            accepts, or the measure
            :func:`~pyphi.measures.distribution.resolve_mechanism_measure`
            returns for one.
        parallel : bool or None, optional
            Evaluate the partitions in parallel. ``None`` defers to
            ``config.infrastructure.parallel_partition_evaluation``.
        progress : bool or None, optional
            Show a progress bar. ``None`` defers to the configuration.

        Returns
        -------
        SystemIrreducibilityAnalysis
            The analysis under the selected formalism.

        Raises
        ------
        TypeError
            Under IIT 3.0, if any option other than ``parallel`` or
            ``progress`` is given: that version has no specified states,
            partition schemes or system measures to set. Also if a measure
            is neither a name nor a measure object.
        ValueError
            If a measure name is not a registered measure.
        """
        from pyphi.cache.disk import maybe_disk_cached
        from pyphi.conf import presets
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.measures.distribution import resolve_system_measure

        system_measure = _resolve_measure(
            "system_measure", system_measure, resolve_system_measure
        )
        specification_measure = _resolve_measure(
            "specification_measure", specification_measure, resolve_mechanism_measure
        )
        options = _given(
            system_state=system_state,
            partitions=partitions,
            partition_scheme=partition_scheme,
            directions=directions,
            system_measure=system_measure,
            specification_measure=specification_measure,
            parallel=parallel,
            progress=progress,
        )

        def _compute() -> Any:
            from pyphi.conf import config as _config
            from pyphi.formalism import sia as _sia
            from pyphi.measures.distribution import resolve_mechanism_measure
            from pyphi.measures.distribution import resolve_system_measure

            call_kwargs = dict(options)
            if _config.formalism.iit.version != "IIT_3_0":
                call_kwargs.setdefault(
                    "system_measure",
                    resolve_system_measure(_config.formalism.iit.system_phi_measure),
                )
                call_kwargs.setdefault(
                    "specification_measure",
                    resolve_mechanism_measure(
                        _config.formalism.iit.specification_measure
                    ),
                )
            return _sia(self, **call_kwargs)

        with presets.use(formalism):
            return maybe_disk_cached(self, "sia", options, _compute)

    def ces(
        self,
        *,
        formalism: Any = None,
        sia: Any = None,
        distinctions: Any = None,
    ) -> Any:
        """Return the cause-effect structure of this system (Eq. 57).

        The result is a :class:`CauseEffectStructure`: the distinctions plus
        their relations.

        Parameters
        ----------
        formalism : str, mapping, or None, optional
            The version of IIT, selected as by :func:`pyphi.analyze`, for
            this call only. ``None`` uses the active configuration.
        sia : SystemIrreducibilityAnalysis, optional
            The system's analysis, when already computed. Its specified
            state resolves the distinctions' congruence.
        distinctions : Distinctions, optional
            The system's distinctions, when already computed.

        Returns
        -------
        CauseEffectStructure
            The cause-effect structure under the selected formalism.

        Notes
        -----
        For results computed under an earlier version of IIT (see
        :doc:`/howto/earlier-versions`): IIT 3.0 has no relations, so under it
        the CES holds the distinctions and an empty
        :class:`~pyphi.relations.NullRelations`.
        """
        from pyphi.cache.disk import maybe_disk_cached
        from pyphi.conf import presets

        options = _given(sia=sia, distinctions=distinctions)

        def _compute() -> Any:
            from pyphi.conf import config as _config

            if _config.formalism.iit.version == "IIT_3_0":
                from pyphi.formalism.iit3 import ces as _iit3_ces

                return _iit3_ces(self, **options)
            from pyphi.formalism.iit4 import ces as _ces
            from pyphi.measures.distribution import resolve_mechanism_measure
            from pyphi.measures.distribution import resolve_system_measure

            return _ces(
                self,
                system_measure=resolve_system_measure(
                    _config.formalism.iit.system_phi_measure
                ),
                specification_measure=resolve_mechanism_measure(
                    _config.formalism.iit.specification_measure
                ),
                **options,
            )

        with presets.use(formalism):
            return maybe_disk_cached(self, "ces", options, _compute)

    def distinctions(self, *, congruent: bool = False) -> Any:
        """Return the :class:`Distinctions` of this system.

        The set of irreducible cause-effect distinctions specified by
        mechanisms in the system, without the relations that bind them.

        Parameters
        ----------
        congruent : bool
            Filter to the distinctions congruent with the system's specified
            state, as the cause-effect structure does, but without the
            system-partition search. The result is a
            :class:`~pyphi.models.distinctions.ResolvedDistinctions` equal to
            ``ces().distinctions`` when the specified state is untied, and
            the unfiltered
            :class:`~pyphi.models.distinctions.UnresolvedDistinctions` when it
            ties (see
            :func:`~pyphi.formalism.iit4.congruent_distinctions`).

        Notes
        -----
        Congruence filtering can remove every distinction — a system whose
        mechanisms specify states incongruent with the system's own has a
        nonempty unfiltered set and an empty cause-effect structure. The
        unfiltered count and Σφ_d are therefore upper bounds, not estimates.

        For results computed under an earlier version of IIT (see
        :doc:`/howto/earlier-versions`): IIT 3.0 has no tied specified states,
        so its distinctions are congruent as computed.
        """
        from pyphi.conf import config as _config

        formalism_name = _config.formalism.iit.version
        if formalism_name == "IIT_3_0":
            from pyphi.formalism.iit3 import (
                _compute_distinctions as _distinctions,  # pyright: ignore[reportPrivateUsage]
            )
            from pyphi.models.distinctions import ResolvedDistinctions

            distinctions = _distinctions(self)
            return ResolvedDistinctions(distinctions) if congruent else distinctions
        if congruent:
            from pyphi.formalism.iit4 import congruent_distinctions
            from pyphi.measures.distribution import resolve_mechanism_measure

            return congruent_distinctions(
                self,
                specification_measure=resolve_mechanism_measure(
                    _config.formalism.iit.specification_measure
                ),
            )
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self)

    def find_mip(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        """Return the minimum partition for a mechanism over a purview.

        Resolves mechanism- and specification-level measures from config at
        the public boundary so the active formalism's MIP search is never
        called without explicit measures in normal flow. Explicit
        ``mechanism_measure``/``specification_measure`` kwargs override.
        """
        from pyphi.conf import config as _config
        from pyphi.formalism import find_mip as _find_mip
        from pyphi.measures.distribution import resolve_mechanism_measure

        _reject_unknown("find_mip", kwargs, *_MIP_OPTIONS)
        if _config.formalism.iit.version != "IIT_3_0":
            alphabet_sizes = self.substrate.factored_tpm.alphabet_sizes
            kwargs.setdefault(
                "mechanism_measure",
                resolve_mechanism_measure(
                    _config.formalism.iit.mechanism_phi_measure, alphabet_sizes
                ),
            )
            kwargs.setdefault(
                "specification_measure",
                resolve_mechanism_measure(_config.formalism.iit.specification_measure),
            )
        return _find_mip(self, direction, mechanism, purview, **kwargs)

    def cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import cause_mip as _cause_mip

        _reject_unknown("cause_mip", kwargs, *_MIP_OPTIONS)
        return _cause_mip(self, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import effect_mip as _effect_mip

        _reject_unknown("effect_mip", kwargs, *_MIP_OPTIONS)
        return _effect_mip(self, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_cause_mip as _phi_cause_mip

        _reject_unknown("phi_cause_mip", kwargs, *_MIP_OPTIONS)
        return _phi_cause_mip(self, mechanism, purview, **kwargs)

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_effect_mip as _phi_effect_mip

        _reject_unknown("phi_effect_mip", kwargs, *_MIP_OPTIONS)
        return _phi_effect_mip(self, mechanism, purview, **kwargs)

    def phi(self, mechanism: Any, purview: Any) -> float:
        """Return the φ of a mechanism over one purview.

        φ is the smaller of the cause and effect φ at their minimum
        partitions, both evaluated over ``purview``.

        Parameters
        ----------
        mechanism : tuple[int, ...]
            The mechanism's node indices.
        purview : tuple[int, ...]
            The purview's node indices.

        Returns
        -------
        float
            The mechanism's φ over ``purview``.
        """
        from pyphi.formalism import phi as _phi

        return _phi(self, mechanism, purview)

    def find_mice(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import find_mice as _find_mice

        _reject_unknown("find_mice", kwargs, "purviews")
        return _find_mice(self, direction, mechanism, **kwargs)

    def mic(self, mechanism: Any, *, purviews: Any = None) -> Any:
        """Return the maximally irreducible cause of a mechanism.

        Parameters
        ----------
        mechanism : tuple[int, ...]
            The mechanism's node indices.
        purviews : iterable of tuple[int, ...], optional
            The candidate purviews to search. Candidates outside this system
            or reducible by its connectivity are skipped. ``None`` searches
            every purview of the system that the connectivity does not rule
            out.

        Returns
        -------
        MaximallyIrreducibleCause
            The cause purview of maximal φ, with its minimum partition
            and specified state.
        """
        from pyphi.formalism import mic as _mic

        return _mic(self, mechanism, purviews=purviews)

    def mie(self, mechanism: Any, *, purviews: Any = None) -> Any:
        """Return the maximally irreducible effect of a mechanism.

        Parameters
        ----------
        mechanism : tuple[int, ...]
            The mechanism's node indices.
        purviews : iterable of tuple[int, ...], optional
            The candidate purviews to search. Candidates outside this system
            or reducible by its connectivity are skipped. ``None`` searches
            every purview of the system that the connectivity does not rule
            out.

        Returns
        -------
        MaximallyIrreducibleEffect
            The effect purview of maximal φ, with its minimum partition
            and specified state.
        """
        from pyphi.formalism import mie as _mie

        return _mie(self, mechanism, purviews=purviews)

    def phi_max(self, mechanism: Any) -> float:
        from pyphi.formalism import phi_max as _phi_max

        return _phi_max(self, mechanism)

    def distinction(self, mechanism: Any) -> Any:
        from pyphi.formalism import distinction as _distinction

        return _distinction(self, mechanism)

    def all_distinctions(self) -> Any:
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self)

    def evaluate_partition(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.formalism import evaluate_partition as _evaluate_partition

        return _evaluate_partition(
            self, direction, mechanism, purview, partition, **kwargs
        )

    def potential_purviews(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.potential_purviews(self, direction, mechanism, **kwargs)

    def indices2nodes(self, indices: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.indices2nodes(self, indices)

    # ---- cache surface + serialization ----

    def cache_info(self) -> dict[str, Any]:
        from pyphi.core import repertoire_algebra as ra

        return ra.cache_info()

    def clear_caches(self) -> None:
        from pyphi.core import repertoire_algebra as ra

        ra.clear_caches(self)


def _subset(self: System) -> NodeIndices:
    """The units of the system, as passed to ``subset``: ``node_indices``."""
    return self.node_indices


# ``subset`` is the constructor's spelling of ``node_indices``. Reading it
# back gives the resolved units, so ``dataclasses.replace`` and
# ``copy.replace`` keep a system's units. This replaces the InitVar default
# the dataclass leaves behind as a class attribute.
System.subset = property(_subset)  # pyright: ignore[reportAttributeAccessIssue]
