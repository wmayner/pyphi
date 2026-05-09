"""System value type — ``(Substrate, state, node_subset, cut)``.

A System is the unit of analysis for IIT: a substrate evaluated in a specific
state over a specific subset of its nodes, optionally with a cut applied.
Immutable, hashable. The cut is a constructor argument (default
:class:`NullCut`), not a hidden mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

from pyphi import connectivity
from pyphi import utils
from pyphi import validate
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition
from pyphi.substrate import Substrate

from .core.tpm.marginalization import cause_tpm as _marginalize_cause
from .core.tpm.marginalization import effect_tpm as _marginalize_effect

if TYPE_CHECKING:
    from pyphi.types import NodeIndices
    from pyphi.types import State


@dataclass(frozen=True, eq=False)
class System:
    """A substrate evaluated in a specific state over a node subset, with cut."""

    substrate: Substrate
    state: State
    node_indices: NodeIndices = field(default=None)  # type: ignore[assignment]
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        substrate = self.substrate
        validate.state_length(self.state, substrate.size)
        validate.node_states(self.state)
        if self.node_indices is None:
            object.__setattr__(self, "node_indices", tuple(range(substrate.size)))
        else:
            object.__setattr__(
                self,
                "node_indices",
                substrate.node_labels.coerce_to_indices(self.node_indices),
            )
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, substrate.node_labels)
            )
        else:
            cut_idx = getattr(self.cut, "indices", None)
            if cut_idx is not None and set(cut_idx) != set(self.node_indices):
                raise ValueError(
                    f"{self.cut} nodes are not equal to system nodes {self.node_indices}"
                )
        from pyphi.conf import config as _config

        if _config.infrastructure.validate_system_states:
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
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash((self.substrate, self.state, self.node_indices, self.cut))

    def __len__(self) -> int:
        return len(self.node_indices)

    def __str__(self) -> str:
        labels = self.node_labels.coerce_to_labels(self.node_indices)
        return f"System({', '.join(str(label) for label in labels)})"

    def apply_cut(self, cut: SystemPartition) -> System:
        """Return a new System with the given cut applied.

        ``substrate``, ``state``, and ``node_indices`` are unchanged.
        Cached derived properties that don't depend on the cut (``cause_tpm``,
        ``effect_tpm``) are not re-derived in the new instance until first
        access, so the new instance shares the same numerical values.
        """
        from dataclasses import replace

        return replace(self, cut=cut)

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        state: Any,
        nodes: Any | None = None,
        cut: SystemPartition | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> System:
        """Construct a System from a substrate, state, and optional node subset."""
        if nodes is None:
            nodes = tuple(range(substrate.size))
        return cls(
            substrate=substrate,
            state=tuple(state),
            node_indices=tuple(nodes),
            cut=cut,  # type: ignore[arg-type]
        )

    # ---- cached cheap derived properties ----

    @cached_property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        all_indices = set(range(self.substrate.size))
        return tuple(sorted(all_indices - set(self.node_indices)))

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def _typed_tpm(self) -> Any:
        """The typed-kernel ``ExplicitTPM`` used by marginalization."""
        from pyphi.core.tpm.explicit import ExplicitTPM as _TypedTPM

        legacy_tpm = self.substrate.tpm
        if hasattr(legacy_tpm, "to_array"):
            return _TypedTPM(legacy_tpm.to_array())
        return _TypedTPM(legacy_tpm)

    @cached_property
    def cause_tpm(self) -> Any:
        typed = _marginalize_cause(
            self._typed_tpm,  # type: ignore[arg-type]
            self.state,
            self.node_indices,
        )
        return typed._inner if hasattr(typed, "_inner") else typed

    @cached_property
    def effect_tpm(self) -> Any:
        external_state = utils.state_of(self.external_indices, self.state)
        background = dict(zip(self.external_indices, external_state, strict=False))
        typed = _marginalize_effect(
            self._typed_tpm,  # type: ignore[arg-type]
            background,
        )
        return typed._inner if hasattr(typed, "_inner") else typed

    @cached_property
    def proper_effect_tpm(self) -> Any:
        import numpy as np

        return np.asarray(self.effect_tpm.squeeze())[..., list(self.node_indices)]

    @cached_property
    def proper_cause_tpm(self) -> Any:
        import numpy as np

        return np.asarray(self.cause_tpm.squeeze())[..., list(self.node_indices)]

    @cached_property
    def cm(self) -> Any:
        return self.cut.apply_cut(self.substrate.cm)

    @cached_property
    def proper_cm(self) -> Any:
        return connectivity.subadjacency(self.cm, self.node_indices)

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def cut_indices(self) -> NodeIndices:
        return self.node_indices

    @cached_property
    def cut_node_labels(self) -> Any:
        from pyphi.labels import NodeLabels

        if self.cut_indices == self.node_indices:
            return self.node_labels
        labels = self.node_labels.coerce_to_labels(self.cut_indices)
        return NodeLabels(labels, self.cut_indices)

    @cached_property
    def is_cut(self) -> bool:
        return not isinstance(self.cut, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.substrate.size

    @cached_property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        return generate_nodes(
            self.cause_tpm._inner
            if hasattr(self.cause_tpm, "_inner")
            else self.cause_tpm,
            self.effect_tpm._inner
            if hasattr(self.effect_tpm, "_inner")
            else self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def cut_mechanisms(self) -> Any:
        return list(self.cut.all_cut_mechanisms())

    @cached_property
    def _index2node(self) -> dict[int, Any]:
        return {node.index: node for node in self.nodes}

    @cached_property
    def null_concept(self) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.null_concept(self)

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
        self, direction: Any, partition: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.partitioned_repertoire(self, direction, partition, **kwargs)

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
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_info(self, mechanism, purview, **kwargs)

    def effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_info(self, mechanism, purview, **kwargs)

    def cause_effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_effect_info(self, mechanism, purview, **kwargs)

    def intrinsic_information(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.intrinsic_information(self, direction, mechanism, purview, **kwargs)

    # ---- formalism queries ----
    #
    # System-level entry points (``sia``, ``phi_structure``, ``ces``) and the
    # mechanism-level queries used to build them (``find_mip``, ``find_mice``,
    # ``distinction``, ``all_distinctions``, ``evaluate_partition``, …) are
    # exposed as thin convenience methods that dispatch via the active
    # formalism. The same operations live as free functions in
    # :mod:`pyphi.formalism` for callers who prefer that grammar.

    def sia(self, **kwargs: Any) -> Any:
        """Return the system irreducibility analysis under the active formalism."""
        from pyphi.formalism import sia as _sia

        return _sia(self, **kwargs)

    def phi_structure(self, **kwargs: Any) -> Any:
        """Return the IIT 4.0 :class:`PhiStructure` for this system.

        Defined under IIT 4.0 only; the IIT 3.0 analogue is :meth:`ces`.
        """
        from pyphi.formalism.iit4 import phi_structure as _phi_structure

        return _phi_structure(self, **kwargs)

    def ces(self, **kwargs: Any) -> Any:
        """Return the cause-effect structure for this system.

        Under IIT 3.0 this returns concepts; under IIT 4.0 this returns
        distinctions. Both are :class:`CauseEffectStructure` instances.
        """
        from pyphi.conf import config as _config

        formalism_name = _config.formalism.formalism
        if formalism_name == "IIT_3_0":
            from pyphi.formalism.iit3 import ces as _ces

            return _ces(self, **kwargs)
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self, **kwargs)

    def find_mip(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        """Return the minimum information partition for a mechanism over a purview."""
        from pyphi.formalism import find_mip as _find_mip

        return _find_mip(self, direction, mechanism, purview, **kwargs)

    def cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import cause_mip as _cause_mip

        return _cause_mip(self, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import effect_mip as _effect_mip

        return _effect_mip(self, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_cause_mip as _phi_cause_mip

        return _phi_cause_mip(self, mechanism, purview, **kwargs)

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi_effect_mip as _phi_effect_mip

        return _phi_effect_mip(self, mechanism, purview, **kwargs)

    def phi(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        from pyphi.formalism import phi as _phi

        return _phi(self, mechanism, purview, **kwargs)

    def find_mice(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import find_mice as _find_mice

        return _find_mice(self, direction, mechanism, **kwargs)

    def mic(self, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import mic as _mic

        return _mic(self, mechanism, **kwargs)

    def mie(self, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import mie as _mie

        return _mie(self, mechanism, **kwargs)

    def phi_max(self, mechanism: Any) -> float:
        from pyphi.formalism import phi_max as _phi_max

        return _phi_max(self, mechanism)

    def distinction(self, mechanism: Any) -> Any:
        from pyphi.formalism import distinction as _distinction

        return _distinction(self, mechanism)

    def all_distinctions(self, **kwargs: Any) -> Any:
        from pyphi.formalism import all_distinctions as _all_distinctions

        return _all_distinctions(self, **kwargs)

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

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "state": list(self.state),
            "node_indices": list(self.node_indices),
            "cut": self.cut,
        }
