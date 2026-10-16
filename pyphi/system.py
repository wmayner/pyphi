"""System value type — ``(Substrate, state, node_subset, partition)``.

A System is the unit of analysis for IIT: a substrate evaluated in a specific
state over a specific subset of its nodes, optionally with a partition
applied. Immutable, hashable. The partition is a constructor argument
(default :class:`NullCut`, i.e. no edges severed), not a hidden mode.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import KW_ONLY
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from functools import wraps
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
# options below.
_MIP_OPTIONS = ("partitions", "state", "mechanism_measure", "specification_measure")

# The parallel options the mechanism-level queries accept. The other options
# of :func:`pyphi.parallel.map_reduce` (reduce and short-circuit hooks) can
# change the result, so they are not part of the public signature.
_PARALLEL_OPTIONS = ("parallel", "progress", "chunksize", "sequential_threshold")


def _reject_unknown(caller: str, kwargs: Any, *names: str) -> None:
    from pyphi.conf import reject_unknown_kwargs

    reject_unknown_kwargs(f"System.{caller}", kwargs, [*names, *_PARALLEL_OPTIONS])


def _check_direction(direction: Any) -> Any:
    """Return ``direction`` as ``Direction.CAUSE`` or ``Direction.EFFECT``,
    which may also be given by value (``0`` or ``1``)."""
    from pyphi.direction import Direction

    try:
        resolved = None if isinstance(direction, bool) else Direction(direction)
    except (ValueError, TypeError):
        resolved = None
    if resolved not in Direction.both():
        raise ValueError(
            f"direction must be Direction.CAUSE or Direction.EFFECT; got {direction!r}"
        )
    return resolved


def _mip_options(system: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Check and resolve the options of a mechanism-level MIP query.

    Measure names resolve to measures, a measure the active version of IIT
    does not define raises :class:`~pyphi.formalism.base.MeasureNotCompatibleError`,
    and ``state`` must be a specified state.
    """
    from pyphi.conf import config as _config
    from pyphi.formalism.base import FORMALISM_REGISTRY
    from pyphi.formalism.base import check_measure_compatible
    from pyphi.formalism.base import check_specification_measure_compatible
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.models.state_specification import StateSpecification

    formalism = FORMALISM_REGISTRY[_config.formalism.iit.version]
    alphabet_sizes = system.substrate.factored_tpm.alphabet_sizes
    for name, resolve, check in (
        (
            "mechanism_measure",
            lambda m: resolve_mechanism_measure(m, alphabet_sizes),
            check_measure_compatible,
        ),
        (
            "specification_measure",
            resolve_mechanism_measure,
            check_specification_measure_compatible,
        ),
    ):
        if kwargs.get(name) is None:
            continue
        measure = _resolve_measure(name, kwargs[name], resolve)
        measure_name = getattr(measure, "name", None)
        if measure_name is not None:
            check(formalism, measure_name)
        kwargs[name] = measure
    state = kwargs.get("state")
    if state is not None and not isinstance(state, StateSpecification):
        raise TypeError(
            "state must be a specified state (a StateSpecification, such as a "
            "MIP's specified_state); got "
            f"{type(state).__name__} {state!r}"
        )
    return kwargs


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
    "network": "substrate",
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
        """Raise if the state is unreachable under this system's dynamics, and
        record the background conditioning it was checked under."""
        validate.state_reachable(self)
        object.__setattr__(
            self, "_checked_conditioning", self._resolved_background_conditioning()
        )

    def _check_precomputed(self, *, sia: Any = None, distinctions: Any = None) -> None:
        """Raise ``ValueError`` if a precomputed analysis or distinctions
        name other units or another state than this system."""
        if sia is not None:
            units = getattr(sia, "node_indices", None)
            if units is not None and tuple(units) != self.node_indices:
                raise ValueError(
                    f"sia was computed for units {tuple(units)}, not this "
                    f"system's units {self.node_indices}"
                )
            state = getattr(sia, "current_state", None)
            # A whole-substrate state is accepted for analyses saved by
            # releases that stored it under IIT 3.0.
            if state is not None and tuple(state) not in (
                self.proper_state,
                self.state,
            ):
                raise ValueError(
                    f"sia was computed in state {tuple(state)}, not this "
                    f"system's state {self.proper_state}"
                )
        if distinctions is not None:
            units = set(self.node_indices)
            for distinction in distinctions:
                mechanism = tuple(distinction.mechanism)
                purviews = (distinction.cause_purview, distinction.effect_purview)
                if not units.issuperset(mechanism) or any(
                    p is not None and not units.issuperset(p) for p in purviews
                ):
                    raise ValueError(
                        f"the distinction of mechanism {mechanism} lies outside "
                        f"this system's units {self.node_indices}"
                    )
                mechanism_state = getattr(distinction.cause, "mechanism_state", None)
                expected = utils.state_of(mechanism, self.state)
                if mechanism_state is not None and tuple(mechanism_state) != expected:
                    raise ValueError(
                        f"the distinction of mechanism {mechanism} was computed "
                        f"in mechanism state {tuple(mechanism_state)}, not this "
                        f"system's {expected}"
                    )

    def _recheck_state(self) -> None:
        """Check the state again when the background conditioning now in
        effect differs from the one it was checked under.

        Reachability depends on the background conditioning, which the
        active formalism sets unless the System pins it. A System built with
        ``validate_system_states`` off was never checked and is not checked
        here.
        """
        from pyphi.conf import config as _config

        checked = getattr(self, "_checked_conditioning", None)
        if (
            checked is not None
            and _config.infrastructure.validate_system_states
            and checked != self._resolved_background_conditioning()
        ):
            self._validate_state_reachable()

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
        """Return the cause repertoire of a mechanism over a purview.

        The distribution over the purview's previous states that the
        mechanism in its current state specifies (Albantakis et al. 2023,
        Eq. 33).

        Parameters
        ----------
        mechanism : tuple of int or str
            The mechanism's units, as indices or labels, in any order.
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        **kwargs
            ``mechanism_state``: the mechanism's state, in the order of
            ``mechanism``, instead of the system's.

        Returns
        -------
        numpy.ndarray
            The repertoire, with one axis per substrate unit (size 1 off the
            purview), so axes follow ascending unit order whatever order the
            purview was given in.
        """
        from pyphi.core import repertoire_algebra as ra

        mechanism, state = self._coerce_with_states(
            mechanism, kwargs.pop("mechanism_state", None)
        )
        if state is not None:
            kwargs["mechanism_state"] = state
        return ra.cause_repertoire(
            self, mechanism, self._coerce_units(purview), **kwargs
        )

    def effect_repertoire(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        """Return the effect repertoire of a mechanism over a purview.

        The distribution over the purview's next states that the mechanism
        in its current state specifies (Albantakis et al. 2023, Eq. 29).

        Parameters
        ----------
        mechanism : tuple of int or str
            The mechanism's units, as indices or labels, in any order.
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        **kwargs
            ``mechanism_state``: the mechanism's state, in the order of
            ``mechanism``, instead of the system's.

        Returns
        -------
        numpy.ndarray
            The repertoire, with one axis per substrate unit (size 1 off the
            purview), so axes follow ascending unit order whatever order the
            purview was given in.
        """
        from pyphi.core import repertoire_algebra as ra

        mechanism, state = self._coerce_with_states(
            mechanism, kwargs.pop("mechanism_state", None)
        )
        if state is not None:
            kwargs["mechanism_state"] = state
        return ra.effect_repertoire(
            self, mechanism, self._coerce_units(purview), **kwargs
        )

    def repertoire(
        self, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        """Return :meth:`cause_repertoire` or :meth:`effect_repertoire`,
        by ``direction``."""
        from pyphi.direction import Direction

        if direction == Direction.CAUSE:
            return self.cause_repertoire(mechanism, purview, **kwargs)
        if direction == Direction.EFFECT:
            return self.effect_repertoire(mechanism, purview, **kwargs)
        _check_direction(direction)
        raise AssertionError("unreachable")

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        """Return the cause repertoire of the empty mechanism over a purview.

        Parameters
        ----------
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        """
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, self._coerce_units(purview))

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        """Return the effect repertoire of the empty mechanism over a purview.

        Parameters
        ----------
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        """
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, self._coerce_units(purview))

    def unconstrained_repertoire(self, direction: Any, purview: Any) -> Any:
        """Return the unconstrained cause or effect repertoire, by
        ``direction``."""
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, self._coerce_units(purview))

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
        """Return the forward cause repertoire of a mechanism over a purview.

        For each purview state, the probability that it leads to the
        mechanism's current state.

        Parameters
        ----------
        mechanism : tuple of int or str
            The mechanism's units, as indices or labels, in any order.
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        purview_state : tuple of int, optional
            One purview state, in the order of ``purview``, to evaluate
            alone; the other entries of the result are NaN.
        """
        from pyphi.core import repertoire_algebra as ra

        mechanism = self._coerce_units(mechanism)
        purview, purview_state = self._coerce_with_states(purview, purview_state)
        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(
        self, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        """Return the forward effect repertoire of a mechanism over a purview:
        the effect repertoire. Takes the arguments of
        :meth:`effect_repertoire`."""
        from pyphi.core import repertoire_algebra as ra

        mechanism, state = self._coerce_with_states(
            mechanism, kwargs.pop("mechanism_state", None)
        )
        if state is not None:
            kwargs["mechanism_state"] = state
        return ra.forward_effect_repertoire(
            self, mechanism, self._coerce_units(purview), **kwargs
        )

    def forward_repertoire(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return :meth:`forward_cause_repertoire` or
        :meth:`forward_effect_repertoire`, by ``direction``."""
        from pyphi.direction import Direction

        if direction == Direction.CAUSE:
            return self.forward_cause_repertoire(mechanism, purview, purview_state)
        if direction == Direction.EFFECT:
            return self.forward_effect_repertoire(mechanism, purview, **kwargs)
        _check_direction(direction)
        raise AssertionError("unreachable")

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        """Return the forward cause repertoire with the mechanism's state
        unconstrained (Albantakis et al. 2023, Eq. 32). Units are given as
        in :meth:`cause_repertoire`."""
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(
            self, self._coerce_units(mechanism), self._coerce_units(purview)
        )

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        """Return the forward effect repertoire with the mechanism's state
        unconstrained. Units are given as in :meth:`effect_repertoire`."""
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(
            self, self._coerce_units(mechanism), self._coerce_units(purview)
        )

    def unconstrained_forward_repertoire(
        self, direction: Any, mechanism: Any, purview: Any
    ) -> Any:
        """Return the unconstrained forward cause or effect repertoire, by
        ``direction``."""
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(
            self, direction, self._coerce_units(mechanism), self._coerce_units(purview)
        )

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        """Return the probability that a purview state leads to the
        mechanism's state.

        Parameters
        ----------
        mechanism : tuple of int or str
            The mechanism's units, as indices or labels, in any order.
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        purview_state : tuple of int
            The purview state, in the order of ``purview``.
        mechanism_state : tuple of int, optional
            The mechanism's state, in the order of ``mechanism``; the
            system's by default.
        """
        from pyphi.core import repertoire_algebra as ra

        mechanism, mechanism_state = self._coerce_with_states(mechanism, mechanism_state)
        purview, purview_state = self._coerce_with_states(purview, purview_state)
        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        """Return the probability that the mechanism's state leads to a
        purview state.

        Parameters
        ----------
        mechanism : tuple of int or str
            The mechanism's units, as indices or labels, in any order.
        purview : tuple of int or str
            The purview's units, as indices or labels, in any order.
        purview_state : tuple of int
            The purview state, in the order of ``purview``.
        """
        from pyphi.core import repertoire_algebra as ra

        purview, purview_state = self._coerce_with_states(purview, purview_state)
        return ra.forward_effect_probability(
            self, self._coerce_units(mechanism), purview, purview_state
        )

    def forward_probability(
        self,
        direction: Any,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kwargs: Any,
    ) -> float:
        """Return :meth:`forward_cause_probability` or
        :meth:`forward_effect_probability`, by ``direction``."""
        from pyphi.direction import Direction

        if direction == Direction.CAUSE:
            return self.forward_cause_probability(
                mechanism, purview, purview_state, **kwargs
            )
        if direction == Direction.EFFECT:
            return self.forward_effect_probability(mechanism, purview, purview_state)
        _check_direction(direction)
        raise AssertionError("unreachable")

    # ---- info / phi proxies ----

    def _info_kwargs(self, caller: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The options of an information query, with the measure resolved.

        Any option but ``repertoire_distance`` and ``state`` raises
        ``TypeError``. A measure name resolves to its measure. Without an
        explicit measure the configured ``mechanism_phi_measure`` is used,
        which only IIT 3.0 defines for these quantities; under another
        version of IIT that raises ``TypeError``.
        """
        from pyphi.conf import config as _config
        from pyphi.conf import reject_unknown_kwargs
        from pyphi.measures.distribution import resolve_mechanism_measure

        reject_unknown_kwargs(
            f"System.{caller}", kwargs, ["repertoire_distance", "state"]
        )
        alphabet_sizes = self.substrate.factored_tpm.alphabet_sizes
        if kwargs.get("repertoire_distance") is not None:
            kwargs["repertoire_distance"] = _resolve_measure(
                "repertoire_distance",
                kwargs["repertoire_distance"],
                lambda name: resolve_mechanism_measure(name, alphabet_sizes),
            )
            return kwargs
        version = _config.formalism.iit.version
        if version != "IIT_3_0":
            raise TypeError(
                f"System.{caller}() with the configured measure is an IIT 3.0 "
                "quantity (the distance between a mechanism's repertoire and "
                f"the unconstrained repertoire) and is not defined under "
                f"{version}; pass repertoire_distance= to choose a distance, "
                "such as 'KLD'. Under IIT 4.0, a mechanism's intrinsic "
                "specification of its purview is on System.mic() and "
                "System.mie() (mice.specified_state.intrinsic_specification) "
                "or System.intrinsic_specification(), and the system's is "
                "System.sia().intrinsic_specification."
            )
        kwargs["repertoire_distance"] = resolve_mechanism_measure(
            _config.formalism.iit.mechanism_phi_measure, alphabet_sizes
        )
        return kwargs

    def cause_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        """Return the distance between the cause repertoire and the
        unconstrained one. Units are given as in :meth:`cause_repertoire`;
        ``repertoire_distance`` names the distance (IIT 3.0's by default)."""
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_info(
            self,
            self._coerce_units(mechanism),
            self._coerce_units(purview),
            **self._info_kwargs("cause_info", kwargs),
        )

    def effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        """Return the distance between the effect repertoire and the
        unconstrained one. Units are given as in :meth:`effect_repertoire`;
        ``repertoire_distance`` names the distance (IIT 3.0's by default)."""
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_info(
            self,
            self._coerce_units(mechanism),
            self._coerce_units(purview),
            **self._info_kwargs("effect_info", kwargs),
        )

    def cause_effect_info(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        """Return the smaller of :meth:`cause_info` and :meth:`effect_info`."""
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_effect_info(
            self,
            self._coerce_units(mechanism),
            self._coerce_units(purview),
            **self._info_kwargs("cause_effect_info", kwargs),
        )

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
        if kwargs.get("states") is not None:
            given = list(purview)
            kwargs["states"] = [
                self._coerce_with_states(given, state)[1] for state in kwargs["states"]
            ]
        spec = ra.intrinsic_specification(
            self,
            direction,
            self._coerce_units(mechanism),
            self._coerce_units(purview),
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
            If a measure name is not a registered measure, or
            ``partition_scheme`` is not a registered system partition scheme.
        ~pyphi.exceptions.StateUnreachableForwardsError
            If the state is unreachable under the background conditioning of
            the formalism in effect for the call, as :func:`pyphi.analyze`
            would raise. Not raised for a System built with
            ``config.infrastructure.validate_system_states`` off.
        """
        from pyphi.cache.disk import maybe_disk_cached
        from pyphi.conf import presets
        from pyphi.conf._helpers import check_progress_flag
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.measures.distribution import resolve_system_measure

        system_measure = _resolve_measure(
            "system_measure", system_measure, resolve_system_measure
        )
        specification_measure = _resolve_measure(
            "specification_measure", specification_measure, resolve_mechanism_measure
        )
        if partition_scheme is not None:
            from pyphi.partition import system_partition_types

            if partition_scheme not in system_partition_types.all():
                raise ValueError(
                    f"unknown partition_scheme {partition_scheme!r}; expected one "
                    f"of {system_partition_types.all()}"
                )
        options = _given(
            system_state=system_state,
            partitions=partitions,
            partition_scheme=partition_scheme,
            directions=directions,
            system_measure=system_measure,
            specification_measure=specification_measure,
            parallel=parallel,
            progress=check_progress_flag(progress),
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
            self._recheck_state()
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

        Raises
        ------
        ValueError
            If ``sia`` or ``distinctions`` was computed for other units or
            another state: the analysis's units or state differ from this
            system's, or a distinction's mechanism or purviews lie outside
            the system or its mechanism state differs from the system's
            state. Only the system's own units and their state are checked:
            an analysis of the same units in another substrate, or with the
            other units of the substrate in another state, is accepted.
        ~pyphi.exceptions.StateUnreachableForwardsError
            If the state is unreachable under the background conditioning of
            the formalism in effect for the call, as :func:`pyphi.analyze`
            would raise. Not raised for a System built with
            ``config.infrastructure.validate_system_states`` off.

        Notes
        -----
        For results computed under an earlier version of IIT (see
        :doc:`/howto/earlier-versions`): IIT 3.0 has no relations, so under it
        the CES holds the distinctions and an empty
        :class:`~pyphi.relations.NullRelations`.
        """
        from pyphi.cache.disk import maybe_disk_cached
        from pyphi.conf import presets

        self._check_precomputed(sia=sia, distinctions=distinctions)
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
            self._recheck_state()
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
            system-partition search. ``analyze(..., compute="distinctions")``
            returns this filtered set. The result is a
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

        Parameters
        ----------
        direction : Direction
            ``Direction.CAUSE`` or ``Direction.EFFECT``.
        mechanism : tuple[int, ...]
            The mechanism's node indices.
        purview : tuple[int, ...]
            The purview's node indices.
        **kwargs
            ``partitions``: the mechanism partitions to search (default:
            every partition of the configured scheme). ``state``: the
            specified state (a
            :class:`~pyphi.models.state_specification.StateSpecification`)
            to evaluate at, instead of searching the specified states.
            ``mechanism_measure`` and ``specification_measure``: a measure
            name or a measure, overriding the configuration. ``parallel``,
            ``progress``, ``chunksize`` and ``sequential_threshold`` tune the
            parallel evaluation of the partitions.

        Returns
        -------
        RepertoireIrreducibilityAnalysis
            The minimum partition and its φ.

        Raises
        ------
        TypeError
            If an option is not one of these, or ``state`` is not a
            specified state.
        ValueError
            If ``direction`` is neither cause nor effect, or a measure name
            is not registered.
        ~pyphi.formalism.base.MeasureNotCompatibleError
            If a measure is not defined by the active version of IIT.
        """
        return self._find_mip("find_mip", direction, mechanism, purview, kwargs)

    def _find_mip(
        self,
        caller: str,
        direction: Any,
        mechanism: Any,
        purview: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        from pyphi.conf import config as _config
        from pyphi.formalism import find_mip as _find_mip
        from pyphi.measures.distribution import resolve_mechanism_measure

        _reject_unknown(caller, kwargs, *_MIP_OPTIONS)
        direction = _check_direction(direction)
        kwargs = _mip_options(self, kwargs)
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
        return _find_mip(
            self,
            direction,
            self._coerce_units(mechanism),
            self._coerce_units(purview),
            **kwargs,
        )

    def cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        """Return the minimum partition of a mechanism's cause over a purview.

        Takes the options of :meth:`find_mip`.
        """
        from pyphi.direction import Direction

        return self._find_mip("cause_mip", Direction.CAUSE, mechanism, purview, kwargs)

    def effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> Any:
        """Return the minimum partition of a mechanism's effect over a purview.

        Takes the options of :meth:`find_mip`.
        """
        from pyphi.direction import Direction

        return self._find_mip("effect_mip", Direction.EFFECT, mechanism, purview, kwargs)

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        """Return the φ of :meth:`cause_mip`, or 0 when it is null.

        Takes the options of :meth:`find_mip`.
        """
        from pyphi.direction import Direction

        mip = self._find_mip(
            "phi_cause_mip", Direction.CAUSE, mechanism, purview, kwargs
        )
        return mip.phi if mip else 0

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kwargs: Any) -> float:
        """Return the φ of :meth:`effect_mip`, or 0 when it is null.

        Takes the options of :meth:`find_mip`.
        """
        from pyphi.direction import Direction

        mip = self._find_mip(
            "phi_effect_mip", Direction.EFFECT, mechanism, purview, kwargs
        )
        return mip.phi if mip else 0

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

        return _phi(self, self._coerce_units(mechanism), self._coerce_units(purview))

    def _coerce_units(self, nodes: Any) -> tuple[int, ...]:
        """Sorted node indices for ``nodes``, given as indices or labels.

        Raises ``TypeError`` for ``None`` and ``ValueError`` for an unknown
        label, an index outside the substrate, or a unit named twice.
        """
        if nodes is None:
            raise TypeError(
                "expected a sequence of node indices or labels, such as (0, 1) "
                "or ('A', 'B'); got None"
            )
        # Sorted in-range index tuples, the form the formalism's inner loops
        # pass, need no conversion.
        if (
            type(nodes) is tuple
            and all(type(n) is int for n in nodes)
            and all(a < b for a, b in itertools.pairwise(nodes))
            and (not nodes or (nodes[0] >= 0 and nodes[-1] < self.substrate.size))
        ):
            return nodes
        return self.substrate.node_labels.coerce_to_indices(nodes)

    def _coerce_with_states(self, nodes: Any, *states: Any) -> tuple[Any, ...]:
        """Sorted node indices for ``nodes``, and each of ``states`` (given in
        the order of ``nodes``, or None) reordered to match them."""
        units = self._coerce_units(nodes)
        if all(state is None for state in states) or (
            type(nodes) is tuple and nodes == units
        ):
            return (units, *states)
        given = [self._coerce_units((node,))[0] for node in nodes]
        order = sorted(range(len(given)), key=given.__getitem__)
        return (
            units,
            *(
                None if state is None else tuple(state[i] for i in order)
                for state in states
            ),
        )

    def _coerce_purviews(self, purviews: Any) -> Any:
        if purviews is None:
            return None
        return [self._coerce_units(purview) for purview in purviews]

    def find_mice(self, direction: Any, mechanism: Any, **kwargs: Any) -> Any:
        from pyphi.formalism import find_mice as _find_mice

        _reject_unknown("find_mice", kwargs, "purviews")
        direction = _check_direction(direction)
        if "purviews" in kwargs:
            kwargs["purviews"] = self._coerce_purviews(kwargs["purviews"])
        return _find_mice(self, direction, self._coerce_units(mechanism), **kwargs)

    def mic(self, mechanism: Any, *, purviews: Any = None) -> Any:
        """Return the maximally irreducible cause of a mechanism.

        Parameters
        ----------
        mechanism : tuple
            The mechanism's units, as node indices or labels in any order.
        purviews : iterable of tuple, optional
            The candidate purviews to search, each as node indices or labels
            in any order. Candidates outside this system or reducible by its
            connectivity are skipped. ``None`` searches every purview of the
            system that the connectivity does not rule out.

        Returns
        -------
        MaximallyIrreducibleCause
            The cause purview of maximal φ, with its minimum partition
            and specified state.

        Raises
        ------
        ValueError
            If the mechanism or a purview names an unknown label or an index
            outside the substrate, or names a unit more than once.
        """
        from pyphi.formalism import mic as _mic

        return _mic(
            self,
            self._coerce_units(mechanism),
            purviews=self._coerce_purviews(purviews),
        )

    def mie(self, mechanism: Any, *, purviews: Any = None) -> Any:
        """Return the maximally irreducible effect of a mechanism.

        Parameters
        ----------
        mechanism : tuple
            The mechanism's units, as node indices or labels in any order.
        purviews : iterable of tuple, optional
            The candidate purviews to search, each as node indices or labels
            in any order. Candidates outside this system or reducible by its
            connectivity are skipped. ``None`` searches every purview of the
            system that the connectivity does not rule out.

        Returns
        -------
        MaximallyIrreducibleEffect
            The effect purview of maximal φ, with its minimum partition
            and specified state.

        Raises
        ------
        ValueError
            If the mechanism or a purview names an unknown label or an index
            outside the substrate, or names a unit more than once.
        """
        from pyphi.formalism import mie as _mie

        return _mie(
            self,
            self._coerce_units(mechanism),
            purviews=self._coerce_purviews(purviews),
        )

    def phi_max(self, mechanism: Any) -> float:
        from pyphi.formalism import phi_max as _phi_max

        return _phi_max(self, mechanism)

    def distinction(self, mechanism: Any) -> Any:
        from pyphi.formalism import distinction as _distinction

        return _distinction(self, self._coerce_units(mechanism))

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


_dataclass_init = System.__init__


@wraps(_dataclass_init)
def _init(self: System, *args: Any, **kwargs: Any) -> None:
    # Point a ported ``Subsystem(network, state, nodes)`` call at the
    # keyword-only ``subset`` instead of a bare argument-count error.
    # ``wraps`` keeps the dataclass signature for help(), inspect and Sphinx.
    old = [name for name in ("indices", "nodes", "node_indices") if name in kwargs]
    if len(args) > 2 or old:
        given = f"keyword argument {old[0]!r}" if old else "a third positional argument"
        raise TypeError(
            f"{type(self).__name__}() got {given}; pass the units of the "
            "candidate system as the keyword-only subset=..."
        )
    _dataclass_init(self, *args, **kwargs)


System.__init__ = _init  # pyright: ignore[reportAttributeAccessIssue]
