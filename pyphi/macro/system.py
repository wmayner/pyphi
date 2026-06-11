"""MacroSystem: a system of macro units analyzed by the IIT pipeline.

``MacroSystem`` subclasses :class:`~pyphi.system.System` over a
synthetic macro-level :class:`~pyphi.substrate.Substrate` built from the
construction's effect TPM (all-ones connectivity, one binary node per
macro unit). The cause-side TPM properties are overridden with the
construction's cause TPM: the two directions differ in their treatment
of micro background units (Eqs. 33-34), so the cause TPM is not
derivable from the synthetic substrate. Everything else — nodes,
repertoires, partitions, ``sia``/``ces`` — is inherited unchanged, and
the pipeline consumes a ``MacroSystem`` exactly like a ``System``.

Once the macro TPMs are built there is no further reference to the
background units, the units' grains, or their micro constituents; macro
units are perturbed uniformly over their two states like any units.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.tpm import _system_micro_indices
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.substrate import Substrate
from pyphi.system import System


def _validate_units(substrate: Substrate, units: tuple[MacroUnit, ...]) -> None:
    if not units:
        raise ValueError("at least one macro unit is required")
    sizes = substrate.factored_tpm.alphabet_sizes
    if any(size != 2 for size in sizes):
        raise ValueError(f"the substrate must be binary; got alphabet sizes {sizes}")
    n = substrate.size
    claimed: set[int] = set()
    for unit in units:
        footprint = set(unit.micro_constituents) | set(unit.background_apportionment)
        if max(footprint) >= n:
            raise ValueError(
                f"unit references indices outside the substrate (size {n}): "
                f"{sorted(i for i in footprint if i >= n)}"
            )
        if claimed & footprint:
            raise ValueError(
                "units' micro constituents and apportionments must be "
                f"pairwise disjoint (Eq. 18); overlap: {sorted(claimed & footprint)}"
            )
        claimed |= footprint
    system = set(_system_micro_indices(units))
    for unit in units:
        if set(unit.background_apportionment) & system:
            raise ValueError(
                "background apportionment must lie outside the system's "
                "micro constituents: "
                f"{sorted(set(unit.background_apportionment) & system)}"
            )
        _validate_nested_apportionment(unit)


def _validate_nested_apportionment(unit: MacroUnit) -> None:
    """Eq. 12: constituents' apportionments nest within their parent's."""
    parent = set(unit.background_apportionment)
    for c in unit.constituents:
        if isinstance(c, MacroUnit):
            if not set(c.background_apportionment) <= parent:
                raise ValueError(
                    "a constituent's background apportionment must be a "
                    "subset of its parent's (Eq. 12); offending indices: "
                    f"{sorted(set(c.background_apportionment) - parent)}"
                )
            _validate_nested_apportionment(c)


def _normalize_history(units, substrate, micro_history):
    max_grain = max(unit.micro_grain for unit in units)
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        if max_grain == 1:
            history = (history,)
        else:
            raise ValueError(
                "micro_history must be a sequence of states (oldest "
                f"first) of length {max_grain}; got a bare state"
            )
    history = tuple(tuple(s) for s in history)
    if len(history) != max_grain:
        raise ValueError(
            f"micro_history must have {max_grain} entries (the maximum "
            f"micro grain); got {len(history)}"
        )
    n = substrate.size
    for s in history:
        if len(s) != n or any(v not in (0, 1) for v in s):
            raise ValueError(
                f"each history entry must be a binary universe state of "
                f"length {n}; got {s}"
            )
    return history


def _macro_state(units, history):
    state = []
    for unit in units:
        window = tuple(
            tuple(s[u] for u in unit.micro_constituents)
            for s in history[len(history) - unit.micro_grain :]
        )
        state.append(unit.state_from(window))
    return tuple(state)


@dataclass(frozen=True, eq=False)
class MacroSystem(System):
    """A system of macro units, consumed by the pipeline like a System.

    Construct with :meth:`from_micro`. The inherited ``substrate`` field
    holds the synthetic macro substrate; the micro universe lives in
    ``micro_substrate``.
    """

    units: tuple[MacroUnit, ...] = ()
    micro_substrate: Substrate | None = None
    micro_history: tuple[tuple[int, ...], ...] = ()
    macro_cause_tpm: FactoredTPM | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.micro_substrate is None or not self.units:
            raise TypeError("MacroSystem must be constructed via MacroSystem.from_micro")
        super().__post_init__()

    @classmethod
    def from_micro(
        cls,
        substrate: Substrate,
        units,
        micro_history,
        node_labels=None,
    ) -> MacroSystem:
        """Build a MacroSystem from a micro substrate and macro units.

        Args:
            substrate: The binary micro universe.
            units: The system's macro units (Eq. 18 must hold).
            micro_history: Universe micro states, oldest first, of
                length ``max(tau_J)``. A bare state is accepted when
                every unit has micro grain 1.
            node_labels: Labels for the macro units.
        """
        units = tuple(units)
        _validate_units(substrate, units)
        history = _normalize_history(units, substrate, micro_history)
        cause_tpm, effect_tpm = macro_tpms(substrate, units, history)
        macro_substrate = Substrate.from_factored(effect_tpm, node_labels=node_labels)
        return cls(
            substrate=macro_substrate,
            state=_macro_state(units, history),
            units=units,
            micro_substrate=substrate,
            micro_history=history,
            macro_cause_tpm=cause_tpm,
        )

    @classmethod
    def from_substrate(cls, *args: Any, **kwargs: Any) -> MacroSystem:  # noqa: ARG003
        raise TypeError(
            "MacroSystem cannot be built from a substrate alone; use "
            "MacroSystem.from_micro(substrate, units, micro_history)"
        )

    @property
    def cause_tpm(self) -> FactoredTPM:  # type: ignore[override]
        """The construction's cause TPM (Eqs. 26-40, cause weighting)."""
        assert self.macro_cause_tpm is not None
        return self.macro_cause_tpm

    @property
    def proper_cause_tpm(self) -> FactoredTPM:  # type: ignore[override]
        """Identical to :attr:`cause_tpm`: there is no macro background."""
        return self.cause_tpm

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MacroSystem):
            return NotImplemented
        return (
            self.micro_substrate == other.micro_substrate
            and self.units == other.units
            and self.micro_history == other.micro_history
            and self.partition == other.partition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.micro_substrate,
                self.units,
                self.micro_history,
                self.partition,
            )
        )
