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

import hashlib
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import Any

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.tpm import _normalize_history
from pyphi.macro.tpm import _validate_units
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.substrate import Substrate
from pyphi.system import System


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
    macro_cause_marginal: FactoredTPM | None = field(default=None, repr=False)

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

        Parameters
        ----------
        substrate : Substrate
            The binary micro universe.
        units : sequence of MacroUnit
            The system's macro units (Eq. 18 must hold).
        micro_history : sequence of universe states
            Universe micro states, oldest first, of length
            ``max(tau_J)``. A bare state is accepted when every unit has
            micro grain 1.
        node_labels : optional
            Labels for the macro units.
        """
        units = tuple(units)
        _validate_units(substrate, units)
        history = _normalize_history(units, substrate, micro_history)
        cause_marginal, effect_marginal = macro_tpms(substrate, units, history)
        macro_substrate = Substrate.from_factored(
            effect_marginal, node_labels=node_labels
        )
        return cls(
            substrate=macro_substrate,
            state=_macro_state(units, history),
            units=units,
            micro_substrate=substrate,
            micro_history=history,
            macro_cause_marginal=cause_marginal,
        )

    @classmethod
    def from_substrate(cls, *args: Any, **kwargs: Any) -> MacroSystem:  # noqa: ARG003
        raise TypeError(
            "MacroSystem cannot be built from a substrate alone; use "
            "MacroSystem.from_micro(substrate, units, micro_history)"
        )

    @property
    def cause_marginal(self) -> FactoredTPM:  # type: ignore[override]
        """The construction's cause TPM (Eqs. 26-40, cause weighting)."""
        assert self.macro_cause_marginal is not None
        return self.macro_cause_marginal

    @property
    def proper_cause_marginal(self) -> FactoredTPM:  # type: ignore[override]
        """Identical to :attr:`cause_marginal`: there is no macro background."""
        return self.cause_marginal

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MacroSystem):
            # A plain System over the macro substrate is a different analysis
            # (the macro construction overrides the cause TPM), so it is never
            # equal. Returning NotImplemented here would fall back to
            # System.__eq__, which compares only the shared fields and reports
            # equality with a different hash and a different phi. Python tries
            # the subclass's __eq__ first in both comparison directions, so
            # answering False here keeps the comparison symmetric.
            if isinstance(other, System):
                return False
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

    @cached_property
    def _fingerprint(self) -> bytes:  # type: ignore[override]
        """blake2b-256 digest of the macro system's kernel inputs.

        The inherited :attr:`System._fingerprint` is unsound for a macro
        system: it hashes only the (effect-marginal) macro substrate and would
        conflate two groupings that share an effect marginal but differ in their
        cause marginal. This digests both derived marginals the kernel reads —
        the effect-side macro substrate and the overridden cause marginal
        (:attr:`macro_cause_marginal`) — plus the state, indices, and partition,
        so equal fingerprint implies identical repertoires.
        """
        assert self.macro_cause_marginal is not None
        ccm = self.macro_cause_marginal
        h = hashlib.blake2b(digest_size=32)
        h.update(self.substrate._fingerprint)
        h.update(repr(ccm.alphabet_sizes).encode())
        for i in range(ccm.n_nodes):
            h.update((ccm.factor(i) + 0.0).tobytes())
        h.update(repr(tuple(self.state)).encode())
        h.update(repr(tuple(self.node_indices)).encode())
        h.update(repr(tuple(self.external_indices)).encode())
        h.update(repr(tuple(sorted(self.partition.indices))).encode())
        h.update(repr(sorted(self.partition.removed_edges())).encode())
        return h.digest()
