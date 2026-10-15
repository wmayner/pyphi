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

import numpy as np

from pyphi import exceptions
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.tpm import _normalize_history
from pyphi.macro.tpm import _validate_units
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.substrate import Substrate
from pyphi.system import System


def _require_marginalized_background() -> None:
    """Raise unless the cause-side background is causally marginalized.

    The macro cause TPM weights the micro background before the update window
    by its posterior (Marshall et al. 2026, Eq. 34), which matches
    ``CAUSAL_MARGINALIZATION`` and has no counterpart that conditions the
    background on its current state.
    """
    from pyphi.conf import config as _config

    convention = _config.formalism.iit.background_conditioning
    if _config.formalism.iit.version == "IIT_3_0":
        raise ValueError(
            "the intrinsic-units macro framework is not defined for IIT_3_0: "
            "its construction marginalizes the cause-side background (Marshall "
            "et al. 2026, Eq. 34), and IIT 3.0 conditions the background on "
            "its current state"
        )
    if convention != "CAUSAL_MARGINALIZATION":
        raise ValueError(
            "the intrinsic-units macro construction marginalizes the cause-side "
            "background (Marshall et al. 2026, Eq. 34) and has no version for "
            f"background_conditioning={convention!r}; set "
            "config.formalism.iit.background_conditioning to "
            "'CAUSAL_MARGINALIZATION' to build macro systems or search grains"
        )


def _macro_state(units, history):
    state = []
    for unit in units:
        window = tuple(
            tuple(s[u] for u in unit.micro_constituents)
            for s in history[len(history) - unit.micro_grain :]
        )
        state.append(unit.state_from(window))
    return tuple(state)


def _micro_labels(substrate: Substrate, units) -> tuple[str, ...] | None:
    """Each unit's micro constituents' labels, joined; ``None`` when two
    units would share a label."""
    labels = tuple(
        "".join(substrate.node_labels.indices2labels(unit.micro_constituents))
        for unit in units
    )
    return labels if len(set(labels)) == len(labels) else None


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

    def __post_init__(self, subset: Any) -> None:
        if self.micro_substrate is None or not self.units:
            raise TypeError("MacroSystem must be constructed via MacroSystem.from_micro")
        super().__post_init__(subset)

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
            The system's macro units (Eq. 19 must hold).
        micro_history : sequence of universe states
            Universe micro states, oldest first, of length
            ``max(tau_J)``. A bare state is accepted when every unit has
            micro grain 1.
        node_labels : optional
            Labels for the macro units. If None, each unit is labelled by
            its micro constituents' labels in ``substrate``, joined (``"A"``
            for a unit over micro unit ``A``, ``"AB"`` for a unit over ``A``
            and ``B``), so the system prints in the substrate's labels; when
            two units would share a label, the default labels are used.

        Raises
        ------
        ValueError
            If ``config.formalism.iit.background_conditioning`` is not
            ``"CAUSAL_MARGINALIZATION"``, the only background convention the
            macro cause TPM is defined for.
        """
        _require_marginalized_background()
        units = tuple(units)
        _validate_units(substrate, units)
        history = _normalize_history(units, substrate, micro_history)
        cause_marginal, effect_marginal = macro_tpms(substrate, units, history)
        if node_labels is None:
            node_labels = _micro_labels(substrate, units)
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

    def _validate_state_reachable(self) -> None:
        """Raise if no prior macro state can produce the current macro state.

        The check runs on the cause TPM, which weights the pre-window
        background by its posterior (Marshall et al. 2026, Eq. 34). The effect
        TPM that backs the synthetic substrate holds the background at its
        current state (Eq. 33), which would make some reachable states look
        unreachable.
        """
        cause = self.cause_marginal
        joint = np.ones(cause.alphabet_sizes, dtype=np.float64)
        for i in range(cause.n_nodes):
            joint = joint * cause.factor(i)[..., self.state[i]]
        if joint.sum() <= 0.0:
            raise exceptions.StateUnreachableForwardsError(self.state)

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
