"""The four-step macro TPM construction (Marshall et al. 2024, Eqs. 26-40).

Step 1 discounts micro connections extrinsic to the macro unit being
updated (Eqs. 26-30). Step 2 chains the modified probabilities into
micro-update sequences (Eq. 31). Step 3 causally marginalizes the
background, conditioning on the current micro state for effects (Eq. 33)
and Bayesian-weighting the pre-window background state for causes
(Eq. 34). Step 4 compresses sequences into macro states via the mapping
preimages and the sequence-proportion weights ``r(u^S, s)``
(Eqs. 35-40). Steps 2 and 4 are fused: sequence probabilities are
accumulated per chaining step into per-update state classes of the
unit's micro constituents, never materializing the full sequence tensor.
"""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.units import MacroUnit


def _system_micro_indices(units) -> tuple[int, ...]:
    """``U^S``: the sorted union of the units' micro constituents."""
    out: set[int] = set()
    for unit in units:
        out |= set(unit.micro_constituents)
    return tuple(sorted(out))


def _patron_units(units) -> dict[int, int]:
    """Map each apportioned background index to its patron unit's index."""
    out: dict[int, int] = {}
    for k, unit in enumerate(units):
        for w in unit.background_apportionment:
            out[w] = k
    return out


def _discounted_on_probabilities(
    factored: FactoredTPM, units: tuple[MacroUnit, ...], j: int
) -> np.ndarray:
    """Step 1 (Eqs. 26-30): modified ON probabilities for updating unit ``j``.

    Returns:
        np.ndarray: ``(2**n, n)`` — for each universe state (little-endian
        row index) and micro unit, the modified probability that the unit
        is ON at the next micro update.
    """
    n = factored.n_nodes
    constituents = set(units[j].micro_constituents)
    patron = _patron_units(units)
    columns = []
    for i in range(n):
        p_on = factored.factor(i)[..., 1]
        if i in constituents:
            out = p_on  # Eq. 27: connections among U^J kept intact
        elif i in patron:
            k = patron[i]
            keep = set(units[k].micro_constituents) | set(
                units[k].background_apportionment
            )
            axes = tuple(a for a in range(n) if a not in keep)
            if axes:
                out = np.broadcast_to(p_on.mean(axis=axes, keepdims=True), p_on.shape)
            else:
                out = p_on  # Eq. 29 with nothing to noise
        else:
            # Eq. 28: other system units and unapportioned background
            out = np.full(p_on.shape, p_on.mean())
        columns.append(np.asarray(out).reshape(-1, order="F"))
    return np.stack(columns, axis=1)
