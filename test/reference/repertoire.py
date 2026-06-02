"""Independent brute-force cause/effect repertoire reference.

Computes IIT cause/effect repertoires directly from raw per-node forward
factors and a (possibly cut) connectivity matrix, using only NumPy. Deliberately
shares no code with ``pyphi.core.repertoire_algebra`` or ``pyphi.node`` so it
serves as a genuine cross-check (see the k-ary cut bug fixed in 7a8efe62).

Each ``factors[i]`` is the forward conditional ``P(node_i,t+1 | prev_states)``
with shape ``(*alphabet_sizes, k_i)`` (obtainable from
``substrate.factored_tpm.factor(i)``). ``cut_cm[j, i] == 1`` means node ``j`` is
an input to node ``i``; severing an edge in a partition is modelled by zeroing
the corresponding ``cut_cm`` entry.
"""

from __future__ import annotations

import numpy as np


def _canonical_shape(alph, purview, n):
    pv = set(purview)
    return tuple(alph[i] if i in pv else 1 for i in range(n))


def ref_effect(factors, alph, cut_cm, mechanism, mstate, purview, n):
    """Effect repertoire P(purview_{t+1} | mechanism_t = mstate), under cut_cm.

    For each purview node z, condition its forward factor on the mechanism
    nodes that are still its inputs after the cut (delta at their state) and
    average uniformly over every other previous-state axis.
    """
    mech = set(mechanism)
    out = np.ones(_canonical_shape(alph, purview, n))
    for z in purview:
        inputs_z = {j for j in range(n) if cut_cm[j, z] == 1}
        cond = mech & inputs_z
        weight = np.ones(alph)
        for j in range(n):
            shape = [1] * n
            shape[j] = alph[j]
            if j in cond:
                v = np.zeros(alph[j])
                v[mstate[j]] = 1.0
            else:
                v = np.full(alph[j], 1.0 / alph[j])
            weight = weight * v.reshape(shape)
        eff_z = np.tensordot(
            weight, factors[z], axes=(list(range(n)), list(range(n)))
        )  # shape (k_z,)
        canon = [1] * n
        canon[z] = alph[z]
        out = out * eff_z.reshape(canon)
    return out


def ref_cause(factors, alph, cut_cm, mechanism, mstate, purview, n):
    """Cause repertoire P(purview_{t-1} | mechanism_t = mstate), under cut_cm.

    Product over mechanism nodes of the forward factor sliced at the node's
    observed state, averaged over every previous-state axis EXCEPT the purview
    nodes that remain inputs to that mechanism node after the cut, then
    normalized. Averaging over severed inputs (even when they are in the
    purview) is what applies the cut on the cause side — omitting it silently
    ignores the cut.
    """
    pv = set(purview)
    joint = np.ones(_canonical_shape(alph, purview, n))
    for m in mechanism:
        cut_inputs_m = {j for j in range(n) if cut_cm[j, m] == 1}
        g = factors[m][..., mstate[m]]
        for ax in range(n):
            if not (ax in pv and ax in cut_inputs_m):
                g = g.mean(axis=ax, keepdims=True)
        joint = joint * g
    total = joint.sum()
    return joint / total if total != 0 else joint
