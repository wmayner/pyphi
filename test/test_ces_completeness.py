"""CES-completeness: is the cause-effect structure a complete substrate invariant?

This module records the Wave-1 confirmation experiment for ROADMAP P11.95c case
(b): does ``CES(S1) == CES(S2)`` (as a label-independent structure) imply that
S1 and S2 are isomorphic (related by a single node permutation)?  The ``<-``
direction is trivial; the ``->`` direction was the open theoretical question
that decided whether case (b) ("structural CES coincidence without substrate
isomorphism") is empty or a genuine distinct case.

**Finding (brute force on small binary substrates).**

* *In general the CES is NOT a complete invariant.*  Non-isomorphic substrates
  can share an identical CES -- identical down to repertoires and relations,
  under node relabeling.  Concrete n=2 counterexamples are pinned below.  Every
  such counterexample contains a part the CES does not constrain (a causally
  inert unit, or a unit that specifies no distinction at the evaluated state),
  so the residual transition probabilities are free to vary.

* *Restricted to irreducible systems (Phi_s > 0) -- the complexes a CES is
  properly the structure of -- no counterexample is found* (n=2 exhaustive:
  every complex has a distinct CES fingerprint; see the slow test).  Every
  pinned counterexample is reducible (Phi_s = 0).

The label-independent fingerprint matches the roadmap's case-(b) definition:
each distinction is a vertex (mechanism, cause purview + specified-state *tie
set*, effect purview + tie set, cause phi, effect phi); each relation is an edge
(relata, purview, phi); the whole structure is minimised over all node
permutations.  Tie *sets* (not the tie-resolved winner) are used so that
tie-resolution labelling cannot manufacture spurious differences.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.convert import le_index2state
from pyphi.convert import state2le_index

_ROUND = 9  # deterministic phi is bit-exact across identical repertoires


# --------------------------------------------------------------------------- #
# Node-permutation relabeling (perm[i] = new label of old node i)
# --------------------------------------------------------------------------- #
def _relabel_state(state, perm):
    new = [0] * len(state)
    for i, val in enumerate(state):
        new[perm[i]] = val
    return tuple(new)


def _relabel_tpm(tpm, perm):
    n = len(perm)
    new = np.empty_like(tpm)
    for r in range(tpm.shape[0]):
        r_new = state2le_index(_relabel_state(le_index2state(r, n), perm))
        for i in range(n):
            new[r_new, perm[i]] = tpm[r, i]
    return new


def _substrate_state_iso(tpm1, s1, tpm2, s2):
    """True iff a single node permutation maps (tpm1, s1) exactly onto (tpm2, s2)."""
    return any(
        np.array_equal(_relabel_tpm(tpm1, perm), tpm2) and _relabel_state(s1, perm) == s2
        for perm in itertools.permutations(range(len(s1)))
    )


# --------------------------------------------------------------------------- #
# Label-independent CES fingerprint
# --------------------------------------------------------------------------- #
def _tie_states(specified_state):
    return {t.state for t in specified_state.ties} or {specified_state.state}


def _raw_ces(tpm, state):
    """(distinctions, relations) of the system as plain, label-bearing records."""
    system = System(
        Substrate(
            np.asarray(tpm, dtype=float), cm=np.ones((len(state),) * 2, dtype=int)
        ),
        state,
    )
    ces = system.ces()
    dists = [
        {
            "mech": tuple(d.mechanism),
            "mstate": tuple(d.mechanism_state),
            "c_pv": tuple(d.cause.purview),
            "c_ties": frozenset(_tie_states(d.cause.specified_state)),
            "c_phi": round(float(d.cause.phi), _ROUND),
            "e_pv": tuple(d.effect.purview),
            "e_ties": frozenset(_tie_states(d.effect.specified_state)),
            "e_phi": round(float(d.effect.phi), _ROUND),
        }
        for d in ces.distinctions
    ]
    rels = [
        {
            "relata": tuple(
                tuple(int(getattr(x, "index", x)) for x in m) for m in r.mechanisms
            ),
            "pv": tuple(int(getattr(u, "index", u)) for u in r.purview),
            "phi": round(float(r.phi), _ROUND),
        }
        for r in ces.relations
    ]
    return dists, rels, float(ces.sia.phi)


def _states_relabeled(states, nodes, perm):
    return tuple(
        sorted(
            tuple(sorted((perm[node], v) for node, v in zip(nodes, s, strict=True)))
            for s in states
        )
    )


def _ces_fingerprint(tpm, state):
    """Canonical (relabeling-invariant) fingerprint of the CES of (tpm, state)."""
    dists, rels, _phi = _raw_ces(tpm, state)
    best = None
    for perm in itertools.permutations(range(len(state))):
        d_recs = sorted(
            (
                tuple(sorted(perm[i] for i in d["mech"])),
                tuple(
                    sorted(
                        (perm[i], v) for i, v in zip(d["mech"], d["mstate"], strict=True)
                    )
                ),
                tuple(sorted(perm[i] for i in d["c_pv"])),
                d["c_phi"],
                _states_relabeled(d["c_ties"], d["c_pv"], perm),
                tuple(sorted(perm[i] for i in d["e_pv"])),
                d["e_phi"],
                _states_relabeled(d["e_ties"], d["e_pv"], perm),
            )
            for d in dists
        )
        r_recs = sorted(
            (
                tuple(sorted(tuple(sorted(perm[i] for i in m)) for m in r["relata"])),
                tuple(sorted(perm[i] for i in r["pv"])),
                r["phi"],
            )
            for r in rels
        )
        cand = (tuple(d_recs), tuple(r_recs))
        if best is None or cand < best:
            best = cand
    return best


# --------------------------------------------------------------------------- #
# Pinned counterexamples (found by the exhaustive n=2 search; see module docs)
# --------------------------------------------------------------------------- #
# Each is a pair of NON-isomorphic substrates (with a state) whose CESes are
# identical up to relabeling, proving the CES is not a complete substrate
# invariant.  Both members are reducible (Phi_s = 0).
_COUNTEREXAMPLES = {
    # An inert (constant) unit is invisible to the CES: node 0 is constant 0 in A
    # (frozen at state 0) and constant 1 in B (frozen at state 1).  No 2-node
    # permutation turns an all-0 column into an all-1 column, so the substrates
    # are non-isomorphic, yet the CES is identical down to repertoires.
    "inert_unit": (
        ([[0, 0], [0, 0], [0, 0], [0, 1]], (0, 0)),
        ([[1, 0], [1, 0], [1, 0], [1, 1]], (1, 0)),
    ),
    # A fully causally-active, full-support counterexample: every unit is both
    # written and read, and the CES touches every node (node 0 appears in the
    # effect purview), yet only node 1 specifies a distinction, leaving node 0's
    # transition rows free to differ between the two non-isomorphic substrates.
    "active_full_support": (
        ([[0, 0], [0, 0], [0, 1], [1, 0]], (0, 0)),
        ([[0, 0], [0, 0], [1, 1], [0, 0]], (0, 0)),
    ),
}


@pytest.fixture(autouse=True)
def _iit4_2023():
    with config.override(
        **presets.iit4_2023, validate_system_states=False, progress_bars=False
    ):
        yield


@pytest.mark.parametrize("name", list(_COUNTEREXAMPLES))
def test_ces_not_complete_invariant(name):
    """Non-isomorphic substrates can share an identical CES (case (b) is non-empty)."""
    (tpm_a, state_a), (tpm_b, state_b) = _COUNTEREXAMPLES[name]
    tpm_a, tpm_b = np.array(tpm_a, dtype=float), np.array(tpm_b, dtype=float)

    assert not _substrate_state_iso(tpm_a, state_a, tpm_b, state_b), (
        f"{name}: substrates must be non-isomorphic for a valid counterexample"
    )
    fp_a = _ces_fingerprint(tpm_a, state_a)
    fp_b = _ces_fingerprint(tpm_b, state_b)
    assert fp_a == fp_b, f"{name}: CESes differ, not a counterexample"
    # Both members are reducible: the incompleteness lives outside the complexes.
    assert _raw_ces(tpm_a, state_a)[2] <= 1e-9
    assert _raw_ces(tpm_b, state_b)[2] <= 1e-9


def test_fingerprint_isomorphism_sanity():
    """Sanity: an isomorphic (node-swapped) pair shares a fingerprint and is iso."""
    tpm = np.array([[0, 0], [0, 0], [0, 1], [1, 0]], dtype=float)
    perm = (1, 0)  # swap the two nodes
    tpm_swapped = _relabel_tpm(tpm, perm)
    state, state_swapped = (1, 0), _relabel_state((1, 0), perm)
    assert _substrate_state_iso(tpm, state, tpm_swapped, state_swapped)
    assert _ces_fingerprint(tpm, state) == _ces_fingerprint(tpm_swapped, state_swapped)


def _exhaustive_n2(irreducible_only):
    """Group all valid n=2 deterministic (substrate, state) by CES fingerprint."""
    from pyphi import exceptions

    groups: dict = {}
    for flat in itertools.product((0.0, 1.0), repeat=8):
        tpm = np.array(flat, dtype=float).reshape(4, 2)
        for r in range(4):
            state = le_index2state(r, 2)
            try:
                dists, _rels, phi = _raw_ces(tpm, state)
            except exceptions.StateUnreachableError:
                continue
            except Exception:  # unreachable-backwards et al.
                continue
            if not dists or (irreducible_only and phi <= 1e-9):
                continue
            fp = _ces_fingerprint(tpm, state)
            iso_key = min(
                (np.round(_relabel_tpm(tpm, p), 12).tobytes(), _relabel_state(state, p))
                for p in itertools.permutations(range(2))
            )
            groups.setdefault(fp, set()).add(iso_key)
    return groups


@pytest.mark.slow
def test_exhaustive_n2_irreducible_systems_appear_complete():
    """n=2 exhaustive: among complexes (Phi_s > 0) no two non-iso substrates
    share a CES, but among all systems they do."""
    irreducible = _exhaustive_n2(irreducible_only=True)
    irreducible_cex = [fp for fp, iso in irreducible.items() if len(iso) > 1]
    assert irreducible_cex == [], (
        f"unexpected counterexample among irreducible n=2 systems: {irreducible_cex}"
    )

    all_systems = _exhaustive_n2(irreducible_only=False)
    all_cex = [fp for fp, iso in all_systems.items() if len(iso) > 1]
    assert all_cex, "expected CES counterexamples among reducible n=2 systems"
