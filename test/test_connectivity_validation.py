"""Tests for the CM/TPM connectivity-consistency validator (B19).

A substrate's connectivity matrix (CM) is user-supplied and independent of its
transition probabilities. If the CM *omits* an edge the TPM actually implies
(an "under-specified" CM), that input is silently marginalized out during node
construction and excluded from purview search, so φ is under-counted with no
error raised. The validator infers the true functional edge set from the
``FactoredTPM`` and rejects an under-specified CM at substrate construction.
Over-specification (declaring an unused edge) stays legal.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import JointTPM
from pyphi import Substrate
from pyphi import System
from pyphi import config
from pyphi import examples
from pyphi import utils
from pyphi.core.tpm.factored import FactoredTPM

# ============== Substrate builders (convention-safe via update functions) ==============


def _deterministic_substrate(alphabet, update, cm=None, state_space=None):
    """Build a deterministic substrate from per-node update functions.

    Each ``update[i]`` maps a full state tuple to node ``i``'s next value.
    Building factors by indexing ``np.ndindex`` sidesteps state-ordering
    conventions, so an axis/endianness bug cannot hide here.
    """
    marginals = []
    for i, k in enumerate(alphabet):
        factor = np.zeros((*alphabet, k))
        for st in np.ndindex(*alphabet):
            factor[(*st, update[i](*st))] = 1.0
        marginals.append(factor)
    if state_space is None:
        state_space = tuple(tuple(range(k)) for k in alphabet)
    return Substrate(marginals=marginals, state_space=state_space, cm=cm)


def _mutual_copy(cm=None):
    """2-node binary mutual-copy loop: node0' = node1, node1' = node0.

    Fully integrated; true edge set is the off-diagonal ``[[0, 1], [1, 0]]``.
    """
    return _deterministic_substrate(
        (2, 2),
        (lambda _s0, s1: s1, lambda s0, _s1: s0),
        cm=cm,
    )


# ============== Failing-first reproduction: the silent under-count ==============


def test_under_specified_cm_raises():
    """An under-specified CM (omitting a real edge) is rejected by default."""
    under_spec = np.array([[0, 0], [1, 0]])  # omits real edge 0 -> 1
    with pytest.raises(ValueError, match=r"(?i)connectivity|edge|under"):
        _mutual_copy(cm=under_spec)


def test_under_specified_cm_undercounts_phi():
    """With validation off, omitting a real edge silently under-counts φ.

    This is the bug B19 prevents: the correct (and over-specified all-ones) CM
    integrate to φ = 2.0, but omitting edge 0 -> 1 collapses φ to 0.
    """
    state = (1, 1)
    with config.override(validate_connectivity=False):
        all_ones = _mutual_copy(cm=np.ones((2, 2), dtype=int))
        true_cm = _mutual_copy(cm=np.array([[0, 1], [1, 0]]))
        under = _mutual_copy(cm=np.array([[0, 0], [1, 0]]))

        phi_all_ones = System(all_ones, state, all_ones.node_indices).sia().phi
        phi_true = System(true_cm, state, true_cm.node_indices).sia().phi
        phi_under = System(under, state, under.node_indices).sia().phi

        # Over-specification does not change φ: all-ones == true CM.
        assert utils.eq(phi_all_ones, phi_true)
        # Under-specification strictly under-counts φ.
        assert utils.is_positive(phi_true - phi_under)


# ============== Over-specification stays legal ==============


def test_over_specified_cm_allowed():
    """A CM declaring more edges than the TPM implies is accepted."""
    # gomez has a sparse true structure but its default CM is all-ones.
    examples.gomez_p53_mdm2_substrate()  # must not raise

    # Explicit over-specification on a binary net: true edges are off-diagonal,
    # declared CM adds non-existent self-loops.
    _mutual_copy(cm=np.ones((2, 2), dtype=int))  # must not raise


# ============== Opt-out + config plumbing ==============


def test_opt_out_allows_under_specified():
    """``validate_connectivity=False`` restores permissive construction."""
    under_spec = np.array([[0, 0], [1, 0]])
    with config.override(validate_connectivity=False):
        assert _mutual_copy(cm=under_spec) is not None


def test_validate_connectivity_default_on():
    assert config.infrastructure.validate_connectivity is True


def test_validate_connectivity_round_trips_through_override():
    with config.override(validate_connectivity=False):
        assert config.infrastructure.validate_connectivity is False
    assert config.infrastructure.validate_connectivity is True


def test_validate_connectivity_rejects_non_bool():
    with (
        pytest.raises((ValueError, TypeError)),
        config.override(validate_connectivity="yes"),
    ):
        pass


# ============== k-ary soundness (the binary-only trap) ==============


def _ternary_edge_only_at_2(cm=None):
    """2-node substrate: node0 ternary & constant; node1 binary, depends on
    node0 *only* via the difference between node0 = 0 and node0 = 2.

    A binary-only edge test (comparing only state 0 vs state 1 of the source)
    would miss edge 0 -> 1 entirely; the correct k-ary test catches it.
    """
    f0 = np.zeros((3, 2, 3))
    f0[..., 0] = 1.0  # node0 constant at state 0 (no inputs)
    f1 = np.zeros((3, 2, 2))
    f1[0, :, :] = [0.5, 0.5]  # node0 == 0
    f1[1, :, :] = [0.5, 0.5]  # node0 == 1 (identical to == 0)
    f1[2, :, :] = [0.9, 0.1]  # node0 == 2 (differs -> real edge 0 -> 1)
    return Substrate(
        marginals=[f0, f1],
        state_space=((0, 1, 2), (0, 1)),
        cm=cm,
    )


def test_kary_edge_visible_only_at_state_2_is_caught():
    """B19 rejects a CM omitting an edge a binary-only check would miss."""
    under_spec = np.zeros((2, 2), dtype=int)  # declares no edges; true: 0 -> 1
    with pytest.raises(ValueError, match=r"(?i)connectivity|edge|under"):
        _ternary_edge_only_at_2(cm=under_spec)


# ============== A1 regression: weakly-stochastic edge must be detected ==============


def test_weakly_stochastic_edge_detected():
    """A small-but-real dependence (~1e-6) is detected as an edge.

    Guards against using ``np.allclose`` with its default ``rtol=1e-5``, which
    would swallow a dependence below the relative tolerance and miss the edge —
    exactly the under-specification this validator must catch.
    """
    eps = 1e-6
    f0 = np.zeros((2, 2, 2))
    f0[..., 0] = 1.0  # node0 constant (no inputs)
    f1 = np.zeros((2, 2, 2))
    f1[0, :, :] = [0.5, 0.5]
    f1[1, :, :] = [0.5 - eps, 0.5 + eps]  # tiny real dependence on node0
    sub = FactoredTPM(factors=[f0, f1], state_space=((0, 1), (0, 1)))
    inferred = sub.infer_cm()
    assert inferred[0, 1] == 1  # edge 0 -> 1 detected despite tiny magnitude


# ============== FactoredTPM.infer_cm unit tests ==============


def test_infer_cm_binary_matches_legacy():
    """For binary substrates, infer_cm agrees with the legacy JointTPM path."""
    sub = _mutual_copy()
    factored_cm = sub.tpm.infer_cm()
    legacy_cm = JointTPM(sub._legacy_binary_joint()).infer_cm()
    assert np.array_equal(factored_cm, np.asarray(legacy_cm))
    assert np.array_equal(factored_cm, np.array([[0, 1], [1, 0]]))


def test_infer_cm_heterogeneous_alphabet_gomez():
    """infer_cm recovers the known sparse structure of the (3,2,2) p53 net.

    From the Table-3 update functions: P' depends on Mn; Mc' on P; Mn' on P and
    Mc. Node order (P, Mc, Mn) -> edges P->Mc, P->Mn, Mc->Mn, Mn->P.
    """
    sub = examples.gomez_p53_mdm2_substrate()
    expected = np.array(
        [
            [0, 1, 1],  # P -> Mc, P -> Mn
            [0, 0, 1],  # Mc -> Mn
            [1, 0, 0],  # Mn -> P
        ]
    )
    assert np.array_equal(sub.tpm.infer_cm(), expected)


def test_infer_cm_size_one_axis_is_non_input():
    """A genuine size-1 input axis yields no edge (no crash, no false edge)."""
    f0 = np.zeros((1, 2, 2))  # node0: axis0 size-1 (non-input), depends on node1
    f0[0, 0, :] = [1.0, 0.0]
    f0[0, 1, :] = [0.0, 1.0]  # varies along axis1 -> edge 1 -> 0
    f1 = np.zeros((1, 2, 2))  # node1: constant (no inputs)
    f1[..., 0] = 1.0
    sub = FactoredTPM(factors=[f0, f1], state_space=((0, 1), (0, 1)))
    inferred = sub.infer_cm()
    assert inferred[0, 0] == 0  # size-1 axis -> no self-edge
    assert inferred[1, 0] == 1  # edge 1 -> 0
    assert inferred[0, 1] == 0
    assert inferred[1, 1] == 0


def test_infer_cm_detects_self_loop():
    """A node depending on its own previous state is a diagonal edge."""
    sub = _deterministic_substrate((2, 2), (lambda s0, _s1: s0, lambda _s0, s1: s1))
    inferred = sub.tpm.infer_cm()
    assert inferred[0, 0] == 1  # node0 copies itself
    assert inferred[1, 1] == 1  # node1 copies itself
    assert inferred[0, 1] == 0
    assert inferred[1, 0] == 0
