import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import automorphism as auto
from test import example_substrates as es


def test_relabel_joint_maps_and_xor_to_xor_and():
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    relabeled = auto._relabel_joint(s_ax.tpm.to_joint(), (1, 0))
    assert np.array_equal(relabeled, s_xa.tpm.to_joint())


def test_relabel_joint_identity_is_noop():
    s = es.and_xor_substrate()
    arr = s.tpm.to_joint()
    assert np.array_equal(auto._relabel_joint(arr, (0, 1)), arr)


def test_candidate_perms_prunes_by_alphabet_and_cm():
    # and_xor: fully connected, uniform alphabet -> both perms are candidates
    s = es.and_xor_substrate()
    assert set(auto._candidate_perms(s)) == {(0, 1), (1, 0)}


def test_automorphisms_identity_always_present():
    for s in (es.and_xor_substrate(), es.xor_and_substrate()):
        autos = auto.substrate_automorphisms(s)
        assert tuple(range(len(s.tpm.alphabet_sizes))) in autos


def test_automorphisms_distinct_gates_have_only_identity():
    # Fully connected but AND != XOR: no nontrivial automorphism.
    s = es.and_xor_substrate()
    assert auto.substrate_automorphisms(s) == ((0, 1),)


def test_automorphisms_preserve_tpm():
    s = es.and_xor_substrate()
    arr = s.tpm.to_joint()
    for perm in auto.substrate_automorphisms(s):
        assert np.array_equal(auto._relabel_joint(arr, perm), arr)


def test_automorphisms_recover_known_symmetries():
    # Three interchangeable nodes -> full symmetric group S_3 (6 perms).
    triple = auto.substrate_automorphisms(es.symmetric_triple_substrate())
    assert len(triple) == 6
    # Two identical AND-XOR blocks -> identity + the block swap.
    dual = set(auto.substrate_automorphisms(es.dual_and_xor_substrate()))
    assert dual == {(0, 1, 2, 3), (2, 3, 0, 1)}


def test_canonical_form_invariant_under_relabeling():
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    canon_ax, _ = auto.substrate_canonical_form(s_ax)
    canon_xa, _ = auto.substrate_canonical_form(s_xa)
    assert canon_ax == canon_xa


def test_canonical_permutation_maps_to_canonical_form():
    s = es.xor_and_substrate()
    canon, perm = auto.substrate_canonical_form(s)
    relabeled = auto._relabel_joint(s.tpm.to_joint(), perm)
    assert np.array_equal(relabeled, canon.tpm.to_joint())


def test_isomorphic_pair_and_nonisomorphic_pair():
    assert auto.are_substrates_isomorphic(es.and_xor_substrate(), es.xor_and_substrate())
    assert auto.are_substrates_isomorphic(es.and_xor_substrate(), es.and_xor_substrate())
    # Different node counts -> not isomorphic (2 nodes vs 3 nodes).
    assert not auto.are_substrates_isomorphic(
        es.and_xor_substrate(), es.symmetric_triple_substrate()
    )


def test_isomorphism_symmetric():
    a, b = es.and_xor_substrate(), es.xor_and_substrate()
    assert auto.are_substrates_isomorphic(a, b) == auto.are_substrates_isomorphic(b, a)


def test_canonical_state_linchpin():
    # The two permuted substrates' corresponding states must canonicalize equal.
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    assert auto.canonical_state(s_ax, (0, 1)) == auto.canonical_state(s_xa, (1, 0))


def test_canonical_state_idempotent_on_canonical_substrate():
    s = es.and_xor_substrate()
    canon, perm = auto.substrate_canonical_form(s)
    state = (1, 0)
    canon_state = tuple(state[perm[i]] for i in range(len(perm)))
    assert auto.canonical_state(canon, canon_state) == auto.canonical_state(s, state)


@settings(max_examples=50)
@given(bits=st.lists(st.integers(0, 1), min_size=2, max_size=2))
def test_canonical_state_orbit_invariant(bits):
    # and_xor and xor_and are related by sigma=(1,0): state s on and_xor
    # corresponds to s'[i]=s[sigma[i]] on xor_and.
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    sigma = (1, 0)
    s = tuple(bits)
    s_prime = tuple(s[sigma[i]] for i in range(2))
    assert auto.canonical_state(s_ax, s) == auto.canonical_state(s_xa, s_prime)
