import numpy as np

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
