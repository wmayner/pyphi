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
