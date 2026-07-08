"""Tests for the JointTPM view (the joint peer of FactoredTPM)."""

import numpy as np
import pytest

from pyphi import examples
from pyphi.core.tpm import TPM
from pyphi.core.tpm.joint import JointTPM


def _joint(sub):
    return np.asarray(sub.factored_tpm.to_joint())


def test_view_satisfies_protocol():
    v = JointTPM(_joint(examples.basic_substrate()))
    assert isinstance(v, TPM)


def test_view_metadata_binary():
    sub = examples.basic_substrate()
    v = JointTPM(_joint(sub))
    assert v.n_nodes == 3
    assert v.alphabet_sizes == (2, 2, 2)
    assert v.shape == (2, 2, 2, 3, 2)


def test_view_to_array_roundtrip():
    joint = _joint(examples.basic_substrate())
    v = JointTPM(joint)
    np.testing.assert_array_equal(v.to_array(), joint)
    np.testing.assert_array_equal(np.asarray(v), joint)


def test_view_is_eager_snapshot():
    joint = _joint(examples.basic_substrate()).copy()
    v = JointTPM(joint)
    joint[:] = 0.0  # mutate the source
    assert not np.all(v.to_array() == 0.0)  # snapshot is decoupled


def test_view_kary_metadata():
    sub = examples.gomez_p53_mdm2_substrate()  # alphabets (3, 2, 2)
    v = JointTPM(_joint(sub))
    assert v.alphabet_sizes == (3, 2, 2)
    assert v.n_nodes == 3


def test_view_equality_and_hash():
    joint = _joint(examples.basic_substrate())
    assert JointTPM(joint) == JointTPM(joint)
    assert hash(JointTPM(joint)) == hash(JointTPM(joint))


def test_view_condition_fixes_input_axis():
    sub = examples.basic_substrate()
    v = JointTPM(_joint(sub))
    conditioned = v.condition({0: 1})
    assert isinstance(conditioned, JointTPM)
    assert conditioned.shape[0] == 1  # input axis 0 collapsed to a singleton


def test_view_displayable_card_binary():
    v = JointTPM(_joint(examples.basic_substrate()))
    r = repr(v)
    assert r.startswith("╭")
    assert "JointTPM" in r
    assert "P(next unit on | current state)" in r
    assert "pyphi-grid" in v._repr_html_()


def test_view_to_xarray_binary():
    pytest.importorskip("xarray")
    joint = _joint(examples.basic_substrate())
    da = JointTPM(joint).to_xarray()
    assert da.dims == ("u0", "u1", "u2", "unit", "out")
    np.testing.assert_allclose(np.asarray(da), joint)
