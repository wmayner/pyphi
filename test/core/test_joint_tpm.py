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


def _heterogeneous_view():
    """A 2-unit view with alphabets (2, 3): unit A binary, unit B ternary."""
    rng = np.random.default_rng(42)

    def rand_dist(shape):
        x = rng.random(shape)
        return x / x.sum(axis=-1, keepdims=True)

    from pyphi.core.tpm.factored import FactoredTPM

    ft = FactoredTPM(
        factors=[rand_dist((2, 3, 2)), rand_dist((2, 3, 3))],
        node_labels=("A", "B"),
    )
    return JointTPM(ft.to_joint(), node_labels=("A", "B"))


def test_conditioned_view_keeps_true_alphabet_sizes():
    # Conditioning collapses input axes to singletons; the output alphabets
    # are unchanged and must still be reported.
    v = _heterogeneous_view()
    conditioned = v.condition({1: 2})
    assert conditioned.shape == (2, 1, 2, 3)
    assert conditioned.alphabet_sizes == (2, 3)


def test_conditioned_view_to_pandas_conserves_probability_mass():
    # Each unit's next-state distribution must sum to 1 for every remaining
    # input state; deriving alphabets from the collapsed input axes truncated
    # the ternary unit's rows to next_state == 0.
    v = _heterogeneous_view()
    df = v.condition({1: 2}).to_pandas()
    sums = df.groupby([df["state"].astype(str), "unit"])["probability"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)
    b_rows = df[df["unit"] == "B"]
    assert sorted(b_rows["next_state"].unique()) == [0, 1, 2]


def test_conditioned_view_grid_section_lists_full_alphabet():
    v = _heterogeneous_view()
    grid = v.condition({1: 2}).grid_section().body[0]
    assert "B=2" in grid.headers
    # Rows enumerate the remaining input states only.
    assert len(grid.rows) == 2


def test_view_buffer_is_read_only():
    # A documented read-only value type must not hand out a writable buffer:
    # external mutation would change its hash and equality.
    v = JointTPM(_joint(examples.basic_substrate()))
    assert not v.to_array().flags.writeable
    assert not np.asarray(v).flags.writeable
    with pytest.raises(ValueError):
        v.to_array()[(0,) * v.to_array().ndim] = 0.5


def test_view_to_xarray_binary():
    pytest.importorskip("xarray")
    joint = _joint(examples.basic_substrate())
    da = JointTPM(joint).to_xarray()
    assert da.dims == ("u0", "u1", "u2", "unit", "out")
    np.testing.assert_allclose(np.asarray(da), joint)
