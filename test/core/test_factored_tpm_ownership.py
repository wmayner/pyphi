"""FactoredTPM storage ownership: stored factors are read-only and immune
to post-construction mutation of the caller's arrays."""

import numpy as np
import pytest

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.substrate import Substrate


def _uniform_factors():
    """Two binary nodes, each factor shape (2, 2, 2), uniform outputs."""
    return np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)


def test_writable_input_is_copied_and_frozen():
    f0, f1 = _uniform_factors()
    tpm = FactoredTPM([f0, f1])
    before = tpm.factor(0).copy()
    h = hash(tpm)
    f0[...] = 0.9  # caller mutates their own array after construction
    assert np.array_equal(tpm.factor(0), before)
    assert hash(tpm) == h
    assert not tpm.factor(0).flags.writeable
    assert f0.flags.writeable, "the caller's own array must stay writable"


def test_non_float64_input_is_frozen_and_detached():
    f0 = np.zeros((2, 2, 2), dtype=int)
    f0[..., 0] = 1
    f1 = np.zeros((2, 2, 2), dtype=int)
    f1[..., 1] = 1
    tpm = FactoredTPM([f0, f1])
    f0[...] = 7
    assert not tpm.factor(0).flags.writeable
    assert tpm.factor(0).max() <= 1.0


def test_read_only_input_is_stored_without_copy():
    f0, f1 = _uniform_factors()
    f0.flags.writeable = False
    f1.flags.writeable = False
    tpm = FactoredTPM([f0, f1])
    assert tpm.factor(0) is f0


def test_xarray_backend_factors_are_read_only():
    pytest.importorskip("xarray")
    f0, f1 = _uniform_factors()
    tpm = FactoredTPM([f0, f1], backend="xarray")
    assert not tpm.factor(0).flags.writeable
    f0[...] = 0.9
    assert float(tpm.factor(0).max()) == 0.5


def test_substrate_fingerprint_immune_to_caller_mutation():
    f0, f1 = _uniform_factors()
    sub = Substrate(marginals=[f0, f1])
    # Mutate before the (lazily cached) fingerprint is first computed: the
    # digest must reflect the values at construction, not the mutation.
    f0[...] = 0.9
    pristine = Substrate(marginals=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)])
    assert sub._fingerprint == pristine._fingerprint
