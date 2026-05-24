"""K-ary cause/effect marginalization tests."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import cause_tpm


def _k3_two_node_uniform() -> FactoredTPM:
    """A small k=3 2-node uniform FactoredTPM.

    Each factor has shape ``(3, 3, 3)`` — uniform alphabet 3, 2 input nodes,
    trailing axis of size 3 is this factor's per-node alphabet.
    """
    f = np.full((3, 3, 3), 1.0 / 3.0)
    return FactoredTPM(factors=[f, f.copy()], alphabet_sizes=(3, 3))


def test_cause_tpm_k3_returns_cause_posterior() -> None:
    """k=3 cause_tpm returns a CausePosterior (no NotImplementedError)."""
    factored = _k3_two_node_uniform()
    result = cause_tpm(factored, state=(0, 0), node_indices=(0, 1))
    assert isinstance(result, CausePosterior)


def test_cause_tpm_k3_sums_to_one() -> None:
    """k=3 native cause posterior is a valid joint distribution: sums to 1."""
    factored = _k3_two_node_uniform()
    result = cause_tpm(factored, state=(0, 0), node_indices=(0, 1))
    assert np.isclose(np.asarray(result).sum(), 1.0, atol=1e-12)
