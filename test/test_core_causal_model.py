"""Tests for pyphi.core.causal_model — CausalModel dataclass."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def test_causal_model_is_frozen() -> None:
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel

    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cm.tpm = None  # type: ignore[misc]


def test_causal_model_from_network_round_trips() -> None:
    """CausalModel.from_network preserves TPM and connectivity exactly."""
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel

    network = examples.basic_network()
    cm = CausalModel.from_network(network)
    np.testing.assert_array_equal(cm.tpm.to_array(), np.asarray(network.tpm))
    np.testing.assert_array_equal(cm.substrate.connectivity_matrix, network.cm)


def test_causal_model_substrate_units() -> None:
    """from_network produces one Unit per network node, in index order."""
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel

    network = examples.basic_network()
    cm = CausalModel.from_network(network)
    assert cm.substrate.n_units == network.size
    indices = [u.index for u in cm.substrate.units]
    assert indices == list(range(network.size))


def test_causal_model_equality() -> None:
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel

    a = CausalModel.from_network(examples.basic_network())
    b = CausalModel.from_network(examples.basic_network())
    assert a == b
    c = CausalModel.from_network(examples.xor_network())
    assert a != c
