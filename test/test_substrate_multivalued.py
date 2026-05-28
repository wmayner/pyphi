"""End-to-end k>2 IIT analysis smoke tests."""

from __future__ import annotations

import numpy as np

import pyphi
from pyphi import actual
from pyphi.formalism.queries import sia


def _k3_two_node_substrate(seed: int = 2026) -> pyphi.Substrate:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(size=(3, 3, 3))
    f0 = f0 / f0.sum(axis=-1, keepdims=True)
    f1 = rng.uniform(size=(3, 3, 3))
    f1 = f1 / f1.sum(axis=-1, keepdims=True)
    return pyphi.Substrate(
        marginals=[f0, f1],
        state_space=("LOW", "MID", "HIGH"),
    )


def test_kary_sia_end_to_end() -> None:
    """Construct a k=3 substrate; run SIA; receive a phi value."""
    sub = _k3_two_node_substrate()
    sys = pyphi.System(sub, state=(0, 0))
    result = sia(sys)
    assert result.phi >= 0
    assert result.partition is not None


def test_kary_state_as_labels_resolves() -> None:
    """SIA works when state is passed as labels."""
    sub = _k3_two_node_substrate()
    sys = pyphi.System(sub, state=("LOW", "LOW"))
    result = sia(sys)
    assert result.phi >= 0


def test_heterogeneous_alphabet_sia() -> None:
    """SIA works for a heterogeneous-alphabet substrate."""
    rng = np.random.default_rng(2026)
    f_binary = rng.uniform(size=(2, 3, 2))
    f_binary = f_binary / f_binary.sum(axis=-1, keepdims=True)
    f_ternary = rng.uniform(size=(2, 3, 3))
    f_ternary = f_ternary / f_ternary.sum(axis=-1, keepdims=True)
    sub = pyphi.Substrate(
        marginals=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    sys = pyphi.System(sub, state=(0, 0))
    result = sia(sys)
    assert result.phi >= 0


def test_kary_account_end_to_end() -> None:
    """Construct a k=3 substrate; compute an Account.

    AC inherits k-ary substrate support from ``System`` via
    ``TransitionSystem``'s delegation through ``_underlying_system``.
    """
    sub = _k3_two_node_substrate()
    transition = pyphi.Transition(
        substrate=sub,
        before_state=(0, 1),
        after_state=(1, 2),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    account = actual.account(transition)
    assert account is not None
    assert all(link.alpha >= 0 for link in account)
