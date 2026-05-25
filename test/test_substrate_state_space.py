"""Tests for Substrate's state_space and alphabet= keyword parameters."""

from __future__ import annotations

import numpy as np
import pytest

import pyphi


def _k3_marginals():
    f = np.full((3, 3, 3), 1.0 / 3.0)
    return [f, f.copy()]


def test_substrate_default_state_space_binary() -> None:
    """Binary substrate via tpm= has default integer state_space."""
    joint = np.full((2, 2, 2), 0.5)
    sub = pyphi.Substrate(tpm=joint)
    assert sub.state_space == ((0, 1), (0, 1))


def test_substrate_state_space_uniform_string_labels() -> None:
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("LOW", "MID", "HIGH"))
    assert sub.state_space == (("LOW", "MID", "HIGH"), ("LOW", "MID", "HIGH"))


def test_substrate_state_space_per_node_heterogeneous() -> None:
    f_binary = np.full((2, 3, 2), 0.5)
    f_ternary = np.full((2, 3, 3), 1.0 / 3.0)
    sub = pyphi.Substrate(
        marginals=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    assert sub.state_space == (("OFF", "ON"), ("LOW", "MID", "HIGH"))


def test_substrate_alphabet_shortcut() -> None:
    """alphabet=k is sugar for state_space=tuple(range(k))."""
    sub = pyphi.Substrate(marginals=_k3_marginals(), alphabet=3)
    assert sub.state_space == ((0, 1, 2), (0, 1, 2))


def test_substrate_alphabet_and_state_space_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match=r"alphabet.*state_space|state_space.*alphabet"):
        pyphi.Substrate(
            marginals=_k3_marginals(),
            alphabet=3,
            state_space=("L", "M", "H"),
        )


def test_substrate_alphabet_sizes_kwarg_removed() -> None:
    """alphabet_sizes is no longer a Substrate constructor kwarg."""
    joint = np.full((2, 2, 2), 0.5)
    with pytest.raises(TypeError):
        pyphi.Substrate(tpm=joint, alphabet_sizes=(2, 2))  # type: ignore[call-arg]


def test_substrate_state_space_delegates_to_factored_tpm() -> None:
    """Substrate.state_space is a delegated property."""
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("L", "M", "H"))
    assert sub.state_space == sub.factored_tpm.state_space


def test_system_state_as_labels_resolves_to_indices() -> None:
    """System(state=labels) resolves to System(state=int_indices) via state_space."""
    # 2 nodes, each with 3-state alphabet {"L", "M", "H"}
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("L", "M", "H"))
    # State has one entry per node (2), not per alphabet-size (3).
    # Disable state-reachability validation — that check is binary-only and
    # is a separate concern from the label-to-index coercion tested here.
    with pyphi.config.override(validate_system_states=False):
        sys_via_labels = pyphi.System(sub, state=("L", "M"))
        sys_via_indices = pyphi.System(sub, state=(0, 1))
    assert sys_via_labels.state == sys_via_indices.state
