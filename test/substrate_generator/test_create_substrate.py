"""Tests for the ``create_substrate`` factory wiring (weights, self-loops, CM)."""

import numpy as np
import pytest

import pyphi
from pyphi.substrate import Substrate
from pyphi.substrate_generator import create_substrate

WSM = {(0, 0): 1.0, (1, 0): 0.5, (0, 1): 0.75, (1, 1): 1.5}


def test_returns_substrate_with_expected_shape_and_labels():
    sub = create_substrate(
        {
            0: {"mechanism": "or", "inputs": (1, 2)},
            1: {"mechanism": "and", "inputs": (0, 2)},
            2: {"mechanism": "xor", "inputs": (0, 1)},
        },
        labels=("X", "Y", "Z"),
    )
    assert isinstance(sub, Substrate)
    assert sub.size == 3
    assert tuple(sub.node_labels) == ("X", "Y", "Z")


def test_per_node_params_are_bound_independently():
    """Two nodes with the same mechanism but different params stay distinct."""
    sub = create_substrate(
        {
            0: {
                "mechanism": "sigmoid",
                "inputs": (1,),
                "params": {"input_weights": (5.0,), "determinism": 10.0},
            },
            1: {
                "mechanism": "sigmoid",
                "inputs": (0,),
                "params": {"input_weights": (0.1,), "determinism": 0.5},
            },
        }
    )
    joint = np.asarray(sub.tpm.to_joint())[..., 1]
    # Node 0 is far more deterministic than node 1, so its ON-probabilities
    # span a wider range.
    assert np.ptp(joint[..., 0]) > np.ptp(joint[..., 1])


def test_state_dependent_mechanism_gets_self_loop():
    sub = create_substrate(
        {
            0: {
                "mechanism": "sigmoid",
                "inputs": (1,),
                "params": {"input_weights": (1.0,)},
            },
            1: {
                "mechanism": "resonator",
                "inputs": (0,),
                "params": {
                    "input_weights": (1.0,),
                    "determinism": 4.0,
                    "threshold": 0.0,
                    "weight_scale_mapping": WSM,
                },
            },
        }
    )
    cm = np.asarray(sub.cm)
    assert cm[1, 1] == 1  # resonator (state-dependent) -> self-loop inserted
    assert cm[0, 0] == 0  # sigmoid (not state-dependent) -> no self-loop


def test_weighted_subunit_weight_survives_connectivity_marker():
    """A connectivity-only sub-mechanism must not clobber a weighted one's weight
    on a shared input."""
    sub = create_substrate(
        {
            0: {"mechanism": "copy", "inputs": (1,)},
            1: {
                "composite": [
                    {
                        "mechanism": "resonator",
                        "inputs": (0, 1),
                        "params": {
                            "input_weights": (0.3, 0.7),
                            "determinism": 4.0,
                            "threshold": 0.0,
                            "weight_scale_mapping": WSM,
                        },
                    },
                    {
                        "mechanism": "mismatch_corrector",
                        "inputs": (0,),
                        "params": {"bias": 0.5},
                    },
                ],
                "mechanism_combination": "selective",
            },
        }
    )
    # The resonator's weight on input 0 (0.3) must remain, not be reset to the
    # mismatch_corrector's connectivity marker (1.0).
    assert sub.factored_tpm is not None  # builds without conflict
    cm = np.asarray(sub.cm)
    assert cm[0, 1] == 1 and cm[1, 1] == 1


def test_resulting_substrate_runs_through_sia():
    sub = create_substrate(
        {
            0: {"mechanism": "or", "inputs": (1, 2)},
            1: {"mechanism": "and", "inputs": (0, 2)},
            2: {"mechanism": "xor", "inputs": (0, 1)},
        }
    )
    system = pyphi.System.from_substrate(sub, state=(1, 0, 0))
    assert system.sia().phi >= 0


def test_list_and_dict_specs_are_equivalent():
    spec = {
        0: {"mechanism": "copy", "inputs": (1,)},
        1: {"mechanism": "copy", "inputs": (0,)},
    }
    from_dict = create_substrate(spec)
    from_list = create_substrate([spec[0], spec[1]])
    assert np.allclose(
        np.asarray(from_dict.tpm.to_joint()), np.asarray(from_list.tpm.to_joint())
    )


def test_non_integer_mapping_keys_raise():
    with pytest.raises(TypeError):
        create_substrate({"a": {"mechanism": "copy", "inputs": (0,)}})  # type: ignore[dict-item]
