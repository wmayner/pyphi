"""Golden tests: ``create_substrate`` reproduces the original ``substrate_modeler``.

The fixtures in ``test/data/substrate_generator/dynamic_tpms.npz`` are the
``dynamic_tpm`` of the substrate built by Bjørn Juel's original library for each
spec in :data:`GOLDEN_CONFIGS` (regenerate with
``scripts/gen_substrate_generator_goldens.py``). Here we rebuild each substrate
natively and assert the state-by-node TPM is byte-identical.
"""

from pathlib import Path

import numpy as np
import pytest

from pyphi.substrate_generator import build_tpm
from pyphi.substrate_generator import create_substrate
from pyphi.substrate_generator import mechanisms as M
from test.substrate_generator.golden_configs import GOLDEN_CONFIGS

FIXTURE = Path(__file__).parents[1] / "data" / "substrate_generator" / "dynamic_tpms.npz"
GOLDENS = np.load(FIXTURE)


@pytest.mark.parametrize("config_id", list(GOLDEN_CONFIGS))
def test_create_substrate_matches_substrate_modeler(config_id):
    reference = GOLDENS[config_id]
    substrate = create_substrate(GOLDEN_CONFIGS[config_id])
    # State-by-node ON-probabilities: last axis is the binary alphabet.
    native = np.asarray(substrate.tpm.to_joint())[..., 1]
    assert native.shape == reference.shape
    assert np.allclose(native, reference, atol=1e-12)


def test_every_paper_mechanism_has_a_golden():
    """The matching-paper mechanisms must all appear in a golden config."""
    used = set()
    for node_params in GOLDEN_CONFIGS.values():
        for spec in node_params.values():
            subs = spec.get("composite", [spec])
            used.update(sub["mechanism"] for sub in subs)
    assert {"sigmoid", "resonnator", "mismatch_corrector", "sor"} <= used


def test_mismatch_pattern_detector_intended_behavior():
    """The original ``mismatch_pattern_detector`` is broken (a ``Nonee`` typo),
    so there is no golden; assert this port's documented behavior directly.

    Unit ON: weak to matching patterns, strong (floor) to non-patterns.
    Unit OFF: strong (ceiling) to patterns, weak to non-patterns.
    """
    weights = np.zeros((3, 3))
    weights[1, 0] = weights[2, 0] = 1.0
    weights[0, 0] = 1.0  # self-loop (state-dependent)
    patterns = [(1, 0)]

    def p(state):
        return M.mismatch_pattern_detector(
            0,
            weights,
            state,
            pattern_selection=patterns,
            selectivity=2.0,
            floor=0.0,
            ceiling=1.0,
            inputs=(1, 2),
        )

    # Unit OFF (state[0] == 0): pattern -> ceiling, non-pattern -> weak.
    assert p((0, 1, 0)) == 1.0
    assert p((0, 0, 0)) == pytest.approx(0.5 - 0.5 / 2.0)
    # Unit ON (state[0] == 1): pattern -> weak, non-pattern -> floor.
    assert p((1, 1, 0)) == pytest.approx(0.5 + 0.5 / 2.0)
    assert p((1, 0, 0)) == 0.0


def test_build_tpm_uniform_mechanism_shape():
    """A uniform-mechanism substrate via build_tpm has state-by-node shape."""
    weights = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    tpm = build_tpm("sigmoid", weights, determinism=4.0)
    assert tpm.shape == (2, 2, 2, 3)
    assert np.all((tpm >= 0) & (tpm <= 1))
