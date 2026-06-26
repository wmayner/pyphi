"""Tests for k-ary (non-binary) Earth Mover's Distance.

The EMD repertoire-distance was historically binary-only (its ground metric
hard-coded a ``2^N`` Hamming matrix). These tests cover the generalization to
arbitrary, possibly heterogeneous, alphabets:

* binary results are byte-identical (the binary path is untouched);
* the cause-direction full EMD (``hamming_emd``) matches hand-computed values
  and is a metric (symmetric, zero on the diagonal);
* the effect-direction analytic shortcut (``effect_emd``) equals the full EMD on
  product (effect) repertoires for any alphabet -- the two implementations check
  each other; and
* EMD now runs as the IIT 3.0 mechanism measure on a multi-valued substrate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.measures.distribution import _ground_metric
from pyphi.measures.distribution import _hamming_matrix
from pyphi.measures.distribution import effect_emd
from pyphi.measures.distribution import hamming_emd
from pyphi.system import System


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / v.sum()


def _product(marginals: list[np.ndarray]) -> np.ndarray:
    """Build the joint product distribution from independent marginals."""
    joint = marginals[0]
    for m in marginals[1:]:
        joint = np.tensordot(joint, m, axes=0)
    return joint


def test_ground_metric_binary_is_byte_identical():
    """The binary ground metric still delegates to the cached ``_hamming_matrix``."""
    for n in range(1, 5):
        assert np.array_equal(_ground_metric((2,) * n), _hamming_matrix(n))


def test_kary_ground_metric_counts_differing_positions():
    """The k-ary ground metric is the number of nodes whose state differs."""
    # Single ternary node: all distinct states are Hamming distance 1 apart.
    assert np.array_equal(
        _ground_metric((3,)),
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    )


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0),  # all mass moves across the node
        ([0.5, 0.5, 0.0], [0.0, 0.5, 0.5], 0.5),  # half the mass moves
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0),  # identical
    ],
)
def test_single_ternary_node_known_values(p, q, expected):
    """Hand-computed EMD on a single ternary node (Hamming ground metric)."""
    p, q = np.array(p), np.array(q)
    assert hamming_emd(p, q) == pytest.approx(expected)
    assert effect_emd(p, q) == pytest.approx(expected)


def test_emd_is_symmetric_and_zero_on_identity():
    rng = np.random.default_rng(7)
    shape = (3, 2, 3)
    a = _product([_normalize(rng.random(k)) for k in shape])
    b = _product([_normalize(rng.random(k)) for k in shape])
    assert hamming_emd(a, b) == pytest.approx(hamming_emd(b, a))
    assert hamming_emd(a, a) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(25))
def test_effect_shortcut_equals_full_emd_for_kary_products(seed):
    """For product (effect) repertoires, the analytic ``effect_emd`` equals the
    full Wasserstein distance from ``hamming_emd``, for any alphabet.

    This is exact (Kantorovich duality with additive 1-Lipschitz potentials), so
    the two independent implementations validate each other to machine epsilon.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(rng.integers(2, 5)) for _ in range(int(rng.integers(1, 4))))
    p = _product([_normalize(rng.random(k)) for k in shape])
    q = _product([_normalize(rng.random(k)) for k in shape])
    assert effect_emd(p, q) == pytest.approx(hamming_emd(p, q), abs=1e-12)


def test_kary_emd_runs_as_iit3_mechanism_measure():
    """EMD computes a system phi as the IIT 3.0 mechanism measure on a k>2
    substrate (the multi-valued p53-Mdm2 network), which previously raised."""
    with config.override(
        **{k: v for k, v in presets.iit3.items() if k != "iit"},
        iit=replace(
            presets.iit3["iit"],
            mechanism_phi_measure="EMD",
            ces_measure="SUM_SMALL_PHI",
            mechanism_partition_scheme="WEDGE_TRIPARTITION",
        ),
        validate_system_states=False,
        progress_bars=False,
    ):
        system = System(
            examples.gomez_p53_mdm2_substrate(), (0, 0, 1), node_indices=(0, 1, 2)
        )
        assert float(system.sia().phi) == pytest.approx(0.5, abs=1e-3)
