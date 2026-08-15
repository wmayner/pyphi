"""Large fixtures for the performance harness only.

These are not golden-regression fixtures: no φ values are stored for them and
they are absent from ``ALL_FIXTURES``. They exist because the golden zoo tops
out at four units, while several costs in PyPhi grow with the size of the
*whole system* rather than with the size of a mechanism — the specified-state
search sweeps every state of the system, so its work grows as 4ⁿ. A cost that
only appears above eight units is invisible to a suite that never goes there.

Each substrate is a seeded noisy ring: unit *i* takes inputs from itself and
its two neighbours, and its ON probability for each input configuration is
drawn once from a fixed generator, so the substrate is exactly reproducible
from ``(n, seed)``. Sparse connectivity keeps the substrate honest about
purview pruning while leaving the system-wide sweeps at full size.
"""

from __future__ import annotations

import numpy as np

from pyphi import Substrate
from pyphi.conf import presets

from .fixture import GoldenFixture

# Layers whose cost is superexponential in system size. A ten-unit system
# irreducibility analysis or cause-effect structure is out of reach; the
# specified-state sweep is not, and it is the layer these fixtures exist for.
_LARGE_SYSTEM_SKIPS = frozenset(
    {"repertoires", "mechanism_mips", "sia", "phi_structure"}
)


def noisy_ring_arrays(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The state-by-node TPM and connectivity matrix of a seeded noisy ring."""
    rng = np.random.default_rng(seed)
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in (i, (i - 1) % n, (i + 1) % n):
            cm[j, i] = 1
    states = np.arange(2**n)
    columns = []
    for i in range(n):
        inputs = [j for j in range(n) if cm[j, i]]
        factor = rng.uniform(0.05, 0.95, size=2 ** len(inputs))
        index = np.zeros(2**n, dtype=int)
        for bit, j in enumerate(inputs):
            index |= ((states >> j) & 1) << bit
        columns.append(factor[index])
    return np.stack(columns, axis=1), cm


def noisy_ring(n: int, seed: int) -> Substrate:
    """A seeded ``n``-unit ring in which each unit sees itself and its neighbours."""
    tpm, cm = noisy_ring_arrays(n, seed)
    return Substrate(tpm, cm=cm)


def _ring_fixture(n: int, seed: int, *, slow: bool) -> GoldenFixture:
    return GoldenFixture(
        name=f"ring{n}_iit4_2026",
        config_overrides=dict(presets.iit4_2026),
        substrate_factory=lambda: noisy_ring(n, seed),
        state=(0,) * n,
        description=(
            f"Seeded {n}-unit noisy ring. Performance harness only: exercises "
            "the specified-state sweep at a system size the golden zoo never "
            "reaches."
        ),
        skip_layers=_LARGE_SYSTEM_SKIPS,
        slow=slow,
    )


PERF_ONLY_FIXTURES: list[GoldenFixture] = [
    _ring_fixture(10, seed=2026, slow=False),
    _ring_fixture(12, seed=2027, slow=True),
]
