"""Serialize the P17 synthesized 6-7 node fixtures (seed + weights + TPM + CM).

The four ``synth_n{6,7}_{sparse,dense}`` networks are built from fixed seeds by
``harness._synth_system``. This writes each one's generative inputs (seed,
density) and resulting arrays (weights, connectivity, TPM) to
``results/synth_fixtures/{name}.json`` so a run is reproducible from committed
data and the seed lives alongside it (the reproducibility rule). Re-running
this script reproduces byte-identical JSON.

Run (post checkout):  uv run python -m benchmarks.iit_3_vs_4.synth_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.iit_3_vs_4.harness import _SYNTH_COUPLING_MEAN
from benchmarks.iit_3_vs_4.harness import _SYNTH_COUPLING_SD
from benchmarks.iit_3_vs_4.harness import _SYNTH_TEMPERATURE
from benchmarks.iit_3_vs_4.harness import NETWORKS

# Generative parameters, kept in sync with the harness registry entries.
_SPECS = {
    "synth_n6_sparse": {"n": 6, "density": 0.35, "seed": 6001},
    "synth_n6_dense": {"n": 6, "density": 0.85, "seed": 6002},
    "synth_n7_sparse": {"n": 7, "density": 0.30, "seed": 7001},
    "synth_n7_dense": {"n": 7, "density": 0.80, "seed": 7002},
}

_OUT_DIR = Path(__file__).parent / "results" / "synth_fixtures"


def _rebuild_weights(n: int, density: float, seed: int) -> np.ndarray:
    """Reproduce the weight matrix ``_synth_system`` draws for this fixture."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, True)
    return mask * rng.normal(_SYNTH_COUPLING_MEAN, _SYNTH_COUPLING_SD, size=(n, n))


def write_fixtures() -> list[Path]:
    """Write every synthesized fixture JSON; return the written paths."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    from pyphi.substrate_generator import build_tpm

    for name, spec in _SPECS.items():
        system = NETWORKS[name].builder()
        weights = _rebuild_weights(spec["n"], spec["density"], spec["seed"])
        tpm = build_tpm("ising", weights, temperature=_SYNTH_TEMPERATURE)
        record = {
            "name": name,
            "n": spec["n"],
            "density": spec["density"],
            "seed": spec["seed"],
            "temperature": _SYNTH_TEMPERATURE,
            "coupling_mean": _SYNTH_COUPLING_MEAN,
            "coupling_sd": _SYNTH_COUPLING_SD,
            "unit_function": "ising",
            "state": list(system.state),
            "weights": np.asarray(weights).tolist(),
            "cm": np.asarray(system.substrate.cm).astype(int).tolist(),
            "tpm": np.asarray(tpm).tolist(),
        }
        path = _OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def main() -> int:
    for path in write_fixtures():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
