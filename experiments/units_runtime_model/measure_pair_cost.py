"""Measure the CPU cost of one (mechanism, direction, purview) MIP search.

The campaign cost model charges ``1 + partition_count(|m|, |p|)`` "units"
per pair. This script measures what a pair actually costs, so the charge
can be regressed against it.

Records per pair: the model's unit charge, the number of
``evaluate_partition`` calls actually made, the number of tied specified
states the IIT 4.0 MIP search sweeps partitions for, whether the zero-φ
short-circuit fired, and CPU seconds.

Usage
-----
    uv run python experiments/units_runtime_model/measure_pair_cost.py \
        --units 14 --samples 3 --seed 1
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path(__file__).parent / "results"


def ring_ising(units: int, radius: int, temperature: float) -> Any:
    """Periodic 1-D Ising chain coupling each unit to ``2 * radius`` neighbors."""
    from pyphi.substrate_generator import build_substrate
    from pyphi.substrate_generator import ising

    weights = np.zeros((units, units))
    for i in range(units):
        for d in range(-radius, radius + 1):
            if d:
                weights[i, (i + d) % units] = 1.0
    return build_substrate([ising.probability] * units, weights, temperature=temperature)


def unique_path(path: Path) -> Path:
    """``path`` if free, else the same stem with the first free ``_vN`` suffix."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def measure(
    units: int,
    radius: int,
    temperature: float,
    max_order: int,
    samples: int,
    seed: int,
    max_mechanism_order: int,
) -> list[dict]:
    from pyphi.cache import registry
    from pyphi.conf import config
    from pyphi.conf import presets
    from pyphi.cost import partition_sweep_count
    from pyphi.direction import Direction
    from pyphi.formalism import queries
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.system import System

    rng = np.random.default_rng(seed)
    substrate = ring_ising(units, radius, temperature)
    rows: list[dict] = []

    # Count the partition evaluations the sweep actually performs. The
    # closure in _find_mip_single_state resolves this module global, so
    # wrapping it here counts every call the MIP search makes.
    real_evaluate = queries.evaluate_partition
    calls = 0

    def counting_evaluate(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_evaluate(*args, **kwargs)

    queries.evaluate_partition = counting_evaluate
    try:
        with config.override(
            **presets.by_name["IIT_4_0_2026"], progress_bars=False, parallel=False
        ):
            system = System.from_substrate(substrate, (0,) * units)
            specification_measure = resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            )
            for msize in range(1, max_mechanism_order + 1):
                pool = [
                    tuple(sorted(rng.choice(units, size=msize, replace=False).tolist()))
                    for _ in range(samples * 4)
                ]
                seen: set[tuple[int, ...]] = set()
                mechanisms = [
                    m
                    for m in pool
                    if not (m in seen or seen.add(m))  # type: ignore[func-returns-value]
                ][:samples]
                for mechanism in mechanisms:
                    for direction in (Direction.CAUSE, Direction.EFFECT):
                        purviews = system.potential_purviews(
                            direction, mechanism, max_order=max_order
                        )
                        by_size: dict[int, tuple[int, ...]] = {}
                        for purview in purviews:
                            by_size.setdefault(len(purview), purview)
                        for psize, purview in sorted(by_size.items()):
                            registry.clear_all()
                            calls = 0
                            start = time.process_time()
                            ria = queries.find_mip(system, direction, mechanism, purview)
                            cpu = time.process_time() - start
                            evaluations = calls
                            n_states = len(
                                system.intrinsic_information(
                                    direction,
                                    mechanism,
                                    purview,
                                    specification_measure=specification_measure,
                                ).ties
                            )
                            model_units = 1 + partition_sweep_count(msize, psize)
                            rows.append(
                                {
                                    "units": units,
                                    "mechanism": list(mechanism),
                                    "msize": msize,
                                    "direction": direction.name,
                                    "purview": list(purview),
                                    "psize": psize,
                                    "model_units": model_units,
                                    "partition_count": model_units - 1,
                                    "n_states": n_states,
                                    "evaluations": evaluations,
                                    "cpu_s": cpu,
                                    "phi": float(ria.phi),
                                }
                            )
    finally:
        queries.evaluate_partition = real_evaluate
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=14)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--max-mechanism-order", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    rows = measure(
        args.units,
        args.radius,
        args.temperature,
        args.max_order,
        args.samples,
        args.seed,
        args.max_mechanism_order,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    label = f"_{args.run_label}" if args.run_label else ""
    name = (
        f"pair_cost_n{args.units}_r{args.radius}_o{args.max_order}"
        f"_m{args.max_mechanism_order}_s{args.samples}_seed{args.seed}{label}.json"
    )
    path = unique_path(RESULTS / name)
    path.write_text(
        json.dumps(
            {
                "params": vars(args),
                "platform": {
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                },
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
