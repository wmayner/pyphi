"""How configuration options change PyPhi's cost.

Times the same stochastic-majority ring family (from :mod:`scaling`) under
different configuration settings, one knob at a time, to quantify how each
option changes runtime and scaling. Three knobs are compared:

- **relations** (IIT 4.0 cause–effect structure): ``CONCRETE`` enumeration of
  every relation versus the closed-form ``ANALYTICAL`` count and summed φ.
- **system cuts** (IIT 3.0 big Φ): the full ``2**n`` ``DIRECTED_BIPARTITION``
  sweep versus the ``2n`` cut-one approximation.
- **mechanism partition scheme** (IIT 4.0 distinctions, isolated by using
  analytical relations): ``JOINT_BIPARTITION`` versus ``WEDGE_TRIPARTITION``
  versus the default ``JOINT_PARTITION_ALL``.

Run::

    uv run python benchmarks/complexity/options.py --max-seconds 60

Outputs mirror :mod:`scaling`: raw per-trial JSON, aggregate CSV, and a figure
under ``benchmarks/complexity/results/``. Never run at docs-build time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pyphi

from benchmarks.complexity.scaling import RESULTS_DIR
from benchmarks.complexity.scaling import _versioned_path
from benchmarks.complexity.scaling import ring_substrate

PRESETS = {"iit3": pyphi.iit3, "iit4_2023": pyphi.iit4_2023}

# knob, config label, stage, base preset, IIT-field overrides. ``n_cap`` bounds
# a config that would otherwise run one very slow trial before the time budget
# trips (the concrete relation enumeration and the full IIT 3.0 cut sweep both
# blow up a step earlier than the reformulations they are compared against).
CONFIGS: list[dict[str, Any]] = [
    {"knob": "relations", "name": "concrete", "stage": "ces", "preset": "iit4_2023", "changes": {}, "n_cap": 5},
    {"knob": "relations", "name": "analytical", "stage": "ces", "preset": "iit4_2023", "changes": {"relation_computation": "ANALYTICAL"}, "n_cap": 5},
    {"knob": "system cuts", "name": "full 2**n", "stage": "sia", "preset": "iit3", "changes": {}, "n_cap": 5},
    {"knob": "system cuts", "name": "cut-one 2n", "stage": "sia", "preset": "iit3", "changes": {"system_partition_scheme": "DIRECTED_BIPARTITION_CUT_ONE"}, "n_cap": 6},
    {"knob": "mechanism scheme", "name": "JOINT_BIPARTITION", "stage": "ces", "preset": "iit4_2023", "changes": {"relation_computation": "ANALYTICAL", "mechanism_partition_scheme": "JOINT_BIPARTITION"}, "n_cap": 7},
    {"knob": "mechanism scheme", "name": "WEDGE_TRIPARTITION", "stage": "ces", "preset": "iit4_2023", "changes": {"relation_computation": "ANALYTICAL", "mechanism_partition_scheme": "WEDGE_TRIPARTITION"}, "n_cap": 6},
    {"knob": "mechanism scheme", "name": "JOINT_PARTITION_ALL", "stage": "ces", "preset": "iit4_2023", "changes": {"relation_computation": "ANALYTICAL", "mechanism_partition_scheme": "JOINT_PARTITION_ALL"}, "n_cap": 5},
]

# Stacked settings: which *combinations* buy the most units. Each group's first
# entry is the formalism default; later entries add one cost-reducing setting at
# a time. IIT 4.0 needs both a cheaper mechanism scheme and analytical relations;
# IIT 3.0 big Φ needs both cut-one and no-new-concepts.
COMBOS: list[dict[str, Any]] = [
    {"knob": "IIT 4.0 CES", "name": "default (all-partitions + concrete)", "stage": "ces", "preset": "iit4_2023", "changes": {}, "n_cap": 5},
    {"knob": "IIT 4.0 CES", "name": "+ analytical relations", "stage": "ces", "preset": "iit4_2023", "changes": {"relation_computation": "ANALYTICAL"}, "n_cap": 5},
    {"knob": "IIT 4.0 CES", "name": "+ bipartitions", "stage": "ces", "preset": "iit4_2023", "changes": {"mechanism_partition_scheme": "JOINT_BIPARTITION"}, "n_cap": 5},
    {"knob": "IIT 4.0 CES", "name": "bipartitions + analytical", "stage": "ces", "preset": "iit4_2023", "changes": {"mechanism_partition_scheme": "JOINT_BIPARTITION", "relation_computation": "ANALYTICAL"}, "n_cap": 7},
    {"knob": "IIT 3.0 big Φ", "name": "default (full cuts)", "stage": "sia", "preset": "iit3", "changes": {}, "n_cap": 5},
    {"knob": "IIT 3.0 big Φ", "name": "+ cut-one", "stage": "sia", "preset": "iit3", "changes": {"system_partition_scheme": "DIRECTED_BIPARTITION_CUT_ONE"}, "n_cap": 6},
    {"knob": "IIT 3.0 big Φ", "name": "+ no-new-concepts", "stage": "sia", "preset": "iit3", "changes": {"assume_partitions_cannot_create_new_concepts": True}, "n_cap": 6},
    {"knob": "IIT 3.0 big Φ", "name": "cut-one + no-new-concepts", "stage": "sia", "preset": "iit3", "changes": {"assume_partitions_cannot_create_new_concepts": True, "system_partition_scheme": "DIRECTED_BIPARTITION_CUT_ONE"}, "n_cap": 6},
]

CONFIG_SETS = {"knobs": CONFIGS, "combos": COMBOS}


@dataclass
class Trial:
    knob: str
    config: str
    stage: str
    n: int
    trial: int
    seconds: float


def _override(preset_name: str, changes: dict[str, Any]) -> dict[str, Any]:
    preset = dict(PRESETS[preset_name])
    preset["iit"] = dataclasses.replace(preset["iit"], **changes)
    preset["progress_bars"] = False
    return preset


def _time_once(substrate: Any, state: tuple[int, ...], stage: str, over: dict[str, Any]) -> float:
    start = time.perf_counter()
    with pyphi.config.override(**over):
        system = pyphi.System.from_substrate(substrate, state, substrate.node_indices)
        getattr(system, stage)()
    return time.perf_counter() - start


def run(
    configs: list[dict[str, Any]],
    max_seconds: float,
    beta: float,
    trials: int,
    n_min: int,
    n_max: int,
) -> list[Trial]:
    pyphi.config.progress_bars = False
    results: list[Trial] = []
    for cfg in configs:
        over = _override(cfg["preset"], cfg["changes"])
        cfg_n_max = min(n_max, cfg.get("n_cap", n_max))
        for n in range(n_min, cfg_n_max + 1):
            substrate = ring_substrate(n, beta)
            state = tuple([1] * n)
            times: list[float] = []
            over_budget = False
            for t in range(trials):
                try:
                    dt = _time_once(substrate, state, cfg["stage"], over)
                except Exception as exc:  # a scheme may be unsupported for a config
                    print(f"  {cfg['knob']}/{cfg['name']} n={n}: skipped ({exc})", flush=True)
                    times = []
                    break
                times.append(dt)
                results.append(Trial(cfg["knob"], cfg["name"], cfg["stage"], n, t, dt))
                if dt > max_seconds:
                    over_budget = True
                    break
            if not times:
                break
            median = float(np.median(times))
            print(f"{cfg['knob']:16s} {cfg['name']:20s} n={n}  median={median:8.3f}s", flush=True)
            if over_budget or median > max_seconds:
                print(f"  -> stop {cfg['knob']}/{cfg['name']}", flush=True)
                break
    return results


def save(results: list[Trial], beta: float, trials: int, tag: str) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(t) for t in results])
    label = f"{tag}_beta{beta:g}_trials{trials}"

    meta = {
        "beta": beta,
        "trials": trials,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    raw_path = _versioned_path(RESULTS_DIR / f"{label}.raw.json")
    raw_path.write_text(
        json.dumps({"meta": meta, "trials": [asdict(t) for t in results]}, indent=2)
    )

    agg = (
        df.groupby(["knob", "config", "stage", "n"])
        .agg(seconds_median=("seconds", "median"))
        .reset_index()
    )
    agg_path = _versioned_path(RESULTS_DIR / f"{label}.agg.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\nwrote {raw_path.name}, {agg_path.name}")
    return {"raw": raw_path, "agg": agg_path}


def plot(agg_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    agg = pd.read_csv(agg_path)
    knobs = list(dict.fromkeys(agg["knob"]))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(knobs), figsize=(5 * len(knobs), 4.5), squeeze=False)
    for ax, knob in zip(axes[0], knobs, strict=True):
        sub = pd.DataFrame(agg[agg["knob"] == knob])
        sns.lineplot(data=sub, x="n", y="seconds_median", hue="config", marker="o", ax=ax)
        ax.set_yscale("log")
        ax.set_title(knob)
        ax.set_xlabel("system size n")
        ax.set_ylabel("median seconds")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = _versioned_path(agg_path.with_suffix("").with_suffix(".png"))
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path.name}")
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["knobs", "combos"], default="knobs")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--plot-only", type=Path, default=None)
    args = parser.parse_args()

    if args.plot_only is not None:
        plot(args.plot_only)
        return

    results = run(
        CONFIG_SETS[args.mode], args.max_seconds, args.beta, args.trials, args.n_min, args.n_max
    )
    tag = {"knobs": "options", "combos": "combos"}[args.mode]
    paths = save(results, args.beta, args.trials, tag=tag)
    plot(paths["agg"])


if __name__ == "__main__":
    main()
