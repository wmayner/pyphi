"""Empirical scaling of PyPhi computations across formalisms.

Times the per-stage cost of PyPhi's IIT computations on a family of
stochastic-majority ring systems of growing size ``n``, fits the exponential
growth rate, and writes the raw per-trial timings, aggregates, fitted rates,
and figures. The purpose is to *confirm* the published IIT 3.0 complexity
``O(n**5 3**n)`` (Mayner et al., 2018) and to measure the unpublished scaling
of the IIT 4.0 and actual-causation paths.

The test system is a ring of ``n`` stochastic majority gates with bidirectional
edges and self-loops: each unit's next state is a logistic function of the
summed spins of its two neighbours and itself. The construction is deterministic
in ``(n, beta)`` — no randomness — so results reproduce exactly. Wall-clock
timing is itself noisy, so several trials are recorded per point and the raw
values are all saved.

Run::

    uv run python benchmarks/complexity/scaling.py --max-seconds 120

then embed the figures/tables it writes to ``benchmarks/complexity/results/``.
This script is never run at docs-build time.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import pyphi
from pyphi import utils
from pyphi.substrate import Substrate

RESULTS_DIR = Path(__file__).parent / "results"

# Formalism version names understood by ``pyphi.analyze``.
IIT_FORMALISMS = ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026")


def majority_ring(n: int, beta: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """State-by-node TPM and connectivity matrix of a majority-gate ring.

    Each of the ``n`` units takes its two ring neighbours and itself as inputs
    (a strongly connected, bidirectional ring with self-loops). Its next-state
    probability is ``logistic(beta * sum of input spins)`` with spin ``2s - 1``,
    a stochastic majority gate. States are enumerated in PyPhi's little-endian
    order via :func:`pyphi.utils.all_states`.
    """
    inputs = [((i - 1) % n, i, (i + 1) % n) for i in range(n)]
    states = list(utils.all_states(n))
    tpm = np.zeros((len(states), n))
    for row, state in enumerate(states):
        for i in range(n):
            total = sum(2 * state[j] - 1 for j in inputs[i])
            tpm[row, i] = 1.0 / (1.0 + np.exp(-beta * total))
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in inputs[i]:
            cm[j, i] = 1
    return tpm, cm


def ring_substrate(n: int, beta: float = 4.0) -> Substrate:
    tpm, cm = majority_ring(n, beta)
    return Substrate(tpm, cm=cm, node_labels=pyphi.examples.LABELS[:n])


@dataclass
class Trial:
    """One timed computation."""

    formalism: str
    stage: str
    n: int
    trial: int
    seconds: float
    n_distinctions: int
    n_relations: int


def _counts(result: object) -> tuple[int, int]:
    """Distinction and relation counts from a CES/SIA result (−1 if absent)."""
    distinctions = getattr(result, "distinctions", None)
    n_d = len(distinctions) if distinctions is not None else -1
    relations = getattr(result, "relations", None)
    n_r = len(relations) if relations is not None else -1
    return n_d, n_r


def time_stage(
    substrate: Substrate,
    state: tuple[int, ...],
    formalism: str,
    stage: str,
    n: int,
    trials: int,
) -> Iterator[Trial]:
    """Time ``analyze(..., compute=stage)`` for ``trials`` fresh evaluations.

    A fresh :class:`System` is built each trial (via ``analyze``), so no
    repertoire cache carries over and every trial pays the full cost.
    """
    for t in range(trials):
        start = time.perf_counter()
        result = pyphi.analyze(substrate, state, formalism=formalism, compute=stage)
        elapsed = time.perf_counter() - start
        n_d, n_r = _counts(result)
        yield Trial(formalism, stage, n, t, elapsed, n_d, n_r)


def time_actual_causation(
    n: int, beta: float, trials: int
) -> Iterator[Trial]:
    """Time an actual-causation account of the ring's all-on fixed point."""
    substrate = ring_substrate(n, beta)
    on = tuple([1] * n)
    units = tuple(range(n))
    for t in range(trials):
        with pyphi.config.override(**pyphi.iit3):
            transition = pyphi.actual.Transition(substrate, on, on, units, units)
            start = time.perf_counter()
            pyphi.actual.account(transition)
            elapsed = time.perf_counter() - start
            yield Trial("AC_2019", "account", n, t, elapsed, -1, -1)


def run(
    max_seconds: float,
    beta: float,
    trials: int,
    n_min: int,
    n_max: int,
) -> list[Trial]:
    """Sweep ``n`` per (formalism, stage), stopping each once a trial is slow.

    A (formalism, stage) sweep advances to the next ``n`` only while the median
    trial time stays under ``max_seconds``; this keeps the whole run bounded
    without hand-tuning a separate ``n`` ceiling per stage.
    """
    pyphi.config.progress_bars = False
    results: list[Trial] = []
    sweeps = [(f, stage) for f in IIT_FORMALISMS for stage in ("sia", "ces")]
    sweeps.append(("AC_2019", "account"))
    for formalism, stage in sweeps:
        for n in range(n_min, n_max + 1):
            if stage == "account":
                gen = time_actual_causation(n, beta, trials)
            else:
                substrate = ring_substrate(n, beta)
                state = tuple([1] * n)
                gen = time_stage(substrate, state, formalism, stage, n, trials)
            # Consume trials lazily so a single over-budget trial stops the
            # remaining (equally slow) trials at this n immediately.
            trs: list[Trial] = []
            over = False
            for trial in gen:
                trs.append(trial)
                if trial.seconds > max_seconds:
                    over = True
                    break
            results.extend(trs)
            median = float(np.median([t.seconds for t in trs]))
            print(
                f"{formalism:14s} {stage:8s} n={n}  median={median:8.3f}s  "
                f"(d={trs[0].n_distinctions}, r={trs[0].n_relations})",
                flush=True,
            )
            if over or median > max_seconds:
                print(f"  -> stop {formalism}/{stage}: exceeded {max_seconds}s", flush=True)
                break
    return results


def _fit_base(ns: np.ndarray, logs: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``logs = a + b ns``; return (base = exp(b), R²)."""
    slope, intercept = np.polyfit(ns, logs, 1)
    predicted = intercept + slope * ns
    ss_res = float(np.sum((logs - predicted) ** 2))
    ss_tot = float(np.sum((logs - logs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(slope)), r2


def fit_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Fit exponential growth per (formalism, stage).

    Two fits over the per-point median times (``n >= 3``; small ``n`` is
    dominated by fixed overhead):

    - ``base``: the raw factor from ``log(time) = a + b n``.
    - ``base_div_n5``: the factor from ``log(time) - 5 log n = a + b n``, i.e.
      after dividing out an ``n**5`` polynomial. For the IIT 3.0 SIA the
      published complexity is ``O(n**5 3**n)``, so this should approach 3, while
      the raw ``base`` is inflated by the polynomial factor over the small-``n``
      range measured here.
    """
    rows = []
    for formalism in df["formalism"].unique():
        stages = df.loc[df["formalism"] == formalism, "stage"].unique()
        for stage in stages:
            grp = df.loc[(df["formalism"] == formalism) & (df["stage"] == stage)]
            med = grp.groupby("n")["seconds"].median()
            ns = np.asarray(med.index, dtype=float)
            secs = np.asarray(med.to_numpy(), dtype=float)
            mask = ns >= 3
            ns, secs = ns[mask], secs[mask]
            if len(secs) < 2:
                continue
            logs = np.log(secs)
            base, r2 = _fit_base(ns, logs)
            base_div_n5, _ = _fit_base(ns, logs - 5.0 * np.log(ns))
            rows.append(
                {
                    "formalism": formalism,
                    "stage": stage,
                    "n_points": len(secs),
                    "n_max": int(ns.max()),
                    "base": base,
                    "base_div_n5": base_div_n5,
                    "r2": r2,
                }
            )
    return pd.DataFrame(rows)


def _versioned_path(base: Path) -> Path:
    """Return ``base``, or ``base`` with a ``_v2``/``_v3``… suffix if it exists."""
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    v = 2
    while (candidate := base.with_name(f"{stem}_v{v}{suffix}")).exists():
        v += 1
    return candidate


def save(results: list[Trial], beta: float, trials: int) -> dict[str, Path]:
    """Write raw per-trial JSON, aggregate CSV, fitted rates, and metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(t) for t in results])
    label = f"beta{beta:g}_trials{trials}"

    raw_path = _versioned_path(RESULTS_DIR / f"scaling_{label}.raw.json")
    meta = {
        "beta": beta,
        "trials": trials,
        "construction": "stochastic majority ring (deterministic in n, beta)",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "pyphi_version": getattr(pyphi, "__version__", "unknown"),
    }
    raw_path.write_text(
        json.dumps({"meta": meta, "trials": [asdict(t) for t in results]}, indent=2)
    )

    agg = (
        df.groupby(["formalism", "stage", "n"])
        .agg(
            seconds_median=("seconds", "median"),
            seconds_min=("seconds", "min"),
            seconds_max=("seconds", "max"),
            n_distinctions=("n_distinctions", "first"),
            n_relations=("n_relations", "first"),
        )
        .reset_index()
    )
    agg_path = _versioned_path(RESULTS_DIR / f"scaling_{label}.agg.csv")
    agg.to_csv(agg_path, index=False)

    fits = fit_rates(df)
    fits_path = _versioned_path(RESULTS_DIR / f"scaling_{label}.fits.csv")
    fits.to_csv(fits_path, index=False)

    print(f"\nwrote {raw_path.name}, {agg_path.name}, {fits_path.name}")
    print("\nfitted per-unit growth factors:")
    print(fits.to_string(index=False))
    return {"raw": raw_path, "agg": agg_path, "fits": fits_path}


def plot(agg_path: Path) -> Path:
    """Draw log-scale runtime-vs-n curves per formalism/stage from an agg CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    agg = pd.read_csv(agg_path)
    agg["label"] = agg["formalism"] + " / " + agg["stage"]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=agg,
        x="n",
        y="seconds_median",
        hue="label",
        marker="o",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("system size n (units)")
    ax.set_ylabel("wall-clock seconds (median)")
    ax.set_title("PyPhi computation cost vs. system size")
    ax.legend(title="formalism / stage", fontsize=8)
    fig.tight_layout()
    fig_path = _versioned_path(agg_path.with_suffix("").with_suffix(".png"))
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path.name}")
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=9)
    parser.add_argument("--plot-only", type=Path, default=None)
    args = parser.parse_args()

    if args.plot_only is not None:
        plot(args.plot_only)
        return

    results = run(args.max_seconds, args.beta, args.trials, args.n_min, args.n_max)
    paths = save(results, args.beta, args.trials)
    plot(paths["agg"])


if __name__ == "__main__":
    main()
