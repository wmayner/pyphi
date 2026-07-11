# Partial-Distinction Certified Φ Bracket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and empirically validate a two-sided *certified* bracket on IIT 4.0 Φ from a partial (incomplete) distinction set, then — only if the confirmation experiment shows it is both sound and useful — promote it into `pyphi/formalism/iit4/bounds.py`.

**Architecture:** The bracket's computable core (Approach A) lives first in a self-contained experiment module `experiments/certified_bracket_experiments/bracket.py`, imported by a truncation-sweep harness `verify_certified_bracket.py`. The harness computes each system's full CES once for ground truth, then sweeps truncations under two computation orders and records the certified interval at each step. A `FINDINGS.md` states the verdict. Promotion into `bounds.py` is a final, conditional task gated on a positive verdict. This keeps unproven code out of the shipping module until the experiment discharges the gate.

**Tech Stack:** Python 3.13+, `numpy`, `pyphi` (`Substrate`/`System`/`CauseEffectStructure`), the existing `pyphi/formalism/iit4/bounds.py` (`_grouped_subset_min_sum`, `g` via `sum_phi_relations_upper_bound`, `UpperBound`), `pytest`. Run everything with `uv run`.

## Global Constraints

- **Python 3.13+ only.** No backward-compatibility shims.
- **Reproducibility.** Any randomization takes a `seed` and uses an isolated `np.random.default_rng(seed)`; the seed is saved in the output JSON, never module-level.
- **Raw data.** Per-record values (per system, state, order, truncation `k`) are saved to JSON alongside aggregates, never only summaries.
- **No clobbering.** Output filenames encode `seed`/`trials`; if a file exists, increment `_v2`, `_v3`, …. Never overwrite.
- **Certified domain only.** Binary units, conditionally independent TPM, GID/II measure — the existing `bounds.py` domain guards apply. Experiment uses library defaults (IIT_4_0_2023, GID).
- **Docstrings:** NumPy style, final-state impersonal voice, Unicode symbols (`Φ`, `φ`, `𝒵`), no process narrative, no planning-artifact references.
- **Commits** end with the two trailer lines used across this branch (`Co-Authored-By:` and `Claude-Session:`). Never `--no-verify`.
- **Notation** (from `experiments/so_certificate_experiments/FINDINGS.md`): `o` = UnitState pair; `𝒵(o)` = distinctions whose `purview_union` contains `o`; `q_d = φ_d/|purview_union_d|`; `S(o) = Σ_{d∈𝒵(o)} q_d`; `g(k) = (2^k−1−k)/k`.

---

### Task 1: Certificate-terms extraction (measured certificate, identity, self-sum)

Port the proved state-keyed extraction from the S(o) reference into a tested module. This yields the lower endpoint `L_r = identity(D_c)`, the exact self-relation sum, and the measured certificate (the `M_u → ∅` limit of the upper endpoint).

**Files:**
- Create: `experiments/certified_bracket_experiments/__init__.py` (empty)
- Create: `experiments/certified_bracket_experiments/bracket.py`
- Test: `experiments/certified_bracket_experiments/test_bracket.py`

**Interfaces:**
- Produces:
  - `profile_from_distinctions(distinctions) -> Profile` where `Profile` is a dataclass with `state_groups: dict[Any, list[float]]` (o → ascending densities), `self_sum: float`.
  - `identity_cross(profile: Profile) -> float` — the exact Eq. 11 cross sum.
  - `measured_cross_certificate(profile: Profile) -> float` — `Σ_o S(o)·g(|𝒵(o)|)`.
  - `sum_phi_relations_lower(profile: Profile) -> float` — `self_sum + identity_cross` (= `L_r`).

- [ ] **Step 1: Write the failing test** (reference agreement on a fixture)

```python
# experiments/certified_bracket_experiments/test_bracket.py
import math

import pyphi

from experiments.certified_bracket_experiments import bracket as B


def _ces(name):
    system = getattr(pyphi.examples, name)()
    return system.ces()


def test_identity_reconstructs_sum_phi_relations_on_grid3():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    reconstructed = B.sum_phi_relations_lower(profile)
    assert math.isclose(
        reconstructed, float(ces.sum_phi_relations), abs_tol=1e-9
    )


def test_measured_certificate_upper_bounds_true_sum_phi_relations_on_grid3():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    cert = profile.self_sum + B.measured_cross_certificate(profile)
    assert cert >= float(ces.sum_phi_relations) - 1e-9
    # FINDINGS reference: grid3 state-keyed bound ≈ 9.94, true Σφ_r ≈ 3.78.
    assert math.isclose(cert, 9.94, abs_tol=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q`
Expected: FAIL — `bracket` has no `profile_from_distinctions`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/certified_bracket_experiments/bracket.py
"""Approach A of the partial-distinction certified Φ bracket.

The computable core: the exact state-keyed identity (lower endpoint on Σφ_r),
the measured state-keyed certificate (upper endpoint in the complete-distinction
limit), and the wildcard extension that bounds the contribution of un-evaluated
candidate mechanisms. Validated by ``test_bracket.py`` and by the truncation
sweep in ``verify_certified_bracket.py``; promoted into
``pyphi/formalism/iit4/bounds.py`` only if that sweep confirms soundness and
usefulness.

Notation follows ``experiments/so_certificate_experiments/FINDINGS.md``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def g(k: int) -> float:
    """The Eq. 14 per-o linear-program maximum weight, (2^k − 1 − k)/k."""
    return (2.0**k - 1.0 - k) / k if k > 0 else 0.0


@dataclass
class Profile:
    """The measured state-keyed incidence profile of a distinction set.

    Attributes
    ----------
    state_groups : dict
        Maps each UnitState pair ``o`` to the ascending list of densities
        ``q_d = φ_d/|purview_union_d|`` of the distinctions incident to it.
    self_sum : float
        The exact self-relation sum ``Σ_d |z*_c ∩ z*_e|·q_d``.
    """

    state_groups: dict[Any, list[float]] = field(default_factory=dict)
    self_sum: float = 0.0


def profile_from_distinctions(distinctions) -> Profile:
    groups: dict[Any, list[float]] = defaultdict(list)
    self_sum = 0.0
    for d in distinctions:
        union = set(d.purview_union)
        if not union:
            continue
        phi = float(d.phi)
        density = phi / len(union)
        inter = set(d.cause.purview_units) & set(d.effect.purview_units)
        self_sum += len(inter) * density
        for o in union:
            groups[o].append(density)
    for densities in groups.values():
        densities.sort()
    return Profile(state_groups=dict(groups), self_sum=self_sum)


def identity_cross(profile: Profile) -> float:
    total = 0.0
    for densities in profile.state_groups.values():
        k = len(densities)
        total += sum(q * (2.0 ** (k - (i + 1)) - 1.0) for i, q in enumerate(densities))
    return total


def measured_cross_certificate(profile: Profile) -> float:
    total = 0.0
    for densities in profile.state_groups.values():
        total += sum(densities) * g(len(densities))
    return total


def sum_phi_relations_lower(profile: Profile) -> float:
    return profile.self_sum + identity_cross(profile)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/certified_bracket_experiments/
git commit -m "Add measured state-keyed certificate extraction for the Φ bracket"
```

---

### Task 2: Σφ_r wildcard upper endpoint (the crux)

The certified upper bound on Σφ_r for a partial distinction set. Pins the computed distinctions at their measured densities and adds a conservative wildcard budget for the un-evaluated mechanisms, capped at the certified `GENERAL` ceiling so it can never be looser than the shipped worst case.

**Files:**
- Modify: `experiments/certified_bracket_experiments/bracket.py`
- Test: `experiments/certified_bracket_experiments/test_bracket.py`

**Interfaces:**
- Consumes: `Profile`, `g` (Task 1).
- Produces:
  - `sum_phi_relations_partial_upper(profile: Profile, uncomputed_sizes: list[int], n: int) -> float`.
    `uncomputed_sizes` is the list of mechanism sizes `|m|` of the un-evaluated candidates (`M_u`).

**Construction (Approach A).** With `U_mass = Σ_{m∈M_u} |m|·n`, `num_u = |M_u|`, and the measured per-o groups from `profile`:

```
self_upper = profile.self_sum + U_mass
CROSS_A    = Σ_{o measured} (S_c(o) + U_mass)·g(k_c(o) + num_u)
             + extra_empty · U_mass · g(num_u)          # extra_empty = max(0, 2n − |measured o's|)
U_r        = self_upper + min(CROSS_A, GENERAL_cross)
```

This is a valid upper bound: for every o, `S_c(o)+U_mass ≥ S_true(o)` (since each uncomputed distinction's total density mass ≤ its own `|m|·n ≤ U_mass`) and `k_c(o)+num_u ≥ k_true(o)`, and `g` is increasing; new o's created by `M_u` are covered by the `extra_empty` term up to the `GENERAL` count of `2n`. The `min(·, GENERAL_cross)` guarantees `U_r ≤ GENERAL` always. It reduces to the measured certificate when `M_u = ∅` (`U_mass = 0`, `num_u = 0`) and to `GENERAL` when `D_c = ∅` (the cap binds). It is deliberately conservative (the same `U_mass` funds both self and cross terms); the experiment measures the resulting looseness.

- [ ] **Step 1: Write the failing tests** (sandwich + soundness)

```python
# append to test_bracket.py
from pyphi.formalism.iit4 import bounds as PB


def _general_cross(n):
    total = float(PB.sum_phi_relations_upper_bound(n, "GENERAL").value)
    self_ceiling = float(PB.sum_phi_distinctions_upper_bound(n, "I").value)
    return total - self_ceiling


def test_partial_upper_reduces_to_measured_when_none_uncomputed():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    measured = profile.self_sum + B.measured_cross_certificate(profile)
    partial = B.sum_phi_relations_partial_upper(profile, uncomputed_sizes=[], n=3)
    assert math.isclose(partial, measured, rel_tol=1e-12)


def test_partial_upper_never_exceeds_general_ceiling():
    n = 3
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    general = float(PB.sum_phi_relations_upper_bound(n, "GENERAL").value)
    # Even with a large uncomputed set the bound stays under GENERAL.
    partial = B.sum_phi_relations_partial_upper(
        profile, uncomputed_sizes=[1, 2, 3, 2, 1], n=n
    )
    assert partial <= general + 1e-9


def test_partial_upper_brackets_true_sum_on_grid3_with_one_dropped():
    # Drop one real distinction into M_u; the partial upper must still bound
    # the true Σφ_r of the FULL structure.
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    distinctions = list(ces.distinctions)
    dropped = distinctions[-1]
    kept = distinctions[:-1]
    profile = B.profile_from_distinctions(kept)
    n = 3
    upper = B.sum_phi_relations_partial_upper(
        profile, uncomputed_sizes=[len(dropped.mechanism)], n=n
    )
    assert upper >= float(ces.sum_phi_relations) - 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q -k partial`
Expected: FAIL — `sum_phi_relations_partial_upper` undefined.

- [ ] **Step 3: Implement**

```python
# append to bracket.py
from pyphi.formalism.iit4 import bounds as _bounds


def _general_cross(n: int) -> float:
    total = float(_bounds.sum_phi_relations_upper_bound(n, "GENERAL").value)
    self_ceiling = float(_bounds.sum_phi_distinctions_upper_bound(n, "I").value)
    return total - self_ceiling


def sum_phi_relations_partial_upper(
    profile: Profile, uncomputed_sizes: list[int], n: int
) -> float:
    """Certified upper bound on Σφ_r for a partial distinction set.

    Parameters
    ----------
    profile : Profile
        The measured incidence profile of the computed distinctions ``D_c``.
    uncomputed_sizes : list of int
        Mechanism sizes ``|m|`` of the un-evaluated candidate mechanisms
        ``M_u``. Empty for a complete distinction set.
    n : int
        Number of binary units.
    """
    u_mass = float(sum(size * n for size in uncomputed_sizes))
    num_u = len(uncomputed_sizes)
    self_upper = profile.self_sum + u_mass

    cross = 0.0
    for densities in profile.state_groups.values():
        s_c = sum(densities)
        k_c = len(densities)
        cross += (s_c + u_mass) * g(k_c + num_u)
    extra_empty = max(0, 2 * n - len(profile.state_groups))
    cross += extra_empty * u_mass * g(num_u)

    cross = min(cross, _general_cross(n))
    return self_upper + cross
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q -k partial`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/certified_bracket_experiments/
git commit -m "Add wildcard Σφ_r upper endpoint for the partial-distinction bracket"
```

---

### Task 3: The Φ bracket assembly

Combine the Σφ_d sides and the Σφ_r sides into the certified interval on Φ, and verify it contains the true Φ on every fixture and tightens as distinctions move from `M_u` into `D_c`.

**Files:**
- Modify: `experiments/certified_bracket_experiments/bracket.py`
- Test: `experiments/certified_bracket_experiments/test_bracket.py`

**Interfaces:**
- Consumes: `Profile`, `profile_from_distinctions`, `sum_phi_relations_lower`, `sum_phi_relations_partial_upper` (Tasks 1–2).
- Produces:
  - `Bracket` dataclass: `lower: float`, `upper: float`, `sum_phi_d_lower: float`, `sum_phi_d_upper: float`, `sum_phi_r_lower: float`, `sum_phi_r_upper: float`.
  - `phi_bracket(computed_distinctions, uncomputed_sizes: list[int], n: int) -> Bracket`.

- [ ] **Step 1: Write the failing tests**

```python
# append to test_bracket.py
def test_full_bracket_contains_true_phi_on_grid3():
    pyphi.config.progress_bars = False
    system = pyphi.examples.grid3_system()
    ces = system.ces()
    true_phi = float(ces.sum_phi_distinctions) + float(ces.sum_phi_relations)
    br = B.phi_bracket(list(ces.distinctions), uncomputed_sizes=[], n=3)
    assert br.lower <= true_phi + 1e-9
    assert br.upper >= true_phi - 1e-9


def test_bracket_tightens_when_more_distinctions_are_computed():
    pyphi.config.progress_bars = False
    system = pyphi.examples.grid3_system()
    ces = system.ces()
    distinctions = list(ces.distinctions)
    n = 3
    # Fewer computed → at most as tight.
    half = len(distinctions) // 2
    sizes_rest = [len(d.mechanism) for d in distinctions[half:]]
    wide = B.phi_bracket(distinctions[:half], sizes_rest, n)
    narrow = B.phi_bracket(distinctions, [], n)
    assert (narrow.upper - narrow.lower) <= (wide.upper - wide.lower) + 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q -k bracket`
Expected: FAIL — `phi_bracket`/`Bracket` undefined.

- [ ] **Step 3: Implement**

```python
# append to bracket.py
@dataclass
class Bracket:
    """A certified two-sided interval on Φ for a partial distinction set."""

    lower: float
    upper: float
    sum_phi_d_lower: float
    sum_phi_d_upper: float
    sum_phi_r_lower: float
    sum_phi_r_upper: float


def phi_bracket(computed_distinctions, uncomputed_sizes: list[int], n: int) -> Bracket:
    """Certified [lower, upper] on Φ from a computed distinction subset.

    ``computed_distinctions`` are the resolved distinctions ``D_c``;
    ``uncomputed_sizes`` are the mechanism sizes of the un-evaluated
    candidate mechanisms ``M_u``.
    """
    computed = list(computed_distinctions)
    profile = profile_from_distinctions(computed)

    sum_phi_d_lower = sum(float(d.phi) for d in computed)
    sum_phi_d_upper = sum_phi_d_lower + sum(size * n for size in uncomputed_sizes)

    sum_phi_r_lower = sum_phi_relations_lower(profile)
    sum_phi_r_upper = sum_phi_relations_partial_upper(profile, uncomputed_sizes, n)

    return Bracket(
        lower=sum_phi_d_lower + sum_phi_r_lower,
        upper=sum_phi_d_upper + sum_phi_r_upper,
        sum_phi_d_lower=sum_phi_d_lower,
        sum_phi_d_upper=sum_phi_d_upper,
        sum_phi_r_lower=sum_phi_r_lower,
        sum_phi_r_upper=sum_phi_r_upper,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest experiments/certified_bracket_experiments/test_bracket.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/certified_bracket_experiments/
git commit -m "Assemble the certified Φ bracket from the Σφ_d and Σφ_r endpoints"
```

---

### Task 4: Truncation-sweep experiment harness

The confirmation experiment. For each system, compute the full CES once, then sweep truncations `k = 0…|distinctions|` under two orders, recording the certified bracket and soundness at each step.

**Files:**
- Create: `experiments/certified_bracket_experiments/verify_certified_bracket.py`

**Interfaces:**
- Consumes: `phi_bracket`, `Bracket` (Task 3).
- Produces: a CLI writing seeded JSON `certified_bracket_seed<seed>_trials<t>.json` with `summary` + raw `records`.

**Design notes.**
- `mechanism` sizes for `M_u` come from the dropped distinctions' `len(d.mechanism)` — the experiment truncates the *resolved* distinction list, so it knows each dropped mechanism's size (in production these are candidate-mechanism sizes, known a priori). This is faithful because `|m|` is the only `M_u` fact Approach A uses.
- Two orders: `oracle` = distinctions by `float(d.phi)` descending; `cheap` = by `len(d.mechanism) * n` descending.
- "Useful" flags recorded per record so the threshold/target is chosen at analysis time, not hard-coded here.

- [ ] **Step 1: Write the harness**

```python
# experiments/certified_bracket_experiments/verify_certified_bracket.py
"""Confirmation experiment for the partial-distinction certified Φ bracket.

Per system and state, computes the full CES once for ground truth, then sweeps
truncations under two computation orders (oracle by φ_d, cheap by |m|·n),
recording at each truncation the certified bracket [L_Φ, U_Φ], its width, the
Σφ_r upper tightness, soundness (true Φ ∈ bracket), and the fraction of
distinctions computed. Answers: does the bracket close usefully before the CES
is complete? Seeded; raw per-record data saved; outputs never overwritten.

Usage:
    uv run python experiments/certified_bracket_experiments/verify_certified_bracket.py --seed 20260711 --trials 120
"""

import argparse
import importlib.metadata
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import pyphi

from experiments.certified_bracket_experiments import bracket as B

EPS = 0.02
OUT_DIR = Path(__file__).parent


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sweep(distinctions, n, true_phi, order):
    if order == "oracle":
        ordered = sorted(distinctions, key=lambda d: float(d.phi), reverse=True)
    elif order == "cheap":
        ordered = sorted(distinctions, key=lambda d: len(d.mechanism) * n, reverse=True)
    else:
        raise ValueError(order)
    m = len(ordered)
    rows = []
    for k in range(m + 1):
        computed = ordered[:k]
        uncomputed_sizes = [len(d.mechanism) for d in ordered[k:]]
        br = B.phi_bracket(computed, uncomputed_sizes, n)
        width = br.upper - br.lower
        rows.append({
            "order": order,
            "k": k,
            "fraction_computed": k / m if m else 1.0,
            "lower": br.lower,
            "upper": br.upper,
            "width": width,
            "sum_phi_r_upper": br.sum_phi_r_upper,
            "sum_phi_r_lower": br.sum_phi_r_lower,
            "sound": (br.lower <= true_phi + 1e-9) and (br.upper >= true_phi - 1e-9),
            "width_over_true": (width / true_phi) if true_phi > 0 else None,
        })
    return rows


def evaluate(system):
    ces = system.ces()
    distinctions = list(ces.distinctions)
    if not distinctions:
        return None
    n = system.substrate.size
    true_phi = float(ces.sum_phi_distinctions) + float(ces.sum_phi_relations)
    rows = []
    for order in ("oracle", "cheap"):
        rows.extend(sweep(distinctions, n, true_phi, order))
    return {
        "n": n,
        "n_distinctions": len(distinctions),
        "true_sum_phi_d": float(ces.sum_phi_distinctions),
        "true_sum_phi_r": float(ces.sum_phi_relations),
        "true_phi": true_phi,
        "sweeps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    pyphi.config.progress_bars = False
    rng = np.random.default_rng(args.seed)
    records = []
    start = time.time()

    for name in ("pqr_system", "grid3_system", "residue_system", "basic_system"):
        system = getattr(pyphi.examples, name)()
        rec = evaluate(system)
        if rec is not None:
            rec["fixture"] = name
            records.append(rec)

    sizes = rng.choice([2, 3, 3, 4], size=args.trials)
    for t, n in enumerate(sizes):
        n = int(n)
        table = rng.uniform(EPS, 1 - EPS, size=(2**n, n))
        sub = pyphi.Substrate(table, cm=np.ones((n, n), dtype=int))
        states = list(itertools.product((0, 1), repeat=n))
        if n == 4:
            states = [states[i] for i in rng.choice(len(states), size=2, replace=False)]
        for state in states:
            rec = evaluate(pyphi.System(sub, state))
            if rec is not None:
                rec.update(trial=t, state=list(state), tpm=table.tolist())
                records.append(rec)

    all_rows = [r for rec in records for r in rec["sweeps"]]
    summary = {
        "n_records": len(records),
        "n_sweep_points": len(all_rows),
        "soundness_violations": sum(1 for r in all_rows if not r["sound"]),
        "wall_time_s": time.time() - start,
    }

    out = {
        "seed": args.seed,
        "trials": args.trials,
        "git_sha": git_sha(),
        "pyphi_version": importlib.metadata.version("pyphi"),
        "config_note": "library defaults (IIT_4_0_2023, GID); precision "
        + str(pyphi.config.numerics.precision),
        "summary": summary,
        "records": records,
    }

    label = f"_{args.run_label}" if args.run_label else ""
    base = OUT_DIR / f"certified_bracket_seed{args.seed}_trials{args.trials}{label}"
    path = base.with_suffix(".json")
    version = 2
    while path.exists():
        path = base.with_name(base.name + f"_v{version}").with_suffix(".json")
        version += 1
    path.write_text(json.dumps(out))
    print(json.dumps(summary, indent=1))
    print("->", path.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the harness**

Run: `uv run python experiments/certified_bracket_experiments/verify_certified_bracket.py --seed 555 --trials 4 --run-label smoke`
Expected: prints a summary with `"soundness_violations": 0` and writes a JSON file. **If `soundness_violations > 0`, STOP** — the construction has a hole; do not proceed to Task 5. Investigate which record/sweep-point violated and reconcile against Task 2's construction before continuing.

- [ ] **Step 3: Commit**

```bash
git add experiments/certified_bracket_experiments/verify_certified_bracket.py
git commit -m "Add truncation-sweep confirmation harness for the Φ bracket"
```

---

### Task 5: Run the confirmation experiment

**Files:** none (produces a seeded JSON output).

- [ ] **Step 1: Main run**

Run: `uv run python experiments/certified_bracket_experiments/verify_certified_bracket.py --seed 20260711 --trials 120`
Expected: `"soundness_violations": 0`; writes `certified_bracket_seed20260711_trials120.json`.

- [ ] **Step 2: Sanity-glance the output**

Run: `uv run python -c "import json,glob; d=json.load(open(sorted(glob.glob('experiments/certified_bracket_experiments/certified_bracket_seed20260711*'))[-1])); print(d['summary'])"`
Expected: `soundness_violations` is 0. If nonzero, STOP and investigate.

- [ ] **Step 3: Commit the raw results**

```bash
git add experiments/certified_bracket_experiments/certified_bracket_seed20260711_trials120.json
git commit -m "Add certified Φ bracket confirmation run (seed 20260711, 120 trials)"
```

---

### Task 6: Analyze, write FINDINGS, set the verdict

Produce the durable result. Compute, from the raw records, the central question: *at what fraction of distinctions computed does the bracket first close to within a useful factor of the true Φ* — under each order. Write `FINDINGS.md`, and flip the ROADMAP Wave 7 row to the measured verdict in the same commit.

**Files:**
- Create: `experiments/certified_bracket_experiments/analyze.py`
- Create: `experiments/certified_bracket_experiments/FINDINGS.md`
- Modify: `ROADMAP.md` (Wave 7 "anytime certified Φ bracket" row)

- [ ] **Step 1: Write the analysis script**

```python
# experiments/certified_bracket_experiments/analyze.py
"""Summarize a certified-bracket run: for each order and target factor, the
median fraction of distinctions that must be computed before the bracket width
falls within ``target × true Φ``. A high fraction means the bracket is useless
for early-stopping (the honest null).

Usage:
    uv run python experiments/certified_bracket_experiments/analyze.py <results.json>
"""

import json
import sys
import statistics


def fraction_to_close(sweep_rows, order, target, true_phi):
    rows = [r for r in sweep_rows if r["order"] == order]
    rows.sort(key=lambda r: r["k"])
    for r in rows:
        if true_phi > 0 and r["width"] <= target * true_phi:
            return r["fraction_computed"]
    return 1.0


def main() -> None:
    data = json.load(open(sys.argv[1]))
    for target in (0.5, 1.0, 2.0):
        for order in ("oracle", "cheap"):
            fracs = [
                fraction_to_close(rec["sweeps"], order, target, rec["true_phi"])
                for rec in data["records"]
                if rec["true_phi"] > 0
            ]
            med = statistics.median(fracs) if fracs else float("nan")
            print(f"target={target:>4} order={order:<7} median_fraction_to_close={med:.3f}")
    print("soundness_violations:", data["summary"]["soundness_violations"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis**

Run: `uv run python experiments/certified_bracket_experiments/analyze.py experiments/certified_bracket_experiments/certified_bracket_seed20260711_trials120.json`
Expected: a table of median fractions per (target, order), and `soundness_violations: 0`.

- [ ] **Step 3: Write `FINDINGS.md`**

Write the verdict from the measured numbers. It MUST state: (1) soundness (violations count — a certified bound requires zero); (2) the median fraction-to-close per order and target; (3) the verdict — *positive* (closes usefully well before completion, e.g. median oracle fraction ≤ ~0.5 at target 2×) or *negative* (does not close until near-complete); (4) the oracle-vs-cheap gap, distinguishing a fundamental limit from an ordering problem. Follow the S(o) `FINDINGS.md` structure (verdict header, setting, results, reproduction). Cite the actual JSON filename and the seeds.

- [ ] **Step 4: Update the ROADMAP row**

Open `ROADMAP.md`, find the Wave 7 "anytime certified Φ bracket" row, and set its status to the measured verdict with a one-line pointer to `experiments/certified_bracket_experiments/FINDINGS.md`. If positive, mark it as proceeding to implementation (Task 7); if negative, mark it resolved-negative (no implementation) and note the measured reason.

- [ ] **Step 5: Commit**

```bash
git add experiments/certified_bracket_experiments/analyze.py \
        experiments/certified_bracket_experiments/FINDINGS.md ROADMAP.md
git commit -m "Add certified Φ bracket FINDINGS and set the Wave 7 verdict"
```

---

### Task 7 (CONDITIONAL — only if Task 6 verdict is POSITIVE): promote into `bounds.py`

Gate: proceed only if `FINDINGS.md` records zero soundness violations **and** a positive usefulness verdict. If the verdict is negative, SKIP this task — the spike is complete at Task 6.

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py`
- Test: `test/test_bounds.py`
- Create: `changelog.d/certified-phi-bracket.feature.md`

**Interfaces:**
- Produces (in `bounds.py`):
  - `Bracket` frozen dataclass: `lower: float`, `upper: float`, `certified: bool`, `assumptions: tuple[str, ...]`, `citation: str`.
  - `certified_big_phi_bracket(distinctions, uncomputed_mechanisms, n) -> Bracket`, where `uncomputed_mechanisms` is an iterable of mechanisms (each supporting `len(...)`).

- [ ] **Step 1: Write the failing test in the real test suite**

```python
# append to test/test_bounds.py
import pyphi
from pyphi.formalism.iit4 import bounds


def test_certified_big_phi_bracket_contains_true_phi_grid3():
    pyphi.config.progress_bars = False
    system = pyphi.examples.grid3_system()
    ces = system.ces()
    true_phi = float(ces.sum_phi_distinctions) + float(ces.sum_phi_relations)
    br = bounds.certified_big_phi_bracket(
        list(ces.distinctions), uncomputed_mechanisms=[], n=3
    )
    assert br.lower <= true_phi + 1e-9 <= br.upper + 2e-9
    assert br.certified is True


def test_certified_big_phi_bracket_brackets_with_dropped_distinction():
    pyphi.config.progress_bars = False
    system = pyphi.examples.grid3_system()
    ces = system.ces()
    ds = list(ces.distinctions)
    true_phi = float(ces.sum_phi_distinctions) + float(ces.sum_phi_relations)
    br = bounds.certified_big_phi_bracket(
        ds[:-1], uncomputed_mechanisms=[ds[-1].mechanism], n=3
    )
    assert br.lower <= true_phi + 1e-9
    assert br.upper >= true_phi - 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_bounds.py -q -k bracket`
Expected: FAIL — `certified_big_phi_bracket` undefined.

- [ ] **Step 3: Port the validated core into `bounds.py`**

Move `profile_from_distinctions`, `identity_cross`, `measured_cross_certificate`, `sum_phi_relations_lower`, `sum_phi_relations_partial_upper`, and `phi_bracket` from the experiment module into `bounds.py`, reusing the module's existing `g` (via `sum_phi_relations_upper_bound`) and `_grouped_subset_min_sum` where they coincide. Add the `Bracket` dataclass (with `certified`/`assumptions`/`citation` fields set from `_CORE_ASSUMPTIONS` and the S(o) FINDINGS citation) and the public `certified_big_phi_bracket(distinctions, uncomputed_mechanisms, n)` wrapper that derives `uncomputed_sizes = [len(m) for m in uncomputed_mechanisms]` and calls the ported core. Apply `_require_valid_domain()` at the entry. Write NumPy-style docstrings citing the S(o) FINDINGS proof and Zaeemzadeh Theorem 1 / Eq. 14. Keep the experiment module as the reproduction reference (it imports from `bounds.py` after promotion, or is left as-is — do not delete it).

- [ ] **Step 4: Run the targeted tests**

Run: `uv run pytest test/test_bounds.py -q -k bracket`
Expected: PASS.

- [ ] **Step 5: Full verification (doctest sweep + slow lane)**

Run: `uv run pytest` (no path argument — this collects the `pyphi/` doctests) and, in the background, `uv run pytest --slow -q`.
Expected: green. Fix any doctest introduced by the new docstrings.

- [ ] **Step 6: Changelog + commit**

```bash
echo 'Added `certified_big_phi_bracket()`: a certified two-sided interval on Φ computable from a partial distinction set.' > changelog.d/certified-phi-bracket.feature.md
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py changelog.d/certified-phi-bracket.feature.md
git commit -m "Add certified_big_phi_bracket to bounds.py"
```

---

## Self-Review

**Spec coverage.** §1 (partial-distinction motivation) → Task 4 harness comment + FINDINGS; §2.1 Σφ_d bracket → Task 3; §2.2 Σφ_r lower/identity → Task 1; §2.3 wildcard upper → Task 2; §2.4 assembly → Task 3; §3 experiment (two orders, records, soundness gate, null) → Tasks 4–6; §4 conditional implementation + negative-result ROADMAP flip → Tasks 6–7; §5 scope (certified-domain guard, no hot-path) → Task 7 Step 3 (`_require_valid_domain`, pure query). Covered.

**Placeholder scan.** No "TBD"/"handle edge cases"; the one intentionally open item is the *content* of `FINDINGS.md` (Task 6 Step 3) and the docstring prose (Task 7 Step 3), which depend on measured numbers and cannot be pre-written — their required structure is fully specified.

**Type consistency.** `Profile`, `Bracket`, `phi_bracket`, `sum_phi_relations_partial_upper(profile, uncomputed_sizes, n)`, `certified_big_phi_bracket(distinctions, uncomputed_mechanisms, n)` names/signatures are consistent across Tasks 1–7. The experiment passes `uncomputed_sizes` (list[int]); the public `bounds.py` wrapper takes `uncomputed_mechanisms` and derives sizes — the one deliberate signature difference, noted in Task 7.
