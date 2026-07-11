# Landscape optimizer driver (`pyphi.optimize`)

**Status:** design approved 2026-07-11, awaiting spec review.
**Roadmap:** Wave 7 — Exploration builds. This is item 3 of the library-surface
sketch in `docs/superpowers/specs/2026-07-07-substrate-parameter-landscapes.md`
§7: "a black-box optimizer driver … objective = signed normalized φₛ, seeded,
raw trajectory saved." Items 1 and 2 (`landscape_section`/`perturb`, selection
margins) have landed as `pyphi.landscape`.

## Purpose

Search over a substrate's continuous parameters (connection weights) for a
substrate that maximizes an IIT quantity — by default the system irreducibility
signed normalized φₛ. Where `pyphi.landscape` *analyzes* the φ landscape along
one axis, this module *searches* it over a bounded parameter box.

## Gate and its constraints

The roughness gate has passed
(`experiments/substrate_landscape_experiments/FINDINGS.md`, n=5): signed
normalized φₛ stays empirically smooth across MIP switches (31 selection regimes
crossed, zero objective discontinuities), so a gradient-free population search
sees no cliffs in the objective. Two constraints the gate imposes on the design:

1. **Population-first, not single local ascent.** A single ascent chain is
   start-dependent — the gate's two seeded starts landed in different basins
   (0.161 vs −0.0021). The driver ships on a population method.
2. **Must not depend on a stable selection identity.** The MIP partition and
   specified states flip at roughly single-grid-point frequency even where the
   objective is smooth. The driver reads only the scalar objective to decide a
   move; it never branches on a selection identity. This holds by construction
   in the design below.

## Placement

New module `pyphi/optimize.py`, sibling to `pyphi/landscape.py`. The two stay
decoupled: the objective evaluates a substrate through the public
`pyphi.analyze.analyze(..., compute="sia")`, the same call
`landscape._eval_point` makes, so there is no shared private helper and no
cross-module coupling. `weight_axis` (single weight → substrate) remains in
`landscape.py` for 1-D sections; the new `weight_axes` (vector → substrate)
lives with the optimizer that consumes vectors.

## Public surface

### `weight_axes`

```python
def weight_axes(
    unit_functions: Any,
    weights: NDArray[Any],
    indices: Sequence[tuple[int, int]],
    **kwargs: Any,
) -> Callable[[NDArray[Any]], Substrate]:
```

Generalizes `landscape.weight_axis` to a vector. Returns a builder mapping a
length-`len(indices)` parameter vector θ to the substrate built from `weights`
with `weights[indices[k]] = θ[k]`. The base matrix is copied on every call; the
input is never mutated. Additional keyword arguments forward to
`build_substrate`. The θ = 0 topology caveat from `weight_axis` carries over
(a zero weight drops the connection from the derived connectivity matrix).

### `optimize`

```python
def optimize(
    builder: Callable[[NDArray[Any]], Substrate],
    state: tuple[int, ...],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int,                                    # required, no default
    objective: str | Callable[[Any], float] = "signed_normalized_phi",
    direction: str = "maximize",
    x0: NDArray[Any] | None = None,
    popsize: int = 15,
    maxiter: int = 100,
    tol: float = 0.01,
    subset: Sequence[int] | None = None,
    formalism: str | None = None,
    parallel: bool | None = None,
    progress: bool | None = None,
) -> OptimizationResult:
```

- **`builder`** — the parameter axis: a vector → `Substrate` map. `weight_axes`
  builds one for the common single-substrate weight case.
- **`state`** — the substrate state analyzed at every candidate.
- **`bounds`** — `(lo, hi)` per parameter dimension. Mandatory: differential
  evolution is a bounded method. `len(bounds)` fixes the search dimension.
- **`seed`** — required, no default. An isolated `np.random.default_rng(seed)`
  drives the optimizer; the seed is recorded on the result. Making it
  unskippable enforces reproducibility at the call site.
- **`objective`** — `"signed_normalized_phi"` (default) or another φₛ variant
  (`"phi"`, `"signed_phi"`, `"normalized_phi"`) selected by name, or an opt-in
  `Callable[[SIA], float]` for objectives the gate did not validate (CES-level
  Σφ_d, Φ, margin-based designs). A named objective reads `getattr(sia, name)`.
- **`direction`** — `"maximize"` (default) or `"minimize"`. Internally the
  problem is always posed to DE as minimization of the negated objective; the
  trajectory is logged in the natural (chosen-direction) convention.
- **`x0`** — optional seed for one population member, e.g. the base weights of a
  known-good substrate. DE fills the rest of the initial population from its
  diverse init.
- **`popsize`, `maxiter`, `tol`** — DE budget/convergence controls, passed
  through and recorded in `settings`.
- **`subset`, `formalism`** — as in `analyze`/`landscape_section`: candidate
  subset node indices, and a formalism preset applied for these evaluations
  only.
- **`parallel`** — defaults to `config.infrastructure.parallel`. When true, each
  generation's whole population is evaluated in one `map_reduce` batch.
- **`progress`** — defaults to `config.infrastructure.progress_bars`.

### `OptimizationResult`

Frozen dataclass:

| field | meaning |
|-------|---------|
| `best_params` | the winning parameter vector |
| `best_objective` | its objective value, in the chosen-direction convention |
| `best_substrate` | `builder(best_params)` |
| `best_sia` | the SIA at the best point (for margins, selection identity) |
| `trajectory` | `pd.DataFrame`, one row per evaluation (see below) |
| `bounds` | the search box |
| `seed` | the recorded seed |
| `direction` | `"maximize"` / `"minimize"` |
| `objective_name` | the quantity name, or `"<callable>"` |
| `settings` | backend, popsize, maxiter, tol |
| `config_snapshot` | pyphi config at run time (precision, formalism) |
| `n_evaluations` | total SIA evaluations |
| `n_unreachable` | count of unreachable candidates |

Methods: `to_pandas()` returns `trajectory` (matching `LandscapeSection` and
`SweepResult`); `save(path)` writes a JSON dump of everything serializable (the
raw trajectory plus all metadata). `save()` writes exactly where told — no-clobber,
parameter-encoded filenames are the calling experiment script's responsibility,
not the library's.

**Trajectory columns** (one row per evaluated candidate): `eval` (index),
`generation`, one column per parameter dimension (`p0`, `p1`, …), `objective`,
`reachable`, plus the cheap IIT-native extras pulled from the SIA — the
selection identity (`partition`, `cause_state`, `effect_state`) and the
selection margins (`partition_margin`, `cause_state_margin`,
`effect_state_margin`). Params and objective are the raw per-evaluation data the
reproducibility policy requires; margins fall out of the same SIA for free and
are the spec's IIT-native sensitivity information. The heavy SIA object is kept
only for the best point, not for every member.

## Backend and parallelism

`scipy.optimize.differential_evolution`, a seeded population method already in
the dependency set, behind an internal `_run_backend()` seam so a real CMA-ES
(`cma` package) can slot in later without touching the interface.

`vectorized=True` hands the objective the whole current population as one
`(D, S)` matrix per generation. The vectorized objective:

1. builds `S` substrates via `builder`;
2. evaluates all `S` SIAs in a single `map_reduce` batch — the exact
   `sweep._run_cells_parallel` pattern: `parallel=True` on the outer
   `map_reduce`, `config.override(**presets.by_name[formalism], parallel=False)`
   inside the worker snapshot (one level of parallelism, no oversubscription),
   `chunksize=1` because each cell is a whole SIA;
3. extracts the objective per member, negating for DE's minimization;
4. appends `S` trajectory rows (params, objective, reachable, selection
   identity, margins) and advances the generation counter.

`workers=1` on DE, since parallelism is owned by the inner `map_reduce`. When
`parallel` is false, the batch is evaluated sequentially with the same formalism
override. `polish=False`: DE's default final L-BFGS-B polish assumes a smooth
objective, which the landscape's kinks and clamp dead-zones break (spec §5,
"hopeless as stated") — and a non-vectorized polish would also bypass the batch
logging.

## Error handling

- **Unreachable candidate.** A parameter vector whose `state` is dynamically
  unreachable (`StateUnreachableForwardsError` / `StateUnreachableBackwardsError`)
  is caught, logged with `reachable=False`, and assigned a fixed penalty
  objective — a large finite value in the unfavorable direction, worse than any
  real φ. Differential evolution's selection is rank-based, so a sentinel that
  simply compares worse than every real candidate is sufficient and stable;
  never a crash, and never `inf`/`nan`, which DE handles poorly. Mirrors how
  `landscape_section` records rather than raises for unreachable points, but as a
  penalty because DE needs a comparable value.
- **Reducible candidate.** A null SIA (φ = 0) is a real landscape value, logged
  as a normal row, not an error.
- **Bad `objective` name.** `ValueError` listing the valid names, matching
  `perturb`'s quantity check.

## Alternatives considered and rejected

- **Single local finite-difference ascent** — the gate showed it is
  start-dependent; a population method is required.
- **Hand-rolled evolution strategy** — scipy already ships a seeded population
  method; reimplementing it adds code with no benefit.
- **New `cma` dependency for v1** — real CMA-ES is the spec's first-named method,
  but scipy's differential evolution is population-first, seeded, and dependency-
  free. CMA-ES slots behind `_run_backend()` if scale demands it.
- **Extending `landscape.py`** — search and analysis are distinct concerns;
  keeping them in separate modules keeps each focused and avoids growing a single
  file past its purpose.

## Out of scope for v1 (YAGNI)

- The analytic-gradient kernel (spec §5 item 3 / §7 item 4): "only if demanded by
  scale."
- Smoothed / annealed surrogate objectives (spec §5 item 4): a research
  direction; the surrogate at τ > 0 is not φₛ.
- CES/Φ objectives as *named* options: the gate validated only φₛ. They remain
  reachable through the callable escape hatch, at the caller's risk.
- Mid-run checkpointing: a dead run is exactly reproducible from its seed.

## Testing

`test/test_optimize.py`, on a tiny 2-node substrate over a small weight box with
a fixed seed and a tiny budget (small `popsize`/`maxiter`):

1. **Beats random.** Best objective ≥ a same-budget seeded random baseline.
2. **Reproducible.** Same seed → byte-identical trajectory.
3. **Unreachable logged, not raised.** A box that includes an unreachable
   candidate yields a `reachable=False` row and completes.

Fast enough for the deterministic (non-Hypothesis) lane.
