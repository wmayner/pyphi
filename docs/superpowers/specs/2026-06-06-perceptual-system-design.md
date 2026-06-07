# PerceptualSystem — environment→system layer — design

**Project:** P14b sub-project 2 (of 4). The substrate-level layer of the
matching formalism: a system embedded in an environment via a sensory
interface, and the machinery that turns a stimulus into a triggered TPM and a
triggered response state. Reference implementation: the external matching
research repo's `tpm.py` / `dynamics.py` / `triggering.py` (pre-2.0 API,
tangled with an old `Model` class — ported, not copied).

**Paper grounding (arXiv 2412.21111v2):** §2.1 splits the full substrate U
into system S and environment E, with sensory interface ∂S ⊆ E (the part of E
that affects S over one update). The triggered TPM is constructed as in the
footnote to §3.3: condition T_U on ∂S=x, evolve τ steps, marginalize over the
initial system state to get Pr(S_t | ∂S_{t−τ}=x). The response state used to
unfold each Φ-structure is the argmax of that distribution (confirmed: the
paper pipeline's `compute_triggered_states` does `triggered_tpm.idxmax(axis=1)`
per stimulus row).

## Scope

In scope:
- `PerceptualSystem` — wraps a `Substrate` (U) + system indices (S) + sensory
  indices (∂S), the central object everything hangs off.
- `TriggeredTPM` — the per-stimulus response distribution, a typed
  pyphi-native wrapper with a provisional `to_pandas()` view.
- Triggered-state computation: argmax-of-row (method A, paper-faithful) and
  iterative settling (method B, physically meaningful, raises on
  non-convergence).

Out of scope (deferred):
- Triggering coefficients, perception (sub-project 3).
- Differentiation, matching, world/noise sampling (sub-project 4).
- The dynamics salvage — `stationary_distribution`, Metropolis Ising sampler —
  deferred to sub-project 4 where they are first consumed. Sub-project 2 needs
  neither: the triggered TPM marginalizes the *initial* system state by a
  uniform average (not a stationary distribution), and there is no sampling
  anywhere in this layer.

## Package location

New top-level package `pyphi/matching/` (the paper frames the whole formalism
as "quantifying matching"; perception is one of five criteria). Modules:

```
pyphi/matching/
  __init__.py          # exports PerceptualSystem, TriggeredTPM
  system.py            # PerceptualSystem
  triggered_tpm.py     # TriggeredTPM + clamp-then-noise construction
```

Triggered-state methods live on `PerceptualSystem` (they are queries about a
system, not properties of the TPM object).

## Reproducibility

This layer is **fully deterministic** — the "noise" in clamp-then-noise is
*marginalization* of ∂S, not sampling. No RNG is used. `settled_state`'s
`seed_state` is an initial system state (not an RNG seed); it is part of the
result's identity and is returned/recorded alongside the settled state.

## Components

### `PerceptualSystem` (`pyphi/matching/system.py`)

```python
@dataclass(frozen=True)
class PerceptualSystem:
    substrate: Substrate          # the full U
    system_indices: tuple[int, ...]    # S
    sensory_indices: tuple[int, ...]   # ∂S ⊆ E = U \ S
```

Validation in `__post_init__`:
- `system_indices` and `sensory_indices` are disjoint subsets of
  `substrate.node_indices`.
- `sensory_indices ⊆ (U \ S)` (the sensory interface is part of the
  environment, not the system).
- both non-empty.

Derived (properties):
- `environment_indices = U \ S`.
- `node_labels` from the substrate.
- `system_labels` / `sensory_labels` via `NodeLabels`.

Methods:
- `triggered_tpm(*, tau, tau_clamp) -> TriggeredTPM` — build the response
  distribution (below). `tau`/`tau_clamp` validated: integers,
  `0 <= tau_clamp <= tau`, `tau >= 1`.
- `triggered_states(*, tau, tau_clamp) -> dict[tuple[int,...], tuple[int,...]]`
  — method A. For each stimulus (state of ∂S), the argmax system state of the
  triggered-TPM row. This is the `{stimulus: response_state}` mapping the
  Φ-structure computation consumes (one structure unfolded per response
  state). Keys are ∂S states; values are S states.
- `triggered_state(stimulus, *, tau, tau_clamp) -> tuple[int,...]` —
  single-stimulus convenience (one row's argmax).
- `settled_state(stimulus, *, seed_state, max_steps=None) -> tuple[int,...]` —
  method B (below).

### `TriggeredTPM` (`pyphi/matching/triggered_tpm.py`)

A thin typed wrapper over a multidimensional ndarray with one axis per unit,
ordered `(∂S axes..., S axes...)`, each axis sized by that unit's alphabet
(binary in the paper models; the design does not assume binary — axis sizes
come from the substrate's state space). Entry `[x..., s...]` is
`Pr(S_t = s | ∂S_{t−τ} = x)`.

```python
@dataclass(frozen=True)
class TriggeredTPM:
    array: np.ndarray             # shape = sensory axes + system axes
    sensory_indices: tuple[int, ...]
    system_indices: tuple[int, ...]
    node_labels: NodeLabels
```

Why multidimensional rather than a 2-D (stimulus × state) matrix: every
operation sub-project 3 needs — `Pr(M=m | ∂S=x)`, `Pr(M=m)` (mean over ∂S),
max over input subsets — is a uniform axis sum/slice, reusing pyphi's existing
little-endian multidimensional-distribution conventions (the same algebra as
repertoires / `JointDistribution.marginalize_out`). No pandas `groupby(axis=)`
(deprecated in the installed pandas 2.3.3) is involved.

Minimal methods for sub-project 2 (kept deliberately small; sub-project 3 adds
marginalization helpers when its exact needs are concrete):
- `row(stimulus) -> np.ndarray` — the system-state distribution for one
  stimulus (slice the ∂S axes).
- `argmax_state(stimulus) -> tuple[int,...]` — the most-probable system state
  for a stimulus (drives `triggered_state`).
- `to_pandas() -> pd.DataFrame` — **provisional** MultiIndex view: rows =
  stimulus multi-states (∂S unit labels), columns = system multi-states (S
  unit labels), values = Pr(s|x). Reproduces the reference DataFrame shape so
  the golden cross-check aligns by label, not raw index. Subsumed by the
  queued unified-`to_pandas` roadmap item (P15 regroup).

### Triggered TPM construction (clamp-then-noise)

For stimulus x, over τ steps with the first τ_clamp clamped:

1. **Clamped segment** — condition the substrate TPM on ∂S=x, restrict outputs
   to S, convert to a state-by-state matrix over S, raise to the τ_clamp
   power. (τ_clamp = 0 → identity.)
2. **Noised segment** — marginalize ∂S out of the substrate TPM (replace the
   input with its marginal — deterministic, not sampled), restrict outputs to
   S, state-by-state, raise to the (τ − τ_clamp) power. (τ_clamp = τ →
   identity.) Independent of x, so computed once and shared across stimuli.
3. **Compose** — matrix-multiply clamped · noised → the τ-step state-by-state
   map over S conditioned on ∂S=x.
4. **Marginalize initial state** — average over rows (initial S states) to get
   the distribution over S at t. This is the row for x.

Stack over all ∂S states → the `TriggeredTPM` array.

Built on 2.0 primitives: `FactoredTPM.condition` / `marginalize_out`,
`pyphi.convert` (state-by-node ↔ state-by-state), `numpy.linalg.matrix_power`.
Exact calls are pinned in the implementation plan after verifying signatures.

### `settled_state` (method B)

Iterate the deterministic most-probable-transition map under a clamped
stimulus:

```
s = seed_state
seen = {s}
loop:
    s_next = argmax over system states of Pr(S_{t+1} | ∂S = x, S_t = s)
    if s_next == s:            # fixed point
        return s
    if s_next in seen:          # limit cycle, no fixed point
        raise NonConvergenceError(naming the cycle states)
    seen.add(s_next); s = s_next
    if max_steps and len(seen) > max_steps:
        raise NonConvergenceError(...)
```

Each step conditions the substrate TPM on (∂S=x, S=s) and takes the argmax of
the next-system-state distribution (for conditionally-independent TPMs the
joint argmax equals the per-unit argmax). The `seen` set both detects cycles
and guarantees termination within |Ω_S| steps; `max_steps` is an optional
early safety for very large state spaces.

**Raises on non-convergence** rather than returning a τ-parity-dependent
state — a triggered state must be a single well-defined state, and silently
picking one from an oscillation would hide a real multi-attractor situation.
The error names the detected cycle for diagnosis. Define a small
`NonConvergenceError(Exception)` in the package.

## Data flow

```
Substrate U  ──PerceptualSystem(U, S, ∂S)──►  PerceptualSystem
                                                  │
                  triggered_tpm(τ, τ_clamp)       │  settled_state(x, seed)
                          ▼                        ▼
                    TriggeredTPM            (iterates substrate TPM directly,
                       │                     independent of TriggeredTPM)
       triggered_states = idxmax per row
                       ▼
        {stimulus: response_state}  ──►  (sub-project 3: unfold Φ-structures,
                                          triggering coefficients, perception)
```

## Error handling

- `PerceptualSystem.__post_init__`: overlapping/empty/out-of-range index sets,
  or `∂S` intersecting `S`, raise `ValueError` naming the offending indices.
- `triggered_tpm`: `tau`/`tau_clamp` not integers, `tau < 1`, or
  `not 0 <= tau_clamp <= tau` raise `ValueError`.
- `settled_state`: non-convergence raises `NonConvergenceError` naming the
  cycle; `seed_state` of wrong length raises `ValueError`.

## Testing

Unit (hand-verifiable on a tiny system — a 3-unit substrate with a 1-unit
sensory interface and 2-unit system, deterministic-enough TPM):
- Triggered-TPM rows sum to 1 (stochastic) for every stimulus.
- A hand-computed entry `Pr(S=s | ∂S=x)` matches for τ_clamp = τ = 1.
- `tau_clamp = 0` reproduces the pure-noise (input-marginalized) evolution;
  `tau_clamp = tau` reproduces the fully-clamped evolution. Verify both limits
  against direct construction.
- `triggered_state` equals the argmax of the corresponding `to_pandas()` row
  (label-aligned), and `triggered_states` is the full `{x: argmax}` mapping.
- `to_pandas()` shape/labels: index over ∂S states, columns over S states,
  values equal `array` entries (label round-trip; guards the little-endian
  ordering).

`settled_state`:
- A substrate designed to settle (monotone relay) reaches the expected fixed
  point from the default seed.
- A substrate designed to oscillate under a clamped input raises
  `NonConvergenceError`, and the error names the cycle.
- A different seed that leads to a different fixed point returns that one
  (seed-dependence is real and tested).

Invariants / property (Hypothesis, small substrates):
- Every triggered-TPM row is a valid distribution (non-negative, sums to 1).
- `triggered_state(x)` is always in the support of `row(x)`.

Optional golden (only if a reference triggered-TPM pickle loads without the
external `substrate_modeler`): compare `to_pandas()` against the saved
DataFrame for a small system. Not a gate — the primary validation is the
hand-computed + invariant tests; the cross-system golden belongs with the
matching layer (sub-projects 3–4) where the substrate fixtures live.

## Files

- `pyphi/matching/__init__.py` — new
- `pyphi/matching/system.py` — new (`PerceptualSystem`, `NonConvergenceError`)
- `pyphi/matching/triggered_tpm.py` — new (`TriggeredTPM` + construction)
- `test/test_matching_system.py` — new
- `test/test_triggered_tpm.py` — new
- `changelog.d/perceptual-system.feature.md` — new

## Notes carried from brainstorming

- The reference `simulate_model_fixed_input` / `sequentially_triggered_state`
  are **not** ported — the paper pipeline used method A; B is reimplemented
  cleanly here, and the per-node-set Gauss-Seidel variant is dropped.
- The reference's pandas `groupby(axis=)` marginalization is **not** ported
  (deprecated); marginalization is uniform ndarray axis ops on the
  multidimensional `TriggeredTPM`.
- `to_pandas()` is provisional pending the unified labeled-export project
  (queued, P15 regroup bundle).
