# Differentiation + matching — design

**Project:** P14b sub-project 4 (of 4), the matching capstone. The
cross-stimulus layer: how richly and diversely an environment triggers
perceptual structures in a system, and the matching `M` between a system and
its environment. Reference implementation: the external matching research
repo's `perception.py` (`PerceptualStructures`) and `__init__.py`
(`MatchingAnalysis`) — pre-2.0 subclass-and-mutate, global-RNG sampling;
ported to immutable views with seeded sampling.

**Paper grounding (arXiv 2412.21111v2):** Eq 15 differentiation structure
(union of components across triggered structures); Eq 16 differentiation `D`
(Σ φ_c over the union); Eq 19 perceptual differentiation `D_p` (Σ over unique
components of the **max** perception across stimuli); Eq 21–23 matching `M`
(max over contiguous subsequences of the expected world-minus-noise D_p gap).

## Scope

In scope (the matching core):
- `Differentiation` — the cross-structure projection over a set of
  `Perception` objects, exposing `D` and `D_p`.
- `MatchingAnalysis` + `MatchingResult` — matching `M` from a world
  distribution over stimuli, with seeded sampling.

Out of scope (deferred to a follow-on, queued on the roadmap):
- **Environment generation** — the dynamics salvage (`stationary_distribution`,
  Metropolis Ising sampler) and the paper's segment/point generators. The core
  takes a **world distribution as input**; the world is hand-specifiable for
  tests and the golden. The matching core needs none of it (noise is uniform,
  computed inline).
- Temporal/Markov environments. The world distribution is over *individual*
  stimuli with i.i.d. sequence sampling (matching the reference).

## Components

### `Differentiation` (`pyphi/matching/differentiation.py`)

```python
@dataclass(frozen=True)
class Differentiation:
    perceptions: tuple[Perception, ...]   # structures triggered by a stimulus set
```

The projection unions the components (distinctions and relations) across the
`Perception` objects, deduped by component identity (the 2.0 `Distinction` /
`Relation` value equality — same mechanism, mechanism state, purviews, and
specified states). Each unique component is counted once.

- `differentiation` (`cached_property`) — `D` (Eq 16): `Σ φ_c` over unique
  components. φ is intrinsic (identical wherever a component appears), so this
  is the summed φ of the union.
- `perceptual_differentiation` (`cached_property`) — `D_p` (Eq 19): for each
  unique component, the **max** perception across the structures containing it;
  summed. The max-over-structures is the cross-structure projection that makes
  concrete relations load-bearing (an analytical form is the queued P14b
  follow-on research item).
- `projection` (`cached_property`) — `{component: max_perception}`, for
  inspection.

Pure and deterministic; no RNG. Duplicate stimuli in `perceptions` collapse in
the union (max is idempotent), so sequence order and repeats do not affect
`D` / `D_p`.

### `MatchingAnalysis` (`pyphi/matching/matching.py`)

```python
@dataclass(frozen=True)
class MatchingAnalysis:
    perceptions: Mapping[tuple[int, ...], Perception]   # {stimulus: Perception}
    world_distribution: Mapping[tuple[int, ...], float] # {stimulus: probability}
```

- `__post_init__` validates: `world_distribution` keys ⊆ `perceptions` keys
  (every sampleable stimulus has a structure); probabilities ≥ 0 and sum to 1
  (within `config.numerics.precision`).
- `noise_distribution` (property) — uniform over `perceptions.keys()` (the
  stimuli for which structures exist; the paper's structureless world).
- `matching(*, seed, n_trials, k, subsequence_max=False) -> MatchingResult`:
  - `rng = np.random.default_rng(seed)` — isolated, never global.
  - Per trial: sample a length-`k` world sequence (i.i.d. from
    `world_distribution`) and a length-`k` noise sequence (i.i.d. uniform);
    compute `D_p` of each via `Differentiation`, and the gap
    `D_p(world) − D_p(noise)`.
  - Default `value` = mean over trials of the full-sequence gap (Eq 21 with
    `(a,b) = (1,k)` — what the paper computes in practice).
  - `subsequence_max=True`: for each contiguous subsequence range `(a,b)`,
    take the trial-mean gap over `world[a:b]` / `noise[a:b]`; `value` = the max
    over `(a,b)` (Eq 21 proper). The winning `(a,b)` is recorded.

### `MatchingResult` (`pyphi/matching/matching.py`)

```python
@dataclass(frozen=True)
class MatchingResult:
    value: float                     # M
    seed: int
    n_trials: int
    k: int
    world_differentiation: tuple[float, ...]   # per-trial D_p(world) — raw
    noise_differentiation: tuple[float, ...]    # per-trial D_p(noise) — raw
    subsequence: tuple[int, int] | None = None  # winning (a,b) if subsequence_max
```

Per the raw-data rule, the **per-trial `D_p` for world and noise** are saved
(the paired observations `M` is derived from), not just the aggregate `M`, and
the `seed` is saved alongside — so `M` is reconstructible without re-running.

## Data flow

```
PerceptualSystem ─► triggered_tpm + triggered_states {stimulus: y}
   (caller computes CES(y) and Perception(CES(y), ttpm, x) per response state)
      ─► {stimulus: Perception}                          (expensive, user-supplied)
Differentiation(perceptions for a stimulus set) ─► D, D_p   (deterministic)
MatchingAnalysis({stimulus: Perception}, world_distribution)
      .matching(seed=, n_trials=, k=) ─► MatchingResult       (seeded sampling)
```

## Reproducibility

- `matching()` takes `seed`; uses `np.random.default_rng(seed)`. No module-level
  `np.random` anywhere.
- `MatchingResult` saves `seed` + per-trial raw `D_p` (world and noise).
- `Differentiation` is fully deterministic; sampling lives only in `matching()`.

## Error handling

- `MatchingAnalysis.__post_init__`: world keys not ⊆ perception keys, negative
  probabilities, or probabilities not summing to 1 → `ValueError`.
- `matching()`: `k < 1` or `n_trials < 1` → `ValueError`.
- `Differentiation`: an empty `perceptions` set → `D = D_p = 0` (well-defined,
  not an error).

## Testing

`Differentiation` (deterministic — hand-computed + invariants):
- On a small set of `Perception` objects (built as in sub-project 3's tests),
  `D` = Σ φ over the unique component set, hand-verified; `D_p` = Σ
  max-perception, hand-verified.
- `D` over a single structure equals that structure's `big_phi`; `D_p` over a
  single structure equals its `richness`.
- Union idempotence: adding a duplicate `Perception` (same stimulus) leaves
  `D` and `D_p` unchanged.
- `D_p ≤ Σ richness` over the structures (max ≤ sum).

`MatchingAnalysis` / `MatchingResult` (seeded determinism + invariants):
- Fixed `seed` → identical `MatchingResult` across runs.
- **`M = 0` (within precision) when `world_distribution` is uniform** (world ≡
  noise), for any seed — the key correctness anchor.
- Per-trial raw arrays have length `n_trials`; their mean difference equals
  `value` (default mode).
- `subsequence_max=True` on a length-1 sequence equals the default; on `k>1`
  returns a `value ≥` the full-sequence value and records `subsequence`.
- Validation: non-normalized world distribution, or a world key absent from
  `perceptions`, raises.

Regression self-golden:
- Freeze a `({stimulus: Perception}, world_distribution, seed, n_trials, k)`
  fixture and its `MatchingResult` (`value` + per-trial arrays).

Old-code golden (attempt, not a gate):
- Try to resurrect the matching repo's pinned environment once and compute
  `D_p` / `M` on a small frozen Φ-structure set; freeze those as a cross-check
  of the projection/union logic against the reference. If the env cannot be
  resurrected, fall back to the hand-computed + self-golden above (the
  projection math is hand-checkable on small inputs). Scoped honestly in the
  plan; does not block the sub-project.

## Files

- `pyphi/matching/differentiation.py` — new (`Differentiation`)
- `pyphi/matching/matching.py` — new (`MatchingAnalysis`, `MatchingResult`)
- `pyphi/matching/__init__.py` — export the new names
- `test/test_differentiation.py` — new
- `test/test_matching.py` — new
- `changelog.d/matching.feature.md` — new
- `ROADMAP.md` — queue the environment-generation follow-on

## Notes carried from brainstorming

- World distribution is **input**; environment generation (Ising sampler,
  `stationary_distribution`, segment generators) is a deferred follow-on.
- Matching is **full-sequence by default**, `subsequence_max` opt-in (Eq 21
  proper) — the paper notes the full sequence attains the max in practice.
- The projection is a pure aggregation over `Perception` objects — no
  subclass-and-mutate. Component dedup uses the 2.0 model equality.
- Closes P14b. After this: P13 (pruning) → P15 surface freeze → P17 → P18 →
  2.0 release.
