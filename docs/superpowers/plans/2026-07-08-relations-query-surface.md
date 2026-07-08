# Relations Query Surface Implementation Plan (Tiers 1–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the query surface from
`docs/superpowers/specs/2026-07-07-relations-without-materialization-design.md`
§4 — exact closed-form (Tier 1), lazy best-first (Tier 2), and seeded
statistical (Tier 3) queries on the `Relations` ABC — so the full relational
structure is queryable without materializing it. Absorbs roadmap wishlist N6
(lazy/top-K relations) and N24 (distinction-importance ranking).

**Architecture:** One internal representation — the UnitState-keyed
atom→distinction incidence plus the per-distinction density vector
(φ_d / |purview_union|) — built once per `AnalyticalRelations` from
`Distinctions.purview_inclusion(max_order=1)` and cached; every query is
derived from it. `ConcreteRelations` gets iterate-by-default implementations
of the same methods, so the two backends answer identical queries and the
whole surface is tested by the existing analytical == concrete invariant
pattern. `AnalyticalRelations` stays non-iterable: enumeration remains the
syntactically loud `strongest()` / `materialize()` choice. No new config
options; `relation_computation` keeps selecting the default backend.

**Tech Stack:** Python 3.13, numpy, pytest, Hypothesis. No new dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start; base on the current working branch).
- **Coordination:** the CES-algebra operations plan
  (`docs/superpowers/plans/2026-07-07-ces-algebra-operations.md`) also
  modifies `pyphi/relations.py` (the `AnalyticalFoldRelations` region,
  ~lines 495–525) and may already have landed. Rebase onto its branch/main
  before starting; this plan touches the `Relations` ABC (~line 324) and
  `AnalyticalRelations` (~line 435) regions — disjoint hunks, but verify at
  rebase time.
- **Source spec availability:** the spec and its appendix (the verified
  reference implementations of every primitive) live on branch
  `worktree-relations-without-materialization`. Copy the spec file into your
  worktree (or read it from that branch) before starting; every formula below
  is specified there and the appendix code is the authoritative reference —
  do not re-derive from scratch.
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on φ values. φ thresholding/bucketing in implementation code goes
  through `config.numerics.precision` / `PyPhiFloat`, never raw `==`.
- Counts are Python ints end-to-end (never numpy float64 — coefficients reach
  `2^|D_a|`). Sorted dot products accumulate via `math.fsum` from the small
  end (spec §4.3).
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts, no design-alternative discussion.
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 8) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).

## Background for implementers (read once)

A `Relations` value is one of `ConcreteRelations` (a frozenset of `Relation`
objects), `AnalyticalRelations` (closed-form, deliberately non-iterable), or
`NullRelations` (`pyphi/relations.py`). A `Relation` is a frozenset of
distinctions; `Relation.phi = |congruent overlap| × min(φ_d/|purview_union_d|)`
where `purview_union` is a set of **UnitState (index, state) pairs** and
congruence is enforced at candidate generation
(`_combinations_with_nonempty_congruent_overlap`). The ABC currently exposes
`sum_phi()`, `apportioned_sum_phi()`, `num_relations()`.

Two facts make the whole surface possible (spec §2, independently re-verified
in `so_certificate_experiments/FINDINGS.md` on 801 random systems):

1. The relation set is a deterministic view of the linear-size summary
   {(purview_union_d, q_d)}: every query is a function of the UnitState-keyed
   incidence structure. **Keying must be on UnitState pairs, not unit
   indices** — index-keying was shown to produce genuinely wrong values (an
   unsound bound witness), not merely looser ones.
2. Within each atom, sorted-density subset sums have closed forms (the S3
   factoring); `combinatorics.sum_of_minimum_among_subsets` is the existing
   kernel and the spec appendix extends it to moments, degree-resolved sums,
   histograms, and max.

Key existing pieces: `Distinctions.purview_inclusion(max_order=1)`
(`pyphi/models/distinctions.py:245`) yields the atom→distinction incidence;
`AnalyticalRelations.self_relations` already handles the |d|=1 terms;
the four small fixtures with full concrete enumeration are
`examples.xor_system()` (4 distinctions/15 relations), `grid3_system` (7/39),
`fig4_system` (4/15), `rule110_system` (4/11).

Test-strategy invariant (used in every task): implement each query on BOTH
backends; assert `analytical == concrete` (approx) on all four fixtures, and
property-test on seeded random 2–3-unit substrates (Hypothesis or a fixed
seeded loop; keep runtime bounded). The concrete implementation is the
oracle; the analytical one is the deliverable.

---

### Task 1: UnitState atom-incidence core

**Files:**
- Modify: `pyphi/relations.py` (private helpers on `AnalyticalRelations`)
- Test: `test/test_relations_queries.py` (create)
- No changelog fragment (internal).

**Interfaces:**
- Produces: `AnalyticalRelations._atom_incidence` (cached) — mapping
  `UnitState → tuple[density, ...]` (densities sorted ascending, floats) and
  a parallel mapping to the contributing distinction indices (needed by
  Tiers 2–3 and the binding matrix). Exact shape is the implementer's choice;
  it must be UnitState-keyed and built once from
  `purview_inclusion(max_order=1)` + the distinctions' φ and
  `purview_union` sizes.

- [ ] **Step 1: failing tests** — for each fixture, brute-force the incidence
  from the distinctions directly (loop over `d.purview_union`) and assert the
  cached structure matches (keys, sorted densities, member indices).
- [ ] **Step 2: implement**; run the tests.
- [ ] **Step 3: commit.**

### Task 2: Tier-1 scalar queries

**Files:**
- Modify: `pyphi/relations.py` (ABC default implementations + analytical
  closed forms)
- Test: `test/test_relations_queries.py` (extend)
- Create: `changelog.d/relations-tier1-queries.feature.md`

**Interfaces:**
- Produces on the ABC (concrete default = iterate; analytical = closed form):
  `sum_phi_moment(k: int = 2) -> float`; `phi_mean_std() -> tuple[float, float]`;
  `num_relations_of_degree(k: int) -> int`; `sum_phi_of_degree(k: int) -> float`;
  `degree_spectrum() -> Mapping[int, tuple[int, float]]` (degree → (count, Σφ));
  `max_phi() -> PyPhiFloat`; `num_faces() -> int`.
- Formulas: spec §3.1 + appendix. `max_phi` uses the proved
  attained-at-degree-2 result → O(|D|²) pair scan over congruently
  overlapping pairs (plus self-relations).

- [ ] **Step 1: failing tests** — analytical == concrete for every method on
  all four fixtures; degree_spectrum sums reconcile with `sum_phi()` and
  `num_relations()`; `max_phi` equals `max(r.phi for r in concrete)`;
  moments: `sum_phi_moment(1) == sum_phi()`; seeded random-substrate
  property loop (n=2–3, ~20 cases, both backends).
- [ ] **Step 2: implement** (ints for counts; `math.fsum` small-end
  accumulation).
- [ ] **Step 3: run, fix, commit.**

### Task 3: histogram, binding matrix, ablation ranking (N24)

**Files:**
- Modify: `pyphi/relations.py`
- Test: `test/test_relations_queries.py` (extend)
- Create: `changelog.d/relations-structure-queries.feature.md`

**Interfaces:**
- `phi_histogram() -> Mapping[PyPhiFloat, int]` — exact value→count via the
  threshold-sweep + Möbius-over-intersection-closure construction (spec
  §3.1.7 / appendix); bucket keys grouped at `config.numerics.precision`.
- `binding_matrix() -> pandas.DataFrame` — unit-state × unit-state B(a,b)
  (spec §3.1.6), labeled per the `to_pandas` conventions
  (`pyphi/models/pandas.py` helpers).
- `distinction_ablation_ranking() -> pandas.DataFrame` — per-distinction
  ΔΣφ_r when that distinction is removed, computed via the existing `fold`
  machinery / incidence recomputation (spec §3.1.9; verified 50 ms for all 27
  Fig-6D folds). This is the N24 importance ranking.

- [ ] **Step 1: failing tests** — histogram == collections.Counter over
  concrete φ values (precision-grouped) on all fixtures; binding matrix spot
  values against brute-force pair counting; ablation ranking equals
  `sum_phi(full) − sum_phi(D − {d})` recomputed concretely per distinction.
- [ ] **Step 2: implement.** Watch intersection-closure growth (spec §6.3):
  if the closure exceeds a size guard, fall back to a documented
  `NotImplementedError` naming `materialize()` — never a silent hang.
- [ ] **Step 3: run, fix, commit.**

### Task 4: Tier-2 `strongest()` — lazy best-first enumeration

**Files:**
- Modify: `pyphi/relations.py`
- Test: `test/test_relations_queries.py` (extend)
- Create: `changelog.d/relations-strongest.feature.md`

**Interfaces:**
- `strongest(k: int | None = None, min_phi: float | None = None,
  max_degree: int | None = None) -> Iterator[Relation]` on the ABC — yields
  real `Relation` objects in non-increasing φ order (ties precision-grouped,
  deterministic order within a tie group); stops after k / below min_phi.
  Analytical implementation: best-first over the atom incidence using
  antitonicity of φ_r on the subset lattice (spec §3.2 + appendix; top-10 of
  1.5M in <1 ms was measured). Concrete implementation: sort.

- [ ] **Step 1: failing tests** — `list(strongest(k))` equals the first k of
  the concretely-sorted relation list on every fixture (φ values approx-equal
  and, within tie groups, same multiset); `min_phi`/`max_degree` filters
  agree with concrete filtering; laziness: pulling 3 items from a large
  random 3-unit structure must not materialize the full set (assert via a
  counting shim on `Relation` construction or a generator-progress check);
  requesting more than exist terminates cleanly.
- [ ] **Step 2: implement.**
- [ ] **Step 3: run, fix, commit.**

### Task 5: Tier-3 `sample()` — seeded unbiased sampling

**Files:**
- Modify: `pyphi/relations.py` (+ a small `RelationSample` result type — may
  live in `pyphi/relations.py`; `Displayable`/`to_pandas` per house style)
- Test: `test/test_relations_queries.py` (extend)
- Create: `changelog.d/relations-sample.feature.md`

**Interfaces:**
- `sample(n: int, *, seed: int, max_degree: int | None = None) ->
  RelationSample` — Karp–Luby proposal over atoms + Horvitz–Thompson
  weights (spec §3.3 + appendix). `seed` is keyword-required with no
  default; RNG is an isolated `np.random.default_rng(seed)`.
  `RelationSample` carries the sampled `Relation`s, HT weights, and
  `estimate(fn)` / `stderr(fn)` for any per-relation statistic; estimates
  are never reported without their standard errors.

- [ ] **Step 1: failing tests** — determinism (same seed → identical sample);
  seed required (TypeError without); unbiasedness: on grid3, the mean of
  `estimate(phi)` over 50 seeds is within 3 combined-stderr of the exact
  `sum_phi()` (a seeded, deterministic loop — no flaky tolerance);
  `max_degree` respected.
- [ ] **Step 2: implement.**
- [ ] **Step 3: run, fix, commit.**

### Task 6: `materialize()` and the non-iterability seam

**Files:**
- Modify: `pyphi/relations.py`
- Test: `test/test_relations_queries.py` (extend)
- Create: `changelog.d/relations-materialize.feature.md`

**Interfaces:**
- `materialize(max_degree: int | None = None, min_phi: float | None = None)
  -> ConcreteRelations` on the ABC (concrete: filtered copy; analytical:
  bounded enumeration via the existing generation path or `strongest`).
- `AnalyticalRelations` stays non-iterable; its iteration error message now
  names `strongest()` and `materialize()` as the explicit alternatives.

- [ ] **Step 1: failing tests** — `materialize()` on analytical equals the
  concrete set on every fixture (as frozensets); bounded variants agree with
  concrete filtering; iteration error message mentions both escape hatches.
- [ ] **Step 2: implement; run; commit.**

### Task 7: consumer rewiring — visualization and diff

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py` (and the hypergraph
  render path that iterates relations)
- Modify: `pyphi/models/diff.py` (relation section of `ResultDiff`)
- Test: `test/visualize/` (extend the affected view tests),
  `test/models/test_result_diff.py` (extend)
- Create: `changelog.d/relations-consumers.change.md`

**Interfaces:**
- Visualization requests `relations.strongest(k)` (k from the existing
  display caps) instead of iterating, and annotates with Tier-1 scalars —
  "top-K shown of N (exact), Σφ_r = x". Behavior change: analytical
  structures become plottable (today they are refused); concrete behavior
  unchanged for structures under the cap.
- `ResultDiff`'s relation section becomes statistic deltas (Σφ_r, count,
  degree spectrum) + distinction-level attribution, with relation-level
  listing only under an explicit materialization bound (spec §3.4.3). Update
  the diff tests deliberately — this is intended surface change, reviewed,
  not silent.

- [ ] **Step 1: failing tests** for both consumers (analytical structure
  plots without raising; capped legend annotation present; diff emits
  statistic deltas for analytical relations instead of raising).
- [ ] **Step 2: implement; run the affected test dirs; commit.**

### Task 8: docs close-out and full verification

**Files:**
- Modify: docstrings on every new public method (final-state voice; the ABC
  docstring gains a short taxonomy note: exact / lazy / sampled, and that
  analytical structures answer everything without enumeration)
- Modify: the spec — add a one-line "implemented" status pointer (do not
  rewrite it)

- [ ] **Step 1:** docstring pass; `uv run pyright pyphi` clean.
- [ ] **Step 2:** full verification: `uv run pytest` with **no path
  argument**; the golden suite; `uv run pytest test/` fast lane green.
- [ ] **Step 3:** confirm no perf-counter pins drifted
  (`test/integration/test_perf_counters.py`) — this plan adds no hot-path
  work, so any drift is a bug in the plan's execution, not a regen case.
- [ ] **Step 4: commit.**

## Self-review notes

- Spec §4.1 coverage: every listed method is assigned to a task (Tier 1 →
  Tasks 2–3, Tier 2 → Task 4, Tier 3 → Task 5, escape hatch → Task 6,
  consumers → Task 7). `fold(seeds)` already exists and is not re-planned;
  `summary_row` capped-table phrasing is folded into Task 7's legend work.
- The state-keying requirement is load-bearing (an index-keyed variant was
  shown unsound in the bounds context); Task 1's brute-force incidence test
  enforces it structurally.
- Highest-risk task is 3 (histogram's intersection closure can blow up —
  guarded fallback specified) and 4 (best-first correctness under ties —
  the tie-group multiset assertion covers it).
- Concrete-backend default implementations make every query doubly
  implemented by construction; the analytical == concrete invariant is the
  oracle throughout, mirroring how `sum_phi` is already tested.
- Runtime: all fixture tests are sub-second; the seeded random property
  loops are bounded (n ≤ 3, fixed counts) to keep the suite fast.
