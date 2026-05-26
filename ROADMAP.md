# PyPhi Strategic Refactoring Roadmap

## Context

PyPhi is a scientific library implementing Integrated Information Theory (IIT). It has
drifted twice without a corresponding engineering rewrite: **(a)** IIT 3.0 → IIT 4.0,
which changed the formalism in deep ways (state-centric `ii(s,s̄)` replacing distribution
distances, directional partitions `Θ(S)` with `δ ∈ {←,→,↔}`, disintegrating partitions
`Θ(M,Z)` for distinctions with product probabilities `π_c`, normalized MIP per Eq. 23,
relations, Φ-structures); and **(b)** the aspirational move toward multi-valued units
(Gómez et al. 2020 — PyPhi once had a `nonbinary` github branch), which never fully
landed and now manifests as ~12 `# TODO extend to nonbinary nodes` breadcrumbs scattered
across `network.py`, `node.py`, `subsystem.py`, `tpm.py`, `metrics/distribution.py`,
and `repertoire.py`.

The result is a codebase where:

- `subsystem.py` is a 1422-line god-object holding conditioned TPMs, four repertoire
  caches, MIP search, φ computation, and both IIT versions behind a `config.IIT_VERSION`
  branch at `subsystem.py:983-1018`, layered over implicit metric-dispatch via
  `config.REPERTOIRE_DISTANCE` (`subsystem.py:1090-1142`). The author's own
  `TODO(4.0) refactor for consistent API across metrics` at line 1089 and
  `TODO(4.0): compute arraywise once, then find max; requires refactoring state kwarg
  to metrics` at line 1144 name the problem exactly.
- Distance metrics have incompatible signatures: IIT 3.0 is `f(rep, rep) → float`,
  IIT 4.0 is `f(forward, partitioned, selectivity, state=None) → Rep|float`. No type
  enforcement; dispatch is by string name in `metrics/distribution.py`.
- Illegal config combinations (`IIT_VERSION=3` + `REPERTOIRE_DISTANCE=INTRINSIC_INFORMATION`)
  are silently accepted and will run to nonsense.
- `models/subsystem.py:187` vs `:283` — `CauseEffectStructure.purviews(direction)` is a
  method but `FlatCauseEffectStructure.purviews` is a property. Liskov violation silenced
  with `# type: ignore[override]`.
- `partition.py:643`: `TODO(4.0) consolidate Cut and SystemPartition logic` — directional
  system partitions, mechanism bipartitions, k-partitions, and the disintegrating
  partitions that distinction computation actually requires are all loosely coexisting
  under one module. The codebase may in fact use the wrong partition family for distinctions
  (needs verification during Project 6).
- `Concept.subsystem` back-reference (`compute/subsystem.py:89,113`) prevents distinctions
  from being clean value types.
- `_backward_tpm()` in `Subsystem.__init__` is PyPhi's implementation of IIT 4.0
  causal marginalization (Eq. 3, 4) but is called unconditionally and documented nowhere
  as such.
- Parallelization uses `joblib + loky` (PROJECTS.md docs calling out "Ray" are stale);
  `parallel/` has a clean `MapReduce` abstraction but only a local backend, and parallel
  tests are excluded from CI.
- `macro.py` (1094 lines) and `actual.py` (953 lines) are flagged "out of date" and import
  from the unstable core.

The ~50+ `TODO(4.0)` and `TODO(nonbinary)` comments in the code constitute the
author's own backlog — this plan absorbs that backlog and orders it against an
architectural north star.

**Intended outcome:** A PyPhi where (i) every Greek letter in Albantakis et al. 2023
maps to a named, typed runtime object; (ii) IIT 3.0 is preserved as a first-class
`PhiFormalism` strategy rather than a contaminant in the hot path; (iii) multi-valued
units become a parameterization, not an aspiration; (iv) the mathematical objects are
immutable value types flowing through stateless algorithm layers; (v) the golden
numerical results are locked in, so every refactor is provably non-regressing.

There is also significant **prior art in open PRs** that should be absorbed:
- **PR #138 (substrate modeler)** (+3577/-452): Adds a `substrate_modeler/` subpackage
  making substrates stateless, unit states as ints, and removing BaseUnit. Directly
  aligned with P7's layered rewrite — review its `unit.py` (364 lines) and
  `substrate.py` (294 lines) as inputs to `core/unit.py` and `core/substrate.py`.
- **PR #105 (implicit TPMs)** (+1918/-620, 100 commits): Adds `ImplicitTPM` as a
  factored per-node TPM representation, with `state_space.py` for per-node state
  tracking. Explicitly supports non-binary ("last dimension must contain entries for
  all states"). **Caution:** This branch diverged ~2019 and predates the entire IIT 4.0
  implementation. It should be treated as **design reference**, not ready-to-merge code.
  The ~6 years of divergence means significant reconciliation work is needed.

**Long-term goal:** PyPhi should also become the reference library for *tractable
approximations* to Φ (φ\*, φ_G, geometric integrated information, etc.). This affects
the Protocol design: `PhiFormalism` must be broad enough for both exact and approximate
methods.

**Language:** Python remains the right choice. Users are researchers in
Python/Jupyter/pandas workflows; the bottleneck is algorithmic (O(2^n) partition
enumeration), not linguistic; NumPy/xarray/dask have no equivalent elsewhere; and
rewriting 15k+ lines of mathematical code in another language carries unacceptable
correctness risk. Consider Rust extensions (pyo3) for specific hot loops only after
profiling the refactored codebase, and only if the Zaeemzadeh bounds (P13) don't
already make the bottleneck irrelevant.

**Python version: target 3.13+** (not 3.12+). Key features:
`copy.replace()` for frozen-dataclass functional updates, experimental free-threaded
mode (no GIL — potentially transformative for parallelization), `match/case` (3.10+)
for partition/formalism dispatch, PEP 695 generic syntax `class Foo[T]:`, `@override`
decorator, `itertools.batched()`, `slots=True` on dataclasses. The 2.0 release won't
ship before late 2026, when 3.13 will be well-established.

This plan ignores time and effort costs (per the task brief) and orders by leverage,
path dependence, and correctness risk.

---

## Guiding Principles

1. **Formalism is the source of truth.** Every first-class runtime object corresponds
   to a named mathematical object from the 4.0 paper: `Unit`, `Substrate`,
   `CausalModel`, `CauseTPM`/`EffectTPM`, `Repertoire`, `IntrinsicInformation`,
   `Distinction`, `Relation`, `PhiStructure`, `Complex`. If it isn't in the paper
   and isn't infrastructure, it's a code smell.

2. **State is part of identity.** IIT 4.0 evaluates `ii(s,s̄)` at a single state.
   Every abstraction above `numpy` should accept or carry a state; the array form
   should be an intermediate, not a primitive.

3. **Separate state from computation.** Today `Subsystem` is simultaneously a
   conditioned TPM, a cache, a repertoire algebra, a MIP searcher, a config
   dispatcher, and a formalism selector. The target is frozen value types flowing
   through stateless algorithm modules, with caches as explicit decorators at known
   memoization boundaries.

4. **Make illegal states unrepresentable.** Replace `config.IIT_VERSION` +
   `config.REPERTOIRE_DISTANCE` double-dispatch with a single `PhiFormalism` object
   that bundles its own compatible metrics, partition schemes, and evaluators.
   Protocol-based dispatch catches wrong-shape calls at type-check time.

5. **Correctness safety before architectural purity.** Before touching any numerical
   code, lock in a golden regression oracle that survives every downstream refactor.
   Without this, every later step carries unacceptable silent-wrongness risk.

6. **Correctness risk scales with call-graph reach.** `subsystem.py` touches almost
   everything. Sequence the high-blast-radius refactors while the test net is freshest
   and defer low-blast-radius work (`macro.py`, `actual.py`, docs, Jupyter display).

---

## Target Architecture

```
pyphi/
  core/                          # typed kernel — no formalism logic
    unit.py                      # Unit(index, state, alphabet_size)
    substrate.py                 # immutable set of Units + connectivity
    causal_model.py              # CausalModel(substrate, TPM) — the zeroth postulate
    tpm/
      base.py                    # TPM Protocol: effect_marginal(), cause_marginal() (Eq. 3, 4)
      explicit.py                # xarray-backed ExplicitTPM; alphabet_size per axis
      implicit.py                # factored per-node TPM (from PR #105); non-binary native
      marginalization.py         # causal_marginalization() — named, documented against Eq. 3/4
    repertoire.py                # Repertoire = labeled tensor + state selector
    candidate_system.py          # (CausalModel, state, node_subset, cut) — frozen
    protocols.py                 # Metric, PartitionScheme, Formalism, Scheduler

  partition/
    algebra.py                   # Partition sum type
    system.py                    # Θ(S) directional — Eq. 14-18
    disintegrating.py            # Θ(M,Z) — Eq. 29 (currently missing as a type)
    mechanism.py                 # legacy bipartitions (IIT 3.0)

  metric/
    base.py                      # Metric Protocol: (repertoire, state|None) → DistanceResult
    intrinsic_information.py     # Eq. 5, 7
    gid.py                       # generalized intrinsic difference
    specification.py
    legacy/                      # IIT 3.0 distribution distances

  combinatorics/                   # refactored from single combinatorics.py + parts of utils.py
    sets.py                      # powerset, pairs, subset operations, only_nonsubsets
    states.py                    # state enumeration, generalized for multi-valued units
    analytical.py                # closed-form Σφ_r formulas (S3 Text of 4.0 paper)
    zdd_family.py                # ZDDFamily Protocol (powerset_family, set_size_family)
    oxidd_family.py              # default ZDD backend (P6b)
    graphillion_family.py        # legacy fallback, removed in 2.1

  formalism/
    base.py                      # PhiFormalism Protocol
    iit3/                        # frozen legacy: bipartitions + distribution metrics
    iit4/
      distinction.py             # build_distinction() — uses Θ(M,Z) + π_c
      relation.py                # relations.py cleaned; uses graphillion + combinatorics
      phi_structure.py           # C = D ∪ R, Φ = Σφ_d + Σφ_r
      sia.py                     # Θ(S) + Eq. 23 normalization
      bounds.py                  # Zaeemzadeh 2024 upper bounds
    approx/                      # future: tractable approximation methods
      base.py                    # ApproximateFormalism(PhiFormalism) with error_bound()
      # phi_star.py, phi_g.py, geometric.py — added incrementally

  models/                        # frozen value types, no back-references
    distinction.py
    phi_structure.py
    sia.py
    ces.py                       # CES + FlatCES as siblings under AbstractCES

  parallel/                      # joblib+loky local, dask.distributed cluster
    backends/ local.py dask.py htcondor.py
    scheduler.py                 # Scheduler Protocol
    chunking.py                  # dynamic cost-sampled chunking

  io/
    config.py                    # split: FormalismConfig / InfrastructureConfig / NumericsConfig
    serialize.py                 # msgspec-based; no custom registry
    pandas.py

  compute/                       # thin orchestration — one-line API over formalism.evaluate_*
```

**Key dependency changes:**
- **Add `xarray`**: labels every repertoire axis with the unit index and state alphabet.
  Retires ~10 binary-assumption TODOs by construction. **Caveat (from review):**
  xarray has significant per-operation overhead on small arrays (16 floats for 4-node
  systems). Benchmark xarray vs raw ndarray on the actual hot path before committing.
  Consider xarray for external-facing API (construction, display, indexing) with
  raw ndarray for internal computation. If benchmarks show >2x overhead, use plain
  ndarrays with metadata carried separately and xarray only at I/O boundaries.
- **Add `msgspec`**: replaces `jsonify.py`'s custom `CLASS_KEY`/`VERSION_KEY`/`ID_KEY`
  scheme for serializable types.
- **Add `dask.distributed` + `dask-jobqueue`** (optional group): cluster backend for
  SLURM / PBS / LSF / SGE / HTCondor. These are the environments scientific users
  actually have access to.
- **Keep `joblib + loky`** for local. The PROJECTS.md entry claiming PyPhi uses Ray
  is stale; it already uses loky.
- **Consider `attrs`** for the value type migration (replacing hand-rolled
  `__eq__`/`__hash__`/`__repr__`); plain frozen `dataclass` is also sufficient.
- **Do NOT add** pydantic, polars, pgmpy, jax, torch. Each is either premature
  (tensor networks, vectorized phi) or scope creep (polars vs pandas).
- **Drop `graphillion`, add `oxidd`** (P6b). graphillion is bus-factor-1, has no
  PyPI wheels for Python 3.13+, and its `_graphillion` C extension does not declare
  GIL safety (blocks free-threaded mode). OxiDD is multi-contributor, ships wheels
  for cp39-cp314 including cp314t (free-threaded), and provides ZBDD primitives
  with cleaner manager-based state management. PyPhi's high-level `setset` family
  algebra (`powerset_family`, `set_size_family`) is reimplemented behind a
  `ZDDFamily` Protocol with both backends shipping in 2.0 (graphillion as fallback
  for one release, removed in 2.1).

**Decision on IIT 3.0: keep it, behind the `PhiFormalism` Protocol.** Reasons:
(a) 3.0 is in the published literature; reproducing 3.0 results is a scientific
requirement; (b) once 3.0 is behind the Protocol, its marginal maintenance cost is
near zero because it stops contaminating the 4.0 hot path; (c) dropping it
forecloses comparative research; (d) the unification work to clean up 4.0 must happen
anyway, so keeping 3.0 is not extra work. The version that's wrong is
"keep 3.0 in the same file as 4.0 with a config branch"; the version that's right is
"keep 3.0 behind a Protocol that 4.0 also implements."

---

## Ordered Project List

### Phase 0 — Prerequisites

**P0. Python 3.13 dependency verification — DONE (2026-04-12)**

Verified that ALL C-extension dependencies compile and pass tests on Python 3.13
(both standard and free-threaded builds). Findings:

**Standard Python 3.13.13:**
- ✓ `graphillion 2.1` installs from source (no wheels published, but compiles cleanly)
- ✓ `numpy 2.4.4`, `scipy 1.17.1`, `joblib 1.5.3` all install
- ✓ pyphi imports and runs
- ✓ Test suite: **874 passed, 27 skipped, 2 xfailed** (vs 850/52/2 on Py 3.12 — the
  delta is mostly EMD tests that were skipped on Py 3.12 due to numpy-2-incompatible
  pyemd; pyemd 2.0 is numpy-2 compatible and pulls in correctly on 3.13)
- ✗ One previously-masked bug surfaces: `PyPhiFloat.__eq__` returns `False` instead
  of `NotImplemented` for non-numeric types, which breaks Python's reflective
  comparison and prevents `pytest.approx` from matching `PyPhiFloat`. Source location:
  `pyphi/data_structures/pyphi_float.py:54-58`. Trivial fix, deferred to **P5**
  (metric API unification).

**Free-threaded Python 3.13.11 (no-GIL, PEP 703):**
- ✓ All deps install
- ⚠️ `graphillion`'s `_graphillion` C extension does **not** declare no-GIL safety
  via `PyMod_GIL_NOT_USED` / `Py_mod_gil`. Python automatically re-enables the GIL
  when `_graphillion` loads, with a `RuntimeWarning`.
- ✓ With GIL fallback, **850 passed, 52 skipped, 2 xfailed** — functionally
  equivalent to standard mode.
- **Conclusion:** Free-threaded mode is currently a no-op for PyPhi until
  `graphillion` ships GIL safety or we migrate the ZDD layer.

**ZDD library survey (also done 2026-04-12):**
The graphillion bus-factor-1 risk is real. Survey findings:

| Library | Last commit | 3.13 wheels | Free-threaded | ZDD primitives | `setset` algebra |
|---------|-------------|-------------|---------------|---|---|
| `graphillion` | 2026-01 | source-only | no | yes | **yes** (unique) |
| `OxiDD` | active (multi-contributor) | yes | **yes (cp314t)** | yes | no (primitives only) |
| `dd` (tulip-control) | 2025-12 | source-only | no | yes (via CUDD) | no |
| `pyeda` | abandoned | n/a | n/a | no (BDD only) | n/a |

Graphillion is unique in providing a high-level `setset` family algebra
(`powerset_family()`, `set_size(k)` filter, `.join()`); replacements would require
~200 lines reimplementing these against ZDD primitives. The math is identical
(Minato's family algebra); the API is one layer below.

**Recommendation:** Stay on graphillion now. Hide it behind a `ZDDFamily` interface
in **P6/P15** (`combinatorics/graphillion_utils.py` already named in the target
architecture). Trigger for migration to OxiDD: graphillion blocks on (a) Python
3.14+ build, (b) free-threaded enablement, or (c) goes >12 months without a release.

**Decisions reflected in code:**
- `pyproject.toml`: `requires-python = ">=3.13"`, `target-version = "py313"`,
  `pythonVersion = "3.13"`. Classifier for 3.12 dropped.
- CI matrix in `.github/workflows/test.yml`: Python 3.13 + 3.14
  (was 3.12 + 3.13). Coverage upload moved to 3.13.
- `lint.yml`, `build.yml`: 3.12 → 3.13.
- P11's free-threaded-mode promise is **conditional** on graphillion or migration.

- *Files:* `pyproject.toml`, `.github/workflows/{test,lint,build}.yml`.
- *Status:* Complete.

### Phase A — Safety net (must be green before any numerical refactor)

**P1. Golden regression harness**

Freeze 15–25 (network × subsystem × config) fixtures covering IIT 3.0 and 4.0 with
every metric/partition scheme combination in use. For each, serialize the **raw
numerical outputs** to `.npz` or minimal JSON independent of `pyphi/jsonify.py`:
repertoires for every (mechanism, purview) pair; every RIA's phi, partition, and
specified states; every Concept's cause/effect purviews and phi; the final SIA phi.
Store as numbers and tuples — never as pickled objects. The format must survive
Project P11 (jsonify retirement).

Generate the fixtures from a known-good commit (the current `develop` head, manually
validated against published IIT results from Albantakis et al. 2023 Fig. 1-7 and
Barbosa et al. 2020 worked examples). Pin the commit hash in a header comment.

- *Why first:* Every later project is either safe because of this, or unsafe without it.
- *Files:* New `test/data/golden/*.npz`, new `test/test_golden_regression.py`. Modifies
  `test/conftest.py` to register fixtures. Uses existing `test/example_networks.py`.
- *Leverage:* Unblocks P4–P10 (anything that touches math).
- *Style:* Incremental — start with 3 networks × 2 versions × 3 metrics, grow.

**P2. Property-based invariant tests with Hypothesis**

Encode every invariant stated in the 4.0 paper as a Hypothesis test:
- Repertoires sum to 1; `phi ≥ 0`; MIP phi ≤ unpartitioned phi.
- Causal marginalization idempotent (Eq. 3 applied twice == once).
- `Direction.CAUSE` and `Direction.EFFECT` dual under `ii` (Eq. 5 vs 7).
- Partition counts match theoretical formulas for |S|, |M|, |Z|.
- `|π_c(z|m)|` = 1 across all sums.
- The invariants listed in the current `repertoire.py:27-31` comment.

Property tests catch bugs that golden tests don't — they explore random edge cases
(empty mechanisms, disconnected networks, deterministic TPMs, unreachable states).
Hypothesis is already a dev dependency but underused.

- *Files:* Expand `test/hypothesis_utils.py`; add `test/test_invariants_hypothesis.py`.
- *Leverage:* Catches regressions in P4–P10.
- *Acceptance criterion for Phase A complete:* a deliberate sign-flip in
  `metrics/distribution.py` must fail at least three golden fixtures AND at least one
  property test.

**P3. Protocol-based type hardening** *(can overlap with P2 — independent of Hypothesis)*

Convert `pyphi/types.py` from a pile of `type` aliases into a set of runtime-checkable
`Protocol` classes: `DistanceMetric`, `PhiFormalism`, `RepertoireComputer`,
`PartitionScheme`, `Scheduler`. Concrete `TypeAlias` remains only for pure
array/tuple types. Enable pyright `strict` mode for `pyphi/types.py`,
`pyphi/metrics/`, `pyphi/partition.py`, `pyphi/direction.py`.

This is pure typing work with zero runtime change, but it builds the vocabulary that
every later project expresses itself in. Writing P4's formalism split as
"Protocol-backed" instead of "duck-typed" is the difference between a clean cutover
and a maintenance disaster.

**Critically:** P3 must define a `SubsystemInterface` Protocol capturing
Subsystem's **full** public surface. The surface is larger than it appears:
- 58 public methods + 10 properties on `Subsystem` itself
- 26+ distinct attribute accesses from `new_big_phi/__init__.py` alone
- `MacroSubsystem` subclasses `Subsystem` and overrides 6 methods, and
  `SystemAttrs.apply()` (macro.py:151-158) directly mutates 7 attributes
  post-construction — incompatible with frozen value types
- 20+ import sites across the codebase (verified by grep)

**Generate this surface programmatically** (AST analysis of all Subsystem attribute
accesses across the codebase), do not hand-count. This Protocol becomes the interface
contract that P7's rewrite must satisfy.

**Also in P3:** Fix the CES/FlatCES Liskov violation — making them siblings under an
`AbstractCES` Protocol is purely a model-layer type change, independent of P7.
This was originally in P8 but has no dependency on the subsystem rewrite.

- *Files:* `pyphi/types.py`, `pyphi/metrics/distribution.py` (top), `pyphi/partition.py`
  (top), `pyphi/models/subsystem.py` (CES/FlatCES fix), new `pyphi/protocols.py`,
  `pyproject.toml` pyright config.
- *Leverage:* High — Projects P4–P9 all express themselves in these Protocols.

### Phase B — Formalism split (the architectural pivot)

**P4. Extract `PhiFormalism` and separate IIT 3.0 / IIT 4.0 into parallel packages**

Define:

```python
class PhiFormalism(Protocol):
    exact: bool                    # True for exact methods, False for approximations
    default_metric: DistanceMetric
    compatible_metrics: frozenset[type[DistanceMetric]]
    partition_scheme: PartitionScheme | None  # None for methods that bypass partitions (e.g. φ*)
    def evaluate_mechanism(self, cs, mechanism, purview) -> RIA: ...
    def evaluate_system(self, cs) -> SIA: ...
    def build_phi_structure(self, cs) -> PhiStructure: ...

class ExactFormalism(PhiFormalism, Protocol):
    """Formalism that computes exact values via exhaustive enumeration."""
    exact: Literal[True]
    partition_scheme: PartitionScheme

class ApproximateFormalism(PhiFormalism, Protocol):
    """Formalism that computes approximate values with error characterization."""
    exact: Literal[False]
    def error_characterization(self, cs) -> ErrorInfo: ...
    # ErrorInfo distinguishes: upper_bound, approximation_error, different_quantity
```

Design `PhiFormalism` broad enough for future approximation methods (φ\*, φ_G,
geometric integrated information). **Key insight from review:** different
approximation types have different error semantics — φ\* computes a *different
quantity* (no error bound), Zaeemzadeh gives *upper bounds*, heuristic CES computes
the *same quantity approximately*. Split the Protocol into `ExactFormalism` and
`ApproximateFormalism` with an `ErrorInfo` return type that distinguishes these cases.
`partition_scheme` is `Optional` because some methods (φ\*) bypass partitions entirely.
`formalism/approx/` is a placeholder for this future work.

Create `IIT3Formalism` and `IIT4Formalism` concrete classes under `pyphi/formalism/`.
Move `new_big_phi/` contents into `formalism/iit4/`. Move IIT 3.0-specific parts of
`subsystem.py` and the 3.0-specific distance metrics into `formalism/iit3/`. The
shared kernel stays in `core/`.

Remove `config.IIT_VERSION` as a runtime switch. Introduce `config.FORMALISM` holding
a `PhiFormalism` instance. Validate at construction that metric and partition scheme
are in `compatible_metrics`/`compatible_partitions` — incompatible combinations become
*impossible*, not silently wrong.

Eliminate the `subsystem.py:983-1018` branch. Dispatch `find_mip` through
`formalism.evaluate_mechanism`.

- *Why here:* Every later refactor has a different (simpler) shape once you don't have
  to preserve both formalisms in the same hot path. Path-dependence: the cost of doing
  this at step P4 is much less than the cost of doing it at step P7, because by step P7
  every intermediate refactor has picked up a new dependency on the conflated code path.
During P4, also sketch the `FormalismConfig` portion of the config split (from P10)
to avoid making config decisions that P10 later reverses.

- *Files:* `pyphi/subsystem.py:983-1018`, `pyphi/new_big_phi/__init__.py` (moved),
  `pyphi/compute/subsystem.py` (contains IIT 3.0 `big_phi` path — must move to
  `formalism/iit3/`), new `pyphi/formalism/` package, `pyphi/conf.py`.
- *Risk:* High — the largest behavioral refactor on the list. Mitigated entirely by P1+P2.
- *Leverage:* Massive. Unblocks P5–P10.
- *Style:* Big-bang at the commit level; optional env-var gate for one release.

**P5. Unify distance metric API**

Promote the 4.0-style signature as the single `DistanceMetric.__call__` signature:

```python
def __call__(
    self,
    repertoire: Repertoire,
    state: State | None = None,
    *,
    partitioned: Repertoire | None = None,
    selectivity: Repertoire | None = None,
) -> DistanceResult: ...
```

When `state is None`, return a `DistanceResult`-valued array so the caller can argmax
once (the `subsystem.py:1144` TODO). IIT 3.0 metrics become a subclass that ignores
`state` and computes distribution distance; IIT 4.0 metrics use it. The branching
`if repertoire_distance in [...]` at `subsystem.py:1090-1142` is deleted; dispatch
is through the metric object.

Also **deprecate `DistanceResult.__array__`** with a warning, and provide a class
method `DistanceResult.values_array(results) -> np.ndarray` as the recommended
replacement. The current design silently drops metadata when coerced into numpy
arrays, defeating the class's purpose in batch workflows. After one release cycle
with the deprecation warning, remove `__array__`.
`__float__` is provided for convenience. `PyPhiFloat` remains for precision-aware
comparison.

**Three `PyPhiFloat` fixes that pair with this work** (surfaced during P0):
1. `__eq__`/`__ne__` return `False`/`True` for non-numeric types instead of
   `NotImplemented` (`pyphi/data_structures/pyphi_float.py:54-58`). This breaks
   Python's reflective comparison fallback and prevents `pytest.approx` from
   matching. Fix: return `NotImplemented` for unknown types.
2. `__hash__` reads `config.PRECISION` at hash time, which is module-global
   mutable state hostile to free-threaded mode and to dict invariants if precision
   ever changes during a session. Fix: snapshot precision into the instance at
   construction (or at first hash; document the choice).
3. The `DistanceResult.__array__` deprecation above.

These three fixes together make `PyPhiFloat` no-GIL safe and well-behaved with
external libraries (pytest, numpy).

**Signed-phi metadata refinement** (surfaced during P1, see grid3 fixture
TODO). PyPhi currently drops the |·|+ operator from Eqs. 19-20 of the IIT 4.0
paper to give users visibility into "preventative" causal structure (mechanism
that decreases probability of a specified state) and the analogous
system-level case. This creates a real internal inconsistency:

- The IIT 3.0 paper (Box 1 Glossary, p. 8) and IIT 4.0 paper (text after
  Eq. 19, Eq. 23) both unambiguously define φ as a *loss* of intrinsic
  information — non-negative by construction. The |·|+ is not optional;
  it's the formal expression of "we only count cases where the system
  raises the probability of the state relative to the partitioned probability".
- PyPhi's MIP selector at `new_big_phi/__init__.py:498` is
  `(normalized_phi, -phi)` → `argmin(signed)` picks the *most negative*
  partition, which is "the partition with the strongest preventative
  effect" — the *opposite* of MIP semantics ("the partition that makes
  the least difference"). For grid3 (1,0,0), this changes the selected
  partition: argmin(signed) picks `2 parts: {A,BC}` (norm = -0.0182);
  argmin(|·|+) would pick `3 parts: {A,B,C}` (smallest signed |·|).
- The 2026 cap (`min(phi, i_diff, i_spec)`) is degenerate when phi is
  signed: `min(positive_caps, negative_phi) = negative_phi`, so the cap
  never enforces anything when 2023 phi is preventative.

The fix:

```python
class SystemIrreducibilityAnalysis:
    # Metadata: raw signed values for visibility
    signed_phi: float
    signed_normalized_phi: float

    # Primary: paper-faithful, non-negative
    @property
    def phi(self) -> float:
        return max(0.0, self.signed_phi)

    @property
    def normalized_phi(self) -> float:
        return max(0.0, self.signed_normalized_phi)


def sia_minimization_key(sia):
    # Operate on |·|+ values: argmin selects smallest deviation
    # (paper-faithful MIP). Tie-break unchanged.
    return (sia.normalized_phi, -sia.phi)  # now non-negative
```

The 2026 cap correspondingly operates on `max(0, ·)` values, so the cap
becomes a meaningful upper bound (`min(non-neg, non-neg) ≥ 0`) rather
than a no-op when any term is negative.

**Distinguish mechanism-level vs system-level signed phi.** At the mechanism
level, negative φ_d means the mechanism is a *preventative cause* (lowers
probability of the purview state). At the system level, negative φ_s means
*the partition increases specification of the system state* (geometric
quirk of factored vs joint distributions). Different phenomena; should be
labeled differently in metadata: `MechanismRIA.preventative_magnitude`
vs `SIA.partition_strengthens_specification`. Both are signed metadata
but they mean philosophically different things.

**Impact on golden fixtures:** `grid3_iit4_2023` and `grid3_iit4_2026`
currently pin SIA phi = -0.0729. Under this redesign they pin SIA phi = 0
(grid3 in (1,0,0) is reducible per the paper-faithful reading) with the
metadata showing the strongest preventative magnitude. P5's
`--regenerate-golden` step will refresh those values. This is the textbook
case where the harness catches an intentional formula change.

- *Why here:* Metric signature inconsistency is the specific seam that forces
  `intrinsic_information()` to have two code paths. Once 3.0 is cordoned off (P4),
  4.0 metrics unify without worrying about 3.0 metric shape. The signed-phi
  refinement piggybacks naturally because it lives at the metric/distance
  result boundary.
- *Files:* `pyphi/metrics/distribution.py` (1041 lines → split into `base.py`,
  `iit3_metrics.py`, `iit4_metrics.py`, `distance_result.py`),
  `pyphi/data_structures/pyphi_float.py`, `pyphi/new_big_phi/__init__.py:498`
  (MIP selector), `pyphi/subsystem.py:800-1200`, `pyphi/repertoire.py`.
- *Risk:* Medium-high — changes user-facing phi values on at least one
  fixture (grid3) and any other state where preventative phi was being
  reported. Covered by P1 fixtures + Hypothesis invariant tests in P2
  (the invariant `phi >= 0` becomes enforced rather than aspirational).
- *Leverage:* High. Unblocks P7 (subsystem rewrite) by removing the biggest
  internal conditional. Also resolves the grid3 anomaly and makes the 2026
  cap mathematically meaningful.

**P6. Partition algebra consolidation with typed sum type**

Unify `Cut`, `SystemPartition`, `GeneralKCut`, `KPartition`, `Bipartition`, and
disintegrating partitions under a single `Partition` algebraic datatype. Critically,
**distinguish `DisintegratingPartition` (Θ(M,Z), Eq. 29) from `SystemPartition`
(Θ(S), Eq. 14-18) in the type system** — they have different mathematical roles and
use different probability constructions (`π_c` product probabilities vs `p_c`).

Verify during implementation that the current 4.0 distinction path's partition
usage is mathematically correct against Eq. 29 of the 4.0 paper. The mechanism-level
bipartitions used by `find_mip()` via `mip_partitions()` appear correct for
distinction computation (mechanism-purview bipartitions are the right family here).
The "disintegrating partition" concept from the paper applies at the **system level**
(already handled via `system_partitions` in `new_big_phi/__init__.py:570-575`).
Earlier analysis may have conflated these two levels. Regardless, verifying partition
correctness against the paper is a mandatory part of P6, not optional.

Resolve the `partition.py:643` TODO by making `Cut` an alias for
`SystemPartition` with `δ = →`. Partition schemes become typed functions returning
`Iterable[Partition]`, registered against typed schemes not strings.

- *Why here:* Partitioning is the combinatorial heart of phi. Doing it after P4-P5 means
  the partition algebra can assume typed TPMs, typed repertoires, and a clean 4.0 boundary.
Move partition-specific generation (`set_partitions()` from `combinatorics.py`) into
`partition/algebra.py`. The broader `combinatorics.py` refactoring into a package
(splitting `sets.py`, `states.py`, `analytical.py`, `graphillion_utils.py`, absorbing
from `utils.py`) is **deferred to P15** to avoid scope creep — `combinatorics.py` is
only 317 lines and the split doesn't block any downstream project.

- *Files:* `pyphi/partition.py` (811 lines), `pyphi/models/cuts.py`,
  `pyphi/subsystem.py:780-930`.
- *Risk:* Medium-high combinatorial (latent bug possible). Property tests on partition
  counts are mandatory.
- *Leverage:* Unblocks P7 and the Zaeemzadeh bounds project (P13).
- **CHECKPOINT:** If P6 discovers the suspected partition-family bug, the golden
  fixtures from P1 canonicalize wrong intermediate values. After P6, **regenerate
  golden fixtures** from the corrected code, validated against Albantakis et al. 2023
  Fig. 2 distinction values. This is a mandatory regeneration checkpoint.

**P6a. Lazy graphillion import + module-level globals audit**

P0 verification confirmed that `graphillion`'s `_graphillion` C extension does not
declare `PyMod_GIL_NOT_USED`, so loading it under free-threaded Python re-enables
the GIL process-wide. Today `import pyphi` eagerly loads graphillion (via
`relations.py:15`), which means even workers that never compute relations pay the
GIL re-enablement cost. P6a addresses both this and the broader globals problem:

1. **Defer graphillion imports** to function bodies in `relations.py` and
   `combinatorics.py`. Workers that compute mechanism-level φ or unfolding without
   relations stay no-GIL safe.
2. **Audit module-level globals that block no-GIL safety:**
   - `pyphi.config` — global mutable singleton; multiple worker threads
     reading/writing different settings is the classic hazard. P10 fixes this
     properly; P6a establishes the audit and adds a test that `config` is not
     written across threads in any code path.
   - `PyPhiFloat.__hash__` reads `config.PRECISION` at hash time → snapshot
     precision into the instance at construction. (Pair with the `__eq__`
     `NotImplemented` fix from P5.)
   - `np.random` global state → audit for any uses; replace with
     `np.random.Generator` instances.
   - `pyphi.log.TqdmHandler` — verify thread safety.
   - Any `lru_cache` or `cache` decorators on module-level functions — verify the
     cache itself is thread-safe (it is, in CPython 3.13t).
3. **Add a no-GIL CI matrix entry** that runs the test suite on Python 3.13t
   (`PYTHON_GIL=0`). Mark known-failing tests `xfail(strict=True)` until P6b lands.

After P6a, `pyphi.iit4.phi_structure(subsystem)` on a small network without
relations enabled should run in a no-GIL Python with no GIL re-enablement
(verifiable via `sys._is_gil_enabled()` returning `False` after the call).

- *Why here:* Independent of P7's critical path; could land any time after P3.
  Placed after P6 so the combinatorics package structure exists.
- *Files:* `pyphi/relations.py`, `pyphi/combinatorics.py` (or the new package after
  P6 split), `pyphi/data_structures/pyphi_float.py`, `pyphi/log.py`,
  `.github/workflows/test.yml` (no-GIL matrix entry).
- *Risk:* Low. Defensive change.
- *Leverage:* Unblocks `LocalThreadScheduler` in P11 for the non-relations
  path (no-GIL workers for mechanism-level parallelism).

**Status (landed 2026-05-06, commit `51062319`):** Complete. Graphillion
import deferred to function bodies in `pyphi.relations` and
`pyphi.combinatorics`; `import pyphi` no longer loads graphillion eagerly.
Globals audit findings: `np.random` global state not used (existing code
uses `np.random.Generator` instances in `tpm.py`/`dynamics.py`);
`log.TqdmHandler` thread-safe via inherited `StreamHandler` lock and
`tqdm.write`'s documented thread safety; `PyPhiFloat.__hash__` already
snapshots `config.PRECISION` at construction (P5 commit 1). Cache decorator
nonlocal hits/misses counters race on 3.13t but only affect counter
accuracy, not correctness — deferred to a P11 cleanup. No-GIL CI matrix
entry deferred to P11 (parallelization redesign already plans a CI
overhaul). Test `test/test_lazy_imports.py` pins the deferred-import
contract.

**P6b. ZDD library migration to OxiDD**

Replace `graphillion`'s `setset` family algebra with `OxiDD`'s `zbdd` primitives.
This addresses three problems:

1. **graphillion bus factor.** Single maintainer, slow release cadence, no PyPI
   wheels for Python 3.13+. OxiDD is multi-contributor, ships wheels for cp39-cp314
   including cp314t (free-threaded), and has been actively developed (last release
   2026-03, recent commits weekly).
2. **Free-threaded compatibility.** Even with P6a's lazy import, workers that
   compute relations re-enable the GIL because graphillion uses module-global
   state (`setset.set_universe()`). OxiDD uses manager objects per-call — fully
   no-GIL safe.
3. **Install ergonomics.** graphillion requires source build on macOS with
   `brew install libomp` workaround. OxiDD ships universal binary wheels.

**Architecture:** Hide the ZDD layer behind a `ZDDFamily` Protocol in
`pyphi/combinatorics/zdd_family.py`. Implement two backends:
- `GraphillionFamily` — current behavior, retained for one release as fallback
- `OxiDDFamily` — new default

Reimplement the high-level operations against ZBDD primitives:
- `powerset_family(X, min_size, max_size, universe)` — recursive `subset0`/
  `subset1` decomposition
- `union_powerset_family(sets, ...)` — fold `union` over per-set powersets
- `set_size_family(family, k)` — size filter, ~30 lines of recursive ZBDD walk
  (no direct primitive in OxiDD; pin with Hypothesis tests on partition counts)

Update `pyphi/relations.py` to use the new abstraction. The `ConcreteRelations` /
`AnalyticalRelations` split stays.

**Migration path (one release of overlap):**
- Default backend selectable via `config.ZDD_BACKEND` (values: `"oxidd"` (default
  in 2.0), `"graphillion"` (legacy fallback))
- If `OxiDDFamily` raises on a workload, user can fall back to graphillion to
  unblock; we triage and fix
- Drop graphillion in 2.1 if no fallback usage reported

- *Why here:* P6 has already created the abstraction seam. P6a has audited and
  fixed module-level globals. P6b is the actual swap.
- *Files:* New `pyphi/combinatorics/zdd_family.py`, `oxidd_family.py`,
  `graphillion_family.py`. `pyphi/relations.py`, `pyphi/combinatorics.py` (now
  package). Tests under `test/test_zdd.py`. `pyproject.toml` (`oxidd` dep added,
  `graphillion` becomes an optional fallback).
- *Risk:* Medium-high. OxiDD is younger than graphillion (~6 months public PyPI
  history). Mitigations: (a) GraphillionFamily fallback retained; (b) Hypothesis
  tests on `set_size_family(k)` partition counts since this is the only
  reimplementation that could introduce mathematical bugs; (c) golden fixtures
  from P1 catch any numerical regressions.
- *Leverage:* High — removes the bus-factor-1 dependency, simplifies install,
  enables free-threaded mode for the relations path. Most other workflows are
  already free-threaded-safe after P6a.
- *Style:* Big-bang behind the abstraction seam.

**Status (deferred to post-roadmap, 2026-05-06):** Deferred until after the
main 2.0 roadmap (P7–P16) ships. Reasoning:

- P6a already cleared the urgent piece (no-GIL-safe `import pyphi`). The
  remaining payoff (no-GIL for relations workflows + macOS install
  ergonomics) is real but not blocking the 2.0 work.
- No downstream project in P7–P10, P12–P16 depends on P6b. P11
  (parallelization redesign) wants `LocalThreadScheduler` for relations
  paths, but P11 can ship without it — relations workflows fall back to
  process-based parallelism, a bounded limitation.
- The OxiDD design will be better-informed after P11 lands (we'll know
  exactly which threading patterns are actually wanted, instead of
  designing speculatively).
- graphillion is used in only 2 files (relations.py, combinatorics.py)
  and a handful of functions. Migration cost won't grow much during the
  rest of the roadmap.

After 2.0 ships, P6b becomes **P17** on the post-2.0 list. graphillion
bus-factor risk is acknowledged: if graphillion goes unmaintained
mid-roadmap, P6b promotes back to in-scope.

### Phase C — Kernel rewrite

**P7. Big-bang layered rewrite of `subsystem.py`**

Replace the 1422-line `Subsystem` god-object with a layered architecture:

- `CausalModel` — immutable: substrate + TPM. Zero computation.
- `CandidateSystem` — immutable: `(CausalModel, state, node_subset, cut)`. This is
  what `Subsystem.__init__` should have been. Exposes cheap derived properties only.
  `cut` is a constructor arg, not a hidden mode.
- `RepertoireAlgebra` — stateless functions taking `CandidateSystem` and computing
  repertoires. Caching is a decorator applied at this layer, **not** hidden state
  inside the system. This retires the `subsystem.py:99` 4-cache TODO by making caching
  one explicit memoization boundary keyed on
  `(CandidateSystem, mechanism, purview)`.
- `MechanismEvaluator` — parameterized by a `PhiFormalism`, implements
  `find_mip`, `phi`, `concept`. Stateless per call.
- `PhiStructureBuilder` — top-level driver, delegates to `formalism.evaluate_system`.

The `Concept.subsystem` back-reference (`compute/subsystem.py:89,113`,
`models/subsystem.py:198` TODOs) vanishes because evaluation takes an explicit
`CandidateSystem` argument.

`CausalModel.causal_marginalization(direction)` explicitly returns `CauseTPM` or
`EffectTPM` subtypes, labeled in the docstring against Eq. 3 and Eq. 4. The current
`_backward_tpm()` call in `Subsystem.__init__` is retired as an implicit side effect
and becomes a named, documented operation.

**Begin with a compatibility analysis of PR #138 and PR #105.** PR #138 (Feb 2026) is
recent and aligned; review its `unit.py` and `substrate.py` as candidates for
`core/unit.py` and `core/substrate.py`. PR #105 (diverged ~2019, predates IIT 4.0)
is a **design reference** for implicit TPMs and non-binary support, not ready code —
expect significant reconciliation work. Verify these PRs' abstractions are compatible
with each other and with the target architecture before committing to either.

P7 must also **port `MacroSubsystem`** as part of the rewrite. This is non-trivial:
`MacroSubsystem` (macro.py:161) subclasses `Subsystem` and its `__init__` calls
`super().__init__()` then **mutates** `self` via `SystemAttrs.apply()` (macro.py:238),
which directly assigns to 7 attributes (`cause_tpm`, `effect_tpm`, `cm`,
`node_indices`, `node_labels`, `nodes`, `state`) post-construction. This is
*fundamentally incompatible* with frozen value types. A "thin adapter" cannot bridge
this gap. The fix is to redesign `MacroSubsystem`'s constructor pipeline as a
`CausalModel → transform → CandidateSystem` chain (coarse-graining as a functor
from one model to another), which is architecturally correct but adds significant
scope to P7. Budget accordingly.

**This is the one project that must be big-bang, not incremental.** Incremental
extraction of a class with shared mutable cache state and conditional version
dispatch preserves the coupling it's trying to remove.

**Note:** Subsystem's real public surface is **not** narrow — there are 58 public
methods, 10 properties, 20+ import sites across the codebase, and MacroSubsystem
subclassing with mutation. However, the `SubsystemInterface` Protocol defined in
P3 serves as the explicit contract. The rewrite must satisfy that Protocol exactly;
everything not in the Protocol is free to change. This is why P3 is load-bearing.

- *Why here:* The center of gravity. Must follow P4–P6 because those remove the
  ambient coupling that would otherwise leak into the new layering.
- *Files:* `pyphi/subsystem.py` (deleted/replaced), new `pyphi/core/` package
  (`causal_model.py`, `candidate_system.py`, `repertoire_algebra.py`,
  `mechanism_evaluator.py`, `phi_structure_builder.py`), `pyphi/models/mechanism.py`
  (remove subsystem backreference), `pyphi/compute/subsystem.py`.
- *Risk:* Very high in absolute terms, but P1–P6 have been specifically sequenced to
  reduce it. Golden fixtures catch regression; Protocol types catch interface
  mistakes at type-check time; the formalism split has already excised the biggest
  conditional.
- *Leverage:* Enormous. Unblocks everything else.
- *Style:* Worktree-based big-bang. One PR, frozen old module until cutover.
  Optional `PYPHI_NEW_CORE=1` env flag for one release.

### Phase D — Model cleanup and consolidation

**P8. `models/mechanism.py` split + Distinction type + Φ-folds**

Split `models/mechanism.py` (1216 lines) into `ria.py`, `state_spec.py`, `mice.py`,
`distinction.py` (new, 4.0-native replacement for `Concept`). Replace hand-rolled
`__eq__`/`__hash__`/`__repr__` with frozen `dataclass` / `attrs` definitions. Now
that `Concept` holds no subsystem backreference (from P7), it is a genuine value type.

**Add `models/phi_fold.py`:** Φ-folds are a core IIT 4.0 concept (Albantakis et al.
2023 Section 2.2.3, Eq. 3-4) — a sub-structure of a Φ-structure consisting of a
single distinction and all relations involving it. Currently implemented only in the
external `~/projects/matching/matching/phi_fold.py` (111 lines), but they belong in
core PyPhi since they characterize how each distinction contributes to total Φ.
The `PhiFold` type is the natural unit for computing Φ_d(C(d(m))) — each
distinction's share of the overall Φ-structure. This is also foundational for the
matching/perception extension (P14b).

(The CES/FlatCES Liskov fix was moved to P3 since it's independent of P7.)

- *Files:* `pyphi/models/mechanism.py`, new `pyphi/models/distinction.py`,
  new `pyphi/models/phi_fold.py`.
- *Risk:* Low-medium. Golden fixtures cover this.
- *Leverage:* Medium. Unblocks P15 (Jupyter display), P14 (macro/actual rewrite),
  and P14b (matching/perception extension).

**P9. Unified cache observability + memory bound + dead-code removal**

**Status (landed 2026-05-08):** Done on `feature/p9-unified-cache`
(`f828d776..294a956f`, 10 commits, branch local-only).

Original plan was to replace the four `DictCache` instances with a single
memoization layer keyed on `(CandidateSystem, mechanism, purview)` plus a
`FrozenMap` canonical key type. After auditing the cache landscape (kernel
`_memoize`, module-level `@cache(...)` for combinatorics, instance-level
`DictCache`, `joblib_memory` for Hamming matrices) we agreed on a narrower
approach: each flavor solves a genuinely different problem, so unification
happens at the *observability/control* layer, not the *decorator* layer.

What landed:

- `CachePolicy` Protocol (`name`, `info()`, `clear()`) + process-local registry in
  `pyphi/cache/{policy,registry}.py`. Public surface
  `pyphi.cache.{info, clear_all, clear, register, unregister}` walks every
  registered cache uniformly.
- Kernel `_memoize` registers under `kernel.<fn>`; bounded by
  `MAXIMUM_CACHE_MEMORY_PERCENTAGE` via per-miss `cache_utils.memory_full()`
  check (closes the unbounded-growth hole in long notebook sessions).
- Module-level `@cache(...)` decorators in `partition.py` / `distribution.py` /
  `combinatorics.py` register under `<module>.<qualname>`.
- `Network.purview_cache` is anonymous (no registration). Originally
  registered under a per-instance name (`f"network.{id(network)}.purview_cache"`)
  for multi-Network visibility, but reversed when the registration was
  found to leak `PurviewCache` instances forever via the adapter's
  closure capture and to interact with `clear_all()` between tests in
  ways that crashed loky workers. Per-Network introspection is still
  available via `network.purview_cache.info()` (the `DictCache.info()`
  method) — `network` is the natural handle for that question.
  Transient `_ObjectCache` instances inside `jsonify.loads()` also stay
  anonymous.
- Threading assumption documented (single-threaded-per-process, Ray-isolated;
  no locks added).
- Dead code removed: `RedisCache` class (never instantiated in pyphi/, test/,
  docs/), `REDIS_CACHE` and `REDIS_CONFIG` config keys (zero live readers),
  conftest Redis fixture, `CACHING.rst` Redis section, benchmark Redis cache
  mode parameter, stale `|MICECache|` Sphinx alias. `joblib_memory` retained
  (live consumer in `metrics/distribution.py` for Hamming matrix disk cache).

What did *not* land (deferred):

- `(CandidateSystem, mechanism, purview)` content-hash key scheme. Kernel cache
  still keys on `id(cs)` per P7's design; works because `weakref.finalize`
  evicts on GC.
- `FrozenMap` as canonical key type in cache. Cosmetic; not load-bearing yet.
- Distributed / cross-process cache (the original `RedisCache` rebuild-on-config-change
  TODO at `cache/redis.py:37`). When this returns, it integrates with the
  `CachePolicy` Protocol established here. Likely surfaces during P11.
- **Mystery (largely resolved by Decision-7 reversal; partial deferral to P11):**
  During acceptance we found that `pyphi.cache.clear_all()` between every
  test + per-instance `Network.purview_cache` name registration caused
  `BrokenProcessPool: failed to un-serialize` worker crashes on
  `loky.get_reusable_executor`-based parallel cuts (golden `basic_iit3_emd`
  / `xor_iit3_emd`, IIT 3.0 + EMD). The registration triggers a closure
  (`lambda: (self.hits, self.misses)`) that keeps every PurviewCache alive
  forever in the registry; combined with `clear_all()` walking that
  growing list between every test, this produced the failure. We
  eliminated the leak by making Network purview caches anonymous (no
  registration). What we still don't fully understand is the mechanism
  by which the parent-side leak propagated to worker-side crashes —
  workers receive Networks via `cloudpickle` and never re-register on
  unpickle, so there's no obvious path from the parent's registry to
  worker state. Most plausible theory: timing/resource interaction
  between slow `clear_all` walks and loky's worker IPC. P11's
  parallelization redesign is the natural place to revisit if it
  surfaces again.

- *Files:* `pyphi/cache/{__init__,policy,registry}.py`,
  `pyphi/core/repertoire_algebra.py`, `pyphi/network.py`, `conftest.py`,
  `pyphi/conf.{py,pyi}`, `CACHING.rst`, `docs/conf.py`,
  `benchmarks/benchmarks/compute.py`. Tests:
  `test/test_cache_{policy,registry,integration}.py`.
- *Risk realized:* The `clear_all`-between-tests interaction described above.
  Worked around; deferred root-cause investigation.
- *Leverage:* `pyphi.cache.info()` enables observability for P13 benchmarking.
  `CachePolicy` Protocol is the integration target for any future distributed
  cache backend.

### Phase E — Infrastructure refresh

**P10. Config split with result-object snapshotting — landed (2026-05-08)**

Three frozen dataclass layers (`FormalismConfig` 18 fields,
`InfrastructureConfig` 24 fields, `NumericsConfig` 1 field) wrapped in a
`ConfigSnapshot` value type, accessed through a `_GlobalConfig` facade
singleton. Hard break on flat uppercase access; layered reads
(`config.numerics.precision`), top-level writes routed via
`_FIELD_TO_LAYER` map (`config.precision = 6`), scoped writes via
`config.override(precision=6, parallel=True)` with build-time field-name
collision detection (zero collisions across all 43 options). Every
top-level result object (SIA for both IIT 3.0 and IIT 4.0, PhiStructure)
carries a `.config: ConfigSnapshot` set at construction time;
`result.config.as_kwargs()` round-trips through `config.override()`.
Layered YAML format supported via `pyphi.config.load_yaml` /
`pyphi.config.to_yaml`; old 1.x flat YAML raises with a rename-map
pointer.

**Deferred to follow-up:** `_GlobalConfig` is currently a thin facade
over the legacy `PyphiConfig` singleton (now `pyphi._conf_legacy`); a
future cleanup will replace it with a self-owning layered config and
delete the legacy module. The auto-load of `pyphi_config.yml` at import
time still uses the 1.x flat format; users opting into the nested format
invoke `pyphi.config.load_yaml(path)` explicitly.

**P10b. Finish the config cutover — drop ``_conf_legacy`` and the facade — landed (2026-05-09)**

Phase 1 (``__dir__`` for tab completion) landed at ``30a39700``; the
self-owning rewrite + module deletions landed across ``0236ea63`` and
``a4885a73``.

The three frozen dataclass layers (``FormalismConfig``,
``InfrastructureConfig``, ``NumericsConfig``) are now stored directly on
``_GlobalConfig`` and replaced via :func:`dataclasses.replace` on field
writes; there is no longer a wrapped legacy ``PyphiConfig`` instance
behind the facade. Validators move to per-layer ``__post_init__`` (the
generic ``Option`` descriptor's value-list / type checks are reproduced
where the suite exercises them); logging callback +
``distinction_phi_normalization`` warning move to
``pyphi/conf/_callbacks.py`` with an explicit ``mark_loaded`` flag that
suppresses warnings during default-state setup. ``fallback`` and
``parallel_kwargs`` move to ``pyphi/conf/_helpers.py``.

YAML auto-load of ``pyphi_config.yml`` at import time uses the layered
nested format; ``pyphi_config.yml`` is migrated. Legacy uppercase keys
raise :class:`ConfigurationError` with a pointer to the rename map.

Public surface: flat (``config.precision``), layered
(``config.numerics.precision``), and legacy uppercase
(``config.PRECISION``) reads/writes all work; the uppercase form is
syntax sugar via case-folding. Wholesale layer replacement
(``config.numerics = NumericsConfig(...)``) is now supported (was
intentionally blocked during the cutover phase). ``override``,
``snapshot``, and ``install_snapshot`` keep their semantics.

``pyphi/_conf_legacy.py`` and its stub deleted (~1100 lines);
``pyphi/conf/legacy_global.py`` renamed to ``pyphi/conf/_global.py``;
the descriptor-pattern tests in ``test/test_config.py`` and the
``Config`` / ``Option`` / ``PyphiConfig`` re-exports are dropped (no
public-surface users remained).

**Acceptance:** golden 17/17 unchanged, hypothesis 21 green, fast unit
lane 926 passed (down from 939 — descriptor-pattern tests deleted),
ruff clean, pyright at 2.0 baseline.

**Original P10 design** (kept for reference):

Split `pyphi/conf.py` (1121 lines) along the three layers that P4–P9 reveal:

- `FormalismConfig` (~15 options) — bundled into the `PhiFormalism` object.
- `InfrastructureConfig` (~20 options) — parallelization, caching, logging.
- `NumericsConfig` (~5 options) — precision, comparison tolerance.

The global singleton becomes a thin facade dispatching to these. **Every result
object gets a snapshot of the relevant config layer attached** (the PROJECTS.md
"attach config to SIA" item, generalized to all result objects). Use frozen
dataclasses; `pydantic` only for YAML loading if desired, not in the hot path.

- *Why here:* Needs to follow P4–P7 because only then do you know which config keys
  are formalism-scoped, which are infrastructure, and which are global.
- *Files:* `pyphi/conf.py` → `pyphi/conf/formalism.py`, `infrastructure.py`,
  `numerics.py`, `legacy_global.py`. `pyphi/conf.pyi`. All call sites using `config.*`.
- *Risk:* Medium. Touches user-facing API; keep `from pyphi import config` working.
- *Leverage:* High for reproducibility; unblocks P11 (parallelization config
  threading to workers).

**P11. Parallelization redesign with `Scheduler` Protocol**

Define `Scheduler` protocol: `map_reduce(fn, items, reducer) -> result`. Concrete
implementations:

- `LocalProcessScheduler` (`joblib + loky`; not Ray — the PROJECTS.md entry is
  stale). Default for GIL-enabled runtimes.
- **`LocalThreadScheduler`** (`concurrent.futures.ThreadPoolExecutor`). Default
  for free-threaded runtimes (Python 3.13t and later). **Requires P6a + P6b
  complete** (graphillion drop and globals audit). Big wins: shared cache
  across workers, no pickling cost for `Subsystem`/`CandidateSystem`/`Repertoire`,
  generators cross thread boundaries without materialization, cheap spawn for
  small tasks.
- `DaskScheduler` using `dask.distributed` + `dask-jobqueue` for SLURM/PBS/LSF/SGE.
- `HTCondorScheduler` via `htcondor-dask` or a direct `condor_submit` adapter.

**Runtime backend selection.** At scheduler construction:
```python
def default_local_scheduler() -> Scheduler:
    if not sys._is_gil_enabled():
        return LocalThreadScheduler()
    return LocalProcessScheduler()
```
Users can override via `config.PARALLEL_BACKEND`.

Clean separation between *algorithmic* tree-reduction (`parallel/tree.py`) and
*backend-specific* work dispatch. Propagate through all the `TODO(4.0) parallelize`
call sites in `compute/subsystem.py`, `new_big_phi/__init__.py:795,802,810`.

Implement generator-aware dynamic chunking — sample the first N tasks, estimate
per-task cost, then chunk the remainder by target batch wall time (~1s). This
addresses the `PROJECTS.md` heterogeneous-chunking concern: IIT iterates over
combinatorial sets whose elements range in size 1..(2^n − 1), so static chunking
is hostile. Generator composition is preserved throughout — dask and joblib both
support streaming. **Note:** chunking matters less for `LocalThreadScheduler`
(no pickling cost) but still matters for cache locality and progress reporting.

Re-enable parallel tests in CI (currently excluded). Until this project lands, mark
them `xfail` instead of `skip`. Add a no-GIL CI matrix entry that runs the
tests with `PYTHON_GIL=0` after P6b lands.

**Possibly investigate the P9 cache/loky mystery here.** P9 hit a
`BrokenProcessPool: failed to un-serialize` on
`loky.get_reusable_executor` parallel cuts (golden `basic_iit3_emd` /
`xor_iit3_emd`) when `pyphi.cache.clear_all()` ran between tests with
per-instance `Network.purview_cache` name registration enabled. We
removed the cause by making Network purview caches anonymous (eliminating
the registry leak), but the precise mechanism by which the parent-side
leak crashed workers was never confirmed — workers receive Networks via
`cloudpickle` and never re-register on unpickle, so there's no obvious
propagation path. If P11's loky/cloudpickle boundary audit doesn't
incidentally explain it, file as a curiosity rather than a P11 deliverable.
See ROADMAP P9 deferred items above for full detail.

- *Why here:* Mostly independent of P4–P9 because `parallel/` has its own clean
  abstraction, but needs P10's config snapshotting so workers receive an explicit
  config (or share one safely under no-GIL) instead of reading globals. Also
  depends on P6a + P6b for the `LocalThreadScheduler` to actually keep the GIL
  off; without those, `LocalThreadScheduler` would still work but graphillion
  imports would re-enable the GIL the moment relations are computed.
- *Files:* `pyphi/parallel/`, new `parallel/backends/{local_process,local_thread,dask,htcondor}.py`,
  `parallel/scheduler.py`.
- *Risk:* Medium for the process backend; high for the thread backend (concurrency
  bugs, race conditions on any remaining shared state). Mitigated by P2 property
  tests running in both modes and by the P6a globals audit.
- *Leverage:* Very high. Enables large-scale experiments; prerequisite for P13.
  No-GIL thread mode is potentially transformative for mechanism-level
  parallelism (small tasks, cache-friendly).

**Status (scope cut for 2.0, 2026-05-09):** P11 ships `LocalProcessScheduler`
+ `LocalThreadScheduler` fully implemented, plus a `DaskScheduler` skeleton
(import lazy, `map_reduce` raises `NotImplementedError`). The skeleton
exists to exercise the `Scheduler` Protocol against three call shapes
(loky pool, thread pool, dask client) so the abstraction is right; cluster
deployment fills it in later. **`HTCondorScheduler` and the full Dask
implementation defer to a post-2.0 follow-up project (`P18`)** — gated by
real user demand for SLURM/PBS/LSF/SGE/HTCondor cluster runs. The
Scheduler Protocol shape is the contract that unblocks them; nothing else
in the 2.0 roadmap depends on cluster backends.

**Status (done, 2026-05-09):** Landed on `feature/p11-parallelization-redesign`
across nine phase commits. Spec at
`docs/superpowers/specs/2026-05-09-p11-parallelization-design.md`; plan at
`docs/superpowers/plans/2026-05-09-p11-parallelization.md`. Notable
deliverables: `Scheduler` Protocol + three concrete schedulers (Process/Thread
implemented, Dask stub); per-call `ConfigSnapshot` propagation via closure
with worker-side dedup; `_GlobalConfig.install_snapshot()` for durable
snapshot application; cost-sampling chunking in `parallel/sampling.py`
replacing the dead-code `chunking.py`; frozen-formalism dataclass conversion
absorbing the deferred P10 Phase 4 follow-through; re-enabled parallel tests
in CI. **Deferred / open:** the P9 loky/cloudpickle `BrokenProcessPool`
intermittent (~50% rate) on `basic_iit3_emd` / `xor_iit3_emd` goldens —
test_golden_regression skips on detection rather than failing CI; root-cause
investigation queued as a separate post-2.0 item. No-GIL CI matrix (`PYTHON_GIL=0`)
still gated on P6b. No-GIL safety audit of `DictCache` likewise gated on P6b.

**P11.5. `pyphi.compute` relocation — done (2026-05-09)**

Folded `pyphi.compute.subsystem` into the IIT 3.0 formalism module
(`pyphi.formalism.iit3`) and `pyphi.compute.network` into substrate-level
helpers (`pyphi.network.systems`, `reachable_systems`, `possible_complexes`).
The legacy `pyphi.compute` namespace is gone.

**P11.6. Paper-aligned user-facing type rename — done (2026-05-09)**

Renamed `pyphi.Network → pyphi.Substrate` and `pyphi.Subsystem →
pyphi.System` to match IIT 4.0 paper terminology. The flat `Substrate`
holds TPM + connectivity matrix + node labels (no separate Topology
layer); `System` wraps a `Substrate` with state, node subset, and an
optional cut. Config keys `SUBSYSTEM_*` renamed to `SYSTEM_*`. The
package `pyphi.network_generator` became `pyphi.substrate_generator`.

**P11.7. Paper-aligned model-class rename: ``CauseEffectStructure`` /
``PhiStructure`` / ``Concept`` / ``Distinction``**

The IIT 4.0 paper's terminology and PyPhi's class names diverge in two
places. Both should be reconciled.

*Concept vs distinction.* IIT 3.0 uses *concept*; IIT 4.0 (Albantakis
et al. 2023) uses *distinction*. The user-facing canonical name in 2.0
is `distinction`. The IIT 3.0 native idiom (``concept(s, m, purviews=...)``)
remains accessible as `pyphi.formalism.iit3.concept` for IIT 3.0
callers; the top-level `pyphi.formalism.distinction` does not carry
the 3.0 ``purviews=`` kwargs. ``System.concept()`` is removed in
favor of ``System.distinction()``. *Status: in progress (split commit
on `feature/concept-distinction-rename`).*

*Cause-effect structure vs Φ-structure (Albantakis 2023, p11).* The
paper distinguishes two terms that PyPhi conflates:

- *Cause-effect structure* — distinctions + relations of *any candidate
  system* (reducible or not).
- *Φ-structure* — the cause-effect structure of a *complex*
  (a maximally irreducible substrate).

Today PyPhi's ``CauseEffectStructure`` class holds *just distinctions*
(no relations) — misnamed for IIT 4.0 — and ``PhiStructure`` holds
distinctions + relations regardless of complex status — misnamed
because Φ-structure carries the complex invariant.

The rename:

- ``CauseEffectStructure`` (current; just distinctions) →
  ``Distinctions`` (or ``DistinctionSet``).
- ``PhiStructure`` (current; distinctions + relations) →
  ``CauseEffectStructure`` — paper-aligned for any candidate system.
- ``System.phi_structure()`` method →
  ``System.cause_effect_structure()``. Available on any system.
- The term *Φ-structure* becomes a semantic / docstring term used in
  SIA contexts to communicate the additional complex invariant
  ("the cause-effect structure of a complex"), not a separate type.
  Optionally: a thin ``PhiStructure(CauseEffectStructure)`` subclass
  that SIA results use as a marker — only if it ends up load-bearing.

This is queued as a follow-on commit after the concept→distinction
rename lands. It touches model definitions, metrics, jsonify, golden
fixtures, and the public surface; expect comparable scope to P11.6.

**P11.8. Performance regression gate + benchmark suite rewrite**

*Motivation.* During the 2.0 work we shipped a 4-day-old structural
change to ``IIT4_2026Formalism`` (5 nested defensive ``config.override``
calls per partition) that interacted catastrophically with a 3-year-old
latent ``atomic_write_yaml`` callback to produce a 60-300x slowdown on
the 2026 hot path. Goldens caught nothing because they were correctness
gates, not performance gates; the suite wall-time crept from minutes to
13 minutes and that felt like *the cost of doing more work* until we
profiled. The 2026 issue is fixed (commit ``7c2e2cd2`` removes the YAML
write), but the lesson is: structural refactors in hot paths need a
perf gate, not just correctness gates. P12 (non-binary) and P13
(Zaeemzadeh pruning) both touch hot paths and are the next likely
sources of perf regressions; the gate should be in place before they
land.

*Why this is bigger than "wire ASV into CI".* The ``benchmarks/``
suite predates the 2.0 architecture by years — every import is broken
(``pyphi.Subsystem``, ``pyphi.compute``, ``examples.basic_subsystem``,
``subsys._repertoire_cache.clear()``). The class names use
"BenchmarkConstellation" — terminology that predates *concept*, let
alone *distinction*. The TODO at the top of
``benchmarks/benchmarks/subsystem.py`` notes that even
``@config.override`` couldn't be used "because it doesn't exist in the
entire project history". The suite is effectively a museum piece. We
have to design what to benchmark from scratch, in the 2.0 vocabulary,
on the 2.0 hot paths.

*Two-tier proposal.*

**Tier 1 — Immediate inline perf budget (small, ships first).** A
handful of ``pytest.mark.perf`` assertions in the fast golden lane:
``basic_iit4_2026 must complete in <5s``,
``basic_iit3_emd must complete in <3s``, etc. Wall-time-based, with
generous margins (~5x typical) so they catch *catastrophic* regressions
without being brittle on slow CI runners. Run on every PR. Cheap
insurance. ~30 lines of code. Catches the class of regression we just
hit (the 2026 path going from 0.3s to 84s wouldn't have survived a 5s
budget).

**Tier 2 — Rewrite the benchmark suite for the 2.0 architecture.**
Design questions worth taking seriously rather than mechanically
porting:

  - *What to benchmark, parameterized by what?* The 2.0 architecture
    has 3 formalisms × N substrates × {sequential, parallel} ×
    {warm cache, cold cache}. A naive cross product is too many
    combinations to track. Pick a small representative grid (e.g.
    {basic, xor} × {iit3_emd, iit4_2023, iit4_2026} × cold) and a
    smaller set of "edge" benchmarks for specific concerns
    (large substrate scaling, parallel speedup, cache hit-rate
    sensitivity).
  - *Which operations?* SIA dominates wall time; but smaller-grain
    operations (``find_mip``, ``find_mice``, single ``distinction``,
    repertoire computation) are useful for localizing regressions. A
    layered set rather than only end-to-end.
  - *Wall time vs counter-based.* Wall time is platform-sensitive
    (~2-3x variance across CI runners). Counter-based measurements
    (``cProfile`` call counts on hot frames, kernel cache hit/miss
    counts, partition evaluation counts) are more stable but require
    upfront design. ASV supports both; the question is which signal
    we trust to fail a build vs only post a warning.
  - *Threshold policy.* When does a 10% regression fail CI vs warn?
    20%? This is policy, not engineering — and the answer changes
    once we have a baseline.
  - *Where does ASV run.* Nightly on develop is the safe answer. PR
    integration is harder because ASV's "compare two commits" mode
    needs both checkouts available, which CI runners don't natively
    do.

*Sequencing.* Tier 1 lands as a small commit (couple hours of work)
*before* P12 starts. Tier 2 is its own multi-day project, scoped
between Tier 1 landing and P12 starting — or punted to P15 if
schedule pressure makes it impractical, but with the inline budget
tier in place to bridge the gap.

*Files.* ``benchmarks/`` (rewrite), ``benchmarks/asv.conf.json``
(point at ``2.0`` not ``develop``), new
``test/test_perf_budget.py`` (Tier 1), ``.github/workflows/`` (new
nightly), ``ROADMAP.md`` (mark P15's Layer D as superseded).

**P11.9. Congruence resolution: make tied-state semantics safe by construction — landed (2026-05-09)**

Implemented as **Option C** (type-level). ``Distinctions`` is now a
shared base class with two marker subtypes:
:class:`pyphi.models.distinctions.UnresolvedDistinctions` (raw
computation result; per-distinction tied states not disambiguated) and
:class:`pyphi.models.distinctions.ResolvedDistinctions` (post
``resolve_congruence``; required by
:func:`pyphi.relations.relations` and the ``distinctions`` field of
:class:`pyphi.models.ces.CauseEffectStructure`). The boolean
``_resolved_congruence`` flag and the runtime ``PyPhiWarning`` in
``relations()`` are deleted — passing an ``UnresolvedDistinctions``
where a ``ResolvedDistinctions`` is required is a static type error.

For degenerate substrates whose SIA is null (NO_STRONG_CONNECTIVITY
etc.), ``phi_structure`` resolves against
``system_intrinsic_information(system)`` instead of skipping
resolution. The fig5b paper-example fixture is regenerated from 2
unfiltered distinctions to 1 resolved distinction; all other phi_structure
fixtures are unchanged.

Landed at commit ``065e7228``.

**Original design discussion** (kept for reference):

In IIT 4.0, a distinction can have multiple *tied* specified states for
its cause and effect MICs. The "true" specified state of each
distinction is determined only after the system-level SIA produces a
``system_state`` direction; ``Distinctions.resolve_congruence(system_state)``
then filters and selects per-distinction states to match that direction.
Until resolution happens, downstream code that operates on the
distinctions (relations, plots, fold counting, ces_distance) can
silently produce wrong results — different distinctions may pick
different unresolved-tied states, relations between them may include
"phantom" overlaps that wouldn't exist after resolution, and the
output looks plausible.

Today's handling is brittle:

- ``Distinctions.__init__`` sets ``_resolved_congruence=False`` by
  default. ``Distinctions.resolve_congruence(system_state)`` returns a
  new ``Distinctions`` with the flag set to ``True``.
- ``pyphi.relations.relations()`` checks the flag and emits a
  ``PyPhiWarning`` (not an error) if it's ``False``. The warning is
  easy to miss, easy to silence, and arrives after the wrong relations
  have already been computed and possibly stored.
- The official ``pyphi.formalism.iit4.phi_structure()`` pipeline calls
  ``distinctions.resolve_congruence(sia.system_state)`` correctly. But
  any caller obtaining distinctions another way (``System.all_distinctions()``,
  loading from JSON, regenerating after a config change, manually
  constructing) gets unresolved distinctions and no enforcement.
- There is no type-level distinction between ``UnresolvedDistinctions``
  and ``ResolvedDistinctions``; the same Python class represents both
  states, with the difference encoded in a Boolean flag.

Three design options, in increasing order of investment:

**Option A — strict runtime (smallest change).** ``relations()``
raises instead of warns when ``resolved_congruence`` is False. Every
other downstream consumer (ces_distance, fold counting, visualize)
adds the same check. Same boolean flag, same pipeline; just
``error`` not ``warn``.

**Option B — lazy resolution.** Drop the flag entirely. ``Distinctions``
stores the unresolved-tied form. Every downstream operation that
needs resolved state takes ``system_state`` as a required argument
and resolves on the fly. Trades the flag for a parameter; less safe
than A if callers default ``system_state`` to something arbitrary.

**Option C — type-level (illegal states unrepresentable).**
Introduce ``UnresolvedDistinctions`` and ``ResolvedDistinctions`` as
distinct types. Functions like ``relations()``, ``ces_distance()``,
``CauseEffectStructure.__init__`` take only ``ResolvedDistinctions``;
trying to pass an ``UnresolvedDistinctions`` is a static type error.
Resolution is a typed transition: ``UnresolvedDistinctions.resolve(system_state)
→ ResolvedDistinctions``.

Recommendation: **Option C**. The IIT 4.0 paper's correctness story
already distinguishes "the bag of distinctions a candidate system
supports" from "the bag of distinctions whose states have been
disambiguated by the SIA winner" — those are different mathematical
objects, and PyPhi should treat them as different runtime types.
Option A is also acceptable as a faster fix if Option C scope is
prohibitive; Option B is not recommended because it pushes safety
onto every caller.

Sequencing: this is correctness work, not feature work, and could go
just before P14 (which will heavily exercise distinctions through
the macro path) or interleaved with P14 if the macro port surfaces
related issues. Either way before P12 — non-binary alphabets
multiply the number of tied states per distinction, making
incorrectly-handled congruence even more dangerous.

*Files (Option C scope):* ``pyphi/models/distinctions.py`` (split
class), ``pyphi/relations.py`` (tighten signature), ``pyphi/models/ces.py``
(``CauseEffectStructure`` requires resolved distinctions), the IIT 4.0
phi_structure pipeline (already correct, just retype), all callers of
``System.all_distinctions()`` (audit how the result is used), tests
for both branches.

---

## Updated 2.0 ordering (2026-05-09)

The original Phase A–H letters captured logical groupings (safety net,
formalism split, kernel rewrite, model cleanup, infrastructure refresh,
features, downstream cleanup, approximations). The phases are still
useful as conceptual labels, but recent work has produced enough new
information — the YAML-write performance bug, the ``actual.py``
breakage, the staleness of the benchmark suite, the size of the dark
test inventory — that a deliberate re-ordering pass is in order.

**Schedule for remaining 2.0 work:**

1. **P11.7 — CES / Φ-structure rename.** *(landed 2026-05-09, commit
   ``7c4bc012``.)* User-facing renames are free while pre-release;
   once 2.0 ships they cost a deprecation cycle. Closes the rename
   trilogy (Network/Subsystem, Concept/Distinction, CES/Φ-structure).

2. **P10b — Finish the config cutover.** *(landed 2026-05-09, commits
   ``30a39700`` / ``0236ea63`` / ``a4885a73``.)* ``_conf_legacy.py``
   deleted, ``_GlobalConfig`` is self-owning,
   ``pyphi.config.<TAB>`` shows leaf settings directly, YAML auto-load
   uses the layered nested format. Sequenced before P14 so the dark
   tests P14 will re-enable migrate exactly once against the final
   config API.

3. **P11.9 — Congruence resolution: type-level safety.** *(landed
   2026-05-09, commit ``065e7228``.)* Implemented as Option C:
   ``Distinctions`` split into ``UnresolvedDistinctions`` /
   ``ResolvedDistinctions`` marker subtypes. Functions that need
   resolved state (``relations``, ``CauseEffectStructure``) accept
   only ``ResolvedDistinctions`` — static type error otherwise. Boolean
   flag and runtime warning deleted. Degenerate-SIA fallback resolves
   against ``system_intrinsic_information`` (regenerated fig5b
   fixture).

4. **P14 — ``actual.py`` resurrection.** *(landed 2026-05-09, commit
   ``71d22611``.)* Scope narrowed to ``actual.py`` only.
   ``pyphi.actual.TransitionSystem`` (frozen dataclass parametric in
   ``Direction``) satisfies ``SystemPublicInterface``; ``Transition``
   becomes a frozen wrapper. The 826 lines of previously-skipped
   ``test/test_actual.py`` are back online, plus paper-fixture
   acceptance tests against Albantakis et al. 2019. Bundled config
   audit nests ``formalism`` into ``iit`` / ``actual_causation``
   sub-namespaces, applies a uniform ``*_distance`` → ``*_measure``
   rename map, and removes the orphaned concept-style cuts machinery.
   Macro work split off into its own paper-faithful project (item 10
   below).

5. **P11.8 Tier 1 — inline pytest perf budget.** *(landed; spec
   ``cad8a967``, plan ``2d88dd78``, implementation ``ea8ebcf6``.)*
   Inline ``@pytest.mark.perf`` wall-time floors on 5 hot-path
   fixtures (basic / xor / rule110 / grid3 / micro_s), catching
   catastrophic regressions of the form previously experienced
   (60–300x on ``IIT4_2026Formalism``). Calibrated to ``max(3.0,
   4× median)`` per fixture.

6. **P12 — Non-binary units.** With perf gate up and macro/actual on
   the new core, alphabet generalization is bounded: PR #105 is the
   reference; binary golden fixtures stay as oracles.

7. **P13 — Zaeemzadeh upper bounds.** Pure feature; depends on
   P12's alphabet generalization.

8. **P14b — Matching/perception fold-in.** Cleanest deferral
   candidate if the schedule slips: self-contained extension whose
   public surface is a new top-level package, so a 2.1 release is
   minimally disruptive.

9. **P11.8 Tier 2 + P15 — Surface-freeze bundle.** Benchmark suite
   rewrite, ASV-in-CI, ``jsonify`` retirement, test reorganization,
   docstring sweep, Sphinx architecture guide,
   ``__repr__`` / ``_repr_html_``, ``ToPandasMixin`` extensions,
   deferred-registry sweep (P9 loky mystery, no-GIL CI matrix entry,
   ``_conf_legacy.py`` confirmed gone), open-PR triage,
   ``migration-2.0.md``. Tier 2 of P11.8 lives here because the
   benchmark rewrite naturally pairs with the docstring/repr/jsonify
   pass.

   **Open subdecision: config storage construct.** ``pyphi.conf``
   today uses frozen :class:`~dataclasses.dataclass` instances for
   the three layers, with substantial custom scaffolding around
   them (``_GlobalConfig`` facade with ``__getattr__`` / ``__setattr__``
   / ``__getitem__`` / ``__setitem__`` / Mapping protocol routing,
   ``FIELD_TO_LAYER`` flat-name mapping, ``_rebuild_nested`` for
   immutable updates, layer-replacement callbacks). The scaffolding
   handles flat-namespace ergonomics that dataclass alone can't
   provide. Candidate alternatives, in increasing migration cost:

   - **dataclass (status quo).** stdlib, no dep, frozen + validators
     via ``__post_init__``, `fields()` introspection used by
     ``_iter_leaf_paths``. Custom scaffolding stays.
   - ``attrs``. Adds the dep; built-in ``@validator`` shortens
     ``__post_init__`` boilerplate; otherwise similar. Net win small.
   - ``msgspec.Struct``. The natural P15 alignment — designed for
     JSON / YAML / msgpack (de)serialization, tagged-union
     discrimination matching the canonical ``__class__`` shape that
     P11.87 documented in :mod:`pyphi.jsonify`, validators-as-types,
     much faster (de)ser. Substantial refactor but pays for itself
     when ``jsonify`` retires. **Recommended target.**
   - ``pydantic.BaseModel``. Rich validators, heavy dep, ergonomic
     ``model_dump`` / ``model_validate``. Overkill for an
     internal config store; better suited to user-facing input
     validation.
   - Flat ``TypedDict`` + dict-of-validators. Drops immutability +
     runtime validation; sidesteps the layered-class shape entirely.
     Loses too much.

   When P15 picks msgspec for serialization, migrate the config
   layers in the same pass — the canonical JSON shape already lines
   up with the tagged-union pattern msgspec supports.

   **Open subdecision: per-formalism IIT config split.** ``IITConfig``
   currently carries fields for both IIT 3.0 and IIT 4.0, with the
   ``version`` string switching dispatch. After P11.95d this is a
   visible code smell: ``presets.iit3`` overrides ~6 fields and 5
   other fields (``system_phi_measure``, ``specification_measure``,
   ``system_partition_include_complete``, ``relation_computation``,
   ``state_tie_resolution``, ``shortcircuit_sia``,
   ``distinction_phi_normalization``) are documented no-ops on the
   3.0 path — they exist on the dataclass but are guarded out at
   the call boundary. The implicit-default audit table in
   ``presets.iit3``'s comment block is the cost of that asymmetry.

   Proposed direction (pair with the construct subdecision above):
   sum-type ``config.formalism.iit: IIT3Config | IIT4Config`` with
   a shared base (or :class:`~typing.Protocol`) for genuinely common
   fields (``mechanism_partition_scheme``,
   ``purview_tie_resolution``, ``mip_tie_resolution``,
   ``assume_partitions_cannot_create_new_concepts``,
   ``single_micro_nodes_with_selfloops_have_phi``). Each variant
   carries only the fields its formalism's code actually consumes;
   IIT 4.0-paper-only fields literally don't exist on
   ``IIT3Config``. Switching formalisms swaps the variant rather
   than flipping a ``version`` string.

   Tradeoff: migration churn. Every consumer reading
   ``config.formalism.iit.X`` for X outside the shared subset has
   to narrow on the variant (or accept :class:`AttributeError` at
   call time). The compensation is that pyright catches "IIT 3.0
   code accidentally reads a 4.0-only field" at type-check time
   rather than via an audit six months later — exactly the class
   of error P11.95d's ``mip_tie_resolution`` finding represented.

   Pairs naturally with P11.95e (post-2.0 code-path-divergence
   audit): if ``IIT3Config`` lacks ``distinction_phi_normalization``
   as a field, the auditor doesn't need to ask "does any 3.0 code
   path read it?" — the type system has answered. Sum-type and
   msgspec migration land in the same P15 pass; the canonical
   JSON shape's tagged-union ``__class__`` discriminator extends
   naturally to discriminating ``IIT3Config`` from ``IIT4Config``.

10. **Macro framework — Marshall 2024 intrinsic units.** Deferred
    candidate, post-2.0 unless it slots in. Paper-faithful rewrite
    of the macro layer per Marshall, Findlay, Albantakis, Tononi
    2024 ("Intrinsic Units"): hierarchical meso constituents,
    sliding-window state mappings $g_J$ over $\tau$ micro updates,
    explicit background apportionment $W^J$, intrinsic-unit search
    via $\varphi_s$. Replaces legacy ``pyphi/macro.py`` outright.
    Disabled ``MacroSystem`` class and 593-line
    ``test/test_macro_system.py`` stay dark until this lands.

11. **P11.85 — Measure-API unification.** *(landed; spec ``90ca10be``,
    plan ``0da5d0ad``, implementation commits ``08a48e5a`` metric
    Protocol types + INTRINSIC_DIFFERENTIATION q-arg drop, ``e382afe7``
    StatefulDistributionMetric Protocol, ``d8ffd83a`` Phase 1 tighten,
    ``67caa112`` + ``6a12a4f6`` registry split + asymmetric-flag fix,
    ``f0ae1baa`` + ``4a4bc7cc`` + ``380dfa2b`` threading + strings →
    objects, ``5d648d94`` resolve_alpha_measure rename, ``1166ab67``
    explicit Protocol dispatch in repertoire_distance.)* Introduces
    calling-shape Protocols (``DistributionMetric``,
    ``StateAwareMetric``, ``StatefulDistributionMetric``,
    ``CompositeMetric``), tags each registered metric with its
    Protocol, and replaces string-based dispatch with Protocol-based
    dispatch. Folds in P5's deferred metric-API cleanups: return-type
    normalization, ``INTRINSIC_DIFFERENTIATION``'s vestigial ``q``
    argument, ``intrinsic_information`` config-read reduction.

12. **P11.86 — Explicit-parameter measure threading.** *(landed; key
    commits ``6e3eafb0`` formalism class methods accept explicit
    metric kwargs, ``003ee133`` Actual Causation metrics threaded
    through internal helpers, ``6c4bbdd9`` DRY the resolve+check
    pattern.)* Eliminated the ``with config.override(...)`` wrapper
    pattern in formalism class methods. Helpers
    (``_evaluate_partition_iit4``, ``_sia``, ``_phi_structure``,
    actual-causation internals) now take explicit measure / scheme /
    strategy parameters; public dispatchers read config and pass
    values down; formalism classes pass their own values when
    calling helpers. Eliminates the override pattern, the string-
    based dispatch handover from P11.85, and the parallel-state
    issue at its root. Cap-regression-impossible test pins the
    architectural guarantee.

13. **P11.87 — Cross-formalism SIA / CES surface unification.**
    *(landed; spec ``e699d031``, plan ``92e39b4c``, implementation
    ``50fdf92b``..``9dbf4bb4``, 17 commits.)* IIT 3.0 adopted IIT
    4.0's CES-wraps-SIA topology: ``iit3.ces()`` returns
    ``CauseEffectStructure(sia, distinctions, NullRelations())``;
    3.0 SIA dropped ``ces`` / ``substrate`` fields and renamed
    ``partitioned_ces`` → ``partitioned_distinctions`` (3.0-specific
    compute receipt). New ``SIAInterface`` /
    ``CauseEffectStructureInterface`` / ``AcSIAInterface`` Protocols
    (runtime_checkable). Common ``__repr__`` template via
    ``fmt_sia_columns`` / ``fmt_ces_columns`` / ``fmt_ac_sia_columns``;
    new ``_repr_html_`` on each class for Jupyter. Cross-class
    ``__eq__`` returns ``NotImplemented``. Canonical JSON shape
    documented in :mod:`pyphi.jsonify` module docstring (the target
    shape for P15's msgspec migration via tagged-union ``__class__``
    discrimination).

14. **P10c — Flat dotted-string config accessor.** *(landed; spec
    ``05f35a92``, plan ``dfadf9cd``, implementation ``575e152a``.)*
    The dotted-path read / write
    (``config["formalism.iit.mechanism_phi_measure"]``,
    ``config["numerics.precision"] = 6``) was already in place pre-
    P10c; this completed the Mapping protocol on ``_GlobalConfig``:
    ``__iter__``, ``__contains__``, ``__len__``, ``keys``,
    ``values``, ``items``, ``get(path, default)``. ``__getitem__``
    extended to accept bare leaf keys (``config["precision"]``) via
    the existing ``FIELD_TO_LAYER`` routing. Internal storage stays
    nested-dataclass.

15. **P11.95a — Deterministic SIA selection.** *(landed; see commits
    ``2a163b6f`` partition ``lex_key`` + ``resolve_ties.sias`` +
    ``sia_tie_resolution`` config, ``18137fef`` IIT 4.0 SIA MIP
    selection through ``resolve_ties.sias``, ``82b778ca`` IIT 3.0
    SIA ``lex_key`` fallback, ``ff064aef`` SIA-determinism property
    test, ``faafa374`` sequential/parallel split, ``fd135387``
    symmetric-substrate fixture regen + drop xfails.)* Added
    structural ``lex_key()`` on ``_PartitionBase`` and the
    ``PARTITION_LEX`` strategy in ``resolve_ties``; routed both IIT
    3.0 and IIT 4.0 SIA MIP selection through a deterministic
    cascade keyed off the new ``sia_tie_resolution`` config. Spec at
    ``docs/superpowers/specs/2026-05-12-sia-tie-breaking-design.md``.

16. **P11.95b — Paper-faithful state-tie resolution.** *(landed; see
    commits ``137a010c`` spec, ``77156e66`` cascade primitive,
    ``3415f199`` ``resolve_state_tie`` helper, ``18c3ff3e`` switch
    from cruelest-cut to paper-faithful, ``834aaf2f`` cruelest-cut
    narrative removal, ``85ae3f56`` perf budgets,  ``1baae7fb``
    distinction-state cascade in ``resolve_congruence``, plus fixture
    regenerations ``2c489439`` / ``b2d14e43`` / ``593adef4`` and
    ``4f802354`` cascade-invariant property tests.)* Per Albantakis
    et al. 2023 Eq. 12 + S1 Text, when multiple cause/effect states
    tie at max ``ii``, the canonical winner is the one with maximum
    unnormalized ``φ_s``. Replaces the prior cruelest-cut convention
    (``min_c min_P integration(P, c)``) with the paper-faithful
    ``max_c min_P`` reading via a generic ``resolve_state_tie``
    cascade. The chosen ``specified_state`` propagates through
    ``resolve_system_state`` into ``SIA.system_state``, filters
    distinctions via ``Concept.resolve_congruence``, and the cascade
    also resolves distinction-state ties.

17. **P11.95c — Substrate canonicalization for intrinsic equivalence.**
    Post-2.0 unless a smaller subset is folded in. PyPhi today computes
    Φ, distinctions, and per-direction breakdowns in a way that is
    sensitive to node labels. Two substrates that are permutations of
    each other (e.g., AND-XOR and XOR-AND with their nodes swapped)
    produce structurally equivalent CESes but PyPhi's outputs can
    differ in incidental fields — the chosen MIP partition, the
    cause/effect RIA's recorded phi at the MIP, the lex-canonical
    state when multiple states tie at φ_s. Cruelest-cut accidentally
    masked this for the per-direction breakdown by making the
    "integration value" symmetric in cause-spec choice; paper-faithful
    state-tie resolution (P11.95b) makes it visible.

    The theoretical desideratum: **PyPhi outputs should depend only on
    the substrate's intrinsic structure, not on its node labels.** The
    paper's S1 Text "extrinsic-tie / intrinsically identical CESes"
    escape clause names a special case of this; the broader
    requirement extends to cross-substrate isomorphism comparison.

    Three sub-cases to handle, in priority order:

    **(a) Cross-substrate isomorphism — primary.** Two distinct
    substrates whose TPMs differ only by a node-permutation π. Detection
    via graph canonicalization: compute a canonical form of each
    substrate's TPM (treated as a labeled, directed, weighted graph or
    appropriately-shaped tensor) and compare canonical forms. Isomorphic
    substrates have identical canonical forms. PyPhi can then:
    - Compute SIA/CES on the canonical form (so output is permutation-
      independent at construction), or
    - Compute normally and use canonicalization only to verify
      equivalence in tests / cross-substrate comparisons.

    The first is more invasive (changes how every result is presented
    to the user) but more honest. The second is a smaller change
    (canonicalization is a sidecar utility) but the user-visible API
    still reports label-dependent values.

    **(b) Structural CES coincidence without substrate isomorphism —
    deferred, opt-in.** Two genuinely-different substrates (not related
    by node permutation) whose CESes happen to have the same multiset
    of distinctions and same relation structure. Detection requires
    graph-isomorphism *on the CES itself* — treating each distinction
    as a labeled vertex (mechanism index in canonical labeling, cause
    state, effect state, φ_d) and relations as labeled edges
    (overlap set, φ_r).

    **Open theoretical question for (b):** is the CES a complete
    invariant of the substrate? That is, does ``CES(S1) = CES(S2)``
    (as labeled structure up to substrate relabeling) imply there
    exists a substrate isomorphism ``π`` with ``π(S1) = S2``? The
    ``←`` direction is trivial. The ``→`` direction is the load-
    bearing claim for case (b) being empty vs. non-empty. The CES
    is built from a very rich set of intrinsic-information
    computations (paper Eqs 5, 7, 19-23, 34-47); intuition says CES
    uniquely determines substrate up to isomorphism, but I have no
    proof and the analogous "graph reconstruction" problems in
    combinatorics have open cases. **Before committing implementation
    effort to (b), settle this question** — either by proof, by
    counterexample construction (search for non-isomorphic substrate
    pairs with matching CES via brute force on small n), or by
    citing existing literature. If CES is a complete invariant,
    case (a) suffices and (b) is empty; otherwise (b) is a genuine
    distinct case warranting an opt-in API.

    Computing the CES-graph-isomorphism check is doable but expensive
    (CES sizes can be in the thousands of distinctions for n=6+); it
    is not needed for the S1 escape clause as written. Reserve this
    for a research-mode opt-in API (e.g.,
    ``pyphi.intrinsic_equivalence(ces1, ces2)``) rather than wiring
    it into the SIA hot path.

    **(c) State-symmetric TPMs — primary.** A TPM may be invariant
    under a state permutation σ (e.g., a bit-flip symmetry on a
    homogeneous network where flipping every node maps each row/column
    to another row/column). This produces CES equivalence between
    states without a node permutation — the "automorphism" is on the
    state space, not the node label space. The canonicalization
    machinery should detect state-level symmetries as well; in
    practice this means computing automorphisms of the TPM viewed as
    a labeled tensor whose two axes are both state-indexed.

    **Dependency: ``pynauty`` (nauty bindings) — primary tool.**
    Hand-rolled graph automorphism is feasible for n ≤ 8 (O(n!) brute
    force ≈ 40k permutations for n=8, ms-level on the typical
    substrate) but scales poorly past n ≈ 10 and does not provide
    canonical labelings. Nauty handles both with mature C
    implementations; pynauty ships wheels and is widely used in
    combinatorics tooling. Alternatives (igraph, networkx) lack the
    canonical-form output. The dep adds ~500KB to install size; it
    fits in a new ``pyphi[symmetry]`` extra if we want it opt-in.

    **Module placement: new ``pyphi/automorphism.py``.** Single-purpose,
    easy to find. Exposes:
    - ``substrate_automorphisms(substrate) -> list[Permutation]`` —
      Aut(TPM) for case (c) and within-substrate ties.
    - ``substrate_canonical_form(substrate) -> Substrate`` — for case
      (a) cross-substrate comparison and canonical labeling.
    - ``are_substrates_isomorphic(s1, s2) -> bool`` — exposed for
      tests + downstream comparison.
    - Possibly ``ces_canonical_fingerprint(ces, substrate) -> bytes``
      for case (b), behind a feature flag.

    **Performance evaluation must answer:**
    - At what substrate size does canonical form computation become
      the SIA bottleneck (vs. partition enumeration, MIP search,
      CES build)? Likely n ≥ 10 if it's the bottleneck at all.
    - Are canonical forms cacheable per ``Substrate`` instance?
      Substrates are immutable; canonical form is a derived
      property; cache once per construction.
    - Is the "compute SIA on canonical form" path uniformly faster,
      slower, or wash vs. the current label-dependent path? Need
      benchmarks on the existing golden fixtures.
    - For case (b) opt-in: what is the CES-fingerprint cost for the
      worst-case fixture (logistic3, large rule110)? If it's a
      multi-second operation, it must stay strictly opt-in.

    **Test surface affected:**
    - ``test_invariants.py::TestPermutationSymmetry`` —
      ``test_sia_phi_c_symmetric``, ``test_sia_phi_e_symmetric``,
      ``test_system_state_symmetric``, ``test_sia_phi_symmetric``.
      Under P11.95b the per-direction breakdown asymmetry surfaces;
      under P11.95c the strict per-direction equality is restored.
      Until P11.95c lands, these tests are relaxed to multiset
      equality on ``(cause.phi, effect.phi)``.
    - Cross-formalism consistency tests benefit similarly:
      canonicalizing the substrate before SIA means the IIT 3.0 and
      IIT 4.0 outputs no longer depend on the arbitrary substrate
      labeling the user passed in.

    Estimated effort: 1-2 weeks for cases (a) and (c) with caching
    and benchmarks. Case (b) is a separate research-scoped project,
    weeks to months depending on the CES isomorphism approach. The
    public surface change for case (a) — "computation on canonical
    form, decanonicalize for output" — is invasive enough that it
    should be paired with a clear migration note in
    ``docs/migration-2.0.md`` (or 3.0.md if it slips post-2.0).

18. **P11.95d — IIT 3.0 tie resolution.** In-scope for 2.0 (required
    by ship criterion #3, which mandates unskipping
    ``test_actual.py``). Design spec at
    ``docs/superpowers/specs/2026-05-17-p11.95d-iit3-tie-resolution-design.md``.

    The IIT 3.0 restoration project (commits
    ``4055a682``–``760bf3bf``) surfaced three tie-resolution issues
    on the IIT 3.0 path, plus the ``purview_tie_resolution``
    diagnostic that the restoration already settled:

    - ``purview_tie_resolution`` (used by ``resolve_ties.purviews``
      at MICE selection) substantively affects ``sia.phi`` for
      substrates with phi-ties at the purview level. Measured at
      basic substrate ``(1,0,0)``: ``"PHI"`` and
      ``["PHI", "PURVIEW_SIZE"]`` converge on ``sia.phi = 2.3125``;
      on ``rule110 (1,0,1)`` they diverge (2.356476 vs 2.406476); on
      ``grid3 (1,0,0)`` they diverge again (0.026700 vs 0.016918).
      The canonical ``pyphi_config_3.0.yml`` value is the two-step
      ``["PHI", "PURVIEW_SIZE"]``. The current ``presets.iit3`` was
      reconciled to match in the restoration work, and the
      ``IITConfig.purview_tie_resolution`` annotation was tightened
      from ``str`` to ``str | list[str]`` to match
      ``resolve_ties.purviews``'s runtime acceptance. No further
      change.

    - **Cross-subsystem complex selection on tie is incidental.**
      When two subsystems tie at maximum ``Φ``,
      ``substrate.complexes`` falls through to ``partition.lex_key()``
      on each within-subsystem MIP — a property of cuts inside one
      subsystem used to compare across different subsystems. The
      post-``82b778ca`` cascade picks ``(0,2)`` for the worked
      example below by this route. The spec replaces this with a
      Determinism-level cascade that flags the tie as
      ``UNRESOLVED_WITHIN_BUDGET`` and signals exclusion-postulate
      failure, mirroring IIT 4.0's ``_resolve_clique_by_big_phi``
      behavior.

    - **``sia_tie_resolution`` is dead code on the IIT 3.0 path.**
      IIT 4.0's ``_find_mip_for_fixed_state`` consults
      ``sia_tie_resolution`` via ``resolve_ties.sias()``. IIT 3.0's
      ``_sia_map_reduce`` uses ``MapReduce(reduce_func=min)``
      directly and never reads the config. The spec refactors the
      3.0 path through ``resolve_ties.sias()`` and adds
      ``sia_tie_resolution = ["PHI", "PARTITION_LEX"]`` to
      ``presets.iit3`` — observable no-op, architectural unification.

    - **``mip_tie_resolution`` default applies IIT 4.0 normalization
      on the IIT 3.0 path** — a latent bug. The shared default
      ``["NORMALIZED_PHI", "NEGATIVE_PHI"]`` selects ``argmin
      normalized_phi`` (then ``argmax raw φ``), where
      ``normalized_phi = phi / num_connections_cut(partition)`` per
      the IIT 4.0 ``NUM_CONNECTIONS_CUT`` registry. The IIT 3.0
      paper (Oizumi et al. 2014) defines the MIP as raw
      ``argmin φ(cut)`` with no normalization — 3.0 dropped the
      system-level normalization that IIT 2.0 (Tononi 2008,
      ``Cut.normalized_phi``) used, and never introduced a
      mechanism-level normalization. The spec sets
      ``mip_tie_resolution = ["PHI", "PARTITION_LEX"]`` in
      ``presets.iit3`` — paper-canonical correction. This may shift
      some 3.0 goldens; the implementation order audits the impact
      between commits.

    **Worked example: standard substrate @ ``current_state=(0,0,1)``
    has two SIAs that genuinely tie at ``phi=1.0`` — the subsystem
    over nodes ``(1,2)`` (B=AND, C=XOR) and the subsystem over nodes
    ``(0,2)`` (A=OR, C=XOR). These are physically distinct subsystems
    (different gate compositions), not symmetry-equivalent. PyPhi
    1.x picked ``(1,2)`` by iteration order; the post-``82b778ca``
    ``partition.lex_key()`` route picks ``(0,2)``. The downstream
    consequence is observable: for the transition ``(1,0,0) → (0,0,1)
    → (1,1,0)``, ``true_events(substrate, *states)`` returns 2 events
    under ``(1,2)`` and 0 events under ``(0,2)``. Under the spec's
    new cascade, neither subsystem qualifies as a complex (the
    clique is indeterminate); ``maximal_complex`` returns a null
    SIA and ``true_events`` returns 0 events. The 2019 AC paper
    (p. 17) explicitly notes such ties are "undetermined".
    ``test/test_actual.py::TestActualCausationIIT30::test_true_events``
    is unskipped with expectations updated to match.

    Independent of P11.95a (deterministic floor) and P11.95b
    (paper-faithful state-tie for 4.0). Estimated 2-3 days
    including goldens regeneration for the
    ``mip_tie_resolution`` correction and the in-scope coverage
    expansion.

    **In-scope coverage expansion.** The
    ``mip_tie_resolution`` bug was caught only after goldens had
    locked the buggy values; the spec's audit-process post-mortem
    flags two coverage gaps that allowed this: the canonical
    SIA-φ reference (``basic_sia_phi_canonical.json``) only covers
    the default BI scheme, and per-(mechanism, direction) MIP
    partition shapes have no independent reference. The spec
    expands coverage in three ways within P11.95d:

    - Add ``basic_tri_sia_phi_canonical.json`` for the
      WEDGE_TRIPARTITION scheme (paper-canonical SIA-φ=2.520833).
    - Extend canonical references with per-mechanism MIP partition
      shapes, wired into the golden regression cross-check.
    - Add ``rule110_iit3_emd`` and ``grid3_iit3_emd`` fixtures —
      the tie-prone substrates the ``purview_tie_resolution`` audit
      already identified.

    **Implicit-default audit.** The natural follow-on question
    ("are other implicit 4.0-flavored defaults silently affecting
    the 3.0 path?") was settled during the P11.95d brainstorm by
    tracing each implicit ``IITConfig`` field for 3.0-path
    reachability. Result: after this spec's fix, every implicit
    4.0-flavored default the 3.0 preset doesn't override
    (``system_phi_measure``, ``specification_measure``,
    ``system_partition_include_complete``,
    ``distinction_phi_normalization``, ``relation_computation``,
    ``shortcircuit_sia``, ``state_tie_resolution``) is a no-op on
    the 3.0 path — each is guarded out at the call boundary or
    only consumed by 4.0-exclusive code. The
    ``mip_tie_resolution`` finding doesn't generalize to a class
    of latent bugs; it was specifically one knob with a real
    consumer on the shared ``formalism.queries.find_mip`` code
    path. Audit table is in the spec.

19. **P11.95e — IIT 3.0 code-path-divergence audit.**
    *Brainstorm only; post-2.0 unless a smoking gun appears.*
    The ``mip_tie_resolution`` bug was an instance of 4.0-paper-
    faithful logic baked into shared code that runs on the 3.0
    path. P11.95d settled the config-knob category exhaustively
    (see audit table). What remains is the broader question: are
    there shared mathematical functions on the 3.0 pipeline that
    use 4.0-paper-faithful logic without going through a config
    knob? Candidate inspection points:

    - ``RepertoireIrreducibilityAnalysis.__init__`` shared path —
      does any 4.0-paper-faithful clamp or filter leak into 3.0
      RIA construction? (The ``positive_part`` clamp at
      ``ria.py:131`` is shared; for 3.0 EMD distances are
      always non-negative, so the clamp should be a no-op —
      worth confirming with a single override-and-diff
      experiment.)
    - ``_compute_distinctions`` on the 3.0 path — does it apply
      any 4.0-only filters (specified-state congruence,
      relation-aware accounting)?
    - Other ``resolve_ties.*`` callers shared between formalisms
      whose strategy defaults are 4.0-flavored — most are now
      paper-canonical for 3.0 via ``presets.iit3``, but
      ``resolve_ties.partitions`` was the prior surprise so
      worth tracing the others (``resolve_ac_*``,
      ``resolve_distinction_tie``).

    Method: for each candidate, run the same 5-minute
    override-and-diff experiment against the IIT 3.0 goldens
    (now coverage-expanded per P11.95d). If no observable diff,
    the candidate is settled. Estimated 1-2 days for the full
    sweep once P11.95d's coverage expansion lands. Probably
    post-2.0 unless an obvious smoking gun appears mid-sweep.

**Ship criterion for 2.0:**

The original roadmap doesn't state a release criterion explicitly.
Codifying one now:

1. Every Greek letter in Albantakis et al. 2023 (and the 2026
   ii(s)-cap addendum) maps to a named runtime type in PyPhi —
   the "mathematician's acceptance test" from the original
   Verification Plan.
2. P11.7, P14, P11.8 Tier 1, P12, P13 are landed; P14b and
   P11.8 Tier 2 are landed *or* explicitly deferred to 2.1
   with a tracking issue.
3. Goldens (fast + slow) green; Hypothesis property suite green
   at default seed and on a 1000-run nightly; ``test_actual.py``
   is no longer skipped (the macro ``test_macro_system.py`` stays
   skipped pending the Marshall-2024 intrinsic-units project,
   which is post-2.0); perf budget green.
4. Sphinx site rebuilt; ``docs/migration-2.0.md`` ships;
   ``pyphi_config.yml`` auto-load uses the layered format
   (legacy YAML rejected with a rename map); ``_GlobalConfig``
   facade is gone (legacy backend self-owning); no
   ``TODO(4.0)`` or ``TODO(nonbinary)`` breadcrumbs survive in
   ``pyphi/``.
5. ``import pyphi; pyphi.iit4.cause_effect_structure(system)``
   runs to completion on the basic 3-node example with no
   ``DeprecationWarning`` from internal code paths.

Conditions 1, 4, and 5 together mean the public surface is stable
enough that 2.1 can be additive — which is the point of releasing
2.0 in the first place.

The phase-letter sections below remain authoritative for **scope**;
this updated ordering is authoritative for **schedule**.

---

### Phase F — Features and new algorithms

**P12. Non-binary (multi-valued) unit support**

Resolve the ~12 `TODO extend to nonbinary nodes` TODOs. Start from **two** existing
references: (a) the experimental `nonbinary` github branch from Gómez et al. 2020,
and (b) **PR #105 (implicit TPMs)**, which already implements per-node state-space
tracking via `state_space.py` and handles non-binary natively in `ImplicitTPM`.
PR #105 has done substantial work here — ~1918 additions with tests passing.

- `TPM` base class with `alphabet_size` per axis (trivial once the TPM is
  xarray-backed, with explicit coordinate labels).
- `Node.n_states` replacing hardcoded `2`.
- Per-unit state space in `Unit.alphabet_size`.
- `DistanceMetric` Protocol gets `state_space` parameter; binary stays default.
- Generalize `repertoire_shape` and all `2**n` calls.
- Golden fixtures remain exclusively binary; property tests get parameterized over
  state-space sizes.

- *Why here:* After P7 because it fundamentally changes what `CausalModel` wraps,
  and after P5 because non-binary exposes the same "implicit binary" assumption in
  multiple distance metrics. Doing it earlier would force P7 to be done twice.
- *Files:* `pyphi/tpm.py`, `pyphi/node.py`, `pyphi/network.py`, `pyphi/repertoire.py`,
  `pyphi/metrics/distribution.py`, `pyphi/distribution.py`, and every file flagged
  with `TODO extend to nonbinary nodes`.
- *Risk:* High — adds a dimension to the math. Mitigated because the binary golden
  fixtures remain valid oracles (nothing binary should change) and property tests
  can be parameterized over alphabet size.
- *Leverage:* Unlocks Gómez-style multivalued research; retires ~12 TODOs.

**P13. Zaeemzadeh upper bounds for pruning**

Implement the Zaeemzadeh & Tononi 2024 upper bounds on Φ. Use them in `find_mip` to
prune partitions that cannot achieve the best-so-far φ. Use them in `all_complexes`
to skip subsystems whose upper bound falls below a threshold. Expose as
`compute_upper_bound(candidate_system) -> float` in the `formalism.iit4.bounds`
module.

Ship in **shadow mode** for one release: compute both pruned and unpruned results,
assert equality in CI. Only switch the default after shadow mode passes 1000+
fixture runs.

- *Why here:* Depends on typed partitions (P6), typed distinctions (P8), working
  parallel backend (P11), and especially on Hypothesis testing (P2) — a pruning
  bug silently produces wrong φ values, and Hypothesis is the only cost-effective
  way to verify that pruning is provably conservative.
- *Files:* new `pyphi/formalism/iit4/bounds.py`,
  `pyphi/formalism/iit4/sia.py` (integration), `pyphi/compute/network.py`
  (complex search integration).
- *Risk:* High if pruning is wrong. Mitigated by shadow mode.
- *Leverage:* Very high for users — order-of-magnitude speedup enabling research
  that is currently intractable.

### Phase G — Downstream cleanup and future extensibility

**P14. `actual.py` resurrection** — ✅ **Landed 2026-05-09 (`71d22611`)**

Resurrected `pyphi.actual` against the frozen `System` value type. The data
layer is unified with IIT via `SystemPublicInterface`: a new frozen
`TransitionSystem` dataclass (parametric in `Direction`) satisfies the protocol,
and `Transition` becomes a frozen wrapper holding two `TransitionSystem`
instances (one per direction). The dispatch layer stays separate — actual
causation is a parallel analysis mode with its own free functions
(`pyphi.actual.sia`, `pyphi.actual.account`, etc.); IIT-formalism dispatchers
raise `NotImplementedError` when called on a `TransitionSystem` (category
errors).

The 826 lines of previously-skipped `test/test_actual.py` are back online,
plus paper-fixture acceptance tests against the worked-example α values from
Albantakis et al. 2019 ("What caused what?", `papers/2019__albantakis-et-al__what-caused-what.pdf`).

The accompanying config audit restructured `config.formalism` into nested
`IITConfig` and `ActualCausationConfig` dataclasses, applied a uniform
rename map (`*_distance` → `*_measure`, `partition_type` →
`mechanism_partition_scheme`, etc.), and added AC-specific knobs
(`mechanism_partition_scheme`, `partitioned_repertoire_scheme`,
`background_strategy`, `alpha_aggregation`). The orphaned concept-style
cuts machinery (`ConceptStyleSystem`, `concept_cuts`, `directional_sia`,
`SystemIrreducibilityAnalysisConceptStyle`, `sia_concept_style`,
`config.system_cuts`) was removed.

Macro work is split off into a separate paper-faithful project tracking
Marshall et al. 2024 (intrinsic units); see "Macro framework — Marshall
2024 intrinsic units" below. The disabled `MacroSystem` class and
`test/test_macro_system.py` (593 lines) remain disabled until that
project lands.

- *Files touched:* `pyphi/actual.py`, `pyphi/models/actual_causation.py`,
  `pyphi/conf/formalism.py`, `pyphi/conf/_global.py`,
  `test/test_actual.py`, new `test/test_actual_paper_fixtures.py`.

**Macro framework — Marshall 2024 intrinsic units**

Paper-faithful macro framework replacing legacy `pyphi/macro.py` outright.
The 2024 formalism organizes macro structure as hierarchical meso
constituents, with sliding-window state mappings $g_J$ over sequences of
$\tau$ micro updates, explicit background apportionment $W^J$, and
intrinsic-unit search via $\varphi_s$ optimization.

> Marshall W, Findlay G, Albantakis L, Tononi G (2024). *Intrinsic Units:
> Identifying a system's causal grain.* (See
> `papers/2024__marshall-et-al__intrinsic-units.pdf`.)

- *Status:* Deferred. The disabled `MacroSystem` class and the 593-line
  `test/test_macro_system.py` remain dark until this lands.
- *Why a fresh project:* The 2024 paper specifies the formalism in terms
  of objects (meso constituents, $g_J$, $W^J$, $\varphi_s$) that don't
  map onto the legacy `MacroSubsystem` design. A faithful port is closer
  to a rewrite than a resurrection.
- *Files:* `pyphi/macro.py` (legacy, ~1094 lines, to be replaced),
  `test/test_macro_system.py` (593 lines, to be unskipped against the
  new design).

**P14b. Fold matching/perception extension into PyPhi**

Integrate the formalism from Mayner, Juel, & Tononi 2024 ("Intrinsic meaning,
perception, and matching") into PyPhi. Currently implemented as research code in
`~/projects/matching/matching/` (~1500 LOC across 9 modules). This extension adds:

- **Triggered TPMs**: How a system's state evolves when a stimulus x is presented
  at its sensory interface ∂S for τ timesteps (conditioning T_U on ∂S=x, then
  evolving for τ steps). Extends the `CausalModel` with environment-system
  partitioning (U = S ∪ E).
- **Triggering coefficient** t(x,m) ∈ [0,1]: normalized causal pointwise mutual
  information measuring how much a stimulus causally determined a mechanism's state
  (Eq. 7 of the paper).
- **Perception value** p(x, d(m)) = t(x,m) × φ_d(m): how much of a distinction's
  causal power was triggered by a stimulus.
- **Perceptual structure**: the portion of the Φ-structure triggered by a stimulus,
  weighted by perception values. A structured interpretation of the stimulus.
- **Perceptual richness** P(x,y): sum of perception values — how much intrinsic
  meaning is triggered.
- **Perceptual differentiation** D: richness and diversity of Φ-structures triggered
  by a stimulus sequence.
- **Matching** M: maximum expected perceptual differentiation above chance — how
  well a system's intrinsic meanings resonate with its environment.

**Architecture:** The matching extension is a *consumer* of core IIT outputs
(`PhiStructure`, `Distinction`, `Relation`, `PhiFold`), not a modifier. It follows
the same pattern as actual causation (P14) and approximation methods (P16). It
should live in `pyphi/formalism/perception/` (or `pyphi/extensions/perception/`)
with modules:

```
formalism/perception/
  __init__.py              # MatchingAnalysis, PerceptualAnalysis facade
  triggering.py            # TriggeringCoefficient computation (Eq. 5-7)
  perception.py            # PerceptualDistinction, PerceptualStructure, richness
  differentiation.py       # Perceptual differentiation (Eq. 15-20)
  matching.py              # Matching computation (Eq. 21-23)
  triggered_tpm.py         # Triggered TPM construction (conditioned evolution)
  dynamics.py              # Environment-system dynamics (Ising, percolation)
```

**Key design decisions for integration:**
- Current matching code subclasses `Concept` as `PerceptualDistinction` and
  `CauseEffectStructure` as `PerceptualDistinctions`. After P8, these should wrap
  the new `Distinction` type instead. The subclassing pattern is clean and should
  be preserved — `PerceptualDistinction` adds a `perception` cached property, not
  different computation.
- `PhiFold` (from P8) is the natural unit for perception computation — perception
  is computed per-distinction Φ-fold then aggregated. The current matching code
  has its own `phi_fold.py`; after P8 promotes Φ-folds to core, the matching code
  should import from core.
- Triggered TPMs are currently pandas DataFrames, not `ExplicitTPM`. This is
  acceptable — they represent environment→system conditioning, which is a different
  object from the core TPM. However, triggered TPM construction should use
  `CausalModel` methods for the conditioning step.
- The `dynamics.py` module (Ising model, stationary distributions) overlaps with
  `pyphi/network_generator/ising.py`. Consolidate during integration.
- Matching code uses its own `utils/parallel.py` (joblib `doublestarmap`). After P11,
  this should use PyPhi's `Scheduler` Protocol.

**PyPhi API surface required (verified):**
- `PhiStructure.distinctions`, `.relations`, `.big_phi`
- `Distinction.mechanism`, `.cause`, `.effect`, `.phi` (or `Concept` equivalents)
- `Relation.phi`
- `CauseEffectStructure` (container)
- `ConcreteRelations`, `AnalyticalRelations`
- `NodeLabels.coerce_to_indices/labels()`
- `ExplicitTPM.condition_tpm()`, `.marginalize_out()`
- `PhiFold` (from P8)

All of these are stable after P8. No core API changes needed.

- *Why here:* Depends on stable `PhiStructure`/`Distinction`/`Relation`/`PhiFold`
  types (P8) and stable `CausalModel` (P7). Could theoretically start as early as
  P8 completion, but placing after P14 keeps the critical path focused on core
  stability.
- *Files:* `~/projects/matching/matching/` (source to adapt), new
  `pyphi/formalism/perception/` package.
- *Risk:* Medium — research code needs significant redesign for production quality.
  The mathematical formalism is well-defined (published paper); the engineering work
  is about adapting ~1500 LOC to the new type system. The triggering coefficient
  computation itself is straightforward (PMI + normalization).
- *Leverage:* High for research impact — enables PyPhi to be the reference library
  for IIT's account of perception, not just consciousness.

**P15. `jsonify` retirement + test reorg + docs / Jupyter / pandas / ASV-in-CI**

The victory-lap bundle. Must be last because everything preceding it changes the
public API surface.

- Replace `pyphi/jsonify.py` with `msgspec`-based serialization for types that need
  it (SIA, PhiStructure, golden fixtures); delete the custom
  `CLASS_KEY`/`VERSION_KEY`/`ID_KEY` registry; provide `to_dict()` via `ToDictMixin`
  for types that don't need round-trip. Migration tool to rewrite old JSON.
- Mirror `pyphi/` structure in `test/`. Split mixed `test_big_phi.py` along
  formalism lines. Kill `test_big_phi_robust.py` by merging into main test files
  with `robust` pytest marker.
- Sphinx site rebuilt against new architecture. New "Architecture" guide explaining
  the `Substrate → System → PhiFormalism → PhiStructure` layering, with
  direct citations to Albantakis et al. 2023 equations on every public method.
- **Systematic docstring cleanup pass.** During the 2.0 refactor, many
  docstrings accumulated migration-context phrasing — "roughly today's
  :class:`pyphi.network.Network` minus the TPM", "The replacement for
  :class:`pyphi.subsystem.Subsystem`. Immutable. Hashable.", "P7 stages: Task
  3.1: skeleton...", "Delegates to the legacy ... future cleanup can relocate
  the implementation if desired", etc. These read fine during the refactor
  but become noise as the project evolves: references decay, planning
  context becomes meaningless, and the docstring fails its primary job.
  Sweep every `pyphi/**/*.py` file (and notebook) and rewrite each
  docstring as if a fresh contributor encountered the file with no
  migration history — describe what the thing IS, not what it WAS,
  REPLACES, or HOW IT WAS BUILT. Project-stage markers (P-numbers,
  "Phase A", task numbers) MUST NOT appear; cross-references to
  legacy/replaced classes are removed unless the docstring is
  explicitly a deprecation notice. This is highest-impact for
  public-API classes (anything in `pyphi.__init__.py` or returned by
  a public function) because they end up in the rendered docs site.
- `__repr__` and `_repr_html_` on all frozen-dataclass results. Since they're
  dataclasses this is ~20 lines each.
- Extend `ToPandasMixin` to `SystemIrreducibilityAnalysis` and `PhiStructure`.
- ASV already configured; wire into CI on a nightly schedule with regression alerts.

- *Files:* `pyphi/jsonify.py`, new `pyphi/models/_serialize.py`, all of `test/`,
  `docs/`, `benchmarks/`, `.github/workflows/`.
- Update `pyphi/examples.py` (1545 lines) — constructs Subsystems throughout;
  must track the new `CandidateSystem` API from P7.
- Update `pyphi/dynamics.py` (117 lines) and `pyphi/timescale.py` (54 lines) —
  import from core pyphi; need updating when TPM types change.
- Update `pyphi/visualize/` (2545+ lines including `phi_structure/`) — tightly
  coupled to current model types (`PhiStructure`, `Distinction`, `SIA`); must
  track model changes from P8.
- Update `pyphi/connectivity.py` and `pyphi/graphs.py` — verify compatibility.
- Also triage and close/merge minor open PRs: #114 (strong connectivity shortcircuit),
  #116 (pandas circular import — likely resolved by file moves), #130 (visualization
  limit), #134 (Jupyter fix), #117 (benchmarking notebook).

### Phase H — Tractable approximations (future, unlocked by the architecture)

**P16. Approximation framework (φ\*, φ_G, and beyond)**

With the `PhiFormalism` Protocol established (P4), the `formalism/approx/` directory
is ready for approximation methods. Each approximation implements `PhiFormalism` with
`exact=False` and provides an `error_bound()`. Initial candidates:

- **φ\* (phi-star)**: Polynomial-time geometric integrated information. Uses KL
  divergence instead of full MIP search. Can be implemented as a formalism that
  replaces the exhaustive partition evaluator with a greedy or spectral method.
- **φ_G**: Graph-theoretic approximation based on effective information.
- **Heuristic CES approximations**: Avoid exhaustive distinction enumeration by
  sampling or bounding.
- **Zaeemzadeh-style bounded-exact**: Use the P13 upper bounds to provide certified
  approximations with provable error margins — not heuristic, but giving up exhaustive
  enumeration in exchange for guaranteed bounds.

The architecture built by P1–P15 supports this naturally: approximation methods
produce the same `SIA`/`PhiStructure` result types (with metadata flagging them as
approximate), can use the same `CandidateSystem` inputs, the same `Repertoire` types,
and the same parallelization infrastructure. The `PartitionScheme` Protocol already
accommodates non-exhaustive search strategies. Users select an approximation via
`pyphi.approx.phi_star(subsystem)` alongside `pyphi.iit4.phi_structure(subsystem)`.

**P17. Cross-formalism performance characterization + targeted optimization**

*Motivation.* The 2.0 work delivered large IIT 4.0 speedups: a cross-temporal
benchmark against commit `b3aaa3e5` (last pre-2.0 commit, parent of P0) shows
4.0 went from 2.5× faster on 3-node networks to 19× faster on `macro` (4n)
and 43× faster on `rule154` (5n) — the speedup widens with network size,
which points at an algorithmic change in the formalism split rather than
constant-factor overhead. Larissa Albantakis's original puzzle — "why isn't
4.0 faster than 3.0?" — is now empirically resolved on the post-refactor
surface, but the *mechanism* behind the gain hasn't been characterized, the
2026 cap variant still returns φ=0 on every standard `pyphi.examples`
network we ran, and the benchmark only covers up to 5 nodes because IIT 3.0
on `rule154` already costs ~9 minutes per trial pre-refactor.

*Scope.*

- **Extended network coverage.** Push the benchmark beyond 5 nodes. Likely
  needs synthesized networks rather than relying on the `pyphi.examples`
  catalog, plus a way to time-bound IIT 3.0 runs (or skip them past a
  size threshold). Identify the network size at which `iit4_sia_2023`
  hits the limits of being interactive (~10s/trial) and at which it
  becomes batch-only (~minutes/trial).
- **2026-cap exercising networks.** All standard 3–5-node example
  networks make `iit4_sia_2026` return φ=0 (cap collapses + short-circuit).
  We can't honestly cost the 2026 variant without a network where it
  produces a non-zero result. Either synthesize one, or pull from the
  paper's worked examples.
- **Mechanism deep-dive.** Take a representative network (likely `macro`
  or `rule154`) and walk through the pre/post profiles function by
  function to identify which 2.0 changes actually produced the 4.0
  speedup. Candidate hypotheses to test: (a) the `formalism.iit4.sia`
  loop no longer recomputes the unpartitioned CES per cut the way
  `compute._sia` did pre-refactor; (b) the layered config split (P10)
  eliminated per-call `__getattr__` overhead in tight loops; (c) the
  P11 parallelization redesign matters at this scale. Write up findings
  as an internal performance-architecture note.
- **Residual shipping-hack sweep.** The construction of the cross-temporal
  benchmark surfaced two real issues (`PARALLEL=False` not actually
  disabling subprocess evaluation in pre-refactor `_sia_map_reduce`;
  IIT 3.0 + `GENERALIZED_INTRINSIC_DIFFERENCE` raising AttributeError).
  The first was post-refactor-relevant; the second was pre-only.
  Spend a 1–2-day sweep looking for analogous issues in the current
  2.0 hot paths — places where a config flag's documented behavior
  doesn't match its actual behavior, or where a config combination
  raises rather than producing a clean result.
- **Optional: top-1 targeted optimization.** If the mechanism deep-dive
  identifies a clear remaining bottleneck (e.g. a redundant repertoire
  recomputation, a parallelism gate that doesn't fire when it should),
  land a targeted fix and re-benchmark. Bound to ~1 week; bigger
  rewrites get their own roadmap item.

*Why here.* Post-surface-freeze (P15 has shipped, the public API is
stable, ASV-in-CI from P11.8 Tier 2 exists). The investigation needs the
clean post-2.0 surface as the reference point, and the benchmark
infrastructure from P11.8 Tier 2 to avoid re-inventing harness work.
Pairs naturally with P16 (approximation framework) because knowing where
the exact-computation cost lives informs which approximations are worth
building — e.g. if 80% of the cost is in mechanism-MIP search, then
`φ*`-style mechanism-level approximations have higher leverage than
`φ_G`-style system-level ones.

*Files.* `benchmarks/iit_3_vs_4/` (extend the existing cross-temporal
harness with larger-network support and synthesized fixtures); new
`benchmarks/iit_3_vs_4/findings.md` (or equivalent) for the mechanism
write-up; possibly small fixes in `pyphi/formalism/` if the targeted
optimization lands.

*Risk.* Low for the measurement portions; medium for any landed
optimization (perf fixes in hot paths have a history of subtle
correctness regressions — see the P11.8 motivating story of the
60–300× slowdown that survived golden tests).

*Leverage.* Documents a publication-relevant claim (the 2.0 work
delivered measurable speedup at scale), informs the P16 approximation
design, and closes the loop on Larissa's question with reproducible
numbers. Existing benchmark harness in `benchmarks/iit_3_vs_4/` already
runs against both pre- (`b3aaa3e5` worktree) and post-refactor
checkouts; the harness needs extension rather than rewrite.

---

## Path-Dependency Graph

![Path-dependency graph](ROADMAP_path_dependency.svg)

<details>
<summary>ASCII version</summary>

```
     ┌────────────────┐
     │ P0 Python 3.13 │
     │   dep verify   │
     └───────┬────────┘
             │
      ┌──────▼───────┐
      │ P1 Golden    │
      │    harness   │
      └──────┬───────┘
             │
      ┌──────┴──────────────┐
      │                     │
 ┌────▼─────┐        ┌─────▼─────┐
 │P2 Hypoth.│        │P3 Protocols│  ← P2 and P3 are independent; can overlap
 │invariants│        │+ CES fix   │
 └────┬─────┘        └─────┬─────┘
      └──────────┬──────────┘
                 │
          ┌──────▼────────────┐
          │ P4 Formalism split│ ◀─── pivot
          └──────┬────────────┘
                 │
          ┌──────▼───────┐
          │ P5 Metric API│
          └──────┬───────┘
                 │
          ┌──────▼───────┐
          │ P6 Partition │
          │   algebra    │
          └──────┬───────┘
                 │
          ┌──────▼─────────────────┐
          │ P6a Lazy graphillion + │ (also unlocks LocalThreadScheduler in P11)
          │   globals audit        │
          └──────┬─────────────────┘
                 │
          ┌──────▼─────────────────┐
          │ P6b ZDD migration      │ (graphillion → OxiDD)
          │   to OxiDD             │
          └──────┬─────────────────┘
                 │
          ┌──────▼───────┐
          │ P7 Subsystem │ ◀─── architectural pivot, big-bang
          │   rewrite    │
          └──────┬───────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
   ┌───▼──┐  ┌───▼──┐  ┌───▼───┐
   │ P8   │  │ P9   │  │ P10   │
   │Models│  │Cache │  │Config │
   └───┬──┘  └──────┘  └───┬───┘
       │                   │
       │            ┌──────▼────────────┐
       │            │P11 Parallel       │
       │            │ (+ ThreadScheduler│
       │            │  if no-GIL)       │
       │            └──────┬────────────┘
       │                   │
       ▼                   ▼
   ┌──────────────────────────┐
   │ P12 Non-binary units      │
   └──────┬────────────────────┘
          │
   ┌──────▼──────────┐
   │ P13 Zaeemzadeh   │
   │  upper bounds    │
   └──────┬──────────┘
          │
   ┌──────▼──────────┐
   │ P14 Macro +      │
   │    actual        │
   └──────┬──────────┘
          │
   ┌──────▼──────────┐
   │ P14b Matching/   │
   │   perception     │
   └──────┬──────────┘
          │
   ┌──────▼──────────┐
   │ P15 jsonify +    │
   │    docs + ASV    │
   └─────────────────┘
```

</details>

**Critical chains:**

- **Correctness safety chain:** P1 → P2 → (everything numerical)
- **Formalism chain:** P3 → P4 → P5 → P6 → P6a → P6b → P7 (core architectural backbone)
- **No-GIL enablement chain:** P6a (lazy import + globals audit) → P6b (graphillion
  → OxiDD swap) → P11 (`LocalThreadScheduler`). All three required for the
  free-threaded benefit; without any one, the thread scheduler degrades to GIL mode.
- **Parallel track (independent of the formalism chain once P10 lands):** P11.
- **Strictly after backbone:** P8, P9, P10, P12.
- **Aggressive features after stability:** P13.
- **Deferred / victory lap:** P14, P15.

---

## Sequencing Rationale

Four interlocking principles drive this order.

**(a) Safety net before truth-preserving refactors.** P1 and P2 are project zero.
Every later project is either safe because of them, or unsafe without them. The
specific engineering call here is that the golden fixture format must be independent
of `pyphi/jsonify.py` — otherwise P15 would invalidate the fixtures P1 depends on.
Store as `.npz` or minimal plain JSON of numbers and tuples, never pickled objects.

**(b) Strategy before algorithm.** P4 (the formalism split) unlocks every subsequent
refactor's shape. Without it, each of P5–P10 would contain its own mini-version of
this project, and the result would be far messier. The cost of doing it at P4 is
much less than the cost of doing it at P7.

**(c) Types before operations.** P3, P5, P6 establish the types that every
formalism-level operation consumes. P7 (subsystem rewrite) is the first
formalism-level operation. You cannot sensibly refactor `find_mip` without first
knowing what a `Repertoire`, `DistanceMetric`, and `Partition` are — and the current
codebase doesn't really commit to an answer on any of those. This is why splitting
`models/mechanism.py` is P8 not P1: its complexity is a symptom, not a cause.

**(d) Correctness before performance.** P1–P9 are correctness-motivated.
P11 (parallelization) is the first performance-motivated project, deliberately
placed after the types settle because parallelization introduces non-determinism
that is much harder to debug on top of unstable types. P13 (Zaeemzadeh pruning) is
even later because its correctness is delicate and requires both typed partitions
and robust property testing.

**On path dependencies nobody at first glance sees:**
P6 (partitions) must precede P7 (subsystem rewrite) because the rewrite needs to
talk to *one* partition abstraction. P9 (cache) and P10 (config) must follow P7
because cache boundaries and config scopes only become clear once the layering
exists. P11 (parallelization) is mostly independent but needs P10 because workers
today read `config.*` globals that silently pickle; post-P10 they receive explicit
config snapshots. P12 (non-binary) must follow P7 because it changes what
`CausalModel` wraps — doing it earlier would force P7 to be done twice.

---

## Test Infrastructure Strategy

Four orthogonal layers of coverage, established in order:

- **Layer A — Golden numerical oracle** (P1). Raw floats, raw partition tuples, raw
  arrays, stored in `.npz`. Format owned by the test harness, **not** by
  `jsonify`. Survives P15's jsonify retirement. Initial fixtures generated from
  a pinned commit manually validated against published IIT results.

- **Layer B — Property invariants** (P2). Hypothesis-driven on random small networks
  (≤4 nodes). Encodes every invariant from the 4.0 paper.
  Runs every PR with fixed seed; nightly with `hypothesis --generate`.

- **Layer C — Behavioral regression** (existing + P15 reorg). Existing test files
  rewired to the new API in P15.

- **Layer D — Performance regression** (P15). ASV runs nightly. Not correctness,
  but tells you when a refactor silently broke memoization.

**CI requirement:** Layers A, B, C run on every PR. Layer D runs nightly.

**Current test suite health (verified):** 950 collected tests, 41s runtime, 855 passed,
93 skipped, 2 xfailed. This is a solid foundation — healthier than CLAUDE.md's
estimate of ~460 test functions.

**Acceptance criterion for "Phase A complete, safe to begin P4":** a deliberate
sign-flip mutation to `metrics/distribution.py` must fail at least three Layer A
fixtures AND at least one Layer B property. Run mutation-testing on a sample of
`subsystem.py` changes and confirm they fail.

---

## Risk Mitigation

**Golden fixture bit-rot.** If fixtures are ever regenerated from a buggy commit,
you've canonicalized a bug. Mitigation: pin the commit hash used for initial fixture
generation in `test/data/golden/README.md`, validate against published results by
hand before committing, and require all future fixture regeneration to cite the
mathematical property it is re-verifying.

**Big-bang subsystem rewrite (P7) produces wrong numbers.** Mitigation: worktree
development; Layers A+B+C all green before merge; optional `PYPHI_NEW_CORE=1`
env-var gate for one release cycle.

**Latent partition-family bug in P6.** The codebase may currently use generic
mechanism bipartitions where distinctions require disintegrating partitions
(Eq. 29). If true, fixing this changes numerical results. Mitigation: triage
before committing — validate the correct answer against Albantakis et al. 2023
Fig. 2 (distinction computation on the 3-node example). Update golden fixtures only
after confirming the fix matches the published values.

**Non-binary refactor (P12) breaks invariants that were always implicitly binary.**
Mitigation: Layer B is binary-parameterized until P12 begins, then parameterized
over state-space size. Layer A remains binary; nothing binary should change.

**Partial progress.** Each numbered project is a landable unit that leaves the
codebase at least as consistent as before. If work stops after P5, the metric API
is clean even though `subsystem.py` isn't yet split. No project requires the next
to have any value.

---

## Critical Files (highest blast radius, touched by multiple projects)

- `/Users/will/projects/pyphi/pyphi/subsystem.py` — P4, P5, P6, P7, P9
- `/Users/will/projects/pyphi/pyphi/new_big_phi/__init__.py` — P4, P7, P11
- `/Users/will/projects/pyphi/pyphi/metrics/distribution.py` — P3, P5, P12
- `/Users/will/projects/pyphi/pyphi/partition.py` — P3, P6, P13
- `/Users/will/projects/pyphi/pyphi/models/mechanism.py` — P7, P8
- `/Users/will/projects/pyphi/pyphi/models/subsystem.py` — P7, P8
- `/Users/will/projects/pyphi/pyphi/tpm.py` — P7, P12
- `/Users/will/projects/pyphi/pyphi/repertoire.py` — P5, P7, P12
- `/Users/will/projects/pyphi/pyphi/conf.py` — P4, P10
- `/Users/will/projects/pyphi/pyphi/parallel/` — P11
- `/Users/will/projects/pyphi/pyphi/cache/` — P9
- `/Users/will/projects/pyphi/pyphi/relations.py` — P6, P8 (already wired to `phi_structure`)

## Existing Utilities to Reuse

- `pyphi/data_structures/pyphi_float.py` — `PyPhiFloat`: keep, extend via P5.
- `pyphi/data_structures/frozen_map.py` — `FrozenMap`: becomes load-bearing in P9.
- `pyphi/direction.py` — `Direction` enum: already cleanly integrated; no changes.
- `pyphi/registry.py` — `Registry[T]`: keep, becomes typed in P3.
- `pyphi/parallel/tree.py` — `TreeSpec`, `TreeConstraints`: reused by P11.
- `pyphi/parallel/__init__.py` — `MapReduce`: becomes Scheduler facade in P11.
- `pyphi/relations.py` — `ConcreteRelations`, `AnalyticalRelations`: move to
  `formalism/iit4/relation.py` in P4/P8; keep the `graphillion`-backed architecture.
- `pyphi/combinatorics.py` — analytical summation formulas: refactored into
  `combinatorics/` package in P6. The `graphillion.setset` integration is load-bearing.
- `pyphi/resolve_ties.py` — tie-resolution logic: reused in new formalism layer.
- `test/example_networks.py` — used by P1 golden fixtures.
- `test/hypothesis_utils.py` — expanded in P2.

## Open PRs to Absorb

- **PR #138 (`feature/substrate_modeler`)** — Reviewed as input to P7. Stateless
  substrate design, `unit.py`, `substrate.py` may be directly usable.
- **PR #105 (`feature/tpm-class`)** — Reviewed as input to P7 and P12. `ImplicitTPM`,
  `state_space.py`, non-binary support. 100 commits, tests passing.
- **PR #135, #133** — Pre-commit autoupdates. Merge into develop.
- **PR #137, #136** — Dependabot bumps. Triage for security relevance.
- **PR #114, #116, #130, #134, #117** — Minor fixes/features. Triage during P15.
- **`~/projects/matching/`** — External research code implementing Mayner, Juel, &
  Tononi 2024 (intrinsic meaning, perception, and matching). ~1500 LOC. Folded into
  PyPhi as P14b. Uses subclassing pattern over PhiStructure/Concept/Relation.

---

## Migration Strategy for Existing Users

PyPhi 2.0 is a **new major release with no backwards compatibility guarantee**. However,
to ease transition:

- **Deprecation warnings in 1.x final release:** Before 2.0, ship a 1.x release that
  adds `FutureWarning` on all APIs that will change (e.g., `config.IIT_VERSION`,
  `Subsystem.concept()`, `DistanceResult.__array__`).
- **Migration guide:** Ship `docs/migration-2.0.md` documenting every API change.
  Auto-generate from the `SubsystemInterface` Protocol (P3) by diffing old vs new
  public surfaces.
- **Updated demo notebook:** Rewrite `docs/examples/IIT_4.0_demo.ipynb` as the
  canonical 2.0 tutorial, showing `pyphi.iit4.phi_structure(subsystem)` rather than
  the old `new_big_phi` imports.
- **`pyphi.compat` module (optional, if demand exists):** Thin shims mapping old API
  (`compute.big_phi()`, `new_big_phi.phi_structure()`) to new entry points. Emits
  `DeprecationWarning`. Can be removed in 2.1.

---

## Verification Plan

- **Each project independently verified:**
  - Run `uv run pytest` — all existing tests pass.
  - Run Layer A (golden fixtures) — all numerical results match to 1e-12.
  - Run Layer B (Hypothesis) — no property violations on 1000 runs.
  - Run `uv run pyright pyphi` — no new type errors.
  - Run `make benchmark` (after P15) — no performance regression >10%.

- **End-to-end verification after P7:**
  - Reproduce Figure 1 from Albantakis et al. 2023 (identifying complexes).
  - Reproduce Figure 2 from Albantakis et al. 2023 (computing distinctions).
  - Reproduce Figure 4 from Albantakis et al. 2023 (computing relations).
  - Reproduce Figure 7 from Albantakis et al. 2023 (state dependence; verify
    φ values for `ABcdE`/`ABcde`/`ABcd` states).
  - Verify IIT 3.0 regression tests in `test_big_phi.py` still pass unchanged.

- **End-to-end verification after P13:**
  - Shadow-mode pruning comparison against unpruned results on all Layer A fixtures
    — exact equality required.
  - Performance: full SIA on a 6-node specialized lattice (Fig. 6D) completes
    in < X% of current wall time, with the concrete threshold set based on
    measurement during P11.

- **Mathematician's acceptance test:** open the post-P7 code and point at every
  Greek letter in Albantakis et al. 2023 to a corresponding class, method, or
  registered function. If that test passes, the refactor has succeeded.


---

## Informal notes — pre-release housekeeping

- clean up / reorganize test suite

- **Condense changelog fragments before 2.0 ships.** Several
  fragments in ``changelog.d/`` describe transitions between
  intermediate 2.0-development states that no released version ever
  exposed (e.g., the cruelest-cut → paper-faithful switch in
  ``paper-faithful-state-tie.change.md`` describes replacing a
  PyPhi-specific convention that itself only existed during the 2.0
  push). A user reading the 2.0 changelog will not have seen those
  intermediate states. Before release, sweep the fragment set and
  rewrite as if 2.0 is the first time the user encounters this
  behaviour — squash transition-describing language into "in 2.0,
  the SIA selects ...". Probably an evening's work just before the
  towncrier-build step of the release.
