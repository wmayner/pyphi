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

**P14. `macro.py` + `actual.py` resurrection**

Port both modules to the new core kernel. Macro subsystems become `CausalModel`
transformations (coarse-graining as a functor from one model to another). Actual
causation becomes a third `PhiFormalism` implementation — the underlying equations
are different but fit the abstraction cleanly.

- *Why last among code projects:* They're flagged "out of date" in `PROJECTS.md`;
  they need to follow every preceding decision; they have the lowest blast radius so
  deferring them is safe; doing them before the core refactor would mean refactoring
  them twice.
- *Files:* `pyphi/macro.py` (1094 lines), `pyphi/actual.py` (953 lines),
  `pyphi/models/actual_causation.py`, `test/test_macro*.py`, `test/test_actual.py`.

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
