# P7 — Subsystem Layered Rewrite (Design)

**Project:** PyPhi 2.0 strategic refactor, Phase C — Kernel rewrite
**Date:** 2026-05-06
**Status:** Design approved by user, pending implementation plan
**Predecessors:** P0–P6, P6a (all complete)
**Successor:** P7b (MacroSubsystem port)

---

## Summary

Replace the 1354-line `Subsystem` god-object with a layered architecture
under `pyphi/core/`. Frozen value types flow through stateless algorithm
modules; caches are explicit decorators at known memoization boundaries.
Numerical behavior is preserved; the Phase A safety net (golden fixtures
+ Hypothesis invariants + surface drift + sign-flip canary) pins
correctness throughout.

This is the largest single project in the 2.0 roadmap. It is executed
big-bang in a worktree, lands as one PR, deletes `pyphi/subsystem.py`,
and renames the public type from `Subsystem` to `CandidateSystem` to
match IIT 4.0 paper terminology.

---

## Decisions (with rationale)

| # | Decision | Rationale |
|---|---|---|
| Q1 | Use PR #138 and PR #105 as **design references only**; cherry-pick concepts into a fresh P7 design without merging either | Both PRs predate the formalism split (P4) and partition consolidation (P6); their abstractions don't match the post-P6 contract. Reading them informs the design; merging them imports stale layering |
| Q2 | P7 ships the **TPM Protocol + numpy-backed `ExplicitTPM`**. ImplicitTPM and xarray are deferred to P12 | ROADMAP §946-948 explicitly places non-binary in P12, after P7, with the warning *"doing it earlier would force P7 to be done twice."* xarray's per-operation overhead caveat (ROADMAP §203-207) is best resolved when alphabet labeling is load-bearing |
| Q3 | **MacroSubsystem deferred to P7b**, immediately after P7. P7 must paper-validate that the design admits a clean `CausalModel → transform → CandidateSystem` macro functor | P7's scope is reduced (kernel only). P7b implements the macro port with kernel context still warm. Per-release shipping (single 2.0 release) means "macro temporarily broken" doesn't bite users |
| Q4 | Cache lives **per-instance** on `CandidateSystem` (option (a)). No cross-cut content-keyed sharing in P7. Redis backend deferred to P9 | Investigation: current `Subsystem.apply_cut` already discards caches across cuts. A global content-keyed cache would have the same hit rate as per-instance for the current MIP search structure. Cross-cut sharing requires content-addressable repertoire keys, which is significant infrastructure beyond P7 |
| Q5 | Public type renamed from `Subsystem` to `CandidateSystem` | Matches IIT 4.0 paper terminology. ROADMAP §744 explicitly says *"This is what `Subsystem.__init__` should have been."* Aligns with no-shims preference. ~20 import sites and 29 test files updated mechanically |
| Q6 | `actual.py` gets a **minimal mechanical port** in P7; full architectural rewrite stays at P14 | Actual-causation is small-scope. Keeping it broken across many projects is a debug-time hazard. Mechanical port is cheap |
| Q7 | `Concept.subsystem` back-reference removed | Concept becomes a value type. Eliminates one class of cross-layer coupling. Serialization changes covered by golden fixtures |
| Q8 | No `PYPHI_NEW_CORE=1` env flag | Single-release shipping makes a transitional flag moot. Hard cutover when P7 lands |
| Q9 | Worktree-based, multiple internal commits, single PR (or squash-merge) to develop | ROADMAP §804-805. Internal commits for review hygiene and bisectability; externally-visible artifact is one PR |

**Approach:** Hybrid (option A from brainstorming). Frozen `@dataclass`
value types + module-of-functions repertoire algebra + existing P4
formalism dispatcher. No new orchestration classes — `PhiFormalism`
already plays the dispatcher role.

---

## Architecture

### Layered split

The `Subsystem` god-object decomposes into three concentric layers,
each with one job:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3 (existing, from P4): formalism/                         │
│   PhiFormalism.evaluate_mechanism(cs, ...) -> RIA               │
│   PhiFormalism.evaluate_system(cs)         -> SIA               │
│   PhiFormalism.build_phi_structure(cs)     -> PhiStructure      │
└─────────────────────────────────────────────────────────────────┘
                              ▲ takes CandidateSystem
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2 (new in P7): core/repertoire_algebra.py                 │
│   pure functions over CandidateSystem                           │
│   cause_repertoire(cs, mechanism, purview) -> Repertoire        │
│   effect_repertoire(cs, mechanism, purview) -> Repertoire       │
│   forward_repertoire(cs, direction, mechanism, purview)         │
│   partitioned_repertoire(cs, partition)                         │
│   ... (everything currently in subsystem.py:355-690)            │
│   @memoize decorator keys on (id(cs), mechanism, purview)       │
└─────────────────────────────────────────────────────────────────┘
                              ▲ takes CandidateSystem
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1 (new in P7): core/ value types                          │
│   CausalModel(substrate: Substrate, tpm: TPM)                   │
│     - immutable; no state; no computation                       │
│   CandidateSystem(causal_model, state, node_subset, cut)        │
│     - immutable; carries cheap derived properties only          │
│     - cut is a constructor arg, not a hidden mode               │
└─────────────────────────────────────────────────────────────────┘
                              ▲ wraps TPM via Protocol
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0 (new in P7): core/tpm/                                  │
│   TPM Protocol: effect_marginal(), cause_marginal(), condition()│
│   ExplicitTPM (numpy-backed; ports current ExplicitTPM)         │
│   marginalization.py (named operations against Eq. 3 / Eq. 4)   │
└─────────────────────────────────────────────────────────────────┘
```

**Direction of dependency:** strictly downward. Layer 3 depends on 2,
2 on 1, 1 on 0. Nothing points up. No back-references.

### What disappears

- `Subsystem` class (1354 lines)
- `MacroSubsystem` (handled by P7b)
- 4 cache instances on `__init__` (one decorator owns the cache)
- `Subsystem._backward_tpm()` implicit side effect (becomes named
  `causal_marginalization(direction)` operation)
- `Concept.subsystem` back-reference

### What's preserved

- All 33 names in `SubsystemPublicInterface` (now methods on
  `CandidateSystem` or functions in `core/repertoire_algebra.py`,
  depending on layer fit)
- All numerical behavior (golden fixtures pin everything)
- `pyphi.config.*` semantics (no config changes in P7)

### File-level inventory

```
NEW:
  pyphi/core/__init__.py
  pyphi/core/unit.py
  pyphi/core/substrate.py
  pyphi/core/causal_model.py
  pyphi/core/candidate_system.py     # ~300 lines: value type, derived props, public methods
  pyphi/core/repertoire_algebra.py   # ~600 lines: all repertoire computation
  pyphi/core/tpm/__init__.py
  pyphi/core/tpm/base.py             # TPM Protocol
  pyphi/core/tpm/explicit.py         # numpy-backed; ports pyphi/tpm.py:ExplicitTPM
  pyphi/core/tpm/marginalization.py  # cause/effect marginalization, backward TPM

DELETED:
  pyphi/subsystem.py                 # 1354 lines
  pyphi/repertoire.py                # 150 lines, folded into core/repertoire_algebra.py

MODIFIED:
  pyphi/__init__.py                  # export CandidateSystem, drop Subsystem
  pyphi/actual.py                    # mechanical Subsystem → CandidateSystem rename
  pyphi/examples.py                  # same
  pyphi/compute/subsystem.py         # update imports + types
  pyphi/compute/network.py           # same
  pyphi/formalism/iit3/__init__.py   # update Subsystem references
  pyphi/formalism/iit4/__init__.py   # same
  pyphi/models/mechanism.py          # remove Concept.subsystem back-reference
  pyphi/models/__init__.py           # same
  pyphi/macro.py                     # disabled in P7, restored in P7b
```

**Line-count delta (estimate):** −1354 (subsystem.py) + ~1400 (new
core/) ≈ flat. The win is locality and testability, not LOC reduction.

---

## Components

All dataclasses are `@dataclass(frozen=True, slots=True)` unless noted.
Type signatures are illustrative; final shapes get nailed down during
implementation.

### `core/tpm/base.py` — TPM Protocol

```python
@runtime_checkable
class TPM(Protocol):
    """A transition probability matrix.

    Implementations: ExplicitTPM (numpy-backed, P7), ImplicitTPM (P12).
    """
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def n_nodes(self) -> int: ...

    def condition(self, fixed: Mapping[int, int]) -> "TPM": ...
    def effect_marginal(self, mechanism, purview, state) -> Repertoire: ...
    def cause_marginal(self, mechanism, purview, state) -> Repertoire: ...
    def squeeze(self) -> "TPM": ...
```

`alphabet_size` introspection deliberately omitted from P7's Protocol —
added by P12 when non-binary lands. Binary is implicit.

### `core/tpm/explicit.py` — `ExplicitTPM`

Direct port of current `pyphi.tpm.ExplicitTPM`. Numpy-backed, no
semantic changes.

### `core/tpm/marginalization.py` — Causal marginalization

```python
def cause_tpm(tpm: TPM, state: State, node_indices: NodeIndices) -> TPM:
    """Backward TPM (Eq. 3 of Albantakis et al. 2023).

    Replaces the implicit `_backward_tpm()` side effect in
    `Subsystem.__init__`. Documented; named.
    """

def effect_tpm(tpm: TPM, external_state: State) -> TPM:
    """Forward TPM conditioned on external state (Eq. 4)."""
```

Free functions, not TPM methods — they're operations *on* TPMs that
happen at `CandidateSystem` construction time, not properties of a TPM
in isolation.

### `core/unit.py` — `Unit`

```python
@dataclass(frozen=True, slots=True)
class Unit:
    index: int
    label: str
    # alphabet_size: int = 2   # added in P12
```

Roughly today's `Node` minus the per-instance TPM caching.

### `core/substrate.py` — `Substrate`

```python
@dataclass(frozen=True, slots=True)
class Substrate:
    units: tuple[Unit, ...]
    connectivity_matrix: ConnectivityMatrix

    @cached_property
    def node_labels(self) -> NodeLabels: ...
    @cached_property
    def n_units(self) -> int: ...
```

Roughly today's `Network` minus the TPM (which moves to `CausalModel`).

### `core/causal_model.py` — `CausalModel`

```python
@dataclass(frozen=True, slots=True)
class CausalModel:
    """Substrate + TPM. The zeroth postulate of IIT 4.0.

    Zero computation. No caches. Operations on a CausalModel are free
    functions in core.tpm.marginalization or core.repertoire_algebra.
    """
    substrate: Substrate
    tpm: TPM

    @classmethod
    def from_network(cls, network: Network) -> "CausalModel":
        """Migration helper: build a CausalModel from a legacy Network."""
```

The `from_network` classmethod stays for the duration of P7+P7b+P8 to
ease the migration; deleted before 2.0 ships if all callers go direct.

### `core/candidate_system.py` — `CandidateSystem`

The replacement for `Subsystem`:

```python
@dataclass(frozen=True, slots=True, eq=False)
class CandidateSystem:
    """A candidate system: (CausalModel, state, node_subset, cut).

    What `Subsystem.__init__` should have been. Immutable. Hashable.
    Cut is a constructor arg, not a hidden mode.

    Carries cheap derived properties only; all repertoire computation
    is in core.repertoire_algebra.
    """
    causal_model: CausalModel
    state: State
    node_indices: NodeIndices
    cut: SystemPartition  # default = NullCut at construction site

    # ---- cached cheap derived properties ----
    @cached_property
    def cause_tpm(self) -> TPM: ...
    @cached_property
    def effect_tpm(self) -> TPM: ...
    @cached_property
    def proper_cause_tpm(self) -> TPM: ...
    @cached_property
    def proper_effect_tpm(self) -> TPM: ...
    @cached_property
    def cm(self) -> ConnectivityMatrix: ...      # cut applied
    @cached_property
    def proper_cm(self) -> ConnectivityMatrix: ...
    @cached_property
    def nodes(self) -> tuple[Node, ...]: ...
    @cached_property
    def proper_state(self) -> State: ...
    # ... (all the @property names from current Subsystem)

    # ---- methods that proxy to repertoire_algebra ----
    def cause_repertoire(self, m, p) -> Repertoire:
        return repertoire_algebra.cause_repertoire(self, m, p)
    def effect_repertoire(self, m, p) -> Repertoire:
        return repertoire_algebra.effect_repertoire(self, m, p)
    # ... etc, every entry in SubsystemPublicInterface that today
    # computes a repertoire

    # ---- methods that proxy to formalism ----
    def find_mip(self, **kw):
        return config.FORMALISM.evaluate_mechanism(self, **kw)
    def sia(self, **kw):
        return config.FORMALISM.evaluate_system(self, **kw)
    # ... etc

    # ---- contract surface kept for back-compat with SubsystemPublicInterface ----
    def apply_cut(self, cut) -> "CandidateSystem":
        return replace(self, cut=cut)
    def cache_info(self) -> dict[str, Any]: ...
    def clear_caches(self) -> None: ...

    def __eq__(self, other) -> bool: ...
    def __hash__(self) -> int: ...
```

The proxy methods are intentional — `SubsystemPublicInterface` is the
contract, and proxying preserves it. Internally everything routes to
`repertoire_algebra` / `formalism` so the layering is real.

### `core/repertoire_algebra.py` — Stateless functions

```python
@_memoize  # see Data Flow section for cache mechanics
def cause_repertoire(cs: CandidateSystem, m: Mechanism, p: Purview) -> Repertoire: ...

@_memoize
def effect_repertoire(cs: CandidateSystem, m: Mechanism, p: Purview) -> Repertoire: ...

@_memoize
def forward_repertoire(cs, direction, m, p) -> Repertoire: ...

@_memoize
def partitioned_repertoire(cs, partition) -> Repertoire: ...

# unconstrained variants (no cache — cheap and rare)
def unconstrained_cause_repertoire(cs, p) -> Repertoire: ...
def unconstrained_effect_repertoire(cs, p) -> Repertoire: ...

# expansion / projection helpers
def expand_repertoire(cs, direction, p, repertoire, ...) -> Repertoire: ...

# probability scalars
def forward_probability(cs, direction, m, p) -> float: ...
```

These are the bodies of `Subsystem._cause_repertoire`,
`_effect_repertoire`, etc. (subsystem.py:355-690), lifted out and given
an explicit `cs` parameter.

### Layer 3 (existing) — minimal changes

`PhiFormalism.evaluate_mechanism(cs, ...)` already takes a
Subsystem-shaped first arg from P4. P7 changes the type annotation to
`CandidateSystem` and updates the body to use `cs.cause_repertoire(...)`
(or equivalently `repertoire_algebra.cause_repertoire(cs, ...)`
directly). No semantic change.

---

## Data Flow

### Path 1: `phi` for a mechanism on a (cut) candidate system

```
User code:
  cm = CausalModel.from_network(network)
  cs = CandidateSystem(cm, state, node_indices)       # NullCut implicit
  phi = cs.phi(mechanism, purview)
                    │
                    ▼
CandidateSystem.phi(m, p)
  → config.FORMALISM.evaluate_mechanism(self, m, p).phi
                    │
                    ▼
PhiFormalism.evaluate_mechanism(cs, m, p)             # P4-existing
  → for each partition: evaluate_mechanism_partition(cs, ..., partition)
    → calls repertoire_algebra.cause_repertoire(cs, m, p)
    → calls repertoire_algebra.partitioned_repertoire(cs, partition)
    → metric(forward, partitioned, selectivity, state)
                    │
                    ▼
repertoire_algebra.cause_repertoire(cs, m, p)
  cache_key = (id(cs), m, p)
  hit?  → return cached Repertoire
  miss? → derive from cs.cause_tpm + cs.cm
        → memoize, return
```

### Path 2: SIA computation

```
User code:
  sia = cs.sia()
                    │
                    ▼
CandidateSystem.sia()
  → config.FORMALISM.evaluate_system(self)
                    │
                    ▼
PhiFormalism.evaluate_system(cs)                       # P4-existing
  → enumerate sia_partitions(cs.cut_indices, cs.cut_node_labels)
  → for each partition (parallel via MapReduce):
       cut_cs = cs.apply_cut(partition)                # → new CandidateSystem
       evaluate_partition(partition, cs, ...)
         → repertoire_algebra.forward_repertoire(cs,     ...)     # uncut
         → repertoire_algebra.forward_repertoire(cut_cs, ...)     # cut
         → metric → integration value
  → take min → SIA(phi, signed_phi, partition, ...)
```

### Cache mechanics

Cache lives in a module-private dict keyed on `(id(cs), *args)`. Because
`id` is reused after GC, the decorator also holds a `WeakValueDictionary`
mapping `id(cs)` → `cs`, with a finalizer that purges cache entries when
a `CandidateSystem` is collected:

```python
# core/repertoire_algebra.py (sketch)

_caches: dict[str, dict[tuple, Any]] = {}        # one dict per memoized fn name
_observers: WeakValueDictionary[int, CandidateSystem] = WeakValueDictionary()

def _memoize(fn):
    cache = _caches.setdefault(fn.__name__, {})
    @wraps(fn)
    def wrapper(cs: CandidateSystem, *args):
        key = (id(cs), args)
        if id(cs) not in _observers:
            _observers[id(cs)] = cs
            weakref.finalize(cs, _evict, id(cs))
        if key in cache:
            return cache[key]
        result = fn(cs, *args)
        cache[key] = result
        return result
    return wrapper

def _evict(cs_id: int) -> None:
    for fn_cache in _caches.values():
        for key in [k for k in fn_cache if k[0] == cs_id]:
            del fn_cache[key]
```

`CandidateSystem.cache_info()` and `CandidateSystem.clear_caches()` both
proxy to module-level helpers in `repertoire_algebra` that filter
`_caches` by `id(self)`. This preserves the public API.

**Memory profile:** identical to today (cache dies with instance), modulo
a few extra weakref/finalizer objects (negligible).

### `apply_cut` semantics

```python
# CandidateSystem
def apply_cut(self, cut: SystemPartition) -> "CandidateSystem":
    return replace(self, cut=cut)
```

Critical contrast with today's `Subsystem.apply_cut` (subsystem.py:323):
the new method does **not** call a constructor that re-runs
`_backward_tpm` or rebuilds nodes from scratch. `causal_model` is
unchanged across `apply_cut`; only the cut field changes. Derived
properties (`cause_tpm`, `effect_tpm`, `cm`) are `@cached_property` so
they re-derive lazily *only when accessed* — and most of them only
depend on `causal_model.tpm` + `state` + `node_indices`, not on cut.

So `cut_cs.cause_tpm is cs.cause_tpm` whenever the cut hasn't touched
the cause-relevant edges. **This is one place the new layering is
genuinely faster than today**, where every `apply_cut` rebuilds
everything.

`cm` does depend on cut (`cut.apply_cut(network.cm)`) so that one
re-derives. `proper_cm` derives from `cm`, also re-derives. Everything
else is shared.

### Construction-time validation

`CandidateSystem.__post_init__` runs the same validation as today's
`Subsystem.__init__`. Validation failures raise the same exceptions as
today (see Error Handling).

---

## Error Handling

### Construction-time errors (preserved)

| Check | Today's location | New location | Exception |
|---|---|---|---|
| `validate.is_network` | `subsystem.py:107` | `CausalModel.from_network` | `TypeError` |
| `validate.state_length` | `subsystem.py:116` | `CandidateSystem.__post_init__` | `ValueError` |
| `validate.node_states` | `subsystem.py:120` | `CandidateSystem.__post_init__` | `ValueError` |
| `validate.state_reachable` (if `VALIDATE_SUBSYSTEM_STATES`) | `subsystem.py:139` | `CandidateSystem.__post_init__` | `StateUnreachableError` |
| `validate.cut` | `subsystem.py:151` | `CandidateSystem.__post_init__` | `ValueError` |

No new exception types. No exception types removed. All existing tests
against these surfaces continue to bite.

### Runtime errors in `repertoire_algebra` functions

Today's `Subsystem._cause_repertoire` etc. let numpy raise
`IndexError`/`ValueError` for malformed mechanism/purview tuples; this
propagates unchanged. The decorator wrapper does **not** catch — failures
pass through. Cache must not retain failed computations:

```python
def wrapper(cs, *args):
    key = (id(cs), args)
    if key in cache:
        return cache[key]
    result = fn(cs, *args)         # raises propagate; key never added
    cache[key] = result
    return result
```

### `MacroSubsystem` import errors during the P7 → P7b gap

`pyphi.macro` keeps importable names (`MacroSubsystem`, `Blackbox`,
`CoarseGrain`, `MacroNetwork`) but constructing any of them raises
`NotImplementedError("MacroSubsystem is undergoing rewrite in P7b; restored after the kernel rewrite lands.")`.

Files that import `pyphi.macro`:
- `test/test_macro_subsystem.py`
- `test/test_macro_blackbox.py`
- `test/example_networks.py` (shared fixture source)
- `test/test_validate.py`

Three test files get a module-level `pytestmark = pytest.mark.skip(reason="P7b: macro port pending")`:
`test_macro_subsystem.py`, `test_macro_blackbox.py`, plus skipping the
macro-using fixtures inside `example_networks.py`. Tests that don't
touch macro are unaffected. Restored in P7b.

### Migration-time errors

`CausalModel.from_network(network)` is the migration helper. It calls
today's `Network` API to read TPM and connectivity. Errors there are
pre-existing `Network` errors (no new surface). Removed in 2.0 once
direct `CausalModel(...)` construction is the documented path.

### Cache eviction failures

If a `CandidateSystem` is GC'd while a thread is mid-call to a memoized
function on it, the finalizer evicts entries that the caller is about
to write. Result: cache miss for that one call, possible spurious extra
computation on the next caller. Not a correctness issue. Worth a unit
test that exercises GC-under-call but no special handling needed beyond
what `WeakValueDictionary` and `weakref.finalize` already provide.

### What does NOT change

- All existing `pyphi.exceptions` types remain
- All existing `validate.*` surface remains
- Numerical-precision errors (existing `PRECISION` config) — same handling
- Parallel computation errors (loky propagation) — unchanged

---

## Testing

### What pins correctness through the rewrite

| Test layer | What it pins | Coverage |
|---|---|---|
| **Golden fixtures (P1)** | numerical match to 1e-12 across IIT 3.0, IIT 4.0 (2023), IIT 4.0 (2026) | 17 fixtures |
| **Hypothesis invariants (P2)** | mathematical properties (non-negativity, Φ ≤ φ_max, formalism dispatch consistency) | 19 properties |
| **Sign-flip canary** | acceptance criterion: a deliberate sign-flip in `metrics/distribution.py` must fail ≥3 fixtures + ≥1 property test | passes today |
| **Surface drift test (P3)** | `SubsystemPublicInterface` Protocol conformance — names, attributes, methods | enforces 33-name contract |

The surface drift test changes its assertion target from `Subsystem` to
`CandidateSystem`. The Protocol body is unchanged.

### New tests added in P7

**`test/test_core_tpm.py`** (~150 lines) — TPM Protocol conformance and
`ExplicitTPM` parity
- `ExplicitTPM` instance satisfies the `TPM` Protocol via
  `runtime_checkable`
- Numerical parity: every method in `ExplicitTPM` produces bit-identical
  output to today's `pyphi.tpm.ExplicitTPM` for ≥10 random networks
- `cause_tpm` / `effect_tpm` (the marginalization functions) match
  today's `_backward_tpm` and `condition_tpm` outputs

**`test/test_core_candidate_system.py`** (~200 lines) — value-type
semantics
- Frozen: `cs.state = ...` raises `FrozenInstanceError`
- Equality: two `CandidateSystem` with identical fields compare equal;
  different cuts compare unequal
- Hash stability: `hash(cs)` is deterministic and survives
  `pickle.dumps/loads`
- `apply_cut` returns a new instance, leaves `self` untouched
- `cause_tpm` / `effect_tpm` `cached_property` survives across
  `apply_cut` when cut doesn't affect cause-relevant edges
- Construction validation: same `ValueError` / `StateUnreachableError`
  as today's `Subsystem`

**`test/test_core_repertoire_algebra.py`** (~250 lines) —
module-of-functions parity + cache mechanics
- Numerical parity: every function produces bit-identical output to
  the corresponding `Subsystem` method (regression test against frozen
  baseline computed from current `Subsystem`)
- Cache hit/miss instrumentation: a memoized call on the same
  `(cs, m, p)` increments hit count
- Cache eviction: GC'ing a `CandidateSystem` evicts its cache entries
- Cache failure: a raise does not poison the cache
- `cache_info()` and `clear_caches()` proxy correctly

**`test/test_core_layering.py`** (~80 lines) — architectural assertions
- `core/causal_model.py` does not import `core/repertoire_algebra`
- `core/repertoire_algebra` does not import `formalism/`
- `core/` package has no `from ..subsystem import` lines
- `Concept.subsystem` no longer exists (regression check)

**`test/test_macro_disabled_during_p7_gap.py`** (~20 lines, deleted at
P7b end) — sentinel
- `pyphi.macro` import succeeds
- `MacroSubsystem(...)` raises `NotImplementedError` with the expected
  message

### Existing tests that need updating

29 test files reference `Subsystem`. The bulk is mechanical:
`Subsystem(network, state, ...)` →
`CandidateSystem.from_network(network, state, ...)` (or the direct
constructor where appropriate).

Three files need attention beyond the mechanical rename:
- `test/test_subsystem.py` — the largest. Most tests pass through to
  `CandidateSystem` methods unchanged. Tests that target
  `_cause_repertoire` directly move to `test_core_repertoire_algebra.py`
- `test/test_subsystem_surface.py` — already targets
  `SubsystemPublicInterface`. Update assertion target to
  `CandidateSystem`
- `test/test_actual.py` — exercises `actual.py` which gets the
  mechanical port (Q6); tests update to match

### Test running strategy during the P7 worktree

Fast/slow lane split (per CLAUDE.md):
- **Fast lane** (under 1 min): `test_core_*` (new),
  `test_subsystem_surface.py`, `test_partition.py`,
  `test_golden_regression.py`, `test_invariants.py` — runs on every
  internal commit
- **Slow lane** (5-10 min): `test_invariants_hypothesis.py` — runs on
  every layer-boundary commit (e.g., after `repertoire_algebra` is
  complete; after `CandidateSystem` is complete; before merge)

Both lanes run green before the worktree merges to develop.

### Acceptance criteria for P7

| Gate | Pass condition |
|---|---|
| Golden fixtures (P1) | all 17 match to 1e-12 |
| Hypothesis invariants (P2) | all 19 properties green |
| Sign-flip canary | mutating `hamming_emd` still fails ≥3 fixtures + ≥1 property |
| Surface drift (P3) | `CandidateSystem` satisfies `SubsystemPublicInterface` |
| Type checking | `uv run pyright pyphi/core` clean (within current strict-mode scope) |
| New tests | `test_core_*.py` all green |
| Macro gap | `pyphi.macro` imports succeed; `MacroSubsystem(...)` raises `NotImplementedError` with the expected message |
| Layering | `test_core_layering.py` confirms downward-only dependency |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `MacroSubsystem`-shaped abstractions leak into `CandidateSystem` design | P7 must paper-validate the `CausalModel → transform → CandidateSystem` macro functor before code freeze. Walk through coarse-graining and blackboxing on whiteboard against the proposed types |
| Cache key based on `id(cs)` collides under concurrent eviction | `WeakValueDictionary` + `weakref.finalize` guarantees the live `CandidateSystem` keeps its `id` while alive; finalizer runs only after collection. Race window during GC is limited to one cache miss per affected key |
| Surface drift: Protocol stays satisfied but semantics drift silently | Hypothesis invariants + golden fixtures + sign-flip canary collectively pin numerical behavior. Type-level conformance alone is not the sole guard |
| `Concept.subsystem` removal breaks third-party code | Acceptable per 2.0 cutoff (single-release shipping, no backward compat shims) |
| `apply_cut` `cached_property` sharing creates hidden coupling | Frozen dataclass + `cached_property` is a well-defined idiom; any "sharing" is read-only structural sharing of immutable objects, not mutable aliasing |
| 29 test files updated mechanically; sed misses an edge case | One commit at the worktree's tail does the rename plus manual review of the 5 files that mix `Subsystem` with construct sites |
| TPM Protocol shape gets the wrong abstraction | Mitigated by reading PR #105's `ImplicitTPM` shape during P7 design — the Protocol must admit both backends. Verified at P12; if Protocol needs adjustment then, that's a P12 cost, not a P7 design failure |

---

## What Does NOT Happen in P7 (Deferred)

- `MacroSubsystem` port — **P7b**
- `ImplicitTPM` and non-binary support — **P12**
- xarray-backed repertoires — **P12** (or later, depending on
  benchmark)
- Redis cache backend — **P9**
- Concept / RIA model split — **P8**
- Full `actual.py` rewrite — **P14** (only mechanical port in P7)
- `core/repertoire_algebra.py` content-keyed (cross-cut) caching —
  no scheduled home; future research direction

---

## Sequencing Notes for the Implementation Plan

The implementation plan (next step) will detail commit-level
sequencing. High-level shape:

1. Worktree setup; create empty `pyphi/core/` skeleton
2. TPM Protocol + `ExplicitTPM` port; tests pass
3. `Unit`, `Substrate`, `CausalModel` value types; tests pass
4. `CandidateSystem` value type with `@cached_property` derived
   properties; constructor parity with `Subsystem.__init__`
5. `core/repertoire_algebra.py` — port repertoire functions one by
   one, each guarded by a parity test
6. `CandidateSystem` proxy methods (the `SubsystemPublicInterface`
   surface)
7. Update `formalism/iit3` and `formalism/iit4` to use
   `CandidateSystem`
8. Update `actual.py` (mechanical), `examples.py`, `compute/`
9. Disable `pyphi.macro` (NotImplementedError on construction)
10. Update tests: rename `Subsystem` → `CandidateSystem`; skip macro
    test files
11. Delete `pyphi/subsystem.py`
12. Final: full suite green; surface drift green against
    `CandidateSystem`; type-check clean; merge

Each step ends with the fast lane running green; layer-boundary steps
(2, 4, 5, 11) also run the slow lane.
