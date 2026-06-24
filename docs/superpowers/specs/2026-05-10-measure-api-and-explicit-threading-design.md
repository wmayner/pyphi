# Measure API + Explicit-Parameter Threading — Design

**Status:** approved (2026-05-10)
**Branch base:** 2.0 head `5f9789c2`
**Scope:** unified bundle covering (a) typed metric Protocols + registry split, and (b) threading measure choices through every internal call as explicit parameters.

---

## Motivation

The `config.override(measure=...)` wrapper pattern in `IIT4_2026Formalism`
(5 sites; `pyphi/formalism/iit4/formalism.py:266-305`) is action-at-a-distance:
internal helpers read `config.formalism.iit.{mechanism,system}_phi_measure`
inside a temporary override block, with no visible parameter announcing which
metric is in play. The dispatch is implicit, scope-dependent, and easy to
miss when renaming or refactoring.

The cap regression in `test_phi_capped_by_ii` (fixed in commit `85d8e029`)
is the canonical instance of the bug class this enables: a mechanical
rename of `repertoire_measure` to `mechanism_phi_measure` left the test's
`II_CONFIG` overriding the wrong field — the cap branch in
`pyphi/formalism/iit4/__init__.py:487` reads `system_phi_measure`, not
`mechanism_phi_measure`. The test silently observed an uncapped value,
the briefing's "all gates green" check missed it (narrow lane), and the
bug only surfaced when the broader fast lane ran during today's perf-budget
work.

Beyond the specific bug: global-config mutation introduces parallel-state
hazards (`ConfigSnapshot.as_kwargs` exists today as a workaround) and
makes every call site harder to reason about — the metric in effect
depends on call-stack-history rather than visible arguments.

This refactor eliminates the bug class by construction: every internal
function receives its metric(s) as explicit parameters, and config is
read at exactly one boundary — the user-facing `System` class methods.

## Goals

1. **Cap-regression bug class impossible by construction.** No internal
   code reads measure config fields. Pyright catches scope mismatches.
2. **Typed metric API.** Three Protocols (`DistributionMetric`,
   `StateAwareMetric`, `CompositeMetric`) capture the real shape
   diversity; one registry per Protocol.
3. **Remove `config.override(measure=...)` wrapper pattern** from
   `IIT4_2026Formalism` and elsewhere.
4. **Fold in P5-deferred metric warts**: `INTRINSIC_DIFFERENTIATION`'s
   vestigial `q` arg removed; return-type discipline normalized within
   each Protocol.
5. **Parallel treatment for Actual Causation** (`alpha_measure`,
   `background_scheme`, `alpha_aggregation`,
   `partitioned_repertoire_scheme`) — same risk class, same fix.

## Non-goals

- Changing `System.sia()` / `System.find_mip()` / `System.phi_structure()`
  public signatures. The class-method API stays stable.
- Removing the `config.formalism.iit.{mechanism,system}_phi_measure`
  config fields. They still exist and are read by the public class
  methods at the boundary.
- Back-compat shims for the old `measures` registry or vestigial
  signatures. Pre-2.0 per CLAUDE.md no-back-compat-shims rule; callers
  migrate.
- Rewriting metric implementations. The math stays; only typing,
  signatures, and resolution change.
- Changing `Subsystem.intrinsic_information()` user-facing signature
  (it's part of public API; remains config-reading).

## Architecture

Three typed `Protocol` classes capture the metric shape diversity that's
currently muddled in one string-keyed registry. Below the user-facing
`System` class methods, every function that uses a metric receives it
as an explicit parameter. The `config.override(measure=...)` wrapper
pattern is removed entirely; formalism classes resolve their
`default_*_metric` ClassVars to concrete metric objects at the
entry-point boundary and thread them through. `pyphi.formalism.iit4.sia`
and equivalents become "internal" — they require explicit metric args,
no config fallback. `config.formalism.iit.{mechanism,system}_phi_measure`
config fields still exist; only `System.*` reads them.

## Components

### Metric Protocols

New file: `pyphi/metrics/protocols.py`

```python
class DistributionMetric(Protocol):
    """(p, q) -> float — symmetric or asymmetric distribution distance."""
    name: str
    asymmetric: bool
    def __call__(self, p: ArrayLike, q: ArrayLike) -> float: ...

class StateAwareMetric(Protocol):
    """(p, state) -> float — pointwise probability at a specified state."""
    name: str
    def __call__(self, p: ArrayLike, state: State) -> float: ...

class CompositeMetric(Protocol):
    """Multi-input metric returning rich DistanceResult metadata.

    Used by GID / INTRINSIC_INFORMATION / INTRINSIC_SPECIFICATION at the
    system level.
    """
    name: str
    def __call__(
        self,
        forward: ArrayLike,
        partitioned: ArrayLike,
        selectivity: ArrayLike | None = None,
        *,
        state: State | None = None,
    ) -> DistanceResult: ...
```

Each existing registered metric function gets retyped against the right
Protocol. `INTRINSIC_DIFFERENTIATION`'s vestigial `q` arg is removed
(becomes `StateAwareMetric`-shaped: `(p, state) -> float`).

### Registry split

`pyphi/metrics/distribution.py::measures` (one mixed registry) splits
into three typed registries:

```python
distribution_metrics: Registry[DistributionMetric]
state_aware_metrics: Registry[StateAwareMetric]
composite_metrics: Registry[CompositeMetric]
```

Each registry's `register(name)` decorator validates the function's
signature against the relevant Protocol at registration time (raises
on mismatch — catches signature drift early).

A flat lookup helper resolves a metric name across all three registries
when the scope is ambiguous (e.g., a `Subsystem.evaluate_partition` call
that uses a metric whose shape isn't fixed by the call site).

### Resolved-metric helpers

```python
def resolve_mechanism_metric(name: str) -> StateAwareMetric | CompositeMetric: ...
def resolve_system_metric(name: str) -> CompositeMetric: ...
def resolve_alpha_measure(name: str) -> DistributionMetric: ...
```

Used at the boundary by `System` class methods and at `default_*_metric`
resolution by formalism classes. Raise on unknown names with a list of
valid options.

### Actual Causation parallel

Symmetric pattern. The AC measures live in
`pyphi/metrics/distribution.py::actual_causation_measures` (registry)
and are consumed by `pyphi/actual.py` (`Transition.alpha` and helpers).
A new `pyphi/metrics/actual_causation.py` may extract the AC-specific
metric Protocol + registry if it cleans up the dependency direction;
that's an implementation call, not a design constraint.

- `Transition.alpha()` (public) resolves `alpha_measure`,
  `background_scheme`, `alpha_aggregation` at the boundary.
- Internal AC helpers (`probability_distance`, partitioned-repertoire
  constructors, etc.) receive these as explicit params.
- AC's `config.override` patterns (if any — to be audited during
  implementation) are removed.

### Removed surface

- `pyphi/metrics/distribution.py::measures` mixed registry — deleted.
- `measures.asymmetric()` method — replaced by per-metric `asymmetric: bool`
  attribute on `DistributionMetric` Protocol.
- `IIT4_2026Formalism`'s 5 `with config.override(...)` blocks — deleted.
- Vestigial `q` arg in `INTRINSIC_DIFFERENTIATION` — deleted.
- `ConfigSnapshot.as_kwargs()` colliding-field exclusion workaround —
  revisited; likely simpler or removable once threading lands.

## Data flow

End-to-end trace of `System.sia()` post-refactor:

```
user code: my_system.sia()
   │
   ▼  ◄── config read happens HERE (and only here)
System.sia(self):
    formalism = get_active_formalism()
    system_metric = resolve_system_metric(
        config.formalism.iit.system_phi_measure
    )
    return formalism.evaluate_system(
        self, system_metric=system_metric
    )
   │
   ▼
IIT4_2026Formalism.evaluate_system(
    self, system, *, system_metric
):
    check_metric_compatible(self, system_metric)
    return _sia(system, system_metric=system_metric)
   │
   ▼  ◄── NO config read; explicit param required
pyphi.formalism.iit4._sia(
    system, *, system_metric: CompositeMetric, **kwargs
):
    ...
    return _evaluate_partition(
        partition,
        system_metric=system_metric,
        ...
    )
   │
   ▼
_evaluate_partition(
    partition, *, system_metric, ...
):
    # The GID-conversion lives once, at this layer, name-typed:
    if system_metric.name == "INTRINSIC_INFORMATION":
        partition_metric = composite_metrics["GENERALIZED_INTRINSIC_DIFFERENCE"]
    else:
        partition_metric = system_metric

    integration = integration_value(
        direction, system, partition,
        metric=partition_metric
    )

    if system_metric.name == "INTRINSIC_INFORMATION":
        # Eq. 23 cap branch, gated on metric.name (not config)
        ...
```

Key shape changes:

- `System.sia(self)` signature **unchanged**. Public API stable.
- `pyphi.formalism.iit4.sia` adds required `system_metric: CompositeMetric`
  kwarg. Tests using raw import need updating.
- Formalism methods add `*, system_metric` (or `mechanism_metric`) kwargs.
- Internal helpers (`_sia`, `_evaluate_partition`, `_find_mip_iit4`,
  `integration_value`, etc.) get explicit metric params. Config reads
  deleted.
- `if measure == "X"` string dispatch becomes `if metric.name == "X"`,
  scoped to the layer that branches on metric identity (notably the
  GID-conversion in `_evaluate_partition`).

## Error surfaces

| Failure | Caught by | When |
|---|---|---|
| Wrong metric type for scope (e.g., `StateAwareMetric` where `CompositeMetric` required) | Pyright | Static analysis |
| Missing required metric arg in internal call | Pyright | Static analysis |
| Unknown metric name in config | `resolve_*_metric` raising `ValueError` | Runtime, at boundary |
| Formalism-metric incompatibility | `check_metric_compatible` raising `MetricNotCompatibleError` | Runtime, at formalism method entry |
| Metric function signature drift from Protocol | Registry's `register` validation | Import time |

## Testing strategy

### Unchanged (passes as-is)

- **Goldens** (17 fixtures): go through `System.sia()` / `System.phi_structure()`,
  which still read config at the boundary. No fixture regeneration.
- **Hypothesis invariants**: same surface.
- **Tier 1 perf budgets**: same surface.
- **Sign-flip canary**: still bites (same hot-path mutation).

### Tests requiring updates

- `test/test_big_phi_robust.py` (3 sites importing `from pyphi.formalism.iit4 import sia`)
  must pass `system_metric=resolve_system_metric(...)` explicitly. The
  `TestEq23IntrinsicInformationCap` tests become more honest about what
  they're testing.
- Any other tests that import raw module-level functions from
  `pyphi.formalism.iit4` (audit during implementation via
  `git grep "from pyphi.formalism.iit4 import"`).

### New tests for the refactor

1. **`test/test_metric_protocols.py`** — `isinstance(emd, DistributionMetric)`
   etc. for every registered metric. Catches signature drift at the
   Protocol level.
2. **`test/test_metric_resolution.py`** — `resolve_*_metric` returns the
   right typed callable; raises descriptively on unknown names; AC
   variants work.
3. **`test/test_formalism_metric_threading.py`** — calling
   `IIT4_2026Formalism.evaluate_system(system, system_metric=gid_metric)`
   with a non-default metric actually uses that metric (no implicit
   override from `default_system_metric`).
4. **Cap-regression-impossible test** — explicitly attempt the bug:
   `config.override(mechanism_phi_measure="INTRINSIC_INFORMATION")` and
   call raw `sia()` without passing `system_metric`; expect pyright /
   runtime to refuse. Pins the new contract.

## Migration phases

Six phases, each ending green (goldens 17/17, fast lane, hypothesis
fast lane, perf budget, pyright + ruff clean). One commit per phase.

### Phase 1: Protocol scaffolding + INTRINSIC_DIFFERENTIATION cleanup (~1 day)

- Create `pyphi/metrics/protocols.py` with the three Protocols.
- Remove `INTRINSIC_DIFFERENTIATION`'s vestigial `q` arg; update its
  one or two callers (small wart, bundles cleanly with type intro
  since otherwise the function doesn't satisfy `StateAwareMetric`).
- Retype existing metric functions against the right Protocol.
- Add `test/test_metric_protocols.py` confirming every registered
  metric satisfies its Protocol.
- Existing code unchanged elsewhere; everything still works via the
  string-keyed `measures` registry.

Acceptance: `uv run pyright pyphi/metrics/` clean; new protocol tests
green; goldens 17/17.

### Phase 2: Registry split + resolvers (~1 day)

- Add `distribution_metrics`, `state_aware_metrics`, `composite_metrics`
  typed registries.
- Move all `@measures.register("X")` registrations to the appropriate
  typed registry.
- Add `resolve_{mechanism,system}_metric` and `resolve_alpha_measure`
  helpers.
- Delete `measures` mixed registry. Update internal callers to use the
  typed registries.
- Add `test/test_metric_resolution.py`.

Acceptance: goldens 17/17; fast lane green; pyright clean.

### Phase 3: Thread internal helpers (~2 days)

- Add explicit metric params to every helper below the formalism class
  boundary:
  - `pyphi/formalism/iit4/__init__.py::{_sia, _evaluate_partition,
    _find_mip_iit4, integration_value, intrinsic_differentiation_value,
    system_intrinsic_information}`
  - `pyphi/formalism/iit3/formalism.py` internal helpers
  - `pyphi/core/repertoire_algebra.py` internal helpers
  - `pyphi/metrics/distribution.py::repertoire_distance` (legacy
    distributor — likely deletable in Phase 5)
- Each helper's config reads removed; callers pass metrics explicitly.
- Public/formalism-class entry points still read config and resolve at
  the boundary; pass to internals.

Acceptance: goldens 17/17; fast lane green; pyright clean; no
`config.formalism.iit.*_phi_measure` reads remain in
`pyphi/{core,metrics}/`.

### Phase 4: Thread formalism class methods (~1.5 days)

- `IIT4_2023Formalism.evaluate_*` and `IIT4_2026Formalism.evaluate_*`
  add `*, mechanism_metric=None, system_metric=None` kwargs. When
  called without explicit metrics (the default case), each method
  resolves from `self.default_{mechanism,system}_metric` ClassVar — no
  config involvement inside.
- IIT4_2026's 5 `with config.override(...)` wrappers deleted.
- `IIT3Formalism` gets the same treatment.
- Add `test/test_formalism_metric_threading.py`.

Acceptance: goldens 17/17; pyright clean. Visual diff: 5
`config.override` blocks in `pyphi/formalism/iit4/formalism.py` gone.

### Phase 5: Thread module-level functions + update raw-import tests (~1.5 days)

- `pyphi.formalism.iit4.{sia, find_mip, phi_structure}` add required
  `system_metric` / `mechanism_metric` kwargs. Their config reads
  removed.
- Audit and update tests that import these directly:
  - `test/test_big_phi_robust.py` (3 known sites)
  - any others surfaced by
    `git grep "from pyphi.formalism.iit4 import"`
- The cap-regression-impossible test (item 4 in Testing strategy)
  added here.

Acceptance: fast lane green; goldens 17/17. The cap regression is now
impossible-by-construction (verified by the new test).

### Phase 6: Actual Causation parallel (~1.5 days)

- Same pattern for `alpha_measure`, `background_scheme`,
  `alpha_aggregation`, `partitioned_repertoire_scheme`.
- `Transition.alpha()` (public) resolves at boundary; helpers
  (`probability_distance`, etc.) receive explicit params.
- Tests updated where needed.
- `ConfigSnapshot.as_kwargs()` colliding-field exclusion workaround
  revisited — likely simpler or removable.

Acceptance: AC tests green (existing `TestActualCausationIIT30`
suite); full fast lane + golden + hypothesis fast green; perf budget
intact.

### Total estimate

~8 days of focused work.

## Files affected (canonical list)

**Created:**
- `pyphi/metrics/protocols.py`
- `test/test_metric_protocols.py`
- `test/test_metric_resolution.py`
- `test/test_formalism_metric_threading.py`

**Modified (substantial):**
- `pyphi/metrics/distribution.py` (Protocol typing; registry split)
- `pyphi/formalism/iit4/__init__.py` (thread params; remove config reads)
- `pyphi/formalism/iit4/formalism.py` (remove `config.override` blocks;
  add kwargs)
- `pyphi/formalism/iit3/formalism.py` (thread params)
- `pyphi/core/repertoire_algebra.py` (thread params; remove config
  reads)
- `pyphi/system.py` / `pyphi/subsystem.py` (resolve at boundary)
- `pyphi/actual.py` (AC parallel)
- `pyphi/conf/snapshot.py` (workaround revisit)
- `test/test_big_phi_robust.py` (raw-import tests)

**Modified (light):**
- `pyphi/metrics/ces.py` (per-metric `asymmetric` attribute read)
- `pyphi/visualize/distribution.py` (same)
- `pyphi/formalism/queries.py` (audit during Phase 3; thread params if
  any reads found — `find_mip` dispatcher is suspicious).

## Risks

| Risk | Mitigation |
|---|---|
| Threading introduces subtle param-passing bugs (wrong metric instance forwarded) | Six-phase migration with goldens + Hypothesis green at each phase; new threading tests pin the wiring. |
| Public API surface drift (someone relies on raw `pyphi.formalism.iit4.sia` working with config alone) | Pre-2.0, no external API contract yet. Audit + update all internal callers. Document the change in changelog. |
| Performance regression from extra arg-passing | Tier 1 perf budget catches catastrophic regressions; expected delta is sub-millisecond (Python attribute lookup cost). |
| Phase 4 leaves IIT3Formalism in an inconsistent intermediate state if a phase is skipped | Each phase is atomic / green; don't skip phases. |
| `ConfigSnapshot.as_kwargs()` colliding-field logic still needed for some other reason | Revisit in Phase 6 with concrete picture; remove or simplify based on observed call sites. |
| AC measure home isn't `pyphi/metrics/` (it's `pyphi/actual.py` or elsewhere) | Resolve location during Phase 6 — design doesn't pin the file. |

## Acceptance criteria

After all six phases:

1. **Cap-regression-impossible test passes** — attempting the original
   bug shape (`config.override(mechanism_phi_measure=...)` + raw `sia()`)
   is statically rejected by pyright or runtime, not silently observed.
2. **All goldens (17) match unchanged** — no fixture regeneration.
3. **Fast lane green** — `pytest test/ -m "not slow"` zero failures.
4. **Hypothesis fast lane green** — invariants intact.
5. **Tier 1 perf budget intact** — 5/5 canaries within budget; no
   regression from threading overhead.
6. **Pyright clean** — no new errors in touched files; baseline
   (2 pre-existing geometry.py) maintained.
7. **Zero internal config reads of measure fields** —
   `git grep 'config\.formalism\.iit\.\(mechanism\|system\)_phi_measure'
   pyphi/` returns only `System.*` class methods.
8. **Zero `config.override(measure=...)` wrappers** —
   `git grep 'config\.override(.*measure' pyphi/` returns nothing
   (or only test code where context-managed override is the test's
   own setup).
9. **No P# markers, "Phase A", "ROADMAP" references** in any new
   source/comments/docstrings.

## What does NOT change

- Existing metric implementations (the math).
- `Subsystem.intrinsic_information()` public signature.
- `System.sia()` / `System.find_mip()` / `System.phi_structure()`
  signatures.
- `config.formalism.iit.{mechanism,system}_phi_measure` config fields.
- `config.formalism.actual_causation.{alpha_measure, background_scheme,
  alpha_aggregation}` config fields.
- Golden fixtures.
- Hypothesis invariants.
- Tier 1 perf budget values.
