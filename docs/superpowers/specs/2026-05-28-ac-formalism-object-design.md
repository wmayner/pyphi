# AC Formalism Object — Design Spec

**Status:** Approved (design); pending implementation plan.
**Date:** 2026-05-28.

## Motivation

PyPhi's measure-configuration architecture is asymmetric between IIT and
Actual Causation (AC).

IIT exposes its measures as first-class, registry-resolved config fields
consumed by a **registered formalism object**: `config.formalism.iit.version`
selects an `IIT4_2023Formalism` / `IIT4_2026Formalism` / `IIT3Formalism`
instance from `FORMALISM_REGISTRY`; that object holds a `config` snapshot and
a `compatible_measures` set, resolves each configured measure name to a
callable, checks compatibility, and then delegates to module-level algorithm
functions (`evaluate_system` → `_sia`, `build_ces` → `_ces`, etc.).

AC has no formalism object. `pyphi/actual.py` resolves its four formalism
knobs (`alpha_measure`, `partitioned_repertoire_scheme`, `background_scheme`,
`alpha_aggregation`) through a loose `_resolve_ac_kwargs()` dict-builder
threaded through free functions (`account`, `directed_account`, `sia`) and
`Transition` methods (`find_mip`, `find_causal_link`). There is no registry
selection, no `compatible_measures` gate, and `alpha_measure` is the only
measure-typed field with no early structural home.

This spec brings AC to **structural parity** with the IIT formalism
abstraction: a registered `AC2019Formalism` object that owns measure/scheme
resolution and evaluation dispatch, selected by a new
`config.formalism.actual_causation.version` field, in a self-contained
`pyphi/formalism/actual_causation/` package mirroring `pyphi/formalism/iit4/`.

## Goals

- A registered `AC2019Formalism` object owning AC evaluation dispatch and
  measure/scheme resolution, mirroring `IIT4_2023Formalism`.
- A `compatible_measures` gate so a misconfigured `alpha_measure` fails with a
  clear error at resolve time (matching IIT's behavior).
- Registry-selectable AC formalism via `config.formalism.actual_causation.version`.
- A self-contained `pyphi/formalism/actual_causation/` package holding the AC
  compute algorithms, leaving `actual.py` as the data layer plus thin public
  dispatchers.
- Zero numeric change to any AC result; the public API is unchanged.

## Non-goals (out of scope)

- **IIT-4.0-style AC measures.** Letting AC compute alpha with generalized
  intrinsic difference / intrinsic information rather than the 2019 PMI/WPMI is
  a separate formalism project, and it intersects the cause-weighting question
  currently pending Larissa's reply. This spec only builds the seam (the
  registry + `version` field) where such a second AC formalism would later
  register; it does not add one.
- Merging AC's measure/scheme registries into the IIT measure registries. They
  stay separate (AC's `(p, q) -> float` alpha measures differ in shape from
  IIT's four typed measure registries).
- Any change to `Transition` / `TransitionSystem` data semantics.

## Current architecture (reference)

**IIT path.** `config.formalism.iit.version` → `FORMALISM_REGISTRY[version]`
(`pyphi/formalism/base.py`) → e.g. `IIT4_2023Formalism`
(`pyphi/formalism/iit4/formalism.py`). The `PhiFormalism` Protocol
(`pyphi/formalism/base.py`) declares `name` / `compatible_measures` /
`partition_scheme` ClassVars, a `config` property, and
`evaluate_mechanism` / `evaluate_mechanism_partition` / `evaluate_system` /
`build_ces`. Each formalism method resolves measures via helpers
(`_resolve_system_measures`, `_resolve_mechanism_measure`) that call
`resolve_*_measure` (`pyphi/measures/distribution.py`) and
`check_measure_compatible`, then delegates to a module-level algorithm
(`_sia`, `_ces`, `_find_mip_iit4`). Dispatch sites live in
`pyphi/formalism/queries.py` (`formalism = FORMALISM_REGISTRY[config.formalism.iit.version]`).

**AC path.** `pyphi/actual.py` defines three bespoke registries
(`partitioned_repertoire_schemes`, `background_strategies`,
`alpha_aggregations`) and resolves all four knobs in `_resolve_ac_kwargs()`
(alpha via `resolve_actual_causation_measure` against the
`actual_causation_measures` registry in `pyphi/measures/distribution.py`). The
compute call graph: `account` → `directed_account` → `Transition.find_causal_link`
→ `Transition.find_mip` → `probability_distance` → resolved alpha measure;
`sia` evaluates system partitions via `_evaluate_partition` / `_get_partitions`
and `account_distance`. No formalism object exists.

## Design

### Package layout

New package `pyphi/formalism/actual_causation/`, mirroring
`pyphi/formalism/iit4/`:

- `__init__.py` — exports `AC2019Formalism`; registers it
  (`ACTUAL_CAUSATION_FORMALISM_REGISTRY.register("AC_2019", AC2019Formalism())`); re-exports the
  algorithm functions used by the public dispatchers.
- `formalism.py` — the `AC2019Formalism` frozen dataclass and the
  `_resolve_ac_measures(...)` helper.
- Algorithm module(s) holding the functions **moved from `actual.py`**:
  `_account`, `_directed_account`, `_find_mip`, `_find_causal_link`, `_sia`,
  `_evaluate_partition`, `_get_partitions`, plus the AC compute utilities
  `probability_distance` and `account_distance`, and the three AC registries
  (`partitioned_repertoire_schemes`, `background_strategies`,
  `alpha_aggregations`) with their registered functions
  (`_partitioned_repertoire_product`, `_background_uniform`,
  `_alpha_subtractive`).

`pyphi/actual.py` retains the **data layer**: `Transition`,
`TransitionSystem`, and their repertoire/probability/structural methods. Its
public compute entry points become thin dispatchers (see *Public API*).

### `AC2019Formalism` object

```python
@dataclass(frozen=True)
class AC2019Formalism:
    """Actual Causation formalism (Albantakis et al. 2019, "What Caused What?")."""

    name: ClassVar[str] = "AC_2019"
    compatible_measures: ClassVar[frozenset[str]] = frozenset({"PMI", "WPMI"})
    config: FormalismConfig = field(default_factory=_default_formalism_config)

    def evaluate_account(self, transition, direction=Direction.BIDIRECTIONAL, **kwargs): ...
    def evaluate_system(self, transition, direction=Direction.BIDIRECTIONAL, **kwargs): ...
    def evaluate_mechanism(self, transition, direction, mechanism, purview, **kwargs): ...
    def evaluate_causal_link(self, transition, direction, mechanism, purviews=None, **kwargs): ...
```

- `evaluate_system` ≙ today's `sia` (system MIP over the transition).
- `evaluate_account` ≙ today's `account` / `directed_account`.
- `evaluate_mechanism` ≙ today's `Transition.find_mip` (mechanism MIP over a purview).
- `evaluate_causal_link` ≙ today's `Transition.find_causal_link`.

Each method resolves measures/schemes once via `_resolve_ac_measures(self, ...)`
(below), then delegates to the moved algorithm functions — the same
resolve-then-delegate shape as `IIT4_2023Formalism.evaluate_system` →
`_resolve_system_measures` → `_sia`.

### Measure/scheme resolution

`_resolve_ac_measures(formalism, *, alpha_measure=None, partitioned_repertoire_scheme=None, ...)`
replaces `_resolve_ac_kwargs()`. It:

1. Reads `formalism.config.actual_causation` (the snapshot), honoring any
   explicit kwarg overrides (mirroring IIT's "explicit arg wins over config").
2. Resolves `alpha_measure` name → callable via
   `resolve_actual_causation_measure(...)` and **checks it against
   `formalism.compatible_measures`** (via the existing
   `check_measure_compatible` used by IIT), raising on unknown/incompatible.
3. Resolves the three schemes from `partitioned_repertoire_schemes` /
   `background_strategies` / `alpha_aggregations`.

### Protocol + registry

Added to `pyphi/formalism/base.py`, beside `PhiFormalism` /
`FormalismRegistry` / `FORMALISM_REGISTRY`:

- `ActualCausationFormalism` — a `@runtime_checkable` Protocol declaring the AC method
  surface (`name`, `compatible_measures` ClassVars; `config` property;
  `evaluate_account` / `evaluate_system` / `evaluate_mechanism` /
  `evaluate_causal_link`). Separate from `PhiFormalism` because AC methods are
  transition-based with different signatures.
- `ActualCausationFormalismRegistry(Registry[ActualCausationFormalism])`
  validating registrations against `ActualCausationFormalism`, and a
  module-level `ACTUAL_CAUSATION_FORMALISM_REGISTRY` instance (exact parallel
  to `ActualCausationMeasureRegistry`). Lookup key is
  `config.formalism.actual_causation.version`.

### Config changes

In `pyphi/conf/formalism.py`, `ActualCausationConfig`:

- Add `version: str = "AC_2019"`.
- **Validation policy mirrors IIT:** `version` and `alpha_measure` are **not**
  validated in `__post_init__` (the config module imports before formalism
  registries populate; IIT defers measure-name validation to resolve time for
  exactly this reason). Validation happens when `AC2019Formalism` resolves the
  measure and checks `compatible_measures`, and when a dispatcher looks the
  formalism up in `ACTUAL_CAUSATION_FORMALISM_REGISTRY` (a missing key raises). The existing
  `__post_init__` frozenset checks for the three scheme fields are unchanged.

### Public API (unchanged surface; dispatch indirection)

These keep their names and signatures and become thin dispatchers to
`ACTUAL_CAUSATION_FORMALISM_REGISTRY[config.formalism.actual_causation.version]`:

- `pyphi.actual.account(transition, direction=..., **kw)` →
  `formalism.evaluate_account(...)`.
- `pyphi.actual.directed_account(...)` → `formalism.evaluate_account(...)`
  (single direction).
- `pyphi.actual.sia(transition, direction=..., **kw)` →
  `formalism.evaluate_system(...)`.
- `Transition.find_mip(...)` → `formalism.evaluate_mechanism(self, ...)`.
- `Transition.find_causal_link(...)` → `formalism.evaluate_causal_link(self, ...)`.
- `Transition.find_actual_cause` / `find_actual_effect` keep delegating to
  `find_causal_link` as today.

The three AC registries and `probability_distance` / `account_distance` move
into the new package. In-repo references are updated to the new import path; if
they are part of the public surface (users register custom schemes), the move
is noted in the changelog (no back-compat re-export shim, per project
convention for unreleased 2.0 work).

## Correctness contract

This is a **pure refactor plus dispatch indirection**. No AC result changes
numerically. The contract is enforced by:

- The full AC suite in `test/test_actual.py` (826 lines, including
  Albantakis et al. 2019 paper-fixture acceptance tests) staying green and
  byte-identical.
- Full `uv run pytest` (no path argument — includes doctests) green.

## Testing strategy

New tests (in `test/test_actual.py` or a new
`test/test_ac_formalism.py`):

- `AC2019Formalism` satisfies `ActualCausationFormalism` and is retrievable from
  `ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]`.
- An unknown or incompatible `alpha_measure` raises at resolve time (parity
  with IIT's `check_measure_compatible`).
- `config.formalism.actual_causation.version` selects the formalism;
  an unknown version raises a clear registry KeyError.
- Parity smoke test: `account` / `sia` on a paper fixture produce results
  equal to the pre-refactor values (covered by the existing acceptance tests,
  which must remain byte-identical).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Import-ordering: `ACTUAL_CAUSATION_FORMALISM_REGISTRY` must be populated before a dispatcher resolves `version`. | Register `AC2019Formalism` at formalism-package import, exactly as IIT formalisms register at `pyphi/formalism/__init__.py` import. |
| Moving the three AC registries changes import paths for any user registering custom schemes. | Update all in-repo references; document the moved paths in the changelog fragment. No silent compat shim. |
| Circular imports between `pyphi/formalism/actual_causation/` (needs `Transition` types) and `pyphi/actual.py` (dispatchers call the formalism). | Keep `Transition`/`TransitionSystem` in `actual.py`; the formalism package imports them; `actual.py` imports the registry lazily inside the dispatcher functions (as `_resolve_ac_kwargs` already imports lazily), mirroring `queries.py`. |
| Numeric drift from accidental logic change during the move. | Move algorithm bodies verbatim; the byte-identical paper-fixture acceptance tests are the gate. |

## Open questions

None blocking. The `version` field is added now (one AC formalism today)
deliberately, as the structural mirror and the registration point for the
deferred IIT-4.0-style AC formalism.
