# Cross-formalism SIA / CES surface unification

## Summary

Unify the surface of `SystemIrreducibilityAnalysis` and
`CauseEffectStructure` across IIT 3.0 and IIT 4.0 so that downstream code
sees the same outer container type, the same `__repr__` / `_repr_html_`
template, and the same canonical JSON shape regardless of which formalism
produced the result. Today the two formalisms return structurally
incompatible result objects: IIT 3.0's `sia()` returns an
`IIT3SystemIrreducibilityAnalysis` that owns its cause-effect structure
via top-level `.ces` / `.partitioned_ces` fields, while IIT 4.0's `sia()`
returns a leaner SIA and the outer container is a separate
`CauseEffectStructure(sia, distinctions, relations)`. This spec
restructures IIT 3.0 to adopt IIT 4.0's CES-wraps-SIA topology, then
unifies the surface via shared Protocols, common rendering, and a single
canonical JSON shape.

`AcSystemIrreducibilityAnalysis` (actual causation) participates in the
unification at the rendering and JSON layers via a sibling
`AcSIAInterface` Protocol; AC has no CES analog, so the topology change
does not affect it.

## Background

### Today's class topology

| Function | IIT 3.0 returns | IIT 4.0 returns |
|---|---|---|
| `sia()` | `IIT3SystemIrreducibilityAnalysis` with `.ces`, `.partitioned_ces`, `.substrate` | `SystemIrreducibilityAnalysis` (no CES fields, no substrate) |
| `ces()` | `UnresolvedDistinctions` — a bag of concepts | `CauseEffectStructure(sia, distinctions, relations)` — wraps SIA |

The structural inversion is a legacy of IIT 3.0's paper era. IIT 3.0
paper terminology calls the constellation of concepts the
*cause-effect structure*, but PyPhi 3.0 stored it on the SIA rather than
wrapping the SIA in a CES container. IIT 4.0 generalized the term to
mean "distinctions + relations specified by a system", and the
`CauseEffectStructure` class (in `pyphi/models/ces.py`) reflects that
reading. IIT 3.0's notion fits the same shape with an empty relations
set — there is no theoretical obstacle to adopting the wrapping topology.

### Compute coupling asymmetry

IIT 3.0 and IIT 4.0 differ in *how* SIA phi is computed in a way that
affects what artifacts naturally survive:

- **IIT 3.0:** `sia.phi = ces_distance(unpartitioned_distinctions,
  partitioned_distinctions, system)`. Both the unpartitioned and
  partitioned distinction bags must exist to produce phi. The
  partitioned bag is a primary scientific output — it is the receipt of
  what the MIP did to the CES.
- **IIT 4.0:** `sia.phi` is the `|·|+`-clamp of signed integration on
  the chosen cause/effect spec at the MIP (Eqs. 19-20). The partitioned
  distinction bag is not produced by the SIA computation; the MIP's
  effect on distinctions is recorded indirectly via the
  `system_state` resolution and the chosen partition.

This asymmetry means that `partitioned_distinctions` is intrinsic to
3.0's compute path but absent from 4.0's. The placement decision below
respects this: the field lives on 3.0's SIA as a compute receipt,
*not* on the shared CES container.

### Shared and divergent fields

| Field | 3.0 SIA | 4.0 SIA | Decision |
|---|---|---|---|
| `phi` | ✓ | ✓ | shared, no change |
| `partition` | ✓ | ✓ | shared, no change |
| `current_state` | ✓ | ✓ | shared, no change |
| `node_indices` | ✓ | ✓ | shared, no change |
| `node_labels` | ✓ | ✓ | shared, no change |
| `config` | ✓ | ✓ | shared, no change |
| `ces` (unpartitioned distinctions) | ✓ | ✗ | dropped from 3.0 SIA; lives on the wrapping `CauseEffectStructure.distinctions` |
| `partitioned_ces` → `partitioned_distinctions` | ✓ | ✗ | renamed; stays on 3.0 SIA as 3.0-specific compute receipt |
| `substrate` | ✓ | ✗ | dropped from 3.0 SIA; neither formalism's SIA or CES carries substrate (callers hold it externally) |
| `normalized_phi`, `signed_phi`, `signed_normalized_phi` | ✗ | ✓ | stays on 4.0 SIA only |
| `cause`, `effect` (RIA objects) | ✗ | ✓ | stays on 4.0 SIA only |
| `system_state` | ✗ | ✓ | stays on 4.0 SIA only |
| `intrinsic_differentiation`, `reasons` | ✗ | ✓ | stays on 4.0 SIA only |

`Distinction` is already a shared class (`pyphi/models/distinction.py:290`
aliases `Concept = Distinction`); `ResolvedDistinctions` is shared. No
class-level changes for distinctions.

## Goals

- IIT 3.0's `ces()` returns a `CauseEffectStructure` (the dominant outer
  container), matching IIT 4.0's topology.
- Both formalism SIAs implement a shared `SIAInterface` Protocol covering
  the fields they have in common; formalism-specific fields live below
  the Protocol line.
- Both formalism SIAs render via a common `__repr__` / `_repr_html_`
  template parametrized by formalism, with formalism-specific columns
  appended where applicable.
- Both formalism SIAs (and the shared CES, and AcSIA) serialize to a
  canonical JSON shape with a formalism discriminator carried by the
  existing `CLASS_KEY` metadata.
- Cross-class `__eq__` returns `NotImplemented` rather than silently
  `False`.
- `AcSystemIrreducibilityAnalysis` participates at the rendering and JSON
  layers via `AcSIAInterface`.

## Non-goals

- No msgspec migration. The canonical JSON shape is documented as the
  target P15 should adopt when it migrates the serialization layer; the
  current `pyphi.jsonify` infrastructure is the implementation today.
- No back-compat shims for the dropped fields (per `feedback_no_unnecessary_compat`).
- No semantic changes to phi computation. The restructure moves artifacts
  between containers; it does not change what numerical values get
  produced. Goldens regenerate for shape changes only.
- Macro `MacroSystem` results stay deferred (the macro framework is
  post-2.0 per the Marshall-2024 project; the existing
  `pyphi/macro.py` surface stays untouched).

## Design

### Phase A — Structural restructure

Three changes:

1. **`iit3.ces()` returns `CauseEffectStructure`.** Replace the current
   `UnresolvedDistinctions` return with a `CauseEffectStructure(sia,
   distinctions, relations=NullRelations())`. The SIA is computed (or
   passed in); the distinctions are the resolved-congruence bag (3.0's
   distinction-state ties are vacuously resolved); relations is a new
   minimal `NullRelations` type.

2. **3.0 SIA drops `.ces` and `.substrate`; renames `.partitioned_ces`
   to `.partitioned_distinctions`.** The unpartitioned distinctions
   live on the wrapping `CauseEffectStructure.distinctions`. The
   partitioned bag stays on the SIA as a 3.0-specific compute receipt.
   Substrate is dropped — callers hold it externally (as 4.0 already
   does).

3. **`NullRelations` added to `pyphi/relations.py`.** A minimal
   `Relations` subclass with `sum_phi() == 0`, `num_relations() == 0`,
   and an empty iterable surface. Used by IIT 3.0's CES return; IIT 4.0
   continues to use `ConcreteRelations` / `AnalyticalRelations`.

#### Side effects of the Phase A restructure

- `OrderableByPhi.unorderable_unless_eq=["substrate"]` on
  `IIT3SystemIrreducibilityAnalysis` becomes vacuous (substrate field
  removed). Remove the entry. If the substrate equality guard is
  load-bearing for any current test, replace it with `node_indices` (a
  weaker, substrate-less proxy) and document the looser semantics.

- 3.0 SIA's `to_json` / `from_json` field set shrinks. Existing
  IIT 3.0 SIA JSON fixtures (the 4 IIT 3.0 EMD goldens plus any inline
  fixtures under `test/data/sia/`) regenerate against the new shape.

- `pyphi.jsonify._loadable_models()` is unchanged in class membership;
  the changes are field-set on `IIT3SystemIrreducibilityAnalysis`.

### Phase B — Field rename audit

After Phase A, the remaining divergence between 3.0 and 4.0 SIA fields
is:

- `current_state` — matched, no change.
- `node_indices`, `node_labels` — matched, no change.
- `config` — matched, no change.

So Phase B is mostly a no-op on field renames *between formalisms*. The
work in this phase is:

- Audit internal use of `_sia_attributes` (currently a module-level
  constant in `pyphi/models/sia.py`) and the corresponding 4.0
  `_sia_attributes` ClassVar — confirm both lists are consistent with
  the new shared shape.

- Audit AcSIA fields against `AcSIAInterface` (which will be defined in
  Phase C). AcSIA already carries `phi` (under the name `alpha`),
  `direction`, `account`, `partitioned_account`, `partition`,
  `before_state`, `after_state`, `node_indices`, `cause_indices`,
  `effect_indices`, `node_labels`, `config`. The Phase B question is
  whether `alpha` should rename to `phi` for Protocol parity, or stay
  as `alpha` per the actual-causation paper's terminology.

  Decision: keep `alpha`. The Albantakis 2019 paper uses α for actual
  causation strength; `AcSIAInterface` declares its measure attribute
  as `alpha` rather than aligning with `phi`. The Protocol surface
  takes the field name from the formalism's own paper terminology.

Phase B is the smallest of the three; folding it into Phase C is
acceptable if implementation finds it more natural.

### Phase C — Protocols + common repr + canonical JSON + __eq__ audit

#### Protocols (`pyphi/models/protocols.py`)

```python
@runtime_checkable
class SIAInterface(Protocol):
    phi: PyPhiFloat | DistanceResult
    partition: _PartitionBase
    current_state: tuple[int, ...] | None
    node_indices: tuple[int, ...] | None
    node_labels: NodeLabels | None
    config: ConfigSnapshot

    def order_by(self) -> Any: ...
    def __bool__(self) -> bool: ...

@runtime_checkable
class CauseEffectStructureInterface(Protocol):
    sia: SIAInterface
    distinctions: ResolvedDistinctions
    relations: Relations
    config: ConfigSnapshot

@runtime_checkable
class AcSIAInterface(Protocol):
    alpha: PyPhiFloat | DistanceResult
    direction: Direction
    account: Account
    partitioned_account: Account
    partition: _PartitionBase
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    node_indices: tuple[int, ...] | None
    cause_indices: tuple[int, ...] | None
    effect_indices: tuple[int, ...] | None
    node_labels: NodeLabels | None
    config: ConfigSnapshot
```

Formalism-specific fields (3.0's `partitioned_distinctions`, 4.0's
`signed_phi` / `normalized_phi` / `cause` / `effect` / etc.) are not in
the Protocol and remain accessible via direct attribute access on the
concrete classes (or `isinstance` dispatch).

#### Common `__repr__` and `_repr_html_`

Add formalism-parametric helpers to `pyphi/models/fmt.py`:

```python
def fmt_sia_columns(sia: SIAInterface) -> list[tuple[str, Any]]:
    """Shared columns for any SIA: System, Current state, phi, Partition."""
    ...

def fmt_ces_columns(ces: CauseEffectStructureInterface) -> list[tuple[str, Any]]:
    """Shared columns for any CES: Φ (=sia.phi), #(distinctions), Σ φ_d, #(relations), Σ φ_r."""
    ...
```

Both `IIT3SystemIrreducibilityAnalysis.__repr__` and
`SystemIrreducibilityAnalysis._repr_columns` route through
`fmt_sia_columns` for the shared portion and extend with formalism-
specific columns (4.0 adds Normalized φ_s, Int. diff. CAUSE/EFFECT, etc.;
3.0 adds nothing). `CauseEffectStructure.__repr__` routes through
`fmt_ces_columns` and works identically for both formalisms.

`_repr_html_` is added in this phase (currently neither SIA implements
it) using the same column structure rendered as HTML. The macro /
Jupyter rendering goal from P15 benefits from this — Phase C lays the
foundation.

#### Canonical JSON shape

The discriminator is the existing `CLASS_KEY` metadata added by
`jsonify._push_metadata` — no new field is introduced. The canonical
shape for each result type:

**`IIT3SystemIrreducibilityAnalysis`**:
```json
{
  "__class__": "IIT3SystemIrreducibilityAnalysis",
  "__version__": "...",
  "__id__": ...,
  "phi": ...,
  "partition": ...,
  "partitioned_distinctions": ...,
  "current_state": ...,
  "node_indices": ...,
  "node_labels": ...,
  "config": ...
}
```

**`SystemIrreducibilityAnalysis` (IIT 4.0)**: existing shape with
`__class__: "SystemIrreducibilityAnalysis"`.

**`CauseEffectStructure`**:
```json
{
  "__class__": "CauseEffectStructure",
  "__version__": "...",
  "__id__": ...,
  "sia": { ... },
  "distinctions": ...,
  "relations": ...,
  "config": ...
}
```

Both formalisms produce the same shape; the SIA inside is the formalism-
specific one (carried by its own `__class__`).

**`AcSystemIrreducibilityAnalysis`**: existing shape, audited for
consistency with `AcSIAInterface`.

P15's msgspec migration adopts these shapes via tagged unions keyed on
`__class__`.

#### Cross-class `__eq__`

Today both `IIT3SystemIrreducibilityAnalysis.__eq__` and
`SystemIrreducibilityAnalysis.__eq__` route through
`cmp.general_eq(self, other, attrs)`. When `other` is a different class,
the attribute comparison may silently return `False` (or worse, may
raise on missing attributes). Audit both `__eq__` paths and have them
return `NotImplemented` when `type(other) is not type(self)`. Python
then falls back to identity comparison, which evaluates `False` — same
end-user behavior, but explicit at the Protocol level.

### AcSIA participation

AcSIA gets `_repr_columns` (currently the class falls back to
`fmt.make_repr` and `fmt.fmt_ac_sia`) routed through a sibling
`fmt_ac_sia_columns` helper, JSON shape audit, and a sibling
`AcSIAInterface` Protocol. AC has no CES wrapper analog and stays out
of `CauseEffectStructureInterface`.

## File map

| File | Phase A | Phase B | Phase C |
|---|---|---|---|
| `pyphi/models/sia.py` | Drop `ces` field; drop `substrate` field; rename `partitioned_ces` → `partitioned_distinctions`; update `_sia_attributes`, `to_json`, `from_json`; drop `unorderable_unless_eq` substrate entry | (cleanup if any) | Adopt `fmt_sia_columns`-based `__repr__`; add `_repr_html_`; `__eq__` returns `NotImplemented` on type mismatch |
| `pyphi/models/ces.py` | No Phase A changes — already correct shape | (cleanup if any) | Adopt `fmt_ces_columns`-based `__repr__`; add `_repr_html_`; canonical `to_json` |
| `pyphi/models/actual_causation.py` | No Phase A changes | Audit field set vs `AcSIAInterface` | Adopt `fmt_ac_sia_columns`-based `__repr__`; add `_repr_html_`; canonical `to_json`; `__eq__` returns `NotImplemented` on type mismatch |
| `pyphi/models/protocols.py` (new) | — | — | Define `SIAInterface`, `CauseEffectStructureInterface`, `AcSIAInterface` |
| `pyphi/models/fmt.py` | — | — | Add `fmt_sia_columns`, `fmt_ces_columns`, `fmt_ac_sia_columns` helpers; HTML render helpers |
| `pyphi/models/__init__.py` | — | — | Export new Protocols |
| `pyphi/formalism/iit3/__init__.py` | `ces()` returns `CauseEffectStructure(...)`; `_sia()` no longer stores unpartitioned distinctions on the SIA but stores `partitioned_distinctions`; update `_null_sia`; remove substrate from SIA construction | — | — |
| `pyphi/formalism/iit4/__init__.py` | No Phase A changes | — | `_repr_columns` routes through `fmt_sia_columns` plus 4.0-specific columns |
| `pyphi/relations.py` | Add `NullRelations` (empty, `sum_phi()==0`, `num_relations()==0`, empty iterator) | — | — |
| `pyphi/jsonify.py` | Audit `_loadable_models()` — `NullRelations` may need entry | — | Document canonical shape (in docstring) |
| `test/data/golden/v1/basic_iit3_emd*.json`, `xor_iit3_emd*.json` | Regenerate for new CES return shape | — | Regenerate for canonical JSON shape changes |
| `test/data/sia/*.json` | Regenerate IIT 3.0 SIA fixtures for new field set | — | — |
| `test/test_big_phi.py`, `test/test_iit3_*.py`, `test/test_complexes.py`, etc. | Update tests reading `sia.ces` / `sia.partitioned_ces` / `sia.substrate` | — | Add Protocol conformance tests; round-trip JSON tests; cross-formalism `__eq__` tests; common-repr tests |
| `changelog.d/*.fix.md` / `*.change.md` | One fragment per phase | — | — |

## Verification

After **Phase A:**

```bash
uv run pytest test/test_big_phi.py test/test_complexes.py test/test_measures_ces.py test/test_golden_regression.py -x -q
uv run pyright pyphi/models/sia.py pyphi/formalism/iit3/
uv run ruff check pyphi/ test/
```

Expected: tests pass (after fixture regen and call-site updates),
pyright clean on modified files. The 4 IIT 3.0 EMD goldens regenerate
with the new CES shape; the `test_canonical_iit3_preset_is_exercised`
guardrail remains green.

After **Phase B:**

```bash
uv run pytest test/test_actual.py -x -q
uv run pyright pyphi/models/actual_causation.py
```

Audit-only phase; tests should remain green throughout.

After **Phase C:**

```bash
uv run pytest test/ -m "not slow" -x -q
uv run pytest test/ --slow -q  # background
uv run pytest test/test_golden_regression.py -q   # all 17 goldens regenerate
uv run pyright pyphi/
uv run ruff check pyphi/ test/
```

Expected: 1256+/0 failures fast lane, 1264+/0 failures slow lane,
17/17 goldens green, pyright 0 errors, ruff clean.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase A regen invalidates 4 IIT 3.0 EMD goldens; downstream consumers reading the old shape break silently | High (expected) | Goldens regenerate with `--regenerate-golden`; shape change reviewed manually in one commit |
| `unorderable_unless_eq=["substrate"]` removal makes cross-substrate phi comparisons valid; some test may have relied on the guard | Low-medium | Audit `OrderableByPhi` callers in Phase A; replace with `node_indices` proxy if a guard is needed |
| `NullRelations` doesn't exist today; need to add | Low | Small, contained addition — `Relations` base class permits it |
| Protocol-conformance type errors surface latent inconsistencies | Medium | Per saved memory `feedback_dont_give_up_on_architectural_refactors` — diagnose latent issues rather than relaxing Protocols |
| JSON shape change forces all 17 goldens to regenerate at Phase C | High (expected) | Document canonical shape in spec before regen; review diff in one commit; guardrail tests stay green |
| msgspec migration (P15) constrains canonical JSON shape | Low | Spec calls out target shape; P15 picks msgspec types matching the shape |
| 3.0 callers reading `sia.ces` or `sia.partitioned_ces` (outside the repo) break | Medium | 2.0 already accepts breaking changes; documented in changelog fragments; migration note in `docs/migration-2.0.md` (planned for P15) |
| AcSIA's `alpha` not renaming to `phi` is inconsistent with the unification goal | Low | Decision noted: AC's paper terminology is α; Protocol surface declares `alpha`. Cross-analysis code that wants a single "magnitude" attribute uses `isinstance` dispatch. |

## Effort

- Phase A: 1-2 days (3.0 CES return type, SIA field changes, fixture regen, call-site updates).
- Phase B: 0.5-1 day (audit-only; mostly no-op).
- Phase C: 2-3 days (Protocols, repr templates, JSON shape, `__eq__` audit, AcSIA folding, goldens regen, conformance tests).
- Total: 4-6 days.

## Sequencing

P11.87 lands after P11.95a + P11.95b (already done) and is independent
of P11.95c / P11.95d. The work pairs naturally with P15 (jsonify
retirement) but does not require P15 to land first — the canonical JSON
shape is implementable via the current `pyphi.jsonify` infrastructure;
P15 later adopts msgspec types matching the shape.

## Open questions

- **AcSIA renaming `alpha` → `phi`?** Decision in spec: keep `alpha` per
  the actual-causation paper. Open to revisiting during Phase C if
  Protocol-surface unification across `SIAInterface` and `AcSIAInterface`
  proves clumsy in practice.
- **Should `_repr_html_` ship in this project or in P15?** Spec says
  here; ship it as part of Phase C since the column infrastructure is
  built then. If timing slips, defer the HTML render to P15 and ship
  the text `__repr__` only.
- **Substrate-less `OrderableByPhi.unorderable_unless_eq`.** Phase A
  audit may reveal a test or downstream consumer that relied on the
  guard. If so, the spec's `node_indices` replacement is the fallback;
  if `node_indices` is insufficient, document the looser semantics in
  the changelog and update the affected test.
