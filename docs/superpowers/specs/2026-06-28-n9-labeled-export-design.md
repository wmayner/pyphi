# N9 — Labeled-export (`to_pandas`) full coverage — design

## Goal

Give every user-facing `Displayable` result type a `to_pandas()` method, so the
labeled-export story is complete and uniform, and lock it with a
coverage-invariant test. This closes the P14d follow-on and removes the reason
`sweep` / `analyze` had to hand-roll private row extractors.

## Audit (the starting point)

Cross-referencing the `Displayable` subclass set against `to_pandas` coverage:

**Already export** (P14d + B8/B15): `RepertoireIrreducibilityAnalysis`,
`MaximallyIrreducibleCauseOrEffect` (+ Cause/Effect), `Distinction`,
`Distinctions` (+ Resolved/Unresolved), `StateSpecification`,
`SystemStateSpecification`, `Explanation`, `ResultDiff`, `Analysis`.

**Missing** (the gaps N9 fills): the SIAs (`SystemIrreducibilityAnalysis` 4.0,
`IIT3SystemIrreducibilityAnalysis` 3.0, `Null*` variants),
`CauseEffectStructure` / `PhiFold`, the relations family (`Relations`,
`Relation`, `RelationFace`, `Concrete`/`Analytical`/`Null` relations),
`Complex` / `ExcludedCandidate`, the actual-causation family
(`AcSystemIrreducibilityAnalysis`, `AcRepertoireIrreducibilityAnalysis`,
`CausalLink`, `Account` / `DirectedAccount`), the structural types (`System`,
`Substrate`, `FactoredTPM`, `JointTPM`), partitions / cuts, and `UnitState`.

**xarray:** confirmed TPM-only by design (`FactoredTPM` / `JointTPM` carry
`to_xarray`; the `to_xarray` result cube and P12c labeled coords were
deliberately deferred / dropped). N9 is a pandas-coverage story; xarray is out
of scope.

## Convention (unchanged from P14d)

`pyphi/models/pandas.py` already defines the contract:

- `ToPandasMixin.to_pandas()` → delegates to `_to_pandas()`.
- Default `_to_pandas()` builds a `Series` from `_pandas_record()` (a
  field→value mapping). **Scalar-record types** implement `_pandas_record()`.
- **Collection / structural types** override `_to_pandas()` directly to return a
  `DataFrame`.
- Pure builders (`record_to_series`, `records_to_frame`, `state_multiindex`)
  own all pandas construction over the `NodeLabels` utilities.

N9 extends coverage by adding `ToPandasMixin` to each uncovered **base** class
and implementing the appropriate hook. Subclasses inherit.

## Per-type export shape

Scalar-record → `Series`; collection / structural → `DataFrame`.

| Type | Shape | Record / rows |
| --- | --- | --- |
| `SystemIrreducibilityAnalysis` (4.0) | Series | `phi`, `normalized_phi`, `system`, `current_state`, `partition`, `n_distinctions` |
| `IIT3SystemIrreducibilityAnalysis` | Series | `phi`, `system`, `current_state`, `partition` |
| `Complex` | Series | the wrapped SIA's record + `is_maximal` |
| `ExcludedCandidate` | Series | `node_indices`, `phi` |
| `CauseEffectStructure` / `PhiFold` | DataFrame | one row per distinction (delegates to `.distinctions`) |
| `Relations` (+ Concrete/Analytical) | DataFrame | one row per relation: `relata`, `phi`, `degree` |
| `Relation` / `RelationFace` | Series | `relata`, `phi`, `degree` |
| `AcSystemIrreducibilityAnalysis` | Series | `alpha`, `system`, `state`, `partition` |
| `AcRepertoireIrreducibilityAnalysis` / `CausalLink` | Series | `alpha`, `mechanism`, `purview`, `direction` |
| `Account` / `DirectedAccount` | DataFrame | one row per `CausalLink` |
| `System` | DataFrame | per-unit: `node`, `label`, `state` |
| `Substrate` | DataFrame | its TPM matrix (delegates to `FactoredTPM`) |
| `FactoredTPM` / `JointTPM` | DataFrame | wide labeled state-by-node matrix (reuses the `state_multiindex` pattern that `TriggeredTPM.to_pandas` already uses) |
| partitions / cuts (`_PartitionBase`, `EdgeCut`, …) | DataFrame | labeled `cut_matrix(n)` grid (from-node × to-node, 1 = severed) |
| `UnitState` | Series | `unit`, `state` |

`Null*` variants (`NullSystemIrreducibilityAnalysis`,
`NullCauseEffectStructure`, `NullRelations`, `NullCut`) inherit their parent's
hook; where a null carries no records (e.g. empty relations) the collection
exporter naturally yields an empty, correctly-typed frame.

Degenerate frames are acceptable: a partition's cut grid or a `UnitState`'s
two-field Series carry little analytic payload, but completing coverage makes
"every result has both a card and a `to_pandas`" a clean, exception-free
invariant for the P15 freeze.

## Enforcement

A coverage-invariant test (mirroring B21's `Displayable`-coverage test) walks
every `Displayable` subclass reachable from the result modules, constructs a
representative instance (reusing existing example fixtures where possible),
calls `to_pandas()`, and asserts the result is a `Series` or `DataFrame` and
does not raise `NotImplementedError`. This is the audit's closing deliverable:
it proves full coverage and prevents a future result type from silently
shipping without an export.

## Deliberately unchanged

- **`sweep` / `analyze` summary rows.** `sweep`'s `_row_sia` / `_row_ces` and
  `Analysis.to_pandas` produce intentionally compact *summary* rows for the
  tidy batch table and the bundle. They are not folded onto the new canonical
  `SIA.to_pandas` / `CES.to_pandas`, to keep `sweep`'s column schema stable.
- **xarray.** TPM-only, out of scope (above).

## Testing

Per-type unit tests assert the shape (`Series` vs `DataFrame`), the expected
columns / index, and a representative value (e.g. `sia.to_pandas()["phi"]`
equals `float(sia.phi)`), grouped by the task clusters below. Plus the single
coverage-invariant test.

## Decomposition

1. **SIA family** — `SystemIrreducibilityAnalysis` (4.0),
   `IIT3SystemIrreducibilityAnalysis` (3.0), `Complex`, `ExcludedCandidate`.
2. **CES + Relations** — `CauseEffectStructure` / `PhiFold`, `Relations` /
   `Relation` / `RelationFace`.
3. **AC family** — `AcSystemIrreducibilityAnalysis`,
   `AcRepertoireIrreducibilityAnalysis`, `CausalLink`, `Account` /
   `DirectedAccount`.
4. **Structural** — `System`, `Substrate`, `FactoredTPM`, `JointTPM`.
5. **Partitions / cuts + `UnitState`**.
6. **Coverage-invariant test + changelog + ROADMAP** (flip N9 to ✅).
