# Actual Causation reads its own mechanism-partition scheme

**Status:** design. Fixes a confirmed correctness bug from the 2026-07-13
whole-library review (Wave 1).

## 0. Summary

Actual-causation partition enumeration silently reads the *IIT* config field
`config.formalism.iit.mechanism_partition_scheme`, so AC α values change with an
unrelated IIT setting, while the dedicated
`ActualCausationConfig.mechanism_partition_scheme` field (default
`JOINT_PARTITION_ALL`) is never read. Under any IIT 3.0 pin (which sets the IIT
field to `JOINT_BIPARTITION`), AC enumerates partitions the 2019 paper forbids
for first-order occurrences, deflating α to 0 where the paper defines α = ρ.

The fix routes AC through its own field, mirroring how AC already threads its
other three schemes (`alpha_measure`, `partitioned_repertoire_scheme`,
`background_scheme`, `alpha_aggregation`). The field remains freely
configurable — users may investigate partition variants deliberately — but the
choice is now an explicit AC setting rather than a value silently inherited
from an unrelated formalism.

## 1. The bug

`pyphi/formalism/actual_causation/compute.py` — `_find_mip` and `_get_partitions`
call `pyphi.partition.mechanism_partitions(mechanism, purview, node_labels)`,
which dispatches on `config.formalism.iit.mechanism_partition_scheme`
(`partition.py:473`) and takes no scheme argument. So:

- AC partition enumeration is governed by the **IIT** field.
- `config.formalism.actual_causation.mechanism_partition_scheme` (defined at
  `formalism.py:148`, default `JOINT_PARTITION_ALL`) has **no reader** — it is
  dead.
- The `iit3` preset sets `iit.mechanism_partition_scheme="JOINT_BIPARTITION"`,
  so every AC computation pinned with `IIT_3_CONFIG` silently uses the IIT
  bipartition family.

**Why this is wrong (paper-grounded).** Albantakis et al. (2019), *What Caused
What?*, Eq. 7 defines a partition ψ of an occurrence into `m` parts as a split
of the mechanism with the constrained purview distributed across the parts
(any part's purview may be empty). Fig 3B (p. 9) and p. 13 add the m=1 rule:
for a first-order occurrence the *only* permitted partition is the full cut
(occurrence completely severed from its purview), giving α = ρ. The paper's
family is therefore *all* partitions of the occurrence with the m=1 non-full-cut
cases excluded — which is precisely what `JOINT_PARTITION_ALL`
(`all_joint_partitions`) implements, via its `partition.py:663` guard that skips
partitions leaving the whole mechanism in one part with a non-empty purview.
`JOINT_BIPARTITION` (m=2) and `WEDGE_TRIPARTITION` (m=3) are IIT-3.0 constructs
that restrict to a fixed part count; the 2019 paper names neither. Under
`JOINT_BIPARTITION`, first-order occurrences over a multi-unit purview get
purview-splitting m=1 partitions with the occurrence intact — the "Not
permitted" case — and minimizing over them drives α to 0.

**Verified.** On the 4-node OR/AND `actual_causation_substrate`, cause link
`_find_mip(t, CAUSE, (0,), (0,1))`: α = 0.415037 under `JOINT_PARTITION_ALL`
(= ρ, paper-correct) vs α = 0.0 under `JOINT_BIPARTITION`.

## 2. The fix

### 2.1 Thread AC's own field

`_resolve_ac_measures` (`formalism.py`) already resolves the AC alpha measure and
the other three AC schemes into callables read from
`config.formalism.actual_causation`. Extend it to also resolve
`mechanism_partition_scheme`:

```python
mp_name = (
    mechanism_partition_scheme_name
    if mechanism_partition_scheme_name is not None
    else ac.mechanism_partition_scheme
)
...
"mechanism_partitions": partition_types[mp_name],
```

`partition_types[mp_name]` is the generator callable with signature
`(mechanism, purview, node_labels) -> Iterable[partition]` — the same signature
the global `mechanism_partitions()` invokes. An unregistered name raises
`KeyError` from the registry (the only validation needed; no allowlist).

Thread the resolved callable through the AC compute entry points
(`_account`, `_directed_account`, `_sia`, `_find_mip`, `_find_causal_link`) as a
`mechanism_partitions` keyword argument, exactly as `partitioned_repertoire_scheme`
is threaded today. `_find_mip` and `_get_partitions` call the passed callable
instead of the global `mechanism_partitions()`.

### 2.2 IIT paths untouched

`pyphi.partition.mechanism_partitions()` keeps reading the IIT field and keeps
its current signature; IIT distinction/MICE computation is unchanged. Only AC
stops routing through it.

### 2.3 Configurable, informed — no restriction

The AC field accepts any registered mechanism-partition scheme; selecting a
non-default scheme is a deliberate variant investigation. The
`ActualCausationConfig.mechanism_partition_scheme` field docstring documents the
semantics so the choice is informed: `JOINT_PARTITION_ALL` is the 2019-paper
family (Eq. 7 + Fig 3B); other schemes deviate — notably `JOINT_BIPARTITION`
admits m=1 partitions the paper forbids for first-order occurrences, so it
produces α below the published values. No config constraint or allowlist is
added.

## 3. Tests and goldens

Fixing the leak changes AC results that were computed under the leaked scheme,
and flips test expectations that codified the bug. All AC tests live under
`test/test_actual.py` (the `TestActualCausationIIT30` class pins `IIT_3_CONFIG`);
AC goldens live in `test/golden/zoo.py` / `test/golden/compute.py`.

### 3.1 New guarantees (regression tests)

- **The IIT field no longer affects AC; the AC field does.** Set
  `iit.mechanism_partition_scheme` and
  `actual_causation.mechanism_partition_scheme` to different values and assert AC
  α tracks the AC field, not the IIT field.
- **First-order occurrence is paper-correct by default.** The §1 repro: under the
  AC default `JOINT_PARTITION_ALL`, `_find_mip(t, CAUSE, (0,), (0,1)).alpha` =
  ρ = 0.415037, not 0.

### 3.2 Expectation changes (bug was codified)

- `test_sia_cause_direction` asserts `alpha == 0.0` under the inherited
  `JOINT_BIPARTITION`. After the fix, AC uses its own default
  `JOINT_PARTITION_ALL`, so the cause α is the paper value. Re-assert the
  recomputed value. A separate test may set the *AC* field to `JOINT_BIPARTITION`
  explicitly to retain coverage of that regime as a deliberate variant.
- `test_prevention` overrides the *IIT* field to `WEDGE_TRIPARTITION`. Repoint the
  override to the *AC* field; `WEDGE_TRIPARTITION` remains a legitimate
  intentional variant. Re-verify the asserted values under the corrected routing.
- The remaining `TestActualCausationIIT30` tests and AC goldens recompute under
  `JOINT_PARTITION_ALL`. Regenerate affected goldens deliberately, each documented
  as previously computed under the leaked IIT scheme.

### 3.3 Verification

`uv run pytest test/test_actual.py` and the golden regression suite green;
`uv run pytest` (no path argument) green for the doctest sweep, since the AC
config field docstring changes.

## 4. Scope boundaries

- **AC only.** IIT and IIT 3.0 mechanism partitioning are untouched.
- **No new partition schemes.** The fix routes existing schemes correctly; it does
  not add or modify any `partition_types` entry.
- **No config-constraint machinery.** The field stays a free choice; the registry
  lookup is the only validation.
- **Not in scope:** the other Wave 1 remainders (macro TPM bugs, factored-backend
  select crash) are separate fixes.
