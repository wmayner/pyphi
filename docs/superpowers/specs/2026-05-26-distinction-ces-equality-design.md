# Per-class `__eq__` migration for IIT result objects

**Date:** 2026-05-26
**Status:** Design

## Problem

The precision-aware comparator project (commit `1bf0bd40`) made `pyphi.models.cmp.numpy_aware_eq` and the production `__eq__` path through `general_eq` tolerance-aware up to `EQUALITY_TOLERANCE = 1e-13`. That fix flows into result objects whose `__eq__` is *implemented as* `cmp.general_eq(self, other, _attribute_list)` — `SystemIrreducibilityAnalysis`, `AcSystemIrreducibilityAnalysis`, `AcRepertoireIrreducibilityAnalysis`, `RepertoireIrreducibilityAnalysis`, `StateSpecification`.

Three result-object classes implement `__eq__` directly without routing through `general_eq`, and so do NOT benefit:

- `Distinction.__eq__` (`pyphi/models/distinction.py:163`) — uses raw `==` on `phi` and `np.array_equal` on repertoires.
- `Distinctions.__eq__` (`pyphi/models/distinctions.py:123`) — delegates to `self.concepts == other.concepts`, which fans out to `Distinction.__eq__`.
- `CauseEffectStructure.__eq__` (`pyphi/models/ces.py:79`) — checks `self.sia == other.sia` (precision-aware) AND `self.distinctions == other.distinctions` (NOT precision-aware) AND `self.relations == other.relations`.

A fourth class, `Relation` (`pyphi/relations.py:126`), inherits `frozenset.__eq__` without an explicit override; line 182 carries a `TODO(4.0) need to also implement __eq__ here`. Element equality cascades through `Distinction.__eq__` (via frozenset's hash-bucket-then-eq lookup), so this gap is downstream of the Distinction gap.

The end-state is to give every result-object class an explicit per-class `__eq__` method that spells out its tolerance-aware and strict-equality attribute checks, replacing the `general_eq` attribute-list pattern. This:

- Closes the precision-aware-comparator gap for `Distinction`, `Distinctions`, `CauseEffectStructure`, and `Relation`.
- Replaces an EMD-era convention (the attribute-list-as-ClassVar pattern, plus the `general_eq` dispatcher) with a Pythonic per-class pattern.
- Closes the line-182 `TODO` on `Relation.__eq__`.

## Approach

### Per-class `__eq__` template

Every affected class gets an explicit method:

```python
def __eq__(self, other: object) -> bool:
    if not isinstance(other, MyClass):
        return NotImplemented
    if self.struct_attr1 != other.struct_attr1:
        return False
    if self.struct_attr2 != other.struct_attr2:
        return False
    if not utils.eq(self.phi, other.phi):
        return False
    if not numpy_aware_eq(self.array_attr, other.array_attr):
        return False
    return True
```

- Strict-equality attributes use `==` (integer indices, tuples of indices, direction enums, etc.).
- Float-valued attributes (``phi``, ``alpha``) use ``utils.eq`` (which respects ``config.numerics.precision``).
- Array-valued attributes (repertoires, distributions) use ``numpy_aware_eq`` (which uses ``EQUALITY_TOLERANCE`` directly).
- Cross-type comparison returns ``NotImplemented`` (per ``feedback_no_unnecessary_compat`` and Python convention; lets Python fall back to the right-hand side).

The method is one block per attribute, each line naming the attribute and the comparison kind. No introspection, no attribute-list dispatch.

### `Distinction.__hash__`

`Distinction` is the only result object that has a meaningful `__hash__` (it's used in `set(self.distinctions)` at `relations.py:99`, in `compositional_state.py:275`, and in test assertions like `set(c) == set(iit3.ces(s).distinctions)`).

Tolerance-aware `__eq__` constrains `__hash__`: Python requires `a == b → hash(a) == hash(b)`. The hash must therefore depend ONLY on attributes that `__eq__` checks with strict equality. For `Distinction`, those are:

- `mechanism`
- `mechanism_state`
- `cause_purview`
- `effect_purview`

(All integer tuples / state-spec objects.)

```python
def __hash__(self) -> int:
    return hash((
        self.mechanism,
        self.mechanism_state,
        self.cause_purview,
        self.effect_purview,
    ))
```

This is provably contract-correct: if two `Distinction` instances satisfy the tolerance-aware `__eq__`, they share these four attributes by exact ``==`` (from `__eq__`'s structural-attr clauses), so their hashes match. The hash drops `phi` and the repertoires because those are tolerance-checked in `__eq__` — including them would risk contract violations at quantization boundaries.

Practical collision rate: within a single substrate, `(mechanism, mechanism_state, cause_purview, effect_purview)` is essentially 1:1 with `(phi, repertoires)` by IIT's construction. Cross-substrate analyses of distinctions may see more hash collisions, but Python's contract only restricts hashes for *equal* items — collisions for non-equal items are permitted, and set/dict lookup falls back to `__eq__` comparison on tuple-typed structural attrs (fast).

### `Orderable` / `OrderableByPhi` refactor

Currently, `Orderable.__lt__` (`pyphi/models/cmp.py:67`) uses `general_eq(self, other, self.unorderable_unless_eq)` to enforce "you can't order across substrates" guards. `unorderable_unless_eq` is a `ClassVar[list[str]]` of attribute names — same attribute-list pattern as the old `__eq__` callers.

Replace with an abstract method:

```python
class Orderable:
    def is_orderable_with(self, other: object) -> bool:
        """Whether ``self`` and ``other`` are mutually orderable.

        Default: any two instances are orderable. Override in subclasses
        that need cross-instance guards (e.g., "distinctions from different
        substrates can't be ordered").
        """
        return True

    def __lt__(self, other: object) -> bool:
        if not self.is_orderable_with(other):
            raise TypeError(
                f"Unorderable: {type(self).__name__} instances do not satisfy "
                f"the orderability constraint of this type."
            )
        return self.order_by() < other.order_by()
```

Subclasses that need cross-instance guards override `is_orderable_with`. Most subclasses use the default and need no override. The `unorderable_unless_eq: ClassVar[list[str]] = []` attribute is removed from the base class.

### `general_eq` deletion

Once all callers are migrated, `general_eq` is unused. Delete it from `pyphi/models/cmp.py`. Also delete the `sametype` decorator if it has no remaining callers (audit during plan-time).

`numpy_aware_eq` is preserved — it's the leaf-array tolerance comparator used by every per-class `__eq__` method and by `test/test_golden_regression.py`. `EQUALITY_TOLERANCE` is preserved for the same reason.

### Affected classes (final list)

| Class | File | Strict attrs | Tolerance attrs | Notes |
|---|---|---|---|---|
| `SystemIrreducibilityAnalysis` (3.0) | `pyphi/models/sia.py` | Per `_sia_attributes` at line 20 | `phi`, `partition_phi` | Plan-time: enumerate from `_sia_attributes` |
| `SystemIrreducibilityAnalysis` (4.0) | `pyphi/formalism/iit4/__init__.py` | Per `_sia_attributes` at line 184 | `phi` | Plan-time enum |
| `AcRepertoireIrreducibilityAnalysis` | `pyphi/models/actual_causation.py` | Per `_acria_attributes_for_eq` at line 33 | `alpha` | Plan-time enum |
| `AcSystemIrreducibilityAnalysis` | `pyphi/models/actual_causation.py` | Per `_ac_sia_attributes` at line 424 | `alpha` | Plan-time enum |
| `RepertoireIrreducibilityAnalysis` | `pyphi/models/ria.py` | Per the `attrs` list at line 355 | `phi` | Plan-time: enumerate from `attrs` at the call site |
| `StateSpecification` | `pyphi/models/state_specification.py` | Per the call at line 96 | (none expected; verify at plan-time) | Plan-time: enumerate from the call site |
| `Distinction` | `pyphi/models/distinction.py` | `mechanism`, `mechanism_state`, `cause_purview`, `effect_purview` | `phi`, `cause_repertoire`, `effect_repertoire` | Also new `__hash__` |
| `Distinctions` | `pyphi/models/distinctions.py` | `concepts` (tuple — cascades) | (none directly) | Explicit method for clarity |
| `CauseEffectStructure` | `pyphi/models/ces.py` | `sia`, `distinctions`, `relations` (all cascade) | (none directly) | Explicit method for clarity |
| `Relation` | `pyphi/relations.py` | `frozenset(distinctions)` (cascades through `Distinction.__eq__`) | (none directly) | Explicit method closes line-182 `TODO` |

Plan-time enumeration of strict vs tolerance attrs for each currently-`general_eq`-routed class is a small audit task in the plan's pre-flight step.

## Testing

### New unit tests

For each class, add tests in `test/test_models.py` (or per-class test files where applicable) covering:

- **Within-tolerance equality**: construct two instances differing by ~1e-15 in a float-valued attr; assert `__eq__` returns True.
- **Outside-tolerance inequality**: differing by ~1e-3 (representative of a real math regression); assert `__eq__` returns False.
- **Cross-type returns NotImplemented**: assert `eq(self, "wrong type")` doesn't crash and falls through to Python's `False`.
- **Hash contract** (Distinction only): assert `d1 == d2 → hash(d1) == hash(d2)` for tolerance-equivalent pairs.

The 12 existing `numpy_aware_eq_*` and the existing `general_eq_*` tests in `test/test_models.py`: keep the `numpy_aware_eq_*` (still the leaf comparator); remove the `general_eq_*` (function being deleted).

### Goldens

The 25 byte-identical fixtures in `test/test_golden_regression.py` should remain byte-identical. The migration changes `__eq__` semantics but not the underlying computed values — fixture data flows through `_compare` in the test, which uses `EQUALITY_TOLERANCE` directly (not `__eq__`). Verify after migration.

### Full-suite check

Run `uv run pytest` (no path argument, includes doctests). Surface any tests that flip from passing to failing OR from failing to passing. Per saved memory `feedback_dont_give_up_on_architectural_refactors`, a test silently flipping under the loosened comparator may reveal a latent inconsistency — investigate before reverting.

Tests that might flip:
- Assertions of `assert obj1 != obj2` where `obj1` and `obj2` were within 1e-13 (probably accidental — these should be re-examined).
- Tests of set/dict behavior on `Distinction` collections — the new `__hash__` may dedupe more aggressively when distinctions share structural attrs but differ in phi/repertoires beyond a single substrate's natural mapping.

## Out of scope

- **Changing `numpy_aware_eq` further.** It's now the leaf comparator; behavior preserved.
- **Changing `utils.eq`.** Still consumes `config.numerics.precision`. The post-IIT-3.0 retirement is a separate ROADMAP item.
- **Adding `__hash__` to result-object classes that currently lack one.** Only `Distinction` has a meaningful `__hash__` used by real code; others remain unhashable or use `object.__hash__`.
- **Refactoring `OrderableByPhi.order_by`.** Unchanged; only `__lt__` and the `unorderable_unless_eq` machinery change.
- **Modifying `Relation.__hash__`.** Inherited from `frozenset` — already tolerance-stable because `frozenset.__hash__` uses element hashes, and once `Distinction.__hash__` is structural-only, the cascade is naturally tolerance-stable.
- **AC k-ary cutover** (deferred from P12b — separate ROADMAP item).

## Acceptance criteria

After the change lands:

1. **Per-class `__eq__` on all 10 classes** named in the affected-classes table. Each method follows the explicit-attribute template (no `general_eq` calls). Each returns `NotImplemented` on cross-type comparison.
2. **`Distinction.__hash__` uses only the four strict-equality attrs** (`mechanism`, `mechanism_state`, `cause_purview`, `effect_purview`).
3. **`Orderable.is_orderable_with` method** replaces `unorderable_unless_eq` ClassVar. Subclasses that need cross-instance guards override `is_orderable_with`.
4. **`general_eq` is deleted** from `pyphi/models/cmp.py`. Plus `unorderable_unless_eq` ClassVar from `Orderable`. `sametype` decorator: deleted if unused (plan-time audit).
5. **Goldens 25/25 byte-identical.**
6. **Full `uv run pytest`** (no path; includes doctests) passes. Any flipped tests investigated and resolved.
7. **`test/test_models.py`** has new per-class `__eq__` / `__hash__` tests. The `general_eq_*` tests are removed.
8. **Pyright + ruff clean.**
9. **Changelog fragment** describes the user-visible expansion (precision-aware structural equality now covers `Distinction`, `Distinctions`, `CauseEffectStructure`, `Relation` in addition to SIA / AcSIA / RIA / StateSpecification).
10. **ROADMAP follow-up entry** from the comparator project (the one labelled "Migrate `Distinction` and `CauseEffectStructure` `__eq__` to the precision-aware path") is removed — its work is now done.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| A test silently flips from failing to passing under the new tolerance-aware comparator, masking a real regression | Low-Medium | Full suite run before and after; investigate flips per `feedback_dont_give_up_on_architectural_refactors`. |
| Hash-contract bug: two tolerance-equal `Distinction`s hash differently | Low | Hash uses only strict-equality attrs from `__eq__`; provably contract-correct. Unit test asserts the invariant for representative pairs. |
| Performance regression: per-class methods slower than `general_eq` dispatch | Very low | Per-class methods are straight-line code; faster than `general_eq`'s `getattr` loop + `try/except AttributeError`. |
| Set/dict deduplication of distinctions changes behavior across substrates | Low | New `__hash__` may collide more across substrates (same structural fingerprint, different repertoires). Documented; `__eq__` still distinguishes by repertoire content. |
| Tests asserting strict inequality (`assert d1 != d2`) break for tolerance-similar items | Medium (theoretical); empirically TBD | Grep for such assertions during implementation; convert to tolerance-aware checks if the assertion was unintentional. |
| `Relation` explicit `__eq__` changes frozenset-inherited behavior subtly | Low | New method should be `isinstance + frozenset.__eq__(self, other)` — same behavior, just explicit. Verify by inspection and test. |

## Estimated scope

~250-300 lines net change, 5-6 commits, 1-2 focused days. Mechanical once the template is fixed; the per-class methods are well-bounded.
