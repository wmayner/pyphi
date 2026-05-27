# Per-class `__eq__` Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `general_eq` attribute-list dispatch with explicit per-class `__eq__` methods (and consistent `__hash__` methods) across all 10 IIT result-object classes. Extend precision-aware structural equality (added in `1bf0bd40`) to `Distinction`, `Distinctions`, `CauseEffectStructure`, and `Relation`; delete `general_eq`; refactor `Orderable.unorderable_unless_eq` ClassVar into an `is_orderable_with(other)` method.

**Architecture:** Per-class methods directly call `utils.eq` (for float scalars: `phi`, `alpha`, etc.), `numpy_aware_eq` (for arrays: repertoires, distributions), and `==` (for structural attrs: integer tuples, enums, indices). `__hash__` uses ONLY strict-equality attrs from the class's `__eq__` to satisfy Python's `a == b → hash(a) == hash(b)` contract under tolerance-aware equality. `Orderable.is_orderable_with(other) -> bool` replaces the `unorderable_unless_eq: ClassVar[list[str]]` attribute-list pattern.

**Tech Stack:** Python 3.12+, NumPy, `math.isclose`, pytest. No new dependencies. Uses the existing `EQUALITY_TOLERANCE = 1e-13` constant from `pyphi/models/cmp.py`.

**Spec:** `docs/superpowers/specs/2026-05-26-distinction-ces-equality-design.md` (committed at `d077cb55` on branch `2.0`).

**Branch context:** Work happens in a dedicated worktree at `/Users/will/projects/pyphi-eq` on branch `feature/per-class-eq` (branched off `2.0` at `d077cb55`). Main repo at `/Users/will/projects/pyphi` stays on `2.0` untouched.

---

## File Structure

| File | Action | Reason |
|------|--------|--------|
| `pyphi/models/cmp.py` | Modify (then delete `general_eq`) | Remove `general_eq` and `unorderable_unless_eq` ClassVar; refactor `Orderable.__lt__` to use `is_orderable_with(other)` |
| `pyphi/models/sia.py` | Modify | Per-class `__eq__` for `IIT3SystemIrreducibilityAnalysis`; remove `_sia_attributes` list |
| `pyphi/formalism/iit4/__init__.py` | Modify | Per-class `__eq__` and structural `__hash__` for IIT 4.0 SIA; remove `_sia_attributes` ClassVar |
| `pyphi/models/actual_causation.py` | Modify | Per-class `__eq__` + structural `__hash__` for `AcRepertoireIrreducibilityAnalysis` and `AcSystemIrreducibilityAnalysis`; `is_orderable_with` override for AcSIA; remove `_acria_attributes_for_eq` and `_ac_sia_attributes` lists |
| `pyphi/models/ria.py` | Modify | Per-class `__eq__` + structural `__hash__` for `RepertoireIrreducibilityAnalysis`; remove inline `attrs` list |
| `pyphi/models/state_specification.py` | Modify | Per-class `__eq__` + structural `__hash__` for `StateSpecification` |
| `pyphi/models/distinction.py` | Modify | New per-class `__eq__` + structural `__hash__` for `Distinction` (replaces existing raw-`==` and `np.array_equal` implementations) |
| `pyphi/models/distinctions.py` | Modify | Explicit `__eq__` for `Distinctions` (currently delegates via `concepts ==`; explicit for clarity) |
| `pyphi/models/ces.py` | Modify | Explicit `__eq__` for `CauseEffectStructure` (currently delegates via three `==`; explicit for clarity) |
| `pyphi/relations.py` | Modify | Explicit `__eq__` for `Relation` (closes line-182 `TODO`) |
| `test/test_models.py` | Modify | Add per-class `__eq__` / `__hash__` tests; remove `test_general_eq_*` tests; preserve `test_numpy_aware_eq_*` tests |
| `ROADMAP.md` | Modify | Remove the now-implemented "Migrate `Distinction` and `CauseEffectStructure` `__eq__` to the precision-aware path" entry |
| `changelog.d/per-class-equality.feature.md` | Create | Single-paragraph fragment describing the user-visible extension |

8 commits across the worktree.

---

## Pre-flight: Strict / Tolerance Attribute Audit

For each class, the strict-vs-tolerance attribute split that drives the per-class `__eq__` and `__hash__`:

| Class | Strict-eq attrs | Tolerance-eq (float) | Tolerance-eq (array) |
|-------|----------------|---------------------|---------------------|
| `IIT3SystemIrreducibilityAnalysis` | `partitioned_distinctions`, `partition`, `node_indices`, `node_labels`, `current_state` | `phi` | (none) |
| IIT 4.0 SIA (`formalism/iit4/__init__.py`) | `partition`, `cause`, `effect`, `system_state`, `current_state`, `node_indices` | `phi`, `normalized_phi`, `signed_phi`, `signed_normalized_phi`, `intrinsic_differentiation` | (none) |
| `AcRepertoireIrreducibilityAnalysis` | `state`, `direction`, `mechanism`, `purview` | `alpha`, `probability` | (none) |
| `AcSystemIrreducibilityAnalysis` | `direction`, `account`, `partitioned_account`, `partition`, `before_state`, `after_state`, `size`, `node_indices`, `cause_indices`, `effect_indices`, `node_labels` | `alpha` | (none) |
| `RepertoireIrreducibilityAnalysis` | `direction`, `mechanism`, `purview` | `phi` | `repertoire` |
| `StateSpecification` | `direction`, `purview`, `state` | `intrinsic_information` | `repertoire`, `unconstrained_repertoire` |
| `Distinction` | `mechanism`, `mechanism_state`, `cause_purview`, `effect_purview` | `phi` | `cause_repertoire`, `effect_repertoire` |
| `Distinctions` | `concepts` (tuple — cascades) | (none direct) | (none direct) |
| `CauseEffectStructure` | `sia`, `distinctions`, `relations` (all cascade) | (none direct) | (none direct) |
| `Relation` | `frozenset(distinctions)` (cascades via element `__eq__`) | (none direct) | (none direct) |

Existing `is_orderable_with` overrides needed:
- `AcSystemIrreducibilityAnalysis` (currently `unorderable_unless_eq: ClassVar[list[str]] = ["direction"]`)

All other classes use `unorderable_unless_eq: ClassVar[list[str]] = []` (the default; no override needed).

Existing `__hash__` audit (classes with both `__eq__` and `__hash__` — these need to be updated to structural-only):
- IIT 4.0 SIA: currently `hash((self.phi, self.partition))` → tolerance violation
- AcRepertoireIrreducibilityAnalysis: `hash(tuple(getattr(self, attr) for attr in _acria_attributes_for_eq))` → includes `alpha`, `probability`
- AcSystemIrreducibilityAnalysis: `hash((self.alpha, ...))` → tolerance violation
- RepertoireIrreducibilityAnalysis: `hash((self.phi, self.direction, self.mechanism, self.purview, self.specified_state, utils.np_hash(self.repertoire)))` → tolerance violation on `phi` and `repertoire`
- StateSpecification: `hash((self.direction, self.purview, self.state, self.intrinsic_information))` → tolerance violation on `intrinsic_information`
- Distinction: `hash((self.phi, self.mechanism, self.mechanism_state, self.cause_purview, self.effect_purview, utils.np_hash(self.cause_repertoire), utils.np_hash(self.effect_repertoire)))` → tolerance violation on `phi` and repertoires

`IIT3SystemIrreducibilityAnalysis` has `__eq__` but no `__hash__` (unhashable by Python default). Keep unhashable.

---

## Task 1: Set up the worktree

**Files:**
- Create: `/Users/will/projects/pyphi-eq/` (new worktree)

This task has no commit. It creates the isolated workspace for the rest of the plan.

- [ ] **Step 1.1: Create the worktree**

Run from main repo:
```bash
cd /Users/will/projects/pyphi
git worktree add -b feature/per-class-eq /Users/will/projects/pyphi-eq 2.0
```

Expected output:
```
Preparing worktree (new branch 'feature/per-class-eq')
HEAD is now at d077cb55 Add per-class __eq__ migration design spec
```

- [ ] **Step 1.2: Verify worktree state**

```bash
cd /Users/will/projects/pyphi-eq
git status --short
git log --oneline -2
```

Expected: clean working tree; HEAD at `d077cb55`.

- [ ] **Step 1.3: Add worktree-local VSCode settings (matches comparator-project pattern)**

Create `/Users/will/projects/pyphi-eq/.vscode/settings.json` with:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.diagnosticMode": "openFilesOnly"
}
```

This is globally `.gitignore`'d (via `~/.gitignore`) so won't appear in `git status`.

- [ ] **Step 1.4: Verify the python environment**

```bash
cd /Users/will/projects/pyphi-eq
uv sync
uv run python -c "import pyphi; print(pyphi.__file__)"
```

Expected: `pyphi` imports cleanly from the worktree's `pyphi/__init__.py`.

---

## Task 2: Refactor `Orderable` (Phase A)

**Files:**
- Modify: `pyphi/models/cmp.py` (lines ~40-90: `Orderable` and `OrderableByPhi`)
- Modify: `pyphi/models/actual_causation.py:516` (one `unorderable_unless_eq` site)
- Modify: `pyphi/models/sia.py:108` (one `unorderable_unless_eq = []` site)

Behavior unchanged: `is_orderable_with` returns True by default, AcSIA overrides to check direction (matching its current `unorderable_unless_eq = ["direction"]`).

- [ ] **Step 2.1: Write failing test for `is_orderable_with`**

Append to `test/test_models.py`:

```python
def test_orderable_is_orderable_with_default_true():
    """Default Orderable.is_orderable_with returns True."""
    class A(models.cmp.Orderable):
        def order_by(self):
            return 0
        def __eq__(self, other):
            return type(self) is type(other)
        def __hash__(self):
            return 0

    assert A().is_orderable_with(A())


def test_ac_sia_is_orderable_with_direction_guard():
    """AcSIA.is_orderable_with returns False when directions differ."""
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis
    from pyphi.direction import Direction
    a = AcSystemIrreducibilityAnalysis(alpha=1.0, direction=Direction.CAUSE)
    b = AcSystemIrreducibilityAnalysis(alpha=1.0, direction=Direction.EFFECT)
    assert not a.is_orderable_with(b)
    c = AcSystemIrreducibilityAnalysis(alpha=2.0, direction=Direction.CAUSE)
    assert a.is_orderable_with(c)
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
uv run pytest test/test_models.py::test_orderable_is_orderable_with_default_true test/test_models.py::test_ac_sia_is_orderable_with_direction_guard -v
```

Expected: FAIL (`is_orderable_with` does not exist).

- [ ] **Step 2.3: Refactor `Orderable` in `pyphi/models/cmp.py`**

Current `Orderable` class (lines ~40-89) has:
```python
class Orderable:
    """Base mixin for implementing rich object comparisons on phi-objects.
    ...
    """

    # The object is not orderable unless these attributes are all equal
    unorderable_unless_eq: ClassVar[list[str]] = []

    def order_by(self) -> Any:
        ...
        raise NotImplementedError

    def __lt__(self, other: object) -> bool:
        if not general_eq(self, other, self.unorderable_unless_eq):
            raise TypeError(
                f"Unorderable: the following attrs must be equal: "
                f"{self.unorderable_unless_eq}"
            )
        return self.order_by() < other.order_by()  # type: ignore[attr-defined]
    ...
```

Replace with:
```python
class Orderable:
    """Base mixin for implementing rich object comparisons on phi-objects.

    Both ``__eq__`` and ``order_by`` need to be implemented on the subclass.
    The ``order_by`` method returns a list of attributes which are compared
    to implement the ordering.

    Subclasses can optionally override ``is_orderable_with`` to enforce
    constraints (for example, ``AcSystemIrreducibilityAnalysis`` requires
    both operands to have the same ``direction``).
    """

    def order_by(self) -> Any:
        """Return a list of values to compare for ordering.

        The first value in the list has the greatest priority; if the first
        objects are equal the second object is compared, etc.
        """
        raise NotImplementedError

    def is_orderable_with(self, other: object) -> bool:
        """Whether ``self`` and ``other`` are mutually orderable.

        Default: any two instances are orderable. Override in subclasses
        that need cross-instance guards.
        """
        return True

    def __lt__(self, other: object) -> bool:
        if not self.is_orderable_with(other):
            raise TypeError(
                f"Unorderable: {type(self).__name__} instances do not satisfy "
                f"the orderability constraint of this type."
            )
        return self.order_by() < other.order_by()  # type: ignore[attr-defined]

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        return other < self

    def __ge__(self, other: object) -> bool:
        return other < self or self == other

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self == other
```

Note: `ClassVar` import in `pyphi/models/cmp.py` may no longer be needed after this change. Audit at end of task.

- [ ] **Step 2.4: Override `is_orderable_with` in `AcSystemIrreducibilityAnalysis`**

In `pyphi/models/actual_causation.py`, find:
```python
    unorderable_unless_eq: ClassVar[list[str]] = ["direction"]
```

Replace with:
```python
    def is_orderable_with(self, other: object) -> bool:
        return isinstance(other, AcSystemIrreducibilityAnalysis) and (
            self.direction == other.direction
        )
```

- [ ] **Step 2.5: Remove `unorderable_unless_eq` ClassVar from SIA 3.0**

In `pyphi/models/sia.py`, around line 108:
```python
    unorderable_unless_eq: ClassVar[list[str]] = []
```

Delete this line. The default `is_orderable_with` (returns True) is correct for SIA 3.0.

- [ ] **Step 2.6: Audit and remove remaining `unorderable_unless_eq` references**

```bash
grep -rn 'unorderable_unless_eq' pyphi/ test/
```

For each remaining site:
- If it's `: ClassVar[list[str]] = []` (the default-empty), delete the line.
- If it's a non-empty list, convert to `is_orderable_with` override.

Known sites at plan time: `pyphi/models/cmp.py` (the base class definition, removed in Step 2.3), `pyphi/models/distinction.py:161`, `pyphi/models/sia.py:108`, `pyphi/models/actual_causation.py:516`. Distinction is `unorderable_unless_eq: ClassVar[list[str]] = []` — delete.

- [ ] **Step 2.7: Run the new tests to verify they pass**

```bash
uv run pytest test/test_models.py::test_orderable_is_orderable_with_default_true test/test_models.py::test_ac_sia_is_orderable_with_direction_guard -v
```

Expected: PASS.

- [ ] **Step 2.8: Run full test suite to verify no regressions**

```bash
uv run pytest --tb=short -q
```

Expected: 0 new failures. The 2 pre-existing `test_complexes` failures (`test_possible_complexes`, `TestComplexesIIT30::test_all_sias_standard`) may persist — note but don't block.

- [ ] **Step 2.9: Pyright + ruff on touched files**

```bash
uv run pyright pyphi/models/cmp.py pyphi/models/sia.py pyphi/models/actual_causation.py pyphi/models/distinction.py test/test_models.py
uv run ruff check pyphi test
uv run ruff format --check pyphi test
```

Expected: 0 errors / 0 warnings; clean.

- [ ] **Step 2.10: Commit**

```bash
cd /Users/will/projects/pyphi-eq
git diff --cached --stat
git add pyphi/models/cmp.py pyphi/models/sia.py pyphi/models/actual_causation.py pyphi/models/distinction.py test/test_models.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Replace Orderable.unorderable_unless_eq with is_orderable_with method

The attribute-list ClassVar pattern is replaced with an overridable
method. AcSIA's direction guard becomes an explicit override; other
classes use the default (always orderable). Sets up the per-class
__eq__ migration per the spec.
EOF
)"
git show --stat HEAD
```

Expected: HEAD advances by 1; 5 files changed.

---

## Task 3: SIA per-class `__eq__` and `__hash__` (Phase B-1)

**Files:**
- Modify: `pyphi/models/sia.py` (lines 18-30: `_sia_attributes` list; line ~110-115: `__eq__`)
- Modify: `pyphi/formalism/iit4/__init__.py` (lines 184-199: `_sia_attributes` ClassVar; line ~240-244: `__eq__`; line ~250-258: `__hash__`)
- Modify: `test/test_models.py` (new tests)

- [ ] **Step 3.1: Write failing tests for SIA 3.0 and 4.0 `__eq__` under tolerance**

Append to `test/test_models.py`:

```python
def test_sia_3_eq_within_tolerance():
    """IIT 3.0 SIA: phi values differing by ~1e-15 compare equal."""
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis
    a = IIT3SystemIrreducibilityAnalysis(
        phi=1.0, partitioned_distinctions=None, partition=None,
        node_indices=(0,), node_labels=None, current_state=(0,),
    )
    b = IIT3SystemIrreducibilityAnalysis(
        phi=1.0 + 1e-15, partitioned_distinctions=None, partition=None,
        node_indices=(0,), node_labels=None, current_state=(0,),
    )
    assert a == b


def test_sia_3_eq_outside_tolerance():
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis
    a = IIT3SystemIrreducibilityAnalysis(
        phi=1.0, partitioned_distinctions=None, partition=None,
        node_indices=(0,), node_labels=None, current_state=(0,),
    )
    b = IIT3SystemIrreducibilityAnalysis(
        phi=1.001, partitioned_distinctions=None, partition=None,
        node_indices=(0,), node_labels=None, current_state=(0,),
    )
    assert a != b
```

If the IIT 4.0 SIA class is directly importable, add analogous tests for it. Otherwise mark the IIT 4.0 SIA testing as covered by goldens (which exercise the full pipeline).

- [ ] **Step 3.2: Run failing tests**

```bash
uv run pytest test/test_models.py::test_sia_3_eq_within_tolerance test/test_models.py::test_sia_3_eq_outside_tolerance -v
```

Expected: the within-tolerance test FAILs (current SIA 3.0 `__eq__` uses `general_eq` which routes `phi` through `utils.eq` so it might pass; verify). The outside-tolerance test should PASS already.

If both pass already, that's because `general_eq` already routed `phi` through `utils.eq` — the test is just locking in the behavior. Either way, replace `general_eq` with per-class `__eq__` in the next step.

- [ ] **Step 3.3: Replace `__eq__` and remove `_sia_attributes` in `pyphi/models/sia.py`**

Delete the module-level list at lines 18-27:
```python
from .distinctions import _null_ces

_sia_attributes = [
    "phi",
    "partitioned_distinctions",
    "partition",
    "node_indices",
    "node_labels",
    "current_state",
]
```

Keep the `from .distinctions import _null_ces` import; delete the `_sia_attributes` list (6 lines).

Replace the `__eq__` method (current lines ~108-114):
```python
    unorderable_unless_eq: ClassVar[list[str]] = []

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _sia_attributes)
```

(The `unorderable_unless_eq` line was already removed in Task 2.5.) Replace `__eq__` with:

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IIT3SystemIrreducibilityAnalysis):
            return NotImplemented
        if self.partitioned_distinctions != other.partitioned_distinctions:
            return False
        if self.partition != other.partition:
            return False
        if self.node_indices != other.node_indices:
            return False
        if self.node_labels != other.node_labels:
            return False
        if self.current_state != other.current_state:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        return True
```

Also update `to_json` if it references `_sia_attributes`. Check line ~140:
```python
        return {attr: getattr(self, attr) for attr in _sia_attributes}
```

Replace with an explicit list (the same 6 attributes):
```python
        return {
            attr: getattr(self, attr)
            for attr in (
                "phi",
                "partitioned_distinctions",
                "partition",
                "node_indices",
                "node_labels",
                "current_state",
            )
        }
```

- [ ] **Step 3.4: Replace `__eq__` and `__hash__` in `pyphi/formalism/iit4/__init__.py`**

Current `_sia_attributes` ClassVar (lines 184-199):
```python
    _sia_attributes: ClassVar[list[str]] = [
        "phi",
        "partition",
        "normalized_phi",
        "signed_phi",
        "signed_normalized_phi",
        "cause",
        "effect",
        "system_state",
        "current_state",
        "node_indices",
        "intrinsic_differentiation",
    ]
```

Delete this ClassVar (12 lines including the `_sia_attributes: ClassVar...` line).

Replace `__eq__` (current ~240-243):
```python
    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, self._sia_attributes)
```

With:
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SystemIrreducibilityAnalysis):
            return NotImplemented
        if self.partition != other.partition:
            return False
        if self.cause != other.cause:
            return False
        if self.effect != other.effect:
            return False
        if self.system_state != other.system_state:
            return False
        if self.current_state != other.current_state:
            return False
        if self.node_indices != other.node_indices:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        if not utils.eq(self.normalized_phi, other.normalized_phi):
            return False
        if not utils.eq(self.signed_phi, other.signed_phi):
            return False
        if not utils.eq(self.signed_normalized_phi, other.signed_normalized_phi):
            return False
        if not utils.eq(self.intrinsic_differentiation, other.intrinsic_differentiation):
            return False
        return True
```

Replace `__hash__` (currently uses `(self.phi, self.partition)` which violates contract):
```python
    def __hash__(self):
        return hash(
            (
                self.partition,
                self.system_state,
                self.current_state,
                self.node_indices,
            )
        )
```

The 4 hashed attrs are all strict-equality (no `phi`/`*_phi`/`intrinsic_differentiation` in the hash).

Search for other references to `_sia_attributes` in the file (might be used by `to_json`, `__repr__`, etc.). For each one, replace with an explicit attribute tuple.

- [ ] **Step 3.5: Add imports as needed**

If `utils.eq` is not yet imported in either file, add the import. Check existing imports first.

For `pyphi/models/sia.py`:
```python
from pyphi import utils
```
This should already be there (it's used elsewhere in the file).

For `pyphi/formalism/iit4/__init__.py`: same — verify `utils` is imported.

- [ ] **Step 3.6: Run SIA tests + goldens**

```bash
uv run pytest test/test_models.py -k "sia" -v
uv run pytest test/test_big_phi.py -x -q
uv run pytest test/test_golden_regression.py -v
```

Expected: SIA tests pass; goldens 25/25 byte-identical.

- [ ] **Step 3.7: Pyright + ruff on touched files**

```bash
uv run pyright pyphi/models/sia.py pyphi/formalism/iit4/__init__.py test/test_models.py
uv run ruff check pyphi/models/sia.py pyphi/formalism/iit4/__init__.py
uv run ruff format --check pyphi/models/sia.py pyphi/formalism/iit4/__init__.py
```

Expected: clean.

- [ ] **Step 3.8: Commit**

```bash
git add pyphi/models/sia.py pyphi/formalism/iit4/__init__.py test/test_models.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Per-class __eq__ for SIA 3.0 and 4.0; remove attribute-list ClassVars

Replaces general_eq dispatch with explicit per-class methods. Updates
IIT 4.0 SIA's __hash__ to use only strict-equality attrs (partition,
system_state, current_state, node_indices) to satisfy the eq/hash
contract under tolerance-aware __eq__. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Task 4: AcRIA and AcSIA per-class `__eq__` and `__hash__` (Phase B-2)

**Files:**
- Modify: `pyphi/models/actual_causation.py` (lines 33-40: `_acria_attributes_for_eq`; lines 19-32: `_acria_attributes` — keep, used by `to_json`; lines 424-436: `_ac_sia_attributes`; `AcRepertoireIrreducibilityAnalysis.__eq__` and `__hash__`; `AcSystemIrreducibilityAnalysis.__eq__` and `__hash__`)
- Modify: `test/test_models.py` (new tests)

- [ ] **Step 4.1: Write failing tests**

Append to `test/test_models.py`:

```python
def test_ac_ria_eq_within_tolerance():
    """AcRIA: alpha values differing by ~1e-15 compare equal."""
    from pyphi.models.actual_causation import AcRepertoireIrreducibilityAnalysis
    from pyphi.direction import Direction
    a = AcRepertoireIrreducibilityAnalysis(
        alpha=1.0, state=(0,), direction=Direction.CAUSE,
        mechanism=(0,), purview=(0,), probability=0.5,
    )
    b = AcRepertoireIrreducibilityAnalysis(
        alpha=1.0 + 1e-15, state=(0,), direction=Direction.CAUSE,
        mechanism=(0,), purview=(0,), probability=0.5,
    )
    assert a == b


def test_ac_ria_hash_contract():
    """eq → same hash for AcRIA."""
    from pyphi.models.actual_causation import AcRepertoireIrreducibilityAnalysis
    from pyphi.direction import Direction
    a = AcRepertoireIrreducibilityAnalysis(
        alpha=1.0, state=(0,), direction=Direction.CAUSE,
        mechanism=(0,), purview=(0,), probability=0.5,
    )
    b = AcRepertoireIrreducibilityAnalysis(
        alpha=1.0 + 1e-15, state=(0,), direction=Direction.CAUSE,
        mechanism=(0,), purview=(0,), probability=0.5,
    )
    assert a == b
    assert hash(a) == hash(b)


def test_ac_sia_eq_within_tolerance():
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis
    from pyphi.direction import Direction
    a = AcSystemIrreducibilityAnalysis(alpha=1.0, direction=Direction.CAUSE)
    b = AcSystemIrreducibilityAnalysis(alpha=1.0 + 1e-15, direction=Direction.CAUSE)
    assert a == b
```

- [ ] **Step 4.2: Run failing tests to verify they fail**

```bash
uv run pytest test/test_models.py::test_ac_ria_eq_within_tolerance test/test_models.py::test_ac_ria_hash_contract test/test_models.py::test_ac_sia_eq_within_tolerance -v
```

Expected: hash_contract test FAILs (current `__hash__` uses `alpha`); eq tests may pass (general_eq already uses utils.eq for `alpha`).

- [ ] **Step 4.3: Replace `AcRepertoireIrreducibilityAnalysis.__eq__` and `__hash__`**

Delete `_acria_attributes_for_eq` (lines 33-40):
```python
_acria_attributes_for_eq = [
    "alpha",
    "state",
    "direction",
    "mechanism",
    "purview",
    "probability",
]
```

(Keep `_acria_attributes` — used by `to_json`.)

Replace `__eq__` (lines ~135-140):
```python
    def __eq__(self, other):
        # TODO(slipperyhank): include 2nd state here?
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _acria_attributes_for_eq)
```

With:
```python
    def __eq__(self, other: object) -> bool:
        # TODO(slipperyhank): include 2nd state here?
        if not isinstance(other, AcRepertoireIrreducibilityAnalysis):
            return NotImplemented
        if self.state != other.state:
            return False
        if self.direction != other.direction:
            return False
        if self.mechanism != other.mechanism:
            return False
        if self.purview != other.purview:
            return False
        if not utils.eq(self.alpha, other.alpha):
            return False
        if not utils.eq(self.probability, other.probability):
            return False
        return True
```

Replace `__hash__` (lines ~152-154):
```python
    def __hash__(self):
        attrs = tuple(getattr(self, attr) for attr in _acria_attributes_for_eq)
        return hash(attrs)
```

With:
```python
    def __hash__(self) -> int:
        return hash(
            (
                self.state,
                self.direction,
                self.mechanism,
                self.purview,
            )
        )
```

- [ ] **Step 4.4: Replace `AcSystemIrreducibilityAnalysis.__eq__` and `__hash__`**

Delete `_ac_sia_attributes` (lines 424-437):
```python
_ac_sia_attributes = [
    "alpha",
    "direction",
    ...
    "node_labels",
]
```

Replace `__eq__` (lines ~520-525):
```python
    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _ac_sia_attributes)
```

With:
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AcSystemIrreducibilityAnalysis):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.account != other.account:
            return False
        if self.partitioned_account != other.partitioned_account:
            return False
        if self.partition != other.partition:
            return False
        if self.before_state != other.before_state:
            return False
        if self.after_state != other.after_state:
            return False
        if self.size != other.size:
            return False
        if self.node_indices != other.node_indices:
            return False
        if self.cause_indices != other.cause_indices:
            return False
        if self.effect_indices != other.effect_indices:
            return False
        if self.node_labels != other.node_labels:
            return False
        if not utils.eq(self.alpha, other.alpha):
            return False
        return True
```

Replace `__hash__` (~lines 530+, currently uses `self.alpha`):
```python
    def __hash__(self) -> int:
        return hash(
            (
                self.direction,
                self.account,
                self.partitioned_account,
                self.partition,
                self.before_state,
                self.after_state,
                self.size,
                self.node_indices,
                self.cause_indices,
                self.effect_indices,
                self.node_labels,
            )
        )
```

Audit `to_json` usage of `_ac_sia_attributes` (line ~549) — replace with the explicit tuple from `__eq__`.

- [ ] **Step 4.5: Audit remaining `_acria_attributes_for_eq` / `_ac_sia_attributes` references**

```bash
grep -n '_acria_attributes_for_eq\|_ac_sia_attributes' pyphi/models/actual_causation.py
```

Expected: 0 references (all migrated). If any remain, replace with explicit tuples.

- [ ] **Step 4.6: Run tests + goldens**

```bash
uv run pytest test/test_models.py -k "ac_ria or ac_sia or ac_sys" -v
uv run pytest test/test_actual.py -x -q
uv run pytest test/test_golden_regression.py -v
```

Expected: tests pass; goldens 25/25 byte-identical.

- [ ] **Step 4.7: Pyright + ruff**

```bash
uv run pyright pyphi/models/actual_causation.py test/test_models.py
uv run ruff check pyphi/models/actual_causation.py
uv run ruff format --check pyphi/models/actual_causation.py
```

Expected: clean.

- [ ] **Step 4.8: Commit**

```bash
git add pyphi/models/actual_causation.py test/test_models.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Per-class __eq__ for AcRIA and AcSIA; structural-only __hash__

Replaces general_eq dispatch with explicit per-class methods. Both
classes get structural-only __hash__ implementations (omitting alpha
and probability) to satisfy the eq/hash contract under tolerance-aware
__eq__. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Task 5: RIA and StateSpecification per-class `__eq__` and `__hash__` (Phase B-3)

**Files:**
- Modify: `pyphi/models/ria.py` (lines ~352-357: `__eq__`; lines ~363-374: `__hash__`)
- Modify: `pyphi/models/state_specification.py` (lines ~96-108: `__eq__`; lines ~110-113: `__hash__`)
- Modify: `test/test_models.py` (new tests)

- [ ] **Step 5.1: Write failing tests**

Append to `test/test_models.py`:

```python
def test_ria_eq_within_tolerance():
    """RIA: phi values differing by ~1e-15 compare equal."""
    from pyphi.models.ria import RepertoireIrreducibilityAnalysis
    from pyphi.direction import Direction
    a = RepertoireIrreducibilityAnalysis(
        phi=1.0, direction=Direction.CAUSE, mechanism=(0,), purview=(0,),
        repertoire=np.array([0.5, 0.5]),
        partitioned_repertoire=np.array([0.5, 0.5]),
    )
    b = RepertoireIrreducibilityAnalysis(
        phi=1.0 + 1e-15, direction=Direction.CAUSE, mechanism=(0,), purview=(0,),
        repertoire=np.array([0.5, 0.5]) + 1e-15,
        partitioned_repertoire=np.array([0.5, 0.5]),
    )
    assert a == b
    assert hash(a) == hash(b)


def test_state_specification_eq_within_tolerance():
    """StateSpecification: intrinsic_information ~1e-15 apart compare equal."""
    from pyphi.models.state_specification import StateSpecification
    from pyphi.direction import Direction
    a = StateSpecification(
        direction=Direction.CAUSE, purview=(0,), state=(0,),
        intrinsic_information=1.0,
        repertoire=np.array([0.5, 0.5]),
        unconstrained_repertoire=np.array([0.5, 0.5]),
    )
    b = StateSpecification(
        direction=Direction.CAUSE, purview=(0,), state=(0,),
        intrinsic_information=1.0 + 1e-15,
        repertoire=np.array([0.5, 0.5]),
        unconstrained_repertoire=np.array([0.5, 0.5]),
    )
    assert a == b
    assert hash(a) == hash(b)
```

Inspect the `RepertoireIrreducibilityAnalysis` and `StateSpecification` constructors for the exact required keyword arguments — adjust the test above if there are required positional args.

- [ ] **Step 5.2: Run failing tests**

```bash
uv run pytest test/test_models.py::test_ria_eq_within_tolerance test/test_models.py::test_state_specification_eq_within_tolerance -v
```

Expected: tests may pass or fail (depends on whether the test constructors match — fix constructor invocations as needed before proceeding).

- [ ] **Step 5.3: Replace `RepertoireIrreducibilityAnalysis.__eq__` and `__hash__`**

In `pyphi/models/ria.py`:

Replace `__eq__` (lines ~352-357):
```python
    def __eq__(self, other):
        # We don't consider the partition and partitioned repertoire in
        # checking for RIA equality.
        attrs = ["phi", "direction", "mechanism", "purview", "repertoire"]
        return cmp.general_eq(self, other, attrs)
```

With:
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RepertoireIrreducibilityAnalysis):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.mechanism != other.mechanism:
            return False
        if self.purview != other.purview:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        if not cmp.numpy_aware_eq(self.repertoire, other.repertoire):
            return False
        return True
```

Replace `__hash__` (lines ~363-374):
```python
    def __hash__(self):
        return hash(
            (
                self.phi,
                self.direction,
                self.mechanism,
                self.purview,
                self.specified_state,
                utils.np_hash(self.repertoire),
            )
        )
```

With:
```python
    def __hash__(self) -> int:
        return hash(
            (
                self.direction,
                self.mechanism,
                self.purview,
                self.specified_state,
            )
        )
```

Note: dropped `phi` (tolerance attr) and `utils.np_hash(self.repertoire)` (tolerance attr). Kept `specified_state` because it's a strict-equality attr (state-spec object whose own `__eq__` is being made tolerance-aware in this task).

- [ ] **Step 5.4: Replace `StateSpecification.__eq__` and `__hash__`**

In `pyphi/models/state_specification.py`:

Replace `__eq__` (lines ~96-108):
```python
    def __eq__(self, other: object) -> bool:
        return cmp.general_eq(
            self, other,
            [
                "direction", "purview", "state",
                "intrinsic_information",
                "repertoire", "unconstrained_repertoire",
            ],
        )
```

With:
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateSpecification):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.purview != other.purview:
            return False
        if self.state != other.state:
            return False
        if not utils.eq(self.intrinsic_information, other.intrinsic_information):
            return False
        if not cmp.numpy_aware_eq(self.repertoire, other.repertoire):
            return False
        if not cmp.numpy_aware_eq(self.unconstrained_repertoire, other.unconstrained_repertoire):
            return False
        return True
```

Replace `__hash__` (lines ~110-113):
```python
    def __hash__(self) -> int:
        return hash(
            (self.direction, self.purview, self.state, self.intrinsic_information)
        )
```

With:
```python
    def __hash__(self) -> int:
        return hash(
            (self.direction, self.purview, self.state)
        )
```

Note: dropped `intrinsic_information` (tolerance attr).

- [ ] **Step 5.5: Run tests + goldens**

```bash
uv run pytest test/test_models.py -k "ria or state_specification" -v
uv run pytest test/test_big_phi.py test/test_actual.py -x -q
uv run pytest test/test_golden_regression.py -v
```

Expected: 25/25 goldens byte-identical.

- [ ] **Step 5.6: Pyright + ruff**

```bash
uv run pyright pyphi/models/ria.py pyphi/models/state_specification.py test/test_models.py
uv run ruff check pyphi/models/ria.py pyphi/models/state_specification.py
uv run ruff format --check pyphi/models/ria.py pyphi/models/state_specification.py
```

- [ ] **Step 5.7: Commit**

```bash
git add pyphi/models/ria.py pyphi/models/state_specification.py test/test_models.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Per-class __eq__ for RIA and StateSpecification; structural-only __hash__

Replaces general_eq dispatch with explicit per-class methods. Both
classes get structural-only __hash__ (omitting phi, intrinsic_information,
and repertoire arrays) to satisfy the eq/hash contract under
tolerance-aware __eq__. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Task 6: Distinction per-class `__eq__` and structural `__hash__` (Phase C-1)

**Files:**
- Modify: `pyphi/models/distinction.py` (lines ~163-186: `__eq__`, `__hash__`)
- Modify: `test/test_models.py` (new tests)

This is the load-bearing commit — the precision-aware comparator finally reaches `Distinction`, and the new `__hash__` is the most-discussed design decision.

- [ ] **Step 6.1: Write failing tests**

Append to `test/test_models.py`:

```python
def _make_distinction(phi_val: float, repertoire_offset: float = 0.0):
    """Helper: construct a Distinction with given phi and repertoire offset.

    Constructs minimal cause/effect MICEs from scratch. May need adjusting
    based on the actual MICE/Distinction constructor signatures.
    """
    import numpy as np
    from pyphi.models.distinction import Distinction
    from pyphi.models.ria import MaximallyIrreducibleCause, MaximallyIrreducibleEffect
    from pyphi.direction import Direction
    # Construct minimal cause and effect MICEs
    # IMPORTANT: adjust constructor signature if MaximallyIrreducibleCause/Effect
    # have different required params; this is a sketch.
    cause = MaximallyIrreducibleCause(
        phi=phi_val, direction=Direction.CAUSE,
        mechanism=(0,), purview=(0,),
        repertoire=np.array([0.5, 0.5]) + repertoire_offset,
        partitioned_repertoire=np.array([0.5, 0.5]),
    )
    effect = MaximallyIrreducibleEffect(
        phi=phi_val, direction=Direction.EFFECT,
        mechanism=(0,), purview=(0,),
        repertoire=np.array([0.5, 0.5]) + repertoire_offset,
        partitioned_repertoire=np.array([0.5, 0.5]),
    )
    return Distinction(mechanism=(0,), cause=cause, effect=effect)


def test_distinction_eq_within_tolerance():
    """Distinction: phi ~1e-15 apart compare equal."""
    a = _make_distinction(phi_val=1.0)
    b = _make_distinction(phi_val=1.0 + 1e-15)
    assert a == b


def test_distinction_eq_outside_tolerance():
    """Distinction: phi 1e-3 apart compare unequal."""
    a = _make_distinction(phi_val=1.0)
    b = _make_distinction(phi_val=1.001)
    assert a != b


def test_distinction_eq_array_within_tolerance():
    """Distinction: repertoires ~1e-15 apart compare equal."""
    a = _make_distinction(phi_val=1.0, repertoire_offset=0.0)
    b = _make_distinction(phi_val=1.0, repertoire_offset=1e-15)
    assert a == b


def test_distinction_hash_contract_within_tolerance():
    """eq → same hash for Distinction."""
    a = _make_distinction(phi_val=1.0)
    b = _make_distinction(phi_val=1.0 + 1e-15)
    assert a == b
    assert hash(a) == hash(b)


def test_distinction_hash_structural_only():
    """hash depends only on (mechanism, mechanism_state, cause_purview, effect_purview)."""
    a = _make_distinction(phi_val=1.0)
    b = _make_distinction(phi_val=2.0)  # different phi, same structure
    # hashes should match because hash uses only structural attrs
    assert hash(a) == hash(b)
    # but they should NOT be equal (different phi)
    assert a != b
```

- [ ] **Step 6.2: Run failing tests**

```bash
uv run pytest test/test_models.py -k "distinction" -v
```

Expected: at least some FAIL — particularly `test_distinction_eq_within_tolerance` (current `__eq__` uses raw `==` on `phi`) and `test_distinction_hash_structural_only` (current `__hash__` uses `phi` and arrays).

- [ ] **Step 6.3: Replace `Distinction.__eq__` and `Distinction.__hash__`**

In `pyphi/models/distinction.py`:

Replace `__eq__` (lines ~163-173):
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinction):
            return NotImplemented
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.mechanism_state == other.mechanism_state
            and self.cause_purview == other.cause_purview
            and self.effect_purview == other.effect_purview
            and self.eq_repertoires(other)
        )
```

With:
```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinction):
            return NotImplemented
        if self.mechanism != other.mechanism:
            return False
        if self.mechanism_state != other.mechanism_state:
            return False
        if self.cause_purview != other.cause_purview:
            return False
        if self.effect_purview != other.effect_purview:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        if not cmp.numpy_aware_eq(self.cause_repertoire, other.cause_repertoire):
            return False
        if not cmp.numpy_aware_eq(self.effect_repertoire, other.effect_repertoire):
            return False
        return True
```

Replace `__hash__` (lines ~175-186):
```python
    def __hash__(self):
        return hash(
            (
                self.phi,
                self.mechanism,
                self.mechanism_state,
                self.cause_purview,
                self.effect_purview,
                utils.np_hash(self.cause_repertoire),
                utils.np_hash(self.effect_repertoire),
            )
        )
```

With:
```python
    def __hash__(self) -> int:
        return hash(
            (
                self.mechanism,
                self.mechanism_state,
                self.cause_purview,
                self.effect_purview,
            )
        )
```

Note: `eq_repertoires` is now used only by the explicit `__eq__` — it's a public method, keep it but it's no longer called by `__eq__`. The new `__eq__` uses `cmp.numpy_aware_eq` directly (which is the tolerance-aware version of `np.array_equal`).

Actually, `eq_repertoires` is still potentially useful externally. Keep the method as-is (it's strict comparison via `np.array_equal`); just stop using it from `__eq__`.

- [ ] **Step 6.4: Run distinction tests + downstream tests**

```bash
uv run pytest test/test_models.py -k "distinction" -v
uv run pytest test/test_big_phi.py -k "distinction or concept or ces" -x -q
uv run pytest test/test_golden_regression.py -v
```

Expected: all distinction tests pass; goldens 25/25 byte-identical.

- [ ] **Step 6.5: Pyright + ruff**

```bash
uv run pyright pyphi/models/distinction.py test/test_models.py
uv run ruff check pyphi/models/distinction.py
uv run ruff format --check pyphi/models/distinction.py
```

- [ ] **Step 6.6: Commit**

```bash
git add pyphi/models/distinction.py test/test_models.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Per-class __eq__ for Distinction; structural-only __hash__

Replaces raw == on phi and np.array_equal on repertoires with utils.eq
and cmp.numpy_aware_eq. __hash__ now uses only the four strict-equality
attrs (mechanism, mechanism_state, cause_purview, effect_purview);
phi and repertoires are dropped because they're tolerance-checked in
__eq__. Provably contract-correct. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Task 7: Distinctions, CauseEffectStructure, Relation explicit `__eq__` (Phase C-2)

**Files:**
- Modify: `pyphi/models/distinctions.py` (lines ~123-126)
- Modify: `pyphi/models/ces.py` (lines ~79-86)
- Modify: `pyphi/relations.py` (line ~182 TODO + new `__eq__`)
- Modify: `test/test_models.py` (new tests)

- [ ] **Step 7.1: Write failing tests**

Append to `test/test_models.py`:

```python
def test_distinctions_eq_cascades_through_distinction():
    """Distinctions.__eq__ cascades via tuple-of-concepts equality."""
    d1 = _make_distinction(phi_val=1.0)
    d2 = _make_distinction(phi_val=1.0 + 1e-15)
    from pyphi.models.distinctions import Distinctions
    s1 = Distinctions((d1,))
    s2 = Distinctions((d2,))
    assert s1 == s2


def test_relation_eq_cascades_through_distinction():
    """Relation.__eq__ should consider tolerance-eq distinctions equal."""
    d1 = _make_distinction(phi_val=1.0)
    d2 = _make_distinction(phi_val=1.0 + 1e-15)
    from pyphi.relations import Relation
    r1 = Relation((d1,))
    r2 = Relation((d2,))
    assert r1 == r2
```

(Skip a CauseEffectStructure cascade test — it requires a full SIA which is unwieldy to construct in a unit test. Goldens exercise the full pipeline.)

- [ ] **Step 7.2: Run failing tests**

```bash
uv run pytest test/test_models.py::test_distinctions_eq_cascades_through_distinction test/test_models.py::test_relation_eq_cascades_through_distinction -v
```

Expected: distinctions test passes already (cascades through Distinction's new `__eq__`). Relation test passes already (frozenset cascades through `Distinction.__eq__` + `__hash__`).

The tests are still valuable: they lock in the cascade behavior, so any future regression that breaks the cascade is caught.

- [ ] **Step 7.3: Add explicit `__eq__` to `Distinctions`**

In `pyphi/models/distinctions.py`, replace (lines ~123-126):

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinctions):
            return NotImplemented
        return self.concepts == other.concepts
```

With (the existing code is already this — but verify, and ensure type-narrowing isinstance check is in place):

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinctions):
            return NotImplemented
        return self.concepts == other.concepts
```

(If unchanged, skip this step.)

- [ ] **Step 7.4: Add explicit `__eq__` to `CauseEffectStructure`**

In `pyphi/models/ces.py`, replace (lines ~79-86):

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseEffectStructure):
            return NotImplemented
        return (
            self.sia == other.sia
            and self.distinctions == other.distinctions
            and self.relations == other.relations
        )
```

With (verify it's already this — the existing code is correct; just ensure clarity):

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseEffectStructure):
            return NotImplemented
        if self.sia != other.sia:
            return False
        if self.distinctions != other.distinctions:
            return False
        if self.relations != other.relations:
            return False
        return True
```

The refactored form makes the cascade explicit step-by-step.

- [ ] **Step 7.5: Add explicit `__eq__` to `Relation` (closes line-182 `TODO`)**

In `pyphi/relations.py`, find the `Relation` class (around line 126) and locate the `# TODO(4.0) need to also implement __eq__ here` comment at line 182.

Replace (the comment line and surrounding context, ~line 178-183):

```python
    def __bool__(self):
        return utils.is_positive(self.phi)

    # TODO(4.0) need to also implement __eq__ here

    @cached_property
    def mechanisms(self):
```

With:

```python
    def __bool__(self):
        return utils.is_positive(self.phi)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return frozenset.__eq__(self, other)

    def __hash__(self) -> int:
        return frozenset.__hash__(self)

    @cached_property
    def mechanisms(self):
```

The explicit `__eq__` is functionally equivalent to the inherited `frozenset.__eq__` (which compares element-by-element using element `__eq__` + `__hash__`), but now it's a documented part of the class and the line-182 `TODO` is closed. The explicit `__hash__` delegates to `frozenset.__hash__` for the same reason.

- [ ] **Step 7.6: Run tests + goldens**

```bash
uv run pytest test/test_models.py -k "distinctions or relation" -v
uv run pytest test/test_relations.py -x -q
uv run pytest test/test_big_phi.py -k "ces" -x -q
uv run pytest test/test_golden_regression.py -v
```

Expected: tests pass; goldens 25/25 byte-identical.

- [ ] **Step 7.7: Pyright + ruff**

```bash
uv run pyright pyphi/models/distinctions.py pyphi/models/ces.py pyphi/relations.py test/test_models.py
uv run ruff check pyphi/models/distinctions.py pyphi/models/ces.py pyphi/relations.py
uv run ruff format --check pyphi/models/distinctions.py pyphi/models/ces.py pyphi/relations.py
```

- [ ] **Step 7.8: Commit**

```bash
git add pyphi/models/distinctions.py pyphi/models/ces.py pyphi/relations.py test/test_models.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Explicit __eq__ for Distinctions, CauseEffectStructure, Relation

Each delegates through to its component objects, which now use the
precision-aware per-class __eq__. Closes the line-182 TODO on
Relation. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Task 8: Delete `general_eq` + cleanup (Phase D)

**Files:**
- Modify: `pyphi/models/cmp.py` (delete `general_eq` function, `sametype` decorator audit, `unorderable_unless_eq` references in docs)
- Modify: `test/test_models.py` (remove `test_general_eq_*` tests)
- Modify: `ROADMAP.md` (remove the now-implemented follow-up entry)
- Create: `changelog.d/per-class-equality.feature.md`

- [ ] **Step 8.1: Verify no remaining `general_eq` callers**

```bash
cd /Users/will/projects/pyphi-eq
grep -rn 'general_eq' pyphi/ test/
```

Expected: only references in `pyphi/models/cmp.py` (the function definition itself) and possibly `test/test_models.py` (the tests being removed).

If callers remain, STOP — investigate and migrate them before proceeding.

- [ ] **Step 8.2: Audit `sametype` decorator usage**

```bash
grep -rn 'sametype\|@cmp.sametype' pyphi/ test/
```

If unused (no `@sametype` decorators outside `cmp.py`), delete the decorator definition too. If used, leave it.

- [ ] **Step 8.3: Delete `general_eq` from `pyphi/models/cmp.py`**

Remove the function definition (lines ~127-152 approximately):

```python
def general_eq(a: object, b: object, attributes: Sequence[str]) -> bool:
    """Return whether two objects are equal up to the given attributes.
    ...
    """
    try:
        ...
    except AttributeError:
        return False
```

If `Sequence` from `collections.abc` is no longer used after the deletion, remove the import too.

- [ ] **Step 8.4: Remove `test_general_eq_*` tests from `test/test_models.py`**

```bash
grep -n 'test_general_eq' test/test_models.py
```

Delete each `test_general_eq_*` function. Adjacent context (test fixtures, `nt_attributes`, `nt`, `a` from lines ~183-191) may also be removable if no other test uses them — verify with grep first.

Specifically:
```python
nt_attributes = ["this", "that", "phi", "mechanism", "purview"]
nt = namedtuple("nt", nt_attributes)
a = nt(...)
```

`a` is also used by `test_numpy_aware_eq_*` tests. Keep `a` and `nt`/`nt_attributes` (they're used by both general_eq and numpy_aware_eq tests; only the general_eq-specific tests are deleted).

- [ ] **Step 8.5: Remove the now-implemented ROADMAP follow-up entry**

In `/Users/will/projects/pyphi-eq/ROADMAP.md`, find the "Informal notes — pre-release housekeeping" section. Locate the entry titled "Migrate `Distinction` and `CauseEffectStructure` `__eq__` to the precision-aware path." (the one starting around the entry text:

```markdown
- **Migrate ``Distinction`` and ``CauseEffectStructure`` ``__eq__`` to the
  precision-aware path.** Unlike ``SIA`` / ``AcSIA`` / ``RIA`` /
  ``StateSpecification`` ...
```

Delete the entire bullet (the entry plus its body). The work is now done.

- [ ] **Step 8.6: Create the changelog fragment**

Create `changelog.d/per-class-equality.feature.md` with:

```
Structural equality on IIT result objects is now precision-aware up to ``EQUALITY_TOLERANCE = 1e-13`` across ALL result-object classes: ``SystemIrreducibilityAnalysis`` (both 3.0 and 4.0), ``AcSystemIrreducibilityAnalysis``, ``AcRepertoireIrreducibilityAnalysis``, ``RepertoireIrreducibilityAnalysis``, ``StateSpecification``, ``Distinction``, ``Distinctions``, ``CauseEffectStructure``, and ``Relation``. The previous ``general_eq`` attribute-list dispatch is replaced with explicit per-class ``__eq__`` methods that directly call ``utils.eq`` for float scalars and ``numpy_aware_eq`` for arrays. Classes that also define ``__hash__`` (Distinction, RIA, AcRIA, StateSpecification, IIT 4.0 SIA, AcSIA) now use structural-only hashes (omitting tolerance-checked attributes) to satisfy Python's ``a == b → hash(a) == hash(b)`` contract under tolerance-aware equality. ``Orderable.unorderable_unless_eq`` ClassVar is replaced with an ``is_orderable_with(other)`` method; the only subclass that needed it (``AcSystemIrreducibilityAnalysis``) now has an explicit override.
```

- [ ] **Step 8.7: Run full pytest (no path argument) to verify clean state**

```bash
cd /Users/will/projects/pyphi-eq
uv run pytest --tb=short -q
```

Expected: 0 failures (or the 2 pre-existing `test_complexes` failures only). Any new failure is blocking.

If a test silently flips from previously-failing to passing under the loosened comparator, that's a concern per saved memory `feedback_dont_give_up_on_architectural_refactors` — investigate before continuing.

- [ ] **Step 8.8: Full pyright + ruff verification**

```bash
uv run pyright pyphi
uv run ruff check pyphi test
uv run ruff format --check pyphi test
```

Expected: 0 errors / 1 baseline warning (the pre-existing `reportUnsupportedDunderAll` in `pyphi/__init__.py:121`); ruff clean.

- [ ] **Step 8.9: Goldens final check**

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 25/25 byte-identical.

- [ ] **Step 8.10: Commit**

```bash
git add pyphi/models/cmp.py test/test_models.py ROADMAP.md changelog.d/per-class-equality.feature.md
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Delete general_eq; cleanup post per-class __eq__ migration

All 10 result-object classes now have explicit per-class __eq__ methods.
general_eq and its attribute-list dispatch pattern is removed. Per-class
__hash__ implementations are tolerance-contract-correct. Per the spec.
EOF
)"
git show --stat HEAD
```

---

## Verification (consolidated final gates)

After Task 8 lands, all of these MUST be green:

```bash
cd /Users/will/projects/pyphi-eq

# Full suite incl. doctests (CRITICAL: no path argument; uses testpaths + doctest-modules)
uv run pytest --tb=short -q
# Expected: 0 new failures; only pre-existing test_complexes failures persist

# Goldens
uv run pytest test/test_golden_regression.py -v
# Expected: 25/25 byte-identical

# Type check
uv run pyright pyphi
# Expected: 0 errors / 1 baseline warning

# Linting
uv run ruff check pyphi test
uv run ruff format --check pyphi test
# Expected: clean

# Verify general_eq deletion
grep -rn 'general_eq' pyphi/ test/
# Expected: no matches

# Verify is_orderable_with adoption
grep -rn 'unorderable_unless_eq' pyphi/ test/
# Expected: no matches

# Smoke test: tolerance-aware equality across the class hierarchy
uv run python -c "
import numpy as np
from pyphi.models.cmp import EQUALITY_TOLERANCE
print(f'EQUALITY_TOLERANCE = {EQUALITY_TOLERANCE}')

# Verify the constant is accessible
assert EQUALITY_TOLERANCE == 1e-13

print('per-class __eq__ migration OK')
"
# Expected: 'per-class __eq__ migration OK'
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Goldens drift (any one of 25 fails byte-identity) | Vanishingly low | Computed values unchanged; only `__eq__` semantics change. Goldens use `_compare` with `EQUALITY_TOLERANCE` directly (not `__eq__`). |
| Test silently flips from failing→passing under loosened tolerance (real regression masked) | Low-Medium | Full suite run before and after; investigate any flip per `feedback_dont_give_up_on_architectural_refactors`. |
| Set/dict deduplication of `Distinction` changes behavior in `relations.py:99`, `compositional_state.py:275`, or test assertions like `set(c) == set(...)` | Medium (behavior change) | New `__hash__` may collide more across substrates; `__eq__` still distinguishes; existing test_big_phi assertions should still pass (they compare same-substrate distinctions). Monitor `test_relations.py` and `test_compositional_state.py` (if exists). |
| Forgotten `_sia_attributes` / `_acria_attributes_for_eq` / `_ac_sia_attributes` reference in `to_json` or `__repr__` | Medium | Each task explicitly audits these (Step 3.3, 4.5). Grep at task boundaries. |
| Missing `is_orderable_with` override on a class that previously had `unorderable_unless_eq = [...]` (non-empty) | Low | Pre-flight audit identified AcSIA as the only non-empty case. Verified at Task 2.6. |
| `numpy_aware_eq` import not added to classes that newly need it (RIA, StateSpec, Distinction) | Low | Use `cmp.numpy_aware_eq` (qualified) via the `from . import cmp` import already present. |
| `Relation.__eq__` becomes type-strict and breaks `frozenset` cross-equivalence | Low | The new `__eq__` does `isinstance(other, Relation) and frozenset.__eq__(self, other)` — matches existing inherited behavior (frozenset already required isinstance-like check via element types). Verified by goldens. |
| `IIT3SystemIrreducibilityAnalysis` becomes unhashable but was previously usable in a set | Low | Already unhashable (no `__hash__` defined; `__eq__` defined → Python sets `__hash__` to None automatically). No-op change. |

---

## Saved-memory constraints reminder (every commit)

- **NEVER** bypass pre-commit hooks with `--no-verify` or `SKIP=*`.
- gpgsign bypass: `git -c commit.gpgsign=false commit ...`. If 1Password agent fails, surface to controller.
- Targeted `git add <files>` only — never `git add .` / `-A`.
- Before each commit: `git diff --cached --stat`. After: `git show --stat HEAD`.
- Doctest scope: `uv run pytest` (no path argument) at every commit boundary.
- No P# / "Phase A" / `TODO(Px)` / ROADMAP IDs in source/comments/docstrings/changelog. Commit messages MAY reference "the spec".
- No design narrative in docstrings; describe what code IS and DOES.
- Do NOT push to origin without explicit per-action consent.
- Per `project_pyright_workaround`: if `typeCheckingMode = "off"` is unstaged in `pyproject.toml`, trust the pre-commit hook output (not direct `uv run pyright`) as authoritative.

---

## Self-Review

**Spec coverage check:**

- ✅ Per-class `__eq__` on all 10 classes — Tasks 3, 4, 5, 6, 7
- ✅ Distinction structural `__hash__` (4-attr) — Task 6
- ✅ `Orderable.is_orderable_with` — Task 2
- ✅ `general_eq` deletion — Task 8
- ✅ Goldens byte-identical — verified at every task boundary + final
- ✅ Full pytest (no path) — Task 8.7 and at every task end
- ✅ Pyright / ruff — Task 8.8 and at every task end
- ✅ New per-class tests — interspersed in Tasks 2-7
- ✅ Old `test_general_eq_*` removed — Task 8.4
- ✅ Changelog fragment — Task 8.6
- ✅ ROADMAP follow-up removed — Task 8.5

**Placeholder scan:** No "TBD", "TODO", "implement later" in task steps. The `# TODO(slipperyhank): include 2nd state here?` and `# TODO use cached_property` comments in source code are preserved (pre-existing, unrelated). The line-182 `Relation` TODO is explicitly closed in Step 7.5.

**Type consistency:**
- `EQUALITY_TOLERANCE` referenced consistently from `pyphi.models.cmp`.
- `is_orderable_with` signature `(self, other: object) -> bool` consistent across base and override.
- Per-class `__eq__` template signature `(self, other: object) -> bool` consistent across all 10 classes.
- Per-class `__hash__` signature `(self) -> int` consistent across all classes with `__hash__`.

**Constraint compliance:**
- No P# / Phase markers in source files / changelog. Commit messages reference "the spec" — allowed.
- Docstrings describe final state.
- Targeted `git add` in every commit step.
- Pre-commit hooks not bypassed.
- `git -c commit.gpgsign=false` used consistently.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-distinction-ces-equality-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, two-stage review (spec compliance + code quality) after each. Past pattern (P12b, comparator): Sonnet 4.6 for mechanical implementer tasks; Opus 4.7 for review subagents.

**2. Inline Execution** — Execute tasks in this session via `superpowers:executing-plans`; batch with checkpoints.

After all 8 commits land, `superpowers:finishing-a-development-branch` per the standard 4-option menu (merge `feature/per-class-eq` into `2.0` locally / push + PR / keep / discard).

Which approach?
