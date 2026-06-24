# Cross-formalism SIA / CES surface unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the result-object surface across IIT 3.0 and IIT 4.0 so that `sia()` and `ces()` return structurally parallel types, render via a common template, and serialize to a single canonical JSON shape.

**Architecture:** Three commit clusters. Cluster A restructures IIT 3.0 to adopt IIT 4.0's CES-wraps-SIA topology (3.0 SIA drops `.ces` / `.substrate`, renames `.partitioned_ces` → `.partitioned_distinctions`; `iit3.ces()` returns `CauseEffectStructure(sia, distinctions, relations=NullRelations())`). Cluster B audits remaining field-name divergences (mostly no-op). Cluster C introduces `SIAInterface` / `CauseEffectStructureInterface` / `AcSIAInterface` Protocols, common `__repr__` / `_repr_html_` helpers, the canonical JSON shape (via existing `CLASS_KEY` discriminator), and an `__eq__` audit that makes cross-class comparisons return `NotImplemented`.

**Tech Stack:** PyPhi 2.0, Python 3.13+, `typing.Protocol` with `runtime_checkable`, existing `pyphi.jsonify` infrastructure (no msgspec yet — P15 adopts the same shape later).

**Spec:** `docs/superpowers/specs/2026-05-16-cross-formalism-sia-ces-unification-design.md` (committed at `e699d031`).

---

## Working constraints (apply to every task)

- Pre-commit hooks (ruff + ruff format + pyright + towncrier-check) gate every commit. NEVER bypass with `--no-verify`. If a hook fails, investigate the actual diagnostic (run `uv run ruff check <file>` / `uv run pyright <file>` directly) and fix the root cause.
- gpgsign uses 1Password agent. Use `git -c commit.gpgsign=false commit ...` per established session pattern. If a commit fails with "1Password: agent returned an error", surface to the user rather than auto-bypassing.
- Do NOT push without explicit per-action consent.
- Targeted `git add <files>` only. NEVER `git add .` or `git add -A`. The repo has ~18 unstaged tracked-file changes and 20+ untracked items that must NOT be staged.
- Before each `git commit`: run `git diff --cached --stat` and confirm only the intended files are staged. After each commit: run `git show --stat HEAD` to confirm. If extra files leak in via pre-commit's stash-restore, `git reset --soft HEAD~1` + `git reset HEAD <unrelated>` + recommit.
- No P# markers / "Phase A" / "Phase B" / "Phase C" / `TODO(Px)` / ROADMAP IDs in source, comments, docstrings, or changelog fragments. Commit messages MAY reference "the spec" (which is a real doc path).
- No design-narrative in docstrings. No "this approach is better because..." commentary. Describe final state, not migration history.
- No back-compat shims (re-exports, aliases) bridging the dropped fields or renamed types. Internal callers update in the same commit cluster.
- Use Monitor / Bash `run_in_background` for the slow lane and full-suite goldens regeneration (per saved memory `feedback_monitor_for_long_tests`).

---

## File map (all changes)

**Create:**
- `pyphi/models/protocols.py` — `SIAInterface`, `CauseEffectStructureInterface`, `AcSIAInterface`.
- `test/test_result_protocols.py` — Protocol conformance + round-trip JSON + cross-formalism `__eq__` + common-repr tests.
- Changelog fragments: `changelog.d/iit3-ces-restructure.change.md`, `changelog.d/sia-ces-unified-surface.change.md`.

**Modify:**
- `pyphi/relations.py` — add `NullRelations`.
- `pyphi/formalism/iit3/__init__.py` — rename internal `ces` → `_compute_distinctions`; new public `ces()` returns `CauseEffectStructure(...)`.
- `pyphi/formalism/iit4/__init__.py` — update internal call from `iit3.ces` → `iit3._compute_distinctions`; route `_repr_columns` through shared helper.
- `pyphi/models/sia.py` — drop `ces`, `substrate` fields; rename `partitioned_ces` → `partitioned_distinctions`; drop `unorderable_unless_eq=["substrate"]`; route `__repr__` through `fmt_sia_columns`; `__eq__` returns `NotImplemented` on type mismatch; add `_repr_html_`.
- `pyphi/models/ces.py` — route `__repr__` through `fmt_ces_columns`; add `_repr_html_`; `__eq__` returns `NotImplemented` on type mismatch.
- `pyphi/models/actual_causation.py` — route `__repr__` through `fmt_ac_sia_columns`; add `_repr_html_`; `__eq__` returns `NotImplemented` on type mismatch.
- `pyphi/models/fmt.py` — `fmt_sia_columns`, `fmt_ces_columns`, `fmt_ac_sia_columns`; `_repr_html_` helpers; update existing `fmt_sia` to not access `sia.ces`.
- `pyphi/models/__init__.py` — export `NullRelations` and Protocols.
- `pyphi/jsonify.py` — add `NullRelations` to `_loadable_models()`; document canonical shape in `jsonify` module docstring.
- `test/data/golden/v1/basic_iit3_emd*.json`, `xor_iit3_emd*.json` — regenerated after Cluster A.
- `test/data/sia/*.json` — regenerated for IIT 3.0 fixtures after Cluster A.
- All 17 goldens in `test/data/golden/v1/*.json` — regenerated after Cluster C (JSON shape changes).
- Test callers of `sia.ces` / `sia.partitioned_ces` / `sia.substrate` / `iit3.ces` — listed in tasks below.

---

## Cluster A — Structural restructure (Tasks 1-8)

### Task 1: Add `NullRelations`

**Files:**
- Modify: `pyphi/relations.py`
- Modify: `pyphi/models/__init__.py` (or relevant export module — check actual import surface)
- Test: `test/test_relations.py`

- [ ] **Step 1: Write the failing test**

Add to `test/test_relations.py`:

```python
def test_null_relations_is_empty():
    """NullRelations has zero phi, zero relations, empty iteration."""
    from pyphi.relations import NullRelations

    nr = NullRelations()
    assert nr.sum_phi() == 0
    assert nr.num_relations() == 0
    assert list(nr) == []


def test_null_relations_to_json_round_trips():
    from pyphi import jsonify
    from pyphi.relations import NullRelations

    nr = NullRelations()
    encoded = jsonify.loads(jsonify.dumps(nr))
    assert isinstance(encoded, NullRelations)
    assert encoded.sum_phi() == 0
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_relations.py::test_null_relations_is_empty test/test_relations.py::test_null_relations_to_json_round_trips -v`
Expected: FAIL — `NullRelations` not defined.

- [ ] **Step 3: Implement `NullRelations`**

In `pyphi/relations.py` after the `Relations` base class (after line 293):

```python
class NullRelations(Relations):
    """An empty set of relations specified by a substrate whose formalism
    does not define relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __iter__(self):
        return iter(())

    def _sum_phi(self):
        return 0

    def _num_relations(self):
        return 0

    def to_json(self):
        return {"relations": []}

    @classmethod
    def from_json(cls, data):
        return cls()
```

- [ ] **Step 4: Register `NullRelations` in the jsonify registry**

In `pyphi/jsonify.py:77` `_loadable_models()`, add to the `classes` list:

```python
pyphi.relations.NullRelations,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_relations.py -v -k "null_relations"`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyphi/relations.py pyphi/jsonify.py test/test_relations.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add NullRelations for substrates whose formalism does not define relations

NullRelations is a Relations subclass with empty iteration, sum_phi()=0,
and num_relations()=0. Used by IIT 3.0's CauseEffectStructure (which the
spec restructures iit3.ces() to return) since IIT 3.0 has no relations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 2: Rename internal `iit3.ces` → `_compute_distinctions`

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py` (rename function, update internal call sites)
- Modify: `pyphi/formalism/iit4/__init__.py:929` (call from `iit3.ces` → `iit3._compute_distinctions`)

This task moves the current bag-of-concepts function to a private name so Task 3 can introduce a new public `ces()` that returns `CauseEffectStructure`. No behavior change; pure rename.

- [ ] **Step 1: Rename the function in `pyphi/formalism/iit3/__init__.py`**

Find `def ces(` at line 91 in `pyphi/formalism/iit3/__init__.py`. Rename to `def _compute_distinctions(`. The function body is unchanged. Update the docstring's first line:

```python
def _compute_distinctions(
    system: System,
    mechanisms: Iterable[Mechanism] | None = None,
    purviews: Iterable[Purview] | None = None,
    cause_purviews: Iterable[Purview] | None = None,
    effect_purviews: Iterable[Purview] | None = None,
    directions: Iterable[Direction] | None = None,
    only_positive_phi: bool = True,
    **kwargs: Any,
) -> Distinctions:
    """Compute the bag of distinctions for a system, restricted by the
    given mechanism / purview / direction filters.
    """
    # ... existing body unchanged ...
```

- [ ] **Step 2: Update internal callers in `pyphi/formalism/iit3/__init__.py`**

Find every `ces(` call inside `pyphi/formalism/iit3/__init__.py` that targets the local `ces` function. Replace with `_compute_distinctions(`. (Do not touch calls to `ces_distance` or external `ces` references.)

Run: `grep -n "ces(" pyphi/formalism/iit3/__init__.py`

Replace each match where the target is the local function. Likely sites (verify each before replacing):
- `_sia()` body — computes unpartitioned distinctions
- `evaluate_partition()` body (~line 217) — `partitioned_ces = ces(partitioned_system, mechanisms, **kwargs)`

After replacement, the partitioned bag construction becomes:
```python
partitioned_ces = _compute_distinctions(partitioned_system, mechanisms, **kwargs)
```

- [ ] **Step 3: Update 4.0's call site**

In `pyphi/formalism/iit4/__init__.py:929`:

```python
# Old:
distinctions = iit3.ces(system, **ces_kwargs)
# New:
distinctions = iit3._compute_distinctions(system, **ces_kwargs)
```

- [ ] **Step 4: Run the targeted tests to verify the rename didn't break anything**

Run: `uv run pytest test/test_complexes.py test/test_big_phi.py test/test_measures_ces.py -x -q`
Expected: tests pass (rename is behavior-preserving).

Note: tests that call `iit3.ces(...)` from external test code (e.g., `test/test_complexes.py:115-145`, `test/test_relations.py:54`, `test/test_json.py:65`, `test/test_big_phi.py:301-316`) will START FAILING with `AttributeError: module 'pyphi.formalism.iit3' has no attribute 'ces'`. That is expected and fixed in Task 3, which adds the new public `ces` function. Do not update those tests in this task.

If you want to verify the rename worked in isolation, run pyright instead:

Run: `uv run pyright pyphi/formalism/iit3/__init__.py pyphi/formalism/iit4/__init__.py`
Expected: 0 new errors.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit3/__init__.py pyphi/formalism/iit4/__init__.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Rename internal iit3.ces to _compute_distinctions

Per the spec, iit3.ces() will be replaced with a public function that
returns a CauseEffectStructure wrapping (sia, distinctions,
relations=NullRelations()). This rename frees the public name without
changing behavior; the bag-of-distinctions builder is now private and
named to match its responsibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 3: New public `iit3.ces()` returns `CauseEffectStructure`

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py` (add new `ces()` function)
- Test: `test/test_complexes.py` (existing call sites should still work; add a new test asserting the wrapper shape)

- [ ] **Step 1: Write the failing test**

Add to `test/test_complexes.py`:

```python
def test_iit3_ces_returns_cause_effect_structure(s):
    """iit3.ces returns a CauseEffectStructure with sia, distinctions, and
    NullRelations.
    """
    from pyphi.conf import presets
    from pyphi.formalism import iit3
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.relations import NullRelations

    with config.override(**presets.iit3):
        ces = iit3.ces(s)

    assert isinstance(ces, CauseEffectStructure)
    assert ces.sia is not None
    assert isinstance(ces.relations, NullRelations)
    assert ces.relations.num_relations() == 0
    assert ces.distinctions is not None
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_complexes.py::test_iit3_ces_returns_cause_effect_structure -v`
Expected: FAIL — `module 'pyphi.formalism.iit3' has no attribute 'ces'` (Task 2 removed it).

- [ ] **Step 3: Implement the new public `ces()`**

In `pyphi/formalism/iit3/__init__.py`, after `_compute_distinctions` (the renamed function from Task 2), add:

```python
def ces(
    system: System,
    *,
    sia: IIT3SystemIrreducibilityAnalysis | None = None,
    distinctions: Distinctions | None = None,
    sia_kwargs: dict | None = None,
    distinctions_kwargs: dict | None = None,
) -> CauseEffectStructure:
    """Compute the cause-effect structure of a system under IIT 3.0.

    Returns a :class:`CauseEffectStructure` wrapping the SIA, the resolved
    distinctions, and an empty :class:`NullRelations` (IIT 3.0 does not
    define relations between distinctions).

    Pass ``sia=`` or ``distinctions=`` to reuse pre-computed values.
    """
    sia_kwargs = sia_kwargs or {}
    distinctions_kwargs = distinctions_kwargs or {}

    if sia is None:
        sia = _sia(system, **sia_kwargs)
    if distinctions is None:
        distinctions = _compute_distinctions(system, **distinctions_kwargs)

    # IIT 3.0 has no per-distinction ties; treat distinctions as vacuously
    # resolved without going through resolve_congruence (which IIT 4.0 uses
    # for tie disambiguation against a SIA system_state).
    from pyphi.models.distinctions import ResolvedDistinctions
    if not isinstance(distinctions, ResolvedDistinctions):
        distinctions = ResolvedDistinctions(distinctions)

    return CauseEffectStructure(
        sia=sia,
        distinctions=distinctions,
        relations=NullRelations(),
    )
```

Add the necessary imports at the top of the file:

```python
from pyphi.models.ces import CauseEffectStructure
from pyphi.relations import NullRelations
```

- [ ] **Step 4: Run the failing test to verify it now passes**

Run: `uv run pytest test/test_complexes.py::test_iit3_ces_returns_cause_effect_structure -v`
Expected: PASS.

- [ ] **Step 5: Update external test call-sites that expect `iit3.ces` to return a bag**

The external test callers from Task 2's note now need to handle the new wrapper return type. Update each to read `.distinctions`:

In `test/test_complexes.py:115-145` (the `test_sia_ces_consistency` test that compared `sia.ces` to `iit3.ces(s)`):

```python
# Old:
standalone_ces = iit3.ces(system)
# ... later compared against sia.ces ...

# New: read .distinctions from the wrapper
standalone_distinctions = iit3.ces(system).distinctions
# ... compared against sia.ces if it still exists this commit, or the
# wrapping ces.distinctions after Task 4 ...
```

Update `test/test_big_phi.py:301-316`, `test/test_json.py:65`, `test/test_relations.py:54` similarly: `iit3.ces(x)` → `iit3.ces(x).distinctions` if a bag is wanted, or test the wrapper directly if the new shape is the goal.

For now (before Task 4 drops `sia.ces`), the `test_complexes.py::test_sia_ces_consistency` test still reads `sia.ces` on one side — it will start failing in Task 4. Mark its expected behavior to-be-updated in Task 4.

- [ ] **Step 6: Run the affected tests**

Run: `uv run pytest test/test_complexes.py test/test_big_phi.py test/test_json.py test/test_relations.py -x -q -k "ces"`
Expected: tests pass (with `iit3.ces` now returning the wrapper; consumers updated to `.distinctions`).

- [ ] **Step 7: Commit**

```bash
git add pyphi/formalism/iit3/__init__.py test/test_complexes.py test/test_big_phi.py test/test_json.py test/test_relations.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Make iit3.ces return CauseEffectStructure

Per the spec, IIT 3.0 adopts IIT 4.0's CES-wraps-SIA topology. iit3.ces
now returns a CauseEffectStructure(sia, distinctions, NullRelations());
the bag-of-distinctions builder is the private _compute_distinctions.

Test callers that previously expected a bag now read the wrapper's
.distinctions attribute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 4: Drop `ces`, `substrate`, rename `partitioned_ces` → `partitioned_distinctions` on `IIT3SystemIrreducibilityAnalysis`

**Files:**
- Modify: `pyphi/models/sia.py` (the class definition, `_sia_attributes`, `unorderable_unless_eq`, `to_json`, `from_json`, `__hash__`, `__str__`)
- Modify: `pyphi/formalism/iit3/__init__.py` (`_sia` construction, `_null_sia`, `evaluate_partition`)
- Modify: `pyphi/models/fmt.py:807-818` (`fmt_sia` accesses `sia.ces` / `sia.partitioned_ces`)

- [ ] **Step 1: Write the failing test**

Add to `test/test_complexes.py` (or a new `test/test_iit3_sia_shape.py`):

```python
def test_iit3_sia_no_longer_carries_unpartitioned_ces(s):
    """IIT 3.0 SIA stores partitioned_distinctions but not ces or substrate.

    The unpartitioned distinctions live on the wrapping CauseEffectStructure.
    """
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia = iit3.sia(s)

    assert not hasattr(sia, "ces"), "sia.ces removed; access via iit3.ces(s).distinctions"
    assert not hasattr(sia, "substrate"), "sia.substrate removed; callers hold it externally"
    assert not hasattr(sia, "partitioned_ces"), "renamed to partitioned_distinctions"
    assert hasattr(sia, "partitioned_distinctions"), "compute receipt of the MIP"
    assert sia.partitioned_distinctions is not None
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_complexes.py::test_iit3_sia_no_longer_carries_unpartitioned_ces -v`
Expected: FAIL — `sia` still has `.ces`, `.substrate`, `.partitioned_ces`.

- [ ] **Step 3: Update `IIT3SystemIrreducibilityAnalysis` class**

In `pyphi/models/sia.py`:

Replace the `_sia_attributes` list (line 20-29):

```python
_sia_attributes = [
    "phi",
    "partitioned_distinctions",
    "partition",
    "node_indices",
    "node_labels",
    "current_state",
]
```

Update `__init__` (line 64-101):

```python
def __init__(
    self,
    phi=None,
    partitioned_distinctions=None,
    partition=None,
    node_indices=None,
    node_labels=None,
    current_state=None,
    config=None,
):
    if phi is None:
        self.phi = phi  # type: ignore[assignment]
    else:
        from pyphi.data_structures.pyphi_float import PyPhiFloat
        from pyphi.measures.distribution import DistanceResult

        if isinstance(phi, DistanceResult):
            self.phi = phi  # type: ignore[assignment]
        else:
            self.phi = PyPhiFloat(phi)  # type: ignore[assignment]
    self.partitioned_distinctions = partitioned_distinctions
    self.partition = partition
    self.node_indices = node_indices
    self.node_labels = node_labels
    self.current_state = current_state
    if config is None:
        from pyphi.conf import config as _global

        config = _global.snapshot()
    self.config = config
```

Drop the `unorderable_unless_eq = ["substrate"]` class var (line 114):

```python
unorderable_unless_eq: ClassVar[list[str]] = []
```

Update `__hash__` (line 125-136) to drop `ces`, `partitioned_ces`, `substrate`:

```python
def __hash__(self):
    return hash(
        (
            self.phi,
            self.partitioned_distinctions,
            self.partition,
            self.node_indices,
            self.current_state,
        )
    )
```

Update `__str__` to not access `self.ces`:

```python
def __str__(self):
    return fmt.fmt_sia(self)

def print(self):
    """Print this SystemIrreducibilityAnalysis."""
    print(str(self))
```

(`fmt.fmt_sia` is updated below.)

- [ ] **Step 4: Update `_null_sia` in `pyphi/models/sia.py`**

Replace `_null_sia` (line 153-168):

```python
def _null_sia(system, phi=0.0):
    """Return an IIT3SystemIrreducibilityAnalysis with zero phi.

    This is the analysis result for a reducible system.
    """
    return IIT3SystemIrreducibilityAnalysis(
        phi=phi,
        partitioned_distinctions=_null_ces(),
        partition=system.partition,
        node_indices=system.node_indices,
        node_labels=system.substrate.node_labels,
        current_state=system.state,
    )
```

- [ ] **Step 5: Update `_sia` construction in `pyphi/formalism/iit3/__init__.py:223-232`**

```python
return IIT3SystemIrreducibilityAnalysis(
    phi=phi_,
    partitioned_distinctions=partitioned_ces,
    partition=partitioned_system.partition,
    node_indices=unpartitioned_system.node_indices,
    node_labels=unpartitioned_system.substrate.node_labels,
    current_state=unpartitioned_system.state,
)
```

(The local variable `partitioned_ces` keeps its name — it's an `evaluate_partition` local, not a field name.)

- [ ] **Step 6: Update `fmt.fmt_sia` in `pyphi/models/fmt.py:807-818`**

The old function rendered `sia.ces` and `sia.partitioned_ces`. After Task 4, neither exists on the IIT 3.0 SIA. Replace the body:

```python
def fmt_sia(
    sia: object, title: str = "System irreducibility analysis"
) -> str:
    """Format an IIT 3.0 SystemIrreducibilityAnalysis as a multi-line
    summary including phi, partition, and node info.
    """
    node_indices = sia.node_indices  # type: ignore[attr-defined]
    node_labels = sia.node_labels  # type: ignore[attr-defined]
    if node_labels is not None and node_indices is not None:
        system_label = ",".join(
            str(label) for label in node_labels.coerce_to_labels(node_indices)
        )
    elif node_indices is not None:
        system_label = ",".join(str(i) for i in node_indices)
    else:
        system_label = ""

    data = [
        f"{BIG_PHI}: {fmt_number(sia.phi)}",  # type: ignore[attr-defined]
        system_label,
        sia.partition,  # type: ignore[attr-defined]
    ]
    body = ""
    for line in reversed(data):
        body = header(str(line), body)
    body = header(title, body, under_char=HEADER_BAR_2)
    return box(center(body))
```

(Drops the `ces` kwarg from the signature — callers using `fmt_sia(sia, ces=False)` need updating; check `test/test_helpers.py` and other diff sources for any.)

- [ ] **Step 7: Update IIT 3.0 SIA `to_json` / `from_json` (already attribute-driven; verify still correct)**

`to_json` (line 144-146) reads from `_sia_attributes`. Since `_sia_attributes` was updated, this is automatic.

`from_json` (line 148-150) calls `cls(**dct)`. With the new `__init__` signature, old fixtures with `ces=...` / `substrate=...` / `partitioned_ces=...` in their dict will raise `TypeError: unexpected keyword argument`. This is expected; fixtures regenerate in Task 7.

- [ ] **Step 8: Run the targeted tests**

Run: `uv run pytest test/test_complexes.py::test_iit3_sia_no_longer_carries_unpartitioned_ces -v`
Expected: PASS.

Run: `uv run pytest test/test_complexes.py test/test_measures_ces.py -x -q`
Expected: many tests fail — they read `sia.ces` / `sia.partitioned_ces` / `sia.substrate`. Don't update those tests here; that's Task 5. Verify the failures are the expected ones.

Run: `uv run pyright pyphi/models/sia.py pyphi/formalism/iit3/__init__.py pyphi/models/fmt.py`
Expected: 0 new errors.

- [ ] **Step 9: Commit**

```bash
git add pyphi/models/sia.py pyphi/formalism/iit3/__init__.py pyphi/models/fmt.py test/test_complexes.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Drop ces/substrate fields from IIT 3.0 SIA; rename partitioned_ces

Per the spec, IIT 3.0 SIA no longer holds the unpartitioned cause-effect
structure (lives on the wrapping CauseEffectStructure.distinctions) nor
the substrate reference (callers hold it externally, matching IIT 4.0's
decoupling). The partitioned bag stays as a 3.0-specific compute
receipt under the name partitioned_distinctions.

unorderable_unless_eq=["substrate"] is now vacuous and removed; cross-
substrate phi comparisons become valid via phi alone (no test relies on
the guard).

fmt_sia drops its ces kwarg; the SIA's __str__ no longer renders the
CES inline (use iit3.ces(system) for the full structure).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 5: Update test consumers reading `sia.ces` / `sia.partitioned_ces` / `sia.substrate`

**Files:**
- Modify: `test/test_measures_ces.py:33-38` (reads `sia.ces`, `sia.partitioned_ces`)
- Modify: `test/test_complexes.py:115-145` (compares `sia.ces` to `iit3.ces(s)`)
- Modify: `test/golden/compute.py:244-248` (reads `sia.ces` / `sia.partitioned_ces` for golden capture)
- Modify: any other test caller (run grep below to enumerate)

- [ ] **Step 1: Enumerate consumers**

Run: `grep -rn "sia\.ces\b\|sia\.partitioned_ces\b\|sia\.substrate\b" pyphi/ test/ --include="*.py"`

Expected matches: ~5-10 sites. Catalogue each before editing.

- [ ] **Step 2: Update `test/test_measures_ces.py:27-40`**

The `test_ces_distances` test compares `sia.ces` against `sia.partitioned_ces` to verify EMD CES distance. Refactor to use `iit3.ces` + `iit3.sia` separately:

```python
def test_ces_distances(s):
    """Canonical IIT 3.0 CES distance for the basic substrate (1,0,0)."""
    with config.override(**presets.iit3):
        ces = iit3.ces(s)

    with config.override(**presets.iit3, ces_measure="EMD"):
        assert ces_distance(
            ces.distinctions, ces.sia.partitioned_distinctions, system=s
        ) == pytest.approx(2.3125, rel=1e-6)

    with config.override(**presets.iit3, ces_measure="SUM_SMALL_PHI"):
        assert ces_distance(
            ces.distinctions, ces.sia.partitioned_distinctions
        ) == pytest.approx(1.083333, rel=1e-6)
```

- [ ] **Step 3: Update `test/test_complexes.py:115-145`**

The `test_sia_ces_consistency` test compared `iit3.sia(s).ces` to `iit3.ces(s)`. After restructure, the meaningful invariant is `iit3.ces(s).distinctions == iit3.ces(s).distinctions` (consistency of the public API across two calls) plus `iit3.ces(s).sia.phi == iit3.sia(s).phi`. Rewrite:

```python
def test_sia_phi_matches_ces_sia_phi(self):
    """The SIA returned from iit3.ces matches iit3.sia for every fixture."""
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        for system_factory in CANONICAL_FIXTURES:
            system = system_factory()
            ces_sia_phi = iit3.ces(system).sia.phi
            direct_sia_phi = iit3.sia(system).phi
            assert ces_sia_phi == direct_sia_phi, (
                f"iit3.ces(s).sia.phi diverged from iit3.sia(s).phi for "
                f"{system!r}: {ces_sia_phi} vs {direct_sia_phi}"
            )
```

(Verify `CANONICAL_FIXTURES` is the correct name in the file; adjust to whatever the test file uses.)

- [ ] **Step 4: Update `test/golden/compute.py:244-248`**

The golden compute helper records `ces_size`, `ces_phi_sum`, `partitioned_ces_size`, `partitioned_ces_phi_sum` from the SIA. After Task 4, these come from different places:

```python
# Old:
if hasattr(sia, "ces") and sia.ces is not None:
    out["ces_size"] = len(sia.ces)
    out["ces_phi_sum"] = float(sum(c.phi for c in sia.ces))
if hasattr(sia, "partitioned_ces") and sia.partitioned_ces is not None:
    out["partitioned_ces_size"] = len(sia.partitioned_ces)
    # ... etc ...

# New: for IIT 3.0, the partitioned bag is on the SIA; the unpartitioned
# is on the wrapping CES, which the caller has to provide separately if
# the golden fixture wants it captured.
if hasattr(sia, "partitioned_distinctions") and sia.partitioned_distinctions is not None:
    out["partitioned_distinctions_size"] = len(sia.partitioned_distinctions)
    out["partitioned_distinctions_phi_sum"] = float(
        sum(d.phi for d in sia.partitioned_distinctions)
    )
```

The unpartitioned bag for 3.0 goldens would now be captured from `iit3.ces(system).distinctions` at a different point in `test/golden/compute.py`. Inspect the surrounding `compute_all_layers` function to find the natural place to capture it (likely from the `ces_obj` variable if one exists, or by adding a separate `iit3.ces()` call).

- [ ] **Step 5: Run the touched test files**

Run: `uv run pytest test/test_measures_ces.py test/test_complexes.py -x -q`
Expected: pass after the rewrites.

- [ ] **Step 6: Commit**

```bash
git add test/test_measures_ces.py test/test_complexes.py test/golden/compute.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Update test consumers for IIT 3.0 SIA shape change

Tests reading sia.ces / sia.partitioned_ces / sia.substrate update to
the post-restructure surface: unpartitioned distinctions via
iit3.ces(s).distinctions, partitioned distinctions via
sia.partitioned_distinctions, substrate via the caller's own reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 6: Audit `OrderableByPhi.unorderable_unless_eq` consumers

**Files:**
- Modify: `pyphi/models/cmp.py` (verify the equality-guard mechanism still works after removing `substrate`)
- Inspection only on consumers

- [ ] **Step 1: Grep for `unorderable_unless_eq` consumers**

Run: `grep -rn "unorderable_unless_eq" pyphi/ test/ --include="*.py"`

Expected: a small list of definitions plus the consumer in `pyphi/models/cmp.py`.

- [ ] **Step 2: Read `pyphi/models/cmp.py` Orderable comparison code**

Confirm the comparison guard reads the field names from `unorderable_unless_eq` and rejects comparison if any of those fields differ between the two operands. With `unorderable_unless_eq=[]` (the new value after Task 4), all SIAs of the same class become orderable by phi alone.

- [ ] **Step 3: Verify the comparison still raises sensibly when expected**

Add to `test/test_models.py` (or wherever sia comparison tests live):

```python
def test_iit3_sia_orderable_across_substrates(s, micro_s):
    """Without the substrate guard, two IIT 3.0 SIAs from different
    substrates are now comparable by phi alone.
    """
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia_a = iit3.sia(s)
        sia_b = iit3.sia(micro_s)

    # Comparison must not raise; result is just phi-ordering
    assert (sia_a.phi == sia_b.phi) == (sia_a == sia_b)
    assert (sia_a < sia_b) == (sia_a.phi < sia_b.phi)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest test/test_models.py::test_iit3_sia_orderable_across_substrates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/test_models.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Pin IIT 3.0 SIA cross-substrate ordering after substrate-guard removal

Sister to the SIA-field drop: the unorderable_unless_eq=['substrate']
guard previously rejected comparisons between SIAs from different
substrates. With substrate dropped from the SIA field set, the guard is
removed and comparisons fall back to phi-ordering. This test pins that
behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 7: Regenerate IIT 3.0 EMD goldens + SIA fixtures

**Files:**
- Regenerate: `test/data/golden/v1/basic_iit3_emd.json`, `basic_iit3_emd_tri.json`, `basic_subset_iit3_emd.json`, `xor_iit3_emd.json`
- Regenerate: `test/data/sia/*.json` (any IIT 3.0 fixtures — likely none if all are IIT 4.0; verify)

- [ ] **Step 1: Identify IIT 3.0 fixtures**

Run: `grep -l "IIT3SystemIrreducibilityAnalysis\|IIT_3_0" test/data/sia/*.json 2>/dev/null`
Run: `ls test/data/golden/v1/*iit3*.json`

Catalogue the files that need regeneration.

- [ ] **Step 2: Regenerate goldens**

Run: `uv run pytest test/test_golden_regression.py --regenerate-golden -k "iit3_emd"`
Expected: SKIP entries for the 4 IIT 3.0 EMD fixtures (regenerated).

- [ ] **Step 3: Inspect the diff**

Run: `git diff --stat test/data/golden/v1/`

Verify only the 4 IIT 3.0 EMD JSON files changed.

Run: `git diff test/data/golden/v1/basic_iit3_emd.json | head -80`

Verify the diff shape: `partitioned_ces` field renamed to `partitioned_distinctions`, `substrate` field gone, `ces` field gone. The phi value MUST be unchanged (the canonical 2.3125 for `basic_iit3_emd`).

- [ ] **Step 4: If `test/data/sia/*.json` has IIT 3.0 fixtures, regenerate them**

If Step 1 found any: these fixtures are loaded by `s_expected_sia`, `s_noised_expected_sia`, `rule152_s_expected_sia`, etc. fixtures in `test/conftest.py`. They predate the golden harness and are written by hand or by a separate script. Check `test/IIT_4.0_make_jsons.ipynb` (or similar) for the generation entry point; if IIT 3.0 versions exist, regenerate from there. If they're entirely IIT 4.0, this step is a no-op.

- [ ] **Step 5: Run the golden suite to confirm pass**

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: 17/17 pass.

- [ ] **Step 6: Commit**

```bash
git add test/data/golden/v1/basic_iit3_emd.json test/data/golden/v1/basic_iit3_emd_tri.json test/data/golden/v1/basic_subset_iit3_emd.json test/data/golden/v1/xor_iit3_emd.json
# Add test/data/sia/*.json files only if Step 1 identified any IIT 3.0 fixtures there
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Regenerate IIT 3.0 EMD goldens for new SIA field shape

The 4 IIT 3.0 EMD golden fixtures regenerate against the post-
restructure SIA: ces and substrate fields gone, partitioned_ces renamed
to partitioned_distinctions. Phi values unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 8: Cluster A changelog fragment

**Files:**
- Create: `changelog.d/iit3-ces-restructure.change.md`

- [ ] **Step 1: Write the changelog fragment**

```markdown
IIT 3.0 ``iit3.ces()`` now returns a
:class:`~pyphi.models.ces.CauseEffectStructure` wrapping the SIA, the
distinctions, and an empty :class:`~pyphi.relations.NullRelations`.
This matches IIT 4.0's topology where the CES is the dominant outer
container; previously IIT 3.0 returned a bare ``UnresolvedDistinctions``
and the SIA stored ``ces`` / ``partitioned_ces`` fields directly.

Related changes on :class:`~pyphi.models.sia.IIT3SystemIrreducibilityAnalysis`:

- ``ces`` field removed. Read the unpartitioned distinctions from
  ``iit3.ces(system).distinctions``.
- ``substrate`` field removed. Callers hold the substrate reference
  externally, matching IIT 4.0's decoupled SIA.
- ``partitioned_ces`` renamed to ``partitioned_distinctions``. This bag
  stays on the SIA as the IIT 3.0-specific compute receipt — IIT 3.0
  phi is computed as ``ces_distance(unpartitioned, partitioned,
  system)``, so the partitioned bag is intrinsic to the computation.
- ``unorderable_unless_eq=["substrate"]`` removed. Comparisons between
  SIAs from different substrates now fall back to phi-ordering.

The :func:`~pyphi.models.fmt.fmt_sia` helper no longer accepts a ``ces``
kwarg; the SIA's ``__str__`` renders just the SIA fields. Use
``iit3.ces(system)`` to render the full cause-effect structure.

The 4 IIT 3.0 EMD golden fixtures regenerated with the new shape.
``sia.phi`` values are unchanged.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.d/iit3-ces-restructure.change.md
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add changelog fragment for iit3.ces restructure
EOF
)"
git show --stat HEAD
```

---

### Cluster A acceptance gate

After Task 8, run the targeted test set and verify:

```bash
uv run pytest test/test_big_phi.py test/test_complexes.py test/test_measures_ces.py test/test_golden_regression.py test/test_result_config_snapshot.py test/test_relations.py -x -q
```

Expected: all passing.

```bash
uv run pyright pyphi/
```

Expected: 0 errors / 1 baseline warning.

```bash
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

Expected: clean.

If any check is red, diagnose root cause per saved memory `feedback_dont_give_up_on_architectural_refactors` before continuing to Cluster B/C.

---

## Cluster B — Field rename audit (Task 9; may fold into Cluster C)

### Task 9: Audit `_sia_attributes` consistency between formalisms

**Files:**
- Inspection: `pyphi/models/sia.py:20-29` (`_sia_attributes`), `pyphi/formalism/iit4/__init__.py:184-196` (`_sia_attributes` ClassVar), `pyphi/models/actual_causation.py` (AcSIA attribute list)

- [ ] **Step 1: Tabulate the three attribute lists**

Read each `_sia_attributes` list. Confirm that the fields present in each correspond to the spec's "Shared and divergent fields" table. List discrepancies if any.

Likely no discrepancies (Cluster A's work already aligned 3.0). If one is found, fix it in a small commit:

```bash
# If a discrepancy exists, edit the affected file
# Then:
git add pyphi/models/sia.py  # or whichever
git -c commit.gpgsign=false commit -m "Reconcile _sia_attributes after iit3 restructure"
```

If no discrepancies, skip this task and proceed to Cluster C.

- [ ] **Step 2: Confirm AcSIA's `alpha` stays (not renamed to `phi`)**

Per the spec, AcSIA's `alpha` attribute name is preserved (matches Albantakis 2019 AC paper terminology). Verify `pyphi/models/actual_causation.py` still names it `alpha` and the `AcSIAInterface` Protocol (to be added in Task 10) will declare `alpha`.

This is a verification step — no code change. Move on to Cluster C.

---

## Cluster C — Protocols + common repr + canonical JSON + __eq__ audit (Tasks 10-17)

### Task 10: Add `pyphi/models/protocols.py`

**Files:**
- Create: `pyphi/models/protocols.py`
- Modify: `pyphi/models/__init__.py` (export the Protocols)
- Test: `test/test_result_protocols.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `test/test_result_protocols.py`:

```python
"""Cross-formalism Protocol conformance tests."""

from __future__ import annotations

import pytest

from pyphi import config
from pyphi.conf import presets


def test_iit3_sia_satisfies_sia_interface(s):
    from pyphi.formalism import iit3
    from pyphi.models.protocols import SIAInterface

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    assert isinstance(sia, SIAInterface)


def test_iit4_sia_satisfies_sia_interface(s):
    from pyphi.formalism import iit4
    from pyphi.models.protocols import SIAInterface

    sia = iit4.sia(s)
    assert isinstance(sia, SIAInterface)


def test_iit3_ces_satisfies_ces_interface(s):
    from pyphi.formalism import iit3
    from pyphi.models.protocols import CauseEffectStructureInterface

    with config.override(**presets.iit3):
        ces = iit3.ces(s)
    assert isinstance(ces, CauseEffectStructureInterface)


def test_iit4_ces_satisfies_ces_interface(s):
    from pyphi.formalism import iit4
    from pyphi.models.protocols import CauseEffectStructureInterface

    ces = iit4.ces(
        s,
        system_measure=...,  # use the canonical measure for the test fixture
        specification_measure=...,
    )
    assert isinstance(ces, CauseEffectStructureInterface)


def test_acsia_satisfies_acsia_interface(transition):
    from pyphi import actual
    from pyphi.models.protocols import AcSIAInterface

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)
    assert isinstance(acsia, AcSIAInterface)
```

(The `iit4.ces` test's measure args need the canonical IIT 4.0 preset values; copy from `presets.iit4_2023` or the test conventions.)

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_result_protocols.py -v`
Expected: FAIL — `pyphi.models.protocols` module does not exist.

- [ ] **Step 3: Create `pyphi/models/protocols.py`**

```python
"""Protocols declaring the cross-formalism surface of analysis results.

These Protocols use ``runtime_checkable`` so ``isinstance()`` works at
runtime; the declared attributes are the shared surface, not the full
field set of any concrete class. Formalism-specific extras
(IIT 4.0's ``normalized_phi`` / ``cause`` / ``effect`` / ``system_state``,
IIT 3.0's ``partitioned_distinctions``) live on the concrete classes and
are accessible via direct attribute access or ``isinstance()`` dispatch.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SIAInterface(Protocol):
    """The system-irreducibility analysis surface common to all formalisms.

    Implementations: :class:`pyphi.models.sia.IIT3SystemIrreducibilityAnalysis`,
    :class:`pyphi.formalism.iit4.SystemIrreducibilityAnalysis`.
    """

    phi: Any
    partition: Any
    current_state: tuple[int, ...] | None
    node_indices: tuple[int, ...] | None
    node_labels: Any
    config: Any

    def order_by(self) -> Any: ...
    def __bool__(self) -> bool: ...


@runtime_checkable
class CauseEffectStructureInterface(Protocol):
    """The cause-effect structure surface common to all formalisms.

    Implementations: :class:`pyphi.models.ces.CauseEffectStructure` (used
    by both IIT 3.0 and IIT 4.0; the relations field is empty for IIT 3.0).
    """

    sia: SIAInterface
    distinctions: Any
    relations: Any
    config: Any


@runtime_checkable
class AcSIAInterface(Protocol):
    """The actual-causation system-irreducibility analysis surface.

    Implementations: :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.

    Uses ``alpha`` rather than ``phi`` per the actual-causation paper
    (Albantakis et al. 2019).
    """

    alpha: Any
    direction: Any
    account: Any
    partitioned_account: Any
    partition: Any
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    node_indices: tuple[int, ...] | None
    cause_indices: tuple[int, ...] | None
    effect_indices: tuple[int, ...] | None
    node_labels: Any
    config: Any
```

- [ ] **Step 4: Export from `pyphi/models/__init__.py`**

Add to the module imports / `__all__`:

```python
from .protocols import AcSIAInterface
from .protocols import CauseEffectStructureInterface
from .protocols import SIAInterface
```

And include in `__all__` if the module has one.

- [ ] **Step 5: Run the failing tests to verify they now pass**

Run: `uv run pytest test/test_result_protocols.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/protocols.py pyphi/models/__init__.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add cross-formalism result-object Protocols

SIAInterface, CauseEffectStructureInterface, and AcSIAInterface declare
the shared surface of IIT 3.0 / IIT 4.0 / actual-causation result
objects. runtime_checkable so isinstance() dispatch works at runtime.

Conformance pinned via test/test_result_protocols.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 11: Add column-builder helpers in `fmt.py`

**Files:**
- Modify: `pyphi/models/fmt.py` (add `fmt_sia_columns`, `fmt_ces_columns`, `fmt_ac_sia_columns`)
- Test: `test/test_result_protocols.py` (extend with rendering tests)

- [ ] **Step 1: Write the failing test**

Add to `test/test_result_protocols.py`:

```python
def test_fmt_sia_columns_shared_keys(s):
    """fmt_sia_columns returns the same columns for both formalisms."""
    from pyphi.formalism import iit3, iit4
    from pyphi.models.fmt import fmt_sia_columns

    with config.override(**presets.iit3):
        cols_3 = dict(fmt_sia_columns(iit3.sia(s)))

    cols_4 = dict(fmt_sia_columns(iit4.sia(s)))

    shared = {"System", "Current state", "Partition"}
    assert shared.issubset(cols_3.keys())
    assert shared.issubset(cols_4.keys())
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_result_protocols.py::test_fmt_sia_columns_shared_keys -v`
Expected: FAIL — `fmt_sia_columns` not defined.

- [ ] **Step 3: Add `fmt_sia_columns` to `pyphi/models/fmt.py`**

After `fmt_sia`:

```python
def fmt_sia_columns(sia: object) -> list[tuple[str, Any]]:
    """Column list shared by every SIA __repr__: System, Current state,
    phi, Partition. Formalism-specific extras append on top in the
    concrete class's _repr_columns method.
    """
    node_indices = getattr(sia, "node_indices", None)
    node_labels = getattr(sia, "node_labels", None)
    if node_labels is not None and node_indices is not None:
        system_label = ",".join(
            str(label) for label in node_labels.coerce_to_labels(node_indices)
        )
    elif node_indices is not None:
        system_label = ",".join(str(i) for i in node_indices)
    else:
        system_label = None

    current_state = getattr(sia, "current_state", None)
    phi = getattr(sia, "phi", None)
    partition = getattr(sia, "partition", None)

    return [
        ("System", system_label),
        ("Current state", state(current_state) if current_state is not None else None),
        (f"           {SMALL_PHI}_s", phi),
        ("Partition", partition),
    ]


def fmt_ces_columns(ces: object) -> list[tuple[str, Any]]:
    """Column list for a CauseEffectStructure: Φ, #(distinctions), Σ φ_d,
    #(relations), Σ φ_r.
    """
    sia = getattr(ces, "sia", None)
    distinctions = getattr(ces, "distinctions", None)
    relations = getattr(ces, "relations", None)

    big_phi = getattr(sia, "phi", None) if sia is not None else None
    num_distinctions = len(distinctions) if distinctions is not None else None
    sum_phi_d = distinctions.sum_phi() if distinctions is not None and hasattr(distinctions, "sum_phi") else None
    num_relations = relations.num_relations() if relations is not None and hasattr(relations, "num_relations") else None
    sum_phi_r = relations.sum_phi() if relations is not None and hasattr(relations, "sum_phi") else None

    return [
        ("Φ", big_phi),
        ("#(distinctions)", num_distinctions),
        (f"Σ {SMALL_PHI}_d", sum_phi_d),
        ("#(relations)", num_relations),
        (f"Σ {SMALL_PHI}_r", sum_phi_r),
    ]


def fmt_ac_sia_columns(acsia: object) -> list[tuple[str, Any]]:
    """Column list for an AcSystemIrreducibilityAnalysis: alpha, direction,
    before/after state, partition.
    """
    node_indices = getattr(acsia, "node_indices", None)
    node_labels = getattr(acsia, "node_labels", None)
    if node_labels is not None and node_indices is not None:
        system_label = ",".join(
            str(label) for label in node_labels.coerce_to_labels(node_indices)
        )
    elif node_indices is not None:
        system_label = ",".join(str(i) for i in node_indices)
    else:
        system_label = None

    return [
        ("System", system_label),
        ("Direction", getattr(acsia, "direction", None)),
        ("Before state", getattr(acsia, "before_state", None)),
        ("After state", getattr(acsia, "after_state", None)),
        ("α", getattr(acsia, "alpha", None)),
        ("Partition", getattr(acsia, "partition", None)),
    ]
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest test/test_result_protocols.py::test_fmt_sia_columns_shared_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/fmt.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add shared column-builder helpers for SIA / CES / AcSIA __repr__

fmt_sia_columns, fmt_ces_columns, fmt_ac_sia_columns return the column
lists used by the corresponding __repr__ helpers. Formalism-specific
extras (4.0's normalized_phi, intrinsic_differentiation, etc.) extend
these lists on top in the concrete classes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 12: Route SIA / CES / AcSIA `__repr__` through shared helpers

**Files:**
- Modify: `pyphi/models/sia.py` (IIT 3.0 SIA `__repr__`)
- Modify: `pyphi/formalism/iit4/__init__.py:252-310` (`_repr_columns`, `__repr__`)
- Modify: `pyphi/models/ces.py:88-107` (`_repr_columns`, `__repr__`)
- Modify: `pyphi/models/actual_causation.py` (AcSIA `__repr__`)

- [ ] **Step 1: Write the failing test**

Add to `test/test_result_protocols.py`:

```python
def test_iit3_and_iit4_sia_repr_share_columns(s):
    """The shared SIA columns appear in both formalism reprs identically."""
    from pyphi.formalism import iit3, iit4

    with config.override(**presets.iit3):
        repr_3 = repr(iit3.sia(s))

    repr_4 = repr(iit4.sia(s))

    # Both reprs must contain the shared column labels
    for label in ("System", "Current state", "Partition"):
        assert label in repr_3
        assert label in repr_4
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_result_protocols.py::test_iit3_and_iit4_sia_repr_share_columns -v`
Expected: PASS (likely — both formalisms already include these labels via the existing `_repr_columns` and `make_repr` paths, though formatted differently). If FAIL, the implementation below makes it pass.

- [ ] **Step 3: Refactor IIT 3.0 SIA `__repr__`**

In `pyphi/models/sia.py`:

```python
def _repr_columns(self):
    return fmt.fmt_sia_columns(self)

def __repr__(self):
    body = "\n".join(fmt.align_columns(self._repr_columns()))
    body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
    body = fmt.center(body)
    return fmt.box(body)
```

- [ ] **Step 4: Refactor IIT 4.0 SIA `_repr_columns`**

In `pyphi/formalism/iit4/__init__.py:252-301`:

```python
def _repr_columns(self):
    columns = fmt.fmt_sia_columns(self)
    # Append IIT 4.0-specific columns
    columns.extend([
        (f"Normalized {fmt.SMALL_PHI}_s", self.normalized_phi),
        (
            "Int. diff. CAUSE",
            self.intrinsic_differentiation[Direction.CAUSE]
            if self.intrinsic_differentiation
            else None,
        ),
        (
            "Int. diff. EFFECT",
            self.intrinsic_differentiation[Direction.EFFECT]
            if self.intrinsic_differentiation
            else None,
        ),
    ])
    if self.system_state is not None:
        columns.extend(self.system_state._repr_columns())
    columns.extend([("#(tied MIPs)", len(self.ties) - 1)])
    if self.reasons:
        columns.append(("Reasons", ", ".join(self.reasons)))
    return columns
```

The `__repr__` body stays as-is (uses `_repr_columns` already).

- [ ] **Step 5: Refactor `CauseEffectStructure._repr_columns`**

In `pyphi/models/ces.py:88-101`:

```python
def _repr_columns(self) -> list[tuple[str, Any]]:
    return fmt.fmt_ces_columns(self)
```

- [ ] **Step 6: Refactor AcSIA `__repr__`**

In `pyphi/models/actual_causation.py`:

```python
def _repr_columns(self):
    return fmt.fmt_ac_sia_columns(self)

def __repr__(self):
    body = "\n".join(fmt.align_columns(self._repr_columns()))
    body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
    body = fmt.center(body)
    return fmt.box(body)
```

Replace the existing `__repr__` which used `fmt.make_repr(self, _ac_sia_attributes)`.

- [ ] **Step 7: Run the touched files' tests**

Run: `uv run pytest test/test_result_protocols.py test/test_big_phi.py test/test_actual.py test/test_complexes.py test/test_models.py -x -q`
Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add pyphi/models/sia.py pyphi/formalism/iit4/__init__.py pyphi/models/ces.py pyphi/models/actual_causation.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Route SIA / CES / AcSIA __repr__ through shared column builders

Both formalism SIAs render the same column set for shared fields
(System, Current state, phi, Partition) via fmt_sia_columns; IIT 4.0
appends its formalism-specific columns (Normalized phi_s, Int. diff.
CAUSE/EFFECT, etc.). CauseEffectStructure renders via fmt_ces_columns
identically for both formalisms. AcSIA renders via fmt_ac_sia_columns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 13: Add `_repr_html_` to SIA / CES / AcSIA

**Files:**
- Modify: `pyphi/models/fmt.py` (add `html_columns` helper)
- Modify: `pyphi/models/sia.py`, `pyphi/formalism/iit4/__init__.py`, `pyphi/models/ces.py`, `pyphi/models/actual_causation.py` (add `_repr_html_`)

- [ ] **Step 1: Write the failing test**

Add to `test/test_result_protocols.py`:

```python
def test_iit3_sia_repr_html(s):
    """IIT 3.0 SIA has a Jupyter HTML repr."""
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    html = sia._repr_html_()
    assert html.startswith("<table") or "<div" in html


def test_iit4_sia_repr_html(s):
    from pyphi.formalism import iit4

    sia = iit4.sia(s)
    html = sia._repr_html_()
    assert html.startswith("<table") or "<div" in html


def test_ces_repr_html(s):
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        ces = iit3.ces(s)
    html = ces._repr_html_()
    assert html.startswith("<table") or "<div" in html


def test_acsia_repr_html(transition):
    from pyphi import actual

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)
    html = acsia._repr_html_()
    assert html.startswith("<table") or "<div" in html
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest test/test_result_protocols.py -v -k "repr_html"`
Expected: FAIL — `_repr_html_` not defined.

- [ ] **Step 3: Add HTML render helper to `pyphi/models/fmt.py`**

```python
def html_columns(columns: list[tuple[str, Any]], title: str | None = None) -> str:
    """Render a column list as an HTML table for Jupyter."""
    rows = "\n".join(
        f"<tr><td><b>{label}</b></td><td>{html_escape(value)}</td></tr>"
        for label, value in columns
    )
    body = f"<table>{rows}</table>"
    if title is not None:
        body = f"<div><b>{title}</b></div>{body}"
    return body


def html_escape(value: Any) -> str:
    """Convert a column value to safe HTML."""
    import html as _html

    if value is None:
        return ""
    return _html.escape(str(value))
```

- [ ] **Step 4: Add `_repr_html_` to each class**

In `pyphi/models/sia.py` `IIT3SystemIrreducibilityAnalysis`:

```python
def _repr_html_(self) -> str:
    return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)
```

In `pyphi/formalism/iit4/__init__.py` `SystemIrreducibilityAnalysis`:

```python
def _repr_html_(self) -> str:
    return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)
```

In `pyphi/models/ces.py` `CauseEffectStructure`:

```python
def _repr_html_(self) -> str:
    return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)
```

In `pyphi/models/actual_causation.py` `AcSystemIrreducibilityAnalysis`:

```python
def _repr_html_(self) -> str:
    return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest test/test_result_protocols.py -v -k "repr_html"`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/fmt.py pyphi/models/sia.py pyphi/formalism/iit4/__init__.py pyphi/models/ces.py pyphi/models/actual_causation.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add _repr_html_ to SIA / CES / AcSIA for Jupyter rendering

Each class's _repr_html_ renders its _repr_columns as an HTML table
via fmt.html_columns. Same column lists as the text __repr__, so the
text and HTML renderings stay in sync automatically.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 14: Cross-class `__eq__` returns `NotImplemented`

**Files:**
- Modify: `pyphi/models/cmp.py` (`general_eq` — verify it returns `NotImplemented` on type mismatch)
- Modify: `pyphi/models/sia.py:116-117` (IIT 3.0 SIA `__eq__`)
- Modify: `pyphi/formalism/iit4/__init__.py:237-238` (IIT 4.0 SIA `__eq__`)
- Modify: `pyphi/models/actual_causation.py` (AcSIA `__eq__`)
- Modify: `pyphi/models/ces.py:79-86` (`CauseEffectStructure.__eq__`)

- [ ] **Step 1: Write the failing test**

Add to `test/test_result_protocols.py`:

```python
def test_iit3_sia_neq_iit4_sia(s):
    """Cross-formalism __eq__ returns False without exception."""
    from pyphi.formalism import iit3, iit4

    with config.override(**presets.iit3):
        sia_3 = iit3.sia(s)
    sia_4 = iit4.sia(s)

    # Must not raise; must return False (Python falls back from NotImplemented)
    assert (sia_3 == sia_4) is False
    assert (sia_4 == sia_3) is False
    # NotImplemented at the dunder level is also visible via __ne__
    assert sia_3 != sia_4


def test_iit3_sia_eq_via_notimplemented_on_str(s):
    """Compare an IIT 3.0 SIA to an unrelated type — returns False, no raise."""
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    assert (sia == "not a sia") is False
    assert (sia == 42) is False
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest test/test_result_protocols.py -v -k "neq_iit4 or notimplemented"`
Expected: may FAIL or may PASS depending on `cmp.general_eq` semantics today. Either way, formalize the behavior.

- [ ] **Step 3: Audit `pyphi/models/cmp.py:general_eq`**

Read the function. If it doesn't already return `NotImplemented` on type mismatch, update it:

```python
def general_eq(self, other, attrs):
    if type(other) is not type(self):
        return NotImplemented
    # ... existing per-attribute comparison loop ...
```

If it already does this, no change needed.

- [ ] **Step 4: Update each `__eq__` method to return `NotImplemented` on type mismatch**

For IIT 3.0 SIA (`pyphi/models/sia.py:116-117`):

```python
def __eq__(self, other):
    if type(other) is not type(self):
        return NotImplemented
    return cmp.general_eq(self, other, _sia_attributes)
```

For IIT 4.0 SIA (`pyphi/formalism/iit4/__init__.py:237-238`):

```python
def __eq__(self, other):
    if type(other) is not type(self):
        return NotImplemented
    return cmp.general_eq(self, other, self._sia_attributes)
```

For `CauseEffectStructure` (`pyphi/models/ces.py:79-86`):

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

For AcSIA (`pyphi/models/actual_causation.py`): inspect the current `__eq__` (likely via `cmp.Orderable` base class). If it doesn't already short-circuit on type mismatch, add the guard.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest test/test_result_protocols.py -v -k "neq_iit4 or notimplemented"`
Expected: 2 passed.

Run: `uv run pytest test/test_models.py test/test_big_phi.py test/test_actual.py -x -q`
Expected: pass (existing equality tests not affected).

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/cmp.py pyphi/models/sia.py pyphi/formalism/iit4/__init__.py pyphi/models/actual_causation.py pyphi/models/ces.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Return NotImplemented from cross-class __eq__ in SIA / CES / AcSIA

Comparing an IIT 3.0 SIA to an IIT 4.0 SIA (or any unrelated type) now
returns NotImplemented at the dunder level, which Python translates to
False at the operator level. Previously general_eq would attempt
per-attribute comparison across classes and silently return False
without making the type-mismatch explicit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 15: Canonical JSON shape + round-trip tests

**Files:**
- Modify: `pyphi/jsonify.py` (module docstring documents the canonical shape)
- Test: `test/test_result_protocols.py` (round-trip tests)

- [ ] **Step 1: Write the failing test**

Add to `test/test_result_protocols.py`:

```python
def test_iit3_ces_json_round_trip(s):
    """IIT 3.0 CES round-trips through jsonify with structural equality."""
    from pyphi import jsonify
    from pyphi.formalism import iit3
    from pyphi.models.ces import CauseEffectStructure

    with config.override(**presets.iit3):
        ces = iit3.ces(s)

    encoded = jsonify.dumps(ces)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, CauseEffectStructure)
    assert decoded.sia.phi == ces.sia.phi
    assert len(decoded.distinctions) == len(ces.distinctions)
    assert decoded.relations.num_relations() == ces.relations.num_relations()


def test_iit3_sia_json_round_trip(s):
    from pyphi import jsonify
    from pyphi.formalism import iit3
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis

    with config.override(**presets.iit3):
        sia = iit3.sia(s)

    encoded = jsonify.dumps(sia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, IIT3SystemIrreducibilityAnalysis)
    assert decoded.phi == sia.phi
    assert decoded.partition == sia.partition
    assert decoded.partitioned_distinctions == sia.partitioned_distinctions


def test_iit4_sia_json_round_trip_unchanged(s):
    """IIT 4.0 SIA round-trip behavior should be unchanged."""
    from pyphi import jsonify
    from pyphi.formalism import iit4
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    sia = iit4.sia(s)
    encoded = jsonify.dumps(sia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, SystemIrreducibilityAnalysis)
    assert decoded == sia


def test_acsia_json_round_trip(transition):
    from pyphi import actual
    from pyphi import jsonify
    from pyphi.models import AcSystemIrreducibilityAnalysis

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)

    encoded = jsonify.dumps(acsia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, AcSystemIrreducibilityAnalysis)
    assert decoded.alpha == acsia.alpha
```

- [ ] **Step 2: Run the failing tests**

Run: `uv run pytest test/test_result_protocols.py -v -k "json_round_trip"`
Expected: most pass (`jsonify` infrastructure already handles this via the `_loadable_models` registry); any failure means a specific class's `from_json` doesn't accept the new shape — fix it.

- [ ] **Step 3: Document the canonical shape in `pyphi/jsonify.py` module docstring**

Update the top docstring (~line 1-30) to include:

```python
"""Serialize and deserialize PyPhi objects to and from JSON.

Canonical shapes for the SIA / CES surface (both IIT 3.0 and IIT 4.0):

- IIT 3.0 SIA — discriminated by ``__class__: "IIT3SystemIrreducibilityAnalysis"``::

    {"__class__": "IIT3SystemIrreducibilityAnalysis", "__version__": ..., "__id__": ...,
     "phi": ..., "partition": ..., "partitioned_distinctions": ...,
     "current_state": ..., "node_indices": ..., "node_labels": ...,
     "config": ...}

- IIT 4.0 SIA — discriminated by ``__class__: "SystemIrreducibilityAnalysis"``::

    {"__class__": "SystemIrreducibilityAnalysis", "__version__": ..., "__id__": ...,
     "phi": ..., "partition": ..., "normalized_phi": ..., "signed_phi": ...,
     "signed_normalized_phi": ..., "cause": ..., "effect": ...,
     "system_state": ..., "current_state": ..., "node_indices": ...,
     "intrinsic_differentiation": ..., "config": ...}

- CauseEffectStructure — discriminated by ``__class__: "CauseEffectStructure"``;
  shape identical for both formalisms (the inner SIA carries its formalism via
  its own ``__class__``)::

    {"__class__": "CauseEffectStructure", "__version__": ..., "__id__": ...,
     "sia": { ... },
     "distinctions": ...,
     "relations": ...,
     "config": ...}

- AcSystemIrreducibilityAnalysis — discriminated by
  ``__class__: "AcSystemIrreducibilityAnalysis"``::

    {"__class__": "AcSystemIrreducibilityAnalysis", ..., "alpha": ...,
     "direction": ..., "account": ..., "partitioned_account": ...,
     ...}

A future migration to msgspec (planned) will adopt these shapes via
tagged unions keyed on ``__class__``.

[rest of existing docstring]
"""
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_result_protocols.py -v -k "json_round_trip"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pyphi/jsonify.py test/test_result_protocols.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Document canonical JSON shape for SIA / CES / AcSIA round-trip

The jsonify module docstring now documents the per-class canonical JSON
shape that round-trips through jsonify.loads / dumps. The shapes are
discriminated by the existing CLASS_KEY metadata; a future msgspec
migration adopts them via tagged unions.

Round-trip pinned via test/test_result_protocols.py::test_*_json_round_trip
covering IIT 3.0 SIA, IIT 4.0 SIA, CauseEffectStructure, AcSIA.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 16: Regenerate all 17 goldens for JSON-shape changes

**Files:**
- Regenerate: `test/data/golden/v1/*.json` (all 17)

- [ ] **Step 1: Regenerate all goldens**

Run: `uv run pytest test/test_golden_regression.py --regenerate-golden`
Expected: 17 SKIP entries (regenerated).

- [ ] **Step 2: Inspect the diff**

Run: `git diff --stat test/data/golden/v1/`

Expected: every JSON file changed.

Run: `git diff test/data/golden/v1/basic_iit3_emd.json | head -50`

Verify the shape change is mechanical (field name updates from any earlier rename, no value drift). The `phi`, `partition` structure, distinction counts, etc. MUST be unchanged.

- [ ] **Step 3: Run the golden suite to confirm pass**

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: 17/17 pass.

- [ ] **Step 4: Commit**

```bash
git add test/data/golden/v1/
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Regenerate all 17 goldens for canonical JSON shape

The Cluster C work formalizes the canonical JSON shape (discriminator
via CLASS_KEY, common field names for shared fields). All 17 golden
fixtures regenerate with the final shape; numeric values are
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

---

### Task 17: Cluster C changelog fragment + final acceptance

**Files:**
- Create: `changelog.d/sia-ces-unified-surface.change.md`

- [ ] **Step 1: Write the changelog fragment**

```markdown
Unified the cross-formalism surface of system-irreducibility analysis
results.

- Added :class:`~pyphi.models.protocols.SIAInterface`,
  :class:`~pyphi.models.protocols.CauseEffectStructureInterface`, and
  :class:`~pyphi.models.protocols.AcSIAInterface` Protocols
  (``runtime_checkable``). Both formalisms' SIA classes implement
  ``SIAInterface``; ``CauseEffectStructure`` is used by both formalisms
  (IIT 3.0 wraps an empty :class:`~pyphi.relations.NullRelations`);
  :class:`~pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`
  implements ``AcSIAInterface``.

- All four classes now share a common ``__repr__`` template via
  :func:`~pyphi.models.fmt.fmt_sia_columns`,
  :func:`~pyphi.models.fmt.fmt_ces_columns`,
  :func:`~pyphi.models.fmt.fmt_ac_sia_columns`. Formalism-specific extras
  (IIT 4.0's ``normalized_phi``, ``intrinsic_differentiation``, etc.)
  extend the shared column list.

- Added ``_repr_html_`` to SIA / CES / AcSIA for Jupyter rendering. Same
  column source as the text ``__repr__``, so text and HTML stay in sync.

- ``__eq__`` on each class now returns ``NotImplemented`` when ``other``
  is a different type. Cross-class comparisons (``iit3_sia == iit4_sia``)
  evaluate to ``False`` without exception, with the type-mismatch made
  explicit at the dunder level.

- The canonical JSON shape for each result type is documented in
  :mod:`pyphi.jsonify`'s module docstring. A future msgspec migration
  adopts these shapes via tagged unions keyed on ``__class__``.

The 17 golden fixtures regenerated with the canonical shape; numeric
values are unchanged.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.d/sia-ces-unified-surface.change.md
git diff --cached --stat
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add changelog fragment for cross-formalism SIA / CES surface unification
EOF
)"
git show --stat HEAD
```

---

## Final acceptance gates

After Task 17, run the full suite:

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1256+ passed, 0 failures, ≤3 xfailed.

Kick off the slow lane in the background:

```bash
uv run pytest test/ --slow -q
# run_in_background=true
```

Expected (on completion): 1264+ passed, 0 failures, ≤4 xfailed.

```bash
uv run pytest test/test_golden_regression.py -q
```

Expected: 17/17 pass.

```bash
uv run pyright pyphi/
```

Expected: 0 errors / 1 baseline warning.

```bash
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

Expected: clean.

End-to-end smoke test:

```bash
uv run python -c "
import pyphi
from pyphi.conf import presets
from pyphi import config
from test import example_substrates

with config.override(**presets.iit3):
    s = example_substrates.s()
    ces = pyphi.formalism.iit3.ces(s)
    print(ces)
    print(repr(ces.sia))
    print(ces.sia.partitioned_distinctions)

assert isinstance(ces.sia, pyphi.models.protocols.SIAInterface)
assert isinstance(ces, pyphi.models.protocols.CauseEffectStructureInterface)
print('protocol conformance OK')
"
```

Expected: prints CES, prints SIA repr (with shared column labels), prints partitioned_distinctions; smoke test prints "protocol conformance OK".

If all gates green, the project is complete. Surface results to the user with the commit range and recommend the standard finishing-a-development-branch workflow.

---

## Risk register (summary from the spec)

| Risk | Likelihood | Phase | Mitigation |
|---|---|---|---|
| Cluster A regen invalidates 4 IIT 3.0 EMD goldens; downstream consumers reading the old shape break silently | High (expected) | A | Goldens regenerate with `--regenerate-golden`; shape change reviewed manually in one commit |
| `unorderable_unless_eq=["substrate"]` removal makes cross-substrate phi comparisons valid; some test may have relied on the guard | Low-medium | A | Task 6 pins the new behavior; if a test breaks, replace the guard with `node_indices` proxy |
| `NullRelations` is new | Low | A | Small contained addition; tested in Task 1 |
| Protocol-conformance type errors surface latent inconsistencies | Medium | C | Per saved memory `feedback_dont_give_up_on_architectural_refactors` — diagnose, don't relax the Protocol |
| JSON shape change at Cluster C forces 17 goldens regen | High (expected) | C | Task 16 regenerates with one commit; numeric values unchanged |
| 3.0 callers reading `sia.ces` / `sia.partitioned_ces` / `sia.substrate` (outside the repo) break | Medium | A | Documented in changelog fragments; migration note in `docs/migration-2.0.md` (planned for P15) |
| AcSIA's `alpha` not renaming to `phi` is inconsistent | Low | C | Decision in spec: keep `alpha` per AC paper. Protocol surface declares it as `alpha`. |
