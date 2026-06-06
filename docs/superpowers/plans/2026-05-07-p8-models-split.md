# P8 — Models Split + Mechanism-Level Signed-Phi: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `pyphi/models/` into focused single-concept files, move `PhiStructure` from `pyphi/formalism/iit4/` into the models tier and clean it up, add the deferred mechanism-level `signed_phi` field, remove vestigial back-references, disambiguate `Unit`, and pin the models tier as pure data via an architectural test.

**Architecture:** See [`docs/superpowers/specs/2026-05-07-p8-models-split-design.md`](../specs/2026-05-07-p8-models-split-design.md).

**Tech Stack:** Python 3.13, frozen dataclasses, `cmp.Orderable`/`cmp.OrderableByPhi` mixins, `pyphi.utils.positive_part`, `ast.walk`-based layering tests.

---

## Phase 0 — Baseline

### Task 0.1: Confirm clean baseline

**Files:** none; verify worktree state

- [ ] **Step 1: Confirm worktree on `feature/p7-kernel-rewrite`**

```bash
cd ../pyphi-p7-kernel-rewrite
git status
git log --oneline -3
```

Expected: clean tree (or only the deferred-deletion changes from P7), HEAD at `bfb0d9d1` ("P7: update changelog fragment to reflect Option D split") or later.

- [ ] **Step 2: Run fast lane to confirm green start**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: 974 passed, 22 skipped (or comparable; no failures).

- [ ] **Step 3: Pyright clean on `pyphi/core/` and `pyphi/formalism/`**

Run: `uv run pyright pyphi/core/ pyphi/formalism/`
Expected: 0 errors.

### Task 0.2: Decide branch strategy

**Files:** none; user decision

- [ ] **Step 1: Confirm branch**

P8 lands as a sequence of small commits on `feature/p7-kernel-rewrite` (or a new `feature/p8-models-split` branch — user's call). Default: continue on `feature/p7-kernel-rewrite` (P7 work is not yet pushed; P8 stacks on it).

---

## Phase 1 — Architectural test (write first; will fail until split lands)

### Task 1.1: Add the layering test

**Files:**
- Create: `test/test_models_layering.py`

- [ ] **Step 1: Write the layering test**

```python
"""Architectural assertions for the pyphi.models tier.

Models are pure data: no formalism dispatch, no kernel-operation calls.
Walks the AST so lazy imports inside method bodies are also caught
(same pattern as ``test_core_layering``).
"""

from __future__ import annotations

import ast
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "pyphi" / "models"


def _imports_in(path: Path) -> set[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_models_does_not_import_formalism() -> None:
    """`pyphi.models.*` is a pure-data tier: no formalism dispatch."""
    for py in MODELS.rglob("*.py"):
        imports = _imports_in(py)
        offenders = [i for i in imports if i.startswith("pyphi.formalism")]
        assert not offenders, f"{py}: imports {offenders}"


def test_models_does_not_import_kernel_ops() -> None:
    """`pyphi.models.*` does not import the repertoire-algebra kernel.

    Models are containers for results; they do not call computation.
    """
    for py in MODELS.rglob("*.py"):
        imports = _imports_in(py)
        offenders = [
            i
            for i in imports
            if i == "pyphi.core.repertoire_algebra"
            or i.endswith(".repertoire_algebra")
        ]
        assert not offenders, f"{py}: imports {offenders}"
```

- [ ] **Step 2: Run the new layering test against the *current* (unsplit) models tier**

Run: `uv run pytest test/test_models_layering.py -q`
Expected: PASS (current models are already pure data; this test will continue passing through the split).

If this fails, that itself is a useful finding — investigate the offending import before proceeding.

- [ ] **Step 3: Commit**

```bash
git add test/test_models_layering.py
git commit -m "P8: layering test for pyphi.models tier"
```

---

## Phase 2 — `signed_phi` for `RepertoireIrreducibilityAnalysis`

This is a self-contained addition to the existing `models/mechanism.py` (before the split). Doing it before the split means the split is purely mechanical.

### Task 2.1: Add `signed_phi` field to RIA

**Files:**
- Modify: `pyphi/models/mechanism.py:214` (RIA dataclass)
- Test: `test/test_models.py` (existing; add new test cases)

- [ ] **Step 1: Read the SIA `__post_init__` pattern (from P5)**

Run: `grep -n "signed_phi\|positive_part\|__post_init__" pyphi/formalism/iit4/__init__.py | head -30`

Familiarize with the clamp pattern: snapshot `signed_phi` from `phi` if not provided, apply `positive_part` to derive the public `phi`, preserve `DistanceResult` metadata.

- [ ] **Step 2: Write the failing test for RIA signed_phi clamping**

Add to `test/test_models.py`:

```python
def test_ria_signed_phi_clamps_phi_to_positive_part():
    """Negative signed_phi yields phi=0; positive signed_phi yields phi=signed_phi."""
    from pyphi.data_structures import PyPhiFloat
    from pyphi.direction import Direction
    from pyphi.models import RepertoireIrreducibilityAnalysis as RIA

    pos = RIA(
        phi=0.5, direction=Direction.CAUSE,
        mechanism=(0,), purview=(1,),
        partition=None, repertoire=None, partitioned_repertoire=None,
    )
    assert float(pos.phi) == 0.5
    assert float(pos.signed_phi) == 0.5

    neg = RIA(
        phi=-0.3, direction=Direction.CAUSE,
        mechanism=(0,), purview=(1,),
        partition=None, repertoire=None, partitioned_repertoire=None,
    )
    assert float(neg.phi) == 0.0
    assert float(neg.signed_phi) == -0.3
```

Run: `uv run pytest test/test_models.py::test_ria_signed_phi_clamps_phi_to_positive_part -v`
Expected: FAIL (signed_phi field doesn't exist yet).

- [ ] **Step 3: Add the `signed_phi` field and `__post_init__` clamp**

Modify `pyphi/models/mechanism.py:214` (`RepertoireIrreducibilityAnalysis`):

```python
# Add to the dataclass:
signed_phi: PyPhiFloat | DistanceResult | None = None

def __post_init__(self) -> None:
    # ... existing code ...

    # Snapshot the raw signed value before clamping (mirror of SIA pattern).
    if self.signed_phi is None:
        self.signed_phi = self.phi
    clamped = utils.positive_part(self.signed_phi)
    if not isinstance(self.phi, DistanceResult):
        self.phi = PyPhiFloat(clamped)
    else:
        self.phi = type(self.phi)(clamped, **self.phi._public_aux_data())
    if not isinstance(self.signed_phi, DistanceResult):
        self.signed_phi = PyPhiFloat(self.signed_phi)
```

Update `_null_ria()` (around line 570) to accept `signed_phi=None` and pass through.

- [ ] **Step 4: Verify the new test passes**

Run: `uv run pytest test/test_models.py::test_ria_signed_phi_clamps_phi_to_positive_part -v`
Expected: PASS.

- [ ] **Step 5: Run full models tests + golden regression for the most likely-to-shift fixture**

Run: `uv run pytest test/test_models.py test/test_golden_regression.py -k "grid3 or basic or xor" -q`
Expected: PASS. If a fixture shifts (e.g., a preventative-cause RIA's `phi` was negative and is now 0), regenerate that fixture with `--regenerate-golden -k <fixture>` and pin the new contract in the commit message.

- [ ] **Step 6: Add Hypothesis property test for the invariant**

Append to `test/test_invariants_hypothesis.py`:

```python
@given(...)  # use existing subsystem strategy
@settings(max_examples=20, deadline=None)
def test_ria_phi_is_positive_part_of_signed_phi(self, ...):
    """RIA.phi == max(0, RIA.signed_phi) for any subsystem/mechanism/purview."""
    ...  # build RIA via formalism.find_mip; assert phi == positive_part(signed_phi)
```

- [ ] **Step 7: Run the Hypothesis property test**

Run: `uv run pytest test/test_invariants_hypothesis.py::TestSignedPhi -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyphi/models/mechanism.py test/test_models.py test/test_invariants_hypothesis.py
git commit -m "P8: signed_phi metadata for RepertoireIrreducibilityAnalysis"
```

---

## Phase 3 — Remove `Concept.subsystem` and `CauseEffectStructure.subsystem` back-references

### Task 3.1: Audit current uses

**Files:**
- Read: `pyphi/models/mechanism.py:944` (Concept), `pyphi/models/subsystem.py` (CES), `pyphi/compute/subsystem.py:90, 118`, `test/test_big_phi.py:287`

- [ ] **Step 1: Confirm the audit findings**

Run:
```bash
grep -rn "concept\.subsystem\|distinction\.subsystem\|ces\.subsystem\|\.subsystem = subsystem\|self\.subsystem = " pyphi/ test/ --include="*.py" | grep -v candidate_system.py | grep -v "core/"
```

Expected output (from current state):
- `pyphi/models/subsystem.py:110` — commented-out
- `pyphi/models/subsystem.py:303` — `self.subsystem = subsystem` (CES, IIT 3.0 SIA)
- `pyphi/models/subsystem.py:328-329` — `self.subsystem.network` (SIA, **stay** — IIT 3.0 SIA, not in scope)
- `pyphi/models/mechanism.py:1162-1188` — commented-out (already dead)
- `pyphi/compute/subsystem.py:93` — `concept.subsystem = None`
- `pyphi/compute/subsystem.py:118` — `concept.subsystem = subsystem`
- `pyphi/compute/subsystem.py:370` — `self.subsystem = subsystem` (likely a different concern; verify)
- `test/test_big_phi.py:287` — assertion

- [ ] **Step 2: Read `compute/subsystem.py:370` to determine which class it's setting on**

Run: `sed -n '350,375p' pyphi/compute/subsystem.py`

Expected: this is on `_ComputeCausalEffectStructure` (a helper class, not Concept). Out of P3 scope; ignore.

### Task 3.2: Remove `subsystem` from `Concept` and `CauseEffectStructure`

**Files:**
- Modify: `pyphi/models/mechanism.py:944` (Concept class — drop `self.subsystem` if any, drop commented-out methods)
- Modify: `pyphi/models/subsystem.py:103` (CES class — drop `self.subsystem` assignment)
- Modify: `pyphi/compute/subsystem.py:93, 118` (drop the set/null pattern)
- Modify: `test/test_big_phi.py:287` (drop the assertion)

- [ ] **Step 1: Read Concept's current subsystem references**

Run: `sed -n '944,990p' pyphi/models/mechanism.py`
Run: `sed -n '1160,1190p' pyphi/models/mechanism.py`

Confirm: no `self.subsystem = ...` in `Concept.__init__` (it's set externally); the commented-out methods are at lines 1162-1188.

- [ ] **Step 2: Delete the commented-out `expand_*_repertoire` block in Concept**

Edit `pyphi/models/mechanism.py:1162-1188` — delete the entire commented-out region.

- [ ] **Step 3: Drop `self.subsystem` from CauseEffectStructure**

Edit `pyphi/models/subsystem.py`: in `CauseEffectStructure.__init__` (around line 103), remove `self.subsystem = subsystem`. Drop the `subsystem` parameter from the signature; update `_null_ces` (line 365) and any internal callers.

- [ ] **Step 4: Drop the set/null pattern in `compute/subsystem.py`**

Edit `pyphi/compute/subsystem.py`:
- Line 93: remove `concept.subsystem = None`
- Line 118: remove `concept.subsystem = subsystem`

- [ ] **Step 5: Update the test assertion**

Edit `test/test_big_phi.py:287`: drop the line `assert concept.subsystem is ces.subsystem`.

- [ ] **Step 6: Run the affected tests**

Run: `uv run pytest test/test_big_phi.py test/test_models.py test/test_compute_subsystem.py -q`
Expected: PASS.

- [ ] **Step 7: Run the full fast lane**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add pyphi/models/mechanism.py pyphi/models/subsystem.py pyphi/compute/subsystem.py test/test_big_phi.py
git commit -m "P8: remove Concept.subsystem and CES.subsystem back-references"
```

---

## Phase 4 — Rename `Unit` → `UnitState` in `models/mechanism.py`

### Task 4.1: Rename the class and update references

**Files:**
- Modify: `pyphi/models/mechanism.py:50` (rename Unit → UnitState)
- Modify: `pyphi/models/fmt.py` (any references)
- Modify: `pyphi/models/__init__.py` (re-exports)

- [ ] **Step 1: Find all references**

Run: `grep -rn "from pyphi\.models\.mechanism import Unit\|models\.mechanism\.Unit\|models import Unit" pyphi/ test/ --include="*.py"`

Expected: a small set (~5 sites, mostly in mechanism.py and fmt.py).

- [ ] **Step 2: Rename the class**

In `pyphi/models/mechanism.py:50`: rename `class Unit:` → `class UnitState:`.
Update internal references within the same file.

- [ ] **Step 3: Update fmt.py and other importers**

Search for `Unit` references where they refer to the model class (not `pyphi.core.unit.Unit`); replace with `UnitState`.

- [ ] **Step 4: Update `pyphi/models/__init__.py` re-exports**

If `Unit` was re-exported, replace with `UnitState`. (Verify whether `Unit` is currently re-exported; if not, just update internal references.)

- [ ] **Step 5: Run the affected tests**

Run: `uv run pytest test/test_models.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/ test/
git commit -m "P8: rename models.mechanism.Unit -> UnitState"
```

---

## Phase 5 — Split `models/mechanism.py` into focused files

This phase does the file-level reorganization without changing semantics. Each task moves one concept and verifies tests still pass.

### Task 5.1: Create `models/state_specification.py`

**Files:**
- Create: `pyphi/models/state_specification.py`
- Modify: `pyphi/models/mechanism.py` (remove the moved content)
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify the content to move**

`pyphi/models/mechanism.py` lines ~76-156 contain:
- `class StateSpecification` (line 76)
- `class DistinctionPhiNormalizationRegistry` (line 158)
- helper `@register` decorators (~lines 168-186)
- `class UnitState` (line 50, after rename)

- [ ] **Step 2: Move content to new file**

Create `pyphi/models/state_specification.py` with the moved classes + necessary imports.

- [ ] **Step 3: Remove from `mechanism.py`**

Delete the moved classes from `pyphi/models/mechanism.py`.

- [ ] **Step 4: Update `__init__.py` imports**

In `pyphi/models/__init__.py`:
```python
from .state_specification import (
    DistinctionPhiNormalizationRegistry,
    StateSpecification,
    UnitState,
)
```

- [ ] **Step 5: Verify test pass**

Run: `uv run pytest test/test_models.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/state_specification.py pyphi/models/mechanism.py pyphi/models/__init__.py
git commit -m "P8: extract state_specification.py from models/mechanism.py"
```

### Task 5.2: Create `models/ria.py`

**Files:**
- Create: `pyphi/models/ria.py`
- Modify: `pyphi/models/mechanism.py`
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify content**

From `pyphi/models/mechanism.py`:
- `class ShortCircuitConditions` (~line 188)
- `class RepertoireIrreducibilityAnalysis` (line 214)
- `def _null_ria` (~line 570)

- [ ] **Step 2: Move to `models/ria.py`**

Create the new file with the extracted classes/functions and their imports.

- [ ] **Step 3: Update `mechanism.py` (remove moved content)**

- [ ] **Step 4: Update `__init__.py` re-exports**

```python
from .ria import RepertoireIrreducibilityAnalysis, ShortCircuitConditions, _null_ria
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest test/test_models.py test/test_subsystem.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/ria.py pyphi/models/mechanism.py pyphi/models/__init__.py
git commit -m "P8: extract ria.py from models/mechanism.py"
```

### Task 5.3: Create `models/mice.py`

**Files:**
- Create: `pyphi/models/mice.py`
- Modify: `pyphi/models/mechanism.py`
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify content**

From `pyphi/models/mechanism.py`:
- `class MaximallyIrreducibleCauseOrEffect` (~line 590)
- `class MaximallyIrreducibleCause` (~line 887)
- `class MaximallyIrreducibleEffect` (~line 910)

- [ ] **Step 2: Move to `models/mice.py`**

- [ ] **Step 3: Update `mechanism.py`**

- [ ] **Step 4: Update `__init__.py` re-exports**

```python
from .mice import (
    MaximallyIrreducibleCause,
    MaximallyIrreducibleCauseOrEffect,
    MaximallyIrreducibleEffect,
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest test/test_models.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/
git commit -m "P8: extract mice.py from models/mechanism.py"
```

### Task 5.4: Create `models/concept.py`

**Files:**
- Create: `pyphi/models/concept.py`
- Modify: `pyphi/models/mechanism.py`
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify content**

From `pyphi/models/mechanism.py`:
- `class Concept` (line 944)

- [ ] **Step 2: Move to `models/concept.py`**

- [ ] **Step 3: Verify `mechanism.py` is now empty (or near-empty)**

Run: `wc -l pyphi/models/mechanism.py`
Expected: very small (just lingering imports / leftover content). If anything substantive remains, evaluate where it belongs.

- [ ] **Step 4: Delete `pyphi/models/mechanism.py`**

If only imports / docstring remain, delete the file.

- [ ] **Step 5: Update `__init__.py` re-exports**

```python
from .concept import Concept
```

Drop any remaining `from .mechanism import ...` lines.

- [ ] **Step 6: Run full fast lane**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pyphi/models/
git commit -m "P8: extract concept.py from models/mechanism.py and delete the old file"
```

---

## Phase 6 — Split `models/subsystem.py` into focused files

### Task 6.1: Create `models/ces.py`

**Files:**
- Create: `pyphi/models/ces.py`
- Modify: `pyphi/models/subsystem.py`
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify content**

From `pyphi/models/subsystem.py`:
- helpers: `_concept_sort_key`, `defaultdict_set`, `_purview_inclusion`, `_find_multiplicities`, `_get_mechanism`, `_get_state` (~lines 62-101)
- `class CauseEffectStructure` (line 103)
- `def _null_ces` (line 365)

- [ ] **Step 2: Move to `models/ces.py`**

- [ ] **Step 3: Update `subsystem.py`**

- [ ] **Step 4: Update `__init__.py` re-exports**

```python
from .ces import CauseEffectStructure, _null_ces
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest test/test_models.py test/test_subsystem.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/
git commit -m "P8: extract ces.py from models/subsystem.py"
```

### Task 6.2: Create `models/sia.py`

**Files:**
- Create: `pyphi/models/sia.py`
- Modify: `pyphi/models/subsystem.py`
- Modify: `pyphi/models/__init__.py`

- [ ] **Step 1: Identify content**

From `pyphi/models/subsystem.py`:
- `class SystemStateSpecification` (line 26)
- `class SystemIrreducibilityAnalysis` (line 255) — IIT 3.0 SIA
- `def _null_sia` (line 370)

- [ ] **Step 2: Move to `models/sia.py`**

- [ ] **Step 3: Verify `subsystem.py` is empty / near-empty**

If only imports / docstring remain, delete the file.

- [ ] **Step 4: Update `__init__.py` re-exports**

```python
from .sia import SystemIrreducibilityAnalysis, SystemStateSpecification, _null_sia
```

Drop any remaining `from .subsystem import ...` lines.

- [ ] **Step 5: Run full fast lane**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/
git commit -m "P8: extract sia.py from models/subsystem.py and delete the old file"
```

---

## Phase 7 — Move and clean up `PhiStructure`

### Task 7.1: Move `PhiStructure` to `pyphi/models/phi_structure.py`

**Files:**
- Create: `pyphi/models/phi_structure.py`
- Modify: `pyphi/formalism/iit4/__init__.py:689` (remove the class)
- Modify: `pyphi/models/__init__.py` (re-export)

- [ ] **Step 1: Find all callers of `PhiStructure`**

Run: `grep -rn "PhiStructure" pyphi/ test/ --include="*.py" | grep -v "matching/" | head`

Expected: ~10 sites in formalism/iit4 + visualize/phi_structure.

- [ ] **Step 2: Create `pyphi/models/phi_structure.py` with the cleaned-up class**

```python
"""Φ-structure model: a SIA bundled with its cause-effect structure
(distinctions) and relations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyphi.models import cmp

if TYPE_CHECKING:
    from pyphi.data_structures import PyPhiFloat
    from pyphi.models import CauseEffectStructure
    from pyphi.models.sia import SystemIrreducibilityAnalysis
    from pyphi.relations import Relations


@dataclass(frozen=True)
class PhiStructure(cmp.Orderable):
    """A Φ-structure: SIA + distinctions + relations.

    Note: ``phi``, ``partition``, ``system_state`` are accessed via
    ``ps.sia.phi`` etc. Earlier versions of this class proxied those
    attributes via ``__getattr__``; the proxy was removed in P8 in
    favor of explicit access through ``.sia``.
    """

    sia: SystemIrreducibilityAnalysis
    distinctions: CauseEffectStructure
    relations: Relations

    @property
    def components(self) -> Iterable[object]:
        yield from self.distinctions
        yield from list(self.relations)

    def order_by(self) -> PyPhiFloat:
        return self.sia.phi

    def __bool__(self) -> bool:
        return bool(self.sia)
```

- [ ] **Step 3: Migrate the ~5 internal call sites that use `phi_structure.phi` / `.partition` / `.system_state`**

Run: `grep -n "phi_structure\.\(phi\|partition\|system_state\)\|\.phi_structure\.\(phi\|partition\|system_state\)" pyphi/ -r --include="*.py"`

Expected: a handful of sites in `pyphi/formalism/iit4/__init__.py` and `pyphi/visualize/phi_structure/`.

For each, replace `phi_structure.phi` → `phi_structure.sia.phi`, etc.

- [ ] **Step 4: Remove `PhiStructure` from `pyphi/formalism/iit4/__init__.py`**

Delete lines 689-~768 (the original class definition).
Add `from pyphi.models.phi_structure import PhiStructure` at the top of the file (since iit4 still uses `PhiStructure` extensively).

- [ ] **Step 5: Add re-export to `pyphi/models/__init__.py`**

```python
from .phi_structure import PhiStructure
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pyphi/models/phi_structure.py pyphi/models/__init__.py pyphi/formalism/iit4/__init__.py pyphi/visualize/
git commit -m "P8: move PhiStructure to models/ and drop __getattr__ proxy"
```

---

## Phase 8 — Acceptance gates

### Task 8.1: Final smoke

**Files:** none

- [ ] **Step 1: Run pyright on `pyphi/core`, `pyphi/formalism`, `pyphi/models`**

Run: `uv run pyright pyphi/core pyphi/formalism pyphi/models`
Expected: 0 errors.

- [ ] **Step 2: Run the architectural tests**

Run: `uv run pytest test/test_core_layering.py test/test_models_layering.py -q`
Expected: PASS.

- [ ] **Step 3: Run the surface drift test**

Run: `uv run pytest test/test_subsystem_surface.py -q`
Expected: PASS.

- [ ] **Step 4: Run the full fast lane**

Run: `uv run pytest test/ --ignore=test/test_invariants_hypothesis.py --ignore=test/test_macro_subsystem.py --ignore=test/test_macro_blackbox.py --ignore=test/test_golden_regression.py -q`
Expected: all pass.

- [ ] **Step 5: Run the Hypothesis lane**

Run: `uv run pytest test/test_invariants_hypothesis.py -q`
Expected: 21 passed (the 20 existing + 1 new for RIA signed_phi).

- [ ] **Step 6: Run the golden regression lane (~14 minutes)**

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: 17 passed. If any fixture shifts due to RIA signed_phi, regenerate it explicitly with `--regenerate-golden -k <fixture>` and document the change in the commit message.

### Task 8.2: Sign-flip canary

- [ ] **Step 1: Verify the sign-flip canary still bites**

Temporarily patch `hamming_emd` in `pyphi/metrics/distribution.py` to negate its output, run the affected fixtures, and confirm ≥3 fixtures + ≥1 property test fail. Revert.

(Optional but recommended for major architectural commits.)

### Task 8.3: Changelog fragment

**Files:**
- Create: `changelog.d/p8-models-split.refactor.md`

- [ ] **Step 1: Write the fragment**

```bash
cat > changelog.d/p8-models-split.refactor.md <<'EOF'
The ``pyphi.models`` tier reorganized into one-concept-per-file: ``ria.py``,
``mice.py``, ``concept.py``, ``state_specification.py``, ``ces.py``, ``sia.py``,
and ``phi_structure.py``. ``pyphi.models.mechanism`` and ``pyphi.models.subsystem``
removed; the public re-exports through ``pyphi.models`` are preserved
(``from pyphi.models import Concept, RepertoireIrreducibilityAnalysis, …``
continues to work). ``RepertoireIrreducibilityAnalysis`` gains a non-optional
``signed_phi`` field mirroring ``SystemIrreducibilityAnalysis`` (P5);
``phi`` is now ``positive_part(signed_phi)``. ``PhiStructure`` moved from
``pyphi.formalism.iit4`` to ``pyphi.models.phi_structure``; the legacy
``__getattr__`` proxy that surfaced SIA attributes at the top level was
dropped — access via ``.sia`` instead. ``Concept.subsystem`` and
``CauseEffectStructure.subsystem`` back-references removed. The local
``pyphi.models.mechanism.Unit`` renamed to ``UnitState`` to disambiguate
from the substrate-level ``pyphi.core.unit.Unit``.
EOF
```

- [ ] **Step 2: Commit**

```bash
git add changelog.d/p8-models-split.refactor.md
git commit -m "P8: changelog fragment for models split + signed_phi at mechanism level"
```

### Task 8.4: Update memory

- [ ] **Step 1: Update MEMORY.md if any new feedback or design decisions worth persisting**

(Not required unless the user gives feedback during execution that's worth recording.)

---

## Out of scope (tracked for follow-up)

| Item | Future home |
|---|---|
| Φ-fold model class | P14b (matching/perception) |
| Triggering, perception, differentiation, matching | P14b |
| `models/fmt.py` split (1027 lines) | P15 cleanup |
| `models/actual_causation.py` rewrite | P14 |
| `SystemIrreducibilityAnalysis.subsystem` removal (IIT 3.0) | Future SIA-layer commit |
| `pyphi/relations.py` reorganization | P14b |

---

End of plan.
