# P8 — Models Split + Signed-Phi at Mechanism Level: Design

**Status:** approved
**Date:** 2026-05-07
**Predecessor:** P7 (subsystem layered rewrite, including Option D — formalism queries hoisted to `pyphi.formalism`)
**Successors:** P9 (unified cache), …, P14 (macro/actual rewrite), P14b (matching/perception — adds Φ-fold)

---

## 1. Motivation

After P7, the architecture is:

- `pyphi/core/` — value types (`CandidateSystem`, `CausalModel`, `Substrate`, `Unit`, `TPM`) and the kernel (`repertoire_algebra`)
- `pyphi/formalism/` — formalism Protocol + concrete formalisms + free-function queries (`find_mip`, `sia`, `concept`, …)
- `pyphi/models/` — data classes for results (RIA, MICE, Concept, CES, SIA, …)

The models tier is the next layer overdue for cleanup. Three concrete problems:

1. **Files are too large to reason about as units.** `models/mechanism.py` is 1216 lines and bundles five distinct concepts (RIA, MICE, Concept, StateSpecification, a local `Unit`). `models/subsystem.py` is 382 lines and bundles `CauseEffectStructure` + IIT-3.0 `SystemIrreducibilityAnalysis`. The separations are crisp at the type level but blurred at the file level.

2. **`PhiStructure` lives in the wrong place.** It is a data class — `(sia, distinctions, relations)` — with a few derived attributes, but it currently lives in `pyphi/formalism/iit4/__init__.py:689`. After P7's kernel/formalism split, that's a layering smell: `PhiStructure` describes data, not formalism policy. It also uses a legacy `__getattr__`-proxy pattern to inherit attributes from its embedded SIA, which obscures its actual fields.

3. **Two `Unit`s, vestigial back-references, and a deferred signed-phi field.** The codebase has both `pyphi.core.unit.Unit` (substrate identity: `index, label`) and `pyphi.models.mechanism.Unit` (a node with its current state value: `index, state, label`) — the same name for different concepts. `Concept.subsystem` is set externally, then nulled before serialization, and only one test still checks it (the methods that used it are commented out). `RepertoireIrreducibilityAnalysis.signed_phi` was deferred from P5 to P8 in the cross-cutting registry — it should land here so the `|·|+` clamp pattern is uniform across mechanism-level and system-level analyses.

P8 fixes these three things and adds an architectural test that pins the models layer as pure data (no imports of `pyphi.formalism` or `pyphi.core.repertoire_algebra`).

## 2. Goals

- One file per concept in `pyphi/models/`. No file mixes more than one named result type.
- `PhiStructure` lives in `pyphi/models/`, as a frozen dataclass with explicit fields (no `__getattr__` proxy).
- Mechanism-level signed-phi metadata mirrors the system-level pattern from P5.
- `Concept.subsystem` removed; the externally-set / immediately-nulled pattern goes away.
- `pyphi.models.mechanism.Unit` renamed to `UnitState` to disambiguate from `pyphi.core.unit.Unit`.
- The models layer is a pure data layer: it does not import formalism or kernel-operation modules. Pinned by a layering test.
- The public API (`from pyphi.models import Concept, RIA, …`) is preserved through `__init__.py` re-exports.

## 3. Non-goals

- **Φ-folds.** Defined in Mayner et al. 2024 (matching paper). Belongs in P14b, where it has consumers (perception, differentiation, matching). Adding it now is speculative.
- **Triggering, perception, differentiation, matching.** All P14b.
- **Splitting `models/fmt.py` (1027 lines).** Cosmetic; defer to P15 cleanup.
- **`models/actual_causation.py` rewrite.** Couples to `pyphi/actual.py` mutability; P14.
- **Removing `SystemIrreducibilityAnalysis.subsystem` / `.cut_subsystem`.** Unlike `Concept.subsystem`, the IIT-3.0 SIA actively uses these as back-references (its `cut` and `network` properties dereference them, and they appear in `__hash__` and `__eq__`). Removing them requires adding explicit fields and migrating IIT-3.0 SIA computation paths — out of scope for a refactor pass.
- **Reorganizing `pyphi/relations.py`.** Likely a P14b concern (since folds + matching are the main consumers).
- **No new formalism Protocol changes.** The kernel/formalism split established in P7 is the contract for downstream layers.

## 4. Architecture

### 4.1 New file layout

```
pyphi/models/
├── __init__.py                    re-exports preserved (stable user API)
├── ria.py                  NEW    RepertoireIrreducibilityAnalysis,
│                                  _null_ria, ShortCircuitConditions,
│                                  signed_phi field (NEW)
├── mice.py                 NEW    MaximallyIrreducibleCauseOrEffect,
│                                  MaximallyIrreducibleCause,
│                                  MaximallyIrreducibleEffect
├── concept.py              NEW    Concept (without subsystem back-ref)
├── state_specification.py  NEW    StateSpecification,
│                                  DistinctionPhiNormalizationRegistry,
│                                  UnitState (renamed from Unit)
├── ces.py                  NEW    CauseEffectStructure, _null_ces
├── sia.py                  NEW    SystemIrreducibilityAnalysis (IIT 3.0),
│                                  SystemStateSpecification, _null_sia
├── phi_structure.py        NEW    PhiStructure (moved from formalism/iit4
│                                  + cleaned up as a frozen dataclass)
├── cuts.py                        unchanged from P6
├── cmp.py                         unchanged
├── pandas.py                      unchanged
├── fmt.py                         unchanged (split deferred to P15)
└── actual_causation.py            unchanged (deferred to P14)
```

`pyphi/models/mechanism.py` and `pyphi/models/subsystem.py` are deleted at the end of the split. No transitional re-export shims (per "migration cost zero — but order things sensibly"; P14 already requires users to update for actual.py / macro.py changes, so a single migration pass is fine).

### 4.2 `RepertoireIrreducibilityAnalysis.signed_phi` (deferred from P5)

Mirror of `SystemIrreducibilityAnalysis.signed_phi`. The current RIA stores raw `phi` (which can be negative under preventative-cause semantics). Add `signed_phi` as a non-optional sibling field; clamp `phi = max(0, signed_phi)` in `__post_init__` (using `pyphi.utils.positive_part`).

```python
@dataclass
class RepertoireIrreducibilityAnalysis(...):
    phi: PyPhiFloat | DistanceResult
    direction: Direction
    mechanism: tuple[int, ...]
    purview: tuple[int, ...]
    partition: Bipartition
    repertoire: Repertoire
    partitioned_repertoire: Repertoire
    ...
    # NEW:
    signed_phi: PyPhiFloat | DistanceResult | None = None

    def __post_init__(self) -> None:
        # Snapshot raw signed value before clamping (mirror of SIA pattern).
        if self.signed_phi is None:
            self.signed_phi = self.phi
        clamped = utils.positive_part(self.signed_phi)
        # ... (PyPhiFloat / DistanceResult wrapping logic, identical to SIA)
        ...
```

The same `_public_aux_data()` pattern from `SystemIrreducibilityAnalysis.__post_init__` is reused for `DistanceResult`-backed values. `_null_ria()` accepts `signed_phi` symmetrically.

**Acceptance:** existing fixtures unchanged where signed_phi == phi (most positive cases); regenerate the grid3 fixture if it produces a negative-signed RIA (canonicalizes the new contract). Hypothesis property test: `ria.phi == positive_part(ria.signed_phi)` for random subsystems.

### 4.3 `PhiStructure` cleanup

Current (`pyphi/formalism/iit4/__init__.py:689`):

```python
class PhiStructure(cmp.Orderable):
    _SIA_INHERITED_ATTRIBUTES = ["phi", "partition", "system_state"]

    def __init__(self, sia, distinctions, relations):
        self._sia = sia
        self._distinctions = distinctions
        self._relations = relations

    @property
    def sia(self): ...
    @property
    def distinctions(self): ...
    @property
    def relations(self): ...
    @property
    def components(self): ...

    def __getattr__(self, attr):
        if attr in self._SIA_INHERITED_ATTRIBUTES:
            return getattr(self.sia, attr)
        ...
```

Target (`pyphi/models/phi_structure.py`):

```python
@dataclass(frozen=True)
class PhiStructure(cmp.Orderable):
    """A Φ-structure: a SIA bundled with its cause-effect structure (distinctions)
    and relations.

    Note: ``phi``, ``partition``, ``system_state`` are not duplicated as fields
    here; access them via ``ps.sia.phi`` etc. The legacy ``__getattr__`` proxy
    that surfaced them at the top level is removed — see migration note below.
    """
    sia: SystemIrreducibilityAnalysis
    distinctions: CauseEffectStructure
    relations: Relations

    @property
    def components(self) -> Iterable[Component]: ...

    def order_by(self) -> PyPhiFloat:
        return self.sia.phi
```

Migration: replace `phi_structure.phi` → `phi_structure.sia.phi` etc. in the ~5 internal call sites. No external callers are documented in the public API, so this is a low-impact change.

### 4.4 `Concept.subsystem` removal

Audit shows:
- `models/mechanism.py:1162–1188` — five `expand_*_repertoire` methods that referenced `self.subsystem` are already commented out (`TODO(4.0) REMOVE`).
- `compute/subsystem.py:118` sets `concept.subsystem = subsystem` after construction; `compute/subsystem.py:93` sets it back to `None` before serialization.
- `test/test_big_phi.py:287` — one assertion `concept.subsystem is ces.subsystem`.

Remove the field from `Concept`. Drop the set/null pattern in `compute/subsystem.py`. Update the test (the assertion was checking the back-reference was consistent, which is no longer meaningful).

`CauseEffectStructure.subsystem` is in the same situation (`models/subsystem.py:303` `self.subsystem = subsystem`); it is removed in the same pass.

`SystemIrreducibilityAnalysis.subsystem` (IIT 3.0) is **not** removed — it is actively used. See non-goals.

### 4.5 `Unit` disambiguation

| Class | Fields | Concept |
|---|---|---|
| `pyphi.core.unit.Unit` | `index, label` | Substrate-level identity (a node) |
| `pyphi.models.mechanism.Unit` (current) | `index, state, label?` | A node *plus its state value* |

Rename `pyphi.models.mechanism.Unit` → `UnitState`, lifted into `pyphi/models/state_specification.py`. The new name parallels `StateSpecification`'s vocabulary (it's the per-unit projection of a system state). Update the ~5 references in `models/mechanism.py` and `models/fmt.py`.

### 4.6 Layering rule for the models tier

Pure data tier: models don't depend on operations. Pinned by an architectural test:

```python
# test/test_models_layering.py

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
            i for i in imports
            if i == "pyphi.core.repertoire_algebra" or i.endswith(".repertoire_algebra")
        ]
        assert not offenders, f"{py}: imports {offenders}"
```

Both walks use `ast.walk` so lazy imports inside method bodies are also caught (same pattern as P7's `test_core_layering`).

### 4.7 `__init__.py` re-exports

Preserved verbatim. The split is internal organization; user imports of the form `from pyphi.models import Concept, RIA, MaximallyIrreducibleCause, ...` keep working.

Specifically, the new `__init__.py`:

```python
from .ces import CauseEffectStructure, _null_ces
from .concept import Concept
from .mice import (
    MaximallyIrreducibleCause,
    MaximallyIrreducibleCauseOrEffect,
    MaximallyIrreducibleEffect,
)
from .phi_structure import PhiStructure
from .ria import RepertoireIrreducibilityAnalysis, ShortCircuitConditions, _null_ria
from .sia import SystemIrreducibilityAnalysis, SystemStateSpecification, _null_sia
from .state_specification import (
    DistinctionPhiNormalizationRegistry,
    StateSpecification,
    UnitState,
)
# unchanged:
from .actual_causation import ...
from .cuts import ...
```

Internal-private names (`_null_ria`, `_null_ces`, `_null_sia`) stay re-exported because `pyphi.formalism.queries` and `pyphi.formalism.iit4` import them from `pyphi.models` directly.

## 5. Decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| Q1 | Φ-folds in scope? | **No, → P14b** | No consumer in core PyPhi today; matching/perception is its actual home |
| Q2 | Move `PhiStructure` to models? | **Yes, and clean it up** | Touching the file at all means it should leave in good shape; legacy `__getattr__` proxy removed |
| Q3 | Remove `Concept.subsystem` and `CauseEffectStructure.subsystem`? | **Yes** | Vestigial back-references; methods using them are already commented out; one test to update |
| Q4 | Remove `SIA.subsystem` (IIT 3.0)? | **No, deferred** | Actively used (`cut`, `network` properties); requires explicit-field migration of compute paths — separate scope |
| Q5 | Resolve `Unit` duplication? | **Rename `models.mechanism.Unit` → `UnitState`** | The two classes model different concepts; renaming disambiguates without semantic change |
| Q6 | `__init__.py` re-export shape? | **Preserve flat re-exports** | Standard Python pattern; users shouldn't pay for our internal reorg |
| Q7 | Transitional shims for `models/mechanism.py` and `models/subsystem.py` after split? | **No shims; delete** | Migration cost is irrelevant for ordering decisions; bundling with P14's user-facing breaks is fine |
| Q8 | `signed_phi` for RIA — what shape? | **Mirror SIA's pattern from P5** | Consistency with system-level signed-phi; `__post_init__` clamps `phi = positive_part(signed_phi)`; raw value stored as `signed_phi` |
| Q9 | Layering rule | **Models do not import `pyphi.formalism` or `pyphi.core.repertoire_algebra`; pinned by AST-walking architectural test** | Establishes the data tier explicitly; mirrors P7's kernel-layering test |

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Splitting moves classes → unintended import-cycle changes | The model classes already only import from `pyphi.utils`, `pyphi.direction`, `pyphi.data_structures`, `pyphi.metrics.distribution`, and each other (within models). Splits respect this graph. Architectural test catches violations. |
| `signed_phi` for RIA changes user-visible numbers | Same shape as SIA: `phi` clamped, `signed_phi` raw. For RIAs where signed_phi ≥ 0 (the common case), `phi` is unchanged. For preventative-cause RIAs, the current `phi` would become non-negative — flag any golden fixture that captures negative RIA phi for regeneration. |
| `PhiStructure.__getattr__` proxy removal silently breaks call sites | Grep is straightforward (`phi_structure.phi`, `phi_structure.partition`, `phi_structure.system_state` are the only delegated attributes). All ~5 sites updated explicitly. |
| Re-export drift — a class moves but `__init__.py` doesn't update | A pre-existing surface-drift test (`test_subsystem_surface`) plus runtime `from pyphi.models import X` smoke tests catch this immediately. |
| Concept.subsystem removal breaks external users | The field was always documented as a workaround; the methods that used it are commented out; the external user-facing impact is essentially zero. Note in changelog. |

## 7. Acceptance criteria

1. **Golden fixtures (P1):** all 17 match unchanged. (Possible exception: a preventative-cause RIA fixture may need regeneration — flag and re-pin if so.)
2. **Hypothesis invariants (P2):** existing 20 green; new property test for `ria.phi == positive_part(ria.signed_phi)` added and green.
3. **Surface drift:** the retargeted `test_subsystem_surface` (CandidateSystem) stays green; no models-tier surface drift.
4. **Layering tests:** new `test_models_layering.py` green.
5. **Pyright:** `pyphi/models/*` and `pyphi/core/*` clean.
6. **Full suite:** fast lane + Hypothesis lane + golden regression all green.
7. **Sign-flip canary:** mutating `hamming_emd` still fails ≥3 fixtures + ≥1 property test.

## 8. Out-of-scope items tracked for follow-up

| Item | Future home |
|---|---|
| Φ-fold model class | P14b (matching/perception) |
| Triggering, perception, differentiation, matching | P14b |
| `models/fmt.py` split (1027 lines) | P15 cleanup |
| `models/actual_causation.py` rewrite | P14 |
| `SystemIrreducibilityAnalysis.subsystem` removal (IIT 3.0) | Future SIA-layer commit (separate scope from a model split) |
| `pyphi/relations.py` reorganization | P14b (where folds + matching consume relations) |

---

End of design.
