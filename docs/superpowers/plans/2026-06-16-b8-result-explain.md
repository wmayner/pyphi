# B8 — `result.explain()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `.explain()` to every top-level PyPhi result type, returning a typed, displayable account of *why* a φ/α quantity came out as it did, and unify the two divergent `ShortCircuitConditions` enums into one `NullResultReason`.

**Architecture:** A new pure-data module `pyphi/models/explanation.py` holds `NullResultReason` (flat enum + `.level`), `Finding`, `Explanation` (a `Displayable` with `to_pandas()`), and `RunnerUp`. Computation paths are instrumented to *retain* the data `.explain()` reads — null-result reasons on the 3.0/AC paths (the 4.0 path already keeps them) and a lightweight runner-up record at the IIT MIP-selection sites. `.explain()` is a pure read over retained fields; it never recomputes and never changes a φ value.

**Tech Stack:** Python 3.12+, `enum`, `dataclasses`, the `pyphi.display` description/backend model (B21), `pandas` (optional, via the existing `ToPandasMixin` pattern), `pytest` (run with `uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-06-16-b8-result-explain-design.md`

**Two refinements to the spec, decided during planning:**
1. `NullResultReason.level ∈ {"system", "mechanism"}` only (structural). The spec's third value `"actual_causation"` conflated formalism with structural level — e.g. `NO_SYSTEM` is a system-level reason shared by IIT *and* AC, so formalism is an orthogonal axis. The formalism is conveyed by the `Explanation.subject` string, not `.level`.
2. The runner-up / φ-gap finding is IIT-only (3.0 + 4.0) in v1. The AC system path reduces with `reduce_func=min` (`pyphi/formalism/actual_causation/compute.py:545`) and never materializes a candidate list, so a runner-up is not cheaply available there. AC `.explain()` surfaces the null-result reason, winning partition, and binding direction; an AC α-gap is a documented follow-up.

---

## File Structure

- **Create** `pyphi/models/explanation.py` — `NullResultReason`, `Finding`, `Explanation`, `RunnerUp`, and the helper `runner_up_from_candidates`. Pure data + display; imports only `pyphi.display`, `pyphi.utils`, `pyphi.data_structures`. No formalism/kernel imports (a layering test pins this).
- **Create** `test/test_explanation.py` — unit tests for the module and the `.explain()` coverage invariant + per-level numeric assertions.
- **Modify** `pyphi/models/ria.py` — delete the local `ShortCircuitConditions`; import `NullResultReason`; add `_findings()` + `explain()`.
- **Modify** `pyphi/formalism/iit4/__init__.py` — delete the local `ShortCircuitConditions`; import `NullResultReason`; add `runner_up` field + retention; add `_findings()` + `explain()` to the 4.0 SIA.
- **Modify** `pyphi/formalism/queries.py` — update the two `ShortCircuitConditions` references.
- **Modify** `pyphi/models/sia.py` — add `reasons` + `runner_up` to the 3.0 SIA; `_null_sia(reasons=…)`; `_findings()` + `explain()`.
- **Modify** `pyphi/formalism/iit3/__init__.py` — pass `reasons=` at each null site; retain runner-up in `_sia_map_reduce`.
- **Modify** `pyphi/models/actual_causation.py` — add `reasons` to `AcRepertoireIrreducibilityAnalysis` / `AcSystemIrreducibilityAnalysis`; `_null_ac_ria(reasons=…)` / `_null_ac_sia(reasons=…)`; `_findings()` + `explain()` for `AcRIA`/`AcSIA`/`Account`.
- **Modify** `pyphi/formalism/actual_causation/compute.py` — pass `reasons=` at each AC null site.
- **Modify** `pyphi/models/mice.py` — `explain()` delegating to `.ria`.
- **Modify** `pyphi/models/distinction.py` — `explain()` composing cause/effect.
- **Modify** `pyphi/models/__init__.py` — re-export `NullResultReason`, `Finding`, `Explanation`.
- **Create** `changelog.d/b8-result-explain.feature.md`.
- **Modify** `ROADMAP.md` — flip the B8 dashboard row to ✅.

---

## Task 1: `NullResultReason` enum + `.level`

**Files:**
- Create: `pyphi/models/explanation.py`
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_explanation.py
"""Tests for pyphi.models.explanation (B8 result.explain())."""

import pytest

from pyphi.models.explanation import NullResultReason


def test_every_reason_has_a_structural_level():
    for reason in NullResultReason:
        assert reason.level in {"system", "mechanism"}


def test_level_partition_is_correct():
    system = {
        NullResultReason.NO_SYSTEM,
        NullResultReason.NO_STRONG_CONNECTIVITY,
        NullResultReason.MONAD_WITH_NO_SELFLOOP,
        NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI,
        NullResultReason.NO_VALID_PARTITIONS,
        NullResultReason.NO_CAUSE,
        NullResultReason.NO_EFFECT,
        NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE,
    }
    mechanism = {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
    }
    assert {r for r in NullResultReason if r.level == "system"} == system
    assert {r for r in NullResultReason if r.level == "mechanism"} == mechanism
    assert system | mechanism == set(NullResultReason)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.models.explanation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/models/explanation.py
"""Typed explanations of why a result came out as it did (``result.explain()``).

``NullResultReason`` enumerates the conditions under which an analysis yields a
trivial (φ = 0 / α = 0) result. ``Finding`` and ``Explanation`` are the typed
account ``.explain()`` returns; ``RunnerUp`` is the lightweight record of the
second-best partition retained at MIP selection.
"""

from __future__ import annotations

from enum import Enum
from enum import auto
from enum import unique


@unique
class NullResultReason(Enum):
    """A condition under which an analysis returns a trivial null result."""

    # System level
    NO_SYSTEM = auto()
    NO_STRONG_CONNECTIVITY = auto()
    MONAD_WITH_NO_SELFLOOP = auto()
    MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI = auto()
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    EMPTY_CAUSE_EFFECT_STRUCTURE = auto()
    # Mechanism level
    NO_PURVIEWS = auto()
    NO_PARTITIONS = auto()
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()
    REDUCIBLE_OVER_PARTITION = auto()

    @property
    def level(self) -> str:
        """The structural level the reason arises at: ``"system"`` or
        ``"mechanism"``."""
        return _LEVEL_OF[self]


_MECHANISM_REASONS = frozenset(
    {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
    }
)

_LEVEL_OF: dict[NullResultReason, str] = {
    reason: ("mechanism" if reason in _MECHANISM_REASONS else "system")
    for reason in NullResultReason
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/explanation.py test/test_explanation.py
git commit -m "Add NullResultReason enum with structural .level (B8)"
```

---

## Task 2: `Finding`, `RunnerUp`, `Explanation` (+ display + pandas)

**Files:**
- Modify: `pyphi/models/explanation.py`
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
from pyphi.models.explanation import Explanation
from pyphi.models.explanation import Finding


def test_explanation_describe_and_pandas():
    expl = Explanation(
        subject="Φ_s = 0.0",
        level="system",
        findings=(
            Finding(
                kind="null_result",
                label="Reason",
                value=NullResultReason.NO_STRONG_CONNECTIVITY,
            ),
            Finding(kind="binding_direction", label="Binding direction", value="CAUSE"),
        ),
    )
    # repr/HTML render without error and mention the subject.
    assert "Φ_s = 0.0" in repr(expl)
    assert "NO_STRONG_CONNECTIVITY" in repr(expl)
    assert "<" in expl._repr_html_()  # HTML backend produced markup

    df = expl.to_pandas()
    assert list(df.columns) == ["level", "kind", "label", "value"]
    assert len(df) == 2
    assert df.iloc[0]["kind"] == "null_result"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_explanation_describe_and_pandas -q`
Expected: FAIL with `ImportError: cannot import name 'Explanation'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/models/explanation.py` (add the new imports at the top of the file):

```python
# --- add to the top-of-file imports ---
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import tone_of  # noqa: F401  (re-exported for callers)
```

```python
# --- append to pyphi/models/explanation.py ---
@dataclass(frozen=True)
class RunnerUp:
    """The second-best partition at MIP selection: the lowest-φ candidate whose
    φ is strictly greater than the MIP's. ``partition`` is the cut; ``phi`` is
    its (clamped) integrated information."""

    partition: Any
    phi: Any


@dataclass(frozen=True)
class Finding:
    """One element of an explanation: a stable machine ``kind``, a human
    ``label``, the ``value`` it concerns, and optional supporting ``detail``."""

    kind: str
    label: str
    value: Any = None
    detail: tuple[tuple[str, Any], ...] = ()
    tone: str | None = None


def _reason_value(value: Any) -> Any:
    """Render a ``NullResultReason`` by its name; pass other values through."""
    return value.name if isinstance(value, NullResultReason) else value


@dataclass(frozen=True)
class Explanation(Displayable):
    """A typed account of why a result came out as it did.

    ``subject`` names the quantity being explained (e.g. ``"Φ_s = 0.0"``);
    ``level`` is ``"system"`` or ``"mechanism"``; ``findings`` is the ordered
    account.
    """

    subject: str
    level: str
    findings: tuple[Finding, ...] = ()

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        rows = tuple(
            Row(f.label, _reason_value(f.value), tone=f.tone) for f in self.findings
        )
        return Description(
            title=self.subject,
            sections=(Section(label="Why", rows=rows),),
            compact=self.subject,
        )

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "level": self.level,
                    "kind": f.kind,
                    "label": f.label,
                    "value": _reason_value(f.value),
                }
                for f in self.findings
            ],
            columns=["level", "kind", "label", "value"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/explanation.py test/test_explanation.py
git commit -m "Add Finding, RunnerUp, Explanation typed display model (B8)"
```

---

## Task 3: Migrate both `ShortCircuitConditions` → `NullResultReason` (value-neutral)

**Files:**
- Modify: `pyphi/models/ria.py:59-67` (delete the enum), references in `pyphi/formalism/queries.py`
- Modify: `pyphi/formalism/iit4/__init__.py:608-616` (delete the enum) and its uses
- Modify: `pyphi/models/__init__.py` (re-export)

- [ ] **Step 1: Enumerate every reference**

Run: `rg -n "ShortCircuitConditions" pyphi/ test/`
Expected sites: the two definitions (`pyphi/models/ria.py:60`, `pyphi/formalism/iit4/__init__.py:608`), imports/uses in `pyphi/formalism/queries.py` (line ~40 + uses), uses in `pyphi/formalism/iit4/__init__.py` (lines ~623, 713, 717, 725, 730, 764, 766), plus any in `pyphi/jsonify.py` or `test/`. Record the full list before editing.

- [ ] **Step 2: Delete the mechanism-level enum and import the unified one**

In `pyphi/models/ria.py`, delete:

```python
@unique_enum
class ShortCircuitConditions(Enum):
    # MICE level reasons
    NO_PURVIEWS = auto()
    NO_PARTITIONS = auto()
    # MIP level reasons
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()
```

and add, with the other model imports:

```python
from pyphi.models.explanation import NullResultReason
```

Then replace every `ShortCircuitConditions` in `pyphi/formalism/queries.py` (the import and the `ShortCircuitConditions.NO_PURVIEWS` / `.EMPTY_PURVIEW` / `.UNREACHABLE_STATE` uses) with `NullResultReason` (import it from `pyphi.models.explanation`).

- [ ] **Step 3: Delete the system-level enum and import the unified one**

In `pyphi/formalism/iit4/__init__.py`, delete:

```python
@unique
class ShortCircuitConditions(Enum):
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    NO_SYSTEM = auto()
    NO_STRONG_CONNECTIVITY = auto()
    MONAD_WITH_NO_SELFLOOP = auto()
    MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI = auto()
```

add `from pyphi.models.explanation import NullResultReason`, and replace every `ShortCircuitConditions.X` in that file with `NullResultReason.X`.

- [ ] **Step 4: Re-export from `pyphi/models/__init__.py`**

Add `NullResultReason`, `Finding`, `Explanation` to the imports and `__all__` in `pyphi/models/__init__.py` (follow the existing re-export pattern in that file).

- [ ] **Step 5: Run the affected suites — value-neutral checkpoint**

Run: `uv run pytest test/test_golden_regression.py test/test_subsystem_surface.py test/test_result_protocols.py test/test_explanation.py -q`
Expected: PASS, no φ value changes. (If a test asserts on the old enum *name*, update it to `NullResultReason` — that is an intended rename, not a value change.)

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/ria.py pyphi/formalism/iit4/__init__.py pyphi/formalism/queries.py pyphi/models/__init__.py
git commit -m "Unify the two ShortCircuitConditions enums into NullResultReason (B8)"
```

---

## Task 4: Plumb `reasons=` through the IIT 3.0 null path

**Files:**
- Modify: `pyphi/models/sia.py:63-73` (add `reasons` param/attr), `:233-245` (`_null_sia`)
- Modify: `pyphi/formalism/iit3/__init__.py` (null sites: ~377, 384, 397, 405, 414, and `_sia_map_reduce` ~325/338/341)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
import pyphi
from pyphi import examples


def test_iit3_null_sia_carries_reason():
    # A disconnected 2-node substrate is not strongly connected → null SIA.
    with pyphi.config.override(formalism={"iit": {"version": "IIT_3_0"}}):
        substrate = examples.disjoint_network()  # two independent components
        system = substrate.system(substrate.node_indices)
        analysis = pyphi.formalism.iit3.sia(system)
    assert analysis.phi == 0
    assert NullResultReason.NO_STRONG_CONNECTIVITY in (analysis.reasons or [])
```

(If `examples.disjoint_network` does not exist, build a 2-node substrate with an all-zero off-diagonal CM inline; the point is a non-strongly-connected system. Confirm the helper name with `rg "def .*network" pyphi/examples.py` and use a real one.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_iit3_null_sia_carries_reason -q`
Expected: FAIL — `analysis.reasons` raises `AttributeError` (the 3.0 SIA has no `reasons`).

- [ ] **Step 3: Add `reasons` to the 3.0 SIA and `_null_sia`**

In `pyphi/models/sia.py`, extend `IIT3SystemIrreducibilityAnalysis.__init__` to accept and store reasons:

```python
    def __init__(
        self,
        phi=None,
        distinctions: Distinctions | None = None,
        partitioned_distinctions=None,
        partition=None,
        node_indices=None,
        node_labels=None,
        current_state=None,
        config=None,
        reasons=None,
        runner_up=None,
    ):
        ...
        self.reasons = reasons or []
        self.runner_up = runner_up
```

(Place the two assignments alongside the existing attribute assignments in `__init__`. `runner_up` is added now so Task 6 only sets it.)

Update `_null_sia`:

```python
def _null_sia(system, phi=0.0, reasons=None):
    """Return an IIT3SystemIrreducibilityAnalysis with zero phi.

    This is the analysis result for a reducible system. ``reasons`` records
    why (a list of :class:`~pyphi.models.explanation.NullResultReason`).
    """
    return IIT3SystemIrreducibilityAnalysis(
        phi=phi,
        partitioned_distinctions=_null_ces(),
        partition=system.partition,
        node_indices=system.node_indices,
        node_labels=system.substrate.node_labels,
        current_state=system.state,
        reasons=reasons,
    )
```

Also add `reasons` and `runner_up` to the `to_json` field list (the dict comprehension near `:188-204`) and to `from_json` round-tripping, mirroring how other fields are handled.

- [ ] **Step 4: Pass reasons at each 3.0 null site**

In `pyphi/formalism/iit3/__init__.py`, import the enum (`from pyphi.models.explanation import NullResultReason`) and pass the matching member at each `_null_sia` call:

```python
    if not system:
        return _null_sia(system, reasons=[NullResultReason.NO_SYSTEM])
    if not connectivity.is_strong(system.cm, system.node_indices):
        return _null_sia(system, reasons=[NullResultReason.NO_STRONG_CONNECTIVITY])
    ...
    if not system.cm[system.node_indices][system.node_indices]:
        return _null_sia(system, reasons=[NullResultReason.MONAD_WITH_NO_SELFLOOP])
    ...
    if not config.formalism.iit.single_micro_nodes_with_selfloops_have_phi:
        return _null_sia(
            system,
            reasons=[NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI],
        )
    ...
    if not unpartitioned_ces:
        return _null_sia(system, reasons=[NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE])
```

In `_sia_map_reduce`, set the reason on the two no-partition/no-tie fallbacks:

```python
    null = _null_sia(system, reasons=[NullResultReason.NO_VALID_PARTITIONS])
    ...
    if not candidates:
        return null
    ties = tuple(resolve_ties.sias(candidates, default=null))
    if not ties:
        return null
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py::test_iit3_null_sia_carries_reason -q`
Expected: PASS.

- [ ] **Step 6: Run the 3.0 golden + serialization suites (no value drift)**

Run: `uv run pytest test/test_golden_regression.py test/test_serialization.py -q`
Expected: PASS. If a 3.0 SIA repr golden changed because the card now shows a Reasons row, regenerate it deliberately and review the diff as an intended surface change.

- [ ] **Step 7: Commit**

```bash
git add pyphi/models/sia.py pyphi/formalism/iit3/__init__.py test/test_explanation.py
git commit -m "Record null-result reasons on the IIT 3.0 SIA path (B8)"
```

---

## Task 5: Plumb `reasons=` through the actual-causation null path

**Files:**
- Modify: `pyphi/models/actual_causation.py:240-251` (`_null_ac_ria`), `:708` (`_null_ac_sia`), and the `AcRepertoireIrreducibilityAnalysis` / `AcSystemIrreducibilityAnalysis` constructors (add a `reasons` attr)
- Modify: `pyphi/formalism/actual_causation/compute.py:209, 231, 301, 508, 515, 528` (pass reasons)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_ac_null_sia_carries_reason():
    transition = examples.actual_causation_substrate()  # OR/AND 2-unit
    # An empty/over-restricted transition yields a null AC SIA; assert the
    # reason is recorded. Use a transition that short-circuits on connectivity
    # (confirm the right example/state with rg over pyphi/examples.py + actual.py).
    sia = pyphi.actual.sia(transition, direction=pyphi.Direction.CAUSE)
    if sia.alpha == 0:
        assert sia.reasons  # non-empty list of NullResultReason
        assert all(isinstance(r, NullResultReason) for r in sia.reasons)
```

(Pin a concrete short-circuiting transition by reading `test/test_actual.py` for an existing null-AC-SIA fixture and asserting the exact reason; replace the conditional with a direct equality once pinned.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_ac_null_sia_carries_reason -q`
Expected: FAIL — `sia.reasons` raises `AttributeError`.

- [ ] **Step 3: Add `reasons` to the AC result constructors and null helpers**

In `pyphi/models/actual_causation.py`, add a `reasons` parameter (default `None`, stored as `self.reasons = reasons or []`) to `AcRepertoireIrreducibilityAnalysis.__init__` and `AcSystemIrreducibilityAnalysis.__init__`, and thread it through the null helpers:

```python
def _null_ac_ria(state, direction, mechanism, purview, partition=None, reasons=None):
    """The irreducibility AC analysis for a reducible causal link."""
    return AcRepertoireIrreducibilityAnalysis(
        state=state,
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        probability=None,
        partitioned_probability=None,
        alpha=0.0,
        reasons=reasons,
    )


def _null_ac_sia(transition, direction, alpha=0.0, reasons=None):
    ...  # add reasons=reasons to the AcSystemIrreducibilityAnalysis(...) call
```

- [ ] **Step 4: Pass reasons at each AC null site**

In `pyphi/formalism/actual_causation/compute.py`, import `NullResultReason` and pass the matching member:

```python
    # find_causal_link / null RIA sites
    return _null_ac_ria(..., reasons=[NullResultReason.EMPTY_PURVIEW])      # :209
    return _null_ac_ria(..., reasons=[NullResultReason.REDUCIBLE_OVER_PARTITION])  # :231
    max_ria = _null_ac_ria(..., reasons=[NullResultReason.NO_PURVIEWS])     # :301
    # system sia sites
    return _null_ac_sia(transition, direction, reasons=[NullResultReason.NO_SYSTEM])  # :508
    return _null_ac_sia(transition, direction,
                        reasons=[NullResultReason.NO_STRONG_CONNECTIVITY])  # :515
    return _null_ac_sia(transition, direction,
                        reasons=[NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE])  # :528
```

(The `:547` `default=_null_ac_sia(..., alpha=float("inf"))` is the min-reduce identity element, not a real short-circuit — leave its `reasons` empty.)

**Confirmation experiment (do not skip — per CLAUDE.md):** AC's system check is `connectivity.is_weak` (`compute.py:510`), logged as "not strongly/weakly connected". Before assigning `NO_STRONG_CONNECTIVITY` to the `:515` site, confirm whether AC's connectivity predicate is genuinely the same notion as the IIT `is_strong` check or a weaker one. If it is weak-connectivity-specific, add a distinct `NO_WEAK_CONNECTIVITY` member (system level) in `explanation.py` + Task 1's level test instead of reusing `NO_STRONG_CONNECTIVITY`. Record the finding in the commit message.

- [ ] **Step 5: Run test + AC suite**

Run: `uv run pytest test/test_explanation.py::test_ac_null_sia_carries_reason test/test_actual.py -q`
Expected: PASS, no α value drift.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/actual_causation.py pyphi/formalism/actual_causation/compute.py test/test_explanation.py pyphi/models/explanation.py
git commit -m "Record null-result reasons on the actual-causation path (B8)"
```

---

## Task 6: Retain the runner-up at the IIT MIP-selection sites

**Files:**
- Modify: `pyphi/models/explanation.py` (add `runner_up_from_candidates`)
- Modify: `pyphi/formalism/iit4/__init__.py:161-174` (add `runner_up` field), `:980-988` (retain)
- Modify: `pyphi/formalism/iit3/__init__.py:_sia_map_reduce` (retain; field added in Task 4)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_runner_up_retained_on_phi_positive_system():
    substrate = examples.basic_substrate()
    system = substrate.system(substrate.node_indices)
    sia = pyphi.formalism.iit4.sia(
        system,
        system_measure=...,            # use the formalism's default measures, as
        specification_measure=...,     # other iit4 tests construct them; or call
    )                                  # the high-level entry that supplies them.
    # A φ>0 system has at least two distinct partition φ values → a runner-up.
    assert sia.phi > 0
    assert sia.runner_up is not None
    assert float(sia.runner_up.phi) >= float(sia.phi)
```

(Construct `sia` exactly as an existing `test/` IIT 4.0 test does — find the canonical call with `rg "iit4.*sia\(" test/` and reuse its measure wiring rather than inventing it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_runner_up_retained_on_phi_positive_system -q`
Expected: FAIL — `sia.runner_up` is `AttributeError` (4.0 SIA has no such field).

- [ ] **Step 3: Add the selection helper**

Append to `pyphi/models/explanation.py`:

```python
from pyphi import utils  # add to top-of-file imports


def runner_up_from_candidates(candidates, mip_phi) -> "RunnerUp | None":
    """The lowest-φ candidate whose φ is *strictly* greater than ``mip_phi``.

    Ties with the MIP (within ``utils.eq``) are tied peers, not runners-up, so
    they are excluded. Returns ``None`` when the MIP is the unique φ value.
    Each candidate must expose ``.phi`` and ``.partition``.
    """
    mip = float(mip_phi)
    best = None
    for candidate in candidates:
        phi = float(candidate.phi)
        if phi > mip and not utils.eq(phi, mip):
            if best is None or phi < float(best.phi):
                best = candidate
    if best is None:
        return None
    return RunnerUp(partition=best.partition, phi=best.phi)
```

- [ ] **Step 4: Add the `runner_up` field + retention (4.0)**

In `pyphi/formalism/iit4/__init__.py`, add to the SIA dataclass fields (after `reasons`):

```python
    runner_up: Any = None
```

and import the helper: `from pyphi.models.explanation import runner_up_from_candidates`.
In `_find_mip_for_fixed_state`, after the MIP is chosen:

```python
    ties = tuple(resolve_ties.sias(candidates))
    mip_sia = ties[0]
    mip_sia.runner_up = runner_up_from_candidates(candidates, mip_sia.phi)
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)
    return mip_sia
```

- [ ] **Step 5: Add the runner-up retention (3.0)**

In `pyphi/formalism/iit3/__init__.py` `_sia_map_reduce`, after `winner = ties[0]`:

```python
    from pyphi.models.explanation import runner_up_from_candidates

    winner = ties[0]
    winner.runner_up = runner_up_from_candidates(candidates, winner.phi)
```

(The `runner_up` attribute already exists on the 3.0 SIA from Task 4.)

- [ ] **Step 6: Run test + golden suite (no φ drift)**

Run: `uv run pytest test/test_explanation.py::test_runner_up_retained_on_phi_positive_system test/test_golden_regression.py -q`
Expected: PASS, no φ value changes (the runner-up is additive metadata).

- [ ] **Step 7: Commit**

```bash
git add pyphi/models/explanation.py pyphi/formalism/iit4/__init__.py pyphi/formalism/iit3/__init__.py test/test_explanation.py
git commit -m "Retain the lightweight runner-up at IIT MIP selection (B8)"
```

---

## Task 7: `.explain()` on the IIT 4.0 SIA

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (add `_findings()` + `explain()` to `SystemIrreducibilityAnalysis`)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_iit4_sia_explain_short_circuit_and_positive():
    # Short-circuit case: not strongly connected → NO_STRONG_CONNECTIVITY finding.
    null_substrate = examples.disjoint_network()
    null_system = null_substrate.system(null_substrate.node_indices)
    null_sia = pyphi.formalism.iit4.sia(null_system, system_measure=..., specification_measure=...)
    expl = null_sia.explain()
    assert expl.level == "system"
    kinds = {f.kind for f in expl.findings}
    assert "null_result" in kinds
    assert any(f.value is NullResultReason.NO_STRONG_CONNECTIVITY for f in expl.findings)

    # φ>0 case: winning partition + runner-up/gap + binding direction findings.
    sub = examples.basic_substrate()
    sia = pyphi.formalism.iit4.sia(sub.system(sub.node_indices), system_measure=..., specification_measure=...)
    expl = sia.explain()
    kinds = {f.kind for f in expl.findings}
    assert {"winning_partition", "binding_direction"} <= kinds
    if sia.runner_up is not None:
        gap = next(f for f in expl.findings if f.kind == "gap")
        assert float(gap.value) == pytest.approx(float(sia.runner_up.phi) - float(sia.phi))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_iit4_sia_explain_short_circuit_and_positive -q`
Expected: FAIL — `null_sia.explain` is `AttributeError`.

- [ ] **Step 3: Implement `_findings()` + `explain()` on the 4.0 SIA**

Add to `SystemIrreducibilityAnalysis` (in `pyphi/formalism/iit4/__init__.py`), importing `Explanation`, `Finding` from `pyphi.models.explanation`:

```python
    def _findings(self) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        for reason in self.reasons or []:
            findings.append(
                Finding(kind="null_result", label="Null result", value=reason)
            )
        if self.partition is not None and bool(self):
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                    detail=(("connections cut", self.partition.num_connections_cut()),),
                )
            )
        if self.runner_up is not None:
            findings.append(
                Finding(
                    kind="runner_up",
                    label="Runner-up partition",
                    value=concise_partition(self.runner_up.partition),
                )
            )
            findings.append(
                Finding(
                    kind="gap",
                    label="φ-gap to runner-up",
                    value=PyPhiFloat(float(self.runner_up.phi) - float(self.phi)),
                )
            )
        if self.cause is not None and self.effect is not None:
            binding = (
                Direction.CAUSE
                if float(self.cause.phi) <= float(self.effect.phi)
                else Direction.EFFECT
            )
            findings.append(
                Finding(
                    kind="binding_direction",
                    label="Binding direction",
                    value=binding.name,
                    detail=(
                        ("φ_cause", self.cause.phi),
                        ("φ_effect", self.effect.phi),
                    ),
                    tone="cause" if binding is Direction.CAUSE else "effect",
                )
            )
        return tuple(findings)

    def explain(self) -> Explanation:
        """A typed account of why this Φ_s value came out as it did."""
        return Explanation(
            subject=f"Φ_s = {format_value(self.phi)}",
            level="system",
            findings=self._findings(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py::test_iit4_sia_explain_short_circuit_and_positive -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/test_explanation.py
git commit -m "Add .explain() to the IIT 4.0 SIA (B8)"
```

---

## Task 8: `.explain()` on the IIT 3.0 SIA

**Files:**
- Modify: `pyphi/models/sia.py` (`_findings()` + `explain()`)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_iit3_sia_explain():
    with pyphi.config.override(formalism={"iit": {"version": "IIT_3_0"}}):
        sub = examples.basic_substrate()
        sia = pyphi.formalism.iit3.sia(sub.system(sub.node_indices))
    expl = sia.explain()
    assert expl.level == "system"
    assert expl.subject.startswith("Φ")
    if sia.phi > 0:
        assert any(f.kind == "winning_partition" for f in expl.findings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_iit3_sia_explain -q`
Expected: FAIL — `AttributeError: explain`.

- [ ] **Step 3: Implement `_findings()` + `explain()` on the 3.0 SIA**

Add to `IIT3SystemIrreducibilityAnalysis` in `pyphi/models/sia.py` (import `Explanation`, `Finding`, `concise_partition`, `PyPhiFloat`, `format_value` are already imported / available):

```python
    def _findings(self):
        from pyphi.models.explanation import Finding

        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if self.partition is not None and self.phi and self.phi > 0:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                )
            )
        if self.runner_up is not None:
            findings.append(
                Finding(
                    kind="gap",
                    label="Φ-gap to runner-up",
                    value=float(self.runner_up.phi) - float(self.phi),
                )
            )
        return tuple(findings)

    def explain(self):
        from pyphi.models.explanation import Explanation

        return Explanation(
            subject=f"Φ = {format_value(self.phi)}",
            level="system",
            findings=self._findings(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py::test_iit3_sia_explain -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/sia.py test/test_explanation.py
git commit -m "Add .explain() to the IIT 3.0 SIA (B8)"
```

---

## Task 9: `.explain()` on the mechanism level (RIA / MICE / Distinction)

**Files:**
- Modify: `pyphi/models/ria.py` (`_findings()` + `explain()`)
- Modify: `pyphi/models/mice.py` (delegate to `.ria`)
- Modify: `pyphi/models/distinction.py` (compose cause/effect)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_mechanism_explain():
    sub = examples.basic_substrate()
    system = sub.system(sub.node_indices)
    distinction = system.distinction((0,))  # confirm the real accessor with rg
    expl = distinction.explain()
    assert expl.level == "mechanism"
    # The binding direction (cause vs effect) is reported for a distinction.
    assert any(f.kind == "binding_direction" for f in expl.findings)

    # A MICE delegates to its RIA.
    mice = distinction.cause
    assert mice.explain().level == "mechanism"
```

(Confirm `system.distinction(...)` / the canonical distinction constructor with `rg "def distinction" pyphi/` and use the real call.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_mechanism_explain -q`
Expected: FAIL — `AttributeError: explain`.

- [ ] **Step 3: Implement RIA `_findings()` + `explain()`**

Add to `RepertoireIrreducibilityAnalysis` in `pyphi/models/ria.py` (it already imports `Description`/`Row`/`Section`; import `Explanation`, `Finding` from `pyphi.models.explanation`):

```python
    def _findings(self):
        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if self.purview:
            findings.append(
                Finding(kind="purview", label="Purview", value=self.purview)
            )
        if self.partition is not None:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                )
            )
        return tuple(findings)

    def explain(self):
        return Explanation(
            subject=f"φ = {format_value(self.phi)}",
            level="mechanism",
            findings=self._findings(),
        )
```

- [ ] **Step 4: Implement MICE + Distinction delegation/composition**

In `pyphi/models/mice.py`, add to `MaximallyIrreducibleCauseOrEffect`:

```python
    def explain(self):
        return self.ria.explain()
```

In `pyphi/models/distinction.py`, add to `Distinction` (it already exposes `.cause`/`.effect` MICE and `.phi = min`):

```python
    def explain(self):
        from pyphi.models.explanation import Explanation, Finding

        binding = self.cause if float(self.cause.phi) <= float(self.effect.phi) else self.effect
        direction_name = "CAUSE" if binding is self.cause else "EFFECT"
        findings = [
            Finding(
                kind="binding_direction",
                label="Binding direction",
                value=direction_name,
                detail=(("φ_cause", self.cause.phi), ("φ_effect", self.effect.phi)),
                tone="cause" if binding is self.cause else "effect",
            ),
            *binding.explain().findings,
        ]
        return Explanation(
            subject=f"φ = {format_value(self.phi)}",
            level="mechanism",
            findings=tuple(findings),
        )
```

(Confirm `format_value` is importable in `distinction.py`; if not, import it from `pyphi.display.numbers`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py::test_mechanism_explain -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/ria.py pyphi/models/mice.py pyphi/models/distinction.py test/test_explanation.py
git commit -m "Add .explain() to RIA, MICE, and Distinction (B8)"
```

---

## Task 10: `.explain()` on the actual-causation results

**Files:**
- Modify: `pyphi/models/actual_causation.py` (`explain()` on `AcRepertoireIrreducibilityAnalysis`, `AcSystemIrreducibilityAnalysis`, `Account`)
- Test: `test/test_explanation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_explanation.py
def test_ac_explain():
    transition = examples.actual_causation_substrate()
    sia = pyphi.actual.sia(transition, direction=pyphi.Direction.CAUSE)
    expl = sia.explain()
    assert expl.level == "system"
    assert expl.subject.startswith("α")
    # reasons appear when null; winning partition appears when α>0
    if sia.alpha == 0:
        assert any(f.kind == "null_result" for f in expl.findings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_explanation.py::test_ac_explain -q`
Expected: FAIL — `AttributeError: explain`.

- [ ] **Step 3: Implement AC `explain()` (no runner-up — see plan note)**

Add to `AcSystemIrreducibilityAnalysis`:

```python
    def explain(self):
        from pyphi.display.numbers import format_value
        from pyphi.models.explanation import Explanation, Finding

        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if getattr(self, "partition", None) is not None and self.alpha:
            findings.append(
                Finding(kind="winning_partition", label="Partition", value=self.partition)
            )
        return Explanation(
            subject=f"α = {format_value(self.alpha)}",
            level="system",
            findings=tuple(findings),
        )
```

Add the analogous `explain()` to `AcRepertoireIrreducibilityAnalysis` (`level="mechanism"`, `subject=f"α = {format_value(self.alpha)}"`, findings from `self.reasons` + purview), and to `Account`:

```python
    def explain(self):
        from pyphi.models.explanation import Explanation, Finding

        findings = [
            Finding(
                kind="link",
                label=str(link),
                value=getattr(link, "alpha", None),
            )
            for link in self
        ]
        return Explanation(
            subject=f"Account ({len(self)} links)",
            level="system",
            findings=tuple(findings),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_explanation.py::test_ac_explain -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/actual_causation.py test/test_explanation.py
git commit -m "Add .explain() to actual-causation results (B8)"
```

---

## Task 11: `.explain()` coverage invariant + full verification

**Files:**
- Modify: `test/test_explanation.py` (coverage invariant)
- Create: `changelog.d/b8-result-explain.feature.md`
- Modify: `ROADMAP.md` (B8 row → ✅)

- [ ] **Step 1: Write the coverage-invariant test**

```python
# append to test/test_explanation.py
@pytest.mark.parametrize(
    "build",
    [
        # (callable returning a top-level result object); pin each with the
        # canonical constructor used elsewhere in test/. Each must return an
        # Explanation with a non-empty findings tuple OR an empty one only for a
        # genuinely featureless result.
    ],
)
def test_explain_is_total(build):
    result = build()
    expl = result.explain()
    from pyphi.models.explanation import Explanation

    assert isinstance(expl, Explanation)
    assert expl.level in {"system", "mechanism"}
```

Populate the parametrize list with one builder per result type: IIT 4.0 SIA, IIT 3.0 SIA, RIA, MICE, Distinction, AcSIA, AcRIA, Account (reuse the constructions from Tasks 7–10).

- [ ] **Step 2: Run the targeted suite**

Run: `uv run pytest test/test_explanation.py -q`
Expected: PASS (all explanation tests).

- [ ] **Step 3: Full suite incl. the doctest sweep (no path argument!)**

Run: `uv run pytest`
Expected: PASS. This collects the `pyphi/` doctests (per `testpaths`/`--doctest-modules`); any repr/HTML drift from new fields surfaces here. Fix doctests that legitimately changed; investigate any φ/α value change as a bug in the additive plumbing.

- [ ] **Step 4: Type check**

Run: `uv run pyright pyphi/models/explanation.py pyphi/models/sia.py pyphi/formalism/iit4/__init__.py`
Expected: no new errors. (If the repo's local `typeCheckingMode = "off"` workaround is present and unstaged, trust the pre-commit hook instead.)

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Added `result.explain()` — a typed account of why a Φ/φ/α quantity came out as it did — across IIT 4.0, IIT 3.0, and actual causation, and unified the two `ShortCircuitConditions` enums into `NullResultReason`.' > changelog.d/b8-result-explain.feature.md
```

- [ ] **Step 6: Update the ROADMAP dashboard**

In `ROADMAP.md`, change the B8 dashboard row status from `⬜ open` to `✅ landed` and update its one-line to past tense (mirror the B16/B21 landed-row style). Do not touch other rows.

- [ ] **Step 7: Commit**

```bash
git add test/test_explanation.py changelog.d/b8-result-explain.feature.md ROADMAP.md
git commit -m "Add .explain() coverage invariant; changelog + ROADMAP for B8"
```

---

## Self-Review notes

- **Spec coverage:** unified enum (Task 1+3) ✓; `.explain()` on all result types (Tasks 7–10) ✓; runner-up/φ-gap retained for IIT (Task 6) ✓ — AC runner-up explicitly deferred (plan note + Task 10) ✓; reason plumbing on 3.0/AC (Tasks 4–5) ✓; Displayable + to_pandas (Task 2) ✓; no-φ-change invariant checked in Tasks 3/4/6 ✓; doctest sweep (Task 11) ✓; AC connectivity confirmation experiment (Task 5 Step 4) ✓.
- **Deviations from spec, flagged to user:** (1) `.level` is `{"system","mechanism"}` only; (2) runner-up is IIT-only in v1.
- **Type consistency:** `Finding(kind, label, value, detail, tone)`, `Explanation(subject, level, findings)`, `RunnerUp(partition, phi)`, `runner_up_from_candidates(candidates, mip_phi)`, `NullResultReason.<MEMBER>.level` — used consistently across tasks.
- **Open confirmations the implementer must resolve (marked inline):** the exact disconnected-substrate / null-AC-transition example names; the canonical IIT 4.0 `sia(...)` measure wiring used in `test/`; the canonical distinction constructor. Each task says to confirm with `rg` and use the real symbol rather than a guess.
