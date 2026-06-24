# B15 — `result.diff()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `a.diff(b)` to every top-level PyPhi result type, returning a typed, displayable, pandas-exportable `ResultDiff` describing what changed between two analyses (Δφ, a real-vs-tie-reshuffle MIP change, distinctions/relations/links gained/lost/changed, and which config differences could explain it).

**Architecture:** A new pure-data module `pyphi/models/diff.py` holds `Change` (a uniform element-delta record) and `ResultDiff` (named scalar fields + a `Change` tuple), reusing B8's `Displayable` + `records_to_frame` patterns and composing the landed `ConfigSnapshot.diff`. Each result type gets a `.diff(other)` method built from a shared `_diff_common(a, b)` helper plus a per-type `_changes(a, b)` hook (mirroring B8's `_findings()`/`explain()`). `.diff()` is a pure read — no recompute, no value change.

**Tech Stack:** Python 3.12+, `dataclasses`, the `pyphi.display` model (B21), `pandas` (via `records_to_frame`), `pytest` (run with `uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-06-16-b15-result-diff-design.md`

---

## File Structure

- **Create** `pyphi/models/diff.py` — `Change`, `ResultDiff`, the `_diff_common(a, b)` helper, and the `_mip_changed(a, b)` / `_config_diff(a, b)` / `_substrate_note(a, b)` primitives. Imports only `pyphi.utils`, `pyphi.display`, `pyphi.models.pandas`. No formalism/kernel imports (pure data, like `explanation.py`).
- **Create** `test/test_result_diff.py` — unit tests + the `.diff()` coverage invariant + per-type numeric + MIP-reshuffle invariant.
- **Modify** `pyphi/models/__init__.py` — re-export `Change`, `ResultDiff`.
- **Modify** `pyphi/formalism/iit4/__init__.py` — `diff()` on the 4.0 SIA.
- **Modify** `pyphi/models/sia.py` — `diff()` on the 3.0 SIA.
- **Modify** `pyphi/models/ces.py` — `diff()` on `CauseEffectStructure` (distinctions/relations deltas).
- **Modify** `pyphi/models/ria.py` / `pyphi/models/mice.py` / `pyphi/models/distinction.py` — `diff()` at the mechanism level.
- **Modify** `pyphi/models/actual_causation.py` — `diff()` on `AcSystemIrreducibilityAnalysis` and `Account`.
- **Create** `changelog.d/b15-result-diff.feature.md`.
- **Modify** `ROADMAP.md` — flip the B15 dashboard row to ✅.

---

## Task 1: `Change` + `ResultDiff` (types + display + pandas)

**Files:**
- Create: `pyphi/models/diff.py`
- Test: `test/test_result_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_result_diff.py
"""Tests for pyphi.models.diff (B15 result.diff())."""


def test_resultdiff_describe_and_pandas():
    from pyphi.models.diff import Change
    from pyphi.models.diff import ResultDiff

    rd = ResultDiff(
        subject="ΔΦ_s = +0.10",
        level="system",
        delta_phi=0.1,
        mip_changed=True,
        binding_direction_changed=False,
        changes=(
            Change(kind="distinction_gained", key=(0,), a_value=None, b_value=0.25),
        ),
        config_diff={"numerics.precision": (13, 6)},
    )
    assert "ΔΦ_s = +0.10" in repr(rd)
    assert "distinction_gained" in repr(rd)
    assert "<" in rd._repr_html_()  # HTML backend rendered markup

    df = rd.to_pandas()
    assert list(df.columns) == ["category", "key", "a", "b"]
    # one row per change + one per config-diff entry + scalar rows
    assert (df["category"] == "distinction_gained").any()
    assert (df["category"] == "config").any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.models.diff'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/models/diff.py
"""Structured deltas between two results (``result.diff()``).

``a.diff(b)`` returns a :class:`ResultDiff`: the signed Δφ, whether the MIP
genuinely changed (not a co-optimal tie-reshuffle), the distinctions /
relations / account-links gained, lost, or changed, and — composing
:meth:`pyphi.conf.snapshot.ConfigSnapshot.diff` — which config differences
could explain the change. Pairs with :mod:`pyphi.models.explanation` (B8):
``explain`` says why one result is what it is; ``diff`` says what changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import Table


@dataclass(frozen=True)
class Change:
    """One element-level delta between two results.

    ``kind`` is a stable machine key (``"distinction_gained"`` /
    ``"distinction_lost"`` / ``"distinction_changed"`` / ``"relation_gained"`` /
    ``"relation_lost"`` / ``"link_gained"`` / ``"link_lost"`` /
    ``"link_changed"``); ``key`` identifies the element (mechanism, relata, or
    link); ``a_value`` / ``b_value`` are the per-side quantities (``None`` on
    the side where the element is absent).
    """

    kind: str
    key: Any
    a_value: Any = None
    b_value: Any = None
    tone: str | None = None


@dataclass(frozen=True)
class ResultDiff(Displayable):
    """A typed delta from result ``a`` to result ``b`` (``a.diff(b)``)."""

    subject: str
    level: str
    delta_phi: Any
    mip_changed: bool
    binding_direction_changed: bool | None = None
    changes: tuple[Change, ...] = ()
    config_diff: dict[str, tuple[Any, Any]] | None = None
    substrate_note: str | None = None

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        summary = [
            Row("Δφ", self.delta_phi),
            Row("MIP changed", self.mip_changed),
        ]
        if self.binding_direction_changed is not None:
            summary.append(Row("Binding direction changed", self.binding_direction_changed))
        if self.substrate_note:
            summary.append(Row("Note", self.substrate_note))
        sections = [Section(label="Summary", rows=tuple(summary))]
        if self.changes:
            sections.append(
                Section(
                    label="Changes",
                    body=(
                        Table(
                            headers=("Change", "Key", "a", "b"),
                            rows=tuple(
                                (c.kind, c.key, c.a_value, c.b_value)
                                for c in self.changes
                            ),
                            row_tones=tuple(
                                (c.tone, None, None, None) for c in self.changes
                            ),
                        ),
                    ),
                )
            )
        if self.config_diff:
            sections.append(
                Section(
                    label="Config differences",
                    body=(
                        Table(
                            headers=("Setting", "a", "b"),
                            rows=tuple(
                                (path, a, b)
                                for path, (a, b) in self.config_diff.items()
                            ),
                        ),
                    ),
                )
            )
        return Description(
            title=self.subject, sections=tuple(sections), compact=self.subject
        )

    def to_pandas(self):
        """A tidy ``(category, key, a, b)`` frame of every delta."""
        from pyphi.models.pandas import records_to_frame

        rows: list[dict[str, Any]] = [
            {"category": "delta_phi", "key": None, "a": None, "b": self.delta_phi},
            {"category": "mip_changed", "key": None, "a": None, "b": self.mip_changed},
        ]
        rows.extend(
            {"category": c.kind, "key": c.key, "a": c.a_value, "b": c.b_value}
            for c in self.changes
        )
        if self.config_diff:
            rows.extend(
                {"category": "config", "key": path, "a": a, "b": b}
                for path, (a, b) in self.config_diff.items()
            )
        return records_to_frame(rows, columns=["category", "key", "a", "b"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_result_diff.py -q`
Expected: PASS.

- [ ] **Step 5: Re-export + commit**

Add to `pyphi/models/__init__.py` (alphabetically, in the `from .X import` block and `__all__`):

```python
from .diff import Change
from .diff import ResultDiff
```

Run: `uv run python -c "from pyphi.models import Change, ResultDiff; print('ok')"`

```bash
git add pyphi/models/diff.py pyphi/models/__init__.py test/test_result_diff.py
git commit -m "Add Change + ResultDiff typed delta model (B15)"
```

---

## Task 2: `_diff_common` shared helper (Δφ, MIP-reshuffle, config, substrate note)

**Files:**
- Modify: `pyphi/models/diff.py`
- Test: `test/test_result_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_result_diff.py
import pyphi
from pyphi.conf import presets


def test_mip_reshuffle_not_flagged_but_real_change_is(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import _diff_common

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    # Same analysis recomputed: identical phi, MIP is a co-optimal member of a.ties.
    b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    common = _diff_common(a, b)
    assert common["mip_changed"] is False  # identical / tie-equivalent MIP
    assert float(common["delta_phi"]) == 0.0
    assert common["config_diff"] == {}


def test_config_diff_surfaces_precision_change(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import _diff_common

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    with pyphi.config.override(precision=6):
        b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    common = _diff_common(a, b)
    assert "numerics.precision" in common["config_diff"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py::test_mip_reshuffle_not_flagged_but_real_change_is -q`
Expected: FAIL — `ImportError: cannot import name '_diff_common'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/models/diff.py` (add `from pyphi import utils` to the imports):

```python
def _phi_of(result: Any) -> Any:
    """The scalar this result is ordered by (φ for IIT, α for AC)."""
    return getattr(result, "alpha", None) if hasattr(result, "alpha") else result.phi


def _mip_changed(a: Any, b: Any) -> bool:
    """True iff b's MIP partition is not one a could co-optimally have chosen.

    Uses ``lex_key`` over a's tie set (which already encodes EQUALITY_TOLERANCE
    from tie resolution). Falls back to a tolerance-aware lex_key inequality for
    results without a ``.ties`` set.
    """
    a_part = getattr(a, "partition", None)
    b_part = getattr(b, "partition", None)
    if a_part is None or b_part is None:
        return a_part is not b_part
    ties = getattr(a, "ties", None)
    if ties:
        a_tie_keys = {t.partition.lex_key() for t in ties if t.partition is not None}
        return b_part.lex_key() not in a_tie_keys
    # Fallback: a real change only if the partition differs AND phi differs.
    if b_part.lex_key() == a_part.lex_key():
        return False
    return not utils.eq(float(_phi_of(a)), float(_phi_of(b)))


def _config_diff(a: Any, b: Any) -> dict[str, tuple[Any, Any]]:
    a_cfg = getattr(a, "config", None)
    b_cfg = getattr(b, "config", None)
    if a_cfg is None or b_cfg is None:
        return {}
    return a_cfg.diff(b_cfg)


def _substrate_note(a: Any, b: Any) -> str | None:
    a_idx = getattr(a, "node_indices", None)
    b_idx = getattr(b, "node_indices", None)
    if a_idx is not None and b_idx is not None and a_idx != b_idx:
        return f"substrates differ ({a_idx} vs {b_idx}); deltas keyed by mechanism"
    return None


def _diff_common(a: Any, b: Any) -> dict[str, Any]:
    """Shared scalar deltas every result type's diff() reuses."""
    from pyphi.data_structures import PyPhiFloat

    delta = PyPhiFloat(float(_phi_of(b)) - float(_phi_of(a)))
    return {
        "delta_phi": delta,
        "mip_changed": _mip_changed(a, b),
        "config_diff": _config_diff(a, b),
        "substrate_note": _substrate_note(a, b),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_result_diff.py -q`
Expected: PASS. (If `test_mip_reshuffle...` shows `mip_changed is True` because the
two recomputations picked the *same* partition rather than tied alternates, the
assertion still holds since identical lex_key ⇒ in `a.ties` ⇒ `False`.)

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/diff.py test/test_result_diff.py
git commit -m "Add _diff_common: delta-phi, MIP-reshuffle test, config diff (B15)"
```

---

## Task 3: `.diff()` on the IIT 4.0 and IIT 3.0 SIAs

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (4.0 SIA)
- Modify: `pyphi/models/sia.py` (3.0 SIA)
- Test: `test/test_result_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_result_diff.py
def test_iit4_sia_diff(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.diff import ResultDiff

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    b = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    assert float(rd.delta_phi) == 0.0
    assert rd.mip_changed is False


def test_diff_type_mismatch_raises(s):
    from pyphi.formalism import FORMALISM_REGISTRY

    a = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    import pytest

    with pytest.raises(TypeError):
        a.diff("not a result")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py::test_iit4_sia_diff -q`
Expected: FAIL — `AttributeError: 'SystemIrreducibilityAnalysis' object has no attribute 'diff'`.

- [ ] **Step 3: Implement `diff()` on the 4.0 SIA**

In `pyphi/formalism/iit4/__init__.py`, import the diff types
(`from pyphi.models.diff import Change, ResultDiff, _diff_common`) and add to
`SystemIrreducibilityAnalysis`:

```python
    def _binding_direction_changed(self, other) -> bool | None:
        if None in (self.cause, self.effect, other.cause, other.effect):
            return None
        a_dir = "cause" if float(self.cause.phi) <= float(self.effect.phi) else "effect"
        b_dir = "cause" if float(other.cause.phi) <= float(other.effect.phi) else "effect"
        return a_dir != b_dir

    def diff(self, other) -> ResultDiff:
        """Structured delta from this SIA to ``other`` (``a.diff(b)``)."""
        if not isinstance(other, SystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"ΔΦ_s = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            binding_direction_changed=self._binding_direction_changed(other),
            changes=(),
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )
```

- [ ] **Step 4: Implement `diff()` on the 3.0 SIA**

In `pyphi/models/sia.py`, import the diff types
(`from .diff import ResultDiff` and `from .diff import _diff_common`) and add to
`IIT3SystemIrreducibilityAnalysis`:

```python
    def diff(self, other) -> ResultDiff:
        from .diff import ResultDiff, _diff_common

        if not isinstance(other, IIT3SystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"ΔΦ = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )
```

- [ ] **Step 5: Run tests + goldens (no value change)**

Run: `uv run pytest test/test_result_diff.py test/test_golden_regression.py -q`
Expected: PASS, no φ drift.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py pyphi/models/sia.py test/test_result_diff.py
git commit -m "Add .diff() to the IIT 4.0 and 3.0 SIAs (B15)"
```

---

## Task 4: `.diff()` on `CauseEffectStructure` (distinctions + relations deltas)

**Files:**
- Modify: `pyphi/models/ces.py`
- Test: `test/test_result_diff.py`

**Pre-step — confirm the real accessors** (do not guess): run
`uv run python -c "import test.example_substrates as e; from pyphi import config; from pyphi.conf import presets; from pyphi.formalism import iit3; ..."` or read `pyphi/models/distinctions.py` to confirm: a `Distinctions` is iterable yielding `Distinction`s; each `Distinction` has `.mechanism`, `.phi`, `.cause.purview`, `.effect.purview`; `Relations` set difference works on the relation objects (they are hashable `frozenset` subclasses) and each relation exposes `.mechanisms()`. Use the confirmed names in the code below.

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_result_diff.py
def test_ces_diff_distinctions_and_relations(s):
    from pyphi.formalism import iit3
    from pyphi.models.diff import ResultDiff

    with pyphi.config.override(**presets.iit3):
        a = iit3.ces(s)
        b = iit3.ces(s)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    # identical CESs: no element changes
    assert rd.changes == ()
    assert float(rd.delta_phi) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py::test_ces_diff_distinctions_and_relations -q`
Expected: FAIL — `AttributeError: ... has no attribute 'diff'`.

- [ ] **Step 3: Implement `diff()` on `CauseEffectStructure`**

In `pyphi/models/ces.py`, import the diff types
(`from .diff import Change, ResultDiff, _diff_common`) and add:

```python
    def _changes(self, other) -> tuple[Change, ...]:
        from pyphi import utils

        changes: list[Change] = []
        a_by_mech = {d.mechanism: d for d in self.distinctions}
        b_by_mech = {d.mechanism: d for d in other.distinctions}
        for mech in a_by_mech.keys() - b_by_mech.keys():
            changes.append(
                Change("distinction_lost", mech, a_value=a_by_mech[mech].phi)
            )
        for mech in b_by_mech.keys() - a_by_mech.keys():
            changes.append(
                Change("distinction_gained", mech, b_value=b_by_mech[mech].phi)
            )
        for mech in a_by_mech.keys() & b_by_mech.keys():
            da, db = a_by_mech[mech], b_by_mech[mech]
            changed = (
                not utils.eq(float(da.phi), float(db.phi))
                or da.cause.purview != db.cause.purview
                or da.effect.purview != db.effect.purview
            )
            if changed:
                changes.append(
                    Change("distinction_changed", mech, a_value=da.phi, b_value=db.phi)
                )
        a_rels = set(self.relations) if hasattr(self.relations, "__iter__") else set()
        b_rels = set(other.relations) if hasattr(other.relations, "__iter__") else set()
        for r in a_rels - b_rels:
            changes.append(Change("relation_lost", tuple(r.mechanisms()), a_value=r.phi))
        for r in b_rels - a_rels:
            changes.append(Change("relation_gained", tuple(r.mechanisms()), b_value=r.phi))
        return tuple(changes)

    def diff(self, other) -> ResultDiff:
        if not isinstance(other, CauseEffectStructure):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self.sia, other.sia)
        return ResultDiff(
            subject=f"ΔΦ = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=self._changes(other),
            config_diff=(self.config.diff(other.config) if self.config and other.config else {}),
            substrate_note=common["substrate_note"],
        )
```

(`format_value` is already imported in `ces.py`; confirm and add the import if not.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_result_diff.py::test_ces_diff_distinctions_and_relations -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/ces.py test/test_result_diff.py
git commit -m "Add .diff() to CauseEffectStructure (B15)"
```

---

## Task 5: `.diff()` on RIA / MICE / Distinction

**Files:**
- Modify: `pyphi/models/ria.py`, `pyphi/models/mice.py`, `pyphi/models/distinction.py`
- Test: `test/test_result_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_result_diff.py
def test_mechanism_diff(s):
    from pyphi.formalism import iit3
    from pyphi.models.diff import ResultDiff

    with pyphi.config.override(**presets.iit3):
        da = iit3.concept(s, (1,))
        db = iit3.concept(s, (1,))
    rd = da.diff(db)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "mechanism"
    assert float(rd.delta_phi) == 0.0
    # MICE delegates
    assert isinstance(da.cause.diff(db.cause), ResultDiff)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py::test_mechanism_diff -q`
Expected: FAIL — `AttributeError: ... has no attribute 'diff'`.

- [ ] **Step 3: Implement RIA `.diff()`**

In `pyphi/models/ria.py` (it imports `format_value`, `concise_partition`; add
`from .diff import Change, ResultDiff, _diff_common`):

```python
    def diff(self, other) -> ResultDiff:
        if not isinstance(other, RepertoireIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        changes = []
        if self.purview != other.purview:
            changes.append(
                Change("purview_changed", self.mechanism, self.purview, other.purview)
            )
        return ResultDiff(
            subject=f"Δφ = {format_value(common['delta_phi'])}",
            level="mechanism",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=tuple(changes),
            config_diff=common["config_diff"],  # {} — RIA carries no ConfigSnapshot
            substrate_note=common["substrate_note"],
        )
```

(`_diff_common` uses `getattr(a, "ties", None)`; RIA has no `.ties`, so the
fallback path runs — correct. `_config_diff` returns `{}` since RIA has no
`.config`.)

- [ ] **Step 4: Implement MICE + Distinction `.diff()`**

In `pyphi/models/mice.py`, add to `MaximallyIrreducibleCauseOrEffect`:

```python
    def diff(self, other):
        other_ria = other.ria if hasattr(other, "ria") else other
        return self.ria.diff(other_ria)
```

In `pyphi/models/distinction.py` (add `from .diff import ResultDiff, _diff_common`):

```python
    def diff(self, other) -> ResultDiff:
        if not isinstance(other, Distinction):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"Δφ = {format_value(common['delta_phi'])}",
            level="mechanism",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=(),
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )
```

(`Distinction` has no `.partition`/`.ties`; `_mip_changed` returns
`a_part is not b_part` → both `None` → `False`. `format_value` is imported in
`distinction.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test/test_result_diff.py::test_mechanism_diff -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/ria.py pyphi/models/mice.py pyphi/models/distinction.py test/test_result_diff.py
git commit -m "Add .diff() to RIA, MICE, and Distinction (B15)"
```

---

## Task 6: `.diff()` on the actual-causation results

**Files:**
- Modify: `pyphi/models/actual_causation.py`
- Test: `test/test_result_diff.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_result_diff.py
def test_ac_diff():
    from pyphi import actual, examples
    from pyphi.direction import Direction
    from pyphi.models.diff import ResultDiff

    t = examples.prevention_transition()
    a = actual.sia(t, Direction.BIDIRECTIONAL)
    b = actual.sia(t, Direction.BIDIRECTIONAL)
    rd = a.diff(b)
    assert isinstance(rd, ResultDiff)
    assert rd.level == "system"
    assert float(rd.delta_phi) == 0.0

    acc_a = actual.account(t, Direction.BIDIRECTIONAL)
    acc_b = actual.account(t, Direction.BIDIRECTIONAL)
    rd_acc = acc_a.diff(acc_b)
    assert isinstance(rd_acc, ResultDiff)
    assert rd_acc.changes == ()  # identical accounts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_result_diff.py::test_ac_diff -q`
Expected: FAIL — `AttributeError: ... has no attribute 'diff'`.

- [ ] **Step 3: Implement AcSIA + Account `.diff()`**

In `pyphi/models/actual_causation.py` (imports `format_value`, `concise_partition`;
add `from .diff import Change, ResultDiff, _diff_common`), add to
`AcSystemIrreducibilityAnalysis`:

```python
    def diff(self, other) -> ResultDiff:
        if not isinstance(other, AcSystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"Δα = {format_value(common['delta_phi'])}",  # noqa: RUF001
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )
```

and to `Account`:

```python
    def diff(self, other) -> ResultDiff:
        from pyphi import utils

        if not isinstance(other, Account):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )

        def key(link):
            return (str(link.direction), link.mechanism, link.purview)

        a_by = {key(link): link for link in self.causal_links}
        b_by = {key(link): link for link in other.causal_links}
        changes: list[Change] = []
        for k in a_by.keys() - b_by.keys():
            changes.append(Change("link_lost", k, a_value=a_by[k].alpha))
        for k in b_by.keys() - a_by.keys():
            changes.append(Change("link_gained", k, b_value=b_by[k].alpha))
        for k in a_by.keys() & b_by.keys():
            if not utils.eq(a_by[k].alpha, b_by[k].alpha):
                changes.append(
                    Change("link_changed", k, a_by[k].alpha, b_by[k].alpha)
                )
        delta = self._sum_alpha and other._sum_alpha
        from pyphi.data_structures import PyPhiFloat

        return ResultDiff(
            subject=f"ΔΣα ({len(self)} → {len(other)} links)",  # noqa: RUF001
            level="system",
            delta_phi=PyPhiFloat(float(other._sum_alpha) - float(self._sum_alpha)),
            mip_changed=False,
            changes=tuple(changes),
            config_diff={},
            substrate_note=None,
        )
```

(`Account` has no `.partition`/`.config`, so `mip_changed=False` and
`config_diff={}` are set directly; `delta` line is vestigial — drop it. The
`# noqa: RUF001` silences the ambiguous-α lint, matching existing AC code.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_result_diff.py::test_ac_diff -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/actual_causation.py test/test_result_diff.py
git commit -m "Add .diff() to actual-causation results (B15)"
```

---

## Task 7: Coverage invariant + full verification + changelog + ROADMAP

**Files:**
- Modify: `test/test_result_diff.py`
- Create: `changelog.d/b15-result-diff.feature.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write the coverage-invariant test**

```python
# append to test/test_result_diff.py
def test_diff_is_total(s):
    """Every top-level result type diffs against another of its kind into a
    valid, renderable, exportable ResultDiff (the B15 coverage invariant)."""
    from pyphi import actual, examples
    from pyphi.direction import Direction
    from pyphi.formalism import FORMALISM_REGISTRY, iit3
    from pyphi.models.diff import ResultDiff

    pairs = []
    pairs.append(
        (
            FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s),
            FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s),
        )
    )
    with pyphi.config.override(**presets.iit3):
        pairs.append((iit3.sia(s), iit3.sia(s)))
        pairs.append((iit3.ces(s), iit3.ces(s)))
        da, db = iit3.concept(s, (1,)), iit3.concept(s, (1,))
    pairs.append((da, db))  # Distinction
    pairs.append((da.cause, db.cause))  # MICE
    pairs.append((da.cause.ria, db.cause.ria))  # RIA
    t = examples.prevention_transition()
    pairs.append((actual.sia(t, Direction.BIDIRECTIONAL), actual.sia(t, Direction.BIDIRECTIONAL)))
    pairs.append((actual.account(t, Direction.BIDIRECTIONAL), actual.account(t, Direction.BIDIRECTIONAL)))

    for a, b in pairs:
        rd = a.diff(b)
        name = type(a).__name__
        assert isinstance(rd, ResultDiff), name
        assert rd.level in {"system", "mechanism"}, name
        assert repr(rd)
        rd.to_pandas()
```

- [ ] **Step 2: Run the targeted suite**

Run: `uv run pytest test/test_result_diff.py -q`
Expected: PASS.

- [ ] **Step 3: Full suite incl. doctest sweep (no path argument!)**

Run: `uv run pytest`
Expected: PASS. Collects the `pyphi/` doctests (ResultDiff adds public repr/HTML
surface). Investigate any φ/α change as a bug (`.diff()` must be pure).

- [ ] **Step 4: Type check**

Run: `uv run pyright pyphi/models/diff.py`
Expected: no new errors (trust the pre-commit hook if the local pyright workaround is active).

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Added `result.diff(other)` — a typed, displayable `ResultDiff` (Δφ, real-vs-tie-reshuffle MIP change, distinctions/relations/links gained-lost-changed, and config-diff attribution via `ConfigSnapshot.diff`) — across IIT 4.0, IIT 3.0, and actual causation. Pairs with `result.explain()`.' > changelog.d/b15-result-diff.feature.md
```

- [ ] **Step 6: Update the ROADMAP dashboard**

In `ROADMAP.md`, change the B15 row status from `⬜ open` to `✅ landed` and rewrite its
one-line to past tense (mirror the B8 landed-row style). Do not touch other rows.

- [ ] **Step 7: Commit**

```bash
git add test/test_result_diff.py changelog.d/b15-result-diff.feature.md ROADMAP.md
git commit -m "Add .diff() coverage invariant; changelog + ROADMAP for B15"
```

---

## Self-Review notes

- **Spec coverage:** `ResultDiff` named fields + `Change` (Task 1) ✓; `_diff_common` + MIP-reshuffle via tie set (Task 2) ✓; `.diff()` on all types (Tasks 3–6) ✓; distinctions/relations gained-lost-changed (Task 4) ✓; config-diff composition (Task 2/4) ✓; comparability `TypeError` + substrate note (Tasks 2–3) ✓; Displayable + to_pandas (Task 1) ✓; coverage invariant + doctest sweep (Task 7) ✓.
- **Type consistency:** `Change(kind, key, a_value, b_value, tone)`, `ResultDiff(subject, level, delta_phi, mip_changed, binding_direction_changed, changes, config_diff, substrate_note)`, `_diff_common(a, b) -> dict`, `_mip_changed`/`_config_diff`/`_substrate_note`/`_phi_of` — consistent across tasks.
- **Confirmations the implementer must resolve inline (marked):** Task 4's exact `Distinction`/`Relations` accessors (`.mechanism`, `.cause.purview`, relation set-difference + `.mechanisms()`) — confirm with a probe before writing, as in B8. Drop the vestigial `delta` line in Task 6 Step 3.
- **Deviation from B8:** mechanism-level results (RIA/MICE/Distinction) and `Account` carry no `ConfigSnapshot`, so their `config_diff` is always `{}` — documented inline, not a bug.
