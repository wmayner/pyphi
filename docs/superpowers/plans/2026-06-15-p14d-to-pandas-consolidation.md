# to_pandas Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heuristic `json_normalize`-based `to_pandas()` with one labeled-export convention applied to every `ToPandasMixin` user, reconciling the provisional `TriggeredTPM` surface.

**Architecture:** A thin `ToPandasMixin.to_pandas()` delegates to a per-class `_to_pandas()`; scalar-record types implement `_pandas_record()` and inherit a Series-building default; collections and distributions override `_to_pandas()`. Pure helper functions in `models/pandas.py` own all pandas construction and compose the existing `NodeLabels` label utilities. Shapes are determined by object category (scalar-record → `Series`; collection → key-indexed `DataFrame`; 1-D distribution → tidy long `DataFrame`; 2-D conditional → wide matrix).

**Tech Stack:** Python 3.12+, pandas, numpy, pytest. Run everything with `uv run`.

**Design doc:** `docs/superpowers/specs/2026-06-15-p14d-to-pandas-consolidation-design.md`

**Key facts established during design (do not re-derive):**
- `pyphi.utils.all_states(spec)` — `spec` is `int` (binary, n nodes) **or** a sequence of per-node alphabet sizes (k-ary). Little-endian by default (index 0 fastest).
- `pyphi.distribution.flatten(repertoire)` squeezes singleton dims and ravels little-endian (`order="F"`) — its order matches `all_states(purview_alphabet)`.
- A full repertoire array is shaped over all N nodes (purview dims = cardinality, others = 1), so per-purview-unit cardinality is `[repertoire.shape[i] for i in purview]`.
- `NodeLabels(labels, node_indices)`; `node_labels.coerce_to_labels(indices)` returns label strings, falling back to integer indices when labels are absent.
- `str(Direction.CAUSE) == "CAUSE"`, `str(Direction.EFFECT) == "EFFECT"`.
- `StateSpecification` / `SystemStateSpecification` carry **no** `node_labels` → their `purview` column is the integer-index fallback (documented, not a bug).
- `CauseEffectStructure` is **not** a `Distinctions`; the mixin user is `system.ces().distinctions` (a `Distinctions` subclass). Individual objects: `d = next(iter(ces.distinctions))` (Distinction), `d.cause` (MICE), `d.cause.ria` (RIA).
- No internal code calls `.to_pandas()`; only `test/test_triggered_tpm.py` does (kept byte-identical).

---

## File Structure

- **Modify** `pyphi/models/pandas.py` — rewrite `ToPandasMixin`; add helpers `record_to_series`, `records_to_frame`, `state_multiindex`, `distribution_rows`; add `_DISTRIBUTION_COLUMNS`. Remove the `json_normalize` heuristic and the now-unused `Sequence` import.
- **Modify** `pyphi/models/ria.py` — add `RepertoireIrreducibilityAnalysis._pandas_record()`.
- **Modify** `pyphi/models/mice.py` — add `MaximallyIrreducibleCauseOrEffect._pandas_record()` (delegates to ria).
- **Modify** `pyphi/models/distinction.py` — add `Distinction._pandas_record()`.
- **Modify** `pyphi/models/distinctions.py` — add `Distinctions._to_pandas()` + `_DISTINCTION_COLUMNS`.
- **Modify** `pyphi/models/state_specification.py` — add `StateSpecification._to_pandas()` and `SystemStateSpecification._to_pandas()`; add `import pandas as pd` if absent.
- **Modify** `pyphi/matching/triggered_tpm.py` — rebuild `to_pandas()` index/columns via `state_multiindex`; drop the provisional note.
- **Create** `test/test_to_pandas.py` — helper tests, per-type tests, shared contract, k-ary, pair, TriggeredTPM guard.
- **Create** `changelog.d/to-pandas-consolidation.change.md`.

---

### Task 1: Helper layer + rewritten `ToPandasMixin`

**Files:**
- Modify: `pyphi/models/pandas.py`
- Test: `test/test_to_pandas.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_to_pandas.py`:

```python
"""Tests for the unified to_pandas labeled-export convention."""

import numpy as np
import pandas as pd
import pytest

from pyphi import examples
from pyphi.direction import Direction
from pyphi.labels import NodeLabels
from pyphi.models.pandas import (
    distribution_rows,
    record_to_series,
    records_to_frame,
    state_multiindex,
)
from pyphi.utils import all_states


def test_state_multiindex_binary():
    labels = NodeLabels(("A", "B"), (0, 1))
    mi = state_multiindex(labels, (0, 1))
    assert list(mi.names) == ["A", "B"]
    assert list(mi) == list(all_states(2))


def test_state_multiindex_kary():
    labels = NodeLabels(("A", "B"), (0, 1))
    mi = state_multiindex(labels, (0, 1), alphabet=(3, 2))
    assert list(mi) == list(all_states((3, 2)))


def test_distribution_rows_binary():
    # shape (2, 2) over purview (0, 1); flatten is little-endian (order="F")
    rep = np.array([[0.1, 0.4], [0.1, 0.4]])
    rows = distribution_rows(Direction.CAUSE, "repertoire", (0, 1), rep, None)
    assert [r["state"] for r in rows] == list(all_states((2, 2)))
    assert rows[0] == {
        "direction": "CAUSE",
        "kind": "repertoire",
        "purview": (0, 1),
        "state": (0, 0),
        "probability": 0.1,
    }
    assert sum(r["probability"] for r in rows) == pytest.approx(1.0)


def test_distribution_rows_kary_states():
    rep = np.arange(6, dtype=float).reshape(3, 2)
    rep = rep / rep.sum()
    rows = distribution_rows(Direction.EFFECT, "repertoire", (0, 1), rep, None)
    assert [r["state"] for r in rows] == list(all_states((3, 2)))
    assert all(r["direction"] == "EFFECT" for r in rows)


def test_distribution_rows_none_is_empty():
    assert distribution_rows(Direction.CAUSE, "repertoire", (0,), None, None) == []


def test_record_to_series_preserves_order():
    series = record_to_series({"phi": 0.5, "mechanism": ("A",)}, name="X")
    assert list(series.index) == ["phi", "mechanism"]
    assert series.name == "X"


def test_records_to_frame_empty_has_columns():
    frame = records_to_frame([], index="mechanism", columns=["mechanism", "phi"])
    assert frame.index.name == "mechanism"
    assert list(frame.columns) == ["phi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_to_pandas.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'distribution_rows' from 'pyphi.models.pandas'`.

- [ ] **Step 3: Rewrite `pyphi/models/pandas.py`**

Replace the file's imports and the entire `ToPandasMixin` class. Keep `try_to_dict`, `ToDictFromExplicitAttrsMixin`, `ToDictMixin` exactly as they are. The new top-of-file imports and helpers:

```python
# models/pandas.py
"""Utilities for working with Pandas data structures."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
```

(Delete the old `from collections.abc import Sequence` line.) Keep `try_to_dict`, `ToDictFromExplicitAttrsMixin`, and `ToDictMixin` unchanged. Then replace the old `ToPandasMixin` with:

```python
_DISTRIBUTION_COLUMNS = ["direction", "kind", "purview", "state", "probability"]


def record_to_series(record: Mapping[str, Any], name: str | None = None) -> pd.Series:
    """Build a Series from an ordered field-to-value mapping."""
    return pd.Series(dict(record), name=name)


def records_to_frame(
    rows: Iterable[Mapping[str, Any]],
    index: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Stack record mappings into a DataFrame, optionally moving one column to
    the index. ``columns`` fixes the column set so an empty ``rows`` still
    produces the right schema."""
    frame = pd.DataFrame(list(rows), columns=columns)
    if index is not None:
        frame = frame.set_index(index)
    return frame


def state_multiindex(node_labels, indices, alphabet=None) -> pd.MultiIndex:
    """A MultiIndex over all states of ``indices``, level-named by label.

    ``alphabet`` is the per-unit cardinality sequence (k-ary); if ``None`` the
    units are binary.
    """
    from pyphi.utils import all_states

    spec = alphabet if alphabet is not None else len(indices)
    states = list(all_states(spec))
    names = list(node_labels.coerce_to_labels(indices))
    return pd.MultiIndex.from_tuples(states, names=names)


def distribution_rows(
    direction, kind, purview, repertoire, node_labels=None
) -> list[dict[str, Any]]:
    """Tidy ``{direction, kind, purview, state, probability}`` rows for one
    repertoire.

    States are enumerated from the repertoire's per-purview-unit cardinality
    (k-ary aware). ``purview`` renders as labels when ``node_labels`` is given,
    else as integer indices. Returns ``[]`` for a ``None`` repertoire.
    """
    from pyphi import distribution
    from pyphi.utils import all_states

    if repertoire is None:
        return []
    repertoire = np.asarray(repertoire)
    alphabet = [repertoire.shape[i] for i in purview]
    flat = distribution.flatten(repertoire)
    states = list(all_states(alphabet)) if alphabet else [()]
    if node_labels is None:
        purview_labels: tuple[Any, ...] = tuple(purview)
    else:
        purview_labels = tuple(node_labels.coerce_to_labels(purview))
    direction_label = str(direction)
    return [
        {
            "direction": direction_label,
            "kind": kind,
            "purview": purview_labels,
            "state": tuple(state),
            "probability": float(prob),
        }
        for state, prob in zip(states, flat, strict=True)
    ]


class ToPandasMixin:
    """Export a result object to a labeled Pandas structure.

    ``to_pandas()`` returns a ``Series`` for scalar-record types and a
    ``DataFrame`` with a labeled index for collections and distributions.
    Units render as labels. Subclasses implement ``_pandas_record()`` (record
    types, which inherit the Series-building ``_to_pandas``) or override
    ``_to_pandas()`` (collections and distributions).
    """

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Return a labeled Pandas view of this object."""
        return self._to_pandas()

    def _to_pandas(self) -> pd.Series | pd.DataFrame:
        return record_to_series(self._pandas_record(), name=type(self).__name__)

    def _pandas_record(self) -> Mapping[str, Any]:
        raise NotImplementedError(type(self).__name__)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_to_pandas.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/pandas.py test/test_to_pandas.py
git commit -m "Add labeled to_pandas helper layer and rewrite ToPandasMixin"
```

---

### Task 2: Scalar-record types (RIA, MICE, Distinction)

**Files:**
- Modify: `pyphi/models/ria.py`
- Modify: `pyphi/models/mice.py`
- Modify: `pyphi/models/distinction.py`
- Test: `test/test_to_pandas.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_to_pandas.py`:

```python
@pytest.fixture(scope="module")
def basic_ces():
    return examples.basic_system().ces()


def test_distinction_to_pandas_is_labeled_series(basic_ces):
    distinction = next(iter(basic_ces.distinctions))
    series = distinction.to_pandas()
    assert isinstance(series, pd.Series)
    assert {
        "phi",
        "mechanism",
        "mechanism_state",
        "cause_purview",
        "effect_purview",
    } <= set(series.index)
    # units render as label strings, never raw ints
    assert all(isinstance(label, str) for label in series["mechanism"])


def test_mice_to_pandas_is_labeled_series(basic_ces):
    mice = next(iter(basic_ces.distinctions)).cause
    series = mice.to_pandas()
    assert isinstance(series, pd.Series)
    assert all(isinstance(label, str) for label in series["mechanism"])
    assert str(series["direction"]) == "CAUSE"


def test_ria_to_pandas_is_labeled_series(basic_ces):
    ria = next(iter(basic_ces.distinctions)).cause.ria
    series = ria.to_pandas()
    assert isinstance(series, pd.Series)
    assert all(isinstance(label, str) for label in series["purview"])
    assert series["direction"] in ("CAUSE", "EFFECT")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_to_pandas.py -k "labeled_series" -q`
Expected: FAIL — `NotImplementedError: RepertoireIrreducibilityAnalysis` (the base `_pandas_record`).

- [ ] **Step 3a: Add `_pandas_record` to `pyphi/models/ria.py`**

Add this method to `RepertoireIrreducibilityAnalysis` (near `to_json`):

```python
    def _pandas_record(self):
        labels = self.node_labels
        return {
            "phi": float(self.phi),
            "direction": str(self.direction),
            "mechanism": tuple(labels.coerce_to_labels(self.mechanism)),
            "purview": tuple(labels.coerce_to_labels(self.purview)),
            "mechanism_state": (
                None
                if self.mechanism_state is None
                else tuple(self.mechanism_state)
            ),
            "purview_state": (
                None if self.purview_state is None else tuple(self.purview_state)
            ),
            "specified_state": self.specified_state,
        }
```

- [ ] **Step 3b: Add `_pandas_record` to `pyphi/models/mice.py`**

Add this method to `MaximallyIrreducibleCauseOrEffect`:

```python
    def _pandas_record(self):
        return self.ria._pandas_record()
```

- [ ] **Step 3c: Add `_pandas_record` to `pyphi/models/distinction.py`**

Add this method to `Distinction`:

```python
    def _pandas_record(self):
        labels = self.node_labels

        def labelled(nodes):
            return None if nodes is None else tuple(labels.coerce_to_labels(nodes))

        return {
            "phi": float(self.phi),
            "mechanism": labelled(self.mechanism),
            "mechanism_state": (
                None
                if self.mechanism_state is None
                else tuple(self.mechanism_state)
            ),
            "cause_purview": labelled(self.cause_purview),
            "effect_purview": labelled(self.effect_purview),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_to_pandas.py -k "labeled_series" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/ria.py pyphi/models/mice.py pyphi/models/distinction.py test/test_to_pandas.py
git commit -m "Implement labeled Series export for RIA, MICE, and Distinction"
```

---

### Task 3: Collection type (Distinctions)

**Files:**
- Modify: `pyphi/models/distinctions.py`
- Test: `test/test_to_pandas.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_to_pandas.py`:

```python
def test_distinctions_to_pandas_dataframe(basic_ces):
    distinctions = basic_ces.distinctions
    frame = distinctions.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert frame.index.name == "mechanism"
    assert list(frame.columns) == [
        "phi",
        "mechanism_state",
        "cause_purview",
        "effect_purview",
    ]
    assert len(frame) == len(distinctions)
    # the index holds labeled mechanisms (tuples of label strings)
    first_mechanism = frame.index[0]
    assert all(isinstance(label, str) for label in first_mechanism)


def test_empty_distinctions_to_pandas_has_schema():
    from pyphi.models.distinctions import ResolvedDistinctions

    frame = ResolvedDistinctions([]).to_pandas()
    assert frame.index.name == "mechanism"
    assert list(frame.columns) == [
        "phi",
        "mechanism_state",
        "cause_purview",
        "effect_purview",
    ]
    assert len(frame) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_to_pandas.py -k "distinctions_to_pandas or empty_distinctions" -q`
Expected: FAIL — `NotImplementedError: ResolvedDistinctions` (base `_pandas_record` via inherited `_to_pandas`).

- [ ] **Step 3: Add `_to_pandas` to `pyphi/models/distinctions.py`**

Near the top of the module (after imports) add the column constant:

```python
_DISTINCTION_COLUMNS = [
    "phi",
    "mechanism",
    "mechanism_state",
    "cause_purview",
    "effect_purview",
]
```

Add this method to the `Distinctions` class:

```python
    def _to_pandas(self):
        from .pandas import records_to_frame

        rows = [concept._pandas_record() for concept in self.concepts]
        return records_to_frame(
            rows, index="mechanism", columns=_DISTINCTION_COLUMNS
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_to_pandas.py -k "distinctions_to_pandas or empty_distinctions" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/distinctions.py test/test_to_pandas.py
git commit -m "Implement labeled DataFrame export for Distinctions"
```

---

### Task 4: Distribution types (StateSpecification, SystemStateSpecification)

**Files:**
- Modify: `pyphi/models/state_specification.py`
- Test: `test/test_to_pandas.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_to_pandas.py`:

```python
def _make_state_spec(direction, purview):
    # full-shape repertoire over 2 binary nodes; purview drives the cardinality
    repertoire = np.array([[0.1, 0.4], [0.2, 0.3]])
    unconstrained = np.full((2, 2), 0.25)
    from pyphi.models.state_specification import StateSpecification

    return StateSpecification(
        direction=direction,
        purview=purview,
        state=(0, 0),
        intrinsic_information=0.5,
        repertoire=repertoire,
        unconstrained_repertoire=unconstrained,
    )


def test_state_specification_to_pandas_is_tidy():
    spec = _make_state_spec(Direction.CAUSE, (0, 1))
    frame = spec.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "direction",
        "kind",
        "purview",
        "state",
        "probability",
    ]
    assert set(frame["kind"]) == {"repertoire", "unconstrained"}
    assert set(frame["direction"]) == {"CAUSE"}
    # 2 kinds x 4 states
    assert len(frame) == 8
    repertoire_rows = frame[frame["kind"] == "repertoire"]
    assert repertoire_rows["probability"].sum() == pytest.approx(1.0)


def test_system_state_specification_is_concat():
    from pyphi.models.state_specification import SystemStateSpecification

    cause = _make_state_spec(Direction.CAUSE, (0, 1))
    effect = _make_state_spec(Direction.EFFECT, (0, 1))
    system = SystemStateSpecification(cause=cause, effect=effect)
    frame = system.to_pandas()
    assert list(frame.columns) == [
        "direction",
        "kind",
        "purview",
        "state",
        "probability",
    ]
    assert set(frame["direction"]) == {"CAUSE", "EFFECT"}
    assert len(frame) == len(cause.to_pandas()) + len(effect.to_pandas())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_to_pandas.py -k "state_specification or system_state" -q`
Expected: FAIL — `NotImplementedError: StateSpecification`.

- [ ] **Step 3: Add `_to_pandas` to `pyphi/models/state_specification.py`**

Ensure `import pandas as pd` is present at the top of the module (add it if not). Add this method to `StateSpecification`:

```python
    def _to_pandas(self):
        from .pandas import _DISTRIBUTION_COLUMNS, distribution_rows

        rows = []
        for kind, rep in (
            ("repertoire", self.repertoire),
            ("unconstrained", self.unconstrained_repertoire),
        ):
            rows.extend(
                distribution_rows(
                    self.direction, kind, self.purview, rep, node_labels=None
                )
            )
        return pd.DataFrame(rows, columns=_DISTRIBUTION_COLUMNS)
```

Add this method to `SystemStateSpecification`:

```python
    def _to_pandas(self):
        from .pandas import _DISTRIBUTION_COLUMNS

        frames = [
            spec.to_pandas()
            for spec in (self.cause, self.effect)
            if spec is not None
        ]
        if not frames:
            return pd.DataFrame(columns=_DISTRIBUTION_COLUMNS)
        return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_to_pandas.py -k "state_specification or system_state" -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/state_specification.py test/test_to_pandas.py
git commit -m "Implement tidy long-format export for state specifications"
```

---

### Task 5: Reconcile TriggeredTPM onto the shared helper

**Files:**
- Modify: `pyphi/matching/triggered_tpm.py`
- Test: `test/test_to_pandas.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_to_pandas.py`:

```python
def test_triggered_tpm_to_pandas_byte_identical():
    from pyphi.matching.triggered_tpm import TriggeredTPM

    labels = NodeLabels(("A", "B", "C"), (0, 1, 2))
    sensory = (0,)
    system = (1, 2)
    array = np.random.default_rng(0).random((2, 2, 2))
    ttpm = TriggeredTPM(
        array=array,
        sensory_indices=sensory,
        system_indices=system,
        node_labels=labels,
    )

    result = ttpm.to_pandas()

    # expected via the explicit (pre-consolidation) construction
    sensory_states = list(all_states(len(sensory)))
    system_states = list(all_states(len(system)))
    expected = pd.DataFrame(
        [[array[x + s] for s in system_states] for x in sensory_states],
        index=pd.MultiIndex.from_tuples(sensory_states, names=["A"]),
        columns=pd.MultiIndex.from_tuples(system_states, names=["B", "C"]),
    )

    pd.testing.assert_frame_equal(result, expected)
    assert list(result.index.names) == ["A"]
    assert list(result.columns.names) == ["B", "C"]
```

- [ ] **Step 2: Run test to verify it passes already (regression guard)**

Run: `uv run pytest test/test_to_pandas.py -k "triggered_tpm_to_pandas_byte_identical" -q`
Expected: PASS — the current `TriggeredTPM.to_pandas()` already produces this frame. This test locks the output *before* the refactor so Step 4 proves byte-identity.

- [ ] **Step 3: Replace `to_pandas()` in `pyphi/matching/triggered_tpm.py`**

Replace the existing method (lines ~70-81) with:

```python
    def to_pandas(self) -> pd.DataFrame:
        """Labeled view: rows = stimulus states, columns = system states,
        values = Pr(s | x)."""
        from pyphi.models.pandas import state_multiindex

        index = state_multiindex(self.node_labels, self.sensory_indices)
        columns = state_multiindex(self.node_labels, self.system_indices)
        data = [
            [self.array[tuple(x) + tuple(s)] for s in columns] for x in index
        ]
        return pd.DataFrame(data, index=index, columns=columns)
```

The `utils.all_states` calls and the `convert`/`utils` imports used only here are no longer needed by this method, but `utils` is still used elsewhere in the module — do **not** remove the module-level `from pyphi import utils` import.

- [ ] **Step 4: Run test to verify it still passes (now exercising the new code)**

Run: `uv run pytest test/test_to_pandas.py -k "triggered_tpm" test/test_triggered_tpm.py -q`
Expected: PASS — both the byte-identity guard and the existing TriggeredTPM suite.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/triggered_tpm.py test/test_to_pandas.py
git commit -m "Reconcile TriggeredTPM.to_pandas onto the shared state_multiindex helper"
```

---

### Task 6: Shared contract test, changelog, full verification

**Files:**
- Test: `test/test_to_pandas.py`
- Create: `changelog.d/to-pandas-consolidation.change.md`

- [ ] **Step 1: Write the shared contract test**

Append to `test/test_to_pandas.py`:

```python
def test_label_carrying_types_never_emit_raw_int_units(basic_ces):
    """Every label-carrying export renders units as label strings, and no
    export has dotted-path (json_normalize) columns."""
    distinction = next(iter(basic_ces.distinctions))
    series_exports = [
        distinction.to_pandas(),
        distinction.cause.to_pandas(),
        distinction.cause.ria.to_pandas(),
    ]
    for series in series_exports:
        assert isinstance(series, pd.Series)
        assert not any("." in str(label) for label in series.index)

    frame = basic_ces.distinctions.to_pandas()
    assert not any("." in str(column) for column in frame.columns)
    for mechanism in frame.index:
        assert all(isinstance(label, str) for label in mechanism)


def test_base_to_pandas_raises_for_unimplemented():
    from pyphi.models.pandas import ToPandasMixin

    class Unimplemented(ToPandasMixin):
        pass

    with pytest.raises(NotImplementedError, match="Unimplemented"):
        Unimplemented().to_pandas()
```

- [ ] **Step 2: Run the full new test file**

Run: `uv run pytest test/test_to_pandas.py -q`
Expected: PASS (all tests).

- [ ] **Step 3: Create the changelog fragment**

Create `changelog.d/to-pandas-consolidation.change.md` with exactly:

```markdown
Unified `to_pandas()` across result objects into one labeled-export convention.
Scalar-record results (`RepertoireIrreducibilityAnalysis`, the MICE types,
`Distinction`) return a `Series`; `Distinctions` returns a `DataFrame` indexed
by labeled mechanism; `StateSpecification` and `SystemStateSpecification` return
a tidy long-format `DataFrame` with columns `direction`, `kind`, `purview`,
`state`, `probability`. Units render as labels throughout. This replaces the
previous `json_normalize`-based heuristic, whose dotted-path columns and
`Series`-vs-`DataFrame` guessing are gone.
```

- [ ] **Step 4: Full verification (doctests + touched suites)**

Because `pyphi/` source modules changed, run the **doctest-inclusive** sweep with no path argument, plus the directly affected suites:

Run: `uv run pytest -q` (full suite, includes `--doctest-modules`)
Expected: PASS, with no new failures attributable to these changes. (If a pre-existing unrelated failure appears — e.g. a known cache-pollution test — confirm it also fails on a clean checkout before dismissing.)

If the full suite is too slow for the inner loop, the minimum targeted set is:
`uv run pytest test/test_to_pandas.py test/test_triggered_tpm.py pyphi/models/pandas.py --doctest-modules -q`

- [ ] **Step 5: Commit**

```bash
git add test/test_to_pandas.py changelog.d/to-pandas-consolidation.change.md
git commit -m "Add to_pandas contract tests and changelog fragment"
```

---

## Self-Review

**1. Spec coverage:**
- Architecture (thin mixin + `_to_pandas` hook + pure helpers) → Task 1. ✓
- Scalar-record → Series (RIA/MICE/Distinction) → Task 2. ✓
- Collection → key-indexed DataFrame (Distinctions, mechanism index) → Task 3. ✓
- 1-D distribution tidy + pair concat (StateSpecification/SystemStateSpecification) → Task 4. ✓
- 2-D conditional wide matrix, byte-identical reconciliation (TriggeredTPM) → Task 5. ✓
- Units-as-labels + no dotted-path columns invariant → Task 6 contract test. ✓
- k-ary correctness → Task 1 `test_distribution_rows_kary_states` / `test_state_multiindex_kary`. ✓
- Error handling: `NotImplementedError` (Task 6 `test_base_to_pandas_raises`), empty objects (Task 3 empty Distinctions; Task 4 empty SystemStateSpecification path), integer fallback for label-less specs (Task 4, `node_labels=None`). ✓
- `to_json`/`jsonify` unchanged: no `to_json` methods touched; verified by the full suite in Task 6. ✓
- Migration changelog (no planning markers) → Task 6. ✓

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases" — every code step shows complete code. ✓

**3. Type consistency:** Helper names (`record_to_series`, `records_to_frame`, `state_multiindex`, `distribution_rows`, `_DISTRIBUTION_COLUMNS`) are defined in Task 1 and used identically in Tasks 2-5. `_pandas_record()` (record types) and `_to_pandas()` (overrides) are consistent with the Task 1 mixin. `_DISTINCTION_COLUMNS` defined and used within Task 3. ✓

**Note on scope discipline:** `StateSpecification`/`SystemStateSpecification` deliberately use the integer-index fallback for their `purview` column because they carry no `node_labels`. Adding labels to those data classes is **out of scope** for this plan — do not expand their constructors or serialization.
