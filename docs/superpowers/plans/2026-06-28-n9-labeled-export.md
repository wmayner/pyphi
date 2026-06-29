# N9 — Labeled-export (`to_pandas`) full coverage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every user-facing `Displayable` result type a `to_pandas()` method following the P14d scalar→Series / collection→DataFrame convention, and lock full coverage with an invariant test.

**Architecture:** Each uncovered base class gains `ToPandasMixin` and implements either `_pandas_record()` (scalar→Series) or `_to_pandas()` (collection/structural→DataFrame). Pure pandas construction stays in the `record_to_series` / `records_to_frame` / `state_multiindex` helpers. Subclasses (`Null*`, MICE/relations/account variants) inherit.

**Tech Stack:** Python 3.13+, pandas, the existing `pyphi/models/pandas.py` helpers + `NodeLabels`.

## Global Constraints

- Python 3.13+ only; no backward-compat shims.
- No planning-artifact markers (P-/N-/B-numbers, "Wave", roadmap codenames) in `pyphi/` source, docstrings, or `changelog.d/`.
- Commit trailer on EVERY commit, verbatim:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Never `--no-verify`. If pre-commit ruff reformats and aborts and HEAD didn't advance, re-`git add` the same files and re-commit.
- Stage only the named files per task (never `git add -A`). ROADMAP.md is edited by concurrent instances — before committing it, confirm `git diff ROADMAP.md` is only your hunk.
- Convention: scalar-record types add `ToPandasMixin` + `_pandas_record()` (a field→value mapping); they inherit the Series-building `_to_pandas`. Collection/structural types add `ToPandasMixin` + override `_to_pandas()` to return a `DataFrame`.
- Fast-lane: `uv run pytest test/<file> -q -p no:cacheprovider`. Full verification before done: `uv run --all-extras pytest` (no path argument).
- Reuse `record_to_series(record, name=...)` and `records_to_frame(rows, index=..., columns=...)` from `pyphi/models/pandas.py`; don't hand-build Series/DataFrames.

---

### Task 1: SIA family → Series

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (`SystemIrreducibilityAnalysis`)
- Modify: `pyphi/models/sia.py` (`IIT3SystemIrreducibilityAnalysis`)
- Modify: `pyphi/models/complex.py` (`Complex`, `ExcludedCandidate`)
- Test: `test/models/test_to_pandas_coverage.py` (new)

**Interfaces:**
- Produces: `SystemIrreducibilityAnalysis.to_pandas() -> pd.Series`; `IIT3SystemIrreducibilityAnalysis.to_pandas() -> pd.Series`; `Complex.to_pandas() -> pd.Series`; `ExcludedCandidate.to_pandas() -> pd.Series`.
- Consumes: `record_to_series` (`pyphi/models/pandas.py`), `ToPandasMixin`, `concise_partition` (already imported in both SIA modules).

- [ ] **Step 1: Write the failing test**

Create `test/models/test_to_pandas_coverage.py`:

```python
"""N9: every Displayable result type exports to a labeled pandas structure."""

from __future__ import annotations

import pandas as pd

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets


def test_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_iit3_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    sia = substrate.sia(state, formalism="IIT_3_0") if False else None
    with config.override(**presets.iit3):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_complex_and_excluded_to_pandas_series():
    from pyphi.models.complex import Complex
    from pyphi.models.complex import ExcludedCandidate

    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    cx = Complex(sia=sia, substrate=substrate, is_maximal=True, excluded=())
    s = cx.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)
    assert bool(s["is_maximal"]) is True

    ec = ExcludedCandidate(node_indices=(0, 1), phi=0.0)
    es = ec.to_pandas()
    assert isinstance(es, pd.Series)
    assert tuple(es["node_indices"]) == (0, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -q -p no:cacheprovider`
Expected: FAIL — `'SystemIrreducibilityAnalysis' object has no attribute 'to_pandas'`.

- [ ] **Step 3: Implement the SIA records**

In `pyphi/formalism/iit4/__init__.py`, add `ToPandasMixin` to `SystemIrreducibilityAnalysis`'s bases and add `_pandas_record`. Import at top: `from pyphi.models.pandas import ToPandasMixin`. The class is a dataclass; add the method:

```python
    def _pandas_record(self):
        return {
            "phi": float(self.phi),
            "normalized_phi": float(self.normalized_phi),
            "system": self._system_label(),
            "current_state": self.current_state,
            "partition": concise_partition(self.partition)
            if self.partition is not None
            else None,
            "n_distinctions": len(self.distinctions)
            if self.distinctions is not None
            else 0,
        }
```

In `pyphi/models/sia.py`, add `ToPandasMixin` to `IIT3SystemIrreducibilityAnalysis` bases (`from pyphi.models.pandas import ToPandasMixin`) and:

```python
    def _pandas_record(self):
        return {
            "phi": float(self.phi),
            "system": self._system_label(),
            "current_state": self.current_state,
            "partition": concise_partition(self.partition)
            if self.partition is not None
            else None,
        }
```

In `pyphi/models/complex.py`, add `ToPandasMixin` to both classes (`from .pandas import ToPandasMixin`):

```python
    # ExcludedCandidate
    def _pandas_record(self):
        return {"node_indices": self.node_indices, "phi": float(self.phi)}
```

```python
    # Complex
    def _pandas_record(self):
        record = dict(self.sia._pandas_record())
        record["is_maximal"] = self.is_maximal
        return record
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -q -p no:cacheprovider`
Expected: the three Task-1 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py pyphi/models/sia.py pyphi/models/complex.py test/models/test_to_pandas_coverage.py
git commit -m "Export the SIA family (4.0/3.0, Complex, ExcludedCandidate) to pandas

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

### Task 2: CES + Relations → DataFrame

**Files:**
- Modify: `pyphi/models/ces.py` (`CauseEffectStructure`, `PhiFold`)
- Modify: `pyphi/relations.py` (`Relations`, `Relation`, `RelationFace`)
- Test: `test/models/test_to_pandas_coverage.py`

**Interfaces:**
- Produces: `CauseEffectStructure.to_pandas() -> pd.DataFrame` (one row per distinction); `Relations.to_pandas() -> pd.DataFrame` (one row per relation); `Relation.to_pandas()` / `RelationFace.to_pandas() -> pd.Series`.
- Consumes: `Distinctions._to_pandas()` (already exists); `records_to_frame`; `record_to_series`.

- [ ] **Step 1: Write the failing test**

Append to `test/models/test_to_pandas_coverage.py`:

```python
def test_ces_to_pandas_dataframe_of_distinctions():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        ces = substrate.ces(state)
    df = ces.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(ces.distinctions)


def test_relations_to_pandas_dataframe():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        ces = substrate.ces(state)
    relations = ces.relations
    df = relations.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"phi", "degree"}
    assert len(df) == relations.num_relations()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "ces or relations" -q -p no:cacheprovider`
Expected: FAIL — no `to_pandas` on `CauseEffectStructure` / relations.

- [ ] **Step 3: Implement**

In `pyphi/models/ces.py`, add `ToPandasMixin` to `CauseEffectStructure` and `PhiFold` bases (`from .pandas import ToPandasMixin`), and to each:

```python
    def _to_pandas(self):
        return self.distinctions.to_pandas()
```

In `pyphi/relations.py`, add `ToPandasMixin` to the `Relations` base, `Relation`, and `RelationFace`. For `Relations` (collection):

```python
    def _to_pandas(self):
        from pyphi.models.pandas import records_to_frame

        rows = [
            {
                "relata": tuple(sorted(r.purview)),
                "phi": float(r.phi),
                "degree": len(r),
            }
            for r in self
        ]
        return records_to_frame(rows, columns=["relata", "phi", "degree"])
```

For `Relation` and `RelationFace` (single record → Series), add `_pandas_record`:

```python
    def _pandas_record(self):
        return {
            "relata": tuple(sorted(self.purview)),
            "phi": float(self.phi),
            "degree": len(self),
        }
```

If `Relations` is not directly iterable, iterate `self.relations` instead of `self`; confirm at Step 4 (the test asserts `len(df) == num_relations()`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "ces or relations" -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/ces.py pyphi/relations.py test/models/test_to_pandas_coverage.py
git commit -m "Export CauseEffectStructure and the relations family to pandas

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

### Task 3: Actual-causation family → Series / DataFrame

**Files:**
- Modify: `pyphi/models/actual_causation.py` (`AcSystemIrreducibilityAnalysis`, `AcRepertoireIrreducibilityAnalysis`, `CausalLink`, `Account`)
- Test: `test/models/test_to_pandas_coverage.py`

**Interfaces:**
- Produces: `AcSystemIrreducibilityAnalysis.to_pandas() -> pd.Series`; `AcRepertoireIrreducibilityAnalysis.to_pandas() -> pd.Series`; `CausalLink.to_pandas() -> pd.Series`; `Account.to_pandas() -> pd.DataFrame` (`DirectedAccount` inherits).
- Consumes: a transition example fixture. Use `examples.actual_causation()` (returns a `Transition`); confirm the accessor name at Step 1 via `grep "def actual_causation\|def .*transition" pyphi/examples.py`.

- [ ] **Step 1: Write the failing test**

First confirm the AC example accessor:
Run: `grep -n "def .*transition\|def actual_causation\|TransitionSystem\|def .*account" pyphi/examples.py`

Append to `test/models/test_to_pandas_coverage.py` (using the confirmed accessor; the example below assumes `examples.actual_causation()` returns a `Transition` exposing `.account()` / `.sia()`):

```python
def test_ac_family_to_pandas():
    transition = examples.actual_causation()
    acsia = transition.sia()
    s = acsia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["alpha"]) == float(acsia.alpha)

    account = transition.account()
    df = account.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(account)
    if len(account):
        link = account[0]
        ls = link.to_pandas()
        assert isinstance(ls, pd.Series)
        assert float(ls["alpha"]) == float(link.alpha)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "ac_family" -q -p no:cacheprovider`
Expected: FAIL — no `to_pandas` on the AC types.

- [ ] **Step 3: Implement**

In `pyphi/models/actual_causation.py`, add `ToPandasMixin` (`from .pandas import ToPandasMixin`) to the four classes.

`AcRepertoireIrreducibilityAnalysis` and `CausalLink` (scalar → Series) — `_pandas_record`:

```python
    def _pandas_record(self):
        return {
            "alpha": float(self.alpha),
            "direction": str(self.direction),
            "mechanism": tuple(self.mechanism),
            "purview": tuple(self.purview),
        }
```

`AcSystemIrreducibilityAnalysis` (scalar → Series):

```python
    def _pandas_record(self):
        return {
            "alpha": float(self.alpha),
            "direction": str(self.direction),
            "before_state": self.before_state,
            "after_state": self.after_state,
        }
```

`Account` (collection of `CausalLink` → DataFrame); `DirectedAccount` inherits:

```python
    def _to_pandas(self):
        from .pandas import records_to_frame

        rows = [link._pandas_record() for link in self]
        return records_to_frame(
            rows, columns=["alpha", "direction", "mechanism", "purview"]
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "ac_family" -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/actual_causation.py test/models/test_to_pandas_coverage.py
git commit -m "Export the actual-causation family to pandas

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

### Task 4: Structural types (System, Substrate, TPMs) → DataFrame

**Files:**
- Modify: `pyphi/system.py` (`System`)
- Modify: `pyphi/substrate.py` (`Substrate`)
- Modify: `pyphi/core/tpm/factored.py` (`FactoredTPM`)
- Modify: `pyphi/core/tpm/joint_distribution.py` (`JointTPM`)
- Test: `test/models/test_to_pandas_coverage.py`

**Interfaces:**
- Produces: `System.to_pandas() -> pd.DataFrame` (per-unit state); `Substrate.to_pandas() -> pd.DataFrame` (its TPM); `FactoredTPM.to_pandas() -> pd.DataFrame` (state-by-node matrix); `JointTPM.to_pandas() -> pd.DataFrame`.
- Consumes: `state_multiindex(node_labels, indices)`; `FactoredTPM.factor(i)`, `.alphabet_sizes`, `.unit_labels_for_display()`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_factored_tpm_to_pandas_state_by_node():
    substrate = examples.basic_substrate()
    tpm = substrate.factored_tpm
    df = tpm.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == tpm.n_nodes  # one column per unit (binary)


def test_substrate_to_pandas_delegates_to_tpm():
    substrate = examples.basic_substrate()
    df = substrate.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == substrate.factored_tpm.n_nodes


def test_system_to_pandas_per_unit_state():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    from pyphi import System

    system = System(substrate, state)
    df = system.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(system.node_indices)
    assert "state" in df.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "tpm or substrate or system" -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `pyphi/core/tpm/factored.py`, add `ToPandasMixin` to `FactoredTPM` (`from pyphi.models.pandas import ToPandasMixin`) — binary → P(on) matrix; k-ary → long-format:

```python
    def _to_pandas(self):
        import pandas as pd

        from pyphi.models.pandas import state_multiindex

        n = self.n_nodes
        a = self.alphabet_sizes
        labels = self.unit_labels_for_display()
        index = state_multiindex(self.node_labels, tuple(range(n)))
        if all(size == 2 for size in a):
            data = [[float(self.factor(i)[tuple(s)][1]) for i in range(n)] for s in index]
            return pd.DataFrame(data, index=index, columns=list(labels))
        rows = []
        for s in index:
            for i in range(n):
                dist = self.factor(i)[tuple(s)]
                for next_state, p in enumerate(dist):
                    rows.append(
                        {"unit": labels[i], "next_state": next_state, "probability": float(p)}
                    )
        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples([
            tuple(s) for s in index for _ in range(n) for _ in a
        ]) if False else None).reset_index(drop=True) if False else pd.DataFrame(rows)
```

Note: the k-ary branch returns a long-format frame (`unit`, `next_state`, `probability`). Drop the dead `if False` scaffolding when implementing — write only:

```python
        rows = []
        for s in index:
            for i in range(n):
                for next_state, p in enumerate(self.factor(i)[tuple(s)]):
                    rows.append(
                        {
                            "state": tuple(s),
                            "unit": labels[i],
                            "next_state": next_state,
                            "probability": float(p),
                        }
                    )
        return pd.DataFrame(rows)
```

In `pyphi/core/tpm/joint_distribution.py`, add `ToPandasMixin` to `JointTPM` and delegate to the factored form it can produce, or build the same state-by-node matrix from its array. Implement:

```python
    def _to_pandas(self):
        return self.factored().to_pandas()
```

Confirm `JointTPM` exposes a `.factored()` (or equivalent) at Step 4; if not, build via `FactoredTPM.from_joint(self).to_pandas()` using the existing constructor.

In `pyphi/substrate.py`, add `ToPandasMixin` to `Substrate` (`from pyphi.models.pandas import ToPandasMixin`):

```python
    def _to_pandas(self):
        return self.factored_tpm.to_pandas()
```

In `pyphi/system.py`, add `ToPandasMixin` to `System`:

```python
    def _to_pandas(self):
        import pandas as pd

        labels = self.substrate.node_labels
        rows = [
            {"node": idx, "label": labels[idx], "state": self.state[idx]}
            for idx in self.node_indices
        ]
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "tpm or substrate or system" -q -p no:cacheprovider`
Expected: PASS. (Resolve the `JointTPM` factored-accessor name here if the delegate raised.)

- [ ] **Step 5: Commit**

```bash
git add pyphi/system.py pyphi/substrate.py pyphi/core/tpm/factored.py pyphi/core/tpm/joint_distribution.py test/models/test_to_pandas_coverage.py
git commit -m "Export structural types (System, Substrate, TPMs) to pandas

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

### Task 5: Partitions / cuts + UnitState

**Files:**
- Modify: `pyphi/models/partitions.py` (`_PartitionBase` — the shared base for cuts and partitions)
- Modify: `pyphi/partition.py` (`CompleteJointPartition`, `AtomicJointPartition` — if they do not share `_PartitionBase`)
- Modify: `pyphi/models/state_specification.py` (`UnitState`)
- Test: `test/models/test_to_pandas_coverage.py`

**Interfaces:**
- Produces: a labeled cut-grid `DataFrame` on every partition/cut; `UnitState.to_pandas() -> pd.Series`.
- Consumes: `_PartitionBase.cut_matrix(n)` (returns an `n×n` int array; 1 = severed a→b); `_PartitionBase.node_indices` / `node_labels`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_partition_to_pandas_cut_grid():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    partition = sia.partition
    df = partition.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == df.shape[1]  # square from × to grid


def test_unitstate_to_pandas_series():
    from pyphi.models.state_specification import UnitState

    us = UnitState(index=1, state=1, label="B")
    s = us.to_pandas()
    assert isinstance(s, pd.Series)
    assert int(s["state"]) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "partition or unitstate" -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `pyphi/models/partitions.py`, add `ToPandasMixin` to `_PartitionBase` (`from pyphi.models.pandas import ToPandasMixin`):

```python
    def _to_pandas(self):
        import pandas as pd

        indices = tuple(self.node_indices)
        n = max(indices) + 1 if indices else 0
        matrix = self.cut_matrix(n)
        labels = [str(self.node_labels[i]) for i in indices]
        sub = matrix[list(indices)][:, list(indices)]
        return pd.DataFrame(sub, index=labels, columns=labels)
```

Confirm `_PartitionBase` exposes `node_indices` and `node_labels` at Step 4; if a concrete partition stores them under different names, adapt the accessor. If `pyphi/partition.py`'s `CompleteJointPartition` / `AtomicJointPartition` do not inherit `_PartitionBase`, add `ToPandasMixin` + the same `_to_pandas` there (they expose `cut_matrix`).

In `pyphi/models/state_specification.py`, `UnitState` already lists `Displayable`; add `ToPandasMixin` and `_pandas_record`:

```python
    def _pandas_record(self):
        label = str(self.index) if self.label is None else self.label
        return {"unit": label, "state": self.state}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest test/models/test_to_pandas_coverage.py -k "partition or unitstate" -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/partitions.py pyphi/partition.py pyphi/models/state_specification.py test/models/test_to_pandas_coverage.py
git commit -m "Export partitions/cuts and UnitState to pandas

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

### Task 6: Coverage-invariant test + changelog + ROADMAP

**Files:**
- Modify: `test/models/test_to_pandas_coverage.py` (add the invariant)
- Create: `changelog.d/n9-to-pandas-coverage.feature.md`
- Modify: `ROADMAP.md` (flip the N9 dashboard row to ✅ landed)

**Interfaces:**
- Consumes: `pyphi.display.mixin.Displayable` (walk its subclass tree); the example fixtures used in Tasks 1–5.

- [ ] **Step 1: Write the coverage-invariant test**

Append to `test/models/test_to_pandas_coverage.py`. The invariant constructs one representative instance per result type and asserts `to_pandas()` returns a `Series`/`DataFrame`. Because constructing every type generically is brittle, the invariant instead checks the *static* guarantee: every `Displayable` subclass either defines/inherits `to_pandas` returning a `Series`/`DataFrame` (i.e. has a `_pandas_record` or `_to_pandas` in its MRO), with an explicit allow-list of intentionally-excluded abstract/mixin bases.

```python
def test_every_displayable_has_to_pandas():
    import pyphi  # noqa: F401  (ensure result modules are imported)
    from pyphi.display.mixin import Displayable
    from pyphi.models.pandas import ToPandasMixin

    def all_subclasses(cls):
        out = set()
        for sub in cls.__subclasses__():
            out.add(sub)
            out |= all_subclasses(sub)
        return out

    missing = []
    for cls in all_subclasses(Displayable):
        has_export = any(
            "_pandas_record" in base.__dict__ or "_to_pandas" in base.__dict__
            for base in cls.__mro__
        ) or issubclass(cls, ToPandasMixin)
        if not has_export:
            missing.append(f"{cls.__module__}.{cls.__name__}")
    assert not missing, f"Displayable types without to_pandas: {sorted(missing)}"
```

- [ ] **Step 2: Run the invariant**

Run: `uv run pytest test/models/test_to_pandas_coverage.py::test_every_displayable_has_to_pandas -q -p no:cacheprovider`
Expected: PASS (all result types now covered). If it lists any type, that type was missed in Tasks 1–5 — add its `ToPandasMixin` + hook before proceeding.

- [ ] **Step 3: Create the changelog fragment**

Create `changelog.d/n9-to-pandas-coverage.feature.md`:

```markdown
Extended the labeled-export (`to_pandas()`) convention to every user-facing result type: the system irreducibility analyses (IIT 4.0 and 3.0), the cause-effect structure, the relations family, the actual-causation family (`AcSIA`, `AcRIA`, `CausalLink`, `Account`), `Complex` / `ExcludedCandidate`, the structural types (`System`, `Substrate`, `FactoredTPM`, `JointTPM`), partitions/cuts, and `UnitState`. Scalar-record results return a `Series`; collections and structural results return a `DataFrame`. A coverage test now guarantees every displayable result type exports to pandas.
```

- [ ] **Step 4: Flip the N9 row in `ROADMAP.md`**

Find the N9 row (`grep -n "N9 labeled-export" ROADMAP.md`) and change `⬜ open` → `✅ landed`. Confirm `git diff ROADMAP.md` shows only this one-line change before staging.

- [ ] **Step 5: Full verification**

Run: `uv run --all-extras pytest`
Expected: full suite passes (doctests + Hypothesis included). Confirm `test/models/test_to_pandas_coverage.py` is collected and green.

- [ ] **Step 6: Commit**

```bash
git add test/models/test_to_pandas_coverage.py changelog.d/n9-to-pandas-coverage.feature.md ROADMAP.md
git commit -m "Add to_pandas coverage invariant; changelog; mark N9 landed

$(printf 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve')"
```

---

## Self-Review

**Spec coverage:** SIA family → T1; CES + Relations → T2; AC family → T3; structural (System/Substrate/TPMs) → T4; partitions/cuts + UnitState → T5; coverage invariant + changelog + ROADMAP → T6. xarray explicitly out of scope (not implemented). ✓

**Placeholder scan:** The Task-4 FactoredTPM step contains scaffolding marked for deletion (the `if False` block) — the step explicitly instructs writing only the clean k-ary long-format version shown below it. No other TBD/TODO. Each step shows complete code.

**Type consistency:** `_pandas_record` (Series) vs `_to_pandas` (DataFrame) used consistently per the P14d convention. `Complex._pandas_record` consumes `self.sia._pandas_record()` defined in T1. `Account._to_pandas` consumes `CausalLink._pandas_record` defined in the same task. The coverage invariant (T6) consumes `ToPandasMixin` from all prior tasks.

**Known confirmations deferred to execution** (each guarded by its task's test): the AC example accessor name (`examples.actual_causation()`), `Relations` iteration (`for r in self` vs `self.relations`), `JointTPM` factored accessor, and `_PartitionBase` label/index attribute names. These are exact-name lookups the TDD step surfaces immediately; the design and shapes are fixed.
