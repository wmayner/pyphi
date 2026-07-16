# FactoredTPM Sparse-Factor Crash Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two crashes on connectivity-sparse `FactoredTPM`s (size-1 non-input axes): backend `select` IndexError on nonzero conditioning states, and macro TPM construction flattening factors at their stored shape instead of the full universe grid.

**Architecture:** Two one-line semantic fixes at the root sites — `select` clamps size-1 axes to index 0 (both storage backends), and `_discounted_on_probabilities` broadcasts each ON-probability slice to the full universe shape before flattening. Regression tests compare the sparse form against a dense broadcast-to-full-shape control.

**Tech Stack:** Python 3.13, numpy, pytest; xarray optional extra for the second backend.

## Global Constraints

- Run everything via `uv run` from the worktree root (`.claude/worktrees/factored-sparse-factors`).
- Commit messages end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe` trailers. Never `--no-verify`.
- Docstrings: NumPy style, final-state impersonal voice, no planning-artifact references.
- Final gate: pathless `uv run pytest` (no path argument) redirected to a file; read the summary line.

**Background for all tasks.** A `FactoredTPM` stores factor `i` with shape `(a_1, …, a_N, a_i)`; an input axis the node does not depend on is kept at size 1 ("size-1 axes encode the connectivity structure and are never squeezed away" — `pyphi/core/tpm/factored.py` module docstring). A size-1 input axis means the conditional is constant in that input, so conditioning at any state must return the single stored slice, and expansion to the universe grid must broadcast it.

The shared test substrate (used in Tasks 1–3): three binary nodes, 0 and 1 copy each other, 2 has a self-loop; non-input axes size 1.

```python
def _sparse_three_node_factors():
    # 0 <-> 1 copy each other; 2 has a self-loop. Non-input axes have size 1.
    f0 = np.zeros((1, 2, 1, 2))
    f0[0, 0, 0] = [1, 0]
    f0[0, 1, 0] = [0, 1]
    f1 = np.zeros((2, 1, 1, 2))
    f1[0, 0, 0] = [1, 0]
    f1[1, 0, 0] = [0, 1]
    f2 = np.zeros((1, 1, 2, 2))
    f2[0, 0, 0] = [1, 0]
    f2[0, 0, 1] = [0, 1]
    return [f0, f1, f2]
```

---

### Task 1: Backend `select` clamps size-1 axes to index 0

**Files:**
- Modify: `pyphi/core/tpm/_factored_backends.py:60-68` (`_NdarrayBackend.select`)
- Modify: `pyphi/core/tpm/_factored_backends_xarray.py:57-64` (`_XarrayBackend.select`)
- Test: `test/core/test_factored_tpm.py`

**Interfaces:**
- Consumes: existing `_StorageBackend.select(i, fixed)` protocol.
- Produces: `select` that returns the stored slice for size-1 fixed axes regardless of the fixed state; signature unchanged. Tasks 2–3 rely on `FactoredTPM.condition`/`subtpm` no longer raising on sparse factors.

- [ ] **Step 0: Install optional extras into the worktree venv** (needed for the xarray test variant and the final pathless sweep)

```bash
cd /Users/will/projects/pyphi/.claude/worktrees/factored-sparse-factors
WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"
env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray]" pot
```

Expected: installs succeed; `uv run python -c "import xarray, ot"` exits 0.

- [ ] **Step 1: Write the failing tests**

Append to `test/core/test_factored_tpm.py` (the file already imports `numpy as np`, `pytest`, `FactoredTPM`, and defines `requires_xarray` at line ~216):

```python
# --- connectivity-sparse factors (size-1 non-input axes) ---


def _sparse_three_node_factors():
    # 0 <-> 1 copy each other; 2 has a self-loop. Non-input axes have size 1.
    f0 = np.zeros((1, 2, 1, 2))
    f0[0, 0, 0] = [1, 0]
    f0[0, 1, 0] = [0, 1]
    f1 = np.zeros((2, 1, 1, 2))
    f1[0, 0, 0] = [1, 0]
    f1[1, 0, 0] = [0, 1]
    f2 = np.zeros((1, 1, 2, 2))
    f2[0, 0, 0] = [1, 0]
    f2[0, 0, 1] = [0, 1]
    return [f0, f1, f2]


def test_condition_size1_axis_nonzero_state() -> None:
    tpm = FactoredTPM(factors=_sparse_three_node_factors())
    c1 = tpm.condition({2: 1})
    c0 = tpm.condition({2: 0})
    # Factors 0 and 1 are independent of node 2 (size-1 axis): identical slices.
    np.testing.assert_array_equal(c1.factor(0), c0.factor(0))
    np.testing.assert_array_equal(c1.factor(1), c0.factor(1))
    # Factor 2 depends on node 2: conditioning selects the real slice.
    np.testing.assert_array_equal(np.squeeze(c1.factor(2)), [0.0, 1.0])
    np.testing.assert_array_equal(np.squeeze(c0.factor(2)), [1.0, 0.0])


def test_subtpm_size1_axis_nonzero_state() -> None:
    tpm = FactoredTPM(factors=_sparse_three_node_factors())
    # Free units 0 and 1 are independent of node 2, so the conditioned
    # sub-TPM is the same for either fixed state.
    assert tpm.subtpm((2,), (1,)) == tpm.subtpm((2,), (0,))


@requires_xarray
def test_condition_size1_axis_nonzero_state_xarray() -> None:
    tpm = FactoredTPM(factors=_sparse_three_node_factors(), backend="xarray")
    c1 = tpm.condition({2: 1})
    c0 = tpm.condition({2: 0})
    np.testing.assert_array_equal(c1.factor(0), c0.factor(0))
    np.testing.assert_array_equal(c1.factor(1), c0.factor(1))
    np.testing.assert_array_equal(np.squeeze(c1.factor(2)), [0.0, 1.0])


@requires_xarray
def test_subtpm_size1_axis_nonzero_state_xarray() -> None:
    tpm = FactoredTPM(factors=_sparse_three_node_factors(), backend="xarray")
    assert tpm.subtpm((2,), (1,)) == tpm.subtpm((2,), (0,))
```

Note: `requires_xarray` is defined mid-file (~line 216); place the new tests after it.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest test/core/test_factored_tpm.py -k size1 -v > /tmp/t1_fail.log 2>&1; tail -20 /tmp/t1_fail.log
```

Expected: 4 failures (or 2 failures + 2 xarray failures), each `IndexError: index 1 is out of bounds for axis 2 with size 1`.

- [ ] **Step 3: Fix `_NdarrayBackend.select`**

In `pyphi/core/tpm/_factored_backends.py`, replace the loop body of `select`:

```python
    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        idx: list[Any] = [slice(None)] * factor.ndim
        for j, state_j in fixed.items():
            # A size-1 input axis means the factor is constant in that
            # input, so conditioning on any state selects the stored slice.
            idx[j] = 0 if factor.shape[j] == 1 else state_j
        out = factor[tuple(idx)]
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out
```

- [ ] **Step 4: Fix `_XarrayBackend.select`**

In `pyphi/core/tpm/_factored_backends_xarray.py`, replace the `idx` construction:

```python
    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        # A size-1 input axis means the factor is constant in that input,
        # so conditioning on any state selects the stored slice.
        idx: dict[str, int] = {
            f"in_{j}": (0 if factor.sizes[f"in_{j}"] == 1 else state_j)
            for j, state_j in fixed.items()
        }
        sliced = factor.isel(idx)
        out = sliced.values
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest test/core/test_factored_tpm.py -v > /tmp/t1_pass.log 2>&1; tail -5 /tmp/t1_pass.log
```

Expected: all tests in the file PASS (new and pre-existing).

- [ ] **Step 6: Commit**

```bash
git add pyphi/core/tpm/_factored_backends.py pyphi/core/tpm/_factored_backends_xarray.py test/core/test_factored_tpm.py
git commit -m "Fix backend select on size-1 non-input axes

A size-1 input axis of a FactoredTPM factor declares that the node is
independent of that input, so conditioning on any state must return the
single stored slice. Both storage backends indexed fixed axes with the
raw state, raising IndexError when a non-input unit was conditioned in
a nonzero state — breaking condition, subtpm, cause_conditioned,
effect_marginal, and System construction on connectivity-sparse
substrates with nonzero background states.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 2: End-to-end regression — System background conditioning on a sparse substrate

**Files:**
- Modify: `test/core/test_repertoire_sparse_heterogeneous.py` (module docstring + new test)

**Interfaces:**
- Consumes: the Task 1 fix (`condition`/`subtpm` work on sparse factors at nonzero states).
- Produces: nothing new — a pure regression pin. No production code changes.

- [ ] **Step 1: Update the stale module docstring and add the test**

Replace the module docstring of `test/core/test_repertoire_sparse_heterogeneous.py` (it claims the sparse inputs "currently raise a shape error", which is no longer true):

```python
"""Regression: cause/effect repertoires on sparse + heterogeneous-alphabet
networks, and background conditioning on connectivity-sparse factors
(size-1 non-input axes) at nonzero background states."""
```

Add `pytest` and `FactoredTPM` imports at the top (keep isort order):

```python
import pytest

from pyphi.core.tpm.factored import FactoredTPM
```

Append the test:

```python
def _sparse_binary_factors():
    # 0 <-> 1 copy each other; 2 has a self-loop. Non-input axes have size 1.
    f0 = np.zeros((1, 2, 1, 2))
    f0[0, 0, 0] = [1, 0]
    f0[0, 1, 0] = [0, 1]
    f1 = np.zeros((2, 1, 1, 2))
    f1[0, 0, 0] = [1, 0]
    f1[1, 0, 0] = [0, 1]
    f2 = np.zeros((1, 1, 2, 2))
    f2[0, 0, 0] = [1, 0]
    f2[0, 0, 1] = [0, 1]
    return [f0, f1, f2]


def test_background_conditioning_on_size1_axis_matches_dense():
    factors = _sparse_binary_factors()
    sparse_sub = Substrate.from_factored(FactoredTPM(factors=factors))
    dense = [np.broadcast_to(f, (2, 2, 2, 2)).copy() for f in factors]
    dense_sub = Substrate.from_factored(FactoredTPM(factors=dense))
    # Node 2 is background, fixed ON — a size-1 non-input axis of the free
    # units' factors. The sparse and dense forms are the same TPM, so the
    # analysis must agree.
    state = (0, 0, 1)
    phi_sparse = System(sparse_sub, state, (0, 1)).sia().phi
    phi_dense = System(dense_sub, state, (0, 1)).sia().phi
    assert phi_sparse == pytest.approx(phi_dense, abs=1e-12)
```

- [ ] **Step 2: Run the test file**

```bash
uv run pytest test/core/test_repertoire_sparse_heterogeneous.py -v > /tmp/t2.log 2>&1; tail -8 /tmp/t2.log
```

Expected: all PASS (the new test passes because Task 1 landed; before Task 1 it raised `IndexError`).

- [ ] **Step 3: Commit**

```bash
git add test/core/test_repertoire_sparse_heterogeneous.py
git commit -m "Pin System analysis on sparse substrates with nonzero background

Regression for backend select on size-1 axes: a System whose background
unit is fixed ON, where that unit is a non-input of the free units'
factors, must give the same phi as the dense broadcast-to-full-shape
form of the same TPM. Also refreshes the module docstring, which still
described these inputs as raising a shape error.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 3: Macro TPM construction broadcasts sparse factors to the universe grid

**Files:**
- Modify: `pyphi/macro/tpm.py:144` (`_discounted_on_probabilities`)
- Test: `test/macro/test_macro_tpm.py`

**Interfaces:**
- Consumes: `FactoredTPM.factor(i)` (may have size-1 input axes), `FactoredTPM.alphabet_sizes`.
- Produces: `_discounted_on_probabilities` returning shape `(2**n, n)` for sparse and dense factors alike; `macro_tpms` and `MacroSystem.from_micro` work on sparse substrates.

- [ ] **Step 1: Write the failing tests**

Append to `test/macro/test_macro_tpm.py` (already imports `numpy as np`, `_discounted_on_probabilities`, `macro_tpms`, `MacroUnit`, `coarse_grain`, `Substrate`; add `from pyphi.core.tpm.factored import FactoredTPM` to the imports in isort order):

```python
class TestSparseFactors:
    """Connectivity-sparse micro substrates (size-1 non-input factor axes)."""

    @staticmethod
    def _sparse_factors():
        # 0 <-> 1 copy each other; 2 has a self-loop. Non-input axes size 1.
        f0 = np.zeros((1, 2, 1, 2))
        f0[0, 0, 0] = [1, 0]
        f0[0, 1, 0] = [0, 1]
        f1 = np.zeros((2, 1, 1, 2))
        f1[0, 0, 0] = [1, 0]
        f1[1, 0, 0] = [0, 1]
        f2 = np.zeros((1, 1, 2, 2))
        f2[0, 0, 0] = [1, 0]
        f2[0, 0, 1] = [0, 1]
        return [f0, f1, f2]

    def test_discounted_on_probabilities_full_universe_shape(self):
        factored = FactoredTPM(factors=self._sparse_factors())
        unit = MacroUnit((0, 1), 1, coarse_grain(2, (1, 2)))
        on_probs = _discounted_on_probabilities(factored, (unit,), 0)
        assert on_probs.shape == (8, 3)

    def test_macro_tpms_match_dense_control(self):
        factors = self._sparse_factors()
        sparse_sub = Substrate.from_factored(FactoredTPM(factors=factors))
        dense_sub = Substrate.from_factored(
            FactoredTPM(
                factors=[np.broadcast_to(f, (2, 2, 2, 2)).copy() for f in factors]
            )
        )
        unit = MacroUnit((0, 1), 1, coarse_grain(2, (1, 2)))
        history = [(0, 0, 0)]
        cause_sparse, effect_sparse = macro_tpms(sparse_sub, (unit,), history)
        cause_dense, effect_dense = macro_tpms(dense_sub, (unit,), history)
        for got, expected in [
            (cause_sparse, cause_dense),
            (effect_sparse, effect_dense),
        ]:
            for i in range(got.n_nodes):
                np.testing.assert_allclose(
                    got.factor(i), expected.factor(i), rtol=0, atol=1e-12
                )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest test/macro/test_macro_tpm.py::TestSparseFactors -v > /tmp/t3_fail.log 2>&1; tail -15 /tmp/t3_fail.log
```

Expected: 2 failures — the shape test asserts `(2, 3) == (8, 3)`; the control test crashes with the matmul `ValueError` (size 2 vs 8).

- [ ] **Step 3: Fix `_discounted_on_probabilities`**

In `pyphi/macro/tpm.py`, change line 144 from:

```python
        p_on = factored.factor(i)[..., 1]
```

to:

```python
        # Broadcast to the full universe grid: a size-1 (non-input) axis is
        # constant in that input, and the flatten below requires 2**n rows.
        p_on = np.broadcast_to(factored.factor(i)[..., 1], factored.alphabet_sizes)
```

The three downstream branches are unaffected: means over the broadcast array equal means over the stored array (uniform replication), and `reshape(-1, order="F")` copies the non-contiguous view.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest test/macro/ -v > /tmp/t3_pass.log 2>&1; tail -5 /tmp/t3_pass.log
```

Expected: entire macro suite PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/tpm.py test/macro/test_macro_tpm.py
git commit -m "Fix macro TPM construction on connectivity-sparse factors

_discounted_on_probabilities flattened each factor's ON-probability
slice at its stored shape, so factors with size-1 non-input axes
produced columns shorter than the 2**n universe size and macro TPM
construction crashed at the transition-matrix product. The slice is now
broadcast to the full universe grid before flattening; means are
unchanged under uniform replication, so discounting is unaffected.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 4: Changelog fragment and full-suite verification

**Files:**
- Create: `changelog.d/factored-sparse-factors.fix.md`

**Interfaces:**
- Consumes: Tasks 1–3 committed.
- Produces: the branch ready to merge; suite green.

- [ ] **Step 1: Write the changelog fragment**

```bash
cat > changelog.d/factored-sparse-factors.fix.md <<'EOF'
Fixed two crashes on connectivity-sparse `FactoredTPM`s (factors with size-1
non-input axes): conditioning a size-1 axis at a nonzero state raised
`IndexError` (breaking `condition()`, `subtpm()`, and `System` construction
with nonzero background states), and macro TPM construction crashed on the
transition-matrix product because factors were flattened at their stored
shape instead of the full universe grid.
EOF
```

- [ ] **Step 2: Run the full pathless suite**

```bash
uv run pytest -q > /tmp/full_suite.log 2>&1; tail -5 /tmp/full_suite.log
```

Expected: summary line with 0 failures (≈3630+ passed, ~286 skipped). Read the summary line from the file — do not trust the pipeline exit code. If `test/integration/test_perf_counters.py` fails, regenerate the pin with `uv run python scripts/gen_perf_counts.py` and inspect `git diff test/data/perf/call_counts.json` — the select fix must not change call counts on non-sparse fixtures; any non-macro/non-sparse change is a red flag to investigate, not commit.

- [ ] **Step 3: Commit**

```bash
git add changelog.d/factored-sparse-factors.fix.md
git commit -m "Add changelog fragment for the sparse-factor crash fixes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Post-merge bookkeeping (main tree, not the worktree)

`REVIEW-2026-07-13.md` is untracked and lives only in the main tree. After the
branch merges to main, update its top status block to record that the two Wave-1
findings ("Backend select crashes on size-1 (non-input) axes" and "macro_tpms
crashes on FactoredTPM with broadcast (singleton-dim) factors") are fixed, with
the merge commit hash. Update the memory file
`project_whole_library_review_2026_07_13.md` likewise.
