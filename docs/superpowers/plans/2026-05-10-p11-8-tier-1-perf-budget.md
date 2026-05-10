# Tier 1 Inline Perf Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-commit pytest perf-budget smoke gate that trips on catastrophic (≥4x) wall-time regressions in five hot-path golden fixtures.

**Architecture:** A new test file `test/test_perf_budget.py` parametrizes over five canary fixture names, runs each through the existing `compute_all_layers` golden harness inside that fixture's `config_context()`, and asserts elapsed wall time stays under a pinned per-fixture budget (`max(3.0, 4 × typical)`). A `perf` marker is registered in `test/conftest.py` so the tests are addressable but not opt-in.

**Tech Stack:** pytest, `time.perf_counter`, existing `test/golden/` harness (no new dependencies)

---

## Spec

See `docs/superpowers/specs/2026-05-10-p11-8-tier-1-perf-budget-design.md` (committed at `cad8a967`). Read it before starting if any decision feels ambiguous.

## Branch base

`2.0` head `cad8a967`. Land all work as a single commit on `2.0`. Do **not** push without explicit per-action consent.

## File structure

```
test/
├── conftest.py            MODIFY: add one addinivalue_line in pytest_configure (registers `perf` marker)
└── test_perf_budget.py    CREATE: parametrized test (~30 lines)
```

No other files change. No new test data. No CI workflow changes.

## Project conventions (do not violate)

- **No P-number markers, "Phase A", `TODO(Px)`, or "per ROADMAP" anywhere in source / comments / docstrings / commit messages.** The spec itself is the only place planning context lives.
- **Default to no comments unless WHY is non-obvious.** The test file's module docstring carries the rationale (catastrophic-regression smoke gate). Inside the test body, no commentary.
- **Pre-commit hooks (ruff + pyright) must pass on commit.** Never bypass with `--no-verify`. If a hook fails, fix the underlying issue (run `uv run ruff check` and `uv run pyright` directly to see the message).
- **GPG signing bypass is authorized for this session only.** Use `git -c commit.gpgsign=false commit ...` for the final commit. Do not change git config.
- **Use `uv run` for any Python invocation.** The project's venv lives behind uv; `python` / `pytest` directly may or may not pick it up.

---

## Phase 1 — Single-commit implementation

All work lands as one commit. Calibration, marker registration, test file, and acceptance checks happen in order; the commit at the end captures everything together.

### Task 1: Calibrate per-canary wall-time budgets

**Files:** none modified (measurement only)

**Why this is first:** The test file pins concrete budget numbers. Pinning made-up numbers and tuning post-hoc would invert the dependency. Calibration is the input; the test file is the output.

- [ ] **Step 1: Confirm branch state**

```bash
git log --oneline -1
```

Expected: `cad8a967 Tier 1 perf budget spec: inline pytest wall-time floors`

- [ ] **Step 2: Measure each canary three times sequentially**

Run each fixture in isolation (one at a time, sequential, not parallel). Use `-q` so output is grep-friendly; pipe through `time` so wall time lands in stderr. Capture each of three runs per fixture.

```bash
for name in basic_iit3_emd basic_iit4_2023 basic_iit4_2026 xor_iit4_2026 logistic3_k8_iit4_2026; do
  echo "=== $name ==="
  for i in 1 2 3; do
    /usr/bin/time -p uv run pytest test/test_golden_regression.py -k "$name" -q 2>&1 | tail -3
  done
done
```

Expected: each fixture passes 3 times, each run reports a `real` wall time. Extract the `real` values; you want 15 numbers total (5 fixtures × 3 runs).

- [ ] **Step 3: Compute budgets**

For each fixture, take the median of its three wall times, multiply by 4, apply a floor of 3.0 seconds, round up to the nearest clean number from `{3, 4, 5, 8, 10, 15, 20, 30, 45, 60}`.

```
budget = max(3.0, ceil_to_clean(4 × median))
```

Worked example: if `basic_iit4_2026` medians at 0.8s → 4 × 0.8 = 3.2 → ceil to next clean = `4.0`. If `logistic3_k8_iit4_2026` medians at 4.5s → 4 × 4.5 = 18.0 → ceil to `20.0`.

Record the median and budget for each fixture in a scratch file or paste buffer — they go in the commit message at Task 11.

**Expected output: a 5-row table you've written down somewhere.** Example shape (numbers will differ on your machine):

```
basic_iit3_emd:           median 0.6s  → budget 3.0s
basic_iit4_2023:          median 0.9s  → budget 4.0s
basic_iit4_2026:          median 1.0s  → budget 4.0s
xor_iit4_2026:            median 1.8s  → budget 8.0s
logistic3_k8_iit4_2026:   median 4.5s  → budget 20.0s
```

### Task 2: Register the `perf` pytest marker

**Files:**
- Modify: `test/conftest.py` (inside the existing `pytest_configure` function, ~line 86–94)

- [ ] **Step 1: Read the current marker registrations**

Open `test/conftest.py` and locate the `pytest_configure(config)` function. It currently registers `golden` and `robust` via two `config.addinivalue_line("markers", ...)` calls.

- [ ] **Step 2: Add a third addinivalue_line for `perf`**

Insert directly after the existing `robust` registration:

```python
    config.addinivalue_line(
        "markers",
        "perf: Wall-time floor assertion on a hot-path fixture",
    )
```

Indentation matches the surrounding two calls (4 spaces, function-body level). Leading blank line not required — adjacent to the existing blocks.

- [ ] **Step 3: Verify the marker is registered**

```bash
uv run pytest --markers 2>&1 | grep -A1 "^@pytest.mark.perf"
```

Expected output:

```
@pytest.mark.perf: Wall-time floor assertion on a hot-path fixture
```

If you don't see this: the indentation or function placement is off. Re-read the file and confirm the new call lives inside `pytest_configure`, not at module top level.

### Task 3: Add the perf-budget test file (TDD: failing version first)

**Files:**
- Create: `test/test_perf_budget.py`

**Why tight budgets first:** Writing the file with `0.001` budgets forces all five tests to fail with a timing diagnostic. That confirms (a) imports resolve, (b) `_FIXTURES_BY_NAME` finds each fixture, (c) the timing loop runs, and (d) the assertion path executes. If the file is silently broken (e.g., import error swallowed), tight budgets surface it before calibrated values mask the problem.

- [ ] **Step 1: Write the test file with intentionally tight budgets**

Create `test/test_perf_budget.py` with this exact content:

```python
"""Wall-time floor assertions on hot-path fixtures.

A 4x margin over typical wall time, with a 3-second floor, catches
catastrophic regressions (e.g., a previously-shipped 60-300x slowdown
from defensive config.override calls per partition) without being
brittle on slow CI runners. This is a smoke gate, not a benchmark
suite — per-fixture micro-budgets and trend tracking belong in the
wider benchmark rewrite.
"""

from __future__ import annotations

import time

import pytest

from .golden import ALL_FIXTURES
from .golden.compute import compute_all_layers

_FIXTURES_BY_NAME = {f.name: f for f in ALL_FIXTURES}

PERF_BUDGETS_S: dict[str, float] = {
    "basic_iit3_emd": 0.001,
    "basic_iit4_2023": 0.001,
    "basic_iit4_2026": 0.001,
    "xor_iit4_2026": 0.001,
    "logistic3_k8_iit4_2026": 0.001,
}


@pytest.mark.perf
@pytest.mark.parametrize("name", list(PERF_BUDGETS_S))
def test_perf_budget(name: str) -> None:
    budget_s = PERF_BUDGETS_S[name]
    fixture = _FIXTURES_BY_NAME[name]
    start = time.perf_counter()
    with fixture.config_context():
        compute_all_layers(fixture)
    elapsed = time.perf_counter() - start
    assert elapsed < budget_s, (
        f"{name} took {elapsed:.2f}s, exceeded perf budget {budget_s}s. "
        f"4x margin over typical suggests catastrophic regression — "
        f"profile the hot path before retuning."
    )
```

- [ ] **Step 2: Run the test to verify it fails with timing diagnostics**

```bash
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -20
```

Expected: 5 failures. Each failure message contains:
- The fixture name (e.g., `basic_iit3_emd`)
- The elapsed time (e.g., `took 0.62s`)
- The budget (`exceeded perf budget 0.001s`)
- The diagnostic suffix mentioning the 4x margin

If any of the 5 fixtures errors instead of failing (e.g., `KeyError`, `ImportError`): something is wrong with the fixture lookup or import path. Fix before continuing.

### Task 4: Replace tight budgets with calibrated values

**Files:**
- Modify: `test/test_perf_budget.py:21-26` (the `PERF_BUDGETS_S` dict)

- [ ] **Step 1: Update PERF_BUDGETS_S to the calibrated values from Task 1**

Replace the five `0.001` values with the budgets you computed in Task 1, Step 3. Example (your numbers may differ):

```python
PERF_BUDGETS_S: dict[str, float] = {
    "basic_iit3_emd": 3.0,
    "basic_iit4_2023": 4.0,
    "basic_iit4_2026": 4.0,
    "xor_iit4_2026": 8.0,
    "logistic3_k8_iit4_2026": 20.0,
}
```

- [ ] **Step 2: Run the test to verify all five pass**

```bash
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -15
```

Expected: 5 passed. Each in well under its budget. The pytest output also shows per-test duration via `-v`; visually confirm each elapsed wall time is ≤ ~25% of its budget. If a test passes with elapsed ≥ 50% of its budget, the budget is too tight; bump it to the next clean number up and re-run.

### Task 5: Sensitivity check — confirm a doubled wall time trips the gate

**Files:**
- Temporarily modify: `test/test_perf_budget.py` (insert a sleep; revert after)

**Why:** Tasks 3 and 4 confirm the test passes for *current* code. The sensitivity check confirms the gate would *trip* for a regressed-by-2x version of current code, i.e. that the assertion is actually load-bearing.

- [ ] **Step 1: Insert a sleep that approximates a 2x regression for one fixture**

Edit `test/test_perf_budget.py` and add `time.sleep(elapsed)` immediately before the assertion, so the test effectively measures `2 × actual elapsed`:

```python
    elapsed = time.perf_counter() - start
    time.sleep(elapsed)  # temporary sensitivity-check insertion
    elapsed = time.perf_counter() - start
    assert elapsed < budget_s, (
```

Yes, this re-measures `elapsed` after the sleep so the assertion sees the doubled wall time. Otherwise the sleep is invisible to the assertion.

- [ ] **Step 2: Run just the smallest fixture to keep the check fast**

```bash
uv run pytest test/test_perf_budget.py -v -k basic_iit3_emd 2>&1 | tail -15
```

Expected: **1 failure** for `basic_iit3_emd`. The diagnostic shows elapsed ≈ 2× typical (e.g., `took 1.20s, exceeded perf budget 3.0s`) — wait, with a 3s budget and ~0.6s typical, 2x is 1.2s which is still under 3s. The doubled measurement won't trip a 4x budget.

This is a positive finding, not a problem: the 4x budget by design absorbs 2x slowdowns. Continue to Step 3 to make the sensitivity check decisive.

- [ ] **Step 3: Escalate the sensitivity-check insertion to ~5x of typical**

Change the temporary line to:

```python
    elapsed = time.perf_counter() - start
    time.sleep(4 * elapsed)  # temporary sensitivity-check insertion
    elapsed = time.perf_counter() - start
    assert elapsed < budget_s, (
```

This makes the measurement `5 × typical`, which exceeds the 4x budget.

- [ ] **Step 4: Re-run and confirm the test now fails**

```bash
uv run pytest test/test_perf_budget.py -v -k basic_iit3_emd 2>&1 | tail -15
```

Expected: **1 failure** for `basic_iit3_emd`, with the diagnostic showing elapsed ≈ 5 × typical (e.g., `took 3.00s, exceeded perf budget 3.0s` or slightly over).

If the test still passes: either the typical wall time is much lower than expected (rare; sub-0.5s), or the budget is far too loose. Re-check Task 1's calibration.

- [ ] **Step 5: Revert the sensitivity-check insertion**

Remove the `time.sleep(...)` line and the second `elapsed = time.perf_counter() - start`. The function returns to its original form.

- [ ] **Step 6: Re-run to confirm all five pass without the sleep**

```bash
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -15
```

Expected: **5 passed.** Same result as Task 4, Step 2. If anything fails now, the revert was incomplete.

### Task 6: Lint and type-check the new file

**Files:** none modified (verification only)

- [ ] **Step 1: Run ruff on the new file**

```bash
uv run ruff check test/test_perf_budget.py
```

Expected: `All checks passed!` (or no output, meaning clean). If ruff reports issues, fix them in the source (don't bypass).

- [ ] **Step 2: Run ruff format check**

```bash
uv run ruff format --check test/test_perf_budget.py
```

Expected: `1 file already formatted`. If not, run `uv run ruff format test/test_perf_budget.py` to apply, then re-check.

- [ ] **Step 3: Run pyright on the new file**

```bash
uv run pyright test/test_perf_budget.py
```

Expected: `0 errors, 0 warnings, 0 informations`. If pyright reports issues, fix in the source. Common pitfall: missing return type annotation on `test_perf_budget` — the file as written includes `-> None`.

- [ ] **Step 4: Run pyright on the full `pyphi/` to confirm no collateral damage**

```bash
uv run pyright pyphi/ 2>&1 | tail -5
```

Expected: same baseline as `cad8a967` (2 pre-existing errors in `pyphi/visualize/phi_structure/geometry.py`, nothing else). The new test file should not have introduced any pyright noise in `pyphi/` since it doesn't import from anything new.

### Task 7: Measure fast-lane wall-time impact

**Files:** none modified (measurement only)

**Why:** Spec acceptance requires fast lane stays under +20s over baseline. Confirm before commit.

- [ ] **Step 1: Time the fast lane *without* the perf budget tests**

```bash
/usr/bin/time -p uv run pytest test/ -m "not slow and not perf" -q 2>&1 | tail -3
```

Record the `real` wall time. This is the baseline.

- [ ] **Step 2: Time the fast lane *with* the perf budget tests**

```bash
/usr/bin/time -p uv run pytest test/ -m "not slow" -q 2>&1 | tail -3
```

Record this `real` wall time. Subtract baseline. The delta is what `perf` adds.

- [ ] **Step 3: Confirm the delta is within budget**

Expected: delta ≤ 20s. The actual perf-tests-only wall time should be approximately the sum of the medians from Task 1 (typically 8-10s on dev machine). If the delta is over 20s, something else is going on — investigate before committing.

### Task 8: Commit

**Files:**
- Modified: `test/conftest.py`
- Created: `test/test_perf_budget.py`

- [ ] **Step 1: Stage the two intended files**

```bash
git add test/conftest.py test/test_perf_budget.py
```

Avoid `git add -A` or `git add .` — there are untracked detritus files in the repo root (`PLAN.md`, `TODO_*.md`, `test-iit4.ipynb`, etc.) that must not be staged.

- [ ] **Step 2: Verify staged contents**

```bash
git diff --cached --stat
```

Expected:

```
 test/conftest.py        | 4 ++++
 test/test_perf_budget.py | 40 ++++++++++++++++++++++++++++++++++++++++
 2 files changed, 44 insertions(+)
```

Numbers will be close to this (line counts depend on exact whitespace).

- [ ] **Step 3: Commit with calibration table in the message body**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add Tier 1 perf budget: hot-path fixture wall-time gate

Five canary fixtures get a wall-time assertion at max(3.0, 4 * typical):
catches catastrophic regressions (the kind that turned a 0.3s 2026 path
into 84s) before they reach hot-path-touching projects, without being
brittle on slow CI runners.

Typical wall times (median of 3 runs, dev machine):

  basic_iit3_emd:           <fill in> s  -> budget 3.0s
  basic_iit4_2023:          <fill in> s  -> budget <fill in>s
  basic_iit4_2026:          <fill in> s  -> budget <fill in>s
  xor_iit4_2026:            <fill in> s  -> budget <fill in>s
  logistic3_k8_iit4_2026:   <fill in> s  -> budget <fill in>s

Sensitivity check: inserting a sleep that 5x's measured time trips
the gate for basic_iit3_emd as expected.

Fast-lane wall-time impact: +<fill in>s over baseline.
EOF
)"
```

**Before running:** replace each `<fill in>` with the actual numbers from Task 1 and Task 7. The commit will fail (pre-commit hooks) if anything is wrong with the staged files; fix the underlying issue, re-stage, and try again — **do not** use `--no-verify`.

- [ ] **Step 4: Verify the commit landed**

```bash
git log --oneline -2
```

Expected:

```
<new sha> Add Tier 1 perf budget: hot-path fixture wall-time gate
cad8a967 Tier 1 perf budget spec: inline pytest wall-time floors
```

- [ ] **Step 5: Final sanity — run the perf lane one more time**

```bash
uv run pytest -m perf -v 2>&1 | tail -10
```

Expected: **5 passed.** Same as Task 4, Step 2.

---

## Acceptance checklist (mirrors spec)

After all tasks complete, confirm each item from the spec's acceptance section:

- [ ] All 5 perf canaries pass on the dev machine at the pinned budgets.
- [ ] Each elapsed wall time is well under (~25%) its budget — confirmed visually in `-v` output during Task 4, Step 2.
- [ ] Fast-lane delta over baseline is < 20s (Task 7).
- [ ] Sensitivity check trips on a synthetic regression (Task 5, Steps 3–4).
- [ ] `uv run pyright test/test_perf_budget.py` clean (Task 6, Step 3).
- [ ] `uv run ruff check test/test_perf_budget.py` clean (Task 6, Step 1).
- [ ] `uv run ruff format --check test/test_perf_budget.py` clean (Task 6, Step 2).
- [ ] No "Tier 1", "P11.8", "ROADMAP", or planning-artifact references in `test/test_perf_budget.py` source. Confirm:

```bash
grep -iE 'P[0-9]+\b|Tier [0-9]|ROADMAP|Phase [A-Z]\b' test/test_perf_budget.py
```

Expected: zero matches. If a match appears, remove it from the source — the spec is the only place planning context lives.

---

## What does NOT happen in this plan

- No CI workflow file changes
- No `benchmarks/` rewrite
- No baseline tracking / regression-vs-prior-run logic
- No `pytest-benchmark` dependency
- No counter-based stable measurements
- No nightly job
- No push to remote (push requires explicit per-action consent from the user)

All of those live in the follow-on benchmark-suite project.

---

## If something goes wrong

**Calibration runs erroring out:** Likely cache pollution or a side effect from the just-committed renames. Run a single canary in isolation first (`uv run pytest test/test_golden_regression.py -k basic_iit3_emd`) and verify it passes before running the calibration loop.

**A canary is much slower than expected (10s+ for `basic_*`):** Profile with `uv run python -m cProfile -s cumulative -m pytest test/test_golden_regression.py -k <name> | head -30` and surface the bottleneck. Do not just bump the budget to make it green; the point of the gate is to fail loudly on slowdowns.

**Pre-commit hook fails:** Run `uv run ruff check test/test_perf_budget.py` and `uv run pyright test/test_perf_budget.py` directly to see the message. Fix the issue. Re-stage. Try the commit again. **Never** use `--no-verify` (saved memory).

**Sensitivity check still passes after 5x sleep:** Either the budget is way too loose (revisit Task 1 calibration) or the sleep insertion was misplaced (re-read Task 5, Steps 3–4 carefully — the sleep must be between the first `elapsed = ...` and the assertion, AND there must be a second `elapsed = ...` re-measurement after the sleep).
