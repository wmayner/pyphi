# Precision-Aware Structural Equality Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pyphi.models.cmp.numpy_aware_eq` precision-aware so structural-equality on IIT result objects absorbs float64 op-order drift (~1e-15) while still detecting real math regressions (≥1e-3); share the same tolerance constant with `test/test_golden_regression.py`.

**Architecture:** Add a module-level `EQUALITY_TOLERANCE = 1e-13` constant in `pyphi/models/cmp.py`. Replace `np.array_equal` (array branch) with `np.allclose(rtol=ET, atol=ET)` and add a `math.isclose(rel_tol=ET, abs_tol=ET)` branch for float scalars. Wrap `np.allclose` in `try/except (ValueError, TypeError)` for shape / non-numeric robustness. `general_eq`'s phi/alpha (via `utils.eq`) and mechanism/purview (set equality) special-cases are unchanged. `test_golden_regression.py` imports `EQUALITY_TOLERANCE` so the two consumers (production `__eq__` and golden-fixture comparison) stay locked together.

**Tech Stack:** Python 3.12+, NumPy, `math.isclose`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-26-precision-aware-comparator-design.md` (committed at `0650fe2d`; three-vs-two thresholds correction at `418fc3eb`).

**Branch context:** Working in worktree `/Users/will/projects/pyphi-p12b` on `feature/p12b-factored-kary`, head `418fc3eb`. Local-only, not pushed. Main repo `/Users/will/projects/pyphi` (on `2.0`) MUST NOT be touched. The worktree has unrelated unstaged churn (`uv.lock`, `filename`) that MUST NOT be staged.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyphi/models/cmp.py` | Modify | Add `EQUALITY_TOLERANCE` constant; rewrite `numpy_aware_eq` to use `np.allclose` and `math.isclose`. |
| `test/test_models.py` | Modify | Add new unit tests for the precision-aware behavior alongside the four existing `test_numpy_aware_eq_*` tests. |
| `test/test_golden_regression.py` | Modify | Replace hardcoded `RTOL = ATOL = 1e-12` with `from pyphi.models.cmp import EQUALITY_TOLERANCE`. |
| `ROADMAP.md` | Modify | Remove the now-implemented "Precision-aware comparator for `intrinsic_information`" entry (lines ~2617-2630); add the post-IIT-3.0 `config.numerics.precision` retirement entry per spec. |
| `changelog.d/precision-aware-comparator.feature.md` | Create | Single user-facing line describing the effect. |

Single commit at end. ~30 lines net change across the five files.

---

## Task 1: Add precision-aware behavior to `numpy_aware_eq` (TDD)

**Files:**
- Modify: `pyphi/models/cmp.py`
- Modify: `test/test_models.py` (new tests appended near the existing `test_numpy_aware_eq_*` block at lines 194-211)

The change has two parts: (a) introduce a `EQUALITY_TOLERANCE = 1e-13` constant, (b) replace the array `np.array_equal` and the scalar `a == b` branches with tolerance-aware equivalents while preserving every other behavior (iterable recursion, `==` fallback, NaN ≠ NaN, length-mismatch → False, shape-mismatch → False without exception leak).

The four existing tests (`test_numpy_aware_eq_noniterable`, `test_numpy_aware_eq_nparray`, `test_numpy_aware_eq_tuple_nparrays`, `test_numpy_aware_eq_identical`) all use differences ≥1 or shape / type mismatches, so they MUST continue to pass — never weaken them.

- [ ] **Step 1.1: Write failing tests for the new precision-aware behavior**

Append the following tests to `test/test_models.py` right after `test_numpy_aware_eq_identical` (line 211). The tests target the new behavior — they will fail under the current strict implementation:

```python
def test_equality_tolerance_constant_exists():
    """``EQUALITY_TOLERANCE`` is exposed at module level and equals 1e-13."""
    from pyphi.models.cmp import EQUALITY_TOLERANCE

    assert EQUALITY_TOLERANCE == 1e-13


def test_numpy_aware_eq_float_within_tolerance():
    """Float scalars differing by ~1e-15 (op-order noise) compare equal."""
    assert models.cmp.numpy_aware_eq(1.0, 1.0 + 1e-15)


def test_numpy_aware_eq_float_outside_tolerance():
    """Float scalars differing by a real-regression-scale amount compare unequal."""
    assert not models.cmp.numpy_aware_eq(1.0, 1.001)


def test_numpy_aware_eq_array_within_tolerance():
    """Arrays differing by ~1e-15 (op-order noise) compare equal."""
    a_ = np.ones(3)
    b_ = np.ones(3) + 1e-15
    assert models.cmp.numpy_aware_eq(a_, b_)


def test_numpy_aware_eq_array_outside_tolerance():
    """Arrays differing by a real-regression-scale amount compare unequal."""
    a_ = np.zeros(3)
    b_ = np.ones(3)
    assert not models.cmp.numpy_aware_eq(a_, b_)


def test_numpy_aware_eq_array_shape_mismatch_returns_false():
    """Shape mismatch on arrays must return False, not raise."""
    a_ = np.zeros(3)
    b_ = np.zeros(4)
    assert not models.cmp.numpy_aware_eq(a_, b_)


def test_numpy_aware_eq_nan_scalar_not_equal():
    """NaN ≠ NaN preserved (math.isclose default behavior)."""
    assert not models.cmp.numpy_aware_eq(float("nan"), float("nan"))


def test_numpy_aware_eq_nan_array_not_equal():
    """NaN array ≠ NaN array preserved (np.allclose equal_nan=False default)."""
    a_ = np.array([np.nan])
    b_ = np.array([np.nan])
    assert not models.cmp.numpy_aware_eq(a_, b_)
```

- [ ] **Step 1.2: Run new tests to verify they fail**

Run:
```bash
uv run pytest test/test_models.py -k "equality_tolerance_constant_exists or numpy_aware_eq_float or numpy_aware_eq_array or numpy_aware_eq_nan" -v
```

Expected:
- `test_equality_tolerance_constant_exists` → FAIL (`ImportError`: `EQUALITY_TOLERANCE` not exported)
- `test_numpy_aware_eq_float_within_tolerance` → FAIL (`1.0 == 1.0 + 1e-15` is False under strict ==)
- `test_numpy_aware_eq_array_within_tolerance` → FAIL (`np.array_equal` is strict)
- `test_numpy_aware_eq_array_shape_mismatch_returns_false` → may PASS already (strict array_equal returns False on shape mismatch) — that's fine; it must keep passing post-change
- The other tests (`*_outside_tolerance`, `*_nan_*`) may pass or fail depending on current behavior; the load-bearing assertions are that they pass post-change

If `test_equality_tolerance_constant_exists` errors out at collection (ImportError leaking from the test module), that counts as failing — the implementation hasn't run yet.

- [ ] **Step 1.3: Update `pyphi/models/cmp.py`**

The current file has the existing `numpy_aware_eq` at lines 108-124 and uses `Iterable` from `collections.abc` (line 6), `Any` from `typing` (line 8), `np` (line 12). Add `math` to the imports if not present, and `EQUALITY_TOLERANCE` constant just above the `numpy_aware_eq` definition. Replace the body.

Modify `pyphi/models/cmp.py`:

Edit 1 — add `import math` to the standard-library imports near the top of the file. The current top reads:
```python
import functools
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import ClassVar
from typing import TypeVar

import numpy as np
```

Replace with (insert `import math` after `import functools`):
```python
import functools
import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
from typing import ClassVar
from typing import TypeVar

import numpy as np
```

Edit 2 — replace the current `numpy_aware_eq` block (lines 108-124):

```python
# TODO use builtin numpy methods here
def numpy_aware_eq(a: Any, b: Any) -> bool:
    """Return whether two objects are equal via recursion, using
    :func:`numpy.array_equal` for comparing numpy arays.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    # TODO(4.0) this is broken if the iterables are sets
    if (
        (isinstance(a, Iterable) and isinstance(b, Iterable))
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):  # type: ignore[arg-type]
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b, strict=False))
    return a == b
```

With:

```python
EQUALITY_TOLERANCE = 1e-13
"""Tolerance for structural equality on IIT quantities. Absorbs op-order
drift in float64 arithmetic on IIT measures while distinguishing real
math regressions. Used by `numpy_aware_eq` (model `__eq__`) and by
golden-fixture comparisons in the test suite. Independent of
`config.numerics.precision`, which governs user-configurable phi
comparison via `utils.eq`."""


def numpy_aware_eq(a: Any, b: Any) -> bool:
    """Return whether two objects are equal via recursion, with float
    leaves compared up to ``EQUALITY_TOLERANCE``.

    Arrays compare via :func:`numpy.allclose`; float scalars via
    :func:`math.isclose`; other types via ``==``. Shape-mismatched or
    non-numeric arrays compare unequal rather than raising.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.allclose(
                a, b, rtol=EQUALITY_TOLERANCE, atol=EQUALITY_TOLERANCE
            )
        except (ValueError, TypeError):
            return False
    # TODO(4.0) this is broken if the iterables are sets
    if (
        (isinstance(a, Iterable) and isinstance(b, Iterable))
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):  # type: ignore[arg-type]
            return False
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return math.isclose(
            float(a), float(b),
            rel_tol=EQUALITY_TOLERANCE,
            abs_tol=EQUALITY_TOLERANCE,
        )
    return a == b
```

Notes:
- The `# TODO(4.0) this is broken if the iterables are sets` comment is preserved — pre-existing, unrelated to this work.
- The `# TODO use builtin numpy methods here` comment at the top of the original function is *removed* (the new implementation uses `np.allclose` — the TODO is satisfied).
- No design narrative in the docstring (per saved memory `feedback_no_design_narrative_in_docstrings`); the docstring describes what the function IS and DOES.

- [ ] **Step 1.4: Run tests to verify the new and existing ones all pass**

Run:
```bash
uv run pytest test/test_models.py -v
```

Expected: all `test_numpy_aware_eq_*` tests pass (the 4 existing + 8 new = 12 numpy_aware_eq tests), plus the rest of `test_models.py` unchanged. Anywhere from ~30-50 tests in this file; what matters is 0 failures.

- [ ] **Step 1.5: Pyright + ruff on touched files**

Run:
```bash
uv run pyright pyphi/models/cmp.py test/test_models.py
uv run ruff check pyphi/models/cmp.py test/test_models.py
uv run ruff format --check pyphi/models/cmp.py test/test_models.py
```

Expected: 0 errors / 0 warnings; clean.

---

## Task 2: Wire `test/test_golden_regression.py` to the shared constant

**Files:**
- Modify: `test/test_golden_regression.py` (lines 31-45 — the docstring-comment block + `RTOL`/`ATOL` definitions)

The current code has explanatory comments at lines 31-43 about why `1e-12` was chosen, plus the hardcoded constants at lines 44-45. The hardcoded constants get replaced with an import; the explanatory comments should be rewritten to describe the *final state* (the import + what the constant means) rather than the pre-change rationale.

- [ ] **Step 2.1: Replace the `RTOL`/`ATOL` block**

Edit `test/test_golden_regression.py`. The current block (lines 31-45):

```python
# Tolerances for array comparisons. config.numerics.precision default is 13, so
# 1e-12 is one digit looser than the equality threshold — strict enough to
# catch any meaningful numerical drift while absorbing minor LAPACK / BLAS
# variations across platforms.
#
# Platform sensitivity note: fixtures committed to the repository were
# generated on macOS aarch64 (Apple Silicon) with Python 3.13.13, NumPy 2.4,
# SciPy 1.17, and pyemd 2.0. Linux x86_64 may produce drift below 1e-12 due
# to BLAS implementation differences (Accelerate vs OpenBLAS) and EMD
# library backend variations. If CI fails on Linux/Windows with sub-1e-10
# differences and the structural fields (partitions, mechanisms, distinction
# counts) match exactly, raise these tolerances to 1e-10 and document.
# Differences above 1e-10 should be investigated as potential bugs.
RTOL = 1e-12
ATOL = 1e-12
```

Replace with:

```python
# Tolerances for array comparisons. Shared with ``pyphi.models.cmp``'s
# ``numpy_aware_eq`` (the production-side structural-equality comparator)
# so the test-fixture comparator and the model-level ``__eq__`` ask the
# same question: did this value drift by more than float64 op-order noise?
#
# Platform sensitivity note: fixtures committed to the repository were
# generated on macOS aarch64 (Apple Silicon) with Python 3.13.13, NumPy
# 2.4, SciPy 1.17, and pyemd 2.0. Linux x86_64 may produce drift below
# 1e-13 due to BLAS implementation differences (Accelerate vs OpenBLAS)
# and EMD library backend variations. If CI fails on Linux/Windows with
# sub-1e-10 differences and the structural fields (partitions, mechanisms,
# distinction counts) match exactly, raise the constant in ``cmp.py`` and
# document. Differences above 1e-10 should be investigated as potential
# bugs.
RTOL = EQUALITY_TOLERANCE
ATOL = EQUALITY_TOLERANCE
```

Edit the import block at the top of the file (currently lines 16-29) to add the import. Find the existing imports section ending at line 29 (`from .golden.fixture import store_fixture`); add the import after the `pyphi`-relative imports. Insert:

```python
from pyphi.models.cmp import EQUALITY_TOLERANCE
```

Place it between the third-party imports (numpy, pytest at lines 20-21) and the `.golden.*` imports (lines 23-29). Specifically: after `import pytest` and before `from .golden import ALL_FIXTURES`, with a blank line separating the groups per existing style.

- [ ] **Step 2.2: Run the golden regression suite**

Run:
```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 25/25 byte-identical (23 binary preserved + 2 k-ary: `multivalued_k3_tiny_iit4_2023`, `multivalued_2x3x3_iit4_2023`). Tightening from `1e-12` to `1e-13` on byte-identical comparisons cannot break anything.

If any golden fails: STOP. Tightening should be a no-op on byte-identical fixtures. A failure indicates either (a) a fixture had hidden non-byte-identical drift between `1e-13` and `1e-12` (silent regression to investigate; see `feedback_dont_give_up_on_architectural_refactors`) or (b) a subtle bug in the swap. Diagnose before proceeding.

- [ ] **Step 2.3: Pyright + ruff on `test_golden_regression.py`**

Run:
```bash
uv run pyright test/test_golden_regression.py
uv run ruff check test/test_golden_regression.py
uv run ruff format --check test/test_golden_regression.py
```

Expected: 0 errors / 0 warnings; clean.

---

## Task 3: ROADMAP swap + changelog fragment

**Files:**
- Modify: `ROADMAP.md` (lines ~2617-2630 — the existing comparator entry)
- Create: `changelog.d/precision-aware-comparator.feature.md`

The existing ROADMAP entry titled "Precision-aware comparator for ``intrinsic_information`` in test fixtures" (lines 2617-2630) describes the work being implemented in this commit. Per the spec, replace it with the post-IIT-3.0 retirement entry for `config.numerics.precision`. (The old entry also referenced the wrong file — `pyphi/jsonify.py` instead of `pyphi/models/cmp.py` — so the swap is doubly justified.)

- [ ] **Step 3.1: Replace the ROADMAP entry**

Edit `ROADMAP.md`. The current block at lines 2617-2630:

```markdown
- **Precision-aware comparator for ``intrinsic_information`` in test
  fixtures.** The test helpers ``numpy_aware_eq`` / ``general_eq`` in
  ``pyphi/jsonify.py`` do strict ``==`` comparison on
  ``intrinsic_information`` dict values, which is brittle: op-order
  changes that produce drift well below ``config.numerics.precision``
  (1e-13) trip the comparison and force fixture regeneration. The
  ``test_iit4.py`` fixtures (``basic_noisy_selfloop.json``,
  ``grid3.json``) have been regenerated five times across the 2.0
  refactor on this pattern. Fix: replace strict ``==`` with
  ``np.allclose(..., atol=1e-13)`` on intrinsic_information dict
  values in the comparator, with care that
  ``test_golden_regression.py`` (which uses the same helper) doesn't
  loosen unintentionally. Probably half a day; audit
  ``numpy_aware_eq`` callers carefully first.
```

Replace with:

```markdown
- **Retire ``config.numerics.precision`` when IIT 3.0 is dropped.** The
  setting is a holdover from the EMD-era ``pyemd`` C-library noise.
  Once 3.0 support is gone, ``utils.eq`` should migrate to use a fixed
  module-level constant (or import ``cmp.EQUALITY_TOLERANCE`` directly),
  and the config field should be removed. For IIT 4.0 the
  "op-order-noise threshold" is a property of float64 arithmetic on
  IIT quantities, not a user preference.
```

- [ ] **Step 3.2: Create the changelog fragment**

Create `changelog.d/precision-aware-comparator.feature.md` with the following content:

```
Structural equality on IIT result objects (``SystemIrreducibilityAnalysis``, ``CauseEffectStructure``, ``AcSystemIrreducibilityAnalysis``, ``RepertoireIrreducibilityAnalysis``, ``StateSpecification``, distinctions) is now precision-aware up to ``EQUALITY_TOLERANCE = 1e-13`` (exported from ``pyphi.models.cmp``). The comparator absorbs float64 op-order drift while distinguishing real math regressions. ``test/test_golden_regression.py`` shares the constant, so production ``__eq__`` and golden-fixture comparisons stay locked together.
```

- [ ] **Step 3.3: Verify towncrier accepts the fragment**

Run:
```bash
uv run towncrier check --compare-with origin/2.0 || uv run towncrier check
```

Expected: passes (the new fragment satisfies the towncrier-check pre-commit hook). If neither base ref is available, the second invocation falls back to a content check.

---

## Task 4: Full verification + single commit

**Files:** all five from prior tasks (no new modifications)

This task runs the full acceptance gate suite, confirms the staged set is exactly the intended five files, and lands the single commit.

- [ ] **Step 4.1: Full pytest run (no path argument — picks up `pyphi/` doctests)**

Per the doctest-scope note in `CLAUDE.md`: at the commit boundary, run `uv run pytest` with NO path argument. This uses `testpaths = ["pyphi", "test"]` and `--doctest-modules` per `pyproject.toml`, so it sweeps `pyphi/` source doctests.

Run:
```bash
uv run pytest --tb=short -q
```

Expected: 0 failures. Baseline ~1429 tests passing pre-change; new tests bring the count slightly higher.

If failures appear, diagnose per saved memory `feedback_dont_give_up_on_architectural_refactors` before reverting — a test silently flipping from fail to pass under the loosened comparator is itself a concern; investigate whether the fixture is masking real drift before accepting it.

- [ ] **Step 4.2: Pyright over the entire package**

Run:
```bash
uv run pyright pyphi
```

Expected: 0 errors / 0 warnings (matches the baseline at `418fc3eb`).

- [ ] **Step 4.3: Ruff check + format over touched files**

Run:
```bash
uv run ruff check pyphi test
uv run ruff format --check pyphi test
```

Expected: clean.

- [ ] **Step 4.4: Stage the five intended files (targeted, never `git add .`)**

Run, from `/Users/will/projects/pyphi-p12b`:
```bash
git add \
  pyphi/models/cmp.py \
  test/test_models.py \
  test/test_golden_regression.py \
  ROADMAP.md \
  changelog.d/precision-aware-comparator.feature.md
```

The worktree has unrelated unstaged churn (`uv.lock`, `filename`) and untracked items that MUST NOT be staged.

- [ ] **Step 4.5: Confirm the staged set is exactly the five intended files**

Run:
```bash
git diff --cached --stat
```

Expected output shape:
```
 ROADMAP.md                                              | 16 ++++++----------
 changelog.d/precision-aware-comparator.feature.md       |  1 +
 pyphi/models/cmp.py                                     | 30 ++++++++++++++++++++++++------
 test/test_golden_regression.py                          | 22 +++++++++-------------
 test/test_models.py                                     | 50 ++++++++++++++++++++++++++++++++++++++++++++++++++
 5 files changed, ~119 insertions(+), ~30 deletions(-)
```

(Numbers are approximate; what matters is exactly 5 files, all from the expected list, no `uv.lock` / `filename` / other churn.)

- [ ] **Step 4.6: Commit**

Per saved memory `feedback_no_verify_bypass`: NEVER bypass pre-commit hooks. The hooks (ruff + ruff format + pyright + towncrier-check) gate this commit. If a hook fails, diagnose by running `uv run ruff check <file>` / `uv run pyright <file>` directly and fix the root cause.

Per session pattern, gpgsign is bypassed via `-c commit.gpgsign=false`:

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Make structural equality precision-aware via shared EQUALITY_TOLERANCE

Replaces strict np.array_equal / == in numpy_aware_eq with np.allclose
and math.isclose at 1e-13 tolerance, and shares the constant with
test_golden_regression.py so production __eq__ and golden-fixture
comparison stay locked together. Absorbs float64 op-order drift on IIT
quantities while still catching real math regressions (≥1e-3). Per the
spec.
EOF
)"
```

If 1Password agent / gpgsign fails persistently despite the bypass, surface to the user — do NOT silently retry without consent (saved memory `feedback_no_verify_bypass`).

If a pre-commit hook fails *and auto-fixes files in place*: re-run `git add -u` on **only the originally-staged files** (`pyphi/models/cmp.py`, `test/test_models.py`, `test/test_golden_regression.py`, `ROADMAP.md`, `changelog.d/precision-aware-comparator.feature.md`). Never `git add -u` broadly — the hook may have touched files outside the intended set.

- [ ] **Step 4.7: Confirm the commit landed**

Run:
```bash
git show --stat HEAD
```

Expected: HEAD is the new commit; 5 files changed; matches Step 4.5's preview.

- [ ] **Step 4.8: Confirm goldens still byte-identical post-commit**

Run:
```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 25/25 pass (sanity check; should be unchanged from Step 2.2).

---

## Verification (consolidated)

After Task 4 lands, all of the following MUST be green:

```bash
# Spec acceptance gates §'Acceptance criteria'
uv run pytest --tb=short -q                          # 0 failures (full suite incl. doctests)
uv run pytest test/test_golden_regression.py -v      # 25/25 byte-identical
uv run pytest test/test_models.py -v                 # all numpy_aware_eq tests (4 existing + 8 new)
uv run pyright pyphi                                 # 0 errors / 0 warnings
uv run ruff check pyphi test                         # clean
uv run ruff format --check pyphi test                # clean

# Sanity smoke
uv run python -c "from pyphi.models.cmp import EQUALITY_TOLERANCE; print(EQUALITY_TOLERANCE)"
# expected: 1e-13

uv run python -c "
from pyphi.models.cmp import numpy_aware_eq
import numpy as np
# op-order noise → equal
assert numpy_aware_eq(1.0, 1.0 + 1e-15)
# real regression → not equal
assert not numpy_aware_eq(1.0, 1.001)
# shape mismatch → False, not exception
assert not numpy_aware_eq(np.zeros(3), np.zeros(4))
# NaN ≠ NaN preserved
assert not numpy_aware_eq(float('nan'), float('nan'))
print('comparator OK')
"
# expected: 'comparator OK'
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| A previously-passing test silently starts passing when it should fail (loosened comparator masks a real regression) | Low | Existing `test_numpy_aware_eq_*` tests use differences ≥1 — far above any tolerance; safe by construction. Risk is on un-audited tests. Mitigation: full suite run before and after; if any test flips from fail→pass, diagnose per `feedback_dont_give_up_on_architectural_refactors`. |
| Tightening 1e-12 → 1e-13 breaks a golden | Vanishingly low | All 25 current goldens are byte-identical (per HEAD at `418fc3eb`); tightening cannot break byte-identical passes. If it does, STOP and diagnose — implies a hidden non-byte-identical comparison was passing under 1e-12. |
| `np.allclose` raises on namedtuple input (test_numpy_aware_eq_nparray) | Low | The `try/except (ValueError, TypeError)` guard returns False. The existing test passes a namedtuple vs ndarray; existing test continues to assert False (correct). |
| Pre-commit hook auto-rewrite leaks unrelated files into the commit | Low-Medium | Step 4.5 verifies the staged set before commit; the recovery path in Step 4.6 only re-adds the five intended files. |
| `np.allclose(rtol=1e-13, atol=1e-13)` on float32 arrays would reject ULP-scale drift | N/A | PyPhi works in float64 throughout; no float32 arrays in result objects. If a future regression introduces float32, the tolerance becomes the natural place to widen. |

---

## Final Acceptance Gates

Per spec §'Acceptance criteria':

1. ✅ `numpy_aware_eq` precision-aware: arrays use `np.allclose(rtol=ET, atol=ET)`; float scalars use `math.isclose(rel_tol=ET, abs_tol=ET)`.
2. ✅ `test_golden_regression.py` imports `EQUALITY_TOLERANCE` from `pyphi.models.cmp`; hardcoded `RTOL`/`ATOL = 1e-12` removed.
3. ✅ Goldens 25/25 byte-identical.
4. ✅ Full `uv run pytest` (no path; includes doctests) passes.
5. ✅ Pyright 0 errors / 0 warnings.
6. ✅ Ruff check + format clean.
7. ✅ `test_models.py` existing 4 `numpy_aware_eq` tests still pass; 8 new tests pass.
8. ✅ ROADMAP has the post-3.0 `config.numerics.precision` retirement entry.
9. ✅ `changelog.d/precision-aware-comparator.feature.md` exists with the user-visible-effect description.
10. ✅ `general_eq` shape unchanged (phi/alpha via `utils.eq`; mechanism/purview via set equality).

---

## Self-Review

**Spec coverage check:** Every requirement in spec §'Acceptance criteria' (1-8) is covered by a task step. The deprecation of the old ROADMAP item is plan-side cleanup beyond the spec; it's correct to do here because the old item describes the work being implemented, and leaving it would be stale.

**Placeholder scan:** No "TBD", "TODO" inline in task steps (the preserved `# TODO(4.0)` comment in `numpy_aware_eq` is pre-existing, unrelated). All code blocks are complete and copy-pastable. No "implement appropriate error handling" — the `try/except (ValueError, TypeError)` is specified explicitly.

**Type consistency:** `EQUALITY_TOLERANCE` is the same name everywhere (cmp.py constant, test imports, ROADMAP reference, changelog fragment). `numpy_aware_eq` signature unchanged. `RTOL`/`ATOL` names preserved in `test_golden_regression.py` (downstream `_compare` helper code at lines 144 and 158 references them by these names — no rename needed).

**Constraint compliance:** No P# / Phase / Task markers in source files or changelog (commit message references "the spec" — allowed). Docstrings describe final state. No back-compat shims. Targeted `git add` only. Pre-commit hooks not bypassed. Push not initiated.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-precision-aware-comparator-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task, two-stage review (spec compliance + code quality) per task. Model: Sonnet 4.6 for the implementer (mechanical TDD on small surface); Opus 4.7 for both reviewers.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`; batch execution with checkpoints.

After implementation lands, branch state housekeeping for `feature/p12b-factored-kary` per `superpowers:finishing-a-development-branch` 4-option menu (the branch carries all P12b work + 4 cleanups + the spec + this implementation; user's call whether to merge to `2.0`, push, keep local, or discard). Then natural follow-up projects per ROADMAP: (a) `test_complexes.py::test_possible_complexes` pre-existing failure investigation (small focused); (b) AC k-ary cutover (medium follow-up, mirrors System cause/effect migration pattern).

Which approach?
