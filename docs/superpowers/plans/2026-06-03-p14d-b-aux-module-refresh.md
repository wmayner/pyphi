# P14d-B: Auxiliary Module Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead, superseded `graphs.py` (and its `igraph` dependency) and back-fill value-based tests for the living auxiliary modules (`substrate_generator/`, `dynamics.py`, `timescale.py`).

**Architecture:** Tests-and-deletion only — no behavior change to any living code. `graphs.py`'s igraph `maximal_independent_sets` was superseded by the graphillion/`setset` logic in `combinatorics`/`relations`, is unused, and is broken (the `Network→Substrate` rename mangled `from_networkx`→`from_substratex`). The new tests are value-based: analytic truth tables for the unit functions, hand-computed values for parametric functions, exact matrix outputs for timescale, deterministic trajectories for dynamics.

**Tech Stack:** Python 3.12+, NumPy, pytest. Run with `uv run`. No golden/numeric-result impact (these modules don't feed Φ).

**Context (no spec — went straight to plan):** sub-project B of P14d. The other P14d piece (`visualize/` refresh) is sub-project A, separate. `connectivity.py` already has tests and is left alone. A holistic peripheral-API regroup was deliberately deferred to a pre-P15 pass.

---

## Background the engineer needs

- `pyphi/substrate_generator/` builds substrates from a weight matrix + per-node "unit function". `utils.total_weighted_input(element, weights, state) == np.dot(state, weights[:, element])` (sum of `state[i] * weights[i, element]`). `utils.binary2spin` maps `0 → -1`. Logical unit functions return Python `bool`; `build_tpm` writes them into a float array (so `True→1.0`, `False→0.0`).
- The `ising_tpm.npy` fixture's generating config is **not recoverable** (it's only `np.load`-ed in `test/test_tpm.py::test_simulate_tpm`, which tests `dynamics.simulate`, not the generator). So `test_substrate_generator` uses **small hand-computed ising configs**, not that fixture.
- `dynamics.simulate` is already partly covered by `test/test_tpm.py` (the ising stationary-distribution convergence test). `test_dynamics.py` covers the *uncovered* functions and a deterministic `simulate` — it does **not** duplicate the stochastic ising test.
- `graphs.py` is the **only** importer of `igraph`. `networkx` is also needed by `visualize/connectivity.py` and is independently declared in the `visualize` extra, so the whole `graphs` extra (`igraph` + `networkx`) can be deleted.
- Run a single test file: `uv run pytest test/test_x.py -q`. Full sweep incl. doctests: `uv run pytest` (no path argument).

---

## Task 1: Remove the dead `graphs.py` and its `igraph` dependency

**Files:**
- Delete: `pyphi/graphs.py`
- Modify: `pyphi/__init__.py` (the `_skip_import` list)
- Modify: `pyproject.toml` (the `graphs` extra)
- Create: `changelog.d/remove-dead-graphs-module.change.md`

- [ ] **Step 1: Confirm it is truly unused (guard against surprise)**

Run:
```
grep -rn "maximal_independent_sets\|largest_independent_sets\|pyphi.graphs\|from .graphs\|from pyphi import graphs\|import igraph" pyphi/ test/ docs/ --include="*.py" --include="*.rst"
```
Expected: the only matches are inside `pyphi/graphs.py` itself and `pyphi/__init__.py:_skip_import`. If anything else references them, STOP and reassess (it would mean the module is not dead).

- [ ] **Step 2: Delete the module and de-list it**

```bash
git rm pyphi/graphs.py
```
In `pyphi/__init__.py`, change the skip list (currently `_skip_import = ["visualize", "graphs"]`):
```python
_skip_import = ["visualize"]
```

- [ ] **Step 3: Remove the `graphs` extra from `pyproject.toml`**

Delete the line:
```
graphs = ["igraph>=0.9.10", "networkx>=2.6.2"]
```
Then verify nothing else references the `graphs` extra or `igraph`:
```
grep -n "graphs\b\|igraph" pyproject.toml
```
Expected: no remaining `graphs`-extra or `igraph` references. (`networkx` stays declared under the `visualize` extra — do not remove it.)

- [ ] **Step 4: Verify the package still imports and the suite is unaffected**

Run: `uv run python -c "import pyphi; print('ok')"`
Expected: `ok` (no ImportError, no reference to the deleted module).
Run: `uv run pytest test/test_tpm.py -q`
Expected: PASS (sanity that nothing adjacent broke).

- [ ] **Step 5: Changelog fragment**

```bash
cat > changelog.d/remove-dead-graphs-module.change.md <<'EOF'
Removed the unused `pyphi.graphs` module and the `graphs` optional-dependency
extra (`igraph`). Its `maximal_independent_sets` / `largest_independent_sets`
helpers were superseded by the graphillion `setset` logic in
`pyphi.combinatorics` / `pyphi.relations`, had no remaining callers, and were
broken by the 2.0 rename sweep. `networkx` remains available via the
`visualize` extra.
EOF
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/__init__.py pyproject.toml changelog.d/remove-dead-graphs-module.change.md
git -c commit.gpgsign=false commit -m "Remove dead, superseded pyphi.graphs module and igraph extra"
```
If the commit does not land (a hook reformatted a file), re-`git add` the listed files and re-run (no `--no-verify`, no `--amend`).

---

## Task 2: Value-based tests for `substrate_generator/`

**Files:**
- Create: `test/test_substrate_generator.py`

- [ ] **Step 1: Write the unit-function truth-table + value tests**

Create `test/test_substrate_generator.py`:

```python
"""Value-based tests for pyphi.substrate_generator."""

import numpy as np
import pytest

from pyphi import Substrate
from pyphi.substrate_generator import (
    UNIT_FUNCTIONS,
    build_substrate,
    build_tpm,
)
from pyphi.substrate_generator import ising, unit_functions, utils

# 3-node all-to-all weights (no self-loops); element 0 has two inputs (nodes 1, 2),
# each weight 1, so total_weighted_input(0, W, state) == state[1] + state[2].
W3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.parametrize(
    "s1,s2,expected_or,expected_and,expected_parity",
    [
        (0, 0, False, False, False),  # twi = 0
        (1, 0, True, False, True),  # twi = 1
        (0, 1, True, False, True),  # twi = 1
        (1, 1, True, True, False),  # twi = 2 (>=2 inputs; even parity)
    ],
)
def test_logical_unit_functions(s1, s2, expected_or, expected_and, expected_parity):
    state = (0, s1, s2)  # state[0] is irrelevant to element 0 (W[0,0] == 0)
    assert unit_functions.logical_or_function(0, W3, state) == expected_or
    assert unit_functions.logical_and_function(0, W3, state) == expected_and
    assert unit_functions.logical_parity_function(0, W3, state) == expected_parity
    # Negations are the logical complements.
    assert unit_functions.logical_nor_function(0, W3, state) == (not expected_or)
    assert unit_functions.logical_nand_function(0, W3, state) == (not expected_and)
    assert unit_functions.logical_nparity_function(0, W3, state) == (
        not expected_parity
    )


def test_naka_rushton():
    # x = twi**exponent; return x / (x + threshold). exponent=2, threshold=1.
    # state (0,1,1): twi=2 -> x=4 -> 4/5 = 0.8
    assert unit_functions.naka_rushton(0, W3, (0, 1, 1), exponent=2.0, threshold=1.0) == pytest.approx(0.8)
    # state (0,1,0): twi=1 -> x=1 -> 1/2 = 0.5
    assert unit_functions.naka_rushton(0, W3, (0, 1, 0), exponent=2.0, threshold=1.0) == pytest.approx(0.5)


def test_gaussian():
    # gaussian binary2spin's the state first (0 -> -1), then gauss(twi, mu, sigma).
    # state (0,1,1) -> spin (-1,1,1); twi(0) = -1*0 + 1*1 + 1*1 = 2.
    # gauss(2, mu=0, sigma=0.5) = exp(-0.5 * (2/0.5)**2) = exp(-8).
    assert unit_functions.gaussian(0, W3, (0, 1, 1), mu=0.0, sigma=0.5) == pytest.approx(np.exp(-8.0))


def test_ising_energy_and_probability():
    # energy == total_weighted_input on the (already-spin) state.
    assert ising.energy(0, W3, (-1, 1, 1)) == pytest.approx(2.0)
    # probability binary2spin's first: state (0,1,1) -> spin (-1,1,1); E=2;
    # sigmoid(E, T=1, field=0) = 1 / (1 + exp(-2)).
    assert ising.probability(0, W3, (0, 1, 1), temperature=1.0, field=0.0) == pytest.approx(
        1.0 / (1.0 + np.exp(-2.0))
    )


def test_unit_functions_registry_keys():
    assert set(UNIT_FUNCTIONS) == {
        "ising", "boolean", "gaussian", "naka_rushton",
        "or", "and", "parity", "nor", "nand", "nparity",
    }
```

- [ ] **Step 2: Run it**

Run: `uv run pytest test/test_substrate_generator.py -q`
Expected: PASS.

- [ ] **Step 3: Add the `build_tpm` / `build_substrate` tests**

Append to `test/test_substrate_generator.py`:

```python
# 2-node ring W[[0,1],[1,0]]: total_weighted_input(0, state)=state[1];
# total_weighted_input(1, state)=state[0].
W2 = np.array([[0, 1], [1, 0]])


def test_build_tpm_parity_exact():
    tpm = build_tpm("parity", W2)
    # tpm[s0, s1, element]; parity(elem) is True iff its single input == 1.
    expected = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],  # (0,0)->[0,0]  (0,1)->[parity(s1=1)=1, parity(s0=0)=0]
            [[0.0, 1.0], [1.0, 1.0]],  # (1,0)->[0,1]  (1,1)->[1,1]
        ]
    )
    assert tpm.shape == (2, 2, 2)
    assert np.array_equal(tpm, expected)


def test_build_tpm_rejects_non_square_weights():
    with pytest.raises(ValueError, match="square"):
        build_tpm("or", np.array([[0, 1, 1], [1, 0, 1]]))


def test_build_tpm_rejects_mismatched_unit_function_count():
    with pytest.raises(ValueError, match="match"):
        build_tpm(["or", "and", "or"], W2)  # 3 funcs for a 2-node weight matrix


def test_build_substrate_returns_substrate_with_correct_cm():
    sub = build_substrate("parity", W2)
    assert isinstance(sub, Substrate)
    assert np.array_equal(np.asarray(sub.cm), np.array([[0, 1], [1, 0]]))
    assert [str(label) for label in sub.node_labels] == ["A", "B"]
```

- [ ] **Step 4: Run + commit**

Run: `uv run pytest test/test_substrate_generator.py -q`
Expected: PASS.
```bash
git add test/test_substrate_generator.py
git -c commit.gpgsign=false commit -m "Add value-based tests for substrate_generator"
```

---

## Task 3: Value-based tests for `dynamics.py`

**Files:**
- Create: `test/test_dynamics.py`

- [ ] **Step 1: Write the tests**

Create `test/test_dynamics.py`:

```python
"""Value-based tests for pyphi.dynamics (the functions not already covered by
test_tpm.py's ising stationary-distribution test)."""

import numpy as np

from pyphi.dynamics import (
    apply_clamp,
    insert_clamp,
    mean_dynamics,
    number_of_units,
    simulate,
)


def test_apply_clamp():
    # apply_clamp overwrites in place by index (no length change).
    assert apply_clamp({1: 0}, (1, 1, 1)) == (1, 0, 1)
    assert apply_clamp({}, (1, 1)) == (1, 1)  # empty clamp is identity


def test_insert_clamp():
    # insert_clamp inserts the clamped values at their indices (length grows).
    assert insert_clamp({1: 9}, (1, 1)) == (1, 9, 1)
    assert insert_clamp({}, (1, 1)) == (1, 1)  # empty clamp is identity


def test_number_of_units():
    tpm = np.zeros((2, 2, 2))  # state-by-node, 2 binary units
    assert number_of_units(tpm) == 2


def test_simulate_deterministic_tpm():
    # A state-by-node TPM with every entry == 1 sends both units to ON each step,
    # independent of the RNG (P(on) = 1 > any threshold in [0, 1)).
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    path = simulate(tpm, initial_state=(0, 0), timesteps=3, rng=rng)
    assert path == [(0, 0), (1, 1), (1, 1)]


def test_simulate_rejects_wrong_length_initial_state():
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    try:
        simulate(tpm, initial_state=(0, 0, 0), timesteps=2, rng=rng)
    except ValueError as e:
        assert "initial_state" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong-length initial_state")


def test_mean_dynamics_deterministic():
    # All-ones TPM -> every trajectory converges to ON; the per-step mean over all
    # initial states converges to 1 for both units after the first transition.
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    mean = mean_dynamics(tpm, repetitions=2, timesteps=3, rng=rng)
    # mean has shape (timesteps+1, N); steps 1.. are all ON.
    assert np.allclose(mean[1:], 1.0)
```

- [ ] **Step 2: Run to verify (and pin `mean_dynamics`' shape if it differs)**

Run: `uv run pytest test/test_dynamics.py -q`
Expected: PASS. If `mean_dynamics` returns a different array shape than `(timesteps+1, N)`, inspect the actual shape once and adjust the final assertion to index the post-initial steps correctly (the invariant — all ON after step 0 — holds regardless of layout). Do not weaken it to a smoke test.

- [ ] **Step 3: Commit**

```bash
git add test/test_dynamics.py
git -c commit.gpgsign=false commit -m "Add value-based tests for dynamics"
```

---

## Task 4: Value-based tests for `timescale.py`

**Files:**
- Create: `test/test_timescale.py`

- [ ] **Step 1: Write the tests**

Create `test/test_timescale.py`:

```python
"""Value-based tests for pyphi.timescale."""

import numpy as np

from pyphi.timescale import dense_time, run_cm, run_tpm, sparse, sparse_time


def test_sparse_density_threshold():
    # sparse() returns whether (#nonzero / size) > threshold.
    assert sparse(np.array([[1, 0], [0, 1]]), threshold=0.1)  # density 0.5 > 0.1
    assert not sparse(np.array([[1, 0], [0, 0]]), threshold=0.4)  # density 0.25 !> 0.4


def test_dense_time_matrix_power():
    m = np.array([[0, 1], [1, 0]])
    assert np.array_equal(dense_time(m, 2), np.eye(2))  # swap^2 == identity


def test_sparse_time_matches_dense():
    m = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(sparse_time(m, 2), dense_time(m, 2))


def test_run_cm_powers_and_clamps_to_one():
    cm = np.array([[1, 1], [1, 1]])
    # cm^2 = [[2,2],[2,2]]; values > 1 are clamped back to 1.
    assert np.array_equal(run_cm(cm, 2), np.array([[1, 1], [1, 1]]))


def test_run_tpm_one_step_is_identity_roundtrip():
    # A deterministic 2-node state-by-node TPM; running it for 1 step is the
    # convert -> matrix_power(1) -> convert round-trip, which returns the input.
    tpm = np.array(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    assert np.allclose(run_tpm(tpm, 1), tpm)
```

- [ ] **Step 2: Run to verify (pin `sparse`/`run_tpm` exactly)**

Run: `uv run pytest test/test_timescale.py -q`
Expected: PASS. If `run_tpm(tpm, 1)` does not equal `tpm` for the chosen matrix (e.g. the sparse/dense branch alters dtype), pick a permutation-style deterministic TPM whose one-step round-trip is provably the identity and pin that; do not weaken to a smoke test.

- [ ] **Step 3: Commit**

```bash
git add test/test_timescale.py
git -c commit.gpgsign=false commit -m "Add value-based tests for timescale"
```

---

## Task 5: Cruft cleanup, roadmap note, full verification, finish

**Files:**
- Modify: `ROADMAP.md` (P14d entry)
- Local-only: stale `__pycache__` directories

- [ ] **Step 1: Remove stale `__pycache__`-only leftover directories (local only)**

These hold no tracked files (leftovers from the rename sweeps); removal produces no git diff but de-clutters the tree:
```bash
rm -rf pyphi/network_generator pyphi/new_big_phi pyphi/metrics
```
Run: `uv run python -c "import pyphi; print('ok')"`
Expected: `ok` (confirms nothing imported from those paths).

- [ ] **Step 2: Update the P14d roadmap entry**

In `ROADMAP.md`, find the `**P14d.` entry. Add a status note directly under its header recording the sub-project-B outcome:
```markdown
> **Status (2026-06-03), sub-project B (aux hygiene) — done:** Back-filled
> value-based tests for `substrate_generator/`, `dynamics.py`, `timescale.py`.
> Removed the dead, superseded `graphs.py` and the `igraph` extra (its
> `maximal_independent_sets` was replaced by the graphillion `setset` logic in
> `combinatorics`/`relations`). The "dynamics.py / ising overlap" this entry
> describes does not exist (`dynamics.py` has no ising code) — nothing to
> consolidate. The holistic peripheral-API regroup is deferred to a dedicated
> pre-P15 pass. **Remaining for P14d:** sub-project A (`visualize/` refresh).
```

- [ ] **Step 3: Full verification**

Run (no path argument, so doctests are collected): `uv run pytest -q`
Expected: 0 failures.
Run: `uv run pyright pyphi` (expect 0 errors) and `uv run ruff check pyphi test` (expect clean).

- [ ] **Step 4: Commit the roadmap note**

```bash
git add ROADMAP.md
git -c commit.gpgsign=false commit -m "Mark P14d sub-project B (aux hygiene) done in roadmap"
```

- [ ] **Step 5: Finish**

Use superpowers:finishing-a-development-branch. If executing inline on `2.0` (no worktree), this collapses to the verification above — report completion. Otherwise merge `--ff-only` into `2.0`, remove the worktree, delete the branch. Do not push without explicit consent.

---

## Notes for the implementer

- **No source behavior changes** beyond deleting dead `graphs.py`. The four living-module test files only *characterize* existing behavior; if a value test fails, first check the hand-computed expectation, then suspect (and investigate) a genuine post-rename bug — do not silently relax the assertion.
- Value assertions are analytic (truth tables, matrix algebra) or hand-derived from the documented formulas, not captured from the code under test.
- `connectivity.py` is core (9 internal importers) and already tested — out of scope.
- Sub-project A (`visualize/` refresh) and the holistic peripheral-API regroup are separate, later work.
