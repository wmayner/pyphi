# Deterministic SIA Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Substrate.sia()` deterministic across runs by adding a structural lex tie-break on partitions tied at the MIP minimisation key, closing the two currently-xfailed `big_subsys_all_complete` tests.

**Architecture:** Add a `lex_key()` method on `_PartitionBase` returning a canonical byte representation of the induced edge cut. Register a `PARTITION_LEX` strategy in `pyphi/resolve_ties.py` and a `sias()` resolver wrapping the existing `resolve()` machinery. Add a `sia_tie_resolution` config key. Replace the manual MIP-finding loop in IIT 4.0 with a call through `resolve_ties.sias`. Extend `OrderableByPhi.order_by` on the IIT 3.0 SIA subclass so `min()` over OrderableByPhi inherits the structural fallback.

**Tech Stack:** Python 3.12+, numpy, pytest, ruff, pyright, uv. PyPhi codebase conventions per `CLAUDE.md`.

---

## Branch baseline (verify before starting)

- Branch: `2.0` head `0941b62a` (or later — verify with `git log --oneline -1`).
- Working in the main repo at `../pyphi`. No worktree.
- Approximately 9 unstaged tracked-file changes from earlier sessions exist. **Do not** stage or touch any file your task doesn't explicitly modify. Use targeted `git add <paths>` only.
- Verify gates green pre-start:

```bash
uv run pytest test/test_resolve_ties.py test/test_big_phi.py -m "not slow" -q   # baseline
uv run pyright pyphi/                                                            # 0 errors / 1 baseline warning
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

## Constraints (load-bearing)

- **Never bypass pre-commit hooks.** No `--no-verify`, no `SKIP=pyright`. If a hook fails, fix the underlying issue.
- **Do not push.** Pushing requires explicit per-action consent from the user.
- **No P# markers / "Phase A" / `TODO(Px)` / ROADMAP-project IDs** in source, comments, docstrings, or changelog. Commit messages may reference "the spec" since that points at `docs/superpowers/specs/2026-05-12-sia-tie-breaking-design.md`.
- **No design-decision narrative in docstrings.** Don't enumerate constructors, explain naming choices, compare to alternatives, or foreshadow extensions. Describe what the code IS and DOES.
- **Docstrings describe final state**, not migration journey.
- **No back-compat shims for unpushed dev work.**
- **gpgsign:** the spec commit signed cleanly. If signing fails mid-plan, stop and ask the user for per-action consent — do not bypass.

## File responsibilities (touch list)

**New files:**
- `changelog.d/sia-determinism.fix.md` — single user-facing changelog fragment.

**Modified — production:**
- `pyphi/models/partitions.py` — add `lex_key()` method on `_PartitionBase`.
- `pyphi/resolve_ties.py` — register `PARTITION_LEX` strategy and add `sias()` resolver.
- `pyphi/conf/formalism.py` — add `sia_tie_resolution: list[str]` field on `IITConfig`.
- `pyphi/formalism/iit4/__init__.py` — replace manual MIP loop (lines 718-735) with `resolve_ties.sias`.
- `pyphi/models/sia.py` — override `order_by` on the IIT 3.0 `SystemIrreducibilityAnalysis` subclass.

**Modified — tests:**
- `test/test_resolve_ties.py` — add `lex_key` and `sias()` unit tests.
- `test/test_big_phi.py` — drop the two `_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` markers.
- `test/test_invariants.py` — add `test_sia_is_deterministic_across_runs` property test.

**Possibly modified — fixtures:**
- `test/data/sia/big_subsys_all_complete.json` — regenerate via `test/IIT_4.0_make_jsons.ipynb` if the new deterministic MIP differs from what was captured.

---

## Task 1: Infrastructure — partition lex key, `sias()` resolver, config field

**Files:**
- Modify: `pyphi/models/partitions.py:67-89` (add `lex_key` method on `_PartitionBase`)
- Modify: `pyphi/resolve_ties.py` (add `PARTITION_LEX` registration + `sias()` function after the existing `purviews()` function)
- Modify: `pyphi/conf/formalism.py:53-57` (add `sia_tie_resolution` field on `IITConfig`)
- Test: `test/test_resolve_ties.py` (extend with `lex_key` and `sias()` cases)

### Steps

- [ ] **Step 1.1: Write failing tests for `lex_key()` on `_PartitionBase`**

Append to `test/test_resolve_ties.py`:

```python
import numpy as np

from pyphi.direction import Direction
from pyphi.models.partitions import (
    CompleteEdgeCut,
    DirectedBipartition,
    DirectedSetPartition,
    EdgeCut,
    NullCut,
)


def test_lex_key_nullcut_is_empty_bytes():
    null = NullCut(indices=(0, 1, 2))
    assert null.lex_key() == b""


def test_lex_key_directed_bipartition_matches_cut_matrix_bytes():
    sp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    expected = sp.cut_matrix(3).tobytes()
    assert sp.lex_key() == expected


def test_lex_key_equivalent_partitions_compare_equal():
    """Two partitions inducing the same edge cut compare equal under lex_key.

    A 3-node EFFECT bipartition severing 1→2 has cut_matrix
    [[0,0,0],[0,0,1],[0,0,0]]; an EdgeCut with the same matrix on the
    same node set must produce the same lex_key.
    """
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    matrix = bp.cut_matrix(3)
    ec = EdgeCut(node_indices=(0, 1, 2), cut_matrix=matrix)
    assert bp.lex_key() == ec.lex_key()


def test_lex_key_distinct_partitions_compare_distinct():
    sp_a = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sp_b = DirectedBipartition(Direction.EFFECT, (0,), (2,))
    assert sp_a.lex_key() != sp_b.lex_key()


def test_lex_key_is_total_ordering():
    """Distinct partitions sort under bytes comparison."""
    sp_a = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sp_b = DirectedBipartition(Direction.EFFECT, (0,), (2,))
    assert sp_a.lex_key() < sp_b.lex_key() or sp_b.lex_key() < sp_a.lex_key()
```

- [ ] **Step 1.2: Run lex_key tests to verify they fail**

```bash
uv run pytest test/test_resolve_ties.py -k "lex_key" -v
```

Expected: 5 failures with `AttributeError: ... has no attribute 'lex_key'`.

- [ ] **Step 1.3: Implement `lex_key()` on `_PartitionBase`**

Edit `pyphi/models/partitions.py` — add method to `_PartitionBase` (after `all_cut_mechanisms` at line 114-118, before `class NullCut` at line 121):

```python
    def lex_key(self) -> bytes:
        """Canonical sortable bytes representation of the induced edge cut.

        Two partitions producing the same edge cut on the same node set
        sort identically. For an empty edge cut, returns ``b""`` so it
        sorts before any non-empty cut.
        """
        indices = self.indices
        if not indices:
            return b""
        return self.cut_matrix(max(indices) + 1).tobytes()
```

- [ ] **Step 1.4: Run lex_key tests to verify they pass**

```bash
uv run pytest test/test_resolve_ties.py -k "lex_key" -v
```

Expected: 5 passed.

- [ ] **Step 1.5: Write failing tests for `PARTITION_LEX` strategy and `sias()` resolver**

Append to `test/test_resolve_ties.py`:

```python
class DummySia:
    """Minimal SIA-shaped object for resolve_ties.sias tests."""

    def __init__(self, normalized_phi, phi, partition):
        self.normalized_phi = normalized_phi
        self.phi = phi
        self.partition = partition


def test_partition_lex_strategy_returns_partition_bytes():
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    sia = DummySia(normalized_phi=0.0, phi=0.0, partition=bp)
    key_fn = resolve_ties.phi_object_tie_resolution_strategies["PARTITION_LEX"]
    assert key_fn(sia) == bp.lex_key()


def test_sias_resolves_partition_lex_tertiary_tiebreak():
    """When (normalized_phi, -phi) ties, PARTITION_LEX picks the smallest lex key."""
    bp_lo = DirectedBipartition(Direction.EFFECT, (0,), (2,))   # smaller lex_key
    bp_hi = DirectedBipartition(Direction.EFFECT, (1,), (2,))   # larger lex_key
    assert bp_lo.lex_key() < bp_hi.lex_key()
    a = DummySia(normalized_phi=0.5, phi=1.0, partition=bp_lo)
    b = DummySia(normalized_phi=0.5, phi=1.0, partition=bp_hi)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([b, a]))   # b first to exercise ordering
    assert resolved == [a]


def test_sias_falls_through_to_secondary_when_normalized_phi_differs():
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    smaller = DummySia(normalized_phi=0.3, phi=1.0, partition=bp)
    larger = DummySia(normalized_phi=0.5, phi=1.0, partition=bp)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([smaller, larger]))
    assert resolved == [smaller]


def test_sias_secondary_prefers_larger_unnormalized_phi():
    """At equal normalized_phi, NEGATIVE_PHI minimised picks larger phi."""
    bp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    lower_phi = DummySia(normalized_phi=0.5, phi=1.0, partition=bp)
    higher_phi = DummySia(normalized_phi=0.5, phi=2.0, partition=bp)
    with config.override(
        sia_tie_resolution=["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    ):
        resolved = list(resolve_ties.sias([lower_phi, higher_phi]))
    assert resolved == [higher_phi]
```

- [ ] **Step 1.6: Run sias() tests to verify they fail**

```bash
uv run pytest test/test_resolve_ties.py -k "partition_lex or sias" -v
```

Expected: 4 failures — `PARTITION_LEX` missing from strategy registry; `sias_tie_resolution` config key missing; `resolve_ties.sias` function missing.

- [ ] **Step 1.7: Register `PARTITION_LEX` strategy and add `sias()` resolver**

Edit `pyphi/resolve_ties.py`. After the existing `@phi_object_tie_resolution_strategies.register("NONE")` block (line 65), add:

```python
@phi_object_tie_resolution_strategies.register("PARTITION_LEX")
def _(m):
    return m.partition.lex_key()
```

After the existing `purviews` function (line 140-149), add:

```python
def sias[T](
    sias: Iterable[T], strategy: str | list[str] | None = None, **kwargs: Any
) -> Iterator[T]:
    """Resolve ties among system-level SIAs.

    Controlled by the ``sia_tie_resolution`` configuration option.
    """
    strategy = fallback(strategy, config.formalism.iit.sia_tie_resolution)
    assert strategy is not None, "sia_tie_resolution config must be set"
    return resolve(sias, strategy, operation=min, **kwargs)
```

- [ ] **Step 1.8: Add `sia_tie_resolution` field on `IITConfig`**

Edit `pyphi/conf/formalism.py`. After line 57 (`purview_tie_resolution: str = "PHI"`), add:

```python
    sia_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    )
```

(The `field` import is already in scope at the top of the file — verify before adding.)

- [ ] **Step 1.9: Run sias() tests to verify they pass**

```bash
uv run pytest test/test_resolve_ties.py -v
```

Expected: all tests in the file pass (existing 6 + 5 lex_key + 4 sias = 15 passed).

- [ ] **Step 1.10: Verify config import sanity**

```bash
uv run python -c "from pyphi import config; print(config.formalism.iit.sia_tie_resolution)"
```

Expected output: `['NORMALIZED_PHI', 'NEGATIVE_PHI', 'PARTITION_LEX']`.

- [ ] **Step 1.11: Run pyright and ruff on touched files**

```bash
uv run pyright pyphi/models/partitions.py pyphi/resolve_ties.py pyphi/conf/formalism.py
uv run ruff check pyphi/models/partitions.py pyphi/resolve_ties.py pyphi/conf/formalism.py test/test_resolve_ties.py
uv run ruff format --check pyphi/models/partitions.py pyphi/resolve_ties.py pyphi/conf/formalism.py test/test_resolve_ties.py
```

Expected: all clean. (Repo-wide pyright is part of pre-commit; this is just for fast local feedback.)

- [ ] **Step 1.12: Run the full fast lane to verify no regression**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1099 passed (or current baseline + 9 new tests in `test_resolve_ties.py` = 1108).

- [ ] **Step 1.13: Commit**

```bash
git add pyphi/models/partitions.py pyphi/resolve_ties.py pyphi/conf/formalism.py test/test_resolve_ties.py
git commit -m "$(cat <<'EOF'
Add partition lex_key, resolve_ties.sias, sia_tie_resolution config

Adds the infrastructure pieces from the deterministic-SIA-selection
spec:

- `_PartitionBase.lex_key()` returns canonical bytes of the induced
  edge cut. Two partitions severing the same edges (across class
  boundaries: DirectedBipartition vs EdgeCut) sort identically.
- `PARTITION_LEX` strategy in `pyphi/resolve_ties.py` reads
  `m.partition.lex_key()`.
- `resolve_ties.sias` mirrors the existing `states / partitions /
  purviews` resolvers but reads `config.formalism.iit.sia_tie_resolution`.
- `IITConfig.sia_tie_resolution` defaults to
  `["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]`.

No call sites are wired through yet; this commit just adds the building
blocks. Existing test suite unchanged at 1099 passed; 9 new unit tests
in test_resolve_ties.py pass.
EOF
)"
```

---

## Task 2: Wire `resolve_ties.sias` into IIT 4.0 SIA selection

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py:699-740`

### Steps

- [ ] **Step 2.1: Read the current MIP loop**

Read `pyphi/formalism/iit4/__init__.py:699-740` to confirm the surrounding context. The replacement happens between the `MapReduce(...).run()` call (line 716) and the `clear_system_caches_after_computing_sia` block (line 737).

- [ ] **Step 2.2: Add the `resolve_ties` import**

Verify the import. Edit the top of `pyphi/formalism/iit4/__init__.py` — find the existing `from pyphi import` block and add `resolve_ties` if not present. The conventional style here is to import the module name:

```python
from pyphi import resolve_ties
```

Run:

```bash
grep -n "from pyphi import resolve_ties\|^from pyphi import" pyphi/formalism/iit4/__init__.py | head -5
```

If `resolve_ties` is not already imported, add the line in the existing `from pyphi import ...` block alphabetically.

- [ ] **Step 2.3: Replace the manual MIP loop with `resolve_ties.sias`**

Edit `pyphi/formalism/iit4/__init__.py` lines 718-735. Replace:

```python
    # Find MIP in one pass, keeping track of ties
    # TODO(ties) refactor into resolve_ties module
    mip_sia = default_sia
    mip_key = (float("inf"), float("-inf"))
    ties = [default_sia]
    if sias is None:
        sias = []
    for candidate_mip_sia in sias:
        candidate_key = sia_minimization_key(candidate_mip_sia)
        if candidate_key < mip_key:
            mip_sia = candidate_mip_sia
            mip_key = candidate_key
            ties = [mip_sia]
        elif candidate_key == mip_key:
            ties.append(candidate_mip_sia)
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)
```

with:

```python
    candidates = list(sias) if sias is not None else []
    if not candidates:
        candidates = [default_sia]
    ties = tuple(resolve_ties.sias(candidates))
    mip_sia = ties[0]
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)
```

- [ ] **Step 2.4: Delete the now-unused `sia_minimization_key` helper**

The function at `pyphi/formalism/iit4/__init__.py:591-595`:

```python
def sia_minimization_key(sia):
    """Return a sorting key that minimizes the normalized phi value.

    Ties are broken by maximizing the phi value."""
    return (sia.normalized_phi, -sia.phi)
```

Grep first to confirm no other caller:

```bash
grep -rn "sia_minimization_key" pyphi/ test/
```

If the only references are the definition and the now-replaced call site, delete the function. If there are other callers, leave the function in place.

- [ ] **Step 2.5: Run pyright and ruff on the modified file**

```bash
uv run pyright pyphi/formalism/iit4/__init__.py
uv run ruff check pyphi/formalism/iit4/__init__.py
uv run ruff format --check pyphi/formalism/iit4/__init__.py
```

Expected: all clean.

- [ ] **Step 2.6: Run the fast lane**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1108 passed (Task 1 baseline). If a test fails, inspect — the rewire shouldn't change behaviour on substrates without tied MIPs.

- [ ] **Step 2.7: Run the golden regression suite**

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 17/17 passed. Tied-MIP substrates that previously returned an arbitrary MIP may now return a different one — if so, regenerate fixtures (Task 5). Don't regenerate here; just note any divergence.

- [ ] **Step 2.8: Probe the two xfailed tests**

```bash
uv run pytest test/test_big_phi.py::test_sia_big_subsys_all_complete_sequential -v --runxfail
uv run pytest test/test_big_phi.py::test_sia_big_subsys_all_complete_parallel -v --runxfail
```

`--runxfail` overrides the xfail marker so we see the actual outcome. Record one of three results in a scratch note:

- **Both pass** → the captured fixture already matches the new canonical MIP. Task 5 will just drop the xfail markers.
- **Both fail with identical state mismatch** → the new canonical MIP differs from the captured one but is deterministic. Task 5 will regenerate the fixture.
- **Tests flake (different outputs across two `--runxfail` invocations)** → the fix is incomplete. Stop and diagnose.

- [ ] **Step 2.9: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py
git commit -m "$(cat <<'EOF'
Wire IIT 4.0 SIA MIP selection through resolve_ties.sias

Replaces the manual ``(normalized_phi, -phi)`` min-loop with a call
through `resolve_ties.sias`, which reads the new `sia_tie_resolution`
config and applies the `PARTITION_LEX` tertiary structural break per
the spec. `resolve_system_state` is unchanged — once the MIP is
deterministic, the cruelest-cut canonical state is too.

Drops the now-unused `sia_minimization_key` helper.

Goldens 17/17 still passing. Fast lane unchanged.
EOF
)"
```

---

## Task 3: Override `order_by` on IIT 3.0 `SystemIrreducibilityAnalysis`

**Files:**
- Modify: `pyphi/models/sia.py:22-138` (add `order_by` method on `SystemIrreducibilityAnalysis`)

### Steps

- [ ] **Step 3.1: Add the `order_by` override**

Edit `pyphi/models/sia.py`. The class is `SystemIrreducibilityAnalysis` at line 22, inheriting from `cmp.OrderableByPhi`. After the `partition` property at line 95-99:

```python
    @property
    def partition(self):
        """The partition that makes the least difference to the system."""
        assert self.partitioned_system is not None
        return self.partitioned_system.partition
```

add:

```python
    def order_by(self):
        """Sort key: phi (primary), then partition lex key (structural fallback)."""
        return (self.phi, self.partition.lex_key())
```

- [ ] **Step 3.2: Run pyright and ruff**

```bash
uv run pyright pyphi/models/sia.py
uv run ruff check pyphi/models/sia.py
uv run ruff format --check pyphi/models/sia.py
```

Expected: all clean.

- [ ] **Step 3.3: Verify IIT 3.0 SIA comparison works**

```bash
uv run python -c "
from pyphi import config
from pyphi.models.sia import SystemIrreducibilityAnalysis
from pyphi.models.partitions import NullCut
from pyphi.system import System
a = SystemIrreducibilityAnalysis(phi=0.0)
print('order_by callable:', callable(getattr(a, 'order_by', None)))
"
```

Expected: `order_by callable: True`. (Constructing a real comparable SIA requires a system and partition; this just verifies the method exists.)

- [ ] **Step 3.4: Run the fast lane**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1108 passed. IIT 3.0 SIA tests should still pass; the new structural fallback only fires on tied phi, which is rare for non-symmetric IIT 3.0 substrates.

- [ ] **Step 3.5: Run the golden regression suite**

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 17/17 passed.

- [ ] **Step 3.6: Commit**

```bash
git add pyphi/models/sia.py
git commit -m "$(cat <<'EOF'
Add partition lex_key fallback to IIT 3.0 SIA ordering

Overrides `order_by` on the IIT 3.0 `SystemIrreducibilityAnalysis`
subclass to return `(phi, partition.lex_key())`. `_sia_map_reduce`'s
`reduce_func=min` over `OrderableByPhi` now breaks phi-ties structurally
on the induced edge cut, making IIT 3.0 SIA selection deterministic
across runs.

No changes to the IIT 3.0 entry point — the override flows through
the existing inheritance.

Goldens 17/17. Fast lane 1108 passed.
EOF
)"
```

---

## Task 4: Add the determinism property test

**Files:**
- Modify: `test/test_invariants.py` (add new test at end of file)

### Steps

- [ ] **Step 4.1: Inspect existing test style**

```bash
grep -n "^def test_\|@pytest\.mark\.slow\|big_subsys_all_complete" test/test_invariants.py | head -10
```

Note the imports and fixture-usage pattern used in the file.

- [ ] **Step 4.2: Append the determinism property test**

Edit `test/test_invariants.py`. Append:

```python
@pytest.mark.slow
def test_sia_is_deterministic_across_runs(big_subsys_all_complete):
    """Running .sia() twice on the same substrate must yield equal SIAs.

    The fully-connected 5-node substrate has multiple partitions tied
    at the MIP minimisation key; structural tie-breaking guarantees a
    single canonical SIA across runs.
    """
    s1 = big_subsys_all_complete.sia()
    s2 = big_subsys_all_complete.sia()
    assert s1 == s2
```

If the file doesn't already import `pytest`, add `import pytest` at the top. If `big_subsys_all_complete` isn't a known fixture in the file's conftest scope, verify with:

```bash
grep -n "big_subsys_all_complete" test/conftest.py test/test_invariants.py | head -5
```

`test/conftest.py:326` defines it — it's a session-scope autouse fixture available file-wide.

- [ ] **Step 4.3: Run the new test**

```bash
uv run pytest test/test_invariants.py::test_sia_is_deterministic_across_runs -v --no-header
```

Expected: PASS. (This is slow — may take 30–90 s as it runs the SIA twice. Acceptable.) If it FAILS with `s1 != s2`, the determinism fix isn't complete: inspect which SIA fields differ. Most likely cause: the test fixture's MIP exposes a code path not covered by Task 2.

- [ ] **Step 4.4: Run the full fast lane (sanity)**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1108 passed (no regression from adding a slow-only test).

- [ ] **Step 4.5: Commit**

```bash
git add test/test_invariants.py
git commit -m "$(cat <<'EOF'
Add SIA determinism property test on big_subsys_all_complete

Asserts that two consecutive .sia() calls on the fully-connected
5-node substrate return equal SIAs. This is the architectural
guarantee the partition lex tie-break is designed to deliver; the
test exercises tied-MIP behaviour explicitly.
EOF
)"
```

---

## Task 5: Drop the `_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` markers (regenerate fixture if needed)

This task has two branches depending on what Task 2 Step 2.8 observed.

**Files:**
- Modify: `test/test_big_phi.py:159-185` (drop xfail markers and the `_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` constant)
- Possibly modify: `test/data/sia/big_subsys_all_complete.json` (regenerate)

### Steps

- [ ] **Step 5.1: Re-run the previously-xfailed tests with the marker still in place**

```bash
uv run pytest test/test_big_phi.py::test_sia_big_subsys_all_complete_sequential test/test_big_phi.py::test_sia_big_subsys_all_complete_parallel -v
```

Expected: `XPASS` (both) — the test body now passes, so pytest reports unexpected pass. Or `XFAIL` — the captured fixture differs from the new canonical SIA.

- [ ] **Step 5.2 (branch A — both `XPASS`): Drop xfail markers only**

If both report `XPASS`, edit `test/test_big_phi.py:159-185`. Delete the `_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` constant entirely (lines 159-168) and the `@_BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL` decorator on each test. The two test definitions become:

```python
@pytest.mark.slow
@config.override(parallel=False)
def test_sia_big_subsys_all_complete_sequential(
    big_subsys_all_complete, big_subsys_all_complete_expected_sia
):
    assert big_subsys_all_complete.sia() == big_subsys_all_complete_expected_sia


@pytest.mark.slow
def test_sia_big_subsys_all_complete_parallel(
    big_subsys_all_complete, big_subsys_all_complete_expected_sia
):
    assert big_subsys_all_complete.sia() == big_subsys_all_complete_expected_sia
```

Then jump to Step 5.5.

- [ ] **Step 5.3 (branch B — both `XFAIL`): Regenerate the fixture via the notebook**

The new canonical MIP differs from the captured one. Regenerate `test/data/sia/big_subsys_all_complete.json`:

```bash
uv run jupyter nbconvert --to notebook --execute test/IIT_4.0_make_jsons.ipynb --output IIT_4.0_make_jsons.executed.ipynb
```

This re-runs the full notebook; it writes all fixtures including `big_subsys_all_complete.json`. The notebook's output (`IIT_4.0_make_jsons.executed.ipynb`) is throwaway — delete it:

```bash
rm test/IIT_4.0_make_jsons.executed.ipynb
```

Inspect what changed:

```bash
git diff --stat test/data/sia/
```

Expected: at minimum `big_subsys_all_complete.json` changed. If other fixtures also changed (other symmetric substrates), inspect each — they may have legitimately drifted to a new canonical MIP. Note them.

- [ ] **Step 5.4 (branch B): Verify fixture regen makes the tests pass**

```bash
uv run pytest test/test_big_phi.py -m "not slow" -k "big_subsys" -v --runxfail
```

Expected: `XPASS` on both — the tests now match the regenerated fixture under the xfail decorator. Then drop the xfail markers as in Step 5.2.

If other fixtures also got regenerated in Step 5.3, run the full golden suite:

```bash
uv run pytest test/test_golden_regression.py -v
```

All 17 must pass against the regenerated fixtures.

- [ ] **Step 5.5: Run the (no-longer-xfailed) tests to confirm they pass**

```bash
uv run pytest test/test_big_phi.py::test_sia_big_subsys_all_complete_sequential test/test_big_phi.py::test_sia_big_subsys_all_complete_parallel -v
```

Expected: 2 passed (no XPASS — they're real passes now).

- [ ] **Step 5.6: Run the slow lane in the background**

```bash
uv run pytest test/ --runxfail -q
```

Run with `run_in_background=true`. Use `Monitor` with an `until` loop to know when it completes. Expected: 0 failures, 0 xfails (since this fix closes both xfails).

While the slow lane runs, you can move to Task 6.

- [ ] **Step 5.7: Stage and commit**

The exact `git add` list depends on which branch ran:

**Branch A (no fixture regen):**

```bash
git add test/test_big_phi.py
git commit -m "$(cat <<'EOF'
Drop _BIG_SUBSYS_ALL_COMPLETE_TIE_XFAIL markers

Both `test_sia_big_subsys_all_complete_{sequential,parallel}` now pass
under the deterministic SIA selection from the spec — partition lex
tie-break makes the MIP canonical across runs, and the captured
fixture happens to match the new canonical ordering. Tests no longer
need the xfail decorator.
EOF
)"
```

**Branch B (with fixture regen):**

```bash
git add test/test_big_phi.py test/data/sia/big_subsys_all_complete.json
# Add any other regenerated fixtures from Step 5.3:
# git add test/data/sia/<other>.json ...
git commit -m "$(cat <<'EOF'
Regenerate big_subsys_all_complete fixture and drop xfail markers

The deterministic SIA selection from the spec picks a different (but
now canonical) MIP than the one captured in the legacy fixture.
Regenerated via test/IIT_4.0_make_jsons.ipynb to match. Both
`test_sia_big_subsys_all_complete_{sequential,parallel}` now pass
without the xfail decorator.
EOF
)"
```

---

## Task 6: Add changelog fragment

**Files:**
- Create: `changelog.d/sia-determinism.fix.md`

### Steps

- [ ] **Step 6.1: Read the changelog README**

```bash
cat changelog.d/README.md | head -40
```

Confirm the filename convention (`<name>.<type>.md` with `type` in the allowed set) and the Markdown body style of other recent fragments.

- [ ] **Step 6.2: Create the fragment**

Write `changelog.d/sia-determinism.fix.md`:

```markdown
`Substrate.sia()` results are now deterministic across runs. Previously,
when multiple partitions tied at the MIP minimisation key, the
first-encountered tied partition under `MapReduce` iteration won — a
race that surfaced on symmetric substrates (fully-connected lattices,
the `big_subsys_all_complete` fixture). A structural tie-break on the
induced edge cut now selects the canonical partition. The new
`sia_tie_resolution` config option exposes the ordering;
the default is `["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]`.
```

- [ ] **Step 6.3: Verify towncrier accepts it**

```bash
uv run towncrier build --draft 2>&1 | head -30
```

Expected: the new fragment appears under the "Bug Fixes" section. (The `--draft` flag prints the rendered changelog without writing it.)

- [ ] **Step 6.4: Commit**

```bash
git add changelog.d/sia-determinism.fix.md
git commit -m "$(cat <<'EOF'
Add changelog fragment for deterministic SIA selection

User-facing summary of the partition lex tie-break and the new
`sia_tie_resolution` config option.
EOF
)"
```

---

## Final verification

- [ ] **Wait for the slow lane to complete (from Step 5.6 background run)**

When the `Monitor` notification fires, check the result:

```bash
# (output stream from the background run will have shown the final summary)
```

Expected: 1109 passed (1099 baseline + 9 new resolve_ties tests + 1 new invariant test), 0 failed, 0 xfails. The two formerly-xfailed `big_subsys_all_complete` tests are now in the passing count.

- [ ] **Run pyright on the full pyphi/**

```bash
uv run pyright pyphi/
```

Expected: 0 errors, 1 pre-existing baseline warning.

- [ ] **Run ruff on the full source**

```bash
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

Expected: clean.

- [ ] **Inspect the final commit log**

```bash
git log --oneline 0941b62a..HEAD
```

Expected: 5 or 6 new commits (5 if Task 5 was branch A; 6 if branch B included fixture regen as a separate commit — depends on whether you split). They should read as a coherent, mostly-mechanical implementation of the spec.

- [ ] **Report status to the user**

Summarise: number of commits added, slow-lane result, whether fixture regeneration was needed, any surprises. Do **not** push; pushing requires explicit per-action consent.

---

## Risk-mode cross-checks

These aren't task steps — they're things to watch for while executing.

| Risk | Detection signal | Response |
|---|---|---|
| Pre-commit hook reformats source | `git diff --cached` shows changes after `git commit` failed | Re-stage the auto-formatted files: `git add -u <paths>` then retry commit. Never `--no-verify`. |
| Pre-commit pyright fails on a file the plan didn't touch | Hook output names a file outside the touch list | The repo has stale baseline errors. Run `uv run pyright pyphi/` directly to see the full diagnostic; if it's pre-existing, narrow the pyright config or skip. Do not bypass the hook. |
| Fixture regeneration changes goldens beyond `big_subsys_all_complete` | `git diff --stat test/data/` shows >1 file | Inspect each — symmetric substrates may legitimately re-canonicalise. If a non-symmetric fixture changes, the lex key implementation may have a bug; stop and diagnose. |
| `default_sia` doesn't have `.partition.lex_key()` callable | Task 2 fast lane crashes with `AttributeError` | `NullSystemIrreducibilityAnalysis` constructs `partition=NullCut(...)`, which has `lex_key()` returning `b""`. If it crashes anyway, check `_null_sia` construction path. |
| 1Password gpgsign prompt blocks the commit | Commit hangs or fails with "gpg failed to sign the data" | Stop and ask the user for per-action consent. Do not bypass signing. |
| Other Claude sessions commit to the same branch mid-plan | `git status` shows unexpected modifications | Per saved memory: do not destroy unrelated changes. Rebase if necessary; commit your work under proper attribution. |

---

## Spec coverage check

| Spec component | Task(s) |
|---|---|
| Component 1 — `_PartitionBase.lex_key()` | Task 1 (Steps 1.1–1.4) |
| Component 2 — `resolve_ties.sias()` + `PARTITION_LEX` strategy | Task 1 (Steps 1.5–1.7) |
| Component 3 — `sia_tie_resolution` config field | Task 1 (Step 1.8) |
| Component 4 — IIT 4.0 MIP loop rewrite | Task 2 |
| Component 5 — IIT 3.0 `order_by` override | Task 3 |
| Test: `test_sias_partition_lex` | Task 1 (Step 1.5: `test_sias_resolves_partition_lex_tertiary_tiebreak`) |
| Test: `test_lex_key_equivalent_partitions_compare_equal` | Task 1 (Step 1.1) |
| Test: `test_lex_key_distinct_partitions_compare_distinct` | Task 1 (Step 1.1) |
| Test: drop xfail markers | Task 5 |
| Test: `test_sia_is_deterministic_across_runs` | Task 4 |
| Fixture regeneration (if needed) | Task 5 (branch B) |
| Changelog fragment | Task 6 |
