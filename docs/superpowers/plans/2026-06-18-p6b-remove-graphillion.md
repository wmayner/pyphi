# P6b — Remove graphillion (pure-Python relations enumeration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the graphillion/ZDD dependency by replacing concrete relations candidate generation with a lazy pure-Python enumerator that produces the identical candidate family.

**Architecture:** The only use of graphillion is `pyphi/relations.py`'s candidate generation, which builds a set-family of distinction-combinations sharing a common purview unit and then fully enumerates it. That family is exactly `combinations_with_nonempty_intersection` over each distinction's purview-union. We add a lazy depth-first enumerator in `pyphi/combinatorics.py` (prunes a subtree the moment the running intersection empties), rewire relations to it, then delete the graphillion wrappers, dependency, and lazy-import guard.

**Tech Stack:** Python 3.12+, pytest, Hypothesis, uv.

**Spec:** `docs/superpowers/specs/2026-06-18-p6b-remove-graphillion-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims.
- No new third-party dependency. No OxiDD.
- Relation results must stay byte-identical (relation-count goldens unchanged).
- Use `uv run` for all Python commands.
- Final verification runs `uv run pytest` **with no path argument** (so the `pyphi/` doctest sweep and relation goldens are included). Bare-path invocations skip doctests.
- Do not bypass pre-commit hooks (`--no-verify` forbidden).
- The working tree has unrelated untracked/modified files from other work — stage only the files each task names; never `git add -A` or check out/reset unrelated files.

---

### Task 1: Lazy pure-Python enumerator in `combinatorics.py`

Add the streaming enumerator and reimplement the by-order grouping over it. Nothing is removed yet (the graphillion wrappers stay until relations is rewired in Task 2), so the suite stays green throughout.

**Files:**
- Modify: `pyphi/combinatorics.py` (replace body of `combinations_with_nonempty_intersection` at ~109-130; replace body of `combinations_with_nonempty_intersection_by_order` at ~47-106)
- Test: `test/test_combinatorics.py`

**Interfaces:**
- Produces: `combinations_with_nonempty_intersection(sets: Sequence[frozenset], min_size: int = 0, max_size: int | None = None) -> Generator[frozenset[int]]` — yields each index-combination (as a `frozenset` of indices into `sets`) whose set-intersection is non-empty, with effective size in `[max(2, min_size), max_size]`.
- Produces: `combinations_with_nonempty_intersection_by_order(...) -> dict[int, set[frozenset]]` — the same combinations grouped by size.

- [ ] **Step 1: Write the failing property test**

Add to `test/test_combinatorics.py` (top-of-file imports `from hypothesis import given, strategies as st` already present in the project; add if missing). Append:

```python
import itertools

from hypothesis import given
from hypothesis import strategies as st


def _bruteforce_nonempty_intersection(sets, min_size, max_size):
    n = len(sets)
    upper = n if max_size is None else min(max_size, n)
    expected = set()
    for size in range(max(2, min_size), upper + 1):
        for combo in itertools.combinations(range(n), size):
            inter = sets[combo[0]]
            for i in combo[1:]:
                inter = inter & sets[i]
            if inter:
                expected.add(frozenset(combo))
    return expected


@given(
    sets=st.lists(
        st.frozensets(st.integers(min_value=0, max_value=5), max_size=4),
        min_size=0,
        max_size=8,
    ),
    min_size=st.integers(min_value=0, max_value=5),
    max_size=st.integers(min_value=0, max_value=6),
)
def test_combinations_with_nonempty_intersection_matches_bruteforce(
    sets, min_size, max_size
):
    result = set(
        combinatorics.combinations_with_nonempty_intersection(
            sets, min_size=min_size, max_size=max_size
        )
    )
    expected = _bruteforce_nonempty_intersection(sets, min_size, max_size)
    assert result == expected
```

- [ ] **Step 2: Run the property test against the current (materializing) implementation**

Run: `uv run pytest test/test_combinatorics.py::test_combinations_with_nonempty_intersection_matches_bruteforce -q`
Expected: PASS (the existing implementation already satisfies the oracle; this pins the contract before refactoring). If it fails, stop — the oracle or current behavior is misunderstood; investigate before continuing.

- [ ] **Step 3: Replace `combinations_with_nonempty_intersection` with the lazy DFS**

In `pyphi/combinatorics.py`, replace the existing `combinations_with_nonempty_intersection` function (the one returning `chain[frozenset]`) with:

```python
def combinations_with_nonempty_intersection(
    sets: Sequence[frozenset], min_size: int = 0, max_size: int | None = None
) -> Generator[frozenset[int]]:
    """Yield index-combinations whose set-intersection is nonempty.

    Each yielded ``frozenset`` holds indices ``i`` into ``sets`` such that the
    intersection of the corresponding sets is nonempty. Combinations are
    enumerated by depth-first search over indices in increasing order, pruning a
    whole subtree as soon as the running intersection becomes empty (sound
    because intersection is monotone non-increasing under adding elements).
    Singletons are never yielded; the effective minimum size is
    ``max(2, min_size)``.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider.

    Keyword Arguments:
        min_size (int): Minimum combination size to yield. Defaults to 0.
        max_size (int): Maximum combination size to yield. If None, no upper
            bound.
    """
    n = len(sets)
    effective_min = max(2, min_size)
    upper = n if max_size is None else max_size
    if upper < effective_min:
        return

    def _extend(
        start: int, chosen: list[int], running: frozenset
    ) -> Generator[frozenset[int]]:
        size = len(chosen)
        if size >= effective_min:
            yield frozenset(chosen)
        if size >= upper:
            return
        for i in range(start, n):
            new_running = running & sets[i]
            if new_running:
                chosen.append(i)
                yield from _extend(i + 1, chosen, new_running)
                chosen.pop()

    for i in range(n):
        if sets[i]:
            yield from _extend(i + 1, [i], sets[i])
```

(`Sequence`, `Generator`, and `defaultdict` are already imported at the top of the file; the `chain` import becomes unused once Step 4 lands — remove it if nothing else uses it.)

- [ ] **Step 4: Reimplement `combinations_with_nonempty_intersection_by_order` over the generator**

Replace the existing `combinations_with_nonempty_intersection_by_order` function body with a thin grouping over the generator (single source of truth):

```python
def combinations_with_nonempty_intersection_by_order(
    sets: Sequence[frozenset], min_size: int = 0, max_size: int | None = None
) -> dict[int, set[frozenset]]:
    """Return nonempty-intersection combinations grouped by size.

    Same combinations as :func:`combinations_with_nonempty_intersection`,
    bucketed into ``{size: {combination, ...}}``. Sizes with no combinations are
    omitted.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider.

    Keyword Arguments:
        min_size (int): Minimum combination size. Defaults to 0.
        max_size (int): Maximum combination size. If None, no upper bound.
    """
    by_order: dict[int, set[frozenset]] = defaultdict(set)
    for combination in combinations_with_nonempty_intersection(
        sets, min_size=min_size, max_size=max_size
    ):
        by_order[len(combination)].add(combination)
    return dict(by_order)
```

Keep the original docstring's argument descriptions if they were richer; do not retain the old incremental-closure algorithm.

- [ ] **Step 5: Run the combinatorics tests**

Run: `uv run pytest test/test_combinatorics.py -q`
Expected: PASS — both the existing `test_combinations_with_nonempty_intersection` / `test_explicit_combinations_with_nonempty_intersection` (parametrized over `size_args`) and the new property test.

- [ ] **Step 6: Commit**

```bash
git add pyphi/combinatorics.py test/test_combinatorics.py
git commit -m "Add lazy pure-Python nonempty-intersection enumerator

Replace the materializing combinations_with_nonempty_intersection with a
streaming depth-first enumerator that prunes on empty running intersection;
reimplement the by-order grouping over it. Pinned against a brute-force oracle."
```

---

### Task 2: Rewire relations to the pure-Python enumerator

Point concrete relations candidate generation at the new enumerator and drop graphillion from `relations.py`. The graphillion wrappers in `combinatorics.py` are now unused but stay until Task 3 (so this task is independently green).

**Files:**
- Modify: `pyphi/relations.py` (function `_combinations_with_nonempty_congruent_overlap` at ~265-286; remove `from graphillion import setset` at line ~36 and ~274)
- Test: existing relation goldens (`test/test_paper_reproduction.py`, `test/test_golden_regression.py`, `test/test_relations.py` if present)

**Interfaces:**
- Consumes: `combinatorics.combinations_with_nonempty_intersection` (Task 1).
- Produces: `_combinations_with_nonempty_congruent_overlap(components, min_degree=2, max_degree=None)` returns an iterator of `frozenset[int]` index-combinations positionally aligned with `components` (consumed by `all_relations`'s `worker` via `distinctions[i]`).

- [ ] **Step 1: Capture a relations baseline (characterization)**

Run a quick relation-count check that the rewrite must preserve:

```bash
uv run pytest test/test_paper_reproduction.py -q -k "fig2 or fig4 or relation" 2>&1 | tail -20
```

Expected: PASS (records the current green baseline; if any are `@pytest.mark.slow`, note which and include them in Step 4).

- [ ] **Step 2: Rewrite `_combinations_with_nonempty_congruent_overlap`**

In `pyphi/relations.py`, replace the function with:

```python
def _combinations_with_nonempty_congruent_overlap(
    components, min_degree=2, max_degree=None
):
    """Return combinations of distinctions with nonempty congruent overlap.

    Two distinctions can relate only if their purview-unions share a unit; a
    combination can relate only if all its members share a common unit, i.e. the
    intersection of their purview-unions is nonempty. Congruence of the shared
    state is checked downstream when the ``Relation`` is constructed.

    Arguments:
        components (Distinctions): The distinctions to find overlaps among.
    """
    purview_unions = [frozenset(component.purview_union) for component in components]
    return combinatorics.combinations_with_nonempty_intersection(
        purview_unions, min_size=min_degree, max_size=max_degree
    )
```

Then delete the two graphillion imports: the module-top `from graphillion import setset  # noqa: F401` (line ~36) and the in-function `from graphillion import setset` (line ~274). Leave the `combinatorics` import in place.

- [ ] **Step 3: Verify graphillion is gone from relations.py**

Run: `grep -n "graphillion\|setset\|purview_inclusion(max_order=1)\|set_universe" pyphi/relations.py`
Expected: no matches.

- [ ] **Step 4: Run the relations regression suite**

Run: `uv run pytest test/test_paper_reproduction.py test/test_golden_regression.py -q` (include `-m "slow or not slow"` / drop any `-m` filter so slow relation reproductions run; if the machine is constrained, at minimum run the non-slow relation tests and note the slow ones to run in Task 4's full sweep).
Expected: PASS — relation counts (e.g. Fig 7's 13740 / 13111 relations) byte-identical.

- [ ] **Step 5: Commit**

```bash
git add pyphi/relations.py
git commit -m "Rewire relations candidate generation to pure-Python enumerator

Replace the graphillion union_powerset_family path with
combinations_with_nonempty_intersection over distinction purview-unions
(byte-identical candidate family). Drop graphillion imports from relations.py."
```

---

### Task 3: Delete graphillion wrappers, dependency, and lazy-import guard

Remove the now-dead graphillion surface entirely.

**Files:**
- Modify: `pyphi/combinatorics.py` (delete `powerset_family` ~133-162 and `union_powerset_family` ~165-179; delete the `if TYPE_CHECKING: from graphillion import setset` block ~20-21)
- Modify: `pyproject.toml` (remove `"Graphillion>=1.5"` at line 33; remove the `[tool.uv.sources]` graphillion section, lines ~277-280)
- Modify: `uv.lock` (regenerate)
- Modify: `test/test_lazy_imports.py` (remove `test_graphillion_not_loaded_at_pyphi_import`; rewrite module docstring)

- [ ] **Step 1: Delete the graphillion wrappers from `combinatorics.py`**

Remove the `powerset_family` and `union_powerset_family` functions in full. Remove the `TYPE_CHECKING` import block:

```python
if TYPE_CHECKING:
    from graphillion import setset
```

If `TYPE_CHECKING` is now unused, remove its import too. Confirm:

Run: `grep -n "graphillion\|setset\|powerset_family\|union_powerset_family" pyphi/combinatorics.py`
Expected: no matches.

- [ ] **Step 2: Remove the dependency from `pyproject.toml`**

Delete the line `    "Graphillion>=1.5",` from `[project] dependencies`. Delete the `[tool.uv.sources]` section that contains only graphillion:

```toml
[tool.uv.sources]
# Build graphillion from source to avoid macOS libgomp dependency issues
# The PyPI wheel has hardcoded paths to Homebrew GCC's libgomp library
graphillion = { git = "https://github.com/takemaru/graphillion" }
```

(Remove the section header and the surrounding blank line if no other `[tool.uv.sources]` entries remain.)

- [ ] **Step 3: Regenerate the lock and confirm graphillion is gone**

Run: `uv lock`
Run: `grep -n "graphillion" uv.lock pyproject.toml`
Expected: no matches.

- [ ] **Step 4: Remove the graphillion lazy-import test**

In `test/test_lazy_imports.py`, delete `test_graphillion_not_loaded_at_pyphi_import` in full. Keep `_check_module_after_import` (the xarray test uses it) and `test_xarray_backend_not_loaded_at_pyphi_import`. Rewrite the module docstring to:

```python
"""Lazy-import discipline.

These tests pin the deferred-import contract that keeps free-threaded
CPython safe to use with PyPhi: optional heavy modules must not load at
``import pyphi`` time, only when the user explicitly invokes the code that
needs them.
"""
```

(Remove the `P6a` tag and all graphillion / OxiDD references; the no-GIL gap that graphillion created is closed by its removal.)

- [ ] **Step 5: Verify the whole repo is graphillion-free**

Run: `grep -rn "graphillion\|setset\|Graphillion" pyphi/ test/ pyproject.toml uv.lock`
Expected: no matches.

- [ ] **Step 6: Reinstall the environment and smoke-test import + a relations compute**

Run: `uv sync` (or `env -u VIRTUAL_ENV uv pip install -e ".[dev,visualize,caching,emd]"` if in a worktree — see project memory on worktree/uv mismatch)
Run: `uv run python -c "import pyphi; s=pyphi.examples.grid3_system(); print(s.ces().relations.num_relations())"`
Expected: prints a relation count without error and without importing graphillion.

- [ ] **Step 7: Commit**

```bash
git add pyphi/combinatorics.py pyproject.toml uv.lock test/test_lazy_imports.py
git commit -m "Remove graphillion dependency and its lazy-import guard

Delete the unused powerset_family / union_powerset_family ZDD wrappers, drop
Graphillion from dependencies and the uv git-source override, and remove the
graphillion no-GIL deferral test (moot once the dependency is gone)."
```

---

### Task 4: Changelog, roadmap, and full verification

Document the change and run the complete gate.

**Files:**
- Create: `changelog.d/p6b-remove-graphillion.change.md`
- Modify: `ROADMAP.md` (P6b dashboard row ~line 32; Wave 2 archive bullet ~line 92; P6a row ~line 43)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/p6b-remove-graphillion.change.md`:

```markdown
Removed the `graphillion` dependency. Concrete relations enumeration is now
pure Python, so the macOS libomp source-build is no longer required and the
relations path is free-threading (no-GIL) safe. The internal
`pyphi.combinatorics.powerset_family` / `union_powerset_family` helpers were
removed.
```

- [ ] **Step 2: Update the ROADMAP dashboard row for P6b**

In `ROADMAP.md`, change the P6b table row status from `⬜ open` to `✅ landed` and update the one-line to record the outcome, e.g.:

```markdown
| P6b | ✅ landed | 2 | graphillion removed (not ported to OxiDD). A pre-design spike showed the ZDD is not load-bearing: the relations candidate family equals `combinations_with_nonempty_intersection` over distinction purview-unions (byte-identical on real networks; pure-Python faster in the concrete-relations regime, and the >10⁶ regime uses `AnalyticalRelations`). Concrete relations enumeration is now a lazy pure-Python DFS; closes the last no-GIL gap and drops the macOS libomp source-build. |
```

- [ ] **Step 3: Update the Wave 2 archive bullet and the P6a row**

Update the Wave 2 `P6b` bullet (the `**P6b — graphillion → OxiDD.**` paragraph) to reflect the actual outcome (graphillion removed, OxiDD rejected, spike evidence). In the P6a dashboard row, remove the "xfail relations until P6b, then full" qualifier — relations are no longer the no-GIL gap (note P6a can now run the no-GIL lane with relations included).

- [ ] **Step 4: Add P6b to the "Landed" prose line**

In the `### ✅ Landed` section near the top, append `· P6b` to the landed list.

- [ ] **Step 5: Run the full verification gate**

Run: `uv run pytest`
Expected: PASS with **no path argument** (collects `pyphi/` + `test/` doctests and all relation goldens). If long, run the slow lane in the background per the project's parallel-test guidance, but the gate is the full no-path run.

- [ ] **Step 6: Final graphillion-free assertion**

Run: `grep -rn "graphillion\|Graphillion\|setset" pyphi/ test/ pyproject.toml uv.lock docs/ | grep -v "docs/superpowers/specs/2026-06-18-p6b\|docs/superpowers/plans/2026-06-18-p6b\|changelog"`
Expected: no matches (the spec/plan/changelog references to the removed library are allowed historical context).

- [ ] **Step 7: Commit**

```bash
git add changelog.d/p6b-remove-graphillion.change.md ROADMAP.md
git commit -m "Mark P6b landed: graphillion removed; changelog + roadmap

Record the spike outcome (ZDD not load-bearing, OxiDD rejected) and that the
relations no-GIL blocker P6a/P11 were gated on is now closed."
```

---

## Self-Review

**Spec coverage:**
- Core lazy enumerator (spec 4.1) → Task 1.
- `_by_order` as thin grouping (spec 4.1) → Task 1 Step 4.
- Relations rewire (spec 4.2) → Task 2.
- Delete wrappers + dependency + lock (spec 4.3) → Task 3.
- Hypothesis property test vs brute-force oracle (spec 4.4) → Task 1 Step 1.
- Remove lazy-import test + docstring (spec 4.4) → Task 3 Step 4.
- Relation-count golden safety net + full `uv run pytest` (spec 4.4, 6) → Task 2 Step 4, Task 4 Step 5.
- Roadmap + changelog (spec 4.5) → Task 4.
- Breaking-change note for removed public helpers (spec 5) → Task 4 changelog fragment.

**Cleanup of investigation scratch:** the `p6b_spike/` directory is throwaway. It is referenced by the spec for reproducibility but is not part of the shipped change — leave it untracked (do not `git add` it). The user can delete it after landing.

**Type consistency:** `combinations_with_nonempty_intersection` yields `frozenset[int]` in Task 1 and is consumed as index iterables (`distinctions[i] for i in combination`) in Task 2 — compatible. `_combinations_with_nonempty_congruent_overlap` returns the generator directly; `all_relations`'s `MapReduce` iterates it.

**Placeholder scan:** none — every code step shows complete code.
