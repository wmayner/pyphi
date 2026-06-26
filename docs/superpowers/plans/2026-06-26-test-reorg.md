# Test-suite reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `test/`'s 156 flat files into a single-level tree mirroring `pyphi/`'s subpackages (plus a `test/integration/` for cross-cutting tests and a small set of reviewed renames), preserving every test, fixture, and result — only import paths and a few `__file__`-relative paths change.

**Architecture:** A reviewed `old → new` mapping table drives a one-off deterministic Python script that does `git mv` + a uniform depth-1 import rewrite (`from .X` → `from ..X`) + `__init__.py` per subdir + cross-reference updates. The `~8` `__file__`-relative-path files are flagged and hand-fixed. Correctness is gated by a collected-test-set invariant captured before the move.

**Tech Stack:** Python 3.12+, pytest, ruff + pyright, git.

## Global Constraints

- **Work on a branch off `main`** from a clean working tree (a 156-file move conflicts easily with concurrent instances). Ask before `git push`.
- **Pure relocation + approved renames only** — no test-logic, fixture, split, or merge changes.
- **Single-level subdirs only** (depth 1); shared infra (`conftest.py`, `example_*`, `hypothesis_utils.py`, `test_helpers.py`, `golden/`, `data/`, `reference/`) stays at `test/` root.
- **The collected-test-set must be identical before and after.** This is the primary gate.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Full verification = `uv run --all-extras pytest` with NO path argument.** Never `--no-verify`.

---

## Task 1: Capture the baseline collection invariant

**Files:**
- Create: `/private/tmp/.../scratchpad/reorg_baseline.txt` (scratch, not committed)

- [ ] **Step 1: Record the exact collected-test set and counts**

Run:
```bash
uv run --all-extras pytest --collect-only -q 2>/dev/null \
  | sed -E 's|^test/[^:]*/|test/|' \
  | sort > "$SCRATCH/reorg_baseline_nodes.txt"
wc -l "$SCRATCH/reorg_baseline_nodes.txt"
```
The `sed` normalizes away the directory portion of each node ID (keeping
`test/<basename>::<nodeid>`), so a file moving from `test/` to `test/models/`
yields the *same* normalized line — the set is move-invariant by construction.

- [ ] **Step 2: Record the baseline pass/skip totals**

Run: `uv run --all-extras pytest -q 2>&1 | tail -1` → save the
`N passed, M skipped` line to `$SCRATCH/reorg_baseline.txt`. Expected today:
`2771 passed, 283 skipped` (or current).

---

## Task 2: Build and review the mapping table (REVIEW GATE)

**Files:**
- Create: `scripts/test_reorg_mapping.json`

**Interfaces:**
- Produces: a JSON object `{ "<old_basename>": {"dir": "<subdir or ''>", "rename": "<new_basename or null>", "reason": "<why, for renames/integration>"} }` covering **all** `test/test_*.py` files.

- [ ] **Step 1: Enumerate every test file and its current imports / `__file__` use**

Run:
```bash
for f in test/test_*.py; do
  b=$(basename "$f")
  fileuse=$(grep -lq "__file__" "$f" && echo "FILE" || echo "-")
  printf "%-50s %s\n" "$b" "$fileuse"
done
```

- [ ] **Step 2: Assign each file a bucket and (if needed) a rename**

For each file, decide `dir` per the spec's three buckets:
- subpackage module → `"conf"`/`"core"`/`"display"`/`"formalism"`/`"measures"`/`"models"`/`"parallel"`/`"serialize"`/`"visualize"`/`"macro"`/`"matching"`/`"data_structures"`/`"substrate_generator"` (only where ≥2 files land);
- root module → `""` (stays at root);
- cross-cutting → `"integration"`.

Set `rename` only when the name is genuinely unintelligible/misleading (bar:
"can you predict what it tests?"). Known candidates: `test_inner_retirement.py`
→ `test_inner_unwrap_retired.py`; `test_visualize_aux.py` →
`test_visualize_matplotlib.py`. Leave clear names (`test_install_snapshot.py`,
`test_golden_regression.py`, `test_paper_reproduction.py`) alone.

Write `scripts/test_reorg_mapping.json`.

- [ ] **Step 3: Sanity-check the table mechanically**

Run a check script asserting: every `test/test_*.py` appears exactly once; no
two entries collide on the same final `dir/new_basename`; every `dir` is in the
allowed set; any `dir` with only one file is flagged (should move to root
instead per the ≥2 rule).

- [ ] **Step 4: STOP — present the mapping + the rename list for approval**

Print the proposed renames (old → new + reason) and a per-directory file count.
**Do not proceed to Task 3 until the user approves the table.** (Naming and
bucketing are judgment calls the user signed up to review.)

---

## Task 3: Write the migration script

**Files:**
- Create: `scripts/reorg_tests.py`

**Interfaces:**
- Consumes: `scripts/test_reorg_mapping.json` (Task 2).
- Produces: side effects (`git mv`, edited files, new `__init__.py`); prints a report of `__file__`-using files needing manual fixup.

- [ ] **Step 1: Write the script**

`scripts/reorg_tests.py`:
```python
"""One-off: relocate test files per scripts/test_reorg_mapping.json.

Single-level layout, so every moved file's root-relative imports shift by
exactly one dot (`from .X` -> `from ..X`). Files that build paths from
``__file__`` are reported, not auto-edited (their path expressions need a
depth-aware fix by hand).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

TEST = Path("test")
MAPPING = json.loads(Path("scripts/test_reorg_mapping.json").read_text())

# from .X  /  from . import X   ->   from ..X  /  from .. import X
_REL = re.compile(r"^(\s*)from \.(\s|\w)", flags=re.M)


def shift_imports(text: str) -> str:
    return _REL.sub(r"\1from ..\2", text)


def main(apply: bool) -> int:
    flagged: list[str] = []
    moves: list[tuple[Path, Path]] = []
    for old_base, spec in sorted(MAPPING.items()):
        old = TEST / old_base
        if not old.exists():
            print(f"!! missing: {old}")
            return 1
        new_base = spec.get("rename") or old_base
        subdir = spec["dir"]
        new = TEST / subdir / new_base if subdir else TEST / new_base
        if "__file__" in old.read_text() and subdir:
            flagged.append(str(new))
        moves.append((old, new))

    if not apply:
        for old, new in moves:
            if old != new:
                print(f"mv {old} -> {new}")
        print(f"\n{len(flagged)} files use __file__ and need a manual path fix:")
        print("\n".join(f"  {p}" for p in flagged))
        return 0

    # 1) ensure subdir packages exist
    for _, new in moves:
        if new.parent != TEST:
            new.parent.mkdir(parents=True, exist_ok=True)
            init = new.parent / "__init__.py"
            if not init.exists():
                init.write_text("")
                subprocess.run(["git", "add", str(init)], check=True)
    # 2) move + rewrite imports for files that change directory
    for old, new in moves:
        if old == new:
            continue
        subprocess.run(["git", "mv", str(old), str(new)], check=True)
        if new.parent != TEST:  # moved into a subdir -> shift imports
            new.write_text(shift_imports(new.read_text()))
            subprocess.run(["git", "add", str(new)], check=True)
    print(f"\nMoved {sum(o != n for o, n in moves)} files.")
    print(f"{len(flagged)} __file__ files still need a manual path fix:")
    print("\n".join(f"  {p}" for p in flagged))
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
```

- [ ] **Step 2: Dry-run it**

Run: `uv run python scripts/reorg_tests.py` (no `--apply`).
Expected: the full `mv` list and the `__file__`-flagged list (~8 files). Eyeball
the moves against the approved table.

---

## Task 4: Execute the move and fix `__file__` residuals

- [ ] **Step 1: Branch from a clean tree and apply**

```bash
git status --porcelain --untracked-files=no   # must be empty
git checkout -b test-reorg
uv run python scripts/reorg_tests.py --apply
```

- [ ] **Step 2: Hand-fix each flagged `__file__` file**

For each file the script flagged, adjust the path expression for its new depth.
A file moved to `test/<sub>/` is one level deeper, so a root-anchored path needs
one more `.parent`:
```python
# before (at test/):     Path(__file__).parent / "data" / "bounds" / "x.json"
# after  (at test/integration/):  Path(__file__).parent.parent / "data" / "bounds" / "x.json"
```
`test_presets.py`'s `Path(__file__).resolve().parent.parent / "pyphi_config_3.0.yml"`
(repo root from `test/`) similarly gains one `.parent` when it moves to a subdir.
Re-`git add` each.

- [ ] **Step 3: Update cross-references to renamed/moved files**

```bash
# for each renamed old basename, find and update references in docstrings/comments
grep -rn "test_inner_retirement\|test_visualize_aux" test --include="*.py"
```
Update any hit (e.g. a "see test_…" docstring) to the new name. Run for every
entry whose `rename` is set.

---

## Task 5: Verify the collection invariant

- [ ] **Step 1: Re-collect and diff against baseline**

```bash
uv run --all-extras pytest --collect-only -q 2>/dev/null \
  | sed -E 's|^test/[^:]*/|test/|' | sort > "$SCRATCH/reorg_after_nodes.txt"
diff "$SCRATCH/reorg_baseline_nodes.txt" "$SCRATCH/reorg_after_nodes.txt" && echo "IDENTICAL SET"
```
Expected: `IDENTICAL SET` (no diff). A non-empty diff means a test was dropped,
un-collected, or duplicated — investigate before continuing (usually a missed
import shift or a missing `__init__.py`).

---

## Task 6: Full suite + type/lint gate

- [ ] **Step 1: Full suite (no path argument)**

Run: `uv run --all-extras pytest`
Expected: same `passed`/`skipped` numbers as the Task 1 baseline.

- [ ] **Step 2: pyright + ruff**

Run: `uv run pyright pyphi test 2>&1 | tail -3` and
`SKIP=pyright uv run pre-commit run ruff ruff-format --files $(git diff --cached --name-only)`
Expected: clean (the import rewrites must satisfy isort I001 / F401).

---

## Task 7: Update docs/ROADMAP, remove the one-off script, finish

- [ ] **Step 1: Update references to test paths**

Grep `ROADMAP.md` and `docs/` for any hardcoded `test/test_…` path that moved;
update. Mark the test-reorg item done in the ROADMAP P15 lines.

- [ ] **Step 2: Remove the one-off migration artifacts**

```bash
git rm scripts/reorg_tests.py scripts/test_reorg_mapping.json
```
(They were a one-time tool; the result is the committed tree.)

- [ ] **Step 3: Changelog fragment**

```bash
echo 'Reorganized the test suite to mirror the `pyphi/` package layout (`test/<subpackage>/`), with cross-cutting tests under `test/integration/`. Pure relocation — no test behavior changed.' > changelog.d/test-reorg.misc.md
```

- [ ] **Step 4: Commit and finish the branch**

```bash
git add -A test/ ROADMAP.md changelog.d/test-reorg.misc.md docs/
git commit   # "Reorganize test/ to mirror pyphi/ package layout"
```
Then use superpowers:finishing-a-development-branch. **Ask before pushing.**

---

## Notes for the implementer

- **The collection invariant is the safety net** — trust the normalized
  `--collect-only` diff over eyeballing. If it says IDENTICAL SET and the suite
  is green with the same counts, the relocation is behavior-preserving.
- **`git mv` preserves history/blame** — use it (not `mv` + `git add`).
- **`git add -A test/`** in Task 7 is the one acceptable broad add (it is scoped
  to `test/`), but verify `git status` first shows only the move set.
- **If the collection diff is non-empty**, the usual causes are: a missed
  `from .` that should be `from ..` (a file that imports a helper via an unusual
  form the regex missed), a subdir missing its `__init__.py`, or a `__file__`
  path that now resolves to a non-existent location (a fixture file fails to
  load → its parametrized tests vanish from collection).
