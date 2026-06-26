# Test-suite reorganization — design

Relocate the flat `test/` directory into a tree that mirrors `pyphi/`'s
package layout, so a module's tests are findable by location instead of by
grep. A pure relocation (plus a small set of reviewed renames): every test,
fixture, and result is preserved; the only code changes are import-path
adjustments and a handful of `__file__`-relative path fixups.

## Goal

At 156 flat files, `test/` no longer mirrors `pyphi/`'s 15 subpackages, so
finding the tests for a module means grepping. Group the files into a tree that
mirrors the source layout. This is maintainability, **not a 2.0 ship gate** (the
ship criteria are named types, green goldens/Hypothesis, the frozen public
surface, a clean import, and the Sphinx/migration docs — test layout is not
among them).

## Background — the current state

- `test/` is already a package (`test/__init__.py`); `test/golden/` and
  `test/reference/` are subpackages. There is exactly one `conftest.py`, at the
  `test/` root.
- All test-file basenames are unique (no collision risk under package import).
- 23+ files use root-relative imports — `from .conftest`, `from .golden`,
  `from .hypothesis_utils`, `from .test_helpers`, `from .example_substrates`.
- **No test file imports another test file** (only the shared helpers), so the
  files are independent — the move order does not matter and the import rewrite
  is uniform.
- Data fixtures are referenced two ways: most via `open("./test/data/…")`
  (anchored to the repo-root CWD — survives a move), but ~8 files via
  `Path(__file__).parent / "data" / …` (anchored to the file — **breaks** on a
  move). One file (`test_presets.py`) uses `Path(__file__).resolve().parent.parent`
  to reach a repo-root config file (also shifts).

## Design

### Target layout

A **single level** of subdirectories under `test/`, one per top-level `pyphi`
subpackage that receives **≥2 test files** (a directory for one file is noise;
that file stays at the root). Nested source subpackages (`core/tpm/`,
`formalism/iit4/`, `parallel/backends/`) are **not** mirrored deeper — their
tests live in the top-level group (`test/core/`, `test/formalism/`,
`test/parallel/`). Keeping everything at depth 1 makes the import rewrite a
single uniform transform.

Likely groups (final mapping is the spec deliverable, see below):
`test/conf/`, `test/core/`, `test/display/`, `test/formalism/`,
`test/measures/`, `test/models/`, `test/parallel/`, `test/serialize/`,
`test/visualize/`, `test/macro/`, `test/matching/`, `test/data_structures/`,
`test/substrate_generator/`, and `test/integration/`.

### The mapping principle (three buckets)

1. **Subpackage tests → `test/<subpkg>/`.** Tests for a module under a `pyphi`
   subpackage move to the matching directory (`test_config_*` → `test/conf/`,
   `test_visualize_*` → `test/visualize/`, the model tests → `test/models/`,
   `test_serializable_mixin` / `test_serialize_*` / `test_convert` →
   `test/serialize/`, `test_pyphi_float` → `test/data_structures/`, …).
2. **Root-module tests → `test/` root.** Tests for a `pyphi` *root* module stay
   at the `test/` root, mirroring that the module lives at the package root
   (`test_system_*`, `test_substrate*`, `test_node`, `test_validate`,
   `test_utils`, `test_labels`, `test_direction`, `test_dynamics`,
   `test_connectivity`, …).
3. **Cross-cutting / integration tests → `test/integration/`.** Tests that span
   modules or formalisms and belong to no single module: `test_cross_formalism_invariants`,
   `test_paper_reproduction`, `test_golden_regression`, `test_invariants*`,
   `test_import_surface`, `test_lazy_imports`, `test_perf_*`,
   `test_cap_regression*`, `test_eq23_cap_oracle`, `test_bounds*`. `integration/`
   is the standard umbrella; if a sub-category later grows large (e.g. the
   goldens), it can split into its own dir then — not pre-split now.

### Shared infrastructure stays at the `test/` root

`conftest.py`, `example_networks.py`, `example_substrates.py`,
`hypothesis_utils.py`, `test_helpers.py`, and the `golden/` / `data/` /
`reference/` directories stay at the root. Every moved file points "up" to the
same place, so the import rewrite is one uniform transform and fixtures are not
fragmented (the single root `conftest` applies to all subdirs by pytest's
directory inheritance).

### Import convention

Depth-1 relative: every `from .X` referring to a root helper becomes `from ..X`
(`from .conftest` → `from ..conftest`, `from .golden` → `from ..golden`, …).
Because the layout is single-level, this is a uniform `.` → `..` rewrite for
moved files; root-module tests that stay put are unchanged. Each new subdir gets
an `__init__.py` (matching the existing `test/`, `golden/`, `reference/`
package style).

### Renames (reviewed, in scope)

Files whose names are genuinely unintelligible or misleading — you cannot tell
even roughly what is tested without opening the file — are renamed to
intelligible names *as part of the same move*. The bar is **intelligibility**,
not "names a module": a regression test named after the bug it guards
(`test_golden_regression`, `test_paper_reproduction`, `test_install_snapshot`)
is legitimate and kept. Candidates surfaced so far: `test_inner_retirement` →
e.g. `test_inner_unwrap_retired`; `test_visualize_aux` → e.g.
`test_visualize_matplotlib`; `test_cap_regression_impossible` (borderline).
Naming is subjective, so the **spec's mapping table lists every proposed rename
(old → new + one-line reason) for explicit approval before execution** — no
unilateral renames.

### Execution: a deterministic script, not a workflow

The bulk is a mechanical transform a small Python script performs exactly and
instantly — `git mv` per the approved mapping, the uniform `.` → `..` import
rewrite, and writing `__init__.py` per subdir. This is **not** a good fit for an
LLM fan-out (cheap or otherwise): there is one global mapping decision (not 156
independent ones), verification is the whole-suite collection (not per-file), and
an LLM would only add the risk of a wrong import depth to an otherwise exact
transform. The judgment lives entirely in building the mapping table (done once,
reviewed); the script executes it.

### The mapping table is the spec deliverable

A single curated table, `old_relpath → new_relpath`, covering all 156 files —
incorporating both relocation and the approved renames. The script consumes it.
Building and reviewing this table (including the rename proposals and the
root-vs-subpackage-vs-integration bucket for each file) is the first plan task
and the human review gate.

## Non-goals

- **No test-logic changes** — not a single assertion moved or altered.
- **No file splits or merges** — `test_big_phi.py` stays one mixed-formalism
  file (per the earlier rename-only decision); we relocate, not re-carve.
- **No fixture or `conftest` behavior changes.**
- **No deeper-than-one-level mirroring.**
- **Not a ship gate.**

## Risks

- **`__file__`-relative paths (~8 files) — the main residual risk.** Files using
  `Path(__file__).parent / "data" / …` break when moved (the `.parent` shifts to
  the new subdir). Each needs a depth-aware fixup — add one `.parent` for the
  depth-1 move, or re-anchor to the test root. `test_presets.py`'s
  `parent.parent` reach to a repo-root config shifts similarly. The script
  **flags** every `__file__`-using file in the move set; these are fixed by hand
  (small number) and caught regardless by the collection/suite gate.
- **Cross-references** — docstrings/comments naming another test file (as
  `test_big_phi.py` named the old `robust` file). The script greps for every
  renamed/moved basename and updates the references.
- **Concurrent instances** — other Claude instances work this repo; a 156-file
  move is a large, conflict-prone diff. Execute on a branch from a clean working
  tree, in one focused pass, and stage only the move set.
- **Collection drift** — a missed import or a stray `__init__.py` could silently
  drop a test. Guard with the collected-count invariant (below).

## Verification

- **Collection invariant.** Capture `pytest --collect-only -q` (the exact set
  and count of test node IDs, basename-normalized) *before* the move; assert the
  *same set* after. This is the primary correctness gate — it catches any test
  silently dropped or un-collected.
- **Full suite** `uv run --all-extras pytest` (no path argument — the `pyphi/`
  doctest sweep runs) green, with the same pass/skip numbers as the pre-move
  baseline.
- **pyright + ruff** green (the import rewrites must satisfy isort/F401).
- **References** — `ROADMAP.md` / docs that point at specific `test/…` paths
  updated.

## Out of scope / deferred

- Splitting `test_big_phi.py` by formalism (a content change, declined earlier).
- A `unit/` vs `integration/` top-level split (the unit tests are
  module-mirrored, not under a `unit/` dir — intentional).
- Mirroring nested source subpackages below depth 1.
