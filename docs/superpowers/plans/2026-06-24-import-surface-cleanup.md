# Import-surface cleanup + dead-dependency drops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the eager recursive submodule walk in `pyphi/__init__.py` with explicit registry-population imports plus a PEP-562 lazy `__getattr__`, guarded by a registry-contents test; and drop the `tblib`, `ordered-set`, and `toolz` dependencies.

**Architecture:** A registry-contents test is written first and passes against the current code, pinning what every registry must contain. Then the walk is replaced with seven explicit registrant-module imports (the audit proved all 53 decorator registrations are co-located with their registry) and a `__getattr__` that lazily imports submodules on first access. The three dependency drops are independent: `tblib` is unreferenced, `ordered-set` backs only dead code, and `toolz`'s three used functions map onto stdlib / `more_itertools`.

**Tech Stack:** Python 3.12+, pytest (with doctest sweep), `more_itertools` (already a dependency), `itertools`, `functools`, ruff + pyright via pre-commit.

## Global Constraints

- **Python 3.12+ only.** No backward-compatibility shims for older Python.
- **No planning-artifact markers** (P-numbers, "Wave N", task numbers, `pre-PXX`) in `pyphi/` source, docstrings, or changelog fragments.
- **Branch `import-surface-cleanup`**, off `origin/main`; this ships as one PR into `main` (CI-gated: lint, build, tests on ubuntu/macos/windows × 3.13). **Ask before `git push`.**
- **Stage only the files named in each task** (`git add <paths>`); never `git add -A` — concurrent instances commit to the same trunk. Leave `.claude/`, `graphify-out/`, and untracked scratch alone.
- **Pre-commit = ruff + pyright; never `--no-verify`.** Ruff reformats and aborts the commit when it changes a file — re-`git add` the named files and re-commit. One import per line (I001); mark side-effect imports `# noqa: F401`.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Full verification = `uv run pytest` with NO path argument** (collects the `pyphi/` doctest sweep). Use `uv run --all-extras pytest` to match CI (extras must be present for the doctest imports).

---

### Task 1: Registry-contents test (the safety net)

Write this first. It passes against the current code and must keep passing after the walk is removed — that is what makes removing the walk safe.

**Files:**
- Create: `test/test_registry_contents.py`

**Interfaces:**
- Consumes: the live registries on the imported `pyphi` package.
- Produces: `EXPECTED_REGISTRY_CONTENTS` mapping (registry accessor → expected key set), reused as the canonical pinned list.

- [ ] **Step 1: Write the test**

```python
# test/test_registry_contents.py
"""Pin the built-in contents of every registry.

If a registrant module stops being imported during package initialization,
its registry loses entries and the matching assertion here fails loudly,
instead of a measure or scheme silently vanishing.
"""

import pytest

import pyphi

EXPECTED_REGISTRY_CONTENTS = {
    "partition.partition_types": {
        "JOINT_BIPARTITION",
        "JOINT_PARTITION_ALL",
        "WEDGE_TRIPARTITION",
    },
    "partition.system_partition_types": {
        "DIRECTED_BIPARTITION",
        "DIRECTED_BIPARTITION_CUT_ONE",
        "DIRECTED_BIPARTITION_SEQUENTIAL",
        "DIRECTED_SET_PARTITION",
        "EDGE_CUT_ALL",
        "EDGE_CUT_BIDIRECTIONAL",
        "TEMPORAL_DIRECTED_BIPARTITION",
        "TEMPORAL_DIRECTED_BIPARTITION_CUT_ONE",
    },
    "relations.relation_computations": {"ANALYTICAL", "CONCRETE"},
    "resolve_ties.phi_object_tie_resolution_strategies": {
        "NEGATIVE_NORMALIZED_PHI",
        "NEGATIVE_PHI",
        "NEGATIVE_PURVIEW_SIZE",
        "NONE",
        "NORMALIZED_PHI",
        "PARTITION_LEX",
        "PHI",
        "PURVIEW_SIZE",
    },
    "models.state_specification.distinction_phi_normalizations": {
        "NONE",
        "NUM_CONNECTIONS_CUT",
    },
    "measures.ces.measures": {"EMD", "SUM_SMALL_PHI"},
    "measures.distribution.distribution_measures": {
        "AID",
        "BLD",
        "EMD",
        "ENTROPY_DIFFERENCE",
        "ID",
        "KLD",
        "KLM",
        "L1",
        "MP2Q",
        "PSQ2",
    },
    "measures.distribution.stateful_distribution_measures": {
        "APMI",
        "IIT_4.0_SMALL_PHI",
        "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
    },
    "measures.distribution.state_aware_measures": {"INTRINSIC_DIFFERENTIATION"},
    "measures.distribution.composite_measures": {
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
        "INTRINSIC_SPECIFICATION",
    },
    "measures.distribution.actual_causation_measures": {"PMI", "WPMI"},
    "formalism.actual_causation.compute.partitioned_repertoire_schemes": {"PRODUCT"},
    "formalism.actual_causation.compute.background_strategies": {"UNIFORM"},
    "formalism.actual_causation.compute.alpha_aggregations": {"SUBTRACTIVE"},
    "formalism.FORMALISM_REGISTRY": {"IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"},
    "formalism.ACTUAL_CAUSATION_FORMALISM_REGISTRY": {"AC_2019"},
}


def _resolve(dotted: str):
    obj = pyphi
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("dotted_path, expected", EXPECTED_REGISTRY_CONTENTS.items())
def test_registry_contents(dotted_path, expected):
    registry = _resolve(dotted_path)
    assert set(registry.keys()) == expected
```

- [ ] **Step 2: Run it — expect PASS against current code**

Run: `uv run --all-extras pytest test/test_registry_contents.py -v`
Expected: PASS (every registry already populated by the current eager walk).

- [ ] **Step 3: Commit**

```bash
git add test/test_registry_contents.py
git commit  # message: "Pin built-in registry contents with a contents test"
```

---

### Task 2: Retire the eager submodule walk

**Files:**
- Modify: `pyphi/__init__.py` (replace lines 68–135: the imports block, `_skip_import`, `_import_submodules`, `_submodules`, and the dynamic `__all__`)
- Create: `test/test_import_surface.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except the Task 1 test, which must keep passing.
- Produces: a module-level `__getattr__(name) -> ModuleType` on `pyphi`, and a static `__all__`.

- [ ] **Step 1: Write the failing import-surface test**

```python
# test/test_import_surface.py
"""The top-level package imports lazily and exposes submodules on demand."""

import os
import subprocess
import sys

import pytest

import pyphi


def test_known_submodule_resolves_lazily():
    # examples is not a registrant module, so it is imported only on access.
    assert pyphi.examples.__name__ == "pyphi.examples"


def test_unknown_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        pyphi.definitely_not_a_real_submodule


def test_import_pyphi_does_not_eagerly_import_peripheral_submodules():
    # The robustness property: a broken or heavy peripheral submodule cannot
    # break `import pyphi`, because it is not imported until accessed.
    code = (
        "import sys, pyphi; "
        "leaked = [m for m in "
        "('pyphi.visualize', 'pyphi.examples', 'pyphi.macro', 'pyphi.matching') "
        "if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYPHI_WELCOME_OFF": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run it — expect FAIL on the eager-import test**

Run: `uv run --all-extras pytest test/test_import_surface.py -v`
Expected: `test_import_pyphi_does_not_eagerly_import_peripheral_submodules` FAILS (the current walk imports `examples`, `macro`, `matching` eagerly; `visualize` is skipped). The other two may already pass.

- [ ] **Step 3: Rewrite the import machinery in `pyphi/__init__.py`**

Replace everything from `import importlib` (line 68) through the end of the dynamic `__all__` block (line 135) with:

```python
import importlib
import os
import pkgutil
import types

# Lift main interfaces to the top-level namespace.
from .actual import Transition
from .actual import TransitionSystem
from .conf import config
from .conf import iit3
from .conf import iit4_2023
from .conf import iit4_2026
from .core.tpm import FactoredTPM as FactoredTPM
from .core.tpm import JointDistribution as JointDistribution
from .core.tpm.joint_distribution import JointTPM
from .direction import Direction
from .substrate import Substrate
from .system import System

# Populate the registries. Each built-in measure, partition scheme,
# tie-resolution strategy, relation computation, distinction normalization,
# and formalism is registered by a decorator (or an explicit ``.register``
# call) that runs when its defining module is imported. Importing these
# modules makes every built-in registrant available. Third-party plugins
# register when the user imports them.
import pyphi.measures.ces  # noqa: F401
import pyphi.measures.distribution  # noqa: F401
import pyphi.models.state_specification  # noqa: F401

from . import formalism  # noqa: F401
from . import partition  # noqa: F401
from . import relations  # noqa: F401
from . import resolve_ties  # noqa: F401

# Names of the public depth-1 submodules, available as attributes via the
# lazy ``__getattr__`` below. Computed by listing (not importing) submodules.
_PUBLIC_SUBMODULES = frozenset(
    name
    for _, name, _ in pkgutil.iter_modules(__path__)
    if not name.startswith("_")
)


def __getattr__(name: str) -> types.ModuleType:
    """Lazily import a public submodule on first attribute access (PEP 562).

    Keeps ``pyphi.examples``, ``pyphi.compute``, and the like working after a
    bare ``import pyphi`` without importing the whole package eagerly, so
    ``import pyphi`` is fast and is not broken by an unrelated submodule that
    fails to import.
    """
    if name in _PUBLIC_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Direction",
    "FactoredTPM",
    "JointDistribution",
    "JointTPM",
    "Substrate",
    "System",
    "Transition",
    "TransitionSystem",
    "config",
    "iit3",
    "iit4_2023",
    "iit4_2026",
    *sorted(_PUBLIC_SUBMODULES),
]
```

(The welcome-message block below line 135 stays unchanged.)

- [ ] **Step 4: Run the import-surface and registry tests — expect PASS**

Run: `uv run --all-extras pytest test/test_import_surface.py test/test_registry_contents.py -v`
Expected: all PASS. If a registry assertion fails, a registrant module is no longer reached — add the missing explicit import (most likely `import pyphi.formalism.actual_causation.compute`).

- [ ] **Step 5: Run the full suite (no path argument) — the doctest sweep matters here**

Run: `uv run --all-extras pytest`
Expected: PASS. The no-path run collects `pyphi/` doctests, catching any doctest that relied on a submodule being eagerly present.

- [ ] **Step 6: Add a changelog fragment**

```bash
echo '`import pyphi` no longer eagerly imports every submodule: submodules are imported lazily on first attribute access, so import is faster and is not broken by an unrelated submodule that fails to import.' > changelog.d/import-surface.change.md
```

- [ ] **Step 7: Commit**

```bash
git add pyphi/__init__.py test/test_import_surface.py changelog.d/import-surface.change.md
git commit  # message: "Retire eager submodule walk for explicit imports + lazy __getattr__"
```

---

### Task 3: Drop the unused `tblib` dependency

**Files:**
- Modify: `pyproject.toml` (remove the `"tblib>=1.3.2",` line, ~line 41)

- [ ] **Step 1: Confirm there are no references in source**

Run: `grep -rn tblib pyphi test --include="*.py"`
Expected: no output.

- [ ] **Step 2: Remove the dependency line from `pyproject.toml`**

Delete the line:
```
    "tblib>=1.3.2",
```

- [ ] **Step 3: Re-sync and smoke-test the import**

Run: `uv sync --all-extras && uv run python -c "import pyphi" && uv run --all-extras pytest test/test_import_surface.py -q`
Expected: sync succeeds without `tblib`; import works; tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit  # message: "Drop the unused tblib dependency"
```

---

### Task 4: Drop the `ordered-set` dependency (delete the dead module)

`OrderedSet` and `HashableOrderedSet` are exported from `pyphi/data_structures/__init__.py` but constructed nowhere in the codebase. They were a parked idea for representing units in purviews/mechanisms; a sorted tuple is the better representation there (canonical sorted order for array indexing, immutable and cheaply hashable, far lighter on the hot path), so the dead code and the dependency are removed.

**Files:**
- Delete: `pyphi/data_structures/hashable_ordered_set.py`
- Modify: `pyphi/data_structures/__init__.py` (remove the two re-export lines)
- Modify: `pyproject.toml` (remove the `ordered-set` dependency line)

- [ ] **Step 1: Confirm nothing constructs or imports these types**

Run: `grep -rn "OrderedSet" pyphi test --include="*.py" | grep -v "data_structures/hashable_ordered_set.py\|data_structures/__init__.py"`
Expected: no output.

- [ ] **Step 2: Delete the module and the re-exports**

```bash
git rm pyphi/data_structures/hashable_ordered_set.py
```

In `pyphi/data_structures/__init__.py`, remove these two lines:
```python
from ordered_set import OrderedSet as OrderedSet
```
```python
from .hashable_ordered_set import HashableOrderedSet as HashableOrderedSet
```
If `OrderedSet` / `HashableOrderedSet` appear in an `__all__` in that file, remove those entries too.

- [ ] **Step 3: Remove the `ordered-set` dependency from `pyproject.toml`**

Delete the `ordered-set` (or `ordered_set`) dependency line.

- [ ] **Step 4: Re-sync, import, and run the data-structures and import tests**

Run: `uv sync --all-extras && uv run --all-extras pytest test/ -k "data_structure or import_surface" -q && uv run python -c "import pyphi"`
Expected: PASS; import works. (The deleted module's own doctests disappear with it.)

- [ ] **Step 5: Add a changelog fragment**

```bash
echo 'Removed the unused `OrderedSet`/`HashableOrderedSet` data structures and dropped the `ordered-set` dependency.' > changelog.d/drop-ordered-set.change.md
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/data_structures/__init__.py pyproject.toml uv.lock changelog.d/drop-ordered-set.change.md
git commit  # message: "Remove unused OrderedSet types and drop the ordered-set dependency"
```

---

### Task 5: Migrate `toolz.concat` → `itertools.chain.from_iterable`

`toolz.concat(seqs)` is exactly `itertools.chain.from_iterable(seqs)`. Call sites: `models/fmt.py:218,335`, `models/ria.py:419`, `models/distinctions.py:190`, `models/mice.py:188,212`. (The `pd.concat` calls in `models/state_specification.py` and `visualize/distribution.py` are pandas — leave them.)

**Files:**
- Modify: `pyphi/models/fmt.py`, `pyphi/models/ria.py`, `pyphi/models/distinctions.py`, `pyphi/models/mice.py`

- [ ] **Step 1: In each file, replace the import**

Replace `from toolz import concat` with `from itertools import chain` (place it with the other stdlib imports; if `chain` is already imported, drop the `toolz` line).

- [ ] **Step 2: Replace each `concat(` call with `chain.from_iterable(`**

In each of the six call sites, change `concat(...)` to `chain.from_iterable(...)`. For example in `models/distinctions.py:190`:
```python
return chain.from_iterable([concept.cause, concept.effect] for concept in self)
```
In `models/mice.py:188`:
```python
for tie in chain.from_iterable(filter(None, [self.partition_ties, self.purview_ties])):
```
(`models/ria.py:419` also uses `unique`; that line is finished in Task 6.)

- [ ] **Step 3: Run the model tests**

Run: `uv run --all-extras pytest test/ -k "fmt or ria or distinction or mice or display" -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pyphi/models/fmt.py pyphi/models/ria.py pyphi/models/distinctions.py pyphi/models/mice.py
git commit  # message: "Replace toolz.concat with itertools.chain.from_iterable"
```

---

### Task 6: Migrate `toolz.unique` → `more_itertools.unique_everseen`

`toolz.unique(seq)` yields the first occurrence of each item in order; `more_itertools.unique_everseen(seq)` is the drop-in equivalent (and supports `key=` identically if ever needed). Call sites: `partition.py:884`, `models/ria.py:419`. Neither uses a `key=` argument.

**Files:**
- Modify: `pyphi/partition.py`, `pyphi/models/ria.py`

- [ ] **Step 1: Replace the imports**

In `pyphi/partition.py`, replace `from toolz import unique` with `from more_itertools import unique_everseen` (merge with the existing `from more_itertools import distinct_permutations` line region; keep one import per line).
In `pyphi/models/ria.py`, replace `from toolz import unique` with `from more_itertools import unique_everseen`.

- [ ] **Step 2: Replace each `unique(` call with `unique_everseen(`**

`pyphi/partition.py:884`:
```python
yield from unique_everseen(_directed_set_partitions(node_indices, node_labels=node_labels))
```
`pyphi/models/ria.py:419` (combined with Task 5's `chain`):
```python
return unique_everseen(chain.from_iterable([self._state_ties, self._partition_ties]))
```

- [ ] **Step 3: Run the partition and ria tests**

Run: `uv run --all-extras pytest test/ -k "partition or ria" -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pyphi/partition.py pyphi/models/ria.py
git commit  # message: "Replace toolz.unique with more_itertools.unique_everseen"
```

---

### Task 7: Migrate `toolz.curry` and drop the `toolz` dependency

Two `utils.py` functions are bound by partial application (`all_same(eq)`); the four substrate-generator unit functions are only ever called fully (verified across `pyphi/`, `test/`, `docs/`), with per-unit configuration already supported by `build_tpm`'s `**kwargs` passthrough or an explicit `functools.partial`. So: drop `@curry` everywhere; convert the four `utils.py` module-level bindings to `functools.partial`; the substrate-generator functions need no call-site changes.

**Files:**
- Modify: `pyphi/utils.py`, `pyphi/substrate_generator/unit_functions.py`, `pyphi/substrate_generator/ising.py`, `pyproject.toml`

- [ ] **Step 1: `pyphi/utils.py` — drop `@curry`, use `functools.partial`**

- Replace `from toolz import curry` with `import functools` (if `functools` is already imported, just remove the `toolz` line).
- Remove the `@curry` decorator above `all_same` (~line 410) and above `all_extrema` (~line 430).
- Change the four module-level bindings to `functools.partial` and drop the now-unneeded pyright-ignore comments:
```python
all_are_equal = functools.partial(all_same, eq)
all_are_identical = functools.partial(all_same, operator.is_)
```
```python
all_minima = functools.partial(all_extrema, operator.lt)
all_maxima = functools.partial(all_extrema, operator.gt)
```

- [ ] **Step 2: `pyphi/substrate_generator/unit_functions.py` and `ising.py` — drop `@curry`**

- In `unit_functions.py`: remove `from toolz import curry` and the three `@curry` decorators (above `naka_rushton`, `boolean_function`, `gaussian`).
- In `ising.py`: remove `from toolz import curry` and the `@curry` decorator above `probability`.
- No call-site changes: `build_tpm` calls these with all positional arguments.

- [ ] **Step 3: Confirm no `toolz` references remain, then drop the dependency**

Run: `grep -rn toolz pyphi --include="*.py"`
Expected: no output.
Then remove the `toolz` dependency line from `pyproject.toml`.

- [ ] **Step 4: Run the affected tests**

Run: `uv run --all-extras pytest test/ -k "utils or substrate_generator or substrate or ising" -q`
Expected: PASS (existing `test_substrate_generator.py` exercises `naka_rushton`, `gaussian`, and `ising.probability`; `utils` tests exercise `all_minima`/`all_are_equal`).

- [ ] **Step 5: Re-sync and add a changelog fragment**

```bash
uv sync --all-extras
echo 'Dropped the `toolz` dependency, replacing its uses with `itertools`, `more_itertools`, and `functools`.' > changelog.d/drop-toolz.change.md
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/utils.py pyphi/substrate_generator/unit_functions.py pyphi/substrate_generator/ising.py pyproject.toml uv.lock changelog.d/drop-toolz.change.md
git commit  # message: "Drop the toolz dependency for stdlib and more_itertools"
```

---

### Task 8: Final verification and ROADMAP update

**Files:**
- Modify: `ROADMAP.md` (B17 dashboard row → landed; note P15 import-cleanup progress)

- [ ] **Step 1: Full suite, no path argument**

Run: `uv run --all-extras pytest`
Expected: PASS (this is the load-bearing gate — it runs the `pyphi/` doctest sweep).

- [ ] **Step 2: Pre-commit and pyright**

Run: `SKIP=pyright uv run pre-commit run --all-files && uv run pyright pyphi`
Expected: clean. (If ruff reformats, re-`git add` and amend the relevant commit.)

- [ ] **Step 3: Confirm the install closure shrank**

Run: `uv run python -c "import importlib.util as u; print([d for d in ('tblib','ordered_set','toolz') if u.find_spec(d)])"`
Expected: `[]` (none resolvable in the synced environment).

- [ ] **Step 4: Update the ROADMAP Status Dashboard**

In `ROADMAP.md`, set the **B17 drop dead deps** row status to `✅ landed` (all three dependencies removed). Add a landed-detail bullet in the Wave-5 section noting the import-walk retirement and the `tblib`/`ordered-set`/`toolz` drops, and that the rest of P15 (jsonify→msgspec, docstring sweep, test reorg, to_pandas, PR triage, docs, changelog) remains open. Do not mark the P15 row landed.

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md
git commit  # message: "Record import-surface cleanup and B17 dep drops in ROADMAP"
```

- [ ] **Step 6: Finish the branch**

Use the superpowers:finishing-a-development-branch skill: verify tests pass, then present the merge/PR options. This sub-project ships as a PR into `main`. **Ask before pushing.**

---

## Notes for the implementer

- **Verification gate:** only `uv run pytest` with no path argument runs the `pyphi/` doctest sweep. The `-k`-filtered runs in individual tasks are for fast inner-loop feedback; Task 8 Step 1 is the real gate.
- **If a registry assertion fails in Task 2:** a registrant module is not being reached. Add the specific missing `import pyphi.<module>` to the registry-population block in `pyphi/__init__.py`. The most likely candidate is `import pyphi.formalism.actual_causation.compute` if importing `pyphi.formalism` does not transitively reach it.
- **Ruff import ordering (I001):** one import per line, sorted; stdlib, third-party, then local groups. Side-effect imports carry `# noqa: F401`.
