# ANALYTICAL Relation-Computation Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ANALYTICAL` the default `relation_computation`, with a viz default of the strongest 1000 relations, matching materializing analytical relations internally, and explicit `CONCRETE` pins for tests that enumerate relations.

**Architecture:** Land the two consumer-side accommodations first (viz default-k, matching materialize-on-first-use), each green under an explicit `ANALYTICAL` override; then flip the field default, triage the suite, and pin the concrete-needing tests in the same commit; finish with the docs sweep and changelog. Spec: `docs/superpowers/specs/2026-07-18-analytical-relations-default-design.md`.

**Tech Stack:** Python 3.13, pytest, uv, jupytext (tutorial sync), towncrier changelog fragments.

## Global Constraints

- Work in a git worktree under `.claude/worktrees/` (superpowers:using-git-worktrees). Venv recipe: `uv venv`, then `WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"; env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray,mcp]" 'pot==0.9.6.post1'` — the pot pin is mandatory (0.9.7 fails import under `filterwarnings=error`).
- Completion gate is **pathless** `uv run pytest` (redirect to a log file and read the summary line — never pipe through `tail`, never trust exit codes alone). Slow lane: `uv run pytest -m slow --slow`.
- Commit messages end with the two trailers exactly:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe`
  (watch for the recurring `claude.ac` typo — must be `claude.ai`). Check `git log --oneline -1` after every commit: pre-commit hooks abort silently and ruff-format aborts leave files staged (re-add and re-commit). Never `--no-verify`.
- NumPy-style, final-state docstrings; no planning-artifact references (no "per spec", no phase names) in code, docstrings, or changelog. Docs tone neutral (theory/performance-motivated, no "not a bug" framing). Unicode `φ` is fine in docstrings.
- Do NOT touch `docs/whats-new-in-2.0.md` (held by a concurrent session), nor the untracked `REVIEW-2026-07-13.md`, `TRIAGE-WAVE5.md`, `experiments/`, or benchmark JSONs.
- After editing a `docs/tutorials/*.md` file, run `uv run jupytext --sync <file>` so the paired `.ipynb` stays in step, and stage both.
- zsh: never put a bare `=`-leading word (e.g. `echo ===`) in a Bash command.

---

### Task 1: Viz default `max_relations` for the analytical backend

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py` (constant + `project_ces` branch ~line 276 + docstring ~lines 239–258)
- Modify: `pyphi/visualize/__init__.py` (docstrings: `plot_ces` `max_relations` param ~line 139, `highlight_phi_fold` ~line 245)
- Modify: `pyphi/mcp/server.py` (`max_relations` tool-doc paragraph ~line 584)
- Test: `test/visualize/test_visualize_projection.py`

**Interfaces:**
- Produces: `pyphi.visualize.projection.DEFAULT_MAX_ANALYTICAL_RELATIONS: int = 1000`, imported by tests. `project_ces(ces, node_labels=None, max_relations=None)` no longer raises for a non-enumerable relation set with `max_relations=None`; it renders the strongest 1000.

- [ ] **Step 1: Rewrite the raise-test as the default-k test**

In `test/visualize/test_visualize_projection.py`, replace `test_project_ces_analytical_requires_cap` (line ~259) with:

```python
def test_project_ces_analytical_defaults_to_strongest_1000():
    from pyphi.relations import AnalyticalRelations
    from pyphi.visualize.projection import (
        DEFAULT_MAX_ANALYTICAL_RELATIONS,
        project_ces,
    )

    ces = _xor_ces(analytical=True)
    assert isinstance(ces.relations, AnalyticalRelations)  # precondition
    projection = project_ces(ces)
    assert 0 < len(projection.edges) <= DEFAULT_MAX_ANALYTICAL_RELATIONS
    # The default draws the same relations the concrete backend would
    # (xor's relation set is far below the cap).
    concrete = project_ces(_xor_ces(analytical=False))
    assert {e.relata for e in projection.edges} == {e.relata for e in concrete.edges}
```

And in `test_plot_ces_forwards_max_relations` (line ~289), delete the two lines asserting the raise:

```python
        with pytest.raises(ValueError, match="max_relations"):
            plot_ces(ces)
```

replacing them with a no-argument call that must now succeed:

```python
        fig_default = plot_ces(ces)
        assert fig_default is not None
```

- [ ] **Step 2: Run to verify the new test fails**

Run: `uv run pytest test/visualize/test_visualize_projection.py -q > /tmp/t1a.log 2>&1; cat /tmp/t1a.log`
Expected: `test_project_ces_analytical_defaults_to_strongest_1000` and `test_plot_ces_forwards_max_relations` FAIL with `ValueError` ("relations are not enumerable"); the rest pass.

- [ ] **Step 3: Implement the default**

In `pyphi/visualize/projection/__init__.py`, add below the imports (near the top-level constants):

```python
DEFAULT_MAX_ANALYTICAL_RELATIONS = 1000
"""Relations rendered when ``max_relations`` is None and the relation set is
not enumerable (the analytical backend): the strongest this-many by φ_r."""
```

Replace the branch in `project_ces` (~line 276):

```python
    if max_relations is None:
        try:
            iter(ces.relations)
        except TypeError:
            raise ValueError(
                "relations are not enumerable (analytical backend); pass "
                "max_relations=N to render the strongest N relations by φ_r"
            ) from None
    top = list(ces.relations.strongest(k=max_relations))
```

with:

```python
    if max_relations is None:
        try:
            iter(ces.relations)
        except TypeError:
            max_relations = DEFAULT_MAX_ANALYTICAL_RELATIONS
    top = list(ces.relations.strongest(k=max_relations))
```

Update the `project_ces` docstring: the `max_relations` parameter text becomes

```
    max_relations : int, optional
        Render only the ``max_relations`` strongest relations (and their faces),
        in descending φ_r order. If None, render every relation when the set is
        enumerable; when it is not (the analytical backend), render the
        strongest ``DEFAULT_MAX_ANALYTICAL_RELATIONS``. Node marker sizes and
        the degree spectrum are always computed over the full structure,
        independent of this cap.
```

and in its `Raises` section drop the clause "or if ``max_relations`` is None and ``ces.relations`` is not enumerable" (the `ValueError` entry keeps only the empty-structure case).

- [ ] **Step 4: Update the forwarding docstrings**

`pyphi/visualize/__init__.py`, `plot_ces` `max_relations` entry (~line 139) becomes:

```
    max_relations : int, optional
        Render only the strongest ``max_relations`` relations by φ_r. If None,
        every relation is rendered when the set is enumerable; analytically
        computed relations (not enumerable) default to the strongest 1000.
        Node sizes and the spectrum view remain exact regardless.
```

`highlight_phi_fold` entry (~line 245) becomes:

```
    max_relations : int, optional
        Render only the strongest ``max_relations`` relations by φ_r; defaults
        to the strongest 1000 for analytically-computed relations.
```

`pyphi/mcp/server.py` `max_relations` tool-doc paragraph (~line 584) — replace the "Required when …" sentence so the paragraph reads:

```
    max_relations : int, optional
        For ``kind="ces"`` only, draw just the strongest this-many relations by
        φ_r. When the structure's relations are computed analytically
        (``relation_computation="ANALYTICAL"``), whose relation set cannot be
        enumerated, ``None`` draws the strongest 1000; node sizes and the
        spectrum view stay exact regardless. With enumerable (``"CONCRETE"``)
        relations, ``None`` draws every relation. For the full direct-Python
        surface see ``get_iit_reference("visualization")``.
```

- [ ] **Step 5: Run the viz tests to verify they pass**

Run: `uv run pytest test/visualize/ test/mcp/ -q > /tmp/t1b.log 2>&1; cat /tmp/t1b.log`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyphi/visualize/projection/__init__.py pyphi/visualize/__init__.py pyphi/mcp/server.py test/visualize/test_visualize_projection.py
git commit -m "Default to the strongest 1000 relations when plotting analytical relation sets"
```

(Trailers per Global Constraints; verify with `git log --oneline -1`.)

---

### Task 2: Matching materializes analytical relations on first use

**Files:**
- Modify: `pyphi/matching/perception.py` (new `_relations` cached property; `richness` ~line 111; class docstring)
- Modify: `pyphi/matching/differentiation.py` (`_component_perceptions` ~lines 14–29; `analytical_differentiation` docstring sentence ~line 85)
- Test: `test/matching/test_perception.py`

**Interfaces:**
- Consumes: `pyphi.relations.AnalyticalRelations.materialize() -> ConcreteRelations` (existing).
- Produces: `Perception._relations` — cached property returning the enumerable relation set (`ConcreteRelations`); `_component_perceptions` and `richness` read it. `Differentiation.perceptual_differentiation` and `.differentiation` now work when structures carry `AnalyticalRelations`.

- [ ] **Step 1: Pin the existing module fixture and add the failing parity tests**

In `test/matching/test_perception.py`, the module fixture `perception` computes its CES under `presets.iit4_2026`, which inherits the ambient relation backend; the module's tests verify per-relation formulas directly, so pin the concrete backend explicitly. Change the `with` line in the `perception` fixture to:

```python
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(relation_computation="CONCRETE"),
    ):
```

Then add at the end of the file:

```python
@pytest.fixture(scope="module")
def analytical_perception():
    """The same stimulus/system as ``perception``, with analytical relations."""
    from pyphi.relations import AnalyticalRelations

    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    stimulus = (1,)
    y = ttpm.argmax_state(stimulus)
    with (
        pyphi.config.override(**presets.iit4_2026),
        pyphi.config.override(relation_computation="ANALYTICAL"),
    ):
        ces = substrate.ces(
            state=_full_state(sensory, system, stimulus, y), indices=system
        )
        assert isinstance(ces.relations, AnalyticalRelations)  # precondition
    return Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)


def test_richness_analytical_matches_concrete(perception, analytical_perception):
    assert analytical_perception.richness == pytest.approx(perception.richness)


def test_perceptual_differentiation_accepts_analytical(
    perception, analytical_perception
):
    from pyphi.matching import Differentiation

    d_concrete = Differentiation((perception,)).perceptual_differentiation
    d_analytical = Differentiation((analytical_perception,)).perceptual_differentiation
    assert d_analytical == pytest.approx(d_concrete)
```

(If `Differentiation` is not exported from `pyphi.matching`, import it from `pyphi.matching.differentiation` instead — check `pyphi/matching/__init__.py`, which imports it at line 15.)

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest test/matching/test_perception.py -q > /tmp/t2a.log 2>&1; cat /tmp/t2a.log`
Expected: `test_richness_analytical_matches_concrete` FAILS with `TypeError` ("cannot be iterated") from `richness`; `test_perceptual_differentiation_accepts_analytical` FAILS with `TypeError` ("concrete differentiation requires iterable"); the rest pass.

- [ ] **Step 3: Implement `Perception._relations` and reroute the two sites**

In `pyphi/matching/perception.py`, add to the `Perception` class (after `triggering_coefficients`):

```python
    @cached_property
    def _relations(self):
        """The enumerable relation set of ``ces``.

        Per-relation perception values (Eqs. 9-13, 19) require individual
        relations; an analytical relation summary is materialized here on
        first use, at the same cost as computing the structure with the
        concrete backend.
        """
        from pyphi.relations import AnalyticalRelations

        relations = self.ces.relations
        if isinstance(relations, AnalyticalRelations):
            return relations.materialize()
        return relations
```

In `richness`, replace the relation sum:

```python
        relations = sum(
            self.relation_perception(r)
            for r in self.ces.relations  # pyright: ignore[reportGeneralTypeIssues]  # Relations base lacks __iter__; concrete subclasses provide it
        )
```

with:

```python
        relations = sum(self.relation_perception(r) for r in self._relations)
```

Add one sentence to the `ces` attribute description in the `Perception` class docstring: "An analytical relation summary is materialized on first use of any per-relation quantity."

In `pyphi/matching/differentiation.py`, replace `_component_perceptions`'s body so the analytical guard disappears:

```python
def _component_perceptions(perception):
    """Yield (component, perception) for each component of one structure."""
    for distinction in perception.ces.distinctions:
        yield distinction, perception.distinction_perception(distinction)
    for relation in perception._relations:
        yield relation, perception.relation_perception(relation)
```

(The `from pyphi.relations import AnalyticalRelations` import and the `raise TypeError` block are deleted with it.)

In `analytical_differentiation`'s docstring, the sentence "…and is the path to use when the structures carry ``AnalyticalRelations`` (which cannot be iterated)" becomes: "…and is the cheap path when the structures carry ``AnalyticalRelations``: it never enumerates a relation, where the concrete properties would first materialize the relation sets."

- [ ] **Step 4: Run the matching tests to verify they pass**

Run: `uv run pytest test/matching/ -q > /tmp/t2b.log 2>&1; cat /tmp/t2b.log`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/perception.py pyphi/matching/differentiation.py test/matching/test_perception.py
git commit -m "Materialize analytical relations on first use in matching perception"
```

---

### Task 3: Flip the default, add default-path tests, pin the suite

**Files:**
- Modify: `pyphi/conf/formalism.py:108`
- Modify: `test/conf/test_config_layers.py` (new assertion test after `test_default_formalism_is_iit4_2026`, line ~618)
- Create: `test/test_analytical_default.py`
- Modify: test files identified by the triage run (see Step 6; known in advance: `test/visualize/test_visualize_projection.py::_xor_ces`, `test/test_relations.py::_pin_formalism`)

**Interfaces:**
- Consumes: Tasks 1–2 (viz default and matching materialization must be in place before the flip, or their test modules fail).
- Produces: `IITConfig.relation_computation` default `"ANALYTICAL"`; the pin idiom `config.override(relation_computation="CONCRETE")` used by later docs examples.

- [ ] **Step 1: Write the failing default-assertion test**

In `test/conf/test_config_layers.py`, immediately after `test_default_formalism_is_iit4_2026`:

```python
def test_default_relation_computation_is_analytical():
    """The library default relation backend. As with the formalism default
    above, do NOT pin a config here — this must observe the real default."""
    from pyphi.conf import config

    assert config.formalism.iit.relation_computation == "ANALYTICAL"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/conf/test_config_layers.py::test_default_relation_computation_is_analytical -q > /tmp/t3a.log 2>&1; cat /tmp/t3a.log`
Expected: FAIL — `'CONCRETE' == 'ANALYTICAL'` assertion error.

- [ ] **Step 3: Flip the default**

`pyphi/conf/formalism.py` line 108:

```python
    relation_computation: str = "ANALYTICAL"
```

Run: `uv run pytest test/conf/ -q > /tmp/t3b.log 2>&1; cat /tmp/t3b.log`
Expected: all pass (including the new assertion test).

- [ ] **Step 4: Add the default-path integration tests**

Create `test/test_analytical_default.py`:

```python
"""Default-path integration tests for the analytical relation backend.

Deliberately unpinned: these tests must observe the shipping default."""

import pytest

from pyphi import examples
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations


def test_default_ces_relations_are_analytical():
    ces = examples.xor_system().ces()
    assert isinstance(ces.relations, AnalyticalRelations)


def test_default_summary_matches_concrete_enumeration():
    analytical = examples.xor_system().ces().relations
    concrete = analytical.materialize()
    assert isinstance(concrete, ConcreteRelations)
    assert len(concrete) == analytical.num_relations()
    assert analytical.sum_phi() == pytest.approx(concrete.sum_phi())
```

Run: `uv run pytest test/test_analytical_default.py -q > /tmp/t3c.log 2>&1; cat /tmp/t3c.log`
Expected: both pass.

- [ ] **Step 5: Apply the two known pin updates**

`test/visualize/test_visualize_projection.py` — `_xor_ces` (line ~241) currently returns the ambient-default backend for `analytical=False`; make both branches explicit:

```python
def _xor_ces(analytical):
    import pyphi
    from pyphi import examples

    backend = "ANALYTICAL" if analytical else "CONCRETE"
    with pyphi.config.override(relation_computation=backend):
        return examples.xor_system().ces()
```

`test/test_relations.py` — `_pin_formalism` (line ~69): the module compares against 2023-sourced golden relation files by enumeration, so add the backend to the existing pin:

```python
@pytest.fixture(autouse=True)
def _pin_formalism():
    """Pin IIT 4.0 (2023) with concrete relations: the golden CES/relation
    files in this module are 2023-sourced and compared relation-by-relation,
    so the comparisons must not depend on the ambient default. (Under the
    2026 default, deterministic fixtures cap to φ_s = 0 and their
    congruence-resolved structures are empty.)"""
    with IIT_4_CONFIG, config.override(relation_computation="CONCRETE"):
        yield
```

(Check the module's imports: it must import `config` — `from pyphi.conf import config` — if not already present.)

- [ ] **Step 6: Triage run — pin the remaining failures**

Run the full suite: `uv run pytest -q > /tmp/flip-triage.log 2>&1` then read the failure list (`grep -E "FAILED|ERROR" /tmp/flip-triage.log`). For every failing test, classify:

1. **Enumerates relations (iterates, indexes, `len()`s, compares relation objects):** pin the concrete backend. Whole module → autouse fixture at the top of the file:

   ```python
   @pytest.fixture(autouse=True)
   def _pin_concrete_relations():
       """These tests enumerate individual relations: concrete backend."""
       with config.override(relation_computation="CONCRETE"):
           yield
   ```

   Single test/class → decorator `@config.override(relation_computation="CONCRETE")` or an inline `with`. If the module already has an autouse config fixture (like `test/test_relations.py`), extend that fixture instead of adding a second.
2. **Asserts a summary quantity that the analytical backend answers directly** (`sum_phi`, counts, degree spectra): re-express in one line via the query surface instead of pinning, when that keeps the test's subject intact.
3. **Order-sensitive viz edge assertions** (analytical `strongest()` may order φ_r ties differently from the concrete sort): compare as sets of `e.relata` (the idiom in `test_project_ces_analytical_matches_concrete_and_sizes_faithfully`), or pin concrete if the test is about concrete rendering.
4. **Perf-counter pins** (`test/integration/test_perf_counters.py`): if counts moved because a measured scope now computes analytical relations, regenerate with `uv run python scripts/gen_perf_counts.py` and confirm the test's counted scope still matches the script's exactly before accepting the new pins.
5. **Serialization/golden tests:** round-trips of `AnalyticalRelations` are supported; failures here mean the test compares enumerated relation contents — pin concrete per (1).

Static-grep candidates to expect in the failure list: `test/test_relations.py`, `test/test_helpers.py`, `test/models/test_ces_views.py`, `test/models/test_phi_fold.py`, `test/formalism/test_complexes.py`, `test/integration/test_bounds.py`, `test/integration/test_ces_completeness.py`, `test/integration/test_paper_reproduction.py`, `test/serialize/`. Re-run each touched file after pinning, then repeat the full pathless run until green.

- [ ] **Step 7: Full suite green**

Run: `uv run pytest -q > /tmp/flip-full.log 2>&1` and read the summary line. Expected: 0 failures (pass/skip counts comparable to main's 3821/284 plus the new tests).

- [ ] **Step 8: Commit**

```bash
git add -A pyphi/conf/formalism.py test/
git commit -m "Make ANALYTICAL the default relation computation

The analytical backend answers relation queries in closed form without
enumerating the exponential relation set, and agrees numerically with
CONCRETE everywhere both run. Tests that enumerate individual relations
now pin relation_computation=CONCRETE explicitly."
```

---

### Task 4: Docs sweep and changelog

**Files:**
- Create: `changelog.d/analytical-relations-default.change.md`
- Modify: `docs/tutorials/worked-example.md` (+ `.ipynb` via jupytext)
- Modify: `docs/howto/query-relations.md`, `docs/howto/visualize.md`, `docs/theory/computational-complexity.md`, `pyphi/mcp/content/visualization.md`
- Do NOT touch: `docs/whats-new-in-2.0.md`

**Interfaces:**
- Consumes: the flipped default (Task 3) — the executed tutorials run under it at docs-build time.

- [ ] **Step 1: Changelog fragment**

```bash
cat > changelog.d/analytical-relations-default.change.md <<'EOF'
`relation_computation` now defaults to `"ANALYTICAL"`: `ces.relations` is a
closed-form summary that answers aggregate queries (`sum_phi()`,
`num_relations()`, `degree_spectrum()`, `strongest(k)`, …) without
enumerating the exponentially large relation set, and agrees numerically
with the concrete backend. Iterating or indexing the summary raises
`TypeError`; use `.strongest(k)` for the top-k relations by φ_r,
`.materialize()` to enumerate explicitly, or set
`relation_computation: CONCRETE` under the `formalism` key in
`pyphi_config.yml` to restore enumerated relation sets. Plotting renders
the strongest 1000 relations by default when the set is not enumerable.
EOF
```

- [ ] **Step 2: worked-example tutorial**

In `docs/tutorials/worked-example.md` (~line 129), before the Fig-4 relation code-cell, append to the prose paragraph: "By default PyPhi computes the relation set analytically — a closed-form summary that answers aggregate queries without building relation objects. To examine individual relations, enumerate them explicitly with `materialize()`:". Change the two enumerating cells:

```python
relations = ces.relations.materialize()
relation = next(
    r for r in relations
    if {tuple(m) for m in r.mechanisms} == {(0,), (0, 1)}
)
print("φ_r =", round(float(relation.phi), 4))
print("faces:", relation.num_faces)
```

and (~line 151):

```python
for r in relations:
    mechs = [tuple(m) for m in r.mechanisms]
    print(f"φ_r = {float(r.phi):.4f}  mechanisms {mechs}")
```

Also check the earlier `ces.to_pandas()` cell (~line 125) and the later "Φ-structure, summed" cell still execute (Step 4 verifies by execution). Then:

```bash
uv run jupytext --sync docs/tutorials/worked-example.md
```

- [ ] **Step 3: Default-wording updates in the remaining docs**

`docs/howto/query-relations.md`:
- Backend list (~lines 81–84): annotate the entries — `**\`ANALYTICAL\`** (the default)` and `**\`CONCRETE\`**` (no default marker). Keep both descriptions otherwise.
- Closing passage (~lines 383–390): replace "To use the analytical backend on your own structures, set … `pyphi.config.relation_computation = "ANALYTICAL"` … and every relation set a computation returns will answer these queries in closed form." with: "The analytical backend is the default, so every relation set a computation returns answers these queries in closed form. To work with enumerated relation sets throughout, set `pyphi.config.relation_computation = "CONCRETE"`."
- Scan the rest of the file for other "default" claims about `CONCRETE` and align them.

`docs/howto/visualize.md` (~lines 95–120): the analytical demo no longer needs an override and the raise-demo is obsolete. Replace the passage so that: the CES is computed without an override (`analytical_ces = examples.xor_system().ces()` — under the default it is analytical); the prose says `plot_ces` renders the strongest relations by φ_r, defaulting to the strongest 1000 when the set is not enumerable, with `max_relations` choosing the cap; the `try/except ValueError` cell and its "Without max_relations the call raises" prose are deleted; keep a cell demonstrating an explicit `max_relations=8`.

`docs/theory/computational-complexity.md`:
- Line ~379 (levers table): the row reads `\`relation_computation\` | \`CONCRETE\` → \`ANALYTICAL\``; reword the row to present `ANALYTICAL` as the default and `CONCRETE` as the opt-in enumeration (e.g. lever column: "`relation_computation` (default `ANALYTICAL`; set `CONCRETE` to enumerate)"), keeping the cost-model cells.
- Line ~406 (timing table): move the `(default)` marker from the `CONCRETE` row to the `ANALYTICAL` row.
- Scan the file for other "CONCRETE (default)" phrasing.

`pyphi/mcp/content/visualization.md` (~lines 45–60): replace the two-bullet rule with:

```
- The default `relation_computation` is `"ANALYTICAL"`: `ces.relations` is a
  closed-form summary that cannot be listed. `plot_ces(ces)` works with no
  extra arguments — it draws the strongest 1000 relations by φ_r (via
  `relations.strongest(k)`); pass `max_relations=N` to choose the cap.
- Under `relation_computation="CONCRETE"` the relation set is enumerable and
  `plot_ces(ces)` draws every relation unless `max_relations` caps it.
```

and update the code example below it to drop the override (compute `ces = system.ces()` directly, plot with and without `max_relations`).

- [ ] **Step 4: Verify the executed docs**

```bash
uv run jupyter nbconvert --to notebook --execute docs/tutorials/worked-example.ipynb --stdout > /dev/null
```

Expected: exits 0 (run it in foreground and check `$?` directly — no pipes). If other tutorials fail at full docs build later, fix them the same way (materialize or pin). Then run the doctest sweep early-warning: `uv run pytest pyphi/relations.py pyphi/visualize -q > /tmp/t4doc.log 2>&1; cat /tmp/t4doc.log` (final gate remains the pathless run).

- [ ] **Step 5: Commit**

```bash
git add changelog.d/analytical-relations-default.change.md docs/tutorials/worked-example.md docs/tutorials/worked-example.ipynb docs/howto/query-relations.md docs/howto/visualize.md docs/theory/computational-complexity.md pyphi/mcp/content/visualization.md
git commit -m "Document the analytical relation-computation default"
```

---

### Task 5: Verification

**Files:** none new.

- [ ] **Step 1: Pathless full suite in the worktree**

`uv run pytest -q > /tmp/analytical-full.log 2>&1`, then read the summary line of the log. Expected: green (≈3821+ passed / 284 skipped).

- [ ] **Step 2: Slow lane**

`uv run pytest -m slow --slow -q > /tmp/analytical-slow.log 2>&1` (background it and keep working if convenient; read the summary line before claiming green). Expected: ≈248 passed / 29 skipped.

- [ ] **Step 3: Docs build spot check**

`just docs > /tmp/analytical-docs.log 2>&1` and read the tail of the log for the build status. The known pre-existing `whats-new-in-2.0.md` orphan warning is not a regression; any *error* in an executed tutorial or `-W`-fatal warning introduced by the docs edits must be fixed.

- [ ] **Step 4: Hand off to merge**

Use superpowers:finishing-a-development-branch (standing choice: merge locally to `main`, `--no-ff`, full pathless suite re-run in the main tree after merge). In the same flow: update the ROADMAP "Relations follow-ups" item (c) to landed status, and record the memory entry.
