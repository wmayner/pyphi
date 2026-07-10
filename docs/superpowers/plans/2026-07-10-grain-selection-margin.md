# Grain Selection Margin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report how decisively each complex won the exclusion competition — `Complex.exclusion_margin` and `Complex.effectively_tied` — and fix the `exclusion_records` self-identification bug that hides same-footprint rival grains.

**Architecture:** Derived properties on `pyphi.models.complex.Complex`, computed from the `excluded` records both doors already attach; a one-line identity fix in `pyphi/condensation.py`; a short jupytext-paired tutorial demonstrating recursive exclusion and higher-φₛ shadows. No cascade, schema, or `ComplexesResult` changes.

**Tech Stack:** Python 3.13, pytest, jupytext/MyST (docs).

**Spec:** `docs/superpowers/specs/2026-07-09-grain-selection-margin-design.md`

## Global Constraints

- **No `PyPhiFloat`.** New code uses plain `float | None` and `pyphi.utils.eq` for precision-aware comparison (the wrapper type is being removed in parallel work).
- Margin semantics: **beaten rivals only** — excluded candidates with φₛ less than or precision-equal to the complex's own; higher-φₛ shadows never enter the margin.
- `uv run` for all python commands. Never `--no-verify`. Stage only files this plan touches.
- Docstrings NumPy-style, final-state voice, Unicode symbols (φₛ not `:math:`). No planning artifacts in code.
- Changelog fragment per user-facing change (`changelog.d/<name>.<type>.md`).
- Full verification at the end: `uv run pytest` with NO path argument (doctest sweep).

## Reference values (verified live, 2026-07-10, presets.iit4_2023)

- Decaying chain (4 units, reciprocal couplings 0.6/0.3/0.15, self 0.05, baseline 0.05, state all-OFF), micro door: complexes `{A,B}` φ=0.31994041538707313 and `{C,D}` φ=0.03710393588291759. `{C,D}`'s records hold 4 shadows (`(1,2)` 0.10412076955496435, `(0,1,2)` 0.10002342323823372, `(0,1,2,3)` 0.05636332608474081, `(1,2,3)` 0.05461361319270893) and 2 beaten singletons (`(2,)` 0.02270527941938451, `(3,)` 0.02077346715192418). Margins: `{A,B}` → 0.21581964583210878 (vs `(1,2)`), `{C,D}` → 0.01439865646353308 (vs `(2,)`).
- Min-substrate grain search (`SearchBounds(mappings="EXHAUSTIVE")`, state `(0, 0)`): winner φ=0.7883339770634886 on footprint `(0, 1)`; **7 losing candidates share that exact footprint** (6 rival mappings + the micro pair at 0.005106576483955726) and are currently absent from the records (the bug). Best same-footprint rival: 0.2532971079071088. Post-fix: `winner.excluded` has 9 records (7 same-footprint + 2 singletons); `winner.exclusion_margin` = 0.5350368691563798.

---

### Task 1: `exclusion_records` identity fix

**Files:**
- Modify: `pyphi/condensation.py:268-295` (`exclusion_records`)
- Test: `test/test_condensation.py`
- Create: `changelog.d/exclusion-records-same-footprint.fix.md`

**Interfaces:**
- Produces: `exclusion_records(accepted, candidates)` — same signature; accepted candidates are now identified by object identity (`id()`), so a losing candidate that shares an accepted complex's footprint appears in its records.

- [ ] **Step 1: Write the failing test** — append to `test/test_condensation.py`:

```python
def test_exclusion_records_include_same_footprint_rivals():
    """A losing candidate on the winner's exact footprint is a genuinely
    excluded rival (macro door: a rival grain over the same micro units)
    and must appear in the winner's records."""
    winner = _candidate({0, 1}, 2.0)
    rival_grain = _candidate({0, 1}, 1.0)
    candidates = [winner, rival_grain]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1)]
    records = exclusion_records(outcome.accepted, candidates)
    assert [(r.node_indices, r.phi) for r in records[(0, 1)]] == [((0, 1), 1.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_condensation.py::test_exclusion_records_include_same_footprint_rivals -v`
Expected: FAIL — `records[(0, 1)]` is `()` (the rival's footprint equals the accepted footprint and is skipped).

- [ ] **Step 3: Implement the identity fix** — replace the body of `exclusion_records` (keep the signature and the `ExcludedCandidate` import):

```python
    from pyphi.models.complex import ExcludedCandidate

    accepted_ids = {id(c) for c in accepted}
    records: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for acc in accepted:
        recs = tuple(
            ExcludedCandidate(
                tuple(sorted(cand.footprint)), cand.phi, units=cand.units
            )
            for cand in candidates
            if id(cand) not in accepted_ids and acc.footprint & cand.footprint
        )
        records[tuple(sorted(acc.footprint))] = recs
    return records
```

Update the docstring's final paragraph to read:

```
    A candidate that overlaps several accepted complexes appears in each of
    their exclusion sets. An excluded candidate may carry higher φₛ than a
    complex whose record it appears in, when it was carved away by a
    different overlapping complex. Accepted candidates are identified by
    object identity, not footprint, so a losing candidate that shares an
    accepted complex's exact footprint — a rival grain over the same micro
    units — is recorded. Reads only values the cascade already computed.
```

- [ ] **Step 4: Run the condensation tests**

Run: `uv run pytest test/test_condensation.py -v`
Expected: all PASS (the existing `test_exclusion_records_key_on_footprints` is unaffected — its candidates have unique footprints).

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Fixed `exclusion_records` omitting excluded candidates that share an accepted complex'"'"'s exact micro footprint (rival grains over the same units); accepted candidates are now identified by object identity.' > changelog.d/exclusion-records-same-footprint.fix.md
```

- [ ] **Step 6: Commit**

```bash
git add test/test_condensation.py pyphi/condensation.py changelog.d/exclusion-records-same-footprint.fix.md
git commit -m "Record same-footprint rivals in exclusion records

exclusion_records identified the accepted candidate by footprint
equality, which also silently dropped every losing candidate sharing
that footprint -- at the macro door, rival grains over the same micro
units. Accepted candidates are now identified by object identity."
```

### Task 2: `Complex.exclusion_margin` and `Complex.effectively_tied`

**Files:**
- Modify: `pyphi/models/complex.py` (`Complex` class: new properties, `_describe`, `_pandas_record`)
- Test: `test/models/test_complex_model.py`
- Create: `changelog.d/complex-exclusion-margin.feature.md`

**Interfaces:**
- Consumes: `Complex.excluded: tuple[ExcludedCandidate, ...]`, `pyphi.utils.eq(x, y) -> bool` (both exist).
- Produces: `Complex.exclusion_margin -> float | None` and `Complex.effectively_tied -> bool` (properties); `_pandas_record()` gains keys `exclusion_margin` and `effectively_tied`; the display card gains rows "Selection margin" and "Effectively tied" when the margin exists.

- [ ] **Step 1: Write the failing tests** — append to `test/models/test_complex_model.py` (the file already defines `_StubSIA` with `phi = 1.0` and imports `pytest`, `Complex`, `ExcludedCandidate`):

```python
def test_exclusion_margin_none_when_unopposed():
    c = Complex(sia=_StubSIA(), substrate=None)
    assert c.exclusion_margin is None
    assert c.effectively_tied is False


def test_exclusion_margin_is_gap_to_best_beaten_rival():
    c = Complex(
        sia=_StubSIA(),  # phi = 1.0
        substrate=None,
        excluded=(
            ExcludedCandidate((0,), 0.25),
            ExcludedCandidate((1,), 0.75),
        ),
    )
    assert c.exclusion_margin == pytest.approx(0.25)
    assert c.effectively_tied is False


def test_exclusion_margin_zero_for_precision_equal_rival():
    # A rival within PRECISION of the complex's own phi counts as beaten
    # (the selection was decided beyond phi) and clamps the margin to 0.
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((1, 2), 1.0 + 1e-15),),
    )
    assert c.exclusion_margin == 0.0
    assert c.effectively_tied is True


def test_exclusion_margin_ignores_shadows():
    # A higher-phi overlapping candidate (carved away by a different
    # complex under the recursive cascade) is not a beaten rival.
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((1, 2), 2.0),),
    )
    assert c.exclusion_margin is None
    assert c.effectively_tied is False


def test_exclusion_margin_mixed_shadows_and_rivals():
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(
            ExcludedCandidate((1, 2), 2.0),  # shadow
            ExcludedCandidate((0,), 0.4),  # beaten
        ),
    )
    assert c.exclusion_margin == pytest.approx(0.6)
    assert c.effectively_tied is False


def test_pandas_record_includes_margin_fields():
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((0,), 0.4),),
    )
    record = c._pandas_record()
    assert record["exclusion_margin"] == pytest.approx(0.6)
    assert record["effectively_tied"] is False


def test_describe_margin_rows_present_only_when_margin_exists():
    with_margin = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((0,), 0.4),),
    )
    labels = [
        row.label
        for section in with_margin._describe(verbosity=2).sections
        for row in section.rows
    ]
    assert "Selection margin" in labels
    assert "Effectively tied" in labels

    without = Complex(sia=_StubSIA(), substrate=None)
    labels = [
        row.label
        for section in without._describe(verbosity=2).sections
        for row in section.rows
    ]
    assert "Selection margin" not in labels
    assert "Effectively tied" not in labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/models/test_complex_model.py -v -k margin`
Expected: FAIL with `AttributeError: 'Complex' object has no attribute 'exclusion_margin'`.

- [ ] **Step 3: Implement the properties** — in `pyphi/models/complex.py`, add after the `phi` property (`utils` is already imported):

```python
    @property
    def exclusion_margin(self) -> float | None:
        """The gap in φₛ between this complex and the best overlapping
        rival it beat, or ``None`` when it beat none.

        Rivals are the excluded candidates whose φₛ is less than or
        precision-equal to this complex's own. Because condensation is
        recursive, ``excluded`` may also contain overlapping candidates
        with higher φₛ — carved away by a different complex before this
        one was accepted — and those do not enter the margin. A margin
        of zero means an overlapping rival tied at ``precision``: the
        selection was decided beyond φₛ, either by escalation within the
        tie clique or by the rival's overlap with another complex.
        """
        phi = float(self.phi)
        rivals = [
            float(c.phi)
            for c in self.excluded
            if c.phi < phi or utils.eq(c.phi, phi)
        ]
        if not rivals:
            return None
        return max(0.0, phi - max(rivals))

    @property
    def effectively_tied(self) -> bool:
        """Whether the exclusion margin is within ``precision`` of zero."""
        margin = self.exclusion_margin
        return margin is not None and utils.eq(margin, 0.0)
```

In `_pandas_record`, add before `return record`:

```python
        record["exclusion_margin"] = self.exclusion_margin
        record["effectively_tied"] = self.effectively_tied
```

In `_describe`, after the `rows = [...]` list is built and before the `if self.units is not None:` insert:

```python
        margin = self.exclusion_margin
        if margin is not None:
            rows.append(Row("Selection margin", margin))
            rows.append(Row("Effectively tied", self.effectively_tied))
```

- [ ] **Step 4: Run the model tests**

Run: `uv run pytest test/models/test_complex_model.py -v`
Expected: all PASS.

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Added `Complex.exclusion_margin` (the φₛ gap to the best overlapping rival the complex beat) and `Complex.effectively_tied`, surfaced on the display card and in `to_pandas`. Higher-φₛ shadows in the exclusion records (possible under recursive condensation) do not enter the margin.' > changelog.d/complex-exclusion-margin.feature.md
```

- [ ] **Step 6: Commit**

```bash
git add test/models/test_complex_model.py pyphi/models/complex.py changelog.d/complex-exclusion-margin.feature.md
git commit -m "Add exclusion_margin and effectively_tied to Complex

The margin reports the phi gap between a complex and the best
overlapping rival it beat, extending selection-margin reporting to the
exclusion competition. Beaten-rivals semantics: higher-phi shadows
carried in the exclusion records under recursive condensation do not
enter the margin."
```

### Task 3: Integration pins at both doors

**Files:**
- Test: `test/macro/test_macro_search.py` (add to `TestMinDriver` and `TestRecursiveCondensation`)

**Interfaces:**
- Consumes: `exclusion_records` identity fix (Task 1), `Complex.exclusion_margin` / `effectively_tied` (Task 2), existing fixtures `min_substrate()` / `decaying_chain_substrate()`, existing imports `micro_unit`, `config`, `presets`, `pytest`.

- [ ] **Step 1: Write the macro-door pin** — add to `class TestMinDriver`:

```python
    def test_excluded_records_and_margin_cover_same_footprint_rivals(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        winner = result.complexes[0]
        # The seven losing candidates on the winner's own footprint
        # {0,1} (six rival mappings and the micro pair) plus the two
        # singletons. Golden recorded at implementation time.
        assert len(winner.excluded) == 9
        assert any(
            e.node_indices == (0, 1)
            and e.units == (micro_unit(0), micro_unit(1))
            and e.phi == pytest.approx(0.005106576483955726, abs=1e-13)
            for e in winner.excluded
        )
        # The winning grain beat its best overlapping rival (the
        # (0,1,0,1)/(0,0,1,1) mappings, phi 0.2532971079071088) by:
        assert winner.exclusion_margin == pytest.approx(
            0.5350368691563798, abs=1e-13
        )
        assert winner.effectively_tied is False
```

- [ ] **Step 2: Write the micro-door pin** — add to `class TestRecursiveCondensation`:

```python
    def test_chain_margins_count_beaten_rivals_only(self):
        from pyphi.substrate import complexes as micro_complexes

        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            found = micro_complexes(substrate, (0, 0, 0, 0))
        by_idx = {c.node_indices: c for c in found}
        ab, cd = by_idx[(0, 1)], by_idx[(2, 3)]
        # {A,B} beat the runner-up {B,C} (phi 0.1041...).
        assert ab.exclusion_margin == pytest.approx(
            0.21581964583210878, abs=1e-13
        )
        # {C,D} carries four higher-phi shadows ({B,C} among them); the
        # margin ignores them and measures the gap to the best beaten
        # rival, the singleton {C} (phi 0.0227...).
        shadows = [e for e in cd.excluded if e.phi > float(cd.phi)]
        assert len(shadows) == 4
        assert cd.exclusion_margin == pytest.approx(
            0.01439865646353308, abs=1e-13
        )
        assert cd.effectively_tied is False
```

- [ ] **Step 3: Update the stale invariant comment** — in
  `test_exclusion_invariants_on_the_chain_sweep` (same file), the assertion
  `assert record.node_indices not in accepted` holds for this sweep only
  because `max_depth=0` makes footprints unique per candidate. Extend the
  existing NOTE comment above it with:

```python
                # (With grains in play a record MAY name an accepted
                # complex's exact footprint -- a rival grain over the same
                # micro units; this sweep is micro-only, so footprints are
                # unique and the containment check below is valid.)
```

- [ ] **Step 4: Run both test classes**

Run: `uv run pytest "test/macro/test_macro_search.py::TestMinDriver" "test/macro/test_macro_search.py::TestRecursiveCondensation" -v`
Expected: all PASS (including the pre-existing tests — the identity fix adds records at the macro door but no existing assertion pins `excluded` contents there).

- [ ] **Step 5: Commit**

```bash
git add test/macro/test_macro_search.py
git commit -m "Pin exclusion-margin behavior at both complex doors

Macro door: same-footprint rival grains appear in exclusion records
and drive the winner's margin. Micro door: on the decaying chain,
{C,D}'s margin measures the gap to its best beaten rival and ignores
the four higher-phi shadows in its records."
```

### Task 4: Recursive-exclusion tutorial

**Files:**
- Create: `docs/tutorials/recursive-exclusion.md` (+ jupytext-paired `docs/tutorials/recursive-exclusion.ipynb`, output-free)
- Modify: `docs/tutorials/index.md` (toctree)
- Create: `changelog.d/recursive-exclusion-tutorial.doc.md`

**Interfaces:**
- Consumes: `pyphi.substrate.Substrate.complexes(state)`, `Complex.excluded` / `.exclusion_margin` / `.phi` (Task 2), `pyphi.conf.presets.iit4_2023`.

- [ ] **Step 1: Write the page** — create `docs/tutorials/recursive-exclusion.md`:

````markdown
---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Recursive exclusion: how complexes carve a substrate

{download}`Download this page as a Jupyter notebook <recursive-exclusion.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/recursive-exclusion.ipynb)

The exclusion postulate says that candidate systems sharing units cannot
both exist: among overlapping candidates, only one specifies its
cause–effect structure. PyPhi applies the postulate *recursively*
(Marshall, Albantakis, and Tononi, 2023): candidates are walked in
descending order of system integrated information $\varphi_s$, each
accepted complex claims its units, and — crucially — a candidate excluded
by an accepted complex no longer exists, so it cannot exclude anything
else in turn.

This recursion has a consequence that surprises many readers: **a complex
can coexist with an overlapping candidate of higher $\varphi_s$**, as long
as that candidate was itself excluded by some other complex. This tutorial
builds the smallest substrate where that happens, finds its complexes, and
reads the exclusion records and selection margins that document who beat
whom.

```{code-cell} python
import numpy as np

import pyphi
from pyphi.conf import presets

pyphi.config.progress_bars = False
```

## A chain of decaying couplings

Four units A, B, C, D in a chain, with reciprocal coupling strengths that
decay along it: A–B couple strongly (0.6), B–C moderately (0.3), C–D weakly
(0.15). Each unit's probability of turning on is a baseline 0.05, plus 0.05
times its own state, plus the coupled inputs.

```{code-cell} python
n = 4
weights = np.zeros((n, n))
weights[0, 1] = weights[1, 0] = 0.6
weights[1, 2] = weights[2, 1] = 0.3
weights[2, 3] = weights[3, 2] = 0.15
for i in range(n):
    weights[i, i] = 0.05

tpm = np.zeros((2**n, n))
for row in range(2**n):
    state = np.array([(row >> k) & 1 for k in range(n)])
    tpm[row] = 0.05 + weights @ state

substrate = pyphi.Substrate(tpm, node_labels=("A", "B", "C", "D"))
state = (0, 0, 0, 0)
```

By construction the $\varphi_s$ landscape is a chain:
$\{A,B\} > \{B,C\} > \{C,D\}$, with $\{B,C\}$ overlapping both of its
neighbors.

## Finding the complexes

```{code-cell} python
with pyphi.config.override(**presets.iit4_2023):
    found = substrate.complexes(state)

for complex_ in found:
    print(complex_.node_indices, float(complex_.phi))
```

Both $\{A,B\}$ *and* $\{C,D\}$ are complexes. A non-recursive reading of
exclusion would reject $\{C,D\}$: it overlaps $\{B,C\}$, which has higher
$\varphi_s$. But $\{B,C\}$ overlaps $\{A,B\}$, which has higher
$\varphi_s$ still — so $\{B,C\}$ is excluded first, and once excluded it
does not exist and has no standing to exclude $\{C,D\}$. The recursion
carves the substrate from the top down, and $\{C,D\}$ is the maximum among
the candidates that remain.

## Shadows: excluded candidates with higher φₛ

Each complex records the overlapping candidates excluded in its favor:

```{code-cell} python
cd = found[1]
for record in sorted(cd.excluded, key=lambda r: -r.phi):
    marker = "shadow" if record.phi > float(cd.phi) else "beaten"
    print(f"{record.node_indices}  φₛ={record.phi:.4f}  [{marker}]")
```

$\{C,D\}$'s records contain candidates with **higher** $\varphi_s$ than
$\{C,D\}$ itself — $\{B,C\}$ among them. These are *shadows*: candidates
that out-inform the complex but were carved away by a different complex
before this one was accepted. They document the recursion at work; they
were never rivals that $\{C,D\}$ had to beat.

## Selection margins

How decisively did each complex win? The `exclusion_margin` of a
{class}`~pyphi.models.complex.Complex` reports the $\varphi_s$ gap to the
best overlapping rival the complex actually beat — shadows do not enter
the margin:

```{code-cell} python
for complex_ in found:
    print(
        complex_.node_indices,
        f"φₛ={float(complex_.phi):.4f}",
        f"margin={complex_.exclusion_margin:.4f}",
    )
```

$\{A,B\}$ won by a wide margin over the runner-up on its units. $\{C,D\}$
beat only the singletons $\{C\}$ and $\{D\}$, and its margin measures the
gap to the best of them. A margin of zero (equivalently,
`complex_.effectively_tied`) would mean an overlapping rival tied at the
configured precision and the selection was decided beyond $\varphi_s$; see
{doc}`../howto/tie-breaking` for how ties are resolved.

## References

- Marshall W, Albantakis L, Tononi G (2023). System integrated information.
  *Entropy* 25(2):334, Algorithm A1.
- Albantakis L et al. (2023). Integrated information theory (IIT) 4.0.
  *PLoS Computational Biology* 19(10):e1011465 (the exclusion postulate).
````

- [ ] **Step 2: Verify the citation details** against `papers/` (the Marshall
  2023 Algorithm A1 reference and the IIT 4.0 citation) — correct the entry
  if the venue/number differs. Never cite from memory.

- [ ] **Step 3: Pair and sync the notebook**

Run: `uv run jupytext --sync docs/tutorials/recursive-exclusion.md`
Expected: creates `docs/tutorials/recursive-exclusion.ipynb` (output-free).

- [ ] **Step 4: Add to the toctree** — in `docs/tutorials/index.md`, add
  `recursive-exclusion` on a new line after `macro`.

- [ ] **Step 5: Build the docs and verify execution**

Run: `env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: exit 0. Then confirm the cells actually executed — the printed φₛ of `{A,B}` appears only in executed output, not in the prose: `grep -o "0.31994" docs/_build/html/tutorials/recursive-exclusion.html | head -1` prints `0.31994`.

- [ ] **Step 6: Changelog fragment**

```bash
echo 'Added a tutorial on recursive exclusion: how complexes carve the substrate, why exclusion records can contain higher-φₛ "shadows", and how to read selection margins.' > changelog.d/recursive-exclusion-tutorial.doc.md
```

- [ ] **Step 7: Commit**

```bash
git add docs/tutorials/recursive-exclusion.md docs/tutorials/recursive-exclusion.ipynb docs/tutorials/index.md changelog.d/recursive-exclusion-tutorial.doc.md
git commit -m "Add recursive-exclusion tutorial

Demonstrates the recursive exclusion cascade on a decaying chain:
overlapping complexes carved top-down, higher-phi shadows in the
exclusion records, and selection margins over beaten rivals."
```

### Task 5: Full verification

- [ ] **Step 1: Run the full suite with the doctest sweep**

Run: `uv run pytest` (NO path argument)
Expected: all green (baseline on main: 3081 passed + env-gated `PYPHI_MACRO_FULL_SWEEP` skips; this plan adds ~10 tests).

- [ ] **Step 2: If anything fails,** fix within the task that introduced it and re-run; do not proceed with failures.
