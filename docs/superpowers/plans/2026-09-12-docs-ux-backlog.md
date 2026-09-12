# Documentation UX Backlog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the sixteen backlog items of the documentation review: two new how-tos (build a substrate, read a result), a glossary, a FAQ, a sizing how-to, a configuration reference, an examples gallery, a contributing page, navigation fixes, migration-guide additions, and six small code changes that remove the traps the walkthroughs hit.

**Architecture:** Code changes first (they are small and the new pages describe the changed behavior): results state their formalism, `estimate_analysis` refuses a positional state, the IIT 3.0 structure returns a resolved type, the MCP `analyze` tool accepts a subset and reports the MIP and the intrinsic-information terms. Then the pages, each an executed MyST page in the existing section layout, cross-linked from the pages the walkthroughs started on. Every page executes under `just docs` (Sphinx `-W`), so a wrong claim fails the build.

**Tech Stack:** Python 3.13, MyST-NB executed pages (`docs/*.md` with `{code-cell}`), Sphinx with `sphinx_design`, pytest (`uv run pytest`), FastMCP tools in `pyphi/mcp/server.py`.

**Spec:** `docs/superpowers/specs/2026-09-11-docs-ux-review.md` (findings F1–F16, backlog B1–B16).

## Global Constraints

- Public prose (pages, docstrings, MCP content, changelog fragments) follows the `writing-naturally` skill: plain sentences, neutral tone, no compressed shorthand.
- Terminology: "intrinsic-information requirement"; never "cap". Names of the specified-state term follow the formalism: intrinsic specification (2026), intrinsic information (2023).
- φₛ and Φ are always distinguished; a page that prints a φ value says which formalism produced it.
- Every executed page pins `pyphi.config.progress_bars = False` in its first cell and states its formalism where a number is shown.
- Every user-facing change gets a `changelog.d/<name>.<type>.md` fragment.
- `uv run ruff format <files>` on the specific `.py` files touched (not directories: directory-wide formatting also rewrites code blocks in `pyphi/mcp/content/*.md`), then `uv run ruff check pyphi test`; pre-commit runs ruff and pyright; never `--no-verify`.
- pytest: redirect to a scratch file, echo `$?`, read the summary line; never pipe through `tail`/`grep`. Bare `uv run pytest` (no path) for the final verification. Docs: `just docs` must end in "build succeeded" (delete a stale `docs/reference/_autosummary/` first if the build reports missing attributes).
- Environment for any pyphi script: `PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1`.
- Commit messages end with the session's attribution trailer. No push, no tag.
- Pages under `docs/getting-started/` and `docs/tutorials/` are jupytext-paired: the pre-commit hook regenerates the `.ipynb`; stage it too (`git add docs/tutorials/<page>.ipynb`) and retry the commit once.

---

### Task 1: Worktree

**Files:** none (git only)

- [ ] **Step 1: Create the worktree**

```bash
cd /Users/will/projects/pyphi
git worktree add .claude/worktrees/docs-ux -b docs-ux main
cd .claude/worktrees/docs-ux
uv sync --all-extras > /dev/null 2>&1; echo "sync exit: $?"
```

- [ ] **Step 2: Baseline**

```bash
export PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1
uv run pytest test/test_analyze.py test/display -q > /tmp/baseline.log 2>&1; echo $?; tail -1 /tmp/baseline.log
```
Expected: exit 0.

---

### Task 2: Results state their formalism (B3)

**Files:**
- Modify: `pyphi/analyze.py` (class `Analysis`, lines 33–100)
- Test: `test/test_analyze.py` (append)
- Test: `test/display/test_display.py` (any golden that renders an `Analysis` card gains the new row)

**Interfaces:**
- Produces: `Analysis.formalism -> str` (`"IIT_4_0_2026"`, `"IIT_4_0_2023"`, or `"IIT_3_0"`), and a `Formalism` row as the first row of the analysis card. Tasks 8, 9, 10, 11 cite it.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_analyze.py`:

```python
def test_analysis_reports_its_formalism():
    """A result says which formalism produced it (docs review F3)."""
    from pyphi import analyze, examples
    from pyphi.conf import config, presets

    substrate = examples.iit4_2023_fig1a_substrate()
    for name in ("IIT_4_0_2026", "IIT_4_0_2023", "IIT_3_0"):
        with config.override(**presets.by_name[name], progress_bars=False):
            analysis = analyze(substrate, (0, 1, 1), subset=(0, 1))
        assert analysis.formalism == name
        assert name in repr(analysis)  # the card leads with a Formalism row
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_analyze.py -q -k reports_its_formalism 2>&1 | tail -3
```
Expected: FAIL with `AttributeError: 'Analysis' object has no attribute 'formalism'`.

- [ ] **Step 3: Implement**

In `pyphi/analyze.py`, after the `phi` property of `Analysis`:

```python
    @property
    def formalism(self) -> str:
        """str: The formalism that produced this analysis — ``"IIT_4_0_2026"``,
        ``"IIT_4_0_2023"``, or ``"IIT_3_0"`` — read from the configuration
        snapshot the system irreducibility analysis carries."""
        return self.sia.config.formalism.iit.version
```

In `Analysis._describe`, after `sections = list(desc.sections)`, put the formalism first:

```python
        # Lead with the formalism: a φ value means nothing without it.
        first = sections[0]
        sections[0] = replace(
            first, rows=(Row("Formalism", self.formalism), *first.rows)
        )
```

with `from dataclasses import replace` and `from pyphi.display import Row` added to the imports (check what `pyphi/analyze.py` already imports from `pyphi.display`; `Section` is a frozen dataclass, so `replace` works).

- [ ] **Step 4: Run**

```bash
uv run pytest test/test_analyze.py test/display test/mcp -q > /tmp/t2.log 2>&1; echo $?; tail -1 /tmp/t2.log
```
Expected: exit 0 after updating any display golden that renders an `Analysis` card (search `test/display/test_display.py` for `Analysis(` and for card text beginning with `╭─ Analysis`); the new first row is the intended change.

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/analysis-formalism.feature.md <<'EOF2'
`Analysis.formalism` names the formalism that produced a result (`"IIT_4_0_2026"`, `"IIT_4_0_2023"`, or `"IIT_3_0"`), and the analysis card leads with it.
EOF2
uv run ruff format pyphi/analyze.py test/test_analyze.py && uv run ruff check pyphi test
git add pyphi/analyze.py test/test_analyze.py test/display/test_display.py changelog.d/analysis-formalism.feature.md
git commit -m "Report the formalism on Analysis and lead its card with it"
```

---

### Task 3: `estimate_analysis` refuses a positional state (B10)

**Files:**
- Modify: `pyphi/cost.py:511-517` (signature)
- Test: `test/test_cost.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_estimate_analysis_subset_is_keyword_only():
    """A state passed positionally must not bind to ``subset`` (docs review F8)."""
    import pytest

    from pyphi import examples
    from pyphi.cost import estimate_analysis

    substrate = examples.iit4_2023_fig1a_substrate()
    with pytest.raises(TypeError):
        estimate_analysis(substrate, (0, 1, 1))
    assert estimate_analysis(substrate, subset=(0, 1)).n_units == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_cost.py -q -k keyword_only 2>&1 | tail -3
```
Expected: FAIL (`DID NOT RAISE`).

- [ ] **Step 3: Implement**

Change the signature to

```python
def estimate_analysis(
    substrate: Substrate,
    *,
    subset: Any = None,
    compute: str | None = None,
    limit: int = 1_000_000,
    scope: Any | None = None,
) -> AnalysisEstimate:
```

Then find every positional caller: `grep -rn "estimate_analysis(" pyphi test docs benchmarks --include='*.py' --include='*.md'` and make each pass `subset=`/`compute=` by keyword (the MCP server already does).

- [ ] **Step 4: Run**

```bash
uv run pytest test/test_cost.py test/mcp test/macro -q > /tmp/t3.log 2>&1; echo $?; tail -1 /tmp/t3.log
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/estimate-analysis-keyword-only.change.md <<'EOF2'
`pyphi.cost.estimate_analysis` now takes `subset`, `compute`, `limit`, and `scope` by keyword only, so a state passed by mistake raises instead of being read as the candidate subset.
EOF2
uv run ruff format pyphi/cost.py test/test_cost.py && uv run ruff check pyphi test
git add pyphi/cost.py test/test_cost.py changelog.d/estimate-analysis-keyword-only.change.md
git add -u pyphi test docs benchmarks
git commit -m "Make estimate_analysis's options keyword-only"
```

---

### Task 4: Three docstring and reference fixes (B14 and parts of B6, B1)

**Files:**
- Modify: `pyphi/substrate_generator/mechanisms.py:53-75` (`sigmoid` docstring)
- Modify: `pyphi/mcp/content/gotchas.md` (§5 heading and sentence)
- Modify: `pyphi/analyze.py` (the `analyze` docstring's "MICE")

- [ ] **Step 1: Sigmoid docstring**

Replace `"""Logistic activation of the weighted input."""` with

```python
    """Logistic activation of the weighted input.

    The probability that the unit is ON at the next step is
    ``σ(determinism · (Σ_i w_i s_i − threshold))``, where the inputs are the
    unit's current input states (mapped to ±1 when ``ising`` is True) and
    ``w_i`` their weights. ``determinism`` is the slope ``k`` of the logistic
    unit in Albantakis et al. (2023, Eq. 60) and Marshall et al. (2023,
    Eq. 2): larger values make the unit more deterministic. ``floor`` and
    ``ceiling`` clip the output.
    """
```

- [ ] **Step 2: Gotchas wording**

In `pyphi/mcp/content/gotchas.md`, change `- **The 2026 differentiation cap.**` to `- **The 2026 intrinsic-information requirement.**` and leave the sentence that follows. Then run `grep -rn -i "\bcap\b\|capped" pyphi/mcp/content pyphi/mcp/skills` and fix any other hit about Eq. 23 (count caps, table caps, and cost caps stay).

- [ ] **Step 3: MICE**

In the `analyze` docstring (`pyphi/analyze.py`), replace the first "MICE" with "maximally irreducible cause and effect (MICE)". Do the same at its first occurrence in `docs/theory/computational-complexity.md`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest pyphi/analyze.py pyphi/substrate_generator test/mcp/test_content.py -q > /tmp/t4.log 2>&1; echo $?; tail -1 /tmp/t4.log
uv run ruff format pyphi/substrate_generator/mechanisms.py pyphi/analyze.py && uv run ruff check pyphi
git add pyphi/substrate_generator/mechanisms.py pyphi/mcp/content/gotchas.md pyphi/analyze.py docs/theory/computational-complexity.md
git commit -m "Name the logistic slope k, expand MICE, and drop the last 'cap'"
```

---

### Task 5: IIT 3.0 structures come back resolved (B15)

**Files:**
- Modify: `pyphi/system.py:853-880` (`System.ces`)
- Test: `test/test_system.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_iit3_ces_is_a_resolved_distinctions():
    """Under IIT 3.0 the structure has no tied states to resolve, so the public
    type is ResolvedDistinctions, not the raw UnresolvedDistinctions (docs
    review F6)."""
    import pyphi
    from pyphi.conf import config, presets
    from pyphi.models import ResolvedDistinctions

    with config.override(**presets.iit3, progress_bars=False):
        ces = pyphi.System(pyphi.examples.basic_substrate(), (1, 0, 0)).ces()
        analysis = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 0, 0))
    assert isinstance(ces, ResolvedDistinctions)
    assert isinstance(analysis.ces, ResolvedDistinctions)
    assert len(ces.concepts) == 4
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_system.py -q -k iit3_ces_is_a_resolved 2>&1 | tail -3
```
Expected: FAIL (`isinstance` False).

- [ ] **Step 3: Implement**

In `System.ces._compute`, where the IIT 3.0 branch returns `_ces(self, **call_kwargs)`, wrap it:

```python
            if formalism_name == "IIT_3_0":
                from pyphi.formalism.iit3 import (
                    _compute_distinctions as _ces,  # pyright: ignore[reportPrivateUsage]
                )
                from pyphi.models import ResolvedDistinctions

                # IIT 3.0 has no tied specified states to resolve.
                return ResolvedDistinctions(_ces(self, **call_kwargs))
```

Read the surrounding function first: keep the disk-cache wrapping and the IIT 4.0 branch untouched. Update the `ces` docstring's IIT 3.0 sentence to "returns a `ResolvedDistinctions` — the concepts, as `.concepts`; IIT 3.0 has no relations".

- [ ] **Step 4: Run**

```bash
uv run pytest test/test_system.py test/formalism test/test_analyze.py -q > /tmp/t5.log 2>&1; echo $?; tail -1 /tmp/t5.log
```
Expected: exit 0 (the disk-cache and serialization round-trips must still pass; if a golden pins the type name, `ResolvedDistinctions` is the intended value).

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/iit3-ces-resolved.change.md <<'EOF2'
Under IIT 3.0, `System.ces()` and `analyze().ces` return a `ResolvedDistinctions` (its concepts under `.concepts`) rather than the internal `UnresolvedDistinctions`.
EOF2
uv run ruff format pyphi/system.py test/test_system.py && uv run ruff check pyphi test
git add pyphi/system.py test/test_system.py changelog.d/iit3-ces-resolved.change.md
git commit -m "Return ResolvedDistinctions for IIT 3.0 cause-effect structures"
```

---

### Task 6: MCP `analyze` takes a subset and reports the MIP and the intrinsic-information terms (B4)

**Files:**
- Modify: `pyphi/mcp/server.py:197-260` (`_result_summary`), `:505-600` (`analyze`)
- Modify: `pyphi/analyze.py` (`Analysis._describe`: a compact system section)
- Modify: `pyphi/mcp/content/primer.md` (tool list line for `analyze`)
- Test: `test/mcp/test_server.py` (append)

**Interfaces:**
- Produces: `analyze(handle, state, subset: list[int | str] | None = None, ...)`; summary keys `mip`, `intrinsic_information`, `requirement_binding` (`None` or `{"term": "differentiation"|"specification", "direction": "CAUSE"|"EFFECT"}`); the analysis card gains a `System` section with rows `MIP`, `ii(s)`, and `Requirement binds` (when it does).

- [ ] **Step 1: Write the failing tests**

Read `test/mcp/test_server.py` first for how existing tests call tools (they call the decorated functions directly or through a fixture; follow the same pattern). Append:

```python
def test_analyze_subset_analyzes_the_candidate_system(server_tools):
    """The Fig 1A pair aB reproduces the paper's 0.17 under the 2023 formalism
    through the tool (docs review F2)."""
    handle = server_tools.load_example("iit4_2023_fig1a")["handle"]
    out = server_tools.analyze(
        handle, [0, 1, 1], subset=["A", "B"], formalism="IIT_4_0_2023", compute="sia"
    )
    assert round(out["summary"]["system_phi"], 2) == 0.17
    assert out["summary"]["subset"] == ["A", "B"]


def test_analyze_summary_carries_mip_and_requirement_terms(server_tools):
    handle = server_tools.load_example("basic")["handle"]
    out = server_tools.analyze(handle, [1, 1, 0], compute="sia")
    summary = out["summary"]
    assert summary["mip"]
    assert summary["intrinsic_information"] == 0.0
    assert summary["requirement_binding"] == {
        "term": "differentiation",
        "direction": "EFFECT",
    }
    assert "MIP" in out["card"] and "ii(s)" in out["card"]
```

Adapt the `server_tools` fixture name to whatever the module uses; if tests call `server.analyze(...)` directly, do that. The `basic` network in state `(1, 1, 0)` under the default formalism has φₛ = 0 with positive φ_c and φ_e (the MCP walkthrough's case), so the effect-side differentiation is the binding term; verify by running before pinning, and pin what runs.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/mcp/test_server.py -q -k "analyze_subset or mip_and_requirement" 2>&1 | tail -3
```
Expected: FAIL (`TypeError: unexpected keyword argument 'subset'`).

- [ ] **Step 3: Implement the tool parameter**

In `analyze`, add `subset: list[int | str] | None = None,` after `state`, document it:

```
    subset : list of int or str, optional
        The candidate system: node indices or labels. Default analyzes the
        whole substrate. Units outside the subset are the background and are
        causally marginalized (IIT 4.0) or conditioned on their current
        state (IIT 3.0) — see ``get_iit_reference("theory")``.
```

resolve it once, and thread it through both the guard and the call:

```python
    substrate = _get_substrate(handle)
    indices = (
        tuple(substrate.node_labels.coerce_to_indices(subset))
        if subset is not None
        else None
    )
    ...
            estimate = estimate_analysis(
                substrate, subset=indices, compute=compute_arg, limit=_GUARD_COUNT_BUDGET
            )
    ...
        result = pyphi.analyze(
            substrate, tuple(state), subset=indices, formalism=formalism, compute=compute_arg
        )
```

and add `"subset": list(subset) if subset is not None else None` to `out["summary"]` after `_result_summary` builds it.

- [ ] **Step 4: Extend the summary**

In `_result_summary`, after the `summary["mip"] = ...` line, add:

```python
        ii = getattr(sia, "intrinsic_information", None)
        if ii is not None:
            summary["intrinsic_information"] = float(ii)
        binding = next(
            (f for f in sia.explain().findings if f.kind == "requirement_binding"),
            None,
        ) if hasattr(sia, "explain") else None
        summary["requirement_binding"] = (
            {"term": binding.value, "direction": dict(binding.detail)["direction"]}
            if binding is not None
            else None
        )
```

Guard `sia.explain()` for null and IIT 3.0 analyses (`explain` exists on both; `requirement_binding` never fires there).

- [ ] **Step 5: Extend the card**

In `pyphi/analyze.py`, `Analysis._describe`: after the formalism row from Task 2, append a compact section built from the SIA when the card does not already embed the SIA (i.e. at verbosity below FULL under IIT 4.0):

```python
        if verbosity < FULL and getattr(self.sia, "partition", None) is not None:
            from pyphi.models.partitions import concise_partition

            rows = [Row("MIP", concise_partition(self.sia.partition))]
            ii = getattr(self.sia, "intrinsic_information", None)
            if ii is not None:
                rows.append(Row("ii(s)", ii))
                binding = next(
                    (f for f in self.sia.explain().findings if f.kind == "requirement_binding"),
                    None,
                )
                if binding is not None:
                    rows.append(
                        Row("Requirement binds", f"{binding.value} ({dict(binding.detail)['direction']})")
                    )
            sections.append(Section(label="System", rows=tuple(rows)))
```

Check where `concise_partition` lives (`grep -rn "def concise_partition" pyphi`) and import from there.

- [ ] **Step 6: Primer**

In `pyphi/mcp/content/primer.md`, the `analyze` tool line: add `subset?` to the parameter list and one clause: "`subset` names the candidate system (indices or labels); the rest of the substrate is background."

- [ ] **Step 7: Run**

```bash
uv run pytest test/mcp test/test_analyze.py test/display -q > /tmp/t6.log 2>&1; echo $?; tail -1 /tmp/t6.log
```
Expected: exit 0 (update the `Analysis` card goldens for the new `System` section).

- [ ] **Step 8: Commit**

```bash
cat > changelog.d/mcp-analyze-subset.feature.md <<'EOF2'
The MCP `analyze` tool takes a `subset` (node indices or labels) to analyze a candidate system inside a larger substrate, and its summary and card now carry the minimum information partition, the intrinsic information ii(s), and which term of the intrinsic-information requirement set φₛ when it did.
EOF2
uv run ruff format pyphi/mcp/server.py pyphi/analyze.py test/mcp/test_server.py && uv run ruff check pyphi test
git add pyphi/mcp/server.py pyphi/analyze.py pyphi/mcp/content/primer.md test/mcp/test_server.py test/display/test_display.py changelog.d/mcp-analyze-subset.feature.md
git commit -m "Let the MCP analyze tool take a subset and report the MIP and ii(s)"
```

Note on unknown arguments: FastMCP validates tool arguments against the generated schema; whether it rejects unknown keys depends on the client and the FastMCP version. Do not add a custom rejection layer; the walkthrough's failure is fixed by `subset` existing.

---

### Task 7: Cost caveats in the MCP `performance` topic and `estimate_cost` (B8, MCP part)

**Files:**
- Modify: `pyphi/mcp/content/performance.md`
- Modify: `pyphi/mcp/server.py:603-662` (`estimate_cost` return value)
- Test: `test/mcp/test_server.py` (append)

- [ ] **Step 1: Failing test**

```python
def test_estimate_cost_states_what_the_seconds_cover(server_tools):
    handle = server_tools.load_example("iit4_2023_fig1a")["handle"]
    full = server_tools.estimate_cost(handle, compute="full")
    sia = server_tools.estimate_cost(handle, compute="sia")
    assert "system-partition" in full["estimated_cpu_seconds_covers"]
    assert sia["estimated_cpu_seconds"] is None
    assert "not calibrated" in sia["estimated_cpu_seconds_covers"]
```

- [ ] **Step 2: Implement**

In `estimate_cost`'s return dict add

```python
        "estimated_cpu_seconds_covers": (
            "the distinction axis only (mechanisms, purviews, mechanism "
            "partitions); the system-partition axis is not calibrated to "
            "seconds and is excluded, so a 'sia' estimate has no seconds and "
            "a 'full' estimate is a lower bound"
        ),
```

In `performance.md`, add under the cost-estimate section:

> `estimated_cpu_seconds` converts only the distinction axis (mechanisms,
> purviews, mechanism partitions) to time. The system-partition axis is not
> calibrated to seconds, so a `compute="sia"` estimate reports counts but no
> time, and a `"full"` estimate is a lower bound. Compare axes by their
> counts: the system-partition count grows fastest with the number of units.

- [ ] **Step 3: Run and commit**

```bash
uv run pytest test/mcp -q > /tmp/t7.log 2>&1; echo $?; tail -1 /tmp/t7.log
cat > changelog.d/mcp-estimate-cost-covers.doc.md <<'EOF2'
The MCP `estimate_cost` tool and the `performance` reference say which axes `estimated_cpu_seconds` covers: the distinction axis only, so a system-φ estimate reports counts without a time and a full estimate is a lower bound.
EOF2
uv run ruff format pyphi/mcp/server.py test/mcp/test_server.py && uv run ruff check pyphi test
git add pyphi/mcp/server.py pyphi/mcp/content/performance.md test/mcp/test_server.py changelog.d/mcp-estimate-cost-covers.doc.md
git commit -m "Say what estimate_cost's seconds cover"
```

---

### Task 8: How-to "Build a substrate" (B1)

**Files:**
- Create: `docs/howto/build-substrate.md`
- Modify: `docs/howto/index.md` (toctree: `build-substrate` first)
- Modify: `docs/getting-started/index.md` ("Where to go next": first bullet) and its paired `.ipynb`
- Modify: `docs/theory/substrate-and-system.md` (one sentence after the `substrate.cm` cell)

**Interfaces:**
- Consumes: `sigmoid`'s `determinism` docstring (Task 4).
- Produces: the page other tasks link as `{doc}`/howto/build-substrate``.

- [ ] **Step 1: Write the page**

`docs/howto/build-substrate.md` (MyST front matter as on `docs/howto/sweep.md`):

````markdown
# Build a substrate

A {class}`~pyphi.substrate.Substrate` is a set of units and the probability
of each unit's next state given the current state of all of them. This page
shows the four ways to make one: from a transition probability matrix you
already have, from a weight matrix with logistic units, from a function per
unit, and from recorded transitions. It ends with how to check the result
before analyzing it.

```{code-cell} python
import numpy as np
import pyphi

pyphi.config.progress_bars = False
```

## From a transition probability matrix

The usual input is **state-by-node** form: one row per current state, one
column per unit, each entry the probability that the unit is ON at the next
step. Rows are ordered so that the **first unit changes fastest** (PyPhi's
little-endian convention): for three units the rows are the states
`(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)`.

```{code-cell} python
tpm = np.array([
    [0.1, 0.1, 0.1],   # current state (0, 0, 0)
    [0.1, 0.9, 0.1],   # (1, 0, 0)
    [0.9, 0.1, 0.9],   # (0, 1, 0)
    [0.9, 0.9, 0.1],   # (1, 1, 0)
    [0.1, 0.1, 0.9],   # (0, 0, 1)
    [0.9, 0.1, 0.9],   # (1, 0, 1)
    [0.9, 0.9, 0.9],   # (0, 1, 1)
    [0.1, 0.9, 0.1],   # (1, 1, 1)
])
substrate = pyphi.Substrate(tpm, node_labels=("A", "B", "C"))
substrate
```

A state-by-state matrix (one row and one column per state) is accepted too
and converted; the {ref}`TPM conventions <tpm-conventions>` page describes
every accepted form and the row order in full.

Pass a connectivity matrix (`cm=`; `cm[i, j] = 1` when unit `i` is an input
to unit `j`) only when you know the wiring. Without one PyPhi assumes every
unit may influence every other, which is always correct and only slower. A
wrong connectivity matrix gives a wrong result, not a slow one.

Check one transition you know before trusting anything computed from the
matrix. Index the multidimensional form with a state:

```{code-cell} python
substrate.tpm[1, 0, 0]  # the row for current state (1, 0, 0)
```

## From a weight matrix and logistic units

The networks of the IIT 4.0 papers are logistic (sigmoid) units of their
weighted inputs. Give {func}`~pyphi.substrate_generator.build_substrate` a
weight matrix — `w[i, j]` is the weight from unit `i` to unit `j` — and the
unit function by name. `determinism` is the slope `k` of the logistic
function in Albantakis et al. (2023, Eq. 60) and Marshall et al. (2023,
Eq. 2); inputs enter as ±1.

```{code-cell} python
from pyphi.substrate_generator import build_substrate

w = np.array([
    [0.2, 0.4, 0.1, 0.3],
    [0.3, 0.2, 0.4, 0.1],
    [0.1, 0.3, 0.2, 0.4],
    [0.4, 0.1, 0.3, 0.2],
])
logistic = build_substrate("sigmoid", w, determinism=3.0, node_labels=("A", "B", "C", "D"))
logistic
```

The connectivity matrix is read from the nonzero weights. This is the
construction behind every `iit4_2023_*` and `marshall_2023_*` example in
{mod}`pyphi.examples`; open one of them to see the weights of a published
network.

## From a function per unit

Logic gates and other named mechanisms go through
{func}`~pyphi.substrate_generator.create_substrate`, one specification per
unit: the mechanism's name, its inputs, and any parameters.

```{code-cell} python
from pyphi.substrate_generator import create_substrate

gates = create_substrate(
    [
        {"mechanism": "or", "inputs": (1, 2)},
        {"mechanism": "and", "inputs": (0, 2)},
        {"mechanism": "xor", "inputs": (0, 1)},
    ],
    labels=("A", "B", "C"),
)
gates
```

The mechanism names are the keys of
{data}`~pyphi.substrate_generator.MECHANISMS`. A unit you write yourself is a
function `f(element, weights, state, **params)` returning the probability
that `element` is ON at the next step; pass it (or a list mixing functions
and names) to {func}`~pyphi.substrate_generator.build_substrate` with a
weight matrix.

## From recorded transitions

{func}`pyphi.estimate_substrate` fits a posterior over transition
probabilities to observed `(current, next)` state pairs. IIT's analysis is
defined on an *interventional* matrix — what each unit does when the system
is put into each state — so say which regime produced the data:
`"perturbational"` when states were set, `"observational"` when they were
only recorded.

```{code-cell} python
rng = np.random.default_rng(0)
current = rng.integers(0, 2, size=(200, 3))
next_state = (rng.random((200, 3)) < tpm[np.ravel_multi_index(current.T[::-1], (2, 2, 2))]).astype(int)
posterior = pyphi.estimate_substrate((current, next_state), regime="perturbational")
posterior.mean_substrate()
```

The posterior's mean substrate is a reference point, not an estimate of
φ: analyzing it mixes what is unknown about the matrix with the substrate's
own indeterminism. Sample substrates from the posterior to carry that
uncertainty through an analysis; see {doc}`the what's-new tour
</whats-new-in-2.0>` for the estimation section.

## Units with more than two states

Pass `alphabet=` (one size for every unit) or `state_space=` (a tuple of
state labels per unit). The number of rows is then the product of the
alphabet sizes, still with the first unit changing fastest.

## Check the substrate before analyzing it

- `substrate` prints the units, the connectivity, and (at higher verbosity)
  the matrix; a wrong row order shows up here as a transition you did not
  intend.
- PyPhi rejects a matrix whose units are not conditionally independent
  given the previous state (a hidden common cause); see
  {doc}`../theory/conditional-independence`.
- A state the substrate cannot reach is refused at analysis time
  ({doc}`FAQ <faq>`).
- Before a large run, count the work: {doc}`estimate-cost`.

## Where to go next

- {doc}`Getting started <../getting-started/index>` analyzes a substrate end to end.
- {doc}`Read a result <read-result>` explains what comes back.
- {doc}`../theory/substrate-and-system` gives the theory behind the matrix.
````

The estimation cell: build `next_state` by drawing each unit ON with the row
probability for the sampled current state; the `ravel_multi_index` with the
reversed columns converts a little-endian state to its row index. Run the
cell; if the expression is awkward, replace it with an explicit loop over
rows — clarity matters more than brevity on this page.

- [ ] **Step 2: Wire the links**

`docs/howto/index.md`: add `build-substrate` as the first toctree entry and
`read-result` (Task 9) second, `estimate-cost` (Task 12) third, `faq` (Task 14)
last — add each when its task lands.

`docs/getting-started/index.md`, "Where to go next": insert as the first
bullet "- To analyze a network of your own, {doc}`build a substrate
<../howto/build-substrate>` from a transition probability matrix, a weight
matrix, or a function per unit."

`docs/theory/substrate-and-system.md`, after the `substrate.cm` cell: "To
build a substrate of your own from a matrix, weights, or unit functions, see
{doc}`../howto/build-substrate`."

- [ ] **Step 3: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t8docs.log 2>&1; echo $?; grep -n "build succeeded\|Error\|WARNING: " /tmp/t8docs.log | tail -5
cat > changelog.d/howto-build-substrate.doc.md <<'EOF2'
New how-to, "Build a substrate": from a transition probability matrix, from a weight matrix with logistic units, from a function per unit, and from recorded transitions, with the checks to run before analyzing.
EOF2
git add docs/howto/build-substrate.md docs/howto/index.md docs/getting-started/index.md docs/getting-started/index.ipynb docs/theory/substrate-and-system.md changelog.d/howto-build-substrate.doc.md
git commit -m "Add the how-to for building a substrate"
```

---

### Task 9: How-to "Read a result" (B2)

**Files:**
- Create: `docs/howto/read-result.md`
- Modify: `docs/howto/index.md`
- Modify: `docs/getting-started/index.md` ("Read the results": one sentence + link) and its `.ipynb`
- Modify: `pyphi/mcp/content/interpreting-iit-results.md` (three additions)

**Interfaces:**
- Consumes: `Analysis.formalism` and the card's `Formalism` row (Task 2); the card's `System` section and the summary keys `intrinsic_information`, `requirement_binding` (Task 6).

- [ ] **Step 1: Write the page**

````markdown
# Read a result

{func}`pyphi.analyze` returns an {class}`~pyphi.analyze.Analysis`. This page
walks through what it prints and what each row means, then through the
system irreducibility analysis underneath it and the questions to ask when
a value surprises you.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
analysis = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))
analysis
```

## The analysis card

- **Formalism** — which version of the theory produced the numbers. Every
  other row depends on it; a φ value reported without it cannot be
  compared with anything. It is `analysis.formalism`.
- **Φ** — the structure integrated information: the sum of φ over every
  distinction and every relation. It measures how much structure the system
  specifies. It is `analysis.big_phi`. Under IIT 3.0 there are no relations
  and this row is absent; that formalism's Φ is the next row.
- **φ_s** — the system integrated information: whether the system exists
  as one whole, and how irreducibly. It is `analysis.phi`. Zero means
  *reducible*, not "no structure": a system can have φ_s = 0 and a nonzero Φ.
- **Distinctions, Σφ_d** — how many mechanisms specify an irreducible
  cause–effect state, and their total φ.
- **Relations, Σφ_r** — how many congruent overlaps bind those distinctions,
  and their total φ. Under the default analytical backend these are computed
  in closed form; the individual relations are not enumerated, so they
  cannot be listed one by one. {doc}`query-relations` shows what can be asked
  of them and how to enumerate when you must.
- **The distinction table** — one row per distinction: its mechanism, φ_d,
  and its cause and effect purviews, each written in the *state* the
  distinction specifies: an uppercase letter is a unit ON, lowercase is
  OFF, and a unit with more than two states carries its state as a
  subscript (`A₂`). "Cause purview `b`" means the mechanism specifies unit
  B being OFF in the past.
- **System: MIP, ii(s), Requirement binds** — the minimum information
  partition (the cut that makes the least difference: the system's weakest
  link), the system's intrinsic information, and, when the
  intrinsic-information requirement set φ_s, which term and direction did
  so. The next section explains these.

## The system irreducibility analysis

```{code-cell} python
sia = analysis.sia
sia
```

- **Normalized φ_s** — φ_s divided by the partition's normalization; the
  minimum information partition is chosen on this value.
- **Specified state** (cause and effect) — the past and future states the
  system specifies with maximal intrinsic information.
- **Intrinsic specification** (labelled **Intrinsic information** under the
  2023 formalism, which is that paper's name for the same quantity) — how
  selectively and informatively the specified state is picked out.
- **Intrinsic differentiation** — the surprisal of the specified state: how
  much of a repertoire of alternatives the system provides itself. Zero for
  a deterministic transition.
- **MIP** — the partition and, in the grid, the connections it severs; "Tied
  MIPs" counts partitions tied with it.

Under the default formalism, IIT 4.0 (2026), φ_s is the smallest of three
terms: the cause-side integration φ_c, the effect-side integration φ_e, and
the intrinsic information ii(s), itself the smaller of specification and
differentiation over both directions (see
{doc}`../theory/intrinsic-information`). `sia.explain()` says which term
won:

```{code-cell} python
for finding in sia.explain().findings:
    print(finding.kind, "=", finding.value)
```

## When φ_s is zero

There are two different reasons, and the findings above tell them apart.

**Ordinary reducibility.** One side's integration is already zero: some
partition of the system makes no difference to its cause or effect
repertoire. Then `binding_direction` names that side and there is no
`requirement_binding` finding. This happens under every formalism.

**The intrinsic-information requirement.** Both φ_c and φ_e are positive,
but the system provides itself no repertoire of alternatives — a
deterministic transition has zero differentiation — so ii(s) is zero and
with it φ_s. Then a `requirement_binding` finding names the term
(`differentiation` or `specification`) and the direction. This happens only
under IIT 4.0 (2026); the same system under `formalism="IIT_4_0_2023"` keeps
its `min(φ_c, φ_e)`.

```{code-cell} python
basic = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), compute="sia")
print(float(basic.cause.phi), float(basic.effect.phi), basic.intrinsic_information)
[f.value for f in basic.explain().findings if f.kind == "requirement_binding"]
```

## Reading the numbers across formalisms

The same substrate and state give different φ_s under each formalism,
because each defines it differently — see {doc}`../theory/formalism-versions`.
Compare values only within one formalism, and compare them tolerantly:
`pyphi.numerics.eq(a, b)` respects the configured precision where `==` does
not.

## Where to go next

- {doc}`../theory/overview` for what these quantities are in the theory.
- {doc}`tie-breaking` for the selection margins and what an "effectively
  tied" result means.
- {doc}`sweep` to compute the same quantities over every state at once.
````

Run the `basic` cell before committing and adjust the printed claim to what
it shows (it must show positive φ_c and φ_e with ii(s) = 0 and a
`differentiation` finding; the MCP walkthrough saw exactly that in this
state).

- [ ] **Step 2: Wire the links; mirror into the MCP topic**

`docs/getting-started/index.md`, end of "Read the results": "Every row of
the card is explained in {doc}`Read a result <../howto/read-result>`."

`pyphi/mcp/content/interpreting-iit-results.md`, add three bullets under
"What `analyze` gives back":

```
- `summary.intrinsic_information` — **ii(s)**, the third term of φₛ under
  IIT 4.0 (2026); `summary.requirement_binding` names the term and direction
  when ii(s) is what set φₛ (both φ_c and φ_e positive, φₛ = 0 is the usual
  sign). `None` when integration set it, and under other formalisms.
- Purview and mechanism labels are written in the specified *state*:
  uppercase is ON, lowercase is OFF, and a subscript gives the state of a
  unit with more than two states (`A₂`).
- Under the default analytical backend, relation counts and Σφ_r are
  closed-form; individual relations are not enumerated, so `inspect` cannot
  list them. Say so rather than inventing them.
```

- [ ] **Step 3: Build, test, commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t9docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t9docs.log
uv run pytest test/mcp/test_content.py -q > /tmp/t9.log 2>&1; echo $?; tail -1 /tmp/t9.log
cat > changelog.d/howto-read-result.doc.md <<'EOF2'
New how-to, "Read a result": every row of the analysis and system cards, the letter-case convention of purview labels, and the two different reasons a system's φₛ can be zero.
EOF2
git add docs/howto/read-result.md docs/howto/index.md docs/getting-started/index.md docs/getting-started/index.ipynb pyphi/mcp/content/interpreting-iit-results.md changelog.d/howto-read-result.doc.md
git commit -m "Add the how-to for reading a result"
```

---

### Task 10: Two causes of φ_s = 0 on the theory page; MCP theory caveat (B9)

**Files:**
- Modify: `docs/theory/intrinsic-information.md` (new short section after "Differentiation and determinism"; a link for φ_c/φ_e)
- Modify: `pyphi/mcp/content/theory.md` ("φₛ versus Φ" paragraph)

- [ ] **Step 1: Theory page**

Where the page first uses $\varphi_c$ and $\varphi_e$ (the sentence "Both
directions have substantial integration..."), add: "($\varphi_c$ and
$\varphi_e$ are the cause- and effect-side integrated information of
{doc}`system-integration`.)"

After the "Differentiation and determinism" section, add:

````markdown
## Two ways to reach zero

A system's $\varphi_s$ is zero either because one side's integration is
already zero — some partition makes no difference, under every formalism —
or because the requirement binds: both $\varphi_c$ and $\varphi_e$ are
positive and $\mathit{ii}(s)$ is zero. `explain()` distinguishes them: the
second case carries a `requirement_binding` finding, the first does not.

```{code-cell} python
basic = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), compute="sia")
(float(basic.cause.phi), float(basic.effect.phi), basic.intrinsic_information,
 [f.value for f in basic.explain().findings if f.kind == "requirement_binding"])
```

The XOR network above is the same case. A network whose cause side is
reducible outright shows $\varphi_c = 0$ and no such finding; see
{doc}`Read a result <../howto/read-result>`.
````

- [ ] **Step 2: MCP theory**

In `pyphi/mcp/content/theory.md`, after "φₛ is the smaller of the two — a
system is only as integrated as its weaker direction." add: "Under the
default IIT 4.0 (2026) a third term enters the minimum, the intrinsic
information ii(s), so φₛ can be 0 with both sides positive; see Formalism
versions below and the `requirement_binding` key of the `analyze` summary."

- [ ] **Step 3: Build, test, commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t10docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t10docs.log
uv run pytest test/mcp/test_content.py -q > /tmp/t10.log 2>&1; echo $?; tail -1 /tmp/t10.log
git add docs/theory/intrinsic-information.md pyphi/mcp/content/theory.md
git commit -m "Explain the two ways phi_s reaches zero"
```

---

### Task 11: Migration guide additions (B5)

**Files:**
- Modify: `docs/migration/migration-2.0.md`
- Modify: `pyphi/mcp/content/migration.md` (the same three tables, condensed)
- Modify: `docs/theory/formalism-versions.md` ("What `formalism=` sets": one paragraph)

- [ ] **Step 1: Gather the facts**

```bash
uv run python -c "import pyphi; print(sorted(n for n in dir(pyphi.examples) if not n.startswith('_')))" 2>/dev/null
grep -n "rename\|LEGACY\|legacy" pyphi/conf/*.py | head -20
```
The second command finds the flat-to-nested rename map that the config
loader uses to reject a 1.x `pyphi_config.yml`; the table below is built
from it, one row per legacy key, in the map's order.

- [ ] **Step 2: Three tables**

After "Renames at a glance", add:

````markdown
### Example networks

The example functions follow the same vocabulary: every `*_network` is now
`*_substrate` and every `*_subsystem` is `*_system`.

| Old | New |
| --- | --- |
| `examples.basic_network()` | `examples.basic_substrate()` |
| `examples.basic_subsystem()` | `examples.basic_system()` |
| `examples.xor_network()` | `examples.xor_substrate()` |
| `examples.residue_network()` | `examples.residue_substrate()` |
| `examples.rule110_network()` | `examples.rule110_substrate()` |
| `examples.fig4()` | `examples.fig4_substrate()` |

(Complete the table from the `dir()` listing: every name in 2.0 ending in
`_substrate` or `_system` whose stem existed in 1.x.)

### Configuration options

| 1.x option | 2.0 option |
| --- | --- |
| `MEASURE` | `formalism.iit.mechanism_phi_measure` (the mechanism-level measure) and `formalism.iit.ces_measure` (the structure distance); values are the measure names of `pyphi.measures` |
| `PARTITION_TYPE` | `formalism.iit.mechanism_partition_scheme` |
| `PICK_SMALLEST_PURVIEW` | `formalism.iit.purview_tie_resolution` |
| `PRECISION` | `numerics.precision` |
| `PARALLEL_*` | `infrastructure.parallel` and the per-level options; see {doc}`/howto/parallel` |
| `CACHE_*` | `infrastructure.cache_*`; see {doc}`/howto/cache` |
| `LOG_*` | `pyphi.enable_logging()` |

(Fill the table from the loader's rename map; every legacy key must appear,
and every 2.0 name must exist on the config classes.)

### The quantities

| 1.x | 2.0 |
| --- | --- |
| `compute.big_phi(subsystem)` — IIT 3.0's Φ | `analyze(substrate, state, formalism="IIT_3_0").phi` — the same quantity; `.big_phi` is IIT 4.0's structure integrated information and raises under IIT 3.0 |
| `compute.sia(subsystem).phi` | `analyze(...).sia.phi` |
| `compute.ces(subsystem)` — concepts | `analyze(..., formalism="IIT_3_0").ces` — a `ResolvedDistinctions`; the concepts are `.concepts` |
| `concept.phi`, `concept.mechanism` | unchanged on each concept |
````

- [ ] **Step 3: One place for the three ways to choose a formalism**

In `docs/theory/formalism-versions.md`, "What `formalism=` sets", replace
the paragraph with:

"There are three equivalent ways to select a formalism, and every page
uses one of them: the `formalism=` argument of `pyphi.analyze` (per call);
`pyphi.config.override(**pyphi.iit4_2023)` (a block; the presets are
`pyphi.iit3`, `pyphi.iit4_2023`, `pyphi.iit4_2026`); and assigning a
preset's `iit` field to the configuration (`pyphi.config.formalism.iit =
pyphi.iit4_2023["iit"]`, for a whole session). All three set the version
together with the measures it requires. Setting `version` alone does not."

Link this paragraph from `migration-2.0.md` ("Choosing a formalism") and
from `howto/configure.md` ("Presets").

- [ ] **Step 4: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t11docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t11docs.log
uv run pytest test/mcp/test_content.py -q > /tmp/t11.log 2>&1; echo $?; tail -1 /tmp/t11.log
cat > changelog.d/migration-guide-tables.doc.md <<'EOF2'
The migration guide gains the example-network renames, a table from every 1.x configuration option to its 2.0 location, and a table of the quantities (1.x big Φ is `analysis.phi` under IIT 3.0).
EOF2
git add docs/migration/migration-2.0.md pyphi/mcp/content/migration.md docs/theory/formalism-versions.md docs/howto/configure.md changelog.d/migration-guide-tables.doc.md
git commit -m "Extend the migration guide to examples, options, and quantities"
```

---

### Task 12: How-to "Estimate the cost before you run" (B8, site part)

**Files:**
- Create: `docs/howto/estimate-cost.md`
- Modify: `docs/howto/index.md`
- Modify: `docs/theory/computational-complexity.md` ("Estimating a workload" wording; a warning at the enumerator demo)

- [ ] **Step 1: Write the page**

````markdown
# Estimate the cost before you run

Analyses grow faster than exponentially with the number of units. Count the
work first; it is far cheaper than discovering the answer by waiting.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The practical ceilings

| What | Formalism | About |
| --- | --- | --- |
| system integrated information φ_s | IIT 4.0 | 10–12 units |
| distinctions and relations (the Φ-structure) | IIT 4.0 | 6–8 units |
| cause–effect structure, Φ | IIT 3.0 | 10–12 units |

These are for fully connected substrates on one core; sparser connectivity
raises them, since absent connections shrink every search.

## Count the work

{func}`pyphi.cost.estimate_analysis` walks the same enumerations the
analysis would, without computing any φ. It needs no state.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig6d_substrate()
pyphi.cost.estimate_analysis(substrate)
```

The counting walk has its own budget, `limit` (one million steps by
default). On ten or more fully connected units it can take tens of seconds
and stop early, reporting `capped=True`; the counts are then lower bounds.
Raise the budget for an exact count when you need one:

```{code-cell} python
estimate = pyphi.cost.estimate_analysis(substrate, compute="sia", limit=10_000_000)
estimate.capped, estimate.system_partitions
```

`compute="sia"` counts only the system-partition search; `"distinctions"`
only the distinction axis; the default counts everything. The
system-partition count grows fastest with the number of units and is the
axis to watch for φ_s.

## What to reduce

- **Connectivity.** Pass the real connectivity matrix if you know it.
- **The candidate system.** Analyze a subset (`subset=`) rather than the
  whole substrate; `substrate.complexes(state)` still has to consider every
  subset.
- **What you compute.** `compute="sia"` for φ_s alone.
- **Settings.** The cost-reduction table in
  {doc}`../theory/computational-complexity` lists the partition schemes and
  short-circuit options and what each gives up.
- **Cores and clusters.** {doc}`parallel` divides the constants;
  {doc}`campaigns` shards one analysis across machines.

## Where to go next

- {doc}`../theory/computational-complexity` derives the scaling.
- {doc}`grain-search` has its own pre-flight estimate.
````

- [ ] **Step 2: Fix the complexity page**

In "Estimating a workload before running it", replace the clause that calls
the estimate free with: "It computes no φ, but its counting walk is itself
work: on ten fully connected units it takes tens of seconds, and it stops
at its `limit` budget with `capped=True` — see {doc}`../howto/estimate-cost`
for raising the budget."

Directly above the Bell-number code cell (`for n in range(1, 8)` or
similar): "The enumerators below are shown for small `n` to illustrate the
growth. Do not call `pyphi.partition.directed_set_partitions` or its
relatives directly on ten or more units: they materialize every partition.
Use {func}`pyphi.cost.estimate_analysis`, which counts without
enumerating."

- [ ] **Step 3: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t12docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t12docs.log
cat > changelog.d/howto-estimate-cost.doc.md <<'EOF2'
New how-to, "Estimate the cost before you run": the practical ceilings, `estimate_analysis` with its counting budget, and what to reduce.
EOF2
git add docs/howto/estimate-cost.md docs/howto/index.md docs/theory/computational-complexity.md changelog.d/howto-estimate-cost.doc.md
git commit -m "Add the cost-estimation how-to and correct the complexity page's framing"
```

---

### Task 13: Glossary (B6)

**Files:**
- Create: `docs/reference/glossary.md`
- Modify: `docs/reference/index.md` (toctree: `glossary` first)
- Modify: `docs/index.md` (the Reference card text: "The API reference, the glossary, configuration options, and conventions.")

- [ ] **Step 1: Write the page**

Sphinx's `{glossary}` directive with one entry per term; each definition is
one to three sentences and ends with the page that treats it. Write it as:

````markdown
# Glossary

Terms as PyPhi and the IIT 4.0 papers use them. Each entry points to the
page that treats it.

```{glossary}
substrate
  A set of units and the probability of each unit's next state given the
  current state of all of them: a causal model. {class}`~pyphi.substrate.Substrate`;
  {doc}`../theory/substrate-and-system`.

system
  A candidate subset of a substrate's units in a definite state, analyzed
  from its own perspective; the rest of the substrate is its
  {term}`background`. {class}`~pyphi.system.System`.

background
  The units outside a system. Under IIT 4.0 they are causally marginalized
  (averaged out conditional on the current state); under the IIT 3.0 preset
  they are fixed at their current state. `background_conditioning` in
  {doc}`../howto/configure`.

transition probability matrix (TPM)
  The substrate's probabilities of next states given current states, in
  state-by-node form: one row per current state (first unit changing
  fastest), one column per unit. {doc}`../howto/build-substrate`;
  {ref}`tpm-conventions`.

little-endian
  PyPhi's state order: the first unit is the least significant bit, so the
  state `(1, 0, 0)` is row 1 of the matrix and `(0, 0, 1)` is row 4.

repertoire
  A probability distribution over the states of a purview: the cause
  repertoire over past states, the effect repertoire over future states,
  as constrained by a mechanism in its state.

mechanism
  A subset of the system's units, in their current state, considered for
  the cause–effect state it specifies.

purview
  The subset of units over which a mechanism's cause or effect is assessed.
  In result cards a purview is written in the state it is specified in:
  uppercase ON, lowercase OFF. {doc}`../howto/read-result`.

specified state
  The purview (or system) state that maximizes intrinsic information: the
  cause and effect the mechanism (or system) selects.

intrinsic information
  For a mechanism or a system: how selectively and informatively it picks
  out its specified state. Under IIT 4.0 (2026) the *system's* intrinsic
  information ii(s) is the smaller of its {term}`intrinsic specification`
  and {term}`intrinsic differentiation` over both directions.
  {doc}`../theory/intrinsic-information`.

intrinsic specification
  The selectivity times informativeness of the specified state (Mayner et
  al. 2026, Eqs. 7 and 9); Albantakis et al. (2023) call the same quantity
  intrinsic information, and the card follows the formalism's own name.

intrinsic differentiation
  The surprisal of the specified state: how much of a repertoire of
  alternatives the system provides itself. Zero for a deterministic
  transition. {doc}`../theory/intrinsic-information`.

intrinsic-information requirement
  Under IIT 4.0 (2026), φₛ = min{φ_c, φ_e, ii(s)}: the system's integrated
  information cannot exceed its intrinsic information (Mayner et al. 2026,
  Eq. 23). {doc}`../theory/intrinsic-information`.

partition
  A cut of a system or mechanism into parts whose connections are severed,
  to test irreducibility. {doc}`../theory/system-integration`.

minimum information partition (MIP)
  The partition that makes the least difference, judged on normalized φ:
  the system's weakest link. Reported on every system irreducibility
  analysis.

integrated information (φ)
  How much a partition changes what is specified. φ_d for a distinction,
  φ_r for a relation, φₛ for a system.

system integrated information (φₛ)
  The irreducibility of a system's specified cause–effect state over its
  minimum information partition; whether the system exists as one whole.
  `analysis.phi`. {doc}`../theory/system-integration`.

structure integrated information (Φ)
  The sum of φ over every distinction and relation of a Φ-structure; how
  much structure the system specifies. `analysis.big_phi`; not defined
  under IIT 3.0, whose Φ is the system-level value.
  {doc}`../theory/phi-structure`.

distinction
  A mechanism together with the cause and effect states it irreducibly
  specifies over its maximally irreducible purviews, with its φ_d.
  IIT 3.0's name is *concept*. {doc}`../theory/distinctions-and-relations`.

relation
  A congruent overlap among the purviews of two or more distinctions — the
  same units specified in the same states — with its φ_r.
  {doc}`../theory/distinctions-and-relations`; {doc}`../howto/query-relations`.

Φ-structure (cause–effect structure)
  The distinctions a complex specifies and the relations among them.
  {class}`~pyphi.models.ces.CauseEffectStructure`; {doc}`../theory/phi-structure`.

complex
  A set of units whose φₛ is maximal among all sets overlapping it; the
  exclusion postulate's answer to which units exist as one whole.
  {meth}`~pyphi.substrate.Substrate.complexes`; {doc}`../tutorials/recursive-exclusion`.

maximally irreducible cause and effect (MICE)
  For a mechanism: the cause purview and effect purview with the highest
  φ, the search a distinction's computation performs.

selection margin
  How far the winning partition (or state) was from its nearest competitor;
  zero means an effective tie. {doc}`../howto/tie-breaking`.

formalism
  The set of rules that turns a system into results: IIT 4.0 (2026), IIT
  4.0 (2023), IIT 3.0, or actual causation. {doc}`../theory/formalism-versions`.

actual causation
  The analysis of what caused what in one observed transition, measured in
  α (bits); a separate formalism from φ and Φ.
  {doc}`../tutorials/actual-causation`.

macro unit
  A unit defined over several micro units by coarse-graining or
  blackboxing, admitted when it is maximally irreducible within.
  {doc}`../theory/macro-units`.
```
````

Then replace the first prose use of each defined term on
`getting-started/index.md` and `theory/overview.md` with a `{term}` role
(`{term}`complex``), at most one per term per page.

- [ ] **Step 2: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t13docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t13docs.log
cat > changelog.d/glossary.doc.md <<'EOF2'
A glossary of the terms the documentation and the IIT 4.0 papers use, each entry pointing to the page that treats it.
EOF2
git add docs/reference/glossary.md docs/reference/index.md docs/index.md docs/getting-started/index.md docs/getting-started/index.ipynb docs/theory/overview.md changelog.d/glossary.doc.md
git commit -m "Add a glossary"
```

---

### Task 14: FAQ and troubleshooting (B7)

**Files:**
- Create: `docs/howto/faq.md`
- Modify: `docs/howto/index.md` (last entry)
- Modify: `docs/index.md` (How-to card text: add "and a FAQ")

- [ ] **Step 1: Write the page**

Static page (no execution), one `##` per question, answers of three to
eight sentences each with a link:

````markdown
# FAQ and troubleshooting

## Why is φ_s zero?

Two reasons, and `analysis.sia.explain()` tells them apart. Either one side
of the system is reducible outright (some partition makes no difference to
its cause or its effect repertoire), or, under the default IIT 4.0 (2026),
the system provides itself no repertoire of alternatives, so its intrinsic
information is zero: every deterministic network is in this second case.
{doc}`read-result` walks through both; {doc}`../theory/intrinsic-information`
explains the requirement. To see the value without it, pass
`formalism="IIT_4_0_2023"`.

## My numbers differ from a paper, or from PyPhi 1.x

Check the formalism first: `analysis.formalism`. Published IIT 4.0 values
from 2023 need `formalism="IIT_4_0_2023"`; 1.x values need
`formalism="IIT_3_0"` (1.x's Φ is `analysis.phi` under that formalism,
not `analysis.big_phi`). {doc}`../theory/formalism-versions` lists what
each changes; {doc}`../migration/migration-2.0` covers 1.x. If the formalism
matches and a value still differs in the last decimals, compare with
`pyphi.numerics.eq`, which respects the configured precision.

## `StateUnreachableForwardsError`

The state you asked about cannot be produced by the substrate's own
dynamics: no current state leads to it. IIT evaluates a system in a state
it could have reached, so PyPhi refuses rather than assigning a structure
that rests on an impossible past. Deterministic toy networks hit this
often — in the three-XOR network every reachable state has even parity.
Choose a reachable state (`pyphi.sweep(substrate, states="all").skipped`
lists the unreachable ones) or check the matrix's row order
({doc}`build-substrate`).

## `ConditionallyDependentError`

The matrix says two units' next states depend on each other at the same
time step, which a substrate of conditionally independent units cannot
express; it signals a hidden common cause. Add the shared variable as a
unit, or rebuild the matrix from each unit's own input–output function.
{doc}`../theory/conditional-independence`.

## The estimate says `capped=True`

The counting walk stopped at its budget; the counts are lower bounds. Raise
`limit` for an exact count, or read the counts as "at least this much" and
reduce the work: {doc}`estimate-cost`.

## The analysis is taking hours

It is probably past the practical ceiling: about 10–12 units for φ_s and
6–8 for the full Φ-structure on a fully connected substrate. Stop it,
count the work with `pyphi.cost.estimate_analysis`, and reduce it
({doc}`estimate-cost`). PyPhi has no checkpointing; a killed run loses its
progress, so for long sweeps turn on the disk result cache
({doc}`cache`).

## The result says "effectively tied"

Two partitions or two specified states came within the configured
precision of each other; small symmetric networks do this constantly. The
selection is still deterministic and follows the postulates, and the
margins say how close it was. {doc}`tie-breaking`.

## Two calls with the same inputs gave different results

Something in the configuration changed between them — an `override` block
still open, a `pyphi_config.yml` in one working directory and not the
other, or a preset applied in one session. `pyphi.config` prints the
active settings; pin the formalism per call with `formalism=`.
{doc}`configure`.

## Where is the API reference?

Under {doc}`Reference </reference/index>` on the built site (it is
generated from the docstrings and is not in the source tree), and in the
interpreter with `help()`.
````

- [ ] **Step 2: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t14docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t14docs.log
cat > changelog.d/faq.doc.md <<'EOF2'
A FAQ and troubleshooting page: zero φₛ, numbers that differ from a paper or from 1.x, unreachable states, conditional dependence, capped estimates, long runs, ties, and configuration drift.
EOF2
git add docs/howto/faq.md docs/howto/index.md docs/index.md changelog.d/faq.doc.md
git commit -m "Add a FAQ and troubleshooting page"
```

---

### Task 15: Configuration reference (B11)

**Files:**
- Create: `docs/reference/configuration.md`
- Modify: `docs/reference/index.md` (toctree)
- Modify: `docs/howto/configure.md` (the closing pointer)

- [ ] **Step 1: Write the page**

An executed page: a table of every option with its layer and default,
generated from the dataclasses so it cannot drift, followed by the three
classes' own docstrings (which document each option in prose).

````markdown
# Configuration options

Every option, its layer, and its default. The three layers are described in
{doc}`../howto/configure`; each class's documentation below explains the
options in prose.

```{code-cell} python
:tags: [hide-input]
import dataclasses
import pandas as pd
from pyphi.conf.formalism import ActualCausationConfig, IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig

rows = []
for layer, cls in (
    ("formalism.iit", IITConfig),
    ("formalism.actual_causation", ActualCausationConfig),
    ("infrastructure", InfrastructureConfig),
    ("numerics", NumericsConfig),
):
    instance = cls()
    for field in dataclasses.fields(cls):
        rows.append({"option": field.name, "layer": layer, "default": repr(getattr(instance, field.name))})
pd.set_option("display.max_rows", None)
pd.DataFrame(rows).set_index("option")
```

## Formalism: IIT

```{eval-rst}
.. autoclass:: pyphi.conf.formalism.IITConfig
   :noindex:
```

## Formalism: actual causation

```{eval-rst}
.. autoclass:: pyphi.conf.formalism.ActualCausationConfig
   :noindex:
```

## Infrastructure

```{eval-rst}
.. autoclass:: pyphi.conf.infrastructure.InfrastructureConfig
   :noindex:
```

## Numerics

```{eval-rst}
.. autoclass:: pyphi.conf.numerics.NumericsConfig
   :noindex:
```
````

Check the actual class names and modules with `grep -n "^class" pyphi/conf/*.py`
and adjust. If `pd.DataFrame` renders poorly, print a Markdown table with
`to_markdown()` inside a `{glue}` or just `print` it as text.

In `docs/howto/configure.md`, replace the final sentence's pointer to the
API reference with "see {doc}`the configuration reference
</reference/configuration>` for every option with its default".

- [ ] **Step 2: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t15docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t15docs.log
cat > changelog.d/configuration-reference.doc.md <<'EOF2'
A configuration reference page lists every option with its layer and default, generated from the configuration classes, followed by their documentation.
EOF2
git add docs/reference/configuration.md docs/reference/index.md docs/howto/configure.md changelog.d/configuration-reference.doc.md
git commit -m "Add the configuration reference page"
```

---

### Task 16: Navigation fixes (B12)

**Files:**
- Modify: `docs/tutorials/actual-causation.md` (+ `.ipynb`), `docs/howto/export.md`, `docs/howto/save-load.md`, `docs/conventions.rst`, `docs/migration/from-substrate-modeler.md` (a "Where to go next" section each)
- Modify: `docs/index.md` (What's new card; task index)
- Modify: `docs/howto/index.md` (an "I want to…" list above the toctree)
- Modify: `README.md` (Documentation section; release note link)

- [ ] **Step 1: Onward links**

Append to each dead-end page a `## Where to go next` with two or three
`{doc}` links (`conventions.rst` uses `:doc:` roles):

- actual-causation → `causal-reductionism`, `../theory/formalism-versions`, `../howto/build-substrate`
- export → `save-load`, `sweep`, `visualize`
- save-load → `export`, `cache`, `campaigns`
- conventions → `/howto/build-substrate`, `/reference/glossary`
- from-substrate-modeler → `migration-2.0`, `/howto/build-substrate`

- [ ] **Step 2: Index and how-to landing**

`docs/index.md`: add a seventh grid card, "What's new in 2.0", link
`whats-new-in-2.0`, text "The tour of the release, and what changed from
1.x." (change the grid to `1 2 3 4` or leave `1 2 3 3`; check the render).

`docs/howto/index.md`: above the toctree add

```markdown
I want to…

- analyze my own network → {doc}`build-substrate`
- understand what a result means → {doc}`read-result`
- know whether a run will finish → {doc}`estimate-cost`
- compute over every state → {doc}`sweep`
- reproduce a published number → {doc}`../tutorials/worked-example` and {doc}`../theory/formalism-versions`
- run on many cores or a cluster → {doc}`parallel`, {doc}`campaigns`
- fix an error → {doc}`faq`
```

- [ ] **Step 3: README**

In "Documentation", add a first bullet "- [Getting started](https://pyphi.readthedocs.io/en/stable/getting-started/) — install and a first computation in ten minutes" (check the built URL path). In the release-status note, point "What's new in 2.0" at `https://pyphi.readthedocs.io/en/stable/whats-new-in-2.0.html`.

- [ ] **Step 4: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t16docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t16docs.log
git add docs README.md
git commit -m "Link the dead-end pages onward and add a task index"
```

---

### Task 17: Examples gallery (B13)

**Files:**
- Create: `docs/reference/examples.md`
- Modify: `docs/reference/index.md`

- [ ] **Step 1: Write the page**

````markdown
# Example networks

Every registered example, generated from {mod}`pyphi.examples` at build
time. Substrates and systems load with `pyphi.examples.<name>()`; the first
line of each docstring names its source.

```{code-cell} python
:tags: [hide-input]
import inspect
import pandas as pd
import pyphi
from pyphi.examples import EXAMPLES

pyphi.config.progress_bars = False
rows = []
for category in ("substrate", "system", "transition", "tpm"):
    for name, func in sorted(EXAMPLES[category].items()):
        doc = (inspect.getdoc(func) or "").split("\n")[0]
        size = ""
        if category == "substrate":
            try:
                size = func().size
            except TypeError:  # takes required arguments
                size = "parametrized"
        rows.append({"name": f"{name}_{category}", "category": category, "units": size, "source": doc})
pd.set_option("display.max_rows", None); pd.set_option("display.max_colwidth", None)
pd.DataFrame(rows).set_index("name")
```
````

Building every substrate at docs time is cheap (the largest is eight
units). If any example function has required arguments, the `TypeError`
branch marks it; if any raises otherwise, fix the loop rather than the
example.

- [ ] **Step 2: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t17docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t17docs.log
cat > changelog.d/examples-gallery.doc.md <<'EOF2'
A reference page lists every registered example network with its size and source, generated from the registry at build time.
EOF2
git add docs/reference/examples.md docs/reference/index.md changelog.d/examples-gallery.doc.md
git commit -m "Add the example-network gallery"
```

---

### Task 18: Contributing page (B16)

**Files:**
- Create: `docs/contributing.md`
- Modify: `docs/index.md` (hidden toctree: `contributing` last)
- Modify: `README.md` ("Contributing": one added line linking the page)

- [ ] **Step 1: Write the page**

Start from the README's Contributing section (fork, clone, `uv sync
--all-extras --group dev`, the `just` recipes), then add: how the test suite
is organized (fast lane `uv run pytest`, slow lane `uv run pytest -m slow
--slow`, doctests run from the package), how to add a paper reproduction
(pin the formalism with a preset, quote the figure, perturbation-verify),
changelog fragments (`changelog.d/<name>.<type>.md`), the docs build (`just
docs`, executed pages, `-W`), and where to file issues. Keep it to one
screen; link `RELEASING.md` is not needed.

- [ ] **Step 2: Build and commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t18docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t18docs.log
git add docs/contributing.md docs/index.md README.md
git commit -m "Add a contributing page to the docs"
```

---

### Task 19: Verification and merge

- [ ] **Step 1: Suites**

```bash
export PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1
uv run pytest -q > /tmp/final-fast.log 2>&1; echo $?; tail -1 /tmp/final-fast.log
uv run pytest -m slow --slow -q > /tmp/final-slow.log 2>&1; echo $?; tail -1 /tmp/final-slow.log
uv run pytest test/integration/test_perf_counters.py -q > /tmp/final-perf.log 2>&1; echo $?; tail -1 /tmp/final-perf.log
```
Expected: all exit 0, no failures.

- [ ] **Step 2: Docs and notebook**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/final-docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/final-docs.log
```
Expected: "build succeeded". Open `docs/_build/html/howto/build-substrate.html`,
`read-result.html`, `estimate-cost.html`, `faq.html`, `reference/glossary.html`,
`reference/configuration.html`, `reference/examples.html` in a browser and
read each once through: every code cell shows output, no cell shows a
traceback, tables render.

- [ ] **Step 3: Re-run two walkthrough probes**

Spot-check the two failures the review found, from a neutral directory
with only the docs open: build the computational scientist's 8×3 matrix
from `howto/build-substrate.md` alone, and reproduce 0.17 for aB through
the MCP `analyze` tool with `subset=["A","B"]`. Both must work as the pages
say.

- [ ] **Step 4: Merge (no push)**

```bash
cd /Users/will/projects/pyphi
git merge --no-ff docs-ux -m "Merge the documentation UX backlog"
git worktree remove .claude/worktrees/docs-ux
git branch -d docs-ux
```

Then fold the new fragments into the 2.0.0 changelog section (the release
is not yet tagged): each `.doc.md` fragment goes at the end of
"Documentation", each `.feature.md` at the end of "API additions", each
`.change.md` at the end of "API changes", in the same wrapped-bullet format
with the fragment name in parentheses; then `git rm` the fragments and
commit "Fold the documentation-pass fragments into the 2.0.0 changelog".
