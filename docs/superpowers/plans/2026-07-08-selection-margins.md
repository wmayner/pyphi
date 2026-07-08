# Selection-Margin Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement item 2 of
`docs/superpowers/specs/2026-07-07-substrate-parameter-landscapes.md` §7:
expose two selection margins on the IIT 4.0 SIA and its `explain()` surface —
the normalized-φ gap between the MIP and the second-best system partition, and
the intrinsic-information gap between the specified system state and the
second-best state (per direction) — plus a flag for whether either selection
is effectively tied at `config.numerics.precision`. Margins are the IIT-native
sensitivity analysis: a small margin means the substrate is near a boundary
where what it specifies changes discretely. Three consumers (sensitivity
analysis, uncertainty screening, tie-risk flagging) share this one primitive.
Concrete slice of roadmap N23.

**Architecture:** Both margins are computed from values the SIA search already
produces and currently discards. The state side retains the second-best
per-state intrinsic-information value inside
`pyphi.core.repertoire_algebra.intrinsic_information` (which already builds
the full `state → ii` dict) as two new optional fields on
`StateSpecification`. The partition side computes the margin in
`_find_mip_for_fixed_state` from the candidate SIAs already materialized for
MIP selection (the same list B8's `runner_up` is drawn from) and stores it as
a new optional field on the IIT 4.0 `SystemIrreducibilityAnalysis`. No second
pass over partitions or states, and no change to any computed φ value.
IIT 3.0 and actual causation are out of scope: their selection structure is
different (IIT 3.0 selects on raw Φ over CES distances with no normalized-φ /
specified-state selection pair; AC selects causal links on α), so margin
reporting there is deferred.

**Tech Stack:** Python 3.13, numpy, msgspec, pytest. No new dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start; base on the current working branch).
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on φ or ii values.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts (no task numbers, no "B8"/"N23", no design-alternative
  discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 5) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).

## Background for implementers (read once)

**The two selection sites.** The IIT 4.0 SIA
(`pyphi/formalism/iit4/__init__.py`) selects twice:

1. *State selection* — `system_intrinsic_information` (line ~78) calls
   `System.intrinsic_information`, which delegates to
   `pyphi.core.repertoire_algebra.intrinsic_information` (line ~543). That
   kernel builds `state_to_information` — one ii value per candidate state —
   takes the max, wraps exact-max ties in a `StateSpecification` tie set, and
   **discards every sub-maximal value**. The state margin is the gap between
   the max and the best remaining value; this task retains it.
2. *Partition selection* — `_find_mip_for_fixed_state` (line ~1012)
   materializes one candidate SIA per system partition via `map_reduce`,
   selects the MIP with `resolve_ties.sias` (default key
   `["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]`, `operation=min` —
   i.e. selection is primarily on **clamped normalized φ**), and already
   retains a `RunnerUp(partition, phi)` on `sia.runner_up` via
   `runner_up_from_candidates` (`pyphi/models/explanation.py:83`). The
   partition margin is computed here, from the same `candidates` list.

**What the existing runner-up is and is not.** `RunnerUp` retains the
partition and its **raw clamped φ**, selected as the lowest-φ candidate
*strictly greater* than the MIP's φ (peers equal within `utils.eq` are
excluded). Two consequences: (a) it is keyed on raw φ while MIP selection is
keyed on normalized φ, so `runner_up.phi − phi` is **not** the selection
margin (verified empirically on `grid3_system`: the two lowest normalized-φ
candidates tie at 0.012333 — true margin ≈ 0 — while the by-raw-φ runner-up
gap is ≈ 0.0143); and (b) excluding eq-peers means a gap derived from
`runner_up` can never be near zero, so it cannot express "effectively tied".
Therefore the margin is a **new quantity** — the minimum normalized-φ gap
over *all* other candidates, which is exactly 0 when a peer ties — and
`RunnerUp` / `runner_up_from_candidates` are left untouched (both the IIT 4.0
and IIT 3.0 call sites keep their current behavior and `explain()` findings).

**Margin definition (uniform for both sites).** For a selection with winner
value `v*` and other candidates' values `V`:
`margin = max(0.0, min/max-gap to the best value in V)`, computed with plain
`float` subtraction; `None` when `V` is empty (no competitor — trivially
unique selection, not a tie). Exact ties give margin 0; at-precision near-ties
give a tiny positive margin. The tie-quantization flag is then
`utils.eq(margin, 0.0)` — the same precision discipline the ties machinery
uses (`utils.eq` at `config.numerics.precision`; there is no `utils.is_zero`
in this tree — use `eq(x, 0.0)`). Never compare with raw `==`.

**2026 cap ordering.** Under `IIT_4_0_2026`, `_apply_ii_cap` mutates the
selected MIP's `phi`/`normalized_phi` **after** selection (`sia()` line ~909;
the MIP is selected on uncapped values exactly as in 2023 — see the in-code
comments at `evaluate_partition` and `_apply_ii_cap`). The partition margin
must therefore be **stored at the selection site** (pre-cap), not derived
lazily from post-cap fields. Both 2023 and 2026 share the two selection
sites, so one implementation covers both.

**Short-circuit caveat.** `map_reduce` in `_find_mip_for_fixed_state` uses
`shortcircuit_func=utils.is_falsy`: if any partition yields φ = 0 the sweep
stops early and `candidates` is truncated. In that case the MIP's φ is 0 and
the margin is computed over the evaluated subset — same semantics the
existing `runner_up` already has. Margins are exact whenever the reported
φ_s > 0 (no falsy candidate exists, so the sweep was exhaustive). Document
this in the field's docstring; test with positive-φ fixtures.

**Where the state spec flows.** The winning `StateSpecification` objects are
threaded by reference through `evaluate_partition` →
`queries.evaluate_partition` (passed as `state=`, never reconstructed) and
back-propagated by `resolve_system_state`; the tied-state cascade uses
`dataclasses.replace(spec, _ties=())`, which preserves all other fields. The
only places a `StateSpecification` is **constructed** are
`repertoire_algebra.intrinsic_information` (the selection site),
`serialize/convert.py` (decoder), and `pyphi/relabel.py` — so those three
plus the dataclass itself are the complete set of touch points for the new
fields.

**Serialization.** `pyphi/serialize/schema.py` holds one frozen
`msgspec.Struct` per type; `convert.py` registers encoder/decoder pairs.
msgspec decodes missing fields to their defaults, so appending
defaulted fields **after the existing defaulted fields** (`tie_peers` is last
in both `StateSpecificationSchema` and `IIT4SIASchema`) makes old serialized
results load with margins `None`. `sia.runner_up` is not serialized today;
the margins (scalars) are, so they survive round-trips without serializing
runner-up objects.

**Display.** The IIT 4.0 SIA card has an exact-render ASCII golden
(`test/display/test_display.py::_GOLDEN_IIT4_SIA`) locked at the default
verbosity (`HIGH` = 2). Margin rows are gated at `FULL` (3, "the card plus
all mathematical content" — `pyphi/display/mixin.py`), so the locked default
card is untouched and no golden changes.

**Empirical fixture facts** (measured in this tree, default config,
IIT_4_0_2023):

| fixture | φ_s | state ties | lowest normalized-φ candidates | margin behavior |
|---|---|---|---|---|
| `examples.basic_system()` | 0.415037 | none | 0.207519, 0.283007, … | partition margin ≈ 0.075488; untied |
| `examples.grid3_system()` | 0.024666 | none | 0.012333, 0.012333, … | partition margin ≈ 0 → flag fires |
| `examples.xor_system()` | 1.5 | cause spec has 2 tied states | 0.25 ×4 | state margin 0 **and** partition margin 0 → flag fires |

There are 22 partitions for each of these 3-node systems under the default
`DIRECTED_SET_PARTITION` scheme; a brute-force sweep is ~0.5 s per fixture.

**Brute-force recipes used by the tests.** Partitions: enumerate
`pyphi.partition.system_partitions(...)` and evaluate each with
`pyphi.formalism.iit4.evaluate_partition(partition, system,
sia.system_state, system_measure=resolve_system_measure(
config.formalism.iit.system_phi_measure))`; sort the clamped
`normalized_phi` values; margin = second − first. States: call
`repertoire_algebra.intrinsic_information(system, direction,
mechanism=system.node_indices, purview=system.node_indices,
specification_measure=resolve_mechanism_measure(
config.formalism.iit.specification_measure), states=[s])` once per state
`s` — each call returns that state's ii independently; sort descending;
margin = first − second.

---

### Task 1: Retain the second-best state at the state-selection site

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (`intrinsic_information`, lines ~599-615)
- Modify: `pyphi/models/state_specification.py` (`StateSpecification`, lines ~77-130)
- Test: `test/formalism/test_selection_margins.py` (create)
- Create: `changelog.d/selection-margins.feature.md`

**Interfaces:**
- Produces:
  - `StateSpecification.runner_up_state: tuple[int, ...] | None = None` and
    `StateSpecification.runner_up_intrinsic_information: PyPhiFloat | None = None`
    (new fields, defaults `None`; excluded from `__eq__`/`__hash__`, like `_ties`).
  - `StateSpecification.state_margin -> PyPhiFloat | None` property:
    `max(0.0, ii − runner_up_ii)`; `None` when there was no competing state.
- Consumed by Tasks 2-5.

- [ ] **Step 1: Write the failing tests**

Create `test/formalism/test_selection_margins.py`:

```python
"""Tests for selection-margin reporting on the IIT 4.0 SIA."""

import pytest

import pyphi
from pyphi import examples
from pyphi import utils
from pyphi.conf import config
from pyphi.core import repertoire_algebra as ra
from pyphi.direction import Direction
from pyphi.measures.distribution import resolve_mechanism_measure


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


@pytest.fixture(scope="module")
def basic_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def xor_sia():
    with pyphi.config.override(progress_bars=False):
        return examples.xor_system().sia()


def _per_state_ii(system, direction):
    """Brute force: intrinsic information of every candidate system state."""
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    from pyphi.utils import all_states

    sizes = tuple(alphabet[i] for i in system.node_indices)
    return {
        state: float(
            ra.intrinsic_information(
                system,
                direction,
                mechanism=system.node_indices,
                purview=system.node_indices,
                specification_measure=measure,
                states=[state],
            ).intrinsic_information
        )
        for state in all_states(sizes)
    }


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_state_runner_up_matches_brute_force(basic_sia, direction):
    system = examples.basic_system()
    values = _per_state_ii(system, direction)
    ranked = sorted(values.values(), reverse=True)
    spec = basic_sia.system_state[direction]
    assert float(spec.intrinsic_information) == pytest.approx(ranked[0])
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(ranked[1])
    assert float(spec.state_margin) == pytest.approx(ranked[0] - ranked[1])
    assert spec.runner_up_state in values
    assert values[spec.runner_up_state] == pytest.approx(ranked[1])


def test_state_margin_zero_for_exactly_tied_states(xor_sia):
    # xor at (0, 0, 0): the specified cause state ties exactly (2 tied specs)
    spec = xor_sia.system_state.cause
    assert len(spec.ties) > 1
    assert float(spec.state_margin) == pytest.approx(0.0)
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(
        float(spec.intrinsic_information)
    )


def test_tie_members_share_runner_up_fields(xor_sia):
    specs = xor_sia.system_state.cause.ties
    values = {float(s.runner_up_intrinsic_information) for s in specs}
    assert len(values) == 1


def test_state_margin_none_when_no_competitor():
    system = examples.basic_system()
    measure = resolve_mechanism_measure(config.formalism.iit.specification_measure)
    spec = ra.intrinsic_information(
        system,
        Direction.CAUSE,
        mechanism=system.node_indices,
        purview=system.node_indices,
        specification_measure=measure,
        states=[system.proper_state],
    )
    assert spec.runner_up_intrinsic_information is None
    assert spec.runner_up_state is None
    assert spec.state_margin is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/formalism/test_selection_margins.py -v`
Expected: FAIL with `AttributeError: 'StateSpecification' object has no
attribute 'runner_up_intrinsic_information'`

- [ ] **Step 3: Add the fields and property to `StateSpecification`**

In `pyphi/models/state_specification.py`, add after the `_ties` field:

```python
    runner_up_state: tuple[int, ...] | None = None
    runner_up_intrinsic_information: PyPhiFloat | DistanceResult | None = None
```

Extend `__post_init__` to wrap the runner-up value like
`intrinsic_information`:

```python
        if self.runner_up_intrinsic_information is not None and not isinstance(
            self.runner_up_intrinsic_information, DistanceResult
        ):
            self.runner_up_intrinsic_information = PyPhiFloat(
                self.runner_up_intrinsic_information
            )
```

Add the property (after `ties`):

```python
    @property
    def state_margin(self) -> PyPhiFloat | None:
        """The intrinsic-information gap between this specified state and the
        best competing state over the same purview.

        Zero when another state ties exactly; ``None`` when there was no
        competing state. A margin within ``config.numerics.precision`` of
        zero means the state selection is effectively tied.
        """
        if self.runner_up_intrinsic_information is None:
            return None
        return PyPhiFloat(
            max(
                0.0,
                float(self.intrinsic_information)
                - float(self.runner_up_intrinsic_information),
            )
        )
```

Do **not** add the new fields to `__eq__` or `__hash__` (they are selection
metadata, treated like `_ties`).

- [ ] **Step 4: Retain the runner-up at the selection site**

In `pyphi/core/repertoire_algebra.py`, in `intrinsic_information`, after
`max_information = max(state_to_information.values())` (line ~600), compute
the best competitor (duplicates included, so an exact tie yields the max
itself and a zero margin):

```python
    ranked = sorted(state_to_information.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1:
        runner_up_state, runner_up_information = ranked[1]
    else:
        runner_up_state = runner_up_information = None
```

and pass both to every tie member's constructor:

```python
        StateSpecification(
            ...,
            runner_up_state=runner_up_state,
            runner_up_intrinsic_information=runner_up_information,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_selection_margins.py -v`
Expected: all PASS. Also run the neighboring suites that exercise this
kernel: `uv run pytest test/formalism/test_big_phi.py test/test_resolve_ties.py test/core -q`
Expected: no regressions (the retention adds fields; no computed value
changes).

- [ ] **Step 6: Changelog fragment and commit**

```bash
cat > changelog.d/selection-margins.feature.md <<'EOF'
Added selection-margin reporting to the IIT 4.0 SIA: `partition_margin`
(normalized-φ gap between the MIP and the best competing system partition),
per-direction specified-state margins
(`StateSpecification.state_margin`, from the retained second-best state's
intrinsic information), and `SystemIrreducibilityAnalysis.effectively_tied`
(whether any selection margin is within `config.numerics.precision` of
zero). Margins are computed from values the SIA search already produces,
surface in `explain()` findings, the `FULL`-verbosity card, and
`to_pandas()`, and round-trip through serialization (older serialized
results load with margins absent).
EOF
git add pyphi/core/repertoire_algebra.py pyphi/models/state_specification.py test/formalism/test_selection_margins.py changelog.d/selection-margins.feature.md
git commit -m "Retain the second-best specified state at ii selection"
```

---

### Task 2: Partition margin and SIA-level margin surface

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py`
  (`SystemIrreducibilityAnalysis` dataclass ~line 152;
  `_find_mip_for_fixed_state` ~line 1012)
- Test: `test/formalism/test_selection_margins.py` (extend)

**Interfaces:**
- Consumes: `StateSpecification.state_margin` (Task 1).
- Produces:
  - `SystemIrreducibilityAnalysis.partition_margin: PyPhiFloat | None = None`
    — new dataclass field (default `None`; excluded from `__eq__`/`__hash__`,
    like `runner_up` and `_ties`). Set at MIP selection, pre-2026-cap.
  - `SystemIrreducibilityAnalysis.state_margins -> dict[Direction, PyPhiFloat | None]`.
  - `SystemIrreducibilityAnalysis.effectively_tied -> bool`.
  - `runner_up` / `RunnerUp` semantics unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `test/formalism/test_selection_margins.py`:

```python
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.formalism.iit4 import evaluate_partition
from pyphi.measures.distribution import resolve_system_measure
from pyphi.partition import system_partitions


def _brute_force_partition_values(system, system_state):
    measure = resolve_system_measure(config.formalism.iit.system_phi_measure)
    partitions = system_partitions(
        system.node_indices,
        node_labels=system.node_labels,
        partition_scheme=config.formalism.iit.system_partition_scheme,
    )
    return sorted(
        float(
            evaluate_partition(
                partition, system, system_state, system_measure=measure
            ).normalized_phi
        )
        for partition in partitions
    )


def test_partition_margin_matches_brute_force(basic_sia):
    values = _brute_force_partition_values(
        examples.basic_system(), basic_sia.system_state
    )
    assert float(basic_sia.normalized_phi) == pytest.approx(values[0])
    assert float(basic_sia.partition_margin) == pytest.approx(values[1] - values[0])


def test_partition_margin_zero_for_symmetric_substrate():
    # grid3's two best partitions are symmetry-related and tie in
    # normalized phi, so the partition selection is effectively tied.
    sia = examples.grid3_system().sia()
    assert float(sia.partition_margin) == pytest.approx(0.0)
    assert sia.effectively_tied


def test_effectively_tied_fires_on_state_tie(xor_sia):
    assert utils.eq(float(xor_sia.state_margins[Direction.CAUSE]), 0.0)
    assert xor_sia.effectively_tied


def test_untied_system_is_not_flagged(basic_sia):
    assert basic_sia.partition_margin is not None
    assert not utils.eq(float(basic_sia.partition_margin), 0.0)
    assert all(
        margin is None or not utils.eq(float(margin), 0.0)
        for margin in basic_sia.state_margins.values()
    )
    assert not basic_sia.effectively_tied


def test_state_margins_read_through_system_state(basic_sia):
    for direction in Direction.both():
        expected = basic_sia.system_state[direction].state_margin
        assert float(basic_sia.state_margins[direction]) == pytest.approx(
            float(expected)
        )


def test_null_sia_has_no_margins():
    sia = NullSystemIrreducibilityAnalysis()
    assert sia.partition_margin is None
    assert sia.state_margins == {
        Direction.CAUSE: None,
        Direction.EFFECT: None,
    }
    assert not sia.effectively_tied


def test_2026_cap_does_not_change_margins():
    # The 2026 formalism selects the MIP exactly as 2023 does and applies
    # the ii(s) cap afterwards, so both selection margins are identical.
    with pyphi.config.override(version="IIT_4_0_2026"):
        sia_2026 = examples.basic_system().sia()
    with pyphi.config.override(version="IIT_4_0_2023"):
        sia_2023 = examples.basic_system().sia()
    assert float(sia_2026.partition_margin) == pytest.approx(
        float(sia_2023.partition_margin)
    )
    for direction in Direction.both():
        assert float(sia_2026.state_margins[direction]) == pytest.approx(
            float(sia_2023.state_margins[direction])
        )


def test_runner_up_surface_unchanged(basic_sia):
    # The existing runner-up record keeps its raw-phi semantics.
    assert basic_sia.runner_up is not None
    assert float(basic_sia.runner_up.phi) > float(basic_sia.phi)
```

Note: check the exact config-override spelling for the formalism version
(`config.formalism.iit.version`; top-level `pyphi.config.override(version=...)`
routes to the right layer — confirm against an existing test that switches
version, e.g. in `test/formalism/`, and copy its form). Note also that the
disk result cache keys on config, so the two `sia()` calls do not collide.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/formalism/test_selection_margins.py -v -k "partition_margin or effectively_tied or state_margins or null_sia or 2026 or runner_up_surface"`
Expected: FAIL with `AttributeError: ... no attribute 'partition_margin'`

- [ ] **Step 3: Add the field and properties to the SIA**

In `pyphi/formalism/iit4/__init__.py`, add to the
`SystemIrreducibilityAnalysis` dataclass, directly after the `runner_up`
field:

```python
    partition_margin: PyPhiFloat | None = None
```

Add properties after `set_ties`:

```python
    @property
    def state_margins(self) -> dict[Direction, PyPhiFloat | None]:
        """Per-direction intrinsic-information gap between the specified
        system state and the best competing state
        (:attr:`StateSpecification.state_margin`)."""
        margins: dict[Direction, PyPhiFloat | None] = {}
        for direction in Direction.both():
            spec = (
                self.system_state[direction]
                if self.system_state is not None
                else None
            )
            margins[direction] = spec.state_margin if spec is not None else None
        return margins

    @property
    def effectively_tied(self) -> bool:
        """Whether any selection margin is within
        ``config.numerics.precision`` of zero — i.e. the partition or
        specified-state selection is effectively tied at the configured
        precision."""
        margins = [self.partition_margin, *self.state_margins.values()]
        return any(
            margin is not None and utils.eq(float(margin), 0.0)
            for margin in margins
        )
```

Extend the `partition_margin` field's docstring coverage in the class
docstring: one sentence defining it as the gap in (clamped) normalized φ
between the MIP and the best competing partition, computed at selection
(before the 2026 ii(s) cap), zero when a competitor ties, ``None`` when
there was no competitor, and exact whenever φ_s > 0 (a reducibility
short-circuit stops the partition sweep early, in which case the margin is
over the evaluated subset).

Do **not** add `partition_margin` to `__eq__` or `__hash__`.

- [ ] **Step 4: Set the margin at the selection site**

In `_find_mip_for_fixed_state`, after the
`mip_sia.runner_up = runner_up_from_candidates(...)` line:

```python
    others = [candidate for candidate in candidates if candidate is not mip_sia]
    if others:
        gap = min(float(c.normalized_phi) for c in others) - float(
            mip_sia.normalized_phi
        )
        mip_sia.partition_margin = PyPhiFloat(max(0.0, gap))
```

(The winner minimizes the selection key, so the gap is non-negative up to
tie-resolution ordering; the `max(0.0, ...)` clamps the near-tie case where
a float-smaller candidate lost on a later key component.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_selection_margins.py -v`
Expected: all PASS.

Then run the SIA-adjacent suites:
`uv run pytest test/formalism -q` and
`uv run pytest test/integration/test_golden_regression.py -q`
Expected: no regressions — margins are additive; no computed value changes,
so every existing golden value is byte-identical.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/formalism/test_selection_margins.py
git commit -m "Report the partition selection margin on the IIT 4.0 SIA"
```

---

### Task 3: `explain()` findings, card rows, and `to_pandas`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py`
  (`_findings` ~line 392, `_describe` ~line 324, `_pandas_record` ~line 313)
- Test: `test/formalism/test_selection_margins.py` (extend)
- Test: `test/display/test_display.py` (extend)

**Interfaces:**
- Consumes: `partition_margin`, `state_margins`, `effectively_tied` (Task 2).
- Produces:
  - New `Finding` kinds: `"partition_margin"`, `"state_margin"` (one per
    direction, toned), `"effectively_tied"`.
  - Card rows at `FULL` verbosity: "Selection margin" in the MIP section,
    "State margin" in the Cause/Effect sections, "Effectively tied" in the
    MIP section. The locked default-verbosity card is unchanged.
  - `_pandas_record` keys: `partition_margin`, `cause_state_margin`,
    `effect_state_margin`, `effectively_tied`.

- [ ] **Step 1: Write the failing tests**

Append to `test/formalism/test_selection_margins.py`:

```python
def _findings_by_kind(explanation):
    by_kind: dict[str, list] = {}
    for finding in explanation.findings:
        by_kind.setdefault(finding.kind, []).append(finding)
    return by_kind


def test_explain_reports_margins(basic_sia):
    by_kind = _findings_by_kind(basic_sia.explain())
    assert float(by_kind["partition_margin"][0].value) == pytest.approx(
        float(basic_sia.partition_margin)
    )
    state_margins = by_kind["state_margin"]
    assert len(state_margins) == 2
    assert {f.tone for f in state_margins} == {"cause", "effect"}
    assert by_kind["effectively_tied"][0].value is False


def test_explain_flags_effective_tie(xor_sia):
    by_kind = _findings_by_kind(xor_sia.explain())
    assert by_kind["effectively_tied"][0].value is True


def test_null_sia_explain_has_no_margin_findings():
    by_kind = _findings_by_kind(NullSystemIrreducibilityAnalysis().explain())
    assert "partition_margin" not in by_kind
    assert "state_margin" not in by_kind
    assert "effectively_tied" not in by_kind


def test_to_pandas_includes_margins(basic_sia):
    record = basic_sia.to_pandas()
    assert float(record["partition_margin"]) == pytest.approx(
        float(basic_sia.partition_margin)
    )
    assert float(record["cause_state_margin"]) == pytest.approx(
        float(basic_sia.state_margins[Direction.CAUSE])
    )
    assert float(record["effect_state_margin"]) == pytest.approx(
        float(basic_sia.state_margins[Direction.EFFECT])
    )
    assert bool(record["effectively_tied"]) is False
```

Append to `test/display/test_display.py` (near the existing
`test_iit4_sia_ascii_golden`):

```python
def test_iit4_sia_full_verbosity_shows_margins():
    """Margin rows appear at FULL verbosity; the default card is unchanged
    (covered by test_iit4_sia_ascii_golden)."""
    pyphi.config.progress_bars = False
    sia = pyphi.examples.basic_system().sia()
    with pyphi.config.override(repr_verbosity=3):
        out = repr(sia)
    assert "Selection margin" in out
    assert "State margin" in out
    assert "Effectively tied" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/formalism/test_selection_margins.py -k "explain or pandas" test/display/test_display.py -k margins -v`
Expected: FAIL (`KeyError: 'partition_margin'`, missing card rows).

- [ ] **Step 3: Implement**

In `_findings` (after the existing `runner_up`/`gap` findings, before the
binding-direction finding):

```python
        if self.partition_margin is not None:
            findings.append(
                Finding(
                    kind="partition_margin",
                    label="MIP selection margin (normalized φ)",
                    value=self.partition_margin,
                )
            )
        state_margins = self.state_margins
        for direction in Direction.both():
            margin = state_margins[direction]
            if margin is not None:
                tone = "cause" if direction == Direction.CAUSE else "effect"
                findings.append(
                    Finding(
                        kind="state_margin",
                        label=f"Specified-{tone}-state margin (ii)",
                        value=margin,
                        tone=tone,
                    )
                )
        if self.partition_margin is not None or any(
            margin is not None for margin in state_margins.values()
        ):
            findings.append(
                Finding(
                    kind="effectively_tied",
                    label="Selection effectively tied",
                    value=self.effectively_tied,
                )
            )
```

In `_describe`, gate the new rows on `verbosity >= FULL` (import `FULL` from
`pyphi.display` — check it is exported; if not, import from
`pyphi.display.mixin`):

- Cause section: after the "Intrinsic differentiation" row, add
  `Row("State margin", state.cause.state_margin)` when
  `verbosity >= FULL and state.cause.state_margin is not None`.
- Effect section: same for `state.effect`.
- MIP section (`mip_rows`): after the "Tied MIPs" row, when
  `verbosity >= FULL`, append `Row("Selection margin", self.partition_margin)`
  (when not `None`) and `Row("Effectively tied", self.effectively_tied)`
  (when any margin is not `None`).

In `_pandas_record`, add:

```python
            "partition_margin": (
                float(self.partition_margin)
                if self.partition_margin is not None
                else None
            ),
            "cause_state_margin": _optional_float(
                self.state_margins[Direction.CAUSE]
            ),
            "effect_state_margin": _optional_float(
                self.state_margins[Direction.EFFECT]
            ),
            "effectively_tied": self.effectively_tied,
```

with a small module-level `_optional_float(x)` helper (`None` passes
through).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_selection_margins.py test/display -q`
Expected: all PASS, **including** `test_iit4_sia_ascii_golden` (the locked
default card must be byte-identical — if it changed, a row leaked below
`FULL`).

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/formalism/test_selection_margins.py test/display/test_display.py
git commit -m "Surface selection margins in explain, display, and to_pandas"
```

---

### Task 4: Serialization and relabel pass-through

**Files:**
- Modify: `pyphi/serialize/schema.py`
  (`StateSpecificationSchema` ~line 40, `IIT4SIASchema` ~line 242)
- Modify: `pyphi/serialize/convert.py`
  (`_encode_state_spec`/`_decode_state_spec` ~lines 79-111,
  `_encode_iit4_sia`/`_decode_iit4_sia` ~lines 526-583)
- Modify: `pyphi/relabel.py` (`relabel_state_specification` ~line 87)
- Test: `test/serialize/test_serialize_sia.py` (extend)
- Test: `test/test_relabel.py` (extend)

**Interfaces:**
- Consumes: the new fields from Tasks 1-2.
- Produces:
  - `StateSpecificationSchema.runner_up_state: tuple[int, ...] | None = None`
    and `.runner_up_intrinsic_information: PhiSchema | None = None`
    (appended after `tie_peers`).
  - `IIT4SIASchema.partition_margin: PhiSchema | None = None`
    (appended after `tie_peers`).
  - Old serialized results (fields absent) decode with margins `None`.

- [ ] **Step 1: Write the failing tests**

Append to `test/serialize/test_serialize_sia.py`:

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_margins_round_trip(fmt):
    import pyphi
    from pyphi import examples

    with pyphi.config.override(progress_bars=False):
        sia = examples.basic_system().sia()
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert float(restored.partition_margin) == pytest.approx(
        float(sia.partition_margin)
    )
    spec = restored.system_state.cause
    assert spec.runner_up_state == sia.system_state.cause.runner_up_state
    assert float(spec.runner_up_intrinsic_information) == pytest.approx(
        float(sia.system_state.cause.runner_up_intrinsic_information)
    )
    assert restored.effectively_tied == sia.effectively_tied


def test_iit4_sia_loads_without_margin_fields():
    """Serialized results produced before margins existed decode with the
    margin fields at their defaults."""
    import json

    import pyphi
    from pyphi import examples
    from pyphi import serialize

    with pyphi.config.override(progress_bars=False):
        sia = examples.basic_system().sia()
    data = json.loads(serialize.dumps(sia, format="json"))

    def strip(obj):
        if isinstance(obj, dict):
            for key in (
                "partition_margin",
                "runner_up_state",
                "runner_up_intrinsic_information",
            ):
                obj.pop(key, None)
            for value in obj.values():
                strip(value)
        elif isinstance(obj, list):
            for item in obj:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.partition_margin is None
    assert restored.system_state.cause.runner_up_intrinsic_information is None
    assert restored.system_state.cause.state_margin is None
    assert not restored.effectively_tied
    # All pre-existing fields are untouched by the additions.
    assert float(restored.phi) == pytest.approx(float(sia.phi))
    assert restored.partition == sia.partition
```

Append to `test/test_relabel.py` (match its existing fixture/import style):

```python
def test_relabel_preserves_selection_margins(grid3_ces):
    mapping = {old: PERM.index(old) for old in range(3)}
    relabeled = grid3_ces.relabel(mapping)
    original = grid3_ces.sia
    assert (relabeled.sia.partition_margin is None) == (
        original.partition_margin is None
    )
    if original.partition_margin is not None:
        assert float(relabeled.sia.partition_margin) == pytest.approx(
            float(original.partition_margin)
        )
    for direction, margin in original.state_margins.items():
        relabeled_margin = relabeled.sia.state_margins[direction]
        if margin is None:
            assert relabeled_margin is None
        else:
            assert float(relabeled_margin) == pytest.approx(float(margin))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_sia.py -k margin test/test_relabel.py -k margin -v`
Expected: round-trip FAILs (restored margins are `None` — encoder drops
them); relabel FAILs (relabeled spec loses the runner-up fields).

- [ ] **Step 3: Extend the schemas**

In `pyphi/serialize/schema.py` — msgspec requires defaulted fields to trail,
so append **after** `tie_peers` in each Struct:

`StateSpecificationSchema`:

```python
    runner_up_state: tuple[int, ...] | None = None
    runner_up_intrinsic_information: PhiSchema | None = None
```

`IIT4SIASchema`:

```python
    partition_margin: PhiSchema | None = None
```

- [ ] **Step 4: Extend the converters**

In `pyphi/serialize/convert.py`:

- `_encode_state_spec`: add
  `runner_up_state=_opt_tuple(spec.runner_up_state)` and
  `runner_up_intrinsic_information=_enc_optional(spec.runner_up_intrinsic_information)`
  (check `_opt_tuple`/`_enc_optional` helper names against the file; they are
  used by the SIA encoder).
- `_decode_state_spec`: pass
  `runner_up_state=_opt_tuple(struct.runner_up_state)` and
  `runner_up_intrinsic_information=_dec_optional(struct.runner_up_intrinsic_information)`
  to the `StateSpecification` constructor.
- `_encode_iit4_sia`: add
  `partition_margin=_enc_optional(sia.partition_margin)`.
- `_decode_iit4_sia`: add
  `"partition_margin": _dec_optional(struct.partition_margin)` to `kwargs`.

- [ ] **Step 5: Extend relabel**

In `pyphi/relabel.py`, `relabel_state_specification` rebuilds the spec
field-by-field; `runner_up_state` is position-aligned to the purview, so it
reorders with the same `order` the state uses:

```python
        runner_up_state=_reorder(spec.runner_up_state, order),
        runner_up_intrinsic_information=spec.runner_up_intrinsic_information,
```

`relabel_sia` uses `dataclasses.replace`, which carries `partition_margin`
automatically — no change needed there (verify with the test).

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest test/serialize test/test_relabel.py -q`
Expected: all PASS, including every pre-existing serialization test.

- [ ] **Step 7: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py pyphi/relabel.py test/serialize/test_serialize_sia.py test/test_relabel.py
git commit -m "Serialize and relabel selection margins"
```

---

### Task 5: Acceptance test, roadmap close-out, full verification

**Files:**
- Test: `test/formalism/test_selection_margins.py` (extend)
- Modify: `ROADMAP.md` (Status Dashboard row "Selection-margin reporting",
  line ~60)

**Interfaces:** none new.

- [ ] **Step 1: Add the landscapes-spec demonstration case**

The IIT 4.0 (2023) Fig. 1A substrate is not in `pyphi.examples`; build it
from its published weights via the substrate generator (the same
construction the landscapes exploration used). Its φ_s ≈ 0.134 value sits
near a specified-state switch, so its state margins are the demonstration
target. No reference margin value was recorded (the exploration measured the
switch distance in *weight* space, 0.005 in the A→B weight, not the ii gap),
so the assertion is brute-force consistency plus the flag's behavior, with
the exact margin values captured by the brute force itself.

Append to `test/formalism/test_selection_margins.py`:

```python
def test_fig1a_2023_state_margins_match_brute_force():
    """IIT 4.0 (2023) Fig. 1A: the substrate near a specified-state switch
    reports finite, brute-force-consistent state margins."""
    import numpy as np

    from pyphi.substrate_generator import build_substrate, ising

    weights = np.array(
        [
            [-0.2, 0.7, 0.2],
            [0.7, -0.2, 0.0],
            [0.0, -0.8, 0.2],
        ]
    )
    substrate = build_substrate(
        [ising.probability] * 3, weights, temperature=0.25
    )
    sia = pyphi.analyze(substrate, (1, 0, 0), compute="sia")
    assert float(sia.phi) > 0

    system = pyphi.System(substrate, state=(1, 0, 0))
    for direction in Direction.both():
        values = sorted(_per_state_ii(system, direction).values(), reverse=True)
        margin = sia.state_margins[direction]
        assert margin is not None
        assert float(margin) == pytest.approx(values[0] - values[1])
    # Its selections are near a boundary but not tied at the published point.
    assert not sia.effectively_tied
```

Check the `System` construction and `pyphi.analyze` signatures against
`test/test_analyze.py` and adjust (the SIA and the brute force must be
computed for the same system/state). Run it once in isolation and record the
observed margin values in the test as a comment for future readers.

Run: `uv run pytest test/formalism/test_selection_margins.py -v`
Expected: all PASS (this test is ~1-2 s: n = 3, 22 partitions, 8 states × 2
directions).

- [ ] **Step 2: Update the ROADMAP dashboard row**

In `ROADMAP.md`, change the "Selection-margin reporting" row (line ~60) from
`⬜ open` to `✅ landed`, following the format of neighboring landed rows,
e.g.:

```markdown
| Selection-margin reporting | ✅ landed | 7 | IIT 4.0 SIA reports `partition_margin` (normalized-φ gap to the best competing system partition), per-direction specified-state margins (second-best ii retained on `StateSpecification`), and `effectively_tied` (any margin within `precision` of zero) — surfaced via `explain()`, the FULL-verbosity card, `to_pandas`, and serialization. IIT 3.0/AC deferred (different selection structure). Concrete slice of N23. Spec: `2026-07-07-substrate-parameter-landscapes.md` §7 |
```

(The landscapes spec itself is an untracked file in the main working tree
and will not exist in the worktree; do not attempt to edit or commit it.)

- [ ] **Step 3: Full test suite (no path argument — includes the doctest sweep)**

Run: `uv run pytest -x -q`
Expected: all pass. This is the complete verification recipe per project
convention; bare-path invocations skip doctests. Pay particular attention to:

- `test/integration/test_golden_regression.py` — every existing golden value
  byte-identical (margins are additive metadata; no computed value changes).
- `test/display/test_display.py::test_iit4_sia_ascii_golden` — the locked
  default-verbosity card unchanged.
- `test/serialize` — full round-trip suite.

If any doctest or unrelated-looking test fails, diagnose before touching
anything — other sessions may have concurrent working-tree changes; only
fix failures traceable to this plan's commits.

- [ ] **Step 4: Pre-commit hooks over the changed files**

Run: `uv run pre-commit run --files $(git diff --name-only $(git merge-base HEAD <base-branch>) | tr '\n' ' ')` — substitute the branch this worktree was created from.
Expected: all hooks pass (ruff, pyright, file checks). Fix any findings and
amend or follow-up commit as appropriate.

- [ ] **Step 5: Commit the close-out**

```bash
git add ROADMAP.md test/formalism/test_selection_margins.py
git commit -m "Add Fig 1A margin acceptance test; mark selection margins landed"
```

---

## Self-review notes

- **Spec coverage:** partition margin (Task 2), per-direction state margin
  (Task 1), tie-quantization flag (Task 2), `explain()` findings + card +
  `to_pandas` (Task 3), serialization backward-compat (Task 4), brute-force
  validation on three fixtures including a deliberately-tied one
  (Tasks 1-2: `xor_system` state tie, `grid3_system` partition tie,
  `basic_system` untied — all verified empirically before planning), the
  landscapes demonstration case (Task 5).
- **No recompute:** the state side retains two scalars from a dict the
  kernel already builds; the partition side takes one `min()` over the
  candidate list already materialized for MIP selection. Neither touches a
  computation path.
- **Why the margin is not `runner_up.phi − phi`:** the existing runner-up is
  keyed on raw clamped φ and excludes precision-equal peers; MIP selection is
  keyed on clamped normalized φ. On `grid3_system` these disagree (true
  normalized margin ≈ 0; raw-φ runner-up gap ≈ 0.0143). The margin is
  computed as the minimum normalized-φ gap over *all* other candidates, and
  `RunnerUp` is left untouched so the existing `explain()` surface and the
  IIT 3.0 call site are unaffected.
- **2026 cap:** `_apply_ii_cap` mutates the winner's φ after selection, so
  the margin is stored at the selection site (pre-cap); Task 2 has an
  explicit 2023-vs-2026 equality test.
- **Precision discipline:** margins are raw float gaps;
  `effectively_tied` uses `utils.eq(margin, 0.0)` (there is no
  `utils.is_zero` in this tree); no raw `==` anywhere.
- **Equality/hash:** new fields are excluded from `__eq__`/`__hash__` on
  both `StateSpecification` and the SIA (mirroring `_ties`/`runner_up`), so
  results loaded from pre-margin serialized files still compare equal to
  themselves and hashing is unchanged.
- **Known caveats, documented not fixed:** margins over a short-circuited
  (φ_s = 0) partition sweep cover only the evaluated subset (same as the
  existing runner-up); disk-cached SIA results computed before this change
  return without margins until recomputed.
- **Out of scope:** IIT 3.0 and AC margins (different selection structure,
  deferred); `diff()`/B15 and provenance/N8 surfaces unchanged; no changes
  to `resolve_ties` or tie-set semantics.
