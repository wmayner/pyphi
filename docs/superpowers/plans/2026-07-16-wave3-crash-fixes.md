# Wave 3 Crash Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every Wave 3 finding from the whole-library review: crashes reached through documented, intended API usage (single-direction SIA, IIT 3.0 analysis drivers, k-ary/empty-CES/IIT 3.0 visualization and MCP paths, `settle()` off-by-one).

**Architecture:** Nine independent guard/generalization fixes across four clusters, each landed TDD (failing crash-repro test first). No new modules; every change is a targeted edit to an existing function. Spec: `docs/superpowers/specs/2026-07-16-wave3-crash-fixes-design.md`.

**Tech Stack:** Python 3.13, pytest, numpy, matplotlib (visualize extra), FastMCP server module.

## Global Constraints

- Work in the worktree `/Users/will/projects/pyphi/.claude/worktrees/wave3-crash-fixes` (branch `fix/wave3-documented-usage-crashes`); run every command from that directory. The venv is already set up with `.[visualize,caching,emd,xarray]` and `pot`.
- Always `uv run pytest ...`; never pipe test output through `tail`/`head` when the result matters — for the final full run, redirect to a file and read the summary line.
- Commit messages end with the two trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe`
- Never `git commit --no-verify`. If a pre-commit formatter modifies files (status shows `MM`), re-stage and re-commit.
- Test imports from conftest are absolute (`from test.conftest import ...`), never relative (ruff TID252).
- Docstrings: NumPy style, final-state voice, no planning-artifact references anywhere in code, docstrings, or changelog fragments.
- Tests that compute φ under IIT 3.0 use `@skip_if_no_emd_backend` (from `test/conftest.py`) and pin formalism explicitly (the `formalism=` argument or `IIT_3_CONFIG`).

---

### Task 1: Single-direction SIA shortcircuit guard

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py:802-816` (`_has_no_cause_or_effect`)
- Test: `test/formalism/test_iit4_sia_components.py`

**Interfaces:**
- Consumes: `FORMALISM_REGISTRY["IIT_4_0_2026"].evaluate_system(system, directions=[...])`; `config.override(shortcircuit_sia=True)`.
- Produces: no API change — `_has_no_cause_or_effect(system_state)` now tolerates `None` direction specs.

- [ ] **Step 1: Write the failing test**

Append to `test/formalism/test_iit4_sia_components.py`:

```python
def test_sia_single_direction_with_shortcircuit():
    """A one-direction ``directions=`` restriction must not crash the
    shortcircuit path: a direction that was not requested cannot
    shortcircuit the analysis."""
    from pyphi import Direction
    from pyphi import examples
    from pyphi.formalism import FORMALISM_REGISTRY

    with config.override(shortcircuit_sia=True):
        system = examples.basic_system()
        formalism = FORMALISM_REGISTRY["IIT_4_0_2026"]
        for direction in Direction.both():
            sia = formalism.evaluate_system(system, directions=[direction])
            assert float(sia.phi) >= 0.0
        # Control: the explicit two-direction call matches the default.
        both = formalism.evaluate_system(
            system, directions=list(Direction.both())
        )
        assert float(both.phi) == pytest.approx(
            float(formalism.evaluate_system(system).phi)
        )
```

(`config` and `pytest` are already imported at the top of this file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/formalism/test_iit4_sia_components.py::test_sia_single_direction_with_shortcircuit -v`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'intrinsic_information'`

- [ ] **Step 3: Implement the guard**

In `pyphi/formalism/iit4/__init__.py`, `_has_no_cause_or_effect`, replace:

```python
        if not numerics.is_positive(system_state[direction].intrinsic_information):
            reasons.append(reason)
```

with:

```python
        spec = system_state[direction]
        if spec is not None and not numerics.is_positive(
            spec.intrinsic_information
        ):
            reasons.append(reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/formalism/test_iit4_sia_components.py::test_sia_single_direction_with_shortcircuit -v`
Expected: PASS

- [ ] **Step 5: Add the changelog fragment and commit**

```bash
cat > changelog.d/sia-single-direction.fix.md <<'EOF'
Fixed `sia(directions=[...])` with a single direction crashing with an `AttributeError` when `shortcircuit_sia` is enabled (the default): a direction that was not requested no longer participates in the shortcircuit check.
EOF
git add pyphi/formalism/iit4/__init__.py test/formalism/test_iit4_sia_components.py changelog.d/sia-single-direction.fix.md
git commit -m "Guard the SIA shortcircuit against unrequested directions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 2: Cross-formalism guards in `landscape._eval_point`

**Files:**
- Modify: `pyphi/landscape.py:133-152` (`_eval_point` row construction)
- Test: `test/test_landscape.py`

**Interfaces:**
- Consumes: the `getattr(..., default)` guard contract established by `pyphi/sweep.py` `_row_sia`; `_optional_float` (returns `math.nan` for `None`) and `_part_id` (handles any partition type, including IIT 3.0's `DirectedBipartition`), both already in `pyphi/landscape.py`.
- Produces: `_eval_point` rows valid for every formalism — `phi` and `partition` always populated; `signed_phi`, `normalized_phi`, `signed_normalized_phi`, `partition_margin`, `cause_state_margin`, `effect_state_margin` are `NaN` and `cause_state`, `effect_state`, `effectively_tied` are `None` where the SIA lacks them. `perturb` shares this code path.

- [ ] **Step 1: Write the failing test**

In `test/test_landscape.py`, add to the top-of-file imports:

```python
from test.conftest import skip_if_no_emd_backend
```

Append the test:

```python
@skip_if_no_emd_backend
def test_landscape_section_iit3_rows_carry_defaults():
    """A documented non-4.0 preset produces rows, not AttributeErrors.

    IIT 4.0-only columns carry NaN/None, the contract established by
    ``pyphi.sweep._row_sia``.
    """
    substrate = examples.basic_substrate()
    section = landscape_section(
        lambda theta: substrate, (1, 0, 0), [0.0], formalism="IIT_3_0"
    )
    row = section.df.iloc[0]
    assert math.isfinite(row["phi"])
    assert isinstance(row["partition"], str)
    for column in (
        "signed_phi",
        "normalized_phi",
        "signed_normalized_phi",
        "partition_margin",
        "cause_state_margin",
        "effect_state_margin",
    ):
        assert math.isnan(row[column])
    assert row["cause_state"] is None
    assert row["effect_state"] is None
    assert row["effectively_tied"] is None
```

(`math`, `examples`, and `landscape_section` are already imported at the top of this file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_landscape.py::test_landscape_section_iit3_rows_carry_defaults -v`
Expected: FAIL with `AttributeError: 'IIT3SystemIrreducibilityAnalysis' object has no attribute 'system_state'`

- [ ] **Step 3: Implement the guards**

In `pyphi/landscape.py`, `_eval_point`, replace everything from `system_state = sia.system_state` through the `row = {...}` literal with:

```python
    # Selection margins and state specifications exist only on IIT 4.0
    # SIAs; other formalisms' rows carry None/NaN in those columns.
    system_state = getattr(sia, "system_state", None)
    cause = system_state.cause if system_state is not None else None
    effect = system_state.effect if system_state is not None else None
    state_margins = getattr(sia, "state_margins", None)
    from pyphi.direction import Direction

    row = {
        "phi": float(sia.phi),
        "signed_phi": _optional_float(getattr(sia, "signed_phi", None)),
        "normalized_phi": _optional_float(getattr(sia, "normalized_phi", None)),
        "signed_normalized_phi": _optional_float(
            getattr(sia, "signed_normalized_phi", None)
        ),
        "partition": _part_id(sia.partition),
        "cause_state": None if cause is None else tuple(int(x) for x in cause.state),
        "effect_state": None if effect is None else tuple(int(x) for x in effect.state),
        "partition_margin": _optional_float(getattr(sia, "partition_margin", None)),
        "cause_state_margin": _optional_float(
            state_margins[Direction.CAUSE] if state_margins is not None else None
        ),
        "effect_state_margin": _optional_float(
            state_margins[Direction.EFFECT] if state_margins is not None else None
        ),
        "effectively_tied": getattr(sia, "effectively_tied", None),
    }
```

- [ ] **Step 4: Run test to verify it passes, plus the existing landscape suite**

Run: `uv run pytest test/test_landscape.py -v`
Expected: all PASS (the new test and every pre-existing IIT 4.0 test).

- [ ] **Step 5: Commit**

```bash
git add pyphi/landscape.py test/test_landscape.py
git commit -m "Carry None/NaN in IIT 4.0-only landscape columns for other formalisms

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 3: Cross-formalism guards in `optimize`

**Files:**
- Modify: `pyphi/optimize.py:131-136` (`_objective_value`), `pyphi/optimize.py:177-191` (`_eval_one` post-`try` reads)
- Test: `test/test_optimize.py`

**Interfaces:**
- Consumes: `_optional_float` (returns `math.nan` for `None`) and `_part_id`, both already in `pyphi/optimize.py`; `weight_axes`, `_eval_one`, `_objective_value`, `FIG1A_WEIGHTS`, `STATE` from the existing test module.
- Produces: `_eval_one` rows valid for every formalism (same column contract as Task 2); `_objective_value` raises `ValueError` naming the objective when the SIA lacks the requested attribute (a formalism/objective mismatch — e.g. the default `objective="signed_normalized_phi"` under `IIT_3_0`), instead of an `AttributeError` from inside a scipy run.

- [ ] **Step 1: Write the failing tests**

In `test/test_optimize.py`, add to the top-of-file imports:

```python
from test.conftest import skip_if_no_emd_backend
```

Append the tests:

```python
@skip_if_no_emd_backend
def test_eval_one_iit3_returns_finite_phi_objective():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    row = _eval_one(
        np.array([0.7]),
        builder=axis,
        state=STATE,
        subset=None,
        formalism="IIT_3_0",
        objective="phi",
    )
    assert row["reachable"] is True
    assert np.isfinite(row["objective"])
    assert isinstance(row["partition"], str)
    assert row["cause_state"] is None
    assert row["effect_state"] is None
    assert np.isnan(row["partition_margin"])
    assert np.isnan(row["cause_state_margin"])
    assert np.isnan(row["effect_state_margin"])


def test_objective_value_missing_attribute_raises_clear_error():
    class MinimalSIA:
        phi = 0.5

    assert _objective_value(MinimalSIA(), "phi") == 0.5
    with pytest.raises(ValueError, match="signed_normalized_phi"):
        _objective_value(MinimalSIA(), "signed_normalized_phi")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_optimize.py::test_eval_one_iit3_returns_finite_phi_objective test/test_optimize.py::test_objective_value_missing_attribute_raises_clear_error -v`
Expected: both FAIL — the first with `AttributeError: 'IIT3SystemIrreducibilityAnalysis' object has no attribute 'system_state'`, the second with `AttributeError` (not the expected `ValueError`).

- [ ] **Step 3: Implement**

In `pyphi/optimize.py`, replace `_objective_value` with:

```python
def _objective_value(sia: Any, objective: Any) -> float:
    """The natural (maximization-convention) objective scalar for one SIA."""
    if callable(objective):
        value: Any = objective(sia)
        return float(value)
    try:
        value = getattr(sia, objective)
    except AttributeError:
        raise ValueError(
            f"objective {objective!r} is not available on "
            f"{type(sia).__name__}; choose an objective the requested "
            f"formalism provides (e.g. objective='phi')"
        ) from None
    return _optional_float(value)
```

In `_eval_one`, replace everything from `system_state = sia.system_state` through the closing `}` of the returned dict with:

```python
    # Selection margins and state specifications exist only on IIT 4.0
    # SIAs; other formalisms' rows carry None/NaN in those columns.
    system_state = getattr(sia, "system_state", None)
    cause = system_state.cause if system_state is not None else None
    effect = system_state.effect if system_state is not None else None
    margins = getattr(sia, "state_margins", None)
    return {
        "objective": _objective_value(sia, objective),
        "reachable": True,
        "partition": _part_id(sia.partition),
        "cause_state": None if cause is None else tuple(int(x) for x in cause.state),
        "effect_state": None if effect is None else tuple(int(x) for x in effect.state),
        "partition_margin": _optional_float(getattr(sia, "partition_margin", None)),
        "cause_state_margin": _optional_float(
            margins[Direction.CAUSE] if margins is not None else None
        ),
        "effect_state_margin": _optional_float(
            margins[Direction.EFFECT] if margins is not None else None
        ),
        "_sia": sia,
    }
```

- [ ] **Step 4: Run the optimize suite**

Run: `uv run pytest test/test_optimize.py -v`
Expected: all PASS (including the pre-existing `test_objective_value_named_and_callable` and `test_optimize_rejects_unknown_objective_name`).

- [ ] **Step 5: Add the cluster changelog fragment and commit**

```bash
cat > changelog.d/analysis-drivers-cross-formalism.fix.md <<'EOF'
Fixed `landscape_section`, `perturb`, and `optimize` crashing with `AttributeError` under non-IIT-4.0 formalism presets (e.g. `formalism="IIT_3_0"`): IIT 4.0-only columns now carry `None`/`NaN`, matching the `sweep` contract, and requesting a named objective the formalism does not provide raises a clear `ValueError`.
EOF
git add pyphi/optimize.py test/test_optimize.py changelog.d/analysis-drivers-cross-formalism.fix.md
git commit -m "Support non-4.0 formalisms in optimize rows and objectives

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 4: k-ary node colors and TPM state labels in `visualize.connectivity`

**Files:**
- Modify: `pyphi/visualize/connectivity.py` (`NODE_COLORS` block, `_system_graph`, `plot_tpm`)
- Test: `test/visualize/test_visualize_matplotlib.py`

**Interfaces:**
- Consumes: `system.substrate.tpm.alphabet_sizes` (tuple of per-unit alphabet sizes, verified present on `FactoredTPM`); `all_states_str` (already imported in `connectivity.py`).
- Produces: `_node_color(in_system: bool, state: int, num_states: int) -> str | tuple[float, float, float]` (module-private); `plot_tpm(..., states=None)` where `states` is an optional sequence of state-label strings, applied to an axis when its length equals the axis length. Task 6 passes `states=` from the MCP server.

- [ ] **Step 1: Write the failing tests**

Append to `test/visualize/test_visualize_matplotlib.py`:

```python
def _k3_system():
    """A two-unit ternary system in state (2, 0) — both units in the system."""
    import pyphi

    rng = np.random.default_rng(2026)

    def marginal():
        m = rng.uniform(size=(3, 3, 3))
        return m / m.sum(axis=-1, keepdims=True)

    substrate = pyphi.Substrate(
        marginals=[marginal(), marginal()], state_space=(0, 1, 2)
    )
    return pyphi.System(substrate, state=(2, 0))


def test_system_graph_kary_colors():
    from matplotlib.colors import to_rgb

    from pyphi.visualize.connectivity import _system_graph

    g, colors = _system_graph(_k3_system())
    assert len(colors) == 2
    # In-system units interpolate lightblue -> darkblue by state intensity:
    # state 2 of 3 is the dark end, state 0 the light end.
    assert np.allclose(colors[0], to_rgb("darkblue"))
    assert np.allclose(colors[1], to_rgb("lightblue"))


def test_plot_system_kary_draws(tmp_path):
    from pyphi.visualize.connectivity import plot_system

    fig, ax = plt.subplots()
    g = plot_system(_k3_system(), ax=ax)
    assert len(g.nodes) == 2
    plt.close(fig)


def test_plot_tpm_kary_integer_labels():
    from pyphi.visualize import plot_tpm

    tpm = np.full((3, 3), 1 / 3)
    fig, ax = plot_tpm(tpm)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "1", "2"]
    plt.close(fig)


def test_plot_tpm_nonsquare_uses_integer_labels():
    from pyphi.visualize import plot_tpm

    # A 16x4 array is not a state-by-state TPM; bit-string labels would be
    # wrong on both axes, so integer state indices are used.
    tpm = np.full((16, 4), 0.25)
    fig, ax = plot_tpm(tpm)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        str(i) for i in range(4)
    ]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        str(i) for i in range(16)
    ]
    plt.close(fig)


def test_plot_tpm_explicit_states():
    from pyphi.visualize import plot_tpm

    tpm = np.full((3, 3), 1 / 3)
    fig, ax = plot_tpm(tpm, states=["L", "M", "H"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["L", "M", "H"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["L", "M", "H"]
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/visualize/test_visualize_matplotlib.py -v -k "kary or nonsquare or explicit_states"`
Expected: `test_system_graph_kary_colors` and `test_plot_system_kary_draws` FAIL with `KeyError: (True, 2)`; `test_plot_tpm_kary_integer_labels` and `test_plot_tpm_explicit_states` FAIL with `ValueError` (FixedLocator locations/labels mismatch) or `TypeError` (unexpected keyword `states`); `test_plot_tpm_nonsquare_uses_integer_labels` FAILS on the label assertion (bit strings instead of integers).

- [ ] **Step 3: Implement**

In `pyphi/visualize/connectivity.py`:

Add below the imports:

```python
import matplotlib.colors as mcolors
```

Add below the `NODE_COLORS` dict:

```python
def _node_color(in_system, state, num_states):
    """Color for one unit: hue family by membership, intensity by state.

    Binary units keep the exact ``NODE_COLORS`` entries; units with a larger
    alphabet interpolate the same family from its light end (state 0) to its
    dark end (state ``num_states - 1``).
    """
    if num_states <= 2:
        return NODE_COLORS[(in_system, state)]
    light, dark = (
        ("lightblue", "darkblue") if in_system else ("lightgrey", "darkgrey")
    )
    fraction = state / (num_states - 1)
    return tuple(
        (1 - fraction) * np.array(mcolors.to_rgb(light))
        + fraction * np.array(mcolors.to_rgb(dark))
    )
```

In `_system_graph`, replace the `colors = [...]` list comprehension with:

```python
    sizes = system.substrate.tpm.alphabet_sizes
    colors = [
        _node_color(i in system.node_indices, system.state[i], sizes[i])
        for i in range(system.substrate.size)
    ]
```

Add a module-private label helper above `plot_tpm`:

```python
def _tick_labels(n, square, states):
    """Axis labels for a TPM axis of length ``n``.

    Explicit ``states`` win when their count matches; a square matrix with a
    power-of-two side is labeled with little-endian bit strings (a binary
    state-by-state TPM); anything else gets integer state indices.
    """
    if states is not None and len(states) == n:
        return list(states)
    if square and n >= 2 and (n & (n - 1)) == 0:
        return list(all_states_str(int(np.log2(n))))
    return [str(i) for i in range(n)]
```

In `plot_tpm`, add the keyword parameter `states=None` (after `xticks_top=True`), give the function a docstring, and use the helper for both axes:

```python
def plot_tpm(
    tpm,
    figsize=(10, 12),
    clim=None,
    cmap="viridis",
    label_fontsize=8,
    show_label_threshold=64,
    xticks_top=True,
    states=None,
):
    """Plot a TPM as a heatmap with state tick labels.

    Parameters
    ----------
    tpm : np.ndarray
        A 2-D transition probability matrix, typically state-by-state.
    states : Sequence[str], optional
        Explicit state labels. An axis is labeled with them when its length
        equals ``len(states)``. If None, a square matrix with a power-of-two
        side is labeled with little-endian bit strings, and integer state
        indices are used otherwise.
    """
```

and replace the two `labels=all_states_str(int(np.log2(tpm.shape[...])))` arguments:

```python
    square = tpm.shape[0] == tpm.shape[1]
    if tpm.shape[1] <= show_label_threshold:
        ax.set_xticks(
            list(range(tpm.shape[1])),
            labels=_tick_labels(tpm.shape[1], square, states),
            rotation=90,
            fontsize=label_fontsize,
        )
        ax.xaxis.set_ticks_position("top" if xticks_top else "bottom")
        ax.xaxis.set_label_position("top" if xticks_top else "bottom")
    if tpm.shape[0] <= show_label_threshold:
        ax.set_yticks(
            list(range(tpm.shape[0])),
            labels=_tick_labels(tpm.shape[0], square, states),
            fontsize=label_fontsize,
        )
```

- [ ] **Step 4: Run the matplotlib visualize suite**

Run: `uv run pytest test/visualize/test_visualize_matplotlib.py -v`
Expected: all PASS — including the pre-existing `test_system_graph_exact` (binary strings unchanged) and `test_plot_tpm_exported_and_labeled` (square 4×4 keeps bit-string labels).

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/connectivity.py test/visualize/test_visualize_matplotlib.py
git commit -m "Support k-ary units in plot_system colors and plot_tpm labels

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 5: Empty-CES guard and IIT 3.0 support in `project_ces`

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py:265-292` (`project_ces`)
- Test: `test/visualize/test_visualize_projection.py`

**Interfaces:**
- Consumes: `Distinction.cause_purview` / `.effect_purview` (tuples of node indices in both formalisms, verified); `NullCauseEffectStructure` from `pyphi.formalism.iit4`.
- Produces: `project_ces` raises `ValueError` (message contains "empty cause-effect structure") on zero distinctions; the purview-union inclusion order no longer touches `Distinction.purview_union`, so IIT 3.0 CESes project and plot.

- [ ] **Step 1: Write the failing tests**

In `test/visualize/test_visualize_projection.py`, add to the top-of-file imports:

```python
from test.conftest import skip_if_no_emd_backend
```

Append:

```python
def test_project_ces_empty_raises_clear_error():
    from pyphi.formalism.iit4 import NullCauseEffectStructure
    from pyphi.visualize.projection import project_ces

    with pytest.raises(ValueError, match="empty cause-effect structure"):
        project_ces(NullCauseEffectStructure())


@pytest.fixture(scope="module")
def iit3_xor_ces():
    import pyphi
    from pyphi import examples
    from test.conftest import IIT_3_CONFIG

    with IIT_3_CONFIG:
        return pyphi.formalism.iit3.ces(examples.xor_system())


@skip_if_no_emd_backend
def test_project_ces_iit3(iit3_xor_ces):
    from pyphi.visualize.projection import project_ces

    projection = project_ces(iit3_xor_ces)
    assert len(projection.nodes) == 3
    # The purview-union inclusion order is built from purview indices, which
    # IIT 3.0 distinctions carry (they have no specified states).
    assert len(projection.purview_union_inclusion.rank) == 3


@skip_if_no_emd_backend
@pytest.mark.parametrize("view", ["lattice", "hypergraph", "scatter"])
def test_plot_ces_iit3_views_render(iit3_xor_ces, view):
    pytest.importorskip("plotly")
    from pyphi.visualize import plot_ces

    fig = plot_ces(iit3_xor_ces, view=view)
    assert fig is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/visualize/test_visualize_projection.py -v -k "empty or iit3"`
Expected: `test_project_ces_empty_raises_clear_error` FAILS with `IndexError: list index out of range`; the IIT 3.0 tests FAIL with a bare `AssertionError` (from `ria.purview_units`).

- [ ] **Step 3: Implement**

In `pyphi/visualize/projection/__init__.py`, `project_ces`:

Replace:

```python
    distinctions = list(ces.distinctions)
    if node_labels is None:
        node_labels = distinctions[0].node_labels
```

with:

```python
    distinctions = list(ces.distinctions)
    if not distinctions:
        raise ValueError(
            "cannot project an empty cause-effect structure "
            "(no distinctions: the system is reducible)"
        )
    if node_labels is None:
        node_labels = distinctions[0].node_labels
```

Replace:

```python
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union) for d in distinctions
    )
```

with:

```python
    # Purview unions as index sets, from the purviews directly: the
    # projection needs only unit indices, and IIT 3.0 distinctions carry no
    # specified states for their purview units.
    unions = tuple(
        frozenset(d.cause_purview) | frozenset(d.effect_purview)
        for d in distinctions
    )
```

Also update the `Raises` section of the `project_ces` docstring to document the new ValueError condition (empty CES).

- [ ] **Step 4: Run the projection suite**

Run: `uv run pytest test/visualize/test_visualize_projection.py -v`
Expected: all PASS — including the pre-existing IIT 4.0 `xor_projection` tests (for IIT 4.0 distinctions the index sets are identical to the old unit-derived ones).

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/projection/__init__.py test/visualize/test_visualize_projection.py
git commit -m "Project IIT 3.0 CESes and reject empty ones with a clear error

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 6: k-ary MCP server paths (`build_substrate`, `_state_by_state`, `plot(kind="tpm")`)

**Files:**
- Modify: `pyphi/mcp/server.py:280-282` (`build_substrate` kwargs), `pyphi/mcp/server.py:534-546` (`_state_by_state`), `pyphi/mcp/server.py:624-626` (the `plot` tpm branch)
- Test: `test/mcp/test_server.py`

**Interfaces:**
- Consumes: `Substrate(tpm=<explicit-alphabet joint array>, state_space=...)` (verified accepted); `pyphi.utils.all_states(sizes)` little-endian mixed-radix enumeration; `substrate.tpm.alphabet_sizes` and `.to_joint()`; `plot_tpm(..., states=...)` from Task 4.
- Produces: `build_substrate(tpm, alphabet=[k0, k1, ...])` works as documented; `_state_by_state(substrate)` returns an S×S matrix (S = ∏ alphabet sizes) for any alphabet, binary output unchanged; `plot(handle, kind="tpm")` renders k-ary TPMs with correct state labels.

- [ ] **Step 1: Write the failing tests**

Append to `test/mcp/test_server.py`:

```python
def _kary_joint_tpm_lists():
    """An explicit-alphabet joint TPM for a two-unit ternary substrate,
    as nested lists (the MCP tool's wire format)."""
    rng = np.random.default_rng(7)

    def marginal():
        m = rng.random((3, 3, 3))
        return m / m.sum(axis=-1, keepdims=True)

    substrate = pyphi.Substrate(
        marginals=[marginal(), marginal()], state_space=(0, 1, 2)
    )
    return np.asarray(substrate.tpm.to_joint()).tolist()


def test_build_substrate_kary_alphabet_list():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    assert out["num_nodes"] == 2
    assert srv._get_substrate(out["handle"]).num_states == 9


def test_state_by_state_binary_output_unchanged(basic_handle):
    from pyphi import convert

    substrate = srv._get_substrate(basic_handle)
    on = np.asarray(substrate.tpm.to_joint())[..., 1]
    expected = convert.state_by_node2state_by_state(
        on.reshape(-1, substrate.size, order="F")
    )
    assert np.allclose(srv._state_by_state(substrate), expected)


def test_state_by_state_kary():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    substrate = srv._get_substrate(out["handle"])
    sbs = srv._state_by_state(substrate)
    assert sbs.shape == (9, 9)
    assert np.allclose(sbs.sum(axis=1), 1.0)


@pytest.mark.skipif(not HAS_VIZ, reason="plotting needs the visualize extra")
def test_plot_tpm_kary_substrate():
    out = srv.build_substrate(_kary_joint_tpm_lists(), alphabet=[3, 3])
    result = srv.plot(out["handle"], kind="tpm")
    assert isinstance(result, list)
    assert ".png" in result[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/mcp/test_server.py -v -k "kary or state_by_state"`
Expected: `test_build_substrate_kary_alphabet_list`, `test_state_by_state_kary`, and `test_plot_tpm_kary_substrate` FAIL with `TypeError: '<' not supported between instances of 'list' and 'int'`; `test_state_by_state_binary_output_unchanged` PASSES (it pins current behavior before the rewrite — keep it).

- [ ] **Step 3: Implement**

In `pyphi/mcp/server.py`:

In `build_substrate`, replace:

```python
    if alphabet is not None:
        kwargs["alphabet"] = alphabet
```

with:

```python
    if alphabet is not None:
        kwargs["state_space"] = tuple(tuple(range(k)) for k in alphabet)
```

Replace `_state_by_state` with:

```python
def _state_by_state(substrate: Any) -> Any:
    """Return a substrate's 2-D state-by-state transition probability matrix.

    Valid for any per-unit alphabet: the (current, next) entry is the product
    over units of each unit's next-state probability given the current joint
    state, with states enumerated in little-endian mixed-radix order
    (``pyphi.utils.all_states``). This is what ``visualize.plot_tpm`` expects.
    """
    from pyphi import utils

    joint = np.asarray(substrate.tpm.to_joint())
    states = list(utils.all_states(substrate.tpm.alphabet_sizes))
    sbs = np.empty((len(states), len(states)))
    for i, current in enumerate(states):
        per_unit = joint[current]  # (unit, next-state) probabilities
        for j, nxt in enumerate(states):
            sbs[i, j] = np.prod([per_unit[u, s] for u, s in enumerate(nxt)])
    return sbs
```

In `plot`, replace the tpm branch:

```python
    elif kind == "tpm":
        substrate = _get_substrate(target)
        fig = visualize.plot_tpm(_state_by_state(substrate))[0]
```

with:

```python
    elif kind == "tpm":
        from pyphi import utils

        substrate = _get_substrate(target)
        states = [
            "".join(map(str, s))
            for s in utils.all_states(substrate.tpm.alphabet_sizes)
        ]
        fig = visualize.plot_tpm(_state_by_state(substrate), states=states)[0]
```

- [ ] **Step 4: Run the MCP suite**

Run: `uv run pytest test/mcp/test_server.py -q > /tmp/wave3-mcp.log 2>&1; uv run python -c "print(open('/tmp/wave3-mcp.log').read()[-2000:])"`
Expected: all PASS, including the pre-existing binary `test_plot_substrate_figures` and the binary-pin `test_state_by_state_binary_output_unchanged`.

- [ ] **Step 5: Add the cluster changelog fragment and commit**

```bash
cat > changelog.d/visualize-mcp-kary-empty-ces.fix.md <<'EOF'
Fixed visualization and MCP crashes on valid input: `plot_system` colors k-ary units (KeyError before), `plot_tpm` labels k-ary and non-square TPMs correctly and accepts explicit `states=` labels, `plot_ces`/`project_ces` support IIT 3.0 cause-effect structures and raise a clear `ValueError` on an empty (reducible-system) structure, the MCP `plot(kind="tpm")` path renders k-ary TPMs, and the MCP `build_substrate` tool's documented `alphabet` list parameter works.
EOF
git add pyphi/mcp/server.py test/mcp/test_server.py changelog.d/visualize-mcp-kary-empty-ces.fix.md
git commit -m "Generalize MCP substrate building and TPM plotting to k-ary units

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 7: `settle()` off-by-one

**Files:**
- Modify: `pyphi/dynamics.py:170-172` (the `max_steps` guard in `settle`)
- Test: `test/test_dynamics.py`

**Interfaces:**
- Consumes: `settle`, `NonConvergenceError`, `convert.state_by_state2state_by_node` (`convert` already imported in the test module).
- Produces: `settle(..., max_steps=m)` returns any trajectory whose settling time (`len(result) - 1`, per the docstring) is ≤ `m`, and raises beyond.

- [ ] **Step 1: Write the failing test**

In `test/test_dynamics.py`, add `import pytest` to the top-of-file imports. Append:

```python
def test_settle_in_exactly_max_steps_returns():
    from pyphi.dynamics import settle

    # Every state maps to (1, 1): a one-step settle from (0, 0).
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1
    tpm = convert.state_by_state2state_by_node(sbs)
    trajectory = settle(tpm, (0, 0), max_steps=1)
    assert trajectory == [(0, 0), (1, 1)]
    assert len(trajectory) - 1 == 1  # settling time == max_steps is allowed


def test_settle_raises_when_settling_time_exceeds_max_steps():
    from pyphi.dynamics import settle
    from pyphi.exceptions import NonConvergenceError

    # Chain (0,0) -> (1,0) -> (1,1) -> (1,1): settling time 2.
    # Little-endian state indices: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3.
    sbs = np.zeros((4, 4))
    sbs[0, 1] = 1
    sbs[1, 3] = 1
    sbs[2, 3] = 1
    sbs[3, 3] = 1
    tpm = convert.state_by_state2state_by_node(sbs)
    with pytest.raises(NonConvergenceError, match="max_steps"):
        settle(tpm, (0, 0), max_steps=1)
    assert settle(tpm, (0, 0), max_steps=2) == [(0, 0), (1, 0), (1, 1)]
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `uv run pytest test/test_dynamics.py -v -k "max_steps"`
Expected: `test_settle_in_exactly_max_steps_returns` FAILS with `NonConvergenceError: did not settle within max_steps=1`; `test_settle_raises_when_settling_time_exceeds_max_steps` PASSES already (the raise side is not affected by the off-by-one) — keep it as the regression pin for the raise side.

- [ ] **Step 3: Implement the fix**

In `pyphi/dynamics.py`, `settle`, replace:

```python
        if max_steps is not None and len(trajectory) > max_steps:
```

with:

```python
        # The just-appended state may itself be the fixed point (confirmed on
        # the next iteration), so the best-case settling time here is
        # len(trajectory) - 1; raise only when even that exceeds the cap.
        if max_steps is not None and len(trajectory) - 1 > max_steps:
```

- [ ] **Step 4: Run the dynamics suite**

Run: `uv run pytest test/test_dynamics.py -v`
Expected: all PASS (including the pre-existing `test_settle_reaches_fixed_point` and `test_settle_already_fixed_returns_length_one`).

- [ ] **Step 5: Add the changelog fragment and commit**

```bash
cat > changelog.d/settle-max-steps.fix.md <<'EOF'
Fixed `settle()` falsely raising `NonConvergenceError` when the trajectory settles in exactly `max_steps` steps.
EOF
git add pyphi/dynamics.py test/test_dynamics.py changelog.d/settle-max-steps.fix.md
git commit -m "Permit settle() trajectories that settle in exactly max_steps steps

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the complete pathless suite in the worktree**

```bash
uv run pytest -q > /tmp/wave3-full.log 2>&1; echo "exit: $?"
```

Then Read `/tmp/wave3-full.log` and check the final summary line. Expected: no failures or errors (skips are fine). Do not trust the exit code alone; read the summary.

- [ ] **Step 2: If green, hand off to finishing-a-development-branch**

Use the superpowers:finishing-a-development-branch skill (merge to `main` locally with `--no-ff` has been the standing choice; re-run the pathless suite in the main tree after merging, then update the review status block and project memory in the main tree).
