# Auxiliary Plot Migration Implementation Plan (P14d-A sub-project 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `connectivity`/`distribution`/`dynamics`/`ising` onto the visualize discipline — pure data-extraction split from figure emission, exact-value + figure-structure tests (currently zero coverage), `plot_tpm` exported, `plot_dynamics` side-effect fixed. Closes P14d-A.

**Architecture:** These stay matplotlib/seaborn (maintainer decision) and don't join the φ-structure projection; the pattern applied is the discipline: each plot's model coupling is isolated in a private, testable extraction function, and renderers return figures instead of showing them.

**Tech Stack:** matplotlib (Agg in tests), seaborn, pandas, numpy, pytest.

**Verified facts** (do not re-probe):
- All four modules currently *work* (probed): `xor_system().sia()` takes 0.04s, `sia.partition` is a `DirectedSetPartition`; `plot_repertoires(system, sia)` returns (fig, 2 axes, reps) with reps keyed by `Direction.CAUSE/EFFECT` × {"unpartitioned","partitioned"}, repertoire shape (2,2,2), CAUSE unpartitioned sums to 2.0 (forward repertoires are unnormalized — hence `validate=False`). `plot_system(xor)` → nodes ['A','B','C'], 6 edges; xor state is (0,0,0) so all node colors are `NODE_COLORS[(True, 0)] == "lightblue"`.
- Ising with `w=[[0,1],[1,0]]`, T=1, field=0, spin=0: states 00,01,10,11 → energies −1,1,−1,1; probabilities ≈0.268941, 0.731059, 0.268941, 0.731059.
- `plot_dynamics` currently calls `plt.show()` and returns `(fig, AxesImage)` (not an Axes).
- Test modules MUST set `matplotlib.use("Agg", force=True)` before any pyplot import, and `plt.close(fig)` after each figure assertion.
- Commit dance: hooks may reformat → re-`git add` + fresh commit; never `--no-verify`/amend; targeted `git add`; `git -c commit.gpgsign=false commit`.

---

### Task 1: dynamics — return the figure, no side effects

**Files:**
- Modify: `pyphi/visualize/dynamics.py`
- Test: `test/test_visualize_aux.py` (new)

- [ ] **Step 1: Write the failing test** — create `test/test_visualize_aux.py`:

```python
"""Tests for the auxiliary visualize modules (matplotlib-based)."""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_plot_dynamics_returns_figure_without_showing():
    from pyphi.visualize.dynamics import plot_dynamics

    data = np.array([[0, 1, 0], [1, 0, 1]])  # (timesteps=2, units=3)
    fig, ax = plot_dynamics(data, node_labels=["A", "B", "C"], title="t")
    assert isinstance(ax, plt.Axes)
    # Time runs horizontally: image is (units, timesteps).
    image = ax.get_images()[0].get_array()
    assert image.shape == (3, 2)
    assert np.array_equal(image, data.T)
    assert ax.get_title() == "t"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B", "C"]
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: FAIL — current code returns `(fig, AxesImage)`, so `isinstance(ax, plt.Axes)` fails (and `plt.show()` is a no-op under Agg).

- [ ] **Step 3: Implement** — replace the body of `pyphi/visualize/dynamics.py`:

```python
# visualize/dynamics.py
"""Visualize state trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_dynamics(
    data: ArrayLike, node_labels=None, title="", fig=None, ax=None, figsize=(25, 5)
):
    """Plot an array of states over time.

    Arguments:
        data (ArrayLike): An array of states with shape (timesteps, units).

    Returns:
        tuple: The matplotlib figure and axes.
    """
    # Plot time horizontally.
    data = np.transpose(data)
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif ax is None:
        ax = fig.gca()
    elif fig is None:
        fig = ax.figure
    im = ax.imshow(data, aspect="auto", interpolation="none", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_title(title)
    ax.set_ylabel("Substrate state")
    ax.set_xlabel("Time")
    if node_labels is not None:
        ax.set_yticks(range(len(node_labels)), node_labels)
    fig.colorbar(im, ax=ax)
    return fig, ax
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/dynamics.py test/test_visualize_aux.py
git -c commit.gpgsign=false commit -m "plot_dynamics returns its figure instead of showing it"
```

---

### Task 2: connectivity — extraction split, export plot_tpm

**Files:**
- Modify: `pyphi/visualize/connectivity.py`, `pyphi/visualize/__init__.py`
- Test: `test/test_visualize_aux.py`

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_aux.py`:

```python
@pytest.fixture(scope="module")
def xor_system():
    from pyphi import examples

    return examples.xor_system()


def test_system_graph_exact(xor_system):
    from pyphi.visualize.connectivity import _system_graph

    g, colors = _system_graph(xor_system)
    assert sorted(g.nodes) == ["A", "B", "C"]
    # xor connectivity: all off-diagonal edges.
    assert len(g.edges) == 6
    assert ("A", "A") not in g.edges
    # All units are in the system, state (0, 0, 0).
    assert colors == ["lightblue"] * 3


def test_plot_system_draws_and_returns_graph(xor_system):
    from pyphi.visualize.connectivity import plot_system

    fig, ax = plt.subplots()
    g = plot_system(xor_system, ax=ax)
    assert sorted(g.nodes) == ["A", "B", "C"]
    plt.close(fig)


def test_plot_tpm_exported_and_labeled():
    from pyphi.visualize import plot_tpm

    tpm = np.eye(4)
    fig, ax = plot_tpm(tpm)
    image = ax.get_images()[0].get_array()
    assert np.array_equal(image, tpm)
    # 2-bit state labels on both axes.
    assert [t.get_text() for t in ax.get_xticklabels()] == ["00", "01", "10", "11"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["00", "01", "10", "11"]
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: new tests FAIL (`ImportError: _system_graph` / `ImportError: plot_tpm` from pyphi.visualize)

- [ ] **Step 3: Implement**

In `pyphi/visualize/connectivity.py`, replace `plot_system` with the extraction + renderer pair:

```python
def _system_graph(system):
    """Directed graph of the system's connectivity and per-unit colors."""
    g = nx.from_numpy_array(system.cm, create_using=nx.DiGraph)
    nx.relabel_nodes(
        g,
        dict(zip(range(system.substrate.size), system.node_labels, strict=False)),
        copy=False,
    )
    colors = [
        NODE_COLORS[(i in system.node_indices, system.state[i])]
        for i in range(system.substrate.size)
    ]
    return g, colors


def plot_system(system, **kwargs):
    g, colors = _system_graph(system)
    kwargs.setdefault("node_color", colors)
    plot_graph(g, **kwargs)
    return g
```

In `pyphi/visualize/__init__.py`: add `from .connectivity import plot_tpm` next to the other connectivity imports and `"plot_tpm"` to `__all__` (sorted position).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/connectivity.py pyphi/visualize/__init__.py test/test_visualize_aux.py
git -c commit.gpgsign=false commit -m "Split system-graph extraction from drawing; export plot_tpm"
```

---

### Task 3: distribution — frame extraction for plot_distribution

**Files:**
- Modify: `pyphi/visualize/distribution.py`
- Test: `test/test_visualize_aux.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def test_distribution_frame_exact():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.array([0.5, 0.25, 0.125, 0.125])
    frame, default_label = _distribution_frame([d])
    assert list(frame["state"]) == ["00", "01", "10", "11"]
    assert list(frame["probability"]) == [0.5, 0.25, 0.125, 0.125]
    assert set(frame["hue"]) == {"0"}
    assert default_label == "AB"


def test_distribution_frame_multiple_with_labels():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.array([0.5, 0.5])
    frame, _ = _distribution_frame([d, d], labels=["x", "y"])
    assert len(frame) == 4
    assert list(frame["hue"]) == ["x", "x", "y", "y"]


def test_distribution_frame_validates():
    from pyphi.visualize.distribution import _distribution_frame

    with pytest.raises(ValueError, match="sum to 1"):
        _distribution_frame([np.array([0.5, 0.6])])
    # Disabled validation lets unnormalized data through.
    frame, _ = _distribution_frame([np.array([0.5, 0.6])], validate=False)
    assert len(frame) == 2


def test_distribution_frame_large_uses_integer_states():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.full(128, 1 / 128)
    frame, default_label = _distribution_frame([d])
    assert default_label is None
    assert list(frame["state"])[:3] == [0, 1, 2]


def test_plot_distribution_bars():
    from pyphi.visualize import plot_distribution

    fig, ax = plot_distribution(np.array([0.5, 0.25, 0.125, 0.125]))
    assert len(ax.patches) == 4
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: frame tests FAIL (`ImportError: _distribution_frame`); the bars test passes already.

- [ ] **Step 3: Implement** — in `pyphi/visualize/distribution.py`, add after `all_states_str`:

```python
def _distribution_frame(
    distributions, states=None, labels=None, lineplot_threshold=64, validate=True
):
    """Tidy frame of probabilities by state and series, plus the default
    state-space label (unit names) when bit-string states are inferred."""
    if validate and not all(np.allclose(d.sum(), 1, rtol=1e-4) for d in distributions):
        raise ValueError("a distribution does not sum to 1!")
    series = [pd.Series(distribution.flatten(d)) for d in distributions]
    first = series[0]
    if validate and not all((first.index == s.index).all() for s in series):
        raise ValueError("distribution indices do not match")
    n = log2(np.prod(first.shape))
    default_label = None
    if states is None:
        if n.is_integer() and len(first) <= lineplot_threshold:
            states = list(all_states_str(int(n)))
            default_label = string.ascii_uppercase[: int(n)]
        else:
            states = np.arange(len(first))
    if labels is None:
        labels = list(map(str, range(len(series))))
    frame = pd.concat(
        [
            pd.DataFrame({"probability": s, "state": states, "hue": [lab] * len(s)})
            for s, lab in zip(series, labels, strict=False)
        ]
    ).reset_index(drop=True)
    return frame, default_label
```

Rewrite `plot_distribution`'s body to consume it (drop the inlined validation/series/states/labels/concat logic):

```python
    data, default_label = _distribution_frame(
        distributions,
        states=states,
        labels=labels,
        lineplot_threshold=lineplot_threshold,
        validate=validate,
    )
    if label is None:
        label = default_label

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    n_points = len(data) // len(distributions)
    if n_points > lineplot_threshold:
        ax = _plot_distribution_line(data, ax, hue="hue", **kwargs)
    else:
        ax = _plot_distribution_bar(data, ax, label, hue="hue", **kwargs)
```

(keep the trailing title/labels/legend/return block; delete the now-unused `defaults = {}` boilerplate).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/distribution.py test/test_visualize_aux.py
git -c commit.gpgsign=false commit -m "Split tidy-frame extraction from distribution plotting"
```

---

### Task 4: distribution — repertoire-comparison extraction

**Files:**
- Modify: `pyphi/visualize/distribution.py`
- Test: `test/test_visualize_aux.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
@pytest.fixture(scope="module")
def xor_sia(xor_system):
    return xor_system.sia()


def test_repertoire_comparison_values(xor_system, xor_sia):
    from pyphi.direction import Direction
    from pyphi.visualize.distribution import _repertoire_comparison

    reps = _repertoire_comparison(xor_system, xor_sia)
    assert set(reps) == {Direction.CAUSE, Direction.EFFECT}
    for by_label in reps.values():
        assert set(by_label) == {"unpartitioned", "partitioned"}
        for r in by_label.values():
            assert r.shape == (2, 2, 2)
    # Partitioning changes the repertoires.
    cause = reps[Direction.CAUSE]
    assert not np.allclose(cause["unpartitioned"], cause["partitioned"])
    # Forward repertoires are unnormalized.
    assert cause["unpartitioned"].sum() == pytest.approx(2.0)


def test_plot_repertoires_smoke(xor_system, xor_sia):
    from pyphi.visualize.distribution import plot_repertoires

    fig, axes, reps = plot_repertoires(xor_system, xor_sia)
    assert len(axes) == 2
    assert len(reps) == 2
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: FAIL (`ImportError: _repertoire_comparison`)

- [ ] **Step 3: Implement** — in `pyphi/visualize/distribution.py`, split `plot_repertoires`:

```python
def _repertoire_comparison(system, sia):
    """Forward repertoires of the system and its partitioned counterpart,
    keyed by direction, then by "unpartitioned"/"partitioned"."""
    if config.formalism.iit.mechanism_phi_measure not in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
        raise NotImplementedError(
            "Only mechanism_phi_measure = "
            "GENERALIZED_INTRINSIC_DIFFERENCE or INTRINSIC_INFORMATION is supported"
        )
    systems = {
        "unpartitioned": system,
        "partitioned": system.apply_cut(sia.partition),
    }
    return {
        direction: {
            label: s.forward_repertoire(direction, s.node_indices, s.node_indices)
            for label, s in systems.items()
        }
        for direction in Direction.both()
    }


def plot_repertoires(system, sia, **kwargs):
    repertoires = _repertoire_comparison(system, sia)
    labels = ["unpartitioned", "partitioned"]
    fig = plt.figure(figsize=(12, 9))
    axes = fig.subplots(2, 1)
    for ax, direction in zip(axes, Direction.both(), strict=False):
        plot_distribution(
            repertoires[direction][labels[0]],
            repertoires[direction][labels[1]],
            validate=False,
            title=str(direction),
            labels=labels,
            ax=ax,
            **kwargs,
        )
    fig.tight_layout(h_pad=0.5)
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.1, 1.1))
    return fig, axes, repertoires
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/distribution.py test/test_visualize_aux.py
git -c commit.gpgsign=false commit -m "Isolate repertoire-comparison extraction from its plot"
```

---

### Task 5: ising — state-energies extraction

**Files:**
- Modify: `pyphi/visualize/ising.py`
- Test: `test/test_visualize_aux.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
def test_ising_state_energies_exact():
    from pyphi.visualize.ising import _state_energies

    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    data = _state_energies(w, temperature=1.0, field=0.0)
    assert list(data["state"]) == ["00", "01", "10", "11"]
    assert list(data["energy"]) == [-1.0, 1.0, -1.0, 1.0]
    assert data["probability"][0] == pytest.approx(0.268941, abs=1e-5)
    assert data["probability"][1] == pytest.approx(0.731059, abs=1e-5)


def test_ising_plot_smoke():
    from pyphi.visualize.ising import plot

    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    fig = plot(w, temperature=1.0, field=0.0)
    assert fig is not None
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: FAIL (`ImportError: _state_energies`)

- [ ] **Step 3: Implement** — in `pyphi/visualize/ising.py`, split `plot`:

```python
def _state_energies(weights, temperature, field, N=None, spin=0):
    """Energy and activation probability of one spin across all states."""
    if N is None:
        N = weights.shape[0]
    else:
        weights = weights[:N, :N]
    rows = []
    for state in all_states(N):
        spin_state = utils.binary2spin(state)
        # Probability that the spin is "on" in the next micro-timestep.
        e = energy(spin, weights, spin_state)
        rows.append(
            {
                "energy": e,
                "probability": utils.sigmoid(e, temperature=temperature, field=field),
                "state": "".join(map(str, state)),
            }
        )
    return pd.DataFrame(rows)


def plot(weights, temperature, field, N=None, spin=0):
    data = _state_energies(weights, temperature, field, N=N, spin=spin)
    limit = np.max(np.abs(data["energy"]))
    x = np.linspace(-limit, limit, num=200)
    fig = plt.figure(figsize=(15, 6))
    ax = plot_sigmoid(x, temperature=temperature, field=field)
    ax = plot_inputs(
        data=data, x="energy", y="probability", label="state", ax=ax, sep=0.05
    )
    return fig
```

(Also fix the module docstring — it still says "visualize/__init__.py": first line becomes `# visualize/ising.py`.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_aux.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/ising.py test/test_visualize_aux.py
git -c commit.gpgsign=false commit -m "Isolate Ising state-energy extraction from its plot"
```

---

### Task 6: Changelog, full verification, close P14d-A

**Files:**
- Create: `changelog.d/visualize-aux-plots.change.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Changelog fragment**

```bash
cat > changelog.d/visualize-aux-plots.change.md <<'MD'
`plot_dynamics` now returns its figure and axes instead of calling
`plt.show()`, and accepts `fig`/`ax`/`figsize`. `plot_tpm` is exported from
`pyphi.visualize`. The auxiliary plot modules (connectivity, distribution,
dynamics, ising) now separate data extraction from figure emission and are
covered by tests.
MD
```

- [ ] **Step 2: Full verification**

Run: `uv run pytest -q` (no path — doctest-inclusive); `uv run ruff check pyphi test`.
Expected: all green.

- [ ] **Step 3: Mark P14d-A sub-project 4 (and with it P14d) done in ROADMAP.md** — update the P14d status block: A-4 done, P14d complete; next item per the release gate is P14b.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/visualize-aux-plots.change.md ROADMAP.md
git -c commit.gpgsign=false commit -m "Close out P14d: auxiliary plot migration done"
```
