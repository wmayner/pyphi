# PerceptualSystem — environment→system layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the substrate-level layer of the matching formalism — a `PerceptualSystem` (substrate U + system S + sensory interface ∂S), the fixed-τ `TriggeredTPM` it produces, and the fixed-τ triggered-state mapping — plus a general `dynamics.settle` primitive.

**Architecture:** A new top-level `pyphi/matching/` package. The triggered TPM is constructed by the validated clamp-then-noise recipe (condition the binary state-by-node TPM on ∂S=x, restrict to S, matrix-power over τ_clamp; noise ∂S, restrict to S, matrix-power over τ−τ_clamp; compose; average over the initial system state). `settle` is the deterministic sibling of `simulate`, added to `pyphi/dynamics.py`.

**Tech Stack:** Python 3.12+, numpy, pytest, `uv run`. Builds on `pyphi.convert`, `pyphi.utils.all_states`, `Substrate`, and the existing `pyphi.dynamics`.

**Spec:** `docs/superpowers/specs/2026-06-06-perceptual-system-design.md`

---

## Background the engineer needs (verified facts)

- **2.0 TPM representation is factored:** `substrate.tpm.to_array()` returns shape `(2,…,2, n, 2)` — state axes per node, then a node axis, then an alphabet axis. For **binary** substrates the classic state-by-node array (`(2,…,2, n)`, entry = P(node ON)) is the **ON-probability slice**: `np.asarray(substrate.tpm.to_array())[..., 1]`. Sub-project 2 **assumes binary substrates** (the matching formalism's sensory units and PMI machinery are binary); document this and validate it in `PerceptualSystem.__post_init__`.
- **State-by-state conversion + lag:** `pyphi.convert.state_by_node2state_by_state(sbn)` turns a classic state-by-node array into a `2^k × 2^k` matrix; `np.linalg.matrix_power(sbs, t)` lags it. `convert.state_by_node2state_by_state` and `convert.le_index2state(index, n)` / `convert.state2le_index(state)` are the index helpers. States are **little-endian** (`utils.all_states(n)` yields first unit fastest: `(0,0),(1,0),(0,1),(1,1)`).
- **The construction is validated** (this exact recipe produces rows that sum to 1 on `examples.basic_substrate()` with sensory `(0,)`, system `(1,2)`):
  ```python
  sbn_full = np.asarray(substrate.tpm.to_array())[..., 1]   # (2,..,2,n) binary
  # system-restricted, sensory-clamped one-step TPM over S:
  #   M_clamp(x)[s_sys] = sbn_full[full_state(x, s_sys)][list(system)]
  # system-restricted, sensory-noised one-step TPM over S:
  #   M_noise[s_sys]    = mean over x of sbn_full[full_state(x, s_sys)][list(system)]
  # lag each via state_by_node2state_by_state + matrix_power, compose, mean(axis=0)
  ```
- **`simulate` stepping:** `pyphi.dynamics.simulate_one_timestep_from_explicit_tpm_state_by_node(rng, tpm, state)` does `tpm[state]` → per-node ON-probabilities → stochastic threshold. The deterministic counterpart is `(probs > 0.5).astype(int)`. `apply_clamp(clamp, state)` overwrites clamped indices in a state tuple.
- **Run a test:** `uv run pytest test/test_triggered_tpm.py::test_name -x -q`. **Commit boundary:** `uv run pytest` (no path, doctest-inclusive). Commit with `git -c commit.gpgsign=false commit`; never `--no-verify`. If a commit silently doesn't land, the hook reformatted — `git add` the same files and re-commit. Ruff bans: `dict()` calls, unicode `×`/`−`/en-dash in strings/docstrings (`·` allowed), mid-file imports (E402 — keep all imports at module top).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `pyphi/exceptions.py` | `NonConvergenceError(ValueError)` | modify |
| `pyphi/dynamics.py` | `settle` + `most_probable_next_state` | modify |
| `pyphi/matching/__init__.py` | package exports | create |
| `pyphi/matching/triggered_tpm.py` | `TriggeredTPM` + `build_triggered_tpm` | create |
| `pyphi/matching/system.py` | `PerceptualSystem` | create |
| `test/test_dynamics.py` | `settle` tests | modify |
| `test/test_triggered_tpm.py` | TriggeredTPM + construction tests | create |
| `test/test_matching_system.py` | PerceptualSystem tests | create |
| `changelog.d/dynamics-settle.feature.md` | changelog | create |
| `changelog.d/perceptual-system.feature.md` | changelog | create |

---

## Task 1: `dynamics.settle` + `NonConvergenceError`

**Files:**
- Modify: `pyphi/exceptions.py` (add after `InvalidTPM`, `:65`)
- Modify: `pyphi/dynamics.py` (add after `simulate_one_timestep_from_explicit_tpm_state_by_node`, `:94`)
- Test: `test/test_dynamics.py`

- [ ] **Step 1: Write the failing tests**

Add to `test/test_dynamics.py` (check its imports; it likely has `import numpy as np` and `from pyphi import dynamics` — add what's missing at the top):

```python
import numpy as np
import pytest

from pyphi import convert
from pyphi import dynamics
from pyphi.exceptions import NonConvergenceError


def _sbn_from_sbs(sbs):
    # state-by-state (rows=current, cols=next) -> multidim state-by-node
    return convert.state_by_state2state_by_node(sbs)


def test_settle_reaches_fixed_point():
    # 2-unit system: deterministic map that drives any state to (1, 1)
    # state-by-state over 4 states (little-endian): every row -> state (1,1)=index 3
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    trajectory = dynamics.settle(tpm, initial_state=(0, 0))
    assert trajectory[-1] == (1, 1)
    assert isinstance(trajectory, list)


def test_settle_already_fixed_returns_length_one():
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    trajectory = dynamics.settle(tpm, initial_state=(1, 1))
    assert trajectory == [(1, 1)]


def test_settle_raises_on_limit_cycle():
    # 1-unit system that flips every step: (0,)->(1,)->(0,)->...
    # state-by-state over 2 states: row 0 -> state 1, row 1 -> state 0
    sbs = np.array([[0.0, 1.0], [1.0, 0.0]])
    tpm = _sbn_from_sbs(sbs)
    with pytest.raises(NonConvergenceError, match="cycle"):
        dynamics.settle(tpm, initial_state=(0,))


def test_settle_clamp_holds_units_fixed():
    # 2 units flip toward all-ON, but clamp unit 0 OFF -> fixed point (0, 1)
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    trajectory = dynamics.settle(tpm, initial_state=(0, 0), clamp={0: 0})
    assert trajectory[-1] == (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_dynamics.py -k settle -x -q`
Expected: FAIL — `cannot import name 'NonConvergenceError'` / `module 'pyphi.dynamics' has no attribute 'settle'`.

- [ ] **Step 3: Add the exception**

In `pyphi/exceptions.py`, after `InvalidTPM` (`:65`):

```python
class NonConvergenceError(ValueError):
    """A deterministic trajectory entered a limit cycle instead of a fixed point."""
```

- [ ] **Step 4: Implement `settle` + the deterministic step**

In `pyphi/dynamics.py`, add the import at the top (with the other imports):

```python
from .exceptions import NonConvergenceError
```

and after `simulate_one_timestep_from_explicit_tpm_state_by_node` (`:94`):

```python
def most_probable_next_state(tpm, state):
    """Return the deterministic most-probable next state (binary).

    Counterpart of the sampled `simulate_one_timestep_*`: each unit takes its
    most-probable next value (ON iff P(ON) > 0.5).
    """
    tpm = JointTPM(tpm)
    elementwise_probabilities = tpm[state]
    return tuple((np.asarray(elementwise_probabilities) > 0.5).astype(int))


def settle(tpm, initial_state, *, clamp=None, max_steps=None):
    """Iterate the most-probable-transition map to a fixed point.

    Deterministic complement to `simulate`: each step takes the most-probable
    next state instead of sampling. Returns the trajectory (a list of states)
    ending at the fixed point; the fixed point is the last element and the
    settling time is ``len(result) - 1``. Raises
    :class:`~pyphi.exceptions.NonConvergenceError` on a limit cycle.

    Args:
        tpm: A state-by-node multidimensional TPM (binary).
        initial_state (tuple[int, ...]): The starting state.

    Keyword Args:
        clamp (Mapping[int, int] | None): Units held fixed every step.
        max_steps (int | None): Optional cap; raises if exceeded.
    """
    if clamp is None:
        clamp = {}
    state = apply_clamp(clamp, tuple(initial_state))
    trajectory = [state]
    seen = {state}
    while True:
        nxt = apply_clamp(clamp, most_probable_next_state(tpm, state))
        if nxt == state:
            return trajectory
        if nxt in seen:
            raise NonConvergenceError(
                f"no fixed point; entered a limit cycle at {nxt} "
                f"(trajectory: {trajectory + [nxt]})"
            )
        trajectory.append(nxt)
        seen.add(nxt)
        state = nxt
        if max_steps is not None and len(trajectory) > max_steps:
            raise NonConvergenceError(
                f"did not settle within max_steps={max_steps}"
            )
```

(`JointTPM` and `np` are already imported in `dynamics.py`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_dynamics.py -k settle -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Changelog + commit**

```bash
cat > changelog.d/dynamics-settle.feature.md <<'EOF'
Added `pyphi.dynamics.settle`, the deterministic complement to `simulate`: it
iterates the most-probable-transition map to a fixed point and returns the
trajectory (the fixed point is the last element; the settling time is its
length minus one). Raises `NonConvergenceError` if the trajectory enters a
limit cycle. Supports `clamp` (units held fixed each step), reusing the
existing clamp machinery.
EOF
git add pyphi/exceptions.py pyphi/dynamics.py test/test_dynamics.py changelog.d/dynamics-settle.feature.md
git -c commit.gpgsign=false commit -m "Add deterministic dynamics.settle and NonConvergenceError"
```

---

## Task 2: `TriggeredTPM` + construction

**Files:**
- Create: `pyphi/matching/__init__.py`
- Create: `pyphi/matching/triggered_tpm.py`
- Test: `test/test_triggered_tpm.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_triggered_tpm.py`:

```python
import numpy as np
import pytest

from pyphi import convert
from pyphi import examples
from pyphi import utils
from pyphi.matching.triggered_tpm import TriggeredTPM
from pyphi.matching.triggered_tpm import build_triggered_tpm


@pytest.fixture(scope="module")
def ttpm():
    substrate = examples.basic_substrate()  # 3 binary units A,B,C
    return build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=2, tau_clamp=1
    )


def test_array_shape_is_sensory_then_system_axes(ttpm):
    # 1 sensory unit + 2 system units -> shape (2, 2, 2)
    assert ttpm.array.shape == (2, 2, 2)


def test_rows_are_distributions(ttpm):
    # summing over the system axes for each stimulus gives 1
    for x in utils.all_states(1):
        row = ttpm.row(x)
        assert row.sum() == pytest.approx(1.0)
        assert np.all(row >= 0)


def test_argmax_state_in_support(ttpm):
    for x in utils.all_states(1):
        state = ttpm.argmax_state(x)
        assert len(state) == 2
        # the argmax cell is positive
        assert ttpm.row(x)[state] > 0


def test_tau_clamp_zero_is_pure_noise():
    substrate = examples.basic_substrate()
    noised = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1, 2), tau=2, tau_clamp=0
    )
    # every stimulus row is identical (input fully marginalized)
    rows = [noised.row(x) for x in utils.all_states(1)]
    assert np.allclose(rows[0], rows[1])


def test_to_pandas_round_trips(ttpm):
    df = ttpm.to_pandas()
    # rows indexed by sensory states, columns by system states
    assert df.shape == (2, 4)
    for x in utils.all_states(1):
        for s in utils.all_states(2):
            assert df.loc[x, s] == pytest.approx(ttpm.array[x + s])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_triggered_tpm.py -x -q`
Expected: FAIL — `No module named 'pyphi.matching'`.

- [ ] **Step 3: Create the package init**

Create `pyphi/matching/__init__.py`:

```python
"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm

__all__ = ["TriggeredTPM", "build_triggered_tpm"]
```

- [ ] **Step 4: Implement `TriggeredTPM` + `build_triggered_tpm`**

Create `pyphi/matching/triggered_tpm.py`:

```python
"""The triggered TPM: the system's fixed-lag response to each stimulus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyphi import convert
from pyphi import utils
from pyphi.labels import NodeLabels


@dataclass(frozen=True)
class TriggeredTPM:
    """Pr(S_t = s | ∂S_{t−τ} = x), one distribution over system states per
    stimulus.

    ``array`` is a multidimensional ndarray with one binary axis per unit,
    ordered ``(sensory axes…, system axes…)``; ``array[x + s]`` is
    Pr(S = s | ∂S = x). Marginalization over unit subsets is a uniform axis
    sum.
    """

    array: np.ndarray
    sensory_indices: tuple[int, ...]
    system_indices: tuple[int, ...]
    node_labels: NodeLabels

    def row(self, stimulus: tuple[int, ...]) -> np.ndarray:
        """The system-state distribution for one stimulus."""
        return self.array[tuple(stimulus)]

    def argmax_state(self, stimulus: tuple[int, ...]) -> tuple[int, ...]:
        """The most-probable system state for a stimulus (the triggered state)."""
        row = self.row(stimulus)
        flat = int(np.argmax(row))
        return convert.le_index2state(flat, len(self.system_indices))

    def to_pandas(self) -> pd.DataFrame:
        """Provisional labeled view: rows = stimulus states, columns = system
        states, values = Pr(s | x). Subsumed by the unified to_pandas project.
        """
        sensory_states = list(utils.all_states(len(self.sensory_indices)))
        system_states = list(utils.all_states(len(self.system_indices)))
        sensory_labels = self.node_labels.coerce_to_labels(self.sensory_indices)
        system_labels = self.node_labels.coerce_to_labels(self.system_indices)
        index = pd.MultiIndex.from_tuples(sensory_states, names=list(sensory_labels))
        columns = pd.MultiIndex.from_tuples(system_states, names=list(system_labels))
        data = [[self.array[x + s] for s in system_states] for x in sensory_states]
        return pd.DataFrame(data, index=index, columns=columns)


def _full_state(sensory_indices, system_indices, x, s_sys, n):
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, si in zip(system_indices, s_sys, strict=True):
        full[i] = si
    return tuple(full)


def _system_step_tpm(sbn_full, sensory_indices, system_indices, n, *, clamp_to):
    """A one-step state-by-node TPM over the system, with the sensory interface
    either clamped to a state (``clamp_to=x``) or marginalized
    (``clamp_to=None``)."""
    system = list(system_indices)
    shape_s = (2,) * len(system_indices)
    step = np.zeros(shape_s + (len(system_indices),))
    for s_sys in utils.all_states(len(system_indices)):
        if clamp_to is not None:
            full = _full_state(sensory_indices, system_indices, clamp_to, s_sys, n)
            step[s_sys] = sbn_full[full][system]
        else:
            acc = np.zeros(len(system_indices))
            for x in utils.all_states(len(sensory_indices)):
                full = _full_state(sensory_indices, system_indices, x, s_sys, n)
                acc += sbn_full[full][system]
            step[s_sys] = acc / (2 ** len(sensory_indices))
    return step


def _lagged_sbs(step_sbn, t):
    sbs = convert.state_by_node2state_by_state(step_sbn)
    if t == 0:
        return np.eye(sbs.shape[0])
    return np.linalg.matrix_power(sbs, t)


def build_triggered_tpm(
    substrate, sensory_indices, system_indices, *, tau, tau_clamp
) -> TriggeredTPM:
    """Construct the triggered TPM by clamp-then-noise evolution.

    Clamp ∂S to the stimulus for ``tau_clamp`` steps, then marginalize ∂S for
    the remaining ``tau − tau_clamp`` steps; compose and average over the
    initial system state. Assumes a binary substrate.
    """
    n = len(substrate.node_indices)
    sbn_full = np.asarray(substrate.tpm.to_array())[..., 1]  # binary ON-prob slice

    noised = _lagged_sbs(
        _system_step_tpm(sbn_full, sensory_indices, system_indices, n, clamp_to=None),
        tau - tau_clamp,
    )
    rows = []
    for x in utils.all_states(len(sensory_indices)):
        clamped = _lagged_sbs(
            _system_step_tpm(
                sbn_full, sensory_indices, system_indices, n, clamp_to=x
            ),
            tau_clamp,
        )
        composed = clamped @ noised
        rows.append(composed.mean(axis=0))  # marginalize initial system state

    flat = np.array(rows)  # (n_stimuli, n_system_states)
    shape = (2,) * len(sensory_indices) + (2,) * len(system_indices)
    array = flat.reshape(shape)
    return TriggeredTPM(
        array=array,
        sensory_indices=tuple(sensory_indices),
        system_indices=tuple(system_indices),
        node_labels=substrate.node_labels,
    )
```

(`NodeLabels.coerce_to_labels(indices) -> tuple[str, ...]` is verified to exist at `pyphi/labels.py:113`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_triggered_tpm.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add pyphi/matching/__init__.py pyphi/matching/triggered_tpm.py test/test_triggered_tpm.py
git -c commit.gpgsign=false commit -m "Add TriggeredTPM and clamp-then-noise construction"
```

---

## Task 3: `PerceptualSystem`

**Files:**
- Create: `pyphi/matching/system.py`
- Modify: `pyphi/matching/__init__.py` (export `PerceptualSystem`)
- Test: `test/test_matching_system.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_matching_system.py`:

```python
import pytest

from pyphi import examples
from pyphi import utils
from pyphi.matching import PerceptualSystem


@pytest.fixture(scope="module")
def perceptual_system():
    return PerceptualSystem(
        examples.basic_substrate(), system_indices=(1, 2), sensory_indices=(0,)
    )


def test_environment_indices(perceptual_system):
    assert perceptual_system.environment_indices == (0,)


def test_rejects_overlapping_partition():
    with pytest.raises(ValueError, match="disjoint"):
        PerceptualSystem(
            examples.basic_substrate(), system_indices=(0, 1), sensory_indices=(1,)
        )


def test_rejects_sensory_in_system():
    with pytest.raises(ValueError):
        PerceptualSystem(
            examples.basic_substrate(), system_indices=(0, 1, 2), sensory_indices=(0,)
        )


def test_triggered_tpm_delegates(perceptual_system):
    ttpm = perceptual_system.triggered_tpm(tau=2, tau_clamp=1)
    assert ttpm.array.shape == (2, 2, 2)


def test_triggered_states_mapping(perceptual_system):
    states = perceptual_system.triggered_states(tau=2, tau_clamp=1)
    assert set(states) == set(utils.all_states(1))           # keyed by stimulus
    for stimulus, response in states.items():
        assert len(response) == 2                            # over system units


def test_triggered_state_single(perceptual_system):
    response = perceptual_system.triggered_state((1,), tau=2, tau_clamp=1)
    assert response == perceptual_system.triggered_states(tau=2, tau_clamp=1)[(1,)]


def test_invalid_tau_raises(perceptual_system):
    with pytest.raises(ValueError):
        perceptual_system.triggered_tpm(tau=1, tau_clamp=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_matching_system.py -x -q`
Expected: FAIL — `cannot import name 'PerceptualSystem'`.

- [ ] **Step 3: Implement `PerceptualSystem`**

Create `pyphi/matching/system.py`:

```python
"""PerceptualSystem: a system embedded in an environment via a sensory interface."""

from __future__ import annotations

from dataclasses import dataclass

from pyphi import utils

from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm


@dataclass(frozen=True)
class PerceptualSystem:
    """A system S within a substrate U, coupled to its environment E = U∖S
    through a sensory interface ∂S ⊆ E.

    Produces the fixed-lag triggered TPM and the triggered response state for
    each stimulus (the state of ∂S). Assumes a binary substrate.
    """

    substrate: object  # pyphi.Substrate
    system_indices: tuple[int, ...]
    sensory_indices: tuple[int, ...]

    def __post_init__(self):
        node_indices = set(self.substrate.node_indices)
        system = set(self.system_indices)
        sensory = set(self.sensory_indices)
        if not system <= node_indices:
            raise ValueError(f"system_indices {self.system_indices} not in substrate")
        if not sensory <= node_indices:
            raise ValueError(f"sensory_indices {self.sensory_indices} not in substrate")
        if not system or not sensory:
            raise ValueError("system_indices and sensory_indices must be non-empty")
        if system & sensory:
            raise ValueError(
                "system_indices and sensory_indices must be disjoint; "
                f"got overlap {sorted(system & sensory)}"
            )

    @property
    def environment_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i in self.substrate.node_indices if i not in set(self.system_indices)
        )

    @property
    def node_labels(self):
        return self.substrate.node_labels

    @staticmethod
    def _validate_tau(tau, tau_clamp):
        if not isinstance(tau, int) or not isinstance(tau_clamp, int):
            raise ValueError("tau and tau_clamp must be integers")
        if tau < 1:
            raise ValueError(f"tau must be >= 1; got {tau}")
        if not 0 <= tau_clamp <= tau:
            raise ValueError(f"require 0 <= tau_clamp <= tau; got {tau_clamp}, {tau}")

    def triggered_tpm(self, *, tau, tau_clamp) -> TriggeredTPM:
        """The fixed-lag response distribution Pr(S_t | ∂S_{t−τ}=x)."""
        self._validate_tau(tau, tau_clamp)
        return build_triggered_tpm(
            self.substrate,
            self.sensory_indices,
            self.system_indices,
            tau=tau,
            tau_clamp=tau_clamp,
        )

    def triggered_states(self, *, tau, tau_clamp) -> dict:
        """Mapping {stimulus: response_state} — the argmax system state per
        stimulus. This is what the Φ-structure computation consumes."""
        ttpm = self.triggered_tpm(tau=tau, tau_clamp=tau_clamp)
        return {
            x: ttpm.argmax_state(x)
            for x in utils.all_states(len(self.sensory_indices))
        }

    def triggered_state(self, stimulus, *, tau, tau_clamp) -> tuple[int, ...]:
        """The response state for a single stimulus."""
        ttpm = self.triggered_tpm(tau=tau, tau_clamp=tau_clamp)
        return ttpm.argmax_state(tuple(stimulus))
```

Update `pyphi/matching/__init__.py`:

```python
"""The matching formalism: perception and matching (Mayner, Juel & Tononi)."""

from .system import PerceptualSystem
from .triggered_tpm import TriggeredTPM
from .triggered_tpm import build_triggered_tpm

__all__ = ["PerceptualSystem", "TriggeredTPM", "build_triggered_tpm"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_matching_system.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/system.py pyphi/matching/__init__.py test/test_matching_system.py
git -c commit.gpgsign=false commit -m "Add PerceptualSystem with triggered TPM and fixed-tau triggered states"
```

---

## Task 4: Hand-computed construction check + verification + changelog

**Files:**
- Modify: `test/test_triggered_tpm.py` (add a hand-verifiable deterministic case)
- Create: `changelog.d/perceptual-system.feature.md`

- [ ] **Step 1: Add a hand-computed deterministic-substrate test**

A relay substrate makes the τ=τ_clamp=1 triggered TPM hand-checkable. Append to `test/test_triggered_tpm.py`. (`pyphi.Substrate(sbn)` accepting a raw state-by-node array of shape `(2,2,2)` is verified to round-trip to `to_array()[..., 1]` shape `(2,2,2)`.) Construct a 2-unit substrate where unit 1 (system) copies unit 0 (sensory) next step:

```python
def test_relay_triggered_tpm_hand_computed():
    import numpy as np
    import pyphi

    # 2 binary units; unit 1 (system) next-state = current unit 0 (sensory).
    # state-by-node shape (2, 2, 2): [a, b] -> P(node ON) for [unit0, unit1].
    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 0] = 0.0      # unit 0 (sensory) — value irrelevant (clamped/marginalized)
            sbn[a, b, 1] = a        # unit 1 copies unit 0
    substrate = pyphi.Substrate(sbn)
    from pyphi.matching.triggered_tpm import build_triggered_tpm

    ttpm = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    # stimulus 0 -> system unit OFF with probability 1; stimulus 1 -> ON w.p. 1
    assert ttpm.row((0,))[(0,)] == 1.0
    assert ttpm.row((1,))[(1,)] == 1.0
    assert ttpm.argmax_state((0,)) == (0,)
    assert ttpm.argmax_state((1,)) == (1,)
```

Run: `uv run pytest test/test_triggered_tpm.py::test_relay_triggered_tpm_hand_computed -x -q`
Expected: PASS.

- [ ] **Step 2: Changelog**

```bash
cat > changelog.d/perceptual-system.feature.md <<'EOF'
Added the `pyphi.matching` package with `PerceptualSystem` — a system embedded
in an environment via a sensory interface — and `TriggeredTPM`, the system's
fixed-lag response `Pr(S_t | ∂S_{t−τ} = x)` to each stimulus, constructed by
clamping the interface for `tau_clamp` steps then evolving for the remaining
`tau − tau_clamp`. `PerceptualSystem.triggered_states` gives the
`{stimulus: response_state}` mapping that the Φ-structure computation consumes.
`TriggeredTPM.to_pandas()` provides a provisional labeled view. Binary
substrates only.
EOF
git add changelog.d/perceptual-system.feature.md
git -c commit.gpgsign=false commit -m "Add changelog fragment for PerceptualSystem"
```

- [ ] **Step 3: Full verification (commit boundary)**

```bash
uv run pytest test/test_dynamics.py test/test_triggered_tpm.py test/test_matching_system.py -q
uv run ruff check pyphi/dynamics.py pyphi/exceptions.py pyphi/matching test/test_triggered_tpm.py test/test_matching_system.py
uv run pyright pyphi/matching pyphi/dynamics.py
```
Expected: all green.

- [ ] **Step 4: Doctest-inclusive sweep**

Run the full suite once (no path argument) per the project testing rules; the new package is small so it's quick to add. Run the slow lane in the background if needed:
```bash
uv run pytest -q
```
Expected: green.

---

## Self-review notes

- **Spec coverage:** `PerceptualSystem` + validation (Task 3); `TriggeredTPM` + multidim array + `to_pandas` (Task 2); clamp-then-noise construction with both τ limits (Task 2 + Task 4); `triggered_states`/`triggered_state` fixed-τ (Task 3); general `dynamics.settle` returning a trajectory + `NonConvergenceError` (Task 1); binary-substrate assumption documented + validated. No `settled_state` (correctly absent). All spec sections map to a task.
- **All API points verified end-to-end before writing:** the clamp-then-noise construction (rows sum to 1 on `basic_substrate`), `NodeLabels.coerce_to_labels` (`labels.py:113`), and `pyphi.Substrate(sbn)` accepting a raw state-by-node array. No "verify during implementation" gaps remain.
- **Name consistency:** `PerceptualSystem`, `TriggeredTPM`, `build_triggered_tpm`, `triggered_tpm`/`triggered_states`/`triggered_state`, `dynamics.settle`, `most_probable_next_state`, `NonConvergenceError` used identically across tasks.
- **Deferred (sub-projects 3–4):** triggering coefficients, perception, differentiation, matching, the dynamics salvage (`stationary_distribution`, Ising sampler).
