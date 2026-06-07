# Triggering coefficients + perception — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the single-stimulus perception layer — triggering coefficients from a `TriggeredTPM`, and an immutable `Perception` view exposing per-component and total perception for one stimulus.

**Architecture:** `TriggeredTPM` gains marginalization primitives (`conditional_probability` = p, `marginal_probability` = q). A `TriggeringCoefficient` frozen dataclass + `triggering_coefficient(...)` compute Eq 5–7. An immutable `Perception(ces, triggered_tpm, stimulus)` view layers perception on a cause-effect structure without mutating it, caching the triggering-coefficient map.

**Tech Stack:** Python 3.12+, numpy, pytest, `uv run`. Builds on sub-projects 1 (`PhiFold.big_phi_contribution`) and 2 (`TriggeredTPM`, `PerceptualSystem`).

**Spec:** `docs/superpowers/specs/2026-06-06-perception-design.md`

---

## Background the engineer needs (verified facts)

- **`TriggeredTPM.array`** has axes `(∂S axes…, S axes…)`, one binary axis per unit, ordered by `sensory_indices` then `system_indices`. `array[tuple(stimulus)]` indexes the ∂S axes → the system-state distribution.
- **Distinction mechanisms are substrate-global indices**, ⊆ `system_indices`, and **sorted ascending** (pyphi mechanisms are sorted tuples). `distinction.mechanism`, `distinction.mechanism_state`, `distinction.phi` are the accessors. `system_indices` is also sorted ascending — so after summing out non-mechanism system axes, the remaining axes are already in mechanism order; index directly with `state`.
- **A `Relation`** (`pyphi/relations.py:126`) is iterable over its relata, which are `Distinction`s: `[rel.mechanism for rel in relation]` works; `relation.phi` is the relation φ.
- **The CES's current system state** is `ces.sia.current_state` (a tuple over `ces.sia.node_indices`), set from the System's `state`. This is the response state `y` the CES was computed in — used for the consistency guard. (`ces.sia.system_state` is a *different* thing — the specified cause/effect state — do not use it here.)
- **Build a CES over a sub-system in a chosen state:** `substrate.ces(state=full_state, indices=system_indices)` where `full_state` is the full substrate state (background units conditioned on their state). Returns a `CauseEffectStructure` whose `sia.current_state` equals the system slice of `full_state` and `sia.node_indices == system_indices`.
- States are **little-endian**; `utils.all_states(n)` yields them. **Build a substrate from a raw state-by-node array:** `pyphi.Substrate(sbn)` (verified in sub-project 2).
- **Run a test:** `uv run pytest test/test_triggering.py::test_name -x -q`. **Commit boundary:** `uv run pytest` (no path). Commit `git -c commit.gpgsign=false commit`; never `--no-verify`; re-`git add` + re-commit if the hook reformats. Ruff bans: `dict()` calls, unicode `×`/`−`/en-dash in strings/docstrings (`·` allowed), mid-file imports (E402 — all imports at top), list/tuple concatenation (RUF005 — use `[*a, b]` / `(*a, b)`).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `pyphi/matching/triggered_tpm.py` | `conditional_probability`, `marginal_probability` | modify |
| `pyphi/matching/triggering.py` | `TriggeringCoefficient` + `triggering_coefficient` | create |
| `pyphi/matching/perception.py` | `Perception` view | create |
| `pyphi/matching/__init__.py` | exports | modify |
| `test/test_triggered_tpm.py` | marginalization tests | modify |
| `test/test_triggering.py` | triggering-coefficient tests | create |
| `test/test_perception.py` | perception tests | create |
| `changelog.d/perception.feature.md` | changelog | create |

---

## Task 1: `TriggeredTPM` marginalization primitives

**Files:**
- Modify: `pyphi/matching/triggered_tpm.py` (add methods to `TriggeredTPM`)
- Test: `test/test_triggered_tpm.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_triggered_tpm.py`:

```python
def test_conditional_probability_relay():
    import pyphi

    # unit 1 copies unit 0; sensory=(0,), system=(1,), tau=tau_clamp=1
    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    # Pr(unit1 = 1 | dS = 1) = 1 ; Pr(unit1 = 0 | dS = 1) = 0
    assert t.conditional_probability((1,), (1,), (1,)) == pytest.approx(1.0)
    assert t.conditional_probability((1,), (0,), (1,)) == pytest.approx(0.0)
    assert t.conditional_probability((1,), (0,), (0,)) == pytest.approx(1.0)


def test_marginal_probability_relay():
    import pyphi

    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    # Pr(unit1 = 1) = mean over stimuli = (0 + 1) / 2 = 0.5
    assert t.marginal_probability((1,), (1,)) == pytest.approx(0.5)
    assert t.marginal_probability((1,), (0,)) == pytest.approx(0.5)


def test_conditional_probability_subset_marginalizes(ttpm):
    # for a single-unit mechanism, conditional prob over that unit equals the
    # row summed to that unit
    for x in utils.all_states(1):
        p1 = ttpm.conditional_probability((1,), (1,), x)
        # equals sum over unit-2 axis of the (unit1=1) slice
        row = ttpm.row(x)  # shape (2, 2): axes = unit1, unit2
        assert p1 == pytest.approx(row[1, :].sum())


def test_marginalization_rejects_out_of_system_mechanism(ttpm):
    with pytest.raises(ValueError):
        ttpm.conditional_probability((0,), (1,), (0,))  # unit 0 is sensory, not system
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest test/test_triggered_tpm.py -k "conditional or marginal or marginaliz" -x -q`
Expected: FAIL — `'TriggeredTPM' object has no attribute 'conditional_probability'`.

- [ ] **Step 3: Implement the primitives**

In `pyphi/matching/triggered_tpm.py`, add to the `TriggeredTPM` class (after `argmax_state`):

```python
    def _marginalize_system(self, distribution, mechanism, state):
        """Given a distribution over the system axes, return Pr(mechanism = state)
        by summing out the system units not in `mechanism`."""
        mechanism = tuple(mechanism)
        if not set(mechanism) <= set(self.system_indices):
            raise ValueError(
                f"mechanism {mechanism} is not a subset of system_indices "
                f"{self.system_indices}"
            )
        if len(state) != len(mechanism):
            raise ValueError(f"state {state} length != mechanism {mechanism} length")
        keep = [self.system_indices.index(m) for m in mechanism]
        sum_axes = tuple(
            a for a in range(len(self.system_indices)) if a not in keep
        )
        reduced = distribution.sum(axis=sum_axes) if sum_axes else distribution
        # mechanism and system_indices are both sorted, so `keep` is increasing
        # and the remaining axes are already in mechanism order.
        return float(reduced[tuple(state)])

    def conditional_probability(self, mechanism, state, stimulus) -> float:
        """Pr(mechanism = state | dS = stimulus)."""
        return self._marginalize_system(self.row(stimulus), mechanism, state)

    def marginal_probability(self, mechanism, state) -> float:
        """Pr(mechanism = state), the uniform-prior marginal over stimuli."""
        marginal = self.array.mean(axis=tuple(range(len(self.sensory_indices))))
        return self._marginalize_system(marginal, mechanism, state)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest test/test_triggered_tpm.py -k "conditional or marginal or marginaliz" -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/triggered_tpm.py test/test_triggered_tpm.py
git -c commit.gpgsign=false commit -m "Add conditional/marginal probability primitives to TriggeredTPM"
```

---

## Task 2: `TriggeringCoefficient` + computation

**Files:**
- Create: `pyphi/matching/triggering.py`
- Modify: `pyphi/matching/__init__.py`
- Test: `test/test_triggering.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_triggering.py`:

```python
import numpy as np
import pytest

import pyphi
from pyphi.matching.triggered_tpm import build_triggered_tpm
from pyphi.matching.triggering import TriggeringCoefficient
from pyphi.matching.triggering import triggering_coefficient


@pytest.fixture
def relay_ttpm():
    sbn = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            sbn[a, b, 1] = a
    substrate = pyphi.Substrate(sbn)
    return build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )


def test_triggering_coefficient_fully_triggered(relay_ttpm):
    # unit 1 fully determined by stimulus: p=1, q=0.5, c=log2(2)=1, info=1, t=1
    tc = triggering_coefficient(relay_ttpm, (1,), (1,), (1,))
    assert isinstance(tc, TriggeringCoefficient)
    assert tc.p == pytest.approx(1.0)
    assert tc.q == pytest.approx(0.5)
    assert tc.connectedness == pytest.approx(1.0)
    assert tc.value == pytest.approx(1.0)


def test_triggering_coefficient_in_unit_interval(relay_ttpm):
    for state in [(0,), (1,)]:
        for stimulus in [(0,), (1,)]:
            tc = triggering_coefficient(relay_ttpm, (1,), state, stimulus)
            assert 0.0 <= tc.value <= 1.0
            assert tc.connectedness >= 0.0


def test_triggering_coefficient_no_effect_is_zero():
    # 2-unit substrate where the system unit ignores the sensory unit:
    # unit 1 next-state = 0 regardless -> p == q -> connectedness 0 -> t 0
    sbn = np.zeros((2, 2, 2))  # all next-states 0
    substrate = pyphi.Substrate(sbn)
    t = build_triggered_tpm(
        substrate, sensory_indices=(0,), system_indices=(1,), tau=1, tau_clamp=1
    )
    tc = triggering_coefficient(t, (1,), (0,), (1,))
    assert tc.p == pytest.approx(tc.q)
    assert tc.connectedness == pytest.approx(0.0)
    assert tc.value == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest test/test_triggering.py -x -q`
Expected: FAIL — `No module named 'pyphi.matching.triggering'`.

- [ ] **Step 3: Implement**

Create `pyphi/matching/triggering.py`:

```python
"""Triggering coefficients: how much a stimulus caused a mechanism's state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriggeringCoefficient:
    """The extent to which a stimulus caused a mechanism's state (Eq 7).

    ``value`` is t(x, m) in [0, 1]; ``connectedness`` is c(x, m) (positive PMI,
    Eq 5); ``p`` and ``q`` are Pr(M=m | dS=x) and Pr(M=m).
    """

    value: float
    connectedness: float
    p: float
    q: float


def triggering_coefficient(triggered_tpm, mechanism, state, stimulus):
    """Compute the triggering coefficient for a mechanism state given a stimulus."""
    p = triggered_tpm.conditional_probability(mechanism, state, stimulus)
    q = triggered_tpm.marginal_probability(mechanism, state)
    # Connectedness is the positive PMI: zero unless the stimulus raised the
    # probability of the mechanism state (Eq 5).
    if p > 0 and q > 0 and p >= q:
        connectedness = float(np.log2(p / q))
    else:
        connectedness = 0.0
    # Normalize by the mechanism state's self-information (Eq 7).
    information = -float(np.log2(q)) if q > 0 else 0.0
    value = connectedness / information if information > 0 else 0.0
    return TriggeringCoefficient(
        value=value, connectedness=connectedness, p=p, q=q
    )
```

Update `pyphi/matching/__init__.py` to add:

```python
from .triggering import TriggeringCoefficient
from .triggering import triggering_coefficient
```

and add `"TriggeringCoefficient"`, `"triggering_coefficient"` to `__all__`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest test/test_triggering.py -q`
Expected: PASS (3 tests). The fully-triggered case is the hand-computed absolute check (p=1, q=0.5, c=1, t=1).

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/triggering.py pyphi/matching/__init__.py test/test_triggering.py
git -c commit.gpgsign=false commit -m "Add TriggeringCoefficient and its computation (Eq 5-7)"
```

---

## Task 3: `Perception` view

**Files:**
- Create: `pyphi/matching/perception.py`
- Modify: `pyphi/matching/__init__.py`
- Test: `test/test_perception.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_perception.py`:

```python
import numpy as np
import pytest

from pyphi import examples
from pyphi.matching import PerceptualSystem
from pyphi.matching.perception import Perception


def _full_state(sensory_indices, system_indices, x, y):
    n = len(sensory_indices) + len(system_indices)
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, yi in zip(system_indices, y, strict=True):
        full[i] = yi
    return tuple(full)


@pytest.fixture(scope="module")
def perception():
    # grid3 over (1,2) yields 3 distinctions and 5 relations -> exercises the
    # relation-perception path (basic_substrate over (1,2) has no relations).
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    stimulus = (1,)
    y = ttpm.argmax_state(stimulus)
    ces = substrate.ces(
        state=_full_state(sensory, system, stimulus, y), indices=system
    )
    return Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)


def test_distinction_perception_is_t_times_phi(perception):
    for d in perception.ces.distinctions:
        tc = perception.triggering_coefficients[d.mechanism]
        assert perception.distinction_perception(d) == pytest.approx(
            tc.value * float(d.phi)
        )


def test_distinction_perception_at_most_phi(perception):
    for d in perception.ces.distinctions:
        assert perception.distinction_perception(d) <= float(d.phi) + 1e-12


def test_relation_perception_is_phi_times_mean_t(perception):
    for r in perception.ces.relations:
        mean_t = np.mean(
            [perception.triggering_coefficients[rel.mechanism].value for rel in r]
        )
        assert perception.relation_perception(r) == pytest.approx(
            float(r.phi) * mean_t
        )


def test_richness_is_sum_of_component_perceptions(perception):
    expected = sum(
        perception.distinction_perception(d) for d in perception.ces.distinctions
    ) + sum(perception.relation_perception(r) for r in perception.ces.relations)
    assert perception.richness == pytest.approx(expected)


def test_fold_perception_uses_big_phi_contribution(perception):
    d = next(iter(perception.ces.distinctions))
    fold = perception.ces.fold([d])
    tc = perception.triggering_coefficients[d.mechanism]
    assert perception.fold_perception(fold) == pytest.approx(
        tc.value * fold.big_phi_contribution
    )


def test_consistency_guard_rejects_wrong_state():
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    stimulus = (1,)
    y = ttpm.argmax_state(stimulus)
    wrong = tuple(1 - v for v in y)  # deliberately wrong system state
    ces = substrate.ces(
        state=_full_state(sensory, system, stimulus, wrong), indices=system
    )
    with pytest.raises(ValueError, match="triggered"):
        Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest test/test_perception.py -x -q`
Expected: FAIL — `cannot import name 'Perception'`.

- [ ] **Step 3: Implement**

Create `pyphi/matching/perception.py`:

```python
"""Perception: the portion of a cause-effect structure triggered by a stimulus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .triggering import triggering_coefficient

if TYPE_CHECKING:
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.models.ces import PhiFold

    from .triggered_tpm import TriggeredTPM


@dataclass(frozen=True)
class Perception:
    """The triggering coefficients and perception values for one stimulus.

    A pure view over a cause-effect structure: it computes how much of the
    structure's cause-effect power was triggered by ``stimulus``, without
    modifying the structure. ``ces`` must be the structure triggered by
    ``stimulus`` (its system state equals the stimulus's triggered state).
    """

    ces: CauseEffectStructure
    triggered_tpm: TriggeredTPM
    stimulus: tuple[int, ...]

    def __post_init__(self):
        sia = self.ces.sia
        if tuple(sia.node_indices) != tuple(self.triggered_tpm.system_indices):
            raise ValueError(
                "ces system nodes do not match the triggered TPM system units"
            )
        triggered = self.triggered_tpm.argmax_state(self.stimulus)
        if tuple(sia.current_state) != tuple(triggered):
            raise ValueError(
                f"ces system state {tuple(sia.current_state)} is not the state "
                f"triggered by stimulus {self.stimulus} ({tuple(triggered)})"
            )

    @cached_property
    def triggering_coefficients(self) -> dict:
        """Mapping {mechanism: TriggeringCoefficient}, one per distinction."""
        return {
            d.mechanism: triggering_coefficient(
                self.triggered_tpm, d.mechanism, d.mechanism_state, self.stimulus
            )
            for d in self.ces.distinctions
        }

    def distinction_perception(self, distinction) -> float:
        """t(x, m) * phi_d (Eq 8)."""
        t = self.triggering_coefficients[distinction.mechanism].value
        return t * float(distinction.phi)

    def relation_perception(self, relation) -> float:
        """phi_r * mean over relata of t(x, relatum) (Eq 9-10, full phi_r)."""
        mean_t = float(
            np.mean(
                [self.triggering_coefficients[rel.mechanism].value for rel in relation]
            )
        )
        return float(relation.phi) * mean_t

    def fold_perception(self, fold: PhiFold) -> float:
        """t(x, m) * Phi_d (Eq 11), for the single-distinction fold of m."""
        (seed,) = fold.distinctions
        t = self.triggering_coefficients[seed.mechanism].value
        return t * fold.big_phi_contribution

    @cached_property
    def richness(self) -> float:
        """Total perceptual richness (Eq 13)."""
        distinctions = sum(
            self.distinction_perception(d) for d in self.ces.distinctions
        )
        relations = sum(
            self.relation_perception(r) for r in self.ces.relations
        )
        return distinctions + relations
```

Update `pyphi/matching/__init__.py`: add `from .perception import Perception` and `"Perception"` to `__all__`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest test/test_perception.py -q`
Expected: PASS (6 tests). The `grid3` fixture yields 3 distinctions and 5 relations (verified), so the relation-perception and richness tests genuinely exercise the relation path.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/perception.py pyphi/matching/__init__.py test/test_perception.py
git -c commit.gpgsign=false commit -m "Add Perception view: per-component and total perception for one stimulus"
```

---

## Task 4: Changelog + verification

**Files:**
- Create: `changelog.d/perception.feature.md`

- [ ] **Step 1: Changelog**

```bash
cat > changelog.d/perception.feature.md <<'EOF'
Added the single-stimulus perception layer to `pyphi.matching`:
`TriggeringCoefficient` and `triggering_coefficient` compute t(x,m) (Eq 5-7)
from a `TriggeredTPM` (which gains `conditional_probability` and
`marginal_probability`), and `Perception(ces, triggered_tpm, stimulus)` exposes
the per-distinction (t*phi_d), per-relation (mean-relata-t * phi_r),
per-fold (t * big_phi_contribution), and total perceptual richness for a
stimulus, as an immutable view that never mutates the structure.
EOF
git add changelog.d/perception.feature.md
git -c commit.gpgsign=false commit -m "Add changelog fragment for perception layer"
```

- [ ] **Step 2: Targeted suite + lint + pyright**

```bash
uv run pytest test/test_triggered_tpm.py test/test_triggering.py test/test_perception.py -q
uv run ruff check pyphi/matching test/test_triggering.py test/test_perception.py
uv run pyright pyphi/matching
```
Expected: all green.

- [ ] **Step 3: Doctest-inclusive full sweep (commit boundary)**

```bash
uv run pytest -q
```
Expected: green. Use the background + fast-lane split if it's slow.

---

## Self-review notes

- **Spec coverage:** marginalization primitives (Task 1); `TriggeringCoefficient` + Eq 5–7 with edge cases (Task 2); `Perception` view with cached coefficient map, per-distinction/relation/fold perception (full φ_r), richness, and the consistency guard (Task 3); regression coverage via the hand-computed relay (t=1) + formula-consistency tests. All spec sections map to a task.
- **All API points verified before writing:** marginalization-by-axis-sum, `Distinction.mechanism`/`.mechanism_state`/`.phi`, `Relation` iterating relata, `ces.sia.current_state`/`.node_indices`, `substrate.ces(state=, indices=)`, `cached_property` on a frozen dataclass.
- **Name consistency:** `TriggeringCoefficient`, `triggering_coefficient`, `conditional_probability`, `marginal_probability`, `Perception`, `distinction_perception`, `relation_perception`, `fold_perception`, `richness`, `triggering_coefficients` used identically across tasks.
- **Deferred (sub-project 4):** cross-stimulus projection, differentiation, matching. `Perception` is the per-stimulus unit it will union.
- **Dropped:** `TriggeringCoefficientMax` (paper uses full ∂S).
