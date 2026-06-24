# P14b — Environment generators for matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide library functions that build the matching manuscript's environment world-distributions (segment/point/noise + composition) over a sensory interface, so `MatchingAnalysis` runs without a hand-coded distribution.

**Architecture:** A new `pyphi/matching/environment.py` of pure functions. A generator *is* a `world_distribution` — a `dict[tuple[int, ...], float]` over sensory states (length-`n` 0/1 tuples) summing to 1, computed exactly over all `2^n` states. Primitives (`segment`, `point`, `noise`) and compositions (`superpose` = independent OR, `mixture` = weighted choice) all return distributions, so they nest arbitrarily. A single seeded `sample` helper draws example stimuli.

**Tech Stack:** Python 3.12+, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-p14b-environment-generators-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims. No new third-party dependency.
- Lives in `pyphi/matching/environment.py` (produces stimulus distributions, not substrates — not `substrate_generator`).
- A generator returns `dict[tuple[int, ...], float]`: keys are length-`n` sensory states (0/1 tuples), values are probabilities summing to 1 (within `pyphi.utils.eq` tolerance). Keys align positionally with the caller's `sensory_indices`; "contiguous" means contiguous in that ordering.
- Composition by `superpose` is independent activation combined by elementwise OR; `mixture` is a weighted convex combination.
- All sampling uses an isolated `np.random.default_rng(seed)` — never the global RNG. `seed` is keyword-only and required.
- Use `uv run` for all Python commands. Final verification runs `uv run pytest` **with no path argument** (public surface; doctest sweep).
- Do not bypass pre-commit hooks. Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).

---

### Task 1: Module + primitive generators (`segment`, `point`, `noise`)

**Files:**
- Create: `pyphi/matching/environment.py`
- Test: `test/test_environment.py` (new)

**Interfaces:**
- Produces: `segment(n: int, length: int, p: float) -> dict[tuple[int, ...], float]`
- Produces: `point(n: int, p: float) -> dict[tuple[int, ...], float]`
- Produces: `noise(n: int, p: float) -> dict[tuple[int, ...], float]`
- Produces (internal): `_normalize(dist: dict) -> dict` — validates non-negativity, drops zero-mass states, renormalizes to sum 1.

- [ ] **Step 1: Write the failing tests**

Create `test/test_environment.py`:

```python
import itertools

import numpy as np
import pytest

from pyphi import utils
from pyphi.matching import environment as env


def _sums_to_one(dist):
    return utils.eq(sum(dist.values()), 1.0)


def test_segment_hand_computed():
    # n=4, length=2, p=0.6: 3 positions, each contiguous-2 run gets 0.6/3=0.2;
    # all-off gets 1-0.6=0.4.
    dist = env.segment(4, 2, 0.6)
    assert _sums_to_one(dist)
    assert dist[(0, 0, 0, 0)] == pytest.approx(0.4)
    assert dist[(1, 1, 0, 0)] == pytest.approx(0.2)
    assert dist[(0, 1, 1, 0)] == pytest.approx(0.2)
    assert dist[(0, 0, 1, 1)] == pytest.approx(0.2)
    # No non-contiguous or wrong-length states present.
    assert (1, 0, 1, 0) not in dist
    assert (1, 1, 1, 0) not in dist


def test_segment_full_length_is_all_on_or_off():
    dist = env.segment(3, 3, 0.7)
    assert dist[(1, 1, 1)] == pytest.approx(0.7)
    assert dist[(0, 0, 0)] == pytest.approx(0.3)
    assert _sums_to_one(dist)


def test_point_equals_segment_length_one():
    assert env.point(5, 0.4) == env.segment(5, 1, 0.4)


def test_noise_is_product_bernoulli():
    dist = env.noise(3, 0.25)
    for state in itertools.product((0, 1), repeat=3):
        expected = np.prod([0.25 if s else 0.75 for s in state])
        assert dist[state] == pytest.approx(expected)
    assert _sums_to_one(dist)


def test_noise_half_is_uniform():
    dist = env.noise(4, 0.5)
    assert len(dist) == 16
    for v in dist.values():
        assert v == pytest.approx(1 / 16)


def test_generator_argument_validation():
    with pytest.raises(ValueError):
        env.segment(3, 4, 0.5)  # length > n
    with pytest.raises(ValueError):
        env.segment(3, 0, 0.5)  # length < 1
    with pytest.raises(ValueError):
        env.segment(3, 1, 1.5)  # p out of range
    with pytest.raises(ValueError):
        env.noise(3, -0.1)  # p out of range
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_environment.py -q`
Expected: FAIL (`pyphi.matching.environment` does not exist).

- [ ] **Step 3: Create `pyphi/matching/environment.py`**

```python
# environment.py
"""Environment generators for matching.

A generator is a *world distribution*: a mapping from sensory-interface states
(length-``n`` 0/1 tuples) to probabilities summing to 1, suitable as the
``world_distribution`` of :class:`pyphi.matching.MatchingAnalysis`. Distributions
are computed exactly over the sensory interface. Keys align positionally with the
caller's ``sensory_indices``; "contiguous" refers to that ordering.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from pyphi import utils

Distribution = dict[tuple[int, ...], float]


def _normalize(dist: Distribution) -> Distribution:
    """Validate non-negativity, drop zero-mass states, renormalize to sum 1."""
    if any(p < 0 for p in dist.values()):
        raise ValueError("probabilities must be non-negative")
    total = float(sum(dist.values()))
    if total <= 0:
        raise ValueError("distribution has zero total mass")
    return {state: p / total for state, p in dist.items() if p > 0}


def _check_p(p: float) -> None:
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1]; got {p}")


def segment(n: int, length: int, p: float) -> Distribution:
    """A run of ``length`` contiguous units at a uniformly random location.

    With probability ``p`` a segment is present (its location uniform over the
    ``n - length + 1`` start positions); with probability ``1 - p`` no unit is
    active (all-off).
    """
    _check_p(p)
    if not 1 <= length <= n:
        raise ValueError(f"length must be in [1, n]={n}; got {length}")
    positions = n - length + 1
    dist: Distribution = defaultdict(float)
    dist[tuple([0] * n)] += 1 - p
    for start in range(positions):
        state = [0] * n
        for i in range(start, start + length):
            state[i] = 1
        dist[tuple(state)] += p / positions
    return _normalize(dict(dist))


def point(n: int, p: float) -> Distribution:
    """A single unit active at a uniformly random location with probability ``p``."""
    return segment(n, 1, p)


def noise(n: int, p: float) -> Distribution:
    """Each unit independently active with probability ``p`` (product Bernoulli).

    ``p = 0.5`` yields the uniform "structureless world".
    """
    _check_p(p)
    dist: Distribution = {}
    for state in itertools.product((0, 1), repeat=n):
        prob = 1.0
        for s in state:
            prob *= p if s else (1 - p)
        dist[state] = prob
    return _normalize(dist)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_environment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/environment.py test/test_environment.py
git commit -m "Add matching environment primitive generators

segment/point/noise produce world distributions over a sensory interface
(exact over all 2^n states); _normalize validates and renormalizes."
```

---

### Task 2: Composition (`superpose`, `mixture`)

**Files:**
- Modify: `pyphi/matching/environment.py`
- Test: `test/test_environment.py`

**Interfaces:**
- Consumes: `_normalize` (Task 1); distributions from `segment`/`point`/`noise`.
- Produces: `superpose(*distributions: Distribution) -> Distribution` — independent activation, elementwise OR.
- Produces: `mixture(distributions: list[Distribution], weights: list[float] | None = None) -> Distribution` — weighted convex combination.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_environment.py`:

```python
def test_superpose_or_combines_independently():
    # Deterministic point at index 0 OR deterministic point at index 1.
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    assert env.superpose(a, b) == {(1, 1): 1.0}


def test_superpose_with_all_off_is_identity():
    a = env.segment(4, 2, 0.6)
    off = {(0, 0, 0, 0): 1.0}
    combined = env.superpose(a, off)
    assert set(combined) == set(a)
    for state in a:
        assert combined[state] == pytest.approx(a[state])


def test_superpose_hand_computed_probability():
    # noise(2, 0.5) OR a deterministic point at index 0.
    # Result state (1, x): index 0 always on; index 1 on iff noise set it.
    combined = env.superpose(env.noise(2, 0.5), {(1, 0): 1.0})
    assert combined[(1, 0)] == pytest.approx(0.5)  # noise gave (0,0) or (1,0)
    assert combined[(1, 1)] == pytest.approx(0.5)  # noise gave (0,1) or (1,1)
    assert _sums_to_one(combined)


def test_superpose_requires_matching_n():
    with pytest.raises(ValueError):
        env.superpose({(0, 0): 1.0}, {(0,): 1.0})


def test_mixture_weights():
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    m = env.mixture([a, b], weights=[3, 1])
    assert m[(1, 0)] == pytest.approx(0.75)
    assert m[(0, 1)] == pytest.approx(0.25)
    assert _sums_to_one(m)


def test_mixture_uniform_default():
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    m = env.mixture([a, b])
    assert m[(1, 0)] == pytest.approx(0.5)
    assert m[(0, 1)] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_environment.py -q -k "superpose or mixture"`
Expected: FAIL (`superpose` / `mixture` undefined).

- [ ] **Step 3: Implement `superpose` and `mixture`**

Add to `pyphi/matching/environment.py`:

```python
def _shared_n(distributions) -> int:
    sizes = {len(next(iter(d))) for d in distributions}
    if len(sizes) != 1:
        raise ValueError(f"all distributions must share the same n; got {sizes}")
    return sizes.pop()


def superpose(*distributions: Distribution) -> Distribution:
    """Independent activation of each generator, combined by elementwise OR.

    Each input distribution is drawn independently; a unit is active in the
    result iff any generator activates it. Computed exactly over the product of
    the inputs' supports.
    """
    if not distributions:
        raise ValueError("superpose requires at least one distribution")
    n = _shared_n(distributions)
    result: Distribution = defaultdict(float)
    for combo in itertools.product(*(d.items() for d in distributions)):
        prob = 1.0
        merged = [0] * n
        for state, state_prob in combo:
            prob *= state_prob
            for i, s in enumerate(state):
                if s:
                    merged[i] = 1
        result[tuple(merged)] += prob
    return _normalize(dict(result))


def mixture(
    distributions: list[Distribution], weights: list[float] | None = None
) -> Distribution:
    """A weighted convex combination of distributions (pick one per draw)."""
    if not distributions:
        raise ValueError("mixture requires at least one distribution")
    _shared_n(distributions)
    if weights is None:
        weights = [1.0] * len(distributions)
    if len(weights) != len(distributions):
        raise ValueError("weights must match the number of distributions")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    result: Distribution = defaultdict(float)
    for dist, weight in zip(distributions, weights, strict=True):
        for state, prob in dist.items():
            result[state] += (weight / total) * prob
    return _normalize(dict(result))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_environment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/environment.py test/test_environment.py
git commit -m "Add environment composition: superpose (independent OR) and mixture

superpose draws each generator independently and ORs activations (the paper's
overlay-on-noise construction); mixture is a weighted convex combination."
```

---

### Task 3: Seeded `sample`

**Files:**
- Modify: `pyphi/matching/environment.py`
- Test: `test/test_environment.py`

**Interfaces:**
- Produces: `sample(distribution: Distribution, size: int, *, seed: int) -> list[tuple[int, ...]]`

- [ ] **Step 1: Write the failing tests**

Add to `test/test_environment.py`:

```python
def test_sample_is_seed_deterministic_and_isolated():
    dist = env.noise(3, 0.3)
    a = env.sample(dist, 50, seed=7)
    b = env.sample(dist, 50, seed=7)
    assert a == b  # reproducible
    c = env.sample(dist, 50, seed=8)
    assert a != c  # seed changes output
    # No global-RNG dependence.
    np.random.seed(0)
    d = env.sample(dist, 50, seed=7)
    np.random.seed(123)
    e = env.sample(dist, 50, seed=7)
    assert d == e


def test_sample_only_draws_supported_states():
    dist = env.segment(4, 2, 0.6)
    drawn = set(env.sample(dist, 200, seed=1))
    assert drawn <= set(dist)


def test_sample_empirical_frequencies_converge():
    dist = env.noise(2, 0.5)
    draws = env.sample(dist, 20000, seed=42)
    counts = {state: 0 for state in dist}
    for state in draws:
        counts[state] += 1
    for state, prob in dist.items():
        assert counts[state] / len(draws) == pytest.approx(prob, abs=0.02)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_environment.py -q -k sample`
Expected: FAIL (`sample` undefined).

- [ ] **Step 3: Implement `sample`**

Add to `pyphi/matching/environment.py`:

```python
def sample(
    distribution: Distribution, size: int, *, seed: int
) -> list[tuple[int, ...]]:
    """Draw ``size`` i.i.d. states from a distribution (seeded, isolated RNG).

    Uses ``np.random.default_rng(seed)`` — never the global RNG — so a draw is
    reproducible from ``seed`` alone. A convenience for inspecting an
    environment; ``MatchingAnalysis.matching`` does its own seeded sampling.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative; got {size}")
    rng = np.random.default_rng(seed)
    states = list(distribution.keys())
    probs = np.array(list(distribution.values()), dtype=float)
    probs /= probs.sum()
    indices = rng.choice(len(states), size=size, p=probs)
    return [states[i] for i in indices]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_environment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/environment.py test/test_environment.py
git commit -m "Add seeded sample() for drawing example stimuli from an environment

Uses an isolated np.random.default_rng(seed); reproducible from the seed alone,
no global-RNG dependence."
```

---

### Task 4: Exports, paper environments, and end-to-end

**Files:**
- Modify: `pyphi/matching/__init__.py` (export the generators)
- Test: `test/test_environment.py`

**Interfaces:**
- Consumes: all of `pyphi.matching.environment` (Tasks 1–3); `pyphi.matching.MatchingAnalysis`, `PerceptualSystem`, `Perception`; `pyphi.examples.grid3_substrate`.
- Produces: top-level `from pyphi.matching import segment, point, noise, superpose, mixture, sample`.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_environment.py`:

```python
import pyphi
from pyphi.matching import MatchingAnalysis
from pyphi.matching import Perception
from pyphi.matching import PerceptualSystem


def _e1(n):
    return env.superpose(env.segment(n, 3, 0.6), env.segment(n, 2, 0.9), env.noise(n, 0.05))


def _e2(n):
    return env.superpose(env.segment(n, 3, 0.6), env.point(n, 0.9), env.noise(n, 0.05))


def _e1b(n):
    return env.superpose(env.segment(n, 2, 0.9), env.noise(n, 0.05))


def test_paper_environments_normalized_with_full_support():
    n = 5
    for environment in (_e1(n), _e2(n), _e1b(n), env.noise(n, 0.5)):
        assert _sums_to_one(environment)
        assert all(len(state) == n for state in environment)


def test_e3_pure_noise_is_uniform():
    n = 5
    e3 = env.noise(n, 0.5)
    assert all(v == pytest.approx(1 / 2**n) for v in e3.values())


def test_e1_all_off_probability_hand_computed():
    # All-off occurs iff no segment fires AND the noise background is all-off.
    n = 5
    e1 = _e1(n)
    expected = (1 - 0.6) * (1 - 0.9) * (0.95**n)
    assert e1[(0, 0, 0, 0, 0)] == pytest.approx(expected)


def test_top_level_exports():
    assert pyphi.matching.segment is env.segment
    assert pyphi.matching.superpose is env.superpose
    assert pyphi.matching.sample is env.sample


def test_matching_analysis_runs_on_generated_world_distribution():
    substrate = pyphi.examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    perceptions = {}
    for stimulus in [(0,), (1,)]:
        y = ttpm.argmax_state(stimulus)
        full = [0, 0, 0]
        full[sensory[0]] = stimulus[0]
        for j, idx in enumerate(system):
            full[idx] = y[j]
        ces = substrate.ces(state=tuple(full), indices=system)
        perceptions[stimulus] = Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)
    world = env.noise(1, 0.3)  # {(0,): 0.7, (1,): 0.3}
    analysis = MatchingAnalysis(perceptions=perceptions, world_distribution=world)
    result = analysis.matching(seed=0, n_trials=5, k=3)
    assert result is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_environment.py -q -k "paper or exports or runs_on or e3 or e1_all_off"`
Expected: FAIL (`pyphi.matching.segment` not exported).

- [ ] **Step 3: Export the generators from `pyphi.matching`**

In `pyphi/matching/__init__.py`, add after the existing imports:

```python
from .environment import mixture
from .environment import noise
from .environment import point
from .environment import sample
from .environment import segment
from .environment import superpose
```

And add the names to `__all__`:

```python
    "mixture",
    "noise",
    "point",
    "sample",
    "segment",
    "superpose",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_environment.py -q`
Expected: PASS (all environment tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/matching/__init__.py test/test_environment.py
git commit -m "Export environment generators; pin paper environments + end-to-end

Export segment/point/noise/superpose/mixture/sample from pyphi.matching.
Reproduce the paper's E1/E2/E1b/E3 compositions and run MatchingAnalysis on a
generator-produced world distribution."
```

---

### Task 5: Changelog, roadmap, and full verification

**Files:**
- Create: `changelog.d/p14b-environment-generators.feature.md`
- Modify: `ROADMAP.md` (P14b env-generation dashboard row; Wave 2 archive bullet; "Landed" prose line)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/p14b-environment-generators.feature.md`:

```markdown
Added environment generators for matching (`pyphi.matching.environment`):
`segment`, `point`, and `noise` build world distributions over a sensory
interface; `superpose` (independent OR) and `mixture` (weighted choice) compose
them; `sample` draws example stimuli with a seeded, isolated RNG. These let
`MatchingAnalysis` run without a hand-coded world distribution and reproduce the
manuscript's environments (E1/E2/E1b/E3).
```

- [ ] **Step 2: Update the ROADMAP dashboard row for P14b env-generation**

In `ROADMAP.md`, find the `P14b env-generation` dashboard row and change its status from `⬜ open` to `✅ landed`, updating the one-line to:

```markdown
| P14b env-generation | ✅ landed | 2 | Environment generators in `pyphi/matching/environment.py`: `segment`/`point`/`noise` produce world distributions over a sensory interface, composed via `superpose` (independent OR — the paper's overlay-on-noise) and `mixture`; seeded `sample` for inspection. `MatchingAnalysis` now runs without a hand-coded world distribution; reproduces the manuscript's E1/E2/E1b/E3. **Scope correction:** the paper defines the world distribution via stimulus generators, *not* a substrate stationary distribution — so `stationary_distribution`/Metropolis-Ising (substrate construction) stay out of scope (deferred; would live in `substrate_generator`). |
```

- [ ] **Step 3: Update the Wave 2 archive bullet and the Landed prose line**

In the Wave 2 archive, update the `P14b tail` bullet's env-generation portion to past tense describing what landed (the generators, the inferred scope correction, out-of-scope stationary/Ising). In the `### ✅ Landed` prose line near the top, append `· P14b env-gen`.

- [ ] **Step 4: Run the full verification gate**

Run: `uv run pytest`
Expected: PASS with **no path argument** (collects `pyphi/` + `test/` doctests and `test/test_environment.py`). Run the slow lane in the background per the project's parallel-test guidance if needed; the gate is the full no-path run.

- [ ] **Step 5: Commit**

```bash
git add changelog.d/p14b-environment-generators.feature.md ROADMAP.md
git commit -m "Mark P14b env-generation landed: environment generators; changelog + roadmap"
```

---

## Self-Review

**Spec coverage:**
- Module + core representation (spec 4.1) → Task 1 (`_normalize`).
- Primitive generators (spec 4.2) → Task 1.
- Composition superpose/mixture (spec 4.3) → Task 2.
- Seeded `sample` (spec 4.4) → Task 3.
- Paper environments E1/E2/E1b/E3 (spec 4.5) → Task 4.
- Testing (spec 5): primitives, composition, invariants, paper environments, seeded sampling, end-to-end → Tasks 1–4.
- Exports + roadmap + changelog (spec 7) → Tasks 4–5.

**Placeholder scan:** none — every step shows complete code.

**Type consistency:** every generator and composition returns `Distribution = dict[tuple[int, ...], float]`; `sample` returns `list[tuple[int, ...]]`. `_shared_n`/`_normalize`/`_check_p` are the shared internals. The end-to-end test builds `perceptions` exactly as the existing `test/test_matching.py` fixture does (grid3, sensory=(0,), system=(1,2)).
