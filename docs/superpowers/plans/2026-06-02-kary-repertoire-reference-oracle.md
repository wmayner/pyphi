# k-ary Repertoire Reference Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit an independent brute-force cause/effect repertoire reference and assert pyphi's repertoires match it across topologies × alphabets × cuts, catching the class of k-ary bug fixed in `7a8efe62`.

**Architecture:** A standalone NumPy reference (`test/reference/repertoire.py`) computes cause/effect repertoires from raw per-node factors + a connectivity matrix, independent of pyphi. Tests assert pyphi matches it: a self-anchor against hand-computed values, a deterministic enumerated sweep, an `apply_cut` bridge, and an upgraded Hypothesis value check.

**Tech Stack:** Python 3.12+, NumPy, pytest, Hypothesis. Run with `uv run`. Tests-only — no library source changes, so goldens are trivially unaffected.

**Reference:** spec `docs/superpowers/specs/2026-06-02-kary-repertoire-reference-oracle-design.md`.

---

## Background the engineer needs

- A `Substrate` with heterogeneous alphabets is built as
  `Substrate(marginals=[factor_i, ...], state_space=((0,1,2), ...), cm=cm)`.
  Each factor is full-dimension `(*alphabet_sizes, k_i)`.
- `substrate.factored_tpm.factor(i)` returns the forward per-node conditional
  `P(node_i,t+1 | prev_states)`, shape `(*alphabet_sizes, k_i)`. This same array
  is used for both cause and effect in the reference (pyphi derives the cause
  repertoire from it by Bayesian inversion inside the algebra; the stored factor
  is identical for both directions).
- A cut is realized by building a `Substrate` whose `cm` is the cut matrix and
  evaluating it under the default `NullCut`. Per-node TPMs depend only on the
  resulting `cm`, so this is equivalent to applying a partition (Task 4 proves
  this).
- `System(substrate=sub, state=(...), node_indices=(...)).repertoire(direction,
  mechanism, purview)` returns the repertoire as a NumPy array in canonical
  shape (purview nodes at their alphabet size, all other axes size 1).
- `pyphi.utils.state_of(indices, state)` extracts a sub-state tuple.
- Tests live under `test/` (an importable package; `test/golden/` is a
  subpackage, so `test/reference/` needs an `__init__.py`). Fast lane:
  `uv run pytest test/ -m "not slow"`. Full incl. doctests: `uv run pytest`
  (no path argument).

---

## Task 1: Reference module + self-anchor against hand-computed values

**Files:**
- Create: `test/reference/__init__.py` (empty)
- Create: `test/reference/repertoire.py`
- Create: `test/test_repertoire_reference.py`

- [ ] **Step 1: Create the empty package marker**

```bash
mkdir -p test/reference && : > test/reference/__init__.py
```

- [ ] **Step 2: Write the reference module**

Create `test/reference/repertoire.py`:

```python
"""Independent brute-force cause/effect repertoire reference.

Computes IIT cause/effect repertoires directly from raw per-node forward
factors and a (possibly cut) connectivity matrix, using only NumPy. Deliberately
shares no code with ``pyphi.core.repertoire_algebra`` or ``pyphi.node`` so it
serves as a genuine cross-check (see the k-ary cut bug fixed in 7a8efe62).

Each ``factors[i]`` is the forward conditional ``P(node_i,t+1 | prev_states)``
with shape ``(*alphabet_sizes, k_i)`` (obtainable from
``substrate.factored_tpm.factor(i)``). ``cut_cm[j, i] == 1`` means node ``j`` is
an input to node ``i``; severing an edge in a partition is modelled by zeroing
the corresponding ``cut_cm`` entry.
"""

from __future__ import annotations

import numpy as np


def _canonical_shape(alph, purview, n):
    pv = set(purview)
    return tuple(alph[i] if i in pv else 1 for i in range(n))


def ref_effect(factors, alph, cut_cm, mechanism, mstate, purview, n):
    """Effect repertoire P(purview_{t+1} | mechanism_t = mstate), under cut_cm.

    For each purview node z, condition its forward factor on the mechanism
    nodes that are still its inputs after the cut (delta at their state) and
    average uniformly over every other previous-state axis.
    """
    mech = set(mechanism)
    out = np.ones(_canonical_shape(alph, purview, n))
    for z in purview:
        inputs_z = {j for j in range(n) if cut_cm[j, z] == 1}
        cond = mech & inputs_z
        weight = np.ones(alph)
        for j in range(n):
            shape = [1] * n
            shape[j] = alph[j]
            if j in cond:
                v = np.zeros(alph[j])
                v[mstate[j]] = 1.0
            else:
                v = np.full(alph[j], 1.0 / alph[j])
            weight = weight * v.reshape(shape)
        eff_z = np.tensordot(
            weight, factors[z], axes=(list(range(n)), list(range(n)))
        )  # shape (k_z,)
        canon = [1] * n
        canon[z] = alph[z]
        out = out * eff_z.reshape(canon)
    return out


def ref_cause(factors, alph, cut_cm, mechanism, mstate, purview, n):
    """Cause repertoire P(purview_{t-1} | mechanism_t = mstate), under cut_cm.

    Product over mechanism nodes of the forward factor sliced at the node's
    observed state, averaged over every previous-state axis EXCEPT the purview
    nodes that remain inputs to that mechanism node after the cut, then
    normalized. Averaging over severed inputs (even when they are in the
    purview) is what applies the cut on the cause side — omitting it silently
    ignores the cut.
    """
    pv = set(purview)
    joint = np.ones(_canonical_shape(alph, purview, n))
    for m in mechanism:
        cut_inputs_m = {j for j in range(n) if cut_cm[j, m] == 1}
        g = factors[m][..., mstate[m]]
        for ax in range(n):
            if not (ax in pv and ax in cut_inputs_m):
                g = g.mean(axis=ax, keepdims=True)
        joint = joint * g
    total = joint.sum()
    return joint / total if total != 0 else joint
```

- [ ] **Step 3: Write the self-anchor test (hand-computed, no pyphi)**

Create `test/test_repertoire_reference.py`:

```python
"""pyphi cause/effect repertoires vs an independent reference."""

import itertools
import zlib

import numpy as np
import pytest

from pyphi import Direction, Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System
from pyphi.utils import state_of
from test.reference.repertoire import ref_cause, ref_effect


def _swap_factors():
    # 2 binary nodes, SWAP dynamics: node0' = prev1, node1' = prev0.
    # cm: 1->0 and 0->1. Factor i shape (prev0, prev1, out_i).
    f0 = np.zeros((2, 2, 2))
    f1 = np.zeros((2, 2, 2))
    for p0 in range(2):
        for p1 in range(2):
            f0[p0, p1, p1] = 1.0  # node0' copies prev1
            f1[p0, p1, p0] = 1.0  # node1' copies prev0
    cm = np.array([[0, 1], [1, 0]])
    return [f0, f1], (2, 2), cm


def test_reference_matches_hand_computed_swap():
    factors, alph, cm = _swap_factors()
    # Effect of {node0 = 1} on purview {node1}: node1' = prev0 = 1 -> [0, 1].
    eff = ref_effect(factors, alph, cm, (0,), {0: 1, 1: 0}, (1,), 2)
    assert np.allclose(eff.reshape(-1), [0.0, 1.0])
    # Cause of {node1 = 1} over purview {node0}: node1 = prev0 -> prev0 = 1 -> [0, 1].
    cau = ref_cause(factors, alph, cm, (1,), {0: 0, 1: 1}, (0,), 2)
    assert np.allclose(cau.reshape(-1), [0.0, 1.0])
```

- [ ] **Step 4: Run the self-anchor**

Run: `uv run pytest test/test_repertoire_reference.py -q`
Expected: PASS (proves the reference is correct independently of pyphi).

- [ ] **Step 5: Commit**

```bash
git add test/reference/__init__.py test/reference/repertoire.py test/test_repertoire_reference.py
git -c commit.gpgsign=false commit -m "Add independent k-ary repertoire reference + self-anchor"
```

If the commit does not land (a hook reformatted a file), re-`git add` the listed files and re-run (no `--no-verify`, no `--amend`).

---

## Task 2: Deterministic enumerated sweep (pyphi vs reference)

**Files:**
- Modify: `test/test_repertoire_reference.py`

- [ ] **Step 1: Add the sweep helpers and test**

Append to `test/test_repertoire_reference.py`:

```python
def _make_substrate(seed, alph, connectivity):
    rng = np.random.default_rng(seed)
    n = len(alph)
    if connectivity == "dense":
        cm = np.ones((n, n), dtype=int)
    elif connectivity == "chain":
        cm = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            cm[i, i + 1] = 1
    elif connectivity == "cycle":
        cm = np.zeros((n, n), dtype=int)
        for i in range(n):
            cm[i, (i + 1) % n] = 1
    else:
        raise ValueError(connectivity)
    factors = []
    for i in range(n):
        f = rng.uniform(size=(*alph, alph[i]))
        f /= f.sum(axis=-1, keepdims=True)
        factors.append(f)
    state_space = tuple(tuple(range(k)) for k in alph)
    return factors, cm, state_space


def _cut_cms(base_cm):
    edges = [(a, b) for a in range(base_cm.shape[0]) for b in range(base_cm.shape[1]) if base_cm[a, b] == 1]
    cms = [base_cm.copy()]
    for e in edges:
        c = base_cm.copy()
        c[e] = 0
        cms.append(c)
    if len(edges) >= 2:
        c = base_cm.copy()
        c[edges[0]] = 0
        c[edges[1]] = 0
        cms.append(c)
    return cms


_SWEEP_CASES = [
    (2, (2, 2)),
    (2, (3, 3)),
    (2, (2, 3)),
    (3, (2, 2, 2)),
    (3, (3, 3, 4)),
    (3, (2, 3, 4)),
]
_CONNECTIVITY = ["dense", "chain", "cycle"]


@pytest.mark.parametrize("n,alph", _SWEEP_CASES)
@pytest.mark.parametrize("connectivity", _CONNECTIVITY)
def test_repertoires_match_reference_sweep(n, alph, connectivity):
    # Deterministic seed (zlib.crc32 is stable across runs, unlike hash()).
    seed = zlib.crc32(repr((n, alph, connectivity)).encode())
    factors, base_cm, state_space = _make_substrate(seed, alph, connectivity)
    state = tuple(0 for _ in range(n))
    mstate = dict(enumerate(state_of(range(n), state)))
    subsets = [
        s
        for k in range(1, n + 1)
        for s in itertools.combinations(range(n), k)
    ]
    for cut_cm in _cut_cms(base_cm):
        sub = Substrate(
            marginals=[f.copy() for f in factors], state_space=state_space, cm=cut_cm
        )
        sys = System(substrate=sub, state=state, node_indices=tuple(range(n)))
        for direction, reffn in (
            (Direction.CAUSE, ref_cause),
            (Direction.EFFECT, ref_effect),
        ):
            for mech in subsets:
                for purv in subsets:
                    got = np.asarray(sys.repertoire(direction, mech, purv))
                    expected = reffn(factors, alph, cut_cm, mech, mstate, purv, n)
                    assert got.shape == expected.shape
                    assert np.allclose(got, expected, atol=1e-12)
                    # invariants
                    assert np.isclose(got.sum(), 1.0, atol=1e-12)
                    assert np.all(got >= -1e-12)
                    assert tuple(got.shape) == tuple(
                        repertoire_shape(
                            range(n), purv, alphabet_sizes=alph
                        )
                    )
```

- [ ] **Step 2: Run the sweep and check runtime**

Run: `uv run pytest test/test_repertoire_reference.py::test_repertoires_match_reference_sweep -q --durations=0`
Expected: PASS. If the total wall-time exceeds ~10s, add `@pytest.mark.slow`
above the two `@pytest.mark.parametrize` decorators so it leaves the fast lane;
otherwise leave it in the fast lane.

- [ ] **Step 3: Commit**

```bash
git add test/test_repertoire_reference.py
git -c commit.gpgsign=false commit -m "Add enumerated repertoire-vs-reference sweep"
```

---

## Task 3: `apply_cut` bridge test

**Files:**
- Modify: `test/test_repertoire_reference.py`

- [ ] **Step 1: Add the bridge test**

Append to `test/test_repertoire_reference.py`:

```python
def test_partition_repertoire_matches_cut_cm_substrate():
    # A System with a DirectedBipartition applied yields the same repertoire as
    # a substrate whose cm IS the partition-induced cm. Closes the gap that the
    # sweep (which builds cut-cm substrates directly) does not exercise
    # partition.apply_cut itself.
    from pyphi.models.partitions import DirectedBipartition

    alph = (3, 3, 4)
    n = 3
    factors, base_cm, state_space = _make_substrate(2028, alph, "dense")
    state = (0, 0, 0)
    base = Substrate(
        marginals=[f.copy() for f in factors], state_space=state_space, cm=base_cm
    )
    s = System(substrate=base, state=state, node_indices=(0, 1, 2))
    part = DirectedBipartition(
        Direction.EFFECT, from_nodes=(0,), to_nodes=(1, 2), node_labels=s.node_labels
    )
    cut_cm = part.apply_cut(base.cm)
    s_cut = s.apply_cut(part)
    ref_sub = Substrate(
        marginals=[f.copy() for f in factors], state_space=state_space, cm=cut_cm
    )
    s_ref = System(substrate=ref_sub, state=state, node_indices=(0, 1, 2))
    for direction in (Direction.CAUSE, Direction.EFFECT):
        a = np.asarray(s_cut.repertoire(direction, (0,), (2,)))
        b = np.asarray(s_ref.repertoire(direction, (0,), (2,)))
        assert np.allclose(a, b, atol=1e-12)
```

- [ ] **Step 2: Run the bridge test**

Run: `uv run pytest test/test_repertoire_reference.py::test_partition_repertoire_matches_cut_cm_substrate -q`
Expected: PASS. If `DirectedBipartition`/`apply_cut` rejects the partition
(e.g. an `indices`/`node_indices` mismatch), stop and inspect — `part.indices`
must equal the system's `node_indices` `(0, 1, 2)`; `from_nodes=(0,)` +
`to_nodes=(1, 2)` satisfies that.

- [ ] **Step 3: Commit**

```bash
git add test/test_repertoire_reference.py
git -c commit.gpgsign=false commit -m "Add apply_cut bridge test for repertoires"
```

---

## Task 4: Upgrade the Hypothesis property test to a value check

**Files:**
- Modify: `test/test_repertoire_kary_properties.py`

- [ ] **Step 1: Replace the test body with a value check against the reference**

In `test/test_repertoire_kary_properties.py`, change the imports to add the
reference and `state_of`, and replace the body of
`test_repertoire_canonical_shape_and_normalized` so it also asserts value
equality. The full updated file:

```python
"""Property tests: cause/effect repertoires match the independent reference,
are canonical-shaped, and normalized for arbitrary alphabet sizes and
connectivity."""

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import Direction
from pyphi import Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System
from pyphi.utils import state_of
from test.reference.repertoire import ref_cause, ref_effect


def _random_substrate(seed, alphabets, dense):
    rng = np.random.default_rng(seed)
    n = len(alphabets)
    alph = tuple(alphabets)
    if dense:
        cm = np.ones((n, n), dtype=int)
    else:
        # chain: node i -> node i+1
        cm = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            cm[i, i + 1] = 1
    marginals = []
    for i in range(n):
        f = rng.uniform(size=(*alph, alph[i]))
        f /= f.sum(axis=-1, keepdims=True)
        marginals.append(f)
    state_space = tuple(tuple(range(k)) for k in alph)
    return marginals, cm, state_space


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    alphabets=st.lists(st.integers(2, 4), min_size=2, max_size=3),
    dense=st.booleans(),
    direction=st.sampled_from([Direction.CAUSE, Direction.EFFECT]),
)
def test_repertoire_matches_reference(seed, alphabets, dense, direction):
    marginals, cm, state_space = _random_substrate(seed, alphabets, dense)
    n = len(alphabets)
    alph = tuple(alphabets)
    state = tuple(0 for _ in range(n))
    sub = Substrate(marginals=marginals, state_space=state_space, cm=cm)
    s = System(substrate=sub, state=state, node_indices=tuple(range(n)))
    mechanism, purview = (0,), (n - 1,)
    got = np.asarray(s.repertoire(direction, mechanism, purview))
    reffn = ref_cause if direction == Direction.CAUSE else ref_effect
    mstate = dict(enumerate(state_of(range(n), state)))
    expected = reffn(marginals, alph, cm, mechanism, mstate, purview, n)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-12)
    assert tuple(got.shape) == tuple(
        repertoire_shape(s.node_indices, purview, alphabet_sizes=alph)
    )
    assert np.isclose(got.sum(), 1.0)
```

- [ ] **Step 2: Run the property test**

Run: `uv run pytest test/test_repertoire_kary_properties.py -q`
Expected: PASS (50 Hypothesis examples, value-checked against the reference).

- [ ] **Step 3: Commit**

```bash
git add test/test_repertoire_kary_properties.py
git -c commit.gpgsign=false commit -m "Upgrade k-ary repertoire property test to value-check vs reference"
```

---

## Task 5: Teeth-check, full verification, finish

- [ ] **Step 1: Confirm the oracle has teeth (local, not committed)**

Temporarily break the fix to confirm the sweep catches it. Back up and edit
`pyphi/node.py`: in `Node.__init__`, change `cause_non_inputs = set(range(
cause_factor.ndim - 1)) - self._inputs` back to the old buggy
`cause_non_inputs = set(cause_factor.tpm_indices()) - self._inputs`.

```bash
cp pyphi/node.py /tmp/node_backup.py
```
Make the edit, then run:
`uv run pytest test/test_repertoire_reference.py -q`
Expected: FAIL / error on a sparse-heterogeneous case (the reference rejects the
buggy behavior). Then restore:
```bash
cp /tmp/node_backup.py pyphi/node.py
```
Confirm restored: `git diff --stat pyphi/node.py` shows no changes. Do not
commit anything from this step.

- [ ] **Step 2: Full suite incl. doctests**

Run: `uv run pytest -q` (NO path argument)
Expected: 0 failures.

- [ ] **Step 3: Lint**

Run: `uv run ruff check pyphi test`
Expected: clean. (No pyright step needed — tests only; the pre-commit hook ran
pyright on each commit.)

- [ ] **Step 4: Finish**

Use superpowers:finishing-a-development-branch. If executing inline directly on
`2.0` (no worktree), this collapses to the verification above — report
completion. Otherwise merge `--ff-only` into `2.0`, remove the worktree, delete
the branch. Do not push without explicit consent.

---

## Notes for the implementer

- The reference must stay independent: `test/reference/repertoire.py` imports
  only NumPy. Do not import from `pyphi.core.repertoire_algebra` or
  `pyphi.node`.
- No library source changes anywhere in this plan — only `test/`. Goldens are
  therefore trivially unaffected; do not regenerate any.
- The cause-side cut handling (averaging over severed inputs even in the
  purview) is the subtle part. The self-anchor (Task 1) and the SWAP/cut cases
  in the sweep (Task 2) both exercise it.
