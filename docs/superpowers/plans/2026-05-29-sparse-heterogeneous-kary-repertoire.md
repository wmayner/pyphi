# Sparse-Heterogeneous k-ary Repertoire Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the IIT-wide core bug where cause/effect repertoires raise a shape error on networks that are both sparsely connected and heterogeneous-alphabet (k>2 with differing per-node sizes), and validate the fix against the Albantakis 2019 Figure 11 actual-causation example.

**Architecture:** TDD-first. Lock the failure with a minimal regression and a property-test matrix (alphabet x connectivity x direction); fix the shared root cause so both `_single_node_cause_repertoire` and `_single_node_effect_repertoire` emit arrays conformant to the canonical `repertoire_shape`; then add the paper-faithful fig11 acceptance test, a small fast AC guard, a sparse-heterogeneous IIT golden, and a changelog correction.

**Tech Stack:** Python 3.12+, NumPy, pytest, Hypothesis. Run everything with `uv run`. Work happens in the `pyphi-ac-kary` worktree on branch `ac-kary`.

**Reference:** spec at `docs/superpowers/specs/2026-05-29-sparse-heterogeneous-kary-repertoire-design.md`.

---

## Background the engineer needs

- A `Substrate` with heterogeneous alphabets is built via
  `Substrate(marginals=[factors], state_space=(...), cm=...)`. Each factor must be
  **full-dimension**: shape `(*alphabet_sizes, k_i)` — i.e. it spans every node's
  previous-state dimension plus its own output dimension. Reduced-dimension factors
  are silently accepted but crash downstream (out of scope here; deferred to "B").
- `pyphi/core/repertoire_algebra.py` holds the repertoire builders. `repertoire_shape`
  lives in `pyphi/distribution.py:112`. The per-node TPMs are built in
  `pyphi/node.py:42-101` (`Node.__init__`).
- The canonical repertoire shape over a purview puts each purview node at its alphabet
  size and **every** other node (including the mechanism node's own dimension) at size 1.
- Tests: fast lane `uv run pytest test/ -m "not slow"`; full sweep (includes doctests)
  `uv run pytest` with NO path argument; goldens `uv run pytest test/test_golden_regression.py`.

---

## Task 1: Minimal sparse-heterogeneous regression test (cause + effect)

**Files:**
- Create: `test/test_repertoire_sparse_heterogeneous.py`

- [ ] **Step 1: Write the failing test**

```python
"""Regression: cause/effect repertoires on sparse + heterogeneous-alphabet
networks. These currently raise a shape error (the node's own dimension is
not collapsed for k>2 sparse nodes)."""

import numpy as np

from pyphi import Direction, Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System


def _sparse_het_substrate():
    # node0 (k=3), node1 (k=3) -> node2 (k=4). Sparse cm: 0->2, 1->2.
    alph = (3, 3, 4)
    f0 = np.full(alph + (3,), 1 / 3)
    f1 = np.full(alph + (3,), 1 / 3)
    core = np.zeros((3, 3, 4))
    for a in range(3):
        for b in range(3):
            core[a, b, (a + b) % 4] = 1.0
    f2 = np.broadcast_to(core[:, :, None, :], alph + (4,)).copy()
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(
        marginals=[f0, f1, f2],
        state_space=((0, 1, 2), (0, 1, 2), (0, 1, 2, 3)),
        cm=cm,
    )


def test_sparse_het_cause_repertoire_shape():
    sub = _sparse_het_substrate()
    s = System(substrate=sub, state=(0, 0, 0), node_indices=(0, 1, 2))
    r = s.repertoire(Direction.CAUSE, (2,), (0,))  # mechanism k=4, purview k=3
    expected = repertoire_shape(
        s.node_indices, (0,), alphabet_sizes=sub.factored_tpm.alphabet_sizes
    )
    assert r.shape == tuple(expected)
    assert np.isclose(r.sum(), 1.0)


def test_sparse_het_effect_repertoire_shape():
    sub = _sparse_het_substrate()
    s = System(substrate=sub, state=(0, 0, 0), node_indices=(0, 1, 2))
    r = s.repertoire(Direction.EFFECT, (0,), (2,))  # mechanism k=3, purview k=4
    expected = repertoire_shape(
        s.node_indices, (2,), alphabet_sizes=sub.factored_tpm.alphabet_sizes
    )
    assert r.shape == tuple(expected)
    assert np.isclose(r.sum(), 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/test_repertoire_sparse_heterogeneous.py -x -q`
Expected: both FAIL — cause with `non-broadcastable output operand ... (3,1,1) vs (3,1,4)`, effect with `cannot reshape array of size 16 into shape (1,1,4)`.

- [ ] **Step 3: Commit the failing test**

```bash
git add test/test_repertoire_sparse_heterogeneous.py
git -c commit.gpgsign=false commit -m "Add failing sparse-heterogeneous repertoire regression"
```

---

## Task 2: Property-test matrix (alphabet x connectivity x direction)

**Files:**
- Create: `test/test_repertoire_kary_properties.py`

- [ ] **Step 1: Write the failing property test**

```python
"""Property tests: cause/effect repertoires are canonical-shaped and
normalized for arbitrary alphabet sizes and connectivity."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyphi import Direction, Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System


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
        f = rng.uniform(size=alph + (alph[i],))
        f /= f.sum(axis=-1, keepdims=True)
        marginals.append(f)
    state_space = tuple(tuple(range(k)) for k in alph)
    return Substrate(marginals=marginals, state_space=state_space, cm=cm)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    alphabets=st.lists(st.integers(2, 4), min_size=2, max_size=3),
    dense=st.booleans(),
    direction=st.sampled_from([Direction.CAUSE, Direction.EFFECT]),
)
def test_repertoire_canonical_shape_and_normalized(seed, alphabets, dense, direction):
    sub = _random_substrate(seed, alphabets, dense)
    n = len(alphabets)
    state = tuple(0 for _ in range(n))
    s = System(substrate=sub, state=state, node_indices=tuple(range(n)))
    mechanism, purview = (0,), (n - 1,)
    r = s.repertoire(direction, mechanism, purview)
    expected = repertoire_shape(
        s.node_indices, purview, alphabet_sizes=sub.factored_tpm.alphabet_sizes
    )
    assert r.shape == tuple(expected)
    assert np.isclose(r.sum(), 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/test_repertoire_kary_properties.py -x -q`
Expected: FAIL on a sparse + heterogeneous example (Hypothesis will find one quickly).

- [ ] **Step 3: Commit**

```bash
git add test/test_repertoire_kary_properties.py
git -c commit.gpgsign=false commit -m "Add k-ary repertoire property tests (alphabet x connectivity x direction)"
```

---

## Task 3: Diagnose and fix the shared root cause

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (`_single_node_cause_repertoire`:122-143, `_single_node_effect_repertoire`:147-170)
- Possibly modify: `pyphi/node.py:67-90` (`Node.__init__` per-node TPM construction)

**REQUIRED SUB-SKILL:** Use superpowers:systematic-debugging for the trace.

- [ ] **Step 1: Trace the shape flow on the minimal repro**

Using `_sparse_het_substrate()` from Task 1, print the shape at each stage for the
cause direction, `mechanism={node2}` `purview={node0}`:
- `mechanism_node.cause_tpm.tpm.shape` (the node's marginalized cause factor),
- the shape after `[..., mechanism_node.state]`,
- the shape after `marginalize_out(mechanism_node.inputs - purview_set)`,
- the canonical target `repertoire_shape(node_indices, {0}, alphabet_sizes)`.

Identify the dimension that fails to collapse to size 1 (expected: node2's own
previous-state dimension surviving at size 4). Repeat for the effect direction,
where `_single_node_effect_repertoire` already calls `reshape(repertoire_shape(...))`
but receives an array whose element count does not match the target.

- [ ] **Step 2: Implement the fix at the shared root**

Make both single-node builders emit arrays conformant to the canonical
`repertoire_shape`. The cause builder currently returns `.tpm` with no reshape; the
effect builder reshapes but is fed an oversized array. Fix so that, for any alphabet
and connectivity, the node's own dimension and all non-purview dimensions collapse to
size 1 before/at the reshape. Prefer a single shared collapse-and-reshape helper used
by both builders over two divergent code paths. If the misalignment originates in the
marginalized factor produced in `Node.__init__` (the `cause_non_inputs` /
`effect_non_inputs` marginalization not reducing the own dimension for k>2), fix it
there so the node TPM is already canonical and both builders simply reshape.

Constraint: the binary and fully-connected-k-ary paths must produce byte-identical
results (verified in Step 4 and Task 7).

- [ ] **Step 3: Run the regression + property tests**

Run: `uv run pytest test/test_repertoire_sparse_heterogeneous.py test/test_repertoire_kary_properties.py -q`
Expected: PASS. If the property test surfaces a *sibling* shape bug (a different
alphabet/connectivity combination), repeat Steps 1-2 until the matrix is green.

- [ ] **Step 4: Verify no regression on existing repertoire/golden tests**

Run: `uv run pytest test/test_golden_regression.py test/test_system.py test/test_subsystem_surface.py -q`
Expected: PASS, goldens byte-identical.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py pyphi/node.py
git -c commit.gpgsign=false commit -m "Fix cause/effect repertoires for sparse heterogeneous k-ary networks"
```

---

## Task 4: Small fast AC sparse-heterogeneous guard

**Files:**
- Create: `test/test_actual_kary.py`

- [ ] **Step 1: Write the test (value pinned after first run)**

```python
"""Fast guard: actual causation on a small sparse + heterogeneous-alphabet
transition. Exercises the AC path through the k-ary repertoire fix."""

import numpy as np

from pyphi import Substrate, actual


def _sparse_het_ac_substrate():
    # node0 (k=3), node1 (k=3) -> node2 (k=4). node2 = (v0 + v1) mod 4.
    alph = (3, 3, 4)
    f0 = np.full(alph + (3,), 1 / 3)
    f1 = np.full(alph + (3,), 1 / 3)
    core = np.zeros((3, 3, 4))
    for a in range(3):
        for b in range(3):
            core[a, b, (a + b) % 4] = 1.0
    f2 = np.broadcast_to(core[:, :, None, :], alph + (4,)).copy()
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(
        marginals=[f0, f1, f2],
        state_space=((0, 1, 2), (0, 1, 2), (0, 1, 2, 3)),
        cm=cm,
    )


def test_kary_ac_actual_cause_runs():
    sub = _sparse_het_ac_substrate()
    # voters (1,1) -> node2 = (1+1)%4 = 2
    t = actual.Transition(sub, (1, 1, 0), (1, 1, 2), cause_indices=(0, 1), effect_indices=(2,))
    cause = t.find_actual_cause((2,))
    assert cause.alpha >= 0.0
    assert set(cause.purview).issubset({0, 1})
```

- [ ] **Step 2: Run, then pin the exact alpha**

Run: `uv run pytest test/test_actual_kary.py -x -q`
After it passes, print `cause.alpha` once and tighten the assertion to
`assert cause.alpha == pytest.approx(<observed>, abs=1e-6)` so it becomes a true
regression (record the observed value; do not leave the loose `>= 0`).

- [ ] **Step 3: Commit**

```bash
git add test/test_actual_kary.py
git -c commit.gpgsign=false commit -m "Add fast k-ary actual-causation regression guard"
```

---

## Task 5: fig11 acceptance test (paper oracle, slow)

**Files:**
- Modify: `test/test_actual.py:1294-1298` (replace the skip stub)

- [ ] **Step 1: Replace the skip stub with the real test**

```python
@pytest.mark.slow
def test_paper_fig11_three_candidate_alpha():
    """2019 Fig 11: voting with three candidates. The actual cause of {W=1} is
    an undetermined 4-of-5 set of the candidate-0 voters, alpha_c^max = 1.893 bits."""
    import itertools

    nv, kc, kw = 7, 3, 4
    alph = (kc,) * nv + (kw,)
    wcore = np.zeros((kc,) * nv + (kw,))
    for combo in itertools.product(range(kc), repeat=nv):
        counts = [combo.count(c) for c in range(kc)]
        m = max(counts)
        w = (counts.index(m) + 1) if m >= 4 else 0
        wcore[combo + (w,)] = 1.0
    voter = np.full(alph + (kc,), 1.0 / kc)
    wfull = np.broadcast_to(np.expand_dims(wcore, axis=nv), alph + (kw,)).copy()
    cm = np.zeros((nv + 1, nv + 1), dtype=int)
    cm[0:nv, nv] = 1
    sub = Substrate(
        marginals=[voter] * nv + [wfull],
        state_space=tuple(tuple(range(k)) for k in alph),
        cm=cm,
    )
    before = (0, 0, 0, 0, 0, 1, 1, 0)
    after = (0, 0, 0, 0, 0, 1, 1, 1)
    t = actual.Transition(sub, before, after, cause_indices=tuple(range(nv)), effect_indices=(nv,))
    cause = t.find_actual_cause((nv,))
    assert cause.alpha == pytest.approx(1.893, abs=1e-2)
    assert len(cause.purview) == 4
    assert set(cause.purview).issubset({0, 1, 2, 3, 4})  # candidate-0 voters A-E
```

- [ ] **Step 2: Run (time-boxed)**

Run: `uv run pytest "test/test_actual.py::test_paper_fig11_three_candidate_alpha" -x -q`
Expected: PASS with alpha ~= 1.893. If it does not complete within a few minutes,
follow the spec's fallback: keep Task 4 as the committed validation, re-mark this
`@pytest.mark.skip(reason="fig11 tractability: see spec")` with a note, and record the
observed runtime/partial result in the commit message.

- [ ] **Step 3: Commit**

```bash
git add test/test_actual.py
git -c commit.gpgsign=false commit -m "Validate AC k-ary against 2019 fig 11 (alpha_c=1.893)"
```

---

## Task 6: Sparse-heterogeneous IIT golden fixture

**Files:**
- Modify: `test/golden/zoo.py` (add a sparse-heterogeneous fixture alongside the `multivalued_*` ones)

- [ ] **Step 1: Add the fixture**

Add a `(k3, k3) -> k4` sparse substrate builder mirroring `_multivalued_2x3x3`'s style
(deterministic seed, full-dimension factors, explicit `state_space`, sparse `cm`), and
register it as an IIT golden fixture named `sparse_multivalued_k3k3_to_k4` so the golden
regression suite computes and pins its SIA/CES.

- [ ] **Step 2: Generate and commit the golden data**

Run the project's golden-write entry point to produce the fixture's expected JSON, then:

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: PASS (new fixture green; all pre-existing goldens byte-identical).

- [ ] **Step 3: Commit**

```bash
git add test/golden/zoo.py test/data/
git -c commit.gpgsign=false commit -m "Add sparse-heterogeneous k-ary IIT golden fixture"
```

---

## Task 7: Changelog correction + full verification

**Files:**
- Create: `changelog.d/sparse-kary-repertoire.fix.md`

- [ ] **Step 1: Write the changelog fragment**

```bash
cat > changelog.d/sparse-kary-repertoire.fix.md <<'EOF'
Cause and effect repertoires now compute correctly on networks that are both
sparsely connected and heterogeneous-alphabet (multi-valued units with differing
per-node alphabet sizes). Previously such networks raised a shape error in the
repertoire algebra; only binary and fully-connected k-ary networks worked. This
restores the multi-valued support claimed for the SIA/CES pipeline to the sparse
case and unblocks actual causation on k-ary substrates (validated against the
three-candidate voting example of Albantakis et al. 2019, Figure 11).
EOF
```

- [ ] **Step 2: Full verification sweep**

Run (fast lane): `uv run pytest test/ -m "not slow" -q` — expect 0 failures.
Run (slow lane, background): `uv run pytest test/ --slow -q` — expect 0 failures.
Run (doctests, NO path arg): `uv run pytest -q` — expect 0 failures.
Run: `uv run pyright pyphi` and `uv run ruff check pyphi test` — expect clean.

- [ ] **Step 3: Commit**

```bash
git add changelog.d/sparse-kary-repertoire.fix.md
git -c commit.gpgsign=false commit -m "Changelog: sparse heterogeneous k-ary repertoire fix"
```

---

## Task 8: Finish the branch

- [ ] Use superpowers:finishing-a-development-branch to verify tests, then present merge/PR options.
  Per project convention: merge `--ff-only` into `2.0`, verify, remove the worktree, delete the branch. Do not push without explicit consent.

---

## Notes for the implementer

- The fig11 and sparse-het fixtures use **full-dimension** factors (`(*alphabet_sizes, k_i)`).
  Do not pass reduced-dimension factors — that triggers a separate, deferred bug.
- "Byte-identical existing goldens" is a hard gate: if any pre-existing golden changes,
  stop and diagnose — the fix must not touch binary or fully-connected-k-ary behavior.
- Deferred to the follow-up ("Approach B"), do NOT do here: cause/effect shape-handling
  unification beyond what the fix requires; `marginals=` reduced-factor validation;
  `_cause_tpm_factored` hardening.
