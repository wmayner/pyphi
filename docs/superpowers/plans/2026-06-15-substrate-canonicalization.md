# Substrate Canonicalization (P11.95c a+c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-direction φ reporting permutation-invariant by adding a behavior-aware substrate canonicalization module and using it as a Determinism-level tie-break in the IIT 4.0 SIA fallback; un-xfail `test_sia_per_direction_phi_multiset_symmetric`.

**Architecture:** A leaf module `pyphi/automorphism.py` computes substrate automorphisms / canonical form / state-canonicalization by exact node-permutation enumeration (preserving connectivity **and** TPM **and** alphabet sizes). The IIT 4.0 `sia` cascade fallback (currently `chosen_key = outcome.tied_set[0]`) is replaced by a canonical-state-keyed minimum, which only fires when the existing cascade is unresolved (reducible systems).

**Tech Stack:** Python 3.12+, numpy, pytest, Hypothesis. No new runtime dependency (pynauty rejected — see spec §6).

**Spec:** `docs/superpowers/specs/2026-06-15-substrate-canonicalization-design.md`

---

## Verified facts (from the brainstorm investigation — do not re-derive)

- `substrate.tpm.to_joint()` returns an ndarray of shape `(*alphabet_sizes, n_nodes, max_alphabet)`.
- The node-relabel primitive (π = "destination `i` ← source `π[i]`"), verified to map `and_xor_substrate()` to `xor_and_substrate()` under π=`(1,0)` and to round-trip under inverse for heterogeneous (k-ary) alphabets:
  ```python
  def _relabel_joint(arr, perm):
      n = len(perm)
      return arr.transpose(tuple(perm) + (n, n + 1))[..., list(perm), :]
  ```
- A permutation moving a node of one alphabet onto a node of a different alphabet changes the array shape, so candidate permutations **must** be pruned by `alphabet_sizes[perm[i]] == alphabet_sizes[i]` (and by connectivity) **before** any array comparison.
- `Substrate.__eq__` compares `(factored_tpm, cm)`. `Substrate` is hashable.
- A relabeled `Substrate` is built via `Substrate(tpm=<relabeled joint ndarray>, cm=<permuted cm>, state_space=<permuted state_space>)` — the explicit-alphabet joint shape is accepted by the constructor (verified: the rebuilt substrate `== xor_and_substrate()`).
- `System` exposes `.substrate` (a `Substrate`). The IIT 4.0 cascade keys `per_pair_sias` by `(cause_state | None, effect_state | None)`; `outcome.tied_set` holds those keys.
- `canonical_state(and_xor, (0,1)) == canonical_state(xor_and, (1,0)) == (1,0)`; the tie-break key selects the `φ_c=0.5` pair on both substrates, yielding multiset `{0.5, 0}` on both.

---

## File Structure

- **Create** `pyphi/automorphism.py` — the canonicalization sidecar (leaf utility).
- **Create** `test/test_automorphism.py` — unit + Hypothesis property tests for the sidecar.
- **Modify** `pyphi/formalism/iit4/__init__.py` — replace the `chosen_key = outcome.tied_set[0]` fallback with a canonical-state tie-break (one site).
- **Modify** `test/test_invariants.py` — remove the `@pytest.mark.xfail` from `test_sia_per_direction_phi_multiset_symmetric`.
- **Create** `changelog.d/substrate-canonicalization.feature.md`.
- **Modify** `ROADMAP.md` — flip the P11.95c (a)+(c) dashboard row to landed.

---

## Task 1: Relabel primitive + candidate enumeration

**Files:**
- Create: `pyphi/automorphism.py`
- Test: `test/test_automorphism.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_automorphism.py
import numpy as np
import pytest

from pyphi import automorphism as auto
from test import example_substrates as es


def test_relabel_joint_maps_and_xor_to_xor_and():
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    relabeled = auto._relabel_joint(s_ax.tpm.to_joint(), (1, 0))
    assert np.array_equal(relabeled, s_xa.tpm.to_joint())


def test_relabel_joint_identity_is_noop():
    s = es.and_xor_substrate()
    arr = s.tpm.to_joint()
    assert np.array_equal(auto._relabel_joint(arr, (0, 1)), arr)


def test_candidate_perms_prunes_by_alphabet_and_cm():
    # and_xor: fully connected, uniform alphabet -> both perms are candidates
    s = es.and_xor_substrate()
    assert set(auto._candidate_perms(s)) == {(0, 1), (1, 0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/substrate-canonicalization && uv run pytest test/test_automorphism.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.automorphism'`.

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/automorphism.py
"""Behavior-aware substrate canonicalization.

A substrate's identity is its connectivity, its per-node TPM, and its
per-node alphabet sizes. A node permutation is a substrate **automorphism**
only when it preserves all three — so a node implementing one mechanism is
never identified with a node implementing a different one, even when their
wiring is identical.

Canonicalization is exact: the automorphism group and canonical form are
found by enumerating node permutations. This is tractable because Φ is
``O(2**n)``, so substrates on which it is computed have few nodes; the
asymptotic regime where graph-isomorphism libraries (e.g. nauty) would help
is one in which Φ itself is intractable.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyphi.substrate import Substrate

# Decimal places used when byte-keying TPM arrays for cross-substrate
# comparison, to make canonical-form equality robust to float round-off.
_ROUND = 12


def _relabel_joint(arr: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Relabel a joint TPM array under ``perm`` (destination ``i`` <- source
    ``perm[i]``).

    ``arr`` has shape ``(*alphabet_sizes, n_nodes, max_alphabet)``: the first
    ``n`` axes are input-state axes, axis ``-2`` is the output-node axis, and
    axis ``-1`` is the per-node next-state distribution (which travels with its
    node). Permuting the input axes and reindexing the node axis relabels the
    nodes.
    """
    n = len(perm)
    return arr.transpose(tuple(perm) + (n, n + 1))[..., list(perm), :]


def _candidate_perms(substrate: "Substrate") -> tuple[tuple[int, ...], ...]:
    """Node permutations preserving connectivity and alphabet sizes.

    These are the only permutations that can be substrate automorphisms or
    isomorphisms; pruning here also avoids comparing arrays of mismatched
    shape (a permutation across differing alphabets reshapes the TPM).
    """
    cm = np.asarray(substrate.cm)
    alphabet = substrate.tpm.alphabet_sizes
    n = len(alphabet)
    out = []
    for perm in permutations(range(n)):
        if any(alphabet[perm[i]] != alphabet[i] for i in range(n)):
            continue
        if not np.array_equal(cm[np.ix_(perm, perm)], cm):
            continue
        out.append(perm)
    return tuple(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/automorphism.py test/test_automorphism.py
git commit -m "feat(automorphism): node-relabel primitive and candidate pruning"
```

---

## Task 2: `substrate_automorphisms`

**Files:**
- Modify: `pyphi/automorphism.py`
- Test: `test/test_automorphism.py`

- [ ] **Step 1: Write the failing test**

```python
def test_automorphisms_identity_always_present():
    for s in (es.and_xor_substrate(), es.xor_and_substrate()):
        autos = auto.substrate_automorphisms(s)
        assert tuple(range(len(s.tpm.alphabet_sizes))) in autos


def test_automorphisms_distinct_gates_have_only_identity():
    # Fully connected but AND != XOR: no nontrivial automorphism.
    s = es.and_xor_substrate()
    assert auto.substrate_automorphisms(s) == ((0, 1),)


def test_automorphisms_preserve_tpm():
    s = es.and_xor_substrate()
    arr = s.tpm.to_joint()
    for perm in auto.substrate_automorphisms(s):
        assert np.array_equal(auto._relabel_joint(arr, perm), arr)


def test_automorphisms_recover_known_symmetries():
    # Three interchangeable nodes -> full symmetric group S_3 (6 perms).
    triple = auto.substrate_automorphisms(es.symmetric_triple_substrate())
    assert len(triple) == 6
    # Two identical AND-XOR blocks -> identity + the block swap.
    dual = set(auto.substrate_automorphisms(es.dual_and_xor_substrate()))
    assert dual == {(0, 1, 2, 3), (2, 3, 0, 1)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: FAIL — `AttributeError: module 'pyphi.automorphism' has no attribute 'substrate_automorphisms'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/automorphism.py`:

```python
def substrate_automorphisms(
    substrate: "Substrate",
) -> tuple[tuple[int, ...], ...]:
    """All node permutations preserving connectivity, TPM, and alphabet sizes.

    Always contains the identity permutation.
    """
    arr = substrate.tpm.to_joint()
    return tuple(
        perm
        for perm in _candidate_perms(substrate)
        if np.array_equal(_relabel_joint(arr, perm), arr)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/automorphism.py test/test_automorphism.py
git commit -m "feat(automorphism): substrate_automorphisms"
```

---

## Task 3: Canonical form + isomorphism

**Files:**
- Modify: `pyphi/automorphism.py`
- Test: `test/test_automorphism.py`

- [ ] **Step 1: Write the failing test**

```python
def test_canonical_form_invariant_under_relabeling():
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    canon_ax, _ = auto.substrate_canonical_form(s_ax)
    canon_xa, _ = auto.substrate_canonical_form(s_xa)
    assert canon_ax == canon_xa


def test_canonical_permutation_maps_to_canonical_form():
    s = es.xor_and_substrate()
    canon, perm = auto.substrate_canonical_form(s)
    relabeled = auto._relabel_joint(s.tpm.to_joint(), perm)
    assert np.array_equal(relabeled, canon.tpm.to_joint())


def test_isomorphic_pair_and_nonisomorphic_pair():
    assert auto.are_substrates_isomorphic(
        es.and_xor_substrate(), es.xor_and_substrate()
    )
    assert auto.are_substrates_isomorphic(
        es.and_xor_substrate(), es.and_xor_substrate()
    )
    # Different node counts -> not isomorphic (2 nodes vs 3 nodes).
    assert not auto.are_substrates_isomorphic(
        es.and_xor_substrate(), es.symmetric_triple_substrate()
    )


def test_isomorphism_symmetric():
    a, b = es.and_xor_substrate(), es.xor_and_substrate()
    assert auto.are_substrates_isomorphic(a, b) == auto.are_substrates_isomorphic(b, a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: FAIL — `AttributeError: ... 'substrate_canonical_form'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/automorphism.py`:

```python
def _serialization(substrate: "Substrate", perm: tuple[int, ...]) -> tuple:
    """A relabeling-applied, byte-comparable key for ``substrate`` under
    ``perm``. Rounding makes cross-substrate equality robust to float
    round-off; ``+ 0.0`` normalizes ``-0.0`` to ``0.0``."""
    cm = np.asarray(substrate.cm)
    alphabet = substrate.tpm.alphabet_sizes
    n = len(alphabet)
    arr = _relabel_joint(substrate.tpm.to_joint(), perm)
    cm_p = np.ascontiguousarray(cm[np.ix_(perm, perm)])
    arr_p = np.ascontiguousarray(np.round(arr, _ROUND)) + 0.0
    alpha_p = tuple(alphabet[perm[i]] for i in range(n))
    return (alpha_p, cm_p.tobytes(), arr_p.tobytes())


@lru_cache(maxsize=None)
def _canonical(substrate: "Substrate") -> tuple[tuple, tuple[tuple[int, ...], ...]]:
    """Return ``(canonical_key, achievers)`` where ``canonical_key`` is the
    lexicographically smallest serialization over candidate permutations and
    ``achievers`` is every permutation attaining it (the set mapping
    ``substrate`` to its canonical form)."""
    best_key = None
    achievers: list[tuple[int, ...]] = []
    for perm in _candidate_perms(substrate):
        key = _serialization(substrate, perm)
        if best_key is None or key < best_key:
            best_key, achievers = key, [perm]
        elif key == best_key:
            achievers.append(perm)
    return best_key, tuple(achievers)


def substrate_canonical_form(
    substrate: "Substrate",
) -> tuple["Substrate", tuple[int, ...]]:
    """Return ``(canonical_substrate, canonical_permutation)``.

    ``canonical_substrate`` is the lexicographically smallest relabeling of
    ``substrate``; ``canonical_permutation`` is the smallest permutation
    attaining it (unique by construction)."""
    from pyphi.substrate import Substrate

    _, achievers = _canonical(substrate)
    perm = min(achievers)
    arr = _relabel_joint(substrate.tpm.to_joint(), perm)
    cm = np.asarray(substrate.cm)[np.ix_(perm, perm)]
    state_space = substrate.state_space
    permuted_state_space = tuple(state_space[perm[i]] for i in range(len(perm)))
    canonical = Substrate(tpm=arr, cm=cm, state_space=permuted_state_space)
    return canonical, perm


def are_substrates_isomorphic(s1: "Substrate", s2: "Substrate") -> bool:
    """True iff some node permutation maps ``s1``'s connectivity, TPM, and
    alphabet sizes onto ``s2``'s."""
    if s1.tpm.alphabet_sizes != s2.tpm.alphabet_sizes and sorted(
        s1.tpm.alphabet_sizes
    ) != sorted(s2.tpm.alphabet_sizes):
        return False
    return _canonical(s1)[0] == _canonical(s2)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/automorphism.py test/test_automorphism.py
git commit -m "feat(automorphism): canonical form and isomorphism test"
```

---

## Task 4: `canonical_state` (the tie-break key) + Hypothesis property

**Files:**
- Modify: `pyphi/automorphism.py`
- Test: `test/test_automorphism.py`

- [ ] **Step 1: Write the failing test**

```python
from hypothesis import given, settings
from hypothesis import strategies as st


def test_canonical_state_linchpin():
    # The two permuted substrates' corresponding states must canonicalize equal.
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    assert auto.canonical_state(s_ax, (0, 1)) == auto.canonical_state(s_xa, (1, 0))


def test_canonical_state_idempotent_on_canonical_substrate():
    s = es.and_xor_substrate()
    canon, perm = auto.substrate_canonical_form(s)
    state = (1, 0)
    canon_state = tuple(state[perm[i]] for i in range(len(perm)))
    assert auto.canonical_state(canon, canon_state) == auto.canonical_state(s, state)


@settings(max_examples=50)
@given(bits=st.lists(st.integers(0, 1), min_size=2, max_size=2))
def test_canonical_state_orbit_invariant(bits):
    # and_xor and xor_and are related by sigma=(1,0): state s on and_xor
    # corresponds to s'[i]=s[sigma[i]] on xor_and.
    s_ax = es.and_xor_substrate()
    s_xa = es.xor_and_substrate()
    sigma = (1, 0)
    s = tuple(bits)
    s_prime = tuple(s[sigma[i]] for i in range(2))
    assert auto.canonical_state(s_ax, s) == auto.canonical_state(s_xa, s_prime)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: FAIL — `AttributeError: ... 'canonical_state'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/automorphism.py`:

```python
def canonical_state(
    substrate: "Substrate", state: tuple[int, ...]
) -> tuple[int, ...]:
    """Map ``state`` into canonical coordinates, reduced over the automorphism
    orbit.

    For substrates related by a node permutation, corresponding states'
    canonical-coordinate images agree only up to an automorphism, so the
    permutation-invariant identity of ``state`` is the lexicographically
    smallest image over every permutation that carries ``substrate`` to its
    canonical form."""
    _, achievers = _canonical(substrate)
    return min(
        tuple(state[perm[i]] for i in range(len(perm))) for perm in achievers
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_automorphism.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add pyphi/automorphism.py test/test_automorphism.py
git commit -m "feat(automorphism): canonical_state with orbit reduction"
```

---

## Task 5: Wire the canonical tie-break into the IIT 4.0 SIA fallback

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (the `chosen_key = outcome.tied_set[0]` site, ~line 805-809)
- Modify: `test/test_invariants.py` (un-xfail target test, ~line 399-407)

- [ ] **Step 1: Make the target test the failing test**

Remove the `@pytest.mark.xfail(...)` decorator (the 8-line block at `test/test_invariants.py:399-406`) immediately above `def test_sia_per_direction_phi_multiset_symmetric`. Leave the test body unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_invariants.py::TestPermutationSymmetry::test_sia_per_direction_phi_multiset_symmetric -q`
Expected: FAIL — assertion error, multiset `{0.5, 0}` != `{0, 0}` (the current label-dependent behavior).

- [ ] **Step 3: Implement the canonical tie-break**

In `pyphi/formalism/iit4/__init__.py`, the cascade fallback currently reads:

```python
        chosen_key = outcome.resolved
        if chosen_key is None:
            assert outcome.tied_set, "cascade outcome has neither winner nor ties"
            chosen_key = outcome.tied_set[0]
        mip_sia = per_pair_sias[chosen_key]
```

Replace the `chosen_key = outcome.tied_set[0]` line with a canonical-state-keyed
minimum over the tied keys (each key is `(cause_state | None, effect_state | None)`):

```python
        chosen_key = outcome.resolved
        if chosen_key is None:
            assert outcome.tied_set, "cascade outcome has neither winner nor ties"
            chosen_key = min(
                outcome.tied_set,
                key=lambda key: _canonical_tie_break_key(system.substrate, key),
            )
        mip_sia = per_pair_sias[chosen_key]
```

Add this module-level helper near the other private helpers in the same file
(e.g. just after `_build_untied_system_state`):

```python
def _canonical_tie_break_key(substrate, key):
    """Permutation-invariant sort key for a tied ``(cause_state, effect_state)``
    pair, so per-direction φ reporting is deterministic up to relabeling. A
    ``None`` direction sorts first via the empty tuple."""
    from pyphi.automorphism import canonical_state

    cause_state, effect_state = key
    return (
        canonical_state(substrate, cause_state) if cause_state is not None else (),
        canonical_state(substrate, effect_state) if effect_state is not None else (),
    )
```

Use a deferred (function-local) import of `pyphi.automorphism` to avoid any
import-order coupling in the formalism package.

- [ ] **Step 4: Run the target test to verify it passes**

Run: `uv run pytest test/test_invariants.py::TestPermutationSymmetry -q`
Expected: PASS — all `TestPermutationSymmetry` tests pass, including the un-xfailed one.

- [ ] **Step 5: Run the full invariants + cross-formalism files**

Run: `uv run pytest test/test_invariants.py test/test_cross_formalism_invariants.py -q`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/test_invariants.py
git commit -m "fix(iit4): permutation-invariant per-direction phi via canonical tie-break

When phi_s ties at zero (reducible systems), the SIA state-tie cascade
previously fell through to enumeration order, so per-direction phi for
permutation-related substrates depended on node labeling. Break the
residual tie on canonical_state so the choice is deterministic up to
relabeling. Un-xfails test_sia_per_direction_phi_multiset_symmetric."
```

---

## Task 6: Full regression sweep + reducible-golden drift check

**Files:** none (verification task)

- [ ] **Step 1: Run the golden regression suite**

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: PASS. If any golden fails, it can only be a **reducible** system whose
*reported specified state* changed (φ values are unchanged). Do **not**
regenerate reflexively: capture the before/after state, confirm both states are
in the same tie set and the φ multiset is unchanged, and report it for review
before any regeneration.

- [ ] **Step 2: Run the full suite WITH doctests (no path argument)**

Run: `uv run pytest -q -p no:cacheprovider`
Expected: PASS. (Per project docs, the bare `uv run pytest` uses `testpaths` and
collects `pyphi/` doctests; a path argument would skip them. The slow Hypothesis
lane may take several minutes — acceptable for the completion gate.)

- [ ] **Step 3: Run pyright on the new + changed modules**

Run: `uv run pyright pyphi/automorphism.py pyphi/formalism/iit4/__init__.py`
Expected: no new errors.

- [ ] **Step 4: Commit (only if Step 1-3 required code fixes)**

```bash
git add -A
git commit -m "test: fixes from full regression sweep for substrate canonicalization"
```

---

## Task 7: Changelog + ROADMAP dashboard

**Files:**
- Create: `changelog.d/substrate-canonicalization.feature.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write the changelog fragment**

```bash
cat > changelog.d/substrate-canonicalization.feature.md <<'EOF'
Added `pyphi.automorphism` (`substrate_automorphisms`, `substrate_canonical_form`,
`are_substrates_isomorphic`, `canonical_state`): behavior-aware substrate
canonicalization by exact node-permutation enumeration (preserving connectivity,
TPM, and alphabet sizes). Per-direction φ reporting for permutation-related
substrates is now deterministic up to relabeling — the IIT 4.0 SIA tie-break
among reducible (`φ_s = 0`) states uses a canonical key instead of enumeration
order.
EOF
```

- [ ] **Step 2: Update the ROADMAP dashboard**

In `ROADMAP.md`, change the `P11.95c (a)+(c)` dashboard row (in the "Remaining 2.0
work" table) from `⬜ open` to `✅ landed`, and add it to the "✅ Landed" list with
a one-line note: exact-enumeration engine (pynauty rejected — it canonicalizes
wiring, not behavior, and Φ caps n below where n! matters); the per-direction
asymmetry was a reducible-system-only tie-break determinism issue, fixed via a
`canonical_state` Determinism tie-break. Also update the Wave-2 detail bullet for
P11.95c to reflect that the per-direction asymmetry question is resolved (a
tie-break determinism issue on reducible systems, not a bug in φ).

- [ ] **Step 3: Commit**

```bash
git add changelog.d/substrate-canonicalization.feature.md ROADMAP.md
git commit -m "docs: changelog + ROADMAP for substrate canonicalization (P11.95c a+c)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §4 API (4 functions) → Tasks 1–4. §5 tie-break → Task 5. §7
  testing (un-xfail, sidecar property tests, canonical-state orbit invariance,
  full `uv run pytest`) → Tasks 4–6. §8 risk (golden drift) → Task 6. §9
  changelog, §10 ROADMAP → Task 7. No gaps.
- **Placeholder scan:** none. All fixtures are confirmed present in
  `test/example_substrates.py` (`and_xor`, `xor_and`, `dual_and_xor`,
  `symmetric_triple`) and their automorphism groups verified during planning.
- **Type/name consistency:** `_relabel_joint`, `_candidate_perms`, `_canonical`,
  `_serialization`, `substrate_automorphisms`, `substrate_canonical_form`,
  `are_substrates_isomorphic`, `canonical_state`, `_canonical_tie_break_key`
  used consistently across tasks; permutation convention ("destination i ←
  source perm[i]") is uniform.
