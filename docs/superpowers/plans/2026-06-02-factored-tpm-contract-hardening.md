# Factored-TPM Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the full-dimension factored-TPM contract at `FactoredTPM` construction (reject reduced-dimension factors with a clear error), and document the repertoire broadcast contract.

**Architecture:** Component 1 adds a single dimensionality check to the existing `_validate` in `pyphi/core/tpm/factored.py`, turning a silent acceptance + opaque downstream crash into an actionable `InvalidTPM` at construction. Component 2 is comments only on the two single-node repertoire builders and their callers. No math, no behavior change beyond rejecting inputs that previously crashed.

**Tech Stack:** Python 3.12+, NumPy, pytest. Run everything with `uv run`. Work happens in a worktree off `2.0` (created at execution time via superpowers:using-git-worktrees, per project convention; spec at `docs/superpowers/specs/2026-06-02-factored-tpm-contract-hardening-design.md`).

**Reference:** spec `docs/superpowers/specs/2026-06-02-factored-tpm-contract-hardening-design.md`.

---

## Background the engineer needs

- A `FactoredTPM` stores one factor per substrate unit, each of shape
  `(*alphabet_sizes, k_i)` — every substrate unit's previous-state axis (n of
  them) plus the unit's own output axis. This is the "full-dimension" contract.
- `_validate` (`pyphi/core/tpm/factored.py`, ~line 310) runs at the end of
  every `FactoredTPM.__init__`. It already checks per-axis sizes and that
  factors sum to 1, but does **not** check that each factor has exactly `n + 1`
  axes. A factor with fewer leading axes (a "reduced" factor) is silently
  accepted, then crashes downstream in `_cause_tpm_factored` with an opaque
  broadcasting error.
- `n` inside `_validate` is `factored.n_nodes`, which equals the number of
  factors.
- Tests: the validation home is `test/test_factored_tpm.py` (existing
  rejection tests follow the `with pytest.raises(InvalidTPM, match=...)`
  pattern). Full sweep incl. doctests: `uv run pytest` with NO path argument.
  Goldens: `uv run pytest test/test_golden_regression.py`.
- Byte-identical existing goldens is a hard gate. Component 1 only rejects
  inputs that previously crashed; Component 2 is docs-only. No golden value
  should change — if one does, stop and diagnose.

---

## Task 1: Reject reduced-dimension factors

**Files:**
- Modify: `pyphi/core/tpm/factored.py` (`_validate`, the per-factor loop ~336–348)
- Test: `test/test_factored_tpm.py`
- Create: `changelog.d/factored-tpm-reduced-factor-validation.fix.md`

- [ ] **Step 1: Write the failing test**

Add to `test/test_factored_tpm.py` (it already imports `numpy as np`,
`pytest`, `FactoredTPM`, and `InvalidTPM`):

```python
def test_factored_tpm_rejects_reduced_dimension_factor() -> None:
    # 2-node binary substrate: full-dim factor is (2, 2, 2). A reduced factor
    # (2, 2) spans only one leading axis and is silently accepted today, then
    # crashes downstream. It must be rejected at construction.
    full = np.full((2, 2, 2), 0.5)
    reduced = np.full((2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="leading axes"):
        FactoredTPM(factors=[full, reduced], state_space=((0, 1), (0, 1)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_factored_tpm.py::test_factored_tpm_rejects_reduced_dimension_factor -q`
Expected: FAIL — no exception raised (the FactoredTPM constructs; `DID NOT RAISE`).

- [ ] **Step 3: Implement the dimensionality check**

In `pyphi/core/tpm/factored.py`, in `_validate`, inside the
`for i in range(n):` loop, add the check as the first statement after
`f = factored.factor(i)` (currently line 337), before the `f.shape[-1]` check:

```python
    for i in range(n):
        f = factored.factor(i)
        if f.ndim != n + 1:
            raise exceptions.InvalidTPM(
                f"factor {i} has {f.ndim - 1} leading axes; expected {n} "
                f"(one per substrate unit). Factors must be full-dimension "
                f"(*alphabet_sizes, k_i)."
            )
        if f.shape[-1] != a[i]:
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has {a[i]} labels but "
                f"factor[{i}] last-dim size is {f.shape[-1]}"
            )
        # ... rest of the loop unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_factored_tpm.py::test_factored_tpm_rejects_reduced_dimension_factor -q`
Expected: PASS.

- [ ] **Step 5: Run the full FactoredTPM + repertoire + golden suite (risk gate)**

The stricter check must not reject any *internal* FactoredTPM construction.
Run:
```
uv run pytest test/test_factored_tpm.py test/test_factored_tpm_kary.py \
  test/test_substrate_factored.py test/test_marginalization_kary.py \
  test/test_marginalization_factored.py test/test_golden_regression.py \
  test/test_system.py -q
```
Expected: all PASS, goldens byte-identical. If any internal construction now
raises `InvalidTPM`, **stop and diagnose** — it indicates a real
reduced-factor construction in the codebase to fix at its source, not a reason
to weaken the check. (Reasoned-clear cases: `System.proper_cause_tpm` /
`proper_effect_tpm` `np.squeeze` background axes before rebuilding, leaving
exactly `len(system)` leading axes for `len(system)` factors — full-dimension
relative to the system.)

- [ ] **Step 6: Write the changelog fragment**

```bash
cat > changelog.d/factored-tpm-reduced-factor-validation.fix.md <<'EOF'
`FactoredTPM` now rejects reduced-dimension factors at construction with a
clear `InvalidTPM` error. Each factor must be full-dimension — shape
`(*alphabet_sizes, k_i)`, one leading axis per substrate unit plus the unit's
own output axis. Previously a factor with too few leading axes was silently
accepted and crashed later with an opaque broadcasting error.
EOF
```

- [ ] **Step 7: Commit**

```bash
git add pyphi/core/tpm/factored.py test/test_factored_tpm.py \
  changelog.d/factored-tpm-reduced-factor-validation.fix.md
git -c commit.gpgsign=false commit -m "Reject reduced-dimension FactoredTPM factors at construction"
```

If the commit does not land (a hook reformatted a file), re-`git add` the
listed files and re-run the commit (do not use `--no-verify`, do not `--amend`).

---

## Task 2: Document the repertoire broadcast contract

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (`_single_node_cause_repertoire` ~122–143, `_single_node_effect_repertoire` ~146–170, `_cause_repertoire_inner` ~173–194)

Comments only — no behavior, math, or performance change.

- [ ] **Step 1: Document the cause builder's broadcast dependency**

In `pyphi/core/repertoire_algebra.py`, replace the body of
`_single_node_cause_repertoire` (lines 141–143) — keep the code identical, add
a comment:

```python
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]
    # The result is size 1 on every purview node that is not an input to this
    # mechanism node. It is NOT self-contained canonical: it relies on the
    # ``joint = np.ones(repertoire_shape(...))`` allocation in
    # ``_cause_repertoire_inner`` to broadcast those size-1 axes up to the full
    # purview alphabet. Keeping them size 1 (rather than broadcasting here) is
    # deliberate — the product over mechanism nodes stays cheap.
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm
```

- [ ] **Step 2: Document the effect builder's self-contained canonical output**

In `_single_node_effect_repertoire`, add a comment before the `return`
(line 164), keeping the code identical:

```python
    nonmechanism_inputs = purview_node.inputs - set(condition)
    tpm = tpm.marginalize_out(nonmechanism_inputs)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    # Unlike the cause builder, the effect builder reshapes to canonical here,
    # so its output is self-describingly canonical (this purview node at full
    # alphabet, every other axis size 1) regardless of the caller's allocation.
    return tpm.reshape(
        repertoire_shape(
            cs.substrate.node_indices,
            (purview_node_index,),
            alphabet_sizes=alphabet_sizes,
        )
    ).tpm
```

- [ ] **Step 3: Mark the load-bearing allocation in the cause inner**

In `_cause_repertoire_inner`, add a comment above the `joint = np.ones(...)`
allocation (line 185), keeping the code identical:

```python
    purview_set: frozenset[int] = frozenset(purview)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    # Load-bearing: this canonical-shaped allocation establishes the full
    # purview shape, so per-mechanism-node contributions (which are size 1 on
    # purview nodes they do not constrain — see _single_node_cause_repertoire)
    # broadcast up correctly. Do not replace with a bare product of the
    # per-node contributions.
    joint = np.ones(
        repertoire_shape(
            cs.substrate.node_indices, purview_set, alphabet_sizes=alphabet_sizes
        )
    )
    joint *= functools.reduce(
        np.multiply,
        [_single_node_cause_repertoire(cs, m, purview_set) for m in mechanism],
    )
    return _dist.normalize(joint)
```

- [ ] **Step 4: Verify no behavior change (goldens byte-identical)**

Run: `uv run pytest test/test_golden_regression.py test/test_system.py -q`
Expected: all PASS, byte-identical (comments cannot change behavior; this
confirms no accidental edit).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py
git -c commit.gpgsign=false commit -m "Document repertoire broadcast contract in single-node builders"
```

---

## Task 3: Full verification and finish

- [ ] **Step 1: Full suite incl. doctests**

Run (NO path argument, so doctests in `pyphi/` are collected):
`uv run pytest -q`
Expected: 0 failures.

- [ ] **Step 2: Type and lint checks**

Run: `uv run pyright pyphi` (expect 0 errors) and
`uv run ruff check pyphi test` (expect clean).

- [ ] **Step 3: Finish the branch**

Use superpowers:finishing-a-development-branch. Per project convention: merge
`--ff-only` into `2.0`, verify, remove the worktree, delete the branch. Do not
push without explicit consent.

---

## Notes for the implementer

- Item 3 from the original deferral (`_cause_tpm_factored` argument validation)
  is intentionally **not** in this plan — Task 1 enforces the full-dimension
  contract at construction, and `state` / `node_indices` are already validated
  by the only caller (`System`). Adding redundant guards is out of scope.
- Do not auto-expand or otherwise support reduced/sparse factors. That is a
  separate future project (roadmap P18, native sparse causal inversion).
- The reduced-factor rejection is the only behavior change, and it only affects
  inputs that previously crashed downstream — so no existing golden should
  change. If a golden changes, stop and diagnose.
