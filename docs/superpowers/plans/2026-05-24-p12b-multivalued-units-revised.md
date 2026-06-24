# P12b — Multi-Valued Units Implementation Plan (Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land native k-ary IIT analysis (cause inversion math, hot-path cutover, user-facing multi-valued substrate API, AC parallel cutover) under a unified `FactoredTPM`-based cause/effect representation, preserving byte-identical binary goldens at every commit boundary.

**Architecture:** Both `cause_tpm` and `effect_tpm` return `FactoredTPM` — the cause posterior factors per output system unit under IIT 4.0 Eq. 4's conditional-independence structure, identical in type to the effect TPM under Eq. 2. `CausePosterior` is retired. The migration introduces a `factor(i)` accessor that works against both legacy SBN storage (synthesized `[1-x, x]`) and the new explicit-alphabet storage, then migrates consumers one site per commit, then switches the producer's internal representation. The k-ary cause math is corrected to include background weighting (`Σ_{w̄}` and `pr_bg / norm` per Eq. 4) so it produces byte-identical results to the legacy binary path on binary inputs. State_space metadata lives canonically on `FactoredTPM` with Substrate delegation. Measure registry gains a declarative `supports_alphabet` callable; EMD raises for k>2.

**Supersedes:** Tasks 6-18 of `docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`. Tasks 1-5 of that plan already landed and are not re-executed.

**Authoritative design:** `docs/superpowers/specs/2026-05-24-p12b-unification-amendment.md` (commit `5674cb92`). The amendment supersedes specific sections of `docs/superpowers/specs/2026-05-24-p12b-multivalued-units-design.md` (commit `8d0387c9`); read the amendment first.

**Tech Stack:** Python 3.12+, NumPy, Hypothesis (property tests), pytest, pyright, ruff. Pre-commit hooks gate every commit. `git -c commit.gpgsign=false` for signing this session.

---

## Spec reference

Authoritative spec at `docs/superpowers/specs/2026-05-24-p12b-unification-amendment.md` (committed at `5674cb92`). Original spec at `docs/superpowers/specs/2026-05-24-p12b-multivalued-units-design.md` (committed at `8d0387c9`); the amendment overrides §3.2 (type hierarchy), §3.3 (cause-TPM math), and §7.11 (goldens gate). All remaining sections of the original spec stand.

Supporting audits:
- `docs/superpowers/audits/p12b-sbn-consumer-catalogue.md` (commit `e4985a2d`): 21 consumer sites; 10 category B (real SBN-semantics dependency).
- `docs/superpowers/audits/p12b-unification-math-analysis.md` (commit `9a353a61`): per-consumer invariance categorization; empirical proof (76/76 cases on `basic_system`) that the unified algorithm preserves cause repertoires byte-identically.

---

## Branch state baseline & pre-flight

**Worktree:** `../pyphi-p12b` (separate from main `../pyphi` repo). The main repo MUST NOT be touched.

**Branch:** `feature/p12b-factored-kary`, head `5674cb92`.

**Already landed (tasks 1-5 of the original plan):**

| Commit | Status under amendment |
|---|---|
| `5727f01b` — Audit cause-output shape contract | Kept as historical record; conclusions partially superseded by the amendment. |
| `55b135fb` — Extract `JointDistribution` base class | Kept. `JointDistribution` remains useful as a base for `JointTPM`'s joint-tensor storage. |
| `ffecc9d6` — Add `CausePosterior` as `JointDistribution` sibling | Superseded; this plan retires the class. |
| `e85fa73f` — `cause_tpm` returns `CausePosterior` | Superseded; this plan changes the return type to `FactoredTPM`. |
| `4c00f64c` — Native k-ary cause path; dispatcher routes binary vs k-ary | Math is incomplete (no background weighting); this plan's first substantive task fixes it. |

**Pre-flight check (run before Task 6):**

```bash
cd ../pyphi-p12b
git log -1 --oneline                                # expect: 5674cb92 ...
git status --short                                  # clean or only untracked
uv run pytest test/test_golden_regression.py -q     # 23/23 binary goldens pass
uv run pyright pyphi 2>&1 | tail -3                 # 0 errors / 5 baseline warnings
```

If any of these don't match, surface to the user before proceeding.

**Working-tree hygiene:** the worktree may have unrelated unstaged churn (e.g., `uv.lock`, scratch notes). DO NOT stage anything not enumerated in each task's `git add` step. Before every commit, run `git diff --cached --stat` to confirm only the intended files are staged.

---

## File responsibilities map

**New files (created during this re-plan):**

```
test/test_marginalization_kary.py        # k-ary cause math + binary-equivalence property
test/test_substrate_state_space.py       # state_space construction + delegation
test/test_substrate_multivalued.py       # end-to-end k>2 substrate + SIA/AC smoke
test/test_measure_alphabet_support.py    # measure metadata + dispatcher guard
test/data/golden/v1/multivalued_k3_tiny.{json,npz}
test/data/golden/v1/multivalued_2x3x3.{json,npz}
test/data/golden/v1/multivalued_p53_mdm2.{json,npz}   # conditional on reproducibility
test/golden/generate_p12b_fixtures.py    # generation script
changelog.d/p12b-multivalued.feature.md
```

**Files deleted (during retirement tasks):**

```
pyphi/core/tpm/cause_posterior.py
test/test_cause_posterior.py
```

**Modified files (touched by one or more tasks):**

```
pyphi/core/tpm/marginalization.py        # math fix; FactoredTPM return type; legacy retirement
pyphi/core/tpm/factored.py               # factor(i) accessor extension; state_space field
pyphi/core/tpm/joint_distribution.py     # tpm_indices() semantics fix
pyphi/core/tpm/__init__.py               # drop CausePosterior export
pyphi/core/repertoire_algebra.py         # _single_node_*_repertoire migration
pyphi/node.py                            # Node.__init__ migration; cause_factor accessor
pyphi/system.py                          # cause_tpm return type; proper_cause_tpm redesign
pyphi/actual.py                          # TransitionSystem parallel cutover
pyphi/substrate.py                       # state_space, alphabet=, joint_tpm unification
pyphi/tpm.py                             # backward_tpm retirement (final task)
pyphi/measures/distribution.py           # supports_alphabet metadata
pyphi/__init__.py                        # drop CausePosterior export
pyphi/jsonify.py                         # registry update for FactoredTPM cause path
docs/conventions.rst                     # fix pre-existing broken doctest
```

---

## TDD pattern (applies to every code-changing task)

1. Write the failing test first (or document the regression bar if the change is mechanical).
2. Run it to confirm it fails for the right reason.
3. Implement the minimal code to pass.
4. Run the test to confirm it passes.
5. **Goldens gate:** `uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3` shows `23 passed` (or whatever count is current). If the count changes or any drift is reported, STOP and diagnose. Drift beyond `atol=1e-10` indicates a bug in the change, not a goldens issue — regenerating goldens requires per-instance user approval.
6. Run a wider check (full suite without path argument, for doctest scope).
7. Run pyright + ruff on touched files.
8. Commit.

For mechanical refactor steps (renames, signature changes, marker cleanup) where TDD is awkward: make the change → run pyright + ruff on touched files → run surrounding tests + goldens → commit.

**Every commit must pass pre-commit hooks** (ruff + ruff format + pyright + towncrier-check). Never `--no-verify`. Diagnose failures via `uv run ruff check <file>` / `uv run pyright <file>` directly.

**gpgsign:** use `git -c commit.gpgsign=false commit -m "..."`. If 1Password agent error persists despite the bypass, STOP and surface to controller.

**Staging discipline:** targeted `git add <file>` only. Before commit: `git diff --cached --stat` to confirm only intended files staged. After commit: `git show --stat HEAD` to confirm.

**Doctest scope reminder:** the full-suite verification MUST run `uv run pytest` WITHOUT a path argument at every commit boundary. Bare-path invocations (`pytest test/`) skip `pyphi/` source doctests.

---

## Strengthened goldens gate (from the amendment)

At every commit boundary, **cause repertoires** (the end-to-end output of `cause_repertoire(cs, mech, purv)` over all subsystem × mechanism × purview combinations) MUST remain byte-identical within `atol=1e-10`. Intermediate array layouts (e.g., legacy SBN `(*α, n)` vs. explicit alphabet `(*α, n, k)`) MAY change without violating the gate, provided end-to-end cause repertoires are preserved.

Drift beyond `atol=1e-10` is a bug in the change, not a reason to regenerate goldens. Regeneration requires (a) an explicit algebraic derivation showing the legacy values were a quirk, (b) per-instance user approval before commit, (c) documentation in the changelog with the derivation cited.

---

## Phase B-revised — Math fix + accessor introduction (Tasks 6-8)

### Task 6: Fix `_cause_tpm_factored_kary` math to include background weighting

**Goal:** Add the `Σ_{w̄}` background marginalization and `pr_bg / norm` weighting to `_cause_tpm_factored_kary` per IIT 4.0 Eq. 4. After the fix, the function produces a `FactoredTPM` of shape `(*α, n_system_units, k_per_unit)` whose factors are byte-identical to what `_cause_tpm_factored_binary` produces on binary inputs (modulo trailing-axis representation, addressed in Task 14).

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py`
- Test: `test/test_marginalization_kary.py` (new)

- [ ] **Step 6.1: Write the failing tests.**

Create `test/test_marginalization_kary.py`:

```python
"""K-ary cause inversion math: correctness + binary equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import (
    _cause_tpm_factored_binary,
    _cause_tpm_factored_kary,
    cause_tpm,
)


def _random_kary_factor(n_nodes: int, alphabet: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (alphabet,) * n_nodes + (alphabet,)
    arr = rng.uniform(size=shape)
    return arr / arr.sum(axis=-1, keepdims=True)


def _random_binary_factor(n_nodes: int, seed: int) -> np.ndarray:
    return _random_kary_factor(n_nodes, 2, seed)


def test_cause_kary_returns_factored_tpm() -> None:
    """The native k-ary path returns a FactoredTPM."""
    factors = [_random_kary_factor(2, 3, seed=10 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_tpm_factored_kary(factored, state=(0, 0), node_indices=(0, 1))
    assert isinstance(result, FactoredTPM)


def test_cause_kary_per_factor_sums_to_one() -> None:
    """Each per-output-unit factor of the returned FactoredTPM is a
    probability distribution over its trailing alphabet axis."""
    factors = [_random_kary_factor(2, 3, seed=20 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_tpm_factored_kary(factored, state=(1, 2), node_indices=(0, 1))
    for i in range(result.n_nodes):
        f = result.factor(i)
        assert f.shape[-1] == 3
        np.testing.assert_allclose(f.sum(axis=-1), 1.0, atol=1e-10)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_cause_kary_binary_equivalent_to_binary_path(seed: int) -> None:
    """On binary inputs the k-ary path and the binary path produce
    equivalent factors (within atol=1e-10) per output unit."""
    factors = [_random_binary_factor(3, seed=seed + i) for i in range(3)]
    factored = FactoredTPM(factors=factors)
    state = (0, 1, 0)
    node_indices = (0, 1, 2)
    kary = _cause_tpm_factored_kary(factored, state, node_indices)
    binary = _cause_tpm_factored_binary(factored, state, node_indices)
    for i in range(factored.n_nodes):
        np.testing.assert_allclose(
            kary.factor(i), binary.factor(i), atol=1e-10,
            err_msg=f"factor {i} disagrees",
        )


def test_cause_kary_subset_system_uses_background_weighting() -> None:
    """When system_indices is a proper subset of the substrate, the
    posterior factor for system unit i depends on the background state
    via pr_bg / norm. Verify against a hand-built 2-node binary case."""
    # 2-node binary: node 0 is the mechanism (system), node 1 is background.
    f0 = np.array([[[0.8, 0.2], [0.5, 0.5]],
                   [[0.1, 0.9], [0.4, 0.6]]], dtype=np.float64)
    f1 = np.array([[[0.7, 0.3], [0.2, 0.8]],
                   [[0.6, 0.4], [0.3, 0.7]]], dtype=np.float64)
    factored = FactoredTPM(factors=[f0, f1])
    state = (1, 0)
    binary = _cause_tpm_factored_binary(factored, state, node_indices=(0,))
    kary = _cause_tpm_factored_kary(factored, state, node_indices=(0,))
    np.testing.assert_allclose(kary.factor(0), binary.factor(0), atol=1e-10)


def test_cause_unreachable_state_raises() -> None:
    from pyphi.exceptions import StateUnreachableBackwardsError
    factors = [np.zeros((2, 2, 2)) for _ in range(2)]
    for f in factors:
        f[..., 0] = 1.0  # always outputs 0
    factored = FactoredTPM(factors=factors)
    with pytest.raises(StateUnreachableBackwardsError):
        _cause_tpm_factored_kary(factored, state=(1, 1), node_indices=(0, 1))
```

- [ ] **Step 6.2: Run failing tests.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v 2>&1 | tail -20
```

Expected: most tests fail with `isinstance(..., FactoredTPM)` or binary-equivalence assertion errors, because the current implementation returns `CausePosterior` and omits background weighting.

- [ ] **Step 6.3: Implement the fix.**

Replace `_cause_tpm_factored_kary` in `pyphi/core/tpm/marginalization.py`:

```python
def _cause_tpm_factored_kary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Native k-ary cause TPM via per-output-unit Bayesian inversion with
    background weighting.

    Implements IIT 4.0 Eq. 4. For each system unit ``i`` and each possible
    output value ``s_i`` the returned factor stores

        factor_i(s_t)[s_i] = Σ_{w_t} P(s_i | s_t, w_t) · (pr_bg(s_t, w_t) / norm)

    where ``pr_bg`` is the joint likelihood of the observed mechanism state
    summed over system past, ``norm`` is the joint likelihood summed over
    all past states, and the inner sum runs over background past states.
    Returned as a FactoredTPM with shape ``(*alphabet_sizes, k_i)`` per
    output unit.
    """
    n = factored.n_nodes
    alphabet_sizes = factored.alphabet_sizes
    all_indices = tuple(range(n))
    system_indices = tuple(sorted(node_indices))
    background_indices = tuple(sorted(set(all_indices) - set(system_indices)))

    # Joint Bernoulli/categorical likelihood of the observed state given
    # past: pr_joint(s_t) = ∏_i factor_i(s_t)[state[i]]
    pr_joint = np.ones(alphabet_sizes, dtype=np.float64)
    for i in all_indices:
        pr_joint = pr_joint * factored.factor(i)[..., state[i]]

    # pr_bg(s_t) = Σ_{s_{M,t}} pr_joint(s_t), keepdims preserves shape
    if system_indices:
        pr_bg = pr_joint.sum(axis=system_indices, keepdims=True)
    else:
        pr_bg = pr_joint.copy()

    norm = pr_joint.sum()
    if norm <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(state)

    weight = pr_bg / norm  # shape: keepdims-broadcast over alphabet_sizes

    # Per-output-unit factor: weighted forward marginal, summed over
    # background past.
    out_factors = []
    for i in all_indices:
        forward_i = factored.factor(i)  # shape (*alphabet_sizes, k_i)
        # Broadcast weight over the trailing alphabet axis of forward_i.
        weighted = forward_i * weight[..., np.newaxis]
        if background_indices:
            weighted = weighted.sum(axis=background_indices, keepdims=True)
        out_factors.append(weighted)

    return FactoredTPM(factors=out_factors)
```

Also fix `_cause_tpm_factored_binary` to return `FactoredTPM` so binary and k-ary now produce the same structural type. Replace its body:

```python
def _cause_tpm_factored_binary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Binary cause TPM via SBN-form Bayesian inversion.

    Stacks per-output-unit ``P(node_i = 1 | s_t)`` slices into the
    state-by-node form, applies the legacy backward TPM, and re-expands
    the trailing on-probability axis into explicit ``[P(off), P(on)]``
    factors per output unit.
    """
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    joint = JointTPM(sbn)
    raw = _legacy_backward_tpm(joint._inner, state, node_indices)
    # raw shape: (*alphabet_sizes, n) where trailing axis is per-output-unit
    # P(node_i = 1) under SBN weighting (collapsed background axes are
    # already size 1 here per the legacy keepdims contract).
    out_factors = []
    for i in range(n):
        on = raw[..., i]
        off = 1.0 - on
        out_factors.append(np.stack([off, on], axis=-1))
    return FactoredTPM(factors=out_factors)
```

Update the dispatcher to keep returning `CausePosterior` for now (Task 14 switches it), wrapping the new `FactoredTPM` through the existing helper. Replace `cause_tpm`'s body:

```python
def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Backward TPM — IIT 4.0 Eq. 4.

    Internally the per-output-unit factors are computed under a unified
    FactoredTPM-based path. The dispatcher currently wraps the result in
    CausePosterior to preserve the existing return contract; a later task
    switches the return type to FactoredTPM directly.
    """
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            factored_out = _cause_tpm_factored_binary(tpm, state, node_indices)
        else:
            factored_out = _cause_tpm_factored_kary(tpm, state, node_indices)
        return _as_cause_posterior(factored_out)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return CausePosterior(_legacy_backward_tpm(legacy._inner, state, node_indices))


def _as_cause_posterior(factored: FactoredTPM) -> CausePosterior:
    """Bridge from the new FactoredTPM cause representation to the
    legacy SBN-shaped CausePosterior. Stacks per-unit on-probability
    slices into the legacy ``(*alphabet_sizes, n)`` array.

    Removed in the producer-switch task; exists during the consumer
    migration window only.
    """
    n = factored.n_nodes
    if not all(a == 2 for a in factored.alphabet_sizes):
        # k-ary: there is no SBN-equivalent shape; consumers must use
        # factor(i) directly. The bridge raises until consumers migrate.
        raise NotImplementedError(
            "Non-binary cause TPMs are not representable in SBN-form. "
            "Use the FactoredTPM-returning path."
        )
    on_slices = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    return CausePosterior(on_slices)
```

- [ ] **Step 6.4: Run tests; verify goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: new k-ary tests pass; golden regression shows `23 passed`; full suite no new failures.

If goldens drift: STOP. The bridge in `_as_cause_posterior` must reproduce the pre-fix SBN-form output byte-identically. Diagnose by comparing `_cause_tpm_factored_binary`'s output to `_legacy_backward_tpm`'s raw output on a fixture from `test/example_networks.py`.

- [ ] **Step 6.5: Pyright + ruff.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 6.6: Commit.**

```bash
cd ../pyphi-p12b
git add pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Fix k-ary cause TPM to apply background weighting per IIT 4.0 Eq. 4

The native k-ary cause path previously computed only a partial-likelihood
joint posterior without the Σ_{w_t} background marginalization or the
pr_bg / norm weighting required for subset-system cause inversion. The
fixed implementation matches the legacy backward-TPM math: joint Bernoulli
likelihood × pr_bg / norm, then summed over background past, per output
unit.

Both _cause_tpm_factored_binary and _cause_tpm_factored_kary now return
FactoredTPM with shape (*alphabet_sizes, k_i) per output unit. The
dispatcher temporarily bridges to CausePosterior via _as_cause_posterior
(SBN-form stacking) so consumers can migrate one site at a time.

Binary inputs produce byte-identical per-factor values across both paths
(Hypothesis property test with atol=1e-10)."
git show --stat HEAD
```

---

### Task 7: Remove the defensive k>2 NotImplementedError from `_effect_tpm_factored`

**Goal:** `FactoredTPM.condition` is alphabet-generic. Remove the binary-only guard so the effect path Just Works for k-ary substrates.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py`
- Test: extend `test/test_marginalization_kary.py` (additional cases)

- [ ] **Step 7.1: Write the failing test.**

Append to `test/test_marginalization_kary.py`:

```python
def test_effect_tpm_kary_does_not_raise() -> None:
    """Effect TPM works for k>2 substrates via FactoredTPM.condition."""
    from pyphi.core.tpm.marginalization import effect_tpm
    factors = [_random_kary_factor(2, 3, seed=30 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = effect_tpm(factored, background={1: 1})
    assert result is not None
    # Conditioning fixes node 1's input axis to index 1; per-factor shape
    # collapses that axis to size 1.
    for i in range(factored.n_nodes):
        f = result.factor(i)
        assert f.shape[1] == 1
```

- [ ] **Step 7.2: Run failing test.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py::test_effect_tpm_kary_does_not_raise -v
```

Expected: fails with `NotImplementedError` from the binary-only guard.

- [ ] **Step 7.3: Remove the binary-only guard.**

Replace `_effect_tpm_factored` in `pyphi/core/tpm/marginalization.py`:

```python
def _effect_tpm_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> FactoredTPM:
    """Condition a factored TPM on background nodes via FactoredTPM.condition.

    Alphabet-generic: works for binary and k-ary substrates uniformly.
    """
    return factored.condition(background)
```

Remove the SBN-stacking fallback. Verify `FactoredTPM.condition` returns a `FactoredTPM` (it does per the P12a contract).

- [ ] **Step 7.4: Run tests.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: all marginalization-kary tests pass; goldens at `23 passed`.

- [ ] **Step 7.5: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/core/tpm/marginalization.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/marginalization.py 2>&1 | tail -3
git add pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Effect TPM dispatches to FactoredTPM.condition for any alphabet

Removes the binary-only guard from _effect_tpm_factored. FactoredTPM.condition
is alphabet-generic; the previous SBN-stacking fallback was unreachable for
binary inputs and incorrect for k-ary inputs.

Adds a k>2 effect TPM smoke test exercising background conditioning on a
3-node 3-alphabet substrate."
git show --stat HEAD
```

---

### Task 8: Introduce `factor(i)` accessor on `CausePosterior`

**Goal:** Add a `factor(i)` method on `CausePosterior` that synthesizes per-output-unit factors of shape `(*α, 2)` from the SBN-form storage. This lets consumers migrate to a uniform `factor(i)[..., state]` access pattern before the producer's representation switches. The accessor is pure code addition; goldens unchanged.

**Files:**
- Modify: `pyphi/core/tpm/cause_posterior.py`
- Test: extend `test/test_cause_posterior.py`

- [ ] **Step 8.1: Write the failing test.**

Append to `test/test_cause_posterior.py`:

```python
def test_cause_posterior_factor_returns_per_unit_distribution() -> None:
    """factor(i) returns shape (*alphabet_sizes, 2) with [1-x, x] entries."""
    import numpy as np
    from pyphi.core.tpm.cause_posterior import CausePosterior
    arr = np.array([[0.2, 0.7], [0.3, 0.4]], dtype=np.float64)
    posterior = CausePosterior(arr)  # SBN-form: trailing axis = n_nodes
    f0 = posterior.factor(0)
    np.testing.assert_allclose(f0[..., 1], arr[..., 0])
    np.testing.assert_allclose(f0[..., 0], 1.0 - arr[..., 0])
    f1 = posterior.factor(1)
    np.testing.assert_allclose(f1[..., 1], arr[..., 1])
    np.testing.assert_allclose(f1[..., 0], 1.0 - arr[..., 1])
```

- [ ] **Step 8.2: Run failing test.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_cause_posterior.py::test_cause_posterior_factor_returns_per_unit_distribution -v
```

Expected: `AttributeError: 'CausePosterior' object has no attribute 'factor'`.

- [ ] **Step 8.3: Add the accessor.**

Add to `pyphi/core/tpm/cause_posterior.py`:

```python
import numpy as np
from numpy.typing import NDArray


class CausePosterior(JointDistribution):
    # ... existing body ...

    def factor(self, i: int) -> NDArray[np.float64]:
        """Per-output-unit factor of shape ``(*alphabet_sizes, 2)``.

        Constructs ``[1 - x, x]`` along a new trailing axis from the
        SBN-form ``P(node_i = 1 | s_t)`` slice at output index ``i``.

        Binary-only: the SBN storage encodes only firing probability and
        cannot represent k-ary alphabets.
        """
        arr = np.asarray(self._tpm)
        on = arr[..., i]
        off = 1.0 - on
        return np.stack([off, on], axis=-1)
```

- [ ] **Step 8.4: Run tests; verify goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_cause_posterior.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: new accessor test passes; goldens unchanged.

- [ ] **Step 8.5: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/core/tpm/cause_posterior.py test/test_cause_posterior.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/cause_posterior.py test/test_cause_posterior.py 2>&1 | tail -3
git add pyphi/core/tpm/cause_posterior.py test/test_cause_posterior.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "CausePosterior gains factor(i) accessor synthesizing per-unit distributions

Adds a factor(i) method that constructs the per-output-unit distribution
of shape (*alphabet_sizes, 2) by stacking [1-x, x] over the SBN-form
P(node_i = 1) slice. Lets downstream consumers migrate to a unified
factor(i)[..., state] access pattern before the cause TPM's internal
representation is switched.

Pure code addition; goldens unchanged."
git show --stat HEAD
```

---

## Phase C-revised — Consumer migration (Tasks 9-13)

Per the amendment, each consumer migrates in isolation with a goldens check before commit. This isolates blast radius if drift occurs.

### Task 9: Migrate `Node.__init__` to use `cause_tpm.factor(self.index)`

**Goal:** Replace the SBN-specific `cause_tpm[..., self.index]` + `np.stack([1-x, x])` construction in `Node.__init__` with a single `cause_tpm.factor(self.index)` call. The new code path is mathematically equivalent to the legacy path on `CausePosterior` (via the Task 8 accessor); goldens unchanged. This unblocks `Node.cause_tpm_on/off` cleanup in a later task.

**Files:**
- Modify: `pyphi/node.py`
- Test: `test/test_node.py` already pins byte-identical per-node `cause_tpm` shape; serves as the regression bar.

- [ ] **Step 9.1: Read the current implementation.**

```bash
cd ../pyphi-p12b
sed -n '40,110p' pyphi/node.py
```

Note the three operations to be replaced:
- Line 63: `cause_tpm_on = cause_tpm[..., self.index]`
- Line 72-73: `cause_non_inputs = ... tpm_indices() ... ; cause_tpm_on = cause_tpm_on.marginalize_out(cause_non_inputs).tpm`
- Lines 80-88: `cause_tpm_off = 1 - cause_tpm_on; ... stack([cause_tpm_off, cause_tpm_on], axis=-1)`

- [ ] **Step 9.2: Replace with the unified factor accessor.**

In `pyphi/node.py::Node.__init__`, replace the substrate-level cause slicing block with:

```python
# Per-unit cause factor: shape (*alphabet_sizes, k_i).
# CausePosterior.factor synthesizes [1-x, x] from the SBN-form trailing axis.
cause_factor = cause_tpm.factor(self.index)
# Wrap in a JointDistribution-like container so the existing
# marginalize_out call below works uniformly.
cause_factor_typed = JointTPM(cause_factor)
cause_non_inputs = (
    set(cause_factor_typed.tpm_indices()) - set(self._inputs)
)
cause_factor_typed = cause_factor_typed.marginalize_out(cause_non_inputs)
self._cause_tpm = cause_factor_typed
```

The substitution preserves:
- The per-node TPM shape `(*alphabet_sizes, 2)` for binary.
- The marginalize_out semantics (non-input axes averaged).
- The downstream `node.cause_tpm[..., state]` indexing in `_single_node_cause_repertoire`.

- [ ] **Step 9.3: Run targeted tests + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_node.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: byte-identical pass on `test_node.py` (per-node cause TPM shapes and values pinned) and `23 passed` on goldens.

If goldens drift: diagnose via `pytest test/test_node.py -k 'test_node_init_tpm' -v --tb=long`. The factor(i) construction must produce numerically identical values to the legacy stack(off, on). Suspect axis-ordering or `np.stack` axis= mismatch.

- [ ] **Step 9.4: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/node.py 2>&1 | tail -3
uv run ruff check pyphi/node.py 2>&1 | tail -3
git add pyphi/node.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Node.__init__ reads per-unit cause factor via factor(i) accessor

Replaces the SBN-trailing-axis slice plus np.stack([1-x, x]) construction
with a single CausePosterior.factor(self.index) call. The per-node cause
TPM keeps its (*alphabet_sizes, 2) shape and its byte-identical values
for binary substrates; only the construction path changes.

Removes Node.__init__'s direct dependency on the SBN trailing-axis
representation, enabling later migration of the cause TPM's internal
representation."
git show --stat HEAD
```

---

### Task 10: Migrate `_single_node_cause_repertoire` to alphabet-generic indexing

**Goal:** The hot-path consumer `_single_node_cause_repertoire` indexes per-node `cause_tpm[..., state]`. After Task 9, `node.cause_tpm` is the FactoredTPM-style per-unit factor, so the indexing already works for any alphabet. This task adds an explicit smoke test for k=3 to lock the contract and removes any binary-specific assumptions in the function body.

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Test: extend `test/test_marginalization_kary.py` with a per-node repertoire smoke test.

- [ ] **Step 10.1: Read the current implementation.**

```bash
cd ../pyphi-p12b
sed -n '115,160p' pyphi/core/repertoire_algebra.py
```

- [ ] **Step 10.2: Write the failing/lock-in test.**

Append to `test/test_marginalization_kary.py`:

```python
def test_single_node_cause_repertoire_k3() -> None:
    """Cause repertoire for a single-node mechanism on a k=3 substrate
    returns a valid distribution over the purview's joint state space."""
    import pyphi
    from pyphi.core.repertoire_algebra import _single_node_cause_repertoire
    factors = [_random_kary_factor(2, 3, seed=40 + i) for i in range(2)]
    sub = pyphi.Substrate(marginals=factors)
    sys = pyphi.System(sub, state=(0, 0))
    rep = _single_node_cause_repertoire(sys, 0, frozenset({0}))
    assert rep.ndim == 2
    assert rep.shape == (3, 1)
    np.testing.assert_allclose(rep.sum(), 1.0, atol=1e-10)
```

- [ ] **Step 10.3: Run the test.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py::test_single_node_cause_repertoire_k3 -v
```

If it passes immediately (likely): Task 9 already made the function alphabet-generic. The body needs no changes beyond a docstring update clarifying the contract.

If it fails: read the failure and add the minimal alphabet-generic fix to `_single_node_cause_repertoire`. The expected fix point is `mechanism_node.cause_tpm[..., mechanism_node.state]` — `mechanism_node.state` is an integer index into the per-node alphabet axis, which works for any `k_i`.

- [ ] **Step 10.4: Update the docstring for alphabet genericity.**

In `pyphi/core/repertoire_algebra.py::_single_node_cause_repertoire`:

```python
def _single_node_cause_repertoire(
    cs: Any, mechanism_node_index: int, purview_set: frozenset[int]
) -> NDArray[np.float64]:
    """Single-node cause repertoire over the purview.

    Reads ``mechanism_node.cause_tpm[..., mechanism_node.state]`` to extract
    the per-output-state slice of the per-node cause factor (shape
    ``(*alphabet_sizes,)`` after the trailing alphabet axis is indexed by
    ``mechanism_node.state``), then marginalizes out the mechanism node's
    non-purview inputs.

    Alphabet-generic: works for any per-node alphabet size.
    """
    # ... existing body, unchanged ...
```

- [ ] **Step 10.5: Run targeted tests + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: k=3 smoke passes; goldens at `23 passed`.

- [ ] **Step 10.6: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/core/repertoire_algebra.py 2>&1 | tail -3
uv run ruff check pyphi/core/repertoire_algebra.py 2>&1 | tail -3
git add pyphi/core/repertoire_algebra.py test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Lock alphabet-generic contract for _single_node_cause_repertoire

Adds a k=3 single-node cause repertoire smoke test and clarifies the
docstring: the function reads node.cause_tpm[..., state] where state is
an integer index into the per-node alphabet axis. Works uniformly across
binary and multi-valued alphabets via the Task 9 Node.__init__ migration."
git show --stat HEAD
```

---

### Task 11: Redesign `System.proper_cause_tpm` as a system-level recompute

**Goal:** `System.proper_cause_tpm` currently slices the substrate-level cause TPM's trailing axis (`cause_tpm[..., list(node_indices)]`) — a binary-specific operation that has no meaning under FactoredTPM. Redesign as an on-demand recompute returning per-system-unit factors restricted to `node_indices`. Binary callers see byte-identical values via the recompute; k-ary callers get a meaningful answer.

**Per the amendment (Q1):** redesign rather than preserve the slice-based optimization. If perf regresses, add a cache later; do not preserve the slice-based optimization speculatively.

**Files:**
- Modify: `pyphi/system.py`
- Modify: `pyphi/actual.py` (parallel `TransitionSystem.proper_cause_tpm`)
- Test: extend `test/test_system.py` with a regression case pinning binary byte-identity and a k=3 smoke case.

- [ ] **Step 11.1: Audit current callers of `proper_cause_tpm`.**

```bash
cd ../pyphi-p12b
grep -rn "proper_cause_tpm" pyphi/ test/ --include="*.py" | grep -v ".pyc:"
```

Document each caller. Most are Protocol declarations (typed `Any`) and one public-API read site.

- [ ] **Step 11.2: Write the failing test.**

Append to `test/test_system.py`:

```python
def test_proper_cause_tpm_kary_returns_factored_view() -> None:
    """proper_cause_tpm for a k=3 system returns per-system-unit factors
    restricted to system indices."""
    import pyphi
    import numpy as np
    rng = np.random.default_rng(99)
    factors = []
    for i in range(2):
        arr = rng.uniform(size=(3, 3, 3))
        factors.append(arr / arr.sum(axis=-1, keepdims=True))
    sub = pyphi.Substrate(marginals=factors)
    sys = pyphi.System(sub, state=(0, 0))
    proper = sys.proper_cause_tpm
    # Result is a FactoredTPM with one factor per system unit; trailing
    # axis sized by per-unit alphabet.
    assert proper.n_nodes == len(sys.node_indices)
    for i in range(proper.n_nodes):
        assert proper.factor(i).shape[-1] == 3
```

- [ ] **Step 11.3: Implement the recompute.**

In `pyphi/system.py`:

```python
@cached_property
def proper_cause_tpm(self) -> Any:
    """Cause TPM restricted to system units.

    For each system unit ``i`` in ``node_indices``, returns the per-unit
    cause factor produced by the Bayesian inversion of the substrate's
    forward TPM under the current state. Background units are
    marginalized via ``pr_bg / norm`` weighting per IIT 4.0 Eq. 4.

    Returns a FactoredTPM with one factor per system unit (trailing axis
    sized by the unit's alphabet).
    """
    from pyphi.core.tpm.marginalization import _cause_tpm_factored_kary
    factored = _cause_tpm_factored_kary(
        self._typed_tpm, self.state, self.node_indices,
    )
    # _cause_tpm_factored_kary returns a FactoredTPM with one factor per
    # substrate unit; restrict to system units.
    system_factors = [factored.factor(i) for i in self.node_indices]
    return FactoredTPM(factors=system_factors)
```

Add `from pyphi.core.tpm.factored import FactoredTPM` at the top of the module if not already imported.

Update `TransitionSystem.proper_cause_tpm` in `pyphi/actual.py` to delegate to the underlying System's recompute (the existing pass-through pattern continues to work):

```python
@cached_property
def proper_cause_tpm(self) -> Any:
    return self._underlying_system.proper_cause_tpm
```

- [ ] **Step 11.4: Run targeted tests + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_system.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: k=3 case passes; binary regression unchanged; goldens at `23 passed`.

**Binary byte-identity note:** the recompute uses the same `_cause_tpm_factored_kary` math that Task 6 verified equivalent to the legacy binary path on binary inputs (Hypothesis property test, atol=1e-10). The per-system-unit factors should match the legacy slice's per-firing-probability values up to representation. If `proper_cause_tpm` is consumed downstream as a numpy array, ensure the consumer reads the on-probability (factor(i)[..., 1]) rather than the full FactoredTPM. Document the API change in the changelog.

- [ ] **Step 11.5: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/system.py pyphi/actual.py 2>&1 | tail -3
uv run ruff check pyphi/system.py pyphi/actual.py 2>&1 | tail -3
git add pyphi/system.py pyphi/actual.py test/test_system.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "System.proper_cause_tpm recomputes per-unit factors restricted to system

Previously, proper_cause_tpm sliced the substrate-level cause TPM's
trailing axis by system node indices — a binary-specific operation with
no FactoredTPM analog. The redesigned version computes the per-output-unit
cause factor under IIT 4.0 Eq. 4 (background-weighted Bayesian inversion)
and restricts to the system's node indices, returning a FactoredTPM.

Binary callers see the same numerical content via the recompute;
multi-valued substrates get a meaningful answer. TransitionSystem's
proper_cause_tpm follows via its existing pass-through pattern."
git show --stat HEAD
```

---

### Task 12: Replace `JointDistribution.tpm_indices` heuristic with per-class implementations

**Goal:** Per the amendment (Q5), replace the binary-shape-grep `tpm_indices` heuristic on `JointDistribution` (`np.where(shape[:-1] == 2)`) with per-class implementations that reflect each subclass's storage contract.

- `JointTPM.tpm_indices()` returns `tuple(range(ndim - 1))` (all leading axes are past-state axes).
- `FactoredTPM.tpm_indices()` returns `tuple(range(n_nodes))` (substrate units).
- `CausePosterior.tpm_indices()` returns `tuple(range(ndim - 1))` (until retired in Task 16).

After Task 9's migration, `Node.__init__` calls `tpm_indices` on a JointTPM-wrapped per-unit factor, so the binary-shape-grep behavior is no longer load-bearing.

**Files:**
- Modify: `pyphi/core/tpm/joint_distribution.py`
- Modify: `pyphi/core/tpm/joint.py` (override on JointTPM if not already)
- Modify: `pyphi/core/tpm/factored.py` (add `tpm_indices` on FactoredTPM)
- Modify: `pyphi/core/tpm/cause_posterior.py` (override on CausePosterior)
- Test: extend `test/test_joint_distribution.py` or add `test/test_tpm_indices.py`

- [ ] **Step 12.1: Audit current `tpm_indices` callers.**

```bash
cd ../pyphi-p12b
grep -rn "tpm_indices()" pyphi/ test/ --include="*.py" | grep -v ".pyc:"
```

Document each caller and what semantics it relies on. Typical answers:
- "axes corresponding to per-substrate-unit past states" → `range(ndim - 1)` (joint storage with a single trailing alphabet/output axis).
- "indices of substrate units" → `range(n_nodes)` (FactoredTPM).

- [ ] **Step 12.2: Write the failing test.**

Create `test/test_tpm_indices.py`:

```python
"""tpm_indices() semantics per concrete TPM type."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM


def test_joint_tpm_indices_returns_range_ndim_minus_one() -> None:
    arr = np.zeros((2, 2, 2, 2))  # 3-node binary SBN-form
    j = JointTPM(arr)
    assert j.tpm_indices() == (0, 1, 2)


def test_factored_tpm_indices_returns_range_n_nodes() -> None:
    factors = [np.full((2, 2, 2), 0.5) for _ in range(2)]
    f = FactoredTPM(factors=factors)
    assert f.tpm_indices() == (0, 1)


def test_cause_posterior_indices_returns_range_ndim_minus_one() -> None:
    arr = np.zeros((2, 2, 3))  # 2-node-past × 3 mechanism observations
    c = CausePosterior(arr)
    assert c.tpm_indices() == (0, 1)
```

- [ ] **Step 12.3: Add per-class overrides.**

In `pyphi/core/tpm/joint.py::JointTPM`:

```python
def tpm_indices(self) -> tuple[int, ...]:
    """Substrate-unit indices: all leading axes are per-substrate-unit
    past-state axes; the trailing axis carries per-output-unit firing
    probability (SBN-form) or output state."""
    return tuple(range(self.ndim - 1))
```

In `pyphi/core/tpm/factored.py::FactoredTPM`:

```python
def tpm_indices(self) -> tuple[int, ...]:
    """Substrate-unit indices: one entry per output unit (the leading
    factor axis); the trailing alphabet axis is per-unit."""
    return tuple(range(self.n_nodes))
```

In `pyphi/core/tpm/cause_posterior.py::CausePosterior`:

```python
def tpm_indices(self) -> tuple[int, ...]:
    """Past-state-axis indices: all leading axes are per-substrate-unit
    past-state axes; the trailing axis indexes observed mechanism units."""
    return tuple(range(self.ndim - 1))
```

Remove or deprecate the base-class heuristic on `JointDistribution`:

```python
def tpm_indices(self) -> tuple[int, ...]:
    """Subclasses must override.

    The base JointDistribution stores no semantic axis labels; concrete
    subclasses define which axes are substrate-unit axes vs. trailing
    representation axes.
    """
    raise NotImplementedError(
        f"{type(self).__name__} must override tpm_indices()"
    )
```

- [ ] **Step 12.4: Run targeted tests + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_tpm_indices.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: all three new tests pass; goldens at `23 passed`. If any other test fails with `NotImplementedError` from the base class, add the appropriate override for that concrete subclass.

- [ ] **Step 12.5: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi/core/tpm/ 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/ 2>&1 | tail -3
git add pyphi/core/tpm/joint_distribution.py pyphi/core/tpm/joint.py pyphi/core/tpm/factored.py pyphi/core/tpm/cause_posterior.py test/test_tpm_indices.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Per-class tpm_indices implementations replace shape-grep heuristic

Removes the np.where(shape[:-1] == 2) heuristic on the JointDistribution
base class — a binary-specific conflation of 'axis of size 2' with
'substrate-unit axis'. Each concrete subclass now implements tpm_indices
against its actual storage contract:

  JointTPM        -> range(ndim - 1) (leading axes are past-state)
  FactoredTPM     -> range(n_nodes)  (one entry per output unit)
  CausePosterior  -> range(ndim - 1) (leading axes are past-state)

The base class raises NotImplementedError. No production callers depend
on the old binary-grep behavior post-Task 9 migration."
git show --stat HEAD
```

---

### Task 13: Audit remaining Phase 0 category B sites

**Goal:** Sweep the catalogue's category B sites and verify each has been migrated or documented as dead code. Catch any consumer the previous tasks missed.

**Files:**
- Modify: source files for any straggler sites.
- Audit doc: append a section to `docs/superpowers/audits/p12b-sbn-consumer-catalogue.md` noting the resolved status of each entry.

- [ ] **Step 13.1: Re-run the catalogue search.**

```bash
cd ../pyphi-p12b
grep -rn "\[\.\.\., self\.index\]\|cause_tpm\[\.\.\., \|cause_tpm\.tpm_indices\|1 - cause_tpm_on" pyphi/ --include="*.py" | grep -v ".pyc:"
```

Expected: only `pyphi/macro.py` (dead code) and possibly `pyphi/actual.py::TransitionSystem` if it has its own copy of the per-node-unwrap pattern.

Check `pyphi/actual.py` specifically:

```bash
cd ../pyphi-p12b
grep -n "generate_nodes\|cause_tpm\b" pyphi/actual.py
```

- [ ] **Step 13.2: Migrate any remaining live sites.**

For each non-macro hit, apply the appropriate migration:
- Per-node trailing-axis indexing → `cause_tpm.factor(i)`.
- Substrate-level SBN consumption → `cause_tpm.factor(i)[..., state[i]]`.
- `tpm_indices()` calls already addressed in Task 12.

The likely outcome is no remaining live sites (Task 9 + Task 10 + Task 11 cover the production path). Document the result.

- [ ] **Step 13.3: Update the catalogue with resolution notes.**

Append to `docs/superpowers/audits/p12b-sbn-consumer-catalogue.md`:

```markdown
## Resolution status (post-re-plan migration)

| Site | Status |
|---|---|
| `pyphi/node.py::Node.__init__` | Migrated to `factor(i)` accessor (Task 9). |
| `pyphi/core/repertoire_algebra.py::_single_node_cause_repertoire` | Alphabet-generic indexing locked (Task 10). |
| `pyphi/system.py::proper_cause_tpm` | Recomputed via `_cause_tpm_factored_kary` (Task 11). |
| `pyphi/core/tpm/joint_distribution.py::tpm_indices` | Per-class overrides (Task 12). |
| `pyphi/macro.py:*` | Dead code; deferred to macro rewrite milestone (amendment §"Out of scope"). |
| `pyphi/actual.py::TransitionSystem.*` | Pass-through to System; covered transitively. |
```

- [ ] **Step 13.4: Run full suite + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest 2>&1 | tail -5
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
```

Expected: full suite passes; goldens at `23 passed`.

- [ ] **Step 13.5: Commit (only the audit-doc update if no live migrations were needed).**

```bash
cd ../pyphi-p12b
git add docs/superpowers/audits/p12b-sbn-consumer-catalogue.md
# (add any other files if migrations were needed)
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Catalogue resolution status post-consumer-migration sweep

Records the per-site resolution status of the SBN cause-TPM consumer
catalogue after the migration tasks. All live production sites are
migrated; macro sites remain deferred to the macro rewrite milestone
per the amendment's out-of-scope list."
git show --stat HEAD
```

---

## Phase D-revised — Producer switch + legacy retirement (Tasks 14-16)

### Task 14: Switch `cause_tpm` return type from `CausePosterior` to `FactoredTPM`

**Goal:** Drop the SBN-form bridge from `cause_tpm` and `_cause_tpm_factored_binary`. Both return `FactoredTPM` directly. Update test files that assert `isinstance(_, CausePosterior)` to assert `isinstance(_, FactoredTPM)` (or remove the type check if it was structural).

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py` (drop `_as_cause_posterior`, update `cause_tpm` return type)
- Modify: `pyphi/system.py` (update `cause_tpm` cached_property's type hint and unwrap pattern)
- Modify: `pyphi/actual.py` (parallel cutover)
- Modify: `pyphi/core/repertoire_algebra.py` (read `factor(i)[..., state]` instead of `cause_tpm[..., index]` if needed)
- Modify: `test/test_marginalization_factored.py` (replace `isinstance(_, CausePosterior)` assertions)
- Modify: `test/test_core_tpm.py` (the `test_cause_tpm_parity` test rewrites its assertion against per-unit-factor equality with the legacy backward TPM's on-probability slice)

- [ ] **Step 14.1: Read the existing dispatcher and System/TransitionSystem unwrap.**

```bash
cd ../pyphi-p12b
sed -n '155,180p' pyphi/system.py
sed -n '240,270p' pyphi/actual.py
sed -n '15,35p' pyphi/core/tpm/marginalization.py
```

- [ ] **Step 14.2: Update the dispatcher.**

In `pyphi/core/tpm/marginalization.py`:

```python
def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Cause TPM — IIT 4.0 Eq. 4.

    Returns a FactoredTPM with one factor per substrate unit (each factor
    of shape (*alphabet_sizes, k_i)). Output factors represent
    ``P(s_i,t | s_{M,t+1} = state_M)`` per output unit, with background
    units marginalized under ``pr_bg / norm`` weighting.
    """
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            return _cause_tpm_factored_binary(tpm, state, node_indices)
        return _cause_tpm_factored_kary(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        # Convert legacy JointTPM input to FactoredTPM, then dispatch.
        factored = FactoredTPM.from_joint(tpm._inner)
        return cause_tpm(factored, state, node_indices)
    arr = tpm.to_array()
    factored = FactoredTPM.from_joint(arr)
    return cause_tpm(factored, state, node_indices)
```

Delete `_as_cause_posterior`.

Remove the `from .cause_posterior import CausePosterior` import (the class is still used in tests until Task 16).

- [ ] **Step 14.3: Update `System.cause_tpm` to return the FactoredTPM directly.**

In `pyphi/system.py`:

```python
@cached_property
def cause_tpm(self) -> FactoredTPM:
    """Per-output-unit cause factors for the system; see IIT 4.0 Eq. 4."""
    return _marginalize_cause(
        self._typed_tpm, self.state, self.node_indices,
    )
```

Drop the `_inner if hasattr(...) else typed` unwrap. Drop the `# type: ignore[arg-type]` if no longer needed.

Same change in `pyphi/actual.py::TransitionSystem.cause_tpm` (it's a pass-through; no body change needed beyond updating the type hint).

- [ ] **Step 14.4: Update `Node.__init__` to consume FactoredTPM.**

Task 9 already routed `Node.__init__` through `cause_tpm.factor(i)`. `FactoredTPM.factor(i)` returns the same `(*alphabet_sizes, k_i)` shape as `CausePosterior.factor(i)`, so no change is needed beyond verifying.

- [ ] **Step 14.5: Update tests.**

In `test/test_marginalization_factored.py`, replace:

```python
assert isinstance(result, CausePosterior)
```

with:

```python
assert isinstance(result, FactoredTPM)
```

and update imports.

In `test/test_core_tpm.py::test_cause_tpm_parity`, the assertion compares `cause_tpm(JointTPM, ...)` to `_legacy_backward_tpm`. After Task 14, `cause_tpm` returns a `FactoredTPM`. Rewrite the assertion to compare the FactoredTPM's per-unit on-probability slice against the legacy SBN output:

```python
def test_cause_tpm_parity() -> None:
    # ... setup as before ...
    result = cause_tpm(JointTPM(joint_arr), state, indices)
    legacy = _legacy_backward_tpm(joint_arr, state, indices)
    # Compare per-unit on-probability slices.
    for i in range(result.n_nodes):
        np.testing.assert_allclose(
            result.factor(i)[..., 1], legacy[..., i], atol=1e-10,
        )
```

- [ ] **Step 14.6: Run targeted tests + full suite + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest test/test_marginalization_factored.py test/test_core_tpm.py -v
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
uv run pytest 2>&1 | tail -5
```

Expected: goldens at `23 passed`; full suite passes.

If goldens drift here: this is the riskiest commit. The drift likely comes from a downstream consumer that the Task 9-13 sweep missed. Bisect with `grep -rn "CausePosterior\|cause_tpm\._tpm\|cause_tpm\[\.\.\." pyphi/ --include="*.py"`. Each remaining call site needs to migrate to `factor(i)` or `factor(i)[..., state[i]]`.

- [ ] **Step 14.7: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
git add pyphi/core/tpm/marginalization.py pyphi/system.py pyphi/actual.py test/test_marginalization_factored.py test/test_core_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "cause_tpm returns FactoredTPM directly; bridge removed

cause_tpm and _cause_tpm_factored_binary now return FactoredTPM with one
factor per substrate unit. The SBN-form bridge in _as_cause_posterior is
removed; the dispatcher routes JointTPM and Protocol inputs through
FactoredTPM.from_joint.

System.cause_tpm and TransitionSystem.cause_tpm return FactoredTPM
directly; the _inner unwrap is dropped. Tests that asserted
isinstance(result, CausePosterior) are updated to FactoredTPM; the
test_cause_tpm_parity assertion is rewritten to compare per-unit
on-probability slices against the legacy backward TPM."
git show --stat HEAD
```

---

### Task 15: Retire `_legacy_backward_tpm`

**Goal:** With binary and k-ary both flowing through `_cause_tpm_factored_binary` and `_cause_tpm_factored_kary` (which the math fix verified equivalent), the legacy `_legacy_backward_tpm` is reachable only from `_cause_tpm_factored_binary`'s body and from the JointTPM-input fallback dispatched via `cause_tpm`. Merge `_cause_tpm_factored_binary`'s implementation into `_cause_tpm_factored_kary` (they're mathematically identical) and remove `backward_tpm` from `pyphi/tpm.py`.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py`
- Modify: `pyphi/tpm.py` (remove `backward_tpm`)
- Modify: `test/test_tpm.py` (drop direct backward_tpm tests; the math is exercised via `cause_tpm`)
- Modify: `test/test_core_tpm.py` (drop the bridged comparison; the math is verified by binary-equivalence property test from Task 6)

- [ ] **Step 15.1: Audit `_legacy_backward_tpm` and `backward_tpm` callers.**

```bash
cd ../pyphi-p12b
grep -rn "_legacy_backward_tpm\|backward_tpm\b" pyphi/ test/ --include="*.py" | grep -v ".pyc:"
```

Expected callers: `pyphi/core/tpm/marginalization.py` (the binary path; about to be retired) and tests in `test/test_tpm.py` / `test/test_core_tpm.py`.

- [ ] **Step 15.2: Merge `_cause_tpm_factored_binary` into `_cause_tpm_factored_kary`.**

In `pyphi/core/tpm/marginalization.py`:

```python
def _cause_tpm_factored_binary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Binary cause TPM dispatches to the unified k-ary path.

    Retained as an explicit binary-path entry point for clarity; both
    binary and k-ary substrates use the same Bayesian inversion math.
    """
    return _cause_tpm_factored_kary(factored, state, node_indices)
```

(Or remove the function entirely and have the dispatcher call `_cause_tpm_factored_kary` for both alphabet cases — slightly cleaner.)

Remove the `from pyphi.tpm import backward_tpm as _legacy_backward_tpm` import.

Remove the `JointTPM(sbn)` stack-and-bridge code path (no longer needed).

- [ ] **Step 15.3: Remove `backward_tpm` from `pyphi/tpm.py`.**

Delete the `backward_tpm` function (lines ~464-498) from `pyphi/tpm.py`. Also delete `probability_of_current_state` (the helper used only by `backward_tpm`) if no other production caller exists. Confirm via grep before deleting.

- [ ] **Step 15.4: Update tests.**

In `test/test_tpm.py`, remove `test_backward_tpm` and any other direct test of the legacy function. The math is now exercised via `cause_tpm` and the Task 6 binary-equivalence property test.

In `test/test_core_tpm.py`, remove `test_cause_tpm_parity` (its comparison against `_legacy_backward_tpm` is no longer meaningful since the legacy function is deleted). The Task 6 property test (binary equivalence between paths) replaces this regression bar.

- [ ] **Step 15.5: Run full suite + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest 2>&1 | tail -5
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
```

Expected: full suite passes; goldens at `23 passed`. If goldens drift here: the unified path's math diverges from the legacy on some input not covered by the Task 6 Hypothesis property. Diagnose by re-running Task 6's binary-equivalence test with `max_examples=500` to surface the case.

- [ ] **Step 15.6: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
git add pyphi/core/tpm/marginalization.py pyphi/tpm.py test/test_tpm.py test/test_core_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Retire legacy backward_tpm; binary and k-ary share unified math

Removes pyphi/tpm.py::backward_tpm and its probability_of_current_state
helper. Binary substrates now use the same _cause_tpm_factored_kary code
path as multi-valued substrates; the math is identical (verified by the
Task 6 Hypothesis property test, atol=1e-10).

_cause_tpm_factored_binary is retained as a thin dispatcher for clarity;
both branches call _cause_tpm_factored_kary. The legacy-path-parity test
in test_core_tpm.py is removed (the binary-equivalence property in
test_marginalization_kary.py replaces it)."
git show --stat HEAD
```

---

### Task 16: Retire `CausePosterior`

**Goal:** With `cause_tpm` returning `FactoredTPM` and all consumers migrated, `CausePosterior` is unreferenced in production. Delete the class, its test file, and exports.

**Files:**
- Delete: `pyphi/core/tpm/cause_posterior.py`
- Delete: `test/test_cause_posterior.py`
- Modify: `pyphi/core/tpm/__init__.py` (drop export)
- Modify: `pyphi/__init__.py` (drop export)
- Modify: any remaining references in code or comments.

- [ ] **Step 16.1: Audit remaining references.**

```bash
cd ../pyphi-p12b
grep -rn "CausePosterior" pyphi/ test/ docs/ --include="*.py" --include="*.md" --include="*.rst" | grep -v ".pyc:"
```

Expected: the cause_posterior.py module itself, the export sites, the test file, and possibly one or two stale references in docstrings or comments.

- [ ] **Step 16.2: Delete the module and test file.**

```bash
cd ../pyphi-p12b
git rm pyphi/core/tpm/cause_posterior.py test/test_cause_posterior.py
```

- [ ] **Step 16.3: Drop the exports.**

In `pyphi/core/tpm/__init__.py`:

```python
# Before: from .cause_posterior import CausePosterior
# After: (removed)
```

In `pyphi/__init__.py`:

```python
# Before: from .core.tpm import CausePosterior as CausePosterior
# After: (removed)
```

- [ ] **Step 16.4: Clean up stragglers.**

For each remaining reference identified in 16.1, update or remove:
- Docstrings mentioning `CausePosterior` as the cause-TPM return type → update to `FactoredTPM`.
- Type hints `CausePosterior | ...` → `FactoredTPM`.
- jsonify registry entries → remove the `CausePosterior` entry (replaced by `FactoredTPM`, which already has its own registration).

- [ ] **Step 16.5: Run full suite + goldens.**

```bash
cd ../pyphi-p12b
uv run pytest 2>&1 | tail -5
uv run pytest test/test_golden_regression.py -q 2>&1 | tail -3
```

Expected: goldens at `23 passed`. Any ImportError indicates a missed reference; fix and re-run.

- [ ] **Step 16.6: Pyright + ruff + commit.**

```bash
cd ../pyphi-p12b
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
git add -A
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Retire CausePosterior; FactoredTPM is the unified cause TPM type

Deletes pyphi/core/tpm/cause_posterior.py, test/test_cause_posterior.py,
and the corresponding exports from pyphi/__init__.py and
pyphi/core/tpm/__init__.py. After the consumer migration and the
producer switch, no production code references CausePosterior.

The unification reflects IIT 4.0 Eq. 4's structural factorization: the
cause TPM factors over output system units identically to the forward
TPM (Eq. 2), so both share the FactoredTPM type."
git show --stat HEAD
```

---

## Phase E — state_space + Substrate constructor

The next three tasks (17-19) carry over from the original plan with no structural changes; the amendment does not affect them. Cross-reference Task 11-13 of the original plan for the full step-by-step.

### Task 17: `FactoredTPM` state_space field

**Goal:** Refactor `FactoredTPM` constructor: `state_space=` keyword; `alphabet_sizes=` parameter removed; alphabet sizes derived from state_space. Add `_normalize_state_space` helper.

**Reference:** Identical to Task 11 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 1876-2146). Execute that task verbatim, with one adjustment:

- Goldens check after Step 11.7 must show `23 passed` (no fixture count change yet — k>2 fixtures land in Task 22).

Commit message stays the same as the original Task 11.

### Task 18: `Substrate` constructor: `state_space=` and `alphabet=` kwargs

**Goal:** Substrate constructor accepts `state_space=` and `alphabet=`; drops `alphabet_sizes=`. State-coercion helper for label-state lookup. `Substrate.state_space` delegated property.

**Reference:** Identical to Task 12 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2148-2411). Execute that task verbatim.

### Task 19: `Substrate.joint_tpm()` alphabet-branch cleanup

**Goal:** Unify `Substrate.joint_tpm()` on the explicit-alphabet shape for binary AND k-ary. Migrate legacy callsites.

**Reference:** Identical to Task 13 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2412-2533). Execute that task verbatim.

---

## Phase F — Measure surface

### Task 20: Declarative `supports_alphabet` metadata on measure registry

**Goal:** Each measure declares a `supports_alphabet` callable. EMD-family declares binary-only; intrinsic-difference family declares alphabet-generic.

**Reference:** Identical to Task 14 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2534-2694). Execute that task verbatim.

### Task 21: Operation-level dispatcher guard

**Goal:** At measure resolution before use, check `supports_alphabet(substrate.alphabet_sizes)`; raise `NotImplementedError` with a Gomez 2021 citation for unsupported combinations.

**Reference:** Identical to Task 15 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2695-2811). Execute that task verbatim.

---

## Phase G — Goldens, end-to-end, docs

### Task 22: k>2 golden fixtures

**Goal:** Add k>2 golden fixtures: a small synthetic k=3 substrate, a heterogeneous (2,3,3) substrate, and (conditionally) the p53-Mdm2 network from Gomez 2021.

**Reference:** Identical to Task 16 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2812-2944). Execute that task verbatim.

**After this task:** `uv run pytest test/test_golden_regression.py -q` shows `25 passed` (23 binary + 2 synthetic k>2) or `26 passed` if p53-Mdm2 reproduces.

### Task 23: End-to-end k>2 SIA + AC smoke tests

**Goal:** Direct end-to-end tests confirming the k>2 IIT pipeline works.

**Reference:** Identical to Task 17 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 2945-3072). Execute that task verbatim.

### Task 24: Documentation, changelog, marker sweep, `_inner` grep test

**Goal:** Final tidy-up. Add changelog fragment. Fix the pre-existing broken `docs/conventions.rst` doctest. Add the `_inner` grep regression test. Sweep any planning markers that crept into source/docstrings/changelog.

**Reference:** Adapt Task 18 of the original plan (`docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`, lines 3073-3272), with the following changelog adjustments:

- The changelog fragment text should describe FactoredTPM as the unified cause+effect TPM type, not the "CausePosterior + FactoredTPM asymmetric pair" the original described.
- The `JointDistribution` mention stays (the base class is retained).
- No mention of `CausePosterior` (retired).

Replacement changelog body in `changelog.d/p12b-multivalued.feature.md`:

```markdown
Multi-valued (k-ary) substrates are now supported for IIT 4.0 analysis.

Construct via ``Substrate(marginals=[...], state_space=...)`` or the
``alphabet=`` shortcut. State_space accepts a flat tuple (uniform labels
across nodes) or a per-node tuple-of-tuples (heterogeneous alphabets).
The ``alphabet_sizes=`` parameter is removed; alphabet sizes are derived
from state_space or factor shapes.

Both cause and effect TPMs return :class:`pyphi.FactoredTPM` — under IIT 4.0
Eq. 4's conditional-independence structure, the cause TPM factors per
output system unit identically to the forward TPM under Eq. 2. The
unified representation eliminates the prior dual-path machinery.
:class:`pyphi.JointDistribution` remains as a base class for joint-tensor
storage (used by :class:`pyphi.JointTPM`).

Measure registry: each measure declares ``supports_alphabet`` —
EMD-family is binary-only (per Gomez et al. 2021 §2.3); the
intrinsic-difference family (AID, GID, INTRINSIC_INFORMATION,
GENERALIZED_INTRINSIC_DIFFERENCE) is alphabet-generic. Using EMD on a
k>2 substrate raises ``NotImplementedError`` with a citation to the
relevant paper and pointers to alphabet-generic alternatives.

Macro analysis (``MacroSystem``) stays binary-only — see the macro
rewrite milestone for multi-valued state grouping.
```

**Marker sweep:** before commit, run:

```bash
cd ../pyphi-p12b
grep -rn "P12b\|Phase [A-Z]\b\|TODO(P12\|per ROADMAP\|original plan" pyphi/ test/ changelog.d/ docs/superpowers/audits/ --include="*.py" --include="*.md" --include="*.rst" 2>/dev/null
```

Any hits in source code, docstrings, comments, or changelog entries must be cleaned up (per the saved-memory `feedback_no_planning_artifacts_in_code`). Hits inside `docs/superpowers/plans/` and `docs/superpowers/specs/` are fine — those are planning artifacts by definition.

---

## Final acceptance gates

| Gate | Command | Expected |
|---|---|---|
| Full suite (incl. doctests) | `uv run pytest --tb=short -q` | 0 failures |
| Fast lane | `uv run pytest test/ -m "not slow" -q` | 0 failures |
| Slow lane | `uv run pytest test/ --slow -q` | 0 failures |
| Goldens (binary byte-identical + k>2 new) | `uv run pytest test/test_golden_regression.py -v` | 23 binary byte-identical + 2-3 new k>2 |
| Perf budget | `uv run pytest test/test_perf_budget.py -v` | All within floor |
| Pyright | `uv run pyright pyphi` | 0 errors / 5 baseline warnings |
| Ruff | `uv run ruff check pyphi test` | clean |
| End-to-end SIA k>2 | `test/test_substrate_multivalued.py::test_kary_sia_end_to_end` | Returns valid SIA |
| End-to-end AC k>2 | `test/test_substrate_multivalued.py::test_kary_account_end_to_end` | Returns valid Account |
| `_inner` grep clean | `test/test_inner_unwrap_pattern.py` | No production-code matches |
| `CausePosterior` deleted | `grep -rn "CausePosterior" pyphi/` | No matches in production code |
| Planning-marker sweep | `grep -rn "P12b\|Phase [A-Z]\b\|TODO(P12" pyphi/ test/ changelog.d/` | No matches |

---

## Self-review checklist

Before declaring the project complete:

- [ ] Every commit boundary passed the goldens gate (`23 passed` initially; `25-26 passed` after Task 22).
- [ ] No commit used `--no-verify`.
- [ ] No commit staged unrelated working-tree churn (`uv.lock` etc. left untouched unless explicitly part of the change).
- [ ] No source/docstring/comment/changelog text contains planning markers (`P12b`, `Phase A`, `TODO(P12*)`, `per ROADMAP`, "original plan", etc.).
- [ ] `CausePosterior` has zero production references.
- [ ] `backward_tpm` / `_legacy_backward_tpm` has zero production references.
- [ ] `_inner if hasattr` unwrap pattern has zero production references.
- [ ] Pyright clean (0 errors / 5 baseline warnings).
- [ ] Ruff check + format clean.
- [ ] Full `uv run pytest` (no path argument) passes — confirms source doctests run.
- [ ] `docs/superpowers/audits/p12b-sbn-consumer-catalogue.md` has a resolution-status section.

---

## Execution handoff

This plan is intended for execution via `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Each task is independently committable and verifiable against the goldens gate. Tasks 6-8 (math fix + accessor) must run first; Tasks 9-13 (consumer migration) can in principle run in parallel under subagent dispatch, but the goldens gate at each commit boundary means sequential execution is safer until the per-task drift behavior is well-understood.

After Task 16 (`CausePosterior` retirement), the architecture is fully unified; Tasks 17-24 (state_space, measure surface, goldens, docs) are bag-of-features work parallel to the unification effort.

At completion, the worktree's branch is ready for review per `superpowers:finishing-a-development-branch`. Per the saved-memory `feedback_ask_before_push`, no push to origin without explicit per-instance user consent.
