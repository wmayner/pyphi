# Retire the SBN Bridge in `System.effect_tpm` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `System.effect_tpm` return a `FactoredTPM` for all substrates (retire the binary-only state-by-node bridge), migrating its two live consumers (`proper_effect_tpm`, `validate`) to consume `FactoredTPM` directly. As a bonus, subsystem-level reachability checking starts working for k>2 substrates (currently silently skipped).

**Branch base:** `2.0` at the commit that adds this plan.

## Design decisions (settled in discussion)

1. **k-ary reachability check.** Rewrite `validate._proper_state_in_image_of_conditioned_tpm` to test, directly off the conditioned `FactoredTPM`: *is there an input configuration where the joint probability of `proper_state` is positive* — `∃ input: ∏_{i∈system} factor_i(input)[proper_state_i] > 0`. Verified to reduce **exactly** to today's binary check (binary: `factor_i(input)[s_i] > 0` ⇔ "not certain to differ at node i"), and extends to k>2 for free.

2. **`proper_effect_tpm` becomes a `FactoredTPM`** (not the legacy SBN array), mirroring `proper_cause_tpm` which P12b already modernized this way. It's the effect-side dual — genuine user-facing utility ("forward structure restricted to system units," k-ary-capable). It recomputes from `_typed_tpm` directly (as `proper_cause_tpm` does), so it stops depending on `self.effect_tpm` — which is what lets the bridge retire cleanly.

3. **Macro is out of scope.** `macro.py` consumes `system.effect_tpm` via the `JointTPM` API at ~7 sites, but all of it is `P7b: MacroSystem port pending` (skipped tests). Retiring the bridge leaves those skipped consumers referencing a retired shape — intentionally, so the P7b port consumes `FactoredTPM` rather than perpetuating SBN. We add a note to the P7b ROADMAP context; we do not touch macro.

## Critical files

- Modify: `pyphi/system.py` (`proper_effect_tpm` rewrite; `effect_tpm` bridge deletion + docstring)
- Modify: `pyphi/validate.py` (`_proper_state_in_image_of_conditioned_tpm` rewrite + docstring)
- Test: `test/test_system.py` (new `proper_effect_tpm` tests, mirroring the `proper_cause_tpm` pair)
- Test: `test/test_validate.py` (keep `test_validate_state_subsystem_unreachable` green; add a k=3 reachability test)
- Modify: `ROADMAP.md` (note macro hand-off on the P7b context; remove the now-done collapse entry and this SBN-bridge entry once landed)
- Create: `changelog.d/retire-effect-tpm-sbn-bridge.refactor.md`

## Reference: the pattern to mirror

`System.proper_cause_tpm` (current):
```python
@cached_property
def proper_cause_tpm(self) -> FactoredTPM:
    factored = _cause_tpm_factored(self._typed_tpm, self.state, self.node_indices)
    background_indices = tuple(
        i for i in range(factored.n_nodes) if i not in set(self.node_indices)
    )
    system_factors = []
    for i in self.node_indices:
        f = factored.factor(i)
        if background_indices:
            f = np.squeeze(f, axis=background_indices)
        system_factors.append(f)
    return FactoredTPM(factors=system_factors)
```

---

## Task 0: Pre-flight worktree

- [ ] **Step 1:** Create a worktree at `/Users/will/projects/pyphi-sbn` on branch `feature/retire-sbn-bridge` (off `2.0` HEAD) via the `superpowers:using-git-worktrees` skill. Set up deps: `uv sync --all-extras --all-groups`.
- [ ] **Step 2:** Confirm baseline green: `uv run pytest test/test_system.py test/test_validate.py test/test_golden_regression.py -q --no-header`.

---

## Task 1: `proper_effect_tpm` → `FactoredTPM` (mirror `proper_cause_tpm`)

**Files:** `pyphi/system.py`, `test/test_system.py`

- [ ] **Step 1: Write failing tests** (mirror the `proper_cause_tpm` pair) in `test/test_system.py`:

```python
def test_proper_effect_tpm_kary_returns_factored_view() -> None:
    """proper_effect_tpm for a k=3 system returns per-system-unit factors."""
    from pyphi.core.tpm.factored import FactoredTPM
    from .test_substrate_multivalued import _k3_two_node_substrate  # or inline

    sub = _k3_two_node_substrate()
    sys = System(sub, state=(0, 0))
    proper = sys.proper_effect_tpm
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(sys.node_indices)


def test_proper_effect_tpm_binary_matches_legacy_on_probability(s) -> None:
    """For binary substrates, the per-system-unit on-probability slice of
    proper_effect_tpm matches the legacy SBN forward on-probability."""
    import numpy as np
    from pyphi.core.tpm.factored import FactoredTPM

    proper = s.proper_effect_tpm
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(s.node_indices)
    # Legacy SBN reference: conditioned forward on-probability per system unit.
    # (Compute via the pre-retirement formula for the binary `s` fixture and
    #  compare proper.factor(slot)[..., 1] against it, atol=1e-10.)
```

The binary test's reference value: compute the conditioned-effect on-probability the way the old `proper_effect_tpm` did, for the `s` fixture, and assert `proper.factor(slot)[..., 1]` matches. (Implementer derives the exact reference array during execution — it's the squeezed `effect_tpm[..., node]` on-probabilities.)

- [ ] **Step 2:** Run them — expect failure (current `proper_effect_tpm` returns an ndarray, not a FactoredTPM).

- [ ] **Step 3: Rewrite `proper_effect_tpm`** in `pyphi/system.py` to recompute from `_typed_tpm` and restrict to system output units, mirroring `proper_cause_tpm`:

```python
@cached_property
def proper_effect_tpm(self) -> FactoredTPM:
    """Effect TPM restricted to system units.

    Per system unit ``i`` in ``node_indices``, the returned FactoredTPM
    carries the forward factor conditioned on the background (external)
    units at their observed state, with background input dims dropped, so
    the returned shape is ``(*system_alphabet, k_i)`` per system output
    unit. The effect-side dual of :attr:`proper_cause_tpm`.
    """
    external_state = utils.state_of(self.external_indices, self.state)
    background = dict(zip(self.external_indices, external_state, strict=False))
    factored = _marginalize_effect(self._typed_tpm, background)
    background_indices = tuple(
        i for i in range(factored.n_nodes) if i not in set(self.node_indices)
    )
    system_factors = []
    for i in self.node_indices:
        f = factored.factor(i)
        if background_indices:
            f = np.squeeze(f, axis=background_indices)
        system_factors.append(f)
    return FactoredTPM(factors=system_factors)
```

(Note: `_marginalize_effect` returns a `FactoredTPM` whose background *input* axes are size-1 after conditioning; the squeeze drops them, exactly as `proper_cause_tpm` does for the marginalized cause factors.)

- [ ] **Step 4:** Run the new tests — expect pass.
- [ ] **Step 5:** Run `test/test_system.py test/test_golden_regression.py -q` — expect no regressions (goldens byte-identical; `proper_effect_tpm` had no golden dependency).
- [ ] **Step 6:** `uv run ruff check`/`format` + pyright on `pyphi/system.py test/test_system.py`.
- [ ] **Step 7: Commit** (`git -c commit.gpgsign=false commit`), targeted add of the two files.

---

## Task 2: k-ary reachability check in `validate`

**Files:** `pyphi/validate.py`, `test/test_validate.py`

- [ ] **Step 1: Write a failing k=3 reachability test** in `test/test_validate.py` — a k=3 substrate with a deterministic unit whose conditioned dynamics cannot produce a chosen subsystem state, asserting `System(...)` raises `StateUnreachableForwardsError`. (Today this is silently skipped via `return True`, so the test fails by *not* raising.)

- [ ] **Step 2:** Run it — expect failure (no raise today for k-ary).

- [ ] **Step 3: Rewrite `_proper_state_in_image_of_conditioned_tpm`** in `pyphi/validate.py` to consume the `FactoredTPM` `proper_effect_tpm`:

```python
def _proper_state_in_image_of_conditioned_tpm(system: object) -> bool:
    """Whether the subsystem's ``proper_state`` is in the image of the
    background-conditioned effect dynamics.

    ``proper_effect_tpm`` is a FactoredTPM with one factor per system
    output unit (background fixed at the external state). The state is in
    the image iff some system-input configuration assigns positive joint
    probability to ``proper_state`` — i.e. every system factor gives
    positive probability to its component of ``proper_state`` for that
    input. Alphabet-generic: works for any per-unit alphabet size.
    """
    proper = system.proper_effect_tpm  # type: ignore[attr-defined]
    proper_state = system.proper_state  # type: ignore[attr-defined]
    joint = np.ones(proper.alphabet_sizes, dtype=np.float64)
    for slot in range(proper.n_nodes):
        joint = joint * proper.factor(slot)[..., proper_state[slot]]
    return bool(np.any(joint > 0.0))
```

Update the `state_reachable` docstring only if wording drifts; the two-check structure is unchanged.

- [ ] **Step 4:** Run the new k=3 test — expect pass.
- [ ] **Step 5:** Run `test/test_validate.py` in full — `test_validate_state_subsystem_unreachable` (binary regression guard) and `test_validate_state_no_error_1` must stay green.
- [ ] **Step 6:** ruff/pyright on `pyphi/validate.py test/test_validate.py`.
- [ ] **Step 7: Commit** (targeted).

---

## Task 3: Delete the SBN bridge from `System.effect_tpm`

**Files:** `pyphi/system.py`

- [ ] **Step 1: Delete the bridge** — `effect_tpm` returns the `_marginalize_effect` result directly:

```python
@cached_property
def effect_tpm(self) -> Any:
    """Forward TPM conditioned on the external units at their observed state."""
    external_state = utils.state_of(self.external_indices, self.state)
    background = dict(zip(self.external_indices, external_state, strict=False))
    return _marginalize_effect(self._typed_tpm, background)
```

(Drop the `isinstance(result, FactoredTPM) and all(a == 2 ...)` SBN-stacking branch and the migration-scaffold docstring.)

- [ ] **Step 2:** Confirm `node.py` handles the FactoredTPM `effect_tpm` — it already branches on `isinstance(effect_tpm, _FactoredTPM)` (`pyphi/node.py:78`). No change expected.
- [ ] **Step 3:** Run the live consumers' suites: `test/test_system.py test/test_validate.py test/test_golden_regression.py test/test_actual.py test/test_substrate_multivalued.py test/test_invariants.py -q`. Goldens byte-identical; AC suite green (TransitionSystem delegates effect_tpm through System).
- [ ] **Step 4:** Confirm macro's *green* tests still pass (the effect_tpm-touching macro code is in skipped P7b paths): `test/test_macro.py test/test_macro_system.py test/test_macro_blackbox.py -q` — expect the same passed/skipped/xfailed counts as baseline.
- [ ] **Step 5:** Full `uv run pytest` (no path argument — doctests) green.
- [ ] **Step 6:** ruff/pyright on `pyphi/system.py`.
- [ ] **Step 7: Commit** (targeted).

---

## Task 4: Changelog + ROADMAP

**Files:** `changelog.d/`, `ROADMAP.md`

- [ ] **Step 1:** Create `changelog.d/retire-effect-tpm-sbn-bridge.refactor.md`:

```
``System.effect_tpm`` now returns a ``FactoredTPM`` for all substrates; the binary-only state-by-node (SBN) bridge is retired. ``System.proper_effect_tpm`` likewise returns a ``FactoredTPM`` (per system output unit), mirroring ``proper_cause_tpm``. Subsystem-level state-reachability validation now covers k>2 substrates, which were previously assumed reachable. No change to Φ or AC results.
```

- [ ] **Step 2:** In `ROADMAP.md`: remove the completed "Collapse `TransitionSystem`" entry and this "Retire the SBN bridge" entry; add one line to the macro/P7b context noting `system.effect_tpm` is now a `FactoredTPM` and the MacroSystem port must consume it directly (the legacy `.tpm`/SBN accesses in `macro.py` need migration during P7b).

- [ ] **Step 3:** Full `uv run pytest` (no path); ruff check + format; pyright on `pyphi`.
- [ ] **Step 4: Commit** (targeted).

---

## Acceptance gate

- `System.effect_tpm` and `proper_effect_tpm` both return `FactoredTPM`; no SBN stacking in `system.py`.
- `validate` reachability runs for k>2 (new k=3 test) and keeps the binary guard green.
- Goldens 25/25 byte-identical; full `uv run pytest` (no path) green; macro counts unchanged.
- pyright clean; ruff clean.
- Changelog fragment added; ROADMAP updated (collapse + bridge entries removed, macro note added).

## Execution

Mechanical and well-scoped (~half day, 4 commits). Subagent-driven or inline per preference. The one judgment-heavy spot is Task 2 (reachability math) — already settled and verified above.
