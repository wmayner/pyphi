# Tolerant State-Tie Collection in `intrinsic_information` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `intrinsic_information` collects the specified-state tie family with `numerics.eq` clustering instead of raw float `==`, and returns the first tied state in enumeration order as the winner.

**Architecture:** One function changes (`intrinsic_information` in `pyphi/core/repertoire_algebra.py`): the raw maximum stays as the tie-cluster anchor, membership becomes `numerics.eq`, the winner becomes the first enumerated tie member, and the runner-up search skips the winner. One golden fixture (`multivalued_k3k3_k4_sparse_iit4_2023`) is regenerated because its recorded single "specified state" is actually a three-way float-noise tie at φ = 0.

**Tech Stack:** Python 3.13, pytest, `pyphi.numerics` tolerance helpers.

**Spec:** `docs/superpowers/specs/2026-07-16-ii-state-tie-precision-design.md`

## Global Constraints

- All work happens in the worktree `.claude/worktrees/wave5-ii-tie-precision` (branch `wave5-ii-tie-precision`); run commands from the worktree root.
- Tests computing φ pin the formalism with complete presets: `pyphi.config.override(**presets.iit4_2026)`.
- Never `--no-verify`; after every commit check `git log --oneline -1` (pre-commit hooks abort silently).
- Commit messages end with the two trailer lines shown in each commit step.
- No planning-artifact references (wave numbers, review pointers) in code, docstrings, or the changelog fragment.
- Test-result verdicts come from reading the pytest summary line in a redirected log file, never from exit codes or piped `tail`.

---

### Task 1: Tolerant tie collection (failing tests → fix → changelog)

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (imports block ~line 23-36; `intrinsic_information` body, lines 662-686)
- Test: `test/core/test_core_repertoire_algebra.py` (append)
- Create: `changelog.d/ii-state-tie-precision.fix.md`

**Interfaces:**
- Consumes: `pyphi.numerics.eq(a, b) -> bool` (tolerance comparison at `config.numerics.precision`); `System.intrinsic_information(direction, mechanism, purview, *, specification_measure, **kwargs)` which forwards `states=` to the kernel.
- Produces: `intrinsic_information` returns a `StateSpecification` whose `.ties` contains every state within `numerics.eq` of the raw maximum, `.state` is the first tied state in enumeration order, and `.runner_up_state` is the best-ranked state different from the winner. Task 2 relies on this exact behavior.

- [ ] **Step 1: Append the two failing tests**

Append to `test/core/test_core_repertoire_algebra.py` (this file imports inside test functions — keep that style):

```python
def test_intrinsic_information_collects_noise_tied_states() -> None:
    """States within ``config.numerics.precision`` of the maximum join the
    tie family, even when their float values differ by ulp-level noise."""
    import pyphi
    from pyphi import Direction
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(**presets.iit4_2026):
        measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.specification_measure
        )
        system = examples.iit4_2023_fig1a_system()
        spec = system.intrinsic_information(
            Direction.EFFECT, (2,), (0, 2), specification_measure=measure
        )
    # (1, 1) computes 2 ulp below (0, 1); both belong to the tie family.
    assert spec.state == (0, 1)
    assert sorted(t.state for t in spec.ties) == [(0, 1), (1, 1)]


def test_intrinsic_information_winner_is_first_enumerated_tied_state() -> None:
    """The winner is the first tied state in enumeration order — not the raw
    float argmax — and the runner-up is never the winner itself."""
    import pyphi
    from pyphi import Direction
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(**presets.iit4_2026):
        measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.specification_measure
        )
        system = examples.iit4_2023_fig1a_system()
        # (1, 1) ties the raw argmax (0, 1) within precision; enumerating it
        # first makes it the winner under enumeration-order selection.
        spec = system.intrinsic_information(
            Direction.EFFECT,
            (2,),
            (0, 2),
            specification_measure=measure,
            states=[(1, 1), (0, 1), (1, 0), (0, 0)],
        )
    assert spec.state == (1, 1)
    assert spec.runner_up_state == (0, 1)
    assert spec.runner_up_state != spec.state
```

- [ ] **Step 2: Run the new tests to verify both fail**

Run:
```bash
uv run pytest test/core/test_core_repertoire_algebra.py -k "noise_tied or first_enumerated" -q > /tmp/wave5-t1-red.log 2>&1; true
```
Read `/tmp/wave5-t1-red.log`. Expected: `2 failed` — the first on the `sorted(...) == [(0, 1), (1, 1)]` assertion (family has 1 member), the second on `spec.state == (1, 1)` (raw argmax `(0, 1)` wins today).

- [ ] **Step 3: Implement the fix**

In `pyphi/core/repertoire_algebra.py`, add to the module imports (after `from pyphi.direction import Direction`, keeping alphabetical order):

```python
from pyphi import numerics
```

Then replace the tie-collection block in `intrinsic_information` (currently lines 662-686):

```python
    state_to_information = {state: evaluate_state(state) for state in states}
    max_information = max(state_to_information.values())
    ranked = sorted(state_to_information.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1:
        runner_up_state, runner_up_value = ranked[1]
        runner_up_information = float(runner_up_value)
    else:
        runner_up_state = runner_up_information = None
    ties = [
        StateSpecification(
            direction=direction,
            purview=purview,
            state=state,
            intrinsic_information=float(information),
            repertoire=rep,
            unconstrained_repertoire=unconstrained_rep,
            runner_up_state=runner_up_state,
            runner_up_intrinsic_information=runner_up_information,
        )
        for state, information in state_to_information.items()
        if information == max_information
    ]
    for tie in ties:
        tie.set_ties(ties)
    return ties[0]
```

with:

```python
    state_to_information = {state: evaluate_state(state) for state in states}
    # The raw maximum anchors the tie cluster; membership is tolerance-based,
    # so states whose values differ from the maximum only by float-path noise
    # still join the family. The winner is the first tied state in enumeration
    # order, which keeps the selection independent of that noise.
    max_information = max(state_to_information.values())
    tied_states = [
        (state, information)
        for state, information in state_to_information.items()
        if numerics.eq(information, max_information)
    ]
    winner_state = tied_states[0][0]
    ranked = sorted(state_to_information.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = next(
        ((state, value) for state, value in ranked if state != winner_state),
        None,
    )
    if runner_up is not None:
        runner_up_state = runner_up[0]
        runner_up_information = float(runner_up[1])
    else:
        runner_up_state = runner_up_information = None
    ties = [
        StateSpecification(
            direction=direction,
            purview=purview,
            state=state,
            intrinsic_information=float(information),
            repertoire=rep,
            unconstrained_repertoire=unconstrained_rep,
            runner_up_state=runner_up_state,
            runner_up_intrinsic_information=runner_up_information,
        )
        for state, information in tied_states
    ]
    for tie in ties:
        tie.set_ties(ties)
    return ties[0]
```

Extend the `intrinsic_information` docstring (NumPy style, final-state voice) by appending to its summary paragraph:

```
    States whose intrinsic information ties the maximum within
    ``config.numerics.precision`` form the tie family (available via the
    returned specification's ``ties``); the returned specification is the
    first tied state in enumeration order.
```

- [ ] **Step 4: Run the new tests to verify both pass**

Run:
```bash
uv run pytest test/core/test_core_repertoire_algebra.py -q > /tmp/wave5-t1-green.log 2>&1; true
```
Read `/tmp/wave5-t1-green.log`. Expected: all tests in the file pass, including the two new ones.

- [ ] **Step 5: Write the changelog fragment**

Create `changelog.d/ii-state-tie-precision.fix.md`:

```markdown
Fixed `intrinsic_information` dropping specified states from the tie family
when their intrinsic-information values differed from the maximum only by
float-path noise: tie membership is now clustered within
`config.numerics.precision` (matching the tie-resolution convention), the
returned specification is the first tied state in enumeration order, and the
runner-up is the best-ranked competing state other than the winner.
```

- [ ] **Step 6: Commit (and verify the commit landed)**

```bash
git add pyphi/core/repertoire_algebra.py test/core/test_core_repertoire_algebra.py changelog.d/ii-state-tie-precision.fix.md
git commit -m "Collect intrinsic-information state ties within numeric precision

States whose intrinsic information differs from the maximum only by
float-path noise now join the specified-state tie family, which the
state-MIP sweep and system-state congruence checks consume. The winner
is the first tied state in enumeration order, so selection no longer
depends on ulp-level noise; the runner-up skips the winner.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```
Expected: `git log --oneline -1` shows the new commit (if it shows the previous commit, the pre-commit hook aborted — read its output, fix, re-stage, re-commit).

### Task 2: Golden fixture regeneration

**Files:**
- Modify (regenerated): `test/data/golden/v1/multivalued_k3k3_k4_sparse_iit4_2023.json` (and `.npz` if rewritten)

**Interfaces:**
- Consumes: Task 1's fixed tie collection; the golden harness's `--regenerate-golden` flag (defined in `test/conftest.py`).
- Produces: a regenerated fixture whose `mechanism_mips[1].specified_states` records the honest three-way tie; Task 3's suite runs depend on it.

- [ ] **Step 1: Confirm the fixture now fails for the expected reason**

Run:
```bash
uv run pytest test/integration/test_golden_regression.py -q > /tmp/wave5-t2-red.log 2>&1; true
```
Read the log. Expected: exactly one failure, `multivalued_k3k3_k4_sparse_iit4_2023`, message `mechanism_mips[1].specified_states: length mismatch — actual 3 vs expected 1`.

- [ ] **Step 2: Regenerate the fixture**

```bash
uv run pytest test/integration/test_golden_regression.py --regenerate-golden -k multivalued_k3k3_k4_sparse_iit4_2023 > /tmp/wave5-t2-regen.log 2>&1; true
```
Read the log to confirm regeneration ran without error.

- [ ] **Step 3: Review the diff like a golden**

```bash
git diff --stat test/data/golden/v1/
git diff test/data/golden/v1/multivalued_k3k3_k4_sparse_iit4_2023.json | head -80
```
Expected: the JSON diff shows `specified_states` for the EFFECT MIP of mechanism `(0,)` over purview `(0,)` growing from `[[0]]` to `[[0], [1], [2]]` (canonically sorted), and **no φ value changes anywhere**. If any φ value or unrelated entry changes, STOP and investigate before committing.

- [ ] **Step 4: Verify the golden suite passes**

```bash
uv run pytest test/integration/test_golden_regression.py -q > /tmp/wave5-t2-green.log 2>&1; true
```
Read the log. Expected: all collected fixtures pass, 0 failed.

- [ ] **Step 5: Commit (and verify the commit landed)**

```bash
git add test/data/golden/v1/
git commit -m "Regenerate k-ary sparse golden for tolerant state-tie collection

The EFFECT MIP of mechanism (0,) over purview (0,) is fully reducible
(phi = 0); its three purview states tie at zero intrinsic information
within numeric precision, so the fixture now records all three specified
states instead of the one that float noise happened to rank highest.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```
Expected: `git log --oneline -1` shows the new commit.

### Task 3: Full verification (fast + slow lanes)

**Files:** none (verification only).

**Interfaces:**
- Consumes: Tasks 1-2 complete.
- Produces: green pathless fast suite and green slow lane in the worktree — the completion gate before the finishing skill runs.

- [ ] **Step 1: Start the slow lane in the background**

```bash
uv run pytest -m slow --slow -q > /tmp/wave5-t3-slow.log 2>&1; true
```
(background; ~5-10 min). The slow lane gates the IIT 4.0 (2026) golden fixtures — if one fails with a tie-family length mismatch, that is the same honest correction as Task 2: inspect the tied ii values (they must be within `config.numerics.precision` of each other), regenerate that fixture with `--regenerate-golden -k <name>`, review the diff the same way, and amend the Task 2 commit or add a follow-up commit.

- [ ] **Step 2: Run the pathless fast suite**

```bash
uv run pytest -q > /tmp/wave5-t3-full.log 2>&1; true
```
Read the summary line of `/tmp/wave5-t3-full.log`. Expected: 0 failed (baseline before this branch: 3765 passed, 284 skipped; this branch adds 2 tests).

- [ ] **Step 3: Read the slow-lane summary**

Read the summary line of `/tmp/wave5-t3-slow.log` once the background run completes. Expected: 0 failed.

- [ ] **Step 4: Hand off to the finishing skill**

All tasks complete and verified → use superpowers:finishing-a-development-branch (standing choice: merge locally into `main`, full pathless suite in the main tree after merge, then worktree cleanup and status-block/memory updates).
