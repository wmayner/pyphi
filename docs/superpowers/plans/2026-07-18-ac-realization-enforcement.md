# Realization Enforcement for Actual Causation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the Realization principle of Albantakis et al. (2019) — zero-probability transitions raise `TransitionUnreachableError` at `Transition` construction and at the actual-causation analysis entry points.

**Architecture:** A new `StateUnreachableError` subclass; a full-substrate observed-pair validator in `pyphi/validate.py`; a per-candidate realization check in `Transition.__post_init__` (unpartitioned construction only); eager validation in `transitions()`, `nexus()`, `causal_nexus()`, and `events()`. Spec: `docs/superpowers/specs/2026-07-18-ac-realization-enforcement-design.md`.

**Tech Stack:** Python 3.13, numpy, pytest.

## Global Constraints

- Execute in a worktree at `.claude/worktrees/ac-realization` (create via superpowers:using-git-worktrees). Worktree venv: `uv venv`, then `WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"; env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray,mcp]" pot`.
- Raw `<= 0.0` probability comparisons — no tolerance (matches existing reachability checks).
- NumPy-style, final-state docstrings; no planning-artifact references in code, docstrings, or changelog.
- Commit messages end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe` trailers. Never `--no-verify`. Check `git log --oneline -1` after every commit (hooks can abort silently).
- Completion gate: pathless `uv run pytest` (redirect to a log file and read the summary line) plus slow lane `uv run pytest -m slow --slow`.

---

### Task 1: `TransitionUnreachableError` and `validate.transition_states`

**Files:**
- Modify: `pyphi/exceptions.py` (after `StateUnreachableBackwardsError`, ~line 50)
- Modify: `pyphi/validate.py` (new function after `state_length`, ~line 185)
- Create: `test/test_actual_realization.py`

**Interfaces:**
- Produces: `exceptions.TransitionUnreachableError(before_state, after_state, message=None)` with attributes `.before_state`, `.after_state` (and inherited `.state == after_state`); `validate.transition_states(substrate, before_state, after_state) -> None` which raises `TransitionUnreachableError` on an impossible pair and validates state lengths/alphabets.

- [ ] **Step 1: Write the failing tests**

Create `test/test_actual_realization.py`:

```python
"""Tests for enforcement of the Realization principle (Albantakis et al. 2019)."""

import numpy as np
import pytest

from pyphi import exceptions, validate
from pyphi.substrate import Substrate


@pytest.fixture
def swap_substrate():
    # Each unit copies the other's previous state: unit 0 next = unit 1
    # previous, unit 1 next = unit 0 previous. Deterministic, so from
    # (0, 0) the only successor is (0, 0).
    tpm = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    cm = np.array([
        [0, 1],
        [1, 0],
    ])
    return Substrate(tpm, cm)


def test_transition_unreachable_error_attributes():
    err = exceptions.TransitionUnreachableError((0, 0), (1, 0))
    assert isinstance(err, exceptions.StateUnreachableError)
    assert err.before_state == (0, 0)
    assert err.after_state == (1, 0)
    assert err.state == (1, 0)


def test_transition_states_rejects_impossible_pair(swap_substrate):
    # Unit 0 cannot turn on from (0, 0).
    with pytest.raises(exceptions.TransitionUnreachableError):
        validate.transition_states(swap_substrate, (0, 0), (1, 0))


def test_transition_states_accepts_possible_pair(swap_substrate):
    validate.transition_states(swap_substrate, (0, 0), (0, 0))
    validate.transition_states(swap_substrate, (1, 0), (0, 1))


def test_transition_states_rejects_malformed_states(swap_substrate):
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0, 0), (0, 0))
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0), (0, 2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'TransitionUnreachableError'`.

- [ ] **Step 3: Implement the exception**

In `pyphi/exceptions.py`, after `StateUnreachableBackwardsError`:

```python
class TransitionUnreachableError(StateUnreachableError):
    """The transition has zero probability under the substrate dynamics.

    Raised when a state pair violates the Realization principle of
    Albantakis et al. (2019): a transition is defined only when
    p(after state | before state) > 0.
    """

    def __init__(
        self,
        before_state: tuple[int, ...],
        after_state: tuple[int, ...],
        message: str | None = None,
    ) -> None:
        self.before_state = before_state
        self.after_state = after_state
        if message is None:
            message = (
                f"The transition {before_state} -> {after_state} has zero "
                "probability under the substrate dynamics."
            )
        super().__init__(after_state, message)
```

- [ ] **Step 4: Implement the validator**

In `pyphi/validate.py`, after `state_length` (~line 185):

```python
def transition_states(
    substrate: Any,
    before_state: Sequence[int],
    after_state: Sequence[int],
) -> None:
    """Raise if the observed state pair is impossible under the dynamics.

    Every unit's ``after_state`` must have nonzero probability given the
    full ``before_state``: the Realization principle of Albantakis et al.
    (2019), Section 2.2, requires p(v_t | v_{t−1}) > 0 for a transition
    to be defined.

    Raises
    ------
    pyphi.exceptions.TransitionUnreachableError
        If ``p(after_state | before_state) = 0`` under the substrate's
        factored TPM.
    ValueError
        If either state has the wrong length or is outside the alphabet.
    """
    factored = substrate.factored_tpm
    state_length(before_state, substrate.size)
    state_length(after_state, substrate.size)
    node_states(before_state, factored.alphabet_sizes)
    node_states(after_state, factored.alphabet_sizes)
    for i in range(factored.n_nodes):
        if factored._factor_at(i, before_state)[after_state[i]] <= 0.0:
            raise exceptions.TransitionUnreachableError(
                tuple(before_state), tuple(after_state)
            )
```

(`_factor_at` clamps size-1 input axes, so declared non-inputs are handled; `state_length`/`node_states` are the module's existing helpers.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pyphi/exceptions.py pyphi/validate.py test/test_actual_realization.py
git commit -m "Add TransitionUnreachableError and observed-pair validation

validate.transition_states() rejects state pairs with zero probability
under the substrate's factored TPM, per the Realization principle of
Albantakis et al. (2019).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 2: Per-candidate realization check in `Transition.__post_init__`

**Files:**
- Modify: `pyphi/actual.py:446-458` (`Transition.__post_init__`) and the `Transition` class docstring (~line 410)
- Test: `test/test_actual_realization.py`

**Interfaces:**
- Consumes: `exceptions.TransitionUnreachableError` (Task 1).
- Produces: `Transition(...)` raises `TransitionUnreachableError` when constructed unpartitioned (`partition=None`) with a zero-probability effect occurrence; construction with an explicit `partition` skips the check.

- [ ] **Step 1: Write the failing tests**

Extend the imports in `test/test_actual_realization.py`:

```python
from pyphi import actual, exceptions, validate
from pyphi.direction import Direction
from pyphi.models import NullCut
```

(replacing the existing `from pyphi import exceptions, validate` line), then append:

```python
def test_unrealizable_transition_raises(swap_substrate):
    # Effect side contains unit 0, whose observed after-state (on) is
    # impossible from (0, 0).
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.Transition(swap_substrate, (0, 0), (1, 0), (1,), (0,))


def test_realized_candidate_within_impossible_pair_constructs(swap_substrate):
    # Unit 1's observed after-state (off) has probability 1 from (0, 0),
    # so this candidate transition satisfies Realization even though the
    # full observed pair is impossible. Rejecting the pair is the job of
    # the analysis entry points, not the Transition object.
    t = actual.Transition(swap_substrate, (0, 0), (1, 0), (0,), (1,))
    assert t.effect_indices == (1,)


def test_realizable_transition_constructs(swap_substrate):
    t = actual.Transition(swap_substrate, (1, 0), (0, 1), (0, 1), (0, 1))
    assert t.node_indices == (0, 1)


def test_null_transition_constructs(swap_substrate):
    # An empty effect side is trivially realized (empty product = 1),
    # even within an impossible observed pair.
    t = actual.Transition(swap_substrate, (0, 0), (1, 0), (), ())
    assert len(t) == 0


def test_explicit_partition_bypasses_check(swap_substrate):
    # An explicit partition marks a derived copy (apply_cut path); the
    # unpartitioned parent is where validation happens.
    t = actual.Transition(
        swap_substrate,
        (0, 0),
        (1, 0),
        (1,),
        (0,),
        partition=NullCut((0, 1), swap_substrate.node_labels),
    )
    assert t.effect_indices == (0,)


def test_apply_cut_does_not_recheck(swap_substrate):
    t = actual.Transition(swap_substrate, (1, 0), (0, 1), (0,), (1,))
    cut = NullCut((0, 1), swap_substrate.node_labels)
    assert t.apply_cut(cut).partition is cut


def test_noised_background_can_realize(swap_substrate):
    # Unit 0's next state copies unit 1, which is background here (the
    # transition is over unit 0 alone). Frozen at 0, after-state 1 is
    # impossible; noised, it has probability 1/2.
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.Transition(swap_substrate, (0, 0), (1, 0), (0,), (0,))
    t = actual.Transition(
        swap_substrate, (0, 0), (1, 0), (0,), (0,), noise_background=True
    )
    assert t.probability(Direction.EFFECT, (0,), (0,)) == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: `test_unrealizable_transition_raises` and the first half of `test_noised_background_can_realize` FAIL (no error raised); the constructibility tests pass vacuously.

- [ ] **Step 3: Implement the check**

Replace `Transition.__post_init__` (`pyphi/actual.py:446-458`) with:

```python
    def __post_init__(self) -> None:
        unpartitioned = self.partition is None
        validate.state_length(self.before_state, self.substrate.size)
        validate.state_length(self.after_state, self.substrate.size)
        alphabet_sizes = self.substrate.factored_tpm.alphabet_sizes
        validate.node_states(self.before_state, alphabet_sizes)
        validate.node_states(self.after_state, alphabet_sizes)
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if unpartitioned:
            object.__setattr__(
                self, "partition", NullCut(self.node_indices, self.substrate.node_labels)
            )
            if self.effect_indices and (
                self.probability(
                    Direction.EFFECT, self.cause_indices, self.effect_indices
                )
                <= 0.0
            ):
                raise exceptions.TransitionUnreachableError(
                    self.before_state, self.after_state
                )
```

In the `Transition` class docstring, after the `Parameters` section, add:

```
    Raises
    ------
    TransitionUnreachableError
        If the effect occurrence has zero probability given the before
        state, violating the Realization principle of Albantakis et al.
        (2019): p(v_t | v_{t−1}) > 0. The check applies the transition's
        own background semantics (frozen, or noised when
        ``noise_background`` is ``True``) and runs only for unpartitioned
        construction; a transition built with an explicit ``partition`` is
        a derived copy of an already-validated transition.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: all pass.

- [ ] **Step 5: Check for fixture fallout**

Run: `uv run pytest test/test_actual.py test/test_actual_kary.py test/test_background_conditioning.py test/test_substrate_multivalued.py -x -q > /tmp/ac-realization-t2.log 2>&1; cat /tmp/ac-realization-t2.log`
Expected: all pass. If an existing fixture raises `TransitionUnreachableError`, it encodes an impossible transition — stop and surface it to the user rather than adjusting the check.

- [ ] **Step 6: Commit**

```bash
git add pyphi/actual.py test/test_actual_realization.py
git commit -m "Enforce Realization at Transition construction

Unpartitioned Transition construction now raises
TransitionUnreachableError when the effect occurrence has zero
probability given the before state (Albantakis et al. 2019). The check
uses the transition's own background semantics, so noised backgrounds
are honored; empty effect sets are trivially realized, and partitioned
copies (apply_cut) skip the re-check.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 3: Entry-point validation (`transitions`, `nexus`, `causal_nexus`, `events`)

**Files:**
- Modify: `pyphi/actual.py:888-943` (`transitions`, `nexus`, `causal_nexus`), `pyphi/actual.py:1000-1007` (`events`), `pyphi/actual.py:1095` (remove stale TODO in `true_events`)
- Test: `test/test_actual_realization.py`

**Interfaces:**
- Consumes: `validate.transition_states` (Task 1).
- Produces: all four entry points raise `TransitionUnreachableError` for impossible observed pairs; `transitions()` validates eagerly (raises at call time, not first iteration).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_actual_realization.py`:

```python
def test_transitions_raises_eagerly(swap_substrate):
    # No iteration: the pair is validated at call time.
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.transitions(swap_substrate, (0, 0), (1, 0))


def test_transitions_yields_all_candidates_for_valid_pair(swap_substrate):
    # For a realizable pair, every cm-supported candidate is realized
    # (each unit's factor is positive, so every subset product is), and
    # the sweep yields all (2^2 - 1)^2 of them.
    assert len(list(actual.transitions(swap_substrate, (1, 0), (0, 1)))) == 9


def test_causal_nexus_rejects_impossible_pair(swap_substrate):
    # Review repro: previously reported alpha = 2.0 for this pair.
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.causal_nexus(swap_substrate, (0, 0), (1, 0))


def test_nexus_rejects_impossible_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.nexus(swap_substrate, (0, 0), (1, 0))


def test_causal_nexus_valid_pair_still_works(swap_substrate):
    result = actual.causal_nexus(swap_substrate, (1, 0), (0, 1))
    assert result is not None


def test_events_rejects_impossible_first_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.events(swap_substrate, (0, 0), (1, 0), (0, 1), (0, 1))


def test_events_rejects_impossible_second_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.events(swap_substrate, (0, 0), (0, 0), (1, 0), (0, 1))


def test_events_valid_triplet_still_works(swap_substrate):
    result = actual.events(swap_substrate, (1, 0), (0, 1), (1, 0), (0, 1))
    assert isinstance(result, tuple)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: the six rejection/eagerness tests FAIL (no error raised); the valid-pair tests pass.

- [ ] **Step 3: Implement entry-point validation**

Replace `transitions()` (`pyphi/actual.py:888-903`, including both TODO comments above and inside it) with:

```python
def transitions(substrate, before_state, after_state):
    """Return a generator over all realizable transitions of a substrate.

    Candidate cause sets are subsets of the units with outputs; candidate
    effect sets are subsets of the units with inputs. The observed state
    pair is validated eagerly: calling this function on an impossible
    pair raises immediately, before any iteration.

    Raises
    ------
    TransitionUnreachableError
        If ``p(after_state | before_state) = 0`` under the substrate
        dynamics (Albantakis et al. 2019, Realization).
    """
    validate.transition_states(substrate, before_state, after_state)

    def _generate():
        # Units without inputs are reducible effects; units without
        # outputs are reducible causes.
        possible_causes = np.where(np.sum(substrate.cm, 1) > 0)[0]
        possible_effects = np.where(np.sum(substrate.cm, 0) > 0)[0]
        for cause_subset in utils.powerset(possible_causes, nonempty=True):
            for effect_subset in utils.powerset(possible_effects, nonempty=True):
                # Safety net: with a validated pair and frozen background,
                # every candidate's effect occurrence has positive
                # probability, but construction may still raise for other
                # reachability reasons.
                with contextlib.suppress(exceptions.StateUnreachableError):
                    yield Transition(
                        substrate, before_state, after_state, cause_subset, effect_subset
                    )

    return _generate()
```

In `nexus()` (`pyphi/actual.py:906`), after `validate.is_substrate(substrate)` add:

```python
    validate.transition_states(substrate, before_state, after_state)
```

and add to its docstring:

```
    Raises
    ------
    TransitionUnreachableError
        If the observed state pair is impossible under the substrate
        dynamics.
```

In `causal_nexus()` (`pyphi/actual.py:917`), after `validate.is_substrate(substrate)` add the same `validate.transition_states(...)` line and the same `Raises` docstring section.

In `events()` (`pyphi/actual.py:1000`), add at the top of the function body:

```python
    validate.transition_states(substrate, previous_state, current_state)
    validate.transition_states(substrate, current_state, next_state)
```

and extend its docstring:

```
    Raises
    ------
    TransitionUnreachableError
        If either observed pair of the state triplet
        (``previous_state`` → ``current_state`` or ``current_state`` →
        ``next_state``) is impossible under the substrate dynamics.
    """
```

In `true_events()` (`pyphi/actual.py:1095`), delete the line `# TODO: validate triplet of states` and add to the docstring's `Returns` section a preceding `Raises` section:

```
    Raises
    ------
    TransitionUnreachableError
        If either observed pair of the state triplet is impossible under
        the substrate dynamics.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_actual_realization.py -v`
Expected: all pass.

- [ ] **Step 5: Run the actual-causation test files**

Run: `uv run pytest test/test_actual.py test/test_actual_kary.py test/integration/test_paper_reproduction.py -q > /tmp/ac-realization-t3.log 2>&1; cat /tmp/ac-realization-t3.log`
Expected: all pass (paper examples are realizable transitions). Stop and surface any failure.

- [ ] **Step 6: Commit**

```bash
git add pyphi/actual.py test/test_actual_realization.py
git commit -m "Validate observed state pairs at actual-causation entry points

transitions(), nexus(), causal_nexus(), and events() now raise
TransitionUnreachableError when the observed state pair (or either pair
of the events() triplet) has zero probability under the substrate
dynamics. transitions() validates eagerly, before any iteration.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 4: Documentation, changelog, and full-suite gate

**Files:**
- Create: `changelog.d/ac-realization.change.md`
- Check/modify: `docs/tutorials/actual-causation.md`, `pyphi/mcp/content/`

**Interfaces:**
- Consumes: everything above. Produces: no new code surface.

- [ ] **Step 1: Changelog fragment**

```bash
printf '%s\n' 'Actual causation now enforces the Realization principle of Albantakis et al. (2019): `Transition` construction raises `TransitionUnreachableError` for occurrence pairs with zero probability, and `transitions()`, `nexus()`, `causal_nexus()`, `events()`, and `true_events()` reject observed state pairs that are impossible under the substrate dynamics.' > changelog.d/ac-realization.change.md
```

- [ ] **Step 2: Tutorial and MCP surface check**

Run: `grep -n -i "impossible\|realization\|unreachable" docs/tutorials/actual-causation.md; grep -rn -i "causal_nexus\|actual caus" pyphi/mcp/content/ pyphi/mcp/server.py pyphi/mcp/resources.py`

If the tutorial or an MCP content page describes what transitions are analyzable, add one sentence stating that zero-probability transitions raise `TransitionUnreachableError`; otherwise make no change. (The tutorial is a jupytext pair — edit the `.md`, then run `uv run jupytext --sync docs/tutorials/actual-causation.md` if it is paired.)

- [ ] **Step 3: Full suite (background) + slow lane**

```bash
uv run pytest -q > /tmp/ac-realization-full.log 2>&1
uv run pytest -m slow --slow -q > /tmp/ac-realization-slow.log 2>&1
```

Read both summary lines from the log files (never pipe to tail; never trust exit codes). Expected: no failures. Any `TransitionUnreachableError` from an existing test means a fixture encodes an impossible transition — surface it, don't suppress it.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/ac-realization.change.md docs/tutorials/actual-causation.md pyphi/mcp/
git commit -m "Document Realization enforcement for actual causation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

(Drop unchanged paths from `git add` as needed.)
