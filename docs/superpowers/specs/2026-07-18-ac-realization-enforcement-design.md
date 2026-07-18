# Realization enforcement for actual causation — design

**Date:** 2026-07-18
**Status:** Approved

## Problem

The actual-causation framework of Albantakis et al. (2019) defines a
transition v_{t−1} ≺ v_t only when it is consistent with the system's
dynamics: p_u(v_t | v_{t−1}) > 0 (the Realization principle, Section 2.2 and
the "Realization" passage of the causal-account section; "Only occurrences
within a transition v_{t−1} ≺ v_t may have, or be, an actual cause or actual
effect"). PyPhi checks this nowhere:

- `Transition` accepts impossible occurrence pairs and computes positive α
  for them.
- `actual.transitions()` claims to yield "all **possible** transitions" but
  performs no possibility check; its `suppress(StateUnreachableError)` is
  dead code because nothing below it raises that error.
- `actual.causal_nexus()` on an observed state pair that is impossible under
  the TPM reports a confident nonzero causal nexus (review repro: a 2-unit
  substrate where p((1,0) | (0,0)) = 0 yields α = 2.0).
- `events()`/`true_events()` never validate the state triplet
  (previous → current → next).

Two logically distinct checks are involved, and the repro is fixed only by
the second:

1. **Per-candidate realization** (the paper's principle applied to one
   `Transition`): p(after-state of the effect nodes | before state) > 0,
   with background conditions handled per the transition's own semantics.
   In the repro, the candidate `cause=(0,), effect=(1,)` *passes* this —
   unit 1's observed after-state has probability 1.
2. **Observed-pair consistency** at analysis entry points: the full observed
   substrate state pair must itself be realizable. In the repro it is not
   (unit 0 cannot turn on), so the whole analysis is over data the model
   says never happened.

## Decisions

- **Entry points raise.** An impossible observed pair is a contradiction
  between the user's data and the user's model; the analysis is undefined.
  This matches `System.from_substrate`, which raises `StateUnreachableError`
  for an unreachable current state rather than returning a null result.
- **Per-candidate enforcement lives in `Transition.__post_init__`.**
  Unrealizable transitions become unrepresentable; the sweep in
  `transitions()` filters them via the now-live
  `suppress(StateUnreachableError)`; directly constructed transitions (user
  code, tests) are equally protected.
- **Raw `<= 0.0` comparison, no tolerance** — consistent with the existing
  reachability checks in `validate.py`, `core/tpm/marginalization.py`, and
  `macro/tpm.py`. Zero-probability TPM entries are exact.

## Design

### 1. Exception

`pyphi/exceptions.py` gains:

```python
class TransitionUnreachableError(StateUnreachableError):
    """The transition has zero probability under the substrate dynamics."""
```

Constructor takes `(before_state, after_state)` (and an optional message)
and formats a message naming the zero-probability transition; `self.state`
(inherited) is set to `after_state`. Subclassing `StateUnreachableError`
makes the existing `suppress(StateUnreachableError)` in `transitions()` the
live filter and keeps existing `except StateUnreachableError` callers
working. This mirrors the `StateUnreachableForwardsError` /
`StateUnreachableBackwardsError` pattern.

### 2. Per-candidate check in `Transition.__post_init__`

At the top of `__post_init__`, record whether the transition is being
constructed unpartitioned (`self.partition is None`). After the existing
state validation, index coercion, and NullCut defaulting, and only in the
unpartitioned case:

```python
if self.effect_indices and (
    self.probability(Direction.EFFECT, self.cause_indices, self.effect_indices)
    <= 0.0
):
    raise exceptions.TransitionUnreachableError(self.before_state, self.after_state)
```

(The `self.effect_indices` guard is an optimization only — an empty effect
purview yields probability 1.0, so the null transition used by
`causal_nexus` is trivially realized either way.)

Computing the probability through the transition's own machinery
(`probability` → `effect_repertoire` → `TransitionSystem`) means:

- `noise_background=True` is respected automatically: a transition
  impossible under frozen background but possible under noised background
  is accepted, matching the p_u the analysis will actually use.
- The check is exactly the paper's p_u(v_t | v_{t−1}): the effect nodes'
  observed after-state, conditioned on the cause nodes' before-state with
  background conditions applied.

**Partitioned construction skips the check.** `apply_cut` derives
partitioned copies via `dataclasses.replace`, which re-runs
`__post_init__`; realization does not depend on the partition and the
unpartitioned parent was already validated, so re-checking would add a
repertoire computation to every partition evaluation for nothing. Known
trade-off: constructing a `Transition` directly with an explicit
`partition` argument bypasses validation. That path is internal (user-facing
construction and the null transition pass `partition=None`), and the
serializer always decodes with an explicit partition, so previously saved
transitions load unchanged.

### 3. Observed-pair consistency: `validate.transition_states`

New function in `pyphi/validate.py`:

```python
def transition_states(substrate, before_state, after_state):
    """Raise if the observed state pair is impossible under the dynamics."""
```

For each unit i of the substrate, look up p(unit i in `after_state[i]` |
full `before_state`) from the factored TPM (the per-unit factor evaluated
at the before state, with size-1 input axes clamped, as `_factor_at`
does); if any factor is `<= 0.0` (equivalently, the full-state product is
zero), raise `TransitionUnreachableError(before_state, after_state)`.
State-length and alphabet validation of the two states happens here too, so
entry points fail on malformed states before any sweep begins.

Called at the top of:

- `transitions()` — validated **eagerly**: because a generator function's
  body only runs on first iteration, `transitions()` becomes a plain
  function that validates, then returns an inner generator. Calling it on
  an impossible pair raises immediately;
- `nexus()` and `causal_nexus()` — direct calls, so the error carries
  these entry points' names in the traceback even though `transitions()`
  also validates;
- `events()` — for **both** pairs of the triplet: (previous → current) and
  (current → next). This discharges the long-standing "validate triplet of
  states" TODO; `true_events`, `true_ces`, and `extrinsic_events` all
  funnel through `events()`.

### 4. Resulting behavior

| Situation | Behavior |
|---|---|
| Observed pair impossible (any unit's after-state has zero probability given the before state) | `transitions`/`nexus`/`causal_nexus`/`events` and callers raise `TransitionUnreachableError` |
| Observed pair possible; a candidate (cause set, effect set) unrealizable | Candidate filtered from the sweep by the live `suppress(StateUnreachableError)` |
| Observed pair possible; no irreducible candidates | Null SIA (α = 0), unchanged |
| Direct construction of an unrealizable `Transition` | Raises `TransitionUnreachableError` at construction; `actual.sia`/`account` can never receive one |
| `apply_cut` / decode of a saved transition | No re-check (explicit partition ⇒ derived copy) |

### 5. Testing

- **Regression test** for the review repro: `causal_nexus` on the 2-unit
  substrate with the impossible pair raises `TransitionUnreachableError`.
- Unit tests:
  - unrealizable direct construction raises; realizable construction and
    the empty-effect null transition pass;
  - a pair impossible under frozen background but possible with
    `noise_background=True` constructs only in the noised case;
  - `apply_cut` on a valid transition does not re-run the check (e.g., no
    error when cutting, and construction with explicit partition bypasses);
  - `events()` raises when either pair of the triplet is impossible;
  - `transitions()` yields only realizable candidates for a pair where some
    candidates are unrealizable.
- Full suite: all existing fixtures and both `examples.py` transitions must
  remain constructible. Any existing fixture that turns out to encode an
  impossible transition is a latent bug to surface and fix individually,
  not to suppress.

### 6. Documentation

- `Raises` sections (NumPy style) on `Transition`, `transitions()`,
  `nexus()`, `causal_nexus()`, `events()`/`true_events()`, citing the
  Realization principle (Albantakis et al. 2019, Entropy 21(5):459,
  Section 2.2).
- Remove the stale TODOs in `transitions()` and the triplet TODO in
  `true_events()`.
- Changelog fragment, type `change` (previously accepted inputs now raise).
- Check `docs/tutorials/actual-causation.md` for any passage that should
  mention the new behavior.

## Out of scope

- Multi-step (k > 1) transitions — the paper generalizes realization to
  p_u(v_t | v_{t−k}); PyPhi's `Transition` is single-step.
- Validation of directly constructed `TransitionSystem` objects (internal
  view class; both instances hang off a validated `Transition`).
- Tolerance-based comparison of near-zero probabilities (follows the
  existing exact-zero convention; revisit only if a real underflow case
  appears).
