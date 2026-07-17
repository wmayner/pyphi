# Tolerant state-tie collection in `intrinsic_information`

**Date:** 2026-07-16
**Status:** Draft
**Scope:** `pyphi/core/repertoire_algebra.py` (`intrinsic_information`), one golden fixture regeneration.

## Problem

`intrinsic_information` (`pyphi/core/repertoire_algebra.py:662-686`) selects the
maximally specified purview state with a raw `max()` and collects the tied family
with raw float `==`:

```python
max_information = max(state_to_information.values())
...
if information == max_information
```

States whose intrinsic-information values differ from the maximum only by
float-path noise (well below `config.numerics.precision`) are silently dropped
from `.ties`. Two verified witnesses at tip `2e57e0a5`:

- **Fig. 1A** (IIT 4.0 2023 example system), EFFECT, mechanism `(2,)`, purview
  `(0, 2)`: state `(1, 1)` has ii `0.13773536151905164`, 2 ulp below the maximum
  `0.13773536151905166` at state `(0, 1)`. `numerics.eq` says they tie; raw `==`
  drops `(1, 1)`, so `.ties` has 1 member instead of 2.
- **Golden fixture `multivalued_k3k3_k4_sparse_iit4_2023`**, EFFECT, mechanism
  `(0,)`, purview `(0,)`: the mechanism is fully reducible (φ = 0) and the three
  purview states have ii `1.09e-16`, `0.0`, `−6.39e-17` — a genuine three-way tie
  at zero that raw `==` resolves by float noise, reporting a single arbitrary
  "specified" state.

The tie family is consumed substantively, not just for reporting:

- `_find_mip_iit4` (`pyphi/formalism/iit4/formalism.py`) runs one state-MIP
  evaluation per member of `.ties` when no state is pinned.
- The system-state cascade (`_spec_candidates` in
  `pyphi/formalism/iit4/__init__.py`) gives each tied state its own MIP search,
  and congruence checks accept a match against any ii-tied state — the
  `StateSpecification.ties` docstring itself notes that trimming the family
  changes the resulting φ-structure.

The precision lint (`test/test_precision_lint.py`) is blind to comparisons of
local variables, so this site was never deliberately waived. No sibling
raw-`==`-extremum site exists elsewhere in `pyphi/` (grep for `== max_` /
`== min_` / `information ==`).

## Design

The project convention for float tie clustering is
`resolve_ties._tied_with_extremum` (`pyphi/resolve_ties.py:170`): the raw
extremum anchors the cluster, membership is `numerics.eq` against that anchor,
and the winner is the first member in input order. `intrinsic_information`
adopts the same three rules:

1. **Anchor:** `max_information = max(state_to_information.values())` stays the
   raw maximum (the cluster anchor, matching `resolve_ties.py:208`).
2. **Membership:** a state joins `.ties` when
   `numerics.eq(information, max_information)`, replacing the raw `==`.
   Membership is collected in state-enumeration order, as today.
3. **Winner:** the returned specification (`ties[0]`) is the **first state in
   enumeration order among the tied family** — the same selection rule
   `resolve_ties` applies to its survivors. This makes the winner independent
   of ulp-level float noise: today the winner is whichever state's float value
   is accidentally largest, which can flip under summation-order perturbations
   (the nondeterminism class the Wave 4 work removed from parallel collection).
   In every observed case (both witnesses and the entire fast-lane suite) this
   coincides with today's winner, because the raw argmax is also the first
   enumerated tie member.

### Runner-up and `state_margin`: no tolerance treatment

`runner_up_state` / `runner_up_intrinsic_information` remain the best-ranked
*competing* state by raw value. One adjustment is required by rule 3: the
competitor search skips the winner itself (today `ranked[1]` is always safe
because the winner is always `ranked[0]`; under rule 3 the winner may rank
lower, and without the guard it could be reported as its own runner-up).

The runner-up may be a tie peer. That is intended: `state_margin` then lands
within `config.numerics.precision` of zero, which its docstring already defines
as "effectively tied". Margins are diagnostics interpreted with tolerance
downstream, not selections, so giving the ranked computation its own `eq`
clustering would change `state_margin`'s meaning for no benefit.

### Implementation sketch

```python
state_to_information = {state: evaluate_state(state) for state in states}
# Raw extremum anchors the tie cluster; membership is tolerance-based,
# following the resolve_ties convention.
max_information = max(state_to_information.values())
tied = [
    (state, information)
    for state, information in state_to_information.items()
    if numerics.eq(information, max_information)
]
winner_state = tied[0][0]
ranked = sorted(state_to_information.items(), key=lambda kv: kv[1], reverse=True)
runner_up = next(
    ((s, v) for s, v in ranked if s != winner_state),
    None,
)
```

`StateSpecification` construction over `tied` is unchanged in shape; all tie
members share the same runner-up fields, as today. `pyphi.numerics` is imported
at module level (no circularity: `numerics` depends only on `conf`).

## Impact

- **Goldens:** exactly one fixture changes —
  `multivalued_k3k3_k4_sparse_iit4_2023`, `mechanism_mips[1].specified_states`
  grows from 1 state to 3 (the honest three-way tie at φ = 0). Measured by
  running the full fast lane with a probe: 3735 passed, 1 failed (that
  assertion only). The fixture is regenerated with
  `--regenerate-golden -k multivalued_k3k3_k4_sparse_iit4_2023` and the diff
  reviewed. The slow lane (which gates the 2026 fixture variants) must be green
  before merge.
- **Behavior:** `.ties` families can grow (correct states restored);
  `runner_up`/`state_margin` self-report near-ties as before; winners are
  unchanged in every observed case.

## Testing

TDD; new tests in `test/core/test_core_repertoire_algebra.py`, pinned with
complete presets per the formalism-pinning rule:

1. **Witness regression (fails before the fix):** Fig. 1A, EFFECT, `(2,)` →
   `(0, 2)`: assert the tie family is `{(0, 1), (1, 1)}` and the winner is
   `(0, 1)`.
2. **Winner rule pinned (fails before the fix):** same computation with a
   caller-supplied `states=` order that enumerates `(1, 1)` first; assert the
   winner is `(1, 1)` (first enumerated tie member, not the raw argmax) and
   the runner-up is a different state (the guard against self-runner-up).
3. **Golden regeneration:** the regenerated fixture is the executable record of
   the k-ary three-way-tie correction; its diff is reviewed like a golden.
4. **Suites:** full pathless `uv run pytest` in the worktree and in the main
   tree after merge; slow lane (`uv run pytest -m slow --slow`) for the
   slow-gated golden fixtures.

## Alternatives considered

- **Raw argmax stays the winner; tolerance only widens the family.** Strictly
  smaller diff and provably no winner flips, but the winner selection remains
  float-noise-dependent — the same instability class this fix and the Wave 4
  work exist to remove. Rejected.
- **Reusing `resolve_ties._tied_with_extremum` directly.** No import cycle
  would result (`resolve_ties` depends only on `numerics`/`conf`/`registry`/
  `utils`), but the helper is private, and its only float logic is the
  `numerics.eq` call itself — the rest is type dispatch the kernel does not
  need, since `evaluate_state` always returns `float`. The kernel also stays
  collection-only: tie *resolution* (strategies, escalation, congruence)
  happens downstream in `resolve_ties` and the state cascade, which this fix
  feeds with a correctly collected family rather than invokes. Rejected in
  favor of the inline `numerics.eq` membership test.
- **Tolerance-clustering the ranked/runner-up computation.** Changes
  `state_margin` semantics (it would report the gap to the best
  non-tied state instead of the best competitor) and breaks its documented
  reading. Rejected.
