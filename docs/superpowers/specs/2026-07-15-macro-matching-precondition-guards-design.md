# Macro/matching precondition guards — design

## Problem

Four public entry points in the macro and matching modules accept inputs that
violate their documented (or implicitly assumed) preconditions and return
silently wrong results instead of raising. All four are missing-validation
bugs; none involves a formalism change. Fixing them converts documented
assumptions into enforced preconditions.

### 1. `build_triggered_tpm` accepts non-binary substrates

`pyphi/matching/triggered_tpm.py:138` slices
`substrate.tpm.to_array()[..., 1]` as a binary ON-probability, and the entire
construction (step-TPM shapes `(2,)*n`, `state_by_node2state_by_state`,
little-endian state decoding) assumes binary units. A k-ary substrate is not
rejected: the slice extracts the next-state-1 column of the explicit-alphabet
joint, and the pipeline returns a well-formed but meaningless binary triggered
TPM (rows even sum to 1), silently corrupting all downstream
perception/matching numbers.

The binary assumption is an implementation limit, not a theoretical one — the
clamp-then-noise construction generalizes to k-ary alphabets, but that is a
construction-path rework (explicit-alphabet distributions, mixed-radix state
indexing) planned as separate feature work. Until it lands, non-binary input
must fail loudly.

### 2. Index-order assumptions in the matching query surface

The `TriggeredTPM` array's system axes follow `system_indices` order, and
stimulus tuples are positional relative to `sensory_indices`. The
construction is internally consistent for any order, but
`_marginalize_system` (`triggered_tpm.py:65-70`) assumes both `mechanism` and
`system_indices` are sorted — its own comment states the assumption — and
nothing enforces it. When the requested mechanism's order differs from
`system_indices` order, the state lookup indexes the reduced axes in the
wrong order and returns wrong multi-unit probabilities.

Two sub-cases require different treatment:

- **`system_indices` / `sensory_indices`: reject unsorted.** The array axes
  and every stimulus/state tuple are positional relative to these tuples.
  Silently sorting at construction would silently reinterpret the caller's
  stimulus tuples — the same silent-wrong-number failure class. An explicit
  `ValueError` is the only unambiguous contract.
- **`mechanism` + `state` in query methods: canonicalize.** In
  `conditional_probability(mechanism, state, stimulus)` and
  `marginal_probability(mechanism, state)` the pairing between mechanism
  entries and state entries is explicit in the call, so sorting the pairs
  together is semantics-preserving: `mechanism=(2,1), state=(s2,s1)` is
  internally reordered to `((1,s1),(2,s2))`. Valid calls in any order then
  give identical results, and `_marginalize_system`'s sorted assumption
  becomes true by construction.

### 3. Public `macro_tpms` validates none of its preconditions

`pyphi/macro/tpm.py:417` computes
`earliest = micro_history[len(micro_history) - unit.micro_grain]` with no
length check. For a grain-2 unit with a length-1 history the index is −1, so
Python negative-index wraparound silently reuses the current state as the
earliest state of the window, producing a wrong cause TPM (the q_c background
conditioning, Eq. 34) with no error. A gap of two or more entries raises a
bare `IndexError`. The Eq. 18 pairwise disjointness of the units is likewise
unchecked on this entry point.

The wrappers (`MacroSystem.from_micro`, `criteria`, `search`) already
validate via `_validate_units` and `_normalize_history` in
`pyphi/macro/system.py` — but `macro_tpms` is exported public API
(`pyphi.macro.macro_tpms`) and direct callers get silent wrong results.
Since `macro/system.py` imports from `macro/tpm.py`, the fix moves the three
validation helpers (`_validate_units`, `_validate_nested_apportionment`,
`_normalize_history`) down into `macro/tpm.py`, and `macro_tpms` calls them
on its own inputs. `system.py` (and any other wrapper importing them)
re-imports from their new home. Double validation through the wrappers is
harmless — the checks are cheap relative to the construction.

### 4. `MacroUnit` accepts silently-inert negative apportionment indices

`MacroUnit.__post_init__` (`pyphi/macro/units.py`) rejects negative
constituent indices but checks `background_apportionment` only for duplicates
and overlap. A negative index (e.g. −1) passes construction and every
downstream check (`_validate_units` tests only `max(footprint) >= n`), is
never consulted by the Eq. 29 discounting (axes are drawn from `range(n)`),
and yet perturbs the construction cache key — two semantically identical
units hash differently. Fix: reject negative apportionment indices in
`__post_init__`, mirroring the constituent check.

## Fixes

1. **Binary guard** — in `build_triggered_tpm` and
   `PerceptualSystem.__post_init__`: if any
   `substrate.factored_tpm.alphabet_sizes` entry differs from 2, raise
   `ValueError("only binary substrates are currently supported; got alphabet
   sizes ...")`. Docstrings change from "Assumes a binary substrate" to "Only
   binary substrates are currently supported."
2. **Sorted-indices guard** — in the same two entry points: if
   `system_indices` or `sensory_indices` is not strictly increasing, raise
   `ValueError` naming the offending argument. (Strictly increasing also
   rejects duplicates.)
3. **Mechanism canonicalization** — in `_marginalize_system`: sort the
   `(mechanism, state)` pairs together before the existing subset/length
   checks; the rest of the method is unchanged and its sorted-assumption
   comment is updated to state the invariant is established by the sort.
4. **`macro_tpms` self-validation** — move `_validate_units`,
   `_validate_nested_apportionment`, and `_normalize_history` from
   `pyphi/macro/system.py` to `pyphi/macro/tpm.py` (unchanged bodies); call
   `_validate_units(substrate, units)` and
   `micro_history = _normalize_history(units, substrate, micro_history)` at
   the top of `macro_tpms`; update importers.
5. **Negative apportionment rejection** — in `MacroUnit.__post_init__`, raise
   `ValueError` on any negative `background_apportionment` entry.

## Tests

- **k-ary rejection:** `build_triggered_tpm` and `PerceptualSystem` raise
  `ValueError` on a 3-state substrate; the message names the alphabet sizes.
- **Unsorted-indices rejection:** both entry points raise on
  `system_indices=(2,1)` and on `sensory_indices=(1,0)`; sorted calls are
  unaffected (control).
- **Mechanism canonicalization:** on a sorted-index `PerceptualSystem`,
  `conditional_probability(mechanism=(2,1), state=(s2,s1), stimulus)` equals
  `conditional_probability(mechanism=(1,2), state=(s1,s2), stimulus)`; same
  for `marginal_probability`.
- **Short-history rejection:** direct `macro_tpms` with a grain-2 unit and a
  length-1 history raises `ValueError` (was: silent wrong cause TPM); a
  correctly sized history is unaffected (control).
- **Disjointness on the public entry:** direct `macro_tpms` with overlapping
  units raises `ValueError`.
- **Negative apportionment rejection:** `MacroUnit` construction with
  `background_apportionment=(-1,)` raises `ValueError`.
- **Wrapper regression:** existing macro suite (`MacroSystem.from_micro`,
  criteria, search) passes unchanged — the moved validators behave
  identically.

## Verification gate

Pathless `uv run pytest` in the worktree (optional extras installed first).
If the perf-counter pin moves, regenerate and inspect the diff — these guards
add no repertoire calls, so any change is a red flag.

## Out of scope

- **K-ary matching support** — the construction-path rework. Recorded as a
  ROADMAP wishlist entry in this change so the binary guard reads as a
  current limit, not a permanent decision.
- The mixed-grain earliest-state test gap (review's low-priority note).
- Wave 2+ review findings.

## Follow-up bookkeeping

- ROADMAP wishlist: add the k-ary matching entry.
- Post-merge (main tree): mark the four findings resolved in
  `REVIEW-2026-07-13.md` (Wave 1 complete) and update the review memory.
- Changelog fragment under `changelog.d/`.
