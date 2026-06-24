# Factored-TPM Contract Hardening — Design

Date: 2026-06-02

## Problem

`FactoredTPM` represents a TPM as one per-node conditional factor of shape
`(*alphabet_sizes, k_i)` — "full-dimension": each factor spans every substrate
unit's previous-state axis plus its own output axis. This full-dimension form
is an implicit contract relied on by every downstream consumer
(`_cause_tpm_factored`, the repertoire builders, `Node.__init__`).

The contract is not enforced. `_validate` (`pyphi/core/tpm/factored.py`) checks
that each *existing* leading axis has size `1` or `a[j]`, but never checks that
the *number* of leading axes equals `n` (the substrate size). A
reduced-dimension factor — one spanning fewer than `n` leading axes (e.g. a
`(k, k)` factor in a 3-node substrate) — is silently accepted at construction
and then crashes later with an opaque broadcasting error in
`_cause_tpm_factored`. Confirmed today: `FactoredTPM(factors=[full, reduced],
state_space=...)` constructs without error, then breaks downstream.

Separately, the two single-node repertoire builders in
`pyphi/core/repertoire_algebra.py` reach canonical output shape by
different routes, and the cause builder's route depends on an undocumented
invariant in its caller.

## Goal

1. Reject reduced-dimension factors at `FactoredTPM` construction with a clear,
   actionable error.
2. Document the repertoire broadcast contract so the cause builder's reliance
   on its caller is explicit.

## Non-goals

- **Supporting or auto-expanding reduced/sparse factors.** Sparse factors are
  the natural input to a future native sparse-inversion compute path (roadmap
  P18); that is a separate, large project. Here we enforce the current
  full-dimension contract; the validation relaxes if and when P18 lands.
- **Argument validation inside `_cause_tpm_factored`.** Once construction
  enforces full-dimension factors, the only residual exposure is the shape of
  `state` / `node_indices`, which every caller (`System`) already validates.
  Redundant internal guards are out of scope.
- **Unifying the two single-node repertoire builders.** Their divergence is
  mathematically justified: the cause builder iterates mechanism nodes (a
  product of Bayesian likelihoods, normalized downstream) while the effect
  builder iterates purview nodes (a product of independent forward
  distributions). The cause builder's size-1-on-uncovered-purview output is
  also deliberately more efficient than forcing self-contained canonical shape.

## Design

### Component 1 — Reject reduced-dimension factors

In `_validate` (`pyphi/core/tpm/factored.py`), enforce that each factor has
exactly `n + 1` dimensions:

- `n = factored.n_nodes` (the number of factors).
- For each factor `i`, require `factor(i).ndim == n + 1` (n substrate-unit axes
  plus one output axis).
- On violation, raise `exceptions.InvalidTPM` naming the factor index, its
  actual leading-axis count (`ndim - 1`), the expected `n`, and stating that
  factors must be full-dimension `(*alphabet_sizes, k_i)`.

Placement: with the existing per-factor validation loop (factored.py ~336–348),
before the `enumerate(f.shape[:-1])` axis-size loop — which currently raises an
opaque `IndexError` (indexing `a[j]` past its end) on a too-short or too-long
factor instead of giving a clear message.

This runs at the end of every `FactoredTPM.__init__` (and therefore
`from_joint`, `from_marginals`, and pickle reconstruction). After it, every
downstream consumer can assume full-dimension factors — which is why no further
hardening of `_cause_tpm_factored` is needed.

### Component 2 — Document the broadcast contract

Comments only; no behavior, math, or performance change.

- `_single_node_cause_repertoire`: document that its output is size-1 on every
  purview node that is not an input to this mechanism node, and is therefore
  broadcast to full canonical shape by the `joint = np.ones(repertoire_shape(
  ...))` allocation in `_cause_repertoire_inner`.
- `_single_node_effect_repertoire`: document that its
  `.reshape(repertoire_shape(...))` makes the output self-describingly canonical
  (its purview node at full alphabet, all other axes size 1).
- The `joint = np.ones(repertoire_shape(...))` line in `_cause_repertoire_inner`:
  note it is load-bearing for the cause path — it establishes the canonical
  shape for purview nodes covered by no mechanism node.

## Testing

- **Component 1 (TDD):**
  - Failing-first: `FactoredTPM(factors=[full_dim, reduced_dim], ...)` raises
    `InvalidTPM` with a message naming the factor and the dim counts. (Today it
    constructs silently.)
  - Full-dimension factors still construct (regression).
  - A reduced factor that previously crashed downstream now fails fast at
    construction with the clear message.
- **Component 2:** documentation only; verified by existing goldens remaining
  byte-identical.
- **Whole-suite verification (key risk):** run `uv run pytest` with no path
  argument (includes doctests) plus the golden suite. Internal FactoredTPM
  constructions must still pass — in particular `System.proper_cause_tpm` /
  `proper_effect_tpm`, which `np.squeeze` background axes before rebuilding a
  FactoredTPM (the result has exactly `len(system)` leading axes for
  `len(system)` factors — full-dimension relative to the system).

## Risks

- The stricter `_validate` could reject a currently-working *internal*
  FactoredTPM construction that builds reduced factors. Mitigation: the
  whole-suite + golden run. Any failure is a real contract violation to
  investigate, not a reason to weaken the check. The reasoned-through internal
  cases (`proper_cause_tpm`, `proper_effect_tpm`) are full-dimension relative to
  their substrate.
- Byte-identical goldens are a hard gate. Component 2 is docs-only, and
  Component 1 only rejects inputs that previously crashed, so no golden value
  should change. If any golden changes, stop and diagnose.

## Success criteria

- Reduced-dimension factors raise `InvalidTPM` at construction with an
  actionable message.
- Full suite + doctests green; all goldens byte-identical.
- The repertoire broadcast contract is documented at both single-node builders
  and the load-bearing `joint` allocation.
