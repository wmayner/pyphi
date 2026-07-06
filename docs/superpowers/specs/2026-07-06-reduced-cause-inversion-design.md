# Reduced-dimension cause inversion for factored TPMs — design

## Problem

The cause-side Bayesian inversion (IIT 4.0 Eq. 4) in
`pyphi/core/tpm/marginalization.py::_cause_marginal_factored` materializes the
joint likelihood of the observed state over **all substrate units**:

```python
pr_joint = np.ones(alphabet_sizes)          # dense a^N array
for i in all_indices:
    pr_joint = pr_joint * factored.factor(i)[..., state[i]]
```

This costs `a^N` time and memory in the substrate size `N`, even though the
`FactoredTPM` representation already encodes each factor's true dependence
structure (input axes for non-parents are size 1). Two independent problems
compound:

1. **Per-partition waste.** `System.apply_cut` is
   `dataclasses.replace(self, partition=partition)`; the new instance's
   `cached_property` caches are empty, so every cut system re-derives
   `cause_marginal` from scratch. The inversion depends only on
   `(substrate TPM, state, node_indices)` — none of which the partition
   touches (the cut enters downstream via the cut connectivity matrix, when
   node TPMs marginalize out severed inputs). Measured: 7,897 inversion calls
   for 7,896 partitions, all on identical inputs
   (`benchmarks/iit_3_vs_4/p18_inversion_share.py`, seed 6001).

2. **Dense evaluation of a sparse computation.** For a small system embedded
   in a large sparse substrate — the realistic workflow now that the factored
   representation makes large substrates representable — the inversion's
   output only needs probability mass on the system units' parent axes. The
   dense `pr_joint` hits a memory wall near `N ≈ 30` (binary: 2^30 float64 ≈
   8 GB), far below the representational ceiling.

Measured cost shares for a fixed 6-node system embedded in substrates of
size 6/8/10/12: 6.2% / 8.2% / 10.3% / 17.1% of SIA wall time (raw profiles in
`benchmarks/iit_3_vs_4/results/p18_inversion_share_seed6001.json`). Beyond
`N ≈ 30` the computation is not slow but impossible.

## Scope

- **Target ceiling: `N ≤ 63` substrate units.** `FactoredTPM` stores every
  factor with all `N` input axes (size 1 for non-parents), and numpy caps
  arrays at 64 dimensions. Lifting that ceiling requires a new factor storage
  (parent axes only + axis map) touching `FactoredTPM`'s contract,
  validation, `condition()`, serialization, and all consumers — explicitly
  **out of scope**. The reduced computation alone unlocks the 30–63 unit
  range and speeds up the 15–30 range.
- The effect side (`_effect_marginal_factored`) is already cheap
  (per-factor conditioning, no joint materialization) — untouched.
- **One implementation, no size dispatch.** The reduced path replaces the
  dense path at every substrate size. The dense implementation survives only
  as a test oracle.

## Design

### Part 0 — `apply_cut` shares partition-independent caches (separate commit, first)

`System.apply_cut` copies the already-computed cache entries for
`cause_marginal`, `effect_marginal`, `proper_cause_marginal`,
`proper_effect_marginal`, and `_typed_tpm` from `self.__dict__` into the new
instance's `__dict__` (only keys already present — `functools.cached_property`
stores values there, and frozen dataclasses do not block direct `__dict__`
access). The cut system receives the *same array objects*, so the result is
byte-identical by construction. Cut systems are created in-process at
`pyphi/formalism/iit4/__init__.py` (partition evaluation) and
`pyphi/formalism/iit3/__init__.py`, so the copy covers the sequential hot
path; parallel workers reconstructing systems from pickles recompute as
before.

This lands as its own commit before the main feature, so the feature's
benchmark numbers measure single-inversion cost rather than redundancy.

**Tests:**
- Identity: after accessing `parent.cause_marginal`,
  `parent.apply_cut(p).cause_marginal is parent.cause_marginal`.
- Genuine-difference guard: partition-*dependent* properties (`cm`, `nodes`)
  differ across two distinct cuts of the same system.
- Existing golden regressions confirm SIA results unchanged.

### Part 1 — the inversion becomes an einsum contraction

`_cause_marginal_factored` is reimplemented as a sum-product contraction. The
math is unchanged — per system unit `i`:

```
out_i(s_M, s_i) = Σ_w  P(s_i | s_M, w) · pr_bg(w) / norm
pr_bg(w)        = Σ_{s_M} ∏_j factor_j(s)[state_j]
norm            = Σ pr_bg
```

Only the evaluation strategy changes:

1. **Likelihood slices.** For each unit `j`, take
   `factor_j[..., state[j]]` and squeeze it to its real (parent) axes,
   recording which substrate axes remain. (Validation guarantees alphabet
   sizes ≥ 2, so a size-1 input axis always means "non-parent".)
2. **Relevant background axes.**
   `R = background ∩ ⋃_{i ∈ system} parents(i)`. Only these axes carry
   weight the outputs can see.
3. **One einsum for `pr_bg`.** `np.einsum` with the integer-labels interface
   over all `N` slices, output axes `R`, contraction path from
   `np.einsum_path(..., optimize="greedy")` (deterministic given shapes).
   This *is* variable elimination; the optimizer is numpy's, not hand-rolled.
   Factors of units disconnected from the system collapse to scalars that
   multiply both `pr_bg` and `norm` and cancel in the division. At `N ≤ 63`
   the worst-case `norm` is ~1e-19 — no underflow risk in float64 (a
   500-unit substrate would need log-space; another reason for the ceiling).
4. **Normalization.** `norm = pr_bg.sum()`; `norm <= 0` raises
   `StateUnreachableBackwardsError` exactly as today.
5. **Per-unit outputs.** For each system unit `i`, one small einsum
   contracts `factor_i` (parent axes + output axis) against
   `weight = pr_bg / norm` (axes `R`), output axes
   `(parents(i) ∩ system, s_i)`. The result is re-inflated with size-1 axes
   to the full-ndim substrate-global shape — the exact output shape the
   dense path produces today (its real extent is also `parents(i) ∩ system`).
6. **Pre-flight size guard.** `np.einsum_path` reports the largest
   intermediate before any allocation. Above a module-level constant
   (`2**27` elements ≈ 1 GB float64), raise an informative exception naming
   the predicted size — a densely coupled large substrate fails fast with an
   explanation instead of OOM-ing the machine.

**Implementation-time verify items** (resolve during the plan, not after):
- numpy's einsum operand limit (`NPY_MAXARGS`) with up to 63 operands. If the
  single-call form is rejected, execute the `einsum_path` steps as pairwise
  contractions in a short loop — same math, same path.
- Whether `pyphi.serialize` stores `System.cause_marginal` anywhere
  (marginals should be derived, never serialized — confirm).

### Part 2 — compute only system-unit outputs (one API change)

Today `_cause_marginal_factored` returns a `FactoredTPM` with output factors
for **all `N` units**, but the only compute-path consumers use system-unit
factors exclusively:

- `System.nodes` → `generate_nodes(...)` builds nodes for `node_indices`
  only, and each `Node` reads only `factor(self.index)`.
- `System.proper_cause_marginal` keeps system factors and discards the rest.

Computing background-unit outputs would pull *their* parents into `R` and
defeat the reduction. Therefore:

- `_cause_marginal_factored(factored, state, node_indices)` returns factors
  **only for `node_indices`**, as a mapping `{substrate index → full-ndim
  array}`.
- `System.cause_marginal` keeps its name and its substrate-global axis
  convention but its value becomes that mapping (`dict[int, ndarray]`,
  system units only). `generate_nodes` / `Node` receive the per-node factor
  directly (both IIT 3.0 and 4.0 route
  through this one path). The `System` protocol entry is updated. Pre-release
  code: no back-compat shims, callers are updated in place.
- `proper_cause_marginal` (squeezed, system-local `FactoredTPM`) is unchanged
  in meaning and shape. Display already uses only the proper marginals.
- `cause_marginal()` in `marginalization.py` (the public dispatcher) keeps
  accepting any `TPM` type; the joint/array branches convert to
  `FactoredTPM` first, as today.

### Part 3 — the dense implementation becomes the test oracle

The current dense implementation moves verbatim into a test helper
(`dense_cause_marginal_reference`) under `test/`. It leaves production but
keeps guarding it through cross-validation.

## Validation protocol

This touches a correctness-critical hot path. The einsum contraction computes
mathematically identical quantities in a different floating-point order, so
last-ulp differences from the dense path are possible. The risk that matters
is not drift magnitude (phi comparisons run at `precision=13`; drift is
~1e-16) but **discrete flips**: a near-tie between two partitions can resolve
differently, silently changing which partition is selected as the MIP.

1. **Full-object A/B before dropping the dense path.** Every golden fixture,
   paper reproduction, and example network is computed with both
   implementations and compared as complete result objects: phi to all
   digits, selected MIPs/partitions, repertoires byte-wise.
   - Byte-identical everywhere → goldens untouched.
   - Repertoire-level ulp drift with no discrete changes → acceptable;
     documented with the measured maximum deviation.
   - Any MIP/partition flip → a genuine near-tie; investigated and resolved
     deliberately (never silently regenerated).
2. **Property-based cross-validation.** Hypothesis generates random factored
   TPMs — random parent sets, asymmetric structure, k-ary alphabets, random
   system subsets — and asserts the einsum result matches the dense oracle
   within 1e-12 (also recording whether agreement is exact). Symmetric
   fixtures hide axis-order errors; asymmetric and k-ary cases are mandatory.
3. **New-capability test with an independent reference.** A ~40-unit sparse
   substrate (dense evaluation impossible: 2^40) with a small embedded
   system, checked against the *hand-reduced equivalent*: because only the
   system's relevant closure affects the result, an equivalent small
   substrate containing just that closure must yield identical
   `proper_cause_marginal` values. This validates the large-`N` path without
   trusting the new code.
4. **Genuine-difference rule.** Every comparison test exercises at least one
   input pair with a real nonzero difference (e.g., two different states
   producing different background weights), so no gate derives its coverage
   from the artifact it verifies.
5. **Error-path parity.** `StateUnreachableBackwardsError` on `norm == 0`,
   and the pre-flight guard's exception, each have a direct test.

## Acceptance criteria

- All existing tests and goldens pass (`uv run --all-extras pytest`, no path
  argument), with any golden change individually justified per the protocol
  above.
- `proper_cause_marginal` for a small system in a 40-unit sparse substrate
  computes in well under a second and matches the reduced-equivalent
  reference exactly.
- After Part 0, one SIA computes the inversion once (not once per
  partition); after Part 1, the inversion cost no longer scales with `a^N`
  for fixed system size and sparse coupling.
- Changelog fragments: one `optimization` fragment for the `apply_cut` cache
  sharing, one for the reduced inversion.
