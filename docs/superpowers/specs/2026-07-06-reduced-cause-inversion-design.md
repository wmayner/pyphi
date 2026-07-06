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

### Part 1 — the inversion becomes a greedy sum-product contraction

`_cause_marginal_factored` is reimplemented as a sum-product contraction. The
math is unchanged — per system unit `i`:

```
out_i(s_M, s_i) = Σ_w  P(s_i | s_M, w) · pr_bg(w) / norm
pr_bg(w)        = Σ_{s_M} ∏_j factor_j(s)[state_j]
norm            = Σ pr_bg
```

Only the evaluation strategy changes:

1. **Likelihood slices.** For each unit `j`, take
   `factor_j[..., state[j]]`, kept in the full-ndim substrate-global shape:
   size-1 axes mark non-parents, so broadcasting aligns factors with no
   explicit axis bookkeeping. (Validation guarantees alphabet sizes ≥ 2, so a
   size-1 input axis always means "non-parent".)
2. **Relevant background axes.**
   `R = background ∩ ⋃_{i ∈ system} parents(i)`. Only these axes carry
   weight the outputs can see.
3. **Greedy variable elimination for `pr_bg`.** Repeatedly eliminate the
   axis (∉ `R`) whose merged product of involved slices is smallest:
   multiply the involved slices (ufunc broadcasting; valid up to numpy's
   64-dimension array limit) and `.sum(axis=k, keepdims=True)`. Ties break
   toward the lowest axis index, so the order is deterministic given shapes.
   `np.einsum` cannot be used here: its interfaces cap distinct axis labels
   at 52 and `np.broadcast_shapes` caps at 32 dimensions — both below the
   63-unit ceiling (verified against numpy 2.4). Factors of units
   disconnected from the system collapse to scalars that multiply both
   `pr_bg` and `norm` and cancel in the division. At `N ≤ 63` the worst-case
   `norm` is ~1e-19 — no underflow risk in float64 (a 500-unit substrate
   would need log-space; another reason for the ceiling).
4. **Normalization.** `norm = pr_bg.sum()`; `norm <= 0` raises
   `StateUnreachableBackwardsError` exactly as today.
5. **Per-unit outputs.** For each system unit `i`,
   `factor_i * (pr_bg / norm)` broadcast, summed over the background axes
   with `keepdims` — the dense path's own final stage, except the weight
   carries real extent only on `R`. The output keeps the full-ndim
   substrate-global shape the dense path produces today (its real extent is
   also `parents(i) ∩ system`).
6. **Pre-flight size guard.** Each elimination step's merged-product size is
   known from shapes before any allocation. Above a module-level constant
   (`2**27` elements ≈ 1 GB float64), raise an informative exception naming
   the predicted size — a densely coupled large substrate fails fast with an
   explanation instead of OOM-ing the machine.

A useful exactness property: for a **full-substrate system** (no background
units) the weight is exactly `1.0` (`pr_bg` is `norm`), and multiplying by
exact `1.0` is a float identity — so full-substrate results are
**bit-identical** to the dense path. Floating-point drift is possible only
for systems with background units.

`pyphi.serialize` stores no derived marginals (verified: no encoder touches
`System.cause_marginal`), so the wire format is unaffected.

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
  **only for `node_indices`**, as a small value type `CauseMarginals` defined
  in `marginalization.py`: substrate-global full-ndim factors exposed via
  `.factor(i)` and `.indices`, with array-aware `__eq__`/`__hash__`.
- The `.factor(i)` accessor is deliberately the same surface `FactoredTPM`
  exposes, so the two existing consumers need no code change: `Node` reads
  `cause_marginal.factor(self.index)`, and `MacroSystem`'s override (which
  returns a `FactoredTPM` — a macro system has no background units) stays
  duck-type compatible. Both IIT 3.0 and 4.0 route through this one path.
  Pre-release code: no back-compat shims; annotations and the
  marginalization dispatch tests are updated in place.
- `System.cause_marginal` keeps its name and its substrate-global axis
  convention; its value becomes the system-unit `CauseMarginals`.
- `proper_cause_marginal` (squeezed, system-local `FactoredTPM`) is unchanged
  in meaning and shape, but now **derives from** `cause_marginal` instead of
  running its own inversion — today the two cached properties invert
  independently, so this removes a duplicate full inversion per system.
  Display already uses only the proper marginals.
- `cause_marginal()` in `marginalization.py` (the public dispatcher) keeps
  accepting any `TPM` type; the joint/array branches convert to
  `FactoredTPM` first, as today.

### Part 3 — the dense implementation becomes the test oracle

The current dense implementation moves verbatim into a test helper
(`dense_cause_marginal_reference`) under `test/`. It leaves production but
keeps guarding it through cross-validation.

## Validation protocol

This touches a correctness-critical hot path. The greedy contraction computes
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
   system subsets — and asserts the reduced result matches the dense oracle
   within 1e-12 (also recording whether agreement is exact). Symmetric
   fixtures hide axis-order errors; asymmetric and k-ary cases are mandatory.
3. **New-capability tests with independent references.** Truncating a
   substrate to the system's ancestor closure is *not* a valid reference:
   descendants of system units carry evidence about system pasts through
   their observed states, so removing them changes the background weights.
   Only whole factor-graph components disconnected from the system cancel
   exactly (as constants in both `pr_bg` and `norm`). Two large-`N` checks
   that do not trust the new code:
   - **Disconnected-block substrate**: an 8-unit block containing the system
     plus a separate 32-unit block with no cross-edges (40 units total;
     dense evaluation impossible at 2^40). The reduced result must match the
     dense oracle run on the 8-unit block alone (the disconnected block's
     contribution cancels; agreement to ~1e-15, not bit-exact, because the
     cancellation is a float division).
   - **Connected 40-unit chain** with a small embedded system, checked
     against a transfer-matrix computation written directly in the test —
     an independent sequential evaluation of the same sum-product, valid
     because a chain's elimination order is trivial.
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
  computes in well under a second and matches the independent references
  (disconnected-block dense oracle; chain transfer-matrix) within 1e-12.
- After Part 0, one SIA computes the inversion once (not once per
  partition); after Part 1, the inversion cost no longer scales with `a^N`
  for fixed system size and sparse coupling.
- Changelog fragments: one `optimization` fragment for the `apply_cut` cache
  sharing, one for the reduced inversion.
