# FactoredTPM sparse-factor crashes — design

## Problem

A `FactoredTPM` stores each node's conditional as a factor of shape
`(a_1, …, a_N, a_i)`, where an input axis the node does **not** depend on is
kept at size 1. The module docstring states this explicitly: "these size-1 axes
encode the connectivity structure and are never squeezed away." This
connectivity-sparse form is a documented, validation-accepted input
(`_validate` accepts `dim_size in (1, a[j])`), and it is preserved into a
`Substrate` (`Substrate.from_factored` keeps the singleton dims).

Two independent code paths assume every factor has its full extent along every
input axis, so both crash on the sparse form. They share one root cause: a
size-1 non-input axis is indexed or flattened as if it had the node's full
alphabet size.

### Bug A — `select` crashes when conditioning a size-1 axis at a nonzero state

`_NdarrayBackend.select` indexes every fixed axis with the raw conditioning
state (`idx[j] = state_j`), and `_XarrayBackend.select` does the same via
`isel`. When the fixed unit is a non-input of factor `i`, that axis is size 1,
so any nonzero state raises `IndexError: index 1 is out of bounds for axis j
with size 1`.

This is reachable through ordinary subsystem analysis. When a background unit
is fixed in a nonzero state and is a non-input of a free unit's factor, the
background-conditioning path crashes:

```python
System(substrate, (0, 0, 1), node_indices=(0, 1)).sia()
# IndexError: index 1 is out of bounds for axis 2 with size 1
```

The same crash occurs directly in `FactoredTPM.condition`, `subtpm`,
`condition_factor`, and (through them) `cause_conditioned` and
`effect_marginal`. It is silent-safe only when the fixed state happens to be 0,
which is why every all-zero and fixed-point fixture in the suite missed it.

### Bug B — macro TPM construction flattens the wrong shape

`_discounted_on_probabilities` (`pyphi/macro/tpm.py`) reads
`p_on = factored.factor(i)[..., 1]` — the factor's input shape *including* its
singleton axes — and flattens each column with `reshape(-1, order="F")`. On a
sparse factor this yields a column of length `prod(non-singleton dims)` instead
of the required `2**n` universe size. The per-unit ON-probability matrix comes
out the wrong size (`(2, n)` instead of `(2**n, n)` on the confirming example),
and `MacroSystem.from_micro` / `macro_tpms` then crash at the transition-matrix
matmul:

```python
MacroSystem.from_micro(substrate, (unit,), (0, 0, 0))
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
#             (size 2 is different from 8)
```

Reachable via the public `MacroSystem.from_micro` and `macro_tpms` entry points.

## Why the fixes are forced (no formalism choice)

A size-1 input axis means factor `i`'s conditional distribution is constant in
that input — the node is mathematically independent of it. Therefore:

- **Conditioning** on that input yields the same slice regardless of the fixed
  state, so `select` must return the single stored slice (index 0).
- **Flattening** to the universe grid must expand the constant axis to the
  node's full alphabet size (broadcasting replicates the constant value).

Both are the unique correct behaviors, not design options. The existing code
already encodes this understanding elsewhere: `_varies_along_axis`
(`factored.py:296`) treats a size-1 axis as constant by definition.

## Fix A — `select` clamps size-1 axes to index 0

In both storage backends, index a fixed axis at 0 when it is size 1, otherwise
at the raw state. The `expand_dims` restoration (which both backends already do)
and `subtpm`'s subsequent squeeze integrate unchanged.

`_NdarrayBackend.select`:

```python
for j, state_j in fixed.items():
    idx[j] = 0 if factor.shape[j] == 1 else state_j
```

`_XarrayBackend.select`:

```python
idx = {
    f"in_{j}": (0 if factor.sizes[f"in_{j}"] == 1 else state_j)
    for j, state_j in fixed.items()
}
```

One change per backend covers every downstream consumer:
`condition`, `condition_factor`, `subtpm`, `cause_conditioned`,
`effect_marginal`, and full `System` construction.

## Fix B — macro broadcasts to the full universe shape before flattening

In `_discounted_on_probabilities`, broadcast the ON-probability slice to the
full input grid at the read site:

```python
p_on = np.broadcast_to(factored.factor(i)[..., 1], factored.alphabet_sizes)
```

Every column then flattens to length `prod(alphabet_sizes) == 2**n`. The three
downstream branches are unaffected: `.mean()` over the broadcast array equals
the mean over the stored array (uniform replication preserves the grand mean),
and `mean(axis=…, keepdims=True)` followed by `broadcast_to(p_on.shape)`
likewise preserves the per-axis means. `reshape(-1, order="F")` copies the
non-contiguous broadcast view, so no aliasing results.

Macro construction is binary by design (the Eq. 26–30 machinery slices `[..., 1]`
as an ON probability and works over `2**n` states), so `alphabet_sizes` is
`(2,) * n` here; the k-ary case is explicitly out of scope (see below).

## Tests

- **Bug A regression:** a connectivity-sparse `FactoredTPM` (a factor with a
  size-1 non-input axis) conditioned at a nonzero state — assert
  `subtpm`/`condition` succeed and return the correct slice, and assert
  `System(substrate, state, node_indices=…).sia()` runs (nonzero background unit
  on a size-1 axis). A control at the all-zero state must give the identical
  result, pinning that the fix does not change behavior where the old code
  already worked.
- **Bug B regression:** `MacroSystem.from_micro` on a singleton-dim substrate,
  compared against the identical substrate with factors broadcast to full
  `(2,)*n` shape (the control). The two must produce bit-equal macro TPMs.
- **Stale docstring:** `test/core/test_repertoire_sparse_heterogeneous.py` has a
  module docstring claiming these inputs "currently raise"; update it to reflect
  that conditioning connectivity-sparse factors at nonzero states is supported.

## Verification gate

Run pathless `uv run pytest` (no path argument) in the worktree so the doctest
sweep and the full suite both run. The worktree venv needs the optional extras
installed first (`visualize,caching,emd,xarray` plus `pot`) or the pathless
sweep fails to collect. If the perf-counter pin
(`test/data/perf/call_counts.json`) moves, regenerate it via
`scripts/gen_perf_counts.py` and review the diff (the select fix should not
change call counts on the non-sparse fixtures; a change there is a red flag).

## Out of scope

These belong to the separate "macro/matching precondition gaps" review item, not
this one:

- `build_triggered_tpm` accepting k-ary substrates and unsorted `system_indices`.
- `macro_tpms` reusing the current state as the earliest window state when
  `micro_history` is one entry short (negative-index wraparound).
- `MacroUnit` accepting silently-inert negative apportionment indices.

## Follow-up bookkeeping

Mark the two findings resolved in the review file's status block
(`REVIEW-2026-07-13.md`) and add a Wave-1 completion note; add a changelog
fragment under `changelog.d/`.
