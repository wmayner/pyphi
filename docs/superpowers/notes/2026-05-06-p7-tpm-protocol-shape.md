# P7 TPM Protocol shape — cross-check against PR #105

**Date:** 2026-05-06
**Purpose:** Validate that the `core.tpm.TPM` Protocol designed for P7
admits both today's `ExplicitTPM` and PR #105's `ImplicitTPM`.

## Operations the current Subsystem performs on TPMs

Direct grep of `pyphi/subsystem.py`:

| Operation | Site | Notes |
|---|---|---|
| `tpm.condition_tpm(background_conditions)` | `Subsystem.__init__:136` | builds effect_tpm |
| `_backward_tpm(network.tpm, state, node_indices)` | `Subsystem.__init__:141` | free function in `pyphi.tpm`; builds cause_tpm |
| `tpm.squeeze()` | `:144,145` | drops singleton dims |
| `tpm.shape` | `:246,248` | shape introspection |
| `tpm.condition_tpm(condition)` | `Subsystem._cause/effect_repertoire:431,433` | per-purview conditioning during repertoire computation |
| `np.asarray(tpm)` | implicit via numpy ops | raw array access |

Also: `tpm[..., list(node_indices)]` style indexing — this is numpy
indexing on the underlying array, not a TPM-method operation. It works
because `ExplicitTPM` is `ArrayLike`. Implicit TPM would need to expose
the same indexing surface or an equivalent.

## Operations PR #105's ImplicitTPM provides

From `gh pr diff 105 -- pyphi/tpm.py`:

| Method | Both? | Notes |
|---|---|---|
| `condition_tpm(condition: Mapping[int, int])` | ✅ both | shape-typed argument |
| `marginalize_out(node_indices)` | ✅ both | |
| `backward_tpm(current_state, system_indices)` | ✅ both | becomes a TPM method, not free function |
| `squeeze(dims=None)` | ImplicitTPM only | optional `dims` argument unique to Implicit |
| `is_state_by_state()` | ✅ both | |
| `remove_singleton_dimensions()` | ✅ both | redundant with squeeze; legacy |
| `probability_of_current_state(state)` | ✅ both | |
| `to_multidimensional_state_by_node()` | ✅ both | format conversion |
| `validate(check_independence=True)` | ✅ both | |
| `shape` (property) | ✅ both | |
| `ndim` (property) | Implicit only | numpy-mirror |
| `node_indices` (property) | Implicit only | |
| `tpm` (property) | Implicit only | reconstituted explicit form |

## Common shape — the Protocol body

The intersection of (a) operations subsystem.py performs on TPMs and
(b) operations both ExplicitTPM and ImplicitTPM provide:

```python
@runtime_checkable
class TPM(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def n_nodes(self) -> int: ...

    def condition(self, fixed: Mapping[int, int]) -> "TPM": ...

    def squeeze(self) -> "TPM": ...

    def to_array(self) -> NDArray[np.float64]: ...
```

Naming: I use `condition` not `condition_tpm` because the latter is
redundant in a TPM Protocol context. I use `to_array` not `__array__`
because `__array__` is implicit via numpy's array protocol; making it
explicit avoids surprise when implicit conversion drops metadata.

## Operations deferred to P12

These belong on the Protocol but require non-binary support to land
first; deferred to P12:

- `marginalize_out(node_indices)` — used by macro coarse-graining and
  partition evaluation. Currently subsystem.py uses it indirectly via
  the legacy `_backward_tpm` free function. Keeping the operation
  inside `core/tpm/marginalization.py` for P7 — it can move onto the
  Protocol in P12 once `ImplicitTPM` is the primary use case
- `alphabet_size` per axis — non-binary state spaces. Binary is
  implicit in P7
- `probability_of_current_state(state)` — used by IIT 4.0 forward
  probabilities. Currently a free function path through legacy
  `pyphi.repertoire`; if we add it to the Protocol, both backends
  satisfy. **Recommendation: include in P7's Protocol as an optional
  method** since both already implement it
- `to_multidimensional_state_by_node()` — format conversion needed when
  a state-by-state TPM is provided. Used by `Network.__init__`.
  P12-relevant only

## Operations explicitly NOT on the Protocol

- `validate(...)` — implementation-internal
- `print()`, `__str__`, `__repr__` — display, not contract
- `__hash__` — implementation chooses

## Decision

**Adopt the Protocol body shown in the spec** (Components section, see
`docs/superpowers/specs/2026-05-06-p7-subsystem-layered-rewrite-design.md`).

Add `probability_of_current_state` to the Protocol — both backends
already implement it; refusing to put it on the Protocol pushes callers
into duck-typing.

The `marginalization` operations live as free functions in
`core/tpm/marginalization.py` rather than on the Protocol — they sit
*above* the TPM, transforming one TPM into another. This matches the
ROADMAP architecture (`core/tpm/marginalization.py`).

## What this confirms for P7

PR #105's `ImplicitTPM` will satisfy our Protocol once it implements
the four core methods (`shape`, `condition`, `squeeze`, `to_array`).
Today its method names are `condition_tpm` (not `condition`) and
`squeeze` (with extra optional `dims` arg). When P12 reconciles, the
adapter is:

```python
class ImplicitTPM(TPM):
    def condition(self, fixed):
        return self.condition_tpm(fixed)
    # squeeze already matches (optional positional arg compatible)
    # shape, n_nodes already match
```

This is mechanical — no Protocol redesign needed at P12. ✓
