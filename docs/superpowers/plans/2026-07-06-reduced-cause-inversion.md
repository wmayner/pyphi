# Reduced-Dimension Cause Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cause-side Bayesian inversion (IIT 4.0 Eq. 4) tractable for small systems embedded in large sparse substrates, and stop recomputing it once per partition.

**Architecture:** Two independent changes per the spec (`docs/superpowers/specs/2026-07-06-reduced-cause-inversion-design.md`): (1) `System.apply_cut` shares partition-independent marginal caches with the cut system; (2) `_cause_marginal_factored` becomes a greedy sum-product contraction over the factored TPM's dependence structure (full-ndim arrays with size-1 non-parent axes; ufunc broadcasting aligns factors, `sum(keepdims=True)` eliminates axes), returning a new `CauseMarginals` value type holding factors only for system units. The dense implementation survives as a test oracle.

**Tech Stack:** numpy (ufunc broadcasting only — `np.einsum` is unusable here: 52-distinct-label cap; `np.broadcast_shapes` caps at 32 dims), pytest, Hypothesis.

## Global Constraints

- Every commit ends with (verbatim):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Never `--no-verify`. If pre-commit reformats-and-aborts (no `[main <sha>]` line), re-`git add` the same files and re-commit. Read the FULL hook output on failure.
- Never `git add -A`; stage only the files named in each task. Concurrent sessions leave unrelated edits in shared files.
- All Python commands via `uv run`.
- No roadmap item numbers or planning-phase markers in `pyphi/` source, docstrings, or changelog fragments.
- Correctness bar: existing goldens compare scalars **exactly**; any golden failure is investigated per the spec's validation protocol, never silently regenerated.
- Full verification (Task 6): `uv run --all-extras pytest` with NO path argument, run in background with output redirected to the absolute scratchpad path; judge by the `RC=` line.

---

### Task 1: `apply_cut` shares partition-independent caches

**Files:**
- Modify: `pyphi/system.py:256-266` (`System.apply_cut`)
- Test: `test/test_system.py` (add two tests after `test_apply_cut`, ~line 133)
- Create: `changelog.d/apply-cut-cache-sharing.optimization.md`

**Interfaces:**
- Produces: `System.apply_cut` behavior — already-computed entries for `_typed_tpm`, `cause_marginal`, `effect_marginal`, `proper_cause_marginal`, `proper_effect_marginal` are shared (same objects) with the returned cut system. No signature change.

- [ ] **Step 1: Write the failing tests**

In `test/test_system.py`, directly after `test_apply_cut` (imports `np`, `DirectedBipartition`, `Direction` already present at module top):

```python
def test_apply_cut_shares_partition_independent_caches(s):
    cut = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    # Populate the parent's caches first.
    _ = s.cause_marginal
    _ = s.effect_marginal
    cut_s = s.apply_cut(cut)
    assert cut_s.cause_marginal is s.cause_marginal
    assert cut_s.effect_marginal is s.effect_marginal


def test_apply_cut_partition_dependent_state_still_differs(s):
    cut_a = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    cut_b = DirectedBipartition(Direction.EFFECT, (0, 2), (1,))
    sys_a = s.apply_cut(cut_a)
    sys_b = s.apply_cut(cut_b)
    # Genuine difference: the two cuts sever different edges.
    assert not np.array_equal(sys_a.cm, sys_b.cm)
    assert sys_a.nodes != sys_b.nodes
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `uv run pytest test/test_system.py::test_apply_cut_shares_partition_independent_caches test/test_system.py::test_apply_cut_partition_dependent_state_still_differs -v`
Expected: first FAILS on the `is` assertion (fresh instance recomputes); second PASSES (it documents behavior that must survive the change).

- [ ] **Step 3: Implement cache sharing**

In `pyphi/system.py`, replace `apply_cut` (lines 256-266):

```python
    def apply_cut(self, partition: DirectedBipartition) -> System:
        """Return a new System with the given partition applied.

        ``substrate``, ``state``, and ``node_indices`` are unchanged. The
        cause/effect marginals depend only on those inputs — the cut enters
        downstream through the cut connectivity matrix when node TPMs
        marginalize out severed inputs — so any already-computed marginal
        caches are shared with the new instance rather than re-derived.
        """
        from dataclasses import replace

        new = replace(self, partition=partition)
        for name in (
            "_typed_tpm",
            "cause_marginal",
            "effect_marginal",
            "proper_cause_marginal",
            "proper_effect_marginal",
        ):
            if name in self.__dict__:
                new.__dict__[name] = self.__dict__[name]
        return new
```

(`functools.cached_property` stores values in the instance `__dict__`; frozen dataclasses block `__setattr__`, not direct `__dict__` writes.)

- [ ] **Step 4: Run the tests and the fast system/formalism lanes**

Run: `uv run pytest test/test_system.py test/core/ test/formalism/ -q`
Expected: all PASS.

- [ ] **Step 5: Changelog fragment**

Create `changelog.d/apply-cut-cache-sharing.optimization.md`:

```markdown
`System.apply_cut` now shares the partition-independent cause/effect marginal caches with the cut system, so a SIA's partition search computes the cause-side Bayesian inversion once instead of once per partition.
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/system.py test/test_system.py changelog.d/apply-cut-cache-sharing.optimization.md
git commit -m "Share partition-independent marginal caches across apply_cut

The cause/effect marginals depend only on (substrate TPM, state,
node_indices); the partition enters downstream via the cut connectivity
matrix. Previously every cut system re-derived the expensive cause
inversion from identical inputs — once per partition evaluated.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 2: `CauseMarginals` + greedy sum-product inversion (alongside the dense path)

**Files:**
- Modify: `pyphi/exceptions.py` (new exception after `StateUnreachableBackwardsError`, ~line 50)
- Modify: `pyphi/core/tpm/marginalization.py` (add `CauseMarginals`, `_sum_product`, `_cause_marginal_reduced`; dense `_cause_marginal_factored` untouched in this task)
- Test: `test/core/test_cause_inversion.py` (new)

**Interfaces:**
- Produces:
  - `pyphi.exceptions.IntractableCauseInversionError(ValueError)`.
  - `CauseMarginals` — constructed from `Mapping[int, NDArray[np.float64]]`; exposes `.indices -> tuple[int, ...]`, `.factor(i) -> NDArray[np.float64]`, array-aware `__eq__`/`__hash__`.
  - `_cause_marginal_reduced(factored: FactoredTPM, state: tuple[int, ...], node_indices: tuple[int, ...]) -> CauseMarginals` — factors only for `node_indices`, each in full-ndim substrate-global shape `(*sizes_with_1s, k_i)`, identical shape to the dense path's factor for the same unit.
- Consumes: existing dense `_cause_marginal_factored` (as the in-tree comparison target for this task only).

- [ ] **Step 1: Write the failing tests**

Create `test/core/test_cause_inversion.py`:

```python
"""Reduced cause inversion (greedy sum-product) vs the dense implementation."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import exceptions
from pyphi.core.tpm import marginalization
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored
from pyphi.core.tpm.marginalization import _cause_marginal_reduced
from pyphi.core.tpm.marginalization import CauseMarginals


def _normalized(raw: np.ndarray) -> np.ndarray:
    return raw / raw.sum(axis=-1, keepdims=True)


def _asymmetric_binary_factored(seed: int = 5) -> FactoredTPM:
    """5 binary units with deliberately unequal parent sets.

    parents: 0 <- {0, 3}; 1 <- {0, 1}; 2 <- {1}; 3 <- {2, 3, 4}; 4 <- {4}.
    Asymmetric on purpose: symmetric fixtures hide axis-order errors.
    """
    rng = np.random.default_rng(seed)
    parent_sets = [{0, 3}, {0, 1}, {1}, {2, 3, 4}, {4}]
    factors = []
    for i, parents in enumerate(parent_sets):
        shape = tuple(2 if j in parents else 1 for j in range(5))
        factors.append(_normalized(rng.random((*shape, 2)) + 1e-3))
    return FactoredTPM(factors=factors)


def _kary_factored(seed: int = 7) -> FactoredTPM:
    """4 units with alphabets (2, 3, 2, 4) and unequal parent sets."""
    rng = np.random.default_rng(seed)
    alphabets = (2, 3, 2, 4)
    parent_sets = [{0, 1}, {1, 3}, {0, 2, 3}, {2}]
    factors = []
    for i, parents in enumerate(parent_sets):
        shape = tuple(alphabets[j] if j in parents else 1 for j in range(4))
        factors.append(_normalized(rng.random((*shape, alphabets[i])) + 1e-3))
    return FactoredTPM(
        factors=factors,
        state_space=tuple(tuple(range(a)) for a in alphabets),
    )


def _dense_factors(factored, state, node_indices):
    dense = _cause_marginal_factored(factored, state, node_indices)
    return {i: dense.factor(i) for i in node_indices}


@pytest.mark.parametrize("state", [(0, 1, 0, 1, 0), (1, 0, 1, 1, 1)])
@pytest.mark.parametrize("system", [(1, 2), (0, 3, 4), (2,)])
def test_reduced_matches_dense_asymmetric_binary(state, system):
    factored = _asymmetric_binary_factored()
    reduced = _cause_marginal_reduced(factored, state, system)
    dense = _dense_factors(factored, state, system)
    assert reduced.indices == system
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(reduced.factor(i), dense[i], rtol=0, atol=1e-13)


@pytest.mark.parametrize("state", [(0, 2, 1, 3), (1, 0, 0, 2)])
@pytest.mark.parametrize("system", [(0, 2), (1,), (1, 2, 3)])
def test_reduced_matches_dense_kary(state, system):
    factored = _kary_factored()
    reduced = _cause_marginal_reduced(factored, state, system)
    dense = _dense_factors(factored, state, system)
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(reduced.factor(i), dense[i], rtol=0, atol=1e-13)


def test_different_states_give_different_marginals():
    """Genuine-difference guard: the comparison tests above must not be
    vacuously passing on state-independent outputs."""
    factored = _asymmetric_binary_factored()
    system = (1, 2)
    a = _cause_marginal_reduced(factored, (0, 1, 0, 1, 0), system)
    b = _cause_marginal_reduced(factored, (1, 0, 1, 1, 1), system)
    assert any(not np.allclose(a.factor(i), b.factor(i)) for i in system)


def test_full_substrate_system_is_bit_identical_to_dense():
    """With no background units the weight is exactly 1.0, so the reduced
    path must reproduce the dense path bit-for-bit."""
    factored = _asymmetric_binary_factored()
    state = (0, 1, 1, 0, 1)
    system = (0, 1, 2, 3, 4)
    reduced = _cause_marginal_reduced(factored, state, system)
    dense = _dense_factors(factored, state, system)
    for i in system:
        assert np.array_equal(reduced.factor(i), dense[i])


def test_unreachable_state_raises():
    factors = [np.zeros((2, 2, 2)) for _ in range(2)]
    for f in factors:
        f[..., 0] = 1.0  # every unit always outputs 0
    factored = FactoredTPM(factors=factors)
    with pytest.raises(exceptions.StateUnreachableBackwardsError):
        _cause_marginal_reduced(factored, state=(1, 1), node_indices=(0, 1))


def test_intractable_contraction_raises(monkeypatch):
    """All-to-all coupling with a tiny cap: every elimination step exceeds it."""
    rng = np.random.default_rng(11)
    n = 6
    factors = [_normalized(rng.random(((2,) * n) + (2,)) + 1e-3) for _ in range(n)]
    factored = FactoredTPM(factors=factors)
    monkeypatch.setattr(marginalization, "_MAX_INTERMEDIATE_ELEMENTS", 8)
    with pytest.raises(exceptions.IntractableCauseInversionError, match=r"\d+"):
        _cause_marginal_reduced(factored, state=(0,) * n, node_indices=(0, 1))


def test_cause_marginals_value_semantics():
    rng = np.random.default_rng(3)
    f = rng.random((2, 1, 2))
    a = CauseMarginals({0: f})
    b = CauseMarginals({0: f.copy()})
    c = CauseMarginals({0: rng.random((2, 1, 2))})
    d = CauseMarginals({1: f})
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d
    assert a.indices == (0,)
    assert a.factor(0) is f
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/core/test_cause_inversion.py -q`
Expected: collection error — `ImportError: cannot import name '_cause_marginal_reduced'`.

- [ ] **Step 3: Add the exception**

In `pyphi/exceptions.py`, after `StateUnreachableBackwardsError`:

```python
class IntractableCauseInversionError(ValueError):
    """The cause inversion cannot proceed within the intermediate-size cap.

    Raised when every remaining step of the sum-product contraction would
    materialize an intermediate array larger than the cap — the substrate's
    coupling is too dense for the reduced inversion at this size.
    """
```

- [ ] **Step 4: Implement `CauseMarginals`, `_sum_product`, `_cause_marginal_reduced`**

In `pyphi/core/tpm/marginalization.py`. Add `from numpy.typing import NDArray` to imports. Below the module docstring / imports:

```python
# Cap on any single intermediate array in the sum-product contraction
# (~1 GiB of float64). Densely coupled substrates whose cheapest elimination
# step exceeds this fail fast with an informative error instead of OOM.
_MAX_INTERMEDIATE_ELEMENTS = 2**27


class CauseMarginals:
    """Cause factors for a set of output units — IIT 4.0 Eq. 4.

    Maps each output unit ``i`` to its cause factor of shape
    ``(*alphabet_sizes, k_i)`` in the substrate-global axis convention
    (size-1 input axes mark non-dependence, exactly as in
    :class:`~pyphi.core.tpm.factored.FactoredTPM` factors, and
    ``.factor(i)`` mirrors that class's accessor). Holds only the
    requested output units.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: Mapping[int, NDArray[np.float64]]) -> None:
        self._factors = dict(factors)

    @property
    def indices(self) -> tuple[int, ...]:
        """The output-unit indices, ascending."""
        return tuple(sorted(self._factors))

    def factor(self, i: int) -> NDArray[np.float64]:
        """The cause factor for output unit ``i``."""
        return self._factors[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseMarginals):
            return NotImplemented
        return self._factors.keys() == other._factors.keys() and all(
            np.array_equal(self._factors[i], other._factors[i])
            for i in self._factors
        )

    def __hash__(self) -> int:
        return hash(
            tuple(
                (i, self._factors[i].shape, (self._factors[i] + 0.0).tobytes())
                for i in sorted(self._factors)
            )
        )


def _check_intermediate(size: int) -> None:
    if size > _MAX_INTERMEDIATE_ELEMENTS:
        raise exceptions.IntractableCauseInversionError(
            f"cause inversion would materialize an intermediate of {size} "
            f"elements (cap: {_MAX_INTERMEDIATE_ELEMENTS}); the substrate's "
            f"coupling is too dense for the reduced inversion"
        )


def _merged_elements(shapes: list[tuple[int, ...]]) -> int:
    """Element count of the broadcast product of arrays with these shapes."""
    size = 1
    for k in range(len(shapes[0])):
        size *= max(s[k] for s in shapes)
    return size


def _sum_product(
    slices: list[NDArray[np.float64]],
    keep_axes: frozenset[int],
) -> NDArray[np.float64]:
    """Marginal of ``∏ slices`` over ``keep_axes`` by greedy elimination.

    All arrays are full-ndim with size-1 axes marking non-dependence, so
    ufunc broadcasting aligns factors with no explicit axis bookkeeping
    (valid up to numpy's 64-dimension limit; ``np.einsum`` and
    ``np.broadcast_shapes`` have lower caps and cannot be used here). Each
    step eliminates the axis whose merged product of involved slices is
    smallest, ties breaking toward the lowest axis index — deterministic
    given shapes.
    """
    factors = list(slices)
    n = factors[0].ndim
    remaining = [k for k in range(n) if k not in keep_axes]
    while remaining:
        best_axis = None
        best_size = None
        for k in remaining:
            shapes = [f.shape for f in factors if f.shape[k] > 1]
            size = _merged_elements(shapes) if shapes else 0
            if best_size is None or size < best_size:
                best_axis, best_size = k, size
        _check_intermediate(best_size)
        remaining.remove(best_axis)
        involved = [f for f in factors if f.shape[best_axis] > 1]
        rest = [f for f in factors if f.shape[best_axis] == 1]
        if involved:
            prod = involved[0]
            for f in involved[1:]:
                prod = prod * f
            rest.append(prod.sum(axis=best_axis, keepdims=True))
        factors = rest
    _check_intermediate(_merged_elements([f.shape for f in factors]))
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _cause_marginal_reduced(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CauseMarginals:
    """Cause factors for the system units — IIT 4.0 Eq. 4.

    For each system unit ``i`` and output value ``s_i``:

        factor_i(s_t)[s_i] = Σ_{w_t} P(s_i | s_t, w_t) · (pr_bg(s_t, w_t) / norm)

    where ``pr_bg`` is the joint likelihood of the observed state summed
    over the system past, ``norm`` sums it over all past states, and the
    outer sum runs over background past states. Evaluated as a sum-product
    contraction over the factored TPM's dependence structure: the joint
    likelihood is never materialized over all substrate units, and the
    background weight carries real extent only on background axes some
    system factor depends on. Factors are returned only for output units in
    ``node_indices``.
    """
    n = factored.n_nodes
    system = frozenset(node_indices)
    background_axes = tuple(k for k in range(n) if k not in system)

    # Per-unit likelihood of the observed state given the past, full-ndim
    # with size-1 non-parent axes: factor_j(s_t)[state_j].
    slices = [factored.factor(j)[..., state[j]] for j in range(n)]

    # Background axes some system factor actually depends on — the only
    # axes on which the outputs can see the weight.
    relevant = frozenset(
        k
        for i in node_indices
        for k, dim in enumerate(factored.factor(i).shape[:-1])
        if dim > 1 and k not in system
    )

    pr_bg = _sum_product(slices, keep_axes=relevant)
    norm = pr_bg.sum()
    if norm <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(state)
    weight = pr_bg / norm

    out_factors: dict[int, NDArray[np.float64]] = {}
    for i in node_indices:
        forward_i = factored.factor(i)
        _check_intermediate(
            _merged_elements([forward_i.shape, (*weight.shape, 1)])
        )
        weighted = forward_i * weight[..., np.newaxis]
        if background_axes:
            weighted = weighted.sum(axis=background_axes, keepdims=True)
        out_factors[i] = weighted
    return CauseMarginals(out_factors)
```

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest test/core/test_cause_inversion.py -v`
Expected: all PASS. If `test_full_substrate_system_is_bit_identical_to_dense` fails, the weight is not collapsing to exact `1.0` — debug before proceeding (this exactness is load-bearing for the golden A/B).

- [ ] **Step 6: Run ruff + pyright on the touched files**

Run: `uv run ruff check pyphi/core/tpm/marginalization.py pyphi/exceptions.py test/core/test_cause_inversion.py && uv run pyright pyphi/core/tpm/marginalization.py pyphi/exceptions.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add pyphi/exceptions.py pyphi/core/tpm/marginalization.py test/core/test_cause_inversion.py
git commit -m "Add greedy sum-product cause inversion alongside the dense path

The reduced inversion evaluates IIT 4.0 Eq. 4 as a variable-elimination
contraction over the factored TPM's dependence structure (size-1 axes
mark non-parents; broadcasting aligns factors), computing output factors
only for the requested system units. Cross-validated against the dense
implementation on asymmetric binary and k-ary cases; bit-identical for
full-substrate systems.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 3: Rewire production to the reduced path; dense becomes the oracle

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py` (delete dense `_cause_marginal_factored`; rename `_cause_marginal_reduced` → `_cause_marginal_factored`; update `cause_marginal` dispatcher docstring/annotation)
- Modify: `pyphi/system.py:38` (drop stale import), `:313-320` (`cause_marginal`), `:353-378` (`proper_cause_marginal`)
- Modify: `pyphi/node.py:20-43` (docstring only)
- Create: `test/core/inversion_oracle.py` (dense reference)
- Modify: `test/core/test_cause_inversion.py` (repoint at the oracle)
- Modify: `test/core/test_marginalization_factored.py`, `test/core/test_marginalization_kary.py`, `test/test_system.py:286-300` (return-type expectations)

**Interfaces:**
- Produces:
  - `_cause_marginal_factored(factored, state, node_indices) -> CauseMarginals` (the reduced implementation, final name).
  - `cause_marginal(tpm, state, node_indices) -> CauseMarginals` (dispatcher; joint/array inputs convert to `FactoredTPM` first, as today).
  - `System.cause_marginal -> CauseMarginals` (cached property; system units only).
  - `System.proper_cause_marginal -> FactoredTPM` — unchanged meaning/shape, now derived from `cause_marginal` (one inversion per system instead of two).
  - `test/core/inversion_oracle.py::dense_cause_marginal_reference(factored, state, node_indices) -> dict[int, np.ndarray]`.
- Consumes: `CauseMarginals`, `_cause_marginal_reduced` from Task 2. `Node` and `MacroSystem` consume `.factor(i)` and need no code change.

- [ ] **Step 1: Create the oracle from the dense implementation**

Create `test/core/inversion_oracle.py` (body is the pre-Task-3 dense implementation, returning a dict):

```python
"""Dense reference for the cause inversion (IIT 4.0 Eq. 4).

The production implementation is a greedy sum-product contraction; this
oracle materializes the full joint likelihood over all substrate units
(a^N) and computes the same quantities directly. Only usable for small
substrates, which is exactly its job: an independent implementation the
reduced path is cross-validated against.
"""

from __future__ import annotations

import numpy as np


def dense_cause_marginal_reference(factored, state, node_indices):
    """Return ``{unit index: cause factor}`` for units in ``node_indices``."""
    n = factored.n_nodes
    alphabet_sizes = factored.alphabet_sizes
    all_indices = tuple(range(n))
    system_indices = tuple(sorted(node_indices))
    background_indices = tuple(sorted(set(all_indices) - set(system_indices)))

    pr_joint = np.ones(alphabet_sizes, dtype=np.float64)
    for i in all_indices:
        pr_joint = pr_joint * factored.factor(i)[..., state[i]]

    if system_indices:
        pr_bg = pr_joint.sum(axis=system_indices, keepdims=True)
    else:
        pr_bg = pr_joint.copy()

    norm = pr_joint.sum()
    assert norm > 0.0, "oracle: unreachable state"
    weight = pr_bg / norm

    out = {}
    for i in node_indices:
        weighted = factored.factor(i) * weight[..., np.newaxis]
        if background_indices:
            weighted = weighted.sum(axis=background_indices, keepdims=True)
        out[i] = weighted
    return out
```

- [ ] **Step 2: Repoint the Task 2 tests at the oracle**

In `test/core/test_cause_inversion.py`:
- Replace the import of the dense production function and the `_dense_factors` helper:

```python
from test.core.inversion_oracle import dense_cause_marginal_reference as _dense_factors
```

(delete the old `_dense_factors` def and the `from pyphi.core.tpm.marginalization import _cause_marginal_factored` import). If `from test.core...` fails to resolve under the repo's pytest import mode, use a relative import `from .inversion_oracle import ...` — match whatever other `test/core/` cross-imports do.
- Replace every `_cause_marginal_reduced` with `_cause_marginal_factored` (import and call sites).

- [ ] **Step 3: Swap the production implementation**

In `pyphi/core/tpm/marginalization.py`:
- Delete the dense `_cause_marginal_factored` (the version building `pr_joint = np.ones(alphabet_sizes)`).
- Rename `_cause_marginal_reduced` to `_cause_marginal_factored`.
- Update the `cause_marginal` dispatcher's annotation and docstring:

```python
def cause_marginal(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CauseMarginals:
    """Cause factors for the system units — IIT 4.0 Eq. 4.

    Returns a :class:`CauseMarginals` mapping each unit in ``node_indices``
    to its cause factor of shape ``(*alphabet_sizes, k_i)`` in the
    substrate-global axis convention: ``P(s_i,t | s_{M,t+1} = state_M)``
    per output unit, with background units marginalized under
    ``pr_bg / norm`` weighting. Joint/array inputs are converted to
    :class:`~pyphi.core.tpm.factored.FactoredTPM` first.
    """
```

(dispatch body unchanged).

- [ ] **Step 4: Rewire `System`**

In `pyphi/system.py`:
- Line 38: delete `from .core.tpm.marginalization import _cause_marginal_factored`. Add `from .core.tpm.marginalization import CauseMarginals` (keep the `cause_marginal as _marginalize_cause` import).
- Replace the `cause_marginal` property (~line 313):

```python
    @cached_property
    def cause_marginal(self) -> CauseMarginals:
        """Per-system-unit cause factors; see IIT 4.0 Eq. 4."""
        return _marginalize_cause(
            self._typed_tpm,
            self.state,
            self.node_indices,
        )
```

- Replace `proper_cause_marginal` (~line 353) to derive from `cause_marginal`:

```python
    @cached_property
    def proper_cause_marginal(self) -> FactoredTPM:
        """Cause TPM restricted to system units.

        Per system unit ``i`` in ``node_indices``, the returned FactoredTPM
        carries the cause factor produced by Bayesian inversion of the
        substrate's forward TPM under the observed state. Background units
        are marginalized via ``pr_bg / norm`` weighting per IIT 4.0 Eq. 4
        and dropped from each factor's input dims, so the returned shape
        is ``(*system_alphabet, k_i)`` per system output unit.
        """
        marginals = self.cause_marginal
        background_indices = tuple(
            i
            for i in range(self._typed_tpm.n_nodes)
            if i not in set(self.node_indices)
        )
        system_factors = []
        for i in self.node_indices:
            f = marginals.factor(i)
            if background_indices:
                f = np.squeeze(f, axis=background_indices)
            system_factors.append(f)
        return FactoredTPM(factors=system_factors, node_labels=self._unit_labels())
```

- [ ] **Step 5: Update the `Node`/`generate_nodes` docstrings**

In `pyphi/node.py`, the `Node` class docstring `Args` section currently says `cause_marginal (JointTPM): The cause (backward) TPM of the system.` Change the cause entry (and the matching line in the `generate_nodes` docstring) to:

```
cause_marginal (CauseMarginals): Per-system-unit cause factors; this
    node reads its own factor via ``cause_marginal.factor(index)``.
```

No code change: `Node.__init__` already calls `cause_marginal.factor(self.index)`, which `CauseMarginals` provides (and `MacroSystem`'s `FactoredTPM` override also provides).

- [ ] **Step 6: Update the return-type expectations in existing tests**

- `test/core/test_marginalization_factored.py`:
  - Add `from pyphi.core.tpm.marginalization import CauseMarginals` to imports.
  - `test_cause_marginal_factored_dispatch_matches_joint`: change both `isinstance(..., FactoredTPM)` asserts to `isinstance(..., CauseMarginals)`; change the comparison loop to `for i in node_indices:`.
  - `test_cause_marginal_returns_factored_tpm_for_jointtpm_input` → rename to `test_cause_marginal_returns_cause_marginals_for_jointtpm_input`, assert `isinstance(result, CauseMarginals)`; same for the `_factored_input` twin.
- `test/core/test_marginalization_kary.py`:
  - `test_cause_marginal_factored_returns_factored_tpm` → rename to `test_cause_marginal_factored_returns_cause_marginals`, assert `isinstance(result, CauseMarginals)`.
  - `test_cause_marginal_factored_per_factor_sums_to_one`: replace `for i in range(result.n_nodes):` with `for i in result.indices:`.
  - `test_cause_marginal_factored_binary_gives_valid_distribution`: replace `for i in range(factored.n_nodes):` with `for i in result.indices:`.
  - (The subset-system, unreachable-state, and repertoire tests already use `.factor(i)` on requested units and need no change.)
- `test/test_system.py::test_proper_cause_marginal_binary_matches_legacy_slice` (~line 286): change `assert isinstance(substrate_cause, FactoredTPM)` to `assert isinstance(substrate_cause, CauseMarginals)` (add the import at use site: `from pyphi.core.tpm.marginalization import CauseMarginals`). The `.factor(node)` access is unchanged.

- [ ] **Step 7: Run the affected lanes**

Run: `uv run pytest test/core/ test/test_system.py test/test_node.py test/macro/ -q`
Expected: all PASS (macro tests confirm the `MacroSystem` duck-type path).

- [ ] **Step 8: Fast integration signal**

Run: `uv run pytest test/integration/test_golden_regression.py test/formalism/ -q`
Expected: all PASS. Any golden failure here means background-carrying goldens drifted — STOP and evaluate per the spec's validation protocol before committing (drift magnitude, discrete flips). Do not regenerate goldens.

- [ ] **Step 9: ruff + pyright**

Run: `uv run ruff check pyphi test/core/test_cause_inversion.py test/core/inversion_oracle.py && uv run pyright pyphi`
Expected: clean. (`MacroSystem.cause_marginal` already carries `# type: ignore[override]`; if pyright now flags the `FactoredTPM` vs `CauseMarginals` override mismatch anyway, the fix is to keep the ignore comment and note both types expose `.factor(i)`.)

- [ ] **Step 10: Commit**

```bash
git add pyphi/core/tpm/marginalization.py pyphi/system.py pyphi/node.py \
  test/core/inversion_oracle.py test/core/test_cause_inversion.py \
  test/core/test_marginalization_factored.py test/core/test_marginalization_kary.py \
  test/test_system.py
git commit -m "Make the reduced sum-product inversion the production cause path

_cause_marginal_factored is now the greedy contraction; the dense
implementation moves to test/core/inversion_oracle.py as the
cross-validation reference. System.cause_marginal holds per-system-unit
CauseMarginals (background-unit cause factors were computed but never
consumed), and proper_cause_marginal derives from it, removing a
duplicate inversion per system.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 4: Hypothesis cross-validation against the oracle

**Files:**
- Test: `test/core/test_cause_inversion_hypothesis.py` (new)

**Interfaces:**
- Consumes: `_cause_marginal_factored -> CauseMarginals` (Task 3 final form); `dense_cause_marginal_reference` oracle (Task 3).

- [ ] **Step 1: Write the property test**

Create `test/core/test_cause_inversion_hypothesis.py`:

```python
"""Property-based cross-validation: reduced inversion vs the dense oracle."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored

from .inversion_oracle import dense_cause_marginal_reference


@st.composite
def inversion_cases(draw):
    """Random factored TPM + state + nonempty system subset.

    Parent sets, alphabets, and the system subset all vary independently;
    probabilities come from a seeded generator (bounded away from zero so
    every state is reachable and the oracle's norm > 0).
    """
    n = draw(st.integers(min_value=2, max_value=6))
    alphabets = tuple(
        draw(st.integers(min_value=2, max_value=3)) for _ in range(n)
    )
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    factors = []
    for i in range(n):
        parents = draw(st.frozensets(st.integers(min_value=0, max_value=n - 1)))
        shape = tuple(alphabets[j] if j in parents else 1 for j in range(n))
        raw = rng.random((*shape, alphabets[i])) + 1e-3
        factors.append(raw / raw.sum(axis=-1, keepdims=True))
    factored = FactoredTPM(
        factors=factors,
        state_space=tuple(tuple(range(a)) for a in alphabets),
    )
    state = tuple(
        draw(st.integers(min_value=0, max_value=alphabets[j] - 1))
        for j in range(n)
    )
    size = draw(st.integers(min_value=1, max_value=n))
    system = tuple(sorted(draw(st.permutations(tuple(range(n))))[:size]))
    return factored, state, system


@given(inversion_cases())
@settings(max_examples=300, deadline=None)
def test_reduced_matches_dense_oracle(case):
    factored, state, system = case
    reduced = _cause_marginal_factored(factored, state, system)
    dense = dense_cause_marginal_reference(factored, state, system)
    assert reduced.indices == system
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(
            reduced.factor(i), dense[i], rtol=0, atol=1e-12
        )
```

(Match the oracle-import form chosen in Task 3 Step 2.)

- [ ] **Step 2: Run it**

Run: `uv run pytest test/core/test_cause_inversion_hypothesis.py -q`
Expected: PASS in well under a minute. If Hypothesis finds a counterexample, it is a real divergence — debug the contraction, do not widen the tolerance.

- [ ] **Step 3: Commit**

```bash
git add test/core/test_cause_inversion_hypothesis.py
git commit -m "Property-test the reduced cause inversion against the dense oracle

Random parent sets, k-ary alphabets, states, and system subsets;
agreement within 1e-12 absolute.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 5: Large-substrate capability tests (independent references)

**Files:**
- Test: `test/core/test_cause_inversion_large.py` (new)

**Interfaces:**
- Consumes: `_cause_marginal_factored -> CauseMarginals`; `dense_cause_marginal_reference`; `FactoredTPM`; `Substrate`/`System` for the end-to-end check.

Background (from the spec): truncating to the system's ancestor closure is NOT a valid reference — descendants of system units carry evidence about system pasts. Only whole disconnected factor-graph components cancel (as constants in both `pr_bg` and `norm`). Hence two references: a disconnected-block substrate checked against the small block's dense oracle, and a connected chain checked against a forward-backward (transfer-matrix) computation written in the test.

- [ ] **Step 1: Write the tests**

Create `test/core/test_cause_inversion_large.py`:

```python
"""Large-substrate cause inversion vs independent references.

These substrates are far beyond the dense implementation's reach (the
joint likelihood would have 2^40 entries), so each test checks the
reduced path against an independently written computation instead.
"""

from __future__ import annotations

import time

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored

from .inversion_oracle import dense_cause_marginal_reference

N_LARGE = 40


def _normalized(raw: np.ndarray) -> np.ndarray:
    return raw / raw.sum(axis=-1, keepdims=True)


def _chain_factor(n: int, unit: int, parents: tuple[int, ...], rng) -> np.ndarray:
    """Binary factor over ``n`` axes with real extent on ``parents`` only."""
    shape = tuple(2 if j in parents else 1 for j in range(n))
    return _normalized(rng.random((*shape, 2)) + 0.05)


def test_disconnected_block_matches_small_dense_oracle():
    """8-unit block containing the system + separate 32-unit block.

    The disconnected block's likelihood contributes a constant that
    cancels in pr_bg / norm, so the 40-unit result must match the dense
    oracle run on the 8-unit block alone.
    """
    rng = np.random.default_rng(42)
    system = (2, 3, 4)

    # Block A: units 0..7, a chain (unit j <- {j-1, j}; unit 0 <- {0}).
    # Build each factor twice from the same values: once over 40 axes,
    # once over 8 axes, so the two substrates share identical numbers.
    small_factors = []
    large_factors = []
    for j in range(8):
        parents = (j,) if j == 0 else (j - 1, j)
        f_small = _chain_factor(8, j, parents, rng)
        small_factors.append(f_small)
        pad = (1,) * (N_LARGE - 8)
        large_factors.append(f_small.reshape((*f_small.shape[:-1], *pad, 2)))

    # Block B: units 8..39, a chain among themselves, no cross edges.
    for j in range(8, N_LARGE):
        parents = (j,) if j == 8 else (j - 1, j)
        large_factors.append(_chain_factor(N_LARGE, j, parents, rng))

    state_small = tuple(int(b) for b in rng.integers(0, 2, size=8))
    state_large = state_small + tuple(int(b) for b in rng.integers(0, 2, size=N_LARGE - 8))

    large = FactoredTPM(factors=large_factors)
    small = FactoredTPM(factors=small_factors)

    reduced = _cause_marginal_factored(large, state_large, system)
    oracle = dense_cause_marginal_reference(small, state_small, system)

    for i in system:
        got = np.squeeze(reduced.factor(i))
        want = np.squeeze(oracle[i])
        assert got.shape == want.shape
        # Not bit-exact: the disconnected block's constant cancels in a
        # float division.
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def _forward_backward_reference(g, system, state_relevant_axis):
    """Marginal weight over one relevant axis of a chain, then per-unit
    outputs — an independent sequential evaluation of Eq. 4 for chain
    topology. ``g[j]`` is unit j's likelihood slice as a small dense array
    over (s_{j-1}, s_j) (unit 0: over (s_0,))."""
    n = len(g)
    # Forward messages: m[j](s_j) = sum_{s_{j-1}} m[j-1](s_{j-1}) g[j](s_{j-1}, s_j)
    m = [g[0]]
    for j in range(1, n):
        m.append(m[j - 1] @ g[j])
    # Backward messages: b[j](s_j) = sum_{s_{j+1}} g[j+1](s_j, s_{j+1}) b[j+1](s_{j+1})
    b = [None] * n
    b[n - 1] = np.ones(2)
    for j in range(n - 2, -1, -1):
        b[j] = g[j + 1] @ b[j + 1]
    r = state_relevant_axis
    pr = m[r] * b[r]           # unnormalized marginal over s_r
    return pr / pr.sum()


def test_connected_chain_matches_forward_backward():
    """40-unit connected chain, system in the middle, vs a transfer-matrix
    computation of the background weight and outputs."""
    rng = np.random.default_rng(7)
    system = (18, 19, 20, 21)

    factors = []
    dense_slices = []  # small (2,)- or (2,2)-shaped likelihood slices
    state = tuple(int(bit) for bit in rng.integers(0, 2, size=N_LARGE))
    for j in range(N_LARGE):
        parents = (j,) if j == 0 else (j - 1, j)
        f = _chain_factor(N_LARGE, j, parents, rng)
        factors.append(f)
        sliced = np.squeeze(f[..., state[j]])
        dense_slices.append(sliced)

    factored = FactoredTPM(factors=factors)
    t0 = time.perf_counter()
    reduced = _cause_marginal_factored(factored, state, system)
    elapsed = time.perf_counter() - t0
    # Feasibility gate: dense evaluation would need a 2^40 array. Generous
    # bound so CI noise cannot flake it, while still catching any
    # accidental fallback to dense evaluation.
    assert elapsed < 10.0

    # Relevant background axis: only unit 17 (parent of system unit 18).
    weight = _forward_backward_reference(dense_slices, system, 17)

    # Reference outputs. Unit 18 depends on (s_17, s_18): contract the
    # weight over s_17. Units 19..21 depend only on system axes: their
    # output is the forward factor itself (weight sums to 1).
    f18 = np.squeeze(factors[18])          # (2, 2, 2): s_17, s_18, out
    want_18 = np.einsum("abk,a->bk", f18, weight)
    got_18 = np.squeeze(reduced.factor(18))
    np.testing.assert_allclose(got_18, want_18, rtol=0, atol=1e-12)

    for i in (19, 20, 21):
        got = np.squeeze(reduced.factor(i))
        want = np.squeeze(factors[i])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_two_backgrounds_give_different_weights():
    """Genuine-difference guard for the chain test: flipping the state of
    the relevant background unit changes the system's cause factors."""
    rng = np.random.default_rng(7)
    system = (18, 19, 20, 21)
    factors = [
        _chain_factor(N_LARGE, j, (j,) if j == 0 else (j - 1, j), rng)
        for j in range(N_LARGE)
    ]
    factored = FactoredTPM(factors=factors)
    state = [0] * N_LARGE
    a = _cause_marginal_factored(factored, tuple(state), system)
    state[17] = 1  # the relevant background parent
    b = _cause_marginal_factored(factored, tuple(state), system)
    assert not np.allclose(a.factor(18), b.factor(18))


def test_system_level_proper_cause_marginal_on_large_substrate():
    """End-to-end: Substrate/System over 40 units computes
    proper_cause_marginal without materializing the joint."""
    from pyphi import config
    from pyphi.substrate import Substrate
    from pyphi.system import System

    rng = np.random.default_rng(3)
    factors = [
        _chain_factor(N_LARGE, j, (j,) if j == 0 else (j - 1, j), rng)
        for j in range(N_LARGE)
    ]
    sub = Substrate(marginals=factors)
    state = tuple(int(bit) for bit in rng.integers(0, 2, size=N_LARGE))
    with config.override(validate_system_states=False):
        sys_ = System(sub, state=state, node_indices=(18, 19, 20, 21))
        proper = sys_.proper_cause_marginal
    assert proper.n_nodes == 4
    for slot in range(4):
        np.testing.assert_allclose(
            proper.factor(slot).sum(axis=-1), 1.0, atol=1e-10
        )
```

Note on `test_connected_chain_matches_forward_backward`'s unit-19..21 expectation: for those units every parent is inside the system, so the background sum contracts the weight completely (`Σ weight = 1`) and the output equals the forward factor — this is itself a meaningful identity check, and unit 18 carries the nontrivial weighted case. The `np.einsum` here is fine (3 labels); the production limit only bites at >52 labels.

Check the exact `System` constructor signature before running (`System(sub, state=..., node_indices=...)` — see `test/test_system.py` for the working form; adjust the keyword if it differs).

- [ ] **Step 2: Run the tests**

Run: `uv run pytest test/core/test_cause_inversion_large.py -v`
Expected: all PASS, total runtime a few seconds.

- [ ] **Step 3: Commit**

```bash
git add test/core/test_cause_inversion_large.py
git commit -m "Validate the reduced cause inversion on 40-unit substrates

Two independent references: a disconnected-block substrate checked
against the dense oracle on the small block (the disconnected
component's likelihood cancels in the background weighting), and a
connected chain checked against a forward-backward transfer-matrix
computation. Includes an end-to-end System-level check and a
feasibility time bound.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 6: Full verification, changelog, ROADMAP, benchmark rerun

**Files:**
- Create: `changelog.d/reduced-cause-inversion.optimization.md`
- Modify: `ROADMAP.md` (the junction-tree/reduced-inversion wishlist row in the Status Dashboard and its residue paragraph; verify exact lines by grepping `junction-tree` at execution time)
- Create: `benchmarks/iit_3_vs_4/results/p18_inversion_share_seed6001_post_reduction.json` (new benchmark output; never overwrite the existing JSON)

- [ ] **Step 1: Full suite in background**

```bash
(uv run --all-extras pytest -q; echo "RC=$?") > /private/tmp/claude-501/-Users-will-projects-pyphi/e565ccef-e10a-41c5-9d26-30126f101e2e/scratchpad/full-suite-reduced-inversion.log 2>&1 &
```

While it runs, proceed with Steps 2-4. When it finishes, read the `RC=` line and the tail of the log. Expected: `RC=0`. **Any golden failure**: apply the spec's validation protocol — quantify the deviation, check for discrete MIP/partition flips, report to the user before touching any golden. Full-substrate systems are bit-identical by construction; only background-carrying cases can drift.

- [ ] **Step 2: Changelog fragment**

Create `changelog.d/reduced-cause-inversion.optimization.md`:

```markdown
The cause-side Bayesian inversion (IIT 4.0 Eq. 4) now evaluates as a greedy sum-product contraction over the factored TPM's dependence structure instead of materializing the joint likelihood over all substrate units. Small systems embedded in large sparse substrates (up to numpy's 64-dimension array limit) are now tractable on the cause side. `System.cause_marginal` now holds per-system-unit cause factors (`CauseMarginals`); background-unit cause factors — computed but never consumed before — are no longer produced. Densely coupled substrates whose contraction would exceed the intermediate-size cap raise `IntractableCauseInversionError` instead of exhausting memory.
```

- [ ] **Step 3: Benchmark rerun**

```bash
uv run python -m benchmarks.iit_3_vs_4.p18_inversion_share --help
```

Check the runner's CLI for a run-label/output-name option and use it so the new JSON lands beside (never over) `results/p18_inversion_share_seed6001.json` with `post_reduction` in the name; if the runner has no such option, copy the runner's no-clobber convention (`_v2` suffix) instead. Run it with the same seed (6001). Expected: the inversion's share of SIA wall time collapses (Task 1 removes the per-partition recomputation; the single remaining inversion is also cheaper). Record the measured shares.

- [ ] **Step 4: ROADMAP update**

`grep -n "junction-tree\|reduced-dimension" ROADMAP.md`, then update the wishlist/residue entries that describe the reduced-dimension cause inversion as future work: mark landed (with today's date and the measured numbers from Step 3), keeping the prose impersonal and self-contained. Add a new post-2.0 wishlist item for lifting the substrate-size ceiling:

> **Reduced-dimension factor storage (lifts the 64-dimension ceiling).**
> `FactoredTPM` factors carry all N input axes (size 1 for non-parents), so
> substrates are capped at 63 units by numpy's 64-dimension array limit.
> Lifting the cap means storing each factor over its parent axes only plus
> an axis map, and threading that convention through `FactoredTPM`
> (validation, `condition()`, `infer_cm`, serialization), node TPM
> generation, and the repertoire algebra — the largest remaining
> representation refactor. The cause inversion is already structured for
> it: `_sum_product` in `pyphi/core/tpm/marginalization.py` is the single
> seam whose internals would change (pairwise contraction with explicit
> axis maps), and the inversion's validation apparatus (dense oracle,
> Hypothesis cross-validation, disconnected-block and transfer-matrix
> references in `test/core/`) is the acceptance harness. Independent
> prerequisites for very large substrates regardless of storage: log-space
> likelihoods (float64 underflow near several hundred units) and
> subset-enumeration costs.

Before staging, run `git diff ROADMAP.md` and confirm only these hunks are yours (concurrent sessions edit this file).

- [ ] **Step 5: Verify the full suite passed**

Read the log: the `RC=` line must be `RC=0`. If not, stop and fix before committing (never bypass hooks, never regenerate goldens silently).

- [ ] **Step 6: Commit**

```bash
git add changelog.d/reduced-cause-inversion.optimization.md ROADMAP.md \
  benchmarks/iit_3_vs_4/results/p18_inversion_share_seed6001_post_reduction.json
git commit -m "Record the reduced cause inversion: changelog, roadmap, benchmark

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

## Self-review notes

- Spec coverage: Part 0 → Task 1; Part 1 (contraction, guard, exception, exactness property) → Task 2; Part 2 (CauseMarginals, System rewiring, proper derivation, docstrings, dispatch tests) → Task 3; Part 3 (oracle) → Task 3; validation protocol items 1 (full-object A/B via exact-comparing golden suite) → Tasks 3+6, 2 (Hypothesis) → Task 4, 3 (independent large-N references) → Task 5, 4 (genuine-difference) → Tasks 1/2/5, 5 (error paths) → Task 2. Acceptance criteria → Tasks 5 and 6.
- Type consistency: `CauseMarginals` (Task 2) is consumed by name in Tasks 3-5; `_cause_marginal_reduced` exists only during Task 2 and is renamed in Task 3 Step 3; the oracle's dict return is consumed as `dense[i]` throughout.
