# JointTPM-as-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish `JointTPM` as a clean, read-only, k-ary **view** over `FactoredTPM` — the joint/dense peer of `FactoredTPM` under the `TPM` Protocol — by first removing its two internal-container roles, then collapsing the two `JointTPM` classes into one eager-snapshot view.

**Architecture:** `FactoredTPM` stays the only *stored* TPM. `JointTPM` becomes a read-only value type that eagerly materializes the joint conditional `P(sₜ₊₁ | sₜ)` (explicit-alphabet shape `(*alphabet_sizes, n, max_alphabet)`) at construction and holds no reference to its source. The compute path (`node.py`, `repertoire_algebra.py`) stops using `JointTPM` as a mutable-ish container and instead uses raw ndarrays plus two free helper functions; `dynamics.py` operates on arrays directly. Only then is the numpy-proxy `JointDistribution`/`ArrayLike` base removed and the thin `joint.py` Protocol wrapper folded into the one clean type.

**Tech Stack:** Python 3.13+, NumPy, pytest (with `--doctest-modules`), pyright, ruff. Uses existing `pyphi.convert`, `pyphi/core/tpm/factored.py`, `pyphi/data_structures` (`np_hash`).

## Global Constraints

- **Mathematical correctness is paramount.** No computed φ value may change. The golden suite (`test/data/golden/v1/`) is the φ-invariance oracle: after each φ-path task, `uv run pytest test/integration/test_golden_regression.py --slow` must pass with **no fixture regeneration**. If a golden hash or value moves, the change was unfaithful — stop and diagnose.
- **Verify with the doctest sweep.** Final verification is `uv run pytest` with **no path argument** (collects `pyphi/` doctests). A bare `pytest test/` skips the doctest sweep.
- **No planning artifacts in source.** No `P0`–`Pn`, "Phase A", `TODO(Px)`, or migration-journey narrative in code/docstrings/changelog. Docstrings describe the final state, impersonal, NumPy style (`Parameters`/`Returns`/`Notes`, not `Args:`).
- **No back-compat shims.** 2.0 is unreleased and breaking; delete cleanly, update callers, no aliases.
- **Never `--no-verify`.** Pre-commit (ruff + pyright) must pass. Do not bypass.
- **Changelog fragment** for user-facing changes in `changelog.d/<name>.<type>.md`.
- **Reuse over rewrite.** `pyphi.convert`, `FactoredTPM.to_joint()`/`from_joint()`, and `pyphi.data_structures.np_hash` already exist — use them.

---

## File Structure

- `pyphi/core/tpm/_node_ops.py` — **new.** Two free functions, `marginalize_out(array, node_indices)` and `condition(array, condition)`, lifted verbatim (semantics-preserving) off the legacy `JointDistribution`. The per-node-marginal algebra the compute path needs, operating on plain ndarrays. Internal (leading underscore).
- `pyphi/node.py` — **modify.** `Node.cause_marginal`/`effect_marginal` become plain ndarrays; construction uses `FactoredTPM.factor(i)` + `_node_ops`; hashing/equality use `np_hash`/`np.array_equal`. Delete dead code (binary `else` branch, four `_off`/`_on` properties, `expand_node_tpm`).
- `pyphi/core/repertoire_algebra.py` — **modify.** ~6 call sites switch from `JointTPM` methods to raw-ndarray indexing + `_node_ops` free functions.
- `pyphi/dynamics.py` — **modify.** Drop `JointTPM`; operate on arrays with a small local `_n_units` helper preserving the state-by-state (`log2`) and state-by-node (`shape[-1]`) cases.
- `pyphi/core/tpm/marginalization.py` — **modify.** Drop the `isinstance(tpm, JointTPM)` / `._inner` special-cases; the generic `to_array()` branch subsumes them.
- `pyphi/core/tpm/joint.py` — **replace.** Becomes the single clean `JointTPM` view (read-only, eager snapshot, k-ary, Protocol-satisfying).
- `pyphi/core/tpm/joint_distribution.py` — **delete** (its `JointTPM`, `JointDistribution`, `simulate`, `permute_nodes`, etc. are all removed; anything still needed moves to `joint.py` or `_node_ops.py`).
- `pyphi/core/tpm/__init__.py`, `pyphi/__init__.py` — **modify.** Repoint `JointTPM`; drop the `JointDistribution` export.
- `pyphi/substrate.py` — **modify.** `joint_tpm()` returns a `JointTPM` view.
- `pyphi/core/tpm/_display.py` / `factored.py` — **check.** Keep the B21 joint display card working against the new view.

---

## PHASE 1 — Remove the internal-container roles (φ-path; prerequisite)

### Task 1: Free per-node-marginal ops (`_node_ops.py`)

**Files:**
- Create: `pyphi/core/tpm/_node_ops.py`
- Test: `test/core/test_node_ops.py`

**Interfaces:**
- Produces:
  - `marginalize_out(array: NDArray, node_indices: Iterable[int]) -> NDArray` — sums `array` over `node_indices` with `keepdims=True`, divides by the product of those axes' sizes (uniform/max-entropy marginalization). Byte-identical to the legacy `JointDistribution.marginalize_out` minus the type wrapper.
  - `condition(array: NDArray, condition: Mapping[int, int]) -> NDArray` — fixes each input axis `i` in `condition` to state `condition[i]`, re-inserting a singleton axis (so ndim is preserved); skips axes already size 1. Byte-identical to the legacy `JointDistribution.condition_tpm` minus the type wrapper.

- [ ] **Step 1: Write the failing tests**

```python
# test/core/test_node_ops.py
import numpy as np
from pyphi.core.tpm._node_ops import marginalize_out, condition


def test_marginalize_out_uniform_average_keepdims():
    # 2 binary inputs, trailing size-2 node axis.
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    out = marginalize_out(arr, [0])
    # Axis 0 collapsed to a size-1 mean.
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(out, arr.sum(axis=0, keepdims=True) / 2)


def test_marginalize_out_multiple_axes():
    arr = np.random.default_rng(0).random((2, 2, 2, 2))
    out = marginalize_out(arr, [0, 1])
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out, arr.sum((0, 1), keepdims=True) / 4)


def test_condition_fixes_axis_and_keeps_ndim():
    arr = np.arange(8, dtype=float).reshape(2, 2, 2)
    out = condition(arr, {0: 1})
    assert out.shape == (1, 2, 2)
    np.testing.assert_array_equal(out, arr[1][np.newaxis, ...])


def test_condition_skips_singleton_axis():
    arr = np.arange(4, dtype=float).reshape(1, 2, 2)
    out = condition(arr, {0: 0})
    np.testing.assert_array_equal(out, arr)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/core/test_node_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: pyphi.core.tpm._node_ops`.

- [ ] **Step 3: Implement**

```python
# pyphi/core/tpm/_node_ops.py
"""Per-node-marginal array operations.

Marginalization and conditioning over a single unit's conditional
distribution, stored as a plain ndarray whose leading axes are the joint
input state at |t| and whose trailing axis is that unit's own state. These
are the array-level operations the repertoire algebra composes; they carry
no distribution type of their own.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from itertools import chain

import numpy as np
from numpy.typing import NDArray


def marginalize_out(
    array: NDArray[np.float64], node_indices: Iterable[int]
) -> NDArray[np.float64]:
    """Marginalize the given input axes out of a per-unit conditional.

    Sums ``array`` over ``node_indices`` (keeping those axes as singletons)
    and divides by the product of their sizes, i.e. averages under a uniform
    distribution over the marginalized units.
    """
    indices = list(node_indices)
    if not indices:
        return array
    return array.sum(tuple(indices), keepdims=True) / (
        np.array(array.shape)[indices].prod()
    )


def condition(
    array: NDArray[np.float64], condition: Mapping[int, int]
) -> NDArray[np.float64]:
    """Condition a per-unit conditional on fixed input states.

    Fixes each input axis ``i`` present in ``condition`` to state
    ``condition[i]``, re-inserting a singleton axis so the number of
    dimensions is unchanged. Axes already of size 1 are left untouched.
    """
    selectors: list[list] = [[slice(None)]] * (array.ndim - 1)
    for i, state_i in condition.items():
        if array.shape[i] != 1:
            selectors[i] = [state_i, np.newaxis]
    return array[tuple(chain.from_iterable(selectors))]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/core/test_node_ops.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/tpm/_node_ops.py test/core/test_node_ops.py
git commit -m "Add free per-node-marginal ops (marginalize_out, condition)"
```

---

### Task 2: Migrate `node.py` off the `JointTPM` container

**Files:**
- Modify: `pyphi/node.py`
- Test: existing `test/test_node.py` + golden suite

**Interfaces:**
- Consumes: `_node_ops.marginalize_out`, `_node_ops.condition`; `pyphi.data_structures.np_hash`.
- Produces: `Node.cause_marginal` / `Node.effect_marginal` are now **`numpy.ndarray`** (shape `(*alphabet_sizes-with-non-inputs-collapsed-to-1, k_i)`), not `JointTPM`. Consumers must index/operate on them as arrays (Task 3 updates the only consumer).

- [ ] **Step 1: Rewrite `Node.__init__` TPM construction.** Replace lines 87–123 (the `JointTPM(...)` wrapping, the `isinstance(effect_marginal, _FactoredTPM)` fork, and the dead binary `else`) with the factored-only path storing ndarrays:

```python
        # Per-unit factor: shape (*alphabet_sizes, k_i), leading axes are the
        # substrate units' state at |t|, trailing axis this unit's own state.
        # Marginalize out the units that are not inputs to this node.
        cause_factor = cause_marginal.factor(self.index)
        cause_non_inputs = set(range(cause_factor.ndim - 1)) - self._inputs
        self.cause_marginal = marginalize_out(cause_factor, cause_non_inputs)

        effect_factor = effect_marginal.factor(self.index)
        effect_non_inputs = set(range(effect_factor.ndim - 1)) - self._inputs
        self.effect_marginal = marginalize_out(effect_factor, effect_non_inputs)
```

Update imports: remove `from .core.tpm.joint_distribution import JointTPM` and `import numpy as np` if now unused (it is used by `expand_node_tpm`, which Step 4 deletes — verify and drop if unused); add `from .core.tpm._node_ops import marginalize_out` and `from .data_structures import np_hash`. Remove the local `from .core.tpm.factored import FactoredTPM as _FactoredTPM` (the isinstance fork is gone). Remove the stale `# TODO extend to nonbinary nodes` at line 18 (the path is now k-ary).

- [ ] **Step 2: Fix hashing and equality.** `self.cause_marginal`/`effect_marginal` are ndarrays now:

```python
        self._hash = hash(
            (
                index,
                np_hash(self.cause_marginal),
                np_hash(self.effect_marginal),
                self.state,
                self._inputs,
                self._outputs,
            )
        )
```

and in `__eq__`:

```python
            and np.array_equal(self.cause_marginal, other.cause_marginal)
            and np.array_equal(self.effect_marginal, other.effect_marginal)
```

(keep `import numpy as np` for `np.array_equal`.)

- [ ] **Step 3: Delete the four dead `_off`/`_on` properties** (`cause_marginal_off`, `effect_marginal_off`, `cause_marginal_on`, `effect_marginal_on`, current lines 138–156). Confirm zero readers first:

Run: `grep -rn "marginal_on\b\|marginal_off\b" pyphi/ test/`
Expected: only the definitions (and this plan). If any real reader exists, stop and reassess.

- [ ] **Step 4: Delete `expand_node_tpm`** (lines 250–263). Confirm no production callers:

Run: `grep -rn "expand_node_tpm" pyphi/ test/`
Expected: only its definition (and possibly a test — if a test exercises it, delete that test too, noting it in the commit).

- [ ] **Step 5: Update the `Node` / `generate_nodes` docstrings** so the `Attributes`/`Parameters` say `cause_marginal`/`effect_marginal` are ndarrays and `effect_marginal` parameter is a `FactoredTPM` (drop the "or JointTPM in binary state-by-node form" alternative — that path is gone). NumPy style, final-state voice.

- [ ] **Step 6: Run node + golden tests**

Run: `uv run pytest test/test_node.py test/integration/test_golden_regression.py --slow -q`
Expected: PASS, **no fixture regeneration**. If any golden hash/value moved, the marginalization is unfaithful — diagnose before proceeding.

- [ ] **Step 7: Commit**

```bash
git add pyphi/node.py test/test_node.py
git commit -m "Store node cause/effect marginals as plain ndarrays; drop JointTPM container from node.py"
```

---

### Task 3: Migrate `repertoire_algebra.py` call sites

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Test: golden suite + `test/core/` repertoire tests

**Interfaces:**
- Consumes: `Node.cause_marginal`/`effect_marginal` as ndarrays (Task 2); `_node_ops.marginalize_out`, `_node_ops.condition`.

- [ ] **Step 1: Update `_single_node_cause_repertoire`.** Line 110 `mechanism_node.cause_marginal[..., mechanism_node.state]` still works (ndarray indexing). Line 117 changes from method to free function and drops `.tpm`:

```python
    tpm = mechanism_node.cause_marginal[..., mechanism_node.state]
    return marginalize_out(tpm, mechanism_node.inputs - purview_set)
```

- [ ] **Step 2: Update `_single_node_effect_repertoire`.** Lines 142/144/149/154–160:

```python
    if direction == Direction.CAUSE:
        tpm = condition(purview_node.cause_marginal, condition_map)
    elif direction == Direction.EFFECT:
        tpm = condition(purview_node.effect_marginal, condition_map)
    else:
        _validate.direction(direction)
        raise AssertionError("unreachable")
    nonmechanism_inputs = purview_node.inputs - set(condition_map)
    tpm = marginalize_out(tpm, nonmechanism_inputs)
    alphabet_sizes = cs.substrate.factored_tpm.alphabet_sizes
    return tpm.reshape(
        repertoire_shape(
            cs.substrate.node_indices,
            (purview_node_index,),
            alphabet_sizes=alphabet_sizes,
        )
    )
```

Note: the local variable holding the `condition` mapping argument must be renamed (e.g. `condition_map`) to avoid shadowing the imported `condition` function. Rename the parameter `condition: FrozenMap` → `condition_map: FrozenMap` in this function's signature and body.

- [ ] **Step 3: Add the import**

```python
from .tpm._node_ops import condition
from .tpm._node_ops import marginalize_out
```

(verify the relative path from `pyphi/core/repertoire_algebra.py` is `.tpm._node_ops`.)

- [ ] **Step 4: Run repertoire + golden tests**

Run: `uv run pytest test/core/ test/integration/test_golden_regression.py --slow -q`
Expected: PASS, no fixture regeneration.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py
git commit -m "Consume node marginals as ndarrays in the repertoire algebra"
```

---

### Task 4: Migrate `dynamics.py` off `JointTPM`

**Files:**
- Modify: `pyphi/dynamics.py`
- Test: `test/core/test_tpm.py` (simulate tests) + a new dynamics unit test

**Interfaces:**
- Consumes: `pyphi.convert` (existing).

- [ ] **Step 1: Add a failing test pinning current simulate behavior** (seeded, deterministic):

```python
# test/test_dynamics.py
import numpy as np
from pyphi import examples
from pyphi.dynamics import simulate, most_probable_next_state


def test_simulate_deterministic_seeded():
    sub = examples.basic_substrate()
    tpm = sub.joint_tpm()[..., 1]  # binary multidim state-by-node P(on)
    rng = np.random.default_rng(42)
    traj = simulate(tpm, initial_state=(0, 0, 0), timesteps=5, rng=rng)
    assert len(traj) == 5
    assert all(len(s) == 3 for s in traj)


def test_most_probable_next_state_binary():
    sub = examples.basic_substrate()
    tpm = sub.joint_tpm()[..., 1]
    nxt = most_probable_next_state(tpm, (0, 0, 0))
    assert len(nxt) == 3
    assert set(nxt) <= {0, 1}
```

Run: `uv run pytest test/test_dynamics.py -v` (should pass on the current code — it's a characterization test to guard the refactor).

- [ ] **Step 2: Replace the `JointTPM` wraps with arrays.** In `mean_dynamics` (line 22) and `simulate` (line 53) and `most_probable_next_state` (line 104), replace `tpm = JointTPM(tpm)` with `tpm = np.asarray(tpm, dtype=float)`. Replace the `tpm.number_of_units` property read in `simulate` (line 54) with a local helper that preserves both shapes:

```python
def _n_units(array):
    """Number of units for a state-by-node (trailing axis = N) or square
    state-by-state (2**N rows) TPM array."""
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return int(np.log2(array.shape[1]))
    return array.shape[-1]
```

Use `_n_units(tpm)` at the `number_of_units(tpm)` (line 26) and `tpm.number_of_units` (lines 54, 64) sites. Delete the now-redundant module-level `number_of_units` free function (line 161) **only if** it has no external importers:

Run: `grep -rn "number_of_units" pyphi/ test/ docs/ --include="*.py" --include="*.rst"`
If `dynamics.number_of_units` is documented/imported elsewhere, keep it as a thin wrapper `return _n_units(np.asarray(tpm))` instead of deleting.

- [ ] **Step 3: Remove the `JointTPM` import** (`from .core.tpm.joint_distribution import JointTPM`, line 12).

- [ ] **Step 4: Run dynamics + doctest for the module**

Run: `uv run pytest test/test_dynamics.py test/core/test_tpm.py --doctest-modules pyphi/dynamics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/dynamics.py test/test_dynamics.py
git commit -m "Operate on arrays in dynamics.py instead of the JointTPM container"
```

---

### Task 5: Simplify `marginalization.py` wrapper branches

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py`
- Test: golden suite

**Interfaces:**
- Consumes: the `TPM` Protocol's `to_array()`.

- [ ] **Step 1: Drop the `isinstance(tpm, JointTPM)` / `._inner` special-cases** in `cause_marginal` (lines 192–194) and `cause_conditioned` (lines 217–218). The generic branch already handles any Protocol TPM:

```python
    # cause_marginal:
    if isinstance(tpm, FactoredTPM):
        return _cause_marginal_factored(tpm, state, node_indices)
    factored = FactoredTPM.from_joint(tpm.to_array())
    return cause_marginal(factored, state, node_indices)
```

```python
    # cause_conditioned:
    if not isinstance(tpm, FactoredTPM):
        tpm = FactoredTPM.from_joint(tpm.to_array())
    conditioned = tpm.condition(dict(background))
    return CauseMarginals({i: conditioned.factor(i) for i in node_indices})
```

- [ ] **Step 2: Remove the now-unused `from .joint import JointTPM` import** (line 14).

- [ ] **Step 3: Run golden suite**

Run: `uv run pytest test/integration/test_golden_regression.py --slow -q`
Expected: PASS, no regeneration.

- [ ] **Step 4: Commit**

```bash
git add pyphi/core/tpm/marginalization.py
git commit -m "Route joint TPM inputs through to_array() in marginalization dispatch"
```

**Checkpoint — Phase 1 done.** Nothing outside `joint_distribution.py`/`joint.py` themselves now constructs or calls container methods on `JointTPM`. Verify:

Run: `grep -rn "JointTPM\|JointDistribution" pyphi/ --include="*.py" | grep -v "core/tpm/joint"`
Expected: only `pyphi/__init__.py` (the export) remains. If anything else appears, it was missed above.

---

## PHASE 2 — Build the clean view and delete the legacy classes

### Task 6: The `JointTPM` view (eager snapshot, read-only, k-ary)

**Files:**
- Replace contents of: `pyphi/core/tpm/joint.py`
- Test: `test/core/test_joint_tpm.py`

**Interfaces:**
- Consumes: `FactoredTPM.to_joint()` output shape `(*alphabet_sizes, n_nodes, max_alphabet)`.
- Produces: `class JointTPM` with `shape`, `n_nodes`, `alphabet_sizes`, `to_array()`, `__array__`, `__getitem__`, `condition(fixed)`, `array_equal(other)`, `__eq__`, `__hash__`, `_describe(verbosity)` (B21 display). Satisfies the `TPM` Protocol (`base.py`).

- [ ] **Step 1: Write the failing tests**

```python
# test/core/test_joint_tpm.py
import numpy as np
import pytest
from pyphi import examples
from pyphi.core.tpm import TPM
from pyphi.core.tpm.joint import JointTPM


def _joint(sub):
    return sub.factored_tpm.to_joint()


def test_view_satisfies_protocol():
    v = JointTPM(_joint(examples.basic_substrate()))
    assert isinstance(v, TPM)


def test_view_metadata_binary():
    sub = examples.basic_substrate()
    v = JointTPM(_joint(sub))
    assert v.n_nodes == 3
    assert v.alphabet_sizes == (2, 2, 2)
    assert v.shape == (2, 2, 2, 3, 2)


def test_view_to_array_roundtrip():
    joint = _joint(examples.basic_substrate())
    v = JointTPM(joint)
    np.testing.assert_array_equal(v.to_array(), joint)
    np.testing.assert_array_equal(np.asarray(v), joint)


def test_view_is_eager_snapshot():
    joint = _joint(examples.basic_substrate()).copy()
    v = JointTPM(joint)
    joint[:] = 0.0  # mutate the source
    assert not np.all(v.to_array() == 0.0)  # snapshot is decoupled


def test_view_kary_metadata():
    sub = examples.gomez_p53_mdm2_substrate()  # alphabets (3, 2, 2)
    v = JointTPM(sub.factored_tpm.to_joint())
    assert v.alphabet_sizes == (3, 2, 2)
    assert v.n_nodes == 3


def test_view_equality_and_hash():
    joint = _joint(examples.basic_substrate())
    assert JointTPM(joint) == JointTPM(joint)
    assert hash(JointTPM(joint)) == hash(JointTPM(joint))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/core/test_joint_tpm.py -v`
Expected: FAIL (current `joint.py` `JointTPM` wraps the legacy class; `test_view_is_eager_snapshot`/metadata assertions fail).

- [ ] **Step 3: Implement the view**

```python
# pyphi/core/tpm/joint.py
"""The joint (dense) form of a substrate TPM.

The joint peer of :class:`~pyphi.core.tpm.factored.FactoredTPM` under the
:class:`~pyphi.core.tpm.base.TPM` Protocol. A read-only snapshot of the
joint conditional ``P(sₜ₊₁ | sₜ)`` materialized as one ndarray in the
explicit-alphabet layout ``(*alphabet_sizes, n_nodes, max_alphabet)`` (per
output unit ``i``, the distribution over its next state occupies slots
``[:alphabet_sizes[i]]`` of the trailing axis; trailing slots are zero when
alphabets are heterogeneous). Produced by
:meth:`~pyphi.core.tpm.factored.FactoredTPM.to_joint` and
:meth:`~pyphi.substrate.Substrate.joint_tpm`.

Notes
-----
The array is copied at construction; the view holds no reference to its
source and does not track later mutation of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi.data_structures import np_hash
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.models.pandas import ToPandasMixin

from ._node_ops import condition as _condition


class JointTPM(Displayable, ToPandasMixin):
    __slots__ = ("_array", "_node_labels")

    def __init__(
        self, data: ArrayLike, node_labels: tuple[str, ...] | None = None
    ) -> None:
        self._array = np.array(data, dtype=np.float64)  # copy = eager snapshot
        self._node_labels = tuple(node_labels) if node_labels is not None else None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def n_nodes(self) -> int:
        # Explicit-alphabet layout: second-to-last axis is the unit axis.
        return int(self._array.shape[-2])

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape[: self.n_nodes])

    def to_array(self) -> NDArray[np.float64]:
        return self._array

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self._array
        return arr.astype(dtype) if dtype is not None else arr

    def __getitem__(self, key: Any) -> Any:
        return self._array[key]

    def condition(self, fixed: Mapping[int, int]) -> JointTPM:
        """Return the joint view with the given input units fixed."""
        return JointTPM(_condition(self._array, dict(fixed)), self._node_labels)

    def array_equal(self, other: object) -> bool:
        return np.array_equal(self._array, np.asarray(other))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointTPM):
            return NotImplemented
        return np.array_equal(self._array, other._array)

    def __hash__(self) -> int:
        return np_hash(self._array)

    def _describe(self, verbosity: int) -> Description:
        # Delegate to the shared state-by-node grid used by the B21 TPM card.
        ...  # see Task 8, Step 2 — the grid builder from _display.py

    def __repr__(self) -> str:
        return f"JointTPM(shape={self.shape})"
```

(The `_describe` body is completed in Task 8 once the display path is reconciled; for now return a minimal `Description(title="JointTPM", compact=repr(self))` so the class is importable and tests pass.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/core/test_joint_tpm.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/tpm/joint.py test/core/test_joint_tpm.py
git commit -m "Replace the JointTPM wrapper with a clean read-only joint view"
```

---

### Task 7: Delete `joint_distribution.py`; repoint exports

**Files:**
- Delete: `pyphi/core/tpm/joint_distribution.py`
- Modify: `pyphi/core/tpm/__init__.py`, `pyphi/__init__.py`

**Interfaces:**
- Produces: `pyphi.JointTPM` and `pyphi.core.tpm.JointTPM` both resolve to the Task 6 view. `pyphi.JointDistribution` export removed.

- [ ] **Step 1: Confirm nothing imports the doomed module** (Phase 1 should have cleared it):

Run: `grep -rn "joint_distribution\|JointDistribution" pyphi/ test/ --include="*.py"`
Expected: only the export lines in `pyphi/__init__.py`/`pyphi/core/tpm/__init__.py` and the file itself. Any straggler (e.g. a `simulate` importer) must be repointed or removed first.

- [ ] **Step 2: Delete the module**

```bash
git rm pyphi/core/tpm/joint_distribution.py
```

- [ ] **Step 3: Repoint `pyphi/core/tpm/__init__.py`.** Remove the `JointDistribution` re-export; ensure `from .joint import JointTPM as JointTPM` remains and points at the view.

- [ ] **Step 4: Repoint `pyphi/__init__.py`.** Change line 101 to `from .core.tpm.joint import JointTPM as JointTPM`; delete line 100 (`JointDistribution` import) and its `__all__` entry (line 150).

- [ ] **Step 5: Full suite (no path arg → doctests)**

Run: `uv run pytest -q`
Expected: PASS. Any doctest referencing the deleted `simulate`/`JointDistribution` surfaces gets updated or removed as part of this task.

- [ ] **Step 6: pyright + ruff**

Run: `uv run pyright pyphi && uv run ruff check pyphi`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Delete the legacy JointTPM/JointDistribution container; repoint exports at the view"
```

---

## PHASE 3 — Complete the joint↔factored symmetry

### Task 8: `Substrate.joint_tpm()` returns a `JointTPM`; reconcile display

**Files:**
- Modify: `pyphi/substrate.py` (`joint_tpm`), `pyphi/core/tpm/joint.py` (`_describe`), `pyphi/core/tpm/_display.py` (if needed)
- Test: `test/test_substrate_factored.py`, `test/display/test_display.py`, golden hashing in `test/golden/compute.py`

**Interfaces:**
- Consumes: `FactoredTPM.to_joint()`, the B21 grid builder in `pyphi/core/tpm/_display.py` / `FactoredTPM.grid_section`.

- [ ] **Step 1: Decide the return contract.** `Substrate.joint_tpm()` currently returns `NDArray`. Change it to return a `JointTPM` view. **Audit every caller first** — several tests and `test/golden/compute.py` do `np.asarray(substrate.joint_tpm())` or index it; because the view implements `__array__` and `__getitem__`, `np.asarray(view)` and `view[...]` keep working. Confirm:

Run: `grep -rn "joint_tpm()" pyphi/ test/ --include="*.py"`
Review each: array-consuming callers (np.asarray, indexing, `.shape`) are safe; any caller relying on the raw-ndarray *type* (`isinstance(x, np.ndarray)`) must adapt.

- [ ] **Step 2: Give the view a real display card.** Fill in `JointTPM._describe` to reuse the existing state-by-node grid the B21 `FactoredTPM`/`JointTPM` card used (in `pyphi/core/tpm/_display.py`). Extract the shared grid builder if it currently lives as a `FactoredTPM` method, so both the factored card and the joint view render the same matrix. Keep the ASCII/HTML output identical to the pre-change `JointTPM` card where a golden/doctest asserts on it.

- [ ] **Step 3: Update `joint_tpm()`**

```python
    def joint_tpm(self) -> JointTPM:
        """The joint conditional TPM as a read-only :class:`JointTPM` view.

        Materializes ``P(sₜ₊₁ | sₜ)`` from the factored storage in the
        explicit-alphabet layout ``(*alphabet_sizes, n_nodes, max_alphabet)``,
        for both binary and k-ary substrates. Recomputed on each call.
        """
        return JointTPM(self._factored_tpm.to_joint(), node_labels=self._node_labels)
```

Add `from .core.tpm.joint import JointTPM` to `substrate.py`.

- [ ] **Step 4: Update golden hashing.** `test/golden/compute.py` passes `substrate.joint_tpm()` to `substrate_hash(...)`, which does `np.ascontiguousarray(tpm).tobytes()`. `np.ascontiguousarray(view)` works via `__array__`, so hashes are unchanged. Verify no regeneration is needed:

Run: `uv run pytest test/integration/test_golden_regression.py --slow -q`
Expected: PASS, **no regeneration**.

- [ ] **Step 5: Update the doctest** in the old `subtpm` example (now gone with `joint_distribution.py`) — n/a; instead confirm the `Substrate` class docstring example (`joint_tpm()[(0,0,1)]`) still holds: indexing a view returns an ndarray slice, so it works. Run the substrate doctests:

Run: `uv run pytest --doctest-modules pyphi/substrate.py pyphi/core/tpm/joint.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite + pyright + ruff + display tests**

Run: `uv run pytest -q && uv run pyright pyphi && uv run ruff check pyphi`
Expected: all clean, no golden regeneration.

- [ ] **Step 7: Changelog + commit**

```bash
echo "\`Substrate.joint_tpm()\` returns a read-only \`JointTPM\` view (the joint peer of \`FactoredTPM\`) rather than a bare ndarray; it is array-convertible and indexable, so existing array usage is unchanged." > changelog.d/joint-tpm-view.change.md
git add -A
git commit -m "Return a JointTPM view from Substrate.joint_tpm() and unify the joint display card"
```

---

## Self-Review

- **Spec coverage:** Phase 1 removes both internal-container roles (node.py Task 2, repertoire_algebra Task 3, dynamics Task 4, marginalization Task 5) — the prerequisite. Phase 2 builds the eager-snapshot view (Task 6) and deletes the legacy classes + numpy-proxy base (Task 7). Phase 3 completes the symmetry (Task 8: `joint_tpm()` → view, display reconciled, export repointed). The three ROADMAP sub-bullets (consolidate two classes, remove ArrayLike proxy, k-ary view + `joint_tpm()` returns it) are all covered.
- **Dead code:** binary `else` branch, four `_off`/`_on` properties, `expand_node_tpm` (Task 2); `simulate`/`permute_nodes`/`subtpm`/`is_deterministic`/`to_multidimensional_state_by_node`/`validate*` all vanish with `joint_distribution.py` (Task 7) — confirm no importer in Task 7 Step 1.
- **φ-invariance:** every φ-path task (2, 3, 5, 8) ends with the golden suite asserted green *without* regeneration.
- **Type consistency:** `marginalize_out`/`condition` signatures identical across `_node_ops.py`, node.py, repertoire_algebra.py; the `condition` free function vs the `condition` mapping variable collision is resolved by renaming the variable to `condition_map` (Task 3 Step 2).

## Open items to confirm during execution

- **`_describe` display parity (Task 8 Step 2) — CONFIRMED risk.** The B21 card (`_describe`, `_to_pandas`, `to_xarray`) lives on the legacy `joint_distribution.JointTPM` (lines ~520–560); the `joint.py` wrapper has none. So the view (Task 6) must carry the display, and Task 7's delete removes the only copy. Two consequences: (1) port the display onto the view **before** deleting `joint_distribution.py` (move Task 8 Step 2 ahead of Task 7, or stub-then-fill); (2) the legacy `_describe` uses the *binary* `_display.state_by_node_grid`, but the view holds the k-ary explicit-alphabet array — reuse the **k-ary-aware** grid instead (the `FactoredTPM.grid_section` logic in `factored.py`, which already branches binary vs k-ary), extracting it to a shared builder so the factored card and the joint view render identically. Where a golden/doctest asserts on the old binary card text, keep that output byte-identical.
- **`JointDistribution` public removal:** dropping the `pyphi.JointDistribution` export is a public API removal. Acceptable for unreleased 2.0, but note it in the Task 7 commit body.
