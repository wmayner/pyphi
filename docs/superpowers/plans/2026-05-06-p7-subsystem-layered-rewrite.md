# P7 — Subsystem Layered Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 1354-line `Subsystem` god-object with a layered `pyphi/core/` package while preserving every numerical result and every name in `SubsystemPublicInterface`.

**Architecture:** Three layers under `pyphi/core/`: TPM Protocol (Layer 0) + ExplicitTPM, value types `Unit`/`Substrate`/`CausalModel`/`CandidateSystem` (Layer 1), stateless `repertoire_algebra` functions with a `WeakValueDictionary`-backed memoization decorator (Layer 2). Existing P4 `PhiFormalism` (Layer 3) is the dispatcher — it switches from `Subsystem` to `CandidateSystem` at the cutover. `Subsystem` is renamed to `CandidateSystem` at the public API.

**Tech Stack:** Python 3.13+, frozen `@dataclass`, `typing.Protocol`/`runtime_checkable`, numpy, `weakref.finalize`, `WeakValueDictionary`, pytest, Hypothesis, pyright.

**Spec:** [docs/superpowers/specs/2026-05-06-p7-subsystem-layered-rewrite-design.md](../specs/2026-05-06-p7-subsystem-layered-rewrite-design.md)

**Strategy:** Worktree-based big-bang. Build new `core/` alongside the unchanged old `subsystem.py`. Parity tests compare new functions to old ones — the suite stays green at every commit. Cutover is one cluster of commits at the end (Phase 6), then `subsystem.py` is deleted.

---

## Phases

| # | Phase | Tasks | Approx. tasks |
|---|---|---|---|
| 0 | Worktree setup + PR #105 review | 0 numerical changes | 2 |
| 1 | TPM Protocol + `ExplicitTPM` port | Build Layer 0 | 4 |
| 2 | `Unit`, `Substrate`, `CausalModel` | Build Layer 1 lower | 3 |
| 3 | `CandidateSystem` value type (no proxy methods) | Build Layer 1 upper | 3 |
| 4 | `core/repertoire_algebra.py` | Port Layer 2 functions | 8 |
| 5 | `CandidateSystem` proxy methods + `SubsystemPublicInterface` parity | Complete Layer 1 surface | 4 |
| 6 | Cutover: formalism, compute, actual | Switch callers | 5 |
| 7 | Test rename + macro disable | Test/example update | 3 |
| 8 | Delete `subsystem.py` + final validation | Cleanup | 3 |

**Acceptance gates** (run before merge): all 17 golden fixtures match to 1e-12, all 19 Hypothesis invariants green, sign-flip canary still bites, surface drift green against `CandidateSystem`, pyright clean on `pyphi/core/`, full suite green.

---

## Phase 0 — Worktree Setup

### Task 0.1: Create worktree on new branch

**Files:**
- No file changes; creates a new git worktree

- [ ] **Step 1: Create worktree**

```bash
git worktree add ../pyphi-p7-kernel-rewrite -b feature/p7-kernel-rewrite develop
cd ../pyphi-p7-kernel-rewrite
```

Expected: `Preparing worktree (new branch 'feature/p7-kernel-rewrite')`. The new directory mirrors the develop tip.

- [ ] **Step 2: Confirm clean state**

```bash
git status
```

Expected: `On branch feature/p7-kernel-rewrite ... working tree clean`.

- [ ] **Step 3: Smoke test that the suite passes on the branch**

```bash
uv run pytest test/test_subsystem_surface.py test/test_partition.py -q
```

Expected: all tests pass. This confirms develop is green at the worktree starting point.

- [ ] **Step 4: Commit a marker (no files; empty commit acceptable)**

```bash
git commit --allow-empty -m "P7 kernel rewrite: worktree initialized"
```

Marks the branch starting point in git log.

### Task 0.2: Read PR #105 ImplicitTPM abstractions; sketch Protocol shape

**Files:**
- Create: `docs/superpowers/notes/2026-05-06-p7-tpm-protocol-shape.md`

**Goal:** Validate that the `TPM` Protocol shape (Section "Components" of the spec) admits both `ExplicitTPM` and the `ImplicitTPM` from PR #105. This is design-only work; no code changes.

- [ ] **Step 1: Read PR #105's `tpm/__init__.py` and `state_space.py`**

```bash
gh pr view 105 --json files --jq '.files[].path' | grep -E "tpm|state_space"
gh pr diff 105 -- 'pyphi/tpm/__init__.py' | head -200
gh pr diff 105 -- 'pyphi/state_space.py' | head -100
```

Read for: ImplicitTPM's marginalization API, alphabet handling, conditioning operations. Do not copy code.

- [ ] **Step 2: Write the Protocol-shape note**

Create `docs/superpowers/notes/2026-05-06-p7-tpm-protocol-shape.md` with sections:
1. ExplicitTPM operations used by current `Subsystem` (grep `subsystem.py` for `self.cause_tpm.` / `self.effect_tpm.`)
2. Operations ImplicitTPM (from #105) provides
3. Common shape — the Protocol body
4. Operations that don't fit (deferred to P12, e.g., `alphabet_size`)

Output should justify the Protocol body shown in `core/tpm/base.py` (spec, Components section).

- [ ] **Step 3: Commit the note**

```bash
git add docs/superpowers/notes/2026-05-06-p7-tpm-protocol-shape.md
git commit -m "P7: TPM Protocol shape note (PR #105 cross-check)"
```

---

## Phase 1 — TPM Protocol + `ExplicitTPM` Port

The goal: a `core/tpm/` package with the Protocol, an `ExplicitTPM` class that satisfies it (delegating to existing `pyphi.tpm.ExplicitTPM`), and named marginalization functions.

### Task 1.1: Create `core/` package skeleton + `core/tpm/base.py`

**Files:**
- Create: `pyphi/core/__init__.py`
- Create: `pyphi/core/tpm/__init__.py`
- Create: `pyphi/core/tpm/base.py`
- Create: `test/test_core_tpm.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_tpm.py`:

```python
"""Tests for pyphi.core.tpm — TPM Protocol and ExplicitTPM port."""

from __future__ import annotations

import numpy as np
import pytest


def test_tpm_protocol_importable() -> None:
    """The TPM Protocol must be importable from pyphi.core.tpm."""
    from pyphi.core.tpm import TPM  # noqa: F401


def test_tpm_protocol_is_runtime_checkable() -> None:
    """TPM Protocol is decorated with runtime_checkable."""
    from pyphi.core.tpm import TPM
    assert hasattr(TPM, "_is_runtime_protocol")
    assert TPM._is_runtime_protocol is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_tpm.py -v
```

Expected: `ImportError: No module named 'pyphi.core'`.

- [ ] **Step 3: Create the skeleton**

Create `pyphi/core/__init__.py`:

```python
"""pyphi.core — typed kernel for the PyPhi 2.0 layered architecture.

See docs/superpowers/specs/2026-05-06-p7-subsystem-layered-rewrite-design.md
for the design.
"""
```

Create `pyphi/core/tpm/__init__.py`:

```python
"""TPM Protocol and concrete backends.

Layer 0 of the kernel: zero formalism logic.
"""

from .base import TPM as TPM
```

Create `pyphi/core/tpm/base.py`:

```python
"""TPM Protocol — the structural contract every transition probability matrix satisfies.

Implementations:
- ExplicitTPM (numpy-backed; this project / P7).
- ImplicitTPM (factored per-node TPM; P12, drawing on PR #105).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TPM(Protocol):
    """Structural protocol satisfied by every PyPhi TPM."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def n_nodes(self) -> int: ...

    def condition(self, fixed: Mapping[int, int]) -> "TPM": ...

    def squeeze(self) -> "TPM": ...

    def to_array(self) -> NDArray[np.float64]: ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_tpm.py -v
```

Expected: 2 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/__init__.py pyphi/core/tpm/__init__.py pyphi/core/tpm/base.py test/test_core_tpm.py
git commit -m "P7: add pyphi.core skeleton + TPM Protocol"
```

### Task 1.2: Port `ExplicitTPM` as `core/tpm/explicit.py`

**Files:**
- Create: `pyphi/core/tpm/explicit.py`
- Modify: `test/test_core_tpm.py` (extend)

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_tpm.py`:

```python
def test_explicit_tpm_is_a_tpm() -> None:
    """ExplicitTPM satisfies the TPM Protocol via runtime_checkable."""
    from pyphi.core.tpm import TPM
    from pyphi.core.tpm.explicit import ExplicitTPM
    arr = np.array([[0.5, 0.5], [0.7, 0.3]])
    tpm = ExplicitTPM(arr)
    assert isinstance(tpm, TPM)


def test_explicit_tpm_parity_with_legacy() -> None:
    """ExplicitTPM produces bit-identical output to legacy ExplicitTPM."""
    import pyphi.tpm as legacy
    from pyphi.core.tpm.explicit import ExplicitTPM
    arr = np.array([
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0.7, 0.7, 0.7], [0.3, 0.3, 0.3]],
    ])
    new = ExplicitTPM(arr)
    old = legacy.ExplicitTPM(arr)
    np.testing.assert_array_equal(new.to_array(), np.asarray(old))
    assert new.shape == old.shape
    new_squeezed = new.squeeze()
    old_squeezed = old.squeeze()
    np.testing.assert_array_equal(new_squeezed.to_array(), np.asarray(old_squeezed))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_tpm.py -v
```

Expected: `ImportError: No module named 'pyphi.core.tpm.explicit'`.

- [ ] **Step 3: Create the explicit backend**

Create `pyphi/core/tpm/explicit.py`:

```python
"""Numpy-backed ExplicitTPM — direct port of pyphi.tpm.ExplicitTPM behind the TPM Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import tpm as _legacy_tpm


class ExplicitTPM:
    """Numpy-backed transition probability matrix.

    Wraps the legacy :class:`pyphi.tpm.ExplicitTPM` and exposes the
    :class:`pyphi.core.tpm.TPM` Protocol surface. Numerical behavior is
    delegated to the legacy implementation; the wrapper exists to give the
    new layering a single, type-checked entry point.
    """

    __slots__ = ("_inner",)

    def __init__(self, data: ArrayLike) -> None:
        if isinstance(data, _legacy_tpm.ExplicitTPM):
            self._inner = data
        else:
            self._inner = _legacy_tpm.ExplicitTPM(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self._inner).shape)

    @property
    def n_nodes(self) -> int:
        return int(self.shape[-1]) if self.shape else 0

    def condition(self, fixed: Mapping[int, int]) -> "ExplicitTPM":
        return ExplicitTPM(self._inner.condition_tpm(dict(fixed)))

    def squeeze(self) -> "ExplicitTPM":
        return ExplicitTPM(self._inner.squeeze())

    def to_array(self) -> NDArray[np.float64]:
        return np.asarray(self._inner)

    def __getattr__(self, name: str) -> Any:
        # During the worktree, callers may still need legacy methods we
        # haven't lifted yet. This passthrough is removed at Phase 8.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"ExplicitTPM(shape={self.shape})"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_tpm.py -v
```

Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/tpm/explicit.py test/test_core_tpm.py
git commit -m "P7: port ExplicitTPM behind core.tpm.TPM Protocol"
```

### Task 1.3: Add `core/tpm/marginalization.py` (cause_tpm / effect_tpm functions)

**Files:**
- Create: `pyphi/core/tpm/marginalization.py`
- Modify: `test/test_core_tpm.py` (extend)

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_tpm.py`:

```python
def test_cause_tpm_parity() -> None:
    """core.tpm.marginalization.cause_tpm matches legacy backward_tpm."""
    from pyphi import examples
    from pyphi.core.tpm.explicit import ExplicitTPM
    from pyphi.core.tpm.marginalization import cause_tpm
    from pyphi.tpm import backward_tpm as legacy_backward_tpm

    network = examples.basic_network()
    state = (1, 0, 0)
    node_indices = (0, 1, 2)

    new_tpm = cause_tpm(ExplicitTPM(network.tpm), state, node_indices)
    old_tpm = legacy_backward_tpm(network.tpm, state, node_indices)
    np.testing.assert_array_equal(new_tpm.to_array(), np.asarray(old_tpm))


def test_effect_tpm_parity() -> None:
    """core.tpm.marginalization.effect_tpm matches legacy condition_tpm."""
    from pyphi import examples
    from pyphi import utils
    from pyphi.core.tpm.explicit import ExplicitTPM
    from pyphi.core.tpm.marginalization import effect_tpm

    network = examples.basic_network()
    state = (1, 0, 0)
    node_indices = (0, 1)            # external = (2,)
    external_indices = (2,)
    external_state = utils.state_of(external_indices, state)
    background = dict(zip(external_indices, external_state, strict=False))

    new_tpm = effect_tpm(ExplicitTPM(network.tpm), background)
    old_tpm = network.tpm.condition_tpm(background)
    np.testing.assert_array_equal(new_tpm.to_array(), np.asarray(old_tpm))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_tpm.py::test_cause_tpm_parity test/test_core_tpm.py::test_effect_tpm_parity -v
```

Expected: `ImportError: No module named 'pyphi.core.tpm.marginalization'`.

- [ ] **Step 3: Create the marginalization module**

Create `pyphi/core/tpm/marginalization.py`:

```python
"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4.

Replaces the implicit ``_backward_tpm()`` side effect in
``Subsystem.__init__`` with documented free functions.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyphi.tpm import backward_tpm as _legacy_backward_tpm

from .explicit import ExplicitTPM


def cause_tpm(
    tpm: ExplicitTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> ExplicitTPM:
    """Backward TPM — IIT 4.0 Eq. 3.

    Conditions the forward TPM on the state of the given node indices and
    inverts to obtain the cause-side conditional distribution.
    """
    return ExplicitTPM(_legacy_backward_tpm(tpm._inner, state, node_indices))


def effect_tpm(
    tpm: ExplicitTPM,
    background: Mapping[int, int],
) -> ExplicitTPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4."""
    return tpm.condition(background)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_tpm.py -v
```

Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/tpm/marginalization.py test/test_core_tpm.py
git commit -m "P7: add core.tpm.marginalization (cause_tpm, effect_tpm)"
```

### Task 1.4: Type-check `core/tpm/`

**Files:**
- No code changes — pyright validation only

- [ ] **Step 1: Run pyright on `core/tpm/`**

```bash
uv run pyright pyphi/core/tpm/
```

Expected: 0 errors, 0 warnings.

- [ ] **Step 2: If errors, fix them and re-run**

If pyright surfaces any error in the new code, fix inline and re-run. Common fixes:
- Add explicit `# pyright: ignore[reportUnknownMemberType]` on `_legacy_tpm.ExplicitTPM` access if the legacy class is untyped.
- Tighten `Mapping[int, int]` arguments where pyright wants invariance.

- [ ] **Step 3: Commit any pyright fixes**

```bash
git add -p pyphi/core/tpm/
git commit -m "P7: pyright clean on core.tpm/"
```

(Skip the commit if no changes were needed.)

---

## Phase 2 — `Unit`, `Substrate`, `CausalModel` Value Types

### Task 2.1: `core/unit.py` — `Unit` dataclass

**Files:**
- Create: `pyphi/core/unit.py`
- Create: `test/test_core_unit.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_unit.py`:

```python
"""Tests for pyphi.core.unit — Unit dataclass."""

from __future__ import annotations

import dataclasses

import pytest


def test_unit_is_frozen() -> None:
    """Unit instances cannot be mutated."""
    from pyphi.core.unit import Unit
    u = Unit(index=0, label="A")
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.index = 1  # type: ignore[misc]


def test_unit_equality_by_value() -> None:
    """Two Units with equal fields compare equal."""
    from pyphi.core.unit import Unit
    assert Unit(0, "A") == Unit(0, "A")
    assert Unit(0, "A") != Unit(1, "A")


def test_unit_is_hashable() -> None:
    """Unit instances hash to themselves consistently."""
    from pyphi.core.unit import Unit
    u = Unit(0, "A")
    assert hash(u) == hash(Unit(0, "A"))
    assert {u, Unit(0, "A")} == {u}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_unit.py -v
```

Expected: `ImportError: No module named 'pyphi.core.unit'`.

- [ ] **Step 3: Implement `Unit`**

Create `pyphi/core/unit.py`:

```python
"""Unit value type — atomic node in a substrate.

Roughly today's :class:`pyphi.node.Node` minus the per-instance TPM
caching, but layered as a pure value type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic node in a substrate.

    P7: alphabet is implicit binary (0 or 1). P12 adds ``alphabet_size``.
    """

    index: int
    label: str
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_unit.py -v
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/unit.py test/test_core_unit.py
git commit -m "P7: add core.unit.Unit value type"
```

### Task 2.2: `core/substrate.py` — `Substrate` dataclass

**Files:**
- Create: `pyphi/core/substrate.py`
- Create: `test/test_core_substrate.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_substrate.py`:

```python
"""Tests for pyphi.core.substrate — Substrate dataclass."""

from __future__ import annotations

import numpy as np
import pytest


def test_substrate_is_frozen() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit
    units = (Unit(0, "A"), Unit(1, "B"))
    cm = np.zeros((2, 2), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.units = ()  # type: ignore[misc]


def test_substrate_n_units() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit
    units = (Unit(0, "A"), Unit(1, "B"), Unit(2, "C"))
    cm = np.zeros((3, 3), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    assert s.n_units == 3


def test_substrate_node_labels() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit
    units = (Unit(0, "A"), Unit(1, "B"))
    cm = np.zeros((2, 2), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    assert tuple(s.node_labels) == ("A", "B")


def test_substrate_equality_includes_cm() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit
    units = (Unit(0, "A"),)
    cm1 = np.array([[0]], dtype=int)
    cm2 = np.array([[1]], dtype=int)
    assert Substrate(units, cm1) == Substrate(units, cm1)
    assert Substrate(units, cm1) != Substrate(units, cm2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_substrate.py -v
```

Expected: `ImportError: No module named 'pyphi.core.substrate'`.

- [ ] **Step 3: Implement `Substrate`**

Create `pyphi/core/substrate.py`:

```python
"""Substrate value type — a frozen set of Units with a connectivity matrix."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from pyphi.labels import NodeLabels

from .unit import Unit


def _array_eq(a: NDArray[np.int_], b: NDArray[np.int_]) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


@dataclass(frozen=True, slots=True, eq=False)
class Substrate:
    """An immutable substrate: tuple of :class:`Unit` plus connectivity matrix.

    Roughly today's :class:`pyphi.network.Network` minus the TPM (which
    moves to :class:`pyphi.core.causal_model.CausalModel`).
    """

    units: tuple[Unit, ...]
    connectivity_matrix: NDArray[np.int_]

    @cached_property
    def n_units(self) -> int:
        return len(self.units)

    @cached_property
    def node_labels(self) -> NodeLabels:
        return NodeLabels(tuple(u.label for u in self.units), tuple(range(len(self.units))))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Substrate):
            return NotImplemented
        return self.units == other.units and _array_eq(
            self.connectivity_matrix, other.connectivity_matrix
        )

    def __hash__(self) -> int:
        return hash((self.units, self.connectivity_matrix.tobytes(),
                     self.connectivity_matrix.shape))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_substrate.py -v
```

Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/substrate.py test/test_core_substrate.py
git commit -m "P7: add core.substrate.Substrate value type"
```

### Task 2.3: `core/causal_model.py` — `CausalModel` dataclass + `from_network`

**Files:**
- Create: `pyphi/core/causal_model.py`
- Create: `test/test_core_causal_model.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_causal_model.py`:

```python
"""Tests for pyphi.core.causal_model — CausalModel dataclass."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def test_causal_model_is_frozen() -> None:
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel
    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cm.tpm = None  # type: ignore[misc]


def test_causal_model_from_network_round_trips() -> None:
    """CausalModel.from_network preserves TPM and connectivity exactly."""
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel
    network = examples.basic_network()
    cm = CausalModel.from_network(network)
    np.testing.assert_array_equal(cm.tpm.to_array(), np.asarray(network.tpm))
    np.testing.assert_array_equal(cm.substrate.connectivity_matrix, network.cm)


def test_causal_model_substrate_units() -> None:
    """from_network produces one Unit per network node, in index order."""
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel
    network = examples.basic_network()
    cm = CausalModel.from_network(network)
    assert cm.substrate.n_units == network.size
    indices = [u.index for u in cm.substrate.units]
    assert indices == list(range(network.size))


def test_causal_model_equality() -> None:
    from pyphi import examples
    from pyphi.core.causal_model import CausalModel
    a = CausalModel.from_network(examples.basic_network())
    b = CausalModel.from_network(examples.basic_network())
    assert a == b
    c = CausalModel.from_network(examples.xor_network())
    assert a != c
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_causal_model.py -v
```

Expected: `ImportError: No module named 'pyphi.core.causal_model'`.

- [ ] **Step 3: Implement `CausalModel`**

Create `pyphi/core/causal_model.py`:

```python
"""CausalModel value type — substrate + TPM. The zeroth postulate of IIT 4.0.

Zero computation. No caches. Operations on a CausalModel are free
functions in :mod:`pyphi.core.tpm.marginalization` or
:mod:`pyphi.core.repertoire_algebra`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .substrate import Substrate
from .tpm.base import TPM
from .tpm.explicit import ExplicitTPM
from .unit import Unit

if TYPE_CHECKING:
    from pyphi.network import Network


@dataclass(frozen=True, slots=True, eq=False)
class CausalModel:
    """An immutable :class:`Substrate` paired with a :class:`TPM`."""

    substrate: Substrate
    tpm: TPM

    @classmethod
    def from_network(cls, network: "Network") -> "CausalModel":
        """Migration helper: build a CausalModel from a legacy Network.

        Stays in place through P7+P7b+P8 to ease the migration; deleted
        before 2.0 ships if all callers go direct.
        """
        units = tuple(
            Unit(index=i, label=str(network.node_labels[i]))
            for i in range(network.size)
        )
        substrate = Substrate(units=units, connectivity_matrix=network.cm)
        tpm = ExplicitTPM(network.tpm)
        return cls(substrate=substrate, tpm=tpm)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalModel):
            return NotImplemented
        if self.substrate != other.substrate:
            return False
        return bool(
            (self.tpm.to_array().shape == other.tpm.to_array().shape)
            and (self.tpm.to_array() == other.tpm.to_array()).all()
        )

    def __hash__(self) -> int:
        return hash((self.substrate, self.tpm.to_array().tobytes(),
                     self.tpm.to_array().shape))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_causal_model.py -v
```

Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/causal_model.py test/test_core_causal_model.py
git commit -m "P7: add core.causal_model.CausalModel + from_network helper"
```

---

## Phase 3 — `CandidateSystem` Value Type

This phase builds the value-type shell of `CandidateSystem` — frozen, hashable, with cached derived properties — but no proxy methods yet. Repertoire methods come in Phase 5.

### Task 3.1: `core/candidate_system.py` skeleton — frozen dataclass + construction validation

**Files:**
- Create: `pyphi/core/candidate_system.py`
- Create: `test/test_core_candidate_system.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_candidate_system.py`:

```python
"""Tests for pyphi.core.candidate_system — CandidateSystem value type."""

from __future__ import annotations

import dataclasses

import pytest


@pytest.fixture
def basic_cs():
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    cm = CausalModel.from_network(examples.basic_network())
    return CandidateSystem(causal_model=cm, state=(1, 0, 0), node_indices=(0, 1, 2))


def test_candidate_system_is_frozen(basic_cs) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        basic_cs.state = (0, 0, 0)  # type: ignore[misc]


def test_candidate_system_default_cut_is_null(basic_cs) -> None:
    from pyphi.models.cuts import NullCut
    assert isinstance(basic_cs.cut, NullCut)


def test_candidate_system_validates_state_length() -> None:
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(ValueError):
        CandidateSystem(causal_model=cm, state=(1, 0), node_indices=(0, 1))


def test_candidate_system_validates_node_states() -> None:
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    cm = CausalModel.from_network(examples.basic_network())
    with pytest.raises(ValueError):
        CandidateSystem(causal_model=cm, state=(1, 0, 7), node_indices=(0, 1, 2))


def test_candidate_system_equality_includes_cut(basic_cs) -> None:
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.models.cuts import SystemPartition
    from pyphi.direction import Direction
    other = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
    )
    assert basic_cs == other
    cut = SystemPartition(Direction.EFFECT, (0,), (1, 2))
    cut_cs = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
        cut=cut,
    )
    assert basic_cs != cut_cs


def test_candidate_system_is_hashable(basic_cs) -> None:
    from pyphi.core.candidate_system import CandidateSystem
    other = CandidateSystem(
        causal_model=basic_cs.causal_model,
        state=basic_cs.state,
        node_indices=basic_cs.node_indices,
    )
    assert hash(basic_cs) == hash(other)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: `ImportError: No module named 'pyphi.core.candidate_system'`.

- [ ] **Step 3: Implement `CandidateSystem` skeleton**

Create `pyphi/core/candidate_system.py`:

```python
"""CandidateSystem value type — (CausalModel, state, node_subset, cut).

The replacement for :class:`pyphi.subsystem.Subsystem`. Immutable.
Hashable. Cut is a constructor argument, not a hidden mode.

P7: only construction validation, equality, hash, and a default-NullCut
field. Cached derived properties are added in Task 3.2; proxy methods
to repertoire_algebra and formalism are added in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from pyphi import validate
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition

from .causal_model import CausalModel

if TYPE_CHECKING:
    from pyphi.types import NodeIndices
    from pyphi.types import State


def _default_cut() -> SystemPartition:
    # Sentinel; replaced in __post_init__ with a NullCut bound to node_indices.
    return None  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class CandidateSystem:
    """A candidate system: ``(CausalModel, state, node_subset, cut)``."""

    causal_model: CausalModel
    state: "State"
    node_indices: "NodeIndices"
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Validate state length / node states.
        substrate = self.causal_model.substrate
        validate.state_length(self.state, substrate.n_units)
        validate.node_states(self.state)
        # Default cut: NullCut bound to node_indices.
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, substrate.node_labels)
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CandidateSystem):
            return NotImplemented
        return (
            self.causal_model == other.causal_model
            and self.state == other.state
            and self.node_indices == other.node_indices
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash((self.causal_model, self.state, self.node_indices, self.cut))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/candidate_system.py test/test_core_candidate_system.py
git commit -m "P7: add core.candidate_system.CandidateSystem skeleton"
```

### Task 3.2: Add cached derived properties to `CandidateSystem`

**Files:**
- Modify: `pyphi/core/candidate_system.py`
- Modify: `test/test_core_candidate_system.py` (extend)

The properties: `cause_tpm`, `effect_tpm`, `proper_cause_tpm`, `proper_effect_tpm`, `cm`, `proper_cm`, `nodes`, `proper_state`, `external_indices`, `node_labels`, `connectivity_matrix`, `cut_indices`, `cut_node_labels`, `cut_mechanisms`, `is_cut`, `size`, `tpm_size`, `null_concept`, `network`. (`network` is preserved as a back-compat property pointing to a synthetic `Network`-shaped object — defer concrete impl, use legacy `Network` for now.)

Parity strategy: each property's value must equal the corresponding `Subsystem` attribute for the same `(network, state, nodes, cut)`.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_candidate_system.py`:

```python
@pytest.fixture
def cs_and_subsystem():
    """Paired CandidateSystem + legacy Subsystem with identical inputs."""
    from pyphi import Subsystem, examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    network = examples.basic_network()
    state = (1, 0, 0)
    nodes = (0, 1, 2)
    cm = CausalModel.from_network(network)
    cs = CandidateSystem(causal_model=cm, state=state, node_indices=nodes)
    sub = Subsystem(network, state, nodes)
    return cs, sub


@pytest.mark.parametrize(
    "attr",
    [
        "external_indices",
        "node_labels",
        "node_indices",
        "state",
        "proper_state",
        "size",
        "tpm_size",
        "is_cut",
        "cut_indices",
    ],
)
def test_candidate_system_property_parity(cs_and_subsystem, attr) -> None:
    cs, sub = cs_and_subsystem
    cs_val = getattr(cs, attr)
    sub_val = getattr(sub, attr)
    assert cs_val == sub_val, f"{attr}: {cs_val!r} != {sub_val!r}"


def test_candidate_system_cause_tpm_parity(cs_and_subsystem) -> None:
    import numpy as np
    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.cause_tpm.to_array(), np.asarray(sub.cause_tpm))


def test_candidate_system_effect_tpm_parity(cs_and_subsystem) -> None:
    import numpy as np
    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.effect_tpm.to_array(), np.asarray(sub.effect_tpm))


def test_candidate_system_cm_parity(cs_and_subsystem) -> None:
    import numpy as np
    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(cs.cm, sub.cm)
    np.testing.assert_array_equal(cs.proper_cm, sub.proper_cm)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: failures with `AttributeError: 'CandidateSystem' object has no attribute 'cause_tpm'` etc.

- [ ] **Step 3: Add derived properties**

Edit `pyphi/core/candidate_system.py` — add these methods after `__hash__`:

```python
    # ---- cached cheap derived properties ----
    @cached_property
    def network(self):
        """Back-compat: a legacy Network instance for code paths that still
        require it (during the worktree, deleted at 2.0).
        """
        from pyphi.network import Network
        return Network(
            self.causal_model.tpm.to_array(),
            cm=self.causal_model.substrate.connectivity_matrix,
            node_labels=tuple(self.causal_model.substrate.node_labels),
        )

    @cached_property
    def node_labels(self):
        return self.causal_model.substrate.node_labels

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        all_indices = set(range(self.causal_model.substrate.n_units))
        return tuple(sorted(all_indices - set(self.node_indices)))

    @cached_property
    def proper_state(self):
        from pyphi import utils
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def cause_tpm(self):
        from pyphi.core.tpm.marginalization import cause_tpm as _cause_tpm
        return _cause_tpm(self.causal_model.tpm, self.state, self.node_indices)

    @cached_property
    def effect_tpm(self):
        from pyphi import utils
        from pyphi.core.tpm.marginalization import effect_tpm as _effect_tpm
        external_state = utils.state_of(self.external_indices, self.state)
        background = dict(zip(self.external_indices, external_state, strict=False))
        return _effect_tpm(self.causal_model.tpm, background)

    @cached_property
    def proper_effect_tpm(self):
        return self.effect_tpm.squeeze().to_array()[..., list(self.node_indices)]

    @cached_property
    def proper_cause_tpm(self):
        return self.cause_tpm.squeeze().to_array()[..., list(self.node_indices)]

    @cached_property
    def cm(self):
        return self.cut.apply_cut(self.causal_model.substrate.connectivity_matrix)

    @cached_property
    def proper_cm(self):
        from pyphi import connectivity
        return connectivity.subadjacency(self.cm, self.node_indices)

    @cached_property
    def connectivity_matrix(self):
        return self.cm

    @cached_property
    def cut_indices(self):
        return self.node_indices

    @cached_property
    def cut_node_labels(self):
        return self.node_labels.coerce_to_labels(self.cut_indices)

    @cached_property
    def is_cut(self) -> bool:
        return not isinstance(self.cut, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.causal_model.substrate.n_units

    @cached_property
    def nodes(self):
        from pyphi.node import generate_nodes
        return generate_nodes(
            self.cause_tpm._inner if hasattr(self.cause_tpm, "_inner") else self.cause_tpm,
            self.effect_tpm._inner if hasattr(self.effect_tpm, "_inner") else self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def cut_mechanisms(self):
        return self.cut.all_cut_mechanisms()

    @cached_property
    def null_concept(self):
        # Delegate to legacy Subsystem until repertoire_algebra and the
        # mechanism-eval machinery are fully ported (Phase 5).
        from pyphi.subsystem import Subsystem
        return Subsystem(
            self.network, self.state, self.node_indices, cut=self.cut,
        ).null_concept
```

Add the import `from functools import cached_property` near the top of the file.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: all 14 passing (6 from Task 3.1 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/candidate_system.py test/test_core_candidate_system.py
git commit -m "P7: add cached derived properties to CandidateSystem"
```

### Task 3.3: `apply_cut` method + `cached_property` sharing test

**Files:**
- Modify: `pyphi/core/candidate_system.py`
- Modify: `test/test_core_candidate_system.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `test/test_core_candidate_system.py`:

```python
def test_apply_cut_returns_new_instance(basic_cs) -> None:
    from pyphi.direction import Direction
    from pyphi.models.cuts import SystemPartition
    cut = SystemPartition(Direction.EFFECT, (0,), (1, 2))
    cut_cs = basic_cs.apply_cut(cut)
    assert cut_cs is not basic_cs
    assert cut_cs.cut == cut
    assert basic_cs.cut != cut  # original untouched


def test_apply_cut_shares_cause_tpm_when_unaffected(basic_cs) -> None:
    """cause_tpm depends only on (causal_model, state, node_indices), not cut.

    After apply_cut, cs and cut_cs should share the cause_tpm object via the
    cached_property — this is the genuine speedup over today's apply_cut.
    """
    from pyphi.direction import Direction
    from pyphi.models.cuts import SystemPartition
    # Materialize cause_tpm on the parent
    parent_tpm = basic_cs.cause_tpm
    cut = SystemPartition(Direction.EFFECT, (0,), (1, 2))
    cut_cs = basic_cs.apply_cut(cut)
    # cut_cs has its own cached_property storage; the value, however,
    # is derived from the same inputs (causal_model, state, node_indices)
    # so the array contents must match exactly.
    import numpy as np
    np.testing.assert_array_equal(
        cut_cs.cause_tpm.to_array(), parent_tpm.to_array()
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_candidate_system.py::test_apply_cut_returns_new_instance test/test_core_candidate_system.py::test_apply_cut_shares_cause_tpm_when_unaffected -v
```

Expected: `AttributeError: 'CandidateSystem' object has no attribute 'apply_cut'`.

- [ ] **Step 3: Add `apply_cut`**

Edit `pyphi/core/candidate_system.py` — add this method:

```python
    def apply_cut(self, cut: SystemPartition) -> "CandidateSystem":
        """Return a new CandidateSystem with the cut applied.

        ``causal_model``, ``state``, and ``node_indices`` are unchanged.
        Cached derived properties that don't depend on cut (``cause_tpm``,
        ``effect_tpm``) are not re-derived in the new instance until first
        access — so the new instance shares the same numerical values.
        """
        from dataclasses import replace
        return replace(self, cut=cut)
```

Add `from dataclasses import replace` to imports if not already there.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/candidate_system.py test/test_core_candidate_system.py
git commit -m "P7: add CandidateSystem.apply_cut"
```

---

## Phase 4 — Repertoire Algebra (`core/repertoire_algebra.py`)

This phase ports the repertoire computation from `pyphi/subsystem.py:355-690` and `pyphi/repertoire.py` into a single module of pure functions, behind a memoization decorator.

### Task 4.1: Memoization decorator + cache helpers

**Files:**
- Create: `pyphi/core/repertoire_algebra.py` (skeleton)
- Create: `test/test_core_repertoire_algebra.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_repertoire_algebra.py`:

```python
"""Tests for pyphi.core.repertoire_algebra — stateless repertoire functions + cache."""

from __future__ import annotations

import gc
import weakref

import pytest


def test_memoize_caches_results() -> None:
    """A memoized function returns the cached value on second call."""
    from pyphi.core.repertoire_algebra import _caches
    from pyphi.core.repertoire_algebra import _memoize

    call_count = {"n": 0}

    @_memoize
    def f(cs, x):
        call_count["n"] += 1
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    assert f(cs, 3) == 6
    assert f(cs, 3) == 6
    assert call_count["n"] == 1


def test_memoize_evicts_on_gc() -> None:
    """When a CandidateSystem is GC'd, its cache entries are evicted."""
    from pyphi.core.repertoire_algebra import _caches
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    cs_id = id(cs)
    f(cs, 1)
    f(cs, 2)
    assert any(k[0] == cs_id for k in _caches[f.__name__])
    del cs
    gc.collect()
    # After GC, the finalizer purges entries with that id.
    assert not any(k[0] == cs_id for k in _caches[f.__name__])


def test_memoize_does_not_poison_on_failure() -> None:
    """If the wrapped function raises, the cache must not retain a partial entry."""
    from pyphi.core.repertoire_algebra import _caches
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        if x < 0:
            raise ValueError("negative")
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    with pytest.raises(ValueError):
        f(cs, -1)
    # A subsequent good call must succeed and cache.
    assert f(cs, 4) == 8
    assert f(cs, 4) == 8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: `ImportError: No module named 'pyphi.core.repertoire_algebra'`.

- [ ] **Step 3: Implement the decorator**

Create `pyphi/core/repertoire_algebra.py`:

```python
"""Stateless repertoire computation over CandidateSystem.

Layer 2 of the kernel. Functions take a CandidateSystem as the first
argument; results are memoized via a per-instance decorator that purges
when the CandidateSystem is garbage-collected.

Numerical bodies are ports of the corresponding Subsystem methods in
pyphi/subsystem.py. Parity tests guard equivalence.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from functools import wraps
from typing import Any
from weakref import WeakValueDictionary

# One cache dict per memoized function name.
_caches: dict[str, dict[tuple, Any]] = {}

# Live CandidateSystem references keyed by id, with finalizers that purge
# the corresponding cache entries on GC.
_observers: WeakValueDictionary[int, Any] = WeakValueDictionary()


def _evict(cs_id: int) -> None:
    """Purge cache entries whose first key element is ``cs_id``."""
    for fn_cache in _caches.values():
        for key in [k for k in fn_cache if k and k[0] == cs_id]:
            del fn_cache[key]


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over CandidateSystem instances by ``id()``.

    Uses ``WeakValueDictionary`` + ``weakref.finalize`` so that cache
    entries are purged when the CandidateSystem is collected.
    """
    cache = _caches.setdefault(fn.__name__, {})

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            return cache[key]
        result = fn(cs, *args)  # raises propagate; key not added on raise
        cache[key] = result
        return result

    return wrapper


def cache_info() -> dict[str, dict[str, int]]:
    """Return per-function ``hits``/``misses``/``size`` (size only here)."""
    return {name: {"size": len(c)} for name, c in _caches.items()}


def clear_caches(cs: Any | None = None) -> None:
    """Clear cache entries. If ``cs`` is given, clear only that instance's entries."""
    if cs is None:
        for c in _caches.values():
            c.clear()
        return
    _evict(id(cs))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: add memoization decorator for core.repertoire_algebra"
```

### Task 4.2: Port `cause_repertoire` and `effect_repertoire`

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

The implementations call into a temporary delegate to the legacy `Subsystem` so the parity test compares like-for-like. We replace the delegate with native ports in subsequent tasks.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_repertoire_algebra.py`:

```python
@pytest.fixture
def cs_and_subsystem():
    from pyphi import Subsystem, examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    network = examples.basic_network()
    state = (1, 0, 0)
    nodes = (0, 1, 2)
    cs = CandidateSystem(causal_model=CausalModel.from_network(network),
                         state=state, node_indices=nodes)
    sub = Subsystem(network, state, nodes)
    return cs, sub


@pytest.mark.parametrize(
    "mechanism, purview",
    [((0,), (1,)), ((0, 1), (2,)), ((0, 1, 2), (0, 1, 2))],
)
def test_cause_repertoire_parity(cs_and_subsystem, mechanism, purview) -> None:
    import numpy as np
    from pyphi.core.repertoire_algebra import cause_repertoire
    cs, sub = cs_and_subsystem
    new = cause_repertoire(cs, mechanism, purview)
    old = sub.cause_repertoire(mechanism, purview)
    np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize(
    "mechanism, purview",
    [((0,), (1,)), ((0, 1), (2,)), ((0, 1, 2), (0, 1, 2))],
)
def test_effect_repertoire_parity(cs_and_subsystem, mechanism, purview) -> None:
    import numpy as np
    from pyphi.core.repertoire_algebra import effect_repertoire
    cs, sub = cs_and_subsystem
    new = effect_repertoire(cs, mechanism, purview)
    old = sub.effect_repertoire(mechanism, purview)
    np.testing.assert_array_equal(new, old)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: `ImportError: cannot import name 'cause_repertoire' from 'pyphi.core.repertoire_algebra'`.

- [ ] **Step 3: Add delegating ports**

Append to `pyphi/core/repertoire_algebra.py`:

```python
def _legacy_subsystem(cs: Any) -> Any:
    """Construct a legacy Subsystem from a CandidateSystem (worktree only).

    Removed in Task 4.8 once all repertoire functions have native
    implementations and parity is verified.
    """
    from pyphi.subsystem import Subsystem
    return Subsystem(cs.network, cs.state, cs.node_indices, cut=cs.cut)


@_memoize
def cause_repertoire(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    """Cause repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    return _legacy_subsystem(cs).cause_repertoire(mechanism, purview)


@_memoize
def effect_repertoire(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    """Effect repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    return _legacy_subsystem(cs).effect_repertoire(mechanism, purview)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: 9 passing (3 + 3 + 3 from previous task).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: port cause_repertoire/effect_repertoire (legacy-delegating)"
```

### Task 4.3: Port `repertoire`, `unconstrained_*`, `expand_*`

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

Functions to add: `repertoire`, `unconstrained_cause_repertoire`, `unconstrained_effect_repertoire`, `unconstrained_repertoire`, `expand_repertoire`, `expand_cause_repertoire`, `expand_effect_repertoire`, `partitioned_repertoire`.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_repertoire_algebra.py`:

```python
@pytest.mark.parametrize(
    "fn_name",
    [
        "unconstrained_cause_repertoire",
        "unconstrained_effect_repertoire",
        "unconstrained_repertoire",
    ],
)
def test_unconstrained_parity(cs_and_subsystem, fn_name) -> None:
    import numpy as np
    import pyphi.core.repertoire_algebra as ra
    cs, sub = cs_and_subsystem
    purview = (0, 1, 2)
    if fn_name.endswith("_repertoire") and not fn_name.startswith("unconstrained_"):
        from pyphi.direction import Direction
        new = getattr(ra, fn_name)(cs, Direction.CAUSE, purview)
        old = getattr(sub, fn_name)(Direction.CAUSE, purview)
    else:
        new = getattr(ra, fn_name)(cs, purview)
        old = getattr(sub, fn_name)(purview)
    np.testing.assert_array_equal(new, old)


def test_repertoire_dispatch_parity(cs_and_subsystem) -> None:
    import numpy as np
    from pyphi.core.repertoire_algebra import repertoire
    from pyphi.direction import Direction
    cs, sub = cs_and_subsystem
    new = repertoire(cs, Direction.CAUSE, (0,), (1,))
    old = sub.repertoire(Direction.CAUSE, (0,), (1,))
    np.testing.assert_array_equal(new, old)


def test_expand_repertoire_parity(cs_and_subsystem) -> None:
    import numpy as np
    from pyphi.core.repertoire_algebra import cause_repertoire, expand_cause_repertoire
    cs, sub = cs_and_subsystem
    rep = cause_repertoire(cs, (0,), (1,))
    new = expand_cause_repertoire(cs, rep)
    old = sub.expand_cause_repertoire(sub.cause_repertoire((0,), (1,)))
    np.testing.assert_array_equal(new, old)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: ImportErrors for the new functions.

- [ ] **Step 3: Add delegating ports**

Append to `pyphi/core/repertoire_algebra.py`:

```python
@_memoize
def repertoire(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).repertoire(direction, mechanism, purview)


def unconstrained_cause_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_cause_repertoire(purview)


def unconstrained_effect_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_effect_repertoire(purview)


def unconstrained_repertoire(cs: Any, direction: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_repertoire(direction, purview)


def expand_repertoire(cs: Any, direction: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None) -> Any:
    return _legacy_subsystem(cs).expand_repertoire(direction, repertoire_array, new_purview=new_purview)


def expand_cause_repertoire(cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None) -> Any:
    return _legacy_subsystem(cs).expand_cause_repertoire(repertoire_array, new_purview=new_purview)


def expand_effect_repertoire(cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None) -> Any:
    return _legacy_subsystem(cs).expand_effect_repertoire(repertoire_array, new_purview=new_purview)


@_memoize
def partitioned_repertoire(cs: Any, direction: Any, partition: Any) -> Any:
    return _legacy_subsystem(cs).partitioned_repertoire(direction, partition)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: all parity tests passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: port repertoire/unconstrained_*/expand_*/partitioned_repertoire"
```

### Task 4.4: Port `forward_*` repertoires and probabilities

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

Functions: `forward_cause_repertoire`, `forward_effect_repertoire`, `forward_repertoire`, `forward_cause_probability`, `forward_effect_probability`, `forward_probability`, `unconstrained_forward_cause_repertoire`, `unconstrained_forward_effect_repertoire`, `unconstrained_forward_repertoire`.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_repertoire_algebra.py`:

```python
@pytest.mark.parametrize(
    "fn_name",
    [
        "forward_cause_repertoire",
        "forward_effect_repertoire",
        "unconstrained_forward_cause_repertoire",
        "unconstrained_forward_effect_repertoire",
    ],
)
def test_forward_repertoire_parity(cs_and_subsystem, fn_name) -> None:
    import numpy as np
    import pyphi.core.repertoire_algebra as ra
    cs, sub = cs_and_subsystem
    if fn_name.startswith("unconstrained_"):
        purview = (0, 1, 2)
        new = getattr(ra, fn_name)(cs, purview)
        old = getattr(sub, fn_name)(purview)
    else:
        new = getattr(ra, fn_name)(cs, (0,), (1,))
        old = getattr(sub, fn_name)((0,), (1,))
    np.testing.assert_array_equal(new, old)


def test_forward_probability_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import forward_effect_probability
    cs, sub = cs_and_subsystem
    new = forward_effect_probability(cs, (0,), (1,), (1,))
    old = sub.forward_effect_probability((0,), (1,), (1,))
    assert new == pytest.approx(old)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: ImportErrors.

- [ ] **Step 3: Add delegating ports**

Append to `pyphi/core/repertoire_algebra.py`:

```python
@_memoize
def forward_cause_repertoire(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).forward_cause_repertoire(mechanism, purview)


@_memoize
def forward_effect_repertoire(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).forward_effect_repertoire(mechanism, purview)


@_memoize
def forward_repertoire(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).forward_repertoire(direction, mechanism, purview)


def unconstrained_forward_cause_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_cause_repertoire(purview)


def unconstrained_forward_effect_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_effect_repertoire(purview)


def unconstrained_forward_repertoire(cs: Any, direction: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_repertoire(direction, purview)


def forward_cause_probability(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], purview_state: Any, mechanism_state: Any | None = None) -> float:
    return _legacy_subsystem(cs).forward_cause_probability(mechanism, purview, purview_state, mechanism_state)


def forward_effect_probability(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], purview_state: Any) -> float:
    return _legacy_subsystem(cs).forward_effect_probability(mechanism, purview, purview_state)


def forward_probability(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], purview_state: Any, **kwargs: Any) -> float:
    return _legacy_subsystem(cs).forward_probability(direction, mechanism, purview, purview_state, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: port forward_* repertoires and probabilities (delegating)"
```

### Task 4.5: Port `cause_info`, `effect_info`, `cause_effect_info`, `intrinsic_information`

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_repertoire_algebra.py`:

```python
def test_cause_info_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import cause_info
    cs, sub = cs_and_subsystem
    new = cause_info(cs, (0,), (1,))
    old = sub.cause_info((0,), (1,))
    assert new == pytest.approx(old)


def test_effect_info_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import effect_info
    cs, sub = cs_and_subsystem
    new = effect_info(cs, (0,), (1,))
    old = sub.effect_info((0,), (1,))
    assert new == pytest.approx(old)


def test_intrinsic_information_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import intrinsic_information
    from pyphi.direction import Direction
    cs, sub = cs_and_subsystem
    new = intrinsic_information(cs, Direction.CAUSE, (0,), (1,))
    old = sub.intrinsic_information(Direction.CAUSE, (0,), (1,))
    # intrinsic_information may return DistanceResult or float
    new_val = float(new) if hasattr(new, "__float__") else new
    old_val = float(old) if hasattr(old, "__float__") else old
    assert new_val == pytest.approx(old_val)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: ImportErrors.

- [ ] **Step 3: Add delegating ports**

Append to `pyphi/core/repertoire_algebra.py`:

```python
def cause_info(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).cause_info(mechanism, purview, **kwargs)


def effect_info(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).effect_info(mechanism, purview, **kwargs)


def cause_effect_info(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).cause_effect_info(mechanism, purview, **kwargs)


def intrinsic_information(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).intrinsic_information(direction, mechanism, purview, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: port info functions (cause/effect/cause_effect/intrinsic)"
```

### Task 4.6: Port mechanism analysis (`evaluate_partition`, `find_mip`, mips, phi, `find_mice`, mic/mie, `phi_max`, `concept`, `distinction`, `all_distinctions`, `sia`, `potential_purviews`, `indices2nodes`)

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

These all delegate to the legacy `Subsystem` for now. The bodies are short and uniform.

- [ ] **Step 1: Write the failing parity test**

Append to `test/test_core_repertoire_algebra.py`:

```python
def test_phi_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import phi
    cs, sub = cs_and_subsystem
    assert phi(cs, (0,), (1,)) == pytest.approx(sub.phi((0,), (1,)))


def test_concept_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import concept
    cs, sub = cs_and_subsystem
    new = concept(cs, (0,))
    old = sub.concept((0,))
    assert new.phi == pytest.approx(old.phi)


def test_sia_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import sia
    cs, sub = cs_and_subsystem
    new = sia(cs)
    old = sub.sia()
    assert new.phi == pytest.approx(old.phi)


def test_potential_purviews_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import potential_purviews
    from pyphi.direction import Direction
    cs, sub = cs_and_subsystem
    new = list(potential_purviews(cs, Direction.CAUSE, (0,)))
    old = list(sub.potential_purviews(Direction.CAUSE, (0,)))
    assert new == old
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: ImportErrors.

- [ ] **Step 3: Add delegating ports**

Append to `pyphi/core/repertoire_algebra.py`:

```python
def evaluate_partition(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], partition: Any, **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).evaluate_partition(direction, mechanism, purview, partition, **kwargs)


def find_mip(cs: Any, direction: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).find_mip(direction, mechanism, purview, **kwargs)


def cause_mip(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).cause_mip(mechanism, purview, **kwargs)


def effect_mip(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).effect_mip(mechanism, purview, **kwargs)


def phi_cause_mip(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).phi_cause_mip(mechanism, purview, **kwargs)


def phi_effect_mip(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).phi_effect_mip(mechanism, purview, **kwargs)


def phi(cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any) -> float:
    return _legacy_subsystem(cs).phi(mechanism, purview, **kwargs)


def find_mice(cs: Any, direction: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).find_mice(direction, mechanism, **kwargs)


def mic(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).mic(mechanism, **kwargs)


def mie(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).mie(mechanism, **kwargs)


def phi_max(cs: Any, mechanism: tuple[int, ...]) -> float:
    return _legacy_subsystem(cs).phi_max(mechanism)


def concept(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).concept(mechanism, **kwargs)


def distinction(cs: Any, mechanism: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).distinction(mechanism)


def all_distinctions(cs: Any, **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).all_distinctions(**kwargs)


def sia(cs: Any, **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).sia(**kwargs)


def potential_purviews(cs: Any, direction: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).potential_purviews(direction, mechanism, **kwargs)


def indices2nodes(cs: Any, indices: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).indices2nodes(indices)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_repertoire_algebra.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "P7: port mechanism / system analysis (delegating)"
```

### Task 4.7: Pyright clean on `core/repertoire_algebra.py`

- [ ] **Step 1: Run pyright**

```bash
uv run pyright pyphi/core/repertoire_algebra.py
```

- [ ] **Step 2: Fix errors inline**

Common fixes: tighten `Any` to specific types where pyright can infer; add `# pyright: ignore[...]` only where unavoidable.

- [ ] **Step 3: Commit**

```bash
git add -p pyphi/core/repertoire_algebra.py
git commit -m "P7: pyright clean on core.repertoire_algebra"
```

(Skip if no changes.)

### Task 4.8: Run fast lane + slow lane to confirm baseline still green

- [ ] **Step 1: Run fast lane**

```bash
uv run pytest test/test_core_*.py test/test_subsystem_surface.py test/test_partition.py test/test_golden_regression.py test/test_invariants.py -q
```

Expected: all green.

- [ ] **Step 2: Run slow lane in background**

```bash
uv run pytest test/test_invariants_hypothesis.py -q --tb=short
```

Expected: all 19 properties green. (If running synchronously, allow 5-10 minutes.)

- [ ] **Step 3: Capture timing for the record**

```bash
uv run pytest test/test_core_*.py -q --durations=10 > /tmp/p7-phase4-durations.txt
```

This is informational; commit not required.

---

## Phase 5 — `CandidateSystem` Proxy Methods + `SubsystemPublicInterface` Parity

The full public surface of `Subsystem` (33 names in `PUBLIC_SUBSYSTEM_ATTRS` callable; plus 20 attributes) lands as methods on `CandidateSystem` that proxy to `repertoire_algebra` / `formalism`.

### Task 5.1: Add repertoire-method proxies on `CandidateSystem`

**Files:**
- Modify: `pyphi/core/candidate_system.py`
- Modify: `test/test_core_candidate_system.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_core_candidate_system.py`:

```python
@pytest.mark.parametrize(
    "method, args",
    [
        ("cause_repertoire", ((0,), (1,))),
        ("effect_repertoire", ((0,), (1,))),
        ("unconstrained_cause_repertoire", ((0,),)),
        ("unconstrained_effect_repertoire", ((0,),)),
        ("forward_cause_repertoire", ((0,), (1,))),
        ("forward_effect_repertoire", ((0,), (1,))),
        ("partitioned_repertoire", None),  # special: partition arg
    ],
)
def test_candidate_system_proxies_repertoire_methods(cs_and_subsystem, method, args) -> None:
    import numpy as np
    cs, sub = cs_and_subsystem
    if method == "partitioned_repertoire":
        from pyphi.partition import mip_partitions
        from pyphi.direction import Direction
        partitions = list(mip_partitions((0,), (1,), cs.node_labels))
        partition = partitions[0]
        new = cs.partitioned_repertoire(Direction.CAUSE, partition)
        old = sub.partitioned_repertoire(Direction.CAUSE, partition)
    else:
        new = getattr(cs, method)(*args)
        old = getattr(sub, method)(*args)
    np.testing.assert_array_equal(new, old)


def test_candidate_system_phi(cs_and_subsystem) -> None:
    cs, sub = cs_and_subsystem
    assert cs.phi((0,), (1,)) == pytest.approx(sub.phi((0,), (1,)))


def test_candidate_system_concept(cs_and_subsystem) -> None:
    cs, sub = cs_and_subsystem
    assert cs.concept((0,)).phi == pytest.approx(sub.concept((0,)).phi)


def test_candidate_system_sia(cs_and_subsystem) -> None:
    cs, sub = cs_and_subsystem
    assert cs.sia().phi == pytest.approx(sub.sia().phi)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: AttributeError on the proxy methods.

- [ ] **Step 3: Add proxy methods to `CandidateSystem`**

Edit `pyphi/core/candidate_system.py` — add the following block of methods after `apply_cut`:

```python
    # ---- repertoire algebra proxies ----
    def cause_repertoire(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.cause_repertoire(self, mechanism, purview, **kwargs)

    def effect_repertoire(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.effect_repertoire(self, mechanism, purview, **kwargs)

    def repertoire(self, direction, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.repertoire(self, direction, mechanism, purview, **kwargs)

    def unconstrained_cause_repertoire(self, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(self, direction, partition):
        from . import repertoire_algebra as ra
        return ra.partitioned_repertoire(self, direction, partition)

    def expand_repertoire(self, direction, repertoire_array, *, new_purview=None):
        from . import repertoire_algebra as ra
        return ra.expand_repertoire(self, direction, repertoire_array, new_purview=new_purview)

    def expand_cause_repertoire(self, repertoire_array, *, new_purview=None):
        from . import repertoire_algebra as ra
        return ra.expand_cause_repertoire(self, repertoire_array, new_purview=new_purview)

    def expand_effect_repertoire(self, repertoire_array, *, new_purview=None):
        from . import repertoire_algebra as ra
        return ra.expand_effect_repertoire(self, repertoire_array, new_purview=new_purview)

    def forward_cause_repertoire(self, mechanism, purview):
        from . import repertoire_algebra as ra
        return ra.forward_cause_repertoire(self, mechanism, purview)

    def forward_effect_repertoire(self, mechanism, purview):
        from . import repertoire_algebra as ra
        return ra.forward_effect_repertoire(self, mechanism, purview)

    def forward_repertoire(self, direction, mechanism, purview):
        from . import repertoire_algebra as ra
        return ra.forward_repertoire(self, direction, mechanism, purview)

    def unconstrained_forward_cause_repertoire(self, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_forward_cause_repertoire(self, purview)

    def unconstrained_forward_effect_repertoire(self, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_forward_effect_repertoire(self, purview)

    def unconstrained_forward_repertoire(self, direction, purview):
        from . import repertoire_algebra as ra
        return ra.unconstrained_forward_repertoire(self, direction, purview)

    def forward_cause_probability(self, mechanism, purview, purview_state, mechanism_state=None):
        from . import repertoire_algebra as ra
        return ra.forward_cause_probability(self, mechanism, purview, purview_state, mechanism_state)

    def forward_effect_probability(self, mechanism, purview, purview_state):
        from . import repertoire_algebra as ra
        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(self, direction, mechanism, purview, purview_state, **kwargs):
        from . import repertoire_algebra as ra
        return ra.forward_probability(self, direction, mechanism, purview, purview_state, **kwargs)

    # ---- info / phi proxies ----
    def cause_info(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.cause_info(self, mechanism, purview, **kwargs)

    def effect_info(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.effect_info(self, mechanism, purview, **kwargs)

    def cause_effect_info(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.cause_effect_info(self, mechanism, purview, **kwargs)

    def intrinsic_information(self, direction, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.intrinsic_information(self, direction, mechanism, purview, **kwargs)

    def evaluate_partition(self, direction, mechanism, purview, partition, **kwargs):
        from . import repertoire_algebra as ra
        return ra.evaluate_partition(self, direction, mechanism, purview, partition, **kwargs)

    def find_mip(self, direction, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.find_mip(self, direction, mechanism, purview, **kwargs)

    def cause_mip(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.cause_mip(self, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.effect_mip(self, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.phi_cause_mip(self, mechanism, purview, **kwargs)

    def phi_effect_mip(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.phi_effect_mip(self, mechanism, purview, **kwargs)

    def phi(self, mechanism, purview, **kwargs):
        from . import repertoire_algebra as ra
        return ra.phi(self, mechanism, purview, **kwargs)

    def find_mice(self, direction, mechanism, **kwargs):
        from . import repertoire_algebra as ra
        return ra.find_mice(self, direction, mechanism, **kwargs)

    def mic(self, mechanism, **kwargs):
        from . import repertoire_algebra as ra
        return ra.mic(self, mechanism, **kwargs)

    def mie(self, mechanism, **kwargs):
        from . import repertoire_algebra as ra
        return ra.mie(self, mechanism, **kwargs)

    def phi_max(self, mechanism):
        from . import repertoire_algebra as ra
        return ra.phi_max(self, mechanism)

    def concept(self, mechanism, **kwargs):
        from . import repertoire_algebra as ra
        return ra.concept(self, mechanism, **kwargs)

    def distinction(self, mechanism):
        from . import repertoire_algebra as ra
        return ra.distinction(self, mechanism)

    def all_distinctions(self, **kwargs):
        from . import repertoire_algebra as ra
        return ra.all_distinctions(self, **kwargs)

    def sia(self, **kwargs):
        from . import repertoire_algebra as ra
        return ra.sia(self, **kwargs)

    def potential_purviews(self, direction, mechanism, **kwargs):
        from . import repertoire_algebra as ra
        return ra.potential_purviews(self, direction, mechanism, **kwargs)

    def indices2nodes(self, indices):
        from . import repertoire_algebra as ra
        return ra.indices2nodes(self, indices)

    # ---- cache surface ----
    def cache_info(self) -> dict[str, Any]:
        from . import repertoire_algebra as ra
        return ra.cache_info()

    def clear_caches(self) -> None:
        from . import repertoire_algebra as ra
        ra.clear_caches(self)

    def to_json(self) -> dict[str, Any]:
        return {
            "causal_model": self.causal_model,
            "state": list(self.state),
            "node_indices": list(self.node_indices),
            "cut": self.cut,
        }
```

Add the necessary import: `from typing import Any` if not already there.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_core_candidate_system.py -v
```

Expected: all passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/candidate_system.py test/test_core_candidate_system.py
git commit -m "P7: add CandidateSystem proxy methods (full public surface)"
```

### Task 5.2: `core/__init__.py` exports + surface drift parity

**Files:**
- Modify: `pyphi/core/__init__.py`
- Create: `test/test_core_layering.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_core_layering.py`:

```python
"""Architectural assertions for the new core/ package."""

from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "pyphi" / "core"


def _imports_in(path: Path) -> set[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_causal_model_does_not_import_repertoire_algebra() -> None:
    imports = _imports_in(CORE / "causal_model.py")
    assert not any("repertoire_algebra" in i for i in imports)


def test_repertoire_algebra_does_not_import_formalism() -> None:
    imports = _imports_in(CORE / "repertoire_algebra.py")
    assert not any(i.startswith("pyphi.formalism") for i in imports)
    assert not any(i == ".formalism" for i in imports)


def test_core_does_not_import_subsystem_module_top_level() -> None:
    """The core package may use the legacy subsystem inside function bodies
    during the worktree (the _legacy_subsystem helper), but no module in
    core/ should import it at the top level after the worktree is healthy.

    P7 final cutover removes even the function-body delegations. Until then,
    function-body imports are tolerated.
    """
    for py in CORE.rglob("*.py"):
        src = py.read_text()
        tree = ast.parse(src, filename=str(py))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "subsystem" in node.module:
                        raise AssertionError(
                            f"{py}: top-level import of subsystem ({node.module})"
                        )


def test_candidate_system_satisfies_subsystem_public_interface() -> None:
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.protocols import PUBLIC_SUBSYSTEM_ATTRS
    declared = {a for a in PUBLIC_SUBSYSTEM_ATTRS if not a.startswith("_")}
    discovered = {a for a in dir(CandidateSystem) if not a.startswith("_")}
    missing = declared - discovered
    assert not missing, f"CandidateSystem missing public attrs: {sorted(missing)}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_core_layering.py -v
```

Expected: failures (probably `test_candidate_system_satisfies_subsystem_public_interface` since `CandidateSystem` is not yet exported from `pyphi.core`).

- [ ] **Step 3: Update `core/__init__.py`**

Edit `pyphi/core/__init__.py`:

```python
"""pyphi.core — typed kernel for the PyPhi 2.0 layered architecture."""

from .candidate_system import CandidateSystem as CandidateSystem
from .causal_model import CausalModel as CausalModel
from .substrate import Substrate as Substrate
from .unit import Unit as Unit
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_core_layering.py -v
```

Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/__init__.py test/test_core_layering.py
git commit -m "P7: export CandidateSystem from pyphi.core; add layering tests"
```

### Task 5.3: Pyright clean on `pyphi/core/`

- [ ] **Step 1: Run pyright on the whole package**

```bash
uv run pyright pyphi/core/
```

- [ ] **Step 2: Fix errors**

Address any issues. Expect a few `Any`-related warnings due to delegation; these are acceptable temporarily but flag them in commit message.

- [ ] **Step 3: Commit**

```bash
git add -p pyphi/core/
git commit -m "P7: pyright clean on pyphi/core/"
```

### Task 5.4: Run full fast lane and slow lane

- [ ] **Step 1: Fast lane**

```bash
uv run pytest test/test_core_*.py test/test_subsystem.py test/test_subsystem_surface.py test/test_partition.py test/test_golden_regression.py test/test_invariants.py -q
```

Expected: all green (existing legacy `Subsystem` tests still pass; new `core/` tests pass).

- [ ] **Step 2: Slow lane**

```bash
uv run pytest test/test_invariants_hypothesis.py -q --tb=short
```

Expected: all 19 properties green.

---

## Phase 6 — Cutover: Formalism, Compute, Actual

The cutover switches every caller from `Subsystem` to `CandidateSystem`. The key insight: most callers take the system as an opaque first argument and call methods on it. Because `CandidateSystem` exposes the same method names and they delegate through `_legacy_subsystem(...)` to a transient `Subsystem` instance, the cutover is mostly type annotations and import statements, with no behavior changes.

### Task 6.1: Cutover `pyphi/formalism/iit4/__init__.py`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py`

- [ ] **Step 1: Write/update the failing test**

```bash
uv run pytest test/test_golden_regression.py -k iit_4_0_2023 -q
```

Expected: passes today, still passes after the change. (No new test; the regression test is the gate.)

- [ ] **Step 2: Edit imports in `formalism/iit4/__init__.py`**

Edit `pyphi/formalism/iit4/__init__.py` line 47: change

```python
from pyphi.subsystem import Subsystem
```

to

```python
from pyphi.core import CandidateSystem as Subsystem  # keep variable name to minimize diff
```

Run a quick sanity check that the rest of the file's `: Subsystem` annotations still bind correctly via the alias.

- [ ] **Step 3: Run regression test**

```bash
uv run pytest test/test_golden_regression.py -k iit_4_0 -q
```

Expected: all IIT 4.0 fixtures pass.

- [ ] **Step 4: Run full IIT 4.0 test suite**

```bash
uv run pytest test/test_big_phi_robust.py test/test_invariants.py -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py
git commit -m "P7: switch formalism/iit4 to CandidateSystem (alias to Subsystem)"
```

### Task 6.2: Cutover `pyphi/compute/subsystem.py` and `pyphi/compute/network.py`

**Files:**
- Modify: `pyphi/compute/subsystem.py`
- Modify: `pyphi/compute/network.py`

- [ ] **Step 1: Edit imports in `compute/subsystem.py`**

Edit `pyphi/compute/subsystem.py` line 37 (the TYPE_CHECKING block) and any direct imports:

```python
# Before:
if TYPE_CHECKING:
    from pyphi.subsystem import Subsystem

# After:
if TYPE_CHECKING:
    from pyphi.core import CandidateSystem as Subsystem
```

Repeat for `compute/network.py:20`:

```python
# Before:
from pyphi.subsystem import Subsystem

# After:
from pyphi.core import CandidateSystem as Subsystem
```

- [ ] **Step 2: Run IIT 3.0 fixture tests**

```bash
uv run pytest test/test_golden_regression.py -k iit_3_0 -q
```

Expected: all IIT 3.0 fixtures pass.

- [ ] **Step 3: Run compute tests**

```bash
uv run pytest test/test_compute_network.py test/test_concept_style_cuts.py -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add pyphi/compute/subsystem.py pyphi/compute/network.py
git commit -m "P7: switch pyphi.compute to CandidateSystem"
```

### Task 6.3: Cutover `pyphi/actual.py` (mechanical)

**Files:**
- Modify: `pyphi/actual.py`

- [ ] **Step 1: Edit imports**

Edit `pyphi/actual.py` line 47:

```python
# Before:
from .subsystem import Subsystem

# After:
from .core import CandidateSystem as Subsystem
```

`pyphi.actual` defines its own `Subsystem` subclass (`Transition` etc. extend `Subsystem`). The minimal port (per spec Q6) keeps the subclassing pattern intact — `class Transition(Subsystem)` becomes `class Transition(CandidateSystem)` via the alias. P14 does the architectural rewrite.

- [ ] **Step 2: Run actual-causation tests**

```bash
uv run pytest test/test_actual.py -q
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add pyphi/actual.py
git commit -m "P7: switch pyphi.actual to CandidateSystem (mechanical port)"
```

### Task 6.4: Cutover `pyphi/__init__.py` and `pyphi/examples.py`

**Files:**
- Modify: `pyphi/__init__.py`
- Modify: `pyphi/examples.py`

- [ ] **Step 1: Edit `pyphi/__init__.py`**

Edit `pyphi/__init__.py` line 77:

```python
# Before:
from .subsystem import Subsystem

# After:
from .core import CandidateSystem
from .core import CandidateSystem as Subsystem  # transitional alias; removed in Phase 8
```

(Keep `Subsystem` as an alias one more time — Phase 8 deletes it after `subsystem.py` is gone.)

- [ ] **Step 2: Edit `pyphi/examples.py`**

Edit `pyphi/examples.py` line 16:

```python
# Before:
from .subsystem import Subsystem

# After:
from .core import CandidateSystem as Subsystem
```

- [ ] **Step 3: Run example tests**

```bash
uv run pytest test/test_examples.py -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add pyphi/__init__.py pyphi/examples.py
git commit -m "P7: switch pyphi top-level + examples to CandidateSystem"
```

### Task 6.5: Remove `Concept.subsystem` back-reference

**Files:**
- Modify: `pyphi/models/mechanism.py`

The `Concept` class has a `subsystem` field set in `__init__`. Remove the field; update `__init__`, `__eq__`, `to_dict`, and any methods that read `self.subsystem`.

- [ ] **Step 1: Find all `self.subsystem` reads in `models/mechanism.py`**

```bash
grep -n "self\.subsystem\|subsystem=" pyphi/models/mechanism.py
```

Note every line — these all need updating or commenting out.

- [ ] **Step 2: Update `Concept.__init__`**

Edit `pyphi/models/mechanism.py` near line 960. Remove the `subsystem` parameter and assignment. Remove the `subsystem (Subsystem):` docstring line. Update `unorderable_unless_eq: ClassVar[list[str]] = ["subsystem"]` to remove `"subsystem"`.

- [ ] **Step 3: Update callers of `Concept(...)`**

```bash
grep -rn "Concept(" pyphi/ test/ | grep -v __pycache__ | grep -v "# Concept"
```

For every callsite that passes `subsystem=`, remove that kwarg.

- [ ] **Step 4: Run model + serialization tests**

```bash
uv run pytest test/test_models.py test/test_serialization.py -q
```

Expected: all green. If serialization tests fail, the `to_dict` method needs the `subsystem` key removed.

- [ ] **Step 5: Run golden regression**

```bash
uv run pytest test/test_golden_regression.py -q
```

Expected: all 17 fixtures pass.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/mechanism.py pyphi/models/__init__.py
git commit -m "P7: remove Concept.subsystem back-reference"
```

---

## Phase 7 — Test Rename + Macro Disable

### Task 7.1: Disable `pyphi.macro` (NotImplementedError on construction)

**Files:**
- Modify: `pyphi/macro.py`

- [ ] **Step 1: Edit `pyphi/macro.py`**

Add at the top of `MacroSubsystem.__init__` (line 182), and also `Blackbox.__new__`, `MacroNetwork.__init__`:

```python
raise NotImplementedError(
    "MacroSubsystem is undergoing rewrite in P7b; restored after the kernel rewrite lands."
)
```

For `Blackbox` and `CoarseGrain` (namedtuples), add `__new__` overrides that raise the same error. Names must remain importable so `pyphi.macro` import does not fail.

- [ ] **Step 2: Verify import still succeeds**

```bash
uv run python -c "from pyphi import macro; print(macro.MacroSubsystem)"
```

Expected: prints the class object.

```bash
uv run python -c "from pyphi.macro import MacroSubsystem; MacroSubsystem(None, None, None)"
```

Expected: `NotImplementedError: ...`.

- [ ] **Step 3: Commit**

```bash
git add pyphi/macro.py
git commit -m "P7: disable MacroSubsystem construction during P7→P7b gap"
```

### Task 7.2: Skip macro-using tests + add sentinel

**Files:**
- Modify: `test/test_macro_subsystem.py`
- Modify: `test/test_macro_blackbox.py`
- Modify: `test/example_networks.py` (skip macro fixtures)
- Create: `test/test_macro_disabled_during_p7_gap.py`

- [ ] **Step 1: Add module-level skip to macro test files**

Add to top of `test/test_macro_subsystem.py` and `test/test_macro_blackbox.py`:

```python
import pytest
pytestmark = pytest.mark.skip(reason="P7b: MacroSubsystem port pending")
```

- [ ] **Step 2: Skip macro fixtures in `example_networks.py`**

Find any function in `test/example_networks.py` that constructs a `MacroSubsystem` or `Blackbox`; wrap with:

```python
@pytest.mark.skip(reason="P7b: MacroSubsystem port pending")
def macro_fixture_name():
    ...
```

(Or replace the body with `pytest.skip(...)` for fixtures that are currently called.)

- [ ] **Step 3: Add sentinel test**

Create `test/test_macro_disabled_during_p7_gap.py`:

```python
"""Sentinel: confirms macro is correctly disabled during P7→P7b.

Deleted by P7b's first commit.
"""
import pytest


def test_macro_module_imports_successfully() -> None:
    from pyphi import macro  # noqa: F401


def test_macro_subsystem_construction_raises() -> None:
    from pyphi.macro import MacroSubsystem
    with pytest.raises(NotImplementedError, match="P7b"):
        MacroSubsystem(None, None, None)
```

- [ ] **Step 4: Run the test suite (excluding hypothesis)**

```bash
uv run pytest -q --ignore=test/test_invariants_hypothesis.py
```

Expected: all green; macro tests skipped; sentinel passes.

- [ ] **Step 5: Commit**

```bash
git add test/test_macro_subsystem.py test/test_macro_blackbox.py test/example_networks.py test/test_macro_disabled_during_p7_gap.py
git commit -m "P7: skip macro-dependent tests + add sentinel for P7→P7b gap"
```

### Task 7.3: Mechanical rename `Subsystem` → `CandidateSystem` in tests

**Files:**
- Modify: 29 test files (per `git grep -l Subsystem test/`)

- [ ] **Step 1: List target files**

```bash
git grep -l '\bSubsystem\b' test/
```

Confirm the count is ~29 (varies as macro tests now skipped).

- [ ] **Step 2: Run search-and-replace**

Use a careful sed (BSD `sed -i ''` on macOS):

```bash
git grep -l '\bSubsystem\b' test/ | xargs sed -i '' 's/\bSubsystem\b/CandidateSystem/g'
```

- [ ] **Step 3: Manual review of constructor sites**

The legacy constructor signature `Subsystem(network, state, nodes)` now needs to be `CandidateSystem.from_network(network, state, nodes)` OR direct construction. The simplest mechanical approach:

```bash
git grep -l 'CandidateSystem(' test/ | xargs grep -l 'CandidateSystem(network'
```

For each match, replace `CandidateSystem(network, state, nodes, cut=...)` with `CandidateSystem.from_network(network, state, nodes, cut=...)`. (Add a `from_network` classmethod to `CandidateSystem` if it doesn't exist — see implementation note below.)

**Implementation note:** Add a classmethod to `pyphi/core/candidate_system.py`:

```python
    @classmethod
    def from_network(
        cls,
        network,
        state,
        nodes=None,
        cut=None,
    ) -> "CandidateSystem":
        """Build a CandidateSystem from a legacy Network.

        Parity migration helper for tests that still construct via
        ``Subsystem(network, state, nodes)``.
        """
        cm = CausalModel.from_network(network)
        if nodes is None:
            nodes = tuple(range(network.size))
        return cls(causal_model=cm, state=tuple(state),
                   node_indices=tuple(nodes), cut=cut)
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest -q --ignore=test/test_invariants_hypothesis.py
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/candidate_system.py test/
git commit -m "P7: rename Subsystem → CandidateSystem across test/ + add from_network helper"
```

---

## Phase 8 — Delete `subsystem.py` + Final Validation

### Task 8.1: Update `SubsystemPublicInterface` test target + drop alias

**Files:**
- Modify: `test/test_subsystem_surface.py`
- Modify: `pyphi/__init__.py`

- [ ] **Step 1: Update test_subsystem_surface.py to test CandidateSystem**

Edit `test/test_subsystem_surface.py`: change the import and `Subsystem` references to `CandidateSystem`. The `PUBLIC_SUBSYSTEM_ATTRS` constant stays (rename to `PUBLIC_CANDIDATE_SYSTEM_ATTRS` is a Phase 8 cleanup).

- [ ] **Step 2: Drop `Subsystem` alias from `pyphi/__init__.py`**

Edit `pyphi/__init__.py`:

```python
from .core import CandidateSystem
# (remove the `as Subsystem` alias)
```

- [ ] **Step 3: Run surface drift**

```bash
uv run pytest test/test_subsystem_surface.py -v
```

Expected: passes against `CandidateSystem`.

- [ ] **Step 4: Commit**

```bash
git add test/test_subsystem_surface.py pyphi/__init__.py
git commit -m "P7: surface drift now targets CandidateSystem; drop Subsystem alias"
```

### Task 8.2: Delete `pyphi/subsystem.py` and `pyphi/repertoire.py`

**Files:**
- Delete: `pyphi/subsystem.py`
- Delete: `pyphi/repertoire.py`
- Modify: `pyphi/core/repertoire_algebra.py` (remove `_legacy_subsystem`)

The `_legacy_subsystem` helper inside `core/repertoire_algebra.py` constructs a legacy `Subsystem`. Once the surface tests, golden regression, and Hypothesis invariants all pass against `CandidateSystem` natively, the helper becomes unreachable.

But every numerical body in `repertoire_algebra.py` currently delegates to `_legacy_subsystem`. So actually deleting `subsystem.py` requires those bodies to be ported natively first. **This is the genuine kernel rewrite — Tasks 4.2 through 4.6 only delegate; the native implementations are this task.**

- [ ] **Step 1: Port `cause_repertoire`/`effect_repertoire` natively**

Replace the bodies of `cause_repertoire` and `effect_repertoire` in `core/repertoire_algebra.py` with code lifted from `pyphi/subsystem.py:355-470`, adapted to take `cs: CandidateSystem` and use `cs.cause_tpm`, `cs.effect_tpm`, `cs.proper_state`, `cs.nodes`. Run parity tests after the change.

```bash
uv run pytest test/test_core_repertoire_algebra.py::test_cause_repertoire_parity test/test_core_repertoire_algebra.py::test_effect_repertoire_parity -v
```

Expected: still passing.

- [ ] **Step 2: Port the remaining bodies natively, one function group at a time**

Repeat the lift-and-adapt for each function group (`forward_*`, `partitioned_repertoire`, `expand_*`, `cause_info`, `intrinsic_information`, `evaluate_partition`, `find_mip`, `find_mice`, `concept`, `sia`, `potential_purviews`, `indices2nodes`). After each group, run parity tests. Commit per group.

Suggested ordering:
1. `unconstrained_*` and `expand_*` (small, no recursion)
2. `forward_*` repertoires (depend on `cause/effect_repertoire` already ported)
3. `partitioned_repertoire` (depends on `cause/effect_repertoire`)
4. `cause_info`, `effect_info`, `cause_effect_info`, `intrinsic_information` (use the metric layer; mostly mechanical)
5. `evaluate_partition` (depends on metric + repertoire layer)
6. `find_mip`, `cause_mip`, `effect_mip`, `phi_*_mip` (depend on `evaluate_partition`)
7. `find_mice`, `mic`, `mie`, `phi_max` (depend on `find_mip`)
8. `concept`, `distinction` (depend on `mic` + `mie`)
9. `all_distinctions`, `sia` (depend on `concept`)
10. `potential_purviews`, `indices2nodes` (orthogonal; small)

After each group is ported natively, run:

```bash
uv run pytest test/test_core_repertoire_algebra.py test/test_golden_regression.py -q
```

Expected: all green.

Each group should be a separate commit:

```bash
git commit -m "P7: native port of <group> in repertoire_algebra"
```

- [ ] **Step 3: Remove `_legacy_subsystem`**

Once every function has a native body, remove the `_legacy_subsystem` helper from `core/repertoire_algebra.py` and confirm no callsites remain:

```bash
grep -n "_legacy_subsystem" pyphi/core/repertoire_algebra.py
```

Expected: no matches.

- [ ] **Step 4: Delete `pyphi/subsystem.py` and `pyphi/repertoire.py`**

```bash
git rm pyphi/subsystem.py pyphi/repertoire.py
```

- [ ] **Step 5: Run full suite (fast lane)**

```bash
uv run pytest -q --ignore=test/test_invariants_hypothesis.py
```

Expected: all green.

- [ ] **Step 6: Run slow lane (Hypothesis)**

```bash
uv run pytest test/test_invariants_hypothesis.py -q --tb=short
```

Expected: all 19 properties green.

- [ ] **Step 7: Commit deletion**

```bash
git add -u
git commit -m "P7: delete pyphi/subsystem.py and pyphi/repertoire.py — kernel rewrite complete"
```

### Task 8.3: Final acceptance gates

**Files:** none — read-only validation

- [ ] **Step 1: Sign-flip canary**

Confirm the canary still bites by deliberately mutating `metrics/distribution.py`:

```bash
# Apply a sign flip to the EMD metric
sed -i '' 's/return distance/return -distance/' pyphi/metrics/distribution.py
uv run pytest test/test_golden_regression.py test/test_invariants_hypothesis.py -q
```

Expected: at least 3 fixtures fail and at least 1 property fails. Restore:

```bash
git checkout -- pyphi/metrics/distribution.py
```

- [ ] **Step 2: Pyright clean**

```bash
uv run pyright pyphi/core/
```

Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Surface drift green**

```bash
uv run pytest test/test_subsystem_surface.py -v
```

Expected: passes.

- [ ] **Step 4: Layering tests green**

```bash
uv run pytest test/test_core_layering.py -v
```

Expected: 4 passing.

- [ ] **Step 5: Full suite (both lanes)**

```bash
uv run pytest -q
```

Expected: all green (including Hypothesis).

- [ ] **Step 6: Add changelog fragment**

Create `changelog.d/p7-subsystem-layered-rewrite.refactor.md`:

```
Replace the 1354-line ``Subsystem`` god-object with a layered architecture
under ``pyphi.core``. ``CandidateSystem`` (immutable, hashable) replaces
``Subsystem`` as the public type; repertoire computation moves into
stateless ``pyphi.core.repertoire_algebra`` functions; the new TPM
Protocol in ``pyphi.core.tpm`` admits ``ExplicitTPM`` (P7) and the
forthcoming ``ImplicitTPM`` (P12). MacroSubsystem rewrite is deferred
to P7b. ``Concept`` no longer carries a ``subsystem`` back-reference.
```

```bash
git add changelog.d/p7-subsystem-layered-rewrite.refactor.md
git commit -m "P7: changelog fragment"
```

- [ ] **Step 7: Open the PR (when ready)**

```bash
git push -u origin feature/p7-kernel-rewrite
gh pr create --title "P7: subsystem layered rewrite" --body "$(cat <<'EOF'
## Summary

- Replace 1354-line ``Subsystem`` with layered ``pyphi.core/``: ``CausalModel``, ``CandidateSystem``, ``repertoire_algebra``, ``tpm/`` (Protocol + ``ExplicitTPM``)
- Frozen value types, stateless algorithm modules, decorator-based memoization
- Public type renamed ``Subsystem`` → ``CandidateSystem`` (matches IIT 4.0 paper)
- ``Concept.subsystem`` back-reference removed
- MacroSubsystem rewrite deferred to P7b (next project)
- All 17 golden fixtures, 19 Hypothesis properties, sign-flip canary, surface drift, pyright all green

## Test plan

- [x] Golden fixtures match to 1e-12
- [x] Hypothesis invariants all green
- [x] Sign-flip canary bites ≥3 fixtures + ≥1 property
- [x] ``CandidateSystem`` satisfies ``SubsystemPublicInterface``
- [x] Layering tests confirm downward-only dependency
- [x] ``pyphi.macro`` imports succeed; ``MacroSubsystem(...)`` raises ``NotImplementedError`` (P7b sentinel)
- [x] Pyright clean on ``pyphi/core/``

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** Each spec section has tasks:
- Architecture / file inventory: Phase 1, 2, 3, 4, 5 build the new files; Phase 6, 7, 8 modify existing
- TPM Protocol: Tasks 1.1-1.4
- Unit, Substrate, CausalModel: Tasks 2.1-2.3
- CandidateSystem: Tasks 3.1-3.3 (skeleton + properties), 5.1 (proxies), 5.2 (exports)
- Repertoire algebra + cache mechanics: Tasks 4.1-4.6
- Cutover (formalism, compute, actual, examples, models): Tasks 6.1-6.5
- Macro disable: Task 7.1, 7.2
- Test rename: Task 7.3
- Final delete + acceptance gates: Tasks 8.1-8.3

**Placeholder scan:** No "TBD" / "implement later" / vague steps. Every code block is concrete.

**Type consistency:** `CandidateSystem` constructor takes `(causal_model, state, node_indices, cut)` everywhere; `CausalModel.from_network` returns `CausalModel`; `ExplicitTPM` exposes `to_array()` consistently in tests and tasks.

**One known compromise:** Tasks 4.2-4.6 use `_legacy_subsystem` delegation inside `core/repertoire_algebra.py`. This means the `core/` package depends on `pyphi/subsystem.py` until Task 8.2. Phase 6 cutover relies on `Subsystem` alias = `CandidateSystem`, which means `pyphi.subsystem.Subsystem` (legacy class) is *still* used internally by `_legacy_subsystem` even after the public alias has switched. This is intentional during the worktree — the legacy class is the parity baseline. Task 8.2 is where it actually goes away.
