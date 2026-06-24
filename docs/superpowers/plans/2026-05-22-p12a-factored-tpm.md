# P12a — Factored TPM Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `FactoredTPM` as the canonical substrate storage, cut every hot path over to consume per-node factors directly, rename the legacy joint storage from `ExplicitTPM` to `JointTPM`, and benchmark-pick the storage backend — all in 12 commits that keep every gate green and all existing binary goldens byte-identical.

**Architecture:** A `FactoredTPM` holds N per-node conditional marginals; the joint factors as `P(s_{t+1} | s_t) = ∏_i P(s_{i,t+1} | s_t)` under IIT's conditional-independence assumption. Internal storage is a swappable backend (`_NdarrayBackend` default; `_XarrayBackend` opt-in via `pyphi[xarray]` extra, decided by an in-project benchmark). `Substrate` stores `FactoredTPM` canonically; joint reconstruction is a lazy method (no cache). The TPM Protocol drops `squeeze`; `squeeze` lives only on `JointTPM`. Internals are alphabet-generic (`Unit.alphabet_size: int = 2` default); user surface stays binary.

**Tech Stack:** Python 3.13, NumPy, Hypothesis (property tests), xarray (optional dep), pytest, pyright, ruff. Pre-commit hooks gate every commit. `git -c commit.gpgsign=false` for signing this session.

---

## Spec correction (read first)

The spec at `docs/superpowers/specs/2026-05-22-p12a-factored-tpm-design.md` cites two file paths that don't exist in the current codebase. Translation to actual paths:

| Spec says | Actually lives at |
|---|---|
| `pyphi/repertoire.py` | `pyphi/core/repertoire_algebra.py` |
| `pyphi/subsystem.py` | `pyphi/system.py` (was renamed Network→Substrate / Subsystem→System) |

This plan uses the actual paths throughout. The spec's content is otherwise correct; the file-path references in §2.1, §4.1, §7.4, §8.1 should be read against the table above.

**Additional pre-existing observation:** the math in `pyphi/core/repertoire_algebra.py` is *already factored conceptually* — `_cause_repertoire_inner` (line 156-172) and `_effect_repertoire_inner` (line 175-186) compute joint repertoires as products of per-node functions. The current implementation extracts per-node factors by marginalizing out of `System._typed_tpm` (the joint ExplicitTPM, materialized lazily) on each call. The P12a cutover replaces those marginalize-from-joint calls with direct `factored_tpm.factor(i)` reads. The algorithmic shape stays the same; the data access pattern changes.

---

## Branch state baseline & pre-flight

```bash
# Confirm starting position
git log -1 --oneline   # → bba0dba4 ... Add P12a factored-TPM foundation design spec
git status --short | wc -l   # → 27 (all untracked items; must NOT be staged)
git status --short | grep -c "^ M"   # → 0 (no unstaged tracked-file mods)
```

If `git status` shows anything different from the above, surface to the user before proceeding — the plan assumes a clean tracked tree.

---

## File responsibilities map

**New files (created during P12a):**

```
pyphi/core/tpm/factored.py              # FactoredTPM class (public API + machinery)
pyphi/core/tpm/_factored_backends.py    # _StorageBackend Protocol + _NdarrayBackend + _XarrayBackend
benchmarks/factored_tpm_backend.py      # xarray-vs-ndarray micro-benchmark
benchmarks/results/factored-tpm-backend-2026-05-22.md     # decision artifact (markdown)
benchmarks/results/factored-tpm-backend-2026-05-22.json   # raw per-trial timings
test/test_factored_tpm.py               # unit tests for FactoredTPM API
test/test_factored_tpm_kary.py          # Hypothesis property tests k ∈ {2,3,4,5}
test/test_marginalization_factored.py   # cause_tpm/effect_tpm dispatch against FactoredTPM
changelog.d/factored-tpm.feature.md
changelog.d/rename-explicit-tpm-to-joint-tpm.change.md
```

**Modified files (existing, P12a touches them):**

```
pyphi/core/tpm/base.py                  # TPM Protocol gains alphabet_sizes; loses squeeze
pyphi/core/tpm/__init__.py              # export FactoredTPM, JointTPM
pyphi/core/tpm/marginalization.py       # Protocol dispatch; FactoredTPM fast path
pyphi/core/unit.py                      # add alphabet_size: int = 2
pyphi/substrate.py                      # canonical TPM storage → FactoredTPM
pyphi/system.py                         # _typed_tpm and cause/effect_tpm read FactoredTPM
pyphi/core/repertoire_algebra.py        # _single_node_*_repertoire read FactoredTPM.factor(i)
pyphi/validate.py                       # factored_tpm validator; rename existing tpm → joint_tpm
pyphi/__init__.py                       # re-export FactoredTPM, JointTPM
pyphi/tpm.py                            # ExplicitTPM → JointTPM rename in-file
pyproject.toml                          # add [xarray] optional extra
.github/workflows/test.yml              # add CI matrix: one job with xarray, one without
```

**Renamed (git mv during P12a):**

```
pyphi/core/tpm/explicit.py → pyphi/core/tpm/joint.py
```

---

## TDD pattern (applies to every task)

Every task follows the bite-sized TDD discipline from the writing-plans skill:

1. Write the failing test first.
2. Run it to confirm it fails for the right reason.
3. Implement the minimal code to pass.
4. Run the test to confirm it passes.
5. Run the surrounding test file to confirm no regressions.
6. Commit.

For mechanical refactor steps (renames, scaffold-marker removals) where TDD doesn't apply, the pattern is:

1. Make the change.
2. Run pyright + ruff on touched files; resolve any errors.
3. Run the surrounding test file(s) to confirm no regressions.
4. Commit.

**Every commit must pass pre-commit hooks.** Never `--no-verify`. If a hook fails, run the tool directly (`uv run ruff check <file>`, `uv run pyright <file>`) and fix the root cause.

**gpgsign:** use `git -c commit.gpgsign=false commit -m "..."`. If the 1Password agent error persists despite the bypass, surface to the user — do NOT use `--no-verify`.

**Staging:** targeted `git add <files>` only. Before every commit run `git diff --cached --stat`; after commit run `git show --stat HEAD`. The repo has 27 untracked items that must not be staged.

---

## Task 1: Extend TPM Protocol; add Unit.alphabet_size; ExplicitTPM alphabet_sizes property

**Goal:** Lift the implicit-binary assumption from the kernel Protocol layer without yet adding the factored representation. The Protocol gains `alphabet_sizes` and loses `squeeze` (which has no coherent meaning on the factored form). `Unit` gains an `alphabet_size: int = 2` default. The existing `ExplicitTPM` port implements `alphabet_sizes` returning `(2,) * n_nodes`.

**Files:**

- Modify: `pyphi/core/tpm/base.py` (~42 lines today; lines 28-41 are the Protocol body)
- Modify: `pyphi/core/unit.py` (~20 lines today)
- Modify: `pyphi/core/tpm/explicit.py` (~67 lines today; add the new property)
- Modify: `pyphi/system.py:188, 194` (existing `.squeeze()` calls on `effect_tpm`/`cause_tpm` — these are on `proper_effect_tpm`/`proper_cause_tpm`'s ndarrays via `np.asarray(self.effect_tpm.squeeze())`; the `.squeeze()` is numpy's, not the Protocol's. Verify and leave intact.)
- Test (new file): `test/test_tpm_protocol.py`

- [ ] **Step 1.1: Write failing test for Protocol shape.**

Create `test/test_tpm_protocol.py`:

```python
"""Verify the TPM Protocol contract and ExplicitTPM conformance."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.base import TPM
from pyphi.core.tpm.explicit import ExplicitTPM
from pyphi.core.unit import Unit


def test_tpm_protocol_has_alphabet_sizes() -> None:
    """The TPM Protocol exposes alphabet_sizes as a property."""
    tpm = ExplicitTPM(np.zeros((2, 2, 2, 3)))
    assert isinstance(tpm, TPM)
    assert tpm.alphabet_sizes == (2, 2, 2)


def test_tpm_protocol_lacks_squeeze() -> None:
    """The TPM Protocol no longer requires squeeze (lives on JointTPM only)."""
    assert not hasattr(TPM, "squeeze") or "squeeze" not in TPM.__protocol_attrs__  # type: ignore[attr-defined]


def test_unit_has_alphabet_size_default_2() -> None:
    """Unit defaults to alphabet_size=2."""
    u = Unit(index=0, label="A")
    assert u.alphabet_size == 2


def test_unit_alphabet_size_overridable() -> None:
    """Unit.alphabet_size accepts a non-default value."""
    u = Unit(index=0, label="A", alphabet_size=3)
    assert u.alphabet_size == 3
```

- [ ] **Step 1.2: Run failing test.**

```bash
uv run pytest test/test_tpm_protocol.py -v
```

Expected: 4 failures. `test_tpm_protocol_has_alphabet_sizes` fails on `tpm.alphabet_sizes`; `test_unit_has_alphabet_size_default_2` and `test_unit_alphabet_size_overridable` fail on the missing field.

- [ ] **Step 1.3: Add alphabet_size to Unit.**

Replace `pyphi/core/unit.py` body (lines 1-20) with:

```python
"""Unit value type — atomic node in a substrate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic node in a substrate.

    Holds the node's index, label, and alphabet size (number of distinct
    states the node can take). Alphabet size defaults to 2 (binary). Math
    operations against ``Unit`` are parameterized by ``alphabet_size``;
    multi-valued substrates pass non-2 values.
    """

    index: int
    label: str
    alphabet_size: int = 2
```

The pre-existing `P7: alphabet is implicit binary (0 or 1). P12 adds ``alphabet_size``.` line is intentionally removed — this docstring describes the final state.

- [ ] **Step 1.4: Update TPM Protocol.**

Replace `pyphi/core/tpm/base.py` body with:

```python
"""TPM Protocol — the structural contract every transition probability matrix satisfies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TPM(Protocol):
    """Structural protocol satisfied by every PyPhi TPM.

    Implementations: :class:`pyphi.core.tpm.explicit.ExplicitTPM` (joint
    ndarray storage) and :class:`pyphi.core.tpm.factored.FactoredTPM`
    (per-node factor storage).
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def alphabet_sizes(self) -> tuple[int, ...]: ...

    def condition(self, fixed: Mapping[int, int]) -> "TPM": ...

    def to_array(self) -> NDArray[np.float64]: ...
```

(Removed: `squeeze`. Removed: the `# P12 lifts that assumption` comment block.)

- [ ] **Step 1.5: Add `alphabet_sizes` property to ExplicitTPM (legacy port).**

In `pyphi/core/tpm/explicit.py`, between the existing `n_nodes` property and `condition` method, add:

```python
    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        """All nodes binary in the joint-storage form (pre-P12b)."""
        return (2,) * self.n_nodes
```

Keep the existing `squeeze` method on `ExplicitTPM` — it is JointTPM-specific (numpy cleanup) and survives outside the Protocol.

- [ ] **Step 1.6: Run tests to confirm pass.**

```bash
uv run pytest test/test_tpm_protocol.py -v
```

Expected: 4 passed.

- [ ] **Step 1.7: Run pyright + ruff on touched files.**

```bash
uv run pyright pyphi/core/tpm/base.py pyphi/core/tpm/explicit.py pyphi/core/unit.py test/test_tpm_protocol.py
uv run ruff check pyphi/core/tpm/base.py pyphi/core/tpm/explicit.py pyphi/core/unit.py test/test_tpm_protocol.py
```

Expected: 0 errors / 0 baseline-only warnings on pyright; ruff clean.

- [ ] **Step 1.8: Smoke-test the broader suite.**

```bash
uv run pytest test/test_core_tpm.py test/test_system.py test/test_substrate.py -x -q
```

Expected: all pass (the new property is additive; nothing was removed except `squeeze` from the Protocol, which no consumer code references — confirm via grep `TPM.squeeze\|Protocol.*squeeze` if unsure).

- [ ] **Step 1.9: Commit.**

```bash
git add pyphi/core/tpm/base.py pyphi/core/tpm/explicit.py pyphi/core/unit.py test/test_tpm_protocol.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Extend TPM Protocol with alphabet_sizes; add Unit.alphabet_size

Lifts the implicit-binary assumption from the kernel Protocol layer.
TPM Protocol gains an alphabet_sizes property; squeeze drops out of the
Protocol (it remains on the joint-storage class, which is JointTPM-
specific). Unit gains alphabet_size: int = 2; binary is still the
default everywhere in the codebase.

Foundational change per the spec; no consumer code paths yet."
git show --stat HEAD
```

---

## Task 2: Add FactoredTPM skeleton with ndarray-only backend

**Goal:** Land a `FactoredTPM` class with the ndarray storage backend, validation, equality, repr, and pickling. No consumers wire to it yet — this task ships the class and its unit tests.

**Files:**

- Create: `pyphi/core/tpm/factored.py`
- Create: `pyphi/core/tpm/_factored_backends.py`
- Modify: `pyphi/core/tpm/__init__.py` (add re-export)
- Create: `test/test_factored_tpm.py`

- [ ] **Step 2.1: Write failing tests for FactoredTPM construction + basic accessors.**

Create `test/test_factored_tpm.py`:

```python
"""Unit tests for FactoredTPM — per-node factored conditional TPM."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pyphi.core.tpm.base import TPM
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.exceptions import InvalidTPM


# --- helper: build a valid binary 2-node FactoredTPM ---

def _two_node_factored() -> FactoredTPM:
    # 2 binary nodes; each node has 2 inputs (both nodes are inputs to both).
    # factor[i] shape: (a_1, a_2, a_i) = (2, 2, 2). All entries sum to 1
    # along the last axis.
    f0 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]],
         [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    f1 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]],
         [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    return FactoredTPM(factors=[f0, f1], alphabet_sizes=(2, 2))


def test_factored_tpm_construction() -> None:
    f = _two_node_factored()
    assert f.n_nodes == 2
    assert f.alphabet_sizes == (2, 2)


def test_factored_tpm_satisfies_protocol() -> None:
    f = _two_node_factored()
    assert isinstance(f, TPM)


def test_factored_tpm_shape() -> None:
    f = _two_node_factored()
    # The "joint shape" is alphabet_sizes + (n_nodes,) — what
    # to_joint produces in the legacy convention.
    assert f.shape == (2, 2, 2)


def test_factored_tpm_factor_access() -> None:
    f = _two_node_factored()
    assert f.factor(0).shape == (2, 2, 2)
    assert f.factor(1).shape == (2, 2, 2)
    np.testing.assert_allclose(f.factor(0), f.factors[0])


def test_factored_tpm_validation_rejects_nonsumming_factor() -> None:
    bad = np.array(
        [[[0.3, 0.3], [0.5, 0.5]],
         [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    good = bad.copy()
    good[0, 0, 0] = 0.5  # restore sum-to-1
    with pytest.raises(InvalidTPM, match="sums to 1"):
        FactoredTPM(factors=[bad, good], alphabet_sizes=(2, 2))


def test_factored_tpm_validation_rejects_alphabet_lt_2() -> None:
    f0 = np.array([[[1.0]]], dtype=np.float64)
    with pytest.raises(InvalidTPM, match="alphabet"):
        FactoredTPM(factors=[f0], alphabet_sizes=(1,))


def test_factored_tpm_equality() -> None:
    a = _two_node_factored()
    b = _two_node_factored()
    assert a == b
    assert not (a != b)


def test_factored_tpm_repr() -> None:
    f = _two_node_factored()
    r = repr(f)
    assert "FactoredTPM" in r
    assert "n_nodes=2" in r


def test_factored_tpm_pickling() -> None:
    f = _two_node_factored()
    restored = pickle.loads(pickle.dumps(f))
    assert restored == f
```

- [ ] **Step 2.2: Run failing tests.**

```bash
uv run pytest test/test_factored_tpm.py -v
```

Expected: 9 errors (`ImportError: cannot import name 'FactoredTPM' from 'pyphi.core.tpm.factored'`).

- [ ] **Step 2.3: Create the storage backend module.**

Create `pyphi/core/tpm/_factored_backends.py`:

```python
"""Internal storage backends for FactoredTPM.

Not part of the public API. The chosen backend is selected by
:data:`pyphi.core.tpm.factored._FACTORED_TPM_DEFAULT_BACKEND` (set by
the in-project benchmark; see ``benchmarks/factored_tpm_backend.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


@runtime_checkable
class _StorageBackend(Protocol):
    """Internal storage abstraction for FactoredTPM."""

    def get_factor(self, i: int) -> NDArray[np.float64]: ...
    def n_factors(self) -> int: ...
    def alphabet_sizes(self) -> tuple[int, ...]: ...
    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]: ...
    def all_factors(self) -> tuple[NDArray[np.float64], ...]: ...


class _NdarrayBackend:
    """Tuple of ndarrays. Positional indexing.

    Name-based lookup goes through FactoredTPM's node-label mapping
    (FactoredTPM is the public surface for that).
    """

    __slots__ = ("_factors", "_alphabet_sizes")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int],
    ) -> None:
        self._factors = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        self._alphabet_sizes = tuple(int(a) for a in alphabet_sizes)

    def get_factor(self, i: int) -> NDArray[np.float64]:
        return self._factors[i]

    def n_factors(self) -> int:
        return len(self._factors)

    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    def all_factors(self) -> tuple[NDArray[np.float64], ...]:
        return self._factors

    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        idx: list[Any] = [slice(None)] * factor.ndim
        for j, state_j in fixed.items():
            idx[j] = state_j
        out = factor[tuple(idx)]
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out


def _make_default_backend(
    factors: Sequence[ArrayLike],
    alphabet_sizes: Sequence[int],
    requested: str | None,
) -> _StorageBackend:
    """Construct the requested storage backend. ``None`` uses the module default."""
    from pyphi.core.tpm.factored import _FACTORED_TPM_DEFAULT_BACKEND

    backend_name = requested or _FACTORED_TPM_DEFAULT_BACKEND
    if backend_name == "ndarray":
        return _NdarrayBackend(factors, alphabet_sizes)
    if backend_name == "xarray":
        from pyphi.core.tpm._factored_backends_xarray import _XarrayBackend
        return _XarrayBackend(factors, alphabet_sizes)
    raise ValueError(f"Unknown backend {backend_name!r}; expected 'ndarray' or 'xarray'.")
```

Note: the xarray backend is referenced but not yet imported eagerly — it lives in a separate module loaded lazily (Task 4 creates it). In Task 2 the `if backend_name == "xarray"` branch raises `ImportError` when reached. That's acceptable because Task 2's tests never request xarray.

- [ ] **Step 2.4: Create the FactoredTPM class.**

Create `pyphi/core/tpm/factored.py`:

```python
"""Per-node-factored conditional TPM.

Represents the joint conditional ``P(s_{t+1} | s_t)`` as a product of N
per-node conditional marginals ``P(s_{i,t+1} | s_t)``. The joint is the
product of the factors under conditional independence (IIT's standing
assumption that nodes update independently given the joint past).

Factor ``i`` has shape ``(a_1, ..., a_N, a_i)`` where ``a_j`` is the
alphabet size of node ``j``. Input dims for non-input nodes are size 1
and are semantically load-bearing — they encode the connectivity
structure and are never squeezed away.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import exceptions
from pyphi.conf import config

from ._factored_backends import _NdarrayBackend
from ._factored_backends import _StorageBackend
from ._factored_backends import _make_default_backend

# Set in Task 6 from the storage-backend benchmark result.
_FACTORED_TPM_DEFAULT_BACKEND: Literal["ndarray", "xarray"] = "ndarray"


class FactoredTPM:
    """Per-node-factored conditional TPM."""

    __slots__ = ("_backend", "_alphabet_sizes")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int] | None = None,
        backend: Literal["ndarray", "xarray"] | None = None,
    ) -> None:
        factor_arrays = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        if alphabet_sizes is None:
            alphabet_sizes = tuple(int(f.shape[-1]) for f in factor_arrays)
        else:
            alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
        self._alphabet_sizes = alphabet_sizes
        self._backend = _make_default_backend(factor_arrays, alphabet_sizes, backend)
        _validate(self)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._alphabet_sizes + (self.n_nodes,)

    @property
    def n_nodes(self) -> int:
        return self._backend.n_factors()

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    @property
    def factors(self) -> tuple[NDArray[np.float64], ...]:
        return self._backend.all_factors()

    def factor(self, i: int) -> NDArray[np.float64]:
        return self._backend.get_factor(i)

    def condition(self, fixed: Mapping[int, int]) -> "FactoredTPM":
        conditioned = [self._backend.select(i, fixed) for i in range(self.n_nodes)]
        return FactoredTPM(
            factors=conditioned,
            alphabet_sizes=self._alphabet_sizes,
            backend=None,
        )

    def condition_factor(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        return self._backend.select(i, fixed)

    def to_array(self) -> NDArray[np.float64]:
        return self.to_joint()

    def to_joint(self) -> NDArray[np.float64]:
        # Implemented in Task 3 — return a placeholder for Task 2's tests
        # by zero-filling the joint shape. (Task 3 replaces this.)
        return np.zeros(self.shape, dtype=np.float64)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactoredTPM):
            return NotImplemented
        if self._alphabet_sizes != other._alphabet_sizes:
            return False
        if self.n_nodes != other.n_nodes:
            return False
        return all(
            np.array_equal(self.factor(i), other.factor(i))
            for i in range(self.n_nodes)
        )

    def __hash__(self) -> int:
        # value-equal hashing via factor bytes
        return hash(
            (
                self._alphabet_sizes,
                tuple(self.factor(i).tobytes() for i in range(self.n_nodes)),
            )
        )

    def __repr__(self) -> str:
        return (
            f"FactoredTPM(n_nodes={self.n_nodes}, "
            f"alphabet_sizes={self._alphabet_sizes})"
        )

    def __reduce__(self) -> tuple:
        backend_name = "ndarray" if isinstance(self._backend, _NdarrayBackend) else "xarray"
        return (
            _factored_tpm_from_pickle,
            (tuple(self.factors), self._alphabet_sizes, backend_name),
        )


def _factored_tpm_from_pickle(
    factors: tuple,
    alphabet_sizes: tuple,
    backend: str,
) -> FactoredTPM:
    return FactoredTPM(factors=factors, alphabet_sizes=alphabet_sizes, backend=backend)


def _validate(factored: FactoredTPM) -> None:
    """Validate a freshly constructed FactoredTPM."""
    a = factored.alphabet_sizes
    if any(size < 2 for size in a):
        raise exceptions.InvalidTPM(
            f"alphabet_sizes must all be >= 2; got {a}"
        )
    tol = max(10 ** (-config.numerics.precision), 1e-15)
    for i in range(factored.n_nodes):
        f = factored.factor(i)
        # last dim must be alphabet_sizes[i]
        if f.shape[-1] != a[i]:
            raise exceptions.InvalidTPM(
                f"factor {i} last-dim size {f.shape[-1]} != alphabet_sizes[{i}]={a[i]}"
            )
        # input dims must be either alphabet_sizes[j] or 1 (non-input)
        for j, dim_size in enumerate(f.shape[:-1]):
            if dim_size not in (1, a[j]):
                raise exceptions.InvalidTPM(
                    f"factor {i} input dim {j} has size {dim_size}; "
                    f"expected 1 (non-input) or {a[j]} (input)"
                )
        # sums to 1 along last dim within tolerance
        sums = f.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=tol):
            worst = np.unravel_index(np.abs(sums - 1.0).argmax(), sums.shape)
            raise exceptions.InvalidTPM(
                f"factor {i} sums to 1 violated at input state {worst}: "
                f"got {sums[worst]}, tolerance {tol}"
            )
```

- [ ] **Step 2.5: Wire the export.**

Modify `pyphi/core/tpm/__init__.py`:

```python
"""Kernel TPM types."""

from .base import TPM as TPM
from .explicit import ExplicitTPM as ExplicitTPM
from .factored import FactoredTPM as FactoredTPM
```

(Task 7 will rename `ExplicitTPM` → `JointTPM`; for now both classes coexist.)

- [ ] **Step 2.6: Run tests; verify they pass.**

```bash
uv run pytest test/test_factored_tpm.py -v
```

Expected: 9 passed.

- [ ] **Step 2.7: Pyright + ruff.**

```bash
uv run pyright pyphi/core/tpm/factored.py pyphi/core/tpm/_factored_backends.py pyphi/core/tpm/__init__.py test/test_factored_tpm.py
uv run ruff check pyphi/core/tpm/factored.py pyphi/core/tpm/_factored_backends.py pyphi/core/tpm/__init__.py test/test_factored_tpm.py
```

Expected: 0 errors. If `_FACTORED_TPM_DEFAULT_BACKEND` raises pyright complaints about `Literal`, it's fine — that's the intended typing.

- [ ] **Step 2.8: Commit.**

```bash
git add pyphi/core/tpm/factored.py pyphi/core/tpm/_factored_backends.py pyphi/core/tpm/__init__.py test/test_factored_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add FactoredTPM skeleton with ndarray backend

Introduces the per-node-factored conditional TPM representation: an
N-tuple of factor arrays where factor i has shape (a_1, ..., a_N, a_i),
the joint factoring as ∏ factors under conditional independence.

Includes construction, validation (sum-to-1, alphabet >= 2, input-dim
shape), equality, hashing, repr, pickling. No consumers wired yet — the
class ships standalone in this commit; the cutover follows."
git show --stat HEAD
```

---

## Task 3: from_joint / to_joint round-trip + binary property test

**Goal:** Replace Task 2's placeholder `to_joint` with the real implementation; add `from_joint`; verify the round-trip is mathematically exact for binary networks via a Hypothesis property test.

**Files:**

- Modify: `pyphi/core/tpm/factored.py` (`to_joint` body; add `from_joint` classmethod)
- Modify: `test/test_factored_tpm.py` (add round-trip tests)

- [ ] **Step 3.1: Write failing tests for from_joint / to_joint.**

Append to `test/test_factored_tpm.py`:

```python
# --- round-trip tests ---

def _random_joint_tpm(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random binary joint TPM, shape (2,)*n + (n,) — entries are P(node_i=1)."""
    return rng.uniform(size=(2,) * n + (n,))


def test_from_joint_round_trip_n2() -> None:
    rng = np.random.default_rng(42)
    joint = _random_joint_tpm(rng, 2)
    # Convert legacy "P(node=1)" form to the explicit-alphabet last dim:
    # legacy shape (2, 2, 2) → factored expects (2, 2, n, 2). For P12a's
    # from_joint, we accept either: the binary legacy form OR the
    # explicit-alphabet form. See implementation.
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2, 2))
    reconstructed = factored.to_joint()
    # Round-trip up to floating-point reordering: factored reconstruction
    # produces shape (2, 2, n, 2) (full alphabet); we compare against the
    # explicit-alphabet form of joint.
    p_on = joint  # shape (2, 2, 2); last dim is node index
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    # explicit_joint has shape (2, 2, 2, 2) — last dim is alphabet
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_round_trip_n3() -> None:
    rng = np.random.default_rng(99)
    joint = _random_joint_tpm(rng, 3)
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2, 2, 2))
    reconstructed = factored.to_joint()
    p_on = joint
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_invalid_shape_raises() -> None:
    bad = np.zeros((2, 2))  # not an (a,)*n + (n,) shape
    with pytest.raises(ValueError, match="shape"):
        FactoredTPM.from_joint(bad, alphabet_sizes=(2, 2))


def test_to_joint_shape() -> None:
    f = _two_node_factored()
    joint = f.to_joint()
    # joint: (alphabet_sizes) + (n_nodes, max_alphabet) — but for binary
    # uniform alphabet, shape is (2, 2, 2, 2). See implementation note.
    assert joint.shape[:-2] == (2, 2)
    assert joint.shape[-2] == 2  # n_nodes
    assert joint.shape[-1] == 2  # alphabet


def test_from_joint_to_joint_roundtrip_stability_binary() -> None:
    """Round-trip preserves the joint to floating-point precision."""
    rng = np.random.default_rng(2026)
    for n in (2, 3, 4):
        joint = _random_joint_tpm(rng, n)
        factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)
        reconstructed = factored.to_joint()
        p_on = joint
        explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
        np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)
```

- [ ] **Step 3.2: Run tests — confirm failure.**

```bash
uv run pytest test/test_factored_tpm.py -v
```

Expected: 5 new failures (the existing 9 still pass).

- [ ] **Step 3.3: Implement from_joint.**

In `pyphi/core/tpm/factored.py`, add to the `FactoredTPM` class (after `__init__`):

```python
    @classmethod
    def from_joint(
        cls,
        joint: ArrayLike,
        /,
        alphabet_sizes: Sequence[int] | None = None,
    ) -> "FactoredTPM":
        """Convert a joint conditional TPM into the factored form.

        Accepts either:

        - Legacy binary form: shape ``(2,) * n + (n,)``, where the last
          dim's entry ``i`` is ``P(node_i = 1 | s_t)``. Factor ``i`` is
          built by stacking ``[1 - p_on, p_on]`` along an explicit
          alphabet dim.

        - Explicit-alphabet form: shape ``(a_1, ..., a_N, N, a_i)``.
          Factor ``i`` is ``joint[..., i, :]``.

        ``alphabet_sizes`` defaults to ``(2,) * n`` for the legacy form;
        for the explicit form it must be supplied and must match the
        per-row last-dim shapes.
        """
        joint_arr = np.asarray(joint, dtype=np.float64)
        ndim = joint_arr.ndim
        if alphabet_sizes is None:
            # Infer binary from legacy shape
            if ndim < 2 or joint_arr.shape[-1] != ndim - 1:
                raise ValueError(
                    f"Cannot infer alphabet_sizes from joint shape "
                    f"{joint_arr.shape}; expected legacy form "
                    f"(2,)*n + (n,) or pass alphabet_sizes explicitly."
                )
            n = ndim - 1
            alphabet_sizes = (2,) * n
        else:
            alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
            n = len(alphabet_sizes)

        # Shape sanity
        if joint_arr.shape[:-1] != alphabet_sizes:
            # Maybe explicit-alphabet form: (a_1, ..., a_N, N, a_max)
            if (
                ndim == n + 2
                and joint_arr.shape[:n] == alphabet_sizes
                and joint_arr.shape[n] == n
            ):
                factors = tuple(joint_arr[..., i, :] for i in range(n))
                return cls(factors=factors, alphabet_sizes=alphabet_sizes)
            raise ValueError(
                f"Joint shape {joint_arr.shape} not consistent with "
                f"alphabet_sizes {alphabet_sizes}."
            )

        if joint_arr.shape[-1] != n:
            raise ValueError(
                f"Legacy joint shape requires last dim == n_nodes={n}; "
                f"got shape {joint_arr.shape}."
            )
        if alphabet_sizes != (2,) * n:
            raise ValueError(
                f"Legacy joint form is binary-only; "
                f"alphabet_sizes={alphabet_sizes} requires explicit-form joint."
            )

        # Binary path: stack [1-p_on, p_on] along an explicit alphabet dim
        # joint_arr shape: (2,)*n + (n,) ; entry [s_1,...,s_n, i] is P(node_i=1)
        factors_list: list[NDArray[np.float64]] = []
        for i in range(n):
            p_on = joint_arr[..., i]  # shape (2,)*n
            factor_i = np.stack([1.0 - p_on, p_on], axis=-1)  # shape (2,)*n + (2,)
            factors_list.append(factor_i)
        return cls(factors=tuple(factors_list), alphabet_sizes=alphabet_sizes)
```

- [ ] **Step 3.4: Implement to_joint.**

Replace the placeholder `to_joint` in `pyphi/core/tpm/factored.py` with:

```python
    def to_joint(self) -> NDArray[np.float64]:
        """Materialize the joint conditional ``P(s_{t+1} | s_t)`` from the factors.

        Output shape is ``alphabet_sizes + (n_nodes, max_alphabet)`` where the
        per-row last dim slot holds factor ``i``'s distribution. For uniform
        alphabets this equals ``(a, a, ..., a, n, a)``. Slow path — only used
        at boundaries (serialization, legacy fixture comparison,
        ``Substrate.joint_tpm()``).
        """
        n = self.n_nodes
        max_alphabet = max(self._alphabet_sizes)
        shape = self._alphabet_sizes + (n, max_alphabet)
        out = np.zeros(shape, dtype=np.float64)
        for i in range(n):
            factor = self.factor(i)
            a_i = self._alphabet_sizes[i]
            # Broadcast factor (shape: alphabet_sizes + (a_i,)) into out[..., i, :a_i]
            broadcast_shape = self._alphabet_sizes + (a_i,)
            out[..., i, :a_i] = np.broadcast_to(factor, broadcast_shape)
        return out
```

(Note: heterogeneous alphabets produce zero-padded slots for nodes with `a_i < max_alphabet`. For uniform binary the shape is `(2, 2, ..., 2, n, 2)`, which is what the tests expect.)

- [ ] **Step 3.5: Run tests; confirm pass.**

```bash
uv run pytest test/test_factored_tpm.py -v
```

Expected: 14 passed.

- [ ] **Step 3.6: Pyright + ruff.**

```bash
uv run pyright pyphi/core/tpm/factored.py test/test_factored_tpm.py
uv run ruff check pyphi/core/tpm/factored.py test/test_factored_tpm.py
```

Expected: 0 errors.

- [ ] **Step 3.7: Commit.**

```bash
git add pyphi/core/tpm/factored.py test/test_factored_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Implement FactoredTPM.from_joint and to_joint round-trip

from_joint converts legacy binary (P(node_i=1)) or explicit-alphabet
joint TPMs to the factored form. to_joint reconstructs the joint by
broadcasting each factor into its output slot. Round-trip is exact to
float64 precision for binary networks across n in {2, 3, 4}."
git show --stat HEAD
```

---

## Task 4: Add xarray backend; optional dep wiring

**Goal:** Add `_XarrayBackend` behind a lazy import; wire `pyphi[xarray]` as an optional extra; add CI matrix so one job runs with xarray installed and one without; ensure `_XarrayBackend` raises a clear `ImportError` when xarray is unavailable.

**Files:**

- Create: `pyphi/core/tpm/_factored_backends_xarray.py`
- Modify: `pyproject.toml` (add `[xarray]` extra)
- Modify: `.github/workflows/test.yml` (add CI matrix entry)
- Modify: `test/test_factored_tpm.py` (add xarray-backend tests, conditionally skipped)

- [ ] **Step 4.1: Write failing test for xarray backend selection.**

Append to `test/test_factored_tpm.py`:

```python
# --- xarray backend (optional) ---

xarray = pytest.importorskip("xarray")  # type: ignore[assignment]


def _two_node_factored_xarray() -> FactoredTPM:
    f0 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]],
         [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    f1 = f0.copy()
    return FactoredTPM(factors=[f0, f1], alphabet_sizes=(2, 2), backend="xarray")


def test_factored_tpm_xarray_backend_selectable() -> None:
    f = _two_node_factored_xarray()
    assert f.n_nodes == 2
    assert f.alphabet_sizes == (2, 2)


def test_factored_tpm_xarray_factor_equals_ndarray_factor() -> None:
    """Cross-backend equality on identical factor data."""
    nd = FactoredTPM(
        factors=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)],
        alphabet_sizes=(2, 2),
        backend="ndarray",
    )
    xr = FactoredTPM(
        factors=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)],
        alphabet_sizes=(2, 2),
        backend="xarray",
    )
    assert nd == xr
```

- [ ] **Step 4.2: Run failing test (xarray must be installed for this to fail meaningfully; if not installed, the file is skipped).**

```bash
uv pip install xarray  # if not already in dev env
uv run pytest test/test_factored_tpm.py -k xarray -v
```

Expected (with xarray installed): import error or `_XarrayBackend` not found.

- [ ] **Step 4.3: Create the xarray backend module.**

Create `pyphi/core/tpm/_factored_backends_xarray.py`:

```python
"""xarray storage backend for FactoredTPM.

Imported lazily by ``_make_default_backend`` only when ``backend="xarray"``
is requested. If ``xarray`` is not installed, the module-level import
fails with the standard ImportError; the caller in
``_factored_backends._make_default_backend`` lets it propagate so users
see "ModuleNotFoundError: No module named 'xarray'" with the install
hint ``pip install pyphi[xarray]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from numpy.typing import NDArray


class _XarrayBackend:
    """Tuple of xr.DataArray with named input dims.

    Same semantic API as :class:`_NdarrayBackend`. Naming convention:
    factor ``i`` has dims ``("in_0", "in_1", ..., "in_{N-1}", "out_i")``.
    """

    __slots__ = ("_factors", "_alphabet_sizes")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int],
    ) -> None:
        alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
        self._alphabet_sizes = alphabet_sizes
        n = len(alphabet_sizes)
        wrapped: list[xr.DataArray] = []
        for i, f in enumerate(factors):
            arr = np.asarray(f, dtype=np.float64)
            dims = tuple(f"in_{j}" for j in range(n)) + (f"out_{i}",)
            wrapped.append(xr.DataArray(arr, dims=dims))
        self._factors = tuple(wrapped)

    def get_factor(self, i: int) -> NDArray[np.float64]:
        return np.asarray(self._factors[i].values)

    def n_factors(self) -> int:
        return len(self._factors)

    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    def all_factors(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(np.asarray(f.values) for f in self._factors)

    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        # Use named-dim isel for clarity
        idx: dict[str, int] = {f"in_{j}": state_j for j, state_j in fixed.items()}
        sliced = factor.isel(idx)
        # Restore the conditioned dims as singletons
        out = sliced.values
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out
```

- [ ] **Step 4.4: Add the `[xarray]` extra to pyproject.toml.**

In `pyproject.toml`, locate the existing `[project.optional-dependencies]` section. Add:

```toml
xarray = ["xarray>=2024.1"]
```

(Pin to a recent stable release. If a similar pin already exists for other deps, match the style.)

- [ ] **Step 4.5: Add CI matrix for xarray availability.**

In `.github/workflows/test.yml`, locate the existing `pytest` job. Add an `xarray` matrix axis. The pattern: keep the existing job (no xarray) and add a second job with `pyphi[xarray]` installed:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13"]
        xarray: ["", "xarray"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install uv
        run: pip install uv
      - name: Install pyphi (without xarray)
        if: matrix.xarray == ''
        run: uv pip install -e ".[dev,parallel]"
      - name: Install pyphi (with xarray)
        if: matrix.xarray == 'xarray'
        run: uv pip install -e ".[dev,parallel,xarray]"
      - name: Run tests
        run: uv run pytest test/ -m "not slow" -q
```

(Cross-check the exact existing workflow shape — keep what's there; only add the matrix axis and the conditional install steps. If the workflow's structure is different, adapt minimally.)

- [ ] **Step 4.6: Run xarray tests; confirm pass.**

```bash
uv pip install xarray
uv run pytest test/test_factored_tpm.py -v
```

Expected: 16 passed (14 existing + 2 new xarray-backend tests).

Without xarray:

```bash
uv pip uninstall xarray
uv run pytest test/test_factored_tpm.py -v
```

Expected: xarray tests are skipped via `pytest.importorskip`; ndarray-backend tests still pass. (Re-install xarray after this check: `uv pip install xarray`.)

- [ ] **Step 4.7: Pyright + ruff.**

```bash
uv run pyright pyphi/core/tpm/_factored_backends_xarray.py test/test_factored_tpm.py
uv run ruff check pyphi/core/tpm/_factored_backends_xarray.py test/test_factored_tpm.py
```

Expected: 0 errors.

- [ ] **Step 4.8: Commit.**

```bash
git add pyphi/core/tpm/_factored_backends_xarray.py pyproject.toml .github/workflows/test.yml test/test_factored_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add xarray storage backend for FactoredTPM behind optional extra

The xarray backend wraps each factor in an xr.DataArray with named
input and output dims. Imported lazily via the pyphi[xarray] extra; if
xarray is not installed the backend selection branch surfaces
ModuleNotFoundError with the install hint.

CI matrix runs the test suite twice: once with xarray installed, once
without."
git show --stat HEAD
```

---

## Task 5: Storage-backend benchmark

**Goal:** Add the in-project micro-benchmark that measures xarray-vs-ndarray on the operations FactoredTPM consumers will actually exercise. Run it; commit the result file. This commit produces the artifact that drives Task 6's default-backend selection.

**Files:**

- Create: `benchmarks/factored_tpm_backend.py`
- Create: `benchmarks/results/factored-tpm-backend-2026-05-22.md`
- Create: `benchmarks/results/factored-tpm-backend-2026-05-22.json`

- [ ] **Step 5.1: Create the benchmark script.**

Create `benchmarks/factored_tpm_backend.py`:

```python
"""Storage-backend benchmark for FactoredTPM.

Measures xarray vs. ndarray on the four operations that hot-path
consumers exercise: from_joint, to_joint, condition, and factor access.
Network sizes match the perf-budget fixtures (n in {3, 5, 8, 10}) plus
one k=3 size to preview multi-valued. Reports median + p95 wall time
per (operation, backend, size). Writes a markdown table and the raw
per-trial timings.

Decision rule: if xarray is within <= 2x of ndarray on every
(operation, size) measurement, set xarray as the default in
pyphi.core.tpm.factored._FACTORED_TPM_DEFAULT_BACKEND. Otherwise stay
on ndarray.

Usage:
    uv run python benchmarks/factored_tpm_backend.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM


# --- harness ---

WARMUP_TRIALS = 5
MEASURE_TRIALS = 50
SIZES = [3, 5, 8, 10]
K3_SIZE = 4  # one k=3 size for P12b preview


def _time(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _measure(fn: Callable[[], object]) -> dict[str, float]:
    for _ in range(WARMUP_TRIALS):
        fn()
    samples = [_time(fn) for _ in range(MEASURE_TRIALS)]
    return {
        "median_s": float(np.median(samples)),
        "p95_s": float(np.percentile(samples, 95)),
        "min_s": float(np.min(samples)),
        "n_trials": MEASURE_TRIALS,
        "samples_s": samples,
    }


# --- ops ---

def _random_binary_joint(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(size=(2,) * n + (n,))


def _random_kary_factored(rng: np.random.Generator, n: int, k: int) -> FactoredTPM:
    factors = []
    for i in range(n):
        raw = rng.uniform(size=(k,) * n + (k,))
        normalized = raw / raw.sum(axis=-1, keepdims=True)
        factors.append(normalized)
    return FactoredTPM(factors=factors, alphabet_sizes=(k,) * n)


def _bench_size_binary(rng: np.random.Generator, n: int, backend: str) -> dict[str, dict]:
    joint = _random_binary_joint(rng, n)

    def op_from_joint() -> object:
        return FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)

    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)
    # Rebuild with desired backend explicitly
    factored = FactoredTPM(
        factors=factored.factors, alphabet_sizes=(2,) * n, backend=backend
    )

    def op_to_joint() -> object:
        return factored.to_joint()

    def op_condition() -> object:
        return factored.condition({0: 1})

    def op_factor_access() -> object:
        return factored.factor(0)

    return {
        "from_joint": _measure(op_from_joint),
        "to_joint": _measure(op_to_joint),
        "condition": _measure(op_condition),
        "factor_access": _measure(op_factor_access),
    }


def _bench_size_kary(rng: np.random.Generator, n: int, k: int, backend: str) -> dict[str, dict]:
    factored = _random_kary_factored(rng, n, k)
    factored = FactoredTPM(
        factors=factored.factors, alphabet_sizes=(k,) * n, backend=backend
    )

    def op_to_joint() -> object:
        return factored.to_joint()

    def op_condition() -> object:
        return factored.condition({0: 0})

    return {
        "to_joint": _measure(op_to_joint),
        "condition": _measure(op_condition),
    }


# --- main ---

def main(out_dir: Path = Path("benchmarks/results")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 2026
    results: dict = {
        "seed": seed,
        "warmup_trials": WARMUP_TRIALS,
        "measure_trials": MEASURE_TRIALS,
        "binary_sizes": SIZES,
        "k3_size": K3_SIZE,
        "binary": {},
        "kary_k3": {},
    }

    try:
        import xarray  # noqa: F401
        backends = ("ndarray", "xarray")
    except ImportError:
        backends = ("ndarray",)
        results["xarray_available"] = False
    else:
        results["xarray_available"] = True

    for n in SIZES:
        results["binary"][str(n)] = {}
        for backend in backends:
            rng = np.random.default_rng(seed + n)  # seeded per-size
            results["binary"][str(n)][backend] = _bench_size_binary(rng, n, backend)

    for backend in backends:
        rng = np.random.default_rng(seed + 100 + K3_SIZE)
        results["kary_k3"][backend] = _bench_size_kary(rng, K3_SIZE, 3, backend)

    # Decision rule
    decision = "ndarray"  # default
    if "xarray" in backends:
        ratios = []
        for n_str, per_size in results["binary"].items():
            nd = per_size["ndarray"]
            xr_ = per_size["xarray"]
            for op in nd:
                ratios.append(xr_[op]["median_s"] / nd[op]["median_s"])
        if all(r <= 2.0 for r in ratios):
            decision = "xarray"
        results["max_xarray_ratio"] = float(max(ratios)) if ratios else None
    results["decision"] = decision

    # Write JSON (raw)
    json_path = out_dir / "factored-tpm-backend-2026-05-22.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    # Write Markdown summary
    md_path = out_dir / "factored-tpm-backend-2026-05-22.md"
    with open(md_path, "w") as f:
        f.write("# FactoredTPM storage-backend benchmark\n\n")
        f.write(f"- Seed: {seed}\n")
        f.write(f"- Warmup trials: {WARMUP_TRIALS}\n")
        f.write(f"- Measure trials: {MEASURE_TRIALS}\n")
        f.write(f"- xarray available: {results['xarray_available']}\n")
        f.write(f"- **Decision: `{decision}`**\n\n")
        if "max_xarray_ratio" in results and results["max_xarray_ratio"] is not None:
            f.write(f"Max xarray:ndarray ratio across (op, size): "
                    f"`{results['max_xarray_ratio']:.3f}` "
                    f"(rule: xarray default iff <= 2.0).\n\n")
        f.write("## Binary networks\n\n")
        f.write("| n | op | backend | median (s) | p95 (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for n_str, per_size in results["binary"].items():
            for backend in backends:
                for op, stats in per_size[backend].items():
                    f.write(f"| {n_str} | {op} | {backend} | "
                            f"{stats['median_s']:.6e} | {stats['p95_s']:.6e} |\n")
        f.write("\n## k=3 preview\n\n")
        f.write("| n | k | op | backend | median (s) | p95 (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for backend in backends:
            for op, stats in results["kary_k3"][backend].items():
                f.write(f"| {K3_SIZE} | 3 | {op} | {backend} | "
                        f"{stats['median_s']:.6e} | {stats['p95_s']:.6e} |\n")
    print(f"Wrote {md_path}")
    sys.stdout.write(f"\nDecision: {decision}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run the benchmark.**

```bash
uv pip install xarray  # ensure both backends are testable
uv run python benchmarks/factored_tpm_backend.py
```

Expected: `benchmarks/results/factored-tpm-backend-2026-05-22.{md,json}` are created. The final stdout line reports `Decision: ndarray` or `Decision: xarray`.

- [ ] **Step 5.3: Sanity-check the result files.**

```bash
head -20 benchmarks/results/factored-tpm-backend-2026-05-22.md
python3 -c "import json; d = json.load(open('benchmarks/results/factored-tpm-backend-2026-05-22.json')); print('Decision:', d['decision']); print('Max ratio:', d.get('max_xarray_ratio'))"
```

- [ ] **Step 5.4: Commit benchmark + result files.**

```bash
git add benchmarks/factored_tpm_backend.py benchmarks/results/factored-tpm-backend-2026-05-22.md benchmarks/results/factored-tpm-backend-2026-05-22.json
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add FactoredTPM storage-backend benchmark + result artifact

Micro-benchmark measures xarray vs. ndarray on from_joint, to_joint,
condition, and factor access across n in {3, 5, 8, 10} binary plus one
k=3 size. Decision rule: xarray default iff <= 2x ndarray on every
(operation, size) measurement, else ndarray.

The committed result file is the auditable artifact that drives the
default-backend choice in the next commit."
git show --stat HEAD
```

---

## Task 6: Set `_FACTORED_TPM_DEFAULT_BACKEND` from benchmark result

**Goal:** Apply the benchmark's decision to `pyphi/core/tpm/factored.py`. If the result file reports `"ndarray"`, no code change is needed (the placeholder already says ndarray). If it reports `"xarray"`, update the constant.

**Files:**

- Modify (conditionally): `pyphi/core/tpm/factored.py` (the `_FACTORED_TPM_DEFAULT_BACKEND` constant on line ~30)

- [ ] **Step 6.1: Read the decision from the result file.**

```bash
python3 -c "import json; d = json.load(open('benchmarks/results/factored-tpm-backend-2026-05-22.json')); print(d['decision'])"
```

- [ ] **Step 6.2: If decision is `xarray`, edit the constant.**

In `pyphi/core/tpm/factored.py`, change:

```python
_FACTORED_TPM_DEFAULT_BACKEND: Literal["ndarray", "xarray"] = "ndarray"
```

to:

```python
_FACTORED_TPM_DEFAULT_BACKEND: Literal["ndarray", "xarray"] = "xarray"
```

If decision is `ndarray`, this step is a no-op — proceed to Step 6.3.

- [ ] **Step 6.3: Run the FactoredTPM tests to confirm the chosen default works.**

```bash
uv run pytest test/test_factored_tpm.py -v
```

Expected: all 16 tests pass under the new default.

- [ ] **Step 6.4: Commit.**

If the default changed:

```bash
git add pyphi/core/tpm/factored.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Set FactoredTPM default backend per benchmark result

Benchmark in benchmarks/results/factored-tpm-backend-2026-05-22.md
selected xarray (within 2x of ndarray on every measured (op, size))."
git show --stat HEAD
```

If the default did not change (ndarray remained), the commit is empty — skip it and record in the implementation log that step 6 was a no-op confirming the placeholder default. **Do not create an empty commit.**

---

## Task 7: Rename `ExplicitTPM` → `JointTPM`

**Goal:** Mechanical rename across pyphi/ and tests/. `git mv pyphi/core/tpm/explicit.py pyphi/core/tpm/joint.py`. Update all importers and re-exports. Changelog fragment recording the rename as a breaking change.

**Files:**

- Rename: `pyphi/core/tpm/explicit.py` → `pyphi/core/tpm/joint.py`
- Modify: `pyphi/tpm.py` (the legacy `ExplicitTPM` class definition; rename class itself)
- Modify: `pyphi/core/tpm/__init__.py`
- Modify: `pyphi/core/tpm/marginalization.py` (imports `ExplicitTPM`)
- Modify: `pyphi/core/tpm/_factored_backends.py` (imports `ExplicitTPM`? grep to confirm)
- Modify: ~30 importer / consumer sites across `pyphi/` and `test/`
- Create: `changelog.d/rename-explicit-tpm-to-joint-tpm.change.md`

- [ ] **Step 7.1: Enumerate every `ExplicitTPM` reference.**

```bash
grep -rn "ExplicitTPM" pyphi/ test/ --include="*.py" | grep -v ".pyc:" > /tmp/explicit-tpm-refs.txt
wc -l /tmp/explicit-tpm-refs.txt
cat /tmp/explicit-tpm-refs.txt
```

Expected: ~128 lines based on the spec's count. Audit the list — every site needs to be updated.

- [ ] **Step 7.2: git mv the file.**

```bash
git mv pyphi/core/tpm/explicit.py pyphi/core/tpm/joint.py
```

- [ ] **Step 7.3: Rename the class inside the renamed file.**

In `pyphi/core/tpm/joint.py`:

- Change `class ExplicitTPM:` → `class JointTPM:`
- Change `def __init__(self, data: ArrayLike) -> None:` body's `isinstance(data, _legacy_tpm.ExplicitTPM)` → `isinstance(data, _legacy_tpm.JointTPM)` (Task 7 also renames in `pyphi/tpm.py`).
- Change `return ExplicitTPM(...)` returns to `return JointTPM(...)`.
- Change the docstring's `:class:`pyphi.tpm.ExplicitTPM`` reference to `:class:`pyphi.tpm.JointTPM``.

- [ ] **Step 7.4: Rename the legacy class in `pyphi/tpm.py`.**

In `pyphi/tpm.py`:

- Find the line `class ExplicitTPM(...):` and change to `class JointTPM(...):`.
- Update the docstring at the top of the file to mention JointTPM instead of ExplicitTPM.
- Update every `ExplicitTPM(...)` call site within the file to `JointTPM(...)`.

Verification:

```bash
grep -n "ExplicitTPM" pyphi/tpm.py
```

Expected: no matches.

- [ ] **Step 7.5: Update `pyphi/core/tpm/__init__.py`.**

```python
"""Kernel TPM types."""

from .base import TPM as TPM
from .factored import FactoredTPM as FactoredTPM
from .joint import JointTPM as JointTPM
```

- [ ] **Step 7.6: Update `pyphi/core/tpm/marginalization.py`.**

```bash
grep -n "ExplicitTPM" pyphi/core/tpm/marginalization.py
```

For each match, replace `ExplicitTPM` with `JointTPM` and `from .explicit import ExplicitTPM` with `from .joint import JointTPM`.

- [ ] **Step 7.7: Update `pyphi/core/tpm/_factored_backends.py`.**

```bash
grep -n "ExplicitTPM" pyphi/core/tpm/_factored_backends.py
```

Replace any matches. (Likely none — but verify.)

- [ ] **Step 7.8: Update `pyphi/substrate.py`.**

Replace:

```python
from .tpm import ExplicitTPM
```

with:

```python
from .tpm import JointTPM
```

Replace every `ExplicitTPM` reference in the file (constructor body, properties, type annotations) with `JointTPM`. Run `grep -n ExplicitTPM pyphi/substrate.py` to confirm none remain.

- [ ] **Step 7.9: Update `pyphi/__init__.py`.**

If `pyphi/__init__.py` re-exports `ExplicitTPM`, replace with `JointTPM`. Also add `FactoredTPM` re-export.

```bash
grep -n "ExplicitTPM\|FactoredTPM" pyphi/__init__.py
```

Edit accordingly so the top-level namespace is:

```python
from .core.tpm import FactoredTPM as FactoredTPM
from .core.tpm import JointTPM as JointTPM
# (no ExplicitTPM re-export)
```

- [ ] **Step 7.10: Update remaining pyphi/ importers.**

```bash
grep -rln "ExplicitTPM" pyphi/ --include="*.py"
```

For each match:

- Replace `from pyphi.tpm import ExplicitTPM` → `from pyphi.tpm import JointTPM`
- Replace `from .tpm import ExplicitTPM` → `from .tpm import JointTPM`
- Replace `from pyphi.core.tpm import ExplicitTPM` → `from pyphi.core.tpm import JointTPM`
- Replace `from pyphi.core.tpm.explicit import ExplicitTPM` → `from pyphi.core.tpm.joint import JointTPM`
- Replace every `ExplicitTPM` usage (constructor call, isinstance check, type hint) with `JointTPM`.

Track each file you modify. Run the grep again after each batch to confirm progress.

- [ ] **Step 7.11: Update test/ importers.**

```bash
grep -rln "ExplicitTPM" test/ --include="*.py"
```

Same procedure as Step 7.10 for every test file.

- [ ] **Step 7.12: Final sweep.**

```bash
grep -rn "ExplicitTPM" pyphi/ test/ --include="*.py" | grep -v ".pyc:" | head -20
```

Expected: zero matches.

- [ ] **Step 7.13: Create the changelog fragment.**

Create `changelog.d/rename-explicit-tpm-to-joint-tpm.change.md`:

```markdown
Renamed ``ExplicitTPM`` to ``JointTPM`` for symmetry with ``FactoredTPM``
(per-node-factored representation). The class lives at ``pyphi.JointTPM``
and ``pyphi.tpm.JointTPM``; the kernel port lives at
``pyphi.core.tpm.JointTPM``. No backward-compatibility alias.
```

- [ ] **Step 7.14: Run pyright + ruff across the touched files.**

Since this is a large mechanical rename, run pyright over the whole package:

```bash
uv run pyright pyphi
uv run ruff check pyphi test
```

Expected: 0 errors / 1 baseline warning (the pre-existing `reportUnsupportedDunderAll` in `pyphi/__init__.py`). Ruff clean.

- [ ] **Step 7.15: Run the fast lane to confirm no regressions.**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: same passing count as the baseline (1313+ passed). Any failure is from a missed rename site — find it via the error traceback and update.

- [ ] **Step 7.16: Commit.**

```bash
git add pyphi/ test/ changelog.d/rename-explicit-tpm-to-joint-tpm.change.md
git diff --cached --stat   # large — verify only intended files
git -c commit.gpgsign=false commit -m "Rename ExplicitTPM to JointTPM

The class lives at pyphi.JointTPM, pyphi.tpm.JointTPM, and
pyphi.core.tpm.JointTPM (file renamed from explicit.py to joint.py via
git mv). Paired with FactoredTPM (per-node factored representation),
the joint-vs-factored vocabulary describes the storage shape directly.

No backward-compatibility alias per the project's no-shim policy on
unpushed 2.0 work."
git show --stat HEAD
```

If `git diff --cached --stat` shows files outside `pyphi/`, `test/`, or `changelog.d/` — investigate before committing.

---

## Task 8: K-ary property tests

**Goal:** Add Hypothesis property tests exercising k ∈ {2, 3, 4, 5} math against FactoredTPM. These are the new-territory tests; user surface stays binary in P12a but the math is alphabet-generic, and these tests prove it.

**Files:**

- Create: `test/test_factored_tpm_kary.py`

- [ ] **Step 8.1: Write the k-ary property test module.**

Create `test/test_factored_tpm_kary.py`:

```python
"""Hypothesis property tests for FactoredTPM with k-ary alphabets.

User-facing P12a is binary-only; these tests exercise the alphabet-
generic internals against k in {2, 3, 4, 5}. They are the foundation
that P12b's user-facing multi-valued work builds on.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM

# --- strategies ---

ALPHABET_SIZES = st.integers(min_value=2, max_value=5)


def _draw_alphabets(n: int) -> st.SearchStrategy[tuple[int, ...]]:
    return st.tuples(*([ALPHABET_SIZES] * n))


@st.composite
def _factored_strategy(draw: st.DrawFn, max_nodes: int = 4) -> FactoredTPM:
    n = draw(st.integers(min_value=2, max_value=max_nodes))
    alphabet_sizes = draw(_draw_alphabets(n))
    factors = []
    for i in range(n):
        shape = alphabet_sizes + (alphabet_sizes[i],)
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
        raw = rng.uniform(size=shape)
        normalized = raw / raw.sum(axis=-1, keepdims=True)
        factors.append(normalized)
    return FactoredTPM(factors=factors, alphabet_sizes=alphabet_sizes)


# --- properties ---

FAST_LANE = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

SLOW_LANE = settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@FAST_LANE
@given(_factored_strategy())
def test_kary_to_joint_from_joint_round_trip(factored: FactoredTPM) -> None:
    """to_joint then reconstruct factors via from_joint should yield equal factors."""
    joint = factored.to_joint()
    # joint shape: alphabet_sizes + (n, max_alpha). For from_joint, pass
    # alphabet_sizes explicitly so the explicit-alphabet branch is used.
    n = factored.n_nodes
    a = factored.alphabet_sizes
    # Build the explicit-alphabet form per from_joint's contract
    # (alphabet_sizes + (n, a_i)). For uniform alphabet this is exactly
    # `joint`. For heterogeneous, pad each row to max_alpha then slice.
    if len(set(a)) == 1:
        reconstructed = FactoredTPM.from_joint(joint, alphabet_sizes=a)
    else:
        # heterogeneous: from_joint needs the per-row slot to be a_i wide
        # joint has max_alpha; slice each per-row to its actual a_i
        explicit = np.zeros(a + (n, max(a)))
        for i in range(n):
            explicit[..., i, : a[i]] = factored.factor(i)
        reconstructed = FactoredTPM.from_joint(explicit, alphabet_sizes=a)
    for i in range(n):
        np.testing.assert_allclose(
            reconstructed.factor(i), factored.factor(i), atol=1e-10
        )


@FAST_LANE
@given(_factored_strategy(max_nodes=3))
def test_kary_condition_commutes_with_reconstruction(factored: FactoredTPM) -> None:
    """condition(fixed).to_joint() agrees with the equivalent slice on the joint."""
    n = factored.n_nodes
    if n < 2:
        return
    fixed = {0: 0}  # condition on node 0 = state 0
    cond_factored = factored.condition(fixed)
    joint = factored.to_joint()
    # Slice the joint at node 0 = 0
    sliced = joint[0]  # shape: alphabet_sizes[1:] + (n, max_alpha)
    reconstructed = cond_factored.to_joint()
    # cond_factored has the conditioned dim collapsed to size 1 — squeeze
    # for comparison via the JointTPM-side ndarray squeeze (not the
    # Protocol)
    reconstructed_squeezed = reconstructed.squeeze(axis=0)
    np.testing.assert_allclose(reconstructed_squeezed, sliced, atol=1e-10)


@FAST_LANE
@given(_factored_strategy(max_nodes=3))
def test_kary_factors_each_sum_to_one(factored: FactoredTPM) -> None:
    """Validation invariant — sums must be exactly 1 along the last dim."""
    for i in range(factored.n_nodes):
        sums = factored.factor(i).sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)


@pytest.mark.slow
@SLOW_LANE
@given(_factored_strategy())
def test_kary_round_trip_slow(factored: FactoredTPM) -> None:
    """Slow-lane variant of round-trip with max_examples=500."""
    test_kary_to_joint_from_joint_round_trip(factored)


# --- direct k-ary smoke tests (not Hypothesis) ---

def test_k3_explicit_round_trip() -> None:
    """Spot-check k=3 with a hand-crafted FactoredTPM."""
    f0 = np.full((3, 3, 3), 1.0 / 3.0)
    f1 = np.full((3, 3, 3), 1.0 / 3.0)
    factored = FactoredTPM(factors=[f0, f1], alphabet_sizes=(3, 3))
    joint = factored.to_joint()
    reconstructed = FactoredTPM.from_joint(joint, alphabet_sizes=(3, 3))
    for i in range(2):
        np.testing.assert_allclose(reconstructed.factor(i), factored.factor(i), atol=1e-12)


def test_k4_explicit_construction_validates() -> None:
    """k=4 construction passes validation."""
    f0 = np.full((4, 4), 0.25)  # 1-input
    f1 = np.full((4, 4), 0.25)
    # Need to expand to (4, 4, 4) for 2-input
    f0_full = np.broadcast_to(f0[:, np.newaxis, :], (4, 4, 4)).copy()
    f1_full = np.broadcast_to(f1[np.newaxis, :, :], (4, 4, 4)).copy()
    factored = FactoredTPM(
        factors=[f0_full, f1_full], alphabet_sizes=(4, 4)
    )
    assert factored.n_nodes == 2
    assert factored.alphabet_sizes == (4, 4)
```

- [ ] **Step 8.2: Run the new tests.**

```bash
uv run pytest test/test_factored_tpm_kary.py -v
```

Expected: 5 fast tests + 1 slow test (skipped without `--slow`). All passing.

- [ ] **Step 8.3: Run slow lane variant.**

```bash
uv run pytest test/test_factored_tpm_kary.py --slow -v
```

Expected: all 6 pass; the slow one takes ~10-30 seconds.

- [ ] **Step 8.4: Pyright + ruff.**

```bash
uv run pyright test/test_factored_tpm_kary.py
uv run ruff check test/test_factored_tpm_kary.py
```

Expected: 0 errors.

- [ ] **Step 8.5: Commit.**

```bash
git add test/test_factored_tpm_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add Hypothesis property tests for FactoredTPM with k in {2,3,4,5}

Properties: to_joint then from_joint round-trips exactly; conditioning
commutes with joint reconstruction; sum-to-1 invariant holds. Fast lane
50 examples per property; slow lane 500.

User surface in P12a stays binary; these tests are the alphabet-generic
math foundation that P12b's user-facing multi-valued work builds on."
git show --stat HEAD
```

---

## Task 9: Cut over `pyphi/core/repertoire_algebra.py` to consume FactoredTPM

**Goal:** Replace marginalize-from-joint with direct factor reads in the repertoire kernel. The existing `_single_node_cause_repertoire` and `_single_node_effect_repertoire` extract per-node TPMs from `System._typed_tpm` (a JointTPM) by `tpm.marginalize_out`. After this commit, they read from `cs.substrate.factored_tpm.factor(i)` directly.

**Note:** The substrate doesn't yet hold a FactoredTPM — that's Task 11. So this commit introduces a helper `_factored_view_of_substrate(substrate)` that lazily builds a FactoredTPM from the substrate's joint TPM. Task 11 replaces the helper with direct access.

**Files:**

- Modify: `pyphi/core/repertoire_algebra.py:122-186`
- Modify: `test/test_core_tpm.py` (verify the cutover stays equivalent)

- [ ] **Step 9.1: Write a parity test (factored vs. legacy joint-marginalize) for the per-node repertoire computations.**

Append to `test/test_factored_tpm.py`:

```python
def test_factored_per_node_matches_joint_marginalize() -> None:
    """Per-node factor reads from FactoredTPM match the legacy
    joint-then-marginalize computation."""
    from pyphi.core.tpm.joint import JointTPM

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    joint = JointTPM(joint_arr)
    # Per-node ground truth: stack [1-p_on, p_on] for the legacy form
    p_on_per_node = [joint_arr[..., i] for i in range(3)]
    for i in range(3):
        factor_i = factored.factor(i)
        # factor_i shape (2,2,2,2); factor_i[..., 1] is P(node_i = 1)
        np.testing.assert_allclose(factor_i[..., 1], p_on_per_node[i], atol=1e-12)
```

- [ ] **Step 9.2: Run the parity test.**

```bash
uv run pytest test/test_factored_tpm.py::test_factored_per_node_matches_joint_marginalize -v
```

Expected: passes (FactoredTPM is already correct; this just documents the invariant).

- [ ] **Step 9.3: Add the helper and refactor.**

In `pyphi/core/repertoire_algebra.py`, at the top (after imports), add:

```python
def _factored_view_of_substrate(substrate: Any) -> Any:
    """Return a FactoredTPM view of the substrate's TPM.

    Bridging helper while the substrate still stores a JointTPM
    canonically. Replaced by direct ``substrate.factored_tpm`` access in
    the substrate cutover.
    """
    from pyphi.core.tpm.factored import FactoredTPM

    joint = substrate.tpm
    arr = joint.to_array() if hasattr(joint, "to_array") else np.asarray(joint)
    n = arr.shape[-1]
    return FactoredTPM.from_joint(arr, alphabet_sizes=(2,) * n)
```

- [ ] **Step 9.4: Replace `_single_node_cause_repertoire` body to read factors directly.**

The existing implementation at `pyphi/core/repertoire_algebra.py:122-131` reads `cs._index2node[mechanism_node_index].cause_tpm` (which was built by marginalizing `System._typed_tpm`). The factored-direct path doesn't change the math — it changes the source. The simplest cutover keeps the existing path AND adds a parity assertion path under a `_DEBUG_FACTORED_PARITY` flag during the cutover commits. For Task 9 we leave the existing code intact and introduce the helper for downstream use; the actual hot-path swap happens in Task 11 when the substrate stores FactoredTPM directly.

Concrete edit: append the helper from Step 9.3 to the module; no other changes to the body of `_single_node_cause_repertoire` in this commit.

- [ ] **Step 9.5: Run the fast lane to confirm no regressions.**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: same passing count as the baseline.

- [ ] **Step 9.6: Commit.**

```bash
git add pyphi/core/repertoire_algebra.py test/test_factored_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add _factored_view_of_substrate helper for the repertoire kernel

Bridging helper that constructs a FactoredTPM from the substrate's
current JointTPM storage. Used by the upcoming substrate-storage
cutover; the helper goes away in that commit (replaced by direct
substrate.factored_tpm access).

Includes a parity test confirming per-node factor reads match the
legacy joint-marginalize result for binary TPMs."
git show --stat HEAD
```

---

## Task 10: Cut over `pyphi/system.py` and `pyphi/core/tpm/marginalization.py` to dispatch on Protocol

**Goal:** `pyphi/core/tpm/marginalization.py`'s `cause_tpm` and `effect_tpm` dispatch on whether the input is a FactoredTPM (fast path) or JointTPM (existing path). `pyphi/system.py`'s `_typed_tpm` becomes a FactoredTPM view of the substrate.

**Files:**

- Modify: `pyphi/core/tpm/marginalization.py`
- Modify: `pyphi/system.py:155-194`

- [ ] **Step 10.1: Write failing test for marginalization dispatch.**

Create `test/test_marginalization_factored.py`:

```python
"""cause_tpm and effect_tpm dispatch on TPM Protocol."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM
from pyphi.core.tpm.marginalization import cause_tpm
from pyphi.core.tpm.marginalization import effect_tpm


def test_cause_tpm_factored_dispatch_matches_joint() -> None:
    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    state = (0, 1, 0)
    node_indices = (0, 1, 2)

    via_joint = cause_tpm(joint, state, node_indices)
    via_factored = cause_tpm(factored, state, node_indices)
    np.testing.assert_allclose(via_factored.to_array(), via_joint.to_array(), atol=1e-10)


def test_effect_tpm_factored_dispatch_matches_joint() -> None:
    rng = np.random.default_rng(99)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    background = {0: 1}

    via_joint = effect_tpm(joint, background)
    via_factored = effect_tpm(factored, background)
    np.testing.assert_allclose(via_factored.to_array(), via_joint.to_array(), atol=1e-10)
```

- [ ] **Step 10.2: Run failing test.**

```bash
uv run pytest test/test_marginalization_factored.py -v
```

Expected: failures because the current `cause_tpm` doesn't dispatch on FactoredTPM.

- [ ] **Step 10.3: Update marginalization.py to dispatch on Protocol.**

Replace `pyphi/core/tpm/marginalization.py` with:

```python
"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4."""

from __future__ import annotations

from collections.abc import Mapping

from pyphi.tpm import backward_tpm as _legacy_backward_tpm

from .base import TPM
from .factored import FactoredTPM
from .joint import JointTPM


def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> TPM:
    """Backward TPM — IIT 4.0 Eq. 3."""
    if isinstance(tpm, FactoredTPM):
        return _cause_tpm_factored(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        return JointTPM(_legacy_backward_tpm(tpm._inner, state, node_indices))
    # Fall back via to_array for unfamiliar TPM Protocol implementers
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return JointTPM(_legacy_backward_tpm(legacy._inner, state, node_indices))


def effect_tpm(
    tpm: TPM,
    background: Mapping[int, int],
) -> TPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4."""
    return tpm.condition(background)


def _cause_tpm_factored(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Compute the cause TPM in factored form.

    The cause repertoire of node ``i`` given state is proportional to
    ``factor_i(s_t)[state_i]`` as a function of ``s_t``. Normalizing
    per-input-state and renormalizing yields the cause-side conditional
    in factored form.
    """
    import numpy as np

    new_factors = []
    for i in range(factored.n_nodes):
        factor_i = factored.factor(i)
        # factor_i shape: alphabet_sizes + (a_i,)
        # cause = P(s_t | s_{t+1,i} = state_i) ∝ P(s_{t+1,i} = state_i | s_t)
        cause_i = factor_i[..., state[i]]  # shape: alphabet_sizes
        # Reinterpret as a distribution over s_t — but the cause repertoire
        # is over the joint past, so we keep the factor shape and store the
        # likelihood per input state. The full normalization happens at the
        # repertoire-product step.
        # Add explicit alphabet dim back: shape alphabet_sizes + (a_i,)
        # with the relevant slot containing the cause likelihood
        a = factored.alphabet_sizes
        new = np.zeros(a + (a[i],))
        new[..., state[i]] = cause_i
        # Normalize per input state to get a valid conditional
        sums = new.sum(axis=-1, keepdims=True)
        sums = np.where(sums == 0, 1.0, sums)
        new = new / sums
        new_factors.append(new)
    return FactoredTPM(factors=new_factors, alphabet_sizes=factored.alphabet_sizes)
```

- [ ] **Step 10.4: Run the new test.**

```bash
uv run pytest test/test_marginalization_factored.py -v
```

Expected: 2 passed.

If `test_cause_tpm_factored_dispatch_matches_joint` fails due to a math discrepancy, the `_cause_tpm_factored` implementation needs refinement — read the legacy `_legacy_backward_tpm` in `pyphi/tpm.py` and align. Hypothesis: the factored cause TPM should marginally agree with the legacy one for binary inputs. The cause TPM is a NON-trivial inverse operation; verify the math by reading `pyphi/tpm.py` `backward_tpm` carefully before "fixing" any test that fails here.

- [ ] **Step 10.5: Update `pyphi/system.py:155-194` to use the factored view.**

In `pyphi/system.py`, replace the `_typed_tpm` property (around line 155-163) with:

```python
    @cached_property
    def _typed_tpm(self) -> Any:
        """The typed-kernel TPM used by marginalization.

        Returns a FactoredTPM view of the substrate. The substrate's
        canonical storage becomes FactoredTPM in the substrate-cutover
        commit; this property is a forward-compatible accessor.
        """
        from pyphi.core.tpm.factored import FactoredTPM

        legacy_tpm = self.substrate.tpm
        if hasattr(legacy_tpm, "to_array"):
            arr = legacy_tpm.to_array()
        else:
            arr = np.asarray(legacy_tpm)
        n = arr.shape[-1]
        return FactoredTPM.from_joint(arr, alphabet_sizes=(2,) * n)
```

The `cause_tpm` and `effect_tpm` properties on `System` (lines 165-182) already call into `marginalization.cause_tpm` / `effect_tpm` — they keep working because of the Protocol dispatch.

(The existing `typed._inner if hasattr(typed, "_inner") else typed` pattern at lines 172, 182 will switch — FactoredTPM has no `_inner`. Adjust to `typed.to_array() if isinstance(typed, FactoredTPM) else typed` or similar; verify by running the surrounding tests.)

Specifically, change line 172 and line 182 from:

```python
        return typed._inner if hasattr(typed, "_inner") else typed
```

to:

```python
        if hasattr(typed, "to_array"):
            return typed.to_array()
        return typed
```

- [ ] **Step 10.6: Run the impacted test suites.**

```bash
uv run pytest test/test_system.py test/test_core_tpm.py test/test_marginalization_factored.py -x -q
```

Expected: all pass.

- [ ] **Step 10.7: Run the fast lane.**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1313+ passed / 0 failed.

- [ ] **Step 10.8: Pyright + ruff.**

```bash
uv run pyright pyphi/core/tpm/marginalization.py pyphi/system.py test/test_marginalization_factored.py
uv run ruff check pyphi/core/tpm/marginalization.py pyphi/system.py test/test_marginalization_factored.py
```

Expected: 0 errors.

- [ ] **Step 10.9: Commit.**

```bash
git add pyphi/core/tpm/marginalization.py pyphi/system.py test/test_marginalization_factored.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Marginalization dispatches on TPM Protocol; System reads factored view

cause_tpm and effect_tpm in pyphi.core.tpm.marginalization now dispatch
on whether the input is a FactoredTPM (per-factor fast path) or a
JointTPM (legacy joint-shape path). Unknown Protocol implementers fall
back through to_array.

System._typed_tpm returns a FactoredTPM view of the substrate's TPM.
The substrate still stores JointTPM canonically until the next commit;
this property is forward-compatible."
git show --stat HEAD
```

---

## Task 11: Substrate canonical storage cutover

**Goal:** The biggest commit by diff. `Substrate` stores `FactoredTPM` canonically. The existing `tpm=joint_array` constructor auto-converts. New `marginals=` keyword accepted. `substrate.joint_tpm()` method exposes the joint on demand (no cache). `pyphi/validate.py` gains `factored_tpm`. The `_factored_view_of_substrate` helper from Task 9 is deleted. System's `_typed_tpm` reads `substrate.factored_tpm` directly.

**Files:**

- Modify: `pyphi/substrate.py` (canonical storage; new keyword; `joint_tpm` method)
- Modify: `pyphi/validate.py` (add `factored_tpm`; rename existing `tpm` → `joint_tpm`)
- Modify: `pyphi/system.py:155-194` (read `substrate.factored_tpm` directly; drop the from_joint wrapper)
- Modify: `pyphi/core/repertoire_algebra.py` (delete `_factored_view_of_substrate` helper)
- Modify: ~20-30 test sites that read `substrate.tpm` expecting joint shape

- [ ] **Step 11.1: Enumerate test sites that need migration.**

```bash
grep -rn "substrate\.tpm\|\.substrate\.tpm" pyphi/ test/ --include="*.py" | grep -v ".pyc:" > /tmp/substrate-tpm-readers.txt
wc -l /tmp/substrate-tpm-readers.txt
cat /tmp/substrate-tpm-readers.txt
```

For each match: classify as "still works (auto-conversion handles it)", "needs `.joint_tpm()`", or "should modernize to `factored_tpm.factor(i)`". Most JointTPM-shape readers need `.joint_tpm()`.

- [ ] **Step 11.2: Write failing tests for the new substrate surface.**

Create `test/test_substrate_factored.py`:

```python
"""Substrate cutover to FactoredTPM canonical storage."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.substrate import Substrate


def test_substrate_stores_factored_tpm() -> None:
    joint = np.array([[[0.5, 0.5], [0.5, 0.5]],
                      [[0.5, 0.5], [0.5, 0.5]]])
    s = Substrate(tpm=joint)
    assert isinstance(s.factored_tpm, FactoredTPM)
    assert s.factored_tpm.n_nodes == 2


def test_substrate_joint_tpm_method() -> None:
    joint = np.array([[[0.5, 0.5], [0.5, 0.5]],
                      [[0.5, 0.5], [0.5, 0.5]]])
    s = Substrate(tpm=joint)
    materialized = s.joint_tpm()
    np.testing.assert_allclose(materialized[..., 0], joint[..., 0], atol=1e-12)


def test_substrate_marginals_keyword() -> None:
    f0 = np.full((2, 2, 2), 0.5)
    f1 = np.full((2, 2, 2), 0.5)
    s = Substrate(marginals=[f0, f1])
    assert s.factored_tpm.n_nodes == 2


def test_substrate_mutually_exclusive_tpm_marginals() -> None:
    joint = np.zeros((2, 2, 2))
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError, match="tpm.*marginals.*not both"):
        Substrate(tpm=joint, marginals=[f0])  # type: ignore[call-arg]


def test_substrate_from_factored_factory() -> None:
    f0 = np.full((2, 2, 2), 0.5)
    f1 = np.full((2, 2, 2), 0.5)
    factored = FactoredTPM(factors=[f0, f1], alphabet_sizes=(2, 2))
    s = Substrate.from_factored(factored)
    assert s.factored_tpm is factored or s.factored_tpm == factored
```

- [ ] **Step 11.3: Run failing tests.**

```bash
uv run pytest test/test_substrate_factored.py -v
```

Expected: 5 failures.

- [ ] **Step 11.4: Update `pyphi/substrate.py`.**

In `pyphi/substrate.py`, replace the import and `__init__` body. Replace lines 8-30 imports block as needed; then replace the `__init__` (lines 75-98) with:

```python
    def __init__(
        self,
        tpm: NDArray[np.float64] | dict[str, Any] | None = None,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        purview_cache: cache.PurviewCache | None = None,
        *,
        marginals: Sequence[ArrayLike] | None = None,
        alphabet_sizes: Sequence[int] | None = None,
    ) -> None:
        if tpm is not None and marginals is not None:
            raise ValueError("pass tpm= or marginals=, not both")
        if tpm is None and marginals is None:
            raise ValueError("must pass tpm= (joint) or marginals= (factored)")

        if marginals is not None:
            self._factored_tpm = FactoredTPM(
                factors=marginals, alphabet_sizes=alphabet_sizes
            )
        elif isinstance(tpm, dict):
            arr = np.asarray(tpm["_tpm"])
            n = arr.shape[-1]
            sizes = alphabet_sizes or (2,) * n
            self._factored_tpm = FactoredTPM.from_joint(arr, alphabet_sizes=sizes)
        else:
            arr = (
                tpm.to_array() if hasattr(tpm, "to_array") else np.asarray(tpm)
            )
            n = arr.shape[-1]
            sizes = alphabet_sizes or (2,) * n
            self._factored_tpm = FactoredTPM.from_joint(arr, alphabet_sizes=sizes)

        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        validate.substrate(self)
```

Replace the existing `tpm` property (lines 100-106) with:

```python
    @property
    def tpm(self) -> FactoredTPM:
        """The canonical FactoredTPM storage."""
        return self._factored_tpm

    @property
    def factored_tpm(self) -> FactoredTPM:
        """Alias for :attr:`tpm` — explicit per-node-factored access."""
        return self._factored_tpm

    def joint_tpm(self) -> NDArray[np.float64]:
        """Materialize the joint conditional TPM on demand.

        Slow path — recomputes on every call (no cache). Use only at
        boundaries (serialization, legacy fixture comparison, display).
        """
        return self._factored_tpm.to_joint()

    @classmethod
    def from_factored(
        cls,
        factored: FactoredTPM,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        purview_cache: cache.PurviewCache | None = None,
    ) -> "Substrate":
        """Construct a Substrate from an existing FactoredTPM."""
        s = cls.__new__(cls)
        s._factored_tpm = factored
        s._cm, s._cm_hash = s._build_cm(cm)
        s._node_indices = tuple(range(s.size))
        s._node_labels = NodeLabels(node_labels, s._node_indices)
        s.purview_cache = purview_cache or cache.PurviewCache()
        validate.substrate(s)
        return s
```

Adjust imports at the top of the file:

```python
from .core.tpm.factored import FactoredTPM
from .tpm import JointTPM  # for legacy compat in tests still passing JointTPM instances
```

- [ ] **Step 11.5: Update `pyphi/validate.py`.**

In `pyphi/validate.py`, change the `substrate` function (line 78-90):

```python
def substrate(n: object) -> bool:
    """Validate a |Substrate|."""
    from pyphi.core.tpm.factored import FactoredTPM

    factored = n.factored_tpm  # type: ignore[attr-defined]
    if not isinstance(factored, FactoredTPM):
        raise ValueError("substrate.factored_tpm must be a FactoredTPM")
    # FactoredTPM validates itself on construction; no additional checks here
    connectivity_matrix(n.cm)  # type: ignore[attr-defined]
    if n.cm.shape[0] != n.size:  # type: ignore[attr-defined]
        raise ValueError(
            "Connectivity matrix must be NxN, where N is the "
            "number of nodes in the substrate."
        )
    return True
```

Add a `factored_tpm` validator (insert before `substrate`):

```python
def factored_tpm(factored: object) -> bool:
    """Validate a FactoredTPM. The class validates itself on construction;
    this entry point exists for explicit revalidation."""
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi.core.tpm.factored import _validate

    if not isinstance(factored, FactoredTPM):
        raise ValueError(f"Expected FactoredTPM, got {type(factored).__name__}")
    _validate(factored)
    return True


def joint_tpm(arr: object) -> bool:
    """Validate a joint TPM ndarray shape and probability axiom."""
    import numpy as np

    joint = np.asarray(arr)
    if joint.ndim < 2:
        raise ValueError(f"Joint TPM must be at least 2-D; got shape {joint.shape}")
    return True
```

- [ ] **Step 11.6: Update `pyphi/system.py:155-194` for direct factored access.**

In `pyphi/system.py`, replace the `_typed_tpm` property (the one Task 10 modified) with the direct accessor:

```python
    @cached_property
    def _typed_tpm(self) -> Any:
        """The canonical FactoredTPM stored on the substrate."""
        return self.substrate.factored_tpm
```

The `cause_tpm` / `effect_tpm` properties (lines 165-182) keep working because they call into `marginalization` which now dispatches.

Adjust lines 172, 182 (the `_inner` legacy fallbacks) to:

```python
        return typed  # FactoredTPM is the canonical typed surface
```

For `proper_effect_tpm` / `proper_cause_tpm` (lines 184-194) — these call `.squeeze()` on the ndarray result. Update to:

```python
    @cached_property
    def proper_effect_tpm(self) -> Any:
        import numpy as np

        effect = self.effect_tpm
        arr = effect.to_joint() if hasattr(effect, "to_joint") else np.asarray(effect)
        return arr.squeeze()[..., list(self.node_indices)]
```

(same pattern for `proper_cause_tpm`)

- [ ] **Step 11.7: Delete the bridging helper from repertoire_algebra.py.**

In `pyphi/core/repertoire_algebra.py`, remove the `_factored_view_of_substrate` function added in Task 9. (It's no longer needed because the substrate now stores FactoredTPM canonically.)

- [ ] **Step 11.8: Migrate the test sites enumerated in Step 11.1.**

For each entry in `/tmp/substrate-tpm-readers.txt`:

- If the call is `substrate.tpm.to_array()` or `np.asarray(substrate.tpm)`: keep as-is (FactoredTPM has `to_array`).
- If the call is `substrate.tpm.tpm` (legacy `ExplicitTPM.tpm` attribute): change to `substrate.joint_tpm()`.
- If the call is `substrate.tpm.condition_tpm(...)`: keep as-is (FactoredTPM has `condition`).
- If the call is `substrate.tpm` treated as a numpy array directly (e.g., `np.asarray(substrate.tpm)`): also keep as-is.
- If the call inspects shape `(2,)*n + (n,)`: replace with `substrate.joint_tpm()`.

Specific known sites:

- `pyphi/system.py:160` (already handled in Step 11.6)
- `pyphi/macro.py:1109`: `sbs_tpm = convert.state_by_node2state_by_state(substrate.tpm.tpm)` → `substrate.joint_tpm()`
- `pyphi/actual.py:253`: `legacy_tpm = self.substrate.tpm` — likely needs `.joint_tpm()` if downstream code expects joint shape; check inline
- `test/test_actual.py:252-253`: `transition.cause_system.effect_tpm.array_equal(substrate.tpm)` — substrate.tpm is now FactoredTPM; the test should call `substrate.joint_tpm()` if comparing to a joint shape
- `test/test_validate.py:79`: `Substrate(s.substrate.tpm.tpm, ...)` → `Substrate(tpm=s.substrate.joint_tpm(), ...)`
- `test/test_core_tpm.py:62, 63, 80, 81`: comparisons; update to use FactoredTPM directly or `substrate.joint_tpm()`
- `test/golden/compute.py:55`: `np.asarray(substrate.tpm)` — FactoredTPM's `to_array` returns the joint, but the test may already work; verify

- [ ] **Step 11.9: Run the new substrate tests.**

```bash
uv run pytest test/test_substrate_factored.py -v
```

Expected: 5 passed.

- [ ] **Step 11.10: Run the fast lane.**

```bash
uv run pytest test/ -m "not slow" -x -q
```

Expected: 1313+ passed / 0 failed. **If any test fails, investigate via the traceback before "fixing" the test** — failures are likely missed migration sites in Step 11.8.

- [ ] **Step 11.11: Run the goldens.**

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 23+ passed. **Hard stop if any golden drifts past `config.numerics.precision`.**

- [ ] **Step 11.12: Run perf budget tests.**

```bash
uv run pytest test/test_perf_budget.py -v
```

Expected: all pass within `max(3.0, 4×median)` floors. **Hard stop if any regresses.**

- [ ] **Step 11.13: Kick off slow lane in background; continue while it runs.**

```bash
# Background the slow lane via run_in_background=true in the Bash tool
uv run pytest test/ --slow -q > /tmp/p12a-slow-lane.log 2>&1 &
SLOW_LANE_PID=$!
echo "Slow lane PID: $SLOW_LANE_PID"
```

Then use Monitor's `until` loop to poll for completion while Step 11.14 onwards proceeds:

```bash
until ! kill -0 $SLOW_LANE_PID 2>/dev/null; do sleep 60; done
```

(The Monitor tool replaces the loop in actual execution — the Bash sleep is a fallback.)

- [ ] **Step 11.14: Pyright + ruff on touched files.**

```bash
uv run pyright pyphi/substrate.py pyphi/validate.py pyphi/system.py pyphi/core/repertoire_algebra.py pyphi/macro.py pyphi/actual.py
uv run ruff check pyphi/substrate.py pyphi/validate.py pyphi/system.py pyphi/core/repertoire_algebra.py
```

Expected: 0 errors / baseline-only warnings.

- [ ] **Step 11.15: Commit.**

```bash
git add pyphi/substrate.py pyphi/validate.py pyphi/system.py pyphi/core/repertoire_algebra.py pyphi/macro.py pyphi/actual.py test/test_substrate_factored.py test/test_validate.py test/test_actual.py test/test_core_tpm.py test/golden/compute.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Substrate stores FactoredTPM canonically; joint becomes a derivation

Largest commit in P12a. The substrate's tpm field is now a FactoredTPM
(per-node-factored). Existing Substrate(tpm=joint_array, ...) keeps
working via auto-conversion. New keyword marginals=[...] accepted
(mutually exclusive with tpm=). Factory Substrate.from_factored(...)
for explicit construction.

Adds substrate.joint_tpm() method to materialize the joint on demand.
No internal caching — callers that need it repeatedly cache locally.

pyphi.validate gains factored_tpm and joint_tpm validators; substrate
validator updated to inspect FactoredTPM directly.

System._typed_tpm reads substrate.factored_tpm without conversion.
Test sites reading substrate.tpm expecting joint shape are migrated to
substrate.joint_tpm()."
git show --stat HEAD
```

- [ ] **Step 11.16: Wait for slow lane to finish.**

```bash
wait $SLOW_LANE_PID
tail -20 /tmp/p12a-slow-lane.log
```

Expected: 1322+ passed / 0 failed. If anything fails, investigate before proceeding to Task 12.

---

## Task 12: Changelog + scaffold-marker cleanup

**Goal:** Add the user-facing changelog fragment. Remove the `# P12 lifts that assumption` / `# P12 adds alphabet_size` scaffold markers from `pyphi/core/tpm/base.py` and `pyphi/core/unit.py` (already done as part of Task 1; this step is a final sweep + the changelog).

**Files:**

- Create: `changelog.d/factored-tpm.feature.md`
- Verify: `pyphi/core/tpm/base.py`, `pyphi/core/unit.py` (no P12 markers)

- [ ] **Step 12.1: Create the changelog fragment.**

Create `changelog.d/factored-tpm.feature.md`:

```markdown
Substrate now stores a per-node-factored TPM (``FactoredTPM``) canonically;
the joint conditional ``P(s_{t+1} | s_t)`` is derived on demand via
``substrate.joint_tpm()``. Existing ``Substrate(tpm=joint_array, ...)``
keeps working — the constructor auto-converts. A new keyword
``marginals=[per_node_factors]`` and factory ``Substrate.from_factored()``
provide direct factored construction. The TPM Protocol drops ``squeeze``
(it remains on ``JointTPM`` as a numpy-cleanup affordance, where it has
a coherent meaning). ``Unit`` gains an ``alphabet_size: int = 2`` field;
internal math is parameterized by alphabet size. User surface stays
binary in this release; multi-valued substrates are the next milestone.

Internal storage is a swappable backend (default ndarray; xarray
opt-in via ``pip install pyphi[xarray]``). The default backend was
selected by an in-project benchmark; see
``benchmarks/results/factored-tpm-backend-2026-05-22.md``.
```

- [ ] **Step 12.2: Final scaffold-marker sweep.**

```bash
grep -rn "# P12\|# P7\|TODO(P\|TODO(4\.0)" pyphi/core/tpm/ pyphi/core/unit.py
```

Expected: no matches. If any remain, remove them — they describe migration state, not the final code.

- [ ] **Step 12.3: Final acceptance run.**

```bash
uv run pytest test/ -m "not slow" -q
uv run pytest test/test_golden_regression.py -v
uv run pytest test/test_perf_budget.py -v
uv run pyright pyphi
uv run ruff check pyphi test
```

Expected:
- Fast lane: 1313+ passed / 0 failed
- Goldens: 23+ passed
- Perf budget: all within floors
- Pyright: 0 errors / 1 baseline warning
- Ruff: clean

- [ ] **Step 12.4: End-to-end smoke test.**

```bash
uv run python -c "
import numpy as np
import pyphi
joint = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
s = pyphi.Substrate(tpm=joint)
print('substrate.factored_tpm:', s.factored_tpm)
print('factor(0) shape:', s.factored_tpm.factor(0).shape)
print('joint_tpm shape:', s.joint_tpm().shape)
np.testing.assert_allclose(s.joint_tpm()[..., 0], joint[..., 0], atol=1e-12)
print('END-TO-END OK')
"
```

Expected: prints `END-TO-END OK`.

- [ ] **Step 12.5: Commit.**

```bash
git add changelog.d/factored-tpm.feature.md
# If any markers were swept in Step 12.2, add those touched files too
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add changelog fragment for factored TPM foundation

Records the FactoredTPM cutover, the JointTPM rename, the alphabet_size
extension to Unit, and the xarray optional extra. User surface stays
binary; multi-valued is the next milestone (separate spec)."
git show --stat HEAD
```

---

## Final acceptance gates

After Task 12 lands, these must all be green:

| Gate | Command | Expected |
|---|---|---|
| Fast lane | `uv run pytest test/ -m "not slow" -q` | 1313+ passed, 0 failed |
| Slow lane | `uv run pytest test/ --slow -q` | 1322+ passed, 0 failed |
| Goldens | `uv run pytest test/test_golden_regression.py -v` | 23+ passed (byte-identical) |
| Perf budget | `uv run pytest test/test_perf_budget.py -v` | all within `max(3.0, 4×median)` floor |
| Pyright | `uv run pyright pyphi` | 0 errors / 1 baseline warning |
| Ruff check | `uv run ruff check pyphi test` | clean |
| Ruff format | `uv run ruff format --check pyphi test` | clean |
| Benchmark | `benchmarks/results/factored-tpm-backend-2026-05-22.md` exists | exists; decision committed |
| End-to-end smoke | Step 12.4 invocation | prints `END-TO-END OK` |

**Hard stops** (block landing):
- Any golden drifts past `config.numerics.precision`.
- Any perf-budget fixture regresses past floor.
- Any user-facing binary script breaks under the new auto-conversion.

---

## Risk register (from spec §8.5)

| Risk | Mitigation in this plan |
|---|---|
| Cause-repertoire factoring diverges from joint-form math for binary | Task 10's `test_cause_tpm_factored_dispatch_matches_joint` + golden tests in Task 11 |
| Hot path slower than joint for binary | Task 5's benchmark + perf-budget gate in Task 11 |
| xarray backend > 2× overhead | Task 5's decision rule — xarray default only if ≤2× on every measurement |
| Substrate refactor breaks macro/actual paths | Task 11's enumeration in Step 11.1 + targeted updates in Step 11.8; fast-lane gate in Step 11.10 |
| Joint → factored → joint drift exceeds precision | Task 3's round-trip test + Task 8's k-ary property tests with adversarial Hypothesis examples |
| `Substrate.joint_tpm()` callers proliferate | Comment in `Substrate.joint_tpm()` docstring marks it slow-path; spec-compliance reviewer flags any new call site |
| xarray optional dep complicates CI matrix | Task 4 adds one extra CI job; ~10 min added |

---

## Self-review checklist (run after this plan is committed)

This is for the plan-writer, not the implementer:

**1. Spec coverage:**

- §1 Background → Task 0 (this plan's introduction recaps it)
- §2.1 In scope → Tasks 1-12 collectively
- §2.2 Out of scope → explicitly excluded (no tasks for k>2 goldens etc.)
- §2.4 Success criteria → Final acceptance gates section above
- §3 Architecture → Task 2 builds the class; Task 11 cuts substrate over
- §4 Components & file layout → Task 2 (factored.py + backends); Task 4 (xarray backend); Task 7 (rename)
- §5 Validation → Task 2 implements `_validate`; Task 11 wires `validate.factored_tpm`
- §6 Data flow → Task 3 (from_joint/to_joint); Task 10 (marginalization)
- §7 Testing strategy → Task 2 (unit); Task 3 (round-trip); Task 8 (k-ary); Tasks 9-11 (parity); Final gates
- §8 Migration → Tasks 9-11; Step 11.8 enumerates test sites
- §9 Deferred to P12b → recorded in spec; no tasks

**2. Placeholder scan:** Search for TBD, TODO, FIXME, "implement later", "similar to Task N". None expected.

**3. Type consistency:**
- `FactoredTPM(factors=..., alphabet_sizes=..., backend=...)` — consistent across Tasks 2-12
- `Substrate(tpm=..., marginals=..., alphabet_sizes=...)` — consistent across Tasks 11-12
- `cause_tpm(tpm, state, node_indices)` / `effect_tpm(tpm, background)` — consistent across Tasks 10-12
- Validation: `pyphi.exceptions.InvalidTPM` used throughout

---

## Execution handoff

After this plan is committed, two execution options:

**1. Subagent-Driven (recommended)** — matches P11.95d's pattern. Dispatch a fresh subagent per task, two-stage review (spec compliance, then code quality) between each. Sonnet 4.6 for the mechanical TDD steps (Tasks 1, 2, 3, 4, 7, 8, 12); Sonnet 4.6 for algorithmic implementations (Tasks 6, 9, 10) — judgment is settled in the spec, implementation is TDD-driven; Opus 4.7 for the benchmark execution + decision (Task 5) and the largest cutover (Task 11). Reviewer subagents on Opus 4.7 throughout.

**2. Inline Execution** — execute tasks in this session via the executing-plans skill, batch execution with checkpoints between tasks.

Which approach?
