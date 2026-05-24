# P12b — Multi-Valued Units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land native k-ary IIT analysis (cause inversion math, hot-path cutover, user-facing multi-valued substrate API, AC parallel cutover) while preserving byte-identical binary goldens.

**Architecture:** A new `JointDistribution` base class with `JointTPM` and `CausePosterior` as siblings. `Substrate.cause_tpm` returns `CausePosterior`; `Substrate.effect_tpm` returns `FactoredTPM` (asymmetric, math-honest). Binary cause inversion keeps the SBN bridge to legacy `backward_tpm`; k>2 uses a new native per-factor likelihood product. State_space lives canonically on `FactoredTPM` with Substrate delegation. Measure registry gains a declarative `supports_alphabet` callable; EMD raises for k>2 with a Gomez 2021 citation.

**Tech Stack:** Python 3.13, NumPy, Hypothesis (property tests), pytest, pyright, ruff. Pre-commit hooks gate every commit. `git -c commit.gpgsign=false` for signing this session.

---

## Spec reference

Spec at `docs/superpowers/specs/2026-05-24-p12b-multivalued-units-design.md` (committed at `8d0387c9`). Plan task numbering and code blocks reference spec sections (§N.M) where useful.

---

## Branch state baseline & pre-flight

**Worktree:** `/Users/will/projects/pyphi-p12b` (separate from main `/Users/will/projects/pyphi` repo).

**Branch:** `feature/p12b-factored-kary`, head `8d0387c9`.

**Base:** branched from `2.0` at `15720651` (post-P12a + follow-ups).

**Pre-flight check (run before Task 1):**

```bash
cd /Users/will/projects/pyphi-p12b
git log -1 --oneline  # expect: 8d0387c9 Add P12b multi-valued units design spec
git status --short    # should be clean or have only untracked items
uv run pytest test/test_golden_regression.py -v  # 23/23 binary goldens pass
uv run pyright pyphi 2>&1 | tail -3  # 0 errors / 5 baseline warnings
```

If any of these don't match, surface to the user before proceeding.

---

## File responsibilities map

**New files (created during P12b):**

```
pyphi/core/tpm/joint_distribution.py      # JointDistribution base class + ProxyMetaclass
pyphi/core/tpm/cause_posterior.py         # CausePosterior class
test/test_joint_distribution.py           # base class behavior tests
test/test_cause_posterior.py              # CausePosterior-specific tests
test/test_marginalization_kary.py         # k-ary cause/effect math
test/test_substrate_state_space.py        # state_space construction + delegation
test/test_substrate_multivalued.py        # end-to-end k>2 substrate + SIA/AC smoke
test/test_measure_alphabet_support.py     # measure metadata + dispatcher guard
test/data/golden/v1/multivalued_k3_tiny.{json,npz}
test/data/golden/v1/multivalued_2x3x3.{json,npz}
test/data/golden/v1/multivalued_p53_mdm2.{json,npz}   # conditional on reproducibility
docs/superpowers/audits/p12b-cause-shape-audit.md     # Task 1 output
changelog.d/p12b-multivalued.feature.md
```

**Modified files:**

```
pyphi/tpm.py                              # JointTPM: subclasses JointDistribution; sheds shared methods
pyphi/core/tpm/joint.py                   # JointTPM kernel port: refactored
pyphi/core/tpm/factored.py                # state_space field; alphabet_sizes derived; constructor change
pyphi/core/tpm/__init__.py                # export JointDistribution, CausePosterior
pyphi/core/tpm/marginalization.py         # native k-ary path; CausePosterior return type; remove k>2 NotImplementedError
pyphi/core/repertoire_algebra.py          # _single_node_*_repertoire consume new return types
pyphi/substrate.py                        # state_space, alphabet=, joint_tpm unification; alphabet_sizes removed
pyphi/system.py                           # cause_tpm: -> CausePosterior; effect_tpm: -> FactoredTPM; _inner cleanup
pyphi/actual.py                           # TransitionSystem parallel cutover
pyphi/node.py                             # generate_nodes audited for binary hardcodes
pyphi/validate.py                         # extended factored_tpm validation
pyphi/measures/distribution.py            # supports_alphabet metadata; dispatcher guard
pyphi/__init__.py                         # re-export new types
pyphi/jsonify.py                          # CausePosterior + state_space registry
docs/conventions.rst                      # fix pre-existing broken doctest
```

---

## TDD pattern (applies to every code-changing task)

Every code-changing task follows this discipline:

1. Write the failing test first (or document the regression bar if the change is mechanical).
2. Run it to confirm it fails for the right reason.
3. Implement the minimal code to pass.
4. Run the test to confirm it passes.
5. Run a wider check (fast lane or goldens) to confirm no regressions.
6. Commit.

For mechanical refactor steps (renames, signature changes, scaffold-marker cleanup) where TDD is awkward, the pattern is: make the change → run pyright + ruff on touched files → run surrounding tests → commit.

**Every commit must pass pre-commit hooks** (ruff + ruff format + pyright + towncrier-check). Never `--no-verify`. Diagnose failures via `uv run ruff check <file>` / `uv run pyright <file>` directly.

**gpgsign:** use `git -c commit.gpgsign=false commit -m "..."`. If 1Password agent error persists despite the bypass, STOP and surface to controller.

**Staging discipline:** targeted `git add <file>` only. Before commit: `git diff --cached --stat` to confirm only intended files staged. After commit: `git show --stat HEAD` to confirm.

**Doctest scope reminder:** the verification recipe MUST run `uv run pytest` WITHOUT a path argument at every commit boundary. Bare-path invocations (`pytest test/`) skip `pyphi/` source doctests. See `CLAUDE.md` → "Doctest scope — important".

---

## Task 1: Pre-flight audit — cause output shape contract

**Goal:** Determine the actual shape contract of `pyphi.tpm.backward_tpm` and its consumers in `pyphi.core.repertoire_algebra._single_node_cause_repertoire`. This is an INVESTIGATIVE task — no code changes. Output is a markdown note recording the decision that unblocks Tasks 2-6.

**Files:**
- Read: `pyphi/tpm.py` (look at `backward_tpm` function around line 698-732)
- Read: `pyphi/core/repertoire_algebra.py:122-153` (`_single_node_cause_repertoire`)
- Read: `pyphi/system.py:155-200` (`_typed_tpm`, `cause_tpm`, `proper_cause_tpm`)
- Read: `pyphi/core/tpm/marginalization.py:22-66` (current `cause_tpm` / `_cause_tpm_factored`)
- Create: `docs/superpowers/audits/p12b-cause-shape-audit.md`

- [ ] **Step 1.1: Read `backward_tpm` and trace its output shape.**

Open `pyphi/tpm.py` and read the `backward_tpm` function (around line 698-732). Note:

- The input `tpm` shape contract (state-by-node multidimensional? `(2,)*n + (n,)`?).
- The intermediate computations: how is the joint conditioning performed?
- The output shape — does it include a trailing `n` axis (node-axis), a trailing singleton, or pure `(2,)*n`?
- Does it call `[..., list(system_indices)]` at the end? If so, the output retains a trailing axis sized by `len(system_indices)` (which equals `len(node_indices)` passed in).

Write findings into `docs/superpowers/audits/p12b-cause-shape-audit.md` under a section "## Legacy backward_tpm output shape".

- [ ] **Step 1.2: Read `_single_node_cause_repertoire` and identify the shape contract it consumes.**

Open `pyphi/core/repertoire_algebra.py:122-153`. The body is:

```python
@_memoize
def _single_node_cause_repertoire(
    cs: Any, mechanism_node_index: int, purview_set: frozenset[int]
) -> Any:
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm
```

Note:

- The `[..., mechanism_node.state]` indexes the LAST axis with a scalar state. So the input `mechanism_node.cause_tpm` has a final axis that's indexed by node-states (alphabet axis).
- This means the input shape is `(*alphabet_sizes, a_mechanism_node)` for this node — a *per-node cause TPM*, not the substrate-level cause TPM.
- The function returns `tpm.marginalize_out(...).tpm` — calls a method on the typed wrapper, then extracts the ndarray.

Document the per-node shape contract. Note that `mechanism_node.cause_tpm` is built by `generate_nodes` in `pyphi/node.py` — it's a per-node-projection of the substrate's cause TPM.

Add findings to the audit doc under "## Downstream consumer shape contract".

- [ ] **Step 1.3: Read `System.cause_tpm` (and `proper_cause_tpm`) to understand the substrate-level output shape consumers expect.**

Open `pyphi/system.py:155-200`. Read `_typed_tpm`, `cause_tpm`, `proper_cause_tpm`. Document:

- `System.cause_tpm` returns an ndarray (currently unwrapped via `_inner`).
- The shape consumers expect — does any caller of `System.cause_tpm` index into a trailing axis (e.g., `cause_tpm[..., i]` for a node index)?

Search call sites:

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "\.cause_tpm" pyphi/ test/ --include="*.py" | grep -v "_typed_tpm\|cache" | head -30
```

Document which consumers depend on the trailing-axis shape.

- [ ] **Step 1.4: Make the canonical-shape decision and write it.**

Based on Steps 1.1-1.3, decide:

**Option A — Pure `alphabet_sizes` shape (no trailing axis).** If `_legacy_backward_tpm` already produces this and downstream is agnostic to trailing axes. Native k-ary uses this naturally. No canonicalization needed in `CausePosterior.__init__`.

**Option B — `alphabet_sizes + (n_observed_nodes,)` shape (legacy trailing axis).** If `_legacy_backward_tpm` includes a trailing axis sized by the observed-nodes count AND downstream depends on it. Canonicalization happens in `CausePosterior.__init__`: native k-ary path produces pure `alphabet_sizes`; `CausePosterior.__init__` accepts either form and adds a trailing axis if absent.

**Option C — Some other shape contract** the audit uncovers.

Document the choice in the audit doc under "## Canonical shape decision":

```markdown
## Canonical shape decision

After reading [files above]:

- `_legacy_backward_tpm` produces shape `<actual shape>` for binary `n=N` substrates.
- `_single_node_cause_repertoire` depends on `<dependency>` (e.g., "the trailing axis sized by n_nodes", or "is shape-agnostic via .marginalize_out + .tpm").
- Decision: **Option <A|B|C>**. Rationale: ...
- Implications for Tasks 5-6:
  - Native k-ary path output shape: `<shape>`
  - `CausePosterior.__init__` canonicalization: `<yes/no, what>`
  - Downstream consumer changes needed: `<list>`
```

- [ ] **Step 1.5: Commit the audit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add docs/superpowers/audits/p12b-cause-shape-audit.md
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Audit cause-output shape contract for P12b

Reads pyphi/tpm.py::backward_tpm and pyphi/core/repertoire_algebra.py
::_single_node_cause_repertoire to determine the actual shape produced
by the SBN bridge and the shape downstream consumers depend on.
Records the canonical-shape decision that unblocks the type-hierarchy
and native k-ary math tasks."
git show --stat HEAD
```

Expected: commit hash logged; one new file in `docs/superpowers/audits/`.

---

## Task 2: Extract `JointDistribution` base class

**Goal:** Refactor `pyphi/tpm.py:JointTPM` to inherit from a new `JointDistribution` base class. Shareable methods (storage, marginalize_out, array machinery, ProxyMetaclass) move to the base; TPM-specific methods (`condition_tpm`, `subtpm`, `expand_tpm`, `infer_edge`, `infer_cm`, `conditionally_independent`, the forward-TPM independence half of `validate`) stay on `JointTPM`. Binary goldens stay byte-identical.

**Files:**
- Create: `pyphi/core/tpm/joint_distribution.py`
- Modify: `pyphi/tpm.py` (subclass declaration + remove duplicated methods)
- Modify: `pyphi/core/tpm/joint.py` (the kernel port; minimal change)
- Modify: `pyphi/core/tpm/__init__.py` (export JointDistribution)
- Modify: `pyphi/__init__.py` (top-level re-export)
- Test: extend `test/test_core_tpm.py` with JointDistribution conformance tests

- [ ] **Step 2.1: Write failing test for JointDistribution existence and JointTPM subclass relationship.**

Create or extend `test/test_joint_distribution.py`:

```python
"""Tests for the JointDistribution base class and its subclasses."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.core.tpm.joint_distribution import JointDistribution
from pyphi.tpm import JointTPM


def test_joint_distribution_exists() -> None:
    """JointDistribution is importable from pyphi.core.tpm.joint_distribution."""
    assert JointDistribution is not None


def test_jointtpm_isinstance_jointdistribution() -> None:
    """JointTPM is a subclass of JointDistribution."""
    arr = np.full((2, 2, 2), 0.5)
    jtpm = JointTPM(arr)
    assert isinstance(jtpm, JointDistribution)


def test_joint_distribution_marginalize_out_inherited() -> None:
    """marginalize_out is inherited from JointDistribution by JointTPM."""
    arr = np.full((2, 2, 2), 0.125)
    jtpm = JointTPM(arr)
    result = jtpm.marginalize_out([0])
    assert isinstance(result, JointTPM)
```

- [ ] **Step 2.2: Run failing test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_joint_distribution.py -v
```

Expected: `ImportError: cannot import name 'JointDistribution' from 'pyphi.core.tpm.joint_distribution'` (the module doesn't exist yet).

- [ ] **Step 2.3: Create `pyphi/core/tpm/joint_distribution.py`.**

Read `pyphi/tpm.py` lines 1-170 (imports + ProxyMetaclass + class JointTPM header) and lines 170-260 (storage and basic API). Identify which methods are shareable (per spec §3.2):

**Move to JointDistribution:**
- `__init__`, `tpm` property, `__array__`, `__repr__`
- `marginalize_out`
- `to_multidimensional_state_by_node`
- `permute_nodes`
- `is_deterministic`, `is_state_by_state`
- `array_equal`, `print`
- `tpm_indices`
- `validate(check_independence=False)` — probability-axioms-only check
- The `ProxyMetaclass` itself (move to base)

**Stay on JointTPM:**
- `condition_tpm`
- `subtpm`
- `expand_tpm`
- `infer_edge`, `infer_cm`
- `conditionally_independent`
- `validate(check_independence=True)` — JointTPM overrides to default `True`

Create `pyphi/core/tpm/joint_distribution.py`:

```python
"""Joint distribution base class — multidimensional joint probability storage.

This is the shared base for typed wrappers around multidimensional ndarrays
representing joint probability distributions over substrate state spaces.

Concrete subclasses:

- :class:`pyphi.JointTPM` — joint conditional ``P(s_{t+1} | s_t)`` with
  TPM-specific affordances (``condition_tpm``, ``subtpm``, ``expand_tpm``,
  ``infer_cm``, ``infer_edge``).
- :class:`pyphi.CausePosterior` — joint posterior
  ``P(s_t | s_{t+1,M} = mu)`` over past states given an observed mechanism
  state.

The ``ProxyMetaclass`` auto-overloads numpy arithmetic operators on the
underlying array; this lives at the base level because arithmetic on
joint-stored distributions is shareable between concrete subclasses.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from itertools import chain
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import convert  # noqa: F401  (used by some moved methods)
from pyphi import data_structures
from pyphi import exceptions  # noqa: F401  (used by validate)
from pyphi.conf import config  # noqa: F401  (used by validate)
from pyphi.data_structures import FrozenMap  # noqa: F401  (used by callers; re-exported for compat)
from pyphi.utils import all_states  # noqa: F401
from pyphi.utils import np_hash
from pyphi.utils import np_immutable


# --- ProxyMetaclass (verbatim port from pyphi.tpm) ---

class ProxyMetaclass(type):
    """Metaclass that auto-overloads arithmetic operators on the underlying array."""

    def __init__(cls, type_name, bases, dct):
        __closures__ = frozenset(
            {
                # 1-ary
                "__abs__", "__copy__", "__invert__", "__neg__", "__pos__",
                # 2-ary
                "__add__", "__iadd__", "__radd__",
                "__sub__", "__isub__", "__rsub__",
                "__mul__", "__imul__", "__rmul__",
                "__matmul__", "__imatmul__", "__rmatmul__",
                "__truediv__", "__itruediv__", "__rtruediv__",
                "__floordiv__", "__ifloordiv__", "__rfloordiv__",
                "__mod__", "__imod__", "__rmod__",
                "__and__", "__iand__", "__rand__",
                "__lshift__", "__ilshift__", "__irshift__",
                "__rlshift__", "__rrshift__", "__rshift__",
                "__ior__", "__or__", "__ror__",
                "__xor__", "__ixor__", "__rxor__",
                "__eq__", "__ne__", "__ge__", "__gt__", "__lt__", "__le__",
                # (continue per the original file's full list)
            }
        )
        # ... (rest of the metaclass body, verbatim from pyphi/tpm.py) ...
        super().__init__(type_name, bases, dct)


class JointDistribution(data_structures.ArrayLike, metaclass=ProxyMetaclass):
    """Multidimensional joint probability distribution storage.

    Base class for typed wrappers (``JointTPM``, ``CausePosterior``) over
    multidimensional ndarrays representing joint distributions. Provides
    storage, marginalization, array machinery, equality, and the arithmetic
    operator overloads via the metaclass.
    """

    def __init__(
        self,
        tpm: ArrayLike,
        validate: bool = False,
    ) -> None:
        self._tpm = np_immutable(np.asarray(tpm, dtype=np.float64))
        self._hash = np_hash(self._tpm)
        if validate:
            self.validate(check_independence=False)

    @property
    def tpm(self) -> NDArray[np.float64]:
        """The underlying ndarray."""
        return self._tpm

    def __array__(self, dtype=None) -> NDArray[np.float64]:
        if dtype is not None:
            return self._tpm.astype(dtype)
        return self._tpm

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tpm!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointDistribution):
            return NotImplemented
        return self.array_equal(other)

    def __hash__(self) -> int:
        return self._hash

    @property
    def number_of_units(self) -> int:
        """Number of nodes (axes) in the joint distribution."""
        return self._tpm.ndim - 1  # last axis is the alphabet/output dim

    def validate(self, check_independence: bool = False) -> bool:
        """Validate probability axioms.

        ``check_independence`` is overridden in :class:`JointTPM` to default
        ``True`` (forward-TPM-specific conditional-independence check).
        """
        # Probability axioms: non-negative, sum to 1 along the last axis.
        # (Exact body ported from pyphi.tpm.JointTPM.validate's first half.)
        ...
        return True

    def marginalize_out(self, node_indices: Iterable[int]) -> "JointDistribution":
        """Marginalize out nodes from this distribution.

        Returns a new instance of the same concrete subclass.
        """
        # Body ported from pyphi.tpm.JointTPM.marginalize_out.
        ...
        return type(self)(marginalized)

    def to_multidimensional_state_by_node(self) -> NDArray[np.float64]:
        """Convert to multidimensional state-by-node form."""
        # Body ported from pyphi.tpm.JointTPM.to_multidimensional_state_by_node.
        ...

    def is_deterministic(self) -> bool:
        """True iff every entry is 0 or 1."""
        return bool(np.all((self._tpm == 0) | (self._tpm == 1)))

    def is_state_by_state(self) -> bool:
        """True iff the underlying array has shape (S, S)."""
        return self._tpm.ndim == 2 and self._tpm.shape[0] == self._tpm.shape[1]

    def permute_nodes(self, permutation: tuple[int, ...]) -> "JointDistribution":
        """Reorder node-axes per the given permutation."""
        # Body ported.
        ...
        return type(self)(permuted)

    def array_equal(self, other: object) -> bool:
        """Content-based equality."""
        if not isinstance(other, JointDistribution):
            return False
        return np.array_equal(self._tpm, other._tpm)

    def print(self) -> None:
        """Pretty-print the distribution."""
        # Body ported.
        ...

    def tpm_indices(self) -> tuple[int, ...]:
        """Return the node indices implied by the array's shape."""
        return tuple(range(self.number_of_units))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tpm.shape

    @property
    def ndim(self) -> int:
        return self._tpm.ndim

    @property
    def size(self) -> int:
        return self._tpm.size
```

**IMPORTANT:** The above is the structural sketch. The actual implementation needs to port the EXACT method bodies from `pyphi/tpm.py`'s `JointTPM`. Method bodies must be verbatim copies (not re-implementations) to preserve byte-identical golden outputs.

The implementer should: (a) read the actual method bodies in `pyphi/tpm.py`, (b) move them verbatim into `JointDistribution`, (c) verify the imports and dependencies are intact.

- [ ] **Step 2.4: Refactor `pyphi/tpm.py:JointTPM` to subclass `JointDistribution`.**

Change `class JointTPM(data_structures.ArrayLike):` to `class JointTPM(JointDistribution):`. Remove the method bodies that moved to the base; keep only the TPM-specific ones:

```python
# pyphi/tpm.py

from pyphi.core.tpm.joint_distribution import JointDistribution

class JointTPM(JointDistribution):
    """Joint conditional TPM ``P(s_{t+1} | s_t)`` over a substrate's state space.

    Forward transition probability matrix. Stored as a multidimensional
    state-by-node array. Adds conditional-distribution-specific operations
    to :class:`JointDistribution`'s base machinery.
    """

    def validate(self, check_independence: bool = True) -> bool:
        """Validate the TPM. Default ``check_independence=True`` is forward-TPM-specific."""
        super().validate(check_independence=False)
        if check_independence:
            # The IIT conditional-independence assumption check.
            # (Body ported from the existing validate's independence half.)
            ...
        return True

    def conditionally_independent(self) -> bool:
        """Forward-TPM property: rows are independent across nodes."""
        # Body ported.
        ...

    def condition_tpm(self, condition: Mapping[int, int]) -> "JointTPM":
        """Return a TPM conditioned on fixed node states."""
        # Body verbatim from existing JointTPM.condition_tpm.
        ...
        return type(self)(conditioned)

    def subtpm(self, fixed_nodes: tuple[int, ...], state: tuple[int, ...]) -> "JointTPM":
        """Return the TPM for a subset of nodes."""
        # Body verbatim from existing JointTPM.subtpm.
        ...

    def expand_tpm(self) -> "JointTPM":
        """Broadcast a state-by-node TPM so singleton dimensions are expanded."""
        # Body verbatim.
        ...

    def infer_edge(self, a: int, b: int, contexts: tuple[tuple[int, ...], ...]) -> bool:
        """Infer the presence/absence of an edge from node A to node B."""
        # Body verbatim.
        ...

    def infer_cm(self) -> NDArray[np.int_]:
        """Infer the connectivity matrix from this TPM."""
        # Body verbatim.
        ...
```

- [ ] **Step 2.5: Update `pyphi/core/tpm/joint.py` (the kernel port) for the new base relationship.**

The kernel port currently is a thin wrapper. After Step 2.4, it stays as-is — it imports `JointTPM` from `pyphi.tpm`, which now subclasses `JointDistribution`. No change needed beyond verifying the imports still work.

```bash
cd /Users/will/projects/pyphi-p12b
uv run python -c "from pyphi.core.tpm import JointTPM; print(JointTPM)"
uv run python -c "from pyphi.core.tpm.joint_distribution import JointDistribution; print(JointDistribution)"
```

Expected: both succeed.

- [ ] **Step 2.6: Update exports.**

`pyphi/core/tpm/__init__.py`:

```python
"""Kernel TPM types."""

from .base import TPM as TPM
from .factored import FactoredTPM as FactoredTPM
from .joint import JointTPM as JointTPM
from .joint_distribution import JointDistribution as JointDistribution
```

`pyphi/__init__.py` — add the JointDistribution re-export:

```python
# (existing exports preserved)
from .core.tpm import JointDistribution as JointDistribution
```

- [ ] **Step 2.7: Run the new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_joint_distribution.py -v
```

Expected: 3 tests pass.

- [ ] **Step 2.8: Verify goldens byte-identical (THE BAR FOR THIS COMMIT).**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 binary goldens pass byte-identical. If any drift, the JointTPM method body was modified rather than ported verbatim — investigate.

- [ ] **Step 2.9: Pyright + ruff + fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
uv run pytest --tb=short -q 2>&1 | tail -3
```

Expected: pyright 0 errors / 5 baseline warnings; ruff clean; full suite (including doctests) 0 failures.

- [ ] **Step 2.10: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/joint_distribution.py pyphi/tpm.py pyphi/core/tpm/__init__.py pyphi/__init__.py test/test_joint_distribution.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Extract JointDistribution base class; refactor JointTPM as subclass

Shareable methods (storage, marginalize_out, array machinery, ProxyMetaclass,
to_multidimensional_state_by_node, is_deterministic, is_state_by_state,
permute_nodes, array_equal, print, tpm_indices) move to JointDistribution.
TPM-specific methods (condition_tpm, subtpm, expand_tpm, infer_edge,
infer_cm, conditionally_independent, validate's independence-check half)
stay on JointTPM. Method bodies are verbatim ports — binary goldens
byte-identical."
git show --stat HEAD
```

---

## Task 3: Add `CausePosterior(JointDistribution)`

**Goal:** A sibling-of-JointTPM class for the cause-side posterior return type. Inherits the joint-distribution machinery from `JointDistribution`; has no TPM-specific methods. If Task 1's audit chose Option B (canonicalization in `__init__`), implement that here.

**Files:**
- Create: `pyphi/core/tpm/cause_posterior.py`
- Modify: `pyphi/core/tpm/__init__.py` (export)
- Modify: `pyphi/__init__.py` (top-level re-export)
- Test: `test/test_cause_posterior.py`

- [ ] **Step 3.1: Write failing test for CausePosterior.**

Create `test/test_cause_posterior.py`:

```python
"""Tests for CausePosterior — joint posterior over past states."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.joint_distribution import JointDistribution
from pyphi.tpm import JointTPM


def test_cause_posterior_isinstance_jointdistribution() -> None:
    """CausePosterior is a JointDistribution."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert isinstance(cp, JointDistribution)


def test_cause_posterior_not_isinstance_jointtpm() -> None:
    """CausePosterior is a sibling of JointTPM, not a subtype."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert not isinstance(cp, JointTPM)


def test_jointtpm_not_isinstance_cause_posterior() -> None:
    """JointTPM is not a CausePosterior."""
    jtpm = JointTPM(np.full((2, 2, 2), 0.125))
    assert not isinstance(jtpm, CausePosterior)


def test_cause_posterior_marginalize_out_inherited() -> None:
    """marginalize_out is inherited from JointDistribution; returns CausePosterior."""
    cp = CausePosterior(np.full((2, 2, 2), 0.125))
    result = cp.marginalize_out([0])
    assert isinstance(result, CausePosterior)


def test_cause_posterior_repr() -> None:
    """__repr__ tags the type."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert repr(cp).startswith("CausePosterior(")
```

- [ ] **Step 3.2: Run failing test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_cause_posterior.py -v
```

Expected: ImportError (CausePosterior doesn't exist).

- [ ] **Step 3.3: Create `pyphi/core/tpm/cause_posterior.py`.**

If Task 1's audit chose **Option A (pure alphabet_sizes shape)** — the simpler version:

```python
"""Joint posterior over past states given an observed future state.

Returned by ``cause_tpm`` after Bayesian inversion of a forward TPM. Stored
as a multidimensional probability distribution over the joint past-state
space.

Past nodes are generally NOT conditionally independent in this posterior —
observing the future couples the past inputs. Hence this is a sibling of
:class:`JointTPM` (joint conditional) rather than a subtype: both are joint
distributions, neither IS-A the other.
"""

from __future__ import annotations

from numpy.typing import NDArray

import numpy as np

from .joint_distribution import JointDistribution


class CausePosterior(JointDistribution):
    """Joint posterior distribution ``P(s_t | s_{t+1,M} = mu)`` over past states."""

    def __repr__(self) -> str:
        return f"CausePosterior({self._tpm!r})"
```

If Task 1's audit chose **Option B (canonicalize to legacy trailing-axis shape)**:

```python
class CausePosterior(JointDistribution):
    """..."""

    def __init__(
        self,
        tpm,
        validate: bool = False,
    ) -> None:
        # Canonicalize: ensure the trailing axis (sized by the number of
        # observed mechanism nodes) is present. If absent (native k-ary
        # path produces pure alphabet_sizes shape), add it.
        arr = np.asarray(tpm, dtype=np.float64)
        if not _has_trailing_observation_axis(arr):
            arr = arr[..., np.newaxis]
        super().__init__(arr, validate=validate)

    def __repr__(self) -> str:
        return f"CausePosterior({self._tpm!r})"


def _has_trailing_observation_axis(arr: NDArray) -> bool:
    """Heuristic: trailing axis present if ndim > number_of_alphabet_axes.

    Per the audit, callers always wrap the canonical form. The check exists
    to make CausePosterior(arr_without_axis) idempotent.
    """
    # Specific check depends on the audit's findings; placeholder logic.
    return True  # Replace with the audit-driven check.
```

The implementer should pick the correct variant based on the audit document from Task 1.

- [ ] **Step 3.4: Update exports.**

`pyphi/core/tpm/__init__.py`:

```python
"""Kernel TPM types."""

from .base import TPM as TPM
from .cause_posterior import CausePosterior as CausePosterior
from .factored import FactoredTPM as FactoredTPM
from .joint import JointTPM as JointTPM
from .joint_distribution import JointDistribution as JointDistribution
```

`pyphi/__init__.py`:

```python
from .core.tpm import CausePosterior as CausePosterior
```

- [ ] **Step 3.5: Run tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_cause_posterior.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3.6: Pyright + ruff + goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/core/tpm/cause_posterior.py pyphi/core/tpm/__init__.py pyphi/__init__.py test/test_cause_posterior.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/cause_posterior.py test/test_cause_posterior.py 2>&1 | tail -3
uv run pytest test/test_golden_regression.py -v 2>&1 | tail -3
```

Expected: pyright clean; ruff clean; goldens unchanged (CausePosterior isn't consumed yet).

- [ ] **Step 3.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/cause_posterior.py pyphi/core/tpm/__init__.py pyphi/__init__.py test/test_cause_posterior.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add CausePosterior as JointDistribution sibling

Joint posterior over past states; returned by cause_tpm after Bayesian
inversion of a forward TPM. Subclasses JointDistribution alongside JointTPM
(both are joint distributions; neither IS-A the other). No TPM-specific
methods; inherits marginalize_out and array machinery from the base.

Not yet wired to consumers — marginalization dispatch wraps with this type
in the next commit."
git show --stat HEAD
```

---

## Task 4: Update `marginalization.cause_tpm` binary path to return `CausePosterior`

**Goal:** Replace `JointTPM` wrapping with `CausePosterior` wrapping in `_cause_tpm_factored` and the `cause_tpm` dispatcher (binary branch only). The Liskov substitution principle keeps downstream `_single_node_cause_repertoire` working — it calls `.marginalize_out(...).tpm`, and both `JointTPM` and `CausePosterior` inherit `marginalize_out` from `JointDistribution`.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py` (3 wrap sites)
- Test: extend `test/test_marginalization_factored.py` (P12a) with CausePosterior return-type checks

- [ ] **Step 4.1: Write failing test for CausePosterior return type.**

Append to `test/test_marginalization_factored.py`:

```python
def test_cause_tpm_returns_cause_posterior_for_jointtpm_input() -> None:
    """cause_tpm wraps the legacy backward_tpm result in CausePosterior."""
    from pyphi.core.tpm.cause_posterior import CausePosterior
    from pyphi.core.tpm.joint import JointTPM
    from pyphi.core.tpm.marginalization import cause_tpm

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    result = cause_tpm(joint, state=(0, 1, 0), node_indices=(0, 1, 2))
    assert isinstance(result, CausePosterior)


def test_cause_tpm_returns_cause_posterior_for_factored_input() -> None:
    """cause_tpm wraps the SBN-bridge result in CausePosterior."""
    from pyphi.core.tpm.cause_posterior import CausePosterior
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi.core.tpm.marginalization import cause_tpm

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    result = cause_tpm(factored, state=(0, 1, 0), node_indices=(0, 1, 2))
    assert isinstance(result, CausePosterior)
```

- [ ] **Step 4.2: Run failing test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_factored.py::test_cause_tpm_returns_cause_posterior_for_jointtpm_input test/test_marginalization_factored.py::test_cause_tpm_returns_cause_posterior_for_factored_input -v
```

Expected: fail because `cause_tpm` currently returns `JointTPM`.

- [ ] **Step 4.3: Update `pyphi/core/tpm/marginalization.py` wrap sites.**

Read the file. Update three `JointTPM(...)` wrap sites in `cause_tpm` and `_cause_tpm_factored` to `CausePosterior(...)`:

```python
# pyphi/core/tpm/marginalization.py

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM
from pyphi.tpm import backward_tpm as _legacy_backward_tpm

from .base import TPM


def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Backward TPM — IIT 4.0 Eq. 3. Returns a CausePosterior."""
    if isinstance(tpm, FactoredTPM):
        return _cause_tpm_factored(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return CausePosterior(_legacy_backward_tpm(legacy._inner, state, node_indices))


def _cause_tpm_factored(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Binary cause TPM via SBN bridge + legacy backward_tpm.

    Binary-only; raises for k>2 substrates. Native k-ary path is added
    in a subsequent commit and dispatched in cause_tpm above.
    """
    if not all(a == 2 for a in factored.alphabet_sizes):
        raise NotImplementedError(
            "FactoredTPM marginalization requires binary alphabets; "
            f"got alphabet_sizes={factored.alphabet_sizes}. "
            f"Multi-valued substrate analysis is the next milestone."
        )
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    return CausePosterior(_legacy_backward_tpm(sbn, state, node_indices))
```

- [ ] **Step 4.4: Run new tests; verify pass.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_factored.py -v
```

Expected: all tests pass, including the 2 new CausePosterior return-type checks AND the existing P12a tests (which used `isinstance(_, JointTPM)` previously — confirm those still pass via Liskov).

- [ ] **Step 4.5: Verify goldens byte-identical (CRITICAL BAR for this commit).**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. If any drift, the wrap-type change introduced a behavior difference — investigate. The wrap type IS the only change; underlying math is unchanged.

- [ ] **Step 4.6: Full fast lane + doctests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest --tb=short -q 2>&1 | tail -5
```

Expected: 0 failures.

- [ ] **Step 4.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/marginalization.py test/test_marginalization_factored.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "cause_tpm returns CausePosterior; underlying math unchanged

Updates the three JointTPM(...) wrap sites in cause_tpm and
_cause_tpm_factored to wrap in CausePosterior instead. Liskov keeps
downstream consumers working — both classes inherit marginalize_out
from JointDistribution. Binary goldens byte-identical."
git show --stat HEAD
```

---

## Task 5: Implement native k-ary cause path

**Goal:** Add `_cause_tpm_factored_kary` for k>2 substrates. Dispatcher in `cause_tpm` routes binary → existing SBN-bridge, k>2 → new native path. The native path computes the joint posterior via per-factor likelihood product: `cause(s_t) ∝ ∏ factor_i(s_t)[state[i]]`.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py`
- Test: `test/test_marginalization_kary.py` (new, but stubs added; full tests in Task 6)

- [ ] **Step 5.1: Write a smoke test that fails the k>2 NotImplementedError.**

Create `test/test_marginalization_kary.py`:

```python
"""K-ary cause/effect marginalization tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import cause_tpm


def _k3_two_node_uniform() -> FactoredTPM:
    """A small k=3 2-node uniform FactoredTPM."""
    f = np.full((3, 3, 3), 1.0 / 3.0)
    return FactoredTPM(factors=[f, f.copy()], alphabet_sizes=(3, 3))


def test_cause_tpm_k3_returns_cause_posterior() -> None:
    """k=3 cause_tpm returns a CausePosterior (no NotImplementedError)."""
    factored = _k3_two_node_uniform()
    result = cause_tpm(factored, state=(0, 0), node_indices=(0, 1))
    assert isinstance(result, CausePosterior)


def test_cause_tpm_k3_sums_to_one() -> None:
    """k=3 cause posterior is a valid probability distribution."""
    factored = _k3_two_node_uniform()
    result = cause_tpm(factored, state=(0, 0), node_indices=(0, 1))
    assert np.isclose(np.asarray(result.tpm).sum(), 1.0, atol=1e-12)
```

- [ ] **Step 5.2: Run failing tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
```

Expected: `NotImplementedError: FactoredTPM marginalization requires binary alphabets; got alphabet_sizes=(3, 3). Multi-valued substrate analysis is the next milestone.`

- [ ] **Step 5.3: Add `_cause_tpm_factored_kary` and dispatch logic.**

Update `pyphi/core/tpm/marginalization.py`. Rename existing `_cause_tpm_factored` to `_cause_tpm_factored_binary`. Add `_cause_tpm_factored_kary`. Dispatch in `cause_tpm` first hits the FactoredTPM branch and routes by alphabet:

```python
def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Backward TPM — IIT 4.0 Eq. 3. Returns a CausePosterior."""
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            return _cause_tpm_factored_binary(tpm, state, node_indices)
        return _cause_tpm_factored_kary(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return CausePosterior(_legacy_backward_tpm(legacy._inner, state, node_indices))


def _cause_tpm_factored_binary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Binary cause TPM via SBN bridge + legacy backward_tpm.

    Preserves byte-identical goldens by delegating to the legacy code path
    that produced them.
    """
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    return CausePosterior(_legacy_backward_tpm(sbn, state, node_indices))


def _cause_tpm_factored_kary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Native k-ary cause TPM via per-factor likelihood product.

    Computes P(s_t | s_{M, t+1} = state) over the joint past state space.
    The likelihood at each past joint state is the product of per-mechanism-node
    factor lookups; normalized over s_t.
    """
    from pyphi import exceptions

    alphabet_sizes = factored.alphabet_sizes
    likelihood = np.ones(alphabet_sizes, dtype=np.float64)
    for i in node_indices:
        likelihood = likelihood * factored.factor(i)[..., state[i]]
    total = likelihood.sum()
    if total <= 0:
        raise exceptions.StateUnreachableBackwardsError(state)
    posterior = likelihood / total
    return CausePosterior(posterior)
```

If Task 1's audit chose Option B (canonicalize in CausePosterior.__init__), the native path produces pure `alphabet_sizes` shape and CausePosterior wraps appropriately. If Option A (pure alphabet_sizes everywhere), this is the final form.

- [ ] **Step 5.4: Run new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5.5: Verify binary goldens byte-identical.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23. The binary path now routes through `_cause_tpm_factored_binary` (renamed from `_cause_tpm_factored`); body unchanged.

- [ ] **Step 5.6: Pyright + ruff + fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py 2>&1 | tail -3
uv run pytest test/ -m "not slow" -q 2>&1 | tail -3
```

Expected: all clean.

- [ ] **Step 5.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add native k-ary cause path; dispatcher routes binary vs k-ary

_cause_tpm_factored_binary preserves the byte-identical SBN bridge path
for binary substrates. _cause_tpm_factored_kary is the new native path:
per-factor likelihood product over node_indices, normalized over the
joint past-state space. Dispatcher in cause_tpm routes by alphabet."
git show --stat HEAD
```

---

## Task 6: Add k-ary property tests (Hypothesis)

**Goal:** Comprehensive property tests for the k-ary cause math. Most importantly: the binary-equivalence property — `_cause_tpm_factored_kary(binary)` agrees with `_cause_tpm_factored_binary(binary)` within `atol=1e-10`. If this fails, the two paths' math is inconsistent and the eventual unification (ROADMAP follow-on) becomes harder.

**Files:**
- Modify: `test/test_marginalization_kary.py` (add Hypothesis property tests)

- [ ] **Step 6.1: Add the property tests.**

Append to `test/test_marginalization_kary.py`:

```python
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pyphi.core.tpm.marginalization import (
    _cause_tpm_factored_binary,
    _cause_tpm_factored_kary,
)


# --- strategies ---

ALPHABET_SIZES = st.integers(min_value=2, max_value=4)


@st.composite
def _factored_strategy(draw: st.DrawFn, max_nodes: int = 3) -> FactoredTPM:
    n = draw(st.integers(min_value=2, max_value=max_nodes))
    alphabet_sizes = tuple(draw(ALPHABET_SIZES) for _ in range(n))
    factors = []
    for i in range(n):
        shape = alphabet_sizes + (alphabet_sizes[i],)
        rng_seed = draw(st.integers(min_value=0, max_value=10_000))
        rng = np.random.default_rng(rng_seed)
        raw = rng.uniform(size=shape)
        normalized = raw / raw.sum(axis=-1, keepdims=True)
        factors.append(normalized)
    return FactoredTPM(factors=factors, alphabet_sizes=alphabet_sizes)


FAST_LANE = settings(max_examples=50, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])
SLOW_LANE = settings(max_examples=500, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])


# --- properties ---

@FAST_LANE
@given(_factored_strategy())
def test_cause_kary_sums_to_one(factored: FactoredTPM) -> None:
    """Posterior is a valid probability distribution."""
    state = tuple(0 for _ in range(factored.n_nodes))
    indices = tuple(range(factored.n_nodes))
    result = cause_tpm(factored, state, indices)
    np.testing.assert_allclose(np.asarray(result.tpm).sum(), 1.0, atol=1e-12)


@FAST_LANE
@given(_factored_strategy())
def test_cause_kary_non_negative(factored: FactoredTPM) -> None:
    """Posterior entries are non-negative."""
    state = tuple(0 for _ in range(factored.n_nodes))
    indices = tuple(range(factored.n_nodes))
    result = cause_tpm(factored, state, indices)
    assert (np.asarray(result.tpm) >= 0).all()


@FAST_LANE
@given(_factored_strategy())
def test_cause_kary_binary_equivalence(factored: FactoredTPM) -> None:
    """For binary substrates, native k-ary path agrees with SBN bridge within 1e-10.

    This is the load-bearing math-correctness property. If this fails, the
    two paths are inconsistent and the eventual unification (ROADMAP follow-on)
    has to reconcile them.
    """
    if not all(a == 2 for a in factored.alphabet_sizes):
        return  # Only meaningful for binary inputs
    state = tuple(0 for _ in range(factored.n_nodes))
    indices = tuple(range(factored.n_nodes))
    via_binary = _cause_tpm_factored_binary(factored, state, indices)
    via_kary = _cause_tpm_factored_kary(factored, state, indices)
    np.testing.assert_allclose(
        np.asarray(via_binary.tpm),
        np.asarray(via_kary.tpm),
        atol=1e-10,
        err_msg="Binary SBN bridge and native k-ary cause math disagree.",
    )


@pytest.mark.slow
@SLOW_LANE
@given(_factored_strategy())
def test_cause_kary_binary_equivalence_slow(factored: FactoredTPM) -> None:
    """Slow-lane variant of the binary-equivalence property with max_examples=500."""
    test_cause_kary_binary_equivalence(factored)


# --- direct k-ary spot checks ---

def test_cause_k3_explicit_uniform() -> None:
    """For a uniform k=3 substrate, the cause posterior is uniform over past states."""
    factored = _k3_two_node_uniform()
    result = cause_tpm(factored, state=(0, 0), node_indices=(0, 1))
    arr = np.asarray(result.tpm)
    expected = np.full(arr.shape, 1.0 / arr.size)
    np.testing.assert_allclose(arr, expected, atol=1e-12)


def test_cause_kary_unreachable_state_raises() -> None:
    """A state with zero likelihood under all past states raises."""
    from pyphi import exceptions

    # Build a deterministic factor: factor_i(s_t) = [1, 0, 0] always.
    f = np.zeros((3, 3, 3))
    f[..., 0] = 1.0
    factored = FactoredTPM(factors=[f, f.copy()], alphabet_sizes=(3, 3))
    # Observing state 2 (which factors deterministically rule out) is unreachable.
    with pytest.raises(exceptions.StateUnreachableBackwardsError):
        cause_tpm(factored, state=(2, 2), node_indices=(0, 1))
```

- [ ] **Step 6.2: Run the new property tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
```

Expected: all fast-lane tests pass. The slow one is skipped without `--slow`.

If `test_cause_kary_binary_equivalence` fails, Hypothesis prints the minimal failing example. Read the failure carefully — this is the load-bearing math test. If the disagreement is structural, the native path implementation has a bug. Per saved memory `feedback_dont_give_up_on_architectural_refactors`, diagnose before adjusting tolerance.

- [ ] **Step 6.3: Run the slow lane variant.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py --slow -v
```

Expected: slow test passes (max_examples=500); takes 10-60s.

- [ ] **Step 6.4: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright test/test_marginalization_kary.py 2>&1 | tail -3
uv run ruff check test/test_marginalization_kary.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 6.5: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add Hypothesis property tests for k-ary cause math

Three fast-lane properties: posterior sums to 1, entries are non-negative,
binary-equivalence with the SBN bridge within atol=1e-10. Slow-lane
variant of the binary-equivalence property with max_examples=500.

Plus two direct k=3 spot checks: uniform-substrate uniform posterior,
unreachable-state raises StateUnreachableBackwardsError."
git show --stat HEAD
```

---

## Task 7: Remove the defensive k>2 NotImplementedError from `_effect_tpm_factored`

**Goal:** P12a's review-fix added a defensive `NotImplementedError` for k>2 in `_effect_tpm_factored`. Effect is alphabet-generic via `FactoredTPM.condition` — the guard is now unnecessary. Remove it; add a smoke test confirming k-ary effect_tpm works.

**Files:**
- Modify: `pyphi/core/tpm/marginalization.py:_effect_tpm_factored`
- Test: append to `test/test_marginalization_kary.py`

- [ ] **Step 7.1: Write the failing test for k-ary effect_tpm.**

Append to `test/test_marginalization_kary.py`:

```python
def test_effect_tpm_k3_returns_factored_tpm() -> None:
    """k=3 effect_tpm returns a FactoredTPM via condition (no NotImplementedError)."""
    from pyphi.core.tpm.marginalization import effect_tpm

    factored = _k3_two_node_uniform()
    result = effect_tpm(factored, background={0: 0})
    assert isinstance(result, FactoredTPM)
    # Conditioning collapses node 0's dim to a singleton.
    assert result.factor(0).shape[0] == 1
```

- [ ] **Step 7.2: Run failing test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py::test_effect_tpm_k3_returns_factored_tpm -v
```

Expected: `NotImplementedError` from the defensive P12a guard.

- [ ] **Step 7.3: Remove the guard.**

In `pyphi/core/tpm/marginalization.py`, find `_effect_tpm_factored`:

```python
def _effect_tpm_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> FactoredTPM:
    # Remove this guard:
    if not all(a == 2 for a in factored.alphabet_sizes):
        raise NotImplementedError(
            "FactoredTPM marginalization requires binary alphabets; ..."
        )
    return factored.condition(background)
```

Replace with the unconditional version:

```python
def _effect_tpm_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> FactoredTPM:
    """Effect TPM = factored.condition(background). Alphabet-generic via P12a."""
    return factored.condition(background)
```

Or — since the body is now trivially equivalent to `effect_tpm`'s direct delegation — consider inlining and removing the helper entirely. Read the `effect_tpm` dispatch first to see if it makes sense to drop the helper.

If `effect_tpm`'s current body is:

```python
def effect_tpm(tpm: TPM, background: Mapping[int, int]) -> TPM:
    if isinstance(tpm, FactoredTPM):
        return _effect_tpm_factored(tpm, background)
    return tpm.condition(background)
```

simplify to:

```python
def effect_tpm(tpm: TPM, background: Mapping[int, int]) -> TPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4.

    Delegates to ``tpm.condition(background)``. Alphabet-generic for both
    FactoredTPM (k-ary via P12a's condition impl) and JointTPM (k-ary via
    the legacy condition_tpm).
    """
    return tpm.condition(background)
```

and delete `_effect_tpm_factored`.

- [ ] **Step 7.4: Run tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_marginalization_kary.py -v
uv run pytest test/test_marginalization_factored.py -v
```

Expected: all pass. The k=3 effect_tpm test passes; the P12a effect_tpm tests still pass (unchanged behavior for binary).

- [ ] **Step 7.5: Verify goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical.

- [ ] **Step 7.6: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/marginalization.py test/test_marginalization_kary.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Remove defensive k>2 NotImplementedError from effect_tpm path

P12a added the guard as a defensive measure when the binary-only cause
SBN-bridge couldn't handle k>2. effect_tpm is alphabet-generic via
FactoredTPM.condition (P12a), so the guard was never load-bearing for
effect. Now that the cause path handles k>2 natively (prior commit),
the effect guard comes out — multi-valued effect TPMs Just Work."
git show --stat HEAD
```

---

## Task 8: Hot-path cutover — `System.cause_tpm` / `System.effect_tpm` return precise types

**Goal:** `System.cause_tpm` returns `CausePosterior`; `System.effect_tpm` returns `FactoredTPM`. The `_inner` unwrap pattern (added in P12a's review-fix as `# type: ignore[attr-defined]`) is retired from `pyphi/system.py`.

**Files:**
- Modify: `pyphi/system.py:155-200`
- Test: extend `test/test_system.py` with return-type assertions

- [ ] **Step 8.1: Read `pyphi/system.py:155-200`.**

The current code (post-P12a review-fix) has:

```python
@cached_property
def cause_tpm(self) -> Any:
    typed = _marginalize_cause(self._typed_tpm, self.state, self.node_indices)
    return typed._inner if hasattr(typed, "_inner") else typed  # type: ignore[attr-defined]
```

and similar for `effect_tpm`.

Note: `_typed_tpm` was simplified in P12a Task 11 to return `self.substrate.factored_tpm` directly. So `_marginalize_cause(factored_tpm, ...)` now returns `CausePosterior` (per Task 4) for the binary path and either `CausePosterior` (k-ary) per Task 5.

- [ ] **Step 8.2: Write failing test.**

Append to `test/test_system.py` (or create `test/test_system_return_types.py`):

```python
def test_system_cause_tpm_returns_cause_posterior() -> None:
    """System.cause_tpm returns CausePosterior, not unwrapped ndarray."""
    from pyphi.core.tpm.cause_posterior import CausePosterior
    from pyphi import examples

    sub = examples.basic_substrate()
    sys = pyphi.System(sub, state=(0, 0, 0))
    assert isinstance(sys.cause_tpm, CausePosterior)


def test_system_effect_tpm_returns_factored_tpm() -> None:
    """System.effect_tpm returns FactoredTPM after the cutover."""
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi import examples

    sub = examples.basic_substrate()
    sys = pyphi.System(sub, state=(0, 0, 0))
    assert isinstance(sys.effect_tpm, FactoredTPM)
```

- [ ] **Step 8.3: Run failing tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_system.py::test_system_cause_tpm_returns_cause_posterior test/test_system.py::test_system_effect_tpm_returns_factored_tpm -v
```

Expected: fail because currently they return unwrapped values.

- [ ] **Step 8.4: Update `cause_tpm` and `effect_tpm` properties.**

In `pyphi/system.py` (around lines 165-185, post-P12a):

```python
@cached_property
def cause_tpm(self) -> "CausePosterior":
    """The cause TPM (joint posterior over past states), conditioned on self.state."""
    return _marginalize_cause(self._typed_tpm, self.state, self.node_indices)


@cached_property
def effect_tpm(self) -> "FactoredTPM":
    """The effect TPM (forward conditional, factored), conditioned on external state."""
    external_state = utils.state_of(self.external_indices, self.state)
    background = dict(zip(self.external_indices, external_state, strict=False))
    return _marginalize_effect(self._typed_tpm, background)
```

Note: imports may need adjustment. Add:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyphi.core.tpm.cause_posterior import CausePosterior
    from pyphi.core.tpm.factored import FactoredTPM
```

The `# type: ignore[attr-defined]` comments come out — pyright resolves to precise types now.

- [ ] **Step 8.5: Update downstream callers within `pyphi/system.py`.**

`proper_cause_tpm` / `proper_effect_tpm` previously consumed unwrapped ndarrays. Update:

```python
@cached_property
def proper_effect_tpm(self) -> Any:
    """The effect TPM projected onto the System's node indices."""
    effect = self.effect_tpm
    # effect is a FactoredTPM; materialize the joint for the legacy projection
    arr = effect.to_joint() if hasattr(effect, "to_joint") else np.asarray(effect)
    return arr.squeeze()[..., list(self.node_indices)]


@cached_property
def proper_cause_tpm(self) -> Any:
    """The cause TPM projected onto the System's node indices."""
    cause = self.cause_tpm
    # cause is a CausePosterior; the underlying tpm is the joint posterior ndarray
    arr = np.asarray(cause.tpm) if hasattr(cause, "tpm") else np.asarray(cause)
    return arr.squeeze()[..., list(self.node_indices)] if arr.ndim > len(self.node_indices) else arr.squeeze()
```

(The exact form for `proper_cause_tpm` depends on Task 1's audit — if the cause posterior has a trailing axis, the slicing logic mirrors `proper_effect_tpm`; if not, the slicing is conditional.)

- [ ] **Step 8.6: Grep audit for `_inner` access.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "_inner if hasattr" pyphi/ --include="*.py" | grep -v ".pyc:"
```

Expected: only `actual.py:261` remains (handled in Task 9). No other production-code matches. If there are other surprises, investigate before commit.

- [ ] **Step 8.7: Run tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_system.py -v
```

Expected: existing tests pass; new return-type tests pass.

- [ ] **Step 8.8: Verify goldens byte-identical.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. If any drift, the type change introduced a subtle behavior difference — most likely in the `proper_*_tpm` properties' shape handling. Investigate.

- [ ] **Step 8.9: Full fast lane + perf budget.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -q 2>&1 | tail -3
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -10
```

Expected: 0 failures; perf budget within floors.

- [ ] **Step 8.10: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/system.py test/test_system.py 2>&1 | tail -3
uv run ruff check pyphi/system.py test/test_system.py 2>&1 | tail -3
```

Expected: clean. No more `# type: ignore[attr-defined]` for the `_inner` pattern in `system.py`.

- [ ] **Step 8.11: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/system.py test/test_system.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Hot-path cutover: System.cause_tpm -> CausePosterior, effect_tpm -> FactoredTPM

System.cause_tpm returns the precise CausePosterior typed object;
System.effect_tpm returns FactoredTPM. The _inner unwrap pattern with
# type: ignore[attr-defined] that P12a's review-fix added comes out of
system.py. proper_cause_tpm and proper_effect_tpm adapted to the new
return types.

Pyright is precise; goldens byte-identical."
git show --stat HEAD
```

---

## Task 9: Hot-path cutover — `TransitionSystem.cause_tpm` / `effect_tpm`

**Goal:** Mirror Task 8's changes in `pyphi/actual.py:TransitionSystem`. Remove the `# type: ignore[attr-defined]` at line 261. k>2 AC works via the same dispatcher logic.

**Files:**
- Modify: `pyphi/actual.py` (TransitionSystem.cause_tpm / effect_tpm)
- Test: extend `test/test_actual.py` with return-type assertions

- [ ] **Step 9.1: Read `pyphi/actual.py:230-280`.**

The current code mirrors System's pattern with the `_inner` unwrap and `# type: ignore[attr-defined]`. The line numbers may have shifted; grep:

```bash
cd /Users/will/projects/pyphi-p12b
grep -n "_inner if hasattr" pyphi/actual.py
```

- [ ] **Step 9.2: Write failing test.**

Append to `test/test_actual.py`:

```python
def test_transition_system_cause_tpm_returns_cause_posterior() -> None:
    """TransitionSystem.cause_tpm returns CausePosterior."""
    from pyphi.core.tpm.cause_posterior import CausePosterior
    from pyphi import examples

    sub = examples.basic_substrate()
    transition = pyphi.actual.Transition(
        substrate=sub,
        before_state=(0, 0, 0),
        after_state=(1, 0, 0),
        cause_indices=(0, 1, 2),
        effect_indices=(0, 1, 2),
    )
    assert isinstance(transition.cause_system.cause_tpm, CausePosterior)


def test_transition_system_effect_tpm_returns_factored_tpm() -> None:
    """TransitionSystem.effect_tpm returns FactoredTPM."""
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi import examples

    sub = examples.basic_substrate()
    transition = pyphi.actual.Transition(
        substrate=sub,
        before_state=(0, 0, 0),
        after_state=(1, 0, 0),
        cause_indices=(0, 1, 2),
        effect_indices=(0, 1, 2),
    )
    assert isinstance(transition.effect_system.effect_tpm, FactoredTPM)
```

- [ ] **Step 9.3: Update `TransitionSystem.cause_tpm` / `effect_tpm`.**

Mirror Task 8's changes. Replace each `return result._inner if hasattr(result, "_inner") else result  # type: ignore[attr-defined]` with the direct return of the marginalization result. Update type hints accordingly.

- [ ] **Step 9.4: Grep audit.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "_inner if hasattr" pyphi/ --include="*.py" | grep -v ".pyc:"
```

Expected: zero matches in production code.

- [ ] **Step 9.5: Run AC tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_actual.py -v 2>&1 | tail -10
```

Expected: all pass; new return-type assertions pass.

- [ ] **Step 9.6: Verify goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical.

- [ ] **Step 9.7: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/actual.py test/test_actual.py 2>&1 | tail -3
uv run ruff check pyphi/actual.py 2>&1 | tail -3
```

Expected: clean. No `_inner if hasattr` survivor.

- [ ] **Step 9.8: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/actual.py test/test_actual.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "TransitionSystem cutover parallels System: precise typed returns

TransitionSystem.cause_tpm returns CausePosterior; effect_tpm returns
FactoredTPM. The _inner unwrap pattern with # type: ignore[attr-defined]
that P12a's follow-up added at actual.py:261 is retired.

AC paths now share the same dispatch flow as the IIT 4.0 paths; k>2
AC analysis works for free via the cause_tpm dispatcher."
git show --stat HEAD
```

---

## Task 10: Downstream cutover — `core/repertoire_algebra._single_node_*_repertoire`

**Goal:** Update the per-node repertoire computations to consume `CausePosterior` (via inherited `marginalize_out`) and `FactoredTPM` (via `condition_factor`). This is the most-load-bearing commit for binary-goldens preservation — the per-node math touches everything.

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py:122-186`
- Modify: `pyphi/node.py` (audit and adapt `generate_nodes` for the new return types)

- [ ] **Step 10.1: Read `pyphi/core/repertoire_algebra.py:122-186` and `pyphi/node.py:generate_nodes`.**

Understand the current shape contract — see Task 1's audit.

`_single_node_cause_repertoire` (line 122-131):

```python
@_memoize
def _single_node_cause_repertoire(cs, mechanism_node_index, purview_set):
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm
```

Note: `mechanism_node.cause_tpm` is per-NODE (built by `generate_nodes`), not per-substrate. After the System cutover (Task 8), the per-node `cause_tpm` is now a slice of the substrate's `CausePosterior` — produced by `generate_nodes` in `pyphi/node.py`.

- [ ] **Step 10.2: Audit `generate_nodes` and per-node TPM access.**

Read `pyphi/node.py:generate_nodes`. Find where per-node `cause_tpm` and `effect_tpm` are computed for each Node. Verify:

- Per-node `cause_tpm` slicing logic still works when the substrate-level `cause_tpm` is a `CausePosterior` (which is a `JointDistribution`, has `.tpm` and `.marginalize_out`).
- Per-node `effect_tpm` slicing logic still works when the substrate-level `effect_tpm` is a `FactoredTPM` (which has `.factor(i)`, `.condition_factor`, but NOT `.marginalize_out`).

This is where the cutover gets nuanced — `generate_nodes` may need to bridge the asymmetric return types. Decide:

- For cause-side: per-node `cause_tpm` can be a slice of the substrate's `CausePosterior` (still a JointDistribution; downstream `marginalize_out` works).
- For effect-side: per-node `effect_tpm` could be (a) the substrate's `FactoredTPM` itself (every node accesses its own factor via `factor(i)`), or (b) a per-node slice/projection.

Document the decision in a comment in `generate_nodes`.

- [ ] **Step 10.3: Update `_single_node_cause_repertoire` if needed.**

If `mechanism_node.cause_tpm` is now a `CausePosterior`, the body still works (it calls `.marginalize_out(...).tpm`). Verify:

```python
@_memoize
def _single_node_cause_repertoire(cs, mechanism_node_index, purview_set):
    mechanism_node = cs._index2node[mechanism_node_index]
    # cause_tpm is now a CausePosterior (joint posterior over past states)
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]  # NOTE: may need adjustment if cause has no trailing axis
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm
```

The `[..., mechanism_node.state]` indexing may need to change if Task 1's audit determined the cause posterior has no trailing observation axis. If pure `alphabet_sizes` shape, the per-node slicing logic happens upstream (in `generate_nodes`) and `_single_node_cause_repertoire` operates on the per-node posterior directly.

- [ ] **Step 10.4: Update `_single_node_effect_repertoire` to consume FactoredTPM.**

```python
@_memoize
def _single_node_effect_repertoire(cs, condition, purview_node_index, direction):
    purview_node = cs._index2node[purview_node_index]
    # purview_node.effect_tpm is now a FactoredTPM
    if direction == Direction.CAUSE:
        factored_tpm = purview_node.cause_tpm
    else:
        factored_tpm = purview_node.effect_tpm
    # If per-node *_tpm is a FactoredTPM, get the per-node factor
    if hasattr(factored_tpm, "condition_factor"):
        # FactoredTPM path
        factor = factored_tpm.condition_factor(purview_node_index, condition)
    else:
        # Legacy path (per-node JointTPM); use the existing logic
        factor = factored_tpm.condition_tpm(condition)
        # ... rest of legacy logic ...
    # Marginalize over non-mechanism inputs
    nonmechanism_inputs = purview_node.inputs - set(condition)
    for axis in sorted(nonmechanism_inputs, reverse=True):
        factor = factor.sum(axis=axis) if isinstance(factor, np.ndarray) else factor.marginalize_out([axis]).tpm
    return factor.reshape(
        repertoire_shape(cs.substrate.node_indices, (purview_node_index,))
    )
```

The exact form depends on the audit from Step 10.2. The implementer should pick the simplest consistent form.

- [ ] **Step 10.5: Run the full fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -x -q 2>&1 | tail -10
```

Expected: 0 failures. If failures, investigate via the traceback — likely a shape mismatch in the per-node logic.

- [ ] **Step 10.6: Verify goldens BYTE-IDENTICAL (the most critical bar).**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. If ANY drift, the per-node cutover changed numerical behavior — investigate before commit. Per saved memory `feedback_dont_give_up_on_architectural_refactors`, diagnose before reverting.

- [ ] **Step 10.7: Verify perf budget.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -10
```

Expected: all 5 binary fixtures within `max(3.0, 4×median)` floor.

- [ ] **Step 10.8: Kick off slow lane in background; continue.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ --slow -q > /tmp/p12b-task10-slow.log 2>&1 &
echo "Slow lane PID: $!"
```

(In the implementation, use the Bash tool's `run_in_background=true` flag.)

- [ ] **Step 10.9: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/core/repertoire_algebra.py pyphi/node.py 2>&1 | tail -3
uv run ruff check pyphi/core/repertoire_algebra.py pyphi/node.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 10.10: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/repertoire_algebra.py pyphi/node.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Downstream cutover: repertoire_algebra consumes CausePosterior + FactoredTPM

_single_node_cause_repertoire consumes CausePosterior via inherited
marginalize_out. _single_node_effect_repertoire consumes FactoredTPM via
condition_factor and numpy summation. generate_nodes audited and adapted
for the new asymmetric per-node TPM return types.

Most-load-bearing commit for binary-goldens preservation; verified
byte-identical."
git show --stat HEAD
```

- [ ] **Step 10.11: Wait for slow lane to finish.**

```bash
cd /Users/will/projects/pyphi-p12b
wait $!
tail -20 /tmp/p12b-task10-slow.log
```

Expected: slow lane 0 failures.

---

## Task 11: `FactoredTPM` state_space field

**Goal:** Refactor `FactoredTPM` constructor: `state_space=` keyword; `alphabet_sizes=` parameter removed; alphabet sizes derived from state_space. Add `_normalize_state_space` helper.

**Files:**
- Modify: `pyphi/core/tpm/factored.py`
- Test: `test/test_factored_tpm.py` (extend with state_space tests)

- [ ] **Step 11.1: Write failing tests for state_space.**

Append to `test/test_factored_tpm.py`:

```python
def test_factored_tpm_default_state_space_is_integer_labels() -> None:
    """When state_space is omitted, integer labels 0..k-1 are inferred per node."""
    f = _two_node_factored()  # binary, no state_space
    assert f.state_space == ((0, 1), (0, 1))


def test_factored_tpm_uniform_state_space_string_labels() -> None:
    """A flat tuple of strings is parsed as uniform across all nodes."""
    f0 = np.full((3, 3, 3), 1.0 / 3.0)
    f = FactoredTPM(factors=[f0, f0.copy()], state_space=("LOW", "MID", "HIGH"))
    assert f.state_space == (("LOW", "MID", "HIGH"), ("LOW", "MID", "HIGH"))
    assert f.alphabet_sizes == (3, 3)


def test_factored_tpm_per_node_state_space() -> None:
    """A tuple of tuples is parsed as per-node labels."""
    f_binary = np.full((2, 3, 2), 0.5)
    f_ternary = np.full((2, 3, 3), 1.0 / 3.0)
    f = FactoredTPM(
        factors=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    assert f.state_space == (("OFF", "ON"), ("LOW", "MID", "HIGH"))
    assert f.alphabet_sizes == (2, 3)


def test_factored_tpm_state_space_length_mismatch_raises() -> None:
    """state_space length must match factor count."""
    from pyphi.exceptions import InvalidTPM
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(
            factors=[f0, f0.copy()],
            state_space=(("OFF", "ON"), ("LOW", "HIGH"), ("EXTRA",)),  # 3 entries, 2 factors
        )


def test_factored_tpm_state_space_label_alphabet_mismatch_raises() -> None:
    """state_space[i] length must match factor[i]'s last-dim size."""
    from pyphi.exceptions import InvalidTPM
    f_binary = np.full((2, 2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(
            factors=[f_binary, f_binary.copy()],
            state_space=(("L", "M", "H"), ("L", "M", "H")),  # 3 labels, but factor has alphabet 2
        )


def test_factored_tpm_alphabet_sizes_not_constructor_kwarg() -> None:
    """alphabet_sizes is no longer a constructor parameter."""
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(TypeError, match="alphabet_sizes"):
        FactoredTPM(factors=[f0, f0.copy()], alphabet_sizes=(2, 2))  # type: ignore[call-arg]
```

- [ ] **Step 11.2: Run failing tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_factored_tpm.py -k "state_space or alphabet" -v
```

Expected: most fail because the constructor signature hasn't been updated yet.

- [ ] **Step 11.3: Refactor `FactoredTPM.__init__` and add `_normalize_state_space`.**

In `pyphi/core/tpm/factored.py`:

```python
from collections.abc import Sequence
from typing import Any

StateSpace = (
    Sequence[Any]                       # flat: uniform labels
    | Sequence[Sequence[Any]]           # per-node
    | None
)


def _normalize_state_space(
    raw: StateSpace,
    factors: Sequence[NDArray[np.float64]],
) -> tuple[tuple[Any, ...], ...]:
    """Normalize state_space input to per-node tuple-of-tuples.

    Rules per spec §4.3.
    """
    n_factors = len(factors)
    if raw is None:
        # Default: integer labels 0..k-1 per node; k from factor shapes
        return tuple(tuple(range(int(f.shape[-1]))) for f in factors)

    raw_tuple = tuple(raw)
    if len(raw_tuple) == 0:
        raise ValueError("state_space cannot be empty")

    # Detect uniform vs per-node: if EVERY element is a non-string sequence,
    # treat as per-node; otherwise treat as uniform-flat.
    def _is_sequence_not_string(x: Any) -> bool:
        return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))

    if all(_is_sequence_not_string(elem) for elem in raw_tuple):
        # Per-node form
        if len(raw_tuple) != n_factors:
            from pyphi import exceptions
            raise exceptions.InvalidTPM(
                f"state_space has {len(raw_tuple)} per-node entries; "
                f"factors imply {n_factors} nodes"
            )
        return tuple(tuple(elem) for elem in raw_tuple)
    else:
        # Uniform form: same labels for all nodes
        uniform = tuple(raw_tuple)
        return tuple(uniform for _ in range(n_factors))


class FactoredTPM:
    __slots__ = ("_backend", "_state_space")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        state_space: StateSpace = None,
        backend: Literal["ndarray", "xarray"] | None = None,
    ) -> None:
        factor_arrays = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        self._state_space = _normalize_state_space(state_space, factor_arrays)
        # alphabet_sizes derived from state_space lengths
        alphabet_sizes = tuple(len(s) for s in self._state_space)
        self._backend = _make_default_backend(factor_arrays, alphabet_sizes, backend)
        _validate(self)

    @property
    def state_space(self) -> tuple[tuple[Any, ...], ...]:
        return self._state_space

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        return tuple(len(s) for s in self._state_space)

    # ... existing methods unchanged ...
```

Update `_validate` to check state_space invariants (per spec §6.1):

```python
def _validate(factored: FactoredTPM) -> None:
    a = factored.alphabet_sizes
    if len(factored._state_space) != factored.n_nodes:
        raise exceptions.InvalidTPM(
            f"state_space has {len(factored._state_space)} per-node entries; "
            f"factors imply {factored.n_nodes} nodes"
        )
    for i, labels in enumerate(factored._state_space):
        if len(labels) != a[i]:
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has {len(labels)} labels but factor[{i}] "
                f"has alphabet size {a[i]}"
            )
        if len(set(labels)) != len(labels):
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has duplicate labels: {labels}"
            )
    # Existing P12a invariants:
    if any(size < 2 for size in a):
        raise exceptions.InvalidTPM(
            f"alphabet_sizes must all be >= 2; got {a}"
        )
    # ... rest of P12a validation (factor shape, sum-to-1) ...
```

- [ ] **Step 11.4: Update `FactoredTPM.from_joint` to accept `state_space=` instead of `alphabet_sizes=`.**

```python
@classmethod
def from_joint(
    cls,
    joint: ArrayLike,
    /,
    state_space: StateSpace = None,
) -> "FactoredTPM":
    """Convert a joint conditional TPM into the factored form.

    state_space: see _normalize_state_space. If None, integer labels are
    inferred from the joint's shape (binary unless explicitly k-ary).
    """
    # Body adapts: replace alphabet_sizes=... with state_space=... and pass through
    # to FactoredTPM(...)
    ...
```

Existing usages of `from_joint(arr, alphabet_sizes=...)` (in `Substrate.__init__`, etc.) need updating in subsequent tasks.

- [ ] **Step 11.5: Run new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_factored_tpm.py -v
```

Expected: all pass, including the 6 new state_space tests.

- [ ] **Step 11.6: Check downstream callers of `FactoredTPM(... alphabet_sizes=...)`.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "alphabet_sizes=" pyphi/ --include="*.py" | grep -v ".pyc:"
```

Expected: some callers remain (Substrate.__init__, etc.). Document each one — it'll be updated in Task 12 (Substrate constructor change).

For now, the failing callers will raise TypeError when invoked. The fast lane will surface these.

- [ ] **Step 11.7: Run fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -x -q 2>&1 | tail -10
```

Expected: failures in tests that construct substrates via the legacy `alphabet_sizes=` path. List them — they get fixed in Task 12.

If too many failures and the next task isn't ready, consider keeping `alphabet_sizes=` as a deprecated kwarg in `FactoredTPM.__init__` temporarily, with a warning. But the spec calls for clean removal, and Task 12 should be ready immediately. Prefer the clean break.

- [ ] **Step 11.8: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/core/tpm/factored.py test/test_factored_tpm.py 2>&1 | tail -3
uv run ruff check pyphi/core/tpm/factored.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 11.9: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/core/tpm/factored.py test/test_factored_tpm.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "FactoredTPM: state_space replaces alphabet_sizes; alphabet_sizes derived

FactoredTPM constructor signature changes: state_space= keyword accepts
None (default integer labels), a flat sequence (uniform across nodes),
or a sequence-of-sequences (per-node). alphabet_sizes parameter is
removed (redundant per the P12b design); alphabet_sizes becomes a
@property derived from state_space lengths.

_normalize_state_space helper handles the parsing. _validate extended to
check state_space length matches factor count and labels match per-node
alphabet sizes; duplicate labels rejected.

Substrate constructor migrations follow in the next commit; some test
sites that pass alphabet_sizes= will fail until then."
git show --stat HEAD
```

---

## Task 12: `Substrate` constructor: state_space and alphabet kwargs

**Goal:** Substrate constructor accepts `state_space=` and `alphabet=`; drops `alphabet_sizes=`. State-coercion helper for label-state lookup. `Substrate.state_space` delegated property.

**Files:**
- Modify: `pyphi/substrate.py` (constructor, properties, state coercion)
- Modify: `pyphi/system.py` (state coercion at System construction)
- Test: `test/test_substrate_state_space.py` (new)

- [ ] **Step 12.1: Write failing tests for the Substrate constructor changes.**

Create `test/test_substrate_state_space.py`:

```python
"""Tests for Substrate's state_space and alphabet= keyword parameters."""

from __future__ import annotations

import numpy as np
import pytest

import pyphi
from pyphi.core.tpm.factored import FactoredTPM


def _k3_marginals():
    f = np.full((3, 3, 3), 1.0 / 3.0)
    return [f, f.copy()]


def test_substrate_default_state_space_binary() -> None:
    """Binary substrate via legacy tpm= has default integer state_space."""
    joint = np.full((2, 2, 2), 0.5)
    sub = pyphi.Substrate(tpm=joint)
    assert sub.state_space == ((0, 1), (0, 1))


def test_substrate_state_space_uniform_string_labels() -> None:
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("LOW", "MID", "HIGH"))
    assert sub.state_space == (("LOW", "MID", "HIGH"), ("LOW", "MID", "HIGH"))


def test_substrate_state_space_per_node_heterogeneous() -> None:
    f_binary = np.full((2, 3, 2), 0.5)
    f_ternary = np.full((2, 3, 3), 1.0 / 3.0)
    sub = pyphi.Substrate(
        marginals=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    assert sub.state_space == (("OFF", "ON"), ("LOW", "MID", "HIGH"))


def test_substrate_alphabet_shortcut() -> None:
    """alphabet=k is sugar for state_space=tuple(range(k))."""
    sub = pyphi.Substrate(marginals=_k3_marginals(), alphabet=3)
    assert sub.state_space == ((0, 1, 2), (0, 1, 2))


def test_substrate_alphabet_and_state_space_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="alphabet.*state_space.*not both"):
        pyphi.Substrate(
            marginals=_k3_marginals(),
            alphabet=3,
            state_space=("L", "M", "H"),
        )


def test_substrate_alphabet_sizes_kwarg_removed() -> None:
    """alphabet_sizes is no longer a Substrate constructor kwarg."""
    joint = np.full((2, 2, 2), 0.5)
    with pytest.raises(TypeError, match="alphabet_sizes"):
        pyphi.Substrate(tpm=joint, alphabet_sizes=(2, 2))  # type: ignore[call-arg]


def test_substrate_state_space_delegates_to_factored_tpm() -> None:
    """Substrate.state_space is a delegated property."""
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("L", "M", "H"))
    assert sub.state_space is sub.factored_tpm.state_space or sub.state_space == sub.factored_tpm.state_space


def test_system_state_as_labels_resolves_to_indices() -> None:
    """System(state=labels) resolves to System(state=int_indices) via state_space."""
    sub = pyphi.Substrate(marginals=_k3_marginals(), state_space=("L", "M", "H"))
    sys_via_labels = pyphi.System(sub, state=("L", "M", "H"))
    sys_via_indices = pyphi.System(sub, state=(0, 1, 2))
    assert sys_via_labels.state == sys_via_indices.state
```

- [ ] **Step 12.2: Run failing tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_substrate_state_space.py -v
```

Expected: most fail because the constructor doesn't have the new params yet.

- [ ] **Step 12.3: Update `Substrate.__init__` per spec §4.1.**

In `pyphi/substrate.py`:

```python
from pyphi.core.tpm.factored import StateSpace  # type alias


class Substrate:
    def __init__(
        self,
        tpm: NDArray[np.float64] | JointTPM | dict[str, Any] | None = None,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        purview_cache: cache.PurviewCache | None = None,
        *,
        marginals: Sequence[ArrayLike] | None = None,
        state_space: StateSpace = None,
        alphabet: int | None = None,
    ) -> None:
        # Mutual exclusion
        if tpm is not None and marginals is not None:
            raise ValueError("pass tpm= or marginals=, not both")
        if alphabet is not None and state_space is not None:
            raise ValueError("pass alphabet= or state_space=, not both")
        if tpm is None and marginals is None:
            raise ValueError("must pass tpm= (joint) or marginals= (factored)")
        if isinstance(tpm, FactoredTPM):
            raise ValueError(
                "pass FactoredTPM via marginals= or Substrate.from_factored(...)"
            )

        # Translate alphabet= shortcut to state_space=
        if alphabet is not None:
            if alphabet < 2:
                raise ValueError(f"alphabet must be >= 2; got {alphabet}")
            state_space = tuple(range(alphabet))  # uniform integer labels

        # Construct FactoredTPM
        if marginals is not None:
            self._factored_tpm = FactoredTPM(factors=marginals, state_space=state_space)
        elif isinstance(tpm, dict):
            arr = np.asarray(tpm["_tpm"])
            self._factored_tpm = FactoredTPM.from_joint(arr, state_space=state_space)
        else:
            arr = tpm.to_array() if hasattr(tpm, "to_array") else np.asarray(tpm)
            self._factored_tpm = FactoredTPM.from_joint(arr, state_space=state_space)

        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()
        validate.substrate(self)

    @property
    def state_space(self) -> tuple[tuple[Any, ...], ...]:
        """Delegated from FactoredTPM."""
        return self._factored_tpm.state_space
```

- [ ] **Step 12.4: Add state-coercion helper.**

In `pyphi/substrate.py` (or `pyphi/utils.py`), add:

```python
def _coerce_state_to_indices(
    state: tuple[Any, ...],
    state_space: tuple[tuple[Any, ...], ...],
) -> tuple[int, ...]:
    """Convert a state-tuple to integer indices via state_space lookup.

    If state[i] is already an integer in range(alphabet_sizes[i]), pass through.
    If state[i] is a label in state_space[i], return its index.
    Otherwise raise ValueError.
    """
    if len(state) != len(state_space):
        raise ValueError(
            f"state length {len(state)} != state_space length {len(state_space)}"
        )
    indices = []
    for i, (s, labels) in enumerate(zip(state, state_space, strict=True)):
        # Try label lookup first if labels contain s
        if s in labels:
            indices.append(labels.index(s))
        elif isinstance(s, int) and 0 <= s < len(labels):
            indices.append(s)
        else:
            raise ValueError(
                f"state[{i}] = {s!r} not in state_space[{i}] = {labels!r} "
                f"and not a valid index for alphabet size {len(labels)}"
            )
    return tuple(indices)
```

- [ ] **Step 12.5: Update `System.__init__` to coerce state at construction.**

In `pyphi/system.py:__post_init__`:

```python
def __post_init__(self) -> None:
    substrate = self.substrate
    # Coerce state-as-labels to integer indices via state_space
    coerced_state = _coerce_state_to_indices(self.state, substrate.state_space)
    object.__setattr__(self, "state", coerced_state)
    validate.state_length(self.state, substrate.size)
    # ... rest of existing __post_init__ ...
```

- [ ] **Step 12.6: Run new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_substrate_state_space.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 12.7: Verify goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. Binary substrates with default integer state_space behave identically to before.

- [ ] **Step 12.8: Run full fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -x -q 2>&1 | tail -10
```

Expected: 0 failures. Any test that was calling `Substrate(..., alphabet_sizes=...)` should have been updated to either `state_space=` or `alphabet=`.

- [ ] **Step 12.9: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/substrate.py pyphi/system.py 2>&1 | tail -3
uv run ruff check pyphi/substrate.py pyphi/system.py test/test_substrate_state_space.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 12.10: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/substrate.py pyphi/system.py test/test_substrate_state_space.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Substrate accepts state_space= and alphabet= keywords

Substrate constructor signature: state_space= (flexible: uniform-flat or
per-node-tuple-of-tuples or None for integer defaults); alphabet= shortcut
for uniform integer alphabets; alphabet_sizes= parameter removed.

Substrate.state_space is a delegated @property pointing at the
FactoredTPM's state_space (single source of truth per the design).

System.__post_init__ coerces label-states to integer indices via the
substrate's state_space — users can pass either form."
git show --stat HEAD
```

---

## Task 13: `Substrate.joint_tpm()` alphabet-branch cleanup

**Goal:** Unify `Substrate.joint_tpm()` on the explicit-alphabet shape for binary AND k-ary. Migrate legacy callsites (`convert.state_by_node2state_by_state`, `infer_cm`, etc.) that depend on the binary `(2,)*n + (n,)` shape.

**Files:**
- Modify: `pyphi/substrate.py:joint_tpm` (drop the alphabet branch)
- Modify: `pyphi/convert.py` (audit callsites)
- Modify: `pyphi/macro.py:1109` and similar callsites
- Modify: tests that depend on the binary joint shape

- [ ] **Step 13.1: Audit legacy callsites.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "joint_tpm()\|\.joint_tpm\b" pyphi/ test/ --include="*.py" | grep -v ".pyc:"
```

Document each callsite and what shape it expects. Common patterns:

- `convert.state_by_node2state_by_state(substrate.joint_tpm())` — needs the legacy `(2,)*n + (n,)` shape.
- `infer_cm(substrate.joint_tpm())` — same.
- Tests asserting shape: usually shape check or pass to a legacy function.

For each callsite, decide:

- **Keep using the legacy shape** — wrap with a per-call binary-extraction (e.g., `substrate.joint_tpm()[..., :, 1]` for binary if the result is alphabet-explicit).
- **Migrate the callee to handle explicit-alphabet shape** — `convert.state_by_node2state_by_state` could be generalized.

Prefer migration where straightforward; wrap where the callee is legacy-only.

- [ ] **Step 13.2: Update `Substrate.joint_tpm()` to drop the alphabet branch.**

In `pyphi/substrate.py`:

```python
def joint_tpm(self) -> NDArray[np.float64]:
    """Materialize the joint conditional TPM on demand.

    Returns the explicit-alphabet shape ``(*alphabet_sizes, n_nodes, max_alpha)``
    for both binary and k-ary substrates. Use ``.to_joint()`` on a specific
    FactoredTPM for the same shape.

    Slow path — recomputes on every call (no cache).
    """
    return self._factored_tpm.to_joint()
```

Remove any binary special-casing.

- [ ] **Step 13.3: Migrate or wrap legacy callsites.**

For each callsite from Step 13.1, apply the chosen migration.

Example (`pyphi/macro.py:1109`):

```python
# Before:
sbs_tpm = convert.state_by_node2state_by_state(substrate.joint_tpm())

# After (if keeping convert.state_by_node2state_by_state legacy-only):
joint = substrate.joint_tpm()
# Extract legacy binary form: for binary substrate, joint shape is (2, ..., 2, n, 2)
# legacy form is (2, ..., 2, n) with P(node_i = 1)
if all(a == 2 for a in substrate.factored_tpm.alphabet_sizes):
    legacy_joint = joint[..., 1]  # extract P(node_i=1) slice along alphabet axis
    sbs_tpm = convert.state_by_node2state_by_state(legacy_joint)
else:
    # k-ary path; needs a different conversion or raise
    raise NotImplementedError(
        "state_by_node2state_by_state conversion is binary-only; "
        "k-ary substrates use the factored representation directly."
    )
```

For each affected callsite, write a similar adapter.

- [ ] **Step 13.4: Run tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -x -q 2>&1 | tail -10
```

Expected: 0 failures. Macro tests in particular should pass — the wrapping is structural.

- [ ] **Step 13.5: Verify goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical.

- [ ] **Step 13.6: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/substrate.py pyphi/macro.py 2>&1 | tail -3
uv run ruff check pyphi 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 13.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/substrate.py pyphi/macro.py # plus any other touched callsites
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Substrate.joint_tpm() unifies on explicit-alphabet shape

Drops the alphabet branch in Substrate.joint_tpm(). Both binary and
k-ary substrates return the explicit-alphabet shape
(*alphabet_sizes, n_nodes, max_alpha). Legacy callsites that depended
on the binary (2,)*n + (n,) shape are migrated to extract the legacy
form per-call (binary-only) with k-ary fallback raising explicitly."
git show --stat HEAD
```

---

## Task 14: Declarative `supports_alphabet` metadata on measure registry

**Goal:** Each measure in `pyphi/measures/distribution.py` (and any other measure registry) gains a `supports_alphabet` attribute — a callable `(alphabet_sizes: tuple[int, ...]) -> bool`. Defaults to `_any_alphabet`; EMD-family declares `_binary_only`.

**Files:**
- Modify: `pyphi/measures/distribution.py` (registry decorator + each measure declaration)
- Test: `test/test_measure_alphabet_support.py` (new)

- [ ] **Step 14.1: Write failing test.**

Create `test/test_measure_alphabet_support.py`:

```python
"""Tests for measure-alphabet-support metadata."""

from __future__ import annotations

import pytest

from pyphi.measures.distribution import (
    distribution_measures,
    # whichever registry holds the measure registrations
)


def test_all_measures_declare_supports_alphabet() -> None:
    """Every registered measure has a supports_alphabet callable."""
    for name in distribution_measures.names():
        measure = distribution_measures[name]
        assert hasattr(measure, "supports_alphabet"), (
            f"Measure {name} missing supports_alphabet declaration"
        )
        assert callable(measure.supports_alphabet)


def test_emd_supports_alphabet_binary_only() -> None:
    """EMD's supports_alphabet returns True for binary, False for k>2."""
    emd = distribution_measures["EMD"]
    assert emd.supports_alphabet((2, 2, 2)) is True
    assert emd.supports_alphabet((2, 3, 2)) is False
    assert emd.supports_alphabet((3, 3)) is False


def test_aid_supports_alphabet_alphabet_generic() -> None:
    """AID's supports_alphabet returns True for any alphabet."""
    aid = distribution_measures["AID"]
    assert aid.supports_alphabet((2, 2)) is True
    assert aid.supports_alphabet((3, 4, 5)) is True
```

- [ ] **Step 14.2: Run failing test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_measure_alphabet_support.py -v
```

Expected: AttributeError on `measure.supports_alphabet`.

- [ ] **Step 14.3: Update the measure registry to support `supports_alphabet`.**

In `pyphi/measures/distribution.py`:

```python
# Helpers for common alphabet-support callables
_binary_only = lambda a: all(x == 2 for x in a)
_any_alphabet = lambda a: True


# The registry's register decorator gains a supports_alphabet parameter
class MeasureRegistry(Registry):
    # ... existing body ...

    def register(
        self,
        name: str,
        asymmetric: bool = False,
        supports_alphabet: Callable[[tuple[int, ...]], bool] = _any_alphabet,
    ):
        def decorator(func):
            func.asymmetric = asymmetric
            func.supports_alphabet = supports_alphabet
            self[name] = func
            return func
        return decorator
```

(Adjust to match the actual registry implementation in the codebase.)

Then update each measure declaration:

```python
@distribution_measures.register("EMD", supports_alphabet=_binary_only)
def emd(...):
    """Earth Mover's Distance. Binary-only per Gomez et al. 2021 §2.3."""
    ...


@distribution_measures.register("AID", supports_alphabet=_any_alphabet)
def aid(...): ...


@distribution_measures.register("GENERALIZED_INTRINSIC_DIFFERENCE", supports_alphabet=_any_alphabet)
def gid(...): ...


@distribution_measures.register("INTRINSIC_INFORMATION", supports_alphabet=_any_alphabet)
def ii(...): ...


# ... etc. for every registered measure ...
```

Audit ALL existing measure declarations and add the appropriate `supports_alphabet`.

- [ ] **Step 14.4: Run new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_measure_alphabet_support.py -v
```

Expected: all pass.

- [ ] **Step 14.5: Verify goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. The metadata is additive; no behavior change.

- [ ] **Step 14.6: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi/measures/distribution.py test/test_measure_alphabet_support.py 2>&1 | tail -3
uv run ruff check pyphi/measures/distribution.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 14.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/measures/distribution.py test/test_measure_alphabet_support.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add declarative supports_alphabet metadata to measure registry

Each registered measure declares a supports_alphabet callable
(alphabet_sizes -> bool). EMD-family: binary-only. Intrinsic-difference
family (AID, GID, INTRINSIC_INFORMATION, GENERALIZED_INTRINSIC_DIFFERENCE):
alphabet-generic. Operation-level dispatcher guard follows in the next
commit."
git show --stat HEAD
```

---

## Task 15: Operation-level dispatcher guard

**Goal:** Read the `supports_alphabet` metadata at the measure-resolution boundary and raise `NotImplementedError` with a Gomez 2021 citation for unsupported (measure, alphabet) combinations.

**Files:**
- Modify: wherever the measure is resolved at compute time (likely
  `pyphi/measures/distribution.py:resolve_*` or
  `pyphi/core/repertoire_algebra.py`)
- Test: append to `test/test_measure_alphabet_support.py`

- [ ] **Step 15.1: Find the measure-resolution sites.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "mechanism_phi_measure\|distribution_measures\[" pyphi/ --include="*.py" | head -20
```

Identify the function(s) that take a measure-name string from config and return the callable. This is the natural place for the guard.

- [ ] **Step 15.2: Write failing test for the dispatcher guard.**

Append to `test/test_measure_alphabet_support.py`:

```python
def test_emd_raises_for_kary_substrate_via_sia() -> None:
    """EMD raises NotImplementedError when invoked on a k>2 substrate."""
    f = np.full((3, 3, 3), 1.0 / 3.0)
    sub = pyphi.Substrate(marginals=[f, f.copy()], alphabet=3)
    sys = pyphi.System(sub, state=(0, 0))
    with pyphi.config.override(mechanism_phi_measure="EMD"):
        with pytest.raises(NotImplementedError, match="alphabet|binary|Gomez"):
            pyphi.compute.sia(sys)


def test_aid_works_for_kary_substrate() -> None:
    """AID is alphabet-generic and works on a k>2 substrate."""
    f = np.full((3, 3, 3), 1.0 / 3.0)
    sub = pyphi.Substrate(marginals=[f, f.copy()], alphabet=3)
    sys = pyphi.System(sub, state=(0, 0))
    with pyphi.config.override(mechanism_phi_measure="AID"):
        sia = pyphi.compute.sia(sys)
        assert sia.phi >= 0
```

- [ ] **Step 15.3: Add the guard in the resolution path.**

Wherever the resolved measure is first used against a substrate's alphabet, add:

```python
def resolve_distribution_measure(name: str, alphabet_sizes: tuple[int, ...] = None) -> Callable:
    """Resolve a measure name to a callable; raise if alphabet unsupported."""
    measure = distribution_measures[name]
    if alphabet_sizes is not None and not measure.supports_alphabet(alphabet_sizes):
        raise NotImplementedError(
            f"Measure {name!r} does not support alphabet sizes "
            f"{alphabet_sizes}. For multi-valued substrates, use an "
            f"alphabet-generic measure (AID, GID, INTRINSIC_INFORMATION, "
            f"GENERALIZED_INTRINSIC_DIFFERENCE). "
            f"See Gomez et al. 2021 §2.3 "
            f"(https://doi.org/10.3390/e23010006) for the theoretical "
            f"rationale on EMD specifically."
        )
    return measure
```

Then update the callers (in `_resolve_iit_kwargs` or similar) to pass `alphabet_sizes`.

The exact integration depends on the codebase's measure-resolution flow — read it carefully and add the guard at the natural boundary.

- [ ] **Step 15.4: Run new tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_measure_alphabet_support.py -v
```

Expected: all pass.

- [ ] **Step 15.5: Verify binary goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23/23 byte-identical. The guard fires only on unsupported combinations; binary substrates are unaffected.

- [ ] **Step 15.6: Pyright + ruff + fast lane.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
uv run pytest test/ -m "not slow" -q 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 15.7: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add pyphi/measures/distribution.py # plus the file with the resolution function, if different
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Operation-level guard: unsupported (measure, alphabet) raises with citation

Reads supports_alphabet metadata at the measure-resolution boundary and
raises NotImplementedError for unsupported (measure, alphabet) combinations.
EMD-on-k>2 surfaces a clear error message including the measure name,
the alphabet sizes, the alphabet-generic alternatives (AID, GID,
INTRINSIC_INFORMATION, GENERALIZED_INTRINSIC_DIFFERENCE), and a citation
to Gomez et al. 2021 §2.3 for the theoretical rationale."
git show --stat HEAD
```

---

## Task 16: k>2 golden fixtures

**Goal:** Add k>2 golden fixtures: a small synthetic k=3 substrate, a heterogeneous (2,3,3) substrate, and (conditionally) the p53-Mdm2 network from Gomez 2021.

**Files:**
- Create: `test/data/golden/v1/multivalued_k3_tiny.{json,npz}`
- Create: `test/data/golden/v1/multivalued_2x3x3.{json,npz}`
- Create (conditional): `test/data/golden/v1/multivalued_p53_mdm2.{json,npz}`
- Modify: `test/test_golden_regression.py` (register new fixtures)
- Create: `test/golden/generate_p12b_fixtures.py` (generation script)

- [ ] **Step 16.1: Write the fixture-generation script.**

Create `test/golden/generate_p12b_fixtures.py`:

```python
"""Generate k>2 golden fixtures for P12b.

Run this once to produce the .json/.npz pairs in test/data/golden/v1/.
Subsequent runs of the regression test compare against the pinned values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pyphi

# (Generation code for each fixture)

def generate_multivalued_k3_tiny():
    """Small synthetic 2-node k=3 substrate. Hand-checkable phi."""
    rng = np.random.default_rng(2026)
    f0 = rng.uniform(size=(3, 3, 3))
    f0 = f0 / f0.sum(axis=-1, keepdims=True)
    f1 = rng.uniform(size=(3, 3, 3))
    f1 = f1 / f1.sum(axis=-1, keepdims=True)
    sub = pyphi.Substrate(
        marginals=[f0, f1],
        state_space=("LOW", "MID", "HIGH"),
    )
    sys = pyphi.System(sub, state=(0, 0))
    sia = pyphi.compute.sia(sys)
    # Serialize via the existing golden infrastructure
    # (Use the same mechanism as P12a's test/golden/compute.py)
    ...


def generate_multivalued_2x3x3():
    """3-node heterogeneous (2,3,3) substrate."""
    # ... similar pattern with mixed alphabets
    ...


def generate_multivalued_p53_mdm2():
    """The p53-Mdm2 network from Gomez et al. 2021 §3.

    Reproduces published phi values; conditional on the reproduction
    matching within config.numerics.precision.
    """
    # Encode the substrate per the paper's Section 3 / supplementary
    # ... if reproduction works, pin the phi; else log discrepancy
    ...


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "golden" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    generate_multivalued_k3_tiny()
    generate_multivalued_2x3x3()
    generate_multivalued_p53_mdm2()
```

The generation script's exact form depends on the existing golden infrastructure in `test/golden/`. Read `test/golden/compute.py` (or similar) to see the serialization pattern P12a established.

- [ ] **Step 16.2: Run the generation script.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run python test/golden/generate_p12b_fixtures.py
ls test/data/golden/v1/multivalued_*
```

Expected: 4 or 6 new files (2 per fixture: `.json` + `.npz`).

If p53-Mdm2 reproduction fails within `precision`, log the discrepancy and skip pinning that fixture. Use only the synthetic ones.

- [ ] **Step 16.3: Register new fixtures in `test/test_golden_regression.py`.**

Open the file and find the fixture loop or list. Add entries for the new fixtures. The pattern should match P12a's existing fixture handling.

- [ ] **Step 16.4: Run goldens.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_golden_regression.py -v
```

Expected: 23 + 2-3 new fixtures pass.

- [ ] **Step 16.5: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright test/test_golden_regression.py 2>&1 | tail -3
uv run ruff check test/golden/generate_p12b_fixtures.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 16.6: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add test/data/golden/v1/multivalued_*.json test/data/golden/v1/multivalued_*.npz test/golden/generate_p12b_fixtures.py test/test_golden_regression.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Add k>2 golden fixtures: synthetic k=3 + heterogeneous + p53-Mdm2 (conditional)

Three new golden fixtures: small synthetic k=3 2-node substrate;
heterogeneous (2,3,3) 3-node substrate; the p53-Mdm2 network from
Gomez et al. 2021 §3 (conditional on reproducibility within
config.numerics.precision).

Generation script in test/golden/generate_p12b_fixtures.py for
auditability; pinned outputs in test/data/golden/v1/multivalued_*.{json,npz}."
git show --stat HEAD
```

---

## Task 17: End-to-end k>2 SIA + AC smoke tests

**Goal:** Direct end-to-end tests confirming the k>2 IIT pipeline works: construct a multi-valued substrate, run `pyphi.compute.sia()`, get a valid SIA; same for `pyphi.compute.account()` on the AC side.

**Files:**
- Create: `test/test_substrate_multivalued.py`

- [ ] **Step 17.1: Write the end-to-end smoke tests.**

Create `test/test_substrate_multivalued.py`:

```python
"""End-to-end k>2 IIT analysis smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import pyphi


def _k3_two_node_substrate(seed: int = 2026) -> pyphi.Substrate:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(size=(3, 3, 3))
    f0 = f0 / f0.sum(axis=-1, keepdims=True)
    f1 = rng.uniform(size=(3, 3, 3))
    f1 = f1 / f1.sum(axis=-1, keepdims=True)
    return pyphi.Substrate(
        marginals=[f0, f1],
        state_space=("LOW", "MID", "HIGH"),
    )


def test_kary_sia_end_to_end() -> None:
    """Construct a k=3 substrate; run SIA; receive a phi value."""
    sub = _k3_two_node_substrate()
    sys = pyphi.System(sub, state=(0, 0))
    sia = pyphi.compute.sia(sys)
    assert sia.phi >= 0
    assert sia.partition is not None


def test_kary_account_end_to_end() -> None:
    """Construct a k=3 substrate; compute an Account."""
    sub = _k3_two_node_substrate()
    transition = pyphi.actual.Transition(
        substrate=sub,
        before_state=(0, 1),
        after_state=(1, 2),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    account = pyphi.compute.account(transition)
    assert account is not None
    assert all(link.alpha >= 0 for link in account)


def test_kary_state_as_labels_resolves() -> None:
    """SIA works when state is passed as labels."""
    sub = _k3_two_node_substrate()
    sys = pyphi.System(sub, state=("LOW", "LOW"))
    sia = pyphi.compute.sia(sys)
    assert sia.phi >= 0


def test_heterogeneous_alphabet_sia() -> None:
    """SIA works for a heterogeneous-alphabet substrate."""
    rng = np.random.default_rng(2026)
    f_binary = rng.uniform(size=(2, 3, 2))
    f_binary = f_binary / f_binary.sum(axis=-1, keepdims=True)
    f_ternary = rng.uniform(size=(2, 3, 3))
    f_ternary = f_ternary / f_ternary.sum(axis=-1, keepdims=True)
    sub = pyphi.Substrate(
        marginals=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    sys = pyphi.System(sub, state=(0, 0))
    sia = pyphi.compute.sia(sys)
    assert sia.phi >= 0
```

- [ ] **Step 17.2: Run the smoke tests.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_substrate_multivalued.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 17.3: Run the full fast lane + perf budget.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/ -m "not slow" -q 2>&1 | tail -3
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -10
```

Expected: 0 failures; perf budget within floors.

- [ ] **Step 17.4: Pyright + ruff.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pyright test/test_substrate_multivalued.py 2>&1 | tail -3
uv run ruff check test/test_substrate_multivalued.py 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 17.5: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add test/test_substrate_multivalued.py
git diff --cached --stat
git -c commit.gpgsign=false commit -m "End-to-end k>2 IIT analysis smoke tests

Four direct end-to-end tests: k=3 SIA, k=3 AC account, k=3 SIA with
state-as-labels, and heterogeneous-alphabet SIA. Confirms the full
pipeline (Substrate construction -> System/Transition -> compute.sia /
compute.account -> valid result) works for multi-valued substrates."
git show --stat HEAD
```

---

## Task 18: Documentation + changelog + scaffold-marker sweep + `_inner` grep test

**Goal:** Final tidy-up. Add changelog fragment. Fix the pre-existing broken `docs/conventions.rst` doctest. Add the `_inner` grep regression test. Sweep any remaining P12-prefixed scaffold markers introduced during P12b.

**Files:**
- Create: `changelog.d/p12b-multivalued.feature.md`
- Modify: `docs/conventions.rst` (fix the broken doctest)
- Modify: (potentially) the IIT 4.0 demo notebook if a natural mention point exists
- Create: `test/test_inner_unwrap_pattern.py` (grep regression test)

- [ ] **Step 18.1: Create the changelog fragment.**

Create `changelog.d/p12b-multivalued.feature.md`:

```markdown
Multi-valued (k-ary) substrates are now supported for IIT 4.0 analysis.

Construct via ``Substrate(marginals=[...], state_space=...)`` or the
``alphabet=`` shortcut. State_space accepts a flat tuple (uniform labels
across nodes) or a per-node tuple-of-tuples (heterogeneous alphabets).
The ``alphabet_sizes=`` parameter is removed; alphabet sizes are derived
from state_space or factor shapes.

Cause and effect TPMs now return precise typed objects:
:class:`pyphi.CausePosterior` for cause-side (joint posterior over past
states), :class:`pyphi.FactoredTPM` for effect-side (conditioned factored
form). Both subclass :class:`pyphi.JointDistribution`, a new base class
for multidimensional joint probability storage; :class:`pyphi.JointTPM`
(formerly the standalone joint storage class) is also refactored as a
JointDistribution subclass.

Measure registry: each measure declares ``supports_alphabet`` —
EMD-family is binary-only (per Gomez et al. 2021 §2.3); the
intrinsic-difference family (AID, GID, INTRINSIC_INFORMATION,
GENERALIZED_INTRINSIC_DIFFERENCE) is alphabet-generic. Using EMD on a
k>2 substrate raises ``NotImplementedError`` with a citation to the
relevant paper and pointers to alphabet-generic alternatives.

Macro analysis (``MacroSystem``) stays binary-only — see the P7b
deferral. AC paths (``pyphi.compute.account``) work for k>2 substrates
in this release.
```

- [ ] **Step 18.2: Fix the pre-existing `docs/conventions.rst` doctest.**

Read `docs/conventions.rst:40-44`:

```rst
    >>> from pyphi.examples import basic_noisy_selfloop_network
    >>> tpm = basic_noisy_selfloop_network().tpm
    >>> state = (0, 0, 1)
    >>> tpm[state]
    JointTPM([0.919 0.91  0.756])
```

Issues:
- `basic_noisy_selfloop_network` doesn't exist (renamed to `basic_noisy_selfloop_substrate`).
- `substrate.tpm` is now a `FactoredTPM` which doesn't support `tpm[state]` ndarray-style indexing.

Update to use the explicit-joint form:

```rst
    >>> from pyphi.examples import basic_noisy_selfloop_substrate
    >>> joint = basic_noisy_selfloop_substrate().joint_tpm()
    >>> state = (0, 0, 1)
    >>> # joint shape: (2,2,2,n,a) -- pick out the per-node P(on=1) slice for legacy display
    >>> # For binary, the [..., 1] slice gives P(node_i=1) per node:
    >>> joint[state][:, 1]
    array([0.919, 0.91 , 0.756])
```

(Adjust based on the actual numbers and shape; rerun the doctest to verify.)

- [ ] **Step 18.3: Add the `_inner` grep regression test.**

Create `test/test_inner_unwrap_pattern.py`:

```python
"""Regression test: the _inner unwrap pattern is retired from production code."""

from __future__ import annotations

import subprocess


def test_no_inner_unwrap_pattern_in_production() -> None:
    """No production code contains the `_inner if hasattr` unwrap pattern.

    P12a kept the pattern as a backward-compat shim with # type: ignore
    annotations. P12b's hot-path cutover retired it. This test locks the
    cleanup so future changes don't accidentally reintroduce it.
    """
    result = subprocess.run(
        ["grep", "-rn", "_inner if hasattr", "pyphi/"],
        capture_output=True,
        text=True,
    )
    # Filter out comments (lines where `_inner if hasattr` appears after a `#`)
    offending = [
        line for line in result.stdout.splitlines()
        if "#" not in line.split(":", 2)[-1].split("_inner")[0]
    ]
    # Allow the JointTPM/JointDistribution class definition file to reference
    # `_inner` as the attribute itself (not the unwrap pattern); the pattern
    # is the specific `_inner if hasattr` form.
    assert not offending, (
        f"Found _inner unwrap patterns in production code: {offending}"
    )
```

- [ ] **Step 18.4: Run the regression test.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest test/test_inner_unwrap_pattern.py -v
```

Expected: passes (no production-code matches).

- [ ] **Step 18.5: Scaffold-marker sweep.**

```bash
cd /Users/will/projects/pyphi-p12b
grep -rn "# P12\|# P7\|TODO(P\|TODO(4\.0)" pyphi/core/tpm/ pyphi/core/unit.py pyphi/measures/ pyphi/system.py pyphi/actual.py pyphi/substrate.py 2>&1
```

Expected: no matches. If any P-markers were introduced during the cutover, remove them.

- [ ] **Step 18.6: Final acceptance gates.**

```bash
cd /Users/will/projects/pyphi-p12b
uv run pytest --tb=short -q 2>&1 | tail -5
uv run pytest test/test_golden_regression.py -v 2>&1 | tail -5
uv run pytest test/test_perf_budget.py -v 2>&1 | tail -5
uv run pyright pyphi 2>&1 | tail -3
uv run ruff check pyphi test 2>&1 | tail -3
```

Expected:
- Full suite (including doctests): 0 failures.
- Goldens: 23 byte-identical binary + new k>2 fixtures pass.
- Perf budget: all within floors.
- Pyright: 0 errors / 5 baseline warnings.
- Ruff: clean.

- [ ] **Step 18.7: End-to-end smoke (manual).**

```bash
cd /Users/will/projects/pyphi-p12b
uv run python -c "
import numpy as np
import pyphi

# k=3 substrate
f = np.full((3, 3, 3), 1.0 / 3.0)
sub = pyphi.Substrate(marginals=[f, f.copy()], state_space=('LOW', 'MID', 'HIGH'))
sys = pyphi.System(sub, state=('LOW', 'LOW'))
sia = pyphi.compute.sia(sys)
print(f'k=3 SIA phi: {sia.phi}')

# AC k=3
transition = pyphi.actual.Transition(
    substrate=sub,
    before_state=(0, 1),
    after_state=(1, 2),
    cause_indices=(0, 1),
    effect_indices=(0, 1),
)
account = pyphi.compute.account(transition)
print(f'k=3 Account: {len(list(account))} links')

print('END-TO-END OK')
"
```

Expected: prints two values + `END-TO-END OK`.

- [ ] **Step 18.8: Commit.**

```bash
cd /Users/will/projects/pyphi-p12b
git add changelog.d/p12b-multivalued.feature.md docs/conventions.rst test/test_inner_unwrap_pattern.py # plus any scaffold-marker cleanup files
git diff --cached --stat
git -c commit.gpgsign=false commit -m "Documentation + changelog + _inner regression test for P12b

Adds the feature changelog fragment covering multi-valued substrates,
typed cause/effect return types, JointDistribution sibling hierarchy,
and the declarative measure-alphabet policy.

Fixes the pre-existing broken doctest in docs/conventions.rst (used
basic_noisy_selfloop_network which doesn't exist post the
Network->Substrate rename; updated to substrate.joint_tpm() shape).

Adds the _inner grep regression test that locks in P12b's cleanup."
git show --stat HEAD
```

---

## Final acceptance gates

After all commits land, run:

```bash
cd /Users/will/projects/pyphi-p12b

# Full suite including doctests
uv run pytest --tb=short -q                                # 0 failures expected

# Goldens
uv run pytest test/test_golden_regression.py -v            # 23 binary byte-identical + new k>2 pass

# Perf budget
uv run pytest test/test_perf_budget.py -v                  # all within max(3.0, 4x median)

# Slow lane
uv run pytest test/ --slow -q                              # 0 failures

# Static checks
uv run pyright pyphi                                       # 0 errors / 5 baseline warnings
uv run ruff check pyphi test                               # clean
uv run ruff format --check pyphi test                      # clean

# End-to-end smoke
uv run python -c "
import numpy as np
import pyphi
f = np.full((3, 3, 3), 1.0 / 3.0)
sub = pyphi.Substrate(marginals=[f, f.copy()], state_space=('LOW', 'MID', 'HIGH'))
sys = pyphi.System(sub, state=('LOW', 'LOW'))
sia = pyphi.compute.sia(sys)
assert sia.phi >= 0
print('END-TO-END OK')
"
```

All gates must pass before declaring P12b complete.

---

## Self-review checklist

(For the plan-writer, run after the plan is committed:)

**1. Spec coverage:** every spec section/requirement is implemented by a task.

- Spec §1 (background): no implementation needed; described in plan intro.
- Spec §2 (scope): in/out items covered across Tasks 1-18.
- Spec §3 (architecture): Tasks 2, 3, 4, 5, 8, 9, 11 cover the type hierarchy + math + cutover.
- Spec §4 (user-facing API): Tasks 11, 12 cover state_space, alphabet=, constructor parsing.
- Spec §5 (math implementation): Tasks 1, 4, 5, 7 cover the binary/kary paths and effect.
- Spec §6 (validation): Tasks 11, 12 cover extended validation.
- Spec §7 (testing): Tasks 6, 16, 17, 18 cover property tests, goldens, smoke, grep regression.
- Spec §8 (migration plan): mirrored in Tasks 1-18 with the same phase ordering.
- Spec §9 (out of scope): nothing to implement.

**2. Placeholder scan:** look for TBD/TODO/FIXME/"implement later"/"similar to Task N" patterns. Fix inline.

**3. Type consistency:** types and method names used in later tasks match earlier definitions.

- `JointDistribution` (Task 2), `CausePosterior` (Task 3), `FactoredTPM` (P12a), `JointTPM` (P12a + refactor in Task 2): consistent throughout.
- `state_space` (parameter name, attribute name): consistent in Tasks 11, 12.
- `alphabet=` (kwarg): consistent in Task 12.
- `marginalize_out`, `condition_factor`, `to_joint`: method names match P12a and Task 10's consumers.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-24-p12b-multivalued-units.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, two-stage review after each (spec compliance, then code quality), fast iteration. Matches P11.95d's and P12a's pattern. Sonnet 4.6 for mechanical TDD tasks (most of the 18); Opus 4.7 for the most-load-bearing cutover (Task 10) and the math correctness verification (Task 5/6) where cross-file coordination and math judgment matter most. Reviewer subagents Opus 4.7 throughout.

**2. Inline Execution** — Execute tasks in this session using the executing-plans skill, batch execution with checkpoints between tasks.

**Which approach?**
