# B16 — First-class `Complex` marker type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a first-class `Complex` value type so `complexes()` returns `tuple[Complex, ...]` and `maximal_complex()` returns a `Complex`, with the exclusion postulate enforced by a named `validate.non_overlapping(...)` invariant — no change to the condensation math.

**Architecture:** A frozen `Complex` (`pyphi/models/complex.py`) wraps a `SystemIrreducibilityAnalysis` (IIT 3.0 or 4.0) plus `is_maximal`, the selecting `Substrate`, and a tuple of lightweight `ExcludedCandidate` records. `pyphi/substrate.py:complexes()` keeps the existing cascade and wraps the accepted SIAs, computing exclusion records in a post-pass over `sorted_sias` (so the cascade is untouched). `maximal_complex()` returns the head, or a null-object `Complex` (falsy, `node_indices == ()`).

**Tech Stack:** Python 3.13, pytest, the existing `pyphi.models` value-type conventions (`cmp.OrderableByPhi`, `fmt`, `jsonify`).

**Spec:** `docs/superpowers/specs/2026-06-15-b16-complex-marker-type-design.md`

---

## File structure

- **Create:** `pyphi/models/complex.py` — `ExcludedCandidate` + `Complex` value types.
- **Create:** `test/test_complex_model.py` — unit tests for the two types (construction, delegation, `__bool__`, serialization).
- **Modify:** `pyphi/models/__init__.py` — export `Complex`, `ExcludedCandidate`.
- **Modify:** `pyphi/jsonify.py` — register both in `_loadable_models()`.
- **Modify:** `pyphi/validate.py` — add `non_overlapping(...)`.
- **Modify:** `pyphi/substrate.py` — wrap results in `complexes()` / `maximal_complex()`; add `_exclusion_records(...)`; update the `Substrate.complexes` / `Substrate.maximal_complex` method return hints + docstrings.
- **Modify:** `test/test_complexes.py` — integration tests (both formalisms).
- **Create:** `changelog.d/b16-complex-type.feature.md`.
- **Modify:** `ROADMAP.md` — flip B16 to ✅.

---

## Task 1: `ExcludedCandidate` value type

**Files:**
- Create: `pyphi/models/complex.py`
- Test: `test/test_complex_model.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_complex_model.py`:

```python
from pyphi import jsonify
from pyphi.models.complex import ExcludedCandidate


def test_excluded_candidate_fields():
    e = ExcludedCandidate(node_indices=[1, 2], phi=0.5)
    assert e.node_indices == (1, 2)  # coerced to tuple
    assert e.phi == 0.5
    assert isinstance(e.phi, float)


def test_excluded_candidate_equality_precision_aware():
    a = ExcludedCandidate((1, 2), 0.5)
    b = ExcludedCandidate((1, 2), 0.5 + 1e-15)
    c = ExcludedCandidate((0, 2), 0.5)
    assert a == b           # phi compared up to PRECISION
    assert a != c           # different units


def test_excluded_candidate_hashable_by_units():
    a = ExcludedCandidate((1, 2), 0.5)
    b = ExcludedCandidate((1, 2), 0.9)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1  # same units collapse


def test_excluded_candidate_json_round_trip():
    e = ExcludedCandidate((1, 2), 0.5)
    decoded = jsonify.loads(jsonify.dumps(e))
    assert decoded == e
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_complex_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.models.complex'`

- [ ] **Step 3: Write minimal implementation**

Create `pyphi/models/complex.py`:

```python
# models/complex.py
"""The |Complex| — an irreducible system selected as a local maximum of
|big_phi| under the exclusion postulate — and the lightweight record of a
candidate excluded in its favor."""

from __future__ import annotations

from typing import Any

from pyphi import utils

from . import cmp
from . import fmt

_excluded_candidate_attributes = ["node_indices", "phi"]


class ExcludedCandidate:
    """A candidate system excluded from being a complex in favor of an
    overlapping complex with greater-or-equal |big_phi|.

    Holds plain values only (units and |big_phi|), never a back-reference to
    the excluding |Complex|, so the heavy analysis graph is not retained.

    Attributes:
        node_indices (tuple[int, ...]): The excluded candidate's units.
        phi (float): The candidate's |big_phi| value.
    """

    def __init__(self, node_indices: Any, phi: Any) -> None:
        self.node_indices: tuple[int, ...] = tuple(node_indices)
        self.phi: float = float(phi)

    def __repr__(self) -> str:
        return fmt.make_repr(self, _excluded_candidate_attributes)

    def __str__(self) -> str:
        return f"ExcludedCandidate(node_indices={self.node_indices}, phi={self.phi})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExcludedCandidate):
            return NotImplemented
        return self.node_indices == other.node_indices and utils.eq(
            self.phi, other.phi
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.node_indices)

    def to_json(self) -> dict[str, Any]:
        return {"node_indices": list(self.node_indices), "phi": self.phi}

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> ExcludedCandidate:
        return cls(node_indices=dct["node_indices"], phi=dct["phi"])
```

- [ ] **Step 4: Run test to verify it fails on the JSON round-trip only**

Run: `uv run pytest test/test_complex_model.py -v`
Expected: the first three tests PASS; `test_excluded_candidate_json_round_trip` FAILS (not yet registered as a loadable model — `jsonify.loads` returns a plain dict). This is fixed in Task 3; leave it failing for now.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/complex.py test/test_complex_model.py
git commit -m "Add ExcludedCandidate value type for B16"
```

---

## Task 2: `Complex` value type

**Files:**
- Modify: `pyphi/models/complex.py`
- Test: `test/test_complex_model.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_complex_model.py`:

```python
import pytest

from pyphi import examples
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.substrate import irreducible_sias


def _basic_sia():
    """Return (substrate, a real irreducible SIA) under IIT 4.0 defaults."""
    substrate = examples.basic_substrate()
    sias = irreducible_sias(substrate, (1, 0, 0))
    return substrate, sias[0]


def test_complex_delegates_node_indices_and_phi():
    substrate, s = _basic_sia()
    c = Complex(sia=s, substrate=substrate, is_maximal=True)
    assert c.node_indices == s.node_indices
    assert float(c.phi) == float(s.phi)
    assert c.sia is s
    assert c.substrate is substrate
    assert c.is_maximal is True
    assert c.excluded == ()


def test_complex_is_truthy_when_phi_positive():
    substrate, s = _basic_sia()
    c = Complex(sia=s, substrate=substrate, is_maximal=True)
    assert bool(c) is True


def test_complex_null_object_is_falsy_with_empty_units():
    substrate = examples.basic_substrate()
    null = Complex(
        sia=NullSystemIrreducibilityAnalysis(),
        substrate=substrate,
        is_maximal=True,
    )
    assert bool(null) is False
    assert null.node_indices == ()   # None normalized to ()
    assert float(null.phi) == 0.0


def test_complex_orders_by_phi():
    substrate, s = _basic_sia()
    big = Complex(sia=s, substrate=substrate)
    null = Complex(sia=NullSystemIrreducibilityAnalysis(), substrate=substrate)
    assert null < big
    assert max([null, big]) is big
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_complex_model.py -k "complex" -v`
Expected: FAIL — `ImportError: cannot import name 'Complex'`

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/models/complex.py` (after `ExcludedCandidate`):

```python
_complex_attributes = ["node_indices", "phi", "is_maximal", "excluded"]


class Complex(cmp.OrderableByPhi):
    """An irreducible system selected as a complex: a local maximum of
    |big_phi| over overlapping candidate systems (the exclusion postulate).

    Wraps the system irreducibility analysis (IIT 3.0 or 4.0) and records
    whether it is the |big_phi|-maximal complex of its substrate, the
    candidates excluded in its favor, and the substrate that selected it.
    Ordered by |big_phi| like the wrapped analysis.

    Attributes:
        sia: The wrapped system irreducibility analysis.
        substrate (Substrate): The substrate this complex was selected from.
        is_maximal (bool): Whether this is the |big_phi|-maximal complex.
        excluded (tuple[ExcludedCandidate, ...]): Overlapping candidates
            excluded in this complex's favor.
    """

    def __init__(
        self,
        sia: Any,
        substrate: Any,
        is_maximal: bool = False,
        excluded: Any = (),
    ) -> None:
        self.sia = sia
        self.substrate = substrate
        self.is_maximal = bool(is_maximal)
        self.excluded: tuple[ExcludedCandidate, ...] = tuple(excluded)

    @property
    def node_indices(self) -> tuple[int, ...]:
        """The units of this complex (``()`` for a null complex)."""
        from pyphi.substrate import _sia_node_indices

        return _sia_node_indices(self.sia) or ()

    @property
    def phi(self) -> Any:
        """The |big_phi| value of this complex."""
        return self.sia.phi

    def order_by(self) -> Any:
        return self.sia.order_by()

    def __bool__(self) -> bool:
        """``True`` iff |big_phi > 0| (a null complex is falsy)."""
        return not utils.eq(self.phi, 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented
        return (
            self.sia == other.sia
            and self.is_maximal == other.is_maximal
            and self.excluded == other.excluded
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.node_indices, self.is_maximal))

    def _repr_columns(self) -> Any:
        return [
            ("Complex", ""),
            ("φ_s", self.phi),
            ("Units", self.node_indices),
            ("Maximal", self.is_maximal),
            ("Excluded", tuple(e.node_indices for e in self.excluded)),
        ]

    def __repr__(self) -> str:
        return fmt.make_repr(self, _complex_attributes)

    def __str__(self) -> str:
        return self.__repr__()

    def to_json(self) -> dict[str, Any]:
        return {
            "sia": self.sia,
            "substrate": self.substrate,
            "is_maximal": self.is_maximal,
            "excluded": list(self.excluded),
        }

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> Complex:
        return cls(
            sia=dct["sia"],
            substrate=dct["substrate"],
            is_maximal=dct["is_maximal"],
            excluded=dct["excluded"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_complex_model.py -k "complex" -v`
Expected: PASS (all four `complex` tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/complex.py test/test_complex_model.py
git commit -m "Add Complex value type wrapping a SIA for B16"
```

---

## Task 3: Export + register for serialization

**Files:**
- Modify: `pyphi/models/__init__.py`
- Modify: `pyphi/jsonify.py:132-174` (the `classes` list in `_loadable_models()`)
- Test: `test/test_complex_model.py` (the round-trip tests from Tasks 1–2)

- [ ] **Step 1: Confirm the currently-failing round-trip tests**

Run: `uv run pytest test/test_complex_model.py -k "json or round_trip" -v`
Expected: `test_excluded_candidate_json_round_trip` FAILS (decoded is a plain dict, not an `ExcludedCandidate`).

- [ ] **Step 2: Add a Complex round-trip test**

Append to `test/test_complex_model.py`:

```python
def test_complex_json_round_trip():
    substrate, s = _basic_sia()
    c = Complex(
        sia=s,
        substrate=substrate,
        is_maximal=True,
        excluded=(ExcludedCandidate((1, 2), 0.5),),
    )
    decoded = jsonify.loads(jsonify.dumps(c))
    assert isinstance(decoded, Complex)
    assert decoded.node_indices == c.node_indices
    assert decoded.is_maximal is True
    assert {e.node_indices for e in decoded.excluded} == {(1, 2)}
```

- [ ] **Step 3: Export from `pyphi/models/__init__.py`**

Add these imports next to the other `from .` imports (alphabetical neighborhood — after the `.ces` import at line 51):

```python
from .complex import Complex
from .complex import ExcludedCandidate
```

Add to the `__all__` list (alphabetical position, near `"Concept"`):

```python
    "Complex",
    "ExcludedCandidate",
```

- [ ] **Step 4: Register in `pyphi/jsonify.py`**

In `_loadable_models()`, add to the `classes` list (next to the other `pyphi.models.*` entries, e.g. after `pyphi.models.Concept`):

```python
        pyphi.models.Complex,  # pyright: ignore[reportAttributeAccessIssue]
        pyphi.models.ExcludedCandidate,  # pyright: ignore[reportAttributeAccessIssue]
```

- [ ] **Step 5: Run the round-trip tests**

Run: `uv run pytest test/test_complex_model.py -v`
Expected: PASS (all tests, including both JSON round-trips).

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/__init__.py pyphi/jsonify.py test/test_complex_model.py
git commit -m "Export and register Complex/ExcludedCandidate for serialization"
```

---

## Task 4: `validate.non_overlapping`

**Files:**
- Modify: `pyphi/validate.py`
- Test: `test/test_complex_model.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_complex_model.py`:

```python
from pyphi import validate


def test_non_overlapping_accepts_disjoint():
    substrate, s = _basic_sia()
    a = Complex(sia=s, substrate=substrate)

    class _Stub:
        def __init__(self, idx):
            self.node_indices = idx

    disjoint = [a, _Stub((9,))]  # (0,1,2) vs (9,) — disjoint
    assert validate.non_overlapping(disjoint) is True


def test_non_overlapping_rejects_overlap():
    class _Stub:
        def __init__(self, idx):
            self.node_indices = idx

    overlapping = [_Stub((0, 1)), _Stub((1, 2))]  # share unit 1
    with pytest.raises(ValueError, match="Exclusion violated"):
        validate.non_overlapping(overlapping)
```

`ValueError` matches the other `pyphi/validate.py` validators (e.g. `direction`), so it is the consistent choice.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_complex_model.py -k non_overlapping -v`
Expected: FAIL — `AttributeError: module 'pyphi.validate' has no attribute 'non_overlapping'`

- [ ] **Step 3: Write minimal implementation**

Append to `pyphi/validate.py`:

```python
def non_overlapping(complexes: Iterable[Any]) -> bool:
    """Validate that complexes have pairwise-disjoint units (exclusion).

    The exclusion postulate requires that no unit belongs to more than one
    complex. Raises if any two of ``complexes`` share a unit.

    Args:
        complexes (Iterable): Objects exposing ``node_indices``.

    Returns:
        bool: ``True`` if the complexes are pairwise node-disjoint.
    """
    seen: set[int] = set()
    for c in complexes:
        units = set(c.node_indices or ())
        overlap = units & seen
        if overlap:
            raise ValueError(
                f"Exclusion violated: unit(s) {sorted(overlap)} belong to more "
                f"than one complex."
            )
        seen.update(units)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_complex_model.py -k non_overlapping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/validate.py test/test_complex_model.py
git commit -m "Add validate.non_overlapping for the exclusion postulate"
```

---

## Task 5: Wire `complexes()` → `tuple[Complex, ...]`

**Files:**
- Modify: `pyphi/substrate.py:696-726` (`complexes`), and add `_exclusion_records` near it
- Test: `test/test_complexes.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_complexes.py` (module level, after the imports):

```python
class TestComplexWrapperIIT40:
    """B16: complexes() returns Complex objects under IIT 4.0."""

    def test_complexes_are_complex_objects(self, s):
        from pyphi.models.complex import Complex

        cx = s.substrate.complexes(s.state)
        assert isinstance(cx, tuple)
        assert all(isinstance(c, Complex) for c in cx)

    def test_exactly_one_is_maximal(self, s):
        cx = s.substrate.complexes(s.state)
        assert sum(1 for c in cx if c.is_maximal) == 1
        assert cx[0].is_maximal is True

    def test_dual_and_xor_excluded_records(self):
        from test.example_substrates import dual_and_xor_substrate

        substrate = dual_and_xor_substrate()
        cx = substrate.complexes((1, 0, 1, 0))
        assert {tuple(sorted(c.node_indices)) for c in cx} == {(0, 1), (2, 3)}
        by_units = {tuple(sorted(c.node_indices)): c for c in cx}
        # The single-node candidates (1,) and (3,) are excluded by the
        # 2-node complexes they overlap.
        assert {e.node_indices for e in by_units[(0, 1)].excluded} == {(1,)}
        assert {e.node_indices for e in by_units[(2, 3)].excluded} == {(3,)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_complexes.py::TestComplexWrapperIIT40 -v`
Expected: FAIL — `complexes()` returns a `list` of SIAs (`isinstance(cx, tuple)` is False; elements are not `Complex`).

- [ ] **Step 3: Write the implementation**

In `pyphi/substrate.py`, add `_exclusion_records` immediately above `complexes` (before line 696):

```python
def _exclusion_records(
    accepted: list[Any], sorted_sias: list[Any]
) -> dict[tuple[int, ...], tuple[Any, ...]]:
    """Map each accepted complex (by units) to the ExcludedCandidate records
    it excluded: every irreducible candidate that overlaps it and was not
    itself accepted.

    A candidate that overlaps several accepted complexes appears in each of
    their exclusion sets. Reads only values the cascade already computed.
    """
    from pyphi.models.complex import ExcludedCandidate

    accepted_indices = {tuple(_sia_node_indices(s) or ()) for s in accepted}
    records: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for acc in accepted:
        acc_idx = tuple(_sia_node_indices(acc) or ())
        acc_set = set(acc_idx)
        recs = []
        for cand in sorted_sias:
            cand_idx = tuple(_sia_node_indices(cand) or ())
            if cand_idx == acc_idx or cand_idx in accepted_indices:
                continue
            if acc_set & set(cand_idx):
                recs.append(ExcludedCandidate(cand_idx, float(cand.phi)))
        records[acc_idx] = tuple(recs)
    return records
```

Then replace the body of `complexes` (lines 718-726) with:

```python
    from pyphi import validate
    from pyphi.models.complex import Complex

    sorted_sias = sorted(
        irreducible_sias(substrate, state, candidates, **kwargs), reverse=True
    )
    if not sorted_sias:
        return ()

    if _config_iit_version() == "IIT_3_0":
        accepted = _iit3_exclusion_cascade(sorted_sias, substrate, state)
    else:
        accepted = _substrate_exclusion_cascade(sorted_sias, substrate, state)
    if not accepted:
        return ()

    records = _exclusion_records(accepted, sorted_sias)
    result = tuple(
        Complex(
            sia=sia,
            substrate=substrate,
            is_maximal=(i == 0),
            excluded=records[tuple(_sia_node_indices(sia) or ())],
        )
        for i, sia in enumerate(accepted)
    )
    validate.non_overlapping(result)
    return result
```

Update the `complexes` signature return annotation from `-> list[Any]:` to `-> tuple[Any, ...]:` and update its docstring's "The returned list is..." sentence to "The returned tuple is...".

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_complexes.py::TestComplexWrapperIIT40 -v`
Expected: PASS

- [ ] **Step 5: Run the existing complexes regression tests**

Run: `uv run pytest test/test_complexes.py -v`
Expected: PASS — the existing IIT 3.0 and IIT 4.0 tests read `.node_indices` / `.phi`, which the `Complex` wrapper delegates, so they still pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add pyphi/substrate.py test/test_complexes.py
git commit -m "Wrap complexes() results in Complex with exclusion records (B16)"
```

---

## Task 6: Wire `maximal_complex()` → null-object `Complex`

**Files:**
- Modify: `pyphi/substrate.py:908-934` (`maximal_complex`), and `pyphi/substrate.py:414-430` (the `Substrate.complexes` / `Substrate.maximal_complex` method hints/docstrings)
- Test: `test/test_complexes.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_complexes.py::TestComplexWrapperIIT40`:

```python
    def test_maximal_complex_is_complex(self, s):
        from pyphi.models.complex import Complex

        mc = s.substrate.maximal_complex(s.state)
        assert isinstance(mc, Complex)
        assert mc.is_maximal is True
        assert mc.node_indices == s.substrate.complexes(s.state)[0].node_indices

    def test_maximal_complex_null_object(self, s):
        from pyphi.models.complex import Complex

        # Forcing an empty candidate set yields no complexes.
        mc = s.substrate.maximal_complex(s.state, candidates=[])
        assert isinstance(mc, Complex)
        assert bool(mc) is False
        assert mc.node_indices == ()
        assert float(mc.phi) == 0.0
        assert mc.is_maximal is True
        assert mc.excluded == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_complexes.py::TestComplexWrapperIIT40 -k maximal -v`
Expected: FAIL — `maximal_complex` returns a SIA / `NullCauseEffectStructure`, not a `Complex`.

- [ ] **Step 3: Write the implementation**

Replace the body of `maximal_complex` (lines 919-934) with:

```python
    from pyphi.models.complex import Complex

    found = complexes(substrate, state, candidates, **kwargs)
    if found:
        return found[0]
    # No irreducible candidate; return a null-object Complex over the empty
    # system (falsy, with empty units).
    from pyphi.conf import config as _config

    empty = System.from_substrate(substrate, state, ())
    if _config.formalism.iit.version == "IIT_3_0":
        from pyphi.formalism.iit3 import _null_sia

        null_sia = _null_sia(empty)
    else:
        from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

        null_sia = NullSystemIrreducibilityAnalysis()
    return Complex(sia=null_sia, substrate=substrate, is_maximal=True, excluded=())
```

Update the `maximal_complex` return annotation from `-> Any:` to `-> Any:` (unchanged — `Complex` is constructed lazily, keep `Any`) and update its docstring: change "Returns a null SIA over the empty system when no irreducible candidate exists." to "Returns a null-object |Complex| (falsy, with empty units) when no irreducible candidate exists."

Update the two `Substrate` methods' docstrings/hints at lines 414-430: `complexes` return hint `list[Any]` → `tuple[Any, ...]`; leave `maximal_complex` as `Any`. Update their one-line docstrings to "Return the substrate's complexes as |Complex| objects; see :func:`complexes`." and "Return the maximal |Complex|; see :func:`maximal_complex`.".

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_complexes.py::TestComplexWrapperIIT40 -v`
Expected: PASS

- [ ] **Step 5: Run actual.py's major_complex path (regression)**

Run: `uv run pytest test/test_actual.py -v -k "complex or major or account"`
Expected: PASS — `actual.py` reads `maximal_complex(...).node_indices`, which the `Complex` wrapper delegates.

- [ ] **Step 6: Commit**

```bash
git add pyphi/substrate.py test/test_complexes.py
git commit -m "Return null-object Complex from maximal_complex (B16)"
```

---

## Task 7: IIT 3.0 integration coverage

**Files:**
- Test: `test/test_complexes.py`

- [ ] **Step 1: Write the failing/passing test (IIT 3.0 path)**

Add to the existing `class TestComplexesIIT30` in `test/test_complexes.py` (which is wrapped in `IIT_3_CONFIG` via its autouse fixture):

```python
    def test_complexes_are_complex_objects_iit3(self, s):
        from pyphi.models.complex import Complex

        cx = s.substrate.complexes(s.state)
        assert isinstance(cx, tuple)
        assert len(cx) == 1
        assert isinstance(cx[0], Complex)
        assert cx[0].is_maximal is True
        assert cx[0].node_indices == (0, 1, 2)

    def test_complexes_excluded_iit3(self, s):
        # The single complex (0,1,2) excludes the overlapping lower-phi
        # irreducible candidates (1,2):1.0 and (0,2):0.5.
        cx = s.substrate.complexes(s.state)
        assert {e.node_indices for e in cx[0].excluded} == {(1, 2), (0, 2)}

    def test_maximal_complex_null_object_iit3(self, s):
        from pyphi.models.complex import Complex

        mc = s.substrate.maximal_complex(s.state, candidates=[])
        assert isinstance(mc, Complex)
        assert bool(mc) is False
        assert mc.node_indices == ()
```

- [ ] **Step 2: Run the IIT 3.0 tests**

Run: `uv run pytest test/test_complexes.py::TestComplexesIIT30 -v`
Expected: PASS (the existing `test_complexes_standard` / `test_maximal_complex` still pass through the wrapper; the three new ones pass).

- [ ] **Step 3: Commit**

```bash
git add test/test_complexes.py
git commit -m "Add IIT 3.0 integration coverage for Complex wrapper (B16)"
```

---

## Task 8: Changelog + ROADMAP

**Files:**
- Create: `changelog.d/b16-complex-type.feature.md`
- Modify: `ROADMAP.md` (Status Dashboard row + Wave-2 entry)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/b16-complex-type.feature.md`:

```markdown
Added a first-class `Complex` type. `Substrate.complexes()` now returns
`tuple[Complex, ...]` and `Substrate.maximal_complex()` returns a `Complex`
(a falsy null-object when no system is irreducible). Each `Complex` exposes
`is_maximal`, the selecting `substrate`, and `excluded` — the overlapping
candidates excluded in its favor. The exclusion postulate is enforced by
`validate.non_overlapping()`.
```

- [ ] **Step 2: Update the ROADMAP dashboard**

In `ROADMAP.md`, change the B16 dashboard row from `⬜ open` to `✅ landed`, move `B16` into the "✅ Landed" list, and update the Wave-2 "Remaining 2.0 Work" `B16` bullet to past-tense "landed" prose noting it advances ship-criterion #1. Verify against the code before editing (per the ROADMAP maintenance protocol).

- [ ] **Step 3: Commit**

```bash
git add changelog.d/b16-complex-type.feature.md ROADMAP.md
git commit -m "Changelog + ROADMAP: B16 Complex marker type landed"
```

---

## Task 9: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite WITHOUT a path argument**

Run: `uv run pytest -q`
Expected: PASS. The no-path invocation uses `testpaths` and runs the `pyphi/` doctest sweep (per CLAUDE.md — bare-path runs skip doctests). If the run is too long for one shot, kick off the slow lane (`test/test_invariants_hypothesis.py`) in the background and run the rest in the foreground, but a full no-path run must pass before claiming completion.

- [ ] **Step 2: Type-check and lint the changed files**

Run: `uv run pyright pyphi/models/complex.py pyphi/substrate.py pyphi/validate.py pyphi/jsonify.py pyphi/models/__init__.py`
Expected: no new errors.

Run: `uv run ruff check pyphi/models/complex.py pyphi/substrate.py pyphi/validate.py test/test_complex_model.py test/test_complexes.py`
Expected: clean (no autofixes applied without permission).

- [ ] **Step 3: Final commit if any lint/type fixes were needed**

```bash
git add -A
git commit -m "Fix lint/type issues for B16"
```

---

## Notes for the implementer

- **Do not modify the condensation cascades** (`_substrate_exclusion_cascade`, `_iit3_exclusion_cascade`, `_resolve_clique_*`). Exclusion records are computed in the `_exclusion_records` post-pass; the math is untouched (a spec non-goal).
- **Lazy imports**: `pyphi/models/complex.py` must import `_sia_node_indices` lazily inside the `node_indices` property, and `pyphi/substrate.py` must import `Complex` lazily inside the functions, to avoid a `substrate ⇄ models.complex` import cycle.
- **Macro is out of scope**: `pyphi/macro/search.py:complexes()` is a different function returning `ComplexesResult`; do not touch it.
- **`float(phi)`**: the SIA's `phi` is a `PyPhiFloat`; `float(...)` normalizes it for the lightweight record.
```
