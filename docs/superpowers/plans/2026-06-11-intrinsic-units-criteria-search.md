# Intrinsic-Unit Criteria and Bounded Grain Search — Implementation Plan

**Goal:** Implement macro sub-project 2 of the Marshall et al. 2024
intrinsic-units formalism: the unit criteria (Eqs 15-16) with verdict
objects, the competing-system set f(U^J, W^J), the bottom-up recursion,
the bounded valid-system set P(u), and the Eq 19 `complexes()` driver.

**Architecture:** Two new modules in `pyphi/macro/`. `criteria.py` holds
pure criteria logic (verdicts, the constituent system, the judgment
function); `search.py` holds the bounded enumeration and drivers and
depends on `criteria.py`, never the reverse. All phi_s evaluations in a
driver run share one memo keyed on the hashable `MacroSystem`. Builds
on SP1 (`MacroUnit`, `macro_tpms`, `MacroSystem`) without modifying it.

**Spec:** `docs/superpowers/specs/2026-06-11-intrinsic-units-criteria-search-design.md`

---

## Verified groundwork (experiments run during planning; do not re-derive)

All experiments under `config.override(**presets.iit4_2023)`.

1. **Anchors reproduce at 1e-13** (SP1's convention), not bitwise:
   - min: phi(A) = phi(B) = 0.0; phi(AB) = 0.005106576483955726 (exact);
     both-on macro = **0.7883339770634884** (committed 0.7883339770634886;
     2 ulp).
   - sfn: phi(A) = 0.02363345634846179 (exact); phi(AC) = 0.0.
   - sfnn: phi(A) = 0.02364098835678946, phi(AC) = 0.004863714555961184
     (committed ...627 / ...354; ~2e-16).
   - sfs: phi(A) = 0.023463717711822592, phi(AC) = 0.1675855507736177,
     phi(AB) = 0.6728123807299449 (all within 1e-15 of committed).
   - The dancing-couples TPM rule as given in the spec is confirmed
     correct (anchors reproduce from it).

2. **Complement mappings are exactly tied.** A mapping and its
   complement produce the same one-unit macro system up to state
   relabeling; phi_s came out **bitwise identical** for the min top pair
   and pairwise identical (within 2 ulp) for all 14 surjective tables
   (7 complement pairs). Under literal Eq 19 strict-tie semantics, an
   enumeration containing both members of each pair can never produce a
   complex from mapped systems. See Deviation D1.

3. **The committed bu anchors are internally stale.** With the bu TPM
   exactly as transcribed (little-endian rows, state (0,0,0)):
   - 2.0 pipeline: phi(A) = phi(B) = **1.0** (not 0.0); subsystem (C,)
     raises `StateUnreachableForwardsError` (C is forced ON given
     background (A,B) = (0,0)); pairs all 0.0; phi(ABC) =
     0.8300749985576875 (exact match).
   - Replaying the authors' own code path (old pyphi at their pinned
     rev `941c65a`, their committed `pyphi_config.yml`,
     `VALIDATE_SUBSYSTEM_STATES=False` as in their run script) gives the
     **same**: A = B = 1.0, C = 0.0, pairs 0.0, ABC = 0.8300749985576875.
   - Their committed `summary.txt` zeros reproduce **only** with
     `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI = False` (old pyphi's
     default), while their committed yml sets it **true** — and sfn's
     committed nonzero singletons (0.02363345634846179) require
     **true**. The authors' result sets are mutually inconsistent on
     this flag; bu's summary predates their committed config. The 2.0
     pipeline (iit4_2023 preset) matches the true-convention, which is
     the one SP1 anchored bit-for-bit on cg/bbx. See Deviation D2.

4. **Tie fixture pinned.** 3-unit substrate, A<->C symmetric
   (P(A') = 0.05 + 0.05A + 0.6B; P(B') = 0.05 + 0.05B + 0.3A + 0.3C;
   P(C') = 0.05 + 0.05C + 0.6B; state (0,0,0)): every system on
   footprint {A,B} has an isomorphic twin on {B,C}, overlapping at B.
   Measured: top systems are the one-unit macro (0,1,1,1)-mapped pair at
   0.3881829280978132 / 0.38818292809781296 (equal at precision 13);
   micro pairs at 0.1328031400246416; (A,C) micro pair at 0.0 (so the
   (0,2) footprint is NOT_INTEGRATED and contributes no pool units);
   all-micro triple at 0.08449862433339383. With
   `SearchBounds(max_constituents=2)` the driver outcome is: no
   complexes, exactly one tie pair.

5. **Cost:** cg 4-unit micro sia = 0.52 s; 2-unit macro sia = 0.01 s;
   1-unit = 0.005 s. The default-bounds cg driver evaluates ~350
   systems, nearly all below 3 units — minutes at worst (slow-marked).

6. **Environment.** `.python-version` pins 3.13.13, which is not
   available in this container; the venv is on 3.13.12. Use
   `uv run --no-sync` for every command (plain `uv run` fails trying to
   resolve 3.13.13). SP1 baseline confirmed green (83 passed).

## Deviations and pins requiring sign-off

These amend or refine the spec; each is forced by the experiments above.

- **D1 — Mapping canonicalization (amends spec counts).**
  `candidate_mappings` returns truth tables deduplicated up to
  complementation, canonicalized so the all-OFF sequence-state maps to
  macro state 0. Justification: a mapping and its complement define the
  same physical partition of sequence-states into two classes — the two
  macro state labels are conventional, the formalism is
  relabel-covariant, and phi_s is confirmed identical. Without dedup,
  Eq 19's strict inequality can never be satisfied by any mapped system
  (every system ties its complement-relabeled twin) and battery 2's
  "returns a macro complex" is unsatisfiable. Consequence: EXHAUSTIVE
  yields `2**(S-1) - 1` tables (min: **7**, not 14; cap-8: **127**, not
  254); FAMILIES drops complement duplicates (m=2 grain 1: **5** tables).
- **D2 — Battery 3 (bu) redesigned around the consistent convention.**
  Tests assert the values the 2.0 pipeline (and the authors' own pinned
  code under their committed config) actually produce: phi(A) = phi(B)
  = 1.0, C unreachable, pairs 0.0, ABC = 0.8300749985576875. The
  micro-unit exemption is still exercised (unit C has phi_s = 0 yet
  remains valid ground; the ABC system stays admissible in P(u)). The
  bu driver verdict follows the real values: **complexes = ({A}, {B})**
  — the singletons beat ABC (1.0 > 0.83), so the example's intended
  bottom-up story does not hold under the consistent convention. Queued
  for the authors alongside the SP1 TPM finding and the f subset
  question.
- **D3 — Unreachable-state policy.** A candidate system whose macro
  state is unreachable under its own TPM specifies no cause and cannot
  exist: phi_s = 0 semantics. `unit_integration` returns 0.0 (so such
  candidates are NOT_INTEGRATED); f and P(u) drop such systems (they
  can never beat anything, and they cannot be constructed as
  `MacroSystem` objects to be recorded).
- **D4 — Singleton decompositions emit only grain-raised variants.**
  A |V| = 1 candidate at tau' = 1 is its constituent relabeled, so
  variants are emitted for tau' in 2..max_update_grain only; singleton
  footprints are enumerated only when max_update_grain > 1 (spec).
  A 1-micro-constituent, grain-1 unit is a micro unit for gating
  purposes (trivially VALID) regardless of its mapping.
- **D5 — Recursion pool timing.** Decompositions draw from the pool as
  of the previous level (so `max_depth` bounds composition height); f
  draws from the incrementally updated pool, with footprints processed
  size-ascending within a level, so a candidate competes against every
  unit already validated at strictly finer footprints — including
  same-level ones (the Fig 3E one-shot-vs-meso competition works at
  depth 1).
- **D6 — ENUMERATE apportionment applies to derived candidates only.**
  Micro units always carry empty W (they enter as ground). Eq 12 is
  respected: a candidate's enumerated W always contains the union of
  its constituents' apportionments.
- **D7 — History contract.** Drivers require `micro_history` length
  exactly `max_update_grain ** max_depth` (`is_intrinsic_unit` and
  `competing_systems`: the max of that and the unit's constituent micro
  grain); a bare state is accepted when that is 1. Per-system
  evaluation trims to the trailing window each system needs.

## Standing constraints

- Commits via `git -c commit.gpgsign=false commit`; targeted `git add`
  only. If the hook reformats (output but no commit line), re-add the
  same files and commit again; check `git status` first.
- Ruff: no `dict()` calls; no Unicode math in Python strings/docstrings
  (write "phi_s", "P(u)", "union", "x", "-"); imports at top (tests
  included), added per-task; RUF005; SIM117; raw strings for regex in
  `pytest.raises(match=...)`; no unused args.
- `uv run --no-sync` for every command (see groundwork 6). Full
  verification at the end: `uv run --no-sync pytest` with NO path
  argument, plus the slow lane
  (`uv run --no-sync pytest test/test_macro_search.py --slow -m slow`).
  Baseline: 1954 passed, 20 skipped, 1 xfailed (+13 slow).
- pyright runs in the hook on `pyphi/` only (tests excluded).
- States are little-endian throughout.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `pyphi/macro/criteria.py` | create | `Reason`, `UnitVerdict`, `judge_candidate`, `canonical_units`, `constituent_system`, `unit_integration` |
| `pyphi/macro/search.py` | create | `SearchBounds`, `candidate_mappings`, recursion engine, `competing_systems`, `is_intrinsic_unit`, `intrinsic_units`, `valid_systems`, `complexes`, result types |
| `pyphi/macro/__init__.py` | edit | export the public surface |
| `test/test_macro_criteria.py` | create | Tasks 1-2 batteries; bu fixture |
| `test/test_macro_search.py` | create | Tasks 3-10 batteries; dancing-couples and tie fixtures |
| `changelog.d/intrinsic-units-search.feature.md` | create | changelog fragment |
| `ROADMAP.md` | edit | item-10 SP2 sub-entry marked landed |

---

## Task 1: Verdict types and `judge_candidate`

**Files:** `pyphi/macro/criteria.py` (new), `test/test_macro_criteria.py` (new)

### Step 1: failing test

Create `test/test_macro_criteria.py`:

```python
"""Tests for pyphi.macro.criteria: intrinsic-unit criteria (Eqs 15-16)."""

import pytest

from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import judge_candidate

# judge_candidate never introspects competitor systems, so opaque
# sentinels stand in for MacroSystem objects in these pure-logic tests.
S1, S2, S3 = object(), object(), object()


class TestJudgeCandidate:
    def test_valid_when_integrated_and_maximal(self):
        verdict = judge_candidate(0.5, [(S1, 0.1), (S2, 0.3)])
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        assert verdict.phi == 0.5
        assert verdict.witness is None
        assert verdict.witness_phi is None
        assert verdict.num_competitors == 2

    def test_not_integrated_when_phi_zero(self):
        verdict = judge_candidate(0.0, [(S1, 0.1)])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED
        assert verdict.witness is None
        assert verdict.num_competitors == 1

    def test_not_integrated_at_precision(self):
        # Positive but below precision: not strictly greater than zero.
        verdict = judge_candidate(1e-15, [])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED

    def test_not_maximal_carries_strongest_witness(self):
        verdict = judge_candidate(0.2, [(S1, 0.1), (S2, 0.7), (S3, 0.3)])
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_MAXIMAL
        assert verdict.witness is S2
        assert verdict.witness_phi == 0.7

    def test_tied_at_precision(self):
        verdict = judge_candidate(0.5, [(S1, 0.5 + 1e-15)])
        assert not verdict.valid
        assert verdict.reason is Reason.TIED
        assert verdict.witness is S1
        assert verdict.witness_phi == 0.5 + 1e-15

    def test_exact_tie(self):
        verdict = judge_candidate(0.5, [(S1, 0.5)])
        assert not verdict.valid
        assert verdict.reason is Reason.TIED

    def test_no_competitors_valid_iff_integrated(self):
        assert judge_candidate(0.5, []).valid
        assert not judge_candidate(0.0, []).valid

    def test_first_of_equal_witnesses_kept(self):
        verdict = judge_candidate(0.2, [(S1, 0.7), (S2, 0.7)])
        assert verdict.witness is S1

    def test_verdict_is_frozen(self):
        verdict = judge_candidate(0.5, [])
        with pytest.raises(AttributeError):
            verdict.valid = False
```

Run: `uv run --no-sync pytest test/test_macro_criteria.py -q`
Expect: collection error (module `pyphi.macro.criteria` does not exist).

### Step 2: implementation

Create `pyphi/macro/criteria.py`:

```python
"""Intrinsic-unit criteria (Marshall et al. 2024, Eqs. 15-16).

A candidate macro unit J with direct constituents ``V^J`` exists as one
unit only if its constituent system -- the system of the elements of
``V^J`` over the full universe, with everything else as background --
is integrated (Eq. 15) and strictly more irreducible than every
competing system that can be built within the unit's footprint
(Eq. 16). Both criteria are properties of the pair ``(V^J, W^J)``: the
candidate's own mapping and update grain do not enter, so mapped and
grained variants of one decomposition share a verdict.

This module holds the pure criteria logic. The competitor set
``f(U^J, W^J)`` is materialized by :mod:`pyphi.macro.search`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from pyphi import exceptions
from pyphi import utils
from pyphi.data_structures.pyphi_float import PyPhiFloat
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate


class Reason(Enum):
    """Why a candidate unit is valid or invalid."""

    VALID = "VALID"
    NOT_INTEGRATED = "NOT_INTEGRATED"
    NOT_MAXIMAL = "NOT_MAXIMAL"
    TIED = "TIED"


@dataclass(frozen=True)
class UnitVerdict:
    """The outcome of checking Eqs. 15-16 for one candidate decomposition.

    Attributes:
        valid: Whether the candidate satisfies both criteria.
        reason: ``VALID``, or which criterion failed: ``NOT_INTEGRATED``
            (Eq. 15), ``NOT_MAXIMAL`` or ``TIED`` (Eq. 16).
        phi: ``phi_s(v^J)``, the constituent system's integrated
            information.
        witness: The competitor that beat or tied the candidate, if any.
        witness_phi: The witness's ``phi_s``.
        num_competitors: Size of the competitor set ``f(U^J, W^J)``.
    """

    valid: bool
    reason: Reason
    phi: float
    witness: MacroSystem | None
    witness_phi: float | None
    num_competitors: int


def judge_candidate(
    phi: float, competitors: Iterable[tuple[MacroSystem, float]]
) -> UnitVerdict:
    """Eqs. 15-16 given ``phi_s(v^J)`` and the evaluated competitor set.

    All inequalities are strict at ``config.numerics.precision``; a
    candidate that ties its strongest competitor is invalid with reason
    ``TIED``.

    Args:
        phi: The candidate's ``phi_s(v^J)``.
        competitors: ``(system, phi_s)`` pairs for ``f(U^J, W^J)``.
    """
    competitors = tuple(competitors)
    if not utils.is_positive(phi):
        return UnitVerdict(
            valid=False,
            reason=Reason.NOT_INTEGRATED,
            phi=float(phi),
            witness=None,
            witness_phi=None,
            num_competitors=len(competitors),
        )
    best_system: MacroSystem | None = None
    best_phi = float("-inf")
    for system, competitor_phi in competitors:
        if best_system is None or float(competitor_phi) > best_phi:
            best_system = system
            best_phi = float(competitor_phi)
    if best_system is not None:
        if utils.eq(phi, best_phi):
            return UnitVerdict(
                valid=False,
                reason=Reason.TIED,
                phi=float(phi),
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
        if best_phi > float(phi):
            return UnitVerdict(
                valid=False,
                reason=Reason.NOT_MAXIMAL,
                phi=float(phi),
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
    return UnitVerdict(
        valid=True,
        reason=Reason.VALID,
        phi=float(phi),
        witness=None,
        witness_phi=None,
        num_competitors=len(competitors),
    )
```

(The imports of `exceptions`, `PyPhiFloat`, `MacroUnit`, `micro_unit`,
and `Substrate` are used from Task 2 onward; to keep each commit
ruff-clean, include in Task 1 only the imports Task 1 uses —
`collections.abc.Iterable`, `dataclass`, `Enum`, `utils`,
`MacroSystem` — and add the rest in Task 2.)

Run: `uv run --no-sync pytest test/test_macro_criteria.py -q`
Expect: 9 passed.

### Step 3: commit

```
git add pyphi/macro/criteria.py test/test_macro_criteria.py
git -c commit.gpgsign=false commit -m "Add intrinsic-unit verdict types and judgment (Eqs 15-16 logic)"
```

---

## Task 2: `constituent_system` and `unit_integration`

**Files:** `pyphi/macro/criteria.py`, `test/test_macro_criteria.py`

### Step 1: failing test

Append to `test/test_macro_criteria.py` (and extend its imports — all
imports at top of file):

```python
import numpy as np

from pyphi import config
from pyphi.conf import presets
from pyphi.macro.criteria import canonical_units
from pyphi.macro.criteria import constituent_system
from pyphi.macro.criteria import unit_integration
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from test.test_macro_tpm import MIN_TPM


def min_substrate():
    return Substrate(MIN_TPM, node_labels=("A", "B"))


def bu_substrate():
    """The authors' bottom-up example: 3 units, deterministic TPM.

    State (0, 0, 0). Note: the committed result set for this example was
    generated under old pyphi's SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI
    default (false), which contradicts the authors' committed config and
    their other result sets; the values asserted in this suite are the
    ones the consistent (flag-true) convention produces, verified
    against the authors' pinned pyphi revision during planning.
    """
    rows = [
        [1, 1, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]
    return Substrate(np.array(rows, dtype=float), node_labels=("A", "B", "C"))


class TestConstituentSystem:
    def test_micro_indices_become_micro_units(self):
        system = constituent_system(min_substrate(), (0, 1), ((0, 0),))
        assert system.units == (micro_unit(0), micro_unit(1))

    def test_constituent_order_is_canonical(self):
        a = constituent_system(min_substrate(), (1, 0), ((0, 0),))
        b = constituent_system(min_substrate(), (0, 1), ((0, 0),))
        assert a == b

    def test_bare_state_accepted_at_grain_one(self):
        system = constituent_system(min_substrate(), (0, 1), (0, 0))
        assert system.micro_history == ((0, 0),)

    def test_meso_constituent_keeps_full_definition(self):
        meso = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))
        system = constituent_system(min_substrate(), (meso,), ((0, 0),))
        assert system.units == (meso,)

    def test_history_trimmed_to_constituent_grain(self):
        # Grain-1 constituents with a length-2 history (as supplied for
        # a grain-2 candidate): only the trailing state is used.
        system = constituent_system(
            min_substrate(), (0, 1), ((1, 1), (0, 0))
        )
        assert system.micro_history == ((0, 0),)

    def test_history_too_short_rejected(self):
        meso = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        with pytest.raises(ValueError, match="history"):
            constituent_system(min_substrate(), (meso,), ((0, 0),))


class TestUnitIntegration:
    def test_min_pair_anchor(self):
        with config.override(**presets.iit4_2023):
            phi = unit_integration(min_substrate(), (0, 1), ((0, 0),))
            assert phi == pytest.approx(0.005106576483955726, abs=1e-13)

    def test_min_singletons_zero(self):
        with config.override(**presets.iit4_2023):
            assert unit_integration(min_substrate(), (0,), ((0, 0),)) == 0.0
            assert unit_integration(min_substrate(), (1,), ((0, 0),)) == 0.0

    def test_unreachable_state_gives_zero(self):
        # bu unit C is forced ON when (A, B) = (0, 0): the one-unit
        # system over C cannot exist in state 0, so phi_s is zero.
        with config.override(**presets.iit4_2023):
            assert unit_integration(bu_substrate(), (2,), ((0, 0, 0),)) == 0.0

    def test_bu_singleton_anchors(self):
        # See bu_substrate's docstring for why these are 1.0, not the
        # stale committed 0.0.
        with config.override(**presets.iit4_2023):
            assert unit_integration(bu_substrate(), (0,), ((0, 0, 0),)) == 1.0
            assert unit_integration(bu_substrate(), (1,), ((0, 0, 0),)) == 1.0


class TestCanonicalUnits:
    def test_sorted_by_footprint(self):
        units = canonical_units([micro_unit(1), micro_unit(0)])
        assert units == (micro_unit(0), micro_unit(1))
```

Run: `uv run --no-sync pytest test/test_macro_criteria.py -q`
Expect: ImportError (`canonical_units` etc. missing).

### Step 2: implementation

Append to `pyphi/macro/criteria.py` (with the Task-2 imports added at
the top: `exceptions`, `PyPhiFloat`, `MacroUnit`, `micro_unit`,
`Substrate`):

```python
def _as_unit(constituent: MacroUnit | int) -> MacroUnit:
    """A constituent as a unit: micro indices become identity units."""
    if isinstance(constituent, MacroUnit):
        return constituent
    return micro_unit(constituent)


def canonical_units(units: Iterable[MacroUnit]) -> tuple[MacroUnit, ...]:
    """The units of a system in canonical order.

    Sorting makes systems that differ only in unit order compare and
    hash equal, so memoized evaluations are shared.
    """
    return tuple(
        sorted(
            units,
            key=lambda unit: (
                unit.micro_constituents,
                unit.micro_grain,
                unit.mapping,
                unit.background_apportionment,
            ),
        )
    )


def constituent_system(
    substrate: Substrate,
    constituents: Iterable[MacroUnit | int],
    micro_history,
) -> MacroSystem:
    """The system of a unit's direct constituents (Eq. 15).

    Each element of ``V^J`` participates with its full definition: a
    micro index becomes an identity micro unit; a meso constituent
    keeps its mapping, grain, and apportionment. The system spans the
    full universe, with all remaining micro units as background.

    ``micro_history`` (oldest first; a bare state is accepted when the
    constituents have micro grain 1) may be longer than the
    constituents require; only the trailing window is used.
    """
    units = canonical_units(_as_unit(c) for c in constituents)
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        history = (history,)
    history = tuple(tuple(s) for s in history)
    needed = max(unit.micro_grain for unit in units)
    if len(history) < needed:
        raise ValueError(
            f"micro_history must have at least {needed} entries for "
            f"these constituents; got {len(history)}"
        )
    return MacroSystem.from_micro(
        substrate, units, history[len(history) - needed :]
    )


def unit_integration(
    substrate: Substrate,
    constituents: Iterable[MacroUnit | int],
    micro_history,
) -> PyPhiFloat:
    """``phi_s(v^J)``: the constituent system's integrated information (Eq. 15).

    A constituent system whose state is unreachable specifies no cause
    and cannot exist; its integration is zero.
    """
    try:
        system = constituent_system(substrate, constituents, micro_history)
    except exceptions.StateUnreachableError:
        return PyPhiFloat(0.0)
    return PyPhiFloat(system.sia().phi)
```

Run: `uv run --no-sync pytest test/test_macro_criteria.py -q`
Expect: all pass (~20 tests).

### Step 3: commit

```
git add pyphi/macro/criteria.py test/test_macro_criteria.py
git -c commit.gpgsign=false commit -m "Add constituent-system evaluation for Eq 15"
```

---

## Task 3: `SearchBounds`

**Files:** `pyphi/macro/search.py` (new), `test/test_macro_search.py` (new)

### Step 1: failing test

Create `test/test_macro_search.py`:

```python
"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import pytest

from pyphi.macro.search import SearchBounds


class TestSearchBounds:
    def test_defaults(self):
        bounds = SearchBounds()
        assert bounds.max_constituents == 4
        assert bounds.max_update_grain == 1
        assert bounds.max_depth == 1
        assert bounds.mappings == "FAMILIES"
        assert bounds.exhaustive_cap == 8
        assert bounds.apportionment == "NONE"
        assert bounds.max_background == 0

    def test_frozen(self):
        bounds = SearchBounds()
        with pytest.raises(AttributeError):
            bounds.max_depth = 2

    def test_max_micro_grain_composes(self):
        assert SearchBounds().max_micro_grain == 1
        assert SearchBounds(max_update_grain=2, max_depth=2).max_micro_grain == 4

    def test_max_constituents_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_constituents"):
            SearchBounds(max_constituents=0)

    def test_max_update_grain_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_update_grain"):
            SearchBounds(max_update_grain=0)

    def test_negative_max_depth_rejected(self):
        with pytest.raises(ValueError, match="max_depth"):
            SearchBounds(max_depth=-1)

    def test_unknown_mappings_policy_rejected(self):
        with pytest.raises(ValueError, match="mappings"):
            SearchBounds(mappings="ALL")

    def test_unknown_apportionment_policy_rejected(self):
        with pytest.raises(ValueError, match="apportionment"):
            SearchBounds(apportionment="ALWAYS")

    def test_enumerate_requires_max_background(self):
        with pytest.raises(ValueError, match="max_background"):
            SearchBounds(apportionment="ENUMERATE")
        assert (
            SearchBounds(apportionment="ENUMERATE", max_background=1).max_background
            == 1
        )
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: collection error (module missing).

### Step 2: implementation

Create `pyphi/macro/search.py`:

```python
"""Bounded search for intrinsic units and complexes (Marshall et al.
2024, Sec. 2.2.2).

The recursion starts from the micro units, which are axiomatically
valid (Eqs. 15-16 gate macroing only). Each level derives candidate
decompositions ``V`` from the previous level's pool of valid units and
judges each ``(V, W)`` pair once -- validity is a property of the
decomposition, independent of the candidate's own mapping and update
grain. Valid decompositions emit their mapped and grained variants
into the pool. Footprints are processed smallest-first, so the
competitor set ``f(U^J, W^J)`` always draws on every unit already
validated at strictly finer footprints.

``f(U^J, W^J)`` is the set of systems assembled from valid units whose
micro constituents are proper subsets of ``U^J`` and whose background
apportionments are non-overlapping subsets of ``W^J``, excluding the
candidate's own constituent system. The set ``P(u)`` extends the same
assembly to the whole universe (Eq. 18), and a member is a complex if
it strictly beats every other member whose micro constituents overlap
its own (Eq. 19). Candidate systems whose state is unreachable under
their own TPM specify no cause and cannot exist; they are dropped.

All ``phi_s`` evaluations within one driver run share a memo keyed on
the hashable :class:`~pyphi.macro.system.MacroSystem`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyphi import exceptions
from pyphi import utils
from pyphi.data_structures.pyphi_float import PyPhiFloat
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import UnitVerdict
from pyphi.macro.criteria import _as_unit
from pyphi.macro.criteria import canonical_units
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.system import MacroSystem
from pyphi.macro.tpm import _system_micro_indices
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate

_MAPPING_POLICIES = ("FAMILIES", "EXHAUSTIVE")
_APPORTIONMENT_POLICIES = ("NONE", "ENUMERATE")


@dataclass(frozen=True)
class SearchBounds:
    """Bounds on the intrinsic-unit search space.

    Attributes:
        max_constituents: Cap on ``|U^J|`` per candidate unit.
        max_update_grain: Largest update grain ``tau'`` per level.
        max_depth: Macroing levels above micro.
        mappings: ``"FAMILIES"`` (coarse-grainings and black-boxings)
            or ``"EXHAUSTIVE"`` (every surjective table, capped).
        exhaustive_cap: Largest sequence-state count for EXHAUSTIVE.
        apportionment: ``"NONE"`` or ``"ENUMERATE"`` (assign background
            micro units to derived candidates).
        max_background: Cap on apportioned units when enumerating.
    """

    max_constituents: int = 4
    max_update_grain: int = 1
    max_depth: int = 1
    mappings: str = "FAMILIES"
    exhaustive_cap: int = 8
    apportionment: str = "NONE"
    max_background: int = 0

    def __post_init__(self) -> None:
        if self.max_constituents < 1:
            raise ValueError(
                f"max_constituents must be >= 1; got {self.max_constituents}"
            )
        if self.max_update_grain < 1:
            raise ValueError(
                f"max_update_grain must be >= 1; got {self.max_update_grain}"
            )
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0; got {self.max_depth}")
        if self.mappings not in _MAPPING_POLICIES:
            raise ValueError(
                f"unknown mappings policy {self.mappings!r}; "
                f"expected one of {_MAPPING_POLICIES}"
            )
        if self.apportionment not in _APPORTIONMENT_POLICIES:
            raise ValueError(
                f"unknown apportionment policy {self.apportionment!r}; "
                f"expected one of {_APPORTIONMENT_POLICIES}"
            )
        if self.apportionment == "ENUMERATE" and self.max_background == 0:
            raise ValueError(
                'apportionment="ENUMERATE" requires max_background >= 1'
            )

    @property
    def max_micro_grain(self) -> int:
        """Largest micro grain a derived unit can reach (grains compose
        down the hierarchy)."""
        return self.max_update_grain**self.max_depth
```

(`itertools`, `exceptions`, `utils`, `PyPhiFloat`, the criteria
imports, `MacroSystem`, `_system_micro_indices`, `MacroUnit`,
`blackbox`, `coarse_grain`, `micro_unit`, and `Substrate` are used from
Task 4 onward; as in Task 1, add each import in the task that first
uses it to keep every commit ruff-clean. Task 3 itself needs only
`dataclass`.)

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: 10 passed.

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add SearchBounds for the intrinsic-unit search"
```

---

## Task 4: `candidate_mappings`

**Files:** `pyphi/macro/search.py`, `test/test_macro_search.py`

### Step 1: failing test

Append to `test/test_macro_search.py` (import `candidate_mappings` at
top):

```python
class TestCandidateMappings:
    def test_families_two_constituents_grain_one(self):
        tables = candidate_mappings(2, 1, SearchBounds())
        # Coarse-grainings (canonicalized: complement when the all-OFF
        # state maps to ON), then black-boxings, first-seen order:
        # on_counts {0} -> complement of (1,0,0,0) = at-least-one-ON;
        # {1} -> exactly-one-ON; {2} -> both-ON; {0,1}, {0,2}, {1,2} ->
        # duplicates of the first three; blackbox {0} -> constituent-0;
        # {1} -> constituent-1; {0,1} -> duplicate of both-ON.
        assert tables == (
            (0, 1, 1, 1),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        )

    def test_families_count_three_constituents(self):
        assert len(candidate_mappings(3, 1, SearchBounds())) == 13

    def test_families_higher_grain_blackbox_only(self):
        # Coarse-graining is defined at update grain 1 only.
        tables = candidate_mappings(1, 2, SearchBounds(max_update_grain=2))
        assert tables == ((0, 0, 1, 1),)

    def test_exhaustive_min_shape(self):
        tables = candidate_mappings(
            2, 1, SearchBounds(mappings="EXHAUSTIVE")
        )
        # 2**(4-1) - 1 = 7 canonical surjective tables.
        assert len(tables) == 7
        assert len(set(tables)) == 7
        for table in tables:
            assert table[0] == 0  # canonical: all-OFF maps to OFF
            assert 1 in table  # surjective
        assert (0, 0, 0, 1) in tables

    def test_exhaustive_cap_exceeded(self):
        with pytest.raises(ValueError, match="exhaustive_cap"):
            candidate_mappings(
                2,
                2,
                SearchBounds(mappings="EXHAUSTIVE", max_update_grain=2),
            )

    def test_all_tables_canonical_and_unique(self):
        for policy in ("FAMILIES", "EXHAUSTIVE"):
            tables = candidate_mappings(2, 1, SearchBounds(mappings=policy))
            assert len(set(tables)) == len(tables)
            assert all(t[0] == 0 for t in tables)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: ImportError.

### Step 2: implementation

Append to `pyphi/macro/search.py` (add the `itertools`, `blackbox`,
`coarse_grain` imports):

```python
def _canonical_table(table: tuple[int, ...]) -> tuple[int, ...]:
    """The representative of ``{table, complement}``.

    A mapping and its complement define the same partition of the
    constituents' sequence-states into two classes; the macro unit's
    two state labels are conventional and the analysis is invariant
    under relabeling. The representative maps the all-OFF sequence to
    macro state 0.
    """
    if table[0] == 1:
        return tuple(1 - entry for entry in table)
    return table


def candidate_mappings(
    num_constituents: int, update_grain: int, bounds: SearchBounds
) -> tuple[tuple[int, ...], ...]:
    """Deduplicated candidate truth tables for a unit shape.

    FAMILIES: every non-degenerate ``coarse_grain`` on-count set
    (update grain 1 only, by the family's definition) plus every
    nonempty ``blackbox`` output subset (any grain). EXHAUSTIVE: every
    surjective table when the sequence-state count is within
    ``exhaustive_cap``; ``ValueError`` above it.

    Tables are canonicalized up to state-label complementation (the
    all-OFF sequence maps to macro state 0) and deduplicated,
    preserving first-seen order.
    """
    tables: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(table: tuple[int, ...]) -> None:
        table = _canonical_table(table)
        if table not in seen:
            seen.add(table)
            tables.append(table)

    if bounds.mappings == "FAMILIES":
        if update_grain == 1:
            counts = tuple(range(num_constituents + 1))
            for size in range(1, len(counts)):
                for on_counts in itertools.combinations(counts, size):
                    add(coarse_grain(num_constituents, on_counts))
        for size in range(1, num_constituents + 1):
            for outputs in itertools.combinations(range(num_constituents), size):
                add(blackbox(num_constituents, update_grain, outputs))
    else:  # EXHAUSTIVE
        num_states = (2**num_constituents) ** update_grain
        if num_states > bounds.exhaustive_cap:
            raise ValueError(
                f"EXHAUSTIVE mappings for {num_constituents} constituents "
                f"at update grain {update_grain} require {num_states} "
                f"sequence-states, above exhaustive_cap="
                f"{bounds.exhaustive_cap}"
            )
        for index in range(1, 2**num_states - 1):
            add(tuple((index >> k) & 1 for k in range(num_states)))
    return tuple(tables)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: all pass.

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add bounded candidate-mapping enumeration"
```

---

## Task 5: Criteria checks — engine, `competing_systems`, `is_intrinsic_unit`

**Files:** `pyphi/macro/search.py`, `test/test_macro_search.py`

### Step 1: failing test

Append to `test/test_macro_search.py`, extending the top-of-file
imports to:

```python
import numpy as np

from pyphi import config
from pyphi import utils
from pyphi.conf import presets
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import unit_integration
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings
from pyphi.macro.search import competing_systems
from pyphi.macro.search import is_intrinsic_unit
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from test.test_macro_criteria import bu_substrate
from test.test_macro_criteria import min_substrate
from test.test_macro_tpm import _asymmetric_substrate
```

then the fixtures and tests:

```python
def dancing_couples(w_v):
    """4 units; P(ON next) = 0.05 + 0.05*self + 0.6*horizontal + w_v*vertical.

    Wiring by unit index: 0 -> h=1, v=2; 1 -> h=0, v=3; 2 -> h=3, v=0;
    3 -> h=2, v=1. The authors' Fig 2 scenarios are w_v = 0.0 (sfn),
    0.01 (sfnn), 0.25 (sfs), all in state (0, 0, 0, 0).
    """
    horizontal = {0: 1, 1: 0, 2: 3, 3: 2}
    vertical = {0: 2, 1: 3, 2: 0, 3: 1}
    n = 4
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = tuple((row >> k) & 1 for k in range(n))
        for i in range(n):
            tpm[row, i] = (
                0.05 + 0.05 * s[i] + 0.6 * s[horizontal[i]] + w_v * s[vertical[i]]
            )
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


SF_STATE = (0, 0, 0, 0)
AC = MacroUnit((0, 2), 1, coarse_grain(2, on_counts={2}))
AB = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))


class TestFig2Verdicts:
    """Battery 1: the three dancing-couples scenarios (authors'
    committed values, asserted at 1e-13)."""

    def test_sfn_not_integrated(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.0), AC, SF_STATE)
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED
        assert verdict.phi == pytest.approx(0.0, abs=1e-13)

    def test_sfn_singleton_anchor(self):
        with config.override(**presets.iit4_2023):
            phi = unit_integration(dancing_couples(0.0), (0,), (SF_STATE,))
        assert phi == pytest.approx(0.02363345634846179, abs=1e-13)

    def test_sfnn_not_maximal(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.01), AC, SF_STATE)
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_MAXIMAL
        assert verdict.phi == pytest.approx(0.004863714555961354, abs=1e-13)
        assert verdict.witness is not None
        assert len(verdict.witness.units) == 1
        assert verdict.witness_phi == pytest.approx(
            0.023640988356789627, abs=1e-13
        )
        assert verdict.num_competitors == 2

    def test_sfs_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.25), AC, SF_STATE)
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        assert verdict.phi == pytest.approx(0.16758555077361778, abs=1e-13)
        assert verdict.witness is None
        assert verdict.num_competitors == 2

    def test_sfs_horizontal_pair_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.25), AB, SF_STATE)
        assert verdict.valid
        assert verdict.phi == pytest.approx(0.6728123807299448, abs=1e-13)


class TestMicroExemption:
    def test_micro_unit_trivially_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(min_substrate(), micro_unit(0), (0, 0))
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        # min singletons have phi_s = 0, yet micro units are valid ground.
        assert verdict.phi == 0.0
        assert verdict.num_competitors == 0

    def test_micro_unit_with_unreachable_state_still_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(
                bu_substrate(), micro_unit(2), (0, 0, 0)
            )
        assert verdict.valid
        assert verdict.phi == 0.0


class TestGrainRaisedSingleton:
    def test_no_competitors_and_gated_by_integration(self):
        # Macroing over updates (Fig 3D): a singleton footprint admits
        # no proper-subset competitors, so the verdict reduces to Eq 15.
        unit = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        bounds = SearchBounds(max_update_grain=2)
        history = ((1, 0, 1, 0), (1, 0, 1, 0))
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(
                _asymmetric_substrate(), unit, history, bounds
            )
        assert verdict.num_competitors == 0
        assert verdict.valid == utils.is_positive(verdict.phi)
        assert verdict.reason in (Reason.VALID, Reason.NOT_INTEGRATED)


class TestCompetingSystems:
    def test_sfs_competitors_are_the_singletons(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), AC, SF_STATE)
        assert len(systems) == 2
        footprints = {
            tuple(u.micro_constituents for u in s.units) for s in systems
        }
        assert footprints == {((0,),), ((2,),)}

    def test_own_constituent_system_excluded(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), AC, SF_STATE)
        own = (micro_unit(0), micro_unit(2))
        assert all(s.units != own for s in systems)

    def test_micro_unit_has_no_competitors(self):
        with config.override(**presets.iit4_2023):
            assert competing_systems(min_substrate(), micro_unit(0), (0, 0)) == ()

    def test_all_member_footprints_proper_subsets(self):
        unit = MacroUnit((0, 1, 2), 1, coarse_grain(3, on_counts={3}))
        with config.override(**presets.iit4_2023):
            systems = competing_systems(bu_substrate(), unit, (0, 0, 0))
        footprint = set(unit.micro_constituents)
        for system in systems:
            for member in system.units:
                assert set(member.micro_constituents) < footprint


class TestVerdictMappingIndependence:
    """Battery 4: Eq 15 mapping-independence -- mapped and grained
    variants of one decomposition share the verdict."""

    def test_variants_share_verdict(self):
        variant_a = MacroUnit((0, 2), 1, coarse_grain(2, on_counts={1, 2}))
        variant_b = MacroUnit((0, 2), 1, blackbox(2, 1, (0,)))
        with config.override(**presets.iit4_2023):
            substrate = dancing_couples(0.25)
            verdicts = [
                is_intrinsic_unit(substrate, unit, SF_STATE)
                for unit in (AC, variant_a, variant_b)
            ]
        for verdict in verdicts[1:]:
            assert verdict.valid == verdicts[0].valid
            assert verdict.reason is verdicts[0].reason
            assert verdict.phi == verdicts[0].phi
            assert verdict.num_competitors == verdicts[0].num_competitors
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: ImportError.

### Step 2: implementation

Append to `pyphi/macro/search.py` (add the remaining imports listed in
Task 3's module header — everything except `_system_micro_indices`,
which arrives in Task 8):

```python
def _normalized_history(substrate, micro_history, required: int):
    """Validate and shape ``micro_history`` (oldest first).

    A bare state is accepted when ``required == 1``.
    """
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        if required != 1:
            raise ValueError(
                f"micro_history must be a sequence of {required} universe "
                "states (oldest first); got a bare state"
            )
        history = (history,)
    history = tuple(tuple(s) for s in history)
    if len(history) != required:
        raise ValueError(
            f"micro_history must have {required} entries (the maximum "
            f"micro grain admitted); got {len(history)}"
        )
    n = substrate.size
    for state in history:
        if len(state) != n or any(v not in (0, 1) for v in state):
            raise ValueError(
                f"each history entry must be a binary universe state of "
                f"length {n}; got {state}"
            )
    return history


def _system_of(substrate, units, micro_history) -> MacroSystem | None:
    """The system of ``units`` over the full universe, or None.

    Returns None when the system's state is unreachable under its own
    TPM: such a system specifies no cause and cannot exist (phi_s = 0).
    """
    units = canonical_units(units)
    needed = max(unit.micro_grain for unit in units)
    window = micro_history[len(micro_history) - needed :]
    try:
        return MacroSystem.from_micro(substrate, units, window)
    except exceptions.StateUnreachableError:
        return None


def _phi(substrate, units, micro_history, memo):
    """Memoized ``(system, phi_s)`` of the system of ``units``."""
    system = _system_of(substrate, units, micro_history)
    if system is None:
        return None, None
    if system not in memo:
        memo[system] = PyPhiFloat(system.sia().phi)
    return system, memo[system]


def _as_constituent(unit: MacroUnit) -> MacroUnit | int:
    """A pool unit as a constituent: identity micro units become bare
    indices, so derived units compare equal to hand-built ones."""
    if (
        len(unit.constituents) == 1
        and not isinstance(unit.constituents[0], MacroUnit)
        and unit.micro_grain == 1
        and unit.mapping == (0, 1)
        and not unit.background_apportionment
    ):
        return unit.constituents[0]
    return unit


def _assemble_systems(pool, background_cap: int):
    """Nonempty unit sets with pairwise-disjoint stakes (Eq. 18).

    Yields tuples in depth-first inclusion order over ``pool``.
    """
    out: list[tuple[MacroUnit, ...]] = []

    def extend(start, partial, claimed, apportioned):
        for k in range(start, len(pool)):
            unit = pool[k]
            stake = set(unit.micro_constituents) | set(
                unit.background_apportionment
            )
            if claimed & stake:
                continue
            total = apportioned + len(unit.background_apportionment)
            if total > background_cap:
                continue
            current = (*partial, unit)
            out.append(current)
            extend(k + 1, current, claimed | stake, total)

    extend(0, (), set(), 0)
    return out


def _decompositions(footprint, pool, *, allow_singleton: bool):
    """Sets of pool units with disjoint footprints whose union is
    ``footprint``, all sharing one micro grain."""
    remaining_all = set(footprint)
    candidates = [
        unit
        for unit in pool
        if set(unit.micro_constituents) <= remaining_all
    ]
    out: list[tuple[MacroUnit, ...]] = []

    def extend(partial, remaining):
        if not remaining:
            if len(partial) == 1 and not allow_singleton:
                return
            if len({unit.micro_grain for unit in partial}) == 1:
                out.append(tuple(partial))
            return
        first = min(remaining)
        for unit in candidates:
            fp = set(unit.micro_constituents)
            if first in fp and fp <= remaining:
                extend((*partial, unit), remaining - fp)

    extend((), remaining_all)
    return out


def _apportionments(n, footprint, inherited, bounds: SearchBounds):
    """Candidate ``W^J`` sets for a footprint.

    Always contains the union of the constituents' apportionments
    (Eq. 12). Under ENUMERATE, extends it with subsets of the remaining
    background up to ``max_background`` total.
    """
    inherited = tuple(sorted(inherited))
    if bounds.apportionment == "NONE":
        return (inherited,)
    if len(inherited) > bounds.max_background:
        return ()
    available = sorted(set(range(n)) - set(footprint) - set(inherited))
    out = []
    for size in range(bounds.max_background - len(inherited) + 1):
        for extra in itertools.combinations(available, size):
            out.append(tuple(sorted((*inherited, *extra))))
    return tuple(out)


def _f(substrate, V, W, footprint, pool, micro_history, bounds, memo):
    """``f(U^J, W^J)``: evaluated competitor systems (Eq. 16)."""
    fp = set(footprint)
    allowed = set(W)
    members = [
        unit
        for unit in pool
        if set(unit.micro_constituents) < fp
        and set(unit.background_apportionment) <= allowed
    ]
    own = canonical_units(V)
    competitors = []
    for combo in _assemble_systems(members, bounds.max_background):
        if canonical_units(combo) == own:
            continue
        system, phi = _phi(substrate, combo, micro_history, memo)
        if system is None:
            continue
        competitors.append((system, phi))
    return competitors


def _variants(V, W, bounds: SearchBounds):
    """Mapped and grained unit variants of a valid decomposition."""
    constituents = tuple(_as_constituent(u) for u in canonical_units(V))
    min_grain = 2 if len(V) == 1 else 1
    out = []
    for update_grain in range(min_grain, bounds.max_update_grain + 1):
        for mapping in candidate_mappings(len(V), update_grain, bounds):
            out.append(MacroUnit(constituents, update_grain, mapping, W))
    return out


def _judge(substrate, V, W, footprint, micro_history, bounds, pool, memo):
    _, phi = _phi(substrate, V, micro_history, memo)
    competitors = _f(
        substrate, V, W, footprint, pool, micro_history, bounds, memo
    )
    return judge_candidate(0.0 if phi is None else phi, competitors)


def _trivial_verdict(phi) -> UnitVerdict:
    return UnitVerdict(
        valid=True,
        reason=Reason.VALID,
        phi=0.0 if phi is None else float(phi),
        witness=None,
        witness_phi=None,
        num_competitors=0,
    )


def _is_micro(unit: MacroUnit) -> bool:
    """Micro for gating purposes: one micro constituent at grain 1.

    Eqs. 15-16 gate macroing only; micro units are axiomatically valid.
    """
    return len(unit.micro_constituents) == 1 and unit.micro_grain == 1


def _unit_history_requirement(unit: MacroUnit, bounds: SearchBounds) -> int:
    return max(bounds.max_micro_grain, unit.constituent_micro_grain)


def _f_for_unit(substrate, unit, V, micro_history, bounds, memo):
    pool, _ = _derive_units(
        substrate,
        micro_history,
        bounds,
        memo,
        within=unit.micro_constituents,
        proper=True,
    )
    return _f(
        substrate,
        V,
        unit.background_apportionment,
        unit.micro_constituents,
        pool,
        micro_history,
        bounds,
        memo,
    )


def competing_systems(
    substrate: Substrate,
    unit: MacroUnit,
    micro_history,
    bounds: SearchBounds = SearchBounds(),
) -> tuple[MacroSystem, ...]:
    """``f(U^J, W^J)`` materialized within the unit's footprint (Eq. 16)."""
    history = _normalized_history(
        substrate, micro_history, _unit_history_requirement(unit, bounds)
    )
    if _is_micro(unit):
        return ()
    memo: dict[MacroSystem, PyPhiFloat] = {}
    V = canonical_units(_as_unit(c) for c in unit.constituents)
    return tuple(
        system
        for system, _ in _f_for_unit(substrate, unit, V, history, bounds, memo)
    )


def is_intrinsic_unit(
    substrate: Substrate,
    unit: MacroUnit,
    micro_history,
    bounds: SearchBounds = SearchBounds(),
) -> UnitVerdict:
    """Eqs. 15-16 for one candidate; micro units return VALID trivially.

    The unit's own mapping and update grain are ignored (Eq. 15 is
    mapping-independent); the recursion is run restricted to the unit's
    footprint to build ``f(U^J, W^J)``.
    """
    history = _normalized_history(
        substrate, micro_history, _unit_history_requirement(unit, bounds)
    )
    memo: dict[MacroSystem, PyPhiFloat] = {}
    if _is_micro(unit):
        _, phi = _phi(substrate, (unit,), history, memo)
        return _trivial_verdict(phi)
    V = canonical_units(_as_unit(c) for c in unit.constituents)
    _, phi = _phi(substrate, V, history, memo)
    competitors = _f_for_unit(substrate, unit, V, history, bounds, memo)
    return judge_candidate(0.0 if phi is None else phi, competitors)
```

and the recursion engine plus its result carriers (also used by Task 6;
defined here because `_f_for_unit` needs `_derive_units`):

```python
@dataclass(frozen=True)
class DecompositionVerdict:
    """A judged candidate decomposition ``(V^J, W^J)``."""

    constituents: tuple[MacroUnit | int, ...]
    background_apportionment: tuple[int, ...]
    verdict: UnitVerdict


def _derive_units(
    substrate, micro_history, bounds, memo, *, within=None, proper=False
):
    """The intrinsic-unit recursion (paper p. 9), bounded by ``bounds``.

    Level 0 is the micro units. Each level derives candidate
    decompositions from the previous level's pool; the competitor set
    draws from the incrementally updated pool, with footprints
    processed smallest-first. Returns ``(pool, verdicts)``.
    """
    n = substrate.size
    indices = (
        tuple(range(n)) if within is None else tuple(sorted(within))
    )
    pool: list[MacroUnit] = [micro_unit(i) for i in indices]
    verdicts: list[DecompositionVerdict] = []
    for unit in pool:
        _, phi = _phi(substrate, (unit,), micro_history, memo)
        verdicts.append(
            DecompositionVerdict(
                constituents=(unit.constituents[0],),
                background_apportionment=(),
                verdict=_trivial_verdict(phi),
            )
        )
    seen: set = set()
    min_size = 1 if bounds.max_update_grain > 1 else 2
    for _level in range(bounds.max_depth):
        pool_prev = tuple(pool)
        emitted_any = False
        max_size = min(
            len(indices) - (1 if proper else 0), bounds.max_constituents
        )
        for size in range(min_size, max_size + 1):
            for footprint in itertools.combinations(indices, size):
                new_units: list[MacroUnit] = []
                decompositions = _decompositions(
                    footprint,
                    pool_prev,
                    allow_singleton=bounds.max_update_grain > 1,
                )
                for V in decompositions:
                    inherited = set().union(
                        *(set(u.background_apportionment) for u in V)
                    )
                    for W in _apportionments(n, footprint, inherited, bounds):
                        key = (canonical_units(V), W)
                        if key in seen:
                            continue
                        seen.add(key)
                        verdict = _judge(
                            substrate,
                            V,
                            W,
                            footprint,
                            micro_history,
                            bounds,
                            pool,
                            memo,
                        )
                        verdicts.append(
                            DecompositionVerdict(
                                constituents=tuple(
                                    _as_constituent(u)
                                    for u in canonical_units(V)
                                ),
                                background_apportionment=W,
                                verdict=verdict,
                            )
                        )
                        if verdict.valid:
                            new_units.extend(_variants(V, W, bounds))
                pool.extend(new_units)
                emitted_any = emitted_any or bool(new_units)
        if not emitted_any:
            break
    return tuple(pool), tuple(verdicts)
```

Run: `uv run --no-sync pytest test/test_macro_search.py test/test_macro_criteria.py -q`
Expect: all pass (battery 1 anchors at 1e-13).

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add intrinsic-unit criteria checks with footprint-restricted recursion"
```

---

## Task 6: `intrinsic_units` and `IntrinsicUnitsResult`

**Files:** `pyphi/macro/search.py`, `test/test_macro_search.py`

### Step 1: failing test

Append (import `DecompositionVerdict`, `IntrinsicUnitsResult`,
`intrinsic_units` at top), plus the tie fixture used here and in
Task 9:

```python
def tie_substrate():
    """3 units, exactly symmetric under swapping A and C.

    B couples to A and C identically; A and C couple to B only. Any
    system on footprint {A, B} has an isomorphic twin on {B, C}
    (overlapping at B), forcing exact phi ties.
    """
    n = 3
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = tuple((row >> k) & 1 for k in range(n))
        tpm[row, 0] = 0.05 + 0.05 * s[0] + 0.6 * s[1]
        tpm[row, 1] = 0.05 + 0.05 * s[1] + 0.3 * s[0] + 0.3 * s[2]
        tpm[row, 2] = 0.05 + 0.05 * s[2] + 0.6 * s[1]
    return Substrate(tpm, node_labels=("A", "B", "C"))


class TestIntrinsicUnits:
    def test_min_pool_and_verdicts(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        # 2 micro units + 5 canonical FAMILIES variants of (0, 1).
        assert len(result.units) == 7
        grouped = result.units_by_footprint()
        assert set(grouped) == {(0,), (1,), (0, 1)}
        assert {u.mapping for u in grouped[(0, 1)]} == set(
            candidate_mappings(2, 1, SearchBounds())
        )
        assert all(u.constituents == (0, 1) for u in grouped[(0, 1)])
        # One verdict per decomposition (not per variant): 2 micro + 1.
        assert len(result.verdicts) == 3
        pair = [v for v in result.verdicts if v.constituents == (0, 1)]
        assert len(pair) == 1
        assert pair[0].verdict.valid
        assert pair[0].verdict.phi == pytest.approx(
            0.005106576483955726, abs=1e-13
        )
        assert pair[0].verdict.num_competitors == 2

    def test_micro_units_axiomatically_valid(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        micro = [v for v in result.verdicts if len(v.constituents) == 1]
        assert len(micro) == 2
        for verdict in micro:
            assert verdict.verdict.valid
            assert verdict.verdict.phi == 0.0  # valid despite zero phi

    def test_tie_substrate_excludes_unintegrated_footprint(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(tie_substrate(), (0, 0, 0), bounds)
        grouped = result.units_by_footprint()
        # (0, 2) is causally disconnected: NOT_INTEGRATED, no variants.
        assert (0, 2) not in grouped
        assert set(grouped) == {(0,), (1,), (2,), (0, 1), (1, 2)}
        assert len(result.units) == 3 + 5 + 5
        rejected = [
            v for v in result.verdicts if v.constituents == (0, 2)
        ]
        assert len(rejected) == 1
        assert rejected[0].verdict.reason is Reason.NOT_INTEGRATED

    def test_bu_micro_only_pool(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(bu_substrate(), (0, 0, 0), SearchBounds())
        # Pairs are unintegrated; ABC is beaten by the singleton {A}
        # system at phi 1.0; pool stays micro.
        assert len(result.units) == 3
        full = [v for v in result.verdicts if v.constituents == (0, 1, 2)]
        assert len(full) == 1
        assert full[0].verdict.reason is Reason.NOT_MAXIMAL
        assert full[0].verdict.phi == pytest.approx(
            0.8300749985576875, abs=1e-13
        )
        assert full[0].verdict.witness_phi == 1.0
        # Unit C: unreachable state, phi 0, still valid ground.
        unit_c = [v for v in result.verdicts if v.constituents == (2,)]
        assert unit_c[0].verdict.valid
        assert unit_c[0].verdict.phi == 0.0

    def test_history_length_validated(self):
        with pytest.raises(ValueError, match="1 entries"):
            intrinsic_units(
                min_substrate(), ((0, 0), (0, 0)), SearchBounds()
            )
        with pytest.raises(ValueError, match="bare state"):
            intrinsic_units(
                min_substrate(), (0, 0), SearchBounds(max_update_grain=2)
            )

    def test_result_is_frozen(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        with pytest.raises(AttributeError):
            result.units = ()
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: ImportError (`intrinsic_units`, `IntrinsicUnitsResult`).

### Step 2: implementation

Append to `pyphi/macro/search.py`:

```python
@dataclass(frozen=True)
class IntrinsicUnitsResult:
    """The recursion's output: the valid-unit pool and every verdict.

    Attributes:
        units: All derived intrinsic units, micro units included, in
            derivation order (footprints smallest-first per level).
        verdicts: One :class:`DecompositionVerdict` per judged
            ``(V^J, W^J)`` candidate, micro units included.
    """

    units: tuple[MacroUnit, ...]
    verdicts: tuple[DecompositionVerdict, ...]

    def units_by_footprint(self) -> dict[tuple[int, ...], tuple[MacroUnit, ...]]:
        """The unit pool grouped by micro footprint."""
        grouped: dict[tuple[int, ...], list[MacroUnit]] = {}
        for unit in self.units:
            grouped.setdefault(unit.micro_constituents, []).append(unit)
        return {k: tuple(v) for k, v in grouped.items()}


def intrinsic_units(
    substrate: Substrate, micro_history, bounds: SearchBounds
) -> IntrinsicUnitsResult:
    """The recursion's fixed point: the valid-unit pool plus all verdicts."""
    history = _normalized_history(
        substrate, micro_history, bounds.max_micro_grain
    )
    memo: dict[MacroSystem, PyPhiFloat] = {}
    units, verdicts = _derive_units(substrate, history, bounds, memo)
    return IntrinsicUnitsResult(units=units, verdicts=verdicts)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: all pass.

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add the intrinsic-unit recursion driver"
```

---

## Task 7: `valid_systems` (the bounded P(u))

**Files:** `pyphi/macro/search.py`, `test/test_macro_search.py`

### Step 1: failing test

Append (import `valid_systems` at top):

```python
def assert_eq18(system):
    """Eq 18: stakes (footprint union apportionment) pairwise disjoint."""
    claimed = set()
    for unit in system.units:
        stake = set(unit.micro_constituents) | set(
            unit.background_apportionment
        )
        assert not (claimed & stake)
        claimed |= stake


class TestValidSystems:
    def test_min_count_and_eq18(self):
        with config.override(**presets.iit4_2023):
            systems = valid_systems(min_substrate(), (0, 0), SearchBounds())
        # {A}, {B}, {A,B} plus the 5 one-unit mapped variants.
        assert len(systems) == 8
        for system in systems:
            assert_eq18(system)

    def test_bu_drops_unreachable_singleton(self):
        with config.override(**presets.iit4_2023):
            systems = valid_systems(bu_substrate(), (0, 0, 0), SearchBounds())
        # 7 micro combinations minus the unconstructable {C}.
        assert len(systems) == 6
        assert all(
            tuple(u.micro_constituents for u in s.units) != ((2,),)
            for s in systems
        )

    def test_tie_substrate_count(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            systems = valid_systems(tie_substrate(), (0, 0, 0), bounds)
        # 7 micro combos + 5 [alpha_AB] + 5 [alpha_AB, C] + 5 [alpha_BC]
        # + 5 [alpha_BC, A].
        assert len(systems) == 27
        for system in systems:
            assert_eq18(system)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: ImportError.

### Step 2: implementation

Append to `pyphi/macro/search.py`:

```python
def valid_systems(
    substrate: Substrate, micro_history, bounds: SearchBounds
) -> tuple[MacroSystem, ...]:
    """The bounded ``P(u)``: every Eq-18-compatible system of intrinsic
    units, evaluated over the full universe with everything else as
    background. Systems whose state is unreachable are dropped."""
    history = _normalized_history(
        substrate, micro_history, bounds.max_micro_grain
    )
    memo: dict[MacroSystem, PyPhiFloat] = {}
    units, _ = _derive_units(substrate, history, bounds, memo)
    systems = []
    for combo in _assemble_systems(list(units), bounds.max_background):
        system = _system_of(substrate, combo, history)
        if system is not None:
            systems.append(system)
    return tuple(systems)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: all pass.

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add the bounded valid-system set (Eq 18)"
```

---

## Task 8: `complexes` driver, records, end-to-end goldens

**Files:** `pyphi/macro/search.py`, `test/test_macro_search.py`

### Step 1: failing test

Append (import `ComplexesResult`, `EvaluationRecord`, `complexes`, and
`MacroSystem` plus `from pyphi.macro import MacroSystem` is NOT needed
— import `MacroSystem` from `pyphi.macro.system` at top):

```python
class TestMinDriver:
    """Battery 2: min end-to-end with EXHAUSTIVE mappings (7 canonical
    tables after complement dedup)."""

    def test_macro_complex_found(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        assert len(result.complexes) == 1
        winner = result.complexes[0]
        # The argmax mapping is the authors' both-on coarse-graining,
        # in canonical form. Golden recorded at implementation time;
        # sanity: equals the committed both-on macro phi
        # (0.7883339770634886) at 1e-13.
        assert winner.units == (MacroUnit((0, 1), 1, (0, 0, 0, 1)),)
        phis = {r.system: r.phi for r in result.records}
        assert phis[winner] == pytest.approx(0.7883339770634884, abs=1e-13)
        assert result.ties == ()

    def test_records_contain_micro_pair_anchor(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        by_units = {r.system.units: r.phi for r in result.records}
        assert by_units[(micro_unit(0), micro_unit(1))] == pytest.approx(
            0.005106576483955726, abs=1e-13
        )

    def test_records_match_independent_recomputation(self):
        # Battery 4: memoized phi equals a fresh evaluation.
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
            for record in result.records[:3]:
                fresh = MacroSystem.from_micro(
                    record.system.micro_substrate,
                    record.system.units,
                    record.system.micro_history,
                )
                assert fresh.sia().phi == pytest.approx(
                    record.phi, abs=1e-13
                )

    def test_every_record_satisfies_eq18(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        for record in result.records:
            assert_eq18(record.system)


class TestBuDriver:
    """Battery 3: micro-exemption under the consistent convention (see
    bu_substrate's docstring). The full micro system is admissible and
    reproduces the committed phi, but the singleton systems {A} and {B}
    (phi 1.0) beat it, so they are the complexes -- golden recorded at
    implementation time."""

    def test_micro_system_admissible_and_anchored(self):
        with config.override(**presets.iit4_2023):
            result = complexes(bu_substrate(), (0, 0, 0), SearchBounds())
        by_units = {r.system.units: r.phi for r in result.records}
        full = tuple(micro_unit(i) for i in range(3))
        assert by_units[full] == pytest.approx(0.8300749985576875, abs=1e-13)

    def test_complexes_are_the_strong_singletons(self):
        with config.override(**presets.iit4_2023):
            result = complexes(bu_substrate(), (0, 0, 0), SearchBounds())
        footprints = {
            tuple(u.micro_constituents for u in s.units)
            for s in result.complexes
        }
        assert footprints == {((0,),), ((1,),)}
        phis = {r.system: r.phi for r in result.records}
        assert all(phis[s] == 1.0 for s in result.complexes)
        assert result.ties == ()

    def test_empty_complexes_is_a_result_not_an_error(self):
        # A universe with nothing above precision everywhere would give
        # an empty-complexes result; emulate with max_depth=0 on min,
        # where P(u) = micro systems all at phi 0 except the pair.
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        assert isinstance(result, ComplexesResult)
        # The micro pair (phi 0.0051) beats the overlapping singletons
        # (phi 0): it is the only complex at depth 0.
        assert len(result.complexes) == 1
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q`
Expect: ImportError.

### Step 2: implementation

Append to `pyphi/macro/search.py` (add the `_system_micro_indices` and
`utils` imports):

```python
@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluated system and its ``phi_s``."""

    system: MacroSystem
    phi: float


@dataclass(frozen=True)
class ComplexesResult:
    """The Eq. 19 outcome over the bounded candidate space.

    Attributes:
        complexes: The winners -- members of ``P(u)`` that strictly
            beat every other member with overlapping micro
            constituents. Mutually disjoint by construction.
        records: Every system evaluated during the run (criteria checks
            included) with its ``phi_s``, in evaluation order.
        ties: Pairs of overlapping systems that would each be a complex
            but for their mutual tie at precision.
    """

    complexes: tuple[MacroSystem, ...]
    records: tuple[EvaluationRecord, ...]
    ties: tuple[tuple[MacroSystem, MacroSystem], ...]


def complexes(
    substrate: Substrate,
    micro_history,
    bounds: SearchBounds = SearchBounds(),
) -> ComplexesResult:
    """Eq. 19 over the bounded candidate space -- the one-call driver."""
    history = _normalized_history(
        substrate, micro_history, bounds.max_micro_grain
    )
    memo: dict[MacroSystem, PyPhiFloat] = {}
    units, _ = _derive_units(substrate, history, bounds, memo)
    evaluated: list[tuple[MacroSystem, PyPhiFloat]] = []
    for combo in _assemble_systems(list(units), bounds.max_background):
        system, phi = _phi(substrate, combo, history, memo)
        if system is not None:
            evaluated.append((system, phi))
    footprints = [
        set(_system_micro_indices(system.units)) for system, _ in evaluated
    ]

    def overlapping(i):
        return [
            j
            for j in range(len(evaluated))
            if j != i and footprints[i] & footprints[j]
        ]

    tops = [
        i
        for i, (_, phi) in enumerate(evaluated)
        if all(
            utils.eq(phi, evaluated[j][1])
            or float(phi) > float(evaluated[j][1])
            for j in overlapping(i)
        )
    ]
    ties: list[tuple[MacroSystem, MacroSystem]] = []
    tied: set[int] = set()
    for a, b in itertools.combinations(tops, 2):
        if footprints[a] & footprints[b] and utils.eq(
            evaluated[a][1], evaluated[b][1]
        ):
            ties.append((evaluated[a][0], evaluated[b][0]))
            tied.add(a)
            tied.add(b)
    winners = tuple(evaluated[i][0] for i in tops if i not in tied)
    records = tuple(
        EvaluationRecord(system=system, phi=float(phi))
        for system, phi in memo.items()
    )
    return ComplexesResult(complexes=winners, records=records, ties=tuple(ties))
```

Execution-time check for the two recorded goldens (spec instruction):
before trusting the hard-coded values, print them once —

```
uv run --no-sync python -c "
from pyphi import config; from pyphi.conf import presets
from pyphi.macro.search import SearchBounds, complexes
import sys; sys.path.insert(0, '.')
from test.test_macro_criteria import min_substrate, bu_substrate
with config.override(**presets.iit4_2023):
    r = complexes(min_substrate(), (0, 0), SearchBounds(mappings='EXHAUSTIVE'))
    print('min winners:', [(s.units, p) for s in r.complexes for (sys2, p) in [(s, {rec.system: rec.phi for rec in r.records}[s])]])
    r2 = complexes(bu_substrate(), (0, 0, 0), SearchBounds())
    print('bu winners:', [tuple(u.micro_constituents for u in s.units) for s in r2.complexes])
"
```

If the printed min winner phi differs from 0.7883339770634884 by more
than 1e-13, or the structure differs from the test's assertions, STOP
and report rather than adjusting silently.

Run: `uv run --no-sync pytest test/test_macro_search.py test/test_macro_criteria.py -q`
Expect: all pass.

### Step 3: commit

```
git add pyphi/macro/search.py test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add the Eq 19 complexes driver with evaluation records"
```

---

## Task 9: Tie battery (no-complex-on-tie)

**Files:** `test/test_macro_search.py`

### Step 1: failing test (it should pass immediately if Tasks 5-8 are
correct — run it to confirm; if it fails, that is a real bug to fix,
not a test to adjust)

Append:

```python
class TestTiePath:
    """Battery 5: the exactly-symmetric fixture. Every system on
    footprint {A,B} has a permutation-identical twin on {B,C}; the top
    pair (the (0,1,1,1)-mapped one-unit systems, measured at
    ~0.3881829280978132 during planning) overlap at B and tie at
    precision, so neither is a complex and nothing else can beat them."""

    def test_no_complex_on_tie(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            result = complexes(tie_substrate(), (0, 0, 0), bounds)
        assert result.complexes == ()
        assert len(result.ties) == 1
        a, b = result.ties[0]
        assert {
            tuple(u.micro_constituents for u in s.units) for s in (a, b)
        } == {((0, 1),), ((1, 2),)}
        assert all(
            s.units[0].mapping == (0, 1, 1, 1) for s in (a, b)
        )
        phis = {r.system: r.phi for r in result.records}
        assert utils.eq(phis[a], phis[b])
        assert phis[a] == pytest.approx(0.3881829280978132, abs=1e-13)
```

Run: `uv run --no-sync pytest test/test_macro_search.py -q -k TiePath`
Expect: pass (driver logic already complete). Then run the full file.

### Step 2: commit

```
git add test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add tie-path coverage for the Eq 19 driver"
```

---

## Task 10: Cost guard (slow) on the cg substrate

**Files:** `test/test_macro_search.py`

### Step 1: test

Append (import `CG_TPM` from `test.test_macro_tpm` at top):

```python
@pytest.mark.slow
class TestCostGuard:
    """Battery 6: the full default-bounds driver on the cg substrate
    terminates and its record reproduces the SP1-anchored micro panel."""

    def test_default_driver_on_cg(self):
        with config.override(**presets.iit4_2023):
            substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
            result = complexes(substrate, (0, 0, 0, 0))
        by_units = {r.system.units: r.phi for r in result.records}
        panel = {
            (micro_unit(0),): 0.003976279885291341,
            (micro_unit(0), micro_unit(1)): 0.044088890564147803,
            tuple(micro_unit(i) for i in range(4)): 0.02015654077792439,
        }
        for units, expected in panel.items():
            assert by_units[units] == pytest.approx(expected, abs=1e-13)
        for record in result.records:
            assert_eq18(record.system)
        # Driver-outcome golden: recorded at implementation time after
        # inspecting the result (see plan Task 10); asserted here so the
        # outcome cannot drift silently.
        # <GOLDEN ASSERTIONS ADDED AT EXECUTION TIME>
```

Execution step: run once with
`uv run --no-sync pytest test/test_macro_search.py --slow -m slow -q -s`
after temporarily printing `result.complexes` / `result.ties` (sizes,
unit structures, phis), then replace the
`<GOLDEN ASSERTIONS ADDED AT EXECUTION TIME>` comment with concrete
assertions on what the driver actually returns (number of complexes,
their unit structure, their phi at 1e-13 — or the tie structure if the
cg substrate's A-B / C-D symmetry produces a top tie). Report the
recorded outcome in the final summary.

Run: `uv run --no-sync pytest test/test_macro_search.py --slow -m slow -q`
Expect: 1 passed (within minutes; the 4-unit evaluations dominate).

### Step 2: commit

```
git add test/test_macro_search.py
git -c commit.gpgsign=false commit -m "Add slow cost-guard for the default-bounds driver"
```

---

## Task 11: Exports, changelog, ROADMAP

**Files:** `pyphi/macro/__init__.py`,
`changelog.d/intrinsic-units-search.feature.md`, `ROADMAP.md`

### Step 1: exports

Replace `pyphi/macro/__init__.py` with:

```python
"""Intrinsic-units macro framework (Marshall et al. 2024).

Macro units are defined by sliding-window state mappings over their
micro constituents; macro cause and effect TPMs are built by the
four-step construction (Eqs. 26-40) and analyzed by the IIT 4.0
pipeline exactly as micro systems are. The intrinsic-unit criteria
(Eqs. 15-16) and the bounded grain search (Eq. 19) decide which units
and which grain are intrinsic for a substrate in a state.
"""

from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import UnitVerdict
from pyphi.macro.criteria import constituent_system
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.criteria import unit_integration
from pyphi.macro.search import ComplexesResult
from pyphi.macro.search import DecompositionVerdict
from pyphi.macro.search import EvaluationRecord
from pyphi.macro.search import IntrinsicUnitsResult
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings
from pyphi.macro.search import competing_systems
from pyphi.macro.search import complexes
from pyphi.macro.search import intrinsic_units
from pyphi.macro.search import is_intrinsic_unit
from pyphi.macro.search import valid_systems
from pyphi.macro.system import MacroSystem
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit

__all__ = [
    "ComplexesResult",
    "DecompositionVerdict",
    "EvaluationRecord",
    "IntrinsicUnitsResult",
    "MacroSystem",
    "MacroUnit",
    "Reason",
    "SearchBounds",
    "UnitVerdict",
    "blackbox",
    "candidate_mappings",
    "coarse_grain",
    "competing_systems",
    "complexes",
    "constituent_system",
    "intrinsic_units",
    "is_intrinsic_unit",
    "judge_candidate",
    "macro_tpms",
    "micro_unit",
    "unit_integration",
    "valid_systems",
]
```

Add a smoke test to `test/test_macro_search.py`:

```python
def test_public_surface_importable():
    import pyphi.macro as macro

    for name in (
        "SearchBounds",
        "complexes",
        "intrinsic_units",
        "is_intrinsic_unit",
        "judge_candidate",
        "unit_integration",
        "valid_systems",
    ):
        assert hasattr(macro, name)
```

### Step 2: changelog fragment

`changelog.d/intrinsic-units-search.feature.md`:

```markdown
Added the intrinsic-unit criteria and bounded grain search from
Marshall et al. 2024, Sec. 2.2.2: `pyphi.macro.criteria` (Eqs. 15-16
verdicts with witnesses, `unit_integration`, `judge_candidate`) and
`pyphi.macro.search` (`SearchBounds`, bounded mapping enumeration with
complement-canonical truth tables, the intrinsic-unit recursion
`intrinsic_units`, the valid-system set `valid_systems`, and the
one-call Eq. 19 driver `complexes` returning winners, ties, and the
full evaluation record).
```

### Step 3: ROADMAP

Rewrite the item-10 SP2 bullet in the SP1 sub-entry's landed style:
spec/plan/implementation commit hashes; summary of the two modules; the
mapping-canonicalization pin (complement relabeling confirmed
phi-invariant, enumeration deduplicated); the bu finding (committed
result set generated under old pyphi's
`SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` default, contradicting the
committed config and the other result sets; tests pin the consistent
convention's values, complexes = {A}, {B}); both queued for the
authors together with the f(U^J, W^J) subset question.

### Step 4: full verification

```
uv run --no-sync pytest -q                # full suite incl. doctest sweep
uv run --no-sync pytest test/test_macro_search.py --slow -m slow -q
```

Expect: baseline (1954 passed, 20 skipped, 1 xfailed) plus the new
tests, zero failures; slow lane green.

### Step 5: commit

```
git add pyphi/macro/__init__.py test/test_macro_search.py changelog.d/intrinsic-units-search.feature.md ROADMAP.md
git -c commit.gpgsign=false commit -m "Export the macro search surface; record SP2 in changelog and roadmap"
```

---

## Self-review notes

- Coverage against the spec's six batteries: 1 -> Task 5; 2 -> Task 8
  (min); 3 -> Task 8 (bu, redesigned per D2); 4 -> Tasks 5, 6, 8
  (mapping-independence, memo consistency, Eq 18, micro-trivial,
  recompute); 5 -> Task 1 (TIED verdicts) + Task 9 (driver tie); 6 ->
  Task 10.
- Error cases from the spec: EXHAUSTIVE over cap (Task 4); unknown
  policies, ENUMERATE without budget, bounds domain checks (Task 3);
  short history (Tasks 2, 6); empty/no-complex results are values, not
  errors (Task 8).
- ENUMERATE apportionment is implemented (Task 5: `_apportionments`,
  budget enforcement in `_assemble_systems`) but has no published
  anchor; it is exercised by the Task 3 validation tests and the Eq 12
  inheritance is enforced by construction. First anchors arrive in SP3.
- Names cross-checked across tasks: `Reason`, `UnitVerdict`,
  `judge_candidate`, `canonical_units`, `constituent_system`,
  `unit_integration` (criteria); `SearchBounds`, `candidate_mappings`,
  `competing_systems`, `is_intrinsic_unit`, `DecompositionVerdict`,
  `IntrinsicUnitsResult`, `intrinsic_units`, `valid_systems`,
  `EvaluationRecord`, `ComplexesResult`, `complexes` (search).
- Two goldens are recorded at execution time per the spec: the min
  argmax (pre-measured 0.7883339770634884, hard-coded in Task 8 with a
  verification print) and the cg driver outcome (Task 10). The bu
  driver verdict (complexes = {A}, {B}) is derived from planning
  measurements and asserted in Task 8.
