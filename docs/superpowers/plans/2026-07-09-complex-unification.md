# Complex Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement
`docs/superpowers/specs/2026-07-09-complex-unification-design.md`: one
condensation core shared by the micro and macro complexes drivers (fixing the
macro driver's literal-Eq.-19 semantics, which computes only the first
condensation layer), S1 Composition tie escalation for macro cliques with
big-Φ computed once per content fingerprint, and `Complex` as the winner type
at both doors.

**Architecture:** A new `pyphi/condensation.py` receives the cascade
machinery from `pyphi/substrate.py` and generalizes it over a `Candidate`
record (micro footprint as the overlap currency; providers for the SIA and
the system, so the cascade never rebuilds candidates from indices — a macro
candidate cannot be rebuilt that way). `pyphi.substrate.complexes` and
`pyphi.macro.complexes` both feed candidates to the shared cascade.
`ComplexesResult.complexes` becomes `tuple[Complex, ...]`; `Complex` and
`ExcludedCandidate` gain `units`.

**Tech Stack:** Python 3.13, numpy, msgspec, pytest, Hypothesis. No new
dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start — suggested: `complex-unification`; base on
  `main`).
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on φ values — **except** where a test's claim is exact identity of two
  code paths (the shadow-equality and both-doors tests); those use exact `==`
  deliberately. Do not "fix" them to `approx`.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task that makes the change.
- Docstrings describe final state only — NumPy style, no migration narrative,
  no planning artifacts (no task numbers, no spec references, no
  design-alternative discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- All macro tests run under `config.override(**presets.iit4_2023)` (the
  existing pattern in `test/macro/`).
- The final verification (Task 7) must run `uv run pytest` **with no path
  argument** at least once (testpaths + doctest sweep; bare-path invocations
  skip doctests).

---

### Task 1: Move the condensation machinery to `pyphi/condensation.py`

Pure move, no behavior change: the private cascade functions leave
`pyphi/substrate.py`; the public drivers stay in `substrate.py` and import
from the new module. `_sia_node_indices` moves too (the new module must not
import `substrate` at module level — `substrate` will import `condensation`).

**Files:**
- Create: `pyphi/condensation.py`
- Modify: `pyphi/substrate.py` (delete moved functions; import from
  `pyphi.condensation`)
- Modify: `pyphi/models/complex.py` (one lazy import re-pointed)
- Test: existing suite (`test/test_substrate.py`, `test/macro/`) — a pure
  move needs no new tests

**Interfaces:**
- Produces: module `pyphi.condensation` exposing (same names, same
  signatures as today): `_sia_node_indices`, `_config_iit_version`,
  `_accept`, `_phi_groups`, `_find_overlap_cliques`, `_big_phi_of_sia`,
  `_resolve_clique_by_big_phi`, `_substrate_exclusion_cascade`,
  `_resolve_clique_iit3`, `_iit3_exclusion_cascade`, `_exclusion_records`.

- [ ] **Step 1: Create `pyphi/condensation.py`**

Module docstring plus the functions below moved **verbatim** from
`pyphi/substrate.py` (current locations given for extraction; copy the bodies
exactly, then adjust only imports):

| function | current location in `pyphi/substrate.py` |
|---|---|
| `_sia_node_indices` | line 812 |
| `_exclusion_records` | line 824 |
| `_config_iit_version` | line 903 |
| `_accept` | line 909 |
| `_phi_groups` | line 918 |
| `_find_overlap_cliques` | line 935 |
| `_big_phi_of_sia` | line 962 |
| `_resolve_clique_by_big_phi` | line 977 |
| `_substrate_exclusion_cascade` | line 1005 |
| `_resolve_clique_iit3` | line 1033 |
| `_iit3_exclusion_cascade` | line 1054 |

Module header:

```python
# condensation.py
"""Condensation of candidate systems into complexes.

Implements the recursive exclusion cascade (Marshall, Albantakis, Tononi
2023, Algorithm A1; Albantakis et al. 2023, Exclusion): walk candidates in
descending φₛ tiers, accept each tier's overlap-clique winners, and drop
candidates that overlap an accepted complex. Ties within a clique escalate
to Composition (big Φ) per the S1 tie-resolution supplement; a clique whose
Φ also ties fails exclusion — its members are removed, but their units stay
available to lower-φ candidates in later tiers.
"""

from __future__ import annotations

from typing import Any, Iterable
```

Inside the moved functions, keep the existing lazy imports
(`from pyphi.system import System`, `from pyphi import resolve_ties`, etc.)
exactly as they are; the two type-only references to `Substrate` in
signatures (`_big_phi_of_sia`, `_resolve_clique_by_big_phi`,
`_substrate_exclusion_cascade`, `_iit3_exclusion_cascade`) become `Any` (the
module must not import `pyphi.substrate`).

- [ ] **Step 2: Re-point `pyphi/substrate.py`**

Delete the moved functions and add at the top of the module:

```python
from pyphi.condensation import (
    _exclusion_records,
    _iit3_exclusion_cascade,
    _sia_node_indices,
    _substrate_exclusion_cascade,
)
```

(`complexes` also calls `_config_iit_version` — import it too.) Every other
use of `_sia_node_indices` inside `substrate.py` (e.g. `maximal_complex`,
`_sia_node_indices` consumers) now refers to the imported name.

- [ ] **Step 3: Re-point `pyphi/models/complex.py`**

In `Complex.node_indices` change the lazy import:

```python
from pyphi.condensation import _sia_node_indices
```

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest test/test_substrate.py test/macro/ -x -q`
Expected: PASS (identical behavior; pure move).

- [ ] **Step 5: Commit**

```bash
git add pyphi/condensation.py pyphi/substrate.py pyphi/models/complex.py
git commit -m "Move the exclusion-cascade machinery into pyphi/condensation.py"
```

---

### Task 2: Generalize the cascade over `Candidate` records

The cascade stops consuming raw SIAs and consumes `Candidate` records:
micro footprint (the overlap currency — macro SIA `node_indices` are
macro-unit positions and must not leak into overlap logic), φ, and providers
for the SIA and the system. The clique resolver computes big Φ **once per
content fingerprint** and skips escalation entirely for single-fingerprint
cliques. The cascade returns failed cliques alongside accepted candidates
(the macro door reports them as ties). TDD with stub candidates — the chain
scenario becomes a pure unit test.

**Files:**
- Modify: `pyphi/condensation.py`
- Modify: `pyphi/substrate.py` (micro door builds `Candidate`s)
- Create: `test/test_condensation.py`

**Interfaces:**
- Produces (consumed by Tasks 4 and 6):

```python
@dataclass(frozen=True)
class Candidate:
    footprint: frozenset[int]                # micro units
    phi: float                               # φₛ of the candidate
    sia_provider: Callable[[], Any]          # the candidate's SIA (for Complex)
    system_provider: Callable[[], Any]       # the candidate's System (for Φ escalation)
    units: tuple[Any, ...] | None = None     # macro unit structure, None for micro

@dataclass(frozen=True)
class CondensationOutcome:
    accepted: tuple[Candidate, ...]          # complexes, φ-descending
    failed_cliques: tuple[tuple[Candidate, ...], ...]  # Φ-tied cliques

def exclusion_cascade(candidates: Sequence[Candidate]) -> CondensationOutcome: ...
    # candidates must be sorted φ-descending (stable) by the caller

def iit3_exclusion_cascade(candidates: Sequence[Candidate]) -> CondensationOutcome: ...

def exclusion_records(
    accepted: Sequence[Candidate], candidates: Sequence[Candidate]
) -> dict[tuple[int, ...], tuple[Any, ...]]: ...
    # keys: sorted footprint tuples; values: ExcludedCandidate records
```

- [ ] **Step 1: Write failing unit tests with stub candidates**

Create `test/test_condensation.py`:

```python
"""Unit tests for the exclusion cascade over Candidate records."""

import pytest

from pyphi import config
from pyphi.condensation import Candidate, exclusion_cascade, exclusion_records


class _StubSystem:
    """System stand-in: fingerprint + counted ces() with a fixed big Φ."""

    calls = 0

    def __init__(self, big_phi, fingerprint):
        self._big_phi = big_phi
        self._fingerprint = fingerprint

    def ces(self):
        type(self).calls += 1
        outer = self

        class _CES:
            big_phi = outer._big_phi

        return _CES()


def _candidate(footprint, phi, big_phi=0.0, fingerprint=None):
    system = _StubSystem(big_phi, fingerprint or object())
    return Candidate(
        footprint=frozenset(footprint),
        phi=phi,
        sia_provider=lambda: None,
        system_provider=lambda: system,
    )


def _footprints(outcome):
    return [tuple(sorted(c.footprint)) for c in outcome.accepted]


def test_chain_recursion_accepts_disjoint_lower_candidate():
    """A candidate beaten only by *excluded* rivals is a complex.

    Chain: {0,1} phi=3 overlaps {1,2} phi=2 overlaps {2,3} phi=1;
    {0,1} and {2,3} are disjoint. Recursive carving yields both.
    """
    candidates = [
        _candidate({0, 1}, 3.0),
        _candidate({1, 2}, 2.0),
        _candidate({2, 3}, 1.0),
    ]
    outcome = exclusion_cascade(candidates)
    assert _footprints(outcome) == [(0, 1), (2, 3)]
    assert outcome.failed_cliques == ()


def test_tied_clique_escalates_to_big_phi():
    """phi-tied overlapping candidates resolve by big Φ."""
    winner = _candidate({0, 1}, 1.0, big_phi=5.0)
    loser = _candidate({1, 2}, 1.0, big_phi=3.0)
    outcome = exclusion_cascade([winner, loser])
    assert _footprints(outcome) == [(0, 1)]


def test_phi_tied_clique_fails_exclusion_and_units_stay_available():
    """A Φ-tied clique is removed; its units remain available below."""
    a = _candidate({0, 1}, 1.0, big_phi=2.0)
    b = _candidate({1, 2}, 1.0, big_phi=2.0)
    lower = _candidate({0}, 0.5)
    outcome = exclusion_cascade([a, b, lower])
    assert _footprints(outcome) == [(0,)]
    assert len(outcome.failed_cliques) == 1
    assert {tuple(sorted(c.footprint)) for c in outcome.failed_cliques[0]} == {
        (0, 1),
        (1, 2),
    }


def test_single_fingerprint_clique_skips_escalation():
    """Identical kernel fingerprints ⇒ Φ ties by bit-identity: no ces() runs."""
    _StubSystem.calls = 0
    fp = b"same-digest"
    a = _candidate({0, 1}, 1.0, big_phi=2.0, fingerprint=fp)
    b = _candidate({1, 2}, 1.0, big_phi=2.0, fingerprint=fp)
    outcome = exclusion_cascade([a, b])
    assert outcome.accepted == ()
    assert len(outcome.failed_cliques) == 1
    assert _StubSystem.calls == 0


def test_mixed_fingerprint_clique_computes_big_phi_once_per_fingerprint():
    _StubSystem.calls = 0
    fp = b"shared"
    a = _candidate({0, 1}, 1.0, big_phi=2.0, fingerprint=fp)
    b = _candidate({1, 2}, 1.0, big_phi=2.0, fingerprint=fp)
    c = _candidate({0, 2}, 1.0, big_phi=7.0, fingerprint=b"other")
    outcome = exclusion_cascade([a, b, c])
    assert _footprints(outcome) == [(0, 2)]
    assert _StubSystem.calls == 2  # one per distinct fingerprint, not per member


def test_exclusion_records_key_on_footprints():
    top = _candidate({0, 1}, 3.0)
    beaten = _candidate({1, 2}, 2.0)
    disjoint = _candidate({2, 3}, 1.0)
    candidates = [top, beaten, disjoint]
    outcome = exclusion_cascade(candidates)
    records = exclusion_records(outcome.accepted, candidates)
    assert {r.node_indices for r in records[(0, 1)]} == {(1, 2)}
    assert {r.node_indices for r in records[(2, 3)]} == {(1, 2)}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_condensation.py -x -q`
Expected: FAIL with `ImportError: cannot import name 'Candidate'`.

- [ ] **Step 3: Implement in `pyphi/condensation.py`**

Add the records and rewrite the cascade around them. The tier walk, clique
grouping, and escalation-budget logic are the existing bodies with
`_sia_node_indices(sia)` replaced by `candidate.footprint` and system
reconstruction replaced by the provider:

```python
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A candidate system as the exclusion cascade sees it.

    ``footprint`` holds the candidate's micro units — the currency in which
    overlap is assessed (macro SIAs index macro units of a synthetic
    substrate, so a SIA's own indices cannot be used). The providers defer
    materialization: the SIA is needed only for accepted complexes, the
    system only when a tie escalates to Composition.
    """

    footprint: frozenset[int]
    phi: float
    sia_provider: Callable[[], Any]
    system_provider: Callable[[], Any]
    units: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class CondensationOutcome:
    """The cascade's result: accepted complexes and Φ-tied cliques."""

    accepted: tuple[Candidate, ...]
    failed_cliques: tuple[tuple[Candidate, ...], ...]


def _phi_groups(candidates: Sequence[Candidate]):
    """Yield contiguous groups tying on φ at precision (input φ-descending)."""
    from pyphi import utils

    i = 0
    while i < len(candidates):
        tier_phi = candidates[i].phi
        j = i + 1
        while j < len(candidates) and utils.eq(candidates[j].phi, tier_phi):
            j += 1
        yield list(candidates[i:j])
        i = j


def _find_overlap_cliques(candidates: list[Candidate]) -> list[list[Candidate]]:
    # existing union-find body, with
    #   units = [set(_sia_node_indices(sia) or ()) for sia in sias]
    # replaced by
    #   units = [set(c.footprint) for c in candidates]
    ...


def _fingerprint_key(system: Any):
    """Content digest for Φ dedupe; systems without one never dedupe."""
    key = getattr(system, "_fingerprint", None)
    return key if key is not None else object()


def _resolve_clique_by_big_phi(clique: list[Candidate]) -> Candidate | None:
    """Pick the Φ-maximal member of a φ-tied clique, or None on a Φ tie.

    Φ is computed once per distinct system content fingerprint: equal
    fingerprints are byte-identical kernel computations, so their Φ values
    tie exactly and a single-fingerprint clique fails exclusion with no
    cause-effect-structure computation at all.
    """
    from dataclasses import dataclass as _dc

    from pyphi import resolve_ties

    systems = [c.system_provider() for c in clique]
    keys = [_fingerprint_key(s) for s in systems]
    if len(set(keys)) == 1:
        return None

    big_phis: dict[Any, float] = {}
    for system, key in zip(systems, keys, strict=True):
        if key not in big_phis:
            big_phis[key] = float(system.ces().big_phi)

    @_dc(frozen=True)
    class _Proxy:
        candidate: Candidate
        big_phi: float

    proxies = [
        _Proxy(candidate=c, big_phi=big_phis[k])
        for c, k in zip(clique, keys, strict=True)
    ]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Composition")
    outcome = resolve_ties.resolve_complex_tie(proxies, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        return outcome.resolved.candidate
    return None


def exclusion_cascade(candidates: Sequence[Candidate]) -> CondensationOutcome:
    """Recursive condensation (Marshall 2023 Alg. A1 + S1 escalation).

    ``candidates`` must be sorted φ-descending (stable). Within each φ tier,
    candidates overlapping an accepted complex are dropped; survivors group
    into overlap cliques; multi-member cliques escalate to Composition. A
    Φ-tied clique fails exclusion: its members are removed but their units
    stay available to lower-φ candidates.
    """
    accepted: list[Candidate] = []
    covered: set[int] = set()
    failed: list[tuple[Candidate, ...]] = []
    for tier in _phi_groups(candidates):
        survivors = [c for c in tier if not (c.footprint & covered)]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            if len(clique) == 1:
                winner = clique[0]
            else:
                winner = _resolve_clique_by_big_phi(clique)
                if winner is None:
                    failed.append(tuple(clique))
                    continue
            accepted.append(winner)
            covered |= winner.footprint
    return CondensationOutcome(tuple(accepted), tuple(failed))
```

`iit3_exclusion_cascade` gets the same mechanical translation of its existing
body (clique resolution stays `resolve_iit3_complex_tie`, applied to
candidates' `sia_provider()` values — the IIT 3.0 resolver reads SIAs; wrap
as today). `exclusion_records` is the existing `_exclusion_records` body
with `_sia_node_indices(...)` replaced by `tuple(sorted(candidate.footprint))`
and `ExcludedCandidate(cand_idx, float(cand.phi))` gaining
`units=cand.units` **in Task 3** (until then keep the two-argument call).
Delete the now-unused `_big_phi_of_sia` and the SIA-based cascade variants.

- [ ] **Step 4: Rewire the micro door in `pyphi/substrate.py`**

In `complexes()` replace the cascade calls:

```python
from pyphi.condensation import (
    Candidate,
    exclusion_cascade,
    exclusion_records,
    iit3_exclusion_cascade,
)
from pyphi.system import System

sorted_sias = sorted(
    irreducible_sias(substrate, state, candidates, **kwargs), reverse=True
)
if not sorted_sias:
    return ()

def _as_candidate(sia):
    indices = tuple(_sia_node_indices(sia) or ())
    return Candidate(
        footprint=frozenset(indices),
        phi=float(sia.phi),
        sia_provider=lambda sia=sia: sia,
        system_provider=lambda indices=indices: System.from_substrate(
            substrate, state, indices
        ),
    )

cascade = (
    iit3_exclusion_cascade
    if _config_iit_version() == "IIT_3_0"
    else exclusion_cascade
)
outcome = cascade([_as_candidate(sia) for sia in sorted_sias])
records = exclusion_records(outcome.accepted, [_as_candidate(s) for s in sorted_sias])
result = tuple(
    Complex(
        sia=cand.sia_provider(),
        substrate=substrate,
        is_maximal=(i == 0),
        excluded=records[tuple(sorted(cand.footprint))],
    )
    for i, cand in enumerate(outcome.accepted)
)
```

(Build the candidate list once and reuse it for both calls — do not call
`_as_candidate` twice per SIA as the sketch above abbreviates.)

- [ ] **Step 5: Run unit tests and the affected suites**

Run: `uv run pytest test/test_condensation.py test/test_substrate.py test/macro/ -q`
Expected: PASS, including
`test_complexes_dispatches_to_iit3_cascade_under_iit3_version` (update its
monkeypatch target from `_iit3_exclusion_cascade` to `iit3_exclusion_cascade`
if it patches by name).

- [ ] **Step 6: Commit**

```bash
git add pyphi/condensation.py pyphi/substrate.py test/test_condensation.py test/test_substrate.py
git commit -m "Generalize the exclusion cascade over Candidate records with fingerprint-deduped ties"
```

---

### Task 3: `units` on `Complex` and `ExcludedCandidate`

**Files:**
- Modify: `pyphi/models/complex.py`
- Modify: `pyphi/condensation.py` (`exclusion_records` passes `units`)
- Test: `test/test_models_complex.py` (create if absent; check for an
  existing home first — `grep -rln "ExcludedCandidate" test/`)

**Interfaces:**
- Produces:
  - `Complex(sia, substrate, is_maximal=False, excluded=(), units=None,
    node_indices=None)` — `units: tuple[MacroUnit, ...] | None`;
    `node_indices` when given overrides the SIA-derived indices and means
    the **micro footprint**.
  - `ExcludedCandidate(node_indices, phi, units=None)`.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for units-aware Complex and ExcludedCandidate."""

from pyphi.models.complex import Complex, ExcludedCandidate


class _StubSIA:
    phi = 1.0
    node_indices = (0, 1)

    def order_by(self):
        return self.phi

    def _pandas_record(self):
        return {"phi": self.phi}


def test_complex_node_indices_override_and_units():
    units = ("unit-a", "unit-b")  # opaque to Complex; MacroUnits in practice
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        units=units,
        node_indices=(0, 1, 2, 3),
    )
    assert c.node_indices == (0, 1, 2, 3)
    assert c.units == units


def test_complex_defaults_micro():
    c = Complex(sia=_StubSIA(), substrate=None)
    assert c.node_indices == (0, 1)
    assert c.units is None


def test_excluded_candidate_units_default_none():
    e = ExcludedCandidate((1, 2), 0.5)
    assert e.units is None
    e2 = ExcludedCandidate((1, 2), 0.5, units=("u",))
    assert e2.units == ("u",)
    assert e2 == ExcludedCandidate((1, 2), 0.5)  # units not part of identity
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_models_complex.py -x -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'units'`.

- [ ] **Step 3: Implement**

In `pyphi/models/complex.py`:

```python
class ExcludedCandidate(...):
    def __init__(self, node_indices, phi, units=None):
        self.node_indices = tuple(node_indices)
        self.phi = float(phi)
        self.units = tuple(units) if units is not None else None
```

(`__eq__`/`__hash__` unchanged — identity stays `(node_indices, phi)`; two
descriptions of one excluded footprint are the same exclusion event. Add a
"Units" row to `_describe` and a `"units"` key to `_pandas_record` only when
`units is not None`.)

Also correct the `ExcludedCandidate` class docstring: it currently reads
"excluded ... in favor of an overlapping complex with greater-or-equal Φ",
which is not the recursive semantics — an excluded candidate may carry
higher φₛ than a complex whose record it appears in when it was carved away
by a *different* overlapping complex. Rewrite to: "A candidate system that
overlaps this complex and is not itself a complex: it was beaten (or
Φ-outranked) by an overlapping accepted complex, or belonged to a Φ-tied
clique that failed exclusion."

```python
class Complex(...):
    def __init__(self, sia, substrate, is_maximal=False, excluded=(),
                 units=None, node_indices=None):
        self.sia = sia
        self.substrate = substrate
        self.is_maximal = bool(is_maximal)
        self.excluded = tuple(excluded)
        self.units = tuple(units) if units is not None else None
        self._node_indices = (
            tuple(node_indices) if node_indices is not None else None
        )

    @property
    def node_indices(self):
        """The complex's micro units (``()`` for a null complex)."""
        if self._node_indices is not None:
            return self._node_indices
        from pyphi.condensation import _sia_node_indices

        return _sia_node_indices(self.sia) or ()
```

In `pyphi/condensation.py`, `exclusion_records` now constructs
`ExcludedCandidate(tuple(sorted(cand.footprint)), cand.phi, units=cand.units)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest test/test_models_complex.py test/test_condensation.py test/test_substrate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/complex.py pyphi/condensation.py test/test_models_complex.py
git commit -m "Carry macro unit structure on Complex and ExcludedCandidate"
```

---

### Task 4: Recursive condensation at the macro door

`pyphi.macro.complexes` consumes the shared cascade; winners become
`Complex` objects; `.ties` becomes escalation-failed cliques; the search
drivers reject IIT 3.0 eagerly. This is the behavior fix — the chain
regression is the test that would have caught the defect.

**Files:**
- Modify: `pyphi/macro/search.py`
- Test: `test/macro/test_macro_search.py` (extend)

**Interfaces:**
- Consumes: `Candidate`, `CondensationOutcome`, `exclusion_cascade`,
  `exclusion_records` (Task 2); `Complex(units=..., node_indices=...)`
  (Task 3).
- Produces:
  - `ComplexesResult.complexes: tuple[Complex, ...]` (φ-descending,
    `is_maximal` on the first).
  - `ComplexesResult.ties: tuple[tuple[MacroSystem, ...], ...]` — one inner
    tuple per Φ-tied clique.
  - `ComplexesResult.maximal_complex: Complex` property (falsy null object
    when no complex exists).
  - `_require_iit4()` guard raising `ValueError` under `IIT_3_0` on:
    `complexes`, `intrinsic_units`, `valid_systems`, `is_intrinsic_unit`,
    `competing_systems`.

- [ ] **Step 1: Write the failing chain-regression and guard tests**

Append to `test/macro/test_macro_search.py`:

```python
def decaying_chain_substrate():
    """4 units, reciprocal couplings 0.6 (0-1), 0.3 (1-2), 0.15 (2-3).

    The phi landscape is a chain: {A,B} > {B,C} > {C,D} with {B,C}
    overlapping both. Recursive condensation yields {A,B} and {C,D};
    the literal Eq. 19 predicate would orphan {C,D}.
    """
    n = 4
    weights = np.zeros((n, n))
    weights[0, 1] = weights[1, 0] = 0.6
    weights[1, 2] = weights[2, 1] = 0.3
    weights[2, 3] = weights[3, 2] = 0.15
    for i in range(n):
        weights[i, i] = 0.05
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = np.array([(row >> k) & 1 for k in range(n)])
        tpm[row] = 0.05 + weights @ s
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


class TestRecursiveCondensation:
    def test_chain_yields_both_disjoint_complexes(self):
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
        footprints = {c.node_indices for c in result.complexes}
        assert footprints == {(0, 1), (2, 3)}

    def test_winners_are_complex_objects_with_units_and_records(self):
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
        from pyphi.models.complex import Complex

        top = result.complexes[0]
        assert isinstance(top, Complex)
        assert top.is_maximal
        assert top.node_indices == (0, 1)
        assert top.units is not None and len(top.units) == 2
        assert any(e.node_indices == (1, 2) for e in top.excluded)
        assert result.maximal_complex is top

    def test_matches_micro_door_on_the_chain(self):
        from pyphi.substrate import complexes as micro_complexes

        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            macro = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
            micro = micro_complexes(substrate, (0, 0, 0, 0))
        assert {c.node_indices for c in macro.complexes} == {
            c.node_indices for c in micro
        }

    def test_iit3_rejected_eagerly(self):
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit3):
            with pytest.raises(ValueError, match="IIT_3_0"):
                complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
```

(`presets.iit3` is imported from `pyphi.conf.presets` like the existing
`iit4_2023` uses in this file. Also add one `pytest.raises` guard test per
remaining driver — `intrinsic_units`, `valid_systems`, `is_intrinsic_unit`,
`competing_systems` — same pattern, one line of setup each.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/macro/test_macro_search.py::TestRecursiveCondensation -x -q`
Expected: FAIL — `{(0, 1)} != {(0, 1), (2, 3)}` on the chain test (the
current literal Eq. 19 orphans {C,D}), `AttributeError` on `.node_indices`
(winners are `MacroSystem`s today), no raise on IIT 3.0.

- [ ] **Step 3: Implement in `pyphi/macro/search.py`**

Add the guard and call it first in all five drivers:

```python
def _require_iit4() -> None:
    """The macro formalism is defined for IIT 4.0 only."""
    from pyphi.conf import config as _config

    version = _config.formalism.iit.version
    if version == "IIT_3_0":
        raise ValueError(
            "the intrinsic-units macro framework is not defined for "
            f"IIT_3_0; got config.formalism.iit.version={version!r}"
        )
```

Replace the Eq. 19 filter in `complexes()` (the current `tops`/`ties`
construction, lines 784–813) with the shared cascade:

```python
from pyphi.condensation import Candidate, exclusion_cascade, exclusion_records
from pyphi.models.complex import Complex

evaluated = [(system, memo[system]) for system in sweep_systems if system is not None]

candidates = [
    Candidate(
        footprint=frozenset(_system_micro_indices(system.units)),
        phi=float(phi),
        sia_provider=lambda system=system: system.sia(),
        system_provider=lambda system=system: system,
        units=system.units,
    )
    for system, phi in evaluated
]
by_candidate = dict(zip(candidates, (s for s, _ in evaluated)))
ordered = sorted(candidates, key=lambda c: -c.phi)   # stable: keeps sweep order in ties
outcome = exclusion_cascade(ordered)
records_map = exclusion_records(outcome.accepted, ordered)
winners = tuple(
    Complex(
        sia=cand.sia_provider(),
        substrate=substrate,
        is_maximal=(i == 0),
        excluded=records_map[tuple(sorted(cand.footprint))],
        units=cand.units,
        node_indices=tuple(sorted(cand.footprint)),
    )
    for i, cand in enumerate(outcome.accepted)
)
ties = tuple(
    tuple(by_candidate[c] for c in clique) for clique in outcome.failed_cliques
)
records = tuple(
    EvaluationRecord(system=system, phi=float(phi)) for system, phi in memo.items()
)
from pyphi import validate

validate.non_overlapping(winners)
return ComplexesResult(complexes=winners, records=records, ties=ties)
```

Notes for the implementer:
- `cand.sia_provider()` re-runs `system.sia()` for **accepted winners only**;
  the P9.5 content cache makes the repertoire kernel warm in-process, and the
  number of winners is small. Do not memoize SIA objects in the sweep — the
  parallel protocol deliberately returns bare floats from workers.
- `Complex` ordering (`OrderableByPhi`) reads `sia.phi`; the accepted order
  from the cascade is already φ-descending.
- Update the `ComplexesResult` docstring: `complexes` are
  `Complex` objects from the recursive condensation; `ties` holds the
  Φ-tied cliques (each a tuple of the tied candidate systems). Add:

```python
@property
def maximal_complex(self):
    """The φₛ-maximal complex, or a falsy null Complex when none exists."""
    if self.complexes:
        return self.complexes[0]
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
    from pyphi.models.complex import Complex

    substrate = self.records[0].system.micro_substrate if self.records else None
    return Complex(
        sia=NullSystemIrreducibilityAnalysis(),
        substrate=substrate,
        is_maximal=True,
        excluded=(),
        units=(),
        node_indices=(),
    )
```

- [ ] **Step 4: Run the new tests, then the full macro suite**

Run: `uv run pytest test/macro/test_macro_search.py -q && uv run pytest test/macro/ -q`
Expected: PASS. Existing assertions that indexed `result.complexes[0].units`
directly on a `MacroSystem` still pass (`Complex.units` carries the same
tuple); any test asserting the old pair-shaped `ties` is updated to the
clique shape — **verify the new value is theoretically correct before
editing the assertion** (tie cliques on `tie_substrate` should contain
exactly the two mirror systems).

- [ ] **Step 5: Re-verify the macro goldens (no deferred confirmation)**

Run: `uv run pytest test/macro/test_macro_goldens.py -q`
Expected: PASS byte-stable. Example 1's winner spans the substrate (no
remainder to recurse on) and bu's winners are disjoint, so recursion changes
nothing there. If any golden moves, STOP and investigate the divergence to
root cause before regenerating anything.

- [ ] **Step 6: Commit**

```bash
git add pyphi/macro/search.py test/macro/test_macro_search.py
git commit -m "Condense macro candidate systems recursively, returning Complex winners

The literal Eq. 19 predicate computed only the first condensation
layer: candidates beaten solely by already-excluded rivals were
missing from the result on chain topologies. Macro condensation now
uses the shared recursive cascade with S1 Composition escalation,
and rejects IIT 3.0 eagerly."
```

---

### Task 5: Serialize `units`

**Files:**
- Modify: `pyphi/serialize/schema.py` (new `MacroUnitSchema`; `units` fields)
- Modify: `pyphi/serialize/convert.py` (encoders/decoders)
- Test: `test/test_serialization_complex.py` (create; or extend the existing
  serialization test module — check `grep -rln "ComplexSchema\|_register_complex" test/`)

**Interfaces:**
- Consumes: `MacroUnit` (`pyphi/macro/units.py`), `Complex.units`,
  `ExcludedCandidate.units` (Task 3).
- Produces: round-trip via `pyphi.serialize` for `Complex` and
  `ExcludedCandidate` carrying units, including nested meso constituents.

- [ ] **Step 1: Write the failing round-trip test**

```python
"""Round-trip serialization of units-bearing Complex objects."""

import numpy as np

from pyphi import serialize
from pyphi.macro.units import MacroUnit, micro_unit
from pyphi.models.complex import ExcludedCandidate


def test_excluded_candidate_units_roundtrip():
    meso = MacroUnit((0, 1), 1, (0, 0, 0, 1))
    unit = MacroUnit((meso, 2), 1, (0, 1, 1, 1))
    e = ExcludedCandidate((0, 1, 2), 0.25, units=(unit,))
    restored = serialize.loads(serialize.dumps(e))
    assert restored.node_indices == (0, 1, 2)
    assert restored.units == (unit,)


def test_excluded_candidate_without_units_roundtrip():
    e = ExcludedCandidate((1, 2), 0.5)
    restored = serialize.loads(serialize.dumps(e))
    assert restored.units is None


def test_macro_complex_roundtrip():
    from pyphi import config
    from pyphi.conf import presets
    from pyphi.macro.search import SearchBounds, complexes
    from pyphi.substrate import Substrate

    tpm = np.array(
        [[0.05, 0.05], [0.05, 0.06], [0.06, 0.05], [0.95, 0.95]]
    )
    substrate = Substrate(tpm, node_labels=("A", "B"))
    with config.override(**presets.iit4_2023):
        result = complexes(substrate, (0, 0), SearchBounds())
    top = result.complexes[0]
    restored = serialize.loads(serialize.dumps(top))
    assert restored.node_indices == top.node_indices
    assert restored.units == top.units
    assert restored.is_maximal == top.is_maximal
```

(Adapt `serialize.dumps`/`loads` to the module's actual entry points —
mirror whatever the existing `Complex` round-trip test uses.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_serialization_complex.py -x -q`
Expected: FAIL (no encoder for `MacroUnit` / `units` dropped).

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py` (near `ExcludedCandidateSchema`, line 225):

```python
class MacroUnitSchema(msgspec.Struct, frozen=True, tag="macro_unit"):
    constituents: tuple["MacroUnitSchema | int", ...]
    update_grain: int
    mapping: tuple[int, ...]
    background_apportionment: tuple[int, ...] = ()
```

Extend the two structs (new fields last, with defaults — msgspec requires
defaults after non-defaulted fields):

```python
class ExcludedCandidateSchema(msgspec.Struct, frozen=True, tag="excluded_candidate"):
    node_indices: tuple[int, ...]
    phi: float
    units: tuple[MacroUnitSchema, ...] | None = None


class ComplexSchema(msgspec.Struct, frozen=True, tag="complex"):
    sia: SIASchema
    substrate: SubstrateSchema
    is_maximal: bool
    excluded: tuple[ExcludedCandidateSchema, ...]
    units: tuple[MacroUnitSchema, ...] | None = None
    node_indices: tuple[int, ...] | None = None
```

Add `MacroUnitSchema` to the tagged-union list (near line 489). In
`pyphi/serialize/convert.py` extend `_register_excluded_candidate` /
`_register_complex` and add:

```python
def _register_macro_unit() -> None:
    from pyphi.macro.units import MacroUnit

    def _enc(u):
        return schema.MacroUnitSchema(
            constituents=tuple(
                _enc(c) if isinstance(c, MacroUnit) else int(c)
                for c in u.constituents
            ),
            update_grain=u.update_grain,
            mapping=tuple(u.mapping),
            background_apportionment=tuple(u.background_apportionment),
        )

    def _dec(s):
        return MacroUnit(
            constituents=tuple(
                _dec(c) if isinstance(c, schema.MacroUnitSchema) else int(c)
                for c in s.constituents
            ),
            update_grain=s.update_grain,
            mapping=tuple(s.mapping),
            background_apportionment=tuple(s.background_apportionment),
        )

    _ENCODERS[MacroUnit] = _enc
    _DECODERS[schema.MacroUnitSchema] = _dec
```

and wire the `units=` / `node_indices=` fields through the `Complex` and
`ExcludedCandidate` encoder/decoder lambdas (encode `None` as `None`;
`Complex`'s encoder passes `node_indices=tuple(c.node_indices)` always — the
decoder passing it back is harmless for micro winners and required for macro
ones). Register `_register_macro_unit()` wherever the sibling `_register_*`
calls run.

- [ ] **Step 4: Run tests**

Run: `uv run pytest test/test_serialization_complex.py -q` and the module's
existing serialization suite.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/test_serialization_complex.py
git commit -m "Serialize macro unit structure on Complex and ExcludedCandidate"
```

---

### Task 6: Cross-door equivalence, shadow equality, and parallel determinism

**Files:**
- Test: `test/macro/test_macro_search.py` (extend)

**Interfaces:**
- Consumes: everything landed in Tasks 2–4.

- [ ] **Step 1: Write the both-doors Hypothesis sweep**

```python
from hypothesis import given, settings, strategies as st


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=0, max_value=10**6))
def test_doors_agree_at_micro_grain(seed):
    """substrate.complexes ≡ macro complexes at max_depth=0.

    At max_depth=0 the macro candidate pool is every subset of micro
    units, evaluated as identity-unit systems (which reproduce System
    results exactly), so the two doors condense the same landscape.
    """
    rng = np.random.default_rng(seed)
    n = 3
    tpm = rng.uniform(0.05, 0.95, size=(2**n, n))
    substrate = Substrate(tpm)
    state = tuple(int(v) for v in rng.integers(0, 2, size=n))
    from pyphi.substrate import complexes as micro_complexes

    with config.override(**presets.iit4_2023):
        try:
            micro = micro_complexes(substrate, state)
            macro = complexes(substrate, state, SearchBounds(max_depth=0))
        except Exception:
            # unreachable states abort both doors identically; skip
            return
    assert {c.node_indices for c in macro.complexes} == {
        c.node_indices for c in micro
    }
    micro_phis = {c.node_indices: float(c.phi) for c in micro}
    for c in macro.complexes:
        assert float(c.phi) == micro_phis[c.node_indices]
```

(The exact `==` on φ is deliberate: identity macroing is pinned to reproduce
`System` results exactly, and both doors read the same SIA values.
If the blanket `except Exception` proves too broad in practice, narrow it to
the actual exceptions unreachable states raise — do not let it mask
assertion failures: catch around the two driver calls only, as written.)

- [ ] **Step 2: Write the tie shadow-equality test**

```python
def test_fingerprint_dedupe_shadow_equality(monkeypatch):
    """Forcing full escalation (unique fingerprints) changes nothing."""
    substrate = tie_substrate()
    state = (0, 0, 0)
    with config.override(**presets.iit4_2023):
        with_skip = complexes(substrate, state, SearchBounds())

    import pyphi.condensation as condensation

    monkeypatch.setattr(condensation, "_fingerprint_key", lambda system: object())
    with config.override(**presets.iit4_2023):
        without_skip = complexes(substrate, state, SearchBounds())

    assert {c.node_indices for c in with_skip.complexes} == {
        c.node_indices for c in without_skip.complexes
    }
    assert [
        {tuple(s.units) for s in clique} for clique in with_skip.ties
    ] == [
        {tuple(s.units) for s in clique} for clique in without_skip.ties
    ]
```

- [ ] **Step 3: Write the parallel ≡ sequential test**

```python
def test_complexes_parallel_equals_sequential():
    substrate = decaying_chain_substrate()
    state = (0, 0, 0, 0)
    with config.override(**presets.iit4_2023):
        sequential = complexes(substrate, state, SearchBounds(max_depth=0))
    with config.override(
        **presets.iit4_2023,
        parallel=True,
        parallel_macro_system_evaluation={"parallel": True},
    ):
        parallel = complexes(substrate, state, SearchBounds(max_depth=0))
    assert [c.node_indices for c in sequential.complexes] == [
        c.node_indices for c in parallel.complexes
    ]
    assert [float(c.phi) for c in sequential.complexes] == [
        float(c.phi) for c in parallel.complexes
    ]
```

(Follow the existing parallel-equivalence test in `test/macro/` for the
config-override idiom; if one already covers `complexes()`, extend it to
assert on `Complex` fields instead of adding a duplicate.)

- [ ] **Step 4: Write the exclusion-invariant property test**

```python
def test_exclusion_invariants_on_the_chain_sweep():
    """Accepted complexes are disjoint, beat every overlapping candidate
    that was not itself excluded earlier, and every exclusion record
    points back at a complex with greater-or-equal phi."""
    substrate = decaying_chain_substrate()
    with config.override(**presets.iit4_2023):
        result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))

    footprints = [set(c.node_indices) for c in result.complexes]
    for i, a in enumerate(footprints):
        for b in footprints[i + 1 :]:
            assert not (a & b)

    accepted = {c.node_indices for c in result.complexes}
    for c in result.complexes:
        for record in c.excluded:
            # every exclusion record names a candidate that overlaps this
            # complex and was not itself accepted. NOTE: an excluded
            # candidate may carry HIGHER phi than the complex whose record
            # it appears in — on the chain, {C,D}'s records include {B,C}
            # (phi 0.104 > 0.037), which was carved away by {A,B}. That is
            # the recursive semantics working as intended; do not assert
            # record.phi <= c.phi.
            assert set(record.node_indices) & set(c.node_indices)
            assert record.node_indices not in accepted
    # the chain makes the higher-phi-excluded case concrete:
    by_units = {c.node_indices: c for c in result.complexes}
    assert any(
        record.phi > float(by_units[(2, 3)].phi)
        for record in by_units[(2, 3)].excluded
    )
```

- [ ] **Step 5: Run**

Run: `uv run pytest test/macro/test_macro_search.py -q`
Expected: PASS. The Hypothesis sweep takes tens of seconds (8 examples × two
full condensations at n = 3); if it exceeds ~2 minutes, reduce
`max_examples` to 5, not the substrate size.

- [ ] **Step 5: Commit**

```bash
git add test/macro/test_macro_search.py
git commit -m "Pin cross-door condensation equivalence, tie shadow equality, and parallel determinism"
```

---

### Task 7: Changelog, ROADMAP, full-suite verification

**Files:**
- Create: `changelog.d/macro-condensation-recursive.fix.md`
- Create: `changelog.d/complex-unification.change.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write the changelog fragments**

`changelog.d/macro-condensation-recursive.fix.md`:

```markdown
Fixed `pyphi.macro.complexes` computing only the first condensation layer:
the literal Eq. 19 predicate let already-excluded candidates veto other
candidates, so complexes on chain topologies (e.g. a system beaten only by
rivals that themselves lost to a stronger complex) were missing from the
result. Macro condensation now applies the recursive exclusion cascade
(Marshall et al. 2023, Algorithm A1) with S1 Composition escalation for
φₛ-tied cliques, matching `pyphi.substrate.complexes`.
```

`changelog.d/complex-unification.change.md`:

```markdown
`pyphi.macro.ComplexesResult.complexes` now contains
`pyphi.models.complex.Complex` objects (with `units`, `node_indices` as the
micro footprint, `is_maximal`, and `excluded` records) instead of bare
`MacroSystem`s; `ComplexesResult.ties` holds Φ-tied cliques (tuples of
candidate systems) instead of pairs, and `ComplexesResult.maximal_complex`
returns the winner or a falsy null `Complex`. `Complex` and
`ExcludedCandidate` gained an optional `units` field, and both serialize
with it. The macro search drivers raise under IIT 3.0. During tie
escalation, big Φ is computed once per system content fingerprint, so
symmetric tied cliques skip cause-effect-structure computation entirely.
```

- [ ] **Step 2: Update `ROADMAP.md`**

Add a Status Dashboard row (✅ landed) for "Complex unification — one
exclusion semantics, one winner type", citing the spec path and summarizing:
macro condensation fixed from literal Eq. 19 to the recursive cascade;
shared `pyphi/condensation.py`; `Complex` winners at both doors;
fingerprint-deduped S1 escalation; IIT 3.0 eager reject. In the
"Macro framework — Marshall 2024 intrinsic units" entry, add a line queueing
the Eq. 19-vs-recursion divergence for the paper's authors alongside the
existing SP1/SP2 upstream findings.

- [ ] **Step 3: Full-suite verification**

Run: `uv run pytest`
(no path argument — testpaths + doctest sweep). Expected: PASS. Also run
`uv run pyright pyphi/condensation.py pyphi/macro/search.py pyphi/models/complex.py`
and fix any new diagnostics.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/macro-condensation-recursive.fix.md changelog.d/complex-unification.change.md ROADMAP.md
git commit -m "Record the complex-unification landing in the changelog and ROADMAP"
```
