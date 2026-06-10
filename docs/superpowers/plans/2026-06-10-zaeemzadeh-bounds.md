# Zaeemzadeh Upper Bounds Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `pyphi/formalism/iit4/bounds.py` — the published upper bounds on IIT quantities (Zaeemzadeh & Tononi 2024, PLOS Comput Biol 20(8):e1012323) as a standalone research utility with per-bound certificates, validated by independent recomputation, property tests against the real 2.0 pipeline, reference goldens from the author's code, and an end-to-end measure-parity check.

**Architecture:** A single pure-math module of plain functions returning frozen `UpperBound(value, certified, assumptions, citation)` objects. No registry, no config keys, zero pipeline changes; the only config interaction is a strict read-only domain guard. Bound III is computed in closed form from the S3-appendix binomial formulas (no pipeline dependency); the 2.0 pipeline and the author's original code are used only to *validate* it.

**Tech Stack:** Python 3.13, `math.comb`/`Fraction` exact arithmetic, numpy (construction TPM only), scipy.optimize.linprog (tests only), Hypothesis (property tests), pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-zaeemzadeh-bounds-design.md` (commit `a844b313`).

---

## Mathematical reference

Everything the code implements, in one place. Sources: main paper (`papers/2024__zaeemzadeh-tononi__upper-bounds.pdf`), S3 proofs appendix (`...__s3-appendix.pdf`), author's experiment code (github.com/zaeemzadeh/IIT-bounds, cloned at /tmp/IIT-bounds).

### Per-object bounds (certified)

- **Lemma 2:** φ(m, Z given θ) ≤ 𝒩(θ), the number of connections severed by partition θ. Holds for *any* partitioning, valid or not.
- **Theorem 1:** φ_e(m, E) ≤ |M||E| and φ_c(m, C) ≤ |M||C| for any candidate purview.
- **Relation (§2.2):** φ_r(**d**) ≤ min over relata of φ_d. (In 2.0 this is partially structural: `Relation.phi = len(purview) * min(relatum.phi / len(relatum.purview_union))` is min-based by construction.)
- **System (Table 2, citing Marshall et al. 2023):** φ_s ≤ |S|(|S|−1). Valid only for partition schemes that do not sever self-connections (set partitions cut only between-part edges; max severed = n(n−1) at the atomic partition).

### Σφ_d bounds

- **Bound I (Eq 6, certified, not achievable):** Σφ_d ≤ Σ_K K·N·C(N,K) = N²·2^(N−1).
- **Bound II (Eq 7, conditional on unique purviews):** Σφ_d ≤ Σ_K K²·C(N,K) = N(N+1)·2^N/4. Inequality (a) in Eq 7 assumes each purview is assigned to exactly one mechanism with matching sizes.
- **Bound III (§2.1.3 + S3 §3, conjectured general):** Σ_K C(N,K)·φ\*_e(K), where φ\*_e(K) is the integrated effect information of a size-K mechanism over itself in the *high-selectivity reflexive construction* — the TPM where every size-K mechanism specifies itself with probability 1.

### Bound III closed form (S3 §3)

Construction TPM (Eqs 18, 20): unit *u* turns OFF with probability 1 in exactly the states where *u* is OFF and at least K−1 other units are OFF; otherwise it turns ON with probability 1. Current state: all-zeros. Selectivity is 1, so φ at partition θ equals the informativeness: −Σ_{Z_i∈Z} log₂ π_e^θ(Z_i = z_i | m_i), where m_i is the mechanism part connected to purview unit i after partitioning.

Partitioned single-unit probabilities, for a part of size *a* (equalities hold under the construction):

- part **contains** the purview unit: π(a) = Σ_{b=K−a}^{N−a} C(N−a, b) / 2^(N−a)
- part **does not contain** it (covers a=0, the empty part): π̄(a) = Σ_{b=K−a}^{N−a} C(N−a−1, b−1) / 2^(N−a)

The MIP search needs only ⌊K/2⌋+1 candidates (proof: cutting self-connections has per-connection gain 1 except K=N, so the MIP either cuts no self-connections — and among those, bipartitions minimize the average — or is the one cut severing a single mechanism unit from all K purview units):

1. Non-self-cutting bipartitions with part sizes (j, K−j), j = 1..⌊K/2⌋: value v = −[j·log₂π(j) + (K−j)·log₂π(K−j)], 𝒩 = 2j(K−j).
2. The one-unit-severed cut: v = −[log₂π̄(K−1) + (K−1)·log₂π(K−1)], 𝒩 = K. (For K=1 this is the only candidate and gives v = 1.)
3. K=N special case: only the complete partition; every unit fully marginalized, π = 1/2^N, so φ\* = N².

MIP = argmin of v/𝒩 (matching the paper's Eq 3 normalization and 2.0's default `NUM_CONNECTIONS_CUT`); φ\*_e(K) is the *unnormalized* v at the argmin. **Verified empirically against the 2.0 pipeline during plan prep:** N=3, K=2 gives φ\* = −2·log₂(3/4) = 0.8300749985576875 from both the formula and `System.find_mip(EFFECT, (0,1), (0,1), partitions=[...])` on the construction TPM, with the bipartition winning as predicted.

### Σφ_r machinery (Eqs 8–16, Table 3, S3 §4)

- 𝒵(o) = the set of (z, φ) pairs of distinctions whose purview-union contains unit-state *o* (Eq 10).
- Eq 11 (exact rewrite): Σφ_r = Σ_{self} φ_r + Σ_o Σ_{i=1}^{|𝒵(o)|} (φ_(i)/|z_(i)|)·(2^(|𝒵(o)|−i) − 1), ratios sorted **ascending** (smallest ratio gets the largest weight — it is the min of the most subsets). Self-relations are bounded by Σφ_d (Eq 12, Step 2).
- Eq 13/14 (LP): max Σ y_i·(2^(R−i)−1) s.t. Σy_i ≤ S(o), y ascending ≥ 0 has maximum S(o)·((2^R − 1)/R − 1), attained at the equal-ratio vertex.
- **Eq 16 (certified, general):** Σφ_r ≤ N²·2^(N−1) + N²·2^N·((2^(2^N−1) − 1)/(2^N − 1) − 1). Derivation: self-term ≤ Eq 6; per unit-state o, S(o) ≤ Σ_M |M| = N·2^(N−1) (Theorem 1, certified) and |𝒵(o)| ≤ 2^N−1; LP max is monotone in both; 2N unit-states for binary units.
- **Table 3 closed forms (scenario-conditional):** exact Eq-11 evaluations of specific extremal purview profiles, *excluding* the self-term in the printed table:
  - Bound I profile (all purviews = S, φ = |M|N, ratio = |M|, 𝒵(o) = all 2^N−1 distinctions): non-self term = N·Σ_K K·(2^(E_K) − 2^(E_K − C(N,K)) − C(N,K)) with E_K = Σ_{i≥K} C(N,i). Self term = Σφ_d Bound I.
  - Bound II profile (purview = mechanism, φ = |M|², ratio = |M|, 𝒵(o) = mechanisms containing o, multiplicity C(N−1,K−1)): analogous with E_K = Σ_{i=K−1}^{N−1} C(N−1,i) and multiplicities C(N−1,K−1). Self term = Σφ_d Bound II.
  - Bound III profile (per the author's published code: ratio = φ\*_K/K over all 2^N−1 distinctions; self term = Σ_K C(N,K)·φ\*_K). NOTE: the paper text (p. 16) instead assumes Z_c = S (ratio φ\*_K/N); the code's hybrid is a valid, looser bound on both readings. We implement the code's version so reference goldens match, and document the difference.
- Grouped evaluation of the Eq-11 inner sum for tied ratios (keeps integer exactness for Bounds I/II): for an ascending group of multiplicity m with `after` elements above it, the group weight is 2^after·(2^m − 1) − m.

### Counting (certified, measure-free)

- Possible distinctions: 2^N − 1; of order k: C(N,k).
- Possible relations (nonempty subsets of distinctions, §2.2): 2^(2^N−1) − 1.
- Possible relation *faces* with unique purviews ("Will's theorem" family, from the pre-publication module): in the scenario where every nonempty unit subset appears as exactly one cause and one effect purview, the number of size-≥2 subsets of the 2(2^N−1) cause/effect purview slots whose purviews intersect in exactly k units is C(N,k)·Σ_i (−1)^i·C(N−k,i)·f(N,k+i), where f(N,j) = 2^(2^(N−j+1)) − 1 − 2^(N−j+1) (there are 2^(N−j+1) slots containing a fixed j-set). Hand-verified for N=2: k=2 → 1, k=1 → 20. The brute-force test pins this semantics; **if it fails, STOP — the semantics hypothesis is wrong; investigate before renaming.**

### Hand-verified test constants

| Quantity | Value |
|---|---|
| φ\*_e(N=3, K=2) | 0.8300749985576875 (= −2·log₂(3/4); 2.0-pipeline verified) |
| Σφ_d Bound III, N=2 | 6 (= 2·1 + 1·4) |
| Σφ_d Bound III, N=3 | 12 + 3·0.8300749985576875 ≈ 14.490224995673063 |
| Σφ_r Bound I, N=2 (incl. self) | 16 (non-self 8 + self 8) |
| Σφ_r Bound II, N=2 (incl. self) | 8 (non-self 2 + self 6) |
| Σφ_r Bound III, N=2 (incl. self) | 14 (non-self 8 + self 6) |
| Σφ_r GENERAL (Eq 16), N=2 | 88/3 ≈ 29.333333333333332 |
| Relation faces, N=2 | k=1: 20, k=2: 1 (total 21) |

---

## Math corrections established during planning (surface in review)

1. **Table 3 Σφ_r closed forms are NOT certified-general**, even when built from a certified Σφ_d input: they are exact evaluations of specific extremal purview *profiles*. The certified-general Σφ_r bound is Eq 16 (which is ~N/2 times larger at leading order). The spec's "status inherited from the Σφ_d input" was too generous. Resolution: `sum_phi_relations_upper_bound(n, bound="I"|"II"|"III")` ships `certified=False` with an "extremal purview profile" assumption; `bound="GENERAL"` ships Eq 16 with `certified=True`.
2. **The φ_s ≤ n(n−1) bound presumes the system partition scheme does not sever self-connections.** The system-level domain guard therefore checks `system_partition_scheme == "DIRECTED_SET_PARTITION"` in addition to (version, measure).
3. **The published Fig 4 Bound III formula differs between paper text (Z_c = S) and the released code (hybrid ratio φ\*/K with 𝒵(o) = all).** Both are valid bounds; we implement the code's version (goldens fidelity) and document the difference.

---

## File structure

- **Create:** `pyphi/formalism/iit4/bounds.py` — the module (only new source file; not imported by `__init__.py` — standalone access via `from pyphi.formalism.iit4 import bounds`).
- **Create:** `test/test_bounds.py` — validation batteries (a), (b), (d) + conjecture probes.
- **Create:** `test/test_bounds_reference_golden.py` — battery (c).
- **Create:** `test/data/bounds/reference_goldens.json` — frozen author-code outputs.
- **Create:** `changelog.d/zaeemzadeh-bounds.feature.md`.
- **Modify:** `ROADMAP.md` — P13 entry rewrite (lines ~2172–2204).

## Conventions (repo-specific, binding)

- Run everything with `uv run`. Full-suite verification at the end: `uv run pytest` with **no path argument** (doctest sweep), in background.
- Commits: `git -c commit.gpgsign=false commit`. Targeted `git add <files>` only — ~25 untracked scratch files in repo root must never be staged. If a commit silently doesn't land, the hook reformatted: re-`git add` the same files and commit again. A FAILED commit leaves files staged — check `git status` before the next add+commit.
- Ruff bans in Python source: `dict()` calls (use `{}`); unicode ×, −, – in strings/docstrings (ASCII only; spell out "phi", "pi"); imports at top (E402, tests too); RUF005 (use `[*a, b]`).
- Docstrings describe the final state; no design narrative, no migration history, no planning artifacts (P-numbers).
- NEVER push. NEVER `--no-verify`.

---

### Task 1: `UpperBound` + counting family

**Files:**
- Create: `pyphi/formalism/iit4/bounds.py`
- Create: `test/test_bounds.py`

- [ ] **Step 1.1: Write the failing tests**

Create `test/test_bounds.py`:

```python
"""Tests for pyphi.formalism.iit4.bounds (Zaeemzadeh & Tononi 2024).

Validation batteries per the design spec:
(a) independent recomputation of the formulas (brute force, linprog);
(b) property tests against the real pipeline = domain-whitelist evidence;
(d) Bound III end-to-end measure parity via the construction TPM.
Battery (c), reference goldens, lives in test_bounds_reference_golden.py.
"""

import dataclasses
import itertools
import math

import pytest

from pyphi.formalism.iit4 import bounds

# NOTE: later tasks add imports to this block as they need them; adding
# them all now would fail ruff F401 at this task's commit.


class TestUpperBound:
    def test_float_protocol(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=("binary units",), citation="Eq 6"
        )
        assert float(bound) == 6.0
        assert isinstance(float(bound), float)

    def test_frozen(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=(), citation="Eq 6"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            bound.value = 7  # pyright: ignore[reportAttributeAccessIssue]

    def test_integral_values_stay_int(self):
        bound = bounds.UpperBound(
            value=6, certified=True, assumptions=(), citation="Eq 6"
        )
        assert isinstance(bound.value, int)


class TestCounts:
    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (3, 7), (10, 1023)])
    def test_number_of_possible_distinctions(self, n, expected):
        assert bounds.number_of_possible_distinctions(n) == expected

    def test_number_of_possible_distinctions_of_order(self):
        assert bounds.number_of_possible_distinctions_of_order(4, 2) == 6
        total = sum(
            bounds.number_of_possible_distinctions_of_order(4, k) for k in range(1, 5)
        )
        assert total == bounds.number_of_possible_distinctions(4)

    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 7), (3, 127)])
    def test_number_of_possible_relations(self, n, expected):
        # 2 ** (2 ** n - 1) - 1: nonempty subsets of candidate distinctions.
        assert bounds.number_of_possible_relations(n) == expected

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            bounds.number_of_possible_distinctions(0)
        with pytest.raises(ValueError, match="order"):
            bounds.number_of_possible_distinctions_of_order(3, 4)

    @staticmethod
    def _brute_force_face_counts(n):
        """Enumerate size->count of purview-slot subsets by exact overlap.

        Slots: every nonempty subset of units appears as exactly one cause
        purview and one effect purview (unique purviews). A candidate
        relation face is a subset of at least two slots with nonempty
        common overlap.
        """
        subsets = [
            frozenset(combo)
            for r in range(1, n + 1)
            for combo in itertools.combinations(range(n), r)
        ]
        slots = [*subsets, *subsets]  # cause slot + effect slot per purview
        counts = {}
        for r in range(2, len(slots) + 1):
            for combo in itertools.combinations(range(len(slots)), r):
                overlap = frozenset.intersection(*(slots[i] for i in combo))
                if overlap:
                    counts[len(overlap)] = counts.get(len(overlap), 0) + 1
        return counts

    @pytest.mark.parametrize("n", [2, 3])
    def test_relation_faces_with_unique_purviews_match_brute_force(self, n):
        expected = self._brute_force_face_counts(n)
        for k in range(1, n + 1):
            actual = bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(
                n, k
            )
            assert actual == expected.get(k, 0), f"n={n}, k={k}"
        assert bounds.number_of_possible_relation_faces_with_unique_purviews(
            n
        ) == sum(expected.values())

    def test_relation_faces_hand_values_n2(self):
        assert (
            bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(2, 1)
            == 20
        )
        assert (
            bounds.number_of_possible_relation_faces_with_unique_purviews_of_order(2, 2)
            == 1
        )
```

- [ ] **Step 1.2: Run the tests to verify they fail**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.formalism.iit4.bounds'` (or ImportError).

- [ ] **Step 1.3: Write the module skeleton**

Create `pyphi/formalism/iit4/bounds.py`:

```python
"""Upper bounds for IIT 4.0 quantities.

Implements the published upper bounds from:

    Zaeemzadeh A, Tononi G. (2024). Upper bounds for integrated
    information. PLOS Computational Biology 20(8): e1012323.
    https://doi.org/10.1371/journal.pcbi.1012323

All bounds assume binary units and a conditionally independent TPM (the
system is realizable as a product of unit TPMs) and are derived for the
IIT 4.0 intrinsic-difference family of measures. Each bound function
returns an :class:`UpperBound` carrying the value together with its
certificate: ``certified=True`` means the bound is theorem-backed for
arbitrary systems satisfying its ``assumptions``; ``certified=False``
means it additionally relies on a scenario assumption or an open
conjecture, recorded in ``assumptions``.

The mechanism-level bounds are ceilings over states and hold at any
selected mechanism partition, so they are insensitive to the configured
partition scheme and MIP normalization. The system-level bound
additionally assumes a system partition scheme that does not sever
self-connections.

Functions taking only ``n`` are pure combinatorics of the binary
formalism; the binary-units assumption is the caller's responsibility
there. :func:`report` validates it when given a
:class:`~pyphi.substrate.Substrate`.

Relation-level sums grow like ``2**(2**n)``. They are computed as exact
Python ints where possible; values that are not integral are returned as
floats and overflow for ``n`` greater than about 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CITATION = (
    "Zaeemzadeh A, Tononi G. (2024). Upper bounds for integrated "
    "information. PLOS Comput Biol 20(8): e1012323."
)

_CORE_ASSUMPTIONS = ("binary units", "conditionally independent TPM")


@dataclass(frozen=True)
class UpperBound:
    """An upper bound on an IIT quantity, with its certificate.

    Attributes:
        value: The bound. An exact ``int`` when the quantity is integral.
        certified: Whether the bound is theorem-backed for arbitrary
            systems satisfying ``assumptions``. Non-certified bounds rely
            on a scenario assumption or an open conjecture and are not
            valid pruning certificates.
        assumptions: The assumptions under which the bound holds.
        citation: Locus in Zaeemzadeh & Tononi (2024), e.g. ``"Eq 6"``.
    """

    value: float
    certified: bool
    assumptions: tuple[str, ...]
    citation: str

    def __float__(self) -> float:
        return float(self.value)


def _require_positive(n: int) -> None:
    if n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}")


##############################################################################
# Counting (pure combinatorics; no measure or version dependence)
##############################################################################


def number_of_possible_distinctions(n: int) -> int:
    """Number of candidate distinctions in a system of n units.

    One per nonempty mechanism: 2**n - 1.
    """
    _require_positive(n)
    return 2**n - 1


def number_of_possible_distinctions_of_order(n: int, k: int) -> int:
    """Number of candidate distinctions with mechanism size k."""
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    return math.comb(n, k)


def number_of_possible_relations(n: int) -> int:
    """Number of candidate relations in a system of n units.

    One per nonempty subset of the candidate distinctions (Sec 2.2):
    2**(2**n - 1) - 1.
    """
    _require_positive(n)
    return 2 ** (2**n - 1) - 1


def _f(n: int, j: int) -> int:
    """Size->=2 subsets of the purview slots containing a fixed j-unit set.

    In the unique-purview scenario there are 2**(n - j) purviews
    containing the fixed set, each contributing a cause slot and an
    effect slot: 2**(n - j + 1) slots in total.
    """
    slots = 2 ** (n - j + 1)
    return 2**slots - 1 - slots


def number_of_possible_relation_faces_with_unique_purviews_of_order(
    n: int, k: int
) -> int:
    """Number of candidate relation faces whose overlap has exactly k units.

    Counts subsets of size at least 2 of the 2(2**n - 1) cause/effect
    purview slots whose purviews intersect in exactly k units, in the
    scenario where every nonempty subset of units appears as exactly one
    cause and one effect purview. Computed by inclusion-exclusion over
    the overlap set.
    """
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    return math.comb(n, k) * sum(
        (-1) ** i * math.comb(n - k, i) * _f(n, k + i) for i in range(n - k + 1)
    )


def number_of_possible_relation_faces_with_unique_purviews(n: int) -> int:
    """Number of candidate relation faces in the unique-purview scenario."""
    _require_positive(n)
    return sum(
        number_of_possible_relation_faces_with_unique_purviews_of_order(n, k)
        for k in range(1, n + 1)
    )
```

- [ ] **Step 1.4: Run the tests to verify they pass**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: PASS (all TestUpperBound + TestCounts tests). If `test_relation_faces_with_unique_purviews_match_brute_force` fails, STOP — the face-semantics hypothesis for the old counting formula is wrong; investigate the discrepancy (compare against subsets-of-distinctions semantics) before changing names or formulas, and surface to the user.

- [ ] **Step 1.5: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add UpperBound and counting functions to iit4.bounds

Counting family (possible distinctions, relations, and relation faces
under unique purviews) is pure combinatorics; the face counts are
verified against brute-force enumeration of purview-slot subsets at
small N, pinning the previously undocumented semantics of the
inclusion-exclusion formula."
```

---

### Task 2: Domain guard + per-object bounds

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py`
- Modify: `test/test_bounds.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `test/test_bounds.py` (imports for this task: add `from pyphi import config` and `from pyphi.conf import presets` and `from pyphi.models.partitions import JointPartition, Part` to the import block at the top — imports must be at top of file per E402):

```python
class TestDomainGuard:
    def test_default_config_is_in_domain(self):
        # Default: IIT_4_0_2023 + GENERALIZED_INTRINSIC_DIFFERENCE.
        bound = bounds.distinction_phi_upper_bound((0, 1), (0, 1, 2))
        assert bound.value == 6

    def test_iit3_raises(self):
        with config.override(**presets.iit3):
            with pytest.raises(ValueError, match="Zaeemzadeh"):
                bounds.distinction_phi_upper_bound((0,), (0,))

    def test_unsupported_mechanism_measure_raises(self):
        with config.override(mechanism_phi_measure="EMD"):
            with pytest.raises(ValueError, match="EMD"):
                bounds.sum_phi_distinctions_upper_bound(3)

    def test_system_guard_checks_partition_scheme(self):
        with config.override(system_partition_scheme="DIRECTED_BIPARTITION"):
            with pytest.raises(ValueError, match="partition scheme"):
                bounds.system_phi_upper_bound(3)

    def test_counts_are_measure_free(self):
        with config.override(**presets.iit3):
            assert bounds.number_of_possible_distinctions(3) == 7


class TestObjectBounds:
    def test_distinction_phi_upper_bound(self):
        bound = bounds.distinction_phi_upper_bound((0, 1), (1, 2, 3))
        assert bound.value == 6
        assert bound.certified
        assert bound.citation == "Theorem 1"
        assert "binary units" in bound.assumptions

    def test_distinction_phi_upper_bound_empty_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            bounds.distinction_phi_upper_bound((), (0,))
        with pytest.raises(ValueError, match="nonempty"):
            bounds.distinction_phi_upper_bound((0,), ())

    def test_partition_phi_upper_bound(self):
        # Bipartition of a 2-mechanism over itself severs 2 connections.
        partition = JointPartition(Part((0,), (0,)), Part((1,), (1,)))
        bound = bounds.partition_phi_upper_bound(partition)
        assert bound.value == partition.num_connections_cut() == 2
        assert bound.certified
        assert bound.citation == "Lemma 2"

    def test_relation_phi_upper_bound(self):
        bound = bounds.relation_phi_upper_bound([0.5, 2.0, 1.25])
        assert bound.value == 0.5
        assert bound.certified

    def test_relation_phi_upper_bound_empty_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            bounds.relation_phi_upper_bound([])

    def test_system_phi_upper_bound(self):
        bound = bounds.system_phi_upper_bound(4)
        assert bound.value == 12
        assert bound.certified
        assert bound.citation == "Table 2"
        assert any("self-connections" in a for a in bound.assumptions)
```

- [ ] **Step 2.2: Run to verify the new tests fail**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'distinction_phi_upper_bound'`.

- [ ] **Step 2.3: Implement the guard and the four bound functions**

Add to the import block of `pyphi/formalism/iit4/bounds.py`:

```python
from collections.abc import Iterable
from typing import Any

from pyphi.conf import config
```

Then append after the counting section:

```python
##############################################################################
# Domain guard
##############################################################################

# (version, measure) combinations for which the property-test battery in
# test/test_bounds.py confirms the bounds against the real pipeline.
MECHANISM_MEASURE_DOMAIN = frozenset(
    {
        ("IIT_4_0_2023", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "GENERALIZED_INTRINSIC_DIFFERENCE"),
    }
)
SYSTEM_MEASURE_DOMAIN = frozenset(
    {
        ("IIT_4_0_2023", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "INTRINSIC_INFORMATION"),
    }
)
# The n(n - 1) system bound counts connections severed by set partitions,
# which never cut self-connections; schemes that sever them break it.
SYSTEM_PARTITION_SCHEME_DOMAIN = frozenset({"DIRECTED_SET_PARTITION"})


def _require_valid_domain() -> None:
    """Raise unless the active config is in the confirmed mechanism-level domain."""
    version = config.formalism.iit.version
    measure = config.formalism.iit.mechanism_phi_measure
    if (version, measure) not in MECHANISM_MEASURE_DOMAIN:
        raise ValueError(
            f"the mechanism-level bounds are not confirmed for "
            f"(version={version!r}, mechanism_phi_measure={measure!r}); "
            f"confirmed combinations: {sorted(MECHANISM_MEASURE_DOMAIN)}. "
            f"See {CITATION}"
        )


def _require_valid_system_domain() -> None:
    """Raise unless the active config is in the confirmed system-level domain."""
    version = config.formalism.iit.version
    measure = config.formalism.iit.system_phi_measure
    if (version, measure) not in SYSTEM_MEASURE_DOMAIN:
        raise ValueError(
            f"the system-level bound is not confirmed for "
            f"(version={version!r}, system_phi_measure={measure!r}); "
            f"confirmed combinations: {sorted(SYSTEM_MEASURE_DOMAIN)}. "
            f"See {CITATION}"
        )
    scheme = config.formalism.iit.system_partition_scheme
    if scheme not in SYSTEM_PARTITION_SCHEME_DOMAIN:
        raise ValueError(
            f"the system-level bound assumes a partition scheme that does "
            f"not sever self-connections; got system partition scheme "
            f"{scheme!r}, confirmed: {sorted(SYSTEM_PARTITION_SCHEME_DOMAIN)}. "
            f"See {CITATION}"
        )


##############################################################################
# Per-object bounds
##############################################################################


def distinction_phi_upper_bound(
    mechanism: Iterable[int], purview: Iterable[int]
) -> UpperBound:
    """Upper bound on phi of a mechanism over a candidate purview.

    Theorem 1: phi(m, Z) <= |M| |Z|, the number of potential causal
    connections between the mechanism and the purview. Only the sizes
    matter.
    """
    _require_valid_domain()
    num_mechanism = len(tuple(mechanism))
    num_purview = len(tuple(purview))
    if num_mechanism < 1 or num_purview < 1:
        raise ValueError("mechanism and purview must be nonempty")
    return UpperBound(
        value=num_mechanism * num_purview,
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Theorem 1",
    )


def partition_phi_upper_bound(partition: Any) -> UpperBound:
    """Upper bound on phi of a mechanism-purview pair under a given partition.

    Lemma 2: phi(m, Z given theta) <= N(theta), the number of connections
    severed by the partition. Holds for any partitioning, valid or not.

    Args:
        partition: Any partition exposing ``num_connections_cut()``
            (e.g. :class:`~pyphi.models.partitions.JointPartition`).
    """
    _require_valid_domain()
    return UpperBound(
        value=partition.num_connections_cut(),
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Lemma 2",
    )


def relation_phi_upper_bound(relata_phis: Iterable[float]) -> UpperBound:
    """Upper bound on phi of a relation, given its relata's distinction phis.

    phi_r(d) <= min over relata of phi_d (Sec 2.2): the relation overlap
    is contained in every relatum's purview union.
    """
    _require_valid_domain()
    phis = tuple(float(phi) for phi in relata_phis)
    if not phis:
        raise ValueError("relata_phis must be nonempty")
    return UpperBound(
        value=min(phis),
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Sec 2.2",
    )


def system_phi_upper_bound(n: int) -> UpperBound:
    """Upper bound on system integrated information for n units.

    phi_s <= n(n - 1) (Table 2, citing Marshall et al. 2023): system phi
    is bounded by the number of connections cut by the selected
    partition, and set partitions sever at most n(n - 1) connections
    (all between-part connections at the atomic partition;
    self-connections are never cut).
    """
    _require_valid_system_domain()
    _require_positive(n)
    return UpperBound(
        value=n * (n - 1),
        certified=True,
        assumptions=(
            *_CORE_ASSUMPTIONS,
            "system partitions do not sever self-connections",
        ),
        citation="Table 2",
    )
```

- [ ] **Step 2.4: Run to verify all tests pass**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add domain guard and per-object bounds to iit4.bounds

Theorem 1 (phi <= |M||Z|), Lemma 2 (phi <= connections severed), the
relation bound (phi_r <= min relata phi_d), and the system bound
(phi_s <= n(n-1)). A strict guard raises ValueError outside the
confirmed (version, measure) domain; the system-level guard also
requires a partition scheme that does not sever self-connections, since
the n(n-1) ceiling counts between-part connections only."
```

---

### Task 3: Σφ_d bounds (I, II, III)

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py`
- Modify: `test/test_bounds.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `test/test_bounds.py`:

```python
class TestSumPhiDistinctions:
    @pytest.mark.parametrize("n", range(1, 13))
    def test_bound_i_matches_brute_force(self, n):
        # Eq 6: every mechanism at phi = |M| * n (purview = whole system).
        brute = sum(
            len(mechanism) * n
            for r in range(1, n + 1)
            for mechanism in itertools.combinations(range(n), r)
        )
        bound = bounds.sum_phi_distinctions_upper_bound(n, bound="I")
        assert bound.value == brute == n * n * 2 ** (n - 1)
        assert bound.certified
        assert bound.citation == "Eq 6"
        assert isinstance(bound.value, int)

    @pytest.mark.parametrize("n", range(1, 13))
    def test_bound_ii_matches_brute_force(self, n):
        # Eq 7: every mechanism at phi = |M| ** 2 (purview = mechanism).
        brute = sum(
            len(mechanism) ** 2
            for r in range(1, n + 1)
            for mechanism in itertools.combinations(range(n), r)
        )
        bound = bounds.sum_phi_distinctions_upper_bound(n, bound="II")
        assert bound.value == brute == n * (n + 1) * 2**n // 4
        assert not bound.certified
        assert any("unique purviews" in a for a in bound.assumptions)
        assert bound.citation == "Eq 7"

    def test_phi_e_star_endpoints(self):
        # K = 1: a single self-copy unit; severing the self-connection
        # halves the probability: phi = 1. K = N: complete partition fully
        # marginalizes every unit: phi = N ** 2.
        for n in range(1, 8):
            assert bounds._phi_e_star(n, 1) == pytest.approx(1.0)
            assert bounds._phi_e_star(n, n) == pytest.approx(float(n * n))

    def test_phi_e_star_hand_value(self):
        # N=3, K=2: MIP is the non-self-cutting bipartition;
        # phi = -2 * log2(3/4). Verified against the 2.0 pipeline.
        assert bounds._phi_e_star(3, 2) == pytest.approx(
            0.8300749985576875, abs=1e-12
        )

    def test_phi_e_star_below_theorem_1(self):
        # Theorem 3: for 1 < K < N the construction cannot achieve K ** 2.
        for n in range(3, 9):
            for k in range(2, n):
                assert bounds._phi_e_star(n, k) < k * k

    def test_bound_iii_hand_values(self):
        assert bounds.sum_phi_distinctions_upper_bound(2, bound="III").value == (
            pytest.approx(6.0)
        )
        assert bounds.sum_phi_distinctions_upper_bound(3, bound="III").value == (
            pytest.approx(12 + 3 * 0.8300749985576875)
        )

    def test_bound_iii_certificate(self):
        bound = bounds.sum_phi_distinctions_upper_bound(4, bound="III")
        assert not bound.certified
        assert any("conjecture" in a for a in bound.assumptions)
        assert bound.citation == "Sec 2.1.3"

    @pytest.mark.parametrize("n", range(2, 9))
    def test_bound_ordering(self, n):
        # Fig 3: Bound III <= Bound II <= Bound I (equality at n = 2).
        bound_i = float(bounds.sum_phi_distinctions_upper_bound(n, bound="I"))
        bound_ii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="II"))
        bound_iii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="III"))
        assert bound_iii <= bound_ii + 1e-9
        assert bound_ii <= bound_i

    def test_invalid_bound_id_raises(self):
        with pytest.raises(ValueError, match="bound"):
            bounds.sum_phi_distinctions_upper_bound(3, bound="IV")
```

- [ ] **Step 3.2: Run to verify the new tests fail**

Run: `uv run pytest test/test_bounds.py::TestSumPhiDistinctions -x -q`
Expected: FAIL with AttributeError on `sum_phi_distinctions_upper_bound`.

- [ ] **Step 3.3: Implement the Σφ_d bounds**

Append to `pyphi/formalism/iit4/bounds.py`:

```python
##############################################################################
# Sum of distinction phi
##############################################################################

_CONJECTURE_NOTE = (
    "conjectured: proven for reflexive selectivity-1 systems; "
    "generality is an open question in the paper"
)


def _log2_pi(n: int, k: int, a: int) -> float:
    """log2 of the partitioned single-unit effect probability pi(a).

    In the size-n, order-k high-selectivity construction, when the
    mechanism part connected to a purview unit has size a and contains
    that unit (S3 Appendix, Sec 3).
    """
    numerator = sum(math.comb(n - a, b) for b in range(k - a, n - a + 1))
    return math.log2(numerator) - (n - a)


def _log2_pi_bar(n: int, k: int, a: int) -> float:
    """log2 of pi-bar(a): as :func:`_log2_pi`, but the connected part does
    not contain the purview unit. Covers a = 0 (fully severed unit)."""
    numerator = sum(math.comb(n - a - 1, b - 1) for b in range(k - a, n - a + 1))
    return math.log2(numerator) - (n - a)


def _phi_e_star(n: int, k: int) -> float:
    """Integrated effect information of a size-k mechanism over itself in
    the high-selectivity construction (S3 Appendix, Sec 3).

    Selectivity is 1, so phi at a partition equals the informativeness
    lost. The MIP is among k // 2 + 1 candidates: the bipartitions that
    keep self-connections intact, and the cut severing one mechanism unit
    from all purview units (the only candidate for k = 1). Candidates are
    compared by value normalized by the number of connections severed;
    the returned phi is unnormalized. For k = n only the complete
    partition exists and phi equals n ** 2.
    """
    if k == n:
        return float(n * n)
    candidates: list[tuple[float, int]] = []
    for j in range(1, k // 2 + 1):
        value = -(j * _log2_pi(n, k, j) + (k - j) * _log2_pi(n, k, k - j))
        candidates.append((value, 2 * j * (k - j)))
    value = -(_log2_pi_bar(n, k, k - 1) + (k - 1) * _log2_pi(n, k, k - 1))
    candidates.append((value, k))
    return min(candidates, key=lambda c: (c[0] / c[1], c[0]))[0]


def sum_phi_distinctions_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on the sum of distinction phi for a system of n units.

    Bounds:
        ``"I"`` (Eq 6, certified, not achievable): every mechanism at
            phi = |M| n; equals (n**2 / 2) 2**n.
        ``"II"`` (Eq 7, conditional): assumes each purview is assigned to
            exactly one mechanism with matching sizes; equals
            (n (n+1) / 4) 2**n.
        ``"III"`` (Sec 2.1.3, conjectured): the numerical bound from the
            high-selectivity reflexive construction,
            sum over K of C(n, K) phi*_e(K).
    """
    _require_valid_domain()
    _require_positive(n)
    if bound == "I":
        return UpperBound(
            value=sum(k * n * math.comb(n, k) for k in range(1, n + 1)),
            certified=True,
            assumptions=_CORE_ASSUMPTIONS,
            citation="Eq 6",
        )
    if bound == "II":
        return UpperBound(
            value=sum(k * k * math.comb(n, k) for k in range(1, n + 1)),
            certified=False,
            assumptions=(
                *_CORE_ASSUMPTIONS,
                "unique purviews: each purview assigned to exactly one mechanism",
            ),
            citation="Eq 7",
        )
    if bound == "III":
        return UpperBound(
            value=sum(math.comb(n, k) * _phi_e_star(n, k) for k in range(1, n + 1)),
            certified=False,
            assumptions=(*_CORE_ASSUMPTIONS, _CONJECTURE_NOTE),
            citation="Sec 2.1.3",
        )
    raise ValueError(f"unknown bound id {bound!r}; expected 'I', 'II', or 'III'")
```

- [ ] **Step 3.4: Run to verify all tests pass**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: PASS.

- [ ] **Step 3.5: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add sum-of-distinction-phi bounds I/II/III to iit4.bounds

Bounds I and II are exact integer closed forms verified against
brute-force enumeration. Bound III is computed in closed form from the
S3-appendix binomial formulas for the high-selectivity construction:
the K/2 + 1 candidate partitions are evaluated analytically (selectivity
is 1, so phi reduces to a sum of log-probabilities), removing any
pipeline dependence. Endpoint and hand-derived values are pinned."
```

---

### Task 4: Σφ_r machinery + Eq 16 + big_phi

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py`
- Modify: `test/test_bounds.py`

- [ ] **Step 4.1: Write the failing tests**

Add to the import block of `test/test_bounds.py`:

```python
import numpy as np
import scipy.optimize
```

Then append:

```python
def _naive_subset_min_sum(values):
    """Brute force: sum of the minimum over all subsets of size >= 2."""
    values = list(values)
    total = 0.0
    for r in range(2, len(values) + 1):
        for combo in itertools.combinations(range(len(values)), r):
            total += min(values[i] for i in combo)
    return total


def _weighted_sorted_sum(values):
    """Eq 11 inner sum, expanded: ascending sort, i-th smallest element
    (0-based) is the minimum of 2**(R - 1 - i) - 1 subsets."""
    values = sorted(values)
    count = len(values)
    return sum(v * (2 ** (count - 1 - i) - 1) for i, v in enumerate(values))


def _table3_bound_i_nonself(n):
    """Verbatim Table 3 Bound I sum-of-relation-phi formula (no self term)."""
    total = 0
    for k in range(1, n + 1):
        exponent = sum(math.comb(n, i) for i in range(k, n + 1))
        group = math.comb(n, k)
        total += k * (2**exponent - 2 ** (exponent - group) - group)
    return n * total


def _table3_bound_ii_nonself(n):
    """Verbatim Table 3 Bound II sum-of-relation-phi formula (no self term)."""
    total = 0
    for k in range(1, n + 1):
        exponent = sum(math.comb(n - 1, i) for i in range(k - 1, n))
        group = math.comb(n - 1, k - 1)
        total += k * (2**exponent - 2 ** (exponent - group) - group)
    return n * total


class TestGroupedSubsetMinSum:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_matches_brute_force_bound_i_profile(self, n):
        expanded = [
            k for k in range(1, n + 1) for _ in range(math.comb(n, k))
        ]
        grouped = bounds._grouped_subset_min_sum(
            [(k, math.comb(n, k)) for k in range(1, n + 1)]
        )
        assert grouped == _naive_subset_min_sum(expanded)
        assert grouped == _weighted_sorted_sum(expanded)

    @pytest.mark.parametrize("n", range(2, 9))
    def test_matches_weighted_sum_float_profile(self, n):
        # Bound III ratios are floats; agreement within float tolerance.
        ratios = [bounds._phi_e_star(n, k) / k for k in range(1, n + 1)]
        expanded = [
            ratios[k - 1] for k in range(1, n + 1) for _ in range(math.comb(n, k))
        ]
        grouped = bounds._grouped_subset_min_sum(
            [(ratios[k - 1], math.comb(n, k)) for k in range(1, n + 1)]
        )
        assert grouped == pytest.approx(_weighted_sorted_sum(expanded), rel=1e-12)


class TestSumPhiRelations:
    @pytest.mark.parametrize("n", range(1, 11))
    def test_bound_i_matches_table3_verbatim(self, n):
        bound = bounds.sum_phi_relations_upper_bound(n, bound="I")
        self_term = n * n * 2 ** (n - 1)
        assert bound.value == _table3_bound_i_nonself(n) + self_term
        assert isinstance(bound.value, int)
        assert not bound.certified

    @pytest.mark.parametrize("n", range(1, 11))
    def test_bound_ii_matches_table3_verbatim(self, n):
        bound = bounds.sum_phi_relations_upper_bound(n, bound="II")
        self_term = n * (n + 1) * 2**n // 4
        assert bound.value == _table3_bound_ii_nonself(n) + self_term

    def test_hand_values_n2(self):
        assert bounds.sum_phi_relations_upper_bound(2, bound="I").value == 16
        assert bounds.sum_phi_relations_upper_bound(2, bound="II").value == 8
        assert bounds.sum_phi_relations_upper_bound(2, bound="III").value == (
            pytest.approx(14.0)
        )
        general = bounds.sum_phi_relations_upper_bound(2, bound="GENERAL")
        assert float(general) == pytest.approx(88 / 3)
        assert general.certified

    def test_profile_bounds_are_conditional(self):
        for bound_id in ("I", "II", "III"):
            bound = bounds.sum_phi_relations_upper_bound(3, bound=bound_id)
            assert not bound.certified
            assert any("profile" in a for a in bound.assumptions)

    @pytest.mark.parametrize("n", range(2, 9))
    def test_general_dominates_profile_bound_i(self, n):
        # Eq 16 uses the LP maximum, which dominates any specific profile.
        general = float(bounds.sum_phi_relations_upper_bound(n, bound="GENERAL"))
        profile = float(bounds.sum_phi_relations_upper_bound(n, bound="I"))
        assert general >= profile

    def test_lp_closed_form_matches_linprog(self):
        # Eq 14: max sum(y_i (2**(R - i) - 1)) over ascending y >= 0 with
        # sum(y) <= S equals S ((2**R - 1) / R - 1).
        rng = np.random.default_rng(20260610)
        for _ in range(20):
            num_relata = int(rng.integers(2, 9))
            budget = float(rng.uniform(0.5, 50.0))
            coeffs = np.array(
                [2.0 ** (num_relata - i) - 1 for i in range(1, num_relata + 1)]
            )
            constraints = np.zeros((num_relata, num_relata))
            constraints[0] = 1.0  # budget row
            for i in range(1, num_relata):
                constraints[i, i - 1] = 1.0  # y_{i-1} <= y_i  (ascending)
                constraints[i, i] = -1.0
            limits = np.zeros(num_relata)
            limits[0] = budget
            result = scipy.optimize.linprog(
                c=-coeffs,
                A_ub=constraints,
                b_ub=limits,
                bounds=[(0, None)] * num_relata,
            )
            assert result.success
            expected = budget * ((2.0**num_relata - 1) / num_relata - 1)
            assert -result.fun == pytest.approx(expected, rel=1e-9)


class TestBigPhi:
    def test_general_is_certified(self):
        bound = bounds.big_phi_upper_bound(3, bound="GENERAL")
        assert bound.certified
        expected = float(
            bounds.sum_phi_distinctions_upper_bound(3, bound="I")
        ) + float(bounds.sum_phi_relations_upper_bound(3, bound="GENERAL"))
        assert float(bound) == pytest.approx(expected)

    def test_profile_bounds_are_conditional(self):
        for bound_id in ("I", "II", "III"):
            assert not bounds.big_phi_upper_bound(3, bound=bound_id).certified
```

- [ ] **Step 4.2: Run to verify the new tests fail**

Run: `uv run pytest test/test_bounds.py::TestSumPhiRelations -x -q`
Expected: FAIL with AttributeError.

- [ ] **Step 4.3: Implement the Σφ_r machinery**

Add to the import block of `pyphi/formalism/iit4/bounds.py`:

```python
from fractions import Fraction
```

Then append:

```python
##############################################################################
# Sum of relation phi
##############################################################################


def _grouped_subset_min_sum(groups: list[tuple[float, int]]) -> float:
    """Sum, over all subsets of size >= 2 of a multiset of ratios, of the
    subset's minimum (the inner sum of Eq 11).

    The i-th smallest of R elements (1-based) is the minimum of
    2**(R - i) - 1 subsets. Equal-ratio groups are summed as geometric
    series: a group of multiplicity m with ``after`` elements above it
    has total weight 2**after (2**m - 1) - m. The computation is exact
    (arbitrary-precision int) when the ratios are ints.

    Args:
        groups: ``(ratio, multiplicity)`` pairs; order irrelevant.
    """
    groups = sorted(groups)
    total_count = sum(multiplicity for _, multiplicity in groups)
    result = 0
    position = 0  # number of elements strictly below the current group
    for ratio, multiplicity in groups:
        after = total_count - position - multiplicity
        weight = 2**after * (2**multiplicity - 1) - multiplicity
        result += ratio * weight
        position += multiplicity
    return result


def _relation_profile(
    n: int, bound: str
) -> tuple[list[tuple[float, int]], float, tuple[str, ...]]:
    """Per-unit (ratio, multiplicity) groups, self-relation term, and
    extra assumptions for a sum-of-relation-phi scenario.

    The profiles realize the corresponding sum-of-distinction-phi
    scenarios (Table 3):

    - ``"I"``: every purview is the whole system in a congruent maximal
      state, so every distinction relates over every unit; ratio |M|.
    - ``"II"``: every purview is the mechanism itself; a unit relates the
      mechanisms containing it; ratio |M|.
    - ``"III"``: the high-selectivity construction profile as implemented
      in the paper's published experiment code (ratio phi*_K / K over all
      distinctions). The paper text instead assumes cause purviews span
      the system (ratio phi*_K / n); the implemented profile dominates
      both readings.
    """
    if bound == "I":
        groups: list[tuple[float, int]] = [
            (k, math.comb(n, k)) for k in range(1, n + 1)
        ]
        self_term: float = sum(k * n * math.comb(n, k) for k in range(1, n + 1))
        extra = ("Bound I extremal purview profile (all purviews span the system)",)
    elif bound == "II":
        groups = [(k, math.comb(n - 1, k - 1)) for k in range(1, n + 1)]
        self_term = sum(k * k * math.comb(n, k) for k in range(1, n + 1))
        extra = (
            "unique purviews: each purview assigned to exactly one mechanism",
            "Bound II extremal purview profile (every purview is its mechanism)",
        )
    elif bound == "III":
        phi_star = {k: _phi_e_star(n, k) for k in range(1, n + 1)}
        groups = [(phi_star[k] / k, math.comb(n, k)) for k in range(1, n + 1)]
        self_term = sum(math.comb(n, k) * phi_star[k] for k in range(1, n + 1))
        extra = (
            _CONJECTURE_NOTE,
            "Bound III extremal purview profile (high-selectivity construction)",
        )
    else:
        raise ValueError(
            f"unknown bound id {bound!r}; expected 'I', 'II', 'III', or 'GENERAL'"
        )
    return groups, self_term, extra


def sum_phi_relations_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on the sum of relation phi (self-relations included).

    For ``bound`` in ``"I"``, ``"II"``, ``"III"``: the exact Eq 11
    evaluation of the corresponding extremal purview profile (the Table 3
    closed forms plus the self-relation term). These are scenario
    bounds: they assume the system's distinction profile matches the
    scenario, so they are not certified for arbitrary systems.

    For ``bound="GENERAL"``: the certified growth bound of Eq 16, built
    from S(o) <= n 2**(n-1) (Theorem 1) and |Z(o)| <= 2**n - 1 via the
    Eq 14 linear-program maximum, summed over all 2n unit-states, plus
    the Eq 6 ceiling on self-relations.
    """
    _require_valid_domain()
    _require_positive(n)
    if bound == "GENERAL":
        budget = Fraction(n * 2**n, 2)  # S(o) <= n 2**(n-1)
        num_relata = 2**n - 1  # |Z(o)| <= number of distinctions
        per_unit_state = budget * (Fraction(2**num_relata - 1, num_relata) - 1)
        exact = (
            Fraction(sum(k * n * math.comb(n, k) for k in range(1, n + 1)))
            + 2 * n * per_unit_state
        )
        value = int(exact) if exact.denominator == 1 else float(exact)
        return UpperBound(
            value=value,
            certified=True,
            assumptions=_CORE_ASSUMPTIONS,
            citation="Eq 16",
        )
    groups, self_term, extra = _relation_profile(n, bound)
    value = self_term + n * _grouped_subset_min_sum(groups)
    return UpperBound(
        value=value,
        certified=False,
        assumptions=(*_CORE_ASSUMPTIONS, *extra),
        citation="Eqs 11-15, Table 3",
    )


def big_phi_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on big phi: the sum of all distinction and relation phi.

    For ``bound`` in ``"I"``, ``"II"``, ``"III"``: the profile-consistent
    pair of sum bounds. For ``bound="GENERAL"``: the certified pair
    (Eq 6 + Eq 16).
    """
    distinctions = sum_phi_distinctions_upper_bound(
        n, bound="I" if bound == "GENERAL" else bound
    )
    relations = sum_phi_relations_upper_bound(n, bound=bound)
    assumptions = tuple(
        dict.fromkeys((*distinctions.assumptions, *relations.assumptions))
    )
    return UpperBound(
        value=distinctions.value + relations.value,
        certified=distinctions.certified and relations.certified,
        assumptions=assumptions,
        citation=f"{distinctions.citation} + {relations.citation}",
    )
```

- [ ] **Step 4.4: Run to verify all tests pass**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: PASS. The Table 3 verbatim identity tests are the load-bearing check that the grouped evaluator and the published closed forms agree exactly (integer equality).

- [ ] **Step 4.5: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add sum-of-relation-phi bounds and big-phi bounds to iit4.bounds

The Table 3 closed forms are computed by an exact grouped evaluation of
the Eq 11 subset-minimum sum (integer-exact for Bounds I/II), verified
against brute-force subset enumeration and the verbatim published
formulas. The Eq 14 linear-program maximum is verified against
scipy.optimize.linprog. The Table 3 family is exposed as
scenario-conditional (certified=False): each value is the exact Eq 11
evaluation of a specific extremal purview profile, not a theorem for
arbitrary systems. The certified general bound (Eq 16) is exposed as
bound='GENERAL' and dominates the profile bounds."
```

---

### Task 5: Construction helpers + `report()`

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py`
- Modify: `test/test_bounds.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `test/test_bounds.py`:

```python
class TestConstructionHelpers:
    def test_construction_tpm_n3_k2(self):
        # Hand-checked rows (little-endian state order): a unit turns OFF
        # with probability 1 iff it is OFF and at least one other is OFF.
        tpm = bounds._construction_tpm(3, 2)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(tpm, expected)

    def test_candidate_partition_count(self):
        for n in range(2, 8):
            for k in range(1, n):
                assert len(list(bounds._candidate_partitions(n, k))) == k // 2 + 1
            assert len(list(bounds._candidate_partitions(n, n))) == 1

    def test_candidate_partitions_sever_expected_connections(self):
        partitions = list(bounds._candidate_partitions(5, 4))
        # Bipartitions (1, 3), (2, 2): sever 2 j (k - j); special cut: k.
        assert sorted(p.num_connections_cut() for p in partitions) == [4, 6, 8]


class TestReport:
    def test_report_by_size(self):
        result = bounds.report(n=3)
        assert float(result["system_phi"]) == 6
        assert result["sum_phi_distinctions:I"].value == 36
        assert result["sum_phi_distinctions:II"].value == 24
        assert float(result["sum_phi_distinctions:III"]) == pytest.approx(
            12 + 3 * 0.8300749985576875
        )
        assert result["sum_phi_relations:GENERAL"].certified
        assert result["big_phi:GENERAL"].certified
        assert result["number_of_possible_distinctions"] == 7
        assert result["number_of_possible_relations"] == 127

    def test_report_requires_exactly_one_input(self):
        with pytest.raises(ValueError, match="exactly one"):
            bounds.report()
        with pytest.raises(ValueError, match="exactly one"):
            bounds.report(n=3, substrate=object())  # pyright: ignore[reportArgumentType]

    def test_report_from_substrate(self):
        from pyphi.examples import EXAMPLES

        substrate = EXAMPLES["substrate"]["basic"]()
        result = bounds.report(substrate=substrate)
        assert float(result["system_phi"]) == 6  # 3 binary units

    def test_report_rejects_nonbinary_substrate(self):
        class FakeTPM:
            alphabet_sizes = (2, 3)

        class FakeSubstrate:
            factored_tpm = FakeTPM()
            size = 2

        with pytest.raises(ValueError, match="binary"):
            bounds.report(substrate=FakeSubstrate())  # pyright: ignore[reportArgumentType]
```

- [ ] **Step 5.2: Run to verify the new tests fail**

Run: `uv run pytest test/test_bounds.py::TestConstructionHelpers test/test_bounds.py::TestReport -x -q`
Expected: FAIL with AttributeError.

- [ ] **Step 5.3: Implement construction helpers and report**

Add to the import block of `pyphi/formalism/iit4/bounds.py`:

```python
from typing import TYPE_CHECKING

import numpy as np

from pyphi import utils
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyphi.substrate import Substrate
```

Then append:

```python
##############################################################################
# High-selectivity construction (S3 Appendix, Sec 3)
##############################################################################


def _construction_tpm(n: int, k: int) -> NDArray[np.float64]:
    """State-by-node TPM of the size-n, order-k high-selectivity construction.

    Unit u turns OFF with probability 1 in exactly the states where u is
    OFF and at least k - 1 other units are OFF (S3 Appendix Eqs 18, 20);
    otherwise it turns ON with probability 1. In this TPM every size-k
    mechanism specifies itself (all-OFF) with probability 1. Rows are in
    little-endian state order.
    """
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    rows = []
    for state in utils.all_states(n):
        row = []
        for unit in range(n):
            zeros_elsewhere = sum(
                1 for other in range(n) if other != unit and state[other] == 0
            )
            specifies_off = state[unit] == 0 and zeros_elsewhere >= k - 1
            row.append(0.0 if specifies_off else 1.0)
        rows.append(row)
    return np.array(rows)


def _candidate_partitions(n: int, k: int):
    """Yield the k // 2 + 1 candidate MIPs for a size-k mechanism over
    itself in the size-n high-selectivity construction (S3 Appendix, Sec 3).

    For k < n: the non-self-cutting bipartitions with part sizes
    (j, k - j), then the cut severing mechanism unit 0 from all purview
    units. For k = n: only the complete partition.
    """
    mechanism = tuple(range(k))
    if k == n:
        yield JointPartition(Part(mechanism, ()), Part((), mechanism))
        return
    for j in range(1, k // 2 + 1):
        first = tuple(range(j))
        rest = tuple(range(j, k))
        yield JointPartition(Part(first, first), Part(rest, rest))
    yield JointPartition(Part(tuple(range(1, k)), mechanism), Part((0,), ()))


##############################################################################
# Report
##############################################################################


def report(
    n: int | None = None, substrate: Substrate | None = None
) -> dict[str, Any]:
    """All size-based bounds for a system of n binary units, in one call.

    Args:
        n: Number of binary units. Mutually exclusive with ``substrate``.
        substrate: A substrate whose size is used; its alphabet must be
            binary.

    Returns:
        Mapping from flat keys (e.g. ``"sum_phi_distinctions:I"``,
        ``"big_phi:GENERAL"``) to :class:`UpperBound` values, plus the
        ``int``-valued counting entries.
    """
    if (n is None) == (substrate is None):
        raise ValueError("provide exactly one of n or substrate")
    if substrate is not None:
        alphabet_sizes = substrate.factored_tpm.alphabet_sizes
        if not all(size == 2 for size in alphabet_sizes):
            raise ValueError(
                f"bounds assume binary units; alphabet sizes are {alphabet_sizes}"
            )
        n = substrate.size
    assert n is not None
    _require_positive(n)
    _require_valid_domain()
    _require_valid_system_domain()
    result: dict[str, Any] = {"system_phi": system_phi_upper_bound(n)}
    for bound_id in ("I", "II", "III"):
        result[f"sum_phi_distinctions:{bound_id}"] = sum_phi_distinctions_upper_bound(
            n, bound=bound_id
        )
        result[f"sum_phi_relations:{bound_id}"] = sum_phi_relations_upper_bound(
            n, bound=bound_id
        )
        result[f"big_phi:{bound_id}"] = big_phi_upper_bound(n, bound=bound_id)
    result["sum_phi_relations:GENERAL"] = sum_phi_relations_upper_bound(
        n, bound="GENERAL"
    )
    result["big_phi:GENERAL"] = big_phi_upper_bound(n, bound="GENERAL")
    result["number_of_possible_distinctions"] = number_of_possible_distinctions(n)
    result["number_of_possible_relations"] = number_of_possible_relations(n)
    result["number_of_possible_relation_faces_with_unique_purviews"] = (
        number_of_possible_relation_faces_with_unique_purviews(n)
    )
    return result
```

- [ ] **Step 5.4: Run to verify all tests pass**

Run: `uv run pytest test/test_bounds.py -x -q`
Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add construction helpers and report() to iit4.bounds

The high-selectivity construction TPM and its candidate MIPs are exposed
as module helpers so the tightness validation can drive the real
pipeline on the same objects the closed form describes. report() bundles
every size-based bound and count for a system size or Substrate."
```

---

### Task 6: Property tests against the pipeline (whitelist admission) + conjecture probes

**Files:**
- Modify: `test/test_bounds.py`

- [ ] **Step 6.1: Write the property tests**

Append to `test/test_bounds.py`. Add to the top-of-file import block:

```python
import warnings

from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.conf.formalism import IITConfig
from pyphi.examples import EXAMPLES
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.substrate import Substrate
from pyphi.system import System

from .hypothesis_utils import small_system
```

Then the tests:

```python
# The whitelist admission evidence: under every (version, measure)
# combination shipped in the domain frozensets, structures computed by the
# real pipeline never exceed the certified bounds. A combination whose
# tests are not green here must be removed from the domain.
DOMAIN_CONFIGS = {
    "iit4_2023": presets.iit4_2023,
    "iit4_2026": presets.iit4_2026,  # system measure: INTRINSIC_INFORMATION
    "iit4_2026_gid_system": {"iit": IITConfig(version="IIT_4_0_2026")},
}

PROPERTY_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ],
)

TOL = 1e-9


def _ces(system):
    return new_big_phi.ces(
        system,
        system_measure=resolve_system_measure(
            config.formalism.iit.system_phi_measure
        ),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    )


def _assert_certified_bounds_hold(system):
    n = len(system.node_indices)
    ces = _ces(system)
    sum_phi_d = 0.0
    for distinction in ces.distinctions:
        phi = float(distinction.phi)
        sum_phi_d += phi
        for side in (distinction.cause, distinction.effect):
            side_phi = float(side.phi)
            # Theorem 1.
            theorem_1 = bounds.distinction_phi_upper_bound(
                distinction.mechanism, side.purview
            )
            assert side_phi <= float(theorem_1) + TOL
            # Lemma 2: phi is bounded by the connections the MIP severed.
            lemma_2 = bounds.partition_phi_upper_bound(side.partition)
            assert side_phi <= float(lemma_2) + TOL
    # Eq 6.
    assert sum_phi_d <= float(
        bounds.sum_phi_distinctions_upper_bound(n, bound="I")
    ) + TOL
    # Relation bound. Partially structural in 2.0: Relation.phi is
    # |overlap| * min(phi_d / |purview union|), which is min-based by
    # construction; this is a consistency check, not independent evidence.
    sum_phi_r = 0.0
    for relation in ces.relations:  # pyright: ignore[reportGeneralTypeIssues]
        sum_phi_r += float(relation.phi)
        relata_phis = [float(distinction.phi) for distinction in relation]
        assert float(relation.phi) <= float(
            bounds.relation_phi_upper_bound(relata_phis)
        ) + TOL
    # Eq 6 + Eq 16.
    assert sum_phi_d + sum_phi_r <= float(
        bounds.big_phi_upper_bound(n, bound="GENERAL")
    ) + TOL
    return ces


class TestCertifiedBoundsAgainstPipeline:
    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    @pytest.mark.parametrize("example_name", ["basic", "xor", "grid3"])
    def test_examples(self, config_name, example_name):
        system = EXAMPLES["system"][example_name]()
        with config.override(**DOMAIN_CONFIGS[config_name]):
            _assert_certified_bounds_hold(system)

    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    def test_system_phi_bound_on_examples(self, config_name):
        system = EXAMPLES["system"]["basic"]()
        n = len(system.node_indices)
        with config.override(**DOMAIN_CONFIGS[config_name]):
            sia = new_big_phi.sia(
                system,
                system_measure=resolve_system_measure(
                    config.formalism.iit.system_phi_measure
                ),
                specification_measure=resolve_mechanism_measure(
                    config.formalism.iit.specification_measure
                ),
            )
            assert float(sia.phi) <= float(bounds.system_phi_upper_bound(n)) + TOL

    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    @PROPERTY_SETTINGS
    @given(data=st.data())
    def test_random_systems(self, config_name, data):
        with config.override(
            **DOMAIN_CONFIGS[config_name], validate_system_states=False
        ):
            system = data.draw(small_system(min_size=2, max_size=3))
            n = len(system.node_indices)
            sum_phi_d = 0.0
            for distinction in system.distinctions():
                phi = float(distinction.phi)
                sum_phi_d += phi
                for side in (distinction.cause, distinction.effect):
                    theorem_1 = bounds.distinction_phi_upper_bound(
                        distinction.mechanism, side.purview
                    )
                    assert float(side.phi) <= float(theorem_1) + TOL
                    lemma_2 = bounds.partition_phi_upper_bound(side.partition)
                    assert float(side.phi) <= float(lemma_2) + TOL
            assert sum_phi_d <= float(
                bounds.sum_phi_distinctions_upper_bound(n, bound="I")
            ) + TOL


class TestConjectureProbes:
    """Non-gating probes of the conditional/conjectured bounds.

    A genuine violation of Bound III on a real system would be a finding
    about the conjecture (its generality is an open question in the
    paper), not a test bug: report it, do not fail.
    """

    def test_random_deterministic_systems(self):
        rng = np.random.default_rng(20260610)
        n = 3
        bound_values = {
            bound_id: float(
                bounds.sum_phi_distinctions_upper_bound(n, bound=bound_id)
            )
            for bound_id in ("I", "II", "III")
        }
        violations = {"II": [], "III": []}
        for trial in range(20):
            tpm = (rng.random((2**n, n)) > rng.random()).astype(float)
            substrate = Substrate(tpm, cm=np.ones((n, n)))
            system = System(substrate, state=(0,) * n, node_indices=(0, 1, 2))
            with config.override(validate_system_states=False):
                sum_phi_d = sum(
                    float(d.phi) for d in system.distinctions()
                )
            # Certified: gating.
            assert sum_phi_d <= bound_values["I"] + TOL
            for bound_id in ("II", "III"):
                if sum_phi_d > bound_values[bound_id] + TOL:
                    violations[bound_id].append((trial, sum_phi_d))
        for bound_id, found in violations.items():
            if found:
                warnings.warn(
                    f"Bound {bound_id} exceeded by {len(found)}/20 random "
                    f"deterministic 3-unit systems: {found}. The bound is "
                    f"not certified; this is a data point about its "
                    f"domain, not a bug.",
                    stacklevel=1,
                )
```

- [ ] **Step 6.2: Run the property tests**

Run: `uv run pytest test/test_bounds.py::TestCertifiedBoundsAgainstPipeline test/test_bounds.py::TestConjectureProbes -x -q`
Expected: PASS (these test the *implementation already written*; they are admission evidence, not TDD-first tests). Budget a few minutes — full CES on 3-node systems across 3 configs.

**If a certified-bound assertion fails under some (version, measure) combination:** do NOT weaken the assertion. First check for a harness bug (wrong attribute, tolerance, state handling). If the violation is real, remove that combination from the corresponding `*_DOMAIN` frozenset in `bounds.py`, document why in a comment, and surface the finding to the user — a real violation of a certified bound under GID would mean binary GID does not reduce to the paper's intrinsic difference, which is exactly what this battery exists to detect.

- [ ] **Step 6.3: Check attribute-access assumptions if failures look like harness bugs**

The test code assumes: `ces.distinctions` iterable of objects with `.phi`, `.mechanism`, `.cause`/`.effect` (each with `.phi`, `.purview`, `.partition`); `ces.relations` iterable of `Relation` (frozenset of distinctions, `.phi`); `system.distinctions()` iterable of `Concept`. These were verified against `pyphi/models/mice.py:51-178`, `pyphi/relations.py:126-192`, and `pyphi/formalism/queries.py:315-343` during planning. If an `AttributeError` appears, read the actual model class and fix the test access path, not the module.

- [ ] **Step 6.4: Commit**

```bash
git add test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add pipeline property tests and conjecture probes for iit4.bounds

The certified bounds (Theorem 1, Lemma 2, the relation bound, Eq 6,
Eq 16, and the system bound) are asserted against real structures
computed by the pipeline under every (version, measure) combination in
the shipped domain whitelist, on example networks and Hypothesis-random
small systems. These tests are the whitelist admission evidence. The
conditional/conjectured Bounds II and III get non-gating violation
probes on seeded random systems: violations warn rather than fail,
since the generality of Bound III is an open question in the paper."
```

---

### Task 7: Bound III end-to-end measure parity (battery d)

**Files:**
- Modify: `test/test_bounds.py`

- [ ] **Step 7.1: Write the parity tests**

Append to `test/test_bounds.py`. Add to the import block: `from pyphi.direction import Direction`.

```python
class TestConstructionParity:
    """Battery (d): the closed-form Bound III machinery against the real
    pipeline on the construction TPM. This is the strongest check that
    2.0's measure semantics (binary GID, NUM_CONNECTIONS_CUT
    normalization) match the paper's intrinsic-difference setup."""

    @staticmethod
    def _construction_system(n, k):
        tpm = bounds._construction_tpm(n, k)
        substrate = Substrate(tpm, cm=np.ones((n, n)))
        return System(substrate, state=(0,) * n, node_indices=tuple(range(n)))

    @pytest.mark.parametrize(
        "n,k", [(n, k) for n in (2, 3, 4) for k in range(1, n + 1)]
    )
    def test_phi_e_star_matches_pipeline_on_candidates(self, n, k):
        system = self._construction_system(n, k)
        mechanism = tuple(range(k))
        mip = system.find_mip(
            Direction.EFFECT,
            mechanism,
            mechanism,
            partitions=list(bounds._candidate_partitions(n, k)),
        )
        assert float(mip.phi) == pytest.approx(
            bounds._phi_e_star(n, k), abs=1e-10
        )

    @pytest.mark.parametrize(
        "n,k", [(3, 2), (4, 2), (4, 3)]
    )
    def test_candidates_contain_the_global_mip(self, n, k):
        # The S3 narrowing argument: the MIP over ALL mechanism partitions
        # equals the MIP over the k // 2 + 1 candidates.
        system = self._construction_system(n, k)
        mechanism = tuple(range(k))
        full_mip = system.find_mip(Direction.EFFECT, mechanism, mechanism)
        assert float(full_mip.phi) == pytest.approx(
            bounds._phi_e_star(n, k), abs=1e-10
        )

    @pytest.mark.parametrize("n,k_star", [(3, 2), (3, 3)])
    def test_achieved_sum_phi_e_below_bound_iii(self, n, k_star):
        # The full structure of the construction must respect Bound III
        # (Fig 2: the bound is tight at the best k_star).
        system = self._construction_system(n, k_star)
        bound_iii = float(bounds.sum_phi_distinctions_upper_bound(n, bound="III"))
        achieved = 0.0
        with config.override(validate_system_states=False):
            for distinction in system.distinctions():
                achieved += float(distinction.effect.phi)
        assert achieved <= bound_iii + 1e-9
```

- [ ] **Step 7.2: Run the parity tests**

Run: `uv run pytest test/test_bounds.py::TestConstructionParity -x -q`
Expected: PASS. The N=3, K=2 case was verified during planning (pipeline returned 0.8300749985576875 with the bipartition as MIP). If `test_candidates_contain_the_global_mip` fails with a *lower* full-search phi, the narrowing argument transcription is wrong — STOP and compare the candidate set against S3 page 9 before touching the closed form.

- [ ] **Step 7.3: Run the full bounds test file plus a quick sanity sweep**

Run: `uv run pytest test/test_bounds.py -q`
Expected: all pass.

- [ ] **Step 7.4: Commit**

```bash
git add test/test_bounds.py
git -c commit.gpgsign=false commit -m "Add Bound III end-to-end parity tests for iit4.bounds

The closed-form phi*_e(K) values are asserted equal to the real
pipeline's find_mip on the construction TPM, both restricted to the
K/2 + 1 candidate partitions and against the unrestricted partition
search (verifying the S3 MIP-narrowing argument empirically), and the
construction's achieved sum of effect phi is asserted below Bound III."
```

---

### Task 8: Reference goldens from the author's code (battery c)

**Files:**
- Create: `test/data/bounds/reference_goldens.json`
- Create: `test/test_bounds_reference_golden.py`

- [ ] **Step 8.1: Resurrect the author's environment**

The author's code (github.com/zaeemzadeh/IIT-bounds, already cloned at `/tmp/IIT-bounds`) pins `pyphi @ feature/iit-4.0` (branch verified to exist: `b78d0e34`). Adapt the matching-goldens recipe:

```bash
uv venv /tmp/bounds-ref-env --python 3.11
uv pip install --python /tmp/bounds-ref-env/bin/python \
  "numpy<2" "pandas>=2,<2.3" pyyaml more-itertools ordered-set joblib \
  psutil scipy networkx toolz tqdm ray redis pyemd igraph plotly \
  "xarray<2024.3" ipywidgets
uv pip install --python /tmp/bounds-ref-env/bin/python --no-deps \
  "pyphi @ git+https://github.com/wmayner/pyphi.git@feature/iit-4.0"
```

If import fails on graphillion, create the stub-package shadow on PYTHONPATH (a `graphillion/__init__.py` defining a `setset` class that raises on use) — same workaround as the matching resurrection (see memory `reference_matching_old_env_recipe`). Run from a scratch CWD so no `pyphi_config.yml` is picked up. Smoke-test:

```bash
cd /tmp && /tmp/bounds-ref-env/bin/python -c "import pyphi; print(pyphi.__version__)"
```

- [ ] **Step 8.2: Write and run the driver**

Write `/tmp/bounds-goldens-generate.py`. It drives the author's own functions (transcribed from `bound_figures.ipynb` cells 3, 4, and 12, minus plotting/ray/tqdm) on the author's own `utils.py`:

```python
"""Generate bounds reference goldens by driving the author's code.

Runs ONLY under the resurrected environment (/tmp/bounds-ref-env) with
the IIT-bounds repo on the path. Transcribed from bound_figures.ipynb
(github.com/zaeemzadeh/IIT-bounds); the phi computations go through the
original pyphi pipeline (feature/iit-4.0), NOT through any 2.0 code.
"""

import json
import math
import sys

sys.path.insert(0, "/tmp/IIT-bounds")

import numpy as np
import pyphi
import utils  # the author's utils.py

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False
pyphi.config.WELCOME_OFF = True

N_RANGE = range(2, 8)  # extend to 8 if runtime allows


def generate_subsystem_high_selectivity(n_nodes, order):
    # Verbatim from bound_figures.ipynb cell 3.
    all_states = list(pyphi.utils.all_states(n_nodes))
    state = (0,) * n_nodes
    node_marginalizers = utils.node_marginalizers(n_nodes, all_states)
    sbn_marg = np.array(
        [
            utils.order_k_marginalizer(order, n, node_marginalizers)
            for n in range(n_nodes)
        ]
    ).T
    network = pyphi.Network(1 - sbn_marg)
    return pyphi.Subsystem(network, state)


def selectivity_1_candidate_mips(order, n_nodes):
    # Verbatim from bound_figures.ipynb cell 3.
    if order == n_nodes:
        yield pyphi.models.cuts.KPartition(
            pyphi.models.cuts.Part(mechanism=tuple(range(order)), purview=()),
            pyphi.models.cuts.Part(mechanism=(), purview=tuple(range(order))),
        )
    else:
        for k in range(1, order // 2 + 1):
            part1 = tuple(range(k))
            part2 = tuple(range(k, order))
            yield pyphi.models.cuts.KPartition(
                pyphi.models.cuts.Part(mechanism=part1, purview=part1),
                pyphi.models.cuts.Part(mechanism=part2, purview=part2),
            )
        yield pyphi.models.cuts.KPartition(
            pyphi.models.cuts.Part(
                mechanism=tuple(range(1, order)), purview=tuple(range(order))
            ),
            pyphi.models.cuts.Part(mechanism=(0,), purview=()),
        )


def phi_star(n_nodes, order):
    subsystem = generate_subsystem_high_selectivity(n_nodes, order)
    mechanism = tuple(range(order))
    effect = subsystem.find_mip(
        direction=pyphi.direction.Direction.EFFECT,
        mechanism=mechanism,
        purview=mechanism,
        partitions=list(selectivity_1_candidate_mips(order, n_nodes)),
    )
    return float(effect.phi)


def sum_of_minimum_among_subsets(values):
    # Verbatim from bound_figures.ipynb cell 12.
    counts = 2 ** (np.arange(len(values), 0, -1).astype(np.longdouble) - 1) - 1
    return np.sum(np.sort(values) * counts)


def powerset_sizes(n):
    return [
        len(m) for m in pyphi.utils.powerset(range(n), nonempty=True)
    ]


results = []
for n in N_RANGE:
    print(f"N = {n}", flush=True)
    phi_k = {}
    for k in range(1, n + 1):
        phi_k[k] = phi_star(n, k)
        print(f"  phi*_e({k}) = {phi_k[k]}", flush=True)
    bound_iii = sum(math.comb(n, k) * phi_k[k] for k in range(1, n + 1))
    sizes = powerset_sizes(n)
    # Fig 4 cell, Bound I profile.
    relations_i = float(
        n * sum_of_minimum_among_subsets(sizes) + sum(k * n for k in sizes)
    )
    # Fig 4 cell, Bound II profile.
    sizes_containing = [
        len(m) + 1 for m in pyphi.utils.powerset(range(n - 1), nonempty=False)
    ]
    relations_ii = float(
        n * sum_of_minimum_among_subsets(sizes_containing)
        + sum(k * k for k in sizes)
    )
    # Fig 4 cell, Bound III profile.
    ratios = [phi_k[k] / k for k in sizes]
    relations_iii = float(
        n * sum_of_minimum_among_subsets(ratios) + sum(phi_k[k] for k in sizes)
    )
    results.append(
        {
            "n": n,
            "phi_e_star": {str(k): phi_k[k] for k in range(1, n + 1)},
            "sum_phi_distinctions_iii": bound_iii,
            "sum_phi_relations_i": relations_i,
            "sum_phi_relations_ii": relations_ii,
            "sum_phi_relations_iii": relations_iii,
        }
    )

output = {
    "metadata": {
        "generator": "author's IIT-bounds experiment code",
        "iit_bounds_repo": "github.com/zaeemzadeh/IIT-bounds",
        "pyphi": "feature/iit-4.0 (b78d0e34)",
        "generated": "2026-06-10",
        "recipe": (
            "Driven through the original pyphi pipeline via "
            "bound_figures.ipynb functions (cells 3, 4, 12) on the "
            "author's utils.py; phi*_e from Subsystem.find_mip over the "
            "K/2+1 candidate partitions of the high-selectivity "
            "construction; relation sums via the notebook's "
            "sum_of_minimum_among_subsets profile evaluations."
        ),
    },
    "results": results,
}
with open("/tmp/bounds-reference-goldens.json", "w") as f:
    json.dump(output, f, indent=1)
print("WROTE /tmp/bounds-reference-goldens.json")
```

Run (background, from scratch CWD; budget ~minutes for N up to 7):

```bash
cd /tmp && /tmp/bounds-ref-env/bin/python /tmp/bounds-goldens-generate.py
```

If N=7 completes quickly, extend `N_RANGE` to `range(2, 9)` and rerun. Record the achieved range in the metadata. Then:

```bash
mkdir -p test/data/bounds
cp /tmp/bounds-reference-goldens.json test/data/bounds/reference_goldens.json
```

**Sanity gate before freezing:** spot-check `phi_e_star` for n=3, k=2 equals 0.8300749985576875 and `sum_phi_distinctions_iii` for n=2 equals 6.0. A mismatch means the old pipeline disagrees with the closed form — STOP, do not freeze, investigate (tie semantics and normalization are the likely suspects), and surface to the user.

- [ ] **Step 8.3: Write the golden test**

Create `test/test_bounds_reference_golden.py`:

```python
"""Golden tests against the author's original bounds code.

The reference values in ``test/data/bounds/reference_goldens.json`` were
computed by the experiment code released with Zaeemzadeh & Tononi (2024)
(github.com/zaeemzadeh/IIT-bounds) running on its pinned pre-2.0 pyphi
(branch feature/iit-4.0); see the fixture metadata for the generation
recipe. phi*_e values went through the original pipeline's find_mip on
the high-selectivity construction; this implementation computes them in
closed form from the S3-appendix binomial formulas, so agreement is a
cross-implementation check of both the formulas and the measure
semantics. The relation sums follow the published code's profile
evaluation (see the module docstring for the paper-text difference).
"""

import json
from pathlib import Path

import pytest

from pyphi.formalism.iit4 import bounds

FIXTURE = Path(__file__).parent / "data" / "bounds" / "reference_goldens.json"

with FIXTURE.open() as f:
    _GOLDENS = json.load(f)

RESULTS = _GOLDENS["results"]
REL = 1e-9


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
def test_phi_e_star_matches_reference(entry):
    n = entry["n"]
    for k_str, reference in entry["phi_e_star"].items():
        actual = bounds._phi_e_star(n, int(k_str))
        assert actual == pytest.approx(reference, rel=REL, abs=1e-12), (
            f"n={n}, k={k_str}"
        )


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
def test_sum_phi_distinctions_iii_matches_reference(entry):
    actual = float(
        bounds.sum_phi_distinctions_upper_bound(entry["n"], bound="III")
    )
    assert actual == pytest.approx(entry["sum_phi_distinctions_iii"], rel=REL)


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
@pytest.mark.parametrize("bound_id,key", [
    ("I", "sum_phi_relations_i"),
    ("II", "sum_phi_relations_ii"),
    ("III", "sum_phi_relations_iii"),
])
def test_sum_phi_relations_matches_reference(entry, bound_id, key):
    actual = float(
        bounds.sum_phi_relations_upper_bound(entry["n"], bound=bound_id)
    )
    assert actual == pytest.approx(entry[key], rel=REL)


def test_fixture_is_nonvacuous():
    assert len(RESULTS) >= 5
    assert any(entry["n"] >= 6 for entry in RESULTS)
```

- [ ] **Step 8.4: Run the golden tests**

Run: `uv run pytest test/test_bounds_reference_golden.py -q`
Expected: PASS. Tolerance note: the reference relation sums went through numpy float128 and JSON float64; the implementation is integer-exact for I/II — if those mismatch beyond 1e-9 relative, suspect the float128-to-float64 conversion first and check the integer value against `_table3_*_nonself` before suspecting either implementation.

- [ ] **Step 8.5: Commit**

```bash
git add test/data/bounds/reference_goldens.json test/test_bounds_reference_golden.py
git -c commit.gpgsign=false commit -m "Add reference goldens for iit4.bounds from the author's code

Frozen outputs of the IIT-bounds experiment code (Zaeemzadeh & Tononi
2024) running on its pinned pre-2.0 pyphi: phi*_e(K) per (N, K) through
the original pipeline's find_mip on the high-selectivity construction,
the Bound III distinction sum, and the Fig 4 relation-sum profile
evaluations, for N = 2..7. The closed-form implementation must
reproduce them; phi*_e agreement is a cross-implementation check of
both the S3 binomial formulas and the measure semantics."
```

(Adjust "N = 2..7" in the commit message to the achieved range.)

---

### Task 9: ROADMAP correction + changelog fragment

**Files:**
- Modify: `ROADMAP.md` (the P13 block, currently at lines ~2172–2204)
- Create: `changelog.d/zaeemzadeh-bounds.feature.md`

- [ ] **Step 9.1: Rewrite the ROADMAP P13 entry**

Replace the entire P13 block (from `**P13. Zaeemzadeh upper bounds for pruning**` through the `- *Leverage:*` line that precedes `### Phase G`) with:

```markdown
**P13. Zaeemzadeh upper bounds** — 🟡 **Sub-project 1 landed; sub-project 2 gated**

*Math correction (2026-06-10):* the original entry directed using the bounds in
`find_mip` to "prune partitions that cannot achieve the best-so-far φ". That is
not a valid use: mechanism φ and φ_s are *minima* over partitions, and upper
bounds prune *maximizations*; pruning a minimization needs per-partition lower
bounds, which the paper does not provide (the φ=0 shortcircuit already handles
the floor). The integration candidates that survive scrutiny are maximizations:
skipping candidate systems in `complexes()` whose certified φ_s ceiling cannot
beat the best-so-far, and capping mechanism purview search via Theorem 1
(φ ≤ |M||Z|).

*Sub-project 1 — bounds module (landed):* `pyphi/formalism/iit4/bounds.py`
exposes the published bound inventory as a research utility: per-object bounds
(Theorem 1, Lemma 2, relation, system), the Bound I/II/III families for Σφ_d
and Σφ_r (including the certified Eq 16 growth bound), counting functions, and
`report()`. Every function returns an `UpperBound` carrying its certificate
(certified / scenario-conditional / conjectured); a strict domain guard raises
outside the property-test-confirmed (version, measure) combinations. Bound III
is computed in closed form from the S3-appendix formulas and validated against
the 2.0 pipeline on the construction TPM and against reference goldens from
the author's released code. Spec:
`docs/superpowers/specs/2026-06-10-zaeemzadeh-bounds-design.md`.

*Sub-project 2 — search integration (gated):* candidate-system skipping in
`complexes()` and purview caps, consuming only theorem-certified bounds,
shipped behind a shadow-mode equality gate. Gated on a bite-rate study: per-pair
Theorem 1 bites only when best-so-far φ exceeds |M||Z|, and φ_s caps bite on
small candidates in high-φ regimes — whether integration pays is an empirical
question. If it does not, the sub-project-1 utility surface is the P13
deliverable on its own. Separate spec when the study runs.

- *Files:* `pyphi/formalism/iit4/bounds.py`, `test/test_bounds.py`,
  `test/test_bounds_reference_golden.py`, `test/data/bounds/reference_goldens.json`.
- *Risk:* Low (sub-project 1 is read-only with respect to the pipeline).
  Sub-project 2 risk is high if pruning is wrong; mitigated by shadow mode and
  by consuming certified bounds only.
- *Leverage:* Research utility now; possible complex-search speedup later.
```

- [ ] **Step 9.2: Create the changelog fragment**

Create `changelog.d/zaeemzadeh-bounds.feature.md`:

```markdown
Added `pyphi.formalism.iit4.bounds`: the upper bounds on IIT quantities
published in Zaeemzadeh & Tononi (2024, PLOS Comput Biol 20(8):e1012323)
as a standalone research utility. Bound functions return an `UpperBound`
carrying the value and its certificate (`certified`, `assumptions`,
`citation`): Theorem 1 (`distinction_phi_upper_bound`), Lemma 2
(`partition_phi_upper_bound`), the relation bound
(`relation_phi_upper_bound`), the system bound
(`system_phi_upper_bound`), the Bound I/II/III families for the sums of
distinction and relation phi (`sum_phi_distinctions_upper_bound`,
`sum_phi_relations_upper_bound`, `big_phi_upper_bound`, including the
certified Eq 16 growth bound as `bound="GENERAL"`), counting functions
for possible distinctions, relations, and relation faces, and a one-call
`report()`. Functions raise `ValueError` when the active configuration
is outside the confirmed (version, measure) domain. Bound III is
computed in closed form from the supplementary-material formulas and is
validated against the original published experiment code and the real
pipeline on the construction TPM.
```

- [ ] **Step 9.3: Commit**

```bash
git add ROADMAP.md changelog.d/zaeemzadeh-bounds.feature.md
git -c commit.gpgsign=false commit -m "Correct ROADMAP P13 and add bounds changelog fragment

The P13 entry's find_mip pruning claim is withdrawn: phi quantities are
minima over partitions, and upper bounds prune maximizations only. The
entry now records the landed bounds module (sub-project 1) and gates
search integration (sub-project 2) on a bite-rate study."
```

---

### Task 10: Full verification

- [ ] **Step 10.1: Full suite (doctest-inclusive), in background**

Run with `run_in_background=true`:

```bash
uv run pytest -q
```

Expected: previous baseline (1735 passed, 41 skipped, 3 xfailed) plus the new bounds tests (~60), no new failures. While it runs, proceed to 10.2.

- [ ] **Step 10.2: Lint and type-check the new module directly**

```bash
uv run ruff check pyphi/formalism/iit4/bounds.py test/test_bounds.py test/test_bounds_reference_golden.py
uv run ruff format --check pyphi/formalism/iit4/bounds.py test/test_bounds.py test/test_bounds_reference_golden.py
```

Expected: clean. (Pyright runs in the pre-commit hook; trust the hook per the repo's current pyright workaround.)

- [ ] **Step 10.3: Smoke the public surface**

```bash
uv run python -c "
from pyphi.formalism.iit4 import bounds
report = bounds.report(n=4)
for key, value in report.items():
    print(key, '=', float(value) if isinstance(value, bounds.UpperBound) else value)
"
```

Expected: a full report with no exceptions; `system_phi = 12.0`, `sum_phi_distinctions:I = 64.0`.

- [ ] **Step 10.4: Confirm the background suite is green, then report completion to the user**

No commit here unless fixes were needed.

---

## Verification summary

| Battery | Where | Gates |
|---|---|---|
| (a) formula verification | TestCounts brute force, TestSumPhiDistinctions brute force, TestGroupedSubsetMinSum, Table 3 verbatim, linprog | hard fail |
| (b) pipeline property tests | TestCertifiedBoundsAgainstPipeline (3 configs × examples + Hypothesis) | hard fail = whitelist eviction |
| (c) reference goldens | test_bounds_reference_golden.py | hard fail |
| (d) construction parity | TestConstructionParity (closed form == pipeline; candidates contain global MIP) | hard fail |
| conjecture probes | TestConjectureProbes | warn only |

## Risk register

| Risk | Mitigation |
|---|---|
| 2.0 GID does not reduce to the paper's intrinsic difference on binary systems | Battery (d) catches it exactly (closed form vs pipeline on the construction); battery (b) catches grosser mismatches |
| Old-pipeline tie/normalization semantics differ from the closed-form argmin | Goldens sanity gate in Step 8.2 before freezing; STOP on mismatch |
| Face-count semantics hypothesis wrong | Brute-force test at N=2,3 pins it; STOP on failure |
| Author's env unresolvable (dependency rot) | Recipe mirrors the verified matching resurrection; graphillion stub fallback; worst case, regenerate goldens from the notebook's published Fig 2/3 values and document |
| Float overflow in relation sums for large n | Documented (module docstring); ints exact for I/II; GENERAL falls back to int when integral |
| Property tests slow | max_examples=10, 3-node cap; heavier sweeps can be added under the slow marker later |

## ROADMAP linkage

Completes P13 sub-project 1. Sub-project 2 (search integration) remains gated on the bite-rate study per the corrected ROADMAP entry.
