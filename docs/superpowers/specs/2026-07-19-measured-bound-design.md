# Measured state-keyed certificate for Σφ_r — design

**Date:** 2026-07-19
**Status:** draft for review
**Scope:** `pyphi/formalism/iit4/bounds.py`, `test/integration/test_bounds.py`, docs.

## 1. Problem

`bounds.py` offers one certified ceiling on the sum of relation φ: the
size-based `GENERAL` growth bound (Eq 16), which uses the worst case over
all systems of a given size and is ~100–1000× looser than the true Σφ_r on
measured systems. The S(o)-certificate experiments
(`experiments/so_certificate_experiments/FINDINGS.md`) proved that
evaluating the paper's per-atom linear-program maximum (Eq 14) on the
*measured* per-atom profile of an actual distinction set — rather than the
worst-case profile — yields a bound that is certified unconditionally and
dramatically tighter (fixture `grid3`: measured ≈ 9.94 vs `GENERAL` ceiling
1270.29 vs true 3.78; measured tightness on Σφ_r over 759 nonzero records:
median 1.45×, max 41×). This is the open remainder of the Wave 7
anytime-bracket ROADMAP entry and triage item M3.

## 2. Mathematical basis (proved; equation numbers verified against the paper)

Notation, from Zaeemzadeh & Tononi (2024): atoms o are **state-tagged
units** (a unit in a state — Eq 10's objects; PyPhi's `UnitState`);
𝒵(o) = the distinctions whose `purview_union` contains o;
q_d = φ_d / |purview_union(d)| (the density); S(o) = Σ_{d ∈ 𝒵(o)} q_d;
g(k) = (2^k − 1 − k)/k.

For any resolved distinction set D under PyPhi's concrete relation
definition (Eq 8; self-relations Eq 9):

    Σφ_r ≤ [Σ_d |z*_c(d) ∩ z*_e(d)| · q_d]  +  Σ_o S(o) · g(|𝒵(o)|)

- The first term is the **exact** self-relation sum (Eq 9 summed) — a
  strict tightening over the paper's Eq 15, which bounds it by Σφ_d.
- The second term is Eq 14's LP maximum per atom, evaluated at the
  measured (S(o), |𝒵(o)|): the true per-atom contribution (Eq 11's inner
  sum) satisfies the LP's budget constraint with equality, so it is ≤ the
  LP maximum. Equivalently, by Chebyshev's sum inequality (FINDINGS proof).
- **State-keying is mandatory, not merely tighter**: keying atoms on unit
  indices with index-count denominators is unsound (a witnessed record has
  the index-keyed value below the true Σφ_r). Atoms must be `UnitState`
  pairs — exactly what `purview_union` contains.

The theorem is purely combinatorial over the relation formula
(`Relation.phi = |congruent overlap| · min density`, `pyphi/relations.py`)
and the non-negativity of φ_d. It does **not** assume binary units, a
conditionally independent TPM, or any particular mechanism measure — unlike
every existing bound in the module.

Verification: 801 records (seeds 555, 20260708; n = 2–4 random substrates +
pqr/grid3/residue fixtures), 0 bound violations, identity exact to 1.7e-11.

## 3. API

Two pure functions in `pyphi/formalism/iit4/bounds.py` (approach approved:
self-contained; no imports from `pyphi.relations`, no config-domain guard):

```python
def sum_phi_relations_measured_bound(distinctions) -> UpperBound
def big_phi_measured_bound(distinctions) -> UpperBound
```

- `distinctions`: any iterable of distinctions (a `CauseEffectStructure` /
  `ResolvedDistinctions` works directly; a plain list works too). Must be a
  resolved set — each mechanism at most once. Documented, not policed
  (duplicates would double-count densities).
- Empty input → bound 0.0 (exact: no distinctions, no relations).
- `sum_phi_relations_measured_bound` returns
  `UpperBound(value, certified=True, assumptions=…, citation="Eqs 9, 14")`
  where the assumptions name the measured resolved distinction set and the
  congruent-overlap relation definition — **not** `_CORE_ASSUMPTIONS`
  (binary units / conditionally independent TPM do not apply).
- `big_phi_measured_bound` = exact Σφ_d (a sum, not a bound) + the relation
  certificate; `certified=True`;
  `citation="exact Σφ_d + Eqs 9, 14"`. This is the certified upper endpoint
  on Φ for a complete distinction set.

### Computation (O(|D|·n))

Per distinction: density q = φ/|purview_union|; add
|purview_intersection|·q to the self term; append q to each atom group for
o ∈ purview_union. Then cross term = Σ_o S(o)·g(|𝒵(o)|). Sums via
`math.fsum`. The weight 2.0**k is computed in float: for |𝒵(o)| > 1023 it
overflows to `inf`, a valid (useless) ceiling — consistent with the
module's documented overflow behavior for relation-level sums.

### Docstrings

NumPy style, final-state voice. Notes sections carry the two subject-matter
facts future maintainers need: (1) atoms are state-tagged units and
index-keying is unsound; (2) the bound is certified for the measured
profile with no extremal-profile or binary/measure assumptions, which is
why these functions have no domain guard. References cite Zaeemzadeh &
Tononi (2024) Eqs 9/10/14 (verified against `papers/`).

## 4. Relationship to the exact analytical sum (honest framing)

For a complete distinction set the already-shipped closed-form identity
(`AnalyticalRelations.sum_phi`, Eq 11) gives Σφ_r *exactly* at the same
O(|D|·n) cost, so the measured bound is never the tighter number to quote.
Its role is as a **certificate object**: an `UpperBound` with proof
metadata in the bounds ecosystem, directly comparable against `GENERAL`
(quantifying how loose the worst case is on a real system), and the proved
upper endpoint for future partial-information use. The docstring states
this relationship explicitly and points to the exact sum.

## 5. Testing (`test/integration/test_bounds.py`)

Config pinned with the complete preset context managers where φ is
computed. New tests:

1. **Soundness on fixtures**: concrete-relations Σφ_r ≤ measured bound on
   PQR and a grid fixture; on PQR (no cross relations) the bound collapses
   to the exact value.
2. **Dominance chain**: exact analytical `sum_phi()` ≤ measured bound;
   measured bound ≤ `GENERAL` bound for the same system (binary, in-domain
   config).
3. **Big-phi composition**: `big_phi_measured_bound.value` == Σφ_d +
   `sum_phi_relations_measured_bound.value`; and ≥ the fixture's actual Φ.
4. **No-guard differentiators**: (a) computes and holds on a k-ary
   system, which the size-based `report()` refuses; (b) computes — with
   the same value — under an out-of-domain mechanism measure
   (`mechanism_phi_measure="ID"`), where `sum_phi_relations_upper_bound`
   raises `ValueError`. (The 2026 preset itself is in-domain for the
   size-based bounds, since its mechanism measure is still GID; the
   soundness tests cover it via the domain-config parametrization.)
5. **Empty input** → 0.0, certified.
6. **Hypothesis property (slow lane)**: random small binary substrates,
   concrete Σφ_r ≤ measured bound + tolerance — the verify script's check
   as a permanent property test.

## 6. Docs and integration

- Changelog fragment (`changelog.d/measured-bound.feature.md`).
- A short mention where the `GENERAL` bound is presented in the docs
  (located during planning), stating the measured certificate and its
  measured tightness.
- MCP surfaces: checked during planning; updated only if the bounds report
  is already exposed there.
- ROADMAP (merge flow): mark the Wave 7 anytime-bracket entry's
  still-open remainder landed; mark triage item M3 landed.

## 7. Non-goals

- **No `report()` integration**: `report()` is size-based and has no
  distinction set; adding a CES-computing path there would change its cost
  class.
- **No `check_phi_bound` wiring**: the exact analytical identity is a
  strictly stronger runtime check on relation sums and is already tested;
  a weaker inequality assertion adds no detection power.
- **No partial-information bracket**: combining measured S(o) with caps for
  uncomputed distinctions remains future work per the FINDINGS; its
  tightness is unmeasured.

## 8. Execution

Worktree `.claude/worktrees/measured-bound` (branch `measured-bound` from
`fd2083ea`). Completion gates: pathless `uv run pytest` green in the
worktree and in the main tree after merge; slow lane green; docs build
clean.
