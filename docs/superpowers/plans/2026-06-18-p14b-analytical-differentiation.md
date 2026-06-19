# P14b — Analytical cross-structure differentiation projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Differentiation.analytical_differentiation`, a closed-form differentiation D (Eq 16) computed by inclusion–exclusion over the unique triggered structures (`2^K − 1` `AnalyticalRelations.sum_phi()` calls), reading only each structure's distinctions — the only path to D when relations are represented analytically.

**Architecture:** A `cached_property` on the existing `Differentiation` dataclass (`pyphi/matching/differentiation.py`). D = Σφ_d over the identity-deduplicated union of distinctions, plus Σφ_r over the relation union computed as `Σ_{∅≠T} (−1)^(|T|+1) AnalyticalRelations(∩_{k∈T} D_k).sum_phi()`. Never enumerates a concrete relation. Cross-validated against the concrete `differentiation`.

**Tech Stack:** Python 3.12+, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-p14b-analytical-differentiation-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims; no new dependency.
- Validation only on the formalism side — the result must equal the concrete `differentiation` wherever that is computable. No change to `differentiation` / `perceptual_differentiation` or to any relations/matching machinery.
- Scope is D (Eq 16). The perceptual differentiation D_p stays research — do **not** add an `analytical_perceptual_differentiation`.
- Build on the tested stack: `AnalyticalRelations(...).sum_phi()` (`pyphi/relations.py`), `ResolvedDistinctions` (`pyphi/models/distinctions.py`).
- Use `uv run` for all Python commands. Final verification runs `uv run pytest` **with no path argument** (public surface; doctest sweep).
- Do not bypass pre-commit hooks. Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).

---

### Task 1: `analytical_differentiation` property + cross-validation tests

**Files:**
- Modify: `pyphi/matching/differentiation.py`
- Test: `test/test_differentiation.py`

**Interfaces:**
- Consumes: `Perception.ces.distinctions` (a `ResolvedDistinctions`); `AnalyticalRelations(...).sum_phi()`.
- Produces: `Differentiation.analytical_differentiation -> float`.

- [ ] **Step 1: Write the failing core tests**

Add to `test/test_differentiation.py` (the module already imports `pytest`, `examples`, `Differentiation`, `PerceptualSystem`, `Perception`, and defines the `perceptions` fixture and `_full_state`):

```python
def test_analytical_matches_concrete_disjoint(perceptions):
    # grid3's two structures are disjoint: the |T|>=2 cross terms vanish, so
    # analytical D = sum of each structure's big_phi, same as concrete.
    d = Differentiation(tuple(perceptions.values()))
    assert d.analytical_differentiation == pytest.approx(d.differentiation)


def test_analytical_single_structure_is_big_phi(perceptions):
    p = perceptions[(0,)]
    d = Differentiation((p,))
    assert d.analytical_differentiation == pytest.approx(float(p.ces.big_phi))


def test_analytical_duplicate_structure_is_idempotent(perceptions):
    p = perceptions[(0,)]
    once = Differentiation((p,))
    twice = Differentiation((p, p))
    assert twice.analytical_differentiation == pytest.approx(
        once.analytical_differentiation
    )


def test_analytical_empty_is_zero():
    assert Differentiation(()).analytical_differentiation == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_differentiation.py -q -k analytical`
Expected: FAIL (`Differentiation` has no attribute `analytical_differentiation`).

- [ ] **Step 3: Implement the property**

In `pyphi/matching/differentiation.py`, add `from itertools import combinations` to the imports, then add this `cached_property` to the `Differentiation` class (after `perceptual_differentiation`):

```python
    @cached_property
    def analytical_differentiation(self) -> float:
        """Differentiation D (Eq 16), in closed form without enumerating
        concrete relations.

        Equal to :attr:`differentiation` wherever that is computable, but reads
        only each structure's ``distinctions`` (never ``relations``), so it is
        the path to use when the structures carry ``AnalyticalRelations`` (which
        are not iterable). D splits into the distinction union ``Σφ_d`` plus the
        relation union, the latter computed by inclusion–exclusion over the
        unique structures:

            Σ_r φ_r = Σ_{∅≠T} (−1)^(|T|+1) AnalyticalRelations(∩_{k∈T} D_k).sum_phi()

        Cost is ``2**K − 1`` analytical relation-sum calls for ``K`` unique
        structures; the method targets the small-``K`` (small sensory interface)
        regime where concrete relation enumeration is the bottleneck.
        """
        if not self.perceptions:
            return 0.0

        from pyphi.models.distinctions import ResolvedDistinctions
        from pyphi.relations import AnalyticalRelations

        # Distinction union term: Σφ_d over the identity-deduplicated union.
        union_distinctions: dict = {}
        for perception in self.perceptions:
            for distinction in perception.ces.distinctions:
                union_distinctions.setdefault(distinction, distinction)
        distinction_sum = sum(float(d.phi) for d in union_distinctions)

        # Relation union term: inclusion–exclusion over the unique structures
        # (deduplicated by distinction set — duplicates do not change the union).
        structures = list(
            {frozenset(p.ces.distinctions) for p in self.perceptions}
        )
        relation_sum = 0.0
        for size in range(1, len(structures) + 1):
            sign = 1.0 if size % 2 == 1 else -1.0
            for subset in combinations(structures, size):
                common = frozenset.intersection(*subset)
                if common:
                    relations = AnalyticalRelations(ResolvedDistinctions(common))
                    relation_sum += sign * float(relations.sum_phi())
        return float(distinction_sum + relation_sum)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_differentiation.py -q -k analytical`
Expected: PASS.

- [ ] **Step 5: Write the cross-term + analytical-relations tests**

These validate the already-implemented property on the cases that exercise the strict-subset cross term and the analytical-relations input. Add to `test/test_differentiation.py`:

```python
def _concrete_ces_subset(ces, n_keep):
    """A CES over the first ``n_keep`` distinctions of ``ces``, concrete relations."""
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.relations import relations as compute_relations

    sub_distinctions = ResolvedDistinctions(list(ces.distinctions)[:n_keep])
    sub_relations = compute_relations(sub_distinctions, "CONCRETE")
    return CauseEffectStructure(
        sia=ces.sia, distinctions=sub_distinctions, relations=sub_relations
    )


def test_analytical_partial_overlap_matches_concrete(perceptions):
    # A structure and a strict sub-structure of it: the union is the full
    # structure, so D = big_phi(full). The |T|=2 term restricts to the shared
    # (sub) distinctions, exercising the common-distinction restriction + sign.
    full = perceptions[(0,)]
    sub_ces = _concrete_ces_subset(full.ces, len(full.ces.distinctions) - 1)
    sub = Perception(ces=sub_ces, triggered_tpm=full.triggered_tpm, stimulus=(0,))
    d = Differentiation((full, sub))
    assert d.analytical_differentiation == pytest.approx(d.differentiation)
    assert d.analytical_differentiation == pytest.approx(float(full.ces.big_phi))


def test_analytical_runs_on_analytical_relations_where_concrete_cannot(perceptions):
    from pyphi.models.ces import CauseEffectStructure
    from pyphi.relations import AnalyticalRelations

    concrete = perceptions[(0,)]
    analytical_ces = CauseEffectStructure(
        sia=concrete.ces.sia,
        distinctions=concrete.ces.distinctions,
        relations=AnalyticalRelations(concrete.ces.distinctions),
    )
    analytical_p = Perception(
        ces=analytical_ces,
        triggered_tpm=concrete.triggered_tpm,
        stimulus=(0,),
    )
    d_analytical = Differentiation((analytical_p,))
    d_concrete = Differentiation((concrete,))
    # The analytical path runs and agrees with the concrete value...
    assert d_analytical.analytical_differentiation == pytest.approx(
        d_concrete.differentiation
    )
    # ...while the concrete path cannot walk non-iterable AnalyticalRelations.
    with pytest.raises(TypeError):
        _ = d_analytical.differentiation
```

- [ ] **Step 6: Run the full differentiation test file**

Run: `uv run pytest test/test_differentiation.py -q`
Expected: PASS (all existing + new tests).

- [ ] **Step 7: Commit**

```bash
git add pyphi/matching/differentiation.py test/test_differentiation.py
git commit -m "Add analytical cross-structure differentiation (closed-form D)

Differentiation.analytical_differentiation computes D (Eq 16) by
inclusion-exclusion over unique structures (2^K-1 AnalyticalRelations.sum_phi()
calls), reading only distinctions. Equals the concrete differentiation where
that runs, and is the only path to D when relations are analytical (not
iterable). D_p stays research."
```

---

### Task 2: Changelog, roadmap, and full verification

**Files:**
- Create: `changelog.d/analytical-differentiation.feature.md`
- Modify: `ROADMAP.md` (P14b analytical projection dashboard row; Wave 2 archive note; Landed prose line)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/analytical-differentiation.feature.md`:

```markdown
Added `pyphi.matching.Differentiation.analytical_differentiation`: the
cross-structure differentiation D computed in closed form, by inclusion-exclusion
over the unique triggered structures, without enumerating concrete relations.
It equals the concrete `differentiation` wherever that is computable, and is the
only way to compute D when the structures carry `AnalyticalRelations` (which are
not iterable). The perceptual differentiation stays concrete.
```

- [ ] **Step 2: Update the ROADMAP dashboard row**

In `ROADMAP.md`, change the `P14b analytical projection` row status from `🟡 open` to `✅ landed` and update its one-line to:

```markdown
| P14b analytical projection | ✅ landed | 2 | Closed-form differentiation D (`Differentiation.analytical_differentiation`): inclusion-exclusion over unique triggered structures, `Σ_T (−1)^(|T|+1) AnalyticalRelations(∩ D_k).sum_phi()` (`2^K−1` analytical calls, no concrete relation enumeration). Equals the concrete `differentiation` where that runs, and is the only path to D when relations are `AnalyticalRelations` (not iterable). The perception-maximized projection (D_p; weight has a mean-of-triggering factor that breaks the pure-min algebra) stays open research. |
```

- [ ] **Step 3: Update the Wave 2 archive note and the Landed prose line**

In `ROADMAP.md`, find the Wave 2 archive bullet `**P14b tail — environment generation (landed 2026-06-18) + analytical projection.**` and update its `*Analytical projection — open:*` clause to past tense: record that the φ-maximized differentiation D landed as `Differentiation.analytical_differentiation` via inclusion-exclusion over unique structures (`2^K−1` `AnalyticalRelations.sum_phi()` calls), cross-validated against the concrete `differentiation`, and that the perception-maximized projection (D_p) stays open research. In the `### ✅ Landed` prose line near the top, append `· P14b analytical-projection (closed-form differentiation D)`.

- [ ] **Step 4: Run the full verification gate**

Run: `uv run pytest`
Expected: PASS with **no path argument** (collects `pyphi/` + `test/` doctests and `test/test_differentiation.py`). If it errors at collection on matplotlib, run `uv sync --all-extras` first.

- [ ] **Step 5: Commit**

```bash
git add changelog.d/analytical-differentiation.feature.md ROADMAP.md
git commit -m "Mark P14b analytical projection landed: changelog + roadmap"
```

---

## Self-Review

**Spec coverage:**
- Closed-form D via inclusion–exclusion (spec §4.1–4.2) → Task 1 Step 3.
- API: `analytical_differentiation` property, no D_p variant (spec §4.3) → Task 1 Step 3.
- Cost boundary documented in the docstring (spec §4.4) → Task 1 Step 3.
- Testing (spec §5): cross-validation (disjoint), single-structure, duplicate/idempotent, disjoint sum, analytical-relations input, hand-computed partial-overlap inclusion–exclusion → Task 1 Steps 1 & 5.
- Roadmap + changelog (spec §7) → Task 2.

**Placeholder scan:** none — every code step shows complete code. Task 2 Step 3 is a localized prose edit to an existing ROADMAP bullet (read the bullet, update the named clause).

**Type consistency:** `analytical_differentiation` returns `float`. It reads `perception.ces.distinctions` (a `ResolvedDistinctions`), deduplicates distinctions via their identity hash/eq, and wraps each common-distinction set in `ResolvedDistinctions(...)` before passing to `AnalyticalRelations(...)` — matching the type `ces.distinctions` already carries and that `AnalyticalRelations.sum_phi()` consumes. The `_concrete_ces_subset` test helper builds a `CauseEffectStructure` with `relations(sub_distinctions, "CONCRETE")`, the same concrete-relations constructor used elsewhere.
