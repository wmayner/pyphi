# Measured State-Keyed Certificate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `sum_phi_relations_measured_bound` and `big_phi_measured_bound` to `pyphi/formalism/iit4/bounds.py` — certified upper bounds evaluated on the measured per-atom profile of a distinction set.

**Architecture:** Two pure functions appended to the "Sum of relation phi" section of `bounds.py`, self-contained (no imports from `pyphi.relations`, no config reads, no domain guard). Tests extend `test/integration/test_bounds.py`, reusing its `_ces` helper, `DOMAIN_CONFIGS`, and `TOL`.

**Tech Stack:** Python 3.13, pytest, Hypothesis (existing `small_system` strategy).

**Spec:** `docs/superpowers/specs/2026-07-19-measured-bound-design.md`

## Global Constraints

- Bound: `Σφ_r ≤ [Σ_d |z*_c(d) ∩ z*_e(d)| · q_d] + Σ_o S(o) · g(|𝒵(o)|)` with `q_d = φ_d/|purview_union|`, `g(k) = (2^k − 1 − k)/k`; atoms o are **state-tagged units** (`UnitState` objects from `purview_union`) — never bare unit indices.
- Citations (verified against `papers/2024__zaeemzadeh-tononi__upper-bounds.pdf`): relation formula Eq 8, self-relations Eq 9, 𝒵(o) Eq 10, identity Eq 11, LP + maximum Eqs 13–14, profile bound Eq 15. `UpperBound.citation` strings: `"Eqs 9, 14"` and `"exact Σφ_d + Eqs 9, 14"`.
- `certified=True`; assumptions do **not** include `_CORE_ASSUMPTIONS` (no binary-units / TPM-factorization / measure assumption).
- Docstrings: NumPy style, final-state voice, Unicode symbols (`φ`, `Σ`, `𝒵`); no planning-artifact references anywhere.
- Tests pin formalism with the preset-sourced `DOMAIN_CONFIGS` entries already in the test file — never a hand-listed subset.
- Test runs: redirect to a log and read the summary line; never pipe pytest through `tail`/`head`.
- Commit trailers (exact): `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe`. Check `git log --oneline -1` after every commit (hooks abort silently).
- Worktree: `.claude/worktrees/measured-bound`, branch `measured-bound`. All commands run from the worktree root with `uv run`.

---

### Task 1: Bound functions + core tests

**Files:**
- Modify: `pyphi/formalism/iit4/bounds.py` (insert after `big_phi_upper_bound`, before the "High-selectivity construction" section header)
- Test: `test/integration/test_bounds.py` (new class after `TestCertifiedBoundsAgainstPipeline`; new imports)

**Interfaces:**
- Produces: `sum_phi_relations_measured_bound(distinctions) -> UpperBound` and `big_phi_measured_bound(distinctions) -> UpperBound`, where `distinctions` is any iterable of distinction objects exposing `.phi`, `.purview_union`, `.purview_intersection`. Tasks 2–3 rely on these exact names.

- [ ] **Step 1: Confirm fixture facts before writing assertions**

Run in the worktree (values feed the test expectations; do not skip):

```bash
uv run python - <<'EOF'
import math
import pyphi
from pyphi import config
from pyphi.conf import presets

with config.override(**presets.iit4_2023, relation_computation="CONCRETE"):
    for name in ("pqr_system", "grid3_system"):
        system = getattr(pyphi.examples, name)()
        ces = system.ces()
        ds = list(ces.distinctions)
        sum_r = math.fsum(float(r.phi) for r in ces.relations)
        print(name, "n_distinctions:", len(ds), "sum_phi_r:", sum_r)
EOF
```

Expected: `pqr_system` has distinctions with `sum_phi_r` = 0.0 (the FINDINGS collapse case); `grid3_system` has `sum_phi_r` ≈ 3.78. If pqr's sum is nonzero under this pin, stop and re-derive the collapse expectation before writing Step 2's `test_pqr_collapses_to_exact`.

- [ ] **Step 2: Write the failing tests**

In `test/integration/test_bounds.py`, add to the imports block (alphabetical within the `pyphi` group):

```python
from pyphi import examples
from pyphi.relations import AnalyticalRelations
```

Add after `TestCertifiedBoundsAgainstPipeline` (before `TestConjectureProbes`):

```python
class TestMeasuredBounds:
    """The measured state-keyed certificates: Eq 14's linear-program maximum
    evaluated on the measured per-atom profile, plus the exact Eq 9
    self-relation sum."""

    @staticmethod
    def _concrete(system):
        """CES with enumerated relations; returns (distinctions, Σφ_r)."""
        with config.override(relation_computation="CONCRETE"):
            ces = _ces(system)
        return list(ces.distinctions), math.fsum(float(r.phi) for r in ces.relations)

    @pytest.mark.parametrize("config_name", sorted(DOMAIN_CONFIGS))
    @pytest.mark.parametrize("example_name", ["basic", "grid3"])
    def test_soundness_on_examples(self, config_name, example_name):
        system = EXAMPLES["system"][example_name]()
        with config.override(**DOMAIN_CONFIGS[config_name]):
            distinctions, sum_phi_r = self._concrete(system)
            measured = bounds.sum_phi_relations_measured_bound(distinctions)
        assert sum_phi_r <= float(measured) + TOL
        assert measured.certified
        assert measured.citation == "Eqs 9, 14"

    def test_pqr_collapses_to_exact(self):
        system = examples.pqr_system()
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            distinctions, sum_phi_r = self._concrete(system)
            measured = bounds.sum_phi_relations_measured_bound(distinctions)
        assert sum_phi_r == pytest.approx(0.0, abs=TOL)
        assert float(measured) == pytest.approx(sum_phi_r, abs=TOL)

    def test_dominance_chain(self):
        system = EXAMPLES["system"]["grid3"]()
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            with config.override(relation_computation="CONCRETE"):
                ces = _ces(system)
            distinctions = list(ces.distinctions)
            exact_concrete = math.fsum(float(r.phi) for r in ces.relations)
            exact_analytical = float(AnalyticalRelations(ces.distinctions).sum_phi())
            measured = bounds.sum_phi_relations_measured_bound(distinctions)
            general = bounds.sum_phi_relations_upper_bound(
                len(system.node_indices), bound="GENERAL"
            )
        # Identity: the analytical closed form equals the enumerated sum.
        assert exact_analytical == pytest.approx(exact_concrete, abs=TOL)
        # Chain: exact <= measured certificate <= worst-case ceiling.
        assert exact_concrete <= float(measured) + TOL
        assert float(measured) <= float(general) + TOL
        # The measured certificate is far below the worst case on grid3
        # (FINDINGS: ~9.94 vs 1270.29, a ~128x gap; assert a conservative 50x).
        assert float(measured) < float(general) / 50

    def test_big_phi_composition(self):
        system = EXAMPLES["system"]["basic"]()
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            distinctions, sum_phi_r = self._concrete(system)
            sum_phi_d = math.fsum(float(d.phi) for d in distinctions)
            relation_bound = bounds.sum_phi_relations_measured_bound(distinctions)
            big = bounds.big_phi_measured_bound(distinctions)
        assert float(big) == pytest.approx(
            sum_phi_d + float(relation_bound), abs=1e-12
        )
        # Dominates the structure's actual Φ = Σφ_d + Σφ_r.
        assert sum_phi_d + sum_phi_r <= float(big) + TOL
        assert big.certified
        assert big.citation == "exact Σφ_d + Eqs 9, 14"

    def test_empty_distinctions(self):
        for fn in (
            bounds.sum_phi_relations_measured_bound,
            bounds.big_phi_measured_bound,
        ):
            result = fn(())
            assert float(result) == 0.0
            assert result.certified

    def test_accepts_plain_iterable(self):
        system = EXAMPLES["system"]["basic"]()
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            distinctions, _ = self._concrete(system)
            from_list = bounds.sum_phi_relations_measured_bound(distinctions)
            from_generator = bounds.sum_phi_relations_measured_bound(
                d for d in distinctions
            )
        assert float(from_list) == float(from_generator)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest test/integration/test_bounds.py::TestMeasuredBounds -q > /tmp/t1-fail.log 2>&1; tail -3 /tmp/t1-fail.log
```

Expected: errors with `AttributeError: module ... has no attribute 'sum_phi_relations_measured_bound'`.

- [ ] **Step 4: Implement the functions**

In `pyphi/formalism/iit4/bounds.py`, insert after `big_phi_upper_bound` (before the `# High-selectivity construction` section banner):

```python
##############################################################################
# Measured certificates (computed from a distinction set, not a size)
##############################################################################

_MEASURED_ASSUMPTIONS = (
    "resolved distinction set (each mechanism at most once)",
    "relations defined by congruent overlap of maximal-state purviews (Eqs 8-9)",
)


def sum_phi_relations_measured_bound(distinctions: Iterable[Any]) -> UpperBound:
    """Certified upper bound on Σφ_r from a measured distinction set.

    Evaluates the per-atom linear-program maximum (Eqs 13-14) on the
    measured profile: each atom o — a unit in a state, drawn from the
    distinctions' purview unions (Eq 10) — receives the total φ density
    S(o) of the distinctions containing it, and contributes at most
    S(o)·(2^k − 1 − k)/k, where k is the number of those distinctions.
    The self-relation sum is carried exactly (Eq 9), a strict tightening
    of Eq 15. Cost is O(|D|·n); no relations are enumerated.

    Parameters
    ----------
    distinctions : iterable
        The distinctions of a Φ-structure. Must be a resolved set — each
        mechanism at most once; duplicates would contribute their
        densities twice.

    Returns
    -------
    UpperBound
        Certified. Unlike the size-based bounds, this holds with no
        binary-units, TPM-factorization, or measure assumption: the
        theorem is combinatorial over the relation formula and the
        non-negativity of φ, so there is no config domain guard.

    Notes
    -----
    Atoms are state-tagged units, exactly the objects in each
    distinction's ``purview_union``. Grouping by bare unit index with
    index-count denominators is unsound: when a purview union contains
    the same unit in two states, the merged density can fall below the
    true per-atom contribution, and the resulting value can fall below
    Σφ_r itself.

    The certificate holds because the true per-atom contribution (the
    inner sum of Eq 11) satisfies the linear program's budget constraint
    (Eq 13) with equality at the measured S(o), so it cannot exceed the
    program's maximum (Eq 14). For a complete distinction set the exact
    Σφ_r is available at the same cost via
    :meth:`pyphi.relations.AnalyticalRelations.sum_phi`; this function's
    value is the auditable certificate, directly comparable to the
    worst-case ceiling of :func:`sum_phi_relations_upper_bound`.

    A group of more than 1023 distinctions sharing an atom overflows the
    2^k weight to ``inf``, a valid (if uninformative) ceiling.
    """
    self_terms: list[float] = []
    groups: dict[Any, list[float]] = {}
    for distinction in distinctions:
        union = distinction.purview_union
        if not union:
            continue
        density = float(distinction.phi) / len(union)
        self_terms.append(len(distinction.purview_intersection) * density)
        for atom in union:
            groups.setdefault(atom, []).append(density)
    cross_terms = [
        math.fsum(densities) * (2.0 ** len(densities) - 1.0 - len(densities))
        / len(densities)
        for densities in groups.values()
        if len(densities) > 1
    ]
    return UpperBound(
        value=math.fsum(self_terms) + math.fsum(cross_terms),
        certified=True,
        assumptions=_MEASURED_ASSUMPTIONS,
        citation="Eqs 9, 14",
    )


def big_phi_measured_bound(distinctions: Iterable[Any]) -> UpperBound:
    """Certified upper bound on Φ from a measured distinction set.

    The exact sum of distinction φ plus the relation certificate of
    :func:`sum_phi_relations_measured_bound`. For a complete distinction
    set this is a certified ceiling on the Φ of the structure.

    Parameters
    ----------
    distinctions : iterable
        The distinctions of a Φ-structure. Must be a resolved set — each
        mechanism at most once.

    Returns
    -------
    UpperBound
        Certified; same assumptions as
        :func:`sum_phi_relations_measured_bound`.
    """
    distinctions = tuple(distinctions)
    sum_phi_d = math.fsum(float(d.phi) for d in distinctions)
    relations = sum_phi_relations_measured_bound(distinctions)
    return UpperBound(
        value=sum_phi_d + relations.value,
        certified=True,
        assumptions=relations.assumptions,
        citation=f"exact Σφ_d + {relations.citation}",
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest test/integration/test_bounds.py -q > /tmp/t1-pass.log 2>&1; tail -3 /tmp/t1-pass.log
```

Expected: all pass (142 existing + the new class). Read the summary line, not the exit code.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/bounds.py test/integration/test_bounds.py
git commit -m "Add measured state-keyed certificates for Σφ_r and Φ

sum_phi_relations_measured_bound evaluates the Eq 14 linear-program
maximum on the measured per-atom profile of a distinction set, with the
exact Eq 9 self-relation sum; big_phi_measured_bound adds the exact Σφ_d.
Certified with no binary-units or measure assumption — the theorem is
combinatorial over the relation formula — so unlike the size-based
bounds there is no config domain guard. Verified in the S(o) certificate
experiments (801 records, 0 violations).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 2: Guard-independence tests + slow-lane property

**Files:**
- Test: `test/integration/test_bounds.py` (extend `TestMeasuredBounds`; new slow class after it)

**Interfaces:**
- Consumes: `bounds.sum_phi_relations_measured_bound(distinctions)`, `bounds.big_phi_measured_bound(distinctions)` from Task 1; `_ces`, `DOMAIN_CONFIGS`, `TOL`, `small_system`, `PROPERTY_SETTINGS` conventions already in the file.

- [ ] **Step 1: Write the failing k-ary test**

Add to `TestMeasuredBounds`:

```python
    def test_kary_system_no_guard(self):
        # A ternary 2-unit substrate: the theorem has no binary-units
        # assumption, so the measured bound must compute and hold where
        # the size-based bounds refuse the substrate outright.
        rng = np.random.default_rng(2026)
        marginals = []
        for _ in range(2):
            f = rng.uniform(size=(3, 3, 3))
            marginals.append(f / f.sum(axis=-1, keepdims=True))
        substrate = Substrate(
            marginals=marginals, state_space=("LOW", "MID", "HIGH")
        )
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            system = System(substrate, (0, 0), substrate.node_indices)
            distinctions, sum_phi_r = self._concrete(system)
            measured = bounds.sum_phi_relations_measured_bound(distinctions)
        assert distinctions, "fixture must yield distinctions; adjust seed/state"
        assert sum_phi_r <= float(measured) + TOL
        with pytest.raises(ValueError, match="binary"):
            bounds.report(substrate=substrate)
```

- [ ] **Step 2: Run it; fix the fixture if it yields no distinctions**

```bash
uv run pytest test/integration/test_bounds.py::TestMeasuredBounds::test_kary_system_no_guard -q > /tmp/t2-kary.log 2>&1; tail -5 /tmp/t2-kary.log
```

Expected: PASS. If the `assert distinctions` guard trips, try state `(1, 1)` or seed `2027` until the CES is nonempty (a uniform-random ternary TPM at n=2 almost always yields distinctions), and keep the working combination.

- [ ] **Step 3: Write the measure-guard contrast test**

Add to `TestMeasuredBounds`:

```python
    def test_no_measure_domain_guard(self):
        # The distinctions are computed under a pinned in-domain config;
        # the *bound call* then happens under an out-of-domain mechanism
        # measure. The measured bound reads no config, so it returns the
        # same value; the size-based bound refuses.
        system = EXAMPLES["system"]["basic"]()
        with config.override(**DOMAIN_CONFIGS["iit4_2023"]):
            distinctions, _ = self._concrete(system)
            in_domain = bounds.sum_phi_relations_measured_bound(distinctions)
        with config.override(mechanism_phi_measure="ID"):
            out_of_domain = bounds.sum_phi_relations_measured_bound(distinctions)
            assert float(out_of_domain) == float(in_domain)
            with pytest.raises(ValueError, match="not confirmed"):
                bounds.sum_phi_relations_upper_bound(3, bound="GENERAL")
```

- [ ] **Step 4: Run the class**

```bash
uv run pytest test/integration/test_bounds.py::TestMeasuredBounds -q > /tmp/t2-class.log 2>&1; tail -3 /tmp/t2-class.log
```

Expected: all pass. (`config.override(mechanism_phi_measure="ID")` is a valid registered measure; if the override itself is rejected, use `"L1"` — also registered — and keep the `pytest.raises` expectation unchanged.)

- [ ] **Step 5: Write the slow-lane Hypothesis property**

Add after `TestMeasuredBounds`:

```python
@pytest.mark.slow
class TestMeasuredBoundPropertySlow:
    """The verify-script soundness check as a permanent property test."""

    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def test_random_systems(self, data):
        with config.override(
            **DOMAIN_CONFIGS["iit4_2023"], validate_system_states=False
        ):
            system = data.draw(small_system(min_size=2, max_size=3))
            with config.override(relation_computation="CONCRETE"):
                ces = _ces(system)
            distinctions = list(ces.distinctions)
            sum_phi_r = math.fsum(float(r.phi) for r in ces.relations)
            measured = bounds.sum_phi_relations_measured_bound(distinctions)
            exact = float(AnalyticalRelations(ces.distinctions).sum_phi())
        assert sum_phi_r <= float(measured) + TOL
        assert exact <= float(measured) + TOL
```

- [ ] **Step 6: Run the slow test**

```bash
uv run pytest test/integration/test_bounds.py::TestMeasuredBoundPropertySlow -m slow --slow -q > /tmp/t2-slow.log 2>&1; tail -3 /tmp/t2-slow.log
```

Expected: 1 passed. (The root conftest errors loudly if `--slow` is missing.)

- [ ] **Step 7: Commit**

```bash
git add test/integration/test_bounds.py
git commit -m "Test guard independence and random-system soundness of measured bounds

The measured certificates compute and hold on a k-ary substrate and
under an out-of-domain mechanism measure — both refused by the
size-based bounds — and a slow-lane Hypothesis property checks
soundness against enumerated relations on random small systems.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 3: Docs, changelog, and full-suite gate

**Files:**
- Create: `changelog.d/measured-bound.feature.md`
- Modify: `docs/theory/computational-complexity.md` (the "published φ upper bounds" paragraph, currently lines 164–168)

**Interfaces:**
- Consumes: the function names from Task 1 (referenced in prose).

- [ ] **Step 1: Write the changelog fragment**

```bash
cat > changelog.d/measured-bound.feature.md <<'EOF'
Added `sum_phi_relations_measured_bound()` and `big_phi_measured_bound()` to `pyphi.formalism.iit4.bounds`: certified upper bounds on Σφ_r and Φ evaluated on the measured per-atom profile of a distinction set, in O(|D|·n) with no relation enumeration. Unlike the size-based bounds they carry no binary-units or measure assumption, and they are typically orders of magnitude tighter than the worst-case ceilings.
EOF
```

- [ ] **Step 2: Extend the complexity-page bounds paragraph**

In `docs/theory/computational-complexity.md`, the paragraph ending
"…and the sum of relation φ grows hyper-exponentially, $O(n^2\,2^{\,2^n})$ (Zaeemzadeh & Tononi, 2024)." — append to the same paragraph:

```
Alongside these worst-case, size-based ceilings, the same module provides
*measured* certificates: given a computed distinction set,
`sum_phi_relations_measured_bound` evaluates the paper's linear-program
maximum on the measured per-atom profile — a certified bound on the sum of
relation φ that is typically orders of magnitude tighter than the
worst-case ceiling — and `big_phi_measured_bound` adds the exact
distinction-φ sum to give a certified ceiling on Φ.
```

Match the surrounding line-wrap width and MyST style when inserting.

- [ ] **Step 3: Verify the docs build**

```bash
just docs > /tmp/t3-docs.log 2>&1; tail -3 /tmp/t3-docs.log
```

Expected: build succeeds (the pre-existing `whats-new-in-2.0.md` orphan warning is a known concurrent-session issue; do not touch that file).

- [ ] **Step 4: MCP surface check (verification only)**

```bash
grep -rn "bounds" pyphi/mcp/ | grep -v Binary | head
```

Expected: no bounds surface exists in `pyphi/mcp/` (verified during planning), so no MCP change. If this grep unexpectedly finds one, surface it for discussion instead of editing.

- [ ] **Step 5: Full-suite gate in the worktree**

```bash
uv run pytest -q > /tmp/t3-full.log 2>&1; tail -3 /tmp/t3-full.log
```

Expected: summary line shows all passed (3860+ passed at branch base), no new failures. Pathless invocation is mandatory — it is the only run that collects the `pyphi/` doctests and the precision lint.

- [ ] **Step 6: Commit**

```bash
git add changelog.d/measured-bound.feature.md docs/theory/computational-complexity.md
git commit -m "Document the measured certificates in the complexity page and changelog

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Completion

After all tasks: use superpowers:finishing-a-development-branch (verify full suite + slow lane in the worktree, present the standard options; on merge, run the full suite in the main tree, update the ROADMAP Wave 7 anytime-bracket remainder + triage item M3 rows in the same flow, then remove the worktree and branch).
