# 2.0.0 release pass: scientific-fidelity additions

**Date:** 2026-09-11
**Status:** approved in discussion; pending review of this document

## Purpose

The engineering side of 2.0 is closed: every ROADMAP dashboard item has
landed, the pre-release audit is discharged, and the five verification
gates in `RELEASING.md` are green. A final read of the primary papers
against the code found that the remaining gaps are on the scientific
side — quantities a paper names that the library does not expose by
that name, and published worked examples the acceptance suite does not
pin. This pass closes those gaps before the tag.

Eight items were identified and all eight ship in 2.0.0. Three add
public surface (sections 2–4), four add tests only (section 5), one is
a terminology sweep (section 6).

## 1. What the cross-check found already exists

Two items are smaller than the ROADMAP describes.

- The IIT 4.0 system irreducibility analysis
  (`pyphi.formalism.iit4.SystemIrreducibilityAnalysis`) already carries
  the per-direction intrinsic differentiation
  (`intrinsic_differentiation: dict[Direction, float]`), the
  per-direction specification value (`system_state[d].intrinsic_information`),
  the combined system intrinsic information `ii(s)`
  (`intrinsic_information` property) and the integrated fraction
  φₛ / ii(s) (`integrated_fraction` property). They are computed under
  every IIT 4.0 preset (the differentiation is evaluated in
  `evaluate_partition` regardless of whether the requirement applies),
  serialized, and shown on the display card. ROADMAP N22 is therefore
  reduced to naming, an `explain()` finding, and documentation.
- The S1 terminal tie branch (a clique of overlapping candidates tied
  in both φₛ and Φ fails exclusion; its units remain available to lower
  tiers) is implemented in `pyphi/condensation.py` and unit-tested in
  `test/test_condensation.py`. What is missing is one end-to-end test
  through `Substrate.complexes()` on a substrate whose symmetry drives
  that branch.

## 2. Intrinsic specification (naming)

Mayner, Marshall & Tononi (2026) rename the quantity that Albantakis et
al. (2023) call intrinsic information — the product of selectivity and
informativeness for a specified state (2026 Eqs. 7 and 9) — to
*intrinsic specification*, reserving *intrinsic information* for the
minimum of specification and differentiation (2026 Eq. 13). The code
uses the 2023 name at the state level and the 2026 name at the system
level, so the same word names two different quantities one level apart.

Changes:

- `StateSpecification.intrinsic_specification`: a property returning
  the same value as `intrinsic_information`. `intrinsic_information`
  stays, documented as the name Albantakis et al. (2023) use for this
  quantity (decision D1: alias, not rename).
- `SystemIrreducibilityAnalysis.intrinsic_specification`: a property
  returning `dict[Direction, float]`, one value per direction, parallel
  in shape to the existing `intrinsic_differentiation`. `None` entries
  where a direction's specification is unavailable, matching how
  `intrinsic_differentiation` behaves on null analyses.
- The display card's per-direction row currently labeled "Intrinsic
  information" becomes formalism-aware, using the pattern that already
  selects the φₛ / Φ glyph: "Intrinsic specification" when the
  configuration's formalism is IIT 4.0 (2026), "Intrinsic information"
  under IIT 4.0 (2023). The label reads from the result's own
  `ConfigSnapshot`, as the φₛ label does.
- `to_pandas()` on the SIA gains `cause_intrinsic_specification`,
  `effect_intrinsic_specification`, `cause_intrinsic_differentiation`,
  and `effect_intrinsic_differentiation` columns.
- Serialization is unchanged: the values already round-trip through the
  existing state-specification and `intrinsic_differentiation` fields.

Not in scope: renaming `StateSpecification.intrinsic_information`
(D1), or exposing the two quantities at the mechanism level (the 2026
paper defines them for systems only).

## 3. `explain()` reports when the requirement binds

Under IIT 4.0 (2026), φₛ = min{φ_c, φ_e, ii(s)}. When the reported φₛ
equals ii(s) rather than either direction's integrated information, the
explanation carries a finding that says so, naming the direction and
the term (specification or differentiation) whose value is the minimum.
Equality is decided with `pyphi.numerics` tolerance, never raw float
comparison. Under presets where the requirement is not applied, the
finding never fires. The finding is a new `Finding` in
`pyphi/models/explanation.py` alongside the existing partition and
short-circuit findings, rendered by the existing display path.

## 4. `Substrate.inactivate`

```python
def inactivate(self, fixed: Mapping[int | str, int]) -> Substrate
```

Returns a new substrate in which each unit in `fixed` (given by index
or label) is frozen in the given state: the unit's state is conditioned
into every other unit's transition factor
(`FactoredTPM.condition`), so it has no counterfactual states and
cannot be intervened upon. Node labels are preserved. The construction
is exactly the one `examples.iit4_2023_fig7_inactivated_substrate`
performs by hand today; that example is rewritten to call the method.

The docstring states the distinction the 4.0 paper draws (Fig 7B vs
7C): an *inactive* unit is in its OFF state and still contributes to
the Φ-structure; an *inactivated* unit is removed from the system's
cause–effect power entirely, and the complex that contained it shrinks.
It also states that inactivation differs from holding a unit as a
background condition of a candidate system.

Decision D2: the method is named `inactivate`, the paper's term. No
`System` or `analyze()` convenience is added.

Validation: an unknown label or index raises the same error the
existing state-coercion helpers raise; a state outside the unit's
alphabet raises `ValueError`. Cannot be applied to a unit twice in one
call (a mapping cannot express it), and applying it to a unit already
frozen by an earlier call is a no-op on that unit.

## 5. Paper-reproduction pins

All new tests follow the conventions of
`test/integration/test_paper_reproduction.py`: the formalism is pinned
with the full preset from `pyphi.conf.presets` (plus `replace(...)` for
paper-specific overrides), every pinned number is quoted from the paper
with its figure or equation number, and the module docstring's
"Currently covered" list is extended. Tests slower than roughly ten
seconds are marked slow. Each pin is verified during implementation to
fail when its expected value is perturbed (a pin that cannot move is
not a gate); the verification is recorded in the commit message.

Where a fixture is reconstructed from a paper's prose and a published
value cannot be matched, the test pins what does match and documents
the deviation in its docstring, as the Fig 6D and Fig 7B tests already
do. No value is forced.

### 5.1 Mayner, Marshall & Tononi (2026)

Preset `iit4_2026`.

- **Fig 2C, monad.** A single imperfect-COPY unit with stay probability
  p. φₛ(p = 0.744) = 0.427 to three decimals, and φₛ is lower at p =
  0.70 and p = 0.80 (the peak is interior). Per Eq. 27 the two terms
  are p·log₂(2p) and −log₂ p; both are asserted through the new
  `intrinsic_specification` and `intrinsic_differentiation` accessors.
- **Fig 3D–G, the 6-unit lattice under a temperature sweep.** The
  substrate is `examples.iit4_2023_fig6d_substrate` with the temperature
  parameter varied. Two bracketing assertions: the full 6-unit system is
  a complex at K = 0.85 and the maximal complexes are 2-unit at
  K = 0.70 (crossover K ≈ 0.775, Fig 3F–G); the requirement binds
  (φₛ = ii(s), and the binding term is differentiation) at K = 3.0 and
  does not at K = 2.7 (crossover K ≈ 2.839). The complexes assertions
  are slow-lane.
- **Fig 4C, intrinsic units.** Using the existing
  `examples.differentiation_micro_tpm(p, epsilon)` /
  `differentiation_macro_tpm` fixtures with ε = 0.01: the macro monad
  α has higher φₛ than the micro pair {a, b} at p = 0.10 and lower at
  p = 0.09 (crossover p ≈ 0.096), and {a, b} satisfies the maximally-
  irreducible-within criterion at both.

### 5.2 Marshall et al. (2023), System Integrated Information

Preset `iit4_2023`, with `background_conditioning` set to
`CONDITION_CURRENT_STATE` (the paper conditions background units on
their current state, §2.1). If a panel reproduces only under causal
marginalization, the test pins that and says so (decision D3).

Substrates are reconstructed from the paper's prose and registered as
`examples.marshall_2023_fig{1,2,3}_substrate` (+ `_system` where the
paper fixes a state), built with `substrate_generator.build_substrate`
and the sigmoid unit of Eq. 2 (`ising.probability` with
`temperature = 1/k`).

- **Fig 1, information.** Four deterministic all-to-all units with
  distinct functions in state ABcD: ii_c = ii_e = 4 (1B); with unit D
  noisy at 0.6/0.4, ii_c = ii_e = 1.95 (1C); with D's function copied
  from A, ii_c = 1.5 and ii_e = 3.0 (1D). The paper gives the unit
  functions only as a state-by-node table in the figure; the fixture is
  chosen to reproduce panel B exactly, and panels C and D are derived
  from it as the paper describes.
- **Fig 2, integration.** Four sigmoid units, k = 3, l = 1, all OFF,
  with the three weight patterns given in §3.2: φₛ = 0.3393, 0.0628,
  0.1477 and integrated fractions 48.1 %, 10.0 %, 21.2 % (through
  `integrated_fraction`). Panel A's two co-minimal partitions are
  asserted as an effective tie.
- **Fig 3, exclusion.** The 8-unit universe of §3.3 (k = 2 for A–F,
  k = 0.2 for G–H; weights as given) condenses through
  `Substrate.complexes()` into {F} (φₛ = 0.49), {A, B, C, D, E}
  (0.12), and {G, H} (0.06), and the nested sequence {A} ⊂ … ⊂
  {A, …, E} has monotone ii_c, ii_e with the φₛ jump at five units
  (Fig 3E–F). Slow-lane.

### 5.3 Albantakis et al. (2019), actual causation

Preset `iit3` as the existing AC tests use. Substrates that do not
already exist as examples are built inline in the test module unless
they are worth registering (the three-candidate vote is).

- **Fig 7.** OR, AND, XNOR, and the prevention gate from {AB = 11}:
  the account structure and α values listed in the figure (0.415 /
  1.0, 2.0 / 1.0 / 0.415 bits), including that AND's only actual cause
  is the joint {AB = 11} and XNOR's parts have no links.
- **Fig 8A/B.** The 4-input majority gate from ABCD = 1110: effects
  0.678 (singletons), 0.585 (pairs), 0.415 (ABC), cause {ABC} at
  1.678; with D = 0 as background, all effects 1.0 and cause 3.0.
- **Fig 9A/B.** `examples.disjunction_conjunction_substrate`:
  {A} → {D} at 0.263, {C} → {D} at 0.678, cause {C} at 0.678; with
  B = 0 as background, only {C} at 1.0 in both directions.
- **Fig 10.** The five-input rule: effects 0.70 / 0.46 / 0.30 / 0.30
  and the indeterminate cause {AB = 11} or {ACDE = 1000} at 1.0.
- **Fig 11.** The three-candidate, seven-voter election
  (multi-valued; the suite's first k-ary AC pin): effects 0.718 /
  0.581 / 0.404 / 0.190 by occurrence order, cause set of four-voter
  occurrences at 1.893, and α = 0 for the two votes for candidate 2.
- **Fig 12.** The noisy COPY (0.9): 0.848 both directions for
  {A = 1} ≺ {N = 1}; no links for {A = 1} ≺ {N = 0}.
- **Fig 13.** The dot/segment/line classifier from ABC = 001: the
  three effect links and seven cause links with the listed α values.
- **Figs 15 and 16.** The double bi-conditional (all 1.0 bits) and
  the irreducible-versus-reducible OR/AND pair (0.415 / 0.170; the
  transition irreducibility 𝒜 = 0.17 versus 0). Fig A1C's 0.03 / 0.83
  sub-transition values are pinned if the AC system-level analysis
  exposes them; otherwise recorded as out of scope in the docstring.

### 5.4 Barbosa et al. (2020), the intrinsic difference

Direct pins on `pyphi.measures.distribution.intrinsic_difference`, no
formalism involved.

- Fig 2A–C: the noiseless bit (1 ibit), the noiseless byte (8 ibits),
  and the one-good-wire byte (< 0.1 ibit).
- Fig 2E: the byte with seven wires at s = 0.78 gives 1 ibit to two
  decimals.
- Fig 3: r = 0.88 per wire, N = 1 / 8 / 16 wires → 0.72 / 2.41 / 1.77
  ibits.
- Fig 4C–E: the sigmoid neuron fan-out at t = 1, N = 1 / 8 / 16 → 0.72
  / 2.90 / 2.10 ibits (the channel distribution built per §"Intrinsic
  information among network elements").

### 5.5 S1 terminal tie branch, end to end

A substrate with two overlapping candidate systems that tie in both φₛ
and Φ by construction (a mirror symmetry), run through
`Substrate.complexes()`: neither tied candidate is a complex, the
exclusion records show them failing at the Composition level, and the
next unique candidate is accepted. Complements the unit-level tests in
`test/test_condensation.py`.

## 6. Terminology sweep

The project's terminology for the 2026 change is "intrinsic-information
requirement"; the 2023 formalism is "without the requirement". The
words "cap", "capped", "uncapped", "ii-cap" are replaced in every
public docstring under `pyphi/`, every page under `docs/`, and the MCP
reference content under `pyphi/mcp/content/`, wherever they refer to
Eq. 23. Unrelated uses of the word (purview-order caps, cost caps, the
numpy dimension ceiling) are untouched.

Identifiers: private names (`_apply_ii_cap`, `_cap_one`) stay. The
public Protocol attribute `CompositeMeasure.applies_ii_cap` in
`pyphi/measures/protocols.py` is renamed
`applies_intrinsic_information_requirement`, with every implementer and
call site updated (decision D4). No alias is kept: 2.0 is unreleased.

## 7. Documentation and release notes

- `docs/theory/intrinsic-information.md` documents the two accessors
  and the `explain()` finding, citing 2026 Eqs. 4–13 and 23.
- The macro/lesion material in `docs/tutorials/` and the Fig 7 example
  use `Substrate.inactivate`.
- One changelog fragment per surface item (`intrinsic-specification`,
  `explain-requirement-binding`, `substrate-inactivate`,
  `applies-ii-requirement-rename`), one per reproduction family
  (`paper-reproduction-2026`, `paper-reproduction-marshall-2023`,
  `paper-reproduction-ac-2019`, `paper-reproduction-barbosa-2020`),
  one for the terminology sweep. The already-built 2.0.0 changelog
  section is regenerated from the fragments rather than hand-edited.
- ROADMAP: N22 and N23 rows updated to landed; the "two paper-derived
  validation directions" paragraph updated; the N1 row lists the new
  papers.

## 8. Execution

Work happens on a `release-pass` branch in a worktree under
`.claude/worktrees/`, merged to `main` before the release walk resumes.

Order:

1. Sections 2, 3, 4, and the D4 rename — done sequentially, test-first,
   since they touch the frozen result types and the reproduction tests
   in 5.1 and 5.2 consume the new accessors.
2. Sections 5 and 6 in parallel as three agents with disjoint files:
   (i) 5.1 + 5.2 + 5.5 (examples and `test_paper_reproduction.py`
   additions for IIT); (ii) 5.3 + 5.4 (AC and ID pins in a new
   `test/integration/test_paper_reproduction_ac.py` and additions to
   the measures tests); (iii) section 6 + section 7 docs. Every agent
   asserts its working directory, runs no git-state commands, and
   proves each pin by perturbation.
3. Full suite without a path argument, the slow lane, and the docs
   build; then the changelog is rebuilt and the release walk in
   `RELEASING.md` resumes from gate 1.

## 9. Out of scope

- Matching-paper numerical pins (Φ = 404.44 on the restricted
  computation): a 13-unit scoped CES; 2.x.
- Barbosa et al. (2021) mechanism examples: computed under a pre-4.0
  state-selection convention the current formalism deliberately
  supersedes (Albantakis et al. 2023, S2 Text); not expected to
  reproduce and not attempted.
- k-ary support on the matching path, the perception-maximized
  projection, and the AC 4.0-style formalism: unchanged, per the
  ROADMAP deferred table.
