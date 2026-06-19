# Intrinsic-unit criteria and bounded grain search — design

**Project:** Macro framework, sub-project 2 (of 3). The *decision* layer of
the Marshall et al. 2024 intrinsic-units formalism: the intrinsic-unit
criteria (Eqs 15-16), the admissible-system set function f(U^J, W^J) and
its recursion, complexes across grains (Eq 19), and a bounded search
driver that answers "which units and which grain are intrinsic for this
substrate in this state." Builds on sub-project 1's evaluation machinery
(`pyphi/macro/`: `MacroUnit`, `macro_tpms`, `MacroSystem`) without
modifying it.

Sub-project 3 (separate spec): reference goldens frozen from the
authors' ten committed result sets.

**Sources of truth:**

- The paper, Sec 2.2.2 (Eqs 15-19, pp. 6-7) and the recursion paragraph
  (p. 9): `papers/2024__marshall-et-al__intrinsic-units.pdf`.
- Fig 2 ("Out of nothing, nothing comes") and its three scenarios, which
  are exactly the `sfn`/`sfnn`/`sfs` result sets committed at full
  precision in the authors' repo (github.com/CSC-UW/Marshall_et_al_2024,
  results/); the `min` and `bu` result sets anchor the cross-grain and
  micro-exemption behavior.
- The authors did **not** implement f, the recursion, or any search;
  their code evaluates hand-chosen candidates only. The criteria
  *checks* have published anchors; the recursion and driver semantics do
  not (see Interpretation risk).

## Scope

In scope:

- `pyphi/macro/criteria.py`: unit-level criteria (Eqs 15-16) with
  verdict objects carrying witnesses; the competing-system set
  f(U^J, W^J) materialized within a unit's footprint.
- `pyphi/macro/search.py`: a frozen `SearchBounds` value; bounded
  mapping enumeration (`candidate_mappings`); the intrinsic-unit
  recursion (`intrinsic_units`); the bounded valid-system set
  ℙ(u) (`valid_systems`); the Eq 19 driver (`complexes`) returning
  winners plus the full evaluation record.
- Opt-in background-apportionment enumeration (off by default):
  under `apportionment="ENUMERATE"`, every assignment of background
  micro units to candidate units (each background unit unapportioned or
  apportioned to exactly one unit, at most `max_background` units
  apportioned in total) is searched.

Out of scope: sub-project 3 goldens; parallelization or resumability of
the search (the φ_s pipeline's own parallel config still applies per
evaluation); non-binary alphabets; any change to the SP1 construction.

## Formalism semantics

These pins are load-bearing; each is forced by an argument given below.

### Eq 15 — unit integration

φ_s(v^J) is the φ_s of the unit's **constituent system**: the
`MacroSystem` over the full universe whose units are the elements of
V^J (a micro index i becomes `micro_unit(i)`; a meso constituent
participates with its full definition — mapping, grain, apportionment).
Consequences:

- A unit's validity is independent of its **own** mapping g'_J and
  update grain tau'_J, but depends on its **constituents'** full
  definitions.
- Validity is therefore a property of the pair (V^J, W^J), checked once
  per decomposition; mapped/grained variants of the same decomposition
  share the verdict.
- The unit's W^J beyond its constituents' apportionments does not
  affect φ_s(v^J) (unapportioned background is fully noised either
  way); W^J matters only for which competitors f admits.

### Eq 16 — maximal irreducibility within, and f(U^J, W^J)

**Pin (confirmed by Marshall, see below):** the candidate is valid iff

    φ_s(v^J) > φ_s(v')  for every v' ≠ v^J in f(U^J, W^J).

f(U^J, W^J) is the set of valid systems whose **total micro
constituents are a (not necessarily strict) subset of U^J**, with
background apportionments that are non-overlapping subsets of W^J. The
only system removed from the comparison is v^J itself — the candidate.
Concretely, the comparison includes:

- strict sub-systems (Fig 2's singleton comparisons); and
- alternative same-union decompositions, where the constituents *equal*
  U^J but are grouped differently into several smaller meso/micro units
  (the Fig 3E "in one shot" vs "built on meso units" competition, which
  the paper states is *a consequence of* this requirement).

It excludes v^J and, with it, any single unit that spans the entire
footprint as one macro unit (the candidate's own "wrapping"): a unit
covering all of U^J is the candidate's own grain, not a finer
organization of it.

**How the implementation realizes this.** `f` is built from the pool of
units already validated at strictly finer footprints, assembled into
systems whose members have pairwise-disjoint footprints. Requiring each
*member* to be a proper subset of U^J yields exactly the set above: a
system whose members' union equals U^J then necessarily has two or more
members (a same-union meso reorganization, kept), and a single unit
spanning U^J (the wrapping) can never be a member (dropped). So the
per-member-proper construction and Marshall's "total constituents,
improper subset, exclude v^J" describe the same set.

**Resolution of the prior interpretation risk.** The paper defines f as
"all valid systems V' (ones that satisfy Eqs. 16 and 18) whose micro
constituents are a subset of U^J." Read with "subset" applied to the
competitor system's total constituents *including the improper subset
and without excluding v^J*, the candidate's own one-unit macro wrapping
is a competitor — and since macroing typically *raises* φ_s (the
formalism's central phenomenon), the candidate would have to beat its
own wrapping. Forcing case, **reproduced** (confirming experiment and
results table in
`docs/superpowers/notes/2026-06-18-marshall-f-clarification.md`): in the
authors' `min` example the candidate (A,B) has constituent-system φ_s =
0.005106576483955726 while its own wrapping {alpha} has φ_s =
0.7883339770634886, so admitting the wrapping flips the verdict from
VALID to NOT_MAXIMAL — invalidating the very unit the example is built
to validate. The question was put to William Marshall:

> in f(U^J, W^J), is the subset condition on the competitor system's
> total constituents or on each of its units' constituents — and is it
> proper?

His answer: the condition is on the total constituents and is **not
strict**; the fix for the circularity is to **exclude v^J itself**, not
to require a strict subset — "we do want to consider systems whose
constituents are equal to U^J, but perhaps organized differently at a
meso spatial scale." Eq 16 should read φ_s(v^J) > φ_s(v') ∀ v' ≠ v^J ∈
f(…). The confirming experiment shows the shipped per-member-proper
construction reproduces this reading verdict-for-verdict on all four
published result sets (`min`, `sfn`, `sfnn`, `sfs`), every φ_s matching
the committed value, and at depth 2 includes same-U^J meso
reorganizations while excluding single-unit wrappings.

**A once-residual sub-question, now settled: full-span units are
excluded regardless of internal scaffolding.** One case is not spelled
out word-for-word in Marshall's reply: a competitor that is a *single*
macro unit spanning all of U^J but built from a *different* meso
organization than the candidate. It is excluded. The reason is intrinsic
to Eq 16's "maximally irreducible *within*": that comparison ranges over
*finer* organizations of the same parts — does the candidate grain beat
keeping the parts more divided? A single unit spanning all of U^J is not
finer; it is the *same* coarseness as the candidate (one unit over the
whole footprint), and its internal scaffolding (in one shot vs. through
meso pieces) changes only how its own TPM is built, not its grain. Eq
15's mapping-independence says the same thing from the other side: the
candidate's validity does not depend on its own internal organization,
so a full-span rival differing *only* in internal organization is not a
distinct competitor. Admitting any full-span unit would also re-create
the forcing case (it carries the inflated macro φ_s). So every single
unit spanning the whole footprint is the candidate's own grain and
stays out of f; only genuinely finer decompositions — more, smaller
units — compete. This never arises at the default `max_depth=1` and is
pinned by test.

### Micro units are axiomatically valid

Eqs 15-16 gate **macroing** (any unit with more than one constituent or
micro grain above 1); micro units enter the recursion as valid ground
("each micro unit U_i is a potential unit", p. 9). Forcing case: in the
authors' `bu` example every 1- and 2-unit subsystem has φ_s = 0 and
only φ_s(ABC) = 0.8300749985576875 — if micro units had to satisfy
Eq 15, the example's own micro complex would contain no valid units and
ℙ(u) would be empty. The paper's footnote (earlier IIT versions imposed
the requirement "only on the system") supports the exemption.

A single-constituent candidate at higher grain (macroing over updates,
Fig 3D) **is** gated: its footprint admits no proper-subset
competitors, so it is valid iff φ_s of its constituent's system is
positive — one cannot build persistence-over-updates out of a unit with
no cause-effect power.

### Ties

All inequalities (Eqs 16 and 19) are strict, compared at
`config.numerics.precision` via the standard PyPhi float comparison. A
candidate that ties its best competitor is invalid with
`reason = TIED`; tied top systems in Eq 19 mean neither is a complex.
Verdicts and driver results carry the tying parties explicitly.

### Eq 19 — complexes across grains

ℙ(u) is the bounded set of valid systems: every nonempty set S of
intrinsic units (micro units included) with pairwise-disjoint
footprints (Eq 18), evaluated over the full universe with everything
else as background. Systems need not cover the universe. S is a complex
iff φ_s(s) strictly exceeds φ_s(s') for every other member whose micro
constituents overlap S's. Multiple mutually-disjoint complexes can
coexist. The system-level φ_s **does** depend on each unit's mapping
and grain, so the driver evaluates every mapped/grained variant
admitted by the bounds.

## The recursion (`intrinsic_units`)

Level 0: the micro units. At each level up to `bounds.max_depth`:

1. For every candidate footprint U_c (subsets of the universe with
   2 <= |U_c|, and singletons when max_update_grain > 1, bounded by
   `max_constituents`), enumerate decompositions V: sets of
   already-derived valid units with pairwise-disjoint footprints whose
   union is U_c, all sharing one constituent micro grain (the SP1
   constraint). |V| = 1 is allowed when the single constituent's
   footprint equals U_c (pure grain-raising of a finer unit).
2. For each (V, W) pair (W per the apportionment policy), check
   validity once: Eq 15 via the constituent system's φ_s; Eq 16 against
   f(U_c, W) materialized from the current valid-unit pool restricted
   to proper-subset footprints. Record a `UnitVerdict` either way.
3. For each valid (V, W), emit the mapped/grained unit variants —
   `MacroUnit(V, tau', g', W)` for tau' in 1..`max_update_grain` and g'
   from `candidate_mappings` — into the next level's pool.

Termination: `max_depth` bounds the levels; footprints are finite. All
φ_s evaluations go through one per-run memo keyed on the hashable
`MacroSystem`, shared across criteria checks and the Eq 19 sweep (the
same systems recur constantly — f members at one footprint are ℙ(u)
members later).

f's recursive definition ("valid systems... that satisfy Eq 16") is
realized by construction: competitors are assembled only from units the
recursion has already validated at finer footprints, which is exactly
the paper's bottom-up derivation.

## Components

### `pyphi/macro/criteria.py`

Pure criteria logic; depends only on SP1 machinery, never on
`search.py` (the dependency runs the other way).

```python
def unit_integration(substrate, constituents, micro_history) -> PyPhiFloat
    """phi_s(v^J): the constituent system's integrated information (Eq 15)."""

class Reason(Enum): VALID; NOT_INTEGRATED; NOT_MAXIMAL; TIED

@dataclass(frozen=True)
class UnitVerdict:
    valid: bool
    reason: Reason
    phi: float                     # phi_s(v^J)
    witness: MacroSystem | None    # the competitor that beat or tied it
    witness_phi: float | None
    num_competitors: int

def judge_candidate(phi, competitors) -> UnitVerdict
    """Eqs 15-16 given the candidate's phi and (system, phi) pairs for f."""
```

### `pyphi/macro/search.py`

```python
@dataclass(frozen=True)
class SearchBounds:
    max_constituents: int = 4      # |U^J| cap per candidate unit
    max_update_grain: int = 1
    max_depth: int = 1             # macroing levels above micro
    mappings: str = "FAMILIES"     # or "EXHAUSTIVE"
    exhaustive_cap: int = 8        # max sequence-states for EXHAUSTIVE
    apportionment: str = "NONE"    # or "ENUMERATE"
    max_background: int = 0        # |W| cap when enumerating

def candidate_mappings(num_constituents, update_grain, bounds) -> tuple[tuple[int, ...], ...]
    """Deduplicated candidate truth tables for a unit shape.

    FAMILIES: every non-degenerate coarse_grain on-count set (update
    grain 1 only, by the family's definition) plus every nonempty
    blackbox output subset (any grain). EXHAUSTIVE: every surjective
    table when the sequence-state count is within exhaustive_cap
    (8 states = 254 tables); ValueError above it.
    """

def competing_systems(substrate, unit, micro_history, bounds) -> tuple[MacroSystem, ...]
    """f(U^J, W^J) materialized within the unit's footprint (Eq 16)."""

def is_intrinsic_unit(substrate, unit, micro_history,
                      bounds=SearchBounds()) -> UnitVerdict
    """Eqs 15-16 for one candidate; micro units return VALID trivially.

    Accepts a full MacroUnit (its own mapping and grain are ignored for
    the verdict, by Eq 15's mapping-independence) and runs the
    recursion restricted to the unit's footprint to build f.
    """

def intrinsic_units(substrate, micro_history, bounds) -> IntrinsicUnitsResult
    """The recursion's fixed point: the valid-unit pool plus all verdicts."""

def valid_systems(substrate, micro_history, bounds) -> tuple[MacroSystem, ...]
    """The bounded P(u): every Eq-18-compatible system of intrinsic units."""

@dataclass(frozen=True)
class EvaluationRecord:
    system: MacroSystem
    phi: float

@dataclass(frozen=True)
class ComplexesResult:
    complexes: tuple[MacroSystem, ...]     # Eq 19 winners (disjoint)
    records: tuple[EvaluationRecord, ...]  # every evaluated system + phi_s
    ties: tuple[tuple[MacroSystem, MacroSystem], ...]

def complexes(substrate, micro_history, bounds=SearchBounds()) -> ComplexesResult
    """Eq 19 over the bounded candidate space — the one-call driver."""
```

`IntrinsicUnitsResult` is frozen and carries the unit pool grouped by
footprint plus the per-candidate `UnitVerdict`s, so the derivation is
inspectable.

Defaults are deliberately conservative: `SearchBounds()` searches one
macroing level, grain 1, family mappings, no apportionment — which
covers the cg, min, bu, sfn/sfnn/sfs cases. Example 2's grain-2
blackboxing needs `max_update_grain=2`.

### Memoization

One evaluation cache per driver run (dict keyed on `MacroSystem`,
which SP1 made hashable on (micro substrate, units, history,
partition)), threaded through criteria and search internals. No global
cache; repeated `complexes()` calls re-evaluate.

## Error handling

`ValueError` with a specific message on: `mappings="EXHAUSTIVE"` with a
unit shape exceeding `exhaustive_cap`; unknown `mappings` or
`apportionment` policy strings; `apportionment="ENUMERATE"` with
`max_background == 0`; bounds with `max_constituents < 1`,
`max_update_grain < 1`, or `max_depth < 0`; a `micro_history` shorter
than the maximum micro grain the bounds admit
(`max_update_grain ** max_depth`, since grains compose down the
hierarchy). An empty ℙ(u) (or one
with no complex due to ties) returns an empty-`complexes` result, not
an error.

## Testing

1. **Fig 2 verdict anchors (authors' committed precision,
   sfn/sfnn/sfs result sets):**
   - sfn (w_v = 0): candidate (A,C) -> `NOT_INTEGRATED`
     (phi = 0.0); singletons phi_s = 0.02363345634846179.
   - sfnn (w_v = 0.01): candidate (A,C) -> `NOT_MAXIMAL` with
     phi = 0.004863714555961354 and a singleton witness at
     0.023640988356789627.
   - sfs (w_v = 0.25): candidate (A,C) -> `VALID` with
     phi = 0.16758555077361778 (singletons 0.02346371771182276);
     horizontal pairs (A,B) also `VALID` at 0.6728123807299448.
   The dancing-couples TPM is built from the authors' published rule
   (base 0.05, self 0.05, horizontal 0.6, vertical w_v).
2. **min end-to-end driver:** unit (A,B) `VALID`
   (0.005106576483955726 > 0; singleton competitors at 0.0); the
   both-on mapping's one-unit system reproduces the committed
   phi_s = 0.7883339770634886; `complexes()` with
   `mappings="EXHAUSTIVE"` (14 tables) returns a macro complex, with
   the argmax mapping and its phi_s recorded as this project's golden
   at implementation time.
3. **bu micro-exemption:** the all-micro system is admissible and
   evaluates to the committed 0.8300749985576875 despite every unit
   and proper subsystem having phi_s = 0; candidate macro units over
   {A,B,C} pass Eq 16 against all-zero competitors; the full driver
   verdict (micro vs wrapped macro) recorded as a golden at
   implementation time.
4. **Invariants:** verdicts identical across mapped/grained variants of
   one decomposition (Eq 15 mapping-independence); memo returns
   identical phi for identical systems; every system the driver
   evaluates satisfies Eq 18; `is_intrinsic_unit` on a micro unit is
   trivially `VALID`; every `EvaluationRecord.phi` matches an
   independent `system.sia().phi` recomputation on a sample.
5. **Tie path:** a hand-built exactly-symmetric fixture (two unit
   candidates whose constituent systems are permutation-identical)
   exercising `TIED` verdicts and the no-complex-on-tie driver outcome.
6. **Cost guard:** the full default-bounds driver on the cg substrate
   under the slow marker; assert it terminates and its record contains
   the SP1-anchored micro panel values.

## Files

- `pyphi/macro/criteria.py`, `pyphi/macro/search.py` — new
- `pyphi/macro/__init__.py` — export the public surface
- `test/test_macro_criteria.py`, `test/test_macro_search.py` — new
- `changelog.d/intrinsic-units-search.feature.md` — new
- `ROADMAP.md` — macro SP2 marked landed in the item-10 entry

## Notes carried from brainstorming

- Approach: layered pure functions + frozen result values (two new
  modules), over a stateful search-session object and over bare
  kwargs-configured functions — matches SP1 and house style; the
  evaluation record provides the transparency a session object would
  have.
- Driver scope: full one-call `complexes()` driver (user choice), not
  checkers-only.
- Mapping policy: families by default, capped exhaustive opt-in (user
  choice).
- Apportionment: off by default, opt-in enumeration (user choice); the
  Eq 29 evaluation path exists in SP1 but gets its first published
  anchor only in SP3.
- The f(U^J, W^J) subset semantics were **resolved by William
  Marshall** (see the Eq 16 section): the subset is on total
  constituents and not strict, and v^J is excluded. The follow-on
  question — whether a single macro unit spanning all of U^J but built
  from a different meso organization competes — is also settled: it does
  not, because a full-span unit is the candidate's own grain, not a
  finer "within" competitor (see the Eq 16 section). Nothing on f
  remains queued.
- Question still queued for the authors: SP1's finding that the
  committed Example 1 macro TPM contains a hand-entry error (0.9212 vs
  the construction's 0.9216) and a rounded entry (0.006833 for
  0.0615/9).
