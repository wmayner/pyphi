# Grain discovery — exploration and design

**Question:** how could PyPhi *discover* the spatiotemporal grain at which a
substrate's integrated information is maximal, rather than evaluating candidate
grains one at a time? Is there structure in the space of grains that a search
can exploit, and where does any shortcut break down?

**Status:** exploration/design, not an implementation plan. Every empirical
claim below was run against the 2.0 pipeline (`pyphi/macro/`, IIT 4.0 2023
preset, default `config.numerics.precision = 13`); the experiments are small
enough to re-run in seconds and are described inline with enough code to
reproduce. Claims are tagged **[exists]** (already in PyPhi), **[established]**
(published result), or **[proposed]** (this document's own suggestion, with its
evidence and its certification status stated).

**Sources of truth:** Marshall, Findlay, Albantakis, Tononi 2024 (*Intrinsic
Units*, `papers/2024__marshall-et-al__intrinsic-units.pdf`); Marshall,
Albantakis, Tononi 2023 (*System Integrated Information*,
`papers/2023__marshall-et-al__system-integrated-information.pdf`); Albantakis
et al. 2023 (IIT 4.0, `papers/2023__albantakis-et-al__iit-4.0.pdf`);
Zaeemzadeh & Tononi 2024 (*Upper bounds*,
`papers/2024__zaeemzadeh-tononi__upper-bounds.pdf`); and the landed macro
framework (`pyphi/macro/`) plus the bounds module
(`pyphi/formalism/iit4/bounds.py`).

---

## 1. Where the problem actually stands

Three facts orient everything else.

**The theory puts grain inside the exclusion argmax, not beside it.** IIT 4.0's
exclusion postulate requires "units, updates, and states" to have a definite
grain, fixed by the same maximal-φ_s criterion that fixes the substrate's
border (IIT 4.0, "Implementing the postulates"; Box 2). A micro description and
a macro description of the same physical patch are overlapping candidate
systems; at most one exists. So "find the maximal grain" is not a separate
optimization to bolt on — it is more candidates in the one argmax that
`complexes()` already implements. **[established]**

**Both papers stop at the base grain.** Marshall et al. 2023, Sec. 2.4: "In
principle, the search should include not only subsets of U = u, but also
include systems of units with different spatiotemporal grains. For simplicity,
in this work we restrict consideration to the grain at which the universe is
defined." The 2024 intrinsic-units paper supplies the formalism for the grain
axis but proposes no search beyond "recursively enumerate; convergence is
guaranteed finite by disjointness" (Sec. 2.2.2). No monotonicity, no bounds, no
ordering. The gap this document addresses is real and acknowledged upstream.
**[established]**

**PyPhi already searches — exhaustively within declared bounds.** The macro
framework is landed: `MacroUnit` (Eq. 11: constituents, update grain, mapping,
background apportionment), the four-step macro TPM construction (Eqs. 26–40,
`pyphi/macro/tpm.py`), the intrinsic-unit criteria (Eqs. 15–16,
`pyphi/macro/criteria.py`), and the bounded recursion + Eq. 19 driver
(`pyphi/macro/search.py`: `intrinsic_units`, `valid_systems`, `complexes`,
parallelized, memoized on hashable `MacroSystem`). `SearchBounds` caps
footprint size, update grain, hierarchy depth, mapping family, and
apportionment; within those caps the enumeration is exhaustive. **[exists]**

So the design question is not "how to represent grains" or "what is the
criterion" — both are settled — but: *what lets the bounded exhaustive sweep
reach larger bounds, and what provably cannot help?*

### 1.1 Is the maximization over the wrong object?

Worth confronting the framing directly. A naive reading of the theory suggests
searching "the space of all grains" for max φ_s. That space is ill-posed as a
search domain: without the intrinsic-unit criteria, a macro unit can be
gerrymandered out of nearly-independent micro units ("building something out of
nearly nothing", 2024 paper Fig. 2), and earlier black-boxing work admitted
exactly this failure (their footnote 1). The 2024 recursion is not a
convenience — it is the definition of which grains are *candidates at all*:
units are certified bottom-up (integrated, Eq. 15; maximally irreducible within
their own footprint, Eq. 16), and only systems assembled from certified units
enter the Eq. 19 competition. The right object of the search is therefore the
recursion itself, and the right question is how to make the recursion and the
final sweep cheap. Everything below keeps the Marshall recursion as the outer
loop and changes only evaluation order, sharing, and skipping. **[proposed
framing, consistent with the papers]**

---

## 2. Anatomy of the space and the cost

### 2.1 The candidate space, axis by axis

For a binary micro universe of size *n*:

- **Footprints and decompositions.** Candidate units draw footprints from
  subsets of the universe; a candidate decomposition V splits a footprint into
  pool units with disjoint footprints. Set-partition growth (Bell-like), capped
  by `max_constituents`.
- **Mappings.** For a unit with |V| direct constituents at update grain τ′
  there are 2^(2^(τ′·|V|)−1) − 1 surjective mappings (Eq. 13 count; the
  landed code additionally canonicalizes away state-label complementation).
  The paper's own illustration (Fig. 3E): 32,727 mappings for 4 micro
  constituents at τ′ = 1; ≈ 5.78 × 10⁷⁸ at τ′ = 2. This axis is doubly
  exponential and is the reason `SearchBounds.mappings` defaults to
  `"FAMILIES"` (coarse-grainings + black-boxings) rather than `"EXHAUSTIVE"`.
- **Update grains.** τ′ per level, composing multiplicatively down the
  hierarchy (`max_micro_grain = max_update_grain ** max_depth`).
- **Apportionment.** Background units assigned to a unit (W^J) add a further
  subset-enumeration layer (`apportionment="ENUMERATE"`); default is none.
- **Assembly.** Valid units combine into systems under Eq. 18 disjointness
  (`_assemble_systems`), and Eq. 19 compares overlapping systems.

Two structural reliefs are already built in. First, **the criteria factor out
mappings and grains entirely**: Eqs. 15–16 are properties of the decomposition
(V, W) alone, so one verdict covers every mapped and grained variant
(`pyphi/macro/criteria.py` module docstring). **[exists]** Second, the paper's
own hierarchy argument: a macro unit built from certified meso constituents has
a mapping space of tens, not 10⁷⁸, because the meso mappings constrain it
(Fig. 3E). The recursion realizes this: only valid decompositions spawn
variants into the next level's pool. **[exists / established]**

### 2.2 Measured cost anatomy of the landed search

Sweep: `complexes()` at default `SearchBounds()` on the 2024 paper's Example 1
substrate (n = 4, state all-OFF, `iit4_2023` preset). Results (single M-series
core, sequential):

| quantity | value |
|---|---|
| systems evaluated (memoized, deduplicated) | 80 |
| wall time | 0.85 s |
| complex found | {αβ} = {A,B},{C,D} both-ON coarse grains, φ_s = 1.0040208141253277 (the SP2 golden) |
| time in macro-TPM construction | 0.057 s (≈ 8%) |
| time in SIA (partition sweeps) | 0.67 s (≈ 92%) |
| per-unit TPM constructions performed | 162 |
| **distinct (footprint, grain) construction keys** | **6** |

Padding the same substrate with weakly self-coupled noise units:
n = 5 → 161 systems, 1.9 s; n = 6 → 323 systems, 6.3 s (same winner). Both
factors grow together: the candidate count with the combinatorics, the
per-candidate cost with 4^n (construction) and with the macro unit count m
(partition sweep; `DIRECTED_SET_PARTITION` counts: m = 3 → 22, 4 → 150,
5 → 1,061, 6 → 7,896).

The 92/8 SIA/construction split holds at n = 4 only. The construction builds a
(2^n × 2^n) transition matrix per unit per direction-independent Step 1–2 pass
(`_full_transition_matrix`), so its share grows exponentially with micro size
— see §3.1.

### 2.3 What one evaluation costs, and the floor under it

Evaluating a single candidate system of m macro units over an n-unit micro
universe costs:

1. **Construction** (Eqs. 26–40): Θ(τ · 4^n) per unit for the discounted
   transition matrix and the chained sequence-class accumulation, plus cheap
   compression. Exponential in the *micro* universe, regardless of how coarse
   the candidate is.
2. **SIA**: the system-state specification (ii, over 2^m macro states per
   direction), then the partition sweep (super-exponential in m).

Point 1 is a floor no search cleverness removes: the paper-faithful
construction chains the modified TPM over the whole universe so background can
percolate (their explicit ordering requirement). Any exact grain search pays
Θ(4^n) at least once per distinct (footprint, τ, apportionment-structure). The
honest consequence: **grain discovery inherits the same exponential wall in
micro size that a single macro evaluation already has.** The search problem is
about not paying that floor more often than necessary, and about not paying
the m-partition sweep at all for candidates that cannot win.

---

## 3. Structure that can be exploited

Four levers, in decreasing order of confidence. Each was tested against the
library.

### 3.1 Construction sharing across mappings and candidates **[proposed; bit-identity verified]**

Steps 1–2 of the construction do not read the unit's mapping. Step 1
(`_discounted_on_probabilities`) depends on the unit's micro constituents and
the apportionment structure of the *other* units; with default (empty)
apportionment it depends on the unit's footprint alone. Step 2
(`_unit_sequence_distributions`) adds only the grain τ. The mapping enters at
Step 4, a cheap compression of the sequence-class distribution through the
truth table.

Verified: for a fixed decomposition at grain 2, the Step 1–2 intermediates are
bit-identical across all candidate mappings (`np.array_equal` on the transition
matrix and the sequence-class distribution). Measured share of construction
time that is mapping-independent, for a 2-unit decomposition on Example 1
padded to n micro units:

| n micro | mapping-independent fraction of construction |
|---|---|
| 4 | 0.20 |
| 5 | 0.27 |
| 6 | 0.46 |
| 7 | 0.76 |
| 8 | **0.95** |

And the E1 sweep needed only 6 distinct (footprint, grain) keys for its 162
per-unit constructions — a 27× key redundancy even in a tiny search.

Design: a per-run cache keyed on `(footprint, τ, apportionment-key)` holding
the discounted transition matrix and the per-τ sequence-class distributions,
added beside the existing `system_cache` (which caches whole constructed
systems but not the intermediates that different systems share). The
apportionment-key must capture the patron structure Step 1 reads
(`_patron_units` plus each patron's keep-set); under the default
`apportionment="NONE"` it collapses to the unit's own footprint. Expected
effect: at n ≈ 8+, construction cost per *distinct decomposition* replaces
construction cost per *mapped variant* — a factor approaching the number of
mapped/grained variants per decomposition (5–10 under FAMILIES at the default
bounds, hundreds under EXHAUSTIVE). This is a pure result-preserving
optimization; goldens must stay byte-identical.

Deeper (research-grade): the (2^n × 2^n) matrix is itself a product of per-unit
Bernoulli factors (`_full_transition_matrix` materializes what
`_discounted_on_probabilities` keeps factored). The chained distribution only
needs the joint over (sequence-class of U^J, current universe state), and the
universe-state part is what forces 2^n. For substrates whose discounted
dynamics factor across weakly coupled blocks, a factored/tensor-network
propagation could beat 4^n — but that is an approximation program with a
correctness burden this document does not discharge. **[proposed, unexplored]**

### 3.2 The intrinsic-information gate and best-first scheduling **[proposed; strong evidence; certification depends on formalism version]**

The system-state specification ii (both directions, no partition sweep) is
computed *before* the partition sweep inside every `sia()` call. Marshall 2023
defines φ as informativeness against the *partitioned* repertoire where ii uses
the *unconstrained* repertoire, and its Theorem 1 proof chain passes through
φ_s ≤ φ_e ≤ (log-ratio) — the intrinsic information is the natural ceiling on
integration. Under the 2026 formalism with
`system_phi_measure="INTRINSIC_INFORMATION"`, φ_s ≤ ii(s) holds *by
construction* (the Eq. 23 cap). Under the 2023 GID measure it is not, to my
knowledge, a published theorem.

Measured, `iit4_2023` preset, GID for both φ_s and the specification measure:

- **min(ii_c, ii_e) ≥ φ_s in 262/262 systems** across four substrates
  (Example 1; the sfs dancing-couples variant; an asymmetric 4-unit substrate;
  a deterministic 4-cycle), covering micro subsystems of every size, and
  2-unit macro systems over all FAMILIES mappings at grains 1 and 2. Zero
  violations at config precision.
- ii costs 2.7× (m = 2) to 23× (m = 4) less than the full `sia()`, and the
  gap grows with the partition count (150 partitions at m = 4; 7,896 at
  m = 6).
- Retrospective bite: with the Example 1 winner's φ_s as incumbent, 79/80
  systems in the sweep have ii below it — their partition sweeps were wasted
  work.
- Forward algorithm: order candidates by descending min-direction ii and run
  the full partition sweep only when ii exceeds the incumbent. On the Example
  1 sweep this needs **exactly 1 full partition sweep** (the true winner,
  which ii ranked first) versus 45 in as-encountered order, saving 78% of
  total SIA time even after paying for all 80 ii computations.

Design: a two-stage evaluation inside the Eq. 19 sweep (and inside the Eq. 16
competitor sets):

1. Construct the TPM (already unavoidable — it also performs the
   reachability admissibility check, which drops unreachable-state candidates
   before any analysis **[exists]**).
2. Compute ii (both directions). If `ii_min < incumbent − 10^-precision`,
   record the candidate with its ii ceiling and skip the partition sweep.
3. Otherwise run the sweep and update the incumbent.

Correctness conditions, stated precisely:

- **Eq. 19 winners are preserved.** A gated candidate x overlapping incumbent
  w has φ_x ≤ ii_x < φ_w, so x is not a complex, and w's strict superiority
  over x is certified without knowing φ_x. Ties are the sharp edge: the gate
  must be strict at precision — a candidate with ii_x tolerance-equal to the
  incumbent must be fully evaluated, because a φ tie between overlapping
  candidates excludes *both* (Eq. 19 / the `ComplexesResult.ties` machinery).
- **Eq. 16 verdicts are preserved** by the mirrored argument: a competitor
  whose ii is strictly below the candidate's φ_s cannot beat or tie it.
- **The gate's validity is the ii ≥ φ_s inequality.** Certified under
  2026/INTRINSIC_INFORMATION (definitional); conjecture with 262/262
  empirical support under 2023/GID. Under 2023 it should ship the way P13
  gated pruning was to ship: behind a shadow-mode equality flag (evaluate
  everything, assert the gate would never have changed a verdict) until the
  inequality is proved or a counterexample retires it. A B1-style runtime
  assertion (`phi_s ≤ ii + tol` at SIA construction) would collect evidence
  suite-wide for free.
- **Determinism and the parallel≡sequential invariant.** The incumbent
  depends on evaluation order. To keep parallel runs bit-identical to
  sequential ones (the N2 invariant the search already honors), the gate must
  key on a deterministic schedule — e.g. batch all ii computations for a size
  class (embarrassingly parallel), then apply the gate in the fixed canonical
  order with a deterministically-updated incumbent. Gating decisions then
  never depend on worker timing.
- **API consequence.** `ComplexesResult.records` currently reports exact φ_s
  for every evaluated system. Gated candidates would carry
  `(ii ceiling, gated)` instead of exact φ. That is a reporting change worth
  making opt-in (`prune="certified" | "off"`), with `"off"` the default until
  the inequality's status is settled.

Why this lever and not the published size bounds: ii adapts to the actual TPM.
It is exactly the quantity that distinguishes a promising grain (selective,
deterministic macro dynamics — the property that made Example 1's coarse grain
win) from a hopeless one, and the theory itself already treats it as the cap on
what integration can deliver.

### 3.3 Size ceilings across grains — mostly a negative result **[established bounds; negative bite evidence]**

The one certified system-level bound, φ_s ≤ m(m−1) for m units
(`system_phi_upper_bound`, from Marshall 2023 Thm. 1: φ_s at most the
connections cut by the selected partition), looks purpose-built for grain
search: coarse grains have few units, so a fine-grain incumbent could wipe out
whole coarseness classes without constructing a single TPM. Three findings
temper this:

1. **It is currently out of domain for macro systems** — deliberately.
   `_bounds_apply_to` excludes `MacroSystem`, documented with the singleton
   case: a single macro unit has m(m−1) = 0 yet can carry φ_s > 0 (its SIA
   partitions its self-loop; measured singleton φ_s up to 0.06 on Example 1,
   and the authors' `bu` example has singleton φ_s = 1.0). The corrected
   singleton ceiling is 1 (one self-connection cut, 1 bit for binary units).
   For m ≥ 2 the m(m−1) argument (partitions never sever self-connections)
   carries over to macro systems *if* Theorem 1's per-partition reasoning
   holds for the construction's paired cause/effect TPMs — plausible (the
   pipeline analyzes m binary units exactly as micro ones) but not certified.
   Every macro system in this document's sweeps respected the corrected
   ceilings. Extending the certificate — with the m = 1 correction — is a
   small, well-scoped theory task and a prerequisite for any use here.
2. **It rarely bites.** In Example 1 the winner has φ_s = 1.004 < 2 =
   ceiling(2): no 2-unit grain is ever excluded by size alone, and larger
   sizes have larger ceilings. This mirrors the P13 SP2 bite-rate verdict at
   the micro level (bounds don't prune in the certified domain), and the
   mechanism is the same: certified ceilings are loose exactly where
   candidates are plausible.
3. **When it would bite, ties can defeat it.** On a deterministic 4-cycle the
   micro incumbent (φ_s = 2.0), the 2-unit ceiling (2.0), and the best 2-unit
   temporal grain (φ_s = 2.0 at τ = 2, hitting the ceiling exactly —
   deterministic tightness, as the bounds paper predicts) coincide. Strict
   ceiling pruning cannot fire on equality, and must not: the tie between
   overlapping candidates changes the outcome (mutual exclusion).

Verdict: implement the m(m−1)+singleton ceiling only as a free sanity assert
and as a pre-construction skip for the regime where it provably fires
(incumbent φ_s strictly above m(m−1) at precision — deterministic, dense,
high-φ substrates where fine grains reach φ_s > 2 and coarse pairs are then
dead on arrival, *TPM construction avoided entirely*). Do not build scheduling
around it.

### 3.4 A spectral cap on useful temporal grain **[proposed; quantitative evidence; not yet certified]**

Intuition: the construction chains the *discounted* transition matrix τ times
(Eq. 31). If that chain mixes, dependence of the sequence distribution on the
starting state decays geometrically at the second-largest eigenvalue modulus
(SLEM), so every τ-grain unit's macro TPM rows converge and effect
information — hence φ_s, via §3.2's inequality — dies at rate SLEM^τ.

Measured, best φ_s over all FAMILIES mapped variants of the {A,B},{C,D}
decomposition per τ:

| substrate | τ=1 | τ=2 | τ=3 | τ=4 | SLEM of discounted matrix |
|---|---|---|---|---|---|
| Example 1 (mixing) | 1.0040 | 0.0197 | 0.00204 | 0.000223 | 0.11 |
| deterministic 4-cycle | 0.1813 | 2.0 | 0.0 | 0.0 | ≈ 0 (nilpotent-like) |

In the mixing case the successive ratios from τ = 2 on are 0.104 and 0.109 —
the decay tracks SLEM = 0.11 almost exactly. (The τ = 1 → 2 drop is larger
partly because the coarse-graining mapping family is only defined at grain 1;
at τ ≥ 2 the FAMILIES policy enumerates black-boxings only.)

The deterministic row is the instructive one, in both directions. First, τ = 2
*wins* (2.0 ≫ 0.18): temporal macroing is a real axis, not a formality, and
decay is not monotone from τ = 1 — so this cap prunes the tail, never the
head. Second, φ_s is exactly 0 from τ = 3 although the raw TPM is a
permutation (SLEM 1): what governs is the SLEM of the *discounted* matrix,
because Step 1 noises every connection that does not run inside the unit's own
footprint (or its apportionment) — once the chain is longer than the longest
causal path that stays inside the footprint, the start state is overwritten by
noised units and nothing macro can see it. The discounting is itself the
mixing source. A unit can only sustain long temporal grains if it contains (or
is apportioned) a causal circuit; this is checkable from connectivity alone.

Design: a per-(footprint, apportionment) τ cap. The discounted matrix is built
once anyway (§3.1 cache); its SLEM (or, cheaper and certified for total
variation, the Dobrushin contraction coefficient computable from row-pair
overlaps) yields τ_max ≈ log(tol)/log(rate) beyond which every mapped variant
at that footprint is guaranteed below the incumbent — with the tolerance tied
to `config.numerics.precision` and to the incumbent through the ii inequality.
The chain from "TV distance between conditional rows below ε" to "φ_s below
incumbent" needs one honest lemma (the log-ratio measures diverge as
partitioned probabilities approach 0, so the bound must route through the
mixed floor the discounting guarantees); until that lemma is written this is a
heuristic cap with strong empirical behavior, and should be surfaced as an
explicit bound parameter (`max_update_grain` today) auto-suggested rather than
silently applied.

### 3.5 What is *not* there: no monotonicity, no lattice order **[established, demonstrated]**

The 2024 paper is explicit that whether φ_s rises with more units or coarser
grain is a balance (their Sec. 4), and the numbers show it is not even locally
orderly: on Example 1, φ_s(micro pair {A,B}) = 0.044 > φ_s(micro whole ABCD) =
0.020 < φ_s(macro pair {αβ}) = 1.004, while every mixed 3-unit system stays
below 0.027; on the 4-cycle, τ = 1 < τ = 2 > τ = 3. Refining or coarsening can
move φ_s either way at any point in the lattice. Hill-climbing over the
refinement lattice, greedy agglomeration, or any search that assumes the
maximum has monotone approach paths is unlicensed by the theory and refuted by
these small examples. Any beam/greedy variant PyPhi ever ships must be labeled
approximate, full stop.

---

## 4. The design, assembled

Keep `intrinsic_units` → `valid_systems` → `complexes` exactly as the outer
loop (it is the theory's own definition of the candidate space). Add, in
order of decreasing certainty:

1. **Construction-intermediate cache** keyed on (footprint, τ,
   apportionment-key) (§3.1). Result-preserving; byte-identical goldens;
   pays off from n ≈ 6 up and multiplies with every other lever, since every
   gated-but-constructed candidate shares its Steps 1–2 with its siblings.
2. **Deterministic two-stage scheduling** of every φ_s batch (§3.2): batch
   ii for a size class in parallel; sort canonically by descending ii; sweep
   partitions only while ii exceeds the deterministic incumbent, with strict
   precision margins so ties are never skipped. Certified under
   2026/INTRINSIC_INFORMATION; shadow-mode under 2023/GID until ii ≥ φ_s is
   proved for GID or refuted. Worst case (all candidates' ii above the true
   maximum) it degenerates to today's behavior plus nothing — ii was being
   computed anyway.
3. **Ceiling asserts and the one certified skip** (§3.3): m(m−1) with the
   singleton correction, extended to macro systems if the small theory task
   goes through; used to skip construction only when the incumbent strictly
   exceeds a size class's ceiling.
4. **τ-cap advisory** (§3.4): report the Dobrushin/SLEM-implied τ ceiling per
   footprint; auto-suggest `max_update_grain`; apply as a hard prune only if
   the TV→φ lemma is established.

What this buys, concretely: on the worked Example 1, the sweep's 45
as-encountered partition sweeps drop to 1, construction cost per decomposition
replaces cost per variant, and nothing about the result changes — same
complex, same golden φ_s, same ties. On substrates an order larger, the same
mechanics apply but the floor of §2.3 stands: candidate counts still grow
combinatorially and each distinct footprint still costs Θ(4^n) once.

### What honestly remains intractable

- **The micro-exponential construction floor.** Θ(4^n) per distinct
  (footprint, τ) is inherent to the paper's Eqs. 26–31 (background must
  percolate through the whole universe). Grain discovery on a 20-unit micro
  substrate is out of reach for the exact construction regardless of search
  order. The only exits are approximations (factored propagation, §3.1) with
  their own validity burden.
- **The mapping axis under EXHAUSTIVE.** 2^(2^(τ|V|)−1)−1 is doubly
  exponential; no ordering fixes that. FAMILIES is a declared restriction of
  the hypothesis space, not a completeness result — a substrate whose maximal
  grain uses a mapping outside both families would be missed, and nothing in
  the papers rules that out. (Scope note: at τ ≥ 2 FAMILIES currently
  enumerates black-boxings only — the coarse-grain family is defined at grain
  1 — so e.g. "count of ON constituents at the final update" mappings at
  higher grains are reachable only via EXHAUSTIVE.)
- **Decomposition growth.** Bell-number-like growth of decompositions per
  footprint and of Eq. 18 assemblies per pool survives every lever above; the
  criteria prune it well in practice (Example 1: 11 judged decompositions, 2
  valid, so the mapped-variant explosion never starts — 8 died at Eq. 15
  integration, 1 at Eq. 16), but a substrate with many valid meso units
  regrows the tree.
- **Ties.** Every strict inequality in Eqs. 16/19 forces full evaluation of
  tolerance-tied candidates; highly symmetric substrates (where grain
  questions are most interesting) manufacture exact ties by isomorphism.
  Symmetry-aware deduplication (the `pyphi/automorphism.py` machinery,
  evaluating one representative per orbit) is the principled answer and
  composes with everything above. **[proposed, unexplored]**

### Tractable special cases, precisely delimited

- **Hierarchical substrates** — the paper's own conjecture, realized by the
  recursion: when maximal grains are built from certified meso units, the
  mapping space collapses from 10⁷⁸-scale to tens. This is the one case where
  the search is *already* efficient relative to the space it covers.
- **Mixing substrates**: the temporal axis is effectively finite
  (SLEM/Dobrushin cap), and the spatial sweep is dominated by ii-gated
  skips — the measured 98.75% bite is the expected regime, since mixing keeps
  most candidates' selectivity (hence ii) low.
- **Near-deterministic, densely connected substrates in high-φ states**: the
  only regime where size ceilings can fire across grains (fine
  incumbents can exceed coarse ceilings), *and* the regime of exact ties, so
  gains are real but bounded by tie handling.
- **Sparse effective connectivity**: a macro system whose effect TPM shows a
  unit independent of another has a smaller per-partition cut count; a min-cut
  refinement of the ceiling would discriminate same-size grains. But the MIP
  is selected on *normalized* φ (`normalization_factor`), so "φ_s ≤ min-cut"
  is not implied — the selected partition need not be the min-cut one. A
  usable certified variant: evaluate the single min-cut partition θ_mc (one
  partition, not the sweep); normalized-selection then gives
  φ_s ≤ N(θ′) · φ(θ_mc)/N(θ_mc) ≤ m(m−1) · φ(θ_mc)/N(θ_mc). Whether that
  one-partition certificate bites often enough to matter is an open
  bite-rate question of exactly the P13 kind, and should be answered by the
  same method (shadow-mode study) before any wiring.

---

## 5. Findings to hand upstream, independent of any search work

1. **ii ≥ φ_s under 2023/GID** held in 262/262 diverse system evaluations at
   precision. Either it is provable (making the certified gate available
   under the default measure) or a counterexample exists and would be
   theoretically interesting in itself. Cheap to monitor: a B1-style assert
   at SIA construction.
2. **Singleton ceiling correction**: `system_phi_upper_bound(1) = 0` is
   inconsistent with singleton systems whose self-loop partition carries
   φ_s > 0 (up to 1.0 in the authors' own `bu` example). The macro exclusion
   in `_bounds_apply_to` hides this today; any extension of the certificate
   to macro systems needs m = 1 → 1 (binary self-cut) first.
3. **The discounted matrix, not the raw TPM, governs temporal grain**: the
   construction's Step 1 noising makes long update grains unsustainable for
   any unit whose footprint (plus apportionment) contains no causal circuit —
   a connectivity-checkable admissibility fact about τ that the papers do not
   state and that materially shrinks the temporal axis (deterministic 4-cycle:
   every 2-unit footprint is φ_s = 0 for all mappings from τ = 3 on).
4. **Scope note** worth a line in the macro docs: FAMILIES has no
   coarse-graining mappings at τ ≥ 2 (by the family's grain-1 definition), so
   higher-grain count-based mappings require EXHAUSTIVE within its cap.

## 6. Reproduction

All numbers above come from seven short scripts run against the working tree
(`uv run python`, `iit4_2023` preset override, default precision). Sketch of
each, sufficient to reconstruct:

- **Sweep anatomy**: `complexes(Substrate(CG_TPM), (0,0,0,0), SearchBounds())`
  with `CG_TPM` as in `test/macro/test_macro_tpm.py`; per-record re-timing of
  `MacroSystem.from_micro` vs `.sia()`; construction keys counted as
  `(unit.micro_constituents, unit.micro_grain)` over all recorded systems.
- **Mapping-share**: `_discounted_on_probabilities` +
  `_unit_sequence_distributions` compared across `candidate_mappings(2, 2, …)`
  variants (`np.array_equal`); timing split on Example 1 padded with
  `P(ON) = 0.1 + 0.7·self` units to n = 4…8.
- **ii gate / audit**: `system_intrinsic_information(system,
  specification_measure=resolve_mechanism_measure(config.formalism.iit
  .specification_measure))`, min over directions, vs `system.sia().phi`, over
  the 80 sweep systems plus micro subsets and 2-unit macro variants (grains
  1–2) of the sfs dancing-couples, asymmetric, and 4-cycle substrates.
- **Best-first simulation**: replay of the recorded (ii, φ, timing) rows in
  descending-ii order with an incumbent gate.
- **Temporal decay**: max φ_s over FAMILIES mapping pairs of {0,1},{2,3} at
  τ = 1…4 vs `np.linalg.eigvals` SLEM of the discounted matrix; deterministic
  contrast is the 4-unit cyclic shift in state (0,1,0,1).
- **Recursion anatomy**: `intrinsic_units(...)` verdict `Reason` counts at
  default bounds and `max_update_grain=2`.
- **Scaling**: `complexes()` on the padded substrates at n = 5, 6.
