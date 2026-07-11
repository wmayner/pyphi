# Synthesis of the six 2026-07-07 design explorations

**Status: reviewed 2026-07-08 — decisions settled (see §4a); the schedule
lives in ROADMAP.md under "Wave 7 — Exploration builds".**

Reconciles the six parallel explorations run 2026-07-07, checks them for
contradictions, extracts the places where they compose, collects the decisions
they leave to the maintainer, and proposes a dependency-ordered sequence.

The six documents:

| # | Exploration | Location |
|---|---|---|
| 1 | Formalism meta-theory & certified approximation | `docs/superpowers/specs/2026-07-07-formalism-meta-theory-and-certified-approximation.md` (worktree `formalism-meta-theory`) |
| 2 | TPM from data & uncertainty | `docs/superpowers/specs/2026-07-07-tpm-from-data-and-uncertainty-design.md` (worktree `tpm-uncertainty-exploration`) |
| 3 | Relations without materialization | `docs/superpowers/specs/2026-07-07-relations-without-materialization-design.md` (worktree `relations-without-materialization`) |
| 4 | Substrate parameter landscapes | `docs/superpowers/specs/2026-07-07-substrate-parameter-landscapes.md` (main tree; experiments in `experiments/substrate_landscape_experiments/`) |
| 5 | CES algebra | `docs/superpowers/specs/2026-07-07-ces-algebra-exploration.md` (main tree) |
| 6 | Grain discovery | `docs/superpowers/specs/2026-07-07-grain-discovery.md` (main tree) |

## 1. Verdicts at a glance

| Exploration | Headline result | Verification strength | Graduation status |
|---|---|---|---|
| Meta-theory | Unification real at mechanism level, **thin at system level** (3.0 min-of-transport vs 4.0 sum is a genuine disjunction, as suspected); two-sided certified Φ bracket for 4.0, upper side 126–1016× tighter than the shipped ceiling; **PyPhi's IIT 3.0 background convention diverges from the 2014 paper on proper-subset systems** | Strong on the bracket and the background divergence (scripts in worktree, live-library numbers incl. a pre-refactor baseline run); the axis factorization itself is analysis, not experiment | Bracket wiring + background config knob are build-ready; matrix half is a reference document |
| Uncertainty | Dirichlet posterior per TPM factor (Jeffreys prior, settled by a paired experiment); "distribution over Φ" = mixture (mass at Φ=0 + density conditional on existence) + existence/identity categoricals; **observational data cannot identify the needed object** (twin-substrate proof); Φ of the mean TPM ≠ mean of Φ | Strong on the empirical core (seeds, raw NPZ/JSON committed in worktree); three inline checks (twin substrate, `infer_cm` saturation, ε-boundary) are **not** committed as reproducible artifacts | Minimal build (`estimate_substrate`/`sample`/`phi_posterior`/coverage report) ready pending one policy call |
| Relations | The relation set is a deterministic view of a **linear-size summary** (purview-union + φ-density per distinction); four query tiers: exact closed-form, lazy top-K, sampling, enumeration; verified at Fig-6D scale (1,537,080 relations, Σφ_r digit-for-digit vs enumeration) | Strongest of the six — proofs plus at-scale numerical validation | Tiers 1–3 build-ready; discharges roadmap N6 and N24. Fold optimization stays research (supermodular) |
| Landscapes | Every IIT 4.0 quantity is piecewise-analytic over selection regimes; jumps come only from specified-state switches; `signed_normalized_phi` is the right objective; clamp dead-zones cover 50–85% of sweeps; the published Fig-1A point sits 0.005 from a φ_s→0 cliff; **selection margins are the IIT-native sensitivity analysis** | Strong at n=3 (six experiments, raw JSON in `experiments/substrate_landscape_experiments/`); everything at n>3 untested | Margin reporting + `landscape_section` build-ready; optimizer driver gated on an n=5–6 roughness check |
| CES algebra | Structure operations form a **meet-semilattice with no join** per frame (exclusion postulate as order theory); relation locality makes induced relations pure containment-filtering; select ≠ recompute (context non-locality); share-weighted μ is the additive fold measure | Strong for its scale — laws machine-verified, but only on 2–3-node fixtures | Six small primitives build-ready; μ naming needs a maintainer decision |
| Grain | `min(ii_c, ii_e) ≥ φ_s` in 262/262 evaluations (theorem under the 2026 cap, conjecture under GID); best-first ii-gating: 1 full partition sweep vs 45; construction Steps 1–2 mapping-independent (95% of construction cost at n=8, 27× key redundancy); **no monotonicity/lattice order in grain space** (greedy search unlicensed) | Strong empirically (seven scripts); the two load-bearing generalizations (ii ≥ φ_s under GID; macro ceiling) are unproved | Cache build-ready; ii-gating spec-ready (shadow-mode under GID); τ-cap and ceiling pruning gated on lemmas |

**No contradictions were found.** Where two explorations touch the same
territory they agree, and in four places they compose (§3). The one alarming
finding (§2) is a divergence between the code and the 2014 paper, not between
the explorations.

## 2. One urgent correctness question (before anything else)

The meta-theory exploration found that **PyPhi's IIT 3.0 mode applies the
IIT 4.0 causal-marginalization background convention**, where the 2014 paper
conditions background units on the current state. On full-substrate systems
the two coincide — which is why every committed golden and the N1 Fig-12
reproduction pass — but on a purpose-built proper-subset fixture the
end-to-end IIT 3.0 Φ differs materially (0.416 vs 0.720).

No committed golden currently discriminates the two conventions. This is
precisely the "don't defer confirmation experiments" pattern: before any of
the other work below locks in more IIT 3.0 state, run the confirmation —
anchor a **proper-subset IIT 3.0 SIA against the genuine PyPhi 1.x oracle**
(the CPython 3.9 recipe behind `scripts/gen_iit3_emd_oracle.py`) to establish
what 1.x actually did, then make the background convention an explicit config
knob (`presets.iit3` carrying whichever default the maintainer rules
paper-faithful) with a golden on each side. Until that runs, treat
proper-subset IIT 3.0 results as convention-ambiguous.

## 3. Where the explorations compose

**(a) Anytime certified Φ = meta-theory bracket + relations Tier 2.** The
meta-theory's open problem #1 asks for the two-sided bracket plus "top-K
decreasing-φ_r enumeration" to make it anytime. The relations exploration
built and verified exactly that enumerator (best-first descending φ_r,
top-10 of 1.5M in under 1 ms). Wiring the two together — bracket endpoints
from `bounds.py`, the lower endpoint rising as Tier-2 streams relations —
is one composed build, not two. (Gate: the empirical S(o) upper bound needs
its certification proof, meta-theory open problem #3.)

**(b) Selection margins are one primitive with three consumers.** The
landscapes exploration proposes margin reporting (gap to the second-best
partition and second-best specified state) as the IIT-native sensitivity
analysis. The uncertainty exploration independently needs exactly this as
the "selection-boundary guard" that makes delta-method screening sound
("the guard is the hard part"). And margins quantify the tie-quantization
risk every other document defers (CES algebra scopes ties out; grain
pruning is defeated by exact ties; relations queries require
`ResolvedDistinctions`). Build margins once, on the SIA/`explain()` surface,
and three programs consume it.

**(c) The CES-algebra μ needs the relations degree-spectrum.** Share-weighted
μ under `AnalyticalRelations` requires degree-resolved incident sums — which
is a Tier-1 closed-form query in the relations design. The algebra's
`induce`/fold surface should be specified against the relations query
interface rather than against enumeration.

**(d) ii-gating and the bracket exploit the same inequality family.** The
grain exploration's scheduler (order candidates by ii, sweep partitions only
while ii exceeds the incumbent) and the 2026 cap's `min(φ, ii)` are the same
`ii ≥ φ_s` structure. Proving that inequality under GID (grain open problem
#1) simultaneously upgrades the grain scheduler from shadow-mode to a
certified prune and gives the meta-theory a formalism-level lemma.

**(e) A shared pathological regime: the determinism boundary.** Four
documents hit it independently. Landscapes: gradients vanish toward
determinism (and the Zaeemzadeh suprema live exactly there). Meta-theory:
the error certificate's Lipschitz constant blows up as p_min→0. The 2026
cap: deterministic systems get φ_s = 0 exactly. Uncertainty: Laplace
smoothing is most harmful near deterministic boundaries (hence Jeffreys).
Any future optimization, approximation, or estimation work should treat
near-determinism as the regime where every tool degrades at once.

**(f) Epistemic vs intrinsic indeterminism.** The uncertainty exploration's
"Φ of the posterior-mean TPM is a silent lie" has a formalism reading: a
mean TPM converts *epistemic* uncertainty into what the 2026 formalism
treats as *intrinsic differentiation* — inflating the system's apparent
repertoire of alternatives while suppressing specification. The mixture
semantics (sample, then compute) is what keeps the two kinds of
indeterminism separate. Worth stating in the uncertainty spec's rationale;
possibly paper-adjacent.

**(g) The tractability frontier moves to distinctions.** With Tier-1/2/3
queries, relations cease to be the n≥6 bottleneck ("the bottleneck is now
entirely the distinctions"). The remaining cost centers are the CES/SIA
partition searches — which is where the grain exploration's ii-gating, the
landscapes' observation that perturbed substrates recompute from scratch
(a content-fingerprint-adjacent reuse opportunity), and the meta-theory
bracket all point. The scaling conversation should be reframed accordingly:
N19's "scaling narrative" now has a concrete spine.

## 4. Decisions the maintainer must make

1. **IIT 3.0 background default** (`presets.iit3`): paper-2014 conditioning
   vs current causal marginalization — after the §2 oracle run settles what
   1.x did. Highest stakes; blocks the config knob.
2. **Partial-coverage refusal**: should `phi_posterior` refuse to emit a bare
   scalar Φ under partial state coverage / observational regime (the
   exploration's recommended "honest library" behavior)?
3. **Compound-fold measure naming**: share-weighted μ vs count-once — which
   is `PhiFold`'s headline number, and what the other is called.
4. **Public query surface**: which relations Tier-1/2/3 methods become public
   API in the freeze sense (they extend the frozen 2.0 surface).
5. Minor: `prune="certified"` default (grain scheduler; exploration proposes
   off), and whether `landscape_section`/`perturb` live in `pyphi.sweep` or a
   new module.

## 4a. Decisions (settled 2026-07-08)

1. **IIT 3.0 background**: the §2 oracle ran against genuine PyPhi 1.2.0
   (control passed: anchored `basic` Φ = 2.3125 exact) — **1.x conditions
   background units at their current state** (cause repertoire exactly matches
   the conditioned prediction; end-to-end Φ = 0.72). Ship a
   `background_conditioning` knob; `presets.iit3` defaults to the 1.x
   convention; discriminating goldens on both settings; the 2014
   condition-at-past-state convention (never implemented in any version) is
   documented, not built. Artifacts:
   `pyphi_1x_background_oracle{.json,_runner.py}` beside the meta-theory
   verification scripts.
2. **Partial-coverage semantics**: no bare-float coercion on uncertain results
   anywhere; perturbational-regime summaries are free with the coverage report
   attached; observational-regime summaries are gated on the constructor-time
   `regime="observational"` assertion, stamped into provenance.
3. **Fold measure**: share-weighting replaces `big_phi_contribution` in place
   (the CES-algebra plan's Task 2 as written) — the count-once multi-seed value
   violates the additivity its own docstring promises; share-weighting is the
   unique extension that keeps Eq. 4 on singletons and tiles Φ over every
   partition. A coverage/incidence quantity, if ever wanted, gets its own name.
4. **Scheduling**: the Wave-1/2 builds go **into 2.0** (ROADMAP Wave 7),
   reopening the frozen surface deliberately, once, as a batch.
5. **ii ≥ φ_s under GID**: proved or refuted **as part of the work** (not
   deferred) — pointwise-per-partition argument first (it sidesteps the MIP
   normalization wrinkle), adversarial counterexample hunt as the falsification
   arm; the grain-prune default follows the outcome. The 2026-cap case holds by
   construction.

## 5. Proposed sequence

**Wave 0 — confirmation experiments and record-keeping (cheap; first).**
- §2: proper-subset IIT 3.0 vs the 1.x oracle; then the background config
  knob + discriminating goldens on both conventions.
- Commit the uncertainty exploration's three uncommitted inline checks
  (twin-substrate non-identifiability, `infer_cm` saturation, ε-boundary)
  as reproducible artifacts; fix the stale "three priors" docstring.
- Merge the three exploration worktrees' docs into the main tree (new files
  only; no conflicts expected).

**Wave 1 — independent, verified, small builds.**
- Relations Tier 1–3 query surface on the `Relations` ABC (+ visualization
  switching to `strongest(k)`; diff to statistic deltas). Discharges N6, N24.
- Selection-margin reporting on SIA/`explain()` (§3b's shared primitive).
  Extends N23.
- Grain construction-intermediate cache (result-preserving; golden-byte
  identical).
- CES algebra primitives: `filter`, `induce`, frame check, `relabel`,
  `is_isomorphic`, meet (μ lands once decision #3 is made).

**Wave 2 — composed builds (depend on Wave 1 + one proof each).**
- Anytime certified Φ bracket (§3a): `bounds.py` two-sided bracket + Tier-2
  streaming. Gate: the S(o) certification proof.
- ii-gated grain scheduling: shadow-mode under GID immediately; certified
  prune once the GID inequality is proved.
- Uncertainty minimal build: `estimate_substrate` / `.sample()` /
  `phi_posterior` / coverage report (decision #2), with margins-based
  screening as the tractability route.
- `landscape_section` + `perturb`; the black-box optimizer driver after an
  n=5–6 roughness replication of the landscape experiments.

**Wave 3 — theory ledger (each unlocks a gated item above).**
1. Prove/refute `min(ii_c, ii_e) ≥ φ_s` under GID → certified grain pruning.
2. Certify the empirical S(o) Eq. 15 bound (key on unit states) → ship the
   126–1016× tighter upper bracket.
3. TV-distance → φ_s lemma → hard τ-cap in grain search.
4. Fold-content optimization heuristics (supermodular; branch-and-bound with
   binding-matrix seeding) → content-fold selection.
5. Transport-level error bound for IIT 3.0 Φ, or an explicit decision that
   3.0 gets no certificate.
6. Distribution-over-structures summaries (label-switching/matching) — note
   this is exactly where the external optimal-transport structure-distance
   work would slot in when it becomes available.
7. Quantum system Φ (distinctions/relations/φ_s): tests whether the
   mechanism-level unification extends to the quantum object or forks again.

**Deliberately not scheduled** (explorations' own negative results): anytime
partial-CES lower bounds (exponentially back-loaded), greedy grain search
(no lattice), smoothness-assuming optimizers and autodiff-through-the-code
(non-traceable), a structure join (no such object), exact arbitrary relation
predicates (#P-hard), in-library resolution of observational
non-identifiability (impossible; it is an assumption, not an estimate).

## 6. Follow-through bookkeeping

- ROADMAP wishlist: N6 and N24 are discharged by the relations design when it
  lands; N22/N23 gain concrete content (margins, ii decomposition); N19's
  scaling narrative should be rewritten around §3g. The N12–N24 sweep entries
  should be revised from "candidate direction" to "explored; see spec" as
  each doc is accepted.
- PAPER-IDEAS.md gained candidates from these results: the piecewise-analytic
  landscape taxonomy with the Fig-1A cliff; the meet-semilattice/no-join
  result; the linear-sufficiency of the relation summary; the ii ≥ φ_s
  inequality; the 3.0 background-convention finding (erratum-shaped).
