# IIT quantities as functions of substrate parameters: differentiability, optimization, and sensitivity analysis

**Status: exploration — not a plan, nothing here is scheduled or committed to.**

PyPhi evaluates Φ, φ_s, φ_d, intrinsic information, and relation φ for a
*given* substrate. This document asks what these quantities look like *as
functions of the substrate's continuous parameters* (connection weights, TPM
entries), and what that permits or forbids for optimizing over substrates,
searching substrate space, and sensitivity analysis. The characterization is
derived from the definitions (Albantakis et al. 2023; Barbosa et al. 2020),
verified against the code in `pyphi/core/repertoire_algebra.py`,
`pyphi/measures/distribution.py`, and `pyphi/formalism/iit4/`, and tested
empirically against the library. All experiments below ran on this working
tree under the default `IIT_4_0_2023` configuration (GID measure,
`DIRECTED_SET_PARTITION`, precision 13); scripts and raw data are in
`substrate_landscape_experiments/` (untracked).

## Summary of conclusions

1. **Within a "selection regime" — a region of parameter space where every
   argmax/argmin in the pipeline (specified states, MIPs, purviews) picks the
   same winner — every IIT quantity is an analytic function of the weights,
   and finite-difference derivatives are stable to ~10 significant figures.**
   Gradients exist almost everywhere and are cheap to exploit locally.

2. **The landscape is partitioned into finitely many such regimes, and the
   quantities are discontinuous across some regime boundaries.** The
   discontinuities are not numerical artifacts; each one traces to a specific
   definitional choice, and they differ in kind:
   - *Specified-state switches* (argmax of intrinsic information, Eq. 12)
     produce genuine jumps in φ_s and φ_d, because the state that maximizes
     `ii` is then evaluated under a different function (the partitioned
     probability ratio).
   - *MIP switches* (argmin of **normalized** φ, Eqs. 22–23) produce jumps in
     the reported (unnormalized) φ_s exactly equal to the ratio of the two
     partitions' normalization factors — while **normalized φ_s is continuous
     across these switches** (verified to 3×10⁻⁸ at a bracketed switch point).
   - *Purview switches* (argmax of φ itself, Eqs. 45–46) do **not** produce
     jumps — the selection criterion is the reported value, so the max is
     continuous and the switch is only a kink.
   - *The positive-part clamp* |·|₊ (Eqs. 19–20) creates regions of exactly
     zero φ with exactly zero gradient, over large fractions of parameter
     space (50–85% of the 1-D sweeps below). The *signed* value underneath
     remains smooth and carries usable gradient information.
   - *Distinction existence* is gated by these same mechanisms, so
     distinctions blink in and out of the cause–effect structure carrying
     finite φ_d, and Φ (sum over distinctions and relations) inherits each
     such event amplified by the relations that appear or vanish with the
     distinction (observed single-step changes in Φ of 2.5–2.8 on a 3-node
     substrate). Φ is much rougher than φ_s.

3. **The discontinuity structure is theory, not implementation.** The argmax
   over states is the "principle of maximal existence"; the specificity axiom
   of the intrinsic difference (Barbosa et al. 2020) is precisely what
   replaces a smooth expectation (KL) with a nonsmooth max. A substrate
   crossing a selection boundary changes *what exists* intrinsically
   (which state is specified, where the fault line is). Any smoothing is
   therefore a surrogate for optimization, not a redefinition candidate.

4. **Optimization is viable, and the choice of objective matters more than
   the choice of optimizer.** A naive finite-difference gradient ascent on
   signed φ_s took the Fig. 1A substrate from φ_s = 0.134 to φ_s = 2.396 in
   685 SIA evaluations (~21 s), beating a same-budget random search (best
   1.254). It works because it is implicitly climbing the *continuous*
   normalized-φ landscape; the raw φ_s trace oscillates between normalization
   branches. Conversely, the clamped φ objective stalls immediately in the
   zero region (gradient exactly 0), and even the signed objective can be
   locally *anti-correlated* with where the jumps lead: in the dead zone
   next to the Fig. 1A point, the smooth gradient points away from the
   boundary at which φ_s jumps back to positive values. Local methods are
   blind to jumps; population/zeroth-order methods are the robust default.

5. **The binding constraint is evaluation cost, not smoothness.** A 3-node
   SIA costs ~20 ms; n = 6 costs ~24 s and the system-partition count grows
   ~7.8× per node (measurements from the cross-formalism benchmark). A
   finite-difference gradient needs 2·n² evaluations per step. Beyond n ≈ 6,
   any strategy that treats the SIA as a black box is wall-clock-bound;
   the realistic escapes are surrogate models, bounds-based pruning
   (`pyphi/formalism/iit4/bounds.py`; Zaeemzadeh & Tononi 2024), or
   analytic gradients at one-evaluation cost (feasible per below, but a
   substantial project).

6. **Sensitivity analysis should report selection margins, not just
   derivatives.** The derivative of φ_s at a point describes only the local
   regime. The more informative object is the distance to the nearest
   selection switch. Concretely: the published Fig. 1E value φ_s(ABC) = 0.13
   sits within **0.005** (in the A→B weight) of a specified-state switch at
   which φ_s collapses to exactly 0. A derivative at the published point
   gives no hint of this.

---

## 1. What exists today

In PyPhi (this tree):

- All quantities are computed for a fixed `Substrate`/`System`. The
  parameterization `substrate_generator.build_substrate(unit_functions,
  weights, **kwargs)` already gives a smooth map from a weight matrix to a
  substrate (e.g. `ising.probability`, a sigmoid in the weighted input), so
  "substrate space" has a natural continuous coordinate system that the
  library itself uses to build the 4.0 paper's example systems.
- `pyphi.sweep` iterates the *discrete* axes of a fixed substrate: states,
  candidate subsets, formalisms. Nothing sweeps the substrate's parameters.
- IIT's own pipeline is already full of **discrete** searches over a fixed
  substrate: maximal complexes (`complexes()`), intrinsic-unit grain search
  (`pyphi/macro/system.py`, after Marshall et al. 2024, which is explicitly
  "maximize φ_s over candidate grains"), purview/partition/state selection.
  The continuous complement — moving the substrate itself — is absent.
- `signed_phi` and `signed_normalized_phi` are computed and exposed on every
  SIA (the |·|₊ clamp is applied at the end and both raw values are kept).
  This turns out to be exactly what optimization needs; no new computation is
  required to get the well-behaved objective.
- The tie machinery (`pyphi.resolve_ties`, `PyPhiFloat` comparisons at
  `config.numerics.precision`) detects exact selection ties. It does not
  report *margins* (how close the runner-up was), which is the natural
  sensitivity certificate (§6).
- `pyphi/formalism/iit4/bounds.py` ships analytic upper bounds with
  certificates (after Zaeemzadeh & Tononi 2024) — usable to prune search.
- The ROADMAP wishlist already contains adjacent ideas: N22 explicitly plans
  "φ_s as a function of substrate determinism; the monad and
  inverse-temperature curves" (experiment E5 below is a first instance of
  that curve), N23 plans a "φ_s landscape over all candidate subsystems with
  local-maximum detection" display, and N24 plans distinction-importance /
  relation-removal sensitivity. N23 and N24 are within-substrate (they do not
  vary the substrate's parameters); N22 is the closest existing commitment to
  the questions examined here.

Established outside PyPhi:

- Evolutionary search over substrates tracking integrated information is the
  main prior art (the animat line: Edlund et al. 2011; Albantakis et al.
  2014) — genetic algorithms over deterministic logic-gate genomes, treating
  the quantity as a black-box fitness. No gradient structure was used.
- Zaeemzadeh & Tononi 2024 derive closed-form upper bounds (φ ≤ |M||Z| per
  mechanism, Σφ_d ≤ (N²/2)·2^N, relation-sum bounds with growth O(N²·2^(2^N)))
  and characterize which architectures approach them: deterministic
  (selectivity-1) mechanisms, disjoint or symmetric purview structure,
  homogeneous grid-like connectivity. Their abstract frames this as
  techniques to *design* high-Φ systems. Notably, the bound-achieving regime
  (determinism) is exactly the regime where gradients vanish (experiment E5)
  — bounds and gradients are complementary tools that live at opposite ends
  of parameter space.
- The IIT 4.0 paper itself discusses only the combinatorial cost of
  *evaluating* a given substrate ("nested combinatorial explosions", p. 39)
  and describes architecture classes qualitatively (feed-forward ⇒ φ_s = 0;
  dense grids ⇒ very high Φ). It contains no substrate-optimization
  procedure. No prior gradient-based treatment of these quantities was found
  in the papers surveyed; the differentiability characterization here may be
  novel (stated as belief, not established fact).

## 2. The parameter space

A substrate is (topology, parameters, unit functions):

- **TPM coordinates.** For binary units the state-by-node TPM is a point in
  [0,1]^(n·2^n) (each entry an independent Bernoulli parameter given the
  input state). This is the maximal continuous parameter space; every
  quantity is ultimately a function of these entries and the current state.
- **Weight coordinates.** `build_substrate` composes the TPM entries from a
  weight matrix through the unit function (sigmoid at temperature T). This
  map is analytic, so smoothness statements transfer from TPM space to weight
  space unchanged. Weights are the natural coordinates for design questions
  (n² parameters instead of n·2^n), at the price of restricting to the
  sigmoid submanifold.
- **Topology is derived, and discrete.** `build_substrate` sets
  `cm = (weights != 0)`. A weight crossing 0 changes the connectivity matrix,
  which changes purview filtering and reducibility short-circuits. In the
  sweep below, w[C→B] = 0 exactly yields an immediate `NullCut` SIA
  (system reducible, φ_s = 0) while every neighboring point runs the full
  partition search. This is a measure-zero set, but optimizers that drive
  weights through zero will step on it. Feed-forward topologies are φ_s = 0
  *regions*, not points (4.0 paper, p. 37): topology classes carve exact-zero
  basins in weight space.
- **Boundary of the space: determinism.** TPM entries at 0 or 1 (weights → ∞,
  or T → 0). The informativeness terms are log-ratios
  (`pointwise_mutual_information_vector` is `log2(p/q)` with only 0/0
  mapped to 0), so a deterministic substrate whose partitioned probability
  for the specified state is 0 yields φ = +∞. The interior of the space is
  where calculus works; the boundary is where the suprema live (Zaeemzadeh's
  bounds are achieved at selectivity 1). This tension is structural: **the
  landscape's high ground is at the edge of the region where gradients carry
  information** (experiment E5).

## 3. How the quantities behave: the pipeline, operation by operation

Every IIT 4.0 quantity is built from the same small set of operations.
Regularity is decided entirely by which of these compose it:

| Operation (code / paper) | Effect on regularity |
|---|---|
| Repertoire entries: products of per-node conditional probabilities, cause side normalized (`repertoire_algebra._cause_repertoire_inner`; Eqs. 5–9) | Effect repertoires and forward probabilities: **multilinear polynomials** in TPM entries. Cause repertoires: **rational** (normalized products; denominator > 0 for reachable states). Analytic in weights. |
| Selectivity × informativeness (GID; Eqs. 19–20, `generalized_intrinsic_difference`) | Analytic where all probabilities > 0; diverges as partitioned probability → 0 (determinism boundary). |
| max over purview states of `ii` (Eq. 12, `intrinsic_information`) | The max *value* is continuous (piecewise-analytic, kinks at ties). But downstream φ evaluates a **different function** at the argmax state ⇒ **jump discontinuities** in φ where the argmax switches. |
| argmin over partitions of **normalized** φ; report **unnormalized** φ at the argmin (Eqs. 22–23 system, 42–44 mechanism; `resolve_ties` strategy `NORMALIZED_PHI`) | Normalized φ: continuous with kinks (min of continuous branches). Reported φ: **jumps**, with jump ratio = ratio of the two normalization factors (at the switch the normalized values are equal). |
| argmax over purviews of φ_d (Eqs. 45–46, `resolve_ties.purviews` strategy `PHI`) | Selection criterion = reported value ⇒ max is **continuous**; kink only. (The *identity* of the purview still switches discretely — the CES's shape changes even where its numbers don't.) |
| min over cause/effect directions (Eqs. 21, 47); min with the ii cap (2026 Eq. 23) | Continuous; kinks. |
| positive part \|·\|₊ (Eqs. 19–20; `utils.positive_part`) | Continuous; creates **exact-zero plateaus of positive measure** with zero gradient. The pre-clamp signed value stays smooth. |
| Distinction existence (φ_d > 0 with both MIC and MIE) and relations over the existing set (Eqs. 49–55) | Existence is gated by quantities that themselves jump ⇒ distinctions appear/disappear at **finite** φ_d. φ_r ≤ min φ_d of the relata (4.0 paper, p. 28), so relations cannot create discontinuities on their own — but each distinction-level jump is amplified by every relation touching it. Φ inherits the sum. |

Composite picture: each quantity is **piecewise-analytic over finitely many
open regimes** whose boundaries are the tie sets of the selection operations
(state ties, normalized-MIP ties, purview ties, φ = 0 crossings, cm changes).
Within a regime, derivatives of every order exist and the active branch is an
explicit analytic formula with all selections frozen. Across boundaries:
continuity holds for purview switches and direction/cap minima; it fails for
state switches and (in the reported, unnormalized value) MIP switches.
Gradients therefore exist almost everywhere, but the function is not
continuous, so none of the standard guarantees for nonsmooth-but-continuous
optimization (subgradient methods, bundle methods) apply globally.

Two engineering notes that matter for anyone probing this numerically:

- `PyPhiFloat` stores exact doubles and only *compares* at
  `config.numerics.precision` (1e-13 by default). Finite differences on
  returned values are meaningful at any h ≫ machine epsilon. But tie
  *detection* is quantized at 1e-13: within that distance of a boundary, the
  library treats candidates as exactly tied and the configured tie-break
  strategy, not the parameters, decides the winner.
- The kernel memoizes per-System by content fingerprint, so perturbed
  substrates recompute from scratch; there is no incremental evaluation.
  A perturbation API that reuses unaffected repertoires is a real (unbuilt,
  nontrivial) optimization opportunity — most single-weight perturbations
  only touch the factors of the target node.

### IIT 3.0 in brief

The IIT 3.0 quantities substitute EMD for the GID at both levels. EMD is the
optimum of a transportation LP — piecewise-linear and *continuous* in the two
distributions, hence continuous piecewise-smooth in the TPM. But the same
selection architecture sits on top (purview argmax, MIP argmin over a
normalized criterion in the `BI`-style schemes, concept existence gating the
CES), so the qualitative picture — piecewise regularity, selection kinks,
existence jumps in Φ-like sums — carries over, with the difference that IIT
3.0's per-partition values are only piecewise-linear rather than analytic.
No experiments were run under `iit3`; this paragraph is derivation only.

## 4. Experimental evidence

All experiments: 3-node Ising-sigmoid substrates (`ising.probability`,
T = 1/4, k = 4 equivalent), state (1,0,0), starting from the IIT 4.0 paper's
Fig. 1A weights (A↔B = +0.7, A→C = +0.2, C→B = −0.8, self A,B = −0.2,
self C = +0.2), which reproduce φ_s(ABC) = 0.1339 (paper: 0.13). Scripts
`exp1_sweep.py` … `exp4_saturation_ascent.py`, raw per-point data in the
`*_raw.json` files alongside; the one randomized experiment (the
random-search baseline in `exp4_saturation_ascent.py`) is seeded
(`seed = 20260707`, isolated `np.random.default_rng`), and the seed is stored
in its output file.

### E1 — one-parameter sweeps of φ_s (`exp1_sweep_raw.json`)

Sweeping w[A→B] over [0.02, 1.40] (277 points, with the identity of the MIP
and the specified cause/effect states recorded per point) partitions the
interval into **6 selection regimes**. Within regimes φ_s moves smoothly
(local slope ~0.6/unit weight). At the boundaries:

| boundary | what switches | φ_s behavior |
|---|---|---|
| w ≈ 0.2675 | MIP (same parts, different cut directions) | continuous |
| w ≈ 0.4527 | MIP: total tripartition (norm 5) → {AB}/{C} (norm 2) | **jump 0.678 → 0.271** |
| w ≈ 0.7025 | specified cause state (0,1,1) → (0,1,0) | **jump 0.134 → 0** (signed: +0.134 → −0.037) |
| w ≈ 0.7475, 0.9975 | MIP identity (within the clamped-zero region) | φ_s stays 0 |

50.5% of this sweep and 85.4% of the C→B sweep have φ_s exactly 0 — the
clamp's dead zones dominate even near a published positive-φ example. The
C→B sweep also shows the isolated topology point: at w[C→B] = 0 exactly, the
cm loses the edge and the SIA short-circuits to `NullCut` (φ_s = 0 without a
partition search); at ±0.005 the full search runs.

### E2 — derivative stability (`exp2_derivatives_raw.json`)

At a generic point (w[A→B] = 0.60), central differences of φ_s converge to
dφ_s/dw = −0.55334983; the estimates at h = 1e-5 and h = 1e-6 agree to 10
significant figures, and all estimates across h ∈ [1e-8, 1e-3] agree to 6.
φ_s is, for numerical purposes, exactly smooth inside a regime. Inside the clamped-zero plateau
(w = 0.72): d(φ_s)/dw = 0 identically, while d(signed φ_s)/dw = +0.5475,
stable across the same h range — the clamp, not the underlying function, is
what kills the signal.

### E3 — anatomy of a MIP switch (`exp2_derivatives_raw.json`)

Bisecting the w ≈ 0.4527 boundary to 1e-9 and evaluating at ±1e-7:

- reported φ_s: 0.677676 (left) vs 0.271070 (right) — ratio 2.4999, exactly
  the 5/2 ratio of the two partitions' normalization factors;
- signed normalized φ_s: left − right = 2.6×10⁻⁸ — continuous;
- normalized slopes: −0.028 (left) vs −0.235 (right) — a kink.

This confirms the Eq. 22/23 mechanism precisely: selection happens on the
normalized value (continuous), reporting on the unnormalized one (jumps).

### E4 — the cause–effect structure and Φ (`exp3_ces_sweep_raw.json`)

The same A→B sweep at CES level (distinctions + `sum_phi_relations`,
Φ = Σφ_d + Σφ_r) has **14 structural regimes** (vs 6 for the SIA): purviews
switch more often than system-level selections. Two behaviors, cleanly
distinguished in the data:

- **Purview switches are continuous.** E.g. mechanism (0,1)'s cause purview
  walks (1,) → (0,1) → back at w ≈ 0.09–0.16 with φ_d steps of ≤ 0.003 per
  0.01 grid step (= slope × step, no excess) — including one crossing with a
  step of 0.0000.
- **Existence switches are jumps.** Mechanism (1,) disappears at w ≈ 0.705
  carrying φ_d = 0.523; (0,) disappears at w ≈ 0.75 carrying φ_d = 0.455;
  (0,) reappears at w ≈ 1.145 with φ_d = 0.732 immediately. Each event moves
  Φ by 1.4–2.8 in a single 0.01 step (largest: Φ 5.90 → 8.71), with roughly
  three-quarters of the jump carried by the relations that appear or vanish
  with the distinction. Since φ_r ≤ min φ_d over relata, a distinction fading
  continuously to zero would take its relations with it continuously; the
  observed finite-φ_d blinking is the state-switch/clamp jump at mechanism
  level, amplified by the relation sum. **Φ is the roughest quantity in the
  family, and any Φ-targeting optimization inherits every mechanism-level
  discontinuity times its relation degree.**

### E5 — toward determinism (`exp4_saturation_ascent_raw.json`, E6 section)

Scaling all Fig. 1A weights by s ∈ [0.25, 8] (T fixed): signed φ_s rises to
+0.137 near s ≈ 0.75, crosses into negative territory past s ≈ 1.1, then
decays toward 0 from below with |dφ/ds| shrinking (0.033 at s = 3.75, 0.012
at s = 6.75) as the TPM saturates (mean |p − ½|·2 = 0.80 at s = 3.75). Along
this particular ray the deterministic limit is a vanishing-gradient regime —
consistent with sigmoid saturation: every derivative through the TPM carries
a factor p(1−p) per perturbed entry. This does not contradict the suprema
living at determinism (Zaeemzadeh); it means gradient methods lose traction
exactly where the bounds become tight, so the endgame of any ascent toward
high-φ deterministic structures must be handled by discrete/combinatorial
moves, not gradients.

### E6 — optimization demos (`exp4_saturation_ascent_raw.json`, E7 sections)

Naive central-difference gradient ascent over all 9 weights (h = 1e-4,
step 0.25 with 6-halving backtracking, 30 steps):

- **From Fig. 1A, objective = signed φ_s:** 0.134 → **2.396** in 685 SIA
  evaluations (~21 s). A same-budget uniform random search over
  [−1.2, 1.2]⁹ (seeded) reached 1.254. The trajectory is instructive: raw
  φ_s repeatedly crashes by exactly the 5/2 normalization ratio and recovers
  (1.848 → 0.743 → 2.069 …), while the signed **normalized** φ_s underneath
  ascends near-monotonically 0.067 → 0.479. The optimizer survives the
  cliffs only because they are normalization artifacts invisible to the
  continuous function it is effectively climbing.
- **From a dead-zone start (w[A→B] = 0.9), objective = clamped φ_s:** stalls
  at step 0, |gradient| = 0 exactly. This is the default `phi` attribute —
  the obvious objective is the worst one.
- **Same start, objective = signed φ_s:** climbs −0.044 → −0.004 but never
  crosses into positive φ_s in 30 steps. The local gradient in the dead zone
  points *away* from the state-switch boundary at which φ_s jumps back to
  +0.13 — the smooth signal and the jump structure disagree about the
  direction of improvement. An honest summary: signed φ restores gradient
  *information* but not gradient *guidance*; escaping a dead zone may
  require jump-aware moves (restarts, population methods, or explicitly
  tracking the runner-up state's ii margin).

Cost extrapolation: at n = 6 (~24 s/SIA, from the cross-formalism benchmark),
one FD gradient step (2·36 + 1 evaluations) is ~30 minutes. FD ascent is a
3–5-node tool.

## 5. Implications for optimization and search

Ranked by leverage:

1. **Objective transformation (zero new machinery).** Optimize
   `signed_normalized_phi`, not `phi`. It is continuous across MIP switches,
   has nonzero gradients in the clamped and negative regions, and its
   maximizers coincide with maximizers of φ_s wherever φ_s > 0 within a
   regime (positive scaling). Remaining discontinuities: specified-state
   switches only. For CES-level objectives there is no equally clean fix:
   Σφ_d and Φ jump at every distinction existence event; a smoother proxy
   (e.g. Σ over *all* candidate mechanisms of signed φ_d, no existence gate)
   is a surrogate design choice, not a free transformation.

2. **Zeroth-order population methods are the robust default** (established
   practice: GAs in the animat literature; CMA-ES is the modern equivalent
   for the continuous weight space). They are indifferent to jumps and dead
   zones, parallelize across the population (PyPhi's per-process caches are
   already compatible with that), and at n ≤ 5 the ~20 ms evaluation makes
   populations of hundreds practical. This is the recommended first
   capability: nothing about the landscape obstructs it, and the E6 random
   baseline (best 1.25 in 685 draws) shows even unstructured search moves
   quickly at small n.

3. **Local FD gradients work within regimes** (demonstrated), and Danskin's
   theorem / envelope arguments say the branch derivative with all selections
   frozen *is* the correct one-sided derivative at non-tied points — so an
   **analytic gradient at one-evaluation cost** is possible in principle:
   forward-mode differentiation through the repertoire algebra (products,
   normalizations, log-ratios) with the selections taken from the primal
   evaluation. Two honest caveats: (a) it cannot be bolted onto the existing
   NumPy pipeline without either a parallel JAX/autograd implementation of
   the kernel or hand-derived derivative formulas — a substantial project
   with the usual risk of divergence from the reference implementation; and
   (b) it inherits every limitation gradients have here (blind to jumps,
   vanishing at saturation). Worth doing only if optimization becomes a
   first-class use case at n ≥ 5 where the 2n²-evaluation FD cost bites.

4. **Smoothed surrogates are available but change the quantity.** Every
   nonsmooth operation is a finite min/max over an enumerable candidate set,
   so temperature-softened versions (log-sum-exp over states; softmin over
   partitions; softmax-weighted evaluation over candidate states to heal the
   argmax-composition jumps) give a family of C^∞ surrogates converging to
   the true values as τ → 0 — the standard Nesterov-smoothing /
   perturbed-optimizer construction. This is a *research* direction, not an
   engineering one: the surrogate at τ > 0 is not φ_s, and the specificity
   axiom (Barbosa et al.) is an explicit argument *against* averaging over
   states. Any use should anneal τ → 0 and report final values under the
   true definitions.

5. **Bounds-guided and structure-guided search.** Zaeemzadeh & Tononi's
   results say where the high ground is (deterministic, symmetric, grid-like,
   large disjoint purviews) and `bounds.py` can prune candidates cheaply.
   A sensible design loop for "find me a high-Φ substrate of n units" is
   coarse: search over symmetric weight templates (few parameters), use
   bounds to discard classes, spend evaluations only inside promising
   classes, and finish with discrete refinement in the near-deterministic
   regime where gradients are useless anyway.

**Hopeless as stated, and why:**

- *Global gradient-based optimization of raw φ or Φ with
  smoothness-assuming machinery* (L-BFGS, line searches asserting Wolfe
  conditions, Newton methods): the objective is discontinuous on a dense-in-
  practice arrangement of regime boundaries; line searches will bracket
  cliffs, and curvature estimates straddling a boundary are garbage.
- *Autodiff through the existing code*: `PyPhiFloat` comparisons, content-
  cache lookups, tie resolution, and short-circuit branches make the current
  implementation non-traceable; this is not a "swap in JAX arrays" job.
- *Gradient methods expecting to reach the suprema*: the suprema are on the
  determinism boundary where gradients vanish (E5) and φ can diverge to +∞
  (log-ratio with vanishing partitioned probability). Any ascent that
  approaches the boundary needs a stopping/handoff criterion.
- *Treating Φ as the optimization target at n beyond toy sizes*: Φ requires
  the full CES + relations per evaluation, is the roughest quantity in the
  family, and its cost grows with the same nested combinatorics the 4.0
  paper flags. Optimize φ_s (or a distinction-sum surrogate) and audit Φ on
  candidates.

## 6. Sensitivity analysis as a capability

Two distinct products, both cheaper than optimization:

- **Local sensitivities.** ∂(signed φ_s)/∂w_ij by central differences is
  reliable (E2) and needs 2·n² SIA evaluations — practical to n = 5–6. It
  answers "which connection does this system's integration depend on most",
  a question adjacent to the ROADMAP's N23 covariates and N24
  distinction-importance items but aimed at the substrate rather than the
  structure.
- **Selection margins (the more informative object).** At any point, for
  each selection in the pipeline, the gap between winner and runner-up —
  normalized-φ gap to the second-best partition, ii gap to the second-best
  specified state, φ_d gap to the second-best purview — is already computed
  and discarded inside the argmin/argmax loops. Exposed, these margins are
  robustness certificates: a small ii margin says "this substrate is near a
  boundary where what it specifies changes discretely." For the Fig. 1A
  system, the specified-cause-state switch lies ~0.005 away in the A→B
  weight; its φ_s value is fragile in a way no derivative reports. Since selection
  switches are *the theory's own claims about what exists*, margin reports
  are arguably the correct IIT-native sensitivity analysis, and they fall
  out of computations PyPhi already performs (the ties machinery detects
  margin = 0; this generalizes it to margin = ε).

## 7. Sketch of a library surface (design only — nothing implemented)

Smallest useful increment, in order:

1. `perturb(substrate, state, param_index, h) -> (value, derivative)` and a
   `landscape_section(substrate, state, param_index, grid)` returning a tidy
   DataFrame like `sweep`'s, with per-point selection identities (partition,
   specified states, per-mechanism purviews). Everything in §4 was done with
   ad-hoc versions of these two calls; they are also the natural display
   artifact ("show me φ_s along this weight").
2. Margin reporting on `SIA`/`explain()`: second-best normalized-φ partition
   gap and second-best ii state gap (both directions). Near-zero margins
   flag tie-quantization risk at `precision` as a side effect.
3. A black-box optimizer driver (CMA-ES or even Nelder-Mead over weights,
   objective = signed normalized φ_s, seeded, raw trajectory saved) with the
   substrate-generator parameterization as the search space. Population
   evaluation parallelizes over the existing `parallel` infrastructure.
4. Only if demanded by scale: the analytic-gradient kernel (§5, item 3).

## 8. Open questions

- **Is the specified-state jump ever *the* answer rather than an obstacle?**
  A substrate crossing that boundary changes which state it specifies —
  discretely. For substrate *design* ("build a system that specifies state
  X robustly"), the margin, not φ_s, is the design target, and the
  discontinuity is the feature being engineered. This reframing (optimize
  margins subject to φ_s > threshold) may be more IIT-native than maximizing
  φ_s itself; it is untested.
- **How rough is the landscape at n = 5–6?** Everything here is n = 3. The
  number of regime boundaries grows with the number of candidate selections
  (states × partitions × purviews), i.e. superexponentially; whether regimes
  become so small that "smooth inside a regime" stops being exploitable is
  an empirical question a n = 5 replication of E1/E6 would answer (hours,
  not weeks, of compute).
- **k-ary and composite units.** Nothing above used alphabet sizes > 2 or
  the composite mechanisms in `substrate_generator.mechanisms`; the
  smoothness ladder applies unchanged (the repertoire algebra is
  alphabet-generic), but the selection density and the dead-zone geometry
  could differ substantially.
- **Actual causation.** The AC quantities (α, PMI-based) share the argmax-
  over-occurrences structure and should inherit the same taxonomy; not
  examined.
- **The 2026 ii cap.** The cap (min with a partition-independent ii(s)) adds
  kinks but no new jump mechanism, and it binds exactly when the system's
  intrinsic information, not its fault line, is limiting; its effect on
  landscape geometry (does the cap widen or narrow dead zones?) was not
  measured.

## 9. Reproducibility

```
cd substrate_landscape_experiments
uv run python exp1_sweep.py           # E1: sweeps + selection segments (~12 s)
uv run python exp2_derivatives.py     # E2/E3: FD stability + switch anatomy (~30 s)
uv run python exp3_ces_sweep.py       # E4: CES/relations sweep (~12 s)
uv run python exp4_saturation_ascent.py  # E5/E6: scaling + ascent demos (~2 min)
```

Each script writes its full per-point raw data to a `*_raw.json` next to it
(never overwriting: they are inputs to this document; delete explicitly to
regenerate). The single randomized experiment is seeded and the seed is
stored in the output. All quantities were computed with the repository's
default configuration; float comparisons in the analysis respect
`config.numerics.precision` via the library's own `PyPhiFloat` semantics.
