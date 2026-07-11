# A generative meta-theory of PyPhi's Φ-formalisms, and Φ-approximation on a certified footing

**Status:** research spec, for maintainer review. Not committed by the author.
**Date:** 2026-07-07.
**Scope:** IIT 3.0, IIT 4.0-2023, IIT 4.0-2026, Actual Causation 2019, and the quantum-mechanism formalism; the Zaeemzadeh bounds and the standard Φ-approximations.

### Provenance legend

Every non-trivial claim is tagged by where it comes from, because the value of this document depends on not blurring the three:

- **[L]** — established in the primary literature (paper + equation cited).
- **[C]** — implemented in the codebase (file:line cited); may or may not match [L].
- **[N]** — new synthesis by the author of this spec (the axis factorization, the error-composition derivation, the two-sided bracket, the empty-cell predictions). Verified numerically where a number is given.

Numerical checks were run against the actual library (`uv run python`, `IIT_4_0_2023` unless noted), respecting `config.numerics.precision = 13` via `pyphi.utils.eq`. The verification scripts are reproduced in Appendix A.

---

## 0. Executive verdict (read this first)

The unification is **real at the mechanism level and thin at the system level.** This is an empirical finding from the definitions, not a framing choice, and it is the single most important result here.

**What genuinely unifies [N].** Every one of the five formalisms computes its atomic irreducibility quantity — IIT 3.0's φ, IIT 4.0's distinction φ_d, the 2026 φ_d, AC's α, and the quantum φ — by the *same* five-step recipe:

> select a purview by **max over purviews** (exclusion) of a **state-specific**, **do()-marginalized**, **product-factorized** repertoire, then take the **min over a partition family** of a **difference measure** between the intact and the partitioned repertoire.

The four things that vary between formalisms in that recipe — the **difference measure**, the **partition family**, the **background/marginalization rule**, and the **MIP normalization** — are genuinely free, orthogonal axes: the code already varies three of them through config, and the fourth (background) is a paper-level axis the code has collapsed. This is Deliverable A's positive core, and it holds to machine precision where I could test a coincidence.

**What does not unify [N].** The *system-level* object called "Φ" is a different mathematical shape in 3.0 versus 4.0, and no single parameterization captures both without a disjunction:

- In **IIT 3.0**, system Φ is a **minimum, over unidirectional cuts, of a transport distance (extended EMD) between two constellations of concepts** (Oizumi et al. 2014, Eq. 11 [L]). It is one integrated quantity, and it requires a ground metric on concept space.
- In **IIT 4.0**, the system has **two different** top-level quantities: the system irreducibility φ_s (a min over directed partitions, Albantakis et al. 2023 Eqs. 19–23 [L]) *and* the structure integrated information Φ = Σφ_d + Σφ_r (a **sum**, not a minimum, over the distinction-and-relation structure, Eq. 59 [L]). The paper is explicit: "Φ is not computed based on a partition (as system phi), but rather a sum of the integrated information within the structure" (2023, p. 29 [L]).

A minimum-of-a-transport-distance and a sum-of-irreducibilities are not two settings of one knob. The mechanism-level axes are shared; the top-level integration rule forks, and 4.0 additionally splits "the whole" into two objects (φ_s and Φ) that 3.0 conflates into one. Section 7 states exactly where the fork sits and why bridging it needs a case split rather than a parameter.

So: a rigorous partial unification, honestly bounded. The design matrix below is the coordinate system; its columns are the shared mechanism-level axes plus the *disjunctive* top-level column where the shared structure ends.

---

## 1. The design matrix

Rows are the five formalisms. Columns are the axes derived in Section 2. Cells cite the governing equation **[L]** and, where built, the code path **[C]**. "—" marks an axis the formalism does not engage; **⚠** marks a place where PyPhi's code diverges from the formalism's paper.

| Axis ↓ / Formalism → | **IIT 3.0** (Oizumi 2014) | **IIT 4.0-2023** (Albantakis 2023) | **IIT 4.0-2026** (Mayner 2026) | **AC 2019** (Albantakis 2019) | **Quantum** (Albantakis 2023) |
|---|---|---|---|---|---|
| **1. Repertoire object** | prob. vector over states | prob. vector | prob. vector | prob. vector over transition | **density matrix ρ** |
| **2. Alphabet** | binary [L] / k-ary [C] | binary [L] / k-ary [C] | k-ary [C] | binary/k-ary | qubits / finite-dim Hilbert |
| **3. State treatment** | single current state (Eq. 6–8) | single, state-specific (Eq. 5,7) | single, state-specific | single transition (before→after) | single, current ρ |
| **4. Background / marginalization** | *paper:* fixed (past-state cause, current-state effect) [L]; ⚠ *code:* extended-background causal marginalization (4.0 Eq. 4) [C] | causal marginalization, cond. on current state (Eq. 3–4) [L,C] | same as 2023 [L,C] | uniform do() over complement (Eq. 2) [L,C] | maximally-mixed ⊗, partial trace (Eq. 22) [L] |
| **5. Difference measure** (mechanism) | **EMD** (Wasserstein / Hamming ground metric) [L]; `hamming_emd` [C] | **GID** (selectivity×PMI), = ID on effect side [L]; `generalized_intrinsic_difference` [C] | **GID**, capped by intrinsic differentiation (Eq. 23) [L,C] | **PMI** (log-ratio, no selectivity weight) [L]; `pointwise_mutual_information` [C] | **QID** (von Neumann rel. entropy, max over eigenstates) Eq. 28 [L] |
| **6. Mechanism partition family** | joint bipartition + wedge (Eq. 4–5) [L]; `JOINT_BIPARTITION` [C] | disintegrating partitions Θ(M,Z), k≥2 (Eq. 38) [L]; `JOINT_PARTITION_ALL` [C] | same Θ(M,Z) [L,C] | partitions of x into m parts + unconstrained remainder (Eq. 7–8) [L]; `mechanism_partitions` [C] | same Θ(M,Z), applied on top of the separable partition P\* [L] |
| **7. Mechanism integration rule** | **min over partitions** (MIP), Eq. 6–8 [L] | **min over partitions**, Eq. 42 [L,C] | **min over partitions**, Eq. 21 [L,C] | **min over partitions**, α^MIP [L,C] | **min over partitions**, Eq. 31 [L] |
| **8. MIP normalization** | none | ÷ severed connections Σ\|Sⁱ\|\|Xⁱ\| (Eq. 43) [L]; `normalization_factor` [C] | same (Eq. 21) [L,C] | subtractive (ρ − ρ_MIP), implicit in PMI [L,C] | ÷ max φ (Eq. 16 analog) [L] |
| **9. Purview selection (exclusion)** | max over purviews (MICE), Eq. 9 [L,C] | max over purviews, Eq. 45–46 [L,C] | max over purviews [L,C] | max over occurrences, α^max, Def. 1 [L,C] | max over Z, Eq. 32–33 [L] |
| **10. TOP-LEVEL integration** ⚑ | **min over unidirectional cuts of extended-EMD(C, C_MIP)**, Eq. 11 [L]; `ces_distance` [C] | **two objects:** φ_s = min over directed partitions (Eq. 21) *and* Φ = **Σφ_d + Σφ_r** (Eq. 59) [L,C] | φ_s with the **ii(s) cap** (Eq. 23) *and* Φ = Σφ_d+Σφ_r [L,C] | big-A = **Σα − Σα_MIP** over the account (Eq. A3) [L,C] | (paper stops at mechanism φ; no system Φ defined) |
| **11. Relations computed?** | no (constellation only) [L,C] | **yes**, φ_r Eq. 55 [L,C] | yes [L,C] | no (causal account) [L,C] | no |
| **12. System partition family** | unidirectional cuts; `DIRECTED_BIPARTITION` [C] | directed set partitions (Eq. 14–16) [L]; `DIRECTED_SET_PARTITION` [C] | same [L,C] | bidirectional account partitions Ψ(v₋≺v) [L,C] | — |

⚑ Column 10 is the **disjunctive column**: this is where 3.0 and 4.0 are different shapes, not different points (Section 7).

---

## 2. Axis definitions, with citations and a confidence note each

For every axis I state: what it is, the governing equation, the code path, and — as the task demands — **whether it is a genuinely free and orthogonal degree of freedom, or an artifact I am imposing to make the picture look unified.**

### Axis 5 — Difference measure

**Definition [L].** The function D(p, q) comparing an intact repertoire to a partitioned (or unconstrained) one. The registered measures: EMD, KLD, L1, ID, GID/`INTRINSIC_SPECIFICATION`, PMI, `INTRINSIC_INFORMATION` (the 2026 composite), QID (paper only).

**Equations.** EMD is IIT 3.0's D (Oizumi 2014, Box 1 p. 4 [L]). ID is fixed *uniquely* by three axioms — causality, intrinsicality, specificity — to `D(Pⁿ,Qⁿ) = max_α |k·p_α log(p_α/q_α)|` (Barbosa et al. 2021, Theorem 1 [L]; Barbosa et al. 2020, Eq. 1 [L]). GID is the state-specific three-argument factoring selectivity×informativeness that ID needs on the cause side (IIT 4.0-2023 Eqs. 7, 20, 44 [L]; `generalized_intrinsic_difference`, `pyphi/measures/distribution.py:1088-1111` [C]). PMI is the unweighted log-ratio `log₂(p/q)` (AC 2019 Eq. 11 [L]; `distribution.py:1196-1216` [C]). QID replaces the classical sum in the von Neumann relative entropy with a max over ρ's eigenstates (Quantum 2023 Eq. 28 [L]).

**Verified coincidence [N].** The claim that on the **effect side** the intrinsic information equals the ID between constrained and unconstrained effect repertoires (2023, p. 16 [L]) holds to machine precision: across every non-degenerate (mechanism, purview) pair of `basic_system`, `ID(π_e, π_e^uc) = max GID(π_e, π_e^uc; selectivity=π_e)` with `pyphi.utils.eq` True (e.g. mech (1,2), purview (0,1): both 3.0000000000). On the **cause side** they do *not* coincide, because ID's selectivity would be the forward probability while ii_c's selectivity is the *backward* Bayes probability π_c^←(z|m) (2023, Eqs. 7, 9 [L]) — which is exactly why the code needs the *generalized* three-argument form rather than a two-argument ID.

**Confidence: genuinely free as an engineering axis; theoretically constrained; and it hides a fork.**
- Free at the code level — `config.formalism.iit.mechanism_phi_measure`, `ces_measure`, `alpha_measure` all swap it, and `IIT3Formalism` validates a whole compatible set (`{EMD, L1, KLD, ID, ...}`) [C].
- *Not* free as a theoretical commitment — Barbosa's uniqueness theorem means once you accept the intrinsicality+specificity postulates, the measure is forced to be the ID. The measure axis and the "which postulates" axis (Axis 11/normalization) are therefore **coupled, not orthogonal** [N].
- The axis is **not a line, it is a fork** [N]: EMD requires a **ground metric on the state space** (Hamming), whereas ID/GID/PMI/QID are *intrinsic* — they read only the probability values, never how states relate. This is the sharpest single difference between 3.0 and 4.0's measure choice, and it is a genuine structural branch, not a slider position. I flag it rather than smoothing it over.

### Axis 6 — Mechanism partition family

**Definition [L].** The set of partitions over which the MIP minimizes. IIT 3.0: joint bipartitions of (mechanism, purview) with the directed "wedge" (Eq. 4–5 [L]). IIT 4.0 and quantum: the **disintegrating partitions** Θ(M,Z) into k≥2 parts with the M⁽ⁱ⁾=M ⟹ Z⁽ⁱ⁾=∅ constraint (2023 Eq. 38 [L]; identical to Barbosa 2021 Eq. 4 [L]). AC: partitions of the occurrence into m parts plus an unconstrained remainder (Eq. 7–8 [L]).

**Code [C].** `pyphi/partition.py` registers each generator: `joint_bipartitions` (:436), `all_joint_partitions`/`JOINT_PARTITION_ALL` (:569), `mechanism_partitions` (:428).

**Confidence: genuinely free.** It is a registered strategy keyed by `partition_scheme`, and swapping it changes which formalism you are running without touching anything else. The disintegrating family strictly contains the bipartition family, so 3.0→4.0 is (in part) a family enlargement along this axis. This axis is real.

### Axis 7 — Mechanism integration rule

**Definition [L].** How the per-partition values are aggregated into the irreducibility. In **all five** formalisms this is **minimum over the partition family** (Oizumi Eq. 8; Albantakis 2023 Eq. 42; Mayner 2026 Eq. 21; AC 2019 α^MIP; Quantum Eq. 31 [L]).

**Confidence: this is a shared invariant, not a free axis [N].** I looked for a formalism that aggregates partitions by anything other than min and found none among the five. The min-over-partitions rule is part of the *common core*, not a coordinate. (Approximations relax it — Section 5 — but no exact formalism varies it.) Presenting it as a free axis would be the imposed-artifact failure mode the task warns against, so I explicitly do not.

### Axis 4 — Background / marginalization rule

**Definition [L].** How units outside the mechanism are handled under the causal intervention do(). All five share the do()-intervention (perturb the complement, average). What varies is the averaging distribution and its conditioning:
- IIT 3.0 *paper*: background fixed at actual past state (cause) / current state (effect) (Box 1 p. 4 [L]).
- Post-3.0: fixed at current state for both.
- IIT 4.0: **causally marginalize** the background, conditioned on the current state — the "extended background" (2023 Eq. 4; S2-comparison pp. 2–3 [L]).
- AC: uniform do() over the complement (Eq. 2 [L]).
- Quantum: replace the complement with the maximally-mixed state, then partial-trace (Eq. 22 [L]).

**Code [C].** The shared kernel `pyphi/core/tpm/marginalization.py:122-174` (`_cause_marginal_factored`, "IIT 4.0 Eq. 4") and `pyphi/system.py:330-360` implement the extended-background rule.

**Confidence: a genuinely free axis in the literature that the code has *collapsed* [N].** ⚠ This is the single substantive paper-vs-code divergence found. PyPhi's `IIT3Formalism` reuses the shared repertoire kernel, so it runs with IIT **4.0** background semantics on the cause side (`repertoire_algebra.py:196,214` docstrings cite "IIT 4.0 Eq. 5/7"; `system.py:52-56` "extended background convention of IIT 4.0"; cause factors from `_cause_marginal_factored`, `marginalization.py:122-174` [C]). So along this axis, PyPhi's "3.0" and "4.0" occupy the **same** cell, even though the papers place them in different cells.

**Precise scope of the divergence [N, verified].** The three conventions can only differ when the system is a **proper subset** of the substrate (nonempty `external_indices`); for a full-substrate system there is no background and all conventions coincide trivially. Consequences, each verified:

- **Full-substrate analyses are unaffected.** The 2014 paper's worked ABC example and PyPhi's IIT 3.0 SIA goldens (`basic_iit3_s`, `xor_iit3_s`) are full-substrate systems (`external_indices == ()` confirmed for `basic_system`, `xor_system`), so those published numbers are reproduced regardless of convention.
- **Effect side is convention-invariant between legacy and 4.0.** Both condition the background at its current state (`system.py:339-343` [C]); verified numerically (library == manual current-state conditioning).
- **Cause side diverges for subsystems, and the divergence is large.** Discriminating fixture (substrate A=OR-ish noisy gate with background parent C, system {A,B}, W={C}, u=(1,0,0)): the cause repertoire of mechanism {A} over purview {B} is **(0.40566, 0.59434)** under the library — matching a manual IIT 4.0 Eq. 4 computation to machine precision — versus **(0.1, 0.9)** under legacy current-state conditioning of C. Carried end-to-end through `iit3.sia` with the canonical `presets.iit3` (EMD, directed bipartition): **Φ_3.0 = 0.4160700000 (2.0 semantics) vs 0.7200000000 (legacy semantics, i.e. analyzing the C-conditioned 2-node substrate exactly as pyphi 1.x's `condition_tpm` on externals did)**. Script: `verify_background.py` in the verification directory.
- **No PyPhi version has ever implemented the 2014 paper's own convention.** The 2014 paper fixed the background at its actual **past** state for causes (Box 1 p. 4 [L]); that requires the past state, which PyPhi's API has never taken (`System` accepts only the current state). Legacy 1.x used the post-3.0 convention (current-state conditioning both directions — the convention of "publications since then \[11,12,16,17\]" per the S2 supplement, which include the 2018 PyPhi paper); 2.0 uses 4.0 causal marginalization. So for subsystem analyses there are **three** distinct conventions, and PyPhi has occupied two of them, neither of which is the 2014 paper's.

The motivation for 4.0's change is the unreachable-state pathology: fixing the background at the current state can make the current state unreachable (no cause) — S2 supplement pp. 2–3 [L]; `find_mip` still carries a null-RIA `UNREACHABLE_STATE` path for the all-zero cause repertoire case (`queries.py:166-167` [C]).

**Genealogy [N, verified from git history and a live run of the pre-refactor baseline].** The convention change did **not** happen during the 2.0 refactor, and the golden harness was never violated:

- Through mid-2023, `Subsystem` had a single TPM conditioned on externals at their current state, used for both directions (`condition_tpm(background_conditions)` — the published-1.x convention, per the pre-refactor repo at `3317ab8c^`).
- **2023-06-08**: `backward_tpm()` implemented per IIT 4.0 Eq. 4 (`fb668a3e`) and wired as an opt-in `Subsystem(backward_tpm=False)` kwarg (`3317ab8c`).
- **2024-06-07**: made structural — `cause_tpm = _backward_tpm(...)` unconditionally, cause repertoires always routed through it, for every `IIT_VERSION` (`380e9b9a`, `9c1eb5e2`).
- The 2.0 refactor ported this faithfully: the pre-refactor baseline (`b3aaa3e5`) and 2.0 produce the **identical** cause repertoire (0.40566, 0.59434) on the discriminating fixture. The 2.0 golden harness compares against that baseline, so it correctly reports no divergence; what no golden has ever covered is an IIT-3.0-mode **proper-subset** system compared against *published 1.x* output, because the IIT 3.0 SIA goldens are all full-substrate systems where the convention is invisible.

**Which is correct.** Per-formalism: for IIT 4.0 (2023/2026), Eq. 4 is definitional — the current behavior is correct. For IIT 3.0 mode, the current behavior matches none of the historical conventions: not the 2014 paper's (past-state causes — unimplementable without adding a past-state parameter to the API, and disavowed by the S2 supplement as unavailable from the intrinsic perspective), and not published 1.x's (current-state conditioning — the convention under which the 3.0-era literature's subsystem/complex results were computed). If reproducing 3.0-era published numbers for proper-subset analyses matters, Axis 4 should become an explicit config knob (e.g. `background_conditioning: CAUSAL_MARGINALIZATION | CONDITION_CURRENT_STATE`, with `presets.iit3` selecting the latter), and a proper-subset IIT 3.0 golden should be added so this surface is covered by the harness. Which default the 3.0 preset should carry is a formalism-fidelity policy decision for the maintainer.

### Axis 8 — MIP normalization

**Definition [L].** Whether the per-partition φ is divided by a normalizer before the min. IIT 4.0 divides by the number of severed connections Σ|Sⁱ||Xⁱ| = the max possible φ (2023 Eq. 43; Marshall et al. 2023 Theorem 1 [L]). IIT 3.0 does not. AC's subtraction is a degenerate normalization (the unconstrained term cancels in the PMI, Eq. 15 [L]).

**Code [C].** `normalization_factor` = `1/np.sum(cut_matrix)` (`pyphi/models/partitions.py:474-476`); knob `distinction_phi_normalization` (`state_specification.py:196-199`).

**Confidence: genuinely free, but finer-grained than a paper knob [N].** The code exposes it as a configurable normalization; the paper fixes it. It is orthogonal to the measure and partition axes (you can normalize any measure by any normalizer), so it is a real coordinate — but a secondary one, and it is the place where the code carries a degree of freedom the formalism does not.

### Axis 2/3 — State algebra (alphabet, aggregation) and Axis 1 (object)

**Definition [L].** Three sub-dimensions: the **alphabet** (binary vs k-ary vs continuous), the **state aggregation** (single specified state vs averaged over states), and the **object** (classical probability vector vs quantum density matrix).

**Findings [C/L].**
- *Alphabet:* k-ary is native in the code (verified: `repertoire_algebra.intrinsic_information` enumerates `alphabet_sizes` per node; `_kary_hamming_matrix` for the EMD ground metric [C]), though most papers use binary (Gomez et al. 2020 does multi-valued [L]). **Free, and largely built.**
- *Aggregation:* **every** current formalism is single-state / state-specific (Oizumi Eq. 6–8; 2023 Eqs. 5,19; AC transitions; Quantum current ρ [L]). Nothing in the code averages ii over states. **This is an unengaged degree of freedom** — see the empty-cell analysis (§4).
- *Object:* the quantum formalism moves from probability vectors to density matrices (Quantum 2023 [L]).

**Confidence: the alphabet and aggregation sub-axes are genuinely free; the quantum "object" move is NOT a free coordinate — it is a coupled package [N].** You cannot independently set "density matrix" while keeping "ID measure": a density matrix forces QID (max over *eigenstates*, with the overlap matrix P_ij), forces partial-trace marginalization, and forces the product structure to become a tensor product over the maximal *separable* partition P\* (Quantum Eqs. 22–28 [L]). So the single word "quantum" simultaneously sets Axes 1, 4, 5, and the factorization. Treating it as one free axis would be an imposed simplification; it is a coordinated move across four axes, and I say so.

### Axis 11 — Which postulates / how much structure

**Definition [L].** How much of the compositional structure the formalism builds, and how the whole is integrated (this is Column 10 of the matrix). Three observed settings: distinctions-only with a min-cut constellation Φ (3.0); distinctions **and relations** with a summed Φ (4.0); a causal account of cause/effect links with a subtractive big-A (AC).

**Confidence: genuinely free at the "how much structure" level; but the *integration rule at the top* is where unification fails, so this "axis" is really the seam of the disjunction, not a smooth coordinate [N].** Exclusion (max over purviews) and composition (evaluate the powerset of mechanisms) are shared postulates across 3.0/4.0/AC/quantum — those *are* common core. What is not shared is (a) whether relations exist and (b) whether the whole is a min-of-a-distance (3.0) or a sum-of-irreducibilities (4.0). I keep this as one matrix column for readability, but Section 7 is explicit that it is not one axis: it is the fork.

---

## 3. Placement of each formalism

Each formalism is a point (a tuple of axis settings) plus its top-level rule. Citations are consolidated from Section 2 and the code paths below.

**IIT 3.0** = ⟨prob-vector, binary[L]/k-ary[C], single-state, extended-background⚠, **EMD**, joint-bipartition+wedge, min-over-partitions, no-normalization, MICE-exclusion, **min-over-unidirectional-cuts of extended-EMD(constellation)**, no relations, unidirectional cuts⟩.
Code: `pyphi/formalism/iit3/`, registry `IIT_3_0` (`pyphi/formalism/__init__.py:42`); φ = `min(φ_c, φ_e)` (`queries.py:228`); constellation distance `pyphi/measures/ces.py:248` (extended EMD). Governing: Oizumi 2014 Eqs. 6–11.

**IIT 4.0-2023** = ⟨prob-vector, binary[L]/k-ary[C], state-specific, causal-marginalization, **GID**, disintegrating-partitions, min-over-partitions, ÷-severed-connections, purview-exclusion, **φ_s = min-over-directed-partitions AND Φ = Σφ_d+Σφ_r**, relations yes, directed-set-partitions⟩.
Code: `pyphi/formalism/iit4/`, registry `IIT_4_0_2023`; distinction φ_d `models/distinction.py:134`; system φ_s driver `iit4/__init__.py:744`; Φ `models/ces.py:194` (`sum_phi_distinctions + sum_phi_relations`); φ_r `relations.py:194`. Governing: Albantakis 2023 Eqs. 34–59; Marshall 2023 for φ_s.

**IIT 4.0-2026** = IIT 4.0-2023 **plus** the intrinsic-differentiation cap on φ_s: φ_s = min{φ_c, φ_e, **ii(s)**}, ii(s) = min over directions of min(intrinsic specification, intrinsic differentiation) (Mayner 2026 Eqs. 4–6, 13, 23 [L]).
Code: registry `IIT_4_0_2026`; `_apply_ii_cap` (`iit4/__init__.py:710-741`), applied once to the selected MIP (`:924-928`), gated on the `INTRINSIC_INFORMATION` measure's `applies_ii_cap` flag [C]. The MIP is still selected on the *uncapped* normalized φ (2023 Eqs. 21–22); the cap is a post-hoc floor, so it can only *lower* φ_s relative to 2023, never change which partition is the MIP (`:669-678` [C]).
**Verified [N]:** on `basic_system` (deterministic), φ_s(2023) = 0.4150374993 but φ_s(2026) = **0.0** exactly — the cap binds hard, because a deterministic system has zero intrinsic differentiation (no alternatives), so ii(s)=0 caps φ_s to 0. This is precisely the 2026 paper's motivating point (p. 4–5: "Perfectly deterministic systems could achieve high intrinsic information despite having no intrinsically defined alternatives" [L]), reproduced to machine precision.

**Actual Causation 2019** = ⟨prob-vector over a transition, binary/k-ary, single transition (before→after), uniform-do(), **PMI**, occurrence-partitions+remainder, min-over-partitions, subtractive, α^max-exclusion, **big-A = Σα − Σα_MIP over the account**, no relations, bidirectional account partitions⟩.
Code: `pyphi/formalism/actual_causation/`, **separate** registry `ACTUAL_CAUSATION_FORMALISM_REGISTRY` key `AC_2019` (`actual_causation/__init__.py:20`); α `compute.py:230`; PMI `distribution.py:1196`; big-A `compute.py:498` with `account_distance` = `sum(alpha) − sum(alpha_partitioned)` (`:165`). Governing: Albantakis 2019 Eqs. 2–16, A1–A3.
Direction semantics [L,C]: cause direction conditions on the **after** state (Bayesian-inverting from the observed effect), effect direction on the **before** state (`actual.py:134-139` [C]) — matching the resolved memory that cause uses after-state, effect uses before-state.
**Verified [N]:** α = PMI of intact vs partitioned probability, i.e. `PMI(0.8, 0.5) = log₂(0.8/0.5) = 0.6780719051` to machine precision, confirming α_e = ρ_e − ρ_e,MIP collapses to the log-ratio (Eq. 15 [L]).

**Quantum mechanism 2023** = ⟨**density matrix**, qubits/finite-Hilbert, current ρ, maximally-mixed⊗+partial-trace, **QID**, disintegrating-partitions-on-top-of-P\*, min-over-partitions, ÷-max-φ, eigenstate-exclusion, (no system Φ defined), no relations, —⟩.
Code: **NOT IMPLEMENTED** in PyPhi. The only `quantum` token in `pyphi/` is an unrelated rounding variable (`visualize/render/common.py:27`). Governing: Albantakis 2023 Eqs. 20–33; external reference toolbox `github.com/Albantakis/QIIT` (two/three qubits) [L]. ROADMAP N21 is `research`-tagged, unscheduled [C].

---

## 4. Empty-cell analysis: predicted-new vs incoherent

The matrix predicts cells no current formalism occupies. I sort them into **coherent-but-unbuilt** (a real formalism the axes permit) and **incoherent** (a cell the axes forbid, with the reason).

### Coherent-but-unbuilt (predicted new formalisms)

1. **State-averaged IIT [N, coherent].** Axis 3 = *averaged* instead of *single-state*: replace ii(s) with Σ_s p(s)·ii(s), i.e. the KLD/mutual-information version rather than the intrinsic point value. This is coherent — it is essentially the pre-intrinsic "effective information" reading of IIT, and the measures already support it (KLD is registered). It is unbuilt because IIT 4.0 deliberately chose the intrinsic (single-state) perspective. Predicted, buildable, theory-rejected rather than incoherent.

2. **k-ary IIT 3.0 with true EMD [N, coherent, partially built].** The code already runs k-ary repertoires and a k-ary Hamming EMD ground metric (`_kary_hamming_matrix` [C]), but the 2014 paper only defines binary examples. The cell (3.0 measure, k-ary alphabet) is coherent and mostly built; what is missing is a paper-level justification of the Hamming ground metric for k>2 states.

3. **Normalized IIT 3.0 [N, coherent].** Axis 8 = ÷-severed-connections applied to the 3.0 EMD MIP. Nothing forbids normalizing the 3.0 φ by connection count; it is simply not what the 2014 paper does. Coherent, trivially buildable, untested for whether it changes the MICE.

4. **Quantum IIT 4.0 system Φ [N, coherent, unbuilt].** The quantum paper stops at mechanism φ; it defines no quantum distinctions/relations structure and no quantum system φ_s. The matrix predicts these cells: a quantum distinction set (QID-selected purviews), quantum relations (overlap of density-matrix supports), and a quantum φ_s (min over directed partitions of QID). Coherent by direct analogy, entirely unbuilt in paper and code. This is the largest genuine research frontier the matrix exposes (ROADMAP N21).

5. **AC with relations [N, coherent].** AC computes a causal account but no relations among causal links. The 4.0 relation machinery (overlap of purviews) is defined over distinctions; transplanting it to causal links is coherent and would give a "relational actual causation." Unbuilt; theoretically unexplored.

### Incoherent cells (forbidden, with the reason)

1. **EMD + intrinsic (single-state) exclusion without a ground metric change [N, incoherent as a *drop-in*].** You cannot take the 4.0 intrinsic recipe and simply swap the measure to EMD and keep everything else, because EMD requires a ground metric on the purview state space that the intrinsic formalism deliberately discards. Barbosa 2021 proves EMD fails intrinsicality and specificity (p. 16 [L]). So "4.0 with EMD" is not a point in the shared space — it violates the axioms that pin the rest of 4.0 together. This is the measure-fork of Axis 5 made concrete.

2. **Density matrix + classical ID [N, incoherent].** Setting Axis 1 = density matrix while Axis 5 = classical ID is incoherent: ID reads a probability vector's entries, but a density matrix's off-diagonal coherences carry information ID cannot see. The correct object forces QID (Quantum Eq. 28 [L]); the coupling is not optional. This is the coupled-package finding of Axis 1/5 stated as a forbidden cell.

3. **Sum-integration (4.0 Φ) with no relations [N, incoherent as "4.0"].** Φ = Σφ_d alone (dropping Σφ_r) is a *computable* number, but it is not the 4.0 Φ — it discards the relational structure the composition postulate requires. It is coherent as *a* quantity (indeed it is the lower bracket of Section 6), but incoherent as a *formalism* claiming to satisfy 4.0's postulates. I flag it because the approximation literature sometimes reports Σφ_d as "Φ," which conflates a bracket endpoint with the theory's Φ.

4. **min-over-partitions at the 4.0 structure level [N, incoherent / category error].** One might try to "unify" 3.0 and 4.0 by making 4.0's Φ a min over partitions like 3.0's. This is incoherent: 4.0's Φ is definitionally a sum (Eq. 59 [L]); its *system* φ_s is the min-over-partitions object, and it is a different quantity measuring a different thing (system irreducibility, not structure content). Forcing them together is the exact false-unification the task warns against. See Section 7.

---

## 5. Approximation as principled relaxation

Each standard Φ-approximation is a **relaxation of one shared-core invariant along one identified axis**. This reframes "approximations" as movements in the same coordinate system, which is Deliverable B's first half. Status tags: **[L]** in literature, **[C]** in code, **[N]** my mapping.

| Approximation | Relaxes | Axis | Kind (per `ErrorInfo`) | Built? |
|---|---|---|---|---|
| **φ\*** (Oizumi geometric) | exact min-over-partitions → closed-form KL projection | Axis 7 (integration rule) | `different_quantity` (φ\* ≠ Φ) | **No** — Protocol seam only [C] |
| **φ_G** (graph / effective information) | exact min-over-partitions → spectral/graph surrogate | Axis 7 | `approximation_error` | **No** [C] |
| **Coarse-graining / macro** | micro state grain → macro grain | Axis 2/3 (state algebra) | `different_quantity` (macro Φ can exceed micro) | **Yes** — `pyphi/macro/` [C] |
| **Zaeemzadeh Σφ_r bound** | exact relation enumeration → certified closed-form ceiling | Axis 11 (structure completeness), upper side | `upper_bound` | **Yes** — `bounds.py` [C] |
| **Top-K relations** | exhaustive relation sum → K-largest partial sum | Axis 11, lower side | `approximation_error` (a valid lower bound) | **No** — trivially derivable [N] |

Notes that matter for correctness:

- **φ\* / φ_G are not in the papers assigned nor in the code [C/N].** The `ApproximateFormalism` Protocol (`base.py:148-158`) and the `formalism/approx/` directory are a **declared-but-empty seam**; there are **zero** registered approximate formalisms (`__init__.py:42-44` are all `exact=True` [C]). P16 is deferred post-2.0 (ROADMAP [C]). The `phi_star` symbol that exists in `bounds.py` is the *Zaeemzadeh construction* quantity φ\*_e(K), unrelated to Oizumi's φ\* — a name collision worth not tripping on [C]. So "φ\* relaxes Axis 7" is my mapping [N] onto a method the codebase reserves space for but has not built.

- **`approximate_specified_state` (`distribution.py:838`) is orphaned [C].** It approximates the argmax-over-states of the specified purview state in linear (not exponential) time — a relaxation of the *state-selection* max (part of Axis 3). It has **zero callers**, no config option, no test. It is a real relaxation, currently dead code.

- **Coarse-graining is a `different_quantity`, not an `approximation_error` [L,N].** Macro Φ can be *larger* than micro Φ (the whole point of intrinsic units, Marshall 2024). So calling it an "approximation of micro Φ" is wrong; it moves along the grain axis to a different, legitimate quantity. `bounds.py` correctly excludes `MacroSystem` from the certified domain (`:676-681`: a single macro unit can have φ_s>0 while n(n−1)=0 [C]).

- **The Zaeemzadeh bound is exactly a relaxation of exhaustive enumeration, and it is one-sided [L,C].** It replaces the sum over up to 2^(2^N)−1 relations with a closed-form ceiling (Eq. 16 [L]). There is **no lower bound anywhere** in the paper or the module (verified: `bounds.py` returns only `UpperBound`; the phrase "lower bound" appears only as a proof device in S3 [C]). The lower side is the missing piece Section 6 supplies.

---

## 6. Error composition and the two-sided certified bracket

This is the keystone and the genuinely new synthesis **[N]**. The goal: propagate approximation error up the compositional hierarchy — repertoire → mechanism φ → Σφ_d and Σφ_r → structure Φ — and assemble a *two-sided* certified bracket on the 4.0 Φ, complementing the one upper side `bounds.py` ships.

### 6.1 The hierarchy and the four links

Write the exact structure as Φ = Σφ_d + Σφ_r (2023 Eq. 59 [L]). An approximation perturbs the repertoires; I track how the error flows up.

**Link 0 — approximation → repertoire.** If the marginalization/TPM is approximated, measure the repertoire error in total variation: ‖π̂ − π‖_TV ≤ ε. (This is the input; its size depends on the specific approximation.)

**Link 1 — repertoire → mechanism φ_d.** φ_d = min over partitions, max over purviews and states, of `selectivity · |informativeness|₊`, where informativeness = log₂(π/π^θ) at the specified state. Two facts compose it:

- **The selection operators are 1-Lipschitz [N, verified].** `|min_θ f(θ) − min_θ ĝ(θ)| ≤ max_θ|f(θ)−ĝ(θ)|`, and likewise for max. So the argmin-over-partitions (MIP), the argmax-over-purviews (MICE), and the argmax-over-states (specified state) **pass error through with factor 1 — they do not amplify it.** Verified numerically (min: 0.002092 ≤ 0.014934; max: 0.002202 ≤ 0.014934 on random perturbation).

- **The per-evaluation map is Lipschitz with a probability-dependent constant [N, verified].** Perturbing the partitioned repertoire entry at the specified state by δ changes φ_e by
  `|Δφ_e| ≤ selectivity / (ln2 · p_min) · δ`,
  where p_min is the smallest partitioned probability at that state. Verified on 8 random draws (every case tight to within numerical slack, e.g. sel=0.422, part=0.231, δ=0.010: |Δφ|=0.025825 ≤ bound 0.026381). Combined with **selectivity ≤ 1** and **informativeness ≤ N(θ) ≤ |M||Z|** (Zaeemzadeh Lemma 2 / Theorem 1 [L]), this bounds the per-distinction error.

  **Honest failure mode [N, verified].** The constant carries a log singularity: as p_min → 0 the bound blows up (part=0.500 → L=2.31; part=0.005 → L=230.83). So the repertoire→φ_d error bound **degrades precisely for near-deterministic repertoires**, where a probability approaches zero. This is not a fixable artifact; it is intrinsic to a log-ratio measure, and any certified propagation must either assume p_min bounded away from 0 or accept the degradation. I state it rather than hide it.

**Link 2 — φ_d → Σφ_d.** Linear accumulation: `|Σφ̂_d − Σφ_d| ≤ (#distinctions)·δ ≤ (2^N−1)·δ`. **Caveat [N]:** this holds only if the *support* is stable — which mechanisms have φ_d > 0. An approximation can push a near-threshold φ_d across zero, adding or dropping a whole distinction (and its congruence membership, Eq. 48 [L]). Bounded per-distinction error does **not** imply bounded Σφ_d error near the existence threshold. The bound is `(2^N−1)·δ` *plus* the mass of any threshold-crossing distinctions.

**Link 3 — φ_d → Σφ_r, and why perturbation fails here.** φ_r = |overlap| · min over relata of (φ_d/|purview|) (Eq. 55 [L]). The min is 1-Lipschitz, so each φ_r's error is controlled by its relata's φ_d errors. But Σφ_r sums over up to **2^(2^N)−1** relations, so linear accumulation is catastrophic — a per-relation error of δ gives a Σφ_r error bound of order 2^(2^N)·δ, which is vacuous. **This is the pivot of the whole analysis [N]:** perturbation propagation governs the mechanism level and Σφ_d, but it *cannot* govern Σφ_r by sheer count. Σφ_r must be handled by a **bracket**, not by error propagation. The two halves of Deliverable B meet exactly here.

**Link 4 — Σφ → Φ.** For 4.0, Φ = Σφ_d + Σφ_r exactly, so `Φ-error = Σφ_d-error + Σφ_r-error` — trivial once the two sums are bracketed. (For 3.0 this link does not exist; see §7.)

### 6.2 The two-sided certified bracket on Φ (4.0)

**The construction [N].** Distinctions are the *cheap* part — there are only 2^N−1 of them and they are enumerated exactly. Relations are the *explosive* part and the only real uncertainty. So bracket Φ by bracketing Σφ_r:

- **Lower bound — from nonnegativity + partial enumeration.** Every φ_r ≥ 0 (it is a purview size times a nonnegative density, Eq. 55 [L]). Therefore *any* partial sum of computed relations is a valid lower bound: with distinctions exact,
  **L = Σφ_d + Σ_{r ∈ R_computed} φ_r ≤ Φ.**
  At minimum (no relations computed) L = Σφ_d. This is the missing lower side `bounds.py` does not provide.

- **Upper bound — Zaeemzadeh, tightened by the empirical profile.** The certified worst-case ceiling is the Eq. 16 growth bound (`sum_phi_relations_upper_bound(n, "GENERAL")` [C]). But once the exact distinctions are in hand, the per-unit-state profile S(o) = Σ_{distinctions incident to o} φ_d/|z*_c ∪ z*_e| and the incidence count |Z(o)| are **known exactly**, and Zaeemzadeh Eq. 15 evaluated on the *empirical* profile is still a valid upper bound (Eq. 11 is an exact rewriting; Eq. 12–14 upper-bound each o-term given its true S(o) [L]):
  **U = Σφ_d + [ Σφ_d + Σ_o LP-max(S(o), |Z(o)|) ]**, LP-max from Eq. 14.

**Verified [N].** On `grid3_system` (n=3, 7 distinctions): computed Σφ_d = 2.721709, Σφ_r = 3.776862, Φ = 6.498571.
- The empirical Eq. 15 bound gives **Σφ_r ≤ 10.088208** — holds (3.78 ≤ 10.09), and is **126× tighter** than the `GENERAL` worst-case ceiling (1270.285714).
- The bracket **[Σφ_d, Σφ_d + empirical-Eq15] = [2.7217, 12.8099]** contains Φ = 6.4986. ✓
- On `pqr_system` the empirical bound is **1016× tighter** than worst-case; bracket [1.0000, 2.2500] contains Φ = 1.0.
- On `basic_system` the worst-case bracket [1.0, 1271.29] contains Φ = 1.0 but is astronomically loose — the honest tightness characterization of the *unaided* `GENERAL` bound.

**What this contributes over `bounds.py` today [N].**
1. The **lower side** (nonnegativity + partial enumeration) is new — `bounds.py` is strictly one-sided.
2. The **empirical-profile upper bound** (Eq. 15 on the measured S(o), |Z(o)|) is ~100–1000× tighter than the shipped worst-case `GENERAL` bound on these fixtures, and it is *still certified*: bounding S(o) by its true value rather than by the n·2^(n-1) ceiling removes looseness without removing the certificate. `bounds.py` marks its Bound II/III non-certified only because they assume *extremal* purview profiles; the *measured* profile is exact, so Eq. 15 on it is certifiable. **This is the concrete additional structure the bracket needs**, and it is a small, wireable addition to `bounds.py` (compute S(o)/|Z(o)| from the exact `Distinctions`, feed Eq. 14/15). One caveat: keying o by unit index rather than unit-*state* can only *over*-count |Z(o)|, which makes the bound larger (still valid, slightly looser than ideal); a production version should key on state-tagged units exactly as the paper does. **[Correction: the follow-up certification found index-keying is not merely looser but *unsound* — a witnessed case where the index-keyed bound falls below the true Σφ_r; state-keying is mandatory, not just tighter. See `experiments/so_certificate_experiments/FINDINGS.md`.]**

### 6.3 Where a bound is not derivable, precisely

- **No certified upper bracket exists for IIT 3.0 Φ [N].** Zaeemzadeh's entire machinery is 4.0-specific — it bounds informativeness under causal marginalization and conditional independence, and `bounds.py` hard-excludes non-GID measures and IIT 3.0 by `ValueError` (`:169-218` [C]). The 3.0 Φ is a transport distance (extended EMD) over concept space, min over cuts; its error composes through the EMD's 1-Lipschitz property in each concept's repertoire *coupled by the transport plan*, not by a sum. So the bracket keystone is a **4.0-only** result. Stating a 4.0-style two-sided bracket for 3.0 would require a transport-level bound that neither the paper nor the code provides.

- **The lower bracket is only as good as the relations you enumerate [N].** L = Σφ_d is free but weak; tightening it means computing relations, which is the expensive thing the bracket was meant to avoid. The bracket is therefore most useful as an *anytime* guarantee: enumerate relations in decreasing-φ_r order (top-K), and L rises toward Φ while U stays fixed, giving a shrinking certified interval at each step. Building the decreasing-order enumeration is the "top-K relations" relaxation of §5 — unbuilt, but the natural companion to this bracket.

---

## 7. Where the unification is thin, and what remains open

**Confidence summary per axis** (the task's explicit ask):

| Axis | Genuinely free & orthogonal? | Confidence |
|---|---|---|
| 5 Difference measure | Free as a code knob; **coupled** to the postulate set; **forks** (metric-on-states vs intrinsic) | High |
| 6 Mechanism partition family | Genuinely free (registered strategy) | High |
| 7 Mechanism integration rule (min) | **NOT free — shared invariant** | High |
| 4 Background / marginalization | Free in the papers; **collapsed in the code** (all use 4.0 extended background) | High |
| 8 MIP normalization | Free; secondary; finer in code than in papers | Medium |
| 2 Alphabet (k-ary) | Free, mostly built | High |
| 3 State aggregation (single vs averaged) | Free but **unengaged** — all formalisms single-state | High |
| 1 Object (classical/quantum) | **NOT a free coordinate — a coupled package** (forces Axes 4,5,factorization) | High |
| 11 Top-level integration | **The fork, not an axis** | High |

**The irreducible common core [N].** Across all five formalisms, exactly these are shared: (i) do()-intervention causal marginalization; (ii) product-factorized, conditionally-independent repertoires; (iii) state-specificity; (iv) exclusion by max-over-purviews; (v) composition over the powerset of mechanisms; (vi) mechanism irreducibility as **min over a partition family** of a difference measure between intact and partitioned repertoire. This core is genuine and machine-verifiable — the mechanism-level recipe is one object with four free axes plugged in.

**The fork [N].** Above the mechanism level, the shared structure ends:

1. **3.0's Φ is a minimum of a transport distance; 4.0's Φ is a sum of irreducibilities.** These are different operations on different objects. Oizumi 2014 Eq. 11 minimizes an extended EMD over unidirectional cuts; Albantakis 2023 Eq. 59 sums φ over the distinction-and-relation structure and explicitly says it is "not computed based on a partition." No single parameter interpolates a min-of-EMD and a sum-of-φ.

2. **4.0 splits "the whole" into two objects that 3.0 conflates.** 4.0 has *both* a system irreducibility φ_s (min over directed partitions, the thing that says whether the system is one system) *and* a structure content Φ = Σφ_d + Σφ_r (the thing that says how much the system specifies about itself). 3.0's single Φ plays both roles at once. A meta-theory that wants one "Φ" symbol has to choose which 4.0 object it maps to 3.0's Φ, and neither choice is clean: φ_s matches 3.0's "min over cuts" *shape* but not its *content*; Φ matches 3.0's "content of the constellation" *role* but not its min-over-partitions *shape*.

3. **The measure fork (Axis 5) and the background collapse (Axis 4) compound the split.** 3.0's EMD needs a ground metric that 4.0's intrinsic measures reject on axiomatic grounds (Barbosa Theorem 1), and PyPhi's code has already chosen 4.0 cause-side backgrounds for its 3.0 — so for proper-subset systems (the only place backgrounds exist), the running "3.0" is not the paper's 3.0, nor legacy 1.x's (verified Φ_3.0 divergence in §2, Axis 4; full-substrate analyses are unaffected).

**The honest conclusion.** A single parameterization captures the **mechanism level** of all five formalisms cleanly — that is a real, non-trivial unification, and the design matrix's mechanism rows (Axes 1–9) are its coordinate system. It does **not** capture the **system level** of 3.0 and 4.0 without a disjunction: `Φ = (3.0) min-over-cuts of extended-EMD  OR  (4.0) Σφ_d + Σφ_r`. This is a genuine structural divergence in the definitions, not an artifact of how I drew the axes, and forcing it into one knob would be the false unification the task rightly warns against. The right shape for a meta-theory is: **one shared mechanism kernel, two top-level integration rules selected by formalism.** That is exactly the shape PyPhi's architecture already has — a shared `core/repertoire_algebra.py` kernel under a `PhiFormalism` Protocol whose `evaluate_system` differs per formalism (`base.py:43-90` [C]) — which is mild but real evidence that the thin-unification verdict is the correct one: the codebase converged on it independently.

### Open problems this analysis exposes

1. **Wire the tighter certified bracket** [N, actionable]. Add the empirical-profile Eq. 15 upper bound and the nonnegativity lower bound to `bounds.py`, giving a live two-sided certified interval on 4.0 Φ (~100–1000× tighter than the current worst-case ceiling on tested fixtures). Small, self-contained, verified above. Pair it with decreasing-φ_r (top-K) relation enumeration for an anytime shrinking interval.
2. **A transport-level error bound for 3.0 Φ** [N, open]. No certified bracket exists for the 3.0 constellation distance; deriving one needs a bound on the extended EMD under repertoire perturbation coupled through the transport plan. Neither paper nor code provides it.
3. **Certify the empirical S(o) bound** [N, small]. Prove that Eq. 15 evaluated on the measured (state-tagged) per-unit profile is a valid certificate (it is, by the exactness of Eq. 11 given the true profile), and key o on unit-states not unit indices to reach the ideal tightness.
4. **The p_min degradation** [N, open]. The repertoire→φ_d Lipschitz constant blows up for near-deterministic repertoires. Any production error certificate must either assume p_min bounded away from zero or switch to a different (e.g. additive-smoothing) analysis there.
5. **The quantum system Φ** [L/N, large]. The quantum paper stops at mechanism φ; quantum distinctions, relations, and φ_s are coherent-but-unbuilt cells (ROADMAP N21). Building them would test whether the mechanism-level unification extends to the quantum object or forks again.

---

## Appendix A — Verification scripts and outputs

Scripts: `2026-07-07-formalism-meta-theory-verification/{verify.py,verify2.py}` (alongside this spec). Key results, all against the live library at `config.numerics.precision = 13`:

```
CHECK 1  effect-side ID == GID (measure-axis coincidence), machine precision:
  mech=(1,) purv=(0,):      ID=0.5000000000 GID=0.5000000000 eq=True
  mech=(2,) purv=(1,):      ID=1.0000000000 GID=1.0000000000 eq=True
  mech=(1,2) purv=(0,1):    ID=3.0000000000 GID=3.0000000000 eq=True
  (all non-degenerate (mech,purv) of basic_system: eq=True)

CHECK 2  2026 intrinsic-differentiation cap (basic_system is deterministic):
  phi_s  2023 = 0.4150374993   2026 = 0.0000000000   (CAP BINDS; 2026 <= 2023)

CHECK 3  Zaeemzadeh bounds hold + worst-case bracket (basic_system):
  Sum phi_d = 1.0  Sum phi_r = 0.0  Phi = 1.0
  bracket [Sum phi_d, Sum phi_d + UB_general(rel)] = [1.0, 1271.2857]  contains Phi=1.0

CHECK (tighter bracket, grid3_system, n=3, 7 distinctions):
  Sum phi_d=2.721709  Sum phi_r=3.776862  Phi=6.498571
  empirical Eq.15 bound on Sum phi_r = 10.088208  (holds; 126x tighter than 1270.29)
  bracket [2.7217, 12.8099] contains Phi=6.4986

CHECK 4  selection operators are 1-Lipschitz (no error amplification):
  min: |min a - min b| = 0.002092 <= max|a-b| = 0.014934   True
  max: |max a - max b| = 0.002202 <= max|a-b| = 0.014934   True

CHECK Link-1  repertoire -> phi_e Lipschitz bound |dphi| <= sel/(ln2*p_min)*delta:
  8/8 random draws hold; degrades as p_min -> 0 (L: 2.31 at part=0.5 -> 230.83 at part=0.005)

CHECK 5  Actual Causation alpha == PMI of intact vs partitioned:
  PMI(0.8,0.5) = 0.6780719051 == log2(0.8/0.5)   True

CHECK 6  background-convention divergence (verify_background.py; spec Axis 4):
  proper-subset system {A,B} of ABC, W={C}, u=(1,0,0):
    cause_repertoire mech={A} purv={B}:
      library = (0.405660, 0.594340) == manual IIT 4.0 Eq. 4  (machine precision)
      legacy current-state conditioning = (0.1, 0.9)           (differs)
    effect side: library == current-state conditioning         (invariant)
    end-to-end, presets.iit3 (EMD, directed bipartition):
      Phi_3.0 = 0.4160700000 (2.0 semantics) vs 0.7200000000 (legacy semantics)
  full-substrate fixtures: basic_system, xor_system external_indices = ()
    -> all conventions coincide; 2014 worked numbers and SIA goldens unaffected
  genealogy: pre-refactor baseline (b3aaa3e5) reproduces (0.405660, 0.594340)
    exactly -> 2.0 == its golden baseline; the divergence from published 1.x
    was made on the feature branch (backward_tpm: 2023-06-08 opt-in fb668a3e/
    3317ab8c; 2024-06-07 structural 380e9b9a/9c1eb5e2)
```
