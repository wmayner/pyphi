# Key IIT equations, with citations

Equation numbers below are cited to specific papers. Where a paper numbers only
some of its equations, that is noted — cite unnumbered formulas by name, not by
a number. When in doubt, verify against the paper itself. Never cite a number
from memory.

Symbols: `s`, `s̄` are system states (current, and a candidate cause/effect
state); `m` a mechanism state; `z` a purview state; `𝒯_e`, `𝒯_c` the effect and
cause transition probability matrices; φ integrated information; ii intrinsic
information.

## IIT

IIT 4.0 is stated in two papers, cited together:

- Albantakis et al. (2023), *Integrated information theory (IIT) 4.0*, PLoS
  Comput Biol 19(10): e1011465. The published version numbers its equations
  1–61. The arXiv preprint numbers them differently, so do not cite preprint
  numbers.
- Mayner, Marshall & Tononi (2026), *Intrinsic cause–effect power: the
  tradeoff between differentiation and specification*, Entropy 28(4): 410.
  It defines the system's intrinsic information, which bounds φₛ.

### Albantakis et al. (2023)

**Substrate and conditional independence**
- Eq. 1 — the transition probability matrix: `𝒯_U ≡ p(ū | u)`.
- Eq. 2 — conditional independence: `p(ū | u) = ∏_i p(ū_i | u)`.

**Intrinsic perspective: cause and effect TPMs (causal marginalization)**
- Eq. 3 — effect TPM, background fixed in the current state: `𝒯_e = p_e(s̄ | s)`.
- Eq. 4 — cause TPM, background causally marginalized over its past states,
  weighted by their probability given the current state of the universe (a
  uniform prior updated by conditioning on u). Not a uniform average: the
  weights are in general neither uniform nor deterministic. This is the
  `CAUSAL_MARGINALIZATION` background scheme.

**Intrinsic information (informativeness × selectivity)**
- Eq. 5 — intrinsic effect information: `ii_e(s, s̄) = p_e(s̄ | s) · log(p_e(s̄ | s) / p_e(s̄))`.
- Eq. 6 — the unconstrained (chance) effect probability `p_e(s̄)`.
- Eq. 7 — intrinsic cause information `ii_c(s, s̄)` (uses a backward,
  Bayes-derived selectivity term).
- Eqs. 8–9 — the unconstrained cause probability and the Bayes inversion.
- Eqs. 10–11 — informativeness is the log term; a system has cause–effect power
  when it raises a state's probability above chance.
- Eqs. 12–13 — the **maximal cause–effect state** is `argmax` of ii over
  candidate states, and the system's intrinsic information is ii at that state.

**Integration: φₛ over the minimum partition**
- Eqs. 14–16 — **directional system partitions** Θ(S): each part has its
  inputs, outputs, or both cut.
- Eqs. 17–18 — the partitioned TPMs (cut connections replaced by independent
  noise).
- Eqs. 19–20 — integrated effect and cause information `φ_e`, `φ_c` (the
  positive part of an intrinsic-difference term against the *partitioned*
  repertoire).
- Eq. 21 — for a partition, `φₛ(θ) = min(φ_c(θ), φ_e(θ))`.
- Eq. 22 — the integration at the minimum partition θ′, `φₛ(θ′)`. φₛ is the
  minimum of this and the system's intrinsic information ii(s) (Mayner et al.
  2026, Eq. 23; below).
- Eq. 23 — the **MIP**: the partition minimizing φₛ *normalized* by the maximum
  possible value for that partition (its number of cut connections). The
  normalization makes the MIP find the system's fault line.

**Exclusion: complexes**
- Eqs. 24–26 — the recursive search for maximal substrates (complexes): the set
  with maximal φₛ is a complex; its units are removed; the search recurses.

**Composition: distinctions**
- Eq. 27 — a distinction `d(m) = (m, z*, φ_d)`.
- Eqs. 28–33 — mechanism purview probabilities: the system units outside the
  mechanism (X = S \ M) are causally marginalized with a uniform distribution,
  and the per-unit probabilities are combined as products.
- Eqs. 34–35 — mechanism intrinsic information `ii_e(m, z)`, `ii_c(m, z)`.
- Eqs. 36–37 — the maximal purview state and mechanism intrinsic information.
- Eq. 38 — the **disintegrating partitions** Θ(M, Z) of a mechanism–purview
  pair (the empty set is a permitted part).
- Eqs. 39–44 — partitioned probabilities and `φ_e(m, Z)`, `φ_c(m, Z)`.
- Eqs. 45–46 — the maximally irreducible purview (exclusion).
- Eq. 47 — `φ_d(m) = min(φ_c(m), φ_e(m))`.
- Eq. 48 — the set of distinctions congruent with the system's cause–effect
  state.

**Composition: relations**
- Eqs. 49–52 — a relation, its **faces**, and their overlaps.
- Eqs. 53–55 — relation integrated information `φ_r` (φ_d spread over unique
  purview units, times the joint-overlap size, minimized over the relation's
  distinctions).

**Cause–effect structures and Φ**
- Eq. 56 — `R(D)`, the set of all relations (φ_r > 0) among a set of
  distinctions D.
- Eq. 57 — a cause–effect structure `C(D) = D ∪ R(D)`.
- Eq. 58 — the **Φ-structure**: the cause–effect structure specified by a
  complex.
- Eq. 59 — the structure integrated information `Φ = Σ φ`, the plain sum over
  all distinctions and relations of the Φ-structure.

### Mayner, Marshall & Tononi (2026)

Intrinsic information has two components, differentiation and specification.

- Eq. 3 — the intrinsic difference `ID(p, q) = maxₛ p(s) · log(p(s) / q(s))`.
- Eqs. 4–6 — **intrinsic differentiation** `i_diff = −log p(s′ | s)`: does the
  system provide itself a repertoire of alternatives?
- Eqs. 7–11 — **intrinsic specification** `i_spec` (called intrinsic
  information in Albantakis et al. 2023).
- Eq. 13 — the system intrinsic information `ii(s) = min{ii_c(s), ii_e(s)}`.
  The per-direction value `ii_c/e = min(i_diff, i_spec)` is stated just before
  Eq. 13 and is not numbered.
- Eq. 23 — `φₛ = min(φ_c, φ_e, ii(s))`, the intrinsic-information
  requirement. A deterministic system has i_diff = 0, so its φₛ is 0.

## The intrinsic difference measure

Barbosa et al. (2020), *A measure for intrinsic information*, Sci Rep 10: 18803.
**Only Eq. 1 is numbered** (the causality/specificity/intrinsicality properties
are unnumbered — cite them by name).

- Eq. 1 — `ID(P, Q) = maxₐ pₐ · log(pₐ / qₐ)`. A **max** over states, not a sum;
  this is the difference from KL divergence.

Barbosa et al. (2021), *Mechanism Integrated Information*, Entropy 23: 362.
Eqs. 1–9 plus Theorem 1.
- Eq. 3 — `φ(m) = min(φ_c(m), φ_e(m))`.
- Eq. 8 — specificity **with an absolute value**: the specified state can be one
  whose probability the mechanism *decreases*.

## System integrated information (φₛ)

Marshall et al. (2023), *System Integrated Information*, Entropy 25: 334.
**Only Eqs. 1–2 and "Theorem 1" are numbered**; cite the φₛ / MIP formulas by
section or name.
- Eq. 1 — the system transition function with background conditioning. It
  holds the background at its current state for both causes and effects, the
  pre-4.0 convention that IIT 4.0 Eqs. 3–4 replace.
- Theorem 1 — the maximum φₛ for a partition equals the number of connections
  it cuts.

## Macro units and grain

Marshall et al. (2026), *Intrinsic units: identifying a system's causal
grain*, Neurosci. Conscious. 2026(1): niag013 (PMC13082400,
doi:10.1093/nc/niag013). Eqs. 1–42 are all numbered; the numbers below are
from the published version. The bioRxiv preprint (2024.04.12.589163) numbers
Eqs. 3–20 differently, so do not cite preprint numbers.

- Eqs. 3–8 — background conditions at the micro grain: `q_e` (Eq. 3), the
  Bayes posterior `q_c` (Eqs. 4–6), the cause TPM `𝒯_c` (Eq. 7), the effect
  TPM `𝒯_e` (Eq. 8).
- Eq. 9 — a complex at a single grain; Eq. 20 — a complex across grains
  (overlap judged on micro constituents).
- Eq. 12 — a unit `J = (U^J, V^J, τ′_J, g′_J, W^J)`; Eq. 13 — constituents'
  background apportionments nest within `W^J`; Eq. 14 — the mapping `g′_J`
  from constituent sequences; Eq. 15 — the composed mapping `g_J` from micro
  sequences. The text after Eq. 14 counts `2^(2^(τ′|V|)) − 2` possible
  mappings.
- Eqs. 16–17 — the intrinsic-unit criteria: `φₛ(v^J) > 0`, and `φₛ(v^J)`
  exceeds every competitor in `f(U^J, W^J, τ_J)`.
- Eq. 19 — units of a system have disjoint micro constituents and
  apportionments.
- Eqs. 26–40 — the four-step macro-TPM construction: discount extrinsic
  connections (Eqs. 26–30), extend to update sequences (Eq. 31), causally
  marginalize the background (Eqs. 32–34), compress into macro states
  (Eqs. 35–40). Eqs. 41–42 — the macro cause and effect TPMs.

## Actual causation

Albantakis et al. (2019), *What Caused What?*, Entropy 21: 459. Eqs. 1–17,
Definitions 1–4.
- Eqs. 11–12 — cause and effect information as **pointwise mutual information**
  (`log₂(p/q)`), the `alpha_measure="PMI"`.
- Eqs. 15–16 — integrated cause/effect information α over the MIP.
- Definitions 1–2 — the actual cause and actual effect (the occurrence
  maximizing α, with a minimality condition).

## Analytical relations

Albantakis et al. (2023), IIT 4.0 S3 Text (only Eq. 1 numbered). Gives the
closed-form sum Σφ_r and the relation count directly from the distinction set,
without enumerating relations — the basis of PyPhi's `analytical_relations`.
