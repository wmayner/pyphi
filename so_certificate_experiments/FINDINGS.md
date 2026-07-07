# The empirical S(o) upper bound on Σφ_r: **proved certified** (state-keyed)

**Verdict (2026-07-08): Zaeemzadeh Eq. 15 evaluated on the measured,
state-tagged per-unit profile is a valid certified upper bound on PyPhi's
Σφ_r — proved below, with the paper's per-o maximization reduced to
Chebyshev's sum inequality. Two corrections to the meta-theory spec: the
index-keyed variant its verification script used is *not* provably valid
(its conservativity argument fails; state-keying is mandatory, not merely
tighter), and the self-relation term can be computed exactly instead of
bounded by Σφ_d.**

## Setting

PyPhi's relation integrated information is exactly the paper's Eq. 8/9:
`Relation.phi = |congruent overlap| · min_d(φ_d / |purview_union_d|)`, where
`purview_union` is a set of **UnitState pairs** and the candidate family
enforces congruent overlap at generation (`pyphi/relations.py` —
`_combinations_with_nonempty_congruent_overlap`; self-relations via the
cause∩effect unit-state intersection).

Notation: o ranges over UnitState pairs; 𝒵(o) = distinctions whose
`purview_union` contains o; q_d = φ_d / |purview_union_d| (the density);
S(o) = Σ_{d ∈ 𝒵(o)} q_d; g(k) = (2^k − 1 − k)/k.

## Theorem

For any resolved distinction set D with concrete relations as defined above,

  Σφ_r  =  [Σ_d |z*_c(d) ∩ z*_e(d)| · q_d]  +  Σ_o Σ_{i=1}^{|𝒵(o)|} q_(i) (2^{|𝒵(o)|−i} − 1)   (identity)
  Σφ_r  ≤  [Σ_d |z*_c(d) ∩ z*_e(d)| · q_d]  +  Σ_o S(o) · g(|𝒵(o)|)                             (certificate)

with q_(i) sorted ascending within each o. Both terms are computable from the
distinctions alone in O(|D| · n) — no relation enumeration.

## Proof

**Identity.** A combination **d** with |**d**| ≥ 2 contributes
φ_r(**d**) = |∩_d purview_union| · min q = Σ_{o ∈ ∩} min_{d ∈ **d**} q_d, and
**d** has o in its congruent overlap iff **d** ⊆ 𝒵(o) (congruence is exactly
membership of the same UnitState in every member's union). So

  Σ_{|**d**|≥2} φ_r = Σ_o Σ_{**d** ⊆ 𝒵(o), |**d**|≥2} min q.

For fixed o with densities sorted ascending, q_(i) is the minimum of exactly
the subsets containing (i) plus any nonempty subset of the |𝒵(o)| − i larger
elements: 2^{|𝒵(o)|−i} − 1 of them. Self-relations (|**d**| = 1) are each
exactly |z*_c ∩ z*_e| · q_d by Eq. 9. ∎ *(Numerically confirmed exact to
< 4e-15 across every record — see Reproduction.)*

**Certificate.** The true per-o term is Σ w_i q_(i) with weights
w_i = 2^{k−i} − 1 **descending** while q_(i) is **ascending**. By Chebyshev's
sum inequality, Σ w_i q_(i) ≤ (1/k)(Σ w_i)(Σ q_(i)) = S(o) · g(k), which is
the paper's Eq. 14 value — attained at the uniform profile. Summing over o
gives the certificate; the self term is carried exactly, not bounded. ∎

The bound depends only on the *measured* (S(o), |𝒵(o)|) — no extremal-profile
assumption — which is why it is certified where `bounds.py`'s Bound II/III
are not, and ~100–1000× tighter than the GENERAL worst-case ceiling
(fixture numbers in the results JSON; on `grid3` the state-keyed bound is
≈ 9.94 vs ceiling 1270.29 vs true 3.78; on `pqr` the CES has Σφ_r = 0 and the
bound **collapses to exact**).

## Correction 1 — index-keying is not certifiable

The meta-theory verification script (`verify2.py`) keyed o on unit *indices*
with index-count denominators, under the caveat that this "can only
over-count |𝒵(o)|, making the bound larger (still valid)." **That argument is
false.** When a distinction's purview_union contains the *same unit in both
states* (cause specifies u=0, effect u=1), state-keying contributes its
density to two groups while index-keying contributes once with a merged
denominator — so the merged S can *decrease*, and the index-keyed bound can
fall below the state-keyed bound (9 of 801 records in the main run).
**The failure is not merely a dead proof: the index-keyed bound is
unsound** — on one witnessed record (n=3, state (0,0,0), 2 distinctions,
seed 20260708 trial 32) it falls below the true value itself:
index bound 0.116418 < true Σφ_r 0.122629 ≤ state bound 0.143970.
**Production implementations must key on UnitState pairs — exactly the
paper's objects — which is what PyPhi's `purview_union` already provides.**

## Correction 2 — the self-relation term is free

Each self-relation's φ_r is determined by its own distinction
(|z*_c ∩ z*_e| · φ_d / |z*_c ∪ z*_e|), so the certificate can carry the exact
self sum instead of the Σφ_d ceiling the spec used — a strict tightening at
zero cost.

## Consequences

- **The Wave-7c certified-bracket gate is discharged.** The anytime bracket
  is now fully proved: lower endpoint Σφ_d + exact self sum + any partial
  relation sum (φ_r ≥ 0 termwise); upper endpoint the state-keyed empirical
  Eq. 15. Paired with best-first descending-φ_r enumeration (the relations
  design's Tier 2), the interval shrinks monotonically and is certified at
  every step. Ready to wire into `pyphi/formalism/iit4/bounds.py`. Measured
  tightness of the upper endpoint on Σφ_r: median 1.45×, min 1.000× (exact),
  max 41× over 759 nonzero records — versus the ~100–1000× gap of the
  worst-case ceiling.
- **Scope correction (the identity supersedes the bracket for the total).**
  Because Eq. 11 is exact, a user with a complete distinction set gets Σφ_r —
  and hence Φ — *exactly* in O(|D|·n) with no relation enumeration; this is
  the existing `AnalyticalRelations` sum, and the certificate is strictly
  weaker there. The bound's genuine scope is the **partial-information
  case**: an incomplete or deliberately truncated distinction set, where
  measured S(o) contributions from computed distinctions can be combined
  with certified caps for uncomputed ones (via the paper's mechanism-level
  φ ≤ |M||Z| bounds) to give a certified Φ range during or without a full
  CES computation. That extension is future work; the small measured LP
  looseness (median 1.45×) indicates it can stay usefully tight.
- The identity itself is worth exposing regardless: it reconstructs Σφ_r
  from the distinctions in O(|D|·n) — it *is* the S3 analytical sum, now
  verified unconditionally on adversarial random systems rather than only
  on structured fixtures — and it shares the state-keyed incidence
  structure the relations query design builds anyway.

## Reproduction

`verify_so_certificate.py` — per (substrate, state): exact concrete-relations
Σφ_r vs the identity reconstruction (residual), the state-keyed and
index-keyed bounds, bound and dominance checks; fixtures pqr/grid3/residue
plus seeded random substrates (n = 2–4, `np.random.default_rng`); raw records
with full TPMs in the seeded results JSON (never overwritten). Runs: seed 555
smoke (45 records); seed 20260708 main run — **801 records, identity exact
everywhere (max residual 1.7e-11), 0 state-keyed bound violations, 9
index-dominance violations of which 1 is genuinely unsound (the witness
above)**.
