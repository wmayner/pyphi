# Querying the relational structure without materializing it — design exploration

**Status: exploration / not scheduled.** This document works out what it would
take for PyPhi to answer questions about a system's relational structure at
scales where listing every relation is hopeless. It is grounded in the IIT 4.0
paper and its analytical-relations supplement, in the relation code as it
stands after the 2.0 refactoring, and in worked demonstrations run against the
actual library (every number below was produced by the demonstration code in
the appendix and checked against concrete enumeration wherever enumeration is
feasible). Throughout, claims are
labeled as **[exists]** (already in PyPhi), **[established]** (known method
elsewhere, adapted here), or **[proposed]** (worked out in this document and
verified numerically, but — to the extent the papers and code have been
searched — not previously written down; the mathematical claims should be
read as carefully checked derivation-plus-numerics, not as published
results).

## 1. The problem

A cause-effect structure comprises distinctions and the relations among them
(Albantakis et al. 2023, Eq. 57). Relations are indexed by subsets of the
distinction set: every subset whose purviews overlap congruently is a relation
(Eq. 56). With `|D|` distinctions there can be up to `2^|D| − 1` relations,
and `|D|` itself can reach `2^n − 1` for `n` units — the paper's own worked
example (Fig. 6D, a 6-unit specialized lattice) has 27 distinctions and
1,537,080 relations; its Φ is dominated by `Σφ_r = 11445.76`, three orders of
magnitude larger than `Σφ_d`.

PyPhi currently offers two modes, selected by
`config.formalism.iit.relation_computation`:

- **`CONCRETE`** — enumerate every relation as a `Relation` object
  (`pyphi/relations.py::all_relations`, a lazy DFS over
  `combinatorics.combinations_with_nonempty_intersection` that prunes when the
  running purview intersection empties). Exact and fully general, but the
  output itself is exponential in `|D|`: at 27 distinctions it costs about a
  minute and gigabytes of resident memory (measured in §5.3), and every ~10
  additional distinctions multiply both by ~1000.
- **`ANALYTICAL`** — `AnalyticalRelations`, which stores only the distinction
  set and answers exactly three scalar queries in closed form: `sum_phi()`,
  `apportioned_sum_phi()`, and `num_relations()` (the S3-supplement results).
  It is deliberately not iterable; anything that walks relations raises
  `TypeError`.

Between "everything, explicitly" and "three scalars" there is currently
nothing. The question explored here: how much of the space between can be
recovered — exactly, lazily, or statistically — without ever materializing the
relation set, and where does that program stop?

The answer worked out below: nearly all of it. The relational structure has a
compact generating representation (§2) from which a much richer family of
queries than the current three scalars is answerable exactly in low-order
polynomial time (§3.1), the rest is answerable either output-sensitively
(§3.2) or by unbiased sampling with known error (§3.3), and the residue that
resists all three is small and identifiable (§3.4).

## 2. The relational structure is a view, not a container

### 2.1 What a relation actually depends on

**[exists]** The S3 supplement proves (its p. 2) that the union of a
relation's face overlaps equals the intersection of its relata's
purview-unions, which removes faces from the φ_r formula entirely. In the
notation used from here on: write, for each distinction `d`,

- `u(d)` — the **purview-union**: the set of *state-tagged* units
  `z*_c(d) ∪ z*_e(d)` (a unit in its specified state; a unit specified
  incongruently on the cause and effect sides contributes two distinct
  atoms). In the code this is `Distinction.purview_union`, a set of
  `UnitState` objects; state-congruence is simply set intersection on
  `UnitState`s.
- `q(d) = φ_d / |u(d)|` — the distinction's φ **density** per unique purview
  unit (Eq. 53).

Then for `S ⊆ D`, `|S| ≥ 2`:

```
O(S)  =  ⋂_{d∈S} u(d)                 (the congruent overlap)
S is a relation  ⇔  O(S) ≠ ∅
φ_r(S)  =  |O(S)| · min_{d∈S} q(d)     (Eq. 55, via the S3 identity)
```

Self-relations (`|S| = 1`) exist for each `d` with
`z*_c(d) ∩ z*_e(d) ≠ ∅` and carry `φ_r = |z*_c ∩ z*_e| · q(d)`; there are at
most `|D|` of them, so they never pose a scaling problem and are handled by
direct summation everywhere below.

So the entire relational structure — membership, φ values, and (as shown in
§3.1.6, even the faces in aggregate) — is a deterministic function of the
reduced summary

```
𝒵 = { (u(d), q(d)) : d ∈ D }
```

which is linear in the size of the distinction set. The relation set is not
information; it is a view. PyPhi's serializer already takes this position
implicitly: `AnalyticalRelationsRefSchema` stores nothing and reconstructs
the object from the distinction table on load. The design conclusion is to
take the same position at the *query* layer: the right object is not a
container of relations but a query interface over `𝒵`.

### 2.2 The generating representation: a union of at most `2n` simplices

**[proposed — but elementary]** Index the distinct state-tagged units
("atoms") appearing in any purview-union: `𝒰 = ⋃_d u(d)`, with
`|𝒰| ≤ 2n` for `n` binary units (one cause-side and one effect-side state
per unit; `k`-ary units give at most `k·n` atoms). For each atom `a` let

```
D_a = { d ∈ D : a ∈ u(d) }
```

Then `S` is a relation iff `S ⊆ D_a` for some atom `a` — i.e. **the relation
set is exactly the union of the powersets (size ≥ 2) of at most `2n` subsets
of `D`**. Equivalently, it is the nerve of the cover `{D_a}`: a simplicial
complex that is a union of at most `2n` complete simplices. The bipartite
incidence between atoms and distinctions (`≤ 2n·|D|` bits) generates
everything, and moreover `|O(S)| = #{a : S ⊆ D_a}` is the *coverage count* of
`S` in this union.

This one picture organizes everything that follows:

- Sums over relations weighted by `|O(S)|` **decompose over atoms** and hit
  each relation exactly `|O(S)|` times — which is precisely the factor φ_r
  carries. This is why the S3 closed form works, and it is why many more
  quantities than `Σφ_r` are closed-form (§3.1).
- Sums *not* weighted by `|O(S)|` (e.g. the bare relation count) need
  inclusion–exclusion over atom subsets — which is why `num_relations` is the
  most expensive of the current three scalars.
- φ_r is **antitone along the subset lattice**: adding a relatum can only
  shrink both `O(S)` and `min q`. This yields best-first enumeration in
  descending φ order (§3.2).
- Sampling from a union of explicitly-sized sets with computable coverage
  counts is a solved problem (Karp–Luby; §3.3).

### 2.3 What this changes conceptually

"The relations" as currently conceived — a set of `Relation` objects — is the
wrong unit of design for large systems, in the same way that "the set of all
states" is the wrong unit of design for a TPM library. The proposal is to
treat `Relations` the way the codebase already treats probability
distributions: a mathematical object with an interface of queries, where
materialization (`ConcreteRelations`) is one possible backend — and past
`|D|` in the mid-30s, an impossible one (§5.3).

Two facts from the code inventory support this being a small step rather than
a rewrite:

- **[exists]** The only relation quantities any *pipeline* consumer needs are
  the three scalars: `CauseEffectStructure.big_phi`, folds (`PhiFold`), the
  matching module's differentiation `D` (computed by inclusion–exclusion over
  `AnalyticalRelations(∩ D_k).sum_phi()` — already a nontrivial
  no-materialization computation in production), `analyze`/`sweep` summary
  rows, and display cards. All of these already work with the non-iterable
  `AnalyticalRelations`.
- **[exists]** The only consumers that iterate relations are visualization
  (`pyphi/visualize/projection`) and `CauseEffectStructure.diff`. Both are
  inherently bounded by human legibility — no one reads a plot of 1.5M
  hyperedges — so a lazy top-K stream (§3.2) serves them better than full
  enumeration even at small scale.

## 3. The query taxonomy: exact, lazy, statistical, unreachable

This is the core of the design. Queries about the relational structure fall
into four tiers, and the tier boundaries are provable properties of the
formalism, not implementation accidents.

Everything in §3.1–§3.3 below was implemented as standalone demonstration
code against the current library and verified against concrete enumeration on
`xor` (4 distinctions / 15 relations), `grid3` (7/39), `fig4` (4/15), and
`rule110` (4/11), and then run at scale on the paper's Fig. 6D system (27
distinctions / 1,537,080 relations). §5 shows the demonstrations.

### 3.1 Tier 1 — exact, closed form, polynomial in `|D|` and `|𝒰|`

#### 3.1.1 The three existing scalars **[exists]**

`Σφ_r`, `Σφ_r/|r|`, and the relation count, per the S3 supplement:

```
Σ_{|S|≥2} φ_r(S) = Σ_{a∈𝒰} Σ_{S⊆D_a,|S|≥2} min_{d∈S} q(d)
```

with the inner sum computed by the sorted-coefficient trick
(`combinatorics.sum_of_minimum_among_subsets`: sort the `q` values ascending;
the `i`-th smallest is the minimum of exactly `2^(m−1−i) − 1` subsets). The
count uses inclusion–exclusion over atom subsets `o` with
`|𝒵(o)| = #{d : u(d) ⊇ o}`:

```
#relations = Σ_{o≠∅} (−1)^{|o|−1} (2^{|𝒵(o)|} − |𝒵(o)| − 1)
```

(currently implemented by walking `Distinctions.purview_inclusion(max_order=None)`,
which enumerates all subsets of each purview-union — see §6.3 for a cheaper
alternative over the intersection closure).

#### 3.1.2 Incident ("fold") sums **[exists]**

`AnalyticalFoldRelations(parent, seeds)` computes the sum/count/apportioned
sum over relations *incident to a seed set* `F` as `total(D) − total(D∖F)` —
a relation either touches `F` or it doesn't. This is exactly the paper's
distinction-Φ-fold ("the context of a distinction") and already tiles
`big_phi` across single-distinction folds in `PhiFold`.

#### 3.1.3 Moments of the φ_r distribution **[proposed]**

The second moment decomposes over *ordered pairs of atoms* exactly as the
first moment decomposes over atoms, because `|O(S)|² = Σ_{a,b∈O(S)} 1`:

```
Σ_S φ_r(S)² = Σ_S |O(S)|² (min q)² = Σ_{(a,b)} Σ_{S⊆D_a∩D_b, |S|≥2} (min_{d∈S} q(d))²
```

and since squaring preserves order on positive values, each inner sum is
`sum_of_minimum_among_subsets([q(d)² for d in D_a ∩ D_b])`. Cost:
`O(|𝒰|² · |D| log |D|)`. The `k`-th moment costs `O(|𝒰|^k)` inner sums via
`k`-tuples of atoms. With the count and the first two moments, the exact mean
and standard deviation of φ_r over all (say) 1.5M relations come out in
milliseconds — see §5.2. Verified exactly (float-exact agreement with
enumeration) on all four small systems.

#### 3.1.4 Degree-resolved counts and sums **[proposed]**

The degree of a relation (its number of relata, `h = |S|`) is a quantity the
paper's own figures report distributions over. Both the count and the φ sum
restrict exactly to fixed degree `k`:

- Sum: in the per-atom decomposition, restrict to subsets of size `k`. The
  `i`-th smallest `q` in `D_a` (with `a_i` values above it) is the minimum of
  `C(a_i, k−1)` subsets of size `k`, so
  `Σ_{|S|=k} φ_r = Σ_a Σ_i q_(i) · C(a_i, k−1)`.
- Count: replace `2^m − m − 1` with `C(m, k)` in the inclusion–exclusion
  count.

Verified exactly on all four small systems; at Fig. 6D scale this produces
the full exact degree spectrum of 1.5M relations (peak at degree 10; §5.2) in
about a millisecond per degree.

#### 3.1.5 The maximum φ_r, and why it is a pair **[proposed]**

*Claim: the maximum of φ_r over all relations of degree ≥ 2 is attained at
degree 2.* Proof: let `S` be any relation, `d* = argmin_{d∈S} q(d)`, and
`d'` any other member. Then `O({d*, d'}) = u(d*) ∩ u(d') ⊇ O(S)` and
`min(q(d*), q(d')) = q(d*) = min_S q`, so
`φ_r({d*, d'}) ≥ φ_r(S)`. ∎

So the global maximum is a scan over `C(|D|, 2)` pairs plus the `≤ |D|`
self-relations — `O(|D|² n)`. Verified against enumeration on all four small
systems. (The same argument gives a per-distinction version: the strongest
relation containing `d` is a pair or `d`'s self-relation.)

#### 3.1.6 Unit-pair "binding" marginals — the relational adjacency structure **[proposed]**

For phenomenological analyses (the grid/extendedness program: which units are
bound to which, and how strongly) the natural object is not any single
relation but the marginal

```
B(a, b) = Σ_{S : a,b ∈ O(S)} min_{d∈S} q(d)      (= Σ_{S : a,b ∈ O(S)} φ_r(S)/|O(S)|)
```

— the total apportioned relation strength jointly covering atoms `a` and `b`.
By the same pair decomposition as §3.1.3 this is one
`sum_of_minimum_among_subsets` call per atom pair: a `|𝒰| × |𝒰|` matrix,
exactly, never touching a relation. Its diagonal is the per-atom
decomposition of `Σφ_r/|O|`; summing the whole matrix with the right
weights recovers `Σφ_r`-type scalars. Verified exactly on all four small
systems. This matrix is the closed-form answer to "show me the relational
skeleton of the experience" and is the natural input for comparing the
relational structure of two states or two substrates without enumerating
either (§3.4.3).

The same construction with triples gives 3-way binding tensors, and so on up
— cost `O(|𝒰|^k)`, each entry one sorted dot product.

#### 3.1.7 The exact φ_r histogram **[proposed]**

φ_r takes at most `|𝒰| · |D|` distinct values (`j · t` for overlap size `j`
and min-density `t`), and the full exact histogram is computable by a
threshold sweep with Möbius inversion:

1. For each distinct density value `t` (at most `|D|` of them), let
   `D^{≥t} = {d : q(d) ≥ t}`.
2. Count relations *within* `D^{≥t}` by exact overlap size `j`: the family of
   achievable overlaps is the intersection closure of `{u(d)}`; for each
   closure element `P`, `g(P) = 2^{m_P} − m_P − 1` (with
   `m_P = #{d ∈ D^{≥t} : u(d) ⊇ P}`) counts relations with `O(S) ⊇ P`, and
   Möbius inversion down the closure gives the count with `O(S) = P` exactly.
   Summing over `|P| = j` gives the count at overlap size `j`.
3. A relation has min-density exactly `t` iff it is inside `D^{≥t}` but not
   inside `D^{>t}`; differencing consecutive sweeps assigns counts to
   `φ_r = j·t` buckets.

Verified exactly (bucket-for-bucket) on all four small systems. Two caveats,
both real: the intersection closure can in principle be exponentially large
in `|𝒰|` (it is small for the structured systems tried here; §6.3), and
mathematically-equal densities that differ by float noise split thresholds
harmlessly for sums but must be grouped at `config.numerics.precision` when
reporting distinct φ values (this actually occurred on `grid3`, where two
mechanisms' densities agree to 14 significant figures but not bit-for-bit).

#### 3.1.8 Face counts, and face-level aggregates **[proposed]**

Faces disappear from φ_r by the S3 identity, but the face structure itself is
still part of the formalism (a relation has up to `3^h` faces). The same
machinery covers them at MICE granularity: take as atoms the `2|D|`
individual causes and effects with their state-tagged purviews
`z*_c(d)`, `z*_e(d)`. A face is exactly a subset of MICEs, of size ≥ 2, with
nonempty state-tagged intersection (a direction assignment
cause/effect/both per distinction is the same thing as a choice of a MICE
subset covering the relation's relata). So the total face count across all
relations — and any face-degree-resolved refinement — is the same
inclusion–exclusion/Möbius computation run on `2|D|` sets instead of `|D|`.
Verified exactly on all four small systems (e.g. `grid3`: 297 faces across 39
relations, closed form = enumeration). The per-relation `faces` property
stays lazy in any case **[exists]**.

#### 3.1.9 Sensitivity / ablation queries **[exists in parts; proposed as a family]**

Because every Tier-1 quantity is a cheap function of `𝒵`, *differences* of
Tier-1 quantities under modification of `𝒵` are equally cheap:

- "How much of `Σφ_r` does distinction `d` carry?" —
  `AnalyticalFoldRelations` **[exists]**.
- "What happens to the relational structure if distinction `d` is removed?"
  — recompute on `D∖{d}`; the fold identity gives it as a difference without
  recomputation.
- "Which distinction's removal costs the most Φ?" — `|D|` fold evaluations,
  each `O(|𝒰| · |D| log |D|)`; at Fig. 6D scale, all 27 in 50 ms (§5.2).
  This subsumes the ROADMAP N24 item (Zaeemzadeh's per-purview incidence
  counts and the asymptotic `|Z|/2N` vanishing fraction are the
  structure-agnostic versions of the same query; the fold version is exact
  for the actual structure at hand).

#### 3.1.10 Monotone lower bounds from partial distinction sets **[proposed]**

`Σφ_r` is monotone nondecreasing in `D`: adding a distinction adds relations
and changes no existing φ_r (φ_r depends only on its members). So the
analytical sum over any *subset* of the true distinction set — for example,
distinctions discovered so far during an incremental CES computation, or
mechanisms up to a size cutoff — is a certified lower bound on the true
`Σφ_r`, and it only tightens as distinctions arrive. Combined with the
Zaeemzadeh-style ceilings in `pyphi/formalism/iit4/bounds.py` **[exists]**
(which need only `n`), every Tier-1 scalar can be reported as an interval
during any partial computation. The count and degree-resolved counts are
monotone the same way; *maxima* are too; histograms are not (buckets shift as
minima change).

The bound is certified but, measured on Fig. 6D, **exponentially
back-loaded**: with 25 of 27 distinctions present it certifies only 25% of
the true `Σφ_r` when distinctions arrive in mechanism-size order, and 66%
under the best ordering tried (descending φ or descending density). The
mechanism is structural, not incidental — each distinction added to an atom
shared by `m` others roughly doubles that atom's subset sum — so a partial
distinction set can never certify more than a vanishing fraction until it is
nearly complete. An anytime-Φ mode built on this bound would be sound but
uninformative; the finding mostly closes that direction (§6.2).

### 3.2 Tier 2 — exact, lazy, output-sensitive

**[proposed]** φ_r is antitone on the subset lattice (§2.2), and every
relation of degree `h ≥ 3` contains a degree-`(h−1)` relation obtained by
removing its largest-index member. Best-first search therefore enumerates
relations in exactly descending φ_r order, lazily:

- seed a max-heap with all valid pairs (`O(|D|²)` — this is also the §3.1.5
  scan, so the seeding is never wasted work);
- on popping `S`, yield it and push its canonical extensions
  `S ∪ {d_j}, j > max-index(S)` that still overlap.

Correctness is the standard best-first invariant: every not-yet-yielded
relation has an ancestor in the heap with φ at least as large. The first `K`
relations cost `O(K·|D|)` heap pushes beyond the seed, independent of the
total relation count. Verified: the first 10 yields match the sorted concrete
enumeration on all four small systems, and produce the top-10 of the 1.5M
Fig.6D relations in under a millisecond (§5.2). This is the "lazy / top-K
relations mode" wished for in ROADMAP item N6, and it directly serves the two
iterating consumers (visualization, diff) with the `K` most important
relations plus an "…and N more, Σφ_r = x" line, exact from Tier 1.

The same search with a φ threshold instead of a count `K` answers "all
relations with φ_r ≥ t" output-sensitively (the antitone property means the search can prune every branch whose
root already falls below the threshold); with a degree cap it answers "all pairwise
relations" in `O(|D|²)` (**[exists]** as `all_relations(max_degree=2)`).

Memory is the one real cost: the heap can grow to `O(yielded · |D|)`
entries if the stream is consumed deeply. For top-K use this is negligible;
for "iterate everything lazily" it is not better than DFS enumeration and
should not pretend to be.

### 3.3 Tier 3 — statistical, unbiased, with known error

**[established — Karp–Luby-style union-of-sets sampling; the application to
relations is proposed]** The generating representation is a union of at most
`2n` explicitly-sized families (`N_a = 2^{|D_a|} − |D_a| − 1` relations lie
inside `D_a`), and the coverage count of a sampled relation is computable in
`O(|S| · n)` (`|O(S)|` itself). This is exactly the setting of Karp–Luby DNF
counting:

- draw atom `a` with probability `∝ N_a`;
- draw `S` uniformly among the size-≥2 subsets of `D_a`;
- then `P(S) = |O(S)| / Σ_a N_a` — known exactly per sample.

Horvitz–Thompson reweighting turns this into an unbiased estimator of
**any** per-relation sum `Σ_S f(S)` — including sums no closed form reaches,
e.g. `f` an arbitrary predicate ("relations whose purview contains atom `a`
and whose degree exceeds 8"), with standard-error bars from the sample
variance, i.i.d. samples, no burn-in, exact normalization
`Z = Σ_a N_a` known in closed form. Uniform sampling of relations follows by
rejection (accept with probability `1/|O(S)|` — acceptance ≥ `1/|𝒰|`).

Verified: 2,000 draws estimate the non-self relation count and `Σφ_r` within
~0.5% on the small systems (where exact answers check the estimator), and
5,000 draws land within 0.16% of the exact `Σφ_r` and count at Fig. 6D scale
in 60 ms (§5.2). Two design requirements from the project's standards: the
sampler must take an explicit `seed` and use an isolated
`random.Random(seed)` / `numpy` generator, and estimates must be reported
with their standard errors, never bare.

The variance caveat is real: HT estimates of *rare-event* sums (a
predicate satisfied by a vanishing fraction of relations, e.g. properties of
the extreme φ tail) have large relative error at fixed sample size. For the φ
tail specifically, Tier 2 answers exactly; for rare structural predicates,
stratifying by degree using the exact Tier-1 degree distribution as strata
weights is the standard fix and needs nothing new here (the degree-restricted
sampler just replaces "uniform subset of `D_a`" with "uniform size-`k` subset
of `D_a`", and the exact per-degree counts from §3.1.4 are the strata
totals).

### 3.4 Tier 4 — what resists all three

Being precise about the boundary is the point of this section.

#### 3.4.1 Arbitrary exact queries

Any query that is an arbitrary function of the full relation multiset —
"the exact number of relations satisfying predicate P" for unstructured P —
is #P-hard in general: counting the relations alone is DNF-counting-shaped,
and DNF counting is #P-complete in general (the *exact* count here is
tractable only because the union has ≤ `2n` clauses — inclusion–exclusion
over `2^{2n}` atom subsets — which becomes the binding constraint at the
macro/coarse-grained scale where `n` itself grows). Exactness for arbitrary
predicates short of enumeration is not on offer; sampling with error bars
(§3.3) is the right tool.

#### 3.4.2 Optimization over folds

"Find the seed set `F` of size `k` whose Φ-fold has maximal internal
`Σφ_r`" — the natural formalization of "find the most bound content in this
experience" — evaluates cheaply per candidate (Tier 1 on `F`), but the
search space is `C(|D|, k)`. Internal `Σφ_r(F)` is monotone and
supermodular — a new member's marginal contribution only grows with the
existing seed set — so the classic submodular-greedy approximation guarantee
does not apply; this is a
combinatorial optimization problem with cheap oracle evaluations, and only
heuristics (greedy, local search, seeding from the §3.1.6 binding matrix's
dense blocks) are available. No exactness claim should be made. This is the
clearest concrete open problem this design leaves unsolved, and it is likely
where the phenomenologically interesting queries (content individuation)
land.

#### 3.4.3 Relation-level diff of two structures

`CauseEffectStructure.diff` currently reports relations gained/lost, which
requires iterating both sides. At scale, the *set difference of two
exponential families* is itself exponential — no representation fix changes
what the answer costs to print. The workable replacements are (a) the exact
difference of any Tier-1 statistic (count, Σφ_r, degree spectrum, φ
histogram, binding matrix — the last being the natural "what changed
structurally" object, entry-wise), all closed-form on both sides; and (b)
since the relation set is a deterministic view of `𝒵` (§2.1), the *complete*
relational diff is generated by the distinction-level diff — two structures
with identical `𝒵` have identical relations, and every relational change is
attributable to specific added/removed/re-weighted distinctions. A useful
exact middle granularity: per-atom simplex diffs (`D_a` before vs. after),
of which there are at most `2n`.

#### 3.4.4 Unresolved ties

Relations are only defined over `ResolvedDistinctions` — distinctions whose
tied specified states have been disambiguated by the SIA's system state
**[exists]**; on unresolved distinctions the purview-unions themselves are
ambiguous and everything above is ill-posed (phantom faces). The interface
must keep the type-level guard. Relatedly, *within* this design ties in `q`
values across distinctions are harmless for sums but must be grouped at
`config.numerics.precision` for histograms and for any "distinct values"
reporting (§3.1.7; observed in practice on `grid3`).

#### 3.4.5 Per-relation φ semantics beyond Eq. 55

Everything in §3 leans on the specific product form
`φ_r = |O| · min q` (and on φ_d being distributed uniformly over purview
units). A different φ_r — say one that re-examined partitions per relation,
or weighted faces non-uniformly — would break the per-atom factoring, and
with it every Tier-1 closed form except the count. The S3 supplement's
authors chose this form partly *because* it factors; the design below keeps
the φ-definition dependency explicit (a `RelationalStructure` is constructed
from `(u, q)` pairs, so any future φ_r that still factors through per-atom
statistics slots in; one that does not, falls back to Tiers 2–3, and Tier 2
survives only if the new φ_r is still antitone).

## 4. Interface proposal

### 4.1 Shape

Keep the existing registry and class family; grow the query surface of the
non-materializing backend. Concretely — names indicative, not final:

```python
class Relations(...):            # existing ABC
    # existing: sum_phi(), apportioned_sum_phi(), num_relations()

    # Tier 1 (exact; default implementations raise or enumerate where possible)
    def sum_phi_moment(self, k: int = 2) -> float: ...
    def phi_mean_std(self) -> tuple[float, float]: ...
    def num_relations_of_degree(self, k: int) -> int: ...
    def sum_phi_of_degree(self, k: int) -> float: ...
    def degree_spectrum(self) -> Mapping[int, tuple[int, float]]: ...
    def max_phi(self) -> PyPhiFloat: ...
    def phi_histogram(self) -> Mapping[PyPhiFloat, int]: ...
    def binding_matrix(self) -> "UnitStateMatrix": ...   # §3.1.6
    def num_faces(self) -> int: ...
    def fold(self, seeds) -> "Relations": ...            # exists (models/ces.py)

    # Tier 2 (lazy, exact order)
    def strongest(self, k: int | None = None, min_phi: float | None = None,
                  max_degree: int | None = None) -> Iterator[Relation]: ...

    # Tier 3 (statistical; seed mandatory)
    def sample(self, n: int, *, seed: int, max_degree: int | None = None
               ) -> "RelationSample": ...   # carries HT weights + estimate()/stderr()

    # escape hatch
    def materialize(self, max_degree=None, min_phi=None) -> ConcreteRelations: ...
```

- `AnalyticalRelations` implements Tier 1 in closed form and Tiers 2–3 from
  `𝒵`; it stays non-iterable (iteration remains the explicit, syntactically
  loud `materialize()` / `strongest()` choice, not an implicit `for` loop
  that silently enumerates 2^|D| objects).
- `ConcreteRelations` inherits default implementations that just iterate —
  the two backends answer the same queries, which is also how the whole
  surface gets tested (the invariant suite pattern already used for
  `sum_phi`: analytical == concrete on small systems, property-tested).
- `strongest()` yields real `Relation` objects, so display, `to_pandas`, and
  the visualization pipeline consume them unchanged. A
  `Relations.summary_row()` used by capped tables gains "top-K shown of N
  (exact), Σφ_r = x" phrasing for the analytical backend.
- The internal representation backing all of it is the atom→distinction
  incidence (§2.2) built once from `purview_inclusion(max_order=1)`
  **[exists]** plus the `q` vector; everything else is derived.

### 4.2 What plugs in where

- **Φ, folds, matching, analyze/sweep** — unchanged; they already use Tier 1.
- **Visualization** — `project_ces` currently iterates `ces.relations` and
  refuses analytical structures; it instead requests
  `relations.strongest(k)` (with `k` from the existing display caps) and the
  Tier-1 scalars for annotation. This turns "cannot plot analytical
  structures" into "plots the K strongest relations of any structure, with
  exact totals in the legend."
- **diff** — replaced per §3.4.3: exact statistic deltas + distinction-level
  attribution, with relation-level listing only under an explicit
  materialization bound.
- **`bounds.py`** — unchanged; it answers the "no distinction set at all"
  regime (ceilings from `n` alone), and §3.1.10 connects it to the partial
  regime: `partial-D lower bound ≤ Σφ_r ≤ bounds ceiling`, both certified.
- **Serialization** — already correct: nothing to store beyond the
  distinctions.
- **Config** — no new global options needed. `relation_computation`
  ("CONCRETE" / "ANALYTICAL") keeps selecting the *default* backend of a
  computed CES; every query above is available on both.

### 4.3 Numerical discipline

- The closed forms use coefficients like `2^{m}` with `m = |D_a|`; float64
  overflows past `m ≈ 1023` and loses integer precision far earlier
  (`m > 53`). Counts must be Python ints end-to-end (they already are in
  `_num_relations`; `sum_of_minimum_among_subsets` currently goes through
  `numpy` float64 — fine for the sums, which are floats anyway, but at
  `|D_a| ≳ 50` the largest coefficients dwarf the smallest and the sorted
  dot product should accumulate from the small end or use `math.fsum`;
  at `|D_a| ≳ 1000` the *sum itself* exceeds float range and only
  log-space or rational arithmetic remains meaningful — the same regime
  `bounds.py` already documents as float-hostile).
- All φ comparisons (histogram bucketing, `min_phi` thresholds, tie grouping
  in the sweep of §3.1.7) go through `config.numerics.precision` /
  `PyPhiFloat`, not raw `==`.
- Sampling: mandatory explicit `seed`, isolated RNG instance, estimates
  carried with standard errors.

## 5. Worked demonstrations

All demonstrations run against the current library (IIT 4.0, default
measures). "Verified" means float-exact or integer-exact agreement with
concrete enumeration.

### 5.1 Small systems — every primitive against enumeration

For `xor`, `grid3`, `fig4`, `rule110` (4–7 distinctions, 11–39 relations):

| Query | Method | Result |
|---|---|---|
| `Σφ_r`, `Σφ_r/\|r\|`, count | closed form **[exists]** | == enumeration, all 4 systems |
| fold incident sums, every single-distinction seed | closed form **[exists]** | == enumeration, all seeds, all 4 systems |
| `Σφ_r²` (second moment) | atom-pair closed form | == enumeration (float-exact), all 4 |
| per-degree count & sum, all degrees | closed form | == enumeration, all degrees, all 4 |
| max φ_r | pair scan (§3.1.5) | == enumeration, all 4 |
| descending-φ lazy stream | best-first (§3.2) | first 10 == sorted enumeration, all 4 |
| φ_r histogram | threshold sweep + Möbius (§3.1.7) | bucket-for-bucket == enumeration, all 4 |
| binding matrix `B(a,b)` | atom-pair closed form | entry-wise == enumeration, all 4 |
| total face count | MICE-atom Möbius (§3.1.8) | == enumeration (e.g. grid3: 297; xor: 119), all 4 |
| count & `Σφ_r` via sampling | Karp–Luby + HT, 2000 draws, seed 42 | within ~0.5% of exact, all 4 |

(One instructive failure during verification: the histogram check first
"failed" on `grid3` because two mechanisms' φ densities agree mathematically
but differ in the last two bits of float64, splitting one bucket into two.
The closed form was right; the lesson is §4.3's precision-grouping rule.)

### 5.2 Fig. 6D — 27 distinctions, 1,537,080 relations

The paper's specialized-lattice example, the largest relational structure the
paper reports. Distinction computation (the CES) took 309 s single-core;
everything below is measured after it, on the distinction set alone:

| Query | Time | Result |
|---|---|---|
| count | 3 ms | 1,537,080 — matches the paper exactly |
| `Σφ_r` | (same call) | 11445.7506 (paper: 11445.7586; relative difference 7·10⁻⁷. Concrete enumeration of the same structure reproduces 11445.7506 digit-for-digit (§5.3), so the difference from the paper is upstream — in the distinction φ values — not in the relations) |
| exact mean, std of φ_r over all 1.5M | 1 ms | 0.007446 ± 0.003475 |
| exact degree spectrum (all 26 degrees) | 2 ms | unimodal, peak at degree 10 (265,825 relations) |
| max φ_r | 1 ms | 0.419612 (a 4-way tie of degree-2 relations) |
| top-10 strongest, lazily | 1 ms | all degree ≤ 3 |
| sampled `Σφ_r` & count (5,000 draws, seed 42) | 60 ms | within 0.16% of exact |
| exact incident `Σφ_r` for **all 27** distinctions (ablation ranking) | 50 ms total | top carrier: mechanism (1,2,3,4,5), incident Σφ_r = 5721 over 768,546 relations |

The asymmetry is the finding: with the query interface, the relational
structure of a system whose relations outnumber its distinctions by five
orders of magnitude costs *milliseconds* to interrogate rather than being the
bottleneck — the bottleneck is now entirely the distinctions (the CES), which
is a different scaling problem with different remedies (and where relational machinery
cannot help: §3.1.10's lower bounds are the only relational
statement available before the CES finishes).

Concrete enumeration of the same structure, for comparison, is measured in
§5.3.

### 5.3 Where concrete enumeration lands

For calibration, the same Fig. 6D structure enumerated concretely
(`ConcreteRelations(all_relations(...))`, sequential):

- building the 1,537,080 `Relation` objects: **11.6 s**, **+1.43 GiB** peak
  resident memory;
- computing φ for all of them (`sum_phi()`, which forces every relation's
  lazy φ): a further **38.3 s**;
- `Σφ_r = 11445.7506` — digit-for-digit the analytical value.

Two further at-scale validations against this enumeration:

- the exact φ_r histogram (§3.1.7) — 81 distinct values over 1,537,065
  non-self relations, computed in **8 ms** (the intersection closure has just
  46 elements over 8 atoms) — matches the enumerated histogram
  bucket-for-bucket;
- the lazy descending stream (§3.2) — its first 100 yields match the sorted
  enumeration of all 1.5M φ values.

So at 27 distinctions concrete enumeration is a ~50-second, ~1.4 GiB
operation while the entire Tier-1/2 query battery is milliseconds; and since
the relation count roughly doubles per added distinction on a shared atom,
|D| in the mid-30s puts enumeration out of memory entirely, while the query
interface's costs do not change (they scale with `|D|` and `|𝒰|`, not with
the relation count). Fig. 6D has `|𝒰| = 8` atoms; `|𝒰|` is bounded by `2n`
regardless of `|D|`.

## 6. Open problems

### 6.1 The fold-optimization problem (§3.4.2)

Exact content-fold selection is open. Cheap per-candidate evaluation plus a
structured search space suggests it is *approachable* (branch-and-bound with
the §3.1.10 monotone bounds; the binding matrix as a spectral seed), but
nothing here proves an approximation guarantee, and the problem deserves its
own investigation before any API promises anything beyond "heuristic".

### 6.2 Tightness of the partial-CES interval — measured, mostly negative

The §3.1.10 lower bound converges too late to be useful as an anytime
estimate of `Σφ_r` (25–66% certified at 25/27 distinctions on Fig. 6D,
depending on arrival order). What remains open is whether a *predictive*
(non-certified) partial estimate is possible: the closed forms make
`Σφ_r` a known function of the final atom occupancies `|D_a|`, so an
estimated *count* of remaining distinctions per atom would extrapolate the
sum exponentially better than the bare partial bound. That is an estimation
problem about distinctions, not relations, and is unexplored.

### 6.3 Intersection-closure growth

The exact histogram (§3.1.7) and the exact-overlap Möbius counts are
polynomial in the closure size, which is bounded by `2^{|𝒰|}` but in the
systems tried here stayed small (46 elements at Fig. 6D scale; a few dozen
or fewer on the small systems). A characterization of
when closures blow up (highly heterogeneous purview geometries?) — and a
fallback (the closure computation is itself lazy and can abort to sampling) —
is future work. The same closure would also speed up the existing
`_num_relations`, which currently walks all `2^{|u(d)|}` subsets of every
purview-union rather than the (typically far smaller) closure.

### 6.4 The 2026 formalism

Both registered IIT 4.0 formalisms (2023, 2026) share the relation code and
Eq. 55, so everything here applies to both. Any future revision of φ_r must
be checked against the factoring requirement (§3.4.5) *before* being adopted,
if the analytical path is to survive; that constraint belongs in whatever
document governs formalism changes.

### 6.5 Face-level φ

§3.1.8 counts faces but assigns them no independent φ (the code gives every
face its parent relation's φ). If a future formalism assigns per-face φ with
its own min/overlap structure, the MICE-atom machinery likely extends, but
that is a conjecture, not a result.

## 7. Relation to prior art within the project

- The S3 analytical supplement and `AnalyticalRelations` are the foundation;
  this design is best read as "the S3 factoring trick, taken seriously as an
  interface contract rather than as three special-case formulas."
- The Zaeemzadeh & Tononi bounds paper (and `bounds.py`) covers the
  no-distinction-set regime; §3.1.10 connects the two regimes into intervals.
- The two counting-relations notes (Mayner 2022-02-23; Zaeemzadeh 2022-03-01)
  count *possible* relations over all purviews — structure-agnostic
  combinatorics. The per-structure closed forms here answer the same
  question-shapes (per-overlap-size, per-purview incidence) for the *actual*
  distinction set, which is what N24 asks for.
- ROADMAP N6 ("lazy / top-K relations mode") is §3.2; ROADMAP N24
  (distinction-importance) is §3.1.9; the "generative-relations compression"
  research note (persist distinctions, regenerate relations) is the
  serialization shadow of §2.1's thesis and is already effectively landed for
  the analytical backend.

## 8. Summary of what breaks and where

| Wanted | Verdict |
|---|---|
| Any monomial-weighted sum over relations (`Σ φ_r^k`, degree-restricted, atom-restricted, fold-restricted) | exact, fast, closed form |
| count, degree spectrum, φ histogram, binding matrices, face counts | exact, fast (histogram: exact but closure-size-dependent) |
| max φ_r; top-K by φ; all φ ≥ t | exact, output-sensitive, lazy |
| arbitrary per-relation functionals / predicates | unbiased estimates with error bars; exactness is #P-hard in general |
| full enumeration, relation-level set diff | inherently exponential output; only bounded materialization |
| optimal content-fold selection | open; heuristics only |
| any of the above under a non-factoring φ_r | Tier 1 collapses to enumeration; Tier 2 survives iff antitone; Tier 3 survives |
| anything before the CES finishes | only interval bounds (partial-D lower, `bounds.py` upper) — and the lower bound is measured to be exponentially back-loaded, so the interval stays wide until the CES is nearly done |
## Appendix: demonstration code

The primitives below are the complete implementations behind §5's numbers,
written against the current library. Distinction sets come from the standard
pipeline, e.g.:

```python
from pyphi import config, examples
from pyphi.formalism import iit4
from pyphi.measures.distribution import (
    resolve_mechanism_measure, resolve_system_measure)

def get_distinctions(system):
    return iit4.ces(
        system,
        system_measure=resolve_system_measure(
            config.formalism.iit.system_phi_measure),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure),
        relations_kwargs={"relation_computation": "ANALYTICAL"},
    ).distinctions
```

Ground truth for verification is `ConcreteRelations(all_relations(distinctions))`.

```python
import heapq
import itertools
import math
import random
from collections import Counter, defaultdict

from pyphi import combinatorics


def unit_index(distinctions):
    """Atom -> distinctions whose purview-union contains it (the D_a sets)."""
    index = defaultdict(list)
    for d in distinctions:
        for a in d.purview_union:
            index[a].append(d)
    return index


def ratio(d):
    """The phi density q(d)."""
    return float(d.phi) / len(d.purview_union)


def self_relation_phis(distinctions):
    return [
        len(o) * ratio(d)
        for d in distinctions
        if (o := d.cause.purview_units & d.effect.purview_units)
    ]


# ---- §3.1.3: second moment (k-th: use k-tuples of atoms and q**k) ----
def sum_phi_squared(distinctions):
    index = unit_index(distinctions)
    total = 0.0
    for u in index:
        setu = set(index[u])
        for v in index:
            group = [d for d in index[v] if d in setu]
            if len(group) >= 2:
                total += combinatorics.sum_of_minimum_among_subsets(
                    [ratio(d) ** 2 for d in group])
    return total + sum(p**2 for p in self_relation_phis(distinctions))


# ---- §3.1.4: degree-resolved sum and count ----
def sum_phi_by_degree(distinctions, k):
    index = unit_index(distinctions)
    total = 0.0
    for group in index.values():
        vals = sorted(ratio(d) for d in group)
        for i, v in enumerate(vals):
            a = len(vals) - 1 - i
            if a >= k - 1:
                total += v * math.comb(a, k - 1)
    return total


def num_relations_by_degree(distinctions, k):
    return sum(
        (-1) ** (len(purview) - 1) * math.comb(len(group), k)
        for purview, group in distinctions.purview_inclusion(max_order=None)
    )


# ---- §3.1.5 / §3.2: max phi and lazy descending enumeration ----
def lazy_descending_relations(distinctions):
    """Yield (phi, index-tuple) for all non-self relations, descending phi."""
    ds = list(distinctions)
    pus = [set(d.purview_union) for d in ds]
    rats = [ratio(d) for d in ds]

    def phi_of(idxs):
        overlap = set.intersection(*(pus[i] for i in idxs))
        return len(overlap) * min(rats[i] for i in idxs) if overlap else None

    heap, counter = [], itertools.count()
    for i, j in itertools.combinations(range(len(ds)), 2):
        if (p := phi_of((i, j))):
            heapq.heappush(heap, (-p, next(counter), (i, j)))
    while heap:
        negp, _, idxs = heapq.heappop(heap)
        yield -negp, idxs
        for nxt in range(idxs[-1] + 1, len(ds)):
            if (p := phi_of(idxs + (nxt,))):
                heapq.heappush(heap, (-p, next(counter), idxs + (nxt,)))


# ---- §3.3: Karp-Luby-style sampler with Horvitz-Thompson estimates ----
def sample_relations(distinctions, n_samples, seed):
    """Draw relations with probability proportional to |O(S)| (known per
    sample), so any sum over relations is estimable without bias."""
    rng = random.Random(seed)
    index = unit_index(distinctions)
    units = list(index)
    weights = [2 ** len(index[u]) - len(index[u]) - 1 for u in units]
    Z = sum(weights)  # = sum over non-self relations of |O(S)|
    samples = []
    for _ in range(n_samples):
        u = rng.choices(units, weights=weights)[0]
        group = index[u]
        while True:
            mask = rng.getrandbits(len(group))
            if mask.bit_count() >= 2:
                break
        S = [d for i, d in enumerate(group) if mask >> i & 1]
        overlap = set.intersection(*(set(d.purview_union) for d in S))
        samples.append((S, len(overlap)))
    return samples, Z


def ht_estimate(samples, Z, f):
    """Unbiased estimate of sum over non-self relations of f(S)."""
    return Z * sum(f(S) / o for S, o in samples) / len(samples)


# ---- §3.1.7: exact phi_r histogram (non-self relations) ----
def intersection_closure(sets):
    closure, frontier = set(), {frozenset(x) for x in sets if x}
    while frontier:
        closure |= frontier
        frontier = {
            q for p in frontier for x in sets
            if (q := p & frozenset(x)) and q not in closure
        }
    return closure


def overlap_size_counts(distinctions):
    """Exact counts of non-self relations by overlap size, via Moebius
    inversion over the intersection closure of the purview-unions."""
    pus = [frozenset(d.purview_union) for d in distinctions]
    closure = sorted(intersection_closure(pus), key=len, reverse=True)
    exact, counts = {}, Counter()
    for p in closure:
        m = sum(1 for pu in pus if p <= pu)
        exact[p] = (2**m - m - 1) - sum(exact[q] for q in closure if p < q)
    for p, e in exact.items():
        if e:
            counts[len(p)] += e
    return counts


def phi_histogram(distinctions):
    """{phi_r: count} over non-self relations. NOTE: thresholds should be
    grouped at config.numerics.precision in a production implementation."""
    ds = sorted(distinctions, key=ratio)
    hist, prev = Counter(), Counter()
    for t in sorted({ratio(d) for d in ds}, reverse=True):
        cur = overlap_size_counts([d for d in ds if ratio(d) >= t])
        for j in set(cur) | set(prev):
            if (n := cur[j] - prev[j]):
                hist[j * t] += n
        prev = cur
    return hist


# ---- §3.1.6: unit-pair binding matrix ----
def binding_matrix(distinctions):
    """B[a, b] = sum of min-density over non-self relations whose purview
    contains both atoms (= sum of phi_r/|O| over those relations)."""
    index = unit_index(distinctions)
    M = {}
    for u in index:
        setu = set(index[u])
        for v in index:
            group = [d for d in index[v] if d in setu]
            if len(group) >= 2:
                M[u, v] = combinatorics.sum_of_minimum_among_subsets(
                    [ratio(d) for d in group])
    return M


# ---- §3.1.8: total face count, same machinery at MICE granularity ----
def total_num_faces(distinctions):
    mice = [frozenset(m.purview_units)
            for d in distinctions for m in (d.cause, d.effect)]
    closure = sorted(intersection_closure(mice), key=len, reverse=True)
    exact = {}
    for p in closure:
        m = sum(1 for x in mice if p <= x)
        exact[p] = (2**m - m - 1) - sum(exact[q] for q in closure if p < q)
    return sum(exact.values())
```

These are demonstrations, not production code: a real implementation would
share the atom index across queries, use `PyPhiFloat`-precision grouping for
thresholds and histogram keys (§4.3), handle the large-`|D_a|` numeric
regimes (§4.3), and route parallelism through the existing `map_reduce`
infrastructure where fan-out exists (the atom-pair loops, the fold sweep).
