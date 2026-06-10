# Zaeemzadeh upper bounds — design

**Project:** P13 sub-project 1 (of 2). A standalone bounds module exposing the
published upper bounds on IIT quantities (Zaeemzadeh & Tononi 2024, *Upper
bounds for integrated information*, PLOS Comp Biol 20(8):e1012323) as a
research utility: ceiling estimates for a system's φ, Σφ_d, Σφ_r, Φ, and φ_s
as functions of size and structure. Search integration (pruning) is
sub-project 2, **gated on a bite-rate study** and consuming only
theorem-certified bounds; if pruning turns out not to pay, this utility
surface is the deliverable on its own.

**Sources of truth:** the published paper (`papers/2024__zaeemzadeh-tononi__upper-bounds.pdf`)
and its appendices (`...__s1-appendix.pdf` cause-side translation,
`...__s2-appendix.pdf` other difference measures, `...__s3-appendix.pdf`
proofs). The pre-publication module deleted in `e5e27868`
(`git show e5e27868^:pyphi/upper_bounds.py`) does **not** match the published
bounds one-to-one (e.g. its `PURVIEW_SIZE` bound is (N/2)·2^N where published
Bound I is (N²/2)·2^N) and is consulted only for the relation-counting
combinatorics. The author's experiment code
(github.com/zaeemzadeh/IIT-bounds) is the cross-check reference.

## Scope

In scope:
- `pyphi/formalism/iit4/bounds.py`: the published bound inventory as plain
  functions returning `UpperBound` objects, plus the counting family and a
  one-call `report()`.
- The math-verification work: re-derivation of the load-bearing inequalities
  against the S3 proofs, independent recomputation in tests, and empirical
  domain confirmation (which measures/versions the bounds provably or
  demonstrably cover in 2.0).
- The ROADMAP P13 correction (the `find_mip` pruning claim is mathematically
  wrong as stated; see below).

Out of scope (sub-project 2, separate spec):
- Any wiring into `complexes()`, purview search, or the formalism registry
  (`ErrorInfo` / approximate-formalism objects).
- The shadow-mode bite-rate study that gates that wiring.

## Mathematical inventory and certificate status

All bounds assume **binary units** and a **conditionally independent TPM**
(product of unit TPMs). They are derived for the IIT 4.0 intrinsic-difference
measure φ = selectivity · |informativeness|₊ with selectivity ≤ 1; S2
extends the results to PMI- and KL-style variants. The bound argument holds
at *any* selected mechanism partition (it bounds the informativeness term for
every partition θ), so it is robust to 2.0's partition-scheme and
MIP-normalization configuration. The bounds are ceilings over states, hence
state-independent.

| Bound | Statement | Status | Citation |
|---|---|---|---|
| Partition | φ(m, Z given θ) ≤ 𝒩(θ) (connections severed) | **certified** | Lemma 2 |
| Mechanism-purview | φ(m, Z) ≤ \|M\|·\|Z\| | **certified** | Theorem 1 |
| Relation | φ_r(**d**) ≤ min over relata of φ_d | **certified** (partially structural in 2.0 — see Testing) | §2.2 |
| System | φ_s ≤ \|S\|(\|S\|−1) | **certified** | Table 2 / Marshall et al. 2023 |
| Σφ_d Bound I | Σφ_d ≤ (N²/2)·2^N | **certified** (not achievable) | Eq 6 |
| Σφ_d Bound II | Σφ_d ≤ (N(N+1)/4)·2^N | **conditional** — assumes unique purviews (each purview assigned to one mechanism); real systems need not satisfy this | Eq 7 |
| Σφ_d Bound III | numerical: Σ_K C(N,K)·φ_e*(K) from the high-selectivity reflexive construction (K/2+1 partitions per K) | **conjectured** — proven only for reflexive selectivity-1 systems; generality is an open question in the paper | §2.1.3, S3 |
| Σφ_r Bounds I/II/III | closed forms (Table 3) via the Eq 11–15 linear-program solution, from the corresponding Σφ_d input | status **inherited** from the Σφ_d input (Eq 15 turns any S(o) bound into a Σφ_r bound) | Eqs 11–16, Table 3 |
| Counts | number of possible distinctions (2^N−1), relations (2^(2^N−1)−1), and the unique-purview per-order variants | **certified** (pure combinatorics, measure-free) | §2.2 + old module's counting functions |

**Certificate taxonomy.** `certified` means theorem-backed for arbitrary
systems within the domain assumptions — eligible for pruning in sub-project
2. `conditional` and `conjectured` bounds are estimates: exposed for research
use with their assumptions attached, mechanically ineligible as pruning
certificates.

**ROADMAP correction (deliverable).** The P13 entry says to "use [bounds] in
`find_mip` to prune partitions that cannot achieve the best-so-far φ". φ_s is
a *minimum* over partitions; upper bounds prune maximizations, and pruning a
minimization needs per-partition *lower* bounds, which the paper does not
provide (the φ=0 shortcircuit already handles the floor). The candidates that
survive scrutiny: skipping candidate *systems* in `complexes()`
(maximization, certified φ_s caps) and capping mechanism *purview search*
(maximization, Theorem 1) — both deferred to sub-project 2 and gated on
measured bite rates. The P13 ROADMAP text is rewritten accordingly, with the
research-utility surface as the primary deliverable.

## Components

### `UpperBound` (`pyphi/formalism/iit4/bounds.py`)

```python
@dataclass(frozen=True)
class UpperBound:
    value: float
    certified: bool                 # theorem-backed for arbitrary systems
    assumptions: tuple[str, ...]    # e.g. ("binary units", "unique purviews")
    citation: str                   # paper locus, e.g. "Eq 7"

    def __float__(self) -> float: ...
```

Every bound function returns one; the certificate travels with the number.

### Bound functions

```python
distinction_phi_upper_bound(mechanism, purview) -> UpperBound   # |M||Z|
partition_phi_upper_bound(partition) -> UpperBound              # 𝒩(θ)
relation_phi_upper_bound(relata_phis) -> UpperBound             # min φ_d
system_phi_upper_bound(n) -> UpperBound                         # n(n−1)
sum_phi_distinctions_upper_bound(n, bound="I") -> UpperBound    # "I" | "II" | "III"
sum_phi_relations_upper_bound(n, bound="I") -> UpperBound
big_phi_upper_bound(n, bound="I") -> UpperBound                 # Σφ_d + Σφ_r ceilings
number_of_possible_distinctions(n) -> int
number_of_possible_relations(n) -> int                          # + unique-purview variants
report(n=None, substrate=None) -> dict[str, UpperBound]
```

- Size-based functions take `n` (pure combinatorics of the binary formalism).
  Object-based functions take index tuples (only sizes matter). `report()`
  accepts either `n` or a `Substrate` (in which case the substrate's alphabet
  is validated and `n` is taken from it); keys are flat strings like
  `"sum_phi_distinctions:I"`.
- Bound III is a numerical procedure: build the paper's high-selectivity
  construction per mechanism size K and evaluate its K/2+1 candidate
  partitions (procedure from S3; cross-checked against the IIT-bounds repo if
  S3 underspecifies). Exposed through the same `bound="III"` parameter; its
  helper sub-API (construction TPM builder, partition evaluator) is module-
  private but reused by the tightness validation.
- No config keys, no registry, no imports from the compute pipeline into the
  bound formulas. Python ints are used wherever the quantities are integral
  (these grow hyper-exponentially; arbitrary precision is free correctness).

### Domain guard

A shared `_require_valid_domain()` raises `ValueError` with the paper
citation when the active configuration is outside the proven/confirmed
domain:

- `config.formalism.iit.version` and the relevant measure
  (`mechanism_phi_measure` for mechanism/structure bounds;
  `system_phi_measure` for the φ_s bound — the whitelists may differ) must be
  in the bound's whitelist.
- Binary units are validated wherever a system object is available
  (`report(substrate=...)`); pure-size functions document the assumption.

**Whitelist admission protocol:** the whitelist ships with exactly the
combinations the property tests confirm. Candidates:
IIT_4_0_2023/IIT_4_0_2026 × GENERALIZED_INTRINSIC_DIFFERENCE (binary GID
should reduce to the paper's measure — "should" is what the tests check) and
the 2026 INTRINSIC_INFORMATION system measure for the φ_s bound. Any
combination whose property tests are not green ships outside the whitelist.

## Error handling

- Out-of-domain configuration → `ValueError` citing the paper and the failed
  condition.
- `n < 1`, empty mechanism/purview, invalid `bound` id → `ValueError`.
- `report(substrate=...)` with non-binary alphabet → `ValueError`.

## Testing

(a) **Formula verification (independent recomputation):**
- Brute-force Σ-formulas at small N (enumerate all (M, Z) pairs; compare
  against Eqs 6/7).
- Relation counts vs brute-force enumeration of overlap patterns (N ≤ 4).
- The Eq 14 linear-program solution vs `scipy.optimize.linprog` on small
  random instances (S(o), |𝒵(o)|) — independent check of the Σφ_r machinery.

(b) **Property tests (the certified bounds against the real pipeline):** for
the example networks and Hypothesis-generated small binary substrates,
compute full structures under each candidate (version, measure) combination
and assert: every distinction's φ ≤ |M||Z|; φ_s ≤ n(n−1); Σφ_d ≤ Bound I;
big_phi ≤ big_phi Bound I. These tests are the whitelist admission evidence.
The φ_r ≤ min φ_d check is included but flagged in the test docstring as
partially structural under 2.0's concrete-relations construction (min-based
by definition) — sanity, not independent evidence.

(c) **Reference goldens:** extract the Bound I/II/III values (Fig 3/Fig 4
curves, N = 2..9) from the author's IIT-bounds repo (run or transcribe its
bound computations), freeze as `test/data/bounds/reference_goldens.json`,
and assert our formulas reproduce them (the matching-layer reference-golden
pattern).

(d) **Bound III tightness (end-to-end measure parity):** build the
construction TPM at small N, compute the actual structure with 2.0, assert
achieved Σφ_e ≤ Bound III and within the paper's observed tightness. The
strongest check that 2.0's measure semantics match the paper's.

**Conjecture probes (non-gating):** violation-counting tests for Bounds
II/III over random/deterministic small substrates, reported but not failing
CI — a real-system violation of Bound III would be a genuine finding (its
generality is open), not a test bug.

## Files

- `pyphi/formalism/iit4/bounds.py` — new
- `test/test_bounds.py` — new (a, b, d + conjecture probes)
- `test/data/bounds/reference_goldens.json` — new (c)
- `papers/2024__zaeemzadeh-tononi__upper-bounds__s{1,2,3}-appendix.pdf` — added
- `changelog.d/zaeemzadeh-bounds.feature.md` — new
- `ROADMAP.md` — P13 rewrite (math correction + sub-project structure)

## Notes carried from brainstorming

- Approach: standalone functions module — no registry (the old module's
  registry pattern is not carried over), no config coupling, zero pipeline
  blast radius. Formalism-object integration (`ErrorInfo`) belongs to
  sub-project 2 with the pruning consumer.
- Pruning honesty: per-pair Theorem 1 bites only when best-so-far φ exceeds
  |M||Z| (rare at typical φ magnitudes); φ_s caps bite on small candidates
  and high-φ regimes. Whether integration pays is an empirical question for
  the sub-project-2 bite-rate study — and if it doesn't, this module is the
  P13 deliverable.
- Sub-project 2 may additionally consider the IIT 4.0 2026 Eq 23 cap
  (φ_s ≤ ii under the INTRINSIC_INFORMATION system measure) as a cheap
  certified candidate-skipping bound; it is a formalism-native cap, not a
  Zaeemzadeh bound, and lives with the pruning work.
