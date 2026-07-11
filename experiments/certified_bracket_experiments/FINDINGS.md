# The partial-distinction certified Φ bracket: sound but not useful (negative result)

**Verdict (2026-07-11): The Approach-A certified bracket on Φ for an
incomplete distinction set is empirically *sound* — zero bound violations
across 5570 truncation points on 624 systems — but *not useful* for
early-stopping. Its median fraction of distinctions that must be computed
before the interval closes to within 2× of the true Φ is 1.000: it does not
close until the cause-effect structure is complete, at which point the exact
state-keyed identity already gives Φ for free. The dominant cause is
single-distinction sensitivity: dropping even one distinction into the
uncomputed set inflates the bracket width from a median 0.21× to a median
34.7× of the true Φ. Approach B (the density-budget LP) is not pursued: the
explosion is driven by the incidence-count term g(|𝒵(o)|), which a mass budget
does not tame.**

## Setting

The bracket brackets Σφ_r (distinctions are enumerated exactly, so Σφ_d is
handled by summation plus Theorem-1 `|m|·n` caps for un-evaluated candidates).
Approach A pins the computed distinctions `D_c` at their measured densities
inside the Zaeemzadeh Eq. 16 linear program and leaves only the uncomputed
budget free, capped at the certified `GENERAL` ceiling. Design in
`docs/superpowers/specs/2026-07-11-partial-distinction-certified-phi-bracket-design.md`;
computable core in `bracket.py`. Notation follows
`experiments/so_certificate_experiments/FINDINGS.md`.

The experiment computes each system's full CES once for ground truth, then
sweeps truncations `k = 0…|D|` under two computation orders — **oracle** (by
true φ_d descending) and **cheap** (by `|m|·n` descending) — recording the
certified interval `[L_Φ, U_Φ]` and its soundness at each step. Fixtures
`pqr`/`grid3`/`residue`/`basic` plus 120 seeded random substrates
(`np.random.default_rng`, `n = 2–4`), library defaults (IIT_4_0_2023, GID).

## Results

Run `certified_bracket_seed20260711_trials120.json.gz` (624 records, 5570
sweep points):

**Soundness — passes unconditionally.** 0 soundness violations
(`true Φ ∈ [L_Φ, U_Φ]` at every truncation of every record). The wildcard
construction is a valid certified upper bound, confirming the §2.3 argument
empirically.

**Usefulness — fails.** Median fraction of distinctions computed before the
interval width falls within `target × true Φ`, over the 624 records with
Φ > 0:

| target | oracle | cheap |
|-------:|:------:|:-----:|
| 0.5×   | 1.000  | 1.000 |
| 1.0×   | 1.000  | 1.000 |
| 2.0×   | 1.000  | 1.000 |

The interval only reaches a useful width at fraction 1.000 — the moment the
CES is complete. The oracle and cheap orders are identical, so the failure is
fundamental, not an ordering artifact.

**Mechanism — single-distinction sensitivity.** The bracket width relative to
the true Φ (oracle order):

- With the **complete** distinction set (`M_u` empty): min 0.00, **median
  0.21**, max 1.82 — tight; all 624 within 2×, 608/624 within 1×. (This is the
  measured state-keyed certificate residual; it is the dominated regime, since
  the exact identity gives Σφ_r there in the same O(|D|·n).)
- With **one distinction dropped** into `M_u`: min 2.37, **median 34.7**, max
  1131.6 — and in **all 624/624** records this single drop pushes the width
  above 2× the true Φ.

A single un-evaluated candidate mechanism `m` contributes `|m|·n` to the
uncomputed mass and, more damagingly, one extra incidence to every unit-state
group, so the per-o weight jumps from `g(k_c)` to `g(k_c + 1)` with
`g(k) = (2^k−1−k)/k` roughly doubling per unit. This exponential incidence
inflation dominates before the last distinction lands.

## Consequence for the ROADMAP gate

The Wave 7 "anytime certified Φ bracket" build is **resolved negative for the
partial-distinction case under Approach A**. The construction is a correct
certified bound but has no early-stopping value: it is loose until the CES is
complete, and once complete the exact identity supersedes it (as the S(o)
FINDINGS scope correction already established). No `bounds.py` implementation
follows.

**Approach B is not pursued.** The density-budget LP was the tightening path if
A proved too loose. But this experiment localizes the looseness to the
*incidence-count* term `g(|𝒵(o)|)`, not the density budget: an uncomputed
distinction of unknown purview can be incident to any unit-state, so `|𝒵(o)|`
still jumps by one per uncomputed distinction regardless of how its density
mass is constrained. A mass budget cannot cap the incidence count, so B faces
the same exponential and is not expected to close early either. Establishing a
useful partial-distinction bracket would require *a priori* structural
constraints on which unit-states an uncomputed mechanism can specify — theory
neither the paper nor this experiment provides.

**What remains sound and useful** is the *complete*-distinction certificate
itself: the measured state-keyed upper bound on Σφ_r (median 0.21× width, the
S(o) FINDINGS' median 1.45× tightness on Σφ_r) is ~100–1000× tighter than the
shipped `sum_phi_relations_upper_bound(n, "GENERAL")`. Exposing *that* as a
measured certified bound in `bounds.py` — the modest, fully-proved core of the
original scope — is a separate, still-open opportunity, distinct from the
anytime-bracket ambition this experiment closes.

## Reproduction

```
uv run python experiments/certified_bracket_experiments/verify_certified_bracket.py --seed 20260711 --trials 120
uv run python experiments/certified_bracket_experiments/analyze.py experiments/certified_bracket_experiments/certified_bracket_seed20260711_trials120.json.gz
```

`bracket.py` (Approach A core, unit-tested in `test_bracket.py`),
`verify_certified_bracket.py` (truncation sweep, seeded, raw records saved),
`analyze.py` (fraction-to-close summary). Results JSON never overwritten.
