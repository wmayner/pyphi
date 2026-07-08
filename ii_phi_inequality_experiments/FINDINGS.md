# min(ii_c, ii_e) ≥ φ_s under IIT 4.0 (2023)/GID: **refuted**

**Verdict (2026-07-08): the inequality is false under the 2023 formalism with
the GID system measure.** Violations exist at n = 2 with margins three orders
of magnitude beyond `config.numerics.precision`; a witness is pinned below and
independently reproduced from its saved TPM. Under the 2026 formalism
(`system_phi_measure="INTRINSIC_INFORMATION"`) the inequality holds by
construction (the Eq. 23 cap), and the witness confirms the cap strictly binds
there.

## Witness (minimal, n = 2)

State-by-node TPM (rows little-endian over (n0, n1); columns P(unit = 1)),
`cm = ones`, state `(0, 1)`, library defaults (IIT_4_0_2023, GID, precision 13):

```
[[0.90294049 0.74463958]
 [0.55496427 0.24027935]
 [0.5432427  0.42462247]
 [0.07088583 0.84413472]]
```

- φ_s (2023/GID) = **0.1719279417** (signed value equal; no clamping involved)
- ii_c = 0.9747150019, ii_e = **0.1178469739**
- margin = min(ii_c, ii_e) − φ_s = **−0.0540809678**
- MIP: directed set partition severing the single connection n0 → n1
- 2026-capped φ_s = 0.1178469739 = ii_e (the cap binds exactly)

Full-precision TPM and four further witnesses (three more at n = 2, one at
n = 3): `hunt_random_seed20260708.json.gz` (5 violations / 1960 records).

## Why it fails (the analytic account)

Write C for the complete directed partition (every unit a singleton part,
both directions cut) and n(θ) = Σᵢ|S⁽ⁱ⁾||X⁽ⁱ⁾| for a partition's normalizer.

1. **Normalizer lemma.** n(θ) = Σᵢ aᵢ(n − aᵢ′) is uniquely maximized by C at
   n(n−1). Since the MIP θ′ minimizes φ(θ)/n(θ), it follows that
   φ_s = φ(θ′) ≤ φ(C) · n(θ′)/n(C) ≤ φ(C) — an unconditional bound, but only
   through C: for any other reference partition the normalizer ratio can
   exceed 1, so no analogous bound exists.
2. **Reduction.** GID and ii share the selectivity factor and the specified
   state s′, so φ_d(C) ≤ ii_d reduces to q_C(s′) ≥ q̄(s′), where q_C is the
   complete-partition repertoire (a **product of per-unit noised means**) and
   q̄ is ii's unconstrained reference (`unconstrained_forward_effect_repertoire`:
   the **mean over mechanism states of the forward repertoires** — an average
   of products; on the cause side the reference is the constant mean, Eq. 32).
3. **Obstruction.** An average of products exceeds the product of averages
   whenever the per-unit conditionals are positively correlated through the
   shared current state (e.g. common-driver motifs at the aligned specified
   state, by power-mean convexity: ½(q_hi^k + q_lo^k) > ((q_hi+q_lo)/2)^k).
   There φ(C) > ii_e, so the only unconditional route fails.
4. **Realization.** The witness shows the failure is not merely a failure of
   the proof route: the actual normalized MIP lands on a single-edge cut
   (n(θ′) = 1 vs n(C) = 2) whose raw φ exceeds ii_e. Notably the pure
   common-driver grid itself never crosses the boundary (8712 records, minimum
   margin exactly 0.0) — the violations come from generic asymmetric
   substrates, so a sample of structured fixtures cannot certify the
   inequality (the prior 262/262 audit sampled exactly such fixtures).

## Consequences

- **Grain search ii-gating** (`docs/superpowers/specs/2026-07-07-grain-discovery.md`
  lever 2, ROADMAP Wave 7c): the ii-gated prune is **unsound as a certified
  prune under 2023/GID** — an ii below the incumbent does not imply φ_s below
  the incumbent. It remains **sound by construction under the 2026 cap**.
  Disposition: `prune="certified"` is available only under
  `system_phi_measure="INTRINSIC_INFORMATION"`; under GID the gate may ship
  only as an explicitly heuristic mode (or not at all), and the witness above
  belongs in the scheduler's test suite as the canonical would-have-pruned-
  wrongly case.
- **Formalism note.** The violations are minimal concrete cases where 2023
  φ_s exceeds the system's own intrinsic information — a system "more
  integrated than informative," the incoherence the 2026 revision's cap is
  designed to remove. They double as 2-unit cap-biting fixtures (the cap
  strictly binds with positive φ), far smaller than the `logistic3_k8`
  cap-biting network previously constructed for that purpose.
- The grain-discovery exploration's open question #1 is settled: **refuted**,
  not proved.

## Reproduction

- `hunt.py` — three-arm hunt (random / common-driver grid / adversarial
  coordinate descent), seeded (`np.random.default_rng`), raw per-record
  values (state, φ, signed φ, ii_c, ii_e, margins, full TPM) saved per arm as
  `hunt_<arm>_seed<seed>.json[.gz]`; files are never overwritten.
- Runs recorded here: seed 20260708 — `random` (300 substrates, 1960 records,
  5 violations, min margin −0.0541), `driver` (8712 records, 0 violations,
  min margin exactly 0.0), `adversarial` (6 restarts × 250 steps, min margin
  0.0017, no crossing — the selection-regime discontinuities stall coordinate
  descent near the boundary, so random sampling, not local search, found the
  violations).
- Independent witness reproduction: construct the substrate from the TPM
  above and compare `sia().phi` against
  `sia().system_state.effect.intrinsic_information`.
