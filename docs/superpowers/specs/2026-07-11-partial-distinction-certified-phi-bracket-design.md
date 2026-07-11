# A certified Φ bracket for partial distinction sets

**Status:** design (spike). Experiment gates any `bounds.py` implementation.

## 0. Executive summary

A two-sided *certified* bracket on the IIT 4.0 Φ that is valid before the
cause-effect structure is complete — i.e. from a computed subset of the
distinctions `D_c ⊊ D_full`, with the remaining candidate mechanisms `M_u`
un-evaluated. The goal is an **anytime** guarantee: a certified interval
`[L_Φ, U_Φ]` that brackets the true Φ at every truncation point and tightens
monotonically as distinctions are computed, enabling early decisions such as
"is this complex's Φ above threshold?" without finishing the full computation.

This is a **research spike**, not a feature commitment. The upper endpoint on
Σφ_r for the partial case is unproven-until-measured: it may collapse toward
the loose worst-case ceiling until nearly all distinctions are computed, in
which case it is useless for early-stopping. A confirmation experiment settles
this **before** any code lands in `bounds.py`, per the project rule against
locking state onto unconfirmed assumptions.

## 1. Why the partial case, and why it is the only interesting case

Φ = Σφ_d + Σφ_r. Two independent "partial" axes exist; only one motivates a
bracket:

- **Partial relations, complete distinctions.** With every distinction
  resolved, the state-keyed identity (Eq. 11, proved in
  `experiments/so_certificate_experiments/FINDINGS.md`) reconstructs Σφ_r
  *exactly* in O(|D|·n) from the distinctions alone — this is precisely
  PyPhi's `CauseEffectStructure.sum_phi_relations` via `AnalyticalRelations`.
  So Φ is already exact and cheap; a bracket, and any streaming of concrete
  relations to raise a lower bound, is strictly dominated. **Not the target.**

- **Partial distinctions.** Computing the distinctions (each mechanism's MICE)
  is the expensive part of a CES, and it is where an anytime certified bracket
  on Φ could pay off. **This is the target.**

The complete-distinction bracket sketched in the meta-theory spec
(`2026-07-07-formalism-meta-theory-and-certified-approximation.md` §6.2) is the
dominated case above; the partial-distinction bracket here is new.

## 2. The construction (Approach A: wildcard worst-case)

Notation follows the S(o) FINDINGS: `o` ranges over UnitState pairs;
`𝒵(o)` = distinctions whose `purview_union` contains `o`;
`q_d = φ_d / |purview_union_d|`; `S(o) = Σ_{d∈𝒵(o)} q_d`;
`g(k) = (2^k − 1 − k)/k`.

At a truncation point, `D_c` is the set of computed distinctions and `M_u` the
un-evaluated candidate mechanisms (only `|m|` known for each). `n` is the unit
count.

### 2.1 Σφ_d — certified both sides

- **Lower:** `L_d = Σ_{d∈D_c} φ_d` (each computed φ_d exact and ≥ 0).
- **Upper:** `U_d = L_d + Σ_{m∈M_u} |m|·n`. Theorem 1 (Zaeemzadeh & Tononi
  2024) gives `φ(m, Z) ≤ |M||Z|`; maximizing the purview size `|Z| ≤ n` gives
  `φ_d(m) ≤ |m|·n` for each un-evaluated candidate. Evaluated mechanisms that
  yielded φ = 0 are not distinctions and are excluded from `M_u`, so they add
  nothing.

Both endpoints tighten monotonically as mechanisms move from `M_u` into `D_c`.

### 2.2 Σφ_r — lower endpoint

`L_r = identity(D_c)`: the exact state-keyed Eq. 11 sum over relations *among
the computed distinctions*. Because every φ_r ≥ 0 and `D_c ⊆ D_full`, relations
among a subset are a subset of all relations, all non-negative, so
`identity(D_c) ≤ Σφ_r(D_full)`. This is a certified lower bound, exact in
O(|D_c|·n), and it dominates any partial sum obtained by streaming concrete
relations in decreasing-φ_r order — so no relation-streaming machinery is
required for the lower side.

### 2.3 Σφ_r — upper endpoint (the crux)

Split Σφ_r into the self-relation term and the cross term.

**Self term.** `Σ_d |z*_c(d) ∩ z*_e(d)| · q_d`. For `d ∈ D_c` it is exact
(carried, not bounded — the FINDINGS Correction 2). For each `m ∈ M_u`,
`selfrel(m) ≤ φ_d(m) ≤ |m|·n` (the intersection size never exceeds the union
size), so the self term's upper endpoint adds `Σ_{m∈M_u} |m|·n`.

**Cross term — the wildcard construction.** The certified worst-case ceiling
is the Eq. 16 growth bound (`sum_phi_relations_upper_bound(n, "GENERAL")`),
built from the Eq. 14 per-o linear-program maximum with `S(o) ≤ n·2^(n-1)` and
`|𝒵(o)| ≤ 2^n − 1`. Approach A **interpolates** between that ceiling and the
measured state-keyed certificate by *pinning the computed distinctions*:

> Take the Eq. 16 LP, but enter every computed distinction `d ∈ D_c` at its
> **exact measured** contribution — its density `q_d` in each `o` its
> `purview_union` contains — and leave only the **uncomputed budget** free to
> maximize: total density mass `Σ_{m∈M_u} |m|·n`, distributable across at most
> `2n` unit-states and contributing at most `|M_u|` extra incidences to any one
> `o`'s group before `g` is applied.

The free maximization over the uncomputed budget uses the same Eq. 14 machinery
(`_grouped_subset_min_sum` / `g`) that `bounds.py` already implements.

**Why this is the right "A".** The construction is provably sandwiched:

- As `M_u → ∅`, the free budget vanishes and `U_r →` the exact measured
  state-keyed certificate (proved tight in the FINDINGS, median 1.45×).
- As `D_c → ∅`, nothing is pinned and `U_r →` the `GENERAL` ceiling.
- For any intermediate truncation, `U_r ≤ GENERAL` (some mass is pinned to
  measured-tight values rather than the worst case), and `U_r ≥` the measured
  certificate on `D_c` alone (the uncomputed budget only adds).

Mass conservation — each distinction's total density summed over the `o`'s it
touches equals `φ_d` — is inherited from the `GENERAL` budget, so the
`|purview_union| = 1` density-blowup that a naive per-distinction wildcard would
suffer does not arise: `U_r` can never exceed `GENERAL`.

### 2.4 The Φ bracket

`[L_Φ, U_Φ] = [L_d + L_r, U_d + U_r]`, certified to contain the true Φ at every
truncation point, tightening monotonically as `M_u` shrinks.

**This construction is a hypothesis.** Its soundness (`true Φ ∈ [L_Φ, U_Φ]`
always) and its usefulness (does it close early?) are settled empirically in
§3 before any implementation. The `bounds.py` proof text is written only once
the experiment confirms zero violations.

## 3. The confirmation experiment

New directory `experiments/certified_bracket_experiments/`, following the S(o)
harness conventions: seeded `np.random.default_rng`, seed saved in output, raw
per-record data saved (never only aggregates), output files never overwritten
(incrementing `_v2`, `_v3`, …).

### 3.1 Procedure

For each system — fixtures `pqr_system`, `grid3_system`, `residue_system`,
`basic_system`, plus seeded random substrates at `n = 2–4` (reusing the S(o)
harness's substrate generation) — compute the full CES once for ground truth
(`Σφ_d`, `Σφ_r`, `Φ`, and every distinction's `φ_d` and `purview_union`).

Then sweep a **truncation** `k = 0 … |M|`: the first `k` mechanisms (under a
chosen order) form `D_c`, the rest form `M_u`, and record the certified
`[L_Φ, U_Φ]` from Approach A at each `k`.

### 3.2 Two computation orders

To separate "is the method capable" from "is it realistic":

- **oracle** — mechanisms by true `φ_d` descending. The best case; upper limit
  on achievable tightness at any fraction computed.
- **cheap-priority** — mechanisms by `|m|·n` descending. The only ordering
  knowable in advance; the realistic order.

The oracle-versus-cheap gap distinguishes a fundamental negative result from a
mere ordering problem.

### 3.3 Recorded per (system, state, order, k)

`L_Φ`, `U_Φ`, bracket width `U_Φ − L_Φ`, tightness `U_r / true_Σφr`,
`bound_holds` (`true_Φ ≤ U_Φ + tol` and `true_Φ ≥ L_Φ − tol`), and whether the
interval resolves a threshold `Φ > c`. Aggregates plus the raw per-record
values, saved to seeded JSON.

### 3.4 The question, and the honest null

*At what fraction of distinctions computed does the bracket first become
useful* — width within a target factor (e.g. 2×) of the true Φ, or resolving a
Φ-threshold decision? The null hypothesis is that it does **not** close until
nearly all distinctions are computed, making it useless for early-stopping.
Soundness is a hard gate independent of usefulness: any `bound_holds = False`
record refutes the construction and blocks implementation until the hole is
found.

## 4. Conditional implementation

**If the experiment shows the bracket closes usefully** (and zero soundness
violations): a thin `certified_big_phi_bracket(distinctions,
uncomputed_mechanisms, n) → Bracket` in `pyphi/formalism/iit4/bounds.py`, with
a two-endpoint `Bracket` value type beside `UpperBound` (fields: `lower`,
`upper`, `certified`, `assumptions`, `citation`). It reuses
`_grouped_subset_min_sum` / `g` and the identity reconstruction. Nothing is
wired into the hot CES computation path in this spike; the function is a pure
query over a distinction set plus an uncomputed-mechanism set.

**If the result is negative:** record it in the experiment's `FINDINGS.md`, flip
the ROADMAP Wave 7 "anytime certified Φ bracket" row from open build to the
measured verdict **in the same commit** (per the CLAUDE.md gate-update rule),
and stop — no `bounds.py` change. Either outcome produces a durable, cited
result.

## 5. Scope boundaries

- **4.0 only.** Zaeemzadeh's machinery is 4.0-specific; no certified upper
  bracket exists for IIT 3.0 Φ (meta-theory spec §6.3).
- **Certified domain only.** Binary units, conditionally independent TPM,
  GID/II measure — the existing `bounds.py` domain guards apply unchanged.
- **Approach B (density-budget LP) is out of scope** unless Approach A proves
  too loose *and* the measured structure looks favorable; it is deferred future
  work, not part of this spike.
- **No hot-path integration.** The bracket is a standalone query; driving it
  from a live CES computation (feeding it truncation state as distinctions are
  produced) is a separate follow-up.
