# P12b Unification Amendment

**Date:** 2026-05-24
**Amends:** `2026-05-24-p12b-multivalued-units-design.md` (committed `8d0387c9`)
**Status:** Pending user approval; supersedes specific sections of the original spec where conflicts arise.

## Why this amendment exists

During execution of the original P12b plan, Task 6 surfaced a math/shape mismatch between the legacy SBN-form cause-TPM bridge (`_legacy_backward_tpm`) and the native k-ary cause path (`_cause_tpm_factored_kary`). The two paths were intended to coexist (binary preserves SBN form for byte-identical goldens; k-ary uses joint-posterior shape). Investigation revealed:

1. **The two paths compute fundamentally different mathematical objects.** Legacy produces a state-by-node form (`forward × likelihood / norm`, shape `(*α, n)`, sums to N). The native k-ary path as originally implemented (Task 5) produces a partial-likelihood-only joint posterior with no background marginalization. These are not just different representations of the same object.

2. **Task 1's audit (`p12b-cause-shape-audit.md`) was partially incorrect.** It claimed Option B canonical shape `(*α, n_observed_nodes)` for both paths. The native k-ary path does not naturally produce this shape, and forcing it to would lose mathematical meaning (k-ary nodes have no single firing probability).

3. **Phase 0 catalogue (`p12b-sbn-consumer-catalogue.md`) identified 21 consumers** of SBN-form cause-TPM data, 10 of them category B (real SBN-semantics dependency). The dual-path approach would require shape-dispatched code at every site, propagating fragility.

4. **Phase 1 math analysis (`p12b-unification-math-analysis.md`) proved unification is feasible.** The unified algorithm (joint posterior with `pr_bg / norm` background weighting, per-node marginalize, product, normalize, broadcast) was verified byte-identical (`atol=1e-10`) to `cause_repertoire(cs, mech, purv)` across all 76 subsystem × mechanism × purview combinations on `basic_system`. Phase 1 also found that **Task 5's `_cause_tpm_factored_kary` was mathematically incomplete** — it omitted background-state conditioning, silently corrupting subset-system computations.

5. **Reading IIT 4.0 (Albantakis et al. 2023) confirms** that the system-level cause TPM (Eq. 4) is structurally factored per output unit, identical to the forward TPM (Eq. 2's conditional independence assumption applied in the cause direction). The legacy SBN form `(*α, n)` is the binary-collapsed form of the natural alphabet-generic shape `(*α, n, k_per_unit)`. T_c and T_e have the same structural type.

The original spec's dual-path design was based on an imprecise reading of the formalism. This amendment revises the design to unify the two TPMs under a single structural type, eliminating the dual-path fragility and aligning code structure with the paper's mathematical structure.

## Revisions

### Revision 1: Type hierarchy unified on `FactoredTPM` (supersedes §3.2)

**Original §3.2** introduced `JointDistribution` as a base class with `JointTPM` and `CausePosterior` as sibling subclasses, on the rationale that the cause posterior was a "joint distribution that does not factor over past nodes."

**Amendment.** Per Eq. 4 of IIT 4.0, the system-level cause TPM **does factor over output (system) units** under the conditional-independence assumption applied to the cause direction:

```
p_c(s | s̄) = ∏_{i=1}^{|S|} Σ_{w̄} p(s_i | s̄, w̄) · (Σ_s p(u|ŝ,w̄) / Σ_u p(u|ū))
```

The outer product is the cause-direction analog of Eq. 2's forward conditional independence. The trailing axis on the legacy SBN form carries the per-output-system-unit marginal cause probability — exactly what the FactoredTPM trailing axis carries on the effect side.

**T_c and T_e are structurally identical.** Both are factored per output unit; both have the same shape `(*α_input_state, n_output_units, k_per_unit)` or equivalent per-unit factors list.

**Type hierarchy revisions:**

- `JointDistribution` (base, retained from Task 2) — remains useful as a base for joint tensor storage (e.g., `JointTPM`'s SBS form), but is no longer the parent of `CausePosterior`.
- `JointTPM(JointDistribution)` — retained for the joint state-by-state representation. Used internally by some computations and for substrate-level forward TPM conversions.
- **`CausePosterior` — RETIRED.** The class is deleted. Cause TPMs are returned as `FactoredTPM`.
- `FactoredTPM` — becomes the canonical type for ANY conditional-independence-factored distribution. Both forward (T_e) and cause (T_c) TPMs are `FactoredTPM`.

The work landed in Tasks 1-2 (audit, `JointDistribution` extraction) is retained — `JointDistribution` is still a useful base class even if its sibling-of-FactoredTPM role goes away. Tasks 3-4 (added `CausePosterior` and wrapped cause_tpm in it) become superseded; the re-plan undoes these in a controlled goldens-stable sequence.

### Revision 2: Unified cause-TPM math (supersedes §3.3)

**Original §3.3** said the binary path stays on `_legacy_backward_tpm` for byte-identical golden preservation; the native k-ary path produces a joint posterior `(*α,)`; downstream consumers dispatch by `isinstance` or shape.

**Amendment.** Both binary and k-ary use a single unified path that implements Eq. 4. The per-system-unit factor for unit `i` is:

```
factor_i(s̄)[s_i] = Σ_{w̄} p(s_i | s̄, w̄) · (Σ_s p(u | ŝ, w̄) / Σ_u p(u | ū))
```

For binary substrates, this reduces algebraically to what `_legacy_backward_tpm` computes (per Phase 1's empirical verification, byte-identical cause repertoires across all subsystem combinations on `basic_system`). For k-ary, it generalizes naturally without needing the SBN-form indirection.

**`_legacy_backward_tpm` is retired** once consumer migration completes. The function may temporarily remain in `pyphi/tpm.py` during the migration window, but no production code path calls it at completion.

**`_cause_tpm_factored_kary`'s math is fixed early.** The Task 5 implementation omits background weighting; the re-plan's first substantive task adds the `Σ_{w̄}` and `pr_bg / norm` terms before any consumer migration begins.

### Revision 3: Q1–Q7 design decisions baked in

The Phase 1 math analysis enumerated seven open questions. User-approved answers:

**Q1: `System.proper_cause_tpm`.** Redesigned to recompute T_c for the specified subsystem rather than slice the substrate-level T_c. The current slice-the-trailing-axis optimization is binary-specific and does not generalize. If performance regressions show up, add a cache; do not preserve the slice-based optimization speculatively.

**Q2: `Node.cause_tpm_on` / `cause_tpm_off`.** Retired. The replacement is `Node.cause_factor` (alphabet-generic), indexed via `cause_factor[..., state]` for any alphabet size. For binary, the new accessor returns shape `(*α, 2)` carrying `(P(off), P(on))` explicitly rather than the legacy `P(on)`-with-implicit-complement form. The `np.stack([off, on], axis=-1)` construction in `Node.__init__` is replaced by direct extraction of the alphabet axis from the system-level FactoredTPM.

**Q3: `pr_bg / norm` caching strategy.** Computed once per `cause_tpm(substrate, current_state, system_indices)` call, scoped locally to the function. No cross-call cache. `System.cause_tpm` remains a cached property at the System level (existing pattern), which provides sufficient memoization.

**Q4: Fix `_cause_tpm_factored_kary` math timing.** Fix EARLY, before consumer migration. The Task 5 commit remains in history but its math is corrected in the re-plan's first substantive task. After the fix, `_cause_tpm_factored_kary` returns the same value as `_cause_tpm_factored_binary` on binary inputs (byte-identical, the binary-equivalence property becomes well-defined).

**Q5: `tpm_indices()` semantics.** Move from class-level heuristic (`np.where(shape == 2)`) to per-instance metadata. `JointTPM.tpm_indices()` returns `range(ndim - 1)`. `FactoredTPM.tpm_indices()` returns the system-unit index range (skipping trailing alphabet and unit axes). Per-class implementations replace the binary-specific shape grep.

**Q6: Macro sites.** Deferred entirely. The macro code (`pyphi/macro.py`) is dead under current `MacroSystem`; reviving it is out of P12b scope. Notes in the re-plan flag this as future work.

**Q7: Substrate-level cache vs per-node recompute.** Substrate-level cache, status quo. `System.cause_tpm` remains a cached property; per-node lookups derive from the cached T_c. Phase 1's algorithm validates this access pattern.

### Revision 4: Strengthened goldens gate (supersedes §7.11)

**Original §7.11** specified byte-identical goldens at the array level for binary substrates.

**Amendment.** The acceptance gate is:

> **Cause repertoires** (the end-to-end output of `cause_repertoire(cs, mech, purv)` over all subsystem × mechanism × purview combinations) MUST remain byte-identical within `atol=1e-10` at every commit boundary. **Intermediate array layouts** (e.g., legacy SBN form `(*α, n)` vs. explicit alphabet axis `(*α, n, k)`) MAY change without violating the gate, provided end-to-end cause repertoires are preserved.

**Drift beyond `atol=1e-10` is treated as a bug in the change, not a reason to regenerate goldens.** Regeneration requires (a) an explicit algebraic derivation showing the legacy values were a quirk, (b) user approval per-instance before commit, (c) documentation in the changelog with the derivation cited.

The empirical evidence from Phase 1 (76/76 subsystem × mechanism × purview combinations on `basic_system` byte-identical) is the proof-of-concept for the unified algorithm. The migration plan must preserve this property at every commit boundary, not just at the end.

## Implications for previously-landed commits

Tasks 1-5 already landed on the `feature/p12b-factored-kary` branch. Under this amendment:

- **Task 1 audit (`5727f01b`)** — Kept as historical record. Its conclusions (Option B trailing-axis canonical shape) are partially superseded by this amendment's findings. Re-plan adds a "supersedes" pointer.
- **Task 2 `JointDistribution` extraction (`55b135fb`)** — Kept. `JointDistribution` remains useful as a base for `JointTPM`'s joint-tensor storage. The sibling-of-FactoredTPM role goes away, but the class itself stays.
- **Task 3 `CausePosterior` addition (`ffecc9d6`)** — Superseded. Re-plan retires the class; `pyphi/core/tpm/cause_posterior.py` and `test/test_cause_posterior.py` are deleted.
- **Task 4 `cause_tpm` wraps in `CausePosterior` (`e85fa73f`)** — Superseded. Re-plan changes `cause_tpm`'s return type to `FactoredTPM`. Test updates (`np.asarray` etc.) made in this commit may need to be re-checked but mostly stand.
- **Task 5 native k-ary cause path (`4c00f64c`)** — Math is wrong. Re-plan fixes it as its first substantive task: add `Σ_{w̄}` background marginalization, add `pr_bg / norm` weighting, change return type to `FactoredTPM`.

Worktree state at amendment time: `e4985a2d` (Phase 0 catalogue), `9a353a61` (Phase 1 math analysis), with this amendment to follow.

## Supporting evidence

- **Phase 0 catalogue:** `docs/superpowers/audits/p12b-sbn-consumer-catalogue.md` (commit `e4985a2d`). 21 consumers enumerated; 10 category B (real SBN-semantics dependency); 3 highest-risk sites identified.
- **Phase 1 math analysis:** `docs/superpowers/audits/p12b-unification-math-analysis.md` (commit `9a353a61`). Empirical verification of unification feasibility; identification of Task 5's math gap; per-consumer invariance categorization.
- **IIT 4.0 paper (Albantakis et al. 2023):** `papers/2023__albantakis-et-al__iit-4.0.pdf`. Key equations: Eq. 2 (forward conditional independence), Eq. 3 (T_e), Eq. 4 (T_c with structural factorization over system units), Eq. 28-33 (mechanism-level computations consuming T_c).

## Out of scope for this amendment

- **Macro substrate handling** (`pyphi/macro.py`). Macro code is dead under current `MacroSystem`; revival is a separate project.
- **`FactoredDistribution` as a shared base** for `FactoredTPM` if future types are added. Defer until evidence accrues.
- **Eliminating `JointTPM`'s SBS form.** Used for legacy interop and intermediate computations; retained.
- **AC (actual causation) cause-TPM analog.** AC has its own cause-side computation in `pyphi/actual.py::TransitionSystem`. The amendment applies the same structural-factorization principle there, but specific code changes are enumerated by the re-plan.
