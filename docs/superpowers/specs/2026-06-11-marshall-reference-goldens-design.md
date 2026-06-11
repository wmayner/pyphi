# Marshall 2024 reference goldens — design

**Project:** Macro framework, sub-project 3 (of 3). Freeze the ten
result sets committed in the authors' repository
(github.com/CSC-UW/Marshall_et_al_2024) as regression goldens for the
2.0 pipeline, with verbatim provenance and explicit handling of the
known upstream discrepancies. Builds on SP1 (`pyphi/macro/` evaluation
machinery) and SP2 (criteria and search); changes neither.

**Sources of truth:**

- The authors' committed `results/*/summary.txt` files at repository
  commit `48471b5d43e1453ac536cd4d3a5c48820cbe73cc` (2025-09-22, the
  current main HEAD), fetched and re-verified bit-identical against
  the pinned-commit raw URLs during planning. Ten sets: `cg_micro`,
  `cg_macro`, `bbx_micro`, `bbx_macro`, `min_micro`, `min_macro`,
  `bu_micro`, `sfn_micro`, `sfnn_micro`, `sfs_micro`.
- The authors' example definitions
  (`marshall_intrinsic_units/marshall_intrinsic_units.py` at the same
  commit): every result value is a plain `Subsystem(network, state,
  subset).sia()` over either the micro network or a standalone macro
  network built from a macro TPM (hand-entered for cg, computed for
  bbx, hand-derived for min).
- SP1/SP2's established config mapping: `presets.iit4_2023` reproduces
  the authors' configuration; anchors assert at `abs=1e-13`.

## Planning findings (verified by experiment; the design hinges on these)

1. **The macro result sets are macro-network subsystems, not
   intrinsic-units candidates.** Their `cg_macro`/`bbx_macro`
   singleton values come from subsystems of a standalone 2-node macro
   network (the other macro node conditioned as background at the
   macro level). The intrinsic-units formalism's 1-unit candidate
   system over the micro universe is a different object and gives
   different values (cg: 0.007115237059108961 vs their
   0.013601886288252735; bbx: 3.867619951750597e-05 vs their
   4.4001603967651364e-05). Both interpretations reproduce cleanly in
   2.0 — the former via `MacroSystem`, the latter via `System` over a
   substrate built from the macro TPM — and both are frozen, clearly
   labeled.
2. **There is no published anchor for the apportionment path
   (Eq. 29).** Nothing in the authors' code apportions background.
   The conjecture that their macro singletons might encode it is
   refuted: apportioning the complementary pair to the candidate
   changes nothing at update grain 1 (Step-1 noising cannot feed back)
   and ~1e-15 at grain 2 (bbx: 3.8676199517666156e-05), nowhere near
   their committed singletons. The ROADMAP item-10 SP3 line claiming a
   "first published anchor for the nonempty-apportionment path" is
   corrected; in its place this project freezes a *project-recorded*
   (unpublished) Eq. 29 regression golden: the bbx 1-unit candidate
   with the complementary half apportioned.
3. **Known upstream discrepancies** (first two established in SP1/SP2,
   third found now):
   - `cg_macro` is built from the authors' hand-entered TPM containing
     a rounding (0.006833 for 0.0615/9) and a transcription error
     (0.9212 for 0.96^2 = 0.9216). Their values reproduce bit-for-bit
     from their literal TPM; the exact-construction TPM gives
     different values (pair: 1.0040208141253277, SP1's golden;
     singletons recorded at implementation time).
   - `bu_micro`'s committed singleton zeros were generated under old
     pyphi's `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` default
     (false), contradicting the authors' committed config and their
     other result sets; the consistent convention gives
     phi_s(A) = phi_s(B) = 1.0 and an unreachable {C} (SP2).
   - `bbx_micro` covers 251 of 255 nonempty subsystems: the four
     7-node subsets omitting one of A-D (`BCDEFGH`, `ACDEFGH`,
     `ABDEFGH`, `ABCEFGH`) are absent from their committed results.
     The sweep tests what is committed.
4. **Spot checks all reproduce at 1e-13** (sfn AB = 0.609069627439754,
   sfs ABC/ABCD, bbx macro pair and singletons including their
   1e-16-level alpha/beta asymmetry, which comes from numerical noise
   in the TPM construction itself and reproduces exactly; bbx 7-node
   ABCDEFG = 0.014989020336624108 reproduces at ~5e-18).
5. **The committed bbx large-subsystem values include unclamped float
   noise**: ~21 values of magnitude ~1e-17, several negative (old
   pyphi did not clamp phi at zero). At the working precision these
   are zeros; the goldens treat them as such.

## Scope

In scope:

- `test/data/marshall2024/`: the ten `summary.txt` files committed
  verbatim (UTF-8, Greek phi intact — data files, not Python), plus a
  `README.md` recording source repository, commit hash, fetch date,
  license note (their repo is GPL-3.0, same as PyPhi), and the known
  discrepancies.
- `test/test_marshall_goldens.py`: a parser for the summary format and
  the golden batteries below. Substrate/state definitions are imported
  from the existing SP1/SP2 test fixtures wherever they exist
  (`CG_TPM`, `_bbx_micro_tpm`, `MIN_TPM`, `dancing_couples`,
  `bu_substrate`); nothing is redefined.
- ROADMAP item-10 SP3 marked landed, with the Eq. 29 correction.

Out of scope: any change to `pyphi/`; parallelization of the sweeps;
the authors' pickle files (summaries carry every committed phi value).

## Test batteries

1. **Micro sweeps, fast lane** — `cg_micro` (15 values), `min_micro`
   (3), `sfn`/`sfnn`/`sfs_micro` (15 each): for every line in the
   summary, `System(substrate, state, subset).sia().phi` matches at
   1e-13. Parametrized per (set, subset) so failures localize.
2. **bu_micro, documented-deviation test**: asserts *both* sides of
   the discrepancy — the committed file says 0.0 for every 1- and
   2-unit subsystem (parser-read, so drift in the upstream file
   surfaces) while the pipeline gives 1.0/1.0 for {A}/{B},
   `StateUnreachableForwardsError` for {C}, 0.0 for pairs; and the
   uncontested ABC = 0.8300749985576875 matches. This pins the
   deviation permanently instead of skipping it.
3. **bbx_micro sweep, tiered by measured cost** (see Costs):
   - 3a (slow lane): sizes 1-4, all 162 committed values.
   - 3b (slow lane): the 61 size-5-to-8 values that are zero at
     precision — these short-circuit as reducible in seconds. This
     includes the ~21 committed values that are raw float noise
     (magnitudes ~1e-17, some negative — old pyphi did not clamp);
     at 1e-13 they are zero and the comparison treats them as such.
   - 3c (opt-in, off by default): the 28 genuinely nonzero size-5-to-8
     values, gated behind an environment variable
     (``PYPHI_MARSHALL_FULL_SWEEP``) because the irreducible large
     subsystems cost minutes-to-tens-of-minutes each (hours total).
     Verified sample during planning: ABCDEFG reproduces their
     0.014989020336624108 at ~5e-18.
4. **Macro-network goldens (config-mapping flavor)**: subsystems of
   the authors' literal macro networks through `System`:
   - cg: their hand-entered TPM -> alpha = beta =
     0.013601886288252735, pair = 1.0039763812908649 (pair already in
     SP1; singletons new here).
   - min: their hand-derived TPM -> alpha = 0.7883339770634886.
   - bbx: substrate from the exact construction's effect TPM (equal to
     their computed TPM to ~1e-16) -> alpha = 4.4001603967651364e-05,
     beta = 4.400160396781154e-05, pair = 1.1183776016500528.
5. **Formalism-candidate goldens (project-recorded, not published)**:
   the intrinsic-units 1-unit candidate systems via `MacroSystem`,
   labeled as 2.0-recorded values distinct from battery 4:
   - cg {alpha} = 0.007115237059108961 (and {beta}, recorded at
     implementation; symmetry suggests equality).
   - bbx {alpha} = {beta} = 3.867619951750597e-05 unapportioned.
   - **Eq. 29 regression golden**: bbx {alpha} with
     W = {E, F, G, H} apportioned = 3.8676199517666156e-05 — the first
     end-to-end regression coverage of the nonempty-apportionment
     path, explicitly marked as unpublished (no upstream anchor
     exists).
   - cg exact-construction macro-network singletons (battery 4's cg
     repeated on the *exact* TPM, values recorded at implementation) —
     the construction-vs-hand-entry delta made visible at the
     singleton level.
6. **Parser sanity**: per-set value counts (15/3/15/15/15/7/251/3/1/3),
   subsystem-label parsing against node labels, and the bbx missing-four
   list pinned.

## Files

- `test/data/marshall2024/{cg_micro,cg_macro,bbx_micro,bbx_macro,min_micro,min_macro,bu_micro,sfn_micro,sfnn_micro,sfs_micro}.summary.txt` — new (verbatim)
- `test/data/marshall2024/README.md` — new (provenance)
- `test/test_marshall_goldens.py` — new
- `changelog.d/marshall-reference-goldens.misc.md` — new
- `ROADMAP.md` — SP3 marked landed; Eq. 29 claim corrected

## Costs

Measured during planning (this container, sequential): bbx sizes 1-4
complete sweep = 162 subsystems in ~16 s. Reducible (phi = 0) large
subsystems short-circuit fast: a 5-node in 1.3 s, a 6-node in 2.1 s —
so battery 3b adds ~3 min. Irreducible large subsystems are the
expensive ones: the 7-node ABCDEFG took 556 s and the 8-node full
system exceeded 20 min before the measurement was capped; battery 3c
(28 values) is therefore estimated at several hours and stays opt-in.
The cg/min/sf*/bu micro sweeps and all macro-network goldens are
seconds each.

## Notes

- The summaries are data files: the no-Unicode-math rule applies to
  Python sources, not committed test data; the parser reads UTF-8 and
  matches the Greek phi explicitly.
- Both repositories are GPL-3.0; the verbatim files are attributed in
  the data README.
- Questions queued for the authors (running list): f(U^J, W^J) subset
  semantics (SP2), Example 1 TPM hand-entry error (SP1), bu config
  inconsistency (SP2), and now the four missing bbx 7-node subsystems
  (worth asking whether they failed or were skipped).
