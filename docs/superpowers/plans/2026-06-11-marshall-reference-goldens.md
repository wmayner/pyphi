# Marshall 2024 Reference Goldens — Implementation Plan

**Goal:** Freeze the authors' ten committed result sets as regression
goldens per the approved spec
(`docs/superpowers/specs/2026-06-11-marshall-reference-goldens-design.md`).
No changes to `pyphi/` — test data, tests, changelog, ROADMAP only.

**Verified groundwork (from spec planning; do not re-derive):**
- The ten summaries are in `/tmp/marshall_results/*.txt`, verified
  bit-identical to the pinned upstream commit
  `48471b5d43e1453ac536cd4d3a5c48820cbe73cc`.
- File value counts: cg_micro 15, cg_macro 3, bbx_micro 251,
  bbx_macro 3, min_micro 3, min_macro 1, bu_micro 7, sfn/sfnn/sfs 15
  each (353 total). bbx_micro is missing the four 7-node subsets
  omitting one of A-D.
- bbx tier split: sizes 1-4 = 162 values (~16 s); size>=5 with
  |value| < 1e-13 = 61 values (short-circuit, ~3 min); size>=5
  genuinely nonzero = 28 values (hours; ABCDEFG verified at ~5e-18).
- Pre-measured battery-5 values: cg candidate {alpha} =
  0.007115237059108961; bbx candidates {alpha} = {beta} =
  3.867619951750597e-05; bbx {alpha, W={E,F,G,H}} =
  3.8676199517666156e-05 (equal to unapportioned at 1e-13 -- the Eq 29
  golden is end-to-end path coverage; the TPM-level effect is pinned
  by SP1's apportionment-bites test).
- Greek characters (phi, alpha, beta) appear in the DATA files and are
  matched in Python via `\\u` escapes only (ruff: no Unicode math in
  Python source).

**Standing constraints:** as SP2 (signed commits — plain `git commit`,
the container signs; targeted `git add`; `uv run --no-sync`; full
verification = `uv run --no-sync pytest` with no path + slow lane).

## Tasks

1. **Test data + provenance.** Copy the ten files to
   `test/data/marshall2024/<set>.summary.txt`; write `README.md`
   recording source repo, commit hash, fetch date, GPL-3.0 note, and
   the four known upstream discrepancies (cg_macro hand-entered TPM;
   bu stale config; bbx missing four; unclamped float noise). Commit.
2. **Parser + sanity battery.** `test/test_marshall_goldens.py`:
   `load_summary(name)` returning `{label: value}`; tests pin the ten
   per-set counts, the bbx missing-four list, and that every parsed
   label maps to node indices. Commit.
3. **Fast micro sweeps.** Parametrized per (set, subsystem) over
   cg_micro, min_micro, sfn/sfnn/sfs_micro (63 cases):
   `System(substrate, state, indices).sia().phi == approx(value,
   abs=1e-13)`. Substrates reused from existing fixtures
   (`CG_TPM`, `MIN_TPM`, `dancing_couples`). Commit.
4. **bu documented-deviation battery.** Asserts the file's claims
   (parser-read zeros) AND the pipeline's consistent-convention values
   (A=B=1.0, {C} raises `StateUnreachableForwardsError`, pairs 0.0,
   ABC matches the file). Commit.
5. **bbx tiered sweeps.** 3a (slow): sizes 1-4. 3b (slow): size>=5
   with |value| < 1e-13. 3c (skipif `PYPHI_MARSHALL_FULL_SWEEP` unset):
   the 28 nonzero large values. Run 3a+3b fully; validate 3c's
   mechanism on one selected case. Commit.
6. **Macro-network goldens.** cg literal hand-entered TPM (alpha =
   beta = 0.013601886288252735, pair = 1.0039763812908649); min
   literal TPM (alpha = 0.7883339770634886); bbx construction-TPM
   network (4.4001603967651364e-05, 4.400160396781154e-05,
   1.1183776016500528). Commit.
7. **Formalism-candidate goldens (project-recorded).** Via
   `MacroSystem`: cg {alpha} (and {beta}, recorded at execution,
   symmetry suggests equality); bbx {alpha}/{beta}; the Eq 29
   bbx apportioned candidate; cg exact-construction-TPM network
   singletons (recorded at execution). All labeled as 2.0-recorded,
   unpublished. Commit.
8. **ROADMAP + changelog + verification + push.** SP3 marked landed
   with the Eq 29 correction; `changelog.d/marshall-reference-goldens.misc.md`;
   full no-path pytest + slow lane; push.
