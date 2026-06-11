# Marshall et al. 2024 committed result sets

The `*.summary.txt` files in this directory are verbatim copies of the
`results/*/summary.txt` files from the paper authors' repository:

- **Source:** https://github.com/CSC-UW/Marshall_et_al_2024
- **Commit:** `48471b5d43e1453ac536cd4d3a5c48820cbe73cc` (2025-09-22)
- **Fetched:** 2026-06-11, re-verified bit-identical against the
  pinned-commit raw URLs
- **License:** GPL-3.0 (same as PyPhi); files reproduced with
  attribution
- **Paper:** Marshall, Findlay, Albantakis, Tononi (2024), "System
  Integrated Information" / intrinsic units,
  https://doi.org/10.1101/2024.04.12.589163

Each value is φ_s of one subsystem, computed by the authors with
`Subsystem(network, state, subset).sia()` under old pyphi at their
pinned revision `wmayner/pyphi@941c65a`. The micro sets analyze the
papers' example networks directly; the macro sets analyze standalone
macro *networks* built from their macro TPMs (macro-level background
conditioning — **not** the intrinsic-units candidate-system
construction; see `pyphi.macro`).

These files are consumed by `test/test_marshall_goldens.py`.

## Known upstream discrepancies (do not "fix" the files)

1. **`cg_macro`** was computed from a hand-entered macro TPM containing
   a rounding (`0.006833` for `0.0615/9`) and a transcription error
   (`0.9212` where the construction gives `0.96**2 = 0.9216`). The
   values reproduce bit-for-bit from the literal TPM; the exact
   construction gives different values (see the golden tests).
2. **`bu_micro`'s** 1- and 2-unit zeros were generated under old
   pyphi's `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` default
   (false), contradicting the authors' committed `pyphi_config.yml`
   (true) and their other result sets, which require true. Replaying
   their pinned pyphi with their committed config gives
   φ_s(A) = φ_s(B) = 1.0 and an unreachable state for {C}.
3. **`bbx_micro`** covers 251 of the 255 nonempty subsystems: the four
   7-node subsets omitting one of A–D (`BCDEFGH`, `ACDEFGH`,
   `ABDEFGH`, `ABCEFGH`) are absent from the authors' results.
4. Some committed large-subsystem values are unclamped floating-point
   noise (magnitudes ~1e-17, several negative); at PyPhi's working
   precision these are zeros.
