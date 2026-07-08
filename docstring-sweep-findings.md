# Docstring sweep — findings log

Byproducts of the 2026-07-08 docstring sweep. The sweep itself edited only
docstrings; the items below are things it *surfaced* and are recorded here for
separate follow-up, not fixed as part of the sweep (with two small exceptions,
noted under "Code touched").

## Code-level bugs found (reported, NOT fixed)

1. **`pyphi/distribution.py` `purview()`** identifies purview nodes by a size-2
   dimension (`dim == 2`). With non-binary nodes now supported
   (`repertoire_shape`/`max_entropy_distribution` take `alphabet_sizes`), a
   k-ary purview node has dimension > 2 and is silently dropped. Latent bug for
   k-ary repertoires. The docstring now states the binary assumption.
2. **`pyphi/dynamics.py`** `most_probable_next_state`/`settle` use `> 0.5` for
   the ON threshold, so P(ON) = 0.5 exactly is treated as OFF. The docstring now
   states the rule; confirm it is intended.
3. **`pyphi/actual.py`** (`Transition.repertoire`, ~line 563): user-facing
   `ValueError` message reads "is no a {direction} mechanism" (should be
   "is not a"). Code string, not a docstring.
4. **`pyphi/utils.py:396` `enforce_integer()`**: the raised message is
   hardcoded "{name} must be a positive integer", but the function checks
   `i < min` with `min` defaulting to -inf. The message is wrong when `min` is
   not 0/1.
5. **`pyphi/core/tpm/joint_distribution.py` `JointDistribution.print()`**
   (~line 262) is a no-op: it builds a multidimensional TPM then iterates
   `for _state in ...: pass`, printing nothing.
6. **`pyphi/models/cmp.py:128`** `numpy_aware_eq`: pre-existing
   `# TODO: this is broken if the iterables are sets` — the Iterable branch uses
   `len()`/`zip()`, which misbehaves for set inputs.
7. **Two distinct classes named `JointTPM`, exported inconsistently.**
   `pyphi/core/tpm/joint_distribution.py` defines the full backing `JointTPM`;
   `pyphi/core/tpm/joint.py` defines a thin wrapper class *also* named
   `JointTPM`. The top-level `pyphi.__init__` re-exports the backing class
   (`pyphi.JointTPM`), while `pyphi.core.tpm.__init__` re-exports the wrapper
   (`pyphi.core.tpm.JointTPM`) — so the same name resolves to different classes
   by import path. This is an API-naming footgun (and it makes bare `JointTPM`
   cross-references ambiguous in the docs). Consider renaming one (e.g. the
   wrapper) so the public name is unambiguous.

(The 46 docstring-vs-code disagreements the sweep *corrected* — reversed
connectivity-matrix semantics, wrong equation citations, wrong return types,
stale parameter names — are in the commit diffs and the workflow report, not
repeated here.)

## Docstrings left unconfirmed

- `pyphi/examples.py` `fig16_substrate`: docstring says "Figure 5B of the 2014
  IIT 3.0 paper" but the function is named `fig16_substrate`. Possible
  name/figure mismatch; needs an author check against Oizumi et al. (2014).
- `pyphi/examples.py` `frog_example`'s inner `get_net()` helper has a Google
  Args block listing parameters not in the signature (`gridsize`) and functions
  not implemented here (`MvsG`). Left unchanged — behavior unclear.
- A handful of pre-existing citations were preserved but not independently
  re-verified against the PDFs (relations Eq. 49/56; conf `Eq. 23 cap` and
  `Eq. 4`; AC `Definition 1 outcome 2`; the resonator coupling-factor appendix
  reference). See the workflow report for the full list.

## Format deviations (agent judgment)

Three files were left in Google-style sections rather than converted to NumPy,
by the rewriting agent's judgment that conversion was pure churn with
transcription risk and no accuracy gain:

- `pyphi/estimate.py`, `pyphi/provenance.py` — rich multi-paragraph docstrings.
- `pyphi/models/__init__.py` — a large `Attributes:` alias block.

These render correctly (napoleon parses Google style), but they break the
uniform-NumPy goal. Convert in a follow-up if uniformity is wanted. Note: this
means `napoleon_google_docstring = False` (planned as the final docs step)
cannot be set until these are converted.

## Code touched by the sweep (the two exceptions to prose-only)

1. **`pyphi/models/state_specification.py`**: the class attribute
   `desc = "functions for normalizing distinction |small_phi| values"` had its
   substitution markup replaced with `φ`. This is a display string (shown in
   registry listings), where `|small_phi|` would render as literal garbage — so
   the change is a correctness improvement — but it is a code line, not a
   docstring.
2. **`pyphi/connectivity.py:76`** `causally_significant_nodes`: the return was
   cast to Python `int` (`tuple(int(i) for i in ...)`) to match its
   `tuple[int, ...]` annotation. Required to pass the stricter pinned pyright
   (1.1.411); behavior-preserving.

## Config change that rode along

`pyproject.toml` gained `allowed-confusables = ["α", "𝒜", "σ", "×", "−", "∖"]`
under `[tool.ruff.lint]`, so the deliberate mathematical Unicode in docstrings
passes the ambiguous-character rules while genuine homoglyph typos elsewhere are
still caught.
