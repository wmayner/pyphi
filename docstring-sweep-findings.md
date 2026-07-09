# Docstring sweep — findings log

Byproducts of the 2026-07-08 docstring sweep. The sweep itself edited only
docstrings; the items below are things it *surfaced* and are recorded here for
separate follow-up, not fixed as part of the sweep (with two small exceptions,
noted under "Code touched").

## Code-level bugs found (all resolved)

Every code-level bug the sweep surfaced has since been fixed:

- `pyphi/actual.py`: "is no a {direction} mechanism" `ValueError` typo → "is not a".
- `pyphi/utils.py` `enforce_integer` / `enforce_integer_or_none`: unused, with a
  wrong error message — deleted as dead code.
- `pyphi/core/tpm` `JointDistribution.print()`: a no-op stub with no callers —
  deleted.
- `pyphi/dynamics.py` `> 0.5` ON threshold: settling a P(ON)=0.5 unit to OFF is
  the intended, now-documented convention (no change needed).
- `pyphi/models/cmp.py` `Orderable.__gt__`/`__ge__`: reverse-sort / `max` / `min`
  over a class and its subclass (e.g. concrete vs null SIA) recursed to a stack
  overflow — the operators now compare `order_by()` directly.
- `pyphi/models/cmp.py` `numpy_aware_eq`: compared sets by positional `zip` →
  now compares sets by set equality.
- `pyphi/distribution.py` `purview()`: dropped k-ary purview nodes (`dim == 2`)
  → identifies non-unitary axes (`dim > 1`).
- The two classes named `JointTPM` were consolidated by the JointTPM-as-view
  refactor, so `pyphi.JointTPM` and `pyphi.core.tpm.JointTPM` now resolve to the
  same class.

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
