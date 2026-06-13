Added ``test/test_measure_precision_oracle.py`` (B9): a high-precision
(60-digit ``decimal``) oracle for the φ measure primitives —
``information_density`` (``p·log₂(p/q)``) and the pointwise mutual information
(``log₂(p/q)``) — with a Hypothesis battery over the regimes that were the
catastrophic-cancellation suspects (``p ≈ q`` at scales down to 1e-13, and the
0/1 boundary). The production float64 values agree with the oracle to ≤ ~1 ULP
relative: the primitives form the ratio ``p/q`` first and take a single
``log₂`` (via ``scipy.special.rel_entr`` for the density), so there is no
subtraction of nearly-equal logs to cancel. The catastrophic-cancellation
question is therefore answered by a standing test rather than a caveat, and no
runtime condition-number guard or ``precision``-trust warning is warranted
(those B9 follow-on slices are moot).
