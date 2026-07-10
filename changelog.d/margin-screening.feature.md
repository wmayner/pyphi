Added margin-gated screening to `phi_posterior`: an explicit `screen_margin`
threshold (off by default) computes the maximal complex of the posterior-mean
substrate once (`SubstratePosterior.mean_substrate()`, new) and, when the
complex-identity margin — the φ_s gap between the top two irreducible
candidate systems at the mean — clears the threshold, reuses that identity
per draw instead of re-running the maximal-complex search, while every Φ
sample is still computed in full from an unchanged draw stream. The result
records `screen_margin`, `screened`, and the reference margins (including
the winner's internal selection margins) as an audit trail. The screen is a
compute heuristic, not a bound; use it after the posterior is tight or an
unscreened pilot.
