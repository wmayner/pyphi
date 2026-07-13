Fixed the IIT 4.0 (2026) intrinsic-differentiation cap term (Eq. 23): `i_diff` is
now evaluated at the **specified** state s′ (Mayner et al. 2026, Eqs. 4, 6, and 12)
instead of the system's current state, and the cause side now applies the Eq. 11
Bayes normalization to the forward likelihoods before taking the surprisal.
Previously, φ_s under the 2026 formalism was wrong whenever the specified state
differed from the current state (e.g. the paper's own Fig. 2 monad in the NOT
regime: PyPhi reported 0.7632 where Eq. 27 gives 0.1520) or the dynamics were not
doubly stochastic. Relatedly: the Eq. 23 cap is now applied to every SIA as soon
as its MIP is selected — including each tied specified-state pair and every
partition-tie member — so the per-state tie cascade compares capped φ_s values and
tie sets no longer mix capped and uncapped φ; the composite `INTRINSIC_INFORMATION`
measure's `state=None` path now returns the per-state minimum of specification and
differentiation instead of broadcasting one global minimum surprisal; and the
system-level partition evaluation now threads the caller's measure explicitly
instead of silently re-resolving `mechanism_phi_measure` from config (an explicit
`system_measure` is no longer overridden by an incompatible config setting, which
could previously shift the MIP). Under IIT 4.0 (2023) and IIT 3.0, φ values are
unchanged; the `intrinsic_differentiation` metadata carried on 2023 SIAs reflects
the corrected convention, and the affected test goldens were regenerated with φ
and MIP asserted identical.
