Added ``config.infrastructure.validate_phi_bounds`` (default off): a debug/CI
gate that checks every IIT 4.0 result's φ against the theorem-certified
Zaeemzadeh (2024) upper bound and raises ``BoundViolationError`` when an
in-domain value exceeds its ceiling — a *proof* of a formalism bug rather than
a numerical artifact. Wired at four result-construction sites: the MICE bound
``φ ≤ |M||Z|`` (Theorem 1) and the MIP bound ``φ ≤ N(θ)`` (Lemma 2) in
``find_mice``; the system bound ``φ_s ≤ n(n−1)`` (Table 2) at the SIA; and the
``Σφ_d`` / ``Σφ_r`` / ``Φ`` bounds at the cause-effect structure.

The check (``pyphi.formalism.iit4.bounds.check_phi_bound``) is a no-op outside
the certified domain, so it produces no false positives. Skipped cases: IIT
3.0 and non-GID measures (out of the version/measure domain); non-binary
(k-ary) units; macro coarse-grainings (``MacroSystem`` — a single macro unit
can have φ_s > 0 while the macro-unit count gives ``n(n-1) = 0``); single-unit
systems (the ``n(n-1)`` ceiling is trivially 0 and does not cover the
single-node self-loop-φ convention); and partition inputs the bound functions
don't cover. The bound arithmetic only runs when the flag is set, so the
default hot path is unaffected. The test suite runs with the flag on, turning
any φ overshoot across the goldens and property tests into a loud failure.
