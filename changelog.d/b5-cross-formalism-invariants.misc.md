Added ``test/test_cross_formalism_invariants.py`` (B5): paired/property tests
across PyPhi's formalisms. A Hypothesis property asserts ``φ_2026 ≤ φ_2023``
over random small substrates — which surfaced the Eq-23 cap MIP-selection bug
(fixed separately) and now guards against its regression. The roadmap's
hypothesized "IIT 3.0 and 4.0 agree on reducibility" invariant is **refuted**:
the formalisms disagree on ~70% of random reachable small substrates (IIT 3.0
EMD frequently finds φ>0 where IIT 4.0 GID finds φ=0); a concrete reachable
2-node witness is pinned so the divergence is documented, not assumed. The
AC/IIT sign-agreement invariant and the ``b3aaa3e5`` pre-refactor byte-match
are deferred follow-ons (see the module docstring).
