Fixed a bug in the IIT 4.0 (2026) intrinsic-information cap (Eq. 23). The cap
``φ_s = min{φ_c, φ_e, ii(s)}`` was applied *per partition* inside the
system-MIP search, so it influenced which partition was selected as the MIP
(the search minimizes normalized φ). This could shift the MIP and make the
reported 2026 system φ *exceed* the 2023 value — contradicting both the 4.0
paper (the MIP is selected on the uncapped integration, then read off) and the
2026 refinement (identical to 4.0 until the Eq. 23 cap is applied to the
result). Empirically the 2026 phi exceeded the 2023 phi on ~2.5% of random
reachable small substrates.

The MIP is now selected on the uncapped integration exactly as in 4.0, and the
``ii(s)`` cap is applied once to the chosen MIP. System φ for the 2026
formalism is now always ≤ the 2023 value, as intended. ``logistic3_k8`` still
binds at the same value (0.0032); only the selected ``sia.partition`` of four
2026 goldens changes — φ is unchanged everywhere. Found by the B5
cross-formalism ``φ_2026 ≤ φ_2023`` property test, which now guards against
regression.
