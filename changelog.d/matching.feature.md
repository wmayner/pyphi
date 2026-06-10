Added the cross-stimulus matching layer to `pyphi.matching`:
`Differentiation` exposes the differentiation D (Eq 16) and perceptual
differentiation D_p (Eq 19) of the component union across triggered
structures, and `MatchingAnalysis.matching()` estimates matching M (Eq 21)
as the expected world-minus-noise perceptual-differentiation gap, with
seeded paired sampling and per-trial raw values carried on the frozen
`MatchingResult`.
