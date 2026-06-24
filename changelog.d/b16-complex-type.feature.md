Added a first-class `Complex` type. `Substrate.complexes()` now returns
`tuple[Complex, ...]` and `Substrate.maximal_complex()` returns a `Complex`
(a falsy null-object when no system is irreducible). Each `Complex` exposes
`is_maximal`, the selecting `substrate`, and `excluded` — the overlapping
candidates excluded in its favor. The exclusion postulate is enforced by
`validate.non_overlapping()`.
