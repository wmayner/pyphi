`numpy_aware_eq()` now compares arrays of different shapes as unequal, as documented; previously broadcastable shape pairs (e.g. `(1, n)` vs `(n,)`) compared equal.
