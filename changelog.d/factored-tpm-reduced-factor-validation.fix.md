`FactoredTPM` now rejects reduced-dimension factors at construction with a
clear `InvalidTPM` error. Each factor must be full-dimension — shape
`(*alphabet_sizes, k_i)`, one leading axis per substrate unit plus the unit's
own output axis. Previously a factor with too few leading axes was silently
accepted and crashed later with an opaque broadcasting error.
