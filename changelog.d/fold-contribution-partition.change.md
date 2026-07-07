`PhiFold.big_phi_contribution` now weights each incident relation by the
seeds' share `|r ∩ F| / |r|` instead of counting it once, so the fold
contributions of any partition of a structure's distinctions sum exactly to
`big_phi`. Single-distinction folds are unchanged.
