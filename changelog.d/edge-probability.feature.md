Added `SubstratePosterior.edge_probability`: a graded connectivity oracle
for estimated substrates (the fraction of posterior samples in which a
unit's conditional varies beyond a caller-chosen threshold along each input
axis). The exact-equality `infer_cm` saturates to fully-connected on any
continuously-estimated TPM; this is the replacement for that regime.
