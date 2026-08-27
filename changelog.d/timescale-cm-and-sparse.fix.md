`timescale.run_cm` no longer mutates the caller's connectivity matrix (and
accepts read-only input), and the `sparse` heuristic is fixed: it was
inverted (dense matrices took the scipy-sparse branch) and measured on the
state-by-node TPM rather than the state-by-state matrix actually raised to
a power. Results are unchanged; only which backend computes them.
