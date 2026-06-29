Added `Substrate.to_dbn()` and `Substrate.to_dbn_dict()`, which export a
substrate as a 2-timeslice dynamic Bayesian network (``nodes_t ->
nodes_{t+1}``): each per-node TPM factor becomes a CPD conditioned on the
node's inferred parents. `to_dbn()` returns a `networkx.DiGraph` (acyclic,
d-separation-ready); `to_dbn_dict()` returns a plain dict (no networkx
import). No pgmpy dependency.
