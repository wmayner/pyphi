The Earth Mover's Distance (`EMD`) repertoire measure now supports multi-valued
(k-ary) substrates. Its ground metric — previously a hard-coded `2^N` Hamming
matrix — is generalized to count differing node states over the substrate's
actual (possibly heterogeneous) state space, and the analytic effect-repertoire
shortcut is generalized to the per-node total variation between k-ary marginals.
Binary results are unchanged. `EMD` is therefore usable as the IIT 3.0 mechanism
measure on non-binary substrates (it is no longer registered binary-only).
