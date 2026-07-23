`sum_phi_relations_measured_bound` (and `big_phi_measured_bound`) raised
`OverflowError` when more than 1023 distinctions shared a purview atom —
crashing `collect`'s scope report at production scale. The 2^k weight
now saturates to the documented `inf` ceiling, and zero-density groups
contribute nothing regardless of size.
