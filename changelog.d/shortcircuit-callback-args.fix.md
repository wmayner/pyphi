`map_reduce`'s `shortcircuit_callback_args` is honored on the parallel paths
(previously only the sequential path), and combining `size_func` with a
short-circuit predicate is rejected eagerly — cost-balanced chunking reorders
items, so the truncation could not match the sequential prefix.
