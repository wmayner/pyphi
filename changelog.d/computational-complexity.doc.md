Added a theory page on the computational complexity of PyPhi's computations,
deriving the dominant cost of each stage for IIT 3.0, IIT 4.0 (2023 and 2026),
and actual causation, and confirming the published `O(n**5 3**n)` scaling of the
IIT 3.0 cause-effect structure empirically. The page also measures how the main
configuration options (analytical vs concrete relations, the cut-one
approximation, and the mechanism partition scheme) change cost, and which
combinations of settings extend the tractable system size. The
`benchmarks/complexity/` scripts reproduce the per-stage, per-option, and
combination timings.
