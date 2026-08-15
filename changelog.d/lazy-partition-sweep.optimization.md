The mechanism-MIP search no longer builds every candidate partition before
evaluating any of them. The search stops at the first reducible partition, but
the full set was constructed up front regardless of where it stopped: for a
fully connected 6-unit system, 31.9 million partitions built to evaluate 4.4
million, with the full-system pair alone holding a list of 2.2 million
partition objects. Partitions are now built one at a time as the search
consumes them. The total, which the search needs in order to tell an
exhaustive pass from one that stopped early, comes from
`pyphi.cost.partition_sweep_count`; it is memoized, and its values are already
checked against real enumeration by the test suite. Unfolding the distinctions
of the IIT 4.0 Fig 6D system takes 187 s rather than 233 s and peaks at 0.8 GiB
rather than 2.0 GiB, with every φ, MIP, specified state and partition margin
unchanged.
