Actual-causation partition enumeration now reads its own
`actual_causation.mechanism_partition_scheme` config field instead of silently
inheriting the IIT `mechanism_partition_scheme`. Under an IIT 3.0 pin the AC
alpha of first-order occurrences was deflated by the paper-forbidden bipartition
family; AC now defaults to the 2019-paper `JOINT_PARTITION_ALL` family
regardless of the IIT setting.
