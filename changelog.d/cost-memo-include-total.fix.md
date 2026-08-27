`pyphi.cost.estimate_analysis` now accounts for
`system_partition_include_total`: the partition-count memo is keyed on the
option, so estimates are correct (and the memo cannot be cross-poisoned)
when the total cut is included.
