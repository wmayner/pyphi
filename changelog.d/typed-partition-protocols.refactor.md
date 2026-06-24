Add `MechanismPartition`, `SystemPartitionLike`, `MechanismPartitionScheme`,
and `SystemPartitionScheme` Protocols to `pyphi.protocols` for type-system
distinction between mechanism-level and system-level partitions, with
registration-time validation on both `partition_types` and
`system_partition_types` registries. The legacy `PartitionScheme` Protocol
is removed in favor of the narrower `MechanismPartitionScheme`.
