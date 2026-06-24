The eager config validator now rejects `IIT_3_0` paired with a
`system_partition_scheme` other than `DIRECTED_BIPARTITION` or
`DIRECTED_BIPARTITION_CUT_ONE` at configuration time (previously this failed
only reactively, deep in the compute path). The error names both conflicting
fields and a fix. Configurable off via `validate_config=False`. The IIT 4.0
family accepts any registered system scheme and is unaffected.
