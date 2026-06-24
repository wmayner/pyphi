Renamed every entry in the mechanism- and system-partition-scheme
registries (and the generator functions backing them) for consistency
with the rest of the new partition taxonomy. Schemes now describe what
they yield, not abbreviations or vague qualifiers like "general" or
"simple". Functions in ``pyphi.partition`` lose the redundant
``system_`` prefix (the registry already disambiguates).

**Mechanism-level registry** (``config.formalism.iit.mechanism_partition_scheme``):

| Old | New |
|---|---|
| ``"BI"`` | ``"JOINT_BIPARTITION"`` |
| ``"TRI"`` | ``"WEDGE_TRIPARTITION"`` |
| ``"ALL"`` | ``"JOINT_PARTITION_ALL"`` |

**System-level registry** (``config.formalism.iit.system_partition_scheme``):

| Old | New |
|---|---|
| ``"DIRECTED_BI"`` | ``"DIRECTED_BIPARTITION"`` |
| ``"DIRECTED_BI_CUT_ONE"`` | ``"DIRECTED_BIPARTITION_CUT_ONE"`` |
| ``"DIRECTED_BI_SIMPLE"`` | ``"DIRECTED_BIPARTITION_SEQUENTIAL"`` |
| ``"TEMPORAL_DIRECTED_BI"`` | ``"TEMPORAL_DIRECTED_BIPARTITION"`` |
| ``"TEMPORAL_DIRECTED_BI_CUT_ONE"`` | ``"TEMPORAL_DIRECTED_BIPARTITION_CUT_ONE"`` |
| ``"GENERAL"`` | ``"EDGE_CUT_ALL"`` |
| ``"GENERAL_BIDIRECTIONAL"`` | ``"EDGE_CUT_BIDIRECTIONAL"`` |
| ``"SET_UNI/BI"`` | ``"DIRECTED_SET_PARTITION"`` |

**Function renames in ``pyphi.partition``**:

| Old | New |
|---|---|
| ``mip_partitions`` | ``mechanism_partitions`` |
| ``mip_bipartitions`` | ``joint_bipartitions`` |
| ``wedge_partitions`` | ``wedge_tripartitions`` |
| ``all_partitions`` | ``all_joint_partitions`` |
| ``system_directed_bipartitions`` | ``directed_bipartitions`` |
| ``system_directed_bipartitions_cut_one`` | ``directed_bipartitions_cut_one`` |
| ``system_bipartitions_simple`` | ``directed_bipartitions_sequential`` |
| ``system_temporal_directed_bipartitions`` | ``temporal_directed_bipartitions`` |
| ``system_temporal_directed_bipartitions_cut_one`` | ``temporal_directed_bipartitions_cut_one`` |
| ``general`` | ``all_edge_cuts`` |
| ``general_bidirectional`` | ``bidirectional_edge_cuts`` |
| ``unidirectional_set_partitions`` | ``directed_set_partitions`` |
| ``num_general_partitions`` | ``num_edge_cuts`` |
| ``complete_partition`` | ``complete_joint_partition`` |
| ``atomic_partition`` | ``atomic_joint_partition`` |

User ``pyphi_config.yml`` files setting any of the old string keys will
silently fall through to defaults; update them to the new keys.
