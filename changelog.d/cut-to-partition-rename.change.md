**Breaking (2.0):** Renamed partition and cut classes to reflect the
distinction between *vertex partitions* (IIT 4.0 paper, Eq. 14–18, 38)
and *edge cuts* (graph-theoretic sense). PyPhi now reserves "partition"
for concepts that divide nodes into groups and "cut" for concepts that
sever edges in the connectivity matrix.

The module ``pyphi.models.cuts`` is now ``pyphi.models.partitions``.

**Class renames**

+----------------------------------+----------------------------------+
| Old name                         | New name                         |
+==================================+==================================+
| ``SystemPartition``              | ``DirectedBipartition``          |
+----------------------------------+----------------------------------+
| ``KPartition``                   | ``JointPartition``               |
+----------------------------------+----------------------------------+
| ``Bipartition``                  | ``JointBipartition``             |
+----------------------------------+----------------------------------+
| ``Tripartition``                 | ``JointTripartition``            |
+----------------------------------+----------------------------------+
| ``CompletePartition``            | ``CompleteJointPartition``       |
+----------------------------------+----------------------------------+
| ``AtomicPartition``              | ``AtomicJointPartition``         |
+----------------------------------+----------------------------------+
| ``KCut`` / ``ActualCut``         | ``DirectedJointPartition``       |
+----------------------------------+----------------------------------+
| ``GeneralKCut``                  | ``EdgeCut``                      |
+----------------------------------+----------------------------------+
| ``CompleteSystemPartition`` /    | ``CompleteEdgeCut``              |
| ``CompleteGeneralKCut``          |                                  |
+----------------------------------+----------------------------------+
| ``GeneralSetPartition``          | ``DirectedSetPartition``         |
+----------------------------------+----------------------------------+

**Attribute renames** (on ``System``, ``TransitionSystem``,
``Transition``, and ``SystemIrreducibilityAnalysis``)

+-------------------------------+--------------------------------+
| Old attribute / parameter     | New attribute / parameter      |
+===============================+================================+
| ``.cut``                      | ``.partition``                 |
+-------------------------------+--------------------------------+
| ``.is_cut``                   | ``.is_partitioned``            |
+-------------------------------+--------------------------------+
| ``.cut_indices``              | ``.partition_indices``         |
+-------------------------------+--------------------------------+
| ``.cut_node_labels``          | ``.partition_node_labels``     |
+-------------------------------+--------------------------------+
| ``.cut_mechanisms``           | ``.partitioned_mechanisms``    |
+-------------------------------+--------------------------------+
| ``SIA.cut_system``            | ``SIA.partitioned_system``     |
+-------------------------------+--------------------------------+
| ``uncut_system=`` parameter   | ``unpartitioned_system=``      |
+-------------------------------+--------------------------------+

**Config key rename**

The configuration key ``parallel_cut_evaluation`` is now
``parallel_partition_evaluation``.  Existing ``pyphi_config.yml`` files
that set this key must be updated; PyPhi raises an error on load if the
old key is present.

**Internal helpers renamed** (no direct public API impact):
``pyphi.validate.cut`` → ``validate.system_partition``;
``pyphi.actual._evaluate_cut``/``_get_cuts`` →
``_evaluate_partition``/``_get_partitions``;
``pyphi.formalism.iit3.evaluate_cut`` → ``evaluate_partition``;
``fmt.fmt_cut`` → ``fmt.fmt_partition_arrow``;
``fmt.fmt_kcut`` → ``fmt.fmt_directed_joint_partition``.

**Preserved names** (graph-theoretic or paper-defined terms that remain
"cut"): the ``NullCut`` class; the ``cut_matrix()``,
``apply_cut()``, ``cuts_connections()``, ``all_cut_mechanisms()``,
``splits_mechanism()``, and ``num_connections_cut()`` methods; the
``cut_one_approximation`` scheme; and the registry keys
``"DIRECTED_BI_CUT_ONE"``, ``"TEMPORAL_DIRECTED_BI_CUT_ONE"``, and
``"NUM_CONNECTIONS_CUT"``.

**JSON serialization:** Data serialized under the old class names will
not deserialize correctly. Regenerate any stored fixtures or replace the
class name strings in saved JSON files.
