Restructured the ``formalism`` config layer into nested ``iit`` and
``actual_causation`` sub-namespaces. Field renames applied across the
public API:

- ``config.formalism.formalism`` → ``config.formalism.iit.version``
- ``config.formalism.repertoire_distance`` →
  ``config.formalism.iit.repertoire_measure`` (and ``_specification``,
  ``_differentiation`` siblings)
- ``config.formalism.ces_distance`` → ``config.formalism.iit.ces_measure``
- ``config.formalism.partition_type`` →
  ``config.formalism.iit.mechanism_partition_scheme``
- ``config.formalism.system_partition_type`` →
  ``config.formalism.iit.system_partition_scheme``
- ``config.formalism.assume_cuts_cannot_create_new_concepts`` →
  ``config.formalism.iit.assume_partitions_cannot_create_new_concepts``
- ``config.formalism.actual_causation_measure`` →
  ``config.formalism.actual_causation.measure``

Adds AC-specific knobs (``mechanism_partition_scheme``,
``partitioned_repertoire_scheme``, ``background_strategy``,
``alpha_aggregation``) on the ``actual_causation`` sub-namespace with
paper-faithful defaults.

Naming principles: "measure" matches the metrics-registry idiom;
"scheme" reads naturally for partition-generator registries;
"partition" replaces "cut" at the type/operation level.

The flat-write convenience form (``config.repertoire_measure = "EMD"``)
still works for unique field names. Names that collide between
``iit`` and ``actual_causation`` (currently only
``mechanism_partition_scheme``) require the qualified form, e.g.
``config.iit = replace(config.formalism.iit,
mechanism_partition_scheme="JOINT_BIPARTITION")``.

YAML config files now use the nested formalism format::

    formalism:
      iit:
        repertoire_measure: EMD
      actual_causation:
        measure: PMI
