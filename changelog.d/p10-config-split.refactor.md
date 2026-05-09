**Breaking — Config layered into three frozen dataclasses.**

The flat ``pyphi.config`` singleton is replaced by a layered facade with
three frozen dataclass layers: ``config.formalism``, ``config.infrastructure``,
``config.numerics``. Reads use layered access (``config.numerics.precision``);
persistent writes use the top-level facade (``config.precision = 6`` routes
to the right layer); scoped writes use ``config.override(precision=6,
parallel=True, repertoire_distance="EMD")`` with build-time field-name
collision detection. Per-layer ``config.numerics.override(...)`` also works.

Every top-level result object (``SystemIrreducibilityAnalysis`` for both
IIT 3.0 and IIT 4.0, ``PhiStructure``) carries a ``.config: ConfigSnapshot``
field set at construction time, so reproducibility is self-contained:
``pyphi.config.override(**result.config.as_kwargs())`` reruns the exact
recorded computation. Inner-loop result types (RIA, MICE, Distinction,
Concept) inherit the recorded config transitively via their parent SIA
or PhiStructure.

The layered ``pyphi_config.yml`` format is supported via
``pyphi.config.load_yaml(path)`` and ``pyphi.config.to_yaml(path)``. Old
1.x flat YAML files raise ``ConfigurationError`` with a pointer to the
rename map below.

**Rename map** (1.x flat → 2.0 layered read):

| Old (1.x flat) | New (2.0 layered) | Layer |
|---|---|---|
| ``FORMALISM`` | ``config.formalism.formalism`` | formalism |
| ``REPERTOIRE_DISTANCE`` | ``config.formalism.repertoire_distance`` | formalism |
| ``REPERTOIRE_DISTANCE_DIFFERENTIATION`` | ``config.formalism.repertoire_distance_differentiation`` | formalism |
| ``REPERTOIRE_DISTANCE_SPECIFICATION`` | ``config.formalism.repertoire_distance_specification`` | formalism |
| ``CES_DISTANCE`` | ``config.formalism.ces_distance`` | formalism |
| ``ACTUAL_CAUSATION_MEASURE`` | ``config.formalism.actual_causation_measure`` | formalism |
| ``PARTITION_TYPE`` | ``config.formalism.partition_type`` | formalism |
| ``SYSTEM_PARTITION_TYPE`` | ``config.formalism.system_partition_type`` | formalism |
| ``SYSTEM_PARTITION_INCLUDE_COMPLETE`` | ``config.formalism.system_partition_include_complete`` | formalism |
| ``SYSTEM_CUTS`` | ``config.formalism.system_cuts`` | formalism |
| ``DISTINCTION_PHI_NORMALIZATION`` | ``config.formalism.distinction_phi_normalization`` | formalism |
| ``RELATION_COMPUTATION`` | ``config.formalism.relation_computation`` | formalism |
| ``STATE_TIE_RESOLUTION`` | ``config.formalism.state_tie_resolution`` | formalism |
| ``MIP_TIE_RESOLUTION`` | ``config.formalism.mip_tie_resolution`` | formalism |
| ``PURVIEW_TIE_RESOLUTION`` | ``config.formalism.purview_tie_resolution`` | formalism |
| ``SHORTCIRCUIT_SIA`` | ``config.formalism.shortcircuit_sia`` | formalism |
| ``SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI`` | ``config.formalism.single_micro_nodes_with_selfloops_have_phi`` | formalism |
| ``ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS`` | ``config.formalism.assume_cuts_cannot_create_new_concepts`` | formalism |
| ``PARALLEL`` | ``config.infrastructure.parallel`` | infrastructure |
| ``PARALLEL_*_EVALUATION`` | ``config.infrastructure.parallel_*_evaluation`` | infrastructure |
| ``PARALLEL_WORKERS`` | ``config.infrastructure.parallel_workers`` | infrastructure |
| ``PARALLEL_BACKEND`` | ``config.infrastructure.parallel_backend`` | infrastructure |
| ``MAXIMUM_CACHE_MEMORY_PERCENTAGE`` | ``config.infrastructure.maximum_cache_memory_percentage`` | infrastructure |
| ``CACHE_REPERTOIRES`` | ``config.infrastructure.cache_repertoires`` | infrastructure |
| ``CACHE_POTENTIAL_PURVIEWS`` | ``config.infrastructure.cache_potential_purviews`` | infrastructure |
| ``CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA`` | ``config.infrastructure.clear_subsystem_caches_after_computing_sia`` | infrastructure |
| ``LOG_FILE`` | ``config.infrastructure.log_file`` | infrastructure |
| ``LOG_FILE_LEVEL`` | ``config.infrastructure.log_file_level`` | infrastructure |
| ``LOG_STDOUT_LEVEL`` | ``config.infrastructure.log_stdout_level`` | infrastructure |
| ``PROGRESS_BARS`` | ``config.infrastructure.progress_bars`` | infrastructure |
| ``REPR_VERBOSITY`` | ``config.infrastructure.repr_verbosity`` | infrastructure |
| ``PRINT_FRACTIONS`` | ``config.infrastructure.print_fractions`` | infrastructure |
| ``LABEL_SEPARATOR`` | ``config.infrastructure.label_separator`` | infrastructure |
| ``WELCOME_OFF`` | ``config.infrastructure.welcome_off`` | infrastructure |
| ``VALIDATE_*`` | ``config.infrastructure.validate_*`` | infrastructure |
| ``PRECISION`` | ``config.numerics.precision`` | numerics |

**Deferred to a future release:** the underlying legacy ``PyphiConfig``
backend persists as ``pyphi._conf_legacy`` for now; ``_GlobalConfig`` is
a thin layered facade over it. A future cleanup will replace the facade
with a self-owning layered config and delete the legacy module. Until
then, the auto-load of ``pyphi_config.yml`` at import time still expects
the 1.x flat format; users who want the nested form invoke
``pyphi.config.load_yaml(path)`` explicitly.
