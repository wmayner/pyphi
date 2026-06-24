**Resurrected ``pyphi.actual.Transition`` against the frozen ``System``
value type, with full formalism config audit.**

``pyphi.actual.TransitionSystem`` is a new frozen dataclass parametric in
``Direction`` that satisfies ``pyphi.protocols.SystemPublicInterface``.
``Transition`` becomes a frozen wrapper holding two TransitionSystem
instances (one per direction); IIT-formalism dispatchers raise
``NotImplementedError`` when called on a TransitionSystem (category
errors). The 826 lines of currently-skipped ``test/test_actual.py`` come
back online, plus paper-fixture acceptance tests against the worked-example
α values from Albantakis et al. 2019 (``papers/2019__albantakis-et-al__what-caused-what.pdf``).

Configuration restructured: ``config.formalism`` now holds two nested
frozen dataclasses, ``IITConfig`` and ``ActualCausationConfig``. Field
naming aligned uniformly:

- ``repertoire_distance`` family → ``repertoire_measure`` family
  (matches the ``measures`` registry name)
- ``ces_distance`` → ``ces_measure``
- ``partition_type`` → ``mechanism_partition_scheme``
- ``system_partition_type`` → ``system_partition_scheme``
- ``assume_cuts_cannot_create_new_concepts`` →
  ``assume_partitions_cannot_create_new_concepts``
- ``actual_causation_measure`` → ``actual_causation.measure``

New AC-specific knobs (under ``formalism.actual_causation``):
``mechanism_partition_scheme``, ``partitioned_repertoire_scheme``,
``background_strategy``, ``alpha_aggregation`` — paper-faithful defaults
match the 2019 Albantakis et al. formalism. AC's ``partitioned_repertoire``
no longer branches on ``config.formalism.iit.repertoire_measure``;
behavior is now independent of IIT-formalism configuration.

Removed the orphaned concept-style cuts machinery
(``ConceptStyleSystem``, ``concept_cuts``, ``directional_sia``,
``SystemIrreducibilityAnalysisConceptStyle``, ``sia_concept_style``) and
the ``system_cuts`` config field. The variant has unclear provenance,
its integration tests rotted in the IIT 4.0 transition, and no current
workflows depend on it.

``pyphi/macro.py`` resurrection is deferred to a separate paper-faithful
project tracking Marshall et al. 2024 (intrinsic units).
