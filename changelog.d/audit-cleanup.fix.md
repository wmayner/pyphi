Triaged findings from a config-rename audit sweep:

- **frog_example fixed.** ``pyphi.examples.frog_example`` had a stale
  decorator that called ``config.override(mechanism_partition_scheme=...)``
  on a colliding field name (lives in both ``IITConfig`` and
  ``ActualCausationConfig``), which raises ``ConfigurationError``. Now
  wraps the colliding kwarg in ``iit=replace(...)``.
- **Validator tightening.** ``IITConfig`` / ``ActualCausationConfig``
  validators previously accepted ``"FORWARD_PROBABILITY"``,
  ``"STATIONARY"``, ``"OBSERVED"``, ``"RATIO"`` as valid values, but
  those keys aren't registered in their corresponding registries. The
  validator frozensets now match the actually-registered keys
  (``PRODUCT``, ``UNIFORM``, ``SUBTRACTIVE``); setting an unregistered
  value now fails fast at config write rather than later at registry
  lookup.
- **Dead branches removed.** ``RepertoireIrreducibilityAnalysis.num_state_ties``
  and ``.num_partition_ties`` had ``if _state_ties is None`` /
  ``if _partition_ties is None`` branches that were unreachable —
  those attributes are always assigned a tuple in ``__init__``.
- **Stale docstring fixed** in ``pyphi/relations.py`` referencing
  ``config.RELATION_COMPUTATIONS`` (old top-level form); now points to
  ``config.formalism.iit.relation_computation``.
- **Stale skipped-test kwargs updated** in ``test/test_macro_blackbox.py``:
  ``CUT_ONE_APPROXIMATION=True`` → ``system_partition_scheme=
  "DIRECTED_BIPARTITION_CUT_ONE"``; ``parallel_*_evaluation=False`` →
  ``parallel=False`` (the file remains ``pytest.mark.skip``'d pending
  MacroSystem port).
