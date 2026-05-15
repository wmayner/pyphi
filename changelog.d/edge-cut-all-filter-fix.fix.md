**Fixed latent bug** in IIT 4.0 `sia()` where the
``is_disconnecting_partition`` filter was never installed under
``system_partition_scheme="EDGE_CUT_ALL"``. The check at
``pyphi/formalism/iit4/__init__.py`` was on the wrong variable
(``partitions == "EDGE_CUT_ALL"`` instead of
``partition_scheme == "EDGE_CUT_ALL"``), so the filter sat in an
unreachable branch since the new_big_phi → formalism/iit4 move.
Without the filter, non-disconnecting edge cuts leaked into MIP
search, violating Eq. 14 of Albantakis et al. 2023 (partitions must
yield non-empty disjoint parts covering all units). A regression test
in ``test/test_big_phi.py`` now guards the invariant by monkey-patching
``system_partitions`` and asserting every enumerated partition
disconnects the system.

Cleanup pass on stale references from prior config-renames (no
behavior change):

- ``pyphi_config_3.0.yml`` migrated to the nested 2.0 format
  (``formalism`` / ``infrastructure`` / ``numerics`` top-level keys
  with current lowercase field names); ``SYSTEM_CUTS`` dropped (no
  longer a config field).
- ``test/test_presets.py`` reads the nested structure; test methods
  renamed to match current field names.
- ``test/test_macro_system.py`` ``outdated``-marker fixture now reuses
  ``IIT_3_CONFIG`` instead of stale per-field overrides.
- Error messages in ``pyphi/formalism/base.py``,
  ``pyphi/parallel/__init__.py``, and
  ``pyphi/parallel/backends/local_process.py`` use current field
  names instead of legacy UPPER_CASE forms.
- Docstrings across ``test_invariants.py``, ``test_big_phi.py``,
  ``test_big_phi_robust.py``, ``test_actual.py``, ``test_parallel.py``,
  and ``pyphi/resolve_ties.py`` reference current field names.
- ``AGENTS.md`` config-reference section rewritten against the
  layered ``IITConfig`` / ``ActualCausationConfig`` /
  ``InfrastructureConfig`` / ``NumericsConfig`` taxonomy.
