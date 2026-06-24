Phase 1 of the 2.0 audit cleanup. Each item closes a finding from the
parallel audits (D-public-API, E-serialization, H-IIT3-coverage,
I-config-snapshot, L-skip-markers, plus a per-test slow-marker review).

**Serialization round-trip fixes**

- ``CompleteEdgeCut.from_json`` no longer inherits the ``EdgeCut`` factory
  that passes ``cut_matrix=`` (the subclass constructor doesn't accept it).
  A subclass-local ``from_json`` reconstructs from
  ``(node_indices, node_labels)`` only.
- ``NullCauseEffectStructure`` is now registered in
  ``pyphi.jsonify._loadable_models()`` and its ``__init__`` uses
  ``kwargs.setdefault(...)`` so the hard-coded defaults don't collide with
  parent-class fields during JSON round-trip.
- Removed orphan fixture ``test/data/PQR_CES.json`` (had drifted from
  current ``Distinction`` schema; no remaining consumers).

**Skip-marker hygiene**

- ``test_ces_labeled_mechanisms`` (``test_models.py``) lost its stale
  ``@pytest.mark.outdated``; passes against current model.
- ``test_example_transitions_construct`` (``test_examples.py``) lost an
  in-test ``pytest.skip(...)`` referencing the gone ``_external_indices``
  override; the parametrized family now runs.
- ``test_macro_system.py``: module-level ``pytestmark.skip(P7b)`` narrowed
  to per-test ``@_skip_p7b``. Result: 4 helper tests (``test_sparse_blackbox``,
  ``test_dense_blackbox``, ``test_cycle_blackbox``, ``test_run_tpm``) run
  in fast lane; 3 known-broken tests reveal as ``xfail``; the remaining 15
  carry the P7b skip individually.

**Slow-marker rebalancing** (per per-test timing audit)

Promoted ``slow`` → fast (all run in <1s):
``test_sia_is_deterministic_across_runs_{sequential,parallel}``,
``test_sia_big_subsys_all_complete_{sequential,parallel}``,
``test_parallel_and_sequential_ces_are_equal``,
``TestActualCausationIIT30::test_causal_nexus``,
``TestActualCausationIIT30::test_true_events_on_known_complex``,
``TestPhiValues::test_sia_big_substrate_complete_phi_value``.

Promoted ``veryslow`` → fast (all run in ~1s):
``test_sia_rule152_s_{sequential,parallel}``,
``TestPhiValues::test_sia_rule152_phi_value``.

Demoted ``slow`` → ``veryslow`` (42-106s per case, ~281s total):
the ``rule154`` parametrize cases in ``test_iit4.py`` and
``test_iit4_robust.py`` that had been polluting the slow lane.

Net effect: fast lane gains 11 tests (now 1241, +17 across this session);
the slow lane drops the rule154 cases. The motivating regression
(``test_causal_nexus`` / ``test_true_events`` that quietly broke under
``@slow``) is now CI-gated.
