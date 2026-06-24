IIT 3.0 tie-resolution overhaul:

- **Cross-subsystem complex selection** on a φ-tie now correctly
  flags the clique as exclusion-postulate failure (no complex from
  the clique) rather than picking an arbitrary representative via
  the within-subsystem MIP's lex key. Mirrors IIT 4.0's
  Composition-escalation-failure handling. Lower-tier candidates
  not overlapping any accepted complex can still be major
  complexes — indeterminate cliques skip without poisoning their
  overlap region.

- **System-MIP selection within a subsystem** now routes through
  ``resolve_ties.sias`` and consults the new
  ``sia_tie_resolution = ["PHI", "PARTITION_LEX"]`` default in
  ``presets.iit3``. Observable no-op for the canonical preset;
  unifies the architecture with IIT 4.0.

- **Mechanism MIP selection** now uses raw φ (paper-canonical for
  IIT 3.0) instead of inheriting IIT 4.0's
  ``NUM_CONNECTIONS_CUT``-normalized default. ``presets.iit3``
  adds ``mip_tie_resolution = ["PHI", "PARTITION_LEX"]``. The
  ``basic_iit3_emd_tri`` golden's ``sia.phi`` shifts from 2.8125
  (under the prior 4.0-flavored default) to 2.520833
  (paper-canonical); three of four IIT 3.0 goldens regenerate.
  ``basic_subset_iit3_emd`` is unaffected.

- ``presets.iit3`` adds ``distinction_phi_normalization = "NONE"``
  for documentation — post-fix, the 3.0 path no longer consumes
  ``RIA.normalized_phi`` for any decision.

- ``test/test_actual.py::TestActualCausationIIT30::test_true_events``
  is no longer skipped; it now asserts ``true_events`` returns one
  event on the (0,1,2) major complex that the cascade accepts after
  the indeterminate (1,2) vs (0,2) tie at the top tier.

Coverage expansion:

- New canonical reference
  ``test/data/iit3-canonical/basic_tri_sia_phi_canonical.json``
  for the WEDGE_TRIPARTITION scheme (SIA-φ = 2.520833).
- ``test/data/iit3-canonical/basic_sia_phi_canonical.json``
  extended with per-(mechanism, direction) MIC partition shapes.
- New IIT 3.0 EMD goldens for ``rule110_iit3_emd`` and
  ``grid3_iit3_emd`` — the tie-prone substrates identified by the
  prior ``purview_tie_resolution`` audit.

``substrate._greedy_condensation`` is removed (private; no
deprecation shim).
