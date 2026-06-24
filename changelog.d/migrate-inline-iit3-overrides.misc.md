Migrated three partition-dependent inline IIT 3.0 test overrides to
``@config.override(**presets.iit3, ...)``. ``test_find_mip`` in
``test/test_system_small_phi.py``, ``test_selfloop_phi_depends_on_config``
in ``test/test_invariants.py``, and ``test_iit3_mip_phi_nonnegative`` in
``test/test_invariants_hypothesis.py`` previously set only
``version="IIT_3_0", mechanism_phi_measure="EMD"`` and inherited the
default IIT 4.0 ``mechanism_partition_scheme``. Behavior was masked by
``IIT3Formalism.partition_scheme`` being a ``ClassVar`` set to
``"JOINT_BIPARTITION"`` (so ``find_mip`` ignored the config field), but
sourcing from the canonical preset eliminates the latent drift and the
``from dataclasses import replace`` import is no longer needed.
