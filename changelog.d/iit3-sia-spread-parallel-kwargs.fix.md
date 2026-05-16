Fixed a long-standing bug in IIT 3.0 SIA / CES computation where
``_ces`` and ``_sia_map_reduce`` in ``pyphi.formalism.iit3`` were
passing ``parallel=config.infrastructure.parallel_partition_evaluation``
to ``ces`` and ``MapReduce``. The intent was to forward the parallel-
evaluation settings, but the call passed the *entire dict* as the value
of the boolean ``parallel`` parameter. ``MapReduce`` saw the truthy
dict, enabled parallel execution with the default
``sequential_threshold=1``, and the parallel worker dispatch silently
dropped non-trivial concepts from the unpartitioned ``CES``.

For the basic substrate in state (1,0,0) the symptom was
``sia.phi=0.5`` with a 2-concept CES instead of the canonical
``sia.phi=2.3125`` with a 4-concept CES (`(1,)`, `(2,)`, `(0,1)`,
`(0,1,2)`). The bug entered with the parallel-redesign merge in
January 2026; before that the equivalent Ray-based dispatch handled
the dict-shaped option correctly.

Both call sites now spread the option dict's keys instead of nesting
them under ``parallel``. Test expectations in ``test_complexes.py``
and ``test_metrics_ces.py`` were updated to the now-correct canonical
values; the three ``@pytest.mark.outdated`` markers in
``test_metrics_ces.py`` are removed.
