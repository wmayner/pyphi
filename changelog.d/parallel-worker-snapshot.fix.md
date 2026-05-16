Fixed parallel worker snapshot propagation in ``pyphi.parallel.MapReduce``.
``MapReduce._run_parallel`` now captures the parent's
``ConfigSnapshot`` and wraps ``map_func`` with
``backends.local_process._make_worker_fn`` so each worker installs
the parent's config before applying the map function. Previously
workers computed under their stale default config; in IIT 3.0 SIA
this meant ``r.config.formalism.iit.version`` recorded ``IIT_4_0_2023``
even when the parent had ``IIT_3_0`` set. A regression test in
``test/test_result_config_snapshot.py::TestIIT3SIASnapshot`` pins
the round-trip behavior for both sequential and parallel paths.

Also added a ``.config`` ``ConfigSnapshot`` field on
``AcSystemIrreducibilityAnalysis`` so actual-causation results carry
the same reproducibility record as IIT 3.0 / IIT 4.0 SIA results.
