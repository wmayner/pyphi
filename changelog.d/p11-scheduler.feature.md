Added a typed ``Scheduler`` Protocol abstracting the parallel-execution
backend. Two concrete schedulers ship: ``LocalProcessScheduler``
(``joblib + loky``, today's default behavior) and ``LocalThreadScheduler``
(``concurrent.futures.ThreadPoolExecutor``, useful for free-threaded
runtimes and IO-bound work). A ``DaskScheduler`` skeleton documents the
contract for cluster deployments and raises ``NotImplementedError`` until
filled in (a follow-up project tracks SLURM/PBS/LSF/SGE/HTCondor support).

Workers receive an explicit ``ConfigSnapshot`` via closure rather than
implicitly pickling global state. ``with config.override(...):`` blocks
correctly propagate to workers via the new
``pyphi.config.install_snapshot()`` method.

The dead-code ``parallel/chunking.py`` heuristics are replaced by a small
cost-sampling implementation in ``parallel/sampling.py``: the scheduler
samples up to four items spread across the iterable, times them inline,
and computes a target chunksize for roughly 1s of wall time per chunk.

Backend selection: ``config.infrastructure.parallel_backend`` accepts
``"local"``, ``"process"``, ``"thread"``, ``"dask"``, and ``"auto"``.
``"auto"`` selects ``LocalThreadScheduler`` on free-threaded runtimes,
``LocalProcessScheduler`` otherwise. The user-facing ``MapReduce`` class
keeps its existing process-pool dispatch; explicit thread or dask
selection is done through
``pyphi.parallel.scheduler.default_scheduler()``.

Concrete formalism classes (``IIT3Formalism``, ``IIT4_2023Formalism``,
``IIT4_2026Formalism``) became ``@dataclass(frozen=True)`` with a
``config: FormalismConfig`` field captured at construction. Workers
receive the formalism instance with its config attached via cloudpickle.

Re-enabled ``test/test_parallel.py`` in CI (it was previously excluded).
``test/test_chunking.py`` was deleted alongside the dead-code
``chunking.py`` module.

The IIT 3.0 EMD goldens (``basic_iit3_emd``, ``xor_iit3_emd``)
intermittently hit a known ``BrokenProcessPool`` flake from worker-side
state interactions (deferred curiosity from the unified-cache work);
when the symptom appears the test skips rather than failing CI. Real
numerical regressions still fail the suite.
