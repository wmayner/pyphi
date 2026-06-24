Added ``test/test_parallel_equals_sequential.py`` (the N2 invariant): a
standing CI check that the loky-backed *parallel* SIA and CES paths produce
results identical to *sequential* evaluation, for IIT 3.0 (EMD) and IIT 4.0
(2023, GID) on the golden binary substrates. It forces the loky process
scheduler on the SIA cut and CES concept levels — the path the golden harness
no longer exercises (it runs ``parallel=False``) and that the pre-existing
``test_parallel_and_sequential_ces_are_equal`` misses (its default thresholds
collapse small substrates to sequential) — so any sequential/parallel
divergence now fails CI loudly.

This also closes the long-standing IIT 3.0 loky ``BrokenProcessPool``
curiosity. The historical ~50% intermittent on ``basic_iit3_emd`` /
``xor_iit3_emd`` was the P9 cache-registry leak (``clear_all`` walking a
growing per-``Network`` ``PurviewCache`` registry crashed workers on
un-serialize), already fixed by making those caches anonymous. It no longer
reproduces under forced loky even with ``clear_all`` between every run; the
flaky-skip the changelog once described is gone, and the N2 invariant now
keeps the loky path under standing coverage so any recurrence is caught.
