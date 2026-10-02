# test/

## Performance regressions the φ goldens cannot see

Correctness tests are blind to cost, so two other gates carry it, and a change
that touches caching, hashing, or a full-state sweep should be checked against
both.

`test/integration/test_perf_counters.py` pins deterministic cProfile call
counts for the frames in `test/golden/perf.py::FRAMES`, which cover two
regression classes. *Redundant work* — the same operation performed more often
than necessary — is counted at PyPhi frames. *Cost per operation* — the same
operations, each more expensive — is counted at the frames dictionary collision
handling passes through (`Mapping.__eq__`, `FrozenMap.__getitem__`), because a
cache-key type whose hash stops separating distinct keys leaves every PyPhi
count identical while making each cache operation a linear scan. A new
cache-key type must also be declared in
`test/data_structures/test_hash_quality.py`, whose companion test instruments
the cache during real analyses and fails on an undeclared type.

Neither gate sees memory. Cache *occupancy* is asserted directly in
`test/cache/test_transient_repertoires.py`: a full-state sweep must admit a
number of entries that scales with the unit count, not the state count.

Fixture size is its own axis. The golden zoo tops out at four units, so costs
driven by the size of the whole system rather than of a mechanism are invisible
to it; the `specified_state` grain and the ring fixtures in
`test/golden/perf_fixtures.py` exist for that range and are perf-only (no φ
goldens, absent from `ALL_FIXTURES`).

When adding a guard here, verify it fails against the reverted defect. A pin
that cannot move is not a gate.

## Formalism pinning (tests that assert φ values)

A φ value is only meaningful relative to a formalism. Any test that asserts a
φ value must **pin its formalism explicitly** — never rely on the ambient
default. Pin with the complete preset-sourced context managers
(`IIT_3_CONFIG`, `IIT_4_CONFIG` in `test/conftest.py`, sourced from
`pyphi.conf.presets`), not a hand-listed subset of `iit.*` fields: setting
`iit.version` alone leaves the measures on the ambient default — the
partial-pin trap that silently recomputes under a different formalism when the
default changes. Tests that compute φ at module-fixture setup must pin inside
the fixture (a function-scoped autouse pin does not wrap module-fixture setup).

Exactly one test — `test_default_formalism_is_iit4_2026` — asserts the shipping
default; it is intentionally unpinned. To flip the default formalism: change
the default in `pyphi/conf/formalism.py`, update that assertion plus the two
default-dependent facade tests (`TestGlobalConfigFacade.test_layered_reads_work`
in `test/conf/test_config_layers.py` and `test_2023_omitted_metric_uses_default`
in `test/formalism/test_formalism_measure_threading.py`), and regenerate only
the `docs/` tutorial examples that demonstrate default behavior (CI doctests in
`pyphi/` compute no cap-sensitive φ).
