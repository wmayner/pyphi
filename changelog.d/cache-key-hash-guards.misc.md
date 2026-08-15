Two regression guards for cache-key hashing. `test/data_structures/test_hash_quality.py`
declares every type that reaches a cache key and asserts each separates a
generated population of distinct instances; a companion test instruments the
cache during real analyses and fails if a key type appears that the registry
does not declare, so a new key type cannot enter without a hash-quality check.
The call-count gate (`test/integration/test_perf_counters.py`) additionally
pins the frames that dictionary collision handling passes through, which move
when a key type's hash degrades while every count of PyPhi operations stays
identical. The performance harness gains a `specified_state` grain and two
seeded ring fixtures at 10 and 12 units, above the four-unit ceiling of the
golden zoo.
