# P10 Call-Site Inventory (working scratch — deleted at end of Phase 6)

**Generated:** 2026-05-08, on `feature/p10-config-split` at `3d418fd6`.

**Totals:**
- Total `config.` references in `pyphi/` + `test/`: **293**
- Approximate uppercase reads (`config.X` not in assignment, not override kwarg): **103**
- Persistent writes (`config.X = ...`): **18**
- `config.override(...)` calls: **104**
- Other (method calls, non-config matches): ~68

**Top per-module density:**

| Count | Path |
|------:|---|
| 22 | pyphi/conf.py |
| 20 | test/test_big_phi.py |
| 15 | pyphi/formalism/iit4/formalism.py |
| 12 | test/test_pyphi_float.py |
| 12 | pyphi/compute/subsystem.py |
| 9 | test/test_big_phi_robust.py |
| 9 | pyphi/subsystem.py |
| 9 | pyphi/parallel/__init__.py |
| 9 | pyphi/formalism/iit4/__init__.py |
| 9 | pyphi/actual.py |
| 8 | pyphi/models/fmt.py |
| 7 | test/test_invariants.py |
| 7 | pyphi/formalism/queries.py |
| 7 | pyphi/data_structures/pyphi_float.py |
| 6 | test/test_parallel.py |
| 6 | test/test_metrics_ces.py |
| 6 | pyphi/parallel/backends/local.py |
| 6 | pyphi/metrics/distribution.py |
| 6 | pyphi/metrics/ces.py |

The rest of `pyphi/` and `test/` has 1–5 hits per file.

**Migration order** (dependency-leaf first):

1. `pyphi/utils.py`, `pyphi/combinatorics.py`, `pyphi/data_structures/pyphi_float.py` (numerics-only)
2. `pyphi/distribution.py`, `pyphi/metrics/distribution.py`, `pyphi/metrics/ces.py`
3. `pyphi/partition.py`, `pyphi/repertoire.py`
4. `pyphi/cache/__init__.py`, `pyphi/cache/cache_utils.py`
5. `pyphi/parallel/tree.py`, `pyphi/parallel/__init__.py`, `pyphi/parallel/backends/local.py`, `pyphi/parallel/progress.py`
6. `pyphi/network.py`, `pyphi/jsonify.py`
7. `pyphi/models/*.py`
8. `pyphi/formalism/iit3/`, `pyphi/formalism/iit4/`
9. `pyphi/formalism/queries.py`, `pyphi/formalism/__init__.py`
10. `pyphi/core/*.py`, `pyphi/subsystem.py`, `pyphi/repertoire.py`
11. `pyphi/compute/*.py`
12. `pyphi/actual.py`, `pyphi/relations.py`
13. `pyphi/visualize/*.py`
14. `test/conftest.py` + all `test/test_*.py`

**Note on `pyphi/conf.py`:** the 22 hits are internal to the legacy module's own implementation (option definitions reference `_values`, `_overrides`, etc.); these don't get migrated — `pyphi/conf.py` becomes `pyphi/_conf_legacy.py` and stays internally consistent until Phase 6 deletes it.
