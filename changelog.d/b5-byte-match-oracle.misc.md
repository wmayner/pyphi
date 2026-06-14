Added the IIT 4.0 (2023) pre-refactor non-regression oracle (B5 follow-on):
``TestPreRefactorByteMatch`` in ``test/test_cross_formalism_invariants.py``
recomputes system phi for a 48-state corpus and asserts it reproduces the
values computed at commit ``b3aaa3e5`` (reproducer:
``scripts/gen_iit4_2023_byte_match_oracle.py``; data:
``test/data/iit4_2023_byte_match_oracle.json``), locking the 2.0 refactor as
numerically non-regressing on the 4.0 directed-bipartition hot path. Running the
oracle showed a literal ``float.hex()`` match is infeasible for two expected
reasons, neither a regression: (1) the pre-refactor ``phi`` was the *raw*
un-clamped integration, so it is matched against HEAD's ``signed_phi`` while the
clamped ``phi`` is asserted to equal ``max(0, raw)`` (the Eqs. 19-20 ``|·|+``
correction the refactor added); and (2) sub-ULP floating-point reassociation
(max abs diff 3.3e-16 over the corpus), so the test asserts machine-epsilon
equivalence (``atol=1e-12``). The match uses ``DIRECTED_BIPARTITION`` (preserved
across the refactor), not the era's default ``SET_UNI/BI`` (intentionally
replaced by ``DIRECTED_SET_PARTITION``).
