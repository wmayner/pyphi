# IIT 3.0 test audit — decisions

Originally drafted 2026-01-17 as a list of test changes that needed
maintainer validation on branch `feature/test-fixes`. Audited
2026-04-08; decisions recorded below. The branch reorganized IIT 3.0
regression tests into ``TestXIIT30`` classes guarded by
``IIT_3_CONFIG`` (``test/conftest.py``); several tests were skipped,
weakened, or had expected values changed in the process. Each item
below is now resolved.

## Resolved items

### 1. `test_background_conditions` — unskipped with corrected expected values

**File**: `test/test_actual.py` (in `TestActualCausationIIT30`).

**Was**: `@pytest.mark.skip("Expected values need validation by maintainer")`.
The test asserted ratios ``1, 1, 0, 0`` — three correct, one wrong.

**Decision**: Unskipped. Replaced the incorrect CAUSE-direction
assertion (``all_on._ratio(CAUSE, (1,), (0,)) == 0``) with
``log2(4/3) ≈ 0.4150374992788``. The original ``0`` was wrong because
OR saturates forward (EFFECT direction) but does **not** saturate
backward — three of four previous states lead to ``node_1 = 1`` with
previous ``node_0`` values ``{0, 1, 1}``, so the PMI is
``log2((2/3)/(1/2)) = log2(4/3)``. Derivation is in the test docstring.

### 2. `test_sia_cause_direction` — kept under BI with explicit docstring

**File**: `test/test_actual.py`.

**Was**: New test (no develop counterpart). Asserted
``actual.sia(transition, Direction.CAUSE).alpha == 0.0`` under
``PARTITION_TYPE="BI"`` (inherited from ``IIT_3_CONFIG``).

**Decision**: Kept as-is. Verified empirically that under BI the cause
direction reduces to 0.0 for this fixture while the effect/bidirectional
directions stay at 0.415; under TRI all three are 0.415. The BI-specific
reduction is a valid regression claim. Tightened the docstring to say so
explicitly and cross-reference ``test_sia_effect_direction`` and
``test_prevention`` (which exercises TRI).

### 3. `test_prevention` — TRI override restored

**File**: `test/test_actual.py`.

**Was**: The original develop test had
``@config.override(PARTITION_TYPE="TRI")`` and asserted
``CAUSE alpha == 0.415037``. The branch dropped the override, inherited
BI from the class fixture, and changed the assertion to all-zero.

**Decision**: Restored ``@config.override(PARTITION_TYPE="TRI")`` on the
test method. Verified that under current code TRI still yields
``0.4150374992788`` for CAUSE (matches the original develop test
exactly). Restoring TRI preserves the original regression semantic
("tripartition → nonzero CAUSE alpha on prevention"), which is a
different claim than "BI → all-zero." Both claims are now tested:
``test_sia_cause_direction`` covers BI, ``test_prevention`` covers TRI.

### 4. `test_all_complexes_standard` — exact count restored

**File**: `test/test_compute_network.py`.

**Was**: Assertion weakened from ``len(complexes) == 3`` to
``len(complexes) >= 3``.

**Decision**: Tightened to ``len(complexes) == 5`` plus a phi-value
check ``[0.0, 0.0, 0.5, 1.0, 2.0]`` plus an assertion that exactly three
complexes have ``phi > 0``. The count of 5 comes from
``possible_complexes`` (not ``2**n - 1``), consistent with
``test_possible_complexes``. Three with ``phi > 0`` matches the
reducibility-filtered count in ``test_complexes_standard``.

### 5. `test_blackbox_emergence` — xfail upgraded to `strict=True` + traceback documented

**File**: `test/test_macro_subsystem.py`.

**Was**: ``@pytest.mark.xfail(reason="Broadcasting error in macro
subsystem - needs investigation")``.

**Decision**: Kept xfailed but upgraded to ``strict=True`` so an
accidental fix surfaces as ``XPASS``. Captured the full traceback in the
test docstring. The bug is in ``pyphi/subsystem.py:380``
(``_cause_repertoire``): ``joint`` is preallocated with shape
``repertoire_shape(self.network.node_indices, purview)`` but the reduced
per-node cause repertoires on the blackbox+coarse-grain path come back
with an extra non-singleton dimension, producing
``ValueError: non-broadcastable output operand with shape (2,1,1,1)
doesn't match the broadcast shape (2,1,1,2)``. Not a test bug —
deferred to a proper fix (see Known bugs below).

### 6. Slow-marked tests in `TestActualCausationIIT30`

**File**: `test/test_actual.py`.

**Was**: ``test_causal_nexus``, ``test_true_events``, and
``test_extrinsic_events`` all marked ``@pytest.mark.slow``.

**Decision**: Timed under ``--slow --outdated``:
``test_causal_nexus`` 12.58s (kept slow), ``test_true_events`` 5.04s
(kept slow, right at the threshold), ``test_extrinsic_events`` 1.57s
(**slow marker removed** — it runs comfortably by default now).

## Related source fix landed alongside the audit

**`pyphi/actual.py::state_probability` shape handling**. The branch
already contained a fix to branch on ``repertoire.ndim`` when computing
the indexing tuple. Initial review thought this was a no-op, but that
was wrong: the fix is necessary. ``Subsystem`` returns
network-shaped (``ndim == network.size``) repertoires for non-empty
mechanisms via ``_cause_repertoire``/``_effect_repertoire``, but
subsystem-shaped (``ndim == subsystem.size``) repertoires for empty
mechanisms via ``max_entropy_distribution`` (see Known bugs). When the
subsystem is a strict subset of the network, ``Transition._ratio``
calls ``state_probability`` with both shapes, so the old hardcoded
length-``network.size`` index tuple raised ``IndexError`` on the
unconstrained path. The fix handles both shapes correctly.

Added a regression test ``test_state_probability_strict_subsystem``
that exercises the subsystem-shaped branch by constructing a transition
on nodes ``(2, 3)`` of a 4-node network (verified to fail with the
pre-fix code and pass with the fix).

## Known bugs surfaced during the audit — need proper fixes

### Shape inconsistency in `Subsystem` repertoire methods

**Location**: `pyphi/subsystem.py` — `Subsystem.cause_repertoire` at
line 415 and `Subsystem.effect_repertoire` (empty-mechanism branches).

**Problem**: Both methods short-circuit to
``max_entropy_distribution(self.node_indices, purview)`` when the
mechanism is empty, which calls
``repertoire_shape(self.node_indices, purview)``. That returns a shape
over **subsystem** node indices (``len == subsystem.size``) — whereas
the non-empty-mechanism paths (``_cause_repertoire`` at line 377,
``_effect_repertoire`` at line 453) use
``repertoire_shape(self.network.node_indices, purview)`` which returns
a shape over **network** node indices (``len == network.size``).

**Consequence**: The same method returns arrays of different
dimensionality depending on whether the mechanism is empty. Downstream
consumers (``Transition.state_probability``, possibly others) must
special-case both shapes, which is what the ``pyphi/actual.py`` fix
does locally.

**Proper fix**: Call ``max_entropy_distribution(self.network.node_indices,
purview)`` in both places so every repertoire is network-shaped. That
removes the need for the ``state_probability`` shape branch and any
other defensive code that currently compensates for the inconsistency.

**Why this wasn't fixed in this branch**: Changing the shape of
unconstrained repertoires is a cross-cutting refactor — it touches
every consumer of ``unconstrained_cause_repertoire`` /
``unconstrained_effect_repertoire`` across the codebase (IIT 3.0 and
4.0 paths, compute, metrics, visualization, ...). That's a bigger
audit than this branch is scoped for, and fixing it incorrectly risks
numerical regressions in published results. The local containment in
``actual.py`` lets the test audit land without entangling with the
broader refactor.

**Proposed next step**: Open a dedicated branch, add a test that
asserts ``unconstrained_cause_repertoire(purview).ndim ==
network.size`` for several subsystem configurations, apply the
``self.network.node_indices`` fix in both empty-mechanism branches,
and run the full test suite (including ``--outdated`` and ``--slow``).
If that lands cleanly, revert the
``pyphi/actual.py::state_probability`` shape branch in a follow-up
since it will no longer be needed.

### Pre-existing failures in the `--slow --outdated` regression lane

Surfaced during the final verification run of this audit but **not caused by
this branch** and **out of scope** — left for a dedicated follow-up. These all
still fail with the audit's changes reverted, matching develop's posture:

- `test/test_big_phi.py::test_system_cut_styles` —
  ``ValueError: if no total is given, chunksize must be provided`` at
  ``pyphi/parallel/tree.py:150``. On develop the test is ``@pytest.mark.slow``
  with ``@config.override(SYSTEM_PARTITION_TYPE="DIRECTED_BI")``. Likely
  affected by the recent parallel-redesign refactor (see commit
  ``97ee0fe0 Merge branch 'feature/parallel-redesign' into develop``).
- `test/test_macro_blackbox.py::test_coarsegrain_spatial_degenerate` —
  ``ValueError: IIT 3.0 calculations must use one of the following system
  partition schemes: ['DIRECTED_BI', 'DIRECTED_BI_CUT_ONE']; got SET_UNI/BI``.
  Already ``@pytest.mark.outdated`` on develop — known-broken.
- `test/test_macro_blackbox.py::test_degenerate` —
  ``AttributeError: 'MacroSubsystem' object has no attribute 'tpm'``.
  Already ``@pytest.mark.outdated`` on develop — known-broken.

The default CI lane (``uv run pytest test/ --ignore=test/test_parallel*.py``)
is green: **863 passed, 52 skipped, 2 xfailed**.

### Broadcasting bug in `_cause_repertoire` for blackbox+coarse-grain macro subsystems

**Location**: `pyphi/subsystem.py:380` (`_cause_repertoire`).

**Problem**: ``joint`` is preallocated with
``repertoire_shape(self.network.node_indices, purview_set)`` but the
reduction of per-node cause repertoires on the blackbox+coarse-grain
macro path produces an array with an extra non-singleton dimension.

**Reproducer**: See `test_blackbox_emergence` (currently
``xfail(strict=True)``).

**Proper fix**: Trace through what
``_single_node_cause_repertoire`` is emitting on the macro path and
reconcile the per-node shape with the preallocated joint shape. Likely
requires fixing the macro subsystem's TPM preparation to emit
singletons for non-input purview nodes rather than 2-sized dims.
