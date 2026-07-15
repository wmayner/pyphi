# AC Mechanism-Partition Scheme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Actual-causation partition enumeration reads its own
`ActualCausationConfig.mechanism_partition_scheme` field instead of silently
inheriting the IIT `mechanism_partition_scheme`, so AC α no longer changes with
an unrelated IIT setting.

**Architecture:** `mechanism_partitions()` is called at exactly two leaf sites in
the AC compute module (`_find_mip` and `_get_partitions`). Both are replaced with
a small AC-local helper that resolves the *AC* config field through the
`partition_types` registry at call time. This is a simpler, behavior-identical
alternative to threading a resolved callable through the whole compute chain
(the spec §2.1 described the threading form; the two-site form is chosen because
there are only two call sites, so no threading is needed). The field remains
freely configurable; the default `JOINT_PARTITION_ALL` is the 2019-paper family.

**Tech Stack:** Python 3.13+, `pyphi.partition.partition_types` registry,
`pyphi.conf.config`, `pytest`. Run everything with `uv run`.

## Global Constraints

- **Python 3.13+ only.** No backward-compatibility shims.
- **Correctness is paramount.** This changes computed α values; every changed
  expectation is a fix of a value that codified the leak, cross-checked against
  the paper (α = ρ for first-order occurrences).
- **Docstrings:** NumPy style, final-state impersonal voice, Unicode symbols
  (`α`, `ρ`), paper citations verified against `papers/`.
- **Formalism pinning:** tests that assert α pin `IIT_3_CONFIG` (from
  `test/conftest.py`); the AC formalism reads `config.formalism.actual_causation`.
- **Commits** end with the two trailer lines used across this branch
  (`Co-Authored-By:` and `Claude-Session:`). Never `--no-verify`.
- **Verification:** `uv run pytest` (no path argument — the doctest sweep) plus
  the golden regression suite must be green.

## Confirmed values (measured on the worktree)

Minimal witness — `Substrate([[0,0],[1,0],[1,0],[1,1]])`,
`Transition(sub, (1,0), (1,0), (0,1), (0,1))`, cause link `_find_mip(CAUSE, (0,),
(0,1))`:

- IIT field `JOINT_PARTITION_ALL` → α = 0.4150374992788 (= ρ = log₂(4/3), paper)
- IIT field `JOINT_BIPARTITION` → α = 0.0 (the leaked, paper-forbidden result)
- AC field currently has **no effect** (dead).

`transition` fixture (`test_actual.py:195`, OR gate, cause (1,2), effect (0,)):
- current leak (IIT `JOINT_BIPARTITION`): `sia(CAUSE).alpha` = 0.0
- post-fix (AC `JOINT_PARTITION_ALL`): `sia(CAUSE)` = `sia(EFFECT)` = `sia()` =
  0.415037. Only the CAUSE direction changes (`test_sia`/`test_sia_effect_direction`
  already assert 0.415037).

`prevention` fixture (`examples.prevention_transition()`): under **both**
`JOINT_PARTITION_ALL` and `WEDGE_TRIPARTITION` on the AC field, `sia(CAUSE)` =
0.415037, `sia(EFFECT)` = 0.0, `sia()` = 0.0 (identical).

---

### Task 1: AC reads its own mechanism-partition scheme

**Files:**
- Modify: `pyphi/formalism/actual_causation/compute.py` (import at `:41`; two
  call sites at `:257` and `:560`; new helper)
- Test: `test/test_actual.py`

**Interfaces:**
- Produces (module-private, `compute.py`):
  `_ac_mechanism_partitions(mechanism, purview, node_labels=None) -> Iterable`
  — yields AC mechanism partitions under
  `config.formalism.actual_causation.mechanism_partition_scheme`.

- [ ] **Step 1: Write the failing regression tests**

Append to `test/test_actual.py` (top of file already has `import numpy as np`,
`from dataclasses import replace`, `from pyphi import config`, `from . import
actual`?; verify these imports exist — the file uses `actual`, `config`,
`replace`, `Substrate`, `Direction`, `np`, and `IIT_3_CONFIG` from `.conftest`).

```python
class TestACMechanismPartitionScheme:
    """AC partition enumeration is governed by the AC config field, not IIT."""

    def _witness(self):
        # Minimal first-order-occurrence-over-multi-unit-purview witness.
        sub = Substrate(np.array([[0, 0], [1, 0], [1, 0], [1, 1]]))
        return actual.Transition(sub, (1, 0), (1, 0), (0, 1), (0, 1))

    def test_ac_uses_its_own_default_not_the_iit_field(self):
        # Even with the IIT field pinned to the paper-forbidden bipartition
        # family, AC uses its own default JOINT_PARTITION_ALL and returns ρ.
        t = self._witness()
        from pyphi.formalism.actual_causation.compute import _find_mip

        with config.override(
            iit=replace(
                config.formalism.iit, mechanism_partition_scheme="JOINT_BIPARTITION"
            )
        ):
            ria = _find_mip(t, Direction.CAUSE, (0,), (0, 1))
        assert np.isclose(ria.alpha, np.log2(4 / 3))

    def test_iit_field_does_not_affect_ac_but_ac_field_does(self):
        t = self._witness()
        from pyphi.formalism.actual_causation.compute import _find_mip

        # IIT field varies; AC field fixed at the default -> α is constant.
        vals = []
        for iit_scheme in ("JOINT_PARTITION_ALL", "JOINT_BIPARTITION"):
            with config.override(
                iit=replace(
                    config.formalism.iit, mechanism_partition_scheme=iit_scheme
                )
            ):
                vals.append(_find_mip(t, Direction.CAUSE, (0,), (0, 1)).alpha)
        assert np.isclose(vals[0], vals[1])

        # AC field drives the result: bipartition on the AC field gives 0.
        with config.override(
            actual_causation=replace(
                config.formalism.actual_causation,
                mechanism_partition_scheme="JOINT_BIPARTITION",
            )
        ):
            assert _find_mip(t, Direction.CAUSE, (0,), (0, 1)).alpha == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_actual.py::TestACMechanismPartitionScheme -q > /tmp/ac1.log 2>&1; tail -20 /tmp/ac1.log`
Expected: FAIL — `test_ac_uses_its_own_default_not_the_iit_field` gets α = 0.0
(the leak), and `test_iit_field_does_not_affect_ac_but_ac_field_does` fails
because the IIT field currently drives AC. (Read the summary line, not the exit
code.)

- [ ] **Step 3: Implement the fix in `compute.py`**

Change the import at `pyphi/formalism/actual_causation/compute.py:41`:

```python
# from:
from pyphi.partition import mechanism_partitions
# to:
from pyphi.partition import partition_types
```

Add the helper immediately after the imports/module constants (before
`_find_mip`), using `config` already imported at `:27`:

```python
def _ac_mechanism_partitions(mechanism, purview, node_labels=None):
    """Yield mechanism partitions under the actual-causation partition scheme.

    Resolves ``config.formalism.actual_causation.mechanism_partition_scheme``
    through the partition-scheme registry at call time, so actual-causation
    partitioning is governed by the AC formalism and never by the IIT
    ``mechanism_partition_scheme`` field. The default ``JOINT_PARTITION_ALL`` is
    the partition family of Albantakis et al. (2019), Eq. 7 and Fig. 3B: all
    partitions of the occurrence, excluding the m=1 non-full-cut cases the paper
    forbids for first-order occurrences.
    """
    scheme = config.formalism.actual_causation.mechanism_partition_scheme
    return partition_types[scheme](mechanism, purview, node_labels)
```

Replace the call at `:257` (in `_find_mip`):

```python
# from:
    for partition in mechanism_partitions(mechanism, purview, transition.node_labels):
# to:
    for partition in _ac_mechanism_partitions(
        mechanism, purview, transition.node_labels
    ):
```

Replace the call at `:560` (in `_get_partitions`):

```python
# from:
        for inner_partition in mechanism_partitions(
            mechanism, purview, transition.node_labels
        ):
# to:
        for inner_partition in _ac_mechanism_partitions(
            mechanism, purview, transition.node_labels
        ):
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_actual.py::TestACMechanismPartitionScheme -q > /tmp/ac2.log 2>&1; tail -20 /tmp/ac2.log`
Expected: PASS (2 tests).

- [ ] **Step 5: Reconcile the existing tests that codified the leak**

Run the full AC suite to see what shifts:
`uv run pytest test/test_actual.py -q > /tmp/ac3.log 2>&1; tail -30 /tmp/ac3.log`

Expected failures and their fixes (values measured above):

1. `test_sia_cause_direction` (`test_actual.py:893`) asserts `sia_cause.alpha ==
   0.0`. Under the fix AC uses its own default `JOINT_PARTITION_ALL`, so the
   CAUSE α is the paper value. Update the assertion and docstring:

```python
    def test_sia_cause_direction(self, transition):
        """Cause-direction SIA under the AC default (JOINT_PARTITION_ALL).

        The AC formalism uses its own ``mechanism_partition_scheme`` (default
        ``JOINT_PARTITION_ALL``, the 2019-paper family), independent of the IIT
        ``mechanism_partition_scheme`` pinned by ``IIT_3_CONFIG``. For a
        first-order occurrence the paper permits only the full cut, giving
        α = ρ = log₂(4/3). Selecting ``JOINT_BIPARTITION`` on the AC field is a
        deliberate variant that admits the paper-forbidden m=1 partitions and
        drives the cause direction to 0 — see
        :meth:`test_sia_cause_direction_bipartition_variant`.
        """
        sia_cause = actual.sia(transition, Direction.CAUSE)
        assert np.isclose(sia_cause.alpha, np.log2(4 / 3))
        assert sia_cause.direction == Direction.CAUSE

    def test_sia_cause_direction_bipartition_variant(self, transition):
        """Selecting JOINT_BIPARTITION on the AC field reproduces the m=1
        deflation to α = 0 as a deliberate, opt-in variant."""
        with config.override(
            actual_causation=replace(
                config.formalism.actual_causation,
                mechanism_partition_scheme="JOINT_BIPARTITION",
            )
        ):
            sia_cause = actual.sia(transition, Direction.CAUSE)
        assert sia_cause.alpha == 0.0
```

2. `test_prevention` (`test_actual.py:915-931`) overrides the *IIT* field to
   `WEDGE_TRIPARTITION`. Repoint the override to the *AC* field; the asserted
   values are identical (measured above). Replace the decorator and docstring:

```python
    @config.override(
        actual_causation=replace(
            config.formalism.actual_causation,
            mechanism_partition_scheme="WEDGE_TRIPARTITION",
        )
    )
    def test_prevention(self, prevention):
        """Prevention example under an explicit AC ``WEDGE_TRIPARTITION`` choice.

        The scheme is selected on the actual-causation config field (a deliberate
        variant), not silently inherited from the IIT field. On this example the
        AC default ``JOINT_PARTITION_ALL`` gives the same values.
        """
        assert np.isclose(actual.sia(prevention, Direction.CAUSE).alpha, np.log2(4 / 3))
        assert actual.sia(prevention, Direction.EFFECT).alpha == 0.0
        assert actual.sia(prevention, Direction.BIDIRECTIONAL).alpha == 0.0
```

If `uv run pytest test/test_actual.py -q` reveals any *other* shifted
assertion, update it only to the value computed under the AC default
`JOINT_PARTITION_ALL`, documenting in the docstring that the prior value was
computed under the leaked IIT scheme. Do not change any non-AC test.

- [ ] **Step 6: Run the full AC suite**

Run: `uv run pytest test/test_actual.py -q > /tmp/ac4.log 2>&1; tail -15 /tmp/ac4.log`
Expected: all pass. Read the summary line.

- [ ] **Step 7: Commit**

```bash
git add pyphi/formalism/actual_causation/compute.py test/test_actual.py
git commit -m "Route AC partitioning through the AC mechanism_partition_scheme field"
```

---

### Task 2: Document the field, changelog, and full verification

**Files:**
- Modify: `pyphi/conf/formalism.py` (the `ActualCausationConfig.mechanism_partition_scheme` field, `:148`)
- Create: `changelog.d/ac-partition-scheme.fix.md`

- [ ] **Step 1: Document the field semantics**

In `pyphi/conf/formalism.py`, add a comment above the
`mechanism_partition_scheme` field at `:148` (inside `ActualCausationConfig`)
so the configurable choice is informed:

```python
    # The partition family for actual-causation MIP search. JOINT_PARTITION_ALL
    # is the Albantakis et al. (2019) family (Eq. 7 + Fig. 3B: all partitions of
    # the occurrence, excluding the m=1 non-full-cut cases forbidden for
    # first-order occurrences). Other registered schemes are deliberate
    # variants — notably JOINT_BIPARTITION admits those m=1 partitions and so
    # yields α below the published values on first-order occurrences.
    mechanism_partition_scheme: str = "JOINT_PARTITION_ALL"
```

- [ ] **Step 2: Add the changelog fragment**

```bash
cat > changelog.d/ac-partition-scheme.fix.md <<'EOF'
Actual-causation partition enumeration now reads its own
`actual_causation.mechanism_partition_scheme` config field instead of silently
inheriting the IIT `mechanism_partition_scheme`. Under an IIT 3.0 pin the AC α
of first-order occurrences was deflated by the paper-forbidden bipartition
family; AC now defaults to the 2019-paper `JOINT_PARTITION_ALL` family
regardless of the IIT setting.
EOF
```

- [ ] **Step 3: Full verification — doctest sweep**

Run: `uv run pytest > /tmp/ac_full.log 2>&1; tail -30 /tmp/ac_full.log`
Expected: green (no path argument, so the `pyphi/` doctest sweep runs; the new
helper docstring has no doctest). Read the summary line, not the exit code.

- [ ] **Step 4: Full verification — golden regression + the other IIT_3_CONFIG AC callers**

Run:
`uv run pytest test/integration/test_golden_regression.py test/models/test_result_config_snapshot.py test/formalism/test_big_phi.py test/formalism/test_complexes.py -q > /tmp/ac_gold.log 2>&1; tail -20 /tmp/ac_gold.log`
Expected: green. If any AC-derived golden shifted, regenerate it deliberately
(these files pin `IIT_3_CONFIG`; only AC-derived values may change, and only
toward the `JOINT_PARTITION_ALL` result). Document any regenerated golden in the
commit message.

- [ ] **Step 5: Commit**

```bash
git add pyphi/conf/formalism.py changelog.d/ac-partition-scheme.fix.md
git commit -m "Document the AC mechanism-partition scheme field and add changelog"
```

---

## Self-Review

**Spec coverage.** §1 bug → Task 1 (the two call-site swaps kill the leak). §2.1
fix mechanism → Task 1 Step 3 (implemented as two-site resolution, the simpler
equivalent of the spec's threading; behavior-identical). §2.2 IIT paths untouched
→ the global `mechanism_partitions()` is unchanged; only AC's two call sites move
to the AC helper. §2.3 configurable + informed → Task 1 (no restriction, registry
lookup validates names) + Task 2 Step 1 (field docs). §3.1 regression guarantees
→ Task 1 Step 1 (both tests). §3.2 expectation changes → Task 1 Step 5. §3.3
verification → Task 2 Steps 3–4. §4 scope → no partition schemes added, no config
constraint. Covered.

**Placeholder scan.** No "TBD"/"handle edge cases". Task 1 Step 5's "if any other
assertion shifts" is bounded and instructed (only AC values, only toward
`JOINT_PARTITION_ALL`), not a placeholder — the measured blast radius is
`test_sia_cause_direction` and `test_prevention` only.

**Type consistency.** `_ac_mechanism_partitions(mechanism, purview,
node_labels=None)` matches the signature the two call sites invoke
(`(mechanism, purview, transition.node_labels)`), identical to the global
`mechanism_partitions` it replaces.
