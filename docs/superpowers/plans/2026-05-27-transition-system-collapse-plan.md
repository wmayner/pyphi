# TransitionSystem → System Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `TransitionSystem` (in `pyphi/actual.py`) onto `System` (in `pyphi/system.py`) by exposing the "extended background" as a parameterizable `external_indices` field on `System`, eliminating ~400 lines of duplicated delegation. As a side effect, retire the last call to `Substrate._legacy_binary_joint()` and inherit AC k-ary support automatically.

**Architecture:** Convert `System.external_indices` from a `@cached_property` to an optional dataclass field; resolve `None` to the today-default (`substrate - node_indices`) in `__post_init__`. `TransitionSystem` shrinks to a façade that constructs an `_underlying_system` with `external_indices = substrate - cause_indices` and delegates the System protocol surface via `__getattr__`. AC k-ary works automatically because System already handles k-ary.

**Tech Stack:** Python 3.12+, frozen dataclasses, `cached_property`, `__getattr__` delegation, pytest, ruff, pyright.

**Spec:** [docs/superpowers/specs/2026-05-27-transition-system-collapse-design.md](../specs/2026-05-27-transition-system-collapse-design.md) (committed at `f3f70996`).

**Branch base:** `2.0` at HEAD `f3f70996` (just after the spec commit).

---

## Task 0: Pre-flight worktree setup

**Files:** N/A (worktree creation)

- [ ] **Step 1: Create the worktree using superpowers:using-git-worktrees**

Worktree path: `../pyphi-tscollapse`
Branch: `feature/transition-system-collapse` (branched off `2.0` at `f3f70996`)

Use the `superpowers:using-git-worktrees` skill to create the worktree. After creation, all subsequent work happens inside the worktree.

- [ ] **Step 2: Verify the worktree is clean and at the expected HEAD**

Run:
```bash
cd ../pyphi-tscollapse
git log --oneline -1
git status --short
```
Expected: HEAD shows `f3f70996 Add TransitionSystem collapse design spec`; status shows clean working tree (no untracked or modified files).

- [ ] **Step 3: Confirm the test baseline is green before any changes**

Run:
```bash
uv run pytest test/test_system.py test/test_actual.py -x -q --no-header 2>&1 | tail -5
```
Expected: all tests pass (something like `77 passed, 2 skipped in 0.X s`).

---

## Task 1: System gains `external_indices` field

**Files:**
- Modify: `pyphi/system.py:38-160` (dataclass definition, `__post_init__`, `__eq__`, `__hash__`, `to_json`, delete `@cached_property external_indices`)
- Test: `test/test_system.py` (new tests appended)

- [ ] **Step 1: Write failing tests for the `external_indices` override**

Append to `test/test_system.py`:

```python
def test_system_external_indices_default_matches_today() -> None:
    """When external_indices=None (default), resolves to substrate - node_indices."""
    from pyphi import examples

    s = examples.basic_subsystem()
    expected = tuple(sorted(set(range(s.substrate.size)) - set(s.node_indices)))
    assert s.external_indices == expected


def test_system_external_indices_explicit_override_accepted() -> None:
    """An explicit override is stored and accepted."""
    from pyphi import examples

    base = examples.basic_subsystem()
    override = (0,)
    s = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=base.node_indices,
        external_indices=override,
    )
    assert s.external_indices == override


def test_system_external_indices_rejects_out_of_range() -> None:
    """Indices must be in range(substrate.size)."""
    from pyphi import examples

    base = examples.basic_subsystem()
    with pytest.raises(ValueError, match="out of range"):
        System(
            substrate=base.substrate,
            state=base.state,
            node_indices=base.node_indices,
            external_indices=(99,),
        )


def test_system_external_indices_rejects_unsorted() -> None:
    """Override must be sorted."""
    from pyphi import examples

    base = examples.basic_subsystem()
    with pytest.raises(ValueError, match="sorted"):
        System(
            substrate=base.substrate,
            state=base.state,
            node_indices=(0, 1, 2),
            external_indices=(2, 0),
        )


def test_system_external_indices_rejects_duplicates() -> None:
    """Override must have unique entries."""
    from pyphi import examples

    base = examples.basic_subsystem()
    with pytest.raises(ValueError, match="duplicate"):
        System(
            substrate=base.substrate,
            state=base.state,
            node_indices=(0, 1, 2),
            external_indices=(0, 0),
        )


def test_system_external_indices_included_in_eq() -> None:
    """Two Systems differing only in external_indices are not equal."""
    from pyphi import examples

    base = examples.basic_subsystem()
    s1 = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1, 2),
        external_indices=(),
    )
    s2 = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1, 2),
        external_indices=None,  # resolves to () since node_indices is full substrate
    )
    # Both resolve to () so equal:
    assert s1 == s2

    s3_full = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1),
    )
    s3_override = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1),
        external_indices=(),
    )
    # default resolves to (2,), override is () — should not be equal
    assert s3_full != s3_override


def test_system_external_indices_included_in_hash() -> None:
    """Hash distinguishes Systems with different external_indices."""
    from pyphi import examples

    base = examples.basic_subsystem()
    s1 = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1),
    )
    s2 = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1),
        external_indices=(),
    )
    assert hash(s1) != hash(s2)


def test_system_external_indices_apply_cut_propagates() -> None:
    """apply_cut returns new System with external_indices preserved."""
    from pyphi import examples
    from pyphi.models.partitions import NullCut

    base = examples.basic_subsystem()
    s = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1),
        external_indices=(2,),
    )
    new_cut = NullCut((0, 1), base.substrate.node_labels)
    s2 = s.apply_cut(new_cut)
    assert s2.external_indices == (2,)


def test_system_external_indices_overlap_with_node_indices_allowed() -> None:
    """The override may overlap with node_indices (AC use case)."""
    from pyphi import examples

    base = examples.basic_subsystem()
    # node_indices = (0, 1, 2); override includes 1, which is in node_indices.
    s = System(
        substrate=base.substrate,
        state=base.state,
        node_indices=(0, 1, 2),
        external_indices=(1,),
    )
    assert s.external_indices == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
uv run pytest test/test_system.py::test_system_external_indices_default_matches_today test/test_system.py::test_system_external_indices_explicit_override_accepted -v 2>&1 | tail -20
```
Expected: tests fail with `TypeError: System.__init__() got an unexpected keyword argument 'external_indices'` or similar.

- [ ] **Step 3: Convert `external_indices` from `@cached_property` to dataclass field**

In `pyphi/system.py:38-46`, update the dataclass header:

```python
@dataclass(frozen=True, eq=False)
class System:
    """A substrate evaluated in a specific state over a node subset, with partition."""

    substrate: Substrate
    state: State
    node_indices: NodeIndices = field(default=None)  # type: ignore[assignment]
    partition: DirectedBipartition = field(default=None)  # type: ignore[assignment]
    external_indices: tuple[int, ...] | None = None
```

Delete the `@cached_property external_indices` at lines 155-158:

```python
# DELETE these lines:
@cached_property
def external_indices(self) -> tuple[int, ...]:
    all_indices = set(range(self.substrate.size))
    return tuple(sorted(all_indices - set(self.node_indices)))
```

- [ ] **Step 4: Resolve `external_indices` default + validate override in `__post_init__`**

In `pyphi/system.py:47-76`, add at the end of `__post_init__` (after the existing partition logic, before the `validate.state_reachable` block):

```python
        # Resolve external_indices: None means compute substrate-minus-system default;
        # else validate the explicit override.
        if self.external_indices is None:
            all_indices = set(range(substrate.size))
            object.__setattr__(
                self,
                "external_indices",
                tuple(sorted(all_indices - set(self.node_indices))),
            )
        else:
            ext = tuple(self.external_indices)
            for i in ext:
                if not (0 <= i < substrate.size):
                    raise ValueError(
                        f"external_indices contains out of range index {i}; "
                        f"must satisfy 0 <= i < {substrate.size}"
                    )
            if list(ext) != sorted(ext):
                raise ValueError(
                    f"external_indices must be sorted; got {ext}"
                )
            if len(set(ext)) != len(ext):
                raise ValueError(
                    f"external_indices must not contain duplicates; got {ext}"
                )
            object.__setattr__(self, "external_indices", ext)
```

- [ ] **Step 5: Update `__eq__` and `__hash__` to include `external_indices`**

In `pyphi/system.py:98-109`, replace:

```python
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.state == other.state
            and self.node_indices == other.node_indices
            and self.partition == other.partition
            and self.external_indices == other.external_indices
        )

    def __hash__(self) -> int:
        return hash((
            self.substrate,
            self.state,
            self.node_indices,
            self.partition,
            self.external_indices,
        ))
```

- [ ] **Step 6: Update `to_json` to serialize `external_indices`**

In `pyphi/system.py:728-734`, replace:

```python
    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "state": list(self.state),
            "node_indices": list(self.node_indices),
            "partition": self.partition,
            "external_indices": list(self.external_indices),
        }
```

- [ ] **Step 7: Run new tests, verify they pass**

Run:
```bash
uv run pytest test/test_system.py -k "external_indices" -v 2>&1 | tail -20
```
Expected: all 9 new `test_system_external_indices_*` tests pass.

- [ ] **Step 8: Run the broader test suite, verify no regressions**

Run:
```bash
uv run pytest test/test_system.py test/test_helpers.py test/test_invariants.py test/test_golden_regression.py test/test_validate.py -x -q --no-header 2>&1 | tail -10
```
Expected: all pass; goldens 25/25 byte-identical.

- [ ] **Step 9: Run pyright and ruff on the touched file**

Run:
```bash
uv run ruff check pyphi/system.py test/test_system.py
uv run ruff format --check pyphi/system.py test/test_system.py
.venv/bin/python -m pyright pyphi/system.py
```
Expected: all clean.

- [ ] **Step 10: Commit**

Run:
```bash
git add pyphi/system.py test/test_system.py
git diff --cached --stat
```
Expected: only `pyphi/system.py` and `test/test_system.py` staged.

Then:
```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
System.external_indices becomes an optional dataclass field

Convert the @cached_property to a field with default None; resolve in
__post_init__ to substrate-minus-node_indices (today's behavior).
Callers can now pass an explicit override at construction. Validation
in __post_init__ rejects out-of-range, unsorted, and duplicate
indices. __eq__, __hash__, and to_json updated to include the field.

The override allows external_indices to overlap with node_indices,
which is the AC use case (freeze effect-but-not-cause units at
observed state even though they're nominally in the system).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Then:
```bash
git show --stat HEAD
```
Expected: commit lands with exactly the two files; pre-commit hooks (ruff + ruff format + pyright + towncrier-check) all green.

---

## Task 2: TransitionSystem delegates `effect_tpm` through `_underlying_system`

**Files:**
- Modify: `pyphi/actual.py:239-277` (`_underlying_system`, `effect_tpm`, `proper_effect_tpm`)
- Modify: `test/test_actual.py:252-256` (remove `legacy_joint` shape assertion)

- [ ] **Step 1: Update `TransitionSystem._underlying_system` to pass `external_indices`**

In `pyphi/actual.py:239-247`, replace:

```python
    @cached_property
    def _underlying_system(self) -> Any:
        state = self.after_state if self.direction == Direction.CAUSE else self.before_state
        external = () if self.noise_background else tuple(
            sorted(set(self.substrate.node_indices) - set(self.cause_indices))
        )
        with config.override(validate_system_states=False):
            return System(
                substrate=self.substrate,
                state=state,
                node_indices=self.node_indices,
                partition=self.partition,
                external_indices=external,
            )
```

Note: the underlying System's `state` was already `before_state` (passed in), but per the spec the AC paper's "state for mechanism evaluation" depends on direction. The TransitionSystem's `state` property already returns the direction-dependent state; mirror that in the underlying System.

- [ ] **Step 2: Delete `TransitionSystem.effect_tpm` and `proper_effect_tpm`**

In `pyphi/actual.py:253-277`, delete these two methods entirely:

```python
# DELETE:
@cached_property
def effect_tpm(self) -> Any:
    from pyphi.core.tpm.joint import JointTPM as _TypedTPM
    from pyphi.core.tpm.marginalization import effect_tpm as _marginalize_effect

    typed = _TypedTPM(self.substrate._legacy_binary_joint())
    external_state = utils.state_of(self.external_indices, self.before_state)
    background = dict(zip(self.external_indices, external_state, strict=False))
    result = _marginalize_effect(typed, background)
    return result

# DELETE:
@cached_property
def proper_effect_tpm(self) -> Any:
    return np.asarray(self.effect_tpm.squeeze())[..., list(self.node_indices)]
```

After deletion, `effect_tpm` and `proper_effect_tpm` will resolve through `_underlying_system` because the next task adds `__getattr__` — but Task 2 is intermediate. To keep tests green for this intermediate state, add explicit delegation properties:

```python
@cached_property
def effect_tpm(self) -> Any:
    return self._underlying_system.effect_tpm

@cached_property
def proper_effect_tpm(self) -> Any:
    return self._underlying_system.proper_effect_tpm
```

(Task 3 deletes these explicit properties in favor of `__getattr__`.)

- [ ] **Step 3: Delete the `legacy_joint` shape assertion in `test/test_actual.py`**

In `test/test_actual.py:248-256`, replace:

```python
    transition = actual.Transition(
        substrate, state, state, (0,), (0,), noise_background=True
    )
    # transition.{cause,effect}_system.effect_tpm goes through the legacy
    # binary path, so compare against the legacy binary joint shape.
    legacy_joint = substrate._legacy_binary_joint()
    assert np.array_equal(np.asarray(transition.cause_system.effect_tpm), legacy_joint)
    assert np.array_equal(np.asarray(transition.effect_system.effect_tpm), legacy_joint)
```

with:

```python
    transition = actual.Transition(
        substrate, state, state, (0,), (0,), noise_background=True
    )
    # Behavioral assertion above (np.isclose(transition._ratio(...), 0.415037))
    # is the load-bearing check for noise_background semantics.
```

- [ ] **Step 4: Run AC test suite, verify all pass**

Run:
```bash
uv run pytest test/test_actual.py -x -q --no-header 2>&1 | tail -10
```
Expected: 77 passed, 2 skipped (same as baseline).

- [ ] **Step 5: Grep-audit that `pyphi/actual.py` no longer calls `_legacy_binary_joint`**

Run:
```bash
grep -n "_legacy_binary_joint" pyphi/actual.py
```
Expected: empty output (no matches).

- [ ] **Step 6: Run goldens to verify byte-identical results**

Run:
```bash
uv run pytest test/test_golden_regression.py -v --no-header 2>&1 | tail -10
```
Expected: 25/25 passed.

- [ ] **Step 7: Run pyright and ruff**

Run:
```bash
uv run ruff check pyphi/actual.py test/test_actual.py
uv run ruff format --check pyphi/actual.py test/test_actual.py
.venv/bin/python -m pyright pyphi/actual.py
```
Expected: all clean.

- [ ] **Step 8: Commit**

Run:
```bash
git add pyphi/actual.py test/test_actual.py
git diff --cached --stat
```

Then:
```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
TransitionSystem.effect_tpm delegates through _underlying_system

Replace the legacy binary joint path with delegation to
_underlying_system.effect_tpm, which uses the substrate's FactoredTPM
and supports k-ary alphabets. Behavior preserved for binary
substrates via System's existing SBN bridge. proper_effect_tpm
delegates likewise.

Removes the last call to Substrate._legacy_binary_joint() from the
live AC pipeline.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Then verify:
```bash
git show --stat HEAD
```

---

## Task 3: TransitionSystem `__getattr__` delegation

**Files:**
- Modify: `pyphi/actual.py:170-705` (the full `TransitionSystem` class — delete ~30 passthrough methods)

- [ ] **Step 1: Define `TRANSITION_SYSTEM_OWN_ATTRS` frozenset**

At module level in `pyphi/actual.py`, after the imports and registries, before the `TransitionSystem` class definition (~line 168), add:

```python
# Attributes TransitionSystem handles locally (rather than delegating to
# its underlying System). Anything not in this set falls through __getattr__
# to self._underlying_system.
TRANSITION_SYSTEM_OWN_ATTRS: frozenset[str] = frozenset({
    # Dataclass fields:
    "substrate", "before_state", "after_state", "cause_indices",
    "effect_indices", "direction", "partition", "noise_background",
    # AC-specific computed properties:
    "node_indices", "state", "proper_state", "external_indices",
    "partition_indices", "partition_node_labels", "is_partitioned",
    "node_labels", "size", "tpm_size",
    # AC-specific methods:
    "apply_cut", "from_substrate", "to_json",
    # Caches and internals:
    "_underlying_system",
    # IIT-formalism stubs (raise NotImplementedError):
    "sia", "ces", "distinctions", "find_mip", "cause_mip", "effect_mip",
    "phi_cause_mip", "phi_effect_mip", "phi", "find_mice", "mic", "mie",
    "phi_max", "distinction", "all_distinctions", "evaluate_partition",
})
```

- [ ] **Step 2: Add `__getattr__` to `TransitionSystem`**

After the `__str__` method on `TransitionSystem` (~line 374), add:

```python
def __getattr__(self, name: str) -> Any:
    """Delegate to the underlying System for anything not handled locally.

    See TRANSITION_SYSTEM_OWN_ATTRS for the locally-handled allow-list.
    """
    if name.startswith("_") or name in TRANSITION_SYSTEM_OWN_ATTRS:
        raise AttributeError(name)
    return getattr(self._underlying_system, name)
```

- [ ] **Step 3: Add an empirical verification test for the delegation**

Append to `test/test_actual.py`:

```python
def test_transition_system_delegates_protocol_surface() -> None:
    """Every PUBLIC_SYSTEM_ATTRS attribute is accessible on TransitionSystem,
    either locally or via __getattr__ delegation."""
    from pyphi.protocols import PUBLIC_SYSTEM_ATTRS

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    for attr in PUBLIC_SYSTEM_ATTRS:
        assert hasattr(ts, attr), f"Missing attribute: {attr}"


def test_transition_system_delegated_repertoire_matches_underlying() -> None:
    """Delegated methods return the same values as the underlying System."""
    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    rep_ts = ts.cause_repertoire((1,), (1,))
    rep_us = ts._underlying_system.cause_repertoire((1,), (1,))
    assert np.array_equal(np.asarray(rep_ts), np.asarray(rep_us))
```

- [ ] **Step 4: Run the new delegation tests, verify they pass**

Run:
```bash
uv run pytest test/test_actual.py::test_transition_system_delegates_protocol_surface test/test_actual.py::test_transition_system_delegated_repertoire_matches_underlying -v 2>&1 | tail -10
```
Expected: both pass. Note: at this point all delegated methods still exist explicitly on TransitionSystem; `__getattr__` is added but not yet load-bearing.

- [ ] **Step 5: Delete the explicit passthrough methods on `TransitionSystem`**

In `pyphi/actual.py`, delete these methods (they fall through to `__getattr__`):

Properties to delete (lines roughly 250-336):
- `cause_tpm`, `effect_tpm`, `proper_cause_tpm`, `proper_effect_tpm`,
- `cm`, `proper_cm`, `connectivity_matrix`,
- `nodes`, `_index2node`,
- `null_distinction`, `null_concept`,
- (Keep: `node_indices`, `state`, `external_indices`, `proper_state`,
  `partition_indices`, `partition_node_labels`, `is_partitioned`,
  `node_labels`, `size`, `tpm_size`)

Methods to delete (lines roughly 375-606):
- `cause_repertoire`, `effect_repertoire`, `repertoire`,
- `unconstrained_cause_repertoire`, `unconstrained_effect_repertoire`,
  `unconstrained_repertoire`,
- `partitioned_repertoire`,
- `expand_cause_repertoire`, `expand_effect_repertoire`, `expand_repertoire`,
- `forward_cause_repertoire`, `forward_effect_repertoire`,
  `forward_repertoire`,
- `unconstrained_forward_cause_repertoire`,
  `unconstrained_forward_effect_repertoire`,
  `unconstrained_forward_repertoire`,
- `forward_cause_probability`, `forward_effect_probability`,
  `forward_probability`,
- `cause_info`, `effect_info`, `cause_effect_info`,
- `intrinsic_information`,
- `potential_purviews`,
- `indices2nodes`,
- `cache_info`, `clear_caches`.

Keep these (with their `NotImplementedError` bodies, since the underlying
System's implementations would otherwise be called):
- `sia`, `ces`, `distinctions`,
- `find_mip`, `cause_mip`, `effect_mip`,
- `phi_cause_mip`, `phi_effect_mip`, `phi`,
- `find_mice`, `mic`, `mie`, `phi_max`,
- `distinction`, `all_distinctions`,
- `evaluate_partition`.

- [ ] **Step 6: Run AC suite, verify all pass**

Run:
```bash
uv run pytest test/test_actual.py -x -q --no-header 2>&1 | tail -10
```
Expected: 77 passed, 2 skipped + 2 new delegation tests = 79 passed, 2 skipped.

- [ ] **Step 7: Run System + golden + invariant suite, verify no regressions**

Run:
```bash
uv run pytest test/test_system.py test/test_golden_regression.py test/test_invariants.py test/test_helpers.py test/test_validate.py -x -q --no-header 2>&1 | tail -10
```
Expected: all pass; goldens 25/25 byte-identical.

- [ ] **Step 8: Run pyright on the touched file**

Run:
```bash
.venv/bin/python -m pyright pyphi/actual.py
```
Expected: 0 errors, 0 warnings.

- [ ] **Step 9: Confirm line-count target hit**

Run:
```bash
wc -l pyphi/actual.py
```
Expected: roughly 1400-1450 lines (down from 1802). The `TransitionSystem` class itself should be ~150-200 lines (down from ~550).

- [ ] **Step 10: Run ruff**

Run:
```bash
uv run ruff check pyphi/actual.py test/test_actual.py
uv run ruff format --check pyphi/actual.py test/test_actual.py
```
Expected: clean.

- [ ] **Step 11: Commit**

Run:
```bash
git add pyphi/actual.py test/test_actual.py
git diff --cached --stat
```

Then:
```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
TransitionSystem delegates System protocol surface via __getattr__

Add TRANSITION_SYSTEM_OWN_ATTRS allow-list and __getattr__ that
delegates to _underlying_system for anything not handled locally.
Delete ~30 passthrough methods (cause_repertoire, effect_repertoire,
forward_*, expand_*, cause_info, etc.) that mechanically called into
repertoire_algebra; those calls now go through the underlying System.

Delete the duplicated cause_tpm, effect_tpm, cm, proper_cm, nodes,
null_distinction properties on TransitionSystem (delegated).

NotImplementedError stubs for IIT-formalism methods (sia, ces,
find_mip, etc.) remain — they must override the underlying System's
real implementations.

TransitionSystem shrinks from ~550 to ~150 lines.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: k=3 paper-figure test

**Files:**
- Modify: `test/test_actual.py` (new test function appended)

- [ ] **Step 1: Write the failing test for AC paper Figure 11 (three-candidate voting)**

Append to `test/test_actual.py`:

```python
def test_paper_fig11_three_candidate_voting_alpha_c() -> None:
    """AC paper Figure 11, Section 3.6: three-candidate voting (k=3).

    Substrate: 7 voters (ABCDEFG) ∈ {0, 1, 2} → 1 winner (W) ∈ {0, 1, 2, 3}.
    Transition: ABCDEFG = (1,1,1,1,1,2,2) → W = 1.

    The paper reports α_c = 1.893 for the actual cause
    {ABCD = 1111, BCDE = 1111} → {W = 1}.

    This test locks in k-ary AC support by exercising the full Transition
    → SIA path with a k=3 substrate. The α_c value is compared within
    config.numerics.precision = 13.
    """
    import numpy as np

    from pyphi import Substrate, examples
    from pyphi.actual import Transition

    # The paper does not give a closed-form TPM for the 7-voter, 3-candidate
    # majority gate; build it from the rule "W = majority candidate if any
    # candidate has > n/2 votes, else 0".
    n_voters = 7
    alphabet_size = 3
    # n_states = alphabet_size ** n_voters

    def majority(votes: tuple[int, ...]) -> int:
        from collections import Counter
        counts = Counter(votes)
        # candidate with > 3 votes wins (simple majority for 7 voters)
        for candidate, count in counts.most_common():
            if count > n_voters / 2:
                return candidate
        return 0  # tie

    # Build the joint TPM in factored form: for each input state, W is deterministic.
    # ABCDEFG factors are identity (input voters don't change between t-1 and t).
    # Actually for AC, we only need the TPM that maps input state at t-1 to
    # output state at t. Construct a substrate with 8 nodes (7 voters + 1 W),
    # where the voters are stationary and W follows the majority rule.

    # (Implementation detail: this is straightforward but verbose; see
    # examples.three_candidate_voting() if such a helper exists, else
    # build inline.)

    substrate = examples.three_candidate_voting()  # if available
    # alternative: build inline (see fallback in test fixture)

    before_state = (1, 1, 1, 1, 1, 2, 2, 0)  # 7 voters + W=0 at t-1
    after_state = (1, 1, 1, 1, 1, 2, 2, 1)   # W = 1 at t

    transition = Transition(
        substrate=substrate,
        before_state=before_state,
        after_state=after_state,
        cause_indices=tuple(range(7)),  # ABCDEFG
        effect_indices=(7,),  # W
    )

    # Verify the effect TPM is k-ary (not binary):
    assert transition.cause_system.effect_tpm.alphabet_sizes != (2,) * 8, (
        "effect_tpm.alphabet_sizes should not be all-binary for k=3 substrate"
    )

    # Compute α_c for the candidate cause {ABCD = 1111} ∪ {BCDE = 1111}
    # acting on effect {W = 1}.
    # The paper reports α_c = 1.893 for this transition.

    # Find the actual cause; verify α_c matches paper within tolerance.
    actual_cause = transition.find_actual_cause(  # exact API may differ; consult docs
        effect=(7,),
    )
    paper_alpha_c = 1.893
    assert np.isclose(
        actual_cause.alpha,
        paper_alpha_c,
        atol=10 ** -2,  # paper reports to 3 decimal places; allow 0.01 slack
    ), f"α_c={actual_cause.alpha}, paper reports {paper_alpha_c}"
```

**Implementation note:** the `examples.three_candidate_voting()` helper
may not exist yet. If it doesn't, the implementer should either:
(a) add the helper to `pyphi/examples.py` with the substrate
construction, or
(b) build the substrate inline in the test fixture.

Pick whichever fits the existing pattern in `pyphi/examples.py`.

- [ ] **Step 2: Run test, verify it passes (or document deviation)**

Run:
```bash
uv run pytest test/test_actual.py::test_paper_fig11_three_candidate_voting_alpha_c -v 2>&1 | tail -20
```

Expected: passes if PyPhi's α_c reproduces the paper's 1.893 within 0.01.

**If it doesn't pass:** investigate whether the divergence is (a) a
substrate-construction bug, (b) a code bug exposed by k=3 substrates, or
(c) a genuine numerical mismatch with the paper. Report the case in a
test docstring with the observed value and mark `@pytest.mark.xfail(reason=...)`
if the case is real but non-blocking. Do not gate the collapse on this
test's pass.

- [ ] **Step 3: Run full AC suite, verify no regression**

Run:
```bash
uv run pytest test/test_actual.py -x -q --no-header 2>&1 | tail -10
```
Expected: all pass (with the new test passing or xfail-marked).

- [ ] **Step 4: Commit**

Run:
```bash
git add test/test_actual.py
# also add pyphi/examples.py if the helper was added there:
git add pyphi/examples.py  # only if modified
git diff --cached --stat
```

Then:
```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
AC paper Figure 11: three-candidate voting k=3 test

Exercises k-ary AC through the full Transition → SIA path using the
2019 paper's Section 3.6 example (7 voters, 3 candidates → 1 winner).
Locks in the k-ary code path with an explicit
`effect_tpm.alphabet_sizes != (2,) * n` assertion, and compares α_c
against the paper's reported value within 0.01.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Changelog and docstring polish

**Files:**
- Create: `changelog.d/transition-system-collapse.refactor.md`
- Modify: `pyphi/actual.py` (TransitionSystem class docstring)
- Modify: `pyphi/system.py` (external_indices field docstring)

- [ ] **Step 1: Create the changelog fragment**

Run:
```bash
cat > changelog.d/transition-system-collapse.refactor.md << 'EOF'
Collapsed ``TransitionSystem`` (in ``pyphi/actual.py``) onto ``System`` (in ``pyphi/system.py``) via a new optional ``external_indices`` field on ``System``. The field parameterizes which substrate units are conditioned at observed state when computing repertoires; when omitted, ``System`` resolves it to ``substrate - node_indices`` (today's behavior). ``TransitionSystem`` constructs its underlying ``System`` with ``external_indices = substrate - cause_indices``, then delegates the System protocol surface via ``__getattr__``. The class shrinks from ~550 lines to ~150 lines.

The actual-causation pipeline now inherits k-ary substrate support automatically — the last call to ``Substrate._legacy_binary_joint()`` from ``pyphi/actual.py`` is gone. Paper Figure 11 (three-candidate voting, Section 3.6) is now covered by a regression test.

No public API changes. ``TransitionSystem``'s protocol surface and call signatures are unchanged.
EOF
```

- [ ] **Step 2: Sharpen the `TransitionSystem` class docstring**

In `pyphi/actual.py`, find the `TransitionSystem` class docstring (currently lines 171-183) and replace with:

```python
class TransitionSystem:
    """A directional view of a state transition.

    Implements :class:`pyphi.protocols.SystemPublicInterface` by holding an
    underlying :class:`pyphi.system.System` and delegating the protocol
    surface via :meth:`__getattr__`. The underlying System is constructed
    with ``external_indices = substrate - cause_indices`` (or ``()`` when
    ``noise_background=True``), encoding the AC paper's Section 3.3
    extended-background convention applied to substrate units outside the
    cause set.

    For each :class:`Transition` instance, two :class:`TransitionSystem`
    views exist (one per :class:`Direction`), differing in the ``state``
    passed to the underlying System: ``after_state`` for CAUSE,
    ``before_state`` for EFFECT.
    """
```

- [ ] **Step 3: Add a docstring to the `external_indices` field**

In `pyphi/system.py`, on the dataclass field definition (around line 46), the field doesn't accept inline docstrings naturally, so add a module-level explanation. After the `System` class docstring (around line 40), add a paragraph:

Actually, dataclass fields can't have docstrings directly. Instead, expand the System class docstring (line 40):

```python
class System:
    """A substrate evaluated in a specific state over a node subset, with partition.

    The ``external_indices`` field specifies which substrate units are
    conditioned at observed state when computing repertoires. When
    ``None`` (the default), it resolves in ``__post_init__`` to
    ``substrate - node_indices`` — the "extended background" convention
    of IIT 4.0. An explicit override (used by ``TransitionSystem`` for
    actual-causation analysis) may overlap with ``node_indices``.
    """
```

- [ ] **Step 4: Run full pytest (no path argument, includes doctests)**

Run:
```bash
uv run pytest 2>&1 | tail -10
```
Expected: all tests pass; doctests collected; no failures.

- [ ] **Step 5: Run ruff and pyright on the full package**

Run:
```bash
uv run ruff check pyphi test
uv run ruff format --check pyphi test
.venv/bin/python -m pyright pyphi
```
Expected: all clean.

- [ ] **Step 6: Commit**

Run:
```bash
git add changelog.d/transition-system-collapse.refactor.md pyphi/actual.py pyphi/system.py
git diff --cached --stat
```

Then:
```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Changelog and docstrings for the TransitionSystem collapse

Adds the user-facing changelog fragment describing the collapse and
the k-ary AC inheritance. Sharpens the TransitionSystem and System
docstrings to describe the final state: TransitionSystem as a façade
over System via __getattr__ delegation, and System.external_indices
as the parameterizable extended-background set.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Then verify:
```bash
git show --stat HEAD
git log --oneline -6
```
Expected: 5 new commits since `f3f70996` (the spec commit).

---

## Final verification

After Task 5 lands, run the full acceptance gate:

- [ ] **End-to-end pytest with no path argument**

```bash
uv run pytest 2>&1 | tail -10
```
Expected: all pass, including doctests in `pyphi/` source modules.

- [ ] **Goldens byte-identical**

```bash
uv run pytest test/test_golden_regression.py -v 2>&1 | tail -10
```
Expected: 25/25 passed.

- [ ] **Pyright clean**

```bash
.venv/bin/python -m pyright pyphi
```
Expected: 0 errors, 0 warnings (matches baseline at `f3f70996`).

- [ ] **Ruff clean**

```bash
uv run ruff check pyphi test
uv run ruff format --check pyphi test
```
Expected: all clean.

- [ ] **No `_legacy_binary_joint` calls from `pyphi/actual.py`**

```bash
grep -n "_legacy_binary_joint" pyphi/actual.py
```
Expected: empty.

- [ ] **End-to-end smoke test**

```bash
uv run python <<'EOF'
import pyphi
from pyphi import examples
from pyphi.actual import Transition
from pyphi import Direction

sub = examples.basic_substrate()
state = (1, 0, 0)
ts = pyphi.System(sub, state, (0, 1, 2))
print(f"System.external_indices: {ts.external_indices}")

# AC path
sub2 = examples.actual_causation_substrate()  # or similar 3-node binary
t = Transition(sub2, (0, 1, 1), (1, 0, 0), (1, 2), (0,))
print(f"AC underlying external_indices: {t.cause_system._underlying_system.external_indices}")
print(f"AC cause_system delegation: {t.cause_system.cause_repertoire((1,), (1,))}")

print("Smoke test OK")
EOF
```
Expected: prints System.external_indices, AC underlying external_indices, a repertoire array, and "Smoke test OK".

- [ ] **Line-count win confirmed**

```bash
wc -l pyphi/actual.py
```
Expected: ~1400 lines (down from 1802); net reduction ~400 lines.

---

## Notes on saved-memory constraints

Implementers must observe:

- **Pre-commit hooks**: NEVER bypass with `--no-verify` or `SKIP=*`. If a hook fails, diagnose via `uv run ruff check <file>` / `.venv/bin/python -m pyright <file>` directly. The user's unstaged `typeCheckingMode = "off"` in pyproject.toml may silence `uv run pyright`; the pre-commit hook is authoritative.
- **gpgsign**: Use `git -c commit.gpgsign=false commit ...` for all commits. If 1Password agent errors persist, surface to user — do NOT auto-bypass.
- **Targeted `git add` only**: Never `git add .` or `git add -A`. The main repo has ~20 untracked items the user is working on in parallel — they must not be staged.
- **Doctest scope**: At every commit boundary in this plan, `uv run pytest` (no path argument) must pass. The fast-lane shortcut (`pytest test/`) skips doctests in `pyphi/` source modules.
- **No planning artifacts in code**: No "Phase A", "Task N", "TODO(P_x)", or ROADMAP IDs in source/comments/docstrings/changelog. Commit messages MAY reference "the spec" or "the plan".
- **Docstrings describe final state**: Not the migration journey. Don't say "previously this was..." or "old binary path now retired".
- **Per saved memory feedback_ask_before_push**: Do not push to origin without explicit per-action consent. The 2.0 branch has 18+ unpushed commits including this work.
