# TransitionSystem → System Collapse via `external_indices` Override

**Status:** Design

**Date:** 2026-05-27

**Branch base:** ``2.0`` at HEAD ``0a4ac15c``

## Goal

Collapse ``TransitionSystem`` (in ``pyphi/actual.py``) onto ``System`` (in
``pyphi/system.py``) by exposing the "extended background" set as a
parameterizable field on ``System``. Eliminate the ~400 lines of
duplicated repertoire-algebra delegation in ``TransitionSystem`` and the
copy-pasted ``proper_effect_tpm`` implementation. As a free side effect,
close the AC k-ary cutover by removing the last call to
``Substrate._legacy_binary_joint()``.

## Background

### Why the collapse is sound

``System`` and ``TransitionSystem`` both implement
``pyphi.protocols.SystemPublicInterface``. ``TransitionSystem`` already
holds an ``_underlying_system`` (a ``System`` with
``validate_system_states=False``) and delegates most of its surface to
it. The remaining duplication is mechanical — the only genuine semantic
divergence is which substrate units get frozen at observed state:

- **``System`` (Φ analysis):** freezes ``substrate - node_indices``
- **``TransitionSystem`` (AC, default):** freezes ``substrate - cause_indices``
- **``TransitionSystem`` with ``noise_background=True``:** freezes nothing

Both Φ and AC are specializations of the same operation: "evaluate a
system with some subset of substrate units treated as Section 3.3
extended background." Today ``System`` hardcodes the formula
``external_indices = substrate - node_indices``; ``TransitionSystem``
reimplements ``effect_tpm`` from scratch with a different formula. The
collapse exposes the formula as a field.

### Paper grounding (2019 Albantakis et al., AC)

The 2019 AC paper distinguishes two operations on variables outside the
occurrence:

1. **Causal marginalization (Eq. 2, the paper's default):** uniform-weighted
   average over ``W = V_{t-1} \ X_{t-1}``.
2. **Extended background (Section 3.3, "Distinct Background Conditions",
   presented as an option):** promote a V-variable to the background U',
   fixing it at observed state.

PyPhi has unified Φ and AC around the IIT 4.0 (2023) extended-background
convention by default. The "freeze substrate-outside-cause at observed
state" path is Section 3.3 applied universally rather than per-variable.
This is a settled modeling choice in PyPhi; not changing it here.
(Logged as a separate ROADMAP item: "AC default: extended background vs.
strict Eq. 2.")

The paper explicitly supports non-binary alphabets (Section 3.6, Figure
11: three-candidate voting). The post-collapse AC pipeline inherits
``System``'s k-ary support automatically.

### AC k-ary cutover, subsumed

The only remaining caller of ``Substrate._legacy_binary_joint()`` in the
live AC path is ``TransitionSystem.effect_tpm`` at
``pyphi/actual.py:261``. After the collapse, this method disappears
entirely (delegated through ``_underlying_system``), and AC inherits
k-ary support from the unchanged ``System.effect_tpm`` path. No
separate AC k-ary work is needed.

## Final Naming and Semantics

### `System.external_indices`

A new public optional field on the ``System`` dataclass:

```python
external_indices: tuple[int, ...] | None = None
```

- When ``None`` (default), ``System`` computes
  ``substrate.node_indices - node_indices`` — today's behavior, bit-identical.
- When set, the override is used directly without recomputation.

**Semantics:** ``external_indices`` is the set of substrate units to
condition on at observed state when computing repertoires.

**Invariants enforced in ``__post_init__`` when not ``None``:**

- All elements are valid substrate indices (``0 <= i < substrate.size``).
- Sorted, no duplicates.

**Invariants *not* enforced:**

- Disjointness from ``node_indices``. ``System`` (Φ usage) keeps this
  invariant by construction. ``TransitionSystem`` (AC usage) violates it:
  AC sets ``external_indices = substrate - cause_indices``, which
  overlaps with ``effect_indices \ cause_indices`` (effect-but-not-cause
  nodes get frozen at ``before_state`` even though they're in
  ``node_indices``). This overlap is well-defined: "freeze this
  substrate unit at observed state, regardless of whether it's nominally
  in ``node_indices``." The field docstring documents this.

**Resolution in code:**

Today, ``external_indices`` is a ``@cached_property`` on ``System`` that
computes ``substrate.node_indices - node_indices`` on first access. The
field-vs-property name collision is resolved by **converting the
cached_property to a dataclass field**, with default resolution moved to
``__post_init__``:

```python
@dataclass(frozen=True, eq=False)
class System:
    substrate: Substrate
    state: State
    node_indices: NodeIndices = field(default=None)
    partition: DirectedBipartition = field(default=None)
    external_indices: tuple[int, ...] | None = None  # NEW

    def __post_init__(self):
        # ... existing validation ...
        if self.external_indices is None:
            all_indices = set(range(self.substrate.size))
            object.__setattr__(
                self,
                "external_indices",
                tuple(sorted(all_indices - set(self.node_indices))),
            )
        else:
            # validate the explicit override
            ...
```

After ``__post_init__``, ``self.external_indices`` is always a concrete
``tuple[int, ...]`` — same public type and semantics as today's
cached_property. The ``@cached_property`` definition is deleted.

**Public API surface unchanged for non-override callers:** code reading
``system.external_indices`` gets the same resolved tuple it always did.

**JSON back-compat:** legacy fixtures don't store ``external_indices``;
deserializing into ``System(**dict)`` gives ``external_indices=None``,
which ``__post_init__`` resolves to the default. So two Systems with
identical ``node_indices`` have identical resolved ``external_indices``
and compare equal. Equality semantics preserved.

### `TransitionSystem` shrinks to a façade

The dataclass keeps its AC-specific fields:

```python
substrate, before_state, after_state, cause_indices, effect_indices,
direction, partition, noise_background
```

Plus a single ``_underlying_system`` cached property:

```python
@cached_property
def _underlying_system(self) -> System:
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

Most of the System protocol surface (``cause_tpm``, ``effect_tpm``,
``proper_cause_tpm``, ``proper_effect_tpm``, ``cm``, ``nodes``,
``cause_repertoire``, ``effect_repertoire``, ``repertoire``,
``unconstrained_*``, ``partitioned_repertoire``, ``expand_*``,
``forward_*``, ``cause_info``, ``effect_info``, ``cause_effect_info``,
``intrinsic_information``, ``potential_purviews``, ``indices2nodes``,
``cache_info``, ``clear_caches``, etc.) delegates to
``_underlying_system`` via ``__getattr__``.

**TransitionSystem keeps its own implementations of:**

- ``node_indices`` (computed as ``cause_indices ∪ effect_indices``)
- ``state`` (Direction-aware: ``after_state`` if CAUSE else ``before_state``)
- ``proper_state`` (computed off ``state``)
- ``external_indices`` (Direction-irrelevant, computed off ``cause_indices``)
- ``partition_indices``, ``partition_node_labels`` (AC-specific)
- ``apply_cut`` (returns a new ``TransitionSystem``, not a new ``System``)
- ``__eq__``, ``__hash__`` (on TS fields, not delegated)
- ``__str__``, ``__repr__`` (Direction-aware formatting)
- ``to_json`` (serializes TS-level fields)
- ``__post_init__`` (validates TS-specific invariants)
- The ``NotImplementedError`` stubs for IIT-formalism methods (``sia``,
  ``ces``, ``distinctions``, ``find_mip``, ``cause_mip``, ``effect_mip``,
  ``phi_cause_mip``, ``phi_effect_mip``, ``phi``, ``find_mice``, ``mic``,
  ``mie``, ``phi_max``, ``distinction``, ``all_distinctions``,
  ``evaluate_partition``)
- ``from_substrate`` classmethod

**TransitionSystem own-attrs list** (used by ``__getattr__`` to refuse
delegation): explicit ``frozenset`` listing all of the above, plus the
dataclass fields. This is the load-bearing detail of the delegation
pattern — anything not in this list falls through to
``_underlying_system``.

### Per-method count check

Before: ~550 lines (``pyphi/actual.py`` ``TransitionSystem`` class).
After: ~150 lines. Net delete: ~400 lines.

System changes: ~30 lines added (new field, validation,
``_resolved_external_indices`` property, ``__eq__``/``__hash__`` updates,
``to_json`` update).

## Critical Files

**Class definitions:**

- ``pyphi/system.py`` — add ``external_indices`` field, validation,
  resolution; update ``__eq__``/``__hash__``/``to_json``
- ``pyphi/actual.py`` — collapse ``TransitionSystem`` to façade

**Protocol:**

- ``pyphi/protocols.py`` — no changes needed; the ``SystemPublicInterface``
  surface is unchanged. (The new field on ``System`` is dataclass-level,
  not protocol-level.)

**Validation:**

- ``pyphi/validate.py`` — no changes needed; ``validate.node_states``
  and ``state_reachable`` already handle k-ary substrates from P12b.

**Tests:**

- ``test/test_system.py`` — add tests for ``external_indices`` override
- ``test/test_actual.py`` — add k=3 paper-figure test, delete the
  ``legacy_joint`` shape assertion at lines 252-256
- ``test/test_validate.py`` — no changes needed
- ``test/test_helpers.py`` — no changes needed
- ``test/test_invariants.py`` — no changes needed
- ``test/test_golden_regression.py`` — no changes needed (goldens
  remain byte-identical)
- ``test/test_json.py`` — verify ``System`` JSON round-trip with and
  without ``external_indices`` set; verify legacy JSON fixtures
  deserialize cleanly (with ``external_indices=None``)

**Documentation:**

- Changelog fragment: ``changelog.d/transition-system-collapse.refactor.md``

## Staging

The work decomposes into 5 commits:

### Commit 1 — System gains `external_indices` field (no AC changes)

- **Delete** the existing ``@cached_property external_indices`` on ``System``
- **Add** ``external_indices: tuple[int, ...] | None = None`` as a
  dataclass field
- **In ``__post_init__``:** if ``external_indices is None``, compute the
  default (``substrate.indices - node_indices``) and assign via
  ``object.__setattr__``; if set, validate (all indices in range,
  sorted, no duplicates) and assign as-is
- All internal reads of ``self.external_indices`` continue unchanged —
  after ``__post_init__`` the field holds the same value the
  cached_property used to return
- Update ``__eq__`` to include ``external_indices`` in the tuple
  comparison
- Update ``__hash__`` to include ``external_indices`` in the hash
- Update ``to_json`` to include ``"external_indices": list(self.external_indices)``
  in the serialized dict (after ``__post_init__`` the field is always
  concrete, safe to serialize)
- Tests in ``test/test_system.py`` for: explicit override accepted;
  override validated (rejects out-of-range, unsorted, duplicates);
  equality includes the field; hash includes the field; ``apply_cut``
  propagates; JSON round-trip preserves; legacy JSON without the field
  deserializes cleanly

**Verification:** existing ``test/test_system.py``, ``test/test_helpers.py``,
``test/test_invariants.py``, ``test/test_golden_regression.py`` all pass
with no changes. Goldens byte-identical.

### Commit 2 — TransitionSystem delegates effect_tpm through `_underlying_system`

- Wire ``TransitionSystem._underlying_system`` to construct ``System``
  with explicit ``external_indices`` (computed from
  ``cause_indices`` and ``noise_background``)
- Delete ``TransitionSystem.effect_tpm`` (delegates through
  ``_underlying_system.effect_tpm``)
- Delete ``TransitionSystem.proper_effect_tpm`` (same)
- Keep all other ``TransitionSystem`` methods unchanged for now
- Delete the ``legacy_joint`` assertion at
  ``test/test_actual.py:252-256``

**Verification:** full AC suite (63 tests) green; goldens byte-identical;
no calls to ``Substrate._legacy_binary_joint()`` from
``pyphi/actual.py`` (grep audit).

### Commit 3 — TransitionSystem `__getattr__` delegation

- Add ``TRANSITION_SYSTEM_OWN_ATTRS`` frozenset listing all attrs that
  ``TransitionSystem`` handles locally (dataclass fields + AC-specific
  properties/methods)
- Add ``__getattr__`` that delegates to ``_underlying_system`` for
  anything not in ``TRANSITION_SYSTEM_OWN_ATTRS``
- Delete the ~30 repertoire-algebra passthrough methods
  (``cause_repertoire``, ``effect_repertoire``, ``repertoire``,
  ``partitioned_repertoire``, ``unconstrained_*``, ``forward_*``,
  ``expand_*``, ``cause_info``, ``effect_info``,
  ``cause_effect_info``, ``intrinsic_information``,
  ``potential_purviews``, ``indices2nodes``, ``cache_info``,
  ``clear_caches``)
- Delete the ``cause_tpm``, ``proper_cause_tpm``, ``cm``,
  ``proper_cm``, ``connectivity_matrix``, ``nodes``,
  ``null_distinction``, ``null_concept``, ``size``, ``tpm_size``,
  ``_index2node`` properties (now delegated)
- Keep the ``NotImplementedError`` stubs (they need to override the
  underlying ``System``'s implementations)
- **Empirical verification step:** for each attribute in
  ``PUBLIC_SYSTEM_ATTRS`` (from ``pyphi/protocols.py``), assert
  ``hasattr(ts, attr)`` on a freshly-constructed ``TransitionSystem``,
  and assert the result is either the locally-handled value or matches
  ``ts._underlying_system.<attr>``. Spot-check ``effect_tpm``,
  ``cause_tpm``, ``proper_effect_tpm``, ``proper_cause_tpm``,
  ``cause_repertoire``, ``effect_repertoire``, ``nodes``.

**Verification:** full AC suite green; ``uv run pytest test/test_actual.py
test/test_system.py test/test_golden_regression.py``; goldens
byte-identical.

**Fallback:** if ``__getattr__`` interacts badly with ``cached_property``
or pickling, swap to explicit one-line delegation methods (verbose but
unsurprising). The diff is mechanical to rewrite.

### Commit 4 — k=3 paper-figure test

- Build a k=3 substrate matching the 2019 AC paper's Figure 11 (Section
  3.6, three-candidate voting; ``ABCDEFG = 1111122`` → ``W = 1``)
- Construct a ``Transition`` over the k=3 substrate
- Verify ``α_c`` matches the paper's reported value (``α_c = 1.893``
  for ``ABCD = 1111, BCDE = 1111`` ← ``W = 1``)
- Add a small ``effect_tpm.alphabet_sizes != (2,) * n`` assertion to
  lock in that we're actually exercising the k-ary code path

**Verification:** new test passes; rest of AC suite still green.

### Commit 5 — Changelog and docstring polish

- ``changelog.d/transition-system-collapse.refactor.md``: describe the
  collapse in user-facing terms (System gains optional
  ``external_indices``; ``TransitionSystem`` shrinks to a façade with
  identical public surface; AC inherits k-ary support).
- Sharpen the ``TransitionSystem`` class docstring to describe the
  final state (delegation pattern, AC-specific overrides).
- Sharpen the ``System.external_indices`` field docstring.

**Verification:** full ``uv run pytest`` (no path argument, includes
doctests); ``uv run pyright pyphi``; ``uv run ruff check pyphi test``.

## Acceptance Criteria

- All 5 commits land cleanly with pre-commit hooks green (ruff + ruff
  format + pyright + towncrier-check); no ``--no-verify`` bypass.
- ``Substrate._legacy_binary_joint()`` is no longer called from
  ``pyphi/actual.py`` (grep returns empty in that file).
- AC suite (~63 tests) all pass.
- New k=3 paper-figure test passes.
- ``test/test_golden_regression.py`` byte-identical (25/25).
- Full ``uv run pytest`` (no path argument) passes; doctests collected.
- ``pyright`` clean on ``pyphi/system.py``, ``pyphi/actual.py``,
  ``test/test_system.py``, ``test/test_actual.py``.
- ``ruff check`` and ``ruff format`` clean.
- ``TransitionSystem`` class shrinks from ~550 to ~150 lines.
- Goldens: ``pyphi/data/goldens/*.json`` byte-identical.
- ROADMAP entries: "Collapse ``TransitionSystem`` onto ``System``" and
  "Retire the SBN bridge in ``System.effect_tpm``" remain in
  "Informal notes" (the latter is the follow-up, not closed yet);
  the former gets a note pointing at the commit range as the closer.

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ``__getattr__`` delegation interacts badly with ``cached_property`` or pickling | Medium | Empirical verification step in Commit 3; explicit fallback plan to switch to one-line delegation methods if it fails |
| Hash/eq drift on ``System`` (forget to include ``external_indices``) | Low-Medium | Explicit hash-contract tests in ``test/test_system.py``; covered by the per-class ``__eq__`` test pattern landed in the recent session |
| Goldens drift because new field changes serialization | Low | ``external_indices=None`` is the default; legacy fixtures deserialize identically. JSON round-trip test verifies. |
| AC paper-Figure 11 reproduction is harder than expected (paper's ``α_c`` value doesn't reproduce within ``EQUALITY_TOLERANCE``) | Low-Medium | If the value diverges, document the discrepancy and either (a) lock in PyPhi's reproducible value and note it, or (b) treat as a separate bug investigation. Don't gate the collapse on paper-fidelity of one fixture. |
| ``_underlying_system`` cached at construction; partition mutation in tests | Very low | Frozen dataclass enforces immutability; existing ``apply_cut`` already returns new instances. |
| ``TransitionSystem`` consumers relied on the SBN-form ``effect_tpm`` shape | Low | The SBN bridge in ``System.effect_tpm`` preserves binary-substrate shapes; only k-ary substrates change shape (and there are no k-ary AC callers today). |

## Out of Scope

- Retiring the SBN bridge in ``System.effect_tpm`` — separate ROADMAP
  entry; cleanest done after this collapse lands.
- Changing AC default semantics from Section 3.3 extended background to
  strict Eq. 2 — logged as a separate ROADMAP item; defer to a future
  AC-semantics review.
- Macro path's ``_legacy_binary_joint()`` usage (``pyphi/macro.py:1111``)
  — binary-by-construction, separate scope.

## Effort Estimate

Half a day to one day. Mechanical search-replace and method
deletion dominate. The risky judgment work is concentrated in Commit 3
(``__getattr__`` delegation) and Commit 4 (paper-figure reproduction).

## Execution Strategy

Subagent-driven development. Sonnet 4.6 sufficient for the mechanical
commits (1, 2, 5); Opus 4.7 for Commit 3 (delegation correctness) and
Commit 4 (paper reproduction). Spec-compliance review after each
commit, then code-quality review.
