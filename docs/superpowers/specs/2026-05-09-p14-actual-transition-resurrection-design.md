# P14 — `actual.Transition` Resurrection + Formalism Config Audit (Design)

**Date:** 2026-05-09
**Status:** Brainstormed; awaiting writing-plans handoff
**Branch:** `feature/p14-actual-resurrection` cut from `2.0` head `b5dfb8a5`

## Goal

Bring `pyphi.actual.Transition` and the `pyphi.actual` free functions back online against the frozen `System` value type introduced in 2.0. Re-enable the 826 lines of currently-dark `test/test_actual.py`. While doing so, audit and unify the formalism config namespace so AC and IIT keys live in parallel nested namespaces under `config.formalism`, and retire the orphaned concept-style cuts machinery.

## Theoretical foundation and correctness oracle

`pyphi.actual` implements the formalism of:

> Albantakis L, Marshall W, Hoel E, Tononi G (2019). What Caused What? A Quantitative Account of Actual Causation Using Dynamical Causal Networks. *Entropy* 21(5): 459.

The 2019 paper is the canonical published reference for actual causation in the IIT framework. Its construction is self-contained — causal marginalization (Eq 2), pointwise mutual information for cause/effect info (Eqs 11–12), product partitioned repertoires (Eqs 8, 10), α = ρ − ρ_MIP (Eqs 15–16), α^max with minimality (Definitions 1–2), causal account as the set of all positive-α links (Definition 4), system-level integration A as MIP over partitions of {V_{t-1}, V_t} (Appendix A). The framework cites no IIT 4.0 / 2024 / 2026 reformulations because it predates them and does not depend on them.

The paper's worked-example figures (Figs 5, 6, 7, 8, 10, 11, 12, 13) carry specific computed α values that serve as a correctness oracle independent of the legacy implementation:

| Example | Worked α value |
|---|---|
| Fig 5/6: OR/AND first-order links {OR_{t-1}=1}↔{OR_t=1} and {AND_{t-1}=0}↔{AND_t=0} | α_e^max = α_c^max = 0.415 bits each |
| Fig 5/6: OR/AND second-order cause link {(OR,AND)_{t-1}=10}←{(OR,AND)_t=10} | α_c^max = 0.170 bits (the corresponding effect link is reducible) |
| Fig 7A: Disjunction (OR-gate), {A=1}↔{C=1} | α_e^max = α_c^max = 0.415 bits (with {AB=11} excluded by minimality) |
| Fig 7B: Conjunction (AND-gate), {AB=11}←{D=1} | α_c^max = 2.0 bits |
| Fig 7C: Bi-conditional (XNOR), second-order {AB=11}↔{E=1} | α_e^max = α_c^max = 1.0 bits (first-order links have ρ_e = 0) |
| Fig 8A: Majority gate, {ABC=111}←{M=1} | α_c^max = 1.678 bits |
| Fig 11: Three-candidate vote, {ABCD,BCDE=1111}←{W=1} | α_c^max = 1.893 bits |
| Fig 12: Probabilistic noisy COPY, {A=1}↔{N=1} | α_e^max = α_c^max = 0.848 bits |

The acceptance test suite in P14 will pin a representative subset of these (Figs 5, 6, 7B, 7C, 8A) as paper-fixture regression tests against the resurrected `Transition` API. This gives us a correctness oracle that does not depend on the legacy implementation's behavior.

The 2019 paper cites PyPhi (Mayner et al. 2018) as the software used to compute its figures, providing strong indirect evidence that the legacy implementation aligns with the paper.

### Theoretical scope (with informal-drift caveat)

P14 implements the AC formalism *as published in 2019*. The IIT group has acknowledged informally that subsequent IIT 4.0 / 2024 / 2026 developments raise questions about possible AC refinements (sharper measures, sequence-aware update grains, intrinsic-differentiation/specification analogues), but no formal AC update has been published. Resurrecting AC against the 2019 published formalism is the standard scientific-software contract: the published reference is what users cite and what the implementation tracks.

A future research project may produce an updated AC formalism. The configuration surface designed here (see § Configuration surface) is built so that — to the extent reasonable — switching from 2019-faithful AC toward a hypothetical updated AC amounts to changing config values and registering new measures/schemes, not rewriting the framework.

## Scope decision: macro deferred

The original P14 charter covered both `actual.py` and `macro.py`. After surveying the Marshall, Findlay, Albantakis & Tononi 2024 paper on intrinsic units (`papers/2024__marshall-et-al__intrinsic-units.pdf`), the macro work is deferred to a separate paper-faithful project:

- The 2024 paper formalizes a substantially more general macro framework than `pyphi/macro.py` implements: hierarchical meso constituents, sliding-window state mappings $g_J$ over sequences of $\tau$ micro updates, explicit background apportionment $W^J$, and intrinsic-unit search via $\varphi_s$ optimization.
- Resurrecting today's `MacroSystem` under the new `System` Protocol would bake legacy semantics into 2.0's public surface, then require API churn to migrate to the paper-faithful framework.
- Salvageable bits of `pyphi/macro.py` (`CoarseGrain`, `Blackbox`, `MacroNetwork` namedtuples; `all_partitions` / `all_groupings` / `all_coarse_grains` / `all_blackboxes`; `effective_info`; the `_partitions_list` precomputed data) stay alive without resurrection because the disabled `MacroSystem` class is the only consumer that's broken.
- `test/test_macro_system.py` (593 lines) remains skipped until the intrinsic-units project lands. Trade-off: those tests test legacy semantics that will not survive the paper-faithful rewrite, so refactoring against the new framework is cheaper than maintaining tests against soon-to-be-deleted code.

This deferral creates a follow-up project to schedule alongside P12/P13/P14b. The follow-up gets its own brainstorm + spec + plan cycle.

## Architectural decision: `System` as Protocol-conforming-concretes

The legacy `MacroSystem` and `actual.Transition`'s cause/effect systems both subclassed `Subsystem` and mutated parent state. Going forward, the cross-module type contract for systems is the existing `pyphi.protocols.SystemPublicInterface` Protocol (already declared, `runtime_checkable`). Concrete system kinds are siblings, not an inheritance hierarchy:

- `pyphi.System` — substrate-native granularity. Today's frozen dataclass; concrete; user-facing constructor (kept under the current name, since users construct systems more often than they annotate against an abstract type).
- `pyphi.actual.TransitionSystem` — directional view of a state transition. New in P14. Implements `SystemPublicInterface`. Frozen, parametric in `Direction`.
- *Future*: `IntrinsicUnitSystem` (Marshall 2024), `PerceptualSystem` (P14b), and any further sibling concretes plug in under the same Protocol.

The Protocol carries the IS-A semantics for dispatch sites; each concrete owns its own construction. No subclassing among concretes. Liskov violations like the legacy `MacroSystem.cut_indices` returning micro indices instead of node indices cannot recur, because there is no parent class whose contract a subclass could shadow.

`SystemPublicInterface` stays in `pyphi.protocols` and is not promoted to a public top-level name. The user-facing concrete remains `pyphi.System`.

## Configuration surface — audit, rename, and AC namespace

### Current state and motivation

Today's `config.formalism` is a flat namespace mixing IIT-formalism dispatch, IIT-specific settings, and a single AC key:

```
formalism                                  # IIT formalism dispatch ID — awkward self-referential name
repertoire_distance                        # measure (registry is called `measures` — naming inconsistency)
repertoire_distance_specification          # measure for specification role
repertoire_distance_differentiation        # measure for differentiation role
ces_distance                               # CES-level measure
actual_causation_measure                   # AC measure — only AC key, flat-namespaced
partition_type                             # mechanism-level partition scheme
system_partition_type                      # system-level partition scheme
system_partition_include_complete          # bool flag
system_cuts                                # IIT-3.0 SIA implementation switch (DEAD; see § Concept-style cuts deletion)
distinction_phi_normalization              # IIT 4.0
relation_computation                       # IIT 4.0
assume_cuts_cannot_create_new_concepts     # IIT 4.0 SIA optimization
state_tie_resolution / mip_tie_resolution / purview_tie_resolution
shortcircuit_sia
single_micro_nodes_with_selfloops_have_phi
```

P14 restructures this into nested namespaces under `config.formalism` so IIT-specific keys and AC keys live in parallel sub-namespaces:

```yaml
formalism:
  iit:
    version: IIT_4_0_2023
    repertoire_measure: GENERALIZED_INTRINSIC_DIFFERENCE
    repertoire_measure_specification: GENERALIZED_INTRINSIC_DIFFERENCE
    repertoire_measure_differentiation: GENERALIZED_INTRINSIC_DIFFERENCE
    ces_measure: SUM_SMALL_PHI
    mechanism_partition_scheme: ALL
    system_partition_scheme: SET_UNI/BI
    system_partition_include_complete: false
    distinction_phi_normalization: NUM_CONNECTIONS_CUT
    relation_computation: CONCRETE
    assume_partitions_cannot_create_new_concepts: false
    shortcircuit_sia: true
    single_micro_nodes_with_selfloops_have_phi: true
    state_tie_resolution: PHI
    mip_tie_resolution: [NORMALIZED_PHI, NEGATIVE_PHI]
    purview_tie_resolution: PHI
  actual_causation:
    measure: PMI
    mechanism_partition_scheme: ALL
    partitioned_repertoire_scheme: PRODUCT
    background_strategy: UNIFORM
    alpha_aggregation: SUBTRACTIVE
```

### Full rename map

| Old | New | Notes |
|---|---|---|
| `formalism.formalism` | `formalism.iit.version` | dispatch ID |
| `formalism.repertoire_distance` | `formalism.iit.repertoire_measure` | "measure" matches the registry name |
| `formalism.repertoire_distance_specification` | `formalism.iit.repertoire_measure_specification` | |
| `formalism.repertoire_distance_differentiation` | `formalism.iit.repertoire_measure_differentiation` | |
| `formalism.ces_distance` | `formalism.iit.ces_measure` | |
| `formalism.partition_type` | `formalism.iit.mechanism_partition_scheme` | |
| `formalism.system_partition_type` | `formalism.iit.system_partition_scheme` | |
| `formalism.system_partition_include_complete` | `formalism.iit.system_partition_include_complete` | |
| `formalism.system_cuts` | *(deleted)* | see § Concept-style cuts deletion |
| `formalism.distinction_phi_normalization` | `formalism.iit.distinction_phi_normalization` | |
| `formalism.relation_computation` | `formalism.iit.relation_computation` | |
| `formalism.assume_cuts_cannot_create_new_concepts` | `formalism.iit.assume_partitions_cannot_create_new_concepts` | |
| `formalism.shortcircuit_sia` | `formalism.iit.shortcircuit_sia` | |
| `formalism.single_micro_nodes_with_selfloops_have_phi` | `formalism.iit.single_micro_nodes_with_selfloops_have_phi` | |
| `formalism.state_tie_resolution` | `formalism.iit.state_tie_resolution` | |
| `formalism.mip_tie_resolution` | `formalism.iit.mip_tie_resolution` | |
| `formalism.purview_tie_resolution` | `formalism.iit.purview_tie_resolution` | |
| `formalism.actual_causation_measure` | `formalism.actual_causation.measure` | |
| *(new)* | `formalism.actual_causation.mechanism_partition_scheme` | reuses `pyphi.partition.partition_types` |
| *(new)* | `formalism.actual_causation.partitioned_repertoire_scheme` | new registry; default `PRODUCT` (paper Eq 8) |
| *(new)* | `formalism.actual_causation.background_strategy` | new registry; default `UNIFORM` (paper Eq 2) |
| *(new)* | `formalism.actual_causation.alpha_aggregation` | new registry; default `SUBTRACTIVE` (paper Eq 15) |

Naming principles applied:
- **"measure"** for any `(p, q) -> float` (or `(p, q, ...) -> DistanceResult`) registry — matches the existing `pyphi.metrics.distribution.measures` registry.
- **"scheme"** for partition-generator registries — reads more naturally than "type" (which is overloaded with Python's class meaning).
- **"partition"** uniformly where the legacy used "cut" for type/operation-level concepts. The verb/noun "cut" survives in code as the runtime-state concept (`apply_cut`, `is_cut`, `cut_indices`); config keys describe types/operations.
- Concepts that align across IIT and AC use the same key name in their respective sub-namespaces (`mechanism_partition_scheme` lives in both).

### AC-specific configuration design

The four new AC keys (plus the migrated `measure`) decompose the 2019 formalism into its parameterized choices, with paper-faithful defaults:

| Key | Default | Registry | Alternatives |
|---|---|---|---|
| `measure` | `PMI` | `pyphi.metrics.distribution.actual_causation_measures` (existing) | `KL`, registered measure functions |
| `mechanism_partition_scheme` | `ALL` | `pyphi.partition.partition_types` (existing) | `BI`, `TRI` |
| `partitioned_repertoire_scheme` | `PRODUCT` | new: `pyphi.actual.partitioned_repertoire_schemes` | `FORWARD_PROBABILITY` (IIT 4.0-style state-aware) |
| `background_strategy` | `UNIFORM` | new: `pyphi.actual.background_strategies` | `STATIONARY`, `OBSERVED` (paper page 26 alternatives) |
| `alpha_aggregation` | `SUBTRACTIVE` | new: `pyphi.actual.alpha_aggregations` | `RATIO` |

Each new registry is a small `pyphi.registry.Registry` instance with the same shape as `partition_types` and `measures`; populating it requires only the default + name validation. Validators in `ActualCausationConfig.__post_init__` ensure each value resolves in the appropriate registry at config-construction time, not deep inside a phi computation.

The set of structural choices in the 2019 formalism that are **not** parameterized (because alternatives would change what makes it AC):
- Per-node product for multi-variate cause/effect repertoires (Eqs 3–4) — the construction's whole point is to discount common-input correlations.
- Exclusion principle (one actual cause/effect per occurrence).
- Minimality clause (Definitions 1–2).
- α^max with maximization over candidates.
- Causal account = set of all positive-α links (Definition 4).
- System-level A as MIP over partitions of {V_{t-1}, V_t} (Appendix A).

These are hard-coded; alternatives constitute different frameworks.

### Decoupling AC from IIT formalism config (the Q2 fix)

`pyphi/core/repertoire_algebra.py:297` routes `partitioned_repertoire` to a state-aware forward-probability product when `config.formalism.repertoire_distance ∈ {GID, INTRINSIC_INFORMATION}` — different math from the paper's Eq 8 (which is unambiguously a product of causally-marginalized repertoires).

Legacy `pyphi/actual.py:372` works around this by passing `state=purview_state` when IIT 4.0 metrics are active, but the resulting `partitioned_repertoire` is still computed via the GID-path. **Under default IIT 4.0 config on 2.0, legacy AC computes a partitioned_repertoire that does not match the paper's prescription.** This is an existing bug that P14 fixes.

The fix: AC's `partitioned_repertoire` reads `config.formalism.actual_causation.partitioned_repertoire_scheme` and dispatches via the new registry, ignoring `config.formalism.iit.repertoire_measure` entirely. The default `PRODUCT` scheme is paper-faithful (Eq 8). The optional `FORWARD_PROBABILITY` scheme remains available for users who want to investigate IIT-4.0-style state-aware AC (a non-paper-faithful but registered alternative).

After this fix, AC's behavior is independent of which IIT formalism is configured; switching `formalism.iit.version` between IIT_3_0 and IIT_4_0_2023 has no effect on `actual.sia(transition)`.

### Migration impact

`config.formalism` becomes a frozen dataclass holding two nested frozen dataclasses (`IITConfig`, `ActualCausationConfig`). The top-level `formalism` layer no longer carries any flat keys directly — all knobs live in one of the two sub-namespaces. ~30 call sites across `pyphi/` rename mechanically (`config.formalism.repertoire_distance` → `config.formalism.iit.repertoire_measure` etc.). `pyphi_config.yml` migrates to the nested form. Test files (`test/test_config_layers.py`, `test/test_config.py`) update accordingly.

Since 2.0 has not been released, this is internal churn rather than a user-visible breaking change.

## Concept-style cuts deletion

The legacy `system_cuts` config key (values `3.0_STYLE` / `CONCEPT_STYLE`) controls a single dispatcher in `pyphi/formalism/iit3/__init__.py:393` that selects between the standard IIT 3.0 SIA (`_sia`) and a "concept-style" variant (`sia_concept_style`). Investigation confirms:

- The concept-style code (`ConceptStyleSystem`, `concept_cuts`, `directional_sia`, `SystemIrreducibilityAnalysisConceptStyle`, `sia_concept_style`) implements an asymmetric per-concept cut evaluation that does NOT appear in the published 2014 IIT 3.0 paper main text or figures (verified by reading pages 8–15 of `papers/2014__oizumi-et-al__iit-3.0.pdf`). The 2014 paper describes whole-constellation unidirectional cuts taking the minimum across directions — that's `_sia` (the `3.0_STYLE` path). The asymmetric per-concept-direction cutting in `ConceptStyleSystem` is a finer-grained variant of unclear provenance.
- The integration test (`test/test_big_phi.py::test_system_cut_styles`) is marked `@pytest.mark.outdated` and `@pytest.mark.slow` — not in the active suite.
- `test/test_concept_style_cuts.py:149` notes that `test_sia_concept_style`, `test_unpickle`, and `test_concept_style_phi` were removed because they had outdated expected values incompatible with IIT 4.0. Only the underlying machinery (KCut algebra, system accessors, the `concept_cuts` generator) remains tested.
- The maintainer (W. Mayner) confirmed during P14 brainstorming that no current workflow depends on this code.

P14 deletes:
- `pyphi/formalism/iit3/__init__.py`: `ConceptStyleSystem` class, `concept_cuts` function, `directional_sia` function, `SystemIrreducibilityAnalysisConceptStyle` class, `sia_concept_style` function. The `if config.formalism.system_cuts == "CONCEPT_STYLE"` branch in `sia()` collapses to `return _sia(system, **kwargs)` (and `sia` no longer needs `@functools.wraps(_sia)`).
- `pyphi/conf/formalism.py`: `_VALID_SYSTEM_CUTS` constant, `system_cuts` field, validation in `__post_init__`.
- `test/test_concept_style_cuts.py` — entire file.
- `test/test_big_phi.py::test_system_cut_styles` — single test removed.
- `test/test_config.py:35` — `SYSTEM_CUTS` validation entry removed.
- `pyphi_config.yml` — `SYSTEM_CUTS` line removed.
- Any docs referencing concept-style cuts (`docs/_build/html/configuration.html` is built-output, regenerates).

## The breakage being repaired

`pyphi/actual.py:98-169` — legacy `Transition.__init__`:

```python
self.effect_system = System(  # type: ignore
    substrate, before_state, self.node_indices, self.cut,
    _external_indices=external_indices,  # System has no such param
)
self.cause_system = System(  # type: ignore
    substrate, before_state, self.node_indices, self.cut,
    _external_indices=external_indices,
)
self.cause_system.state = after_state  # frozen forbids mutation
for node in self.cause_system.nodes:
    node.state = after_state[node.index]  # Node mutation
```

Three distinct breaks:

1. `_external_indices` keyword override on `System(...)` — frozen `System` declares only `(substrate, state, node_indices, cut)`; `external_indices` is a derived `cached_property` with no override hook.
2. `self.cause_system.state = after_state` — `@dataclass(frozen=True)` raises `FrozenInstanceError`.
3. Per-node state mutation — `Node` is also frozen post-2.0 P3.

The intent encoded in those mutations was: cause-side analysis needs TPMs marginalized with respect to `before_state` background but mechanism states evaluated in `after_state`. This is a two-state requirement that today's `System` (single `state` field) cannot express.

## The fix: `TransitionSystem` parametric in direction

`pyphi.actual.TransitionSystem` is a frozen dataclass implementing `SystemPublicInterface`. One class, parametric in `Direction.CAUSE` / `Direction.EFFECT`. `Transition` holds two cached instances (one per direction).

### Type signature

```python
@dataclass(frozen=True, eq=False)
class TransitionSystem:
    """A directional view of a state transition. Satisfies SystemPublicInterface.

    The TPMs are conditioned on `before_state` for every substrate index
    outside `cause_indices` (the asymmetric background-conditioning rule from
    the 2019 Albantakis et al. formalism). The mechanism-evaluation `state`
    is `after_state` for the CAUSE direction and `before_state` for the
    EFFECT direction.
    """

    substrate: Substrate
    before_state: State
    after_state: State
    cause_indices: NodeIndices
    effect_indices: NodeIndices
    direction: Direction
    cut: SystemPartition = field(default=None)  # NullCut after __post_init__
    noise_background: bool = False
```

### Derived attributes (cached)

```python
    @cached_property
    def node_indices(self) -> NodeIndices:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @cached_property
    def state(self) -> State:
        return self.after_state if self.direction == Direction.CAUSE else self.before_state

    @cached_property
    def external_indices(self) -> NodeIndices:
        if self.noise_background:
            return ()
        return tuple(sorted(set(self.substrate.node_indices) - set(self.cause_indices)))

    @cached_property
    def cause_tpm(self):
        # Marginalize substrate.tpm against before_state on external_indices,
        # using the configured background_strategy from config.formalism.actual_causation.
        ...

    @cached_property
    def effect_tpm(self):
        # Same pattern.
        ...

    # node_labels, nodes, cm, proper_state, proper_cause_tpm, proper_effect_tpm,
    # is_cut, size, tpm_size, etc. — same machinery as System; bodies copied or
    # reused via shared free functions.
```

### Protocol surface (delegation)

The repertoire-algebra surface is identical to `System`'s — every method body is a thin delegation to `pyphi.core.repertoire_algebra` (or to `pyphi.actual` for the partitioned_repertoire path; see § Decoupling AC from IIT formalism config) with `self` as the system argument.

```python
    def repertoire(self, direction, mechanism, purview, **kw):
        return pyphi.core.repertoire_algebra.repertoire(self, direction, mechanism, purview, **kw)
    def partitioned_repertoire(self, direction, partition, **kw):
        # AC's own partitioned_repertoire — paper-faithful, decoupled from IIT formalism config
        return pyphi.actual.partitioned_repertoire(self, direction, partition, **kw)
    def potential_purviews(self, direction, mechanism, **kw): ...
    def cause_repertoire(self, mechanism, purview, **kw): ...
    def effect_repertoire(self, mechanism, purview, **kw): ...
    # ... full surface per PUBLIC_SYSTEM_ATTRS in pyphi/protocols.py
```

`TransitionSystem` does **not** expose the IIT formalism dispatchers (`sia`, `phi_structure`, `ces`, `find_mip`, `find_mice`, `distinction`, `all_distinctions`, `evaluate_partition`). Calling those on a TransitionSystem is a category error — actual causation is not an IIT-formalism analysis. They raise `NotImplementedError` with a message pointing at the appropriate `pyphi.actual` free functions.

### Validation in `__post_init__`

```python
    def __post_init__(self) -> None:
        validate.state_length(self.before_state, self.substrate.size)
        validate.state_length(self.after_state, self.substrate.size)
        validate.node_states(self.before_state)
        validate.node_states(self.after_state)
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.cut is None:
            object.__setattr__(self, "cut", NullCut(self.node_indices, self.substrate.node_labels))
        if self.direction == Direction.CAUSE and config.infrastructure.validate_system_states:
            validate.state_reachable(self)
```

The legacy `with config.override(validate_system_states=False): ... validate.state_reachable(cause_system)` block collapses to: validation runs implicitly from `__post_init__`, gated on direction. EFFECT-side construction never triggers a reachability check it would fail because its `state` is `before_state` (the substrate's actual current state, which reaches itself by construction).

## `Transition` as frozen wrapper

```python
@dataclass(frozen=True, eq=False)
class Transition:
    """A state transition over a substrate, holding two TransitionSystem views.

    ``eq=False`` so the explicit ``__eq__`` / ``__hash__`` below preserve the
    legacy semantics: two transitions with the same indices/states/substrate/cut
    are equal regardless of ``noise_background`` setting.
    """

    substrate: Substrate
    before_state: State
    after_state: State
    cause_indices: NodeIndices
    effect_indices: NodeIndices
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]
    noise_background: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.before_state == other.before_state
            and self.after_state == other.after_state
            and self.cause_indices == other.cause_indices
            and self.effect_indices == other.effect_indices
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash((
            self.substrate, self.before_state, self.after_state,
            self.cause_indices, self.effect_indices, self.cut,
        ))

    def __post_init__(self) -> None:
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, self.substrate.node_labels)
            )

    @cached_property
    def node_indices(self) -> NodeIndices:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @cached_property
    def cause_system(self) -> TransitionSystem:
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.CAUSE,
            cut=self.cut,
            noise_background=self.noise_background,
        )

    @cached_property
    def effect_system(self) -> TransitionSystem:
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.EFFECT,
            cut=self.cut,
            noise_background=self.noise_background,
        )

    @cached_property
    def system(self) -> Mapping[Direction, TransitionSystem]:
        return MappingProxyType({
            Direction.CAUSE: self.cause_system,
            Direction.EFFECT: self.effect_system,
        })

    def apply_cut(self, cut: SystemPartition) -> "Transition":
        return replace(self, cut=cut)

    # All existing instance methods preserved verbatim except partitioned_repertoire,
    # which is rewritten to drop the IIT-formalism-config branch (see § Decoupling).
    # Method bodies read from self.cause_system / self.effect_system, which are now
    # frozen TransitionSystem instances rather than mutated System instances:
    #
    #   __repr__, __str__, __len__, __bool__, node_labels, to_json,
    #   cause_repertoire, effect_repertoire, unconstrained_cause_repertoire,
    #   unconstrained_effect_repertoire, repertoire, state_probability,
    #   probability, unconstrained_probability, purview_state, mechanism_state,
    #   mechanism_indices, purview_indices, _ratio, cause_ratio, effect_ratio,
    #   partitioned_probability, find_mip, potential_purviews, find_causal_link,
    #   find_actual_cause, find_actual_effect, find_mice
```

`Transition.partitioned_repertoire`, was:

```python
def partitioned_repertoire(self, direction, partition):
    system = self.system[direction]
    if config.formalism.repertoire_distance in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
        purview_state = tuple(
            self.purview_state(direction)[node] for node in partition.purview
        )
        return system.partitioned_repertoire(direction, partition, state=purview_state)
    return system.partitioned_repertoire(direction, partition)
```

Becomes:

```python
def partitioned_repertoire(self, direction, partition):
    return pyphi.actual.partitioned_repertoire(
        self.system[direction], direction, partition,
    )  # dispatches via formalism.actual_causation.partitioned_repertoire_scheme registry
```

## Module-level free functions — bodies unchanged

`directed_account`, `account`, `probability_distance`, `account_distance`, `_evaluate_cut`, `_get_cuts`, `sia`, `transitions`, `nexus`, `causal_nexus`, `nice_true_ces`, `_actual_causes`, `_actual_effects`, `events`, `true_ces`, `true_events`, `extrinsic_events` — bodies unchanged. They consume `Transition` instances; the swap of internal types is invisible.

`pyphi.actual.partitioned_repertoire` is **new** — a free function that dispatches via the `partitioned_repertoire_scheme` registry. Default scheme `PRODUCT` implements the paper's Eq 8 (product of causally-marginalized repertoires).

## Files modified

| File | Change |
|---|---|
| `pyphi/conf/formalism.py` | Restructure: `FormalismConfig` becomes thin holder of nested `IITConfig` + `ActualCausationConfig` frozen dataclasses. Drop `system_cuts` field and `_VALID_SYSTEM_CUTS`. |
| `pyphi/actual.py` | Rewrite `Transition`; add `TransitionSystem`; add `partitioned_repertoire` free function + `partitioned_repertoire_schemes` registry; add `background_strategies` registry; add `alpha_aggregations` registry. |
| `pyphi/formalism/iit3/__init__.py` | Delete `ConceptStyleSystem`, `concept_cuts`, `directional_sia`, `SystemIrreducibilityAnalysisConceptStyle`, `sia_concept_style`; collapse `sia()` to call `_sia` directly. |
| ~30 call sites across `pyphi/` | Mechanical rename: `config.formalism.X` → `config.formalism.iit.X` etc. per the rename map. |
| `pyphi/__init__.py` | Re-export `TransitionSystem` from `pyphi.actual` for top-level access. |
| `pyphi_config.yml` | Migrate to nested form. |
| `test/test_actual.py` | Remove `pytestmark = pytest.mark.skip(...)` at line 16. Add `test_transition_system_*` tests + paper-fixture tests against Figs 5/6/7B/7C/8A. |
| `test/conftest.py` | Remove the `pytest.skip(...)` at line 381–384 inside the `transition` fixture. |
| `test/test_concept_style_cuts.py` | Delete entire file. |
| `test/test_big_phi.py` | Delete `test_system_cut_styles`. |
| `test/test_config.py` | Remove `SYSTEM_CUTS` validation entry; add tests for nested-namespace validation. |
| `test/test_config_layers.py` | Update for nested `formalism.iit` / `formalism.actual_causation` structure. |
| `ROADMAP.md` | Mark P14 complete; update entry to reflect actual-only scope (macro deferred); insert new project entry for the deferred macro/intrinsic-units work. |
| `changelog.d/p14-actual-resurrection.fix.md` | New changelog fragment describing the resurrection + config audit. |

Files **not** modified:

- `pyphi/system.py` — TransitionSystem doesn't subclass or extend System.
- `pyphi/protocols.py` — `SystemPublicInterface` already covers the surface.
- `pyphi/core/repertoire_algebra.py` — already duck-typed; AC's partitioned_repertoire path moves into `pyphi.actual` rather than threading through here.
- `pyphi/formalism/iit4/*` — no AC interaction; just rename-driven call site updates.
- `pyphi/macro.py` — stays disabled until the intrinsic-units project lands.
- `test/test_macro_system.py` — stays skipped.

## Acceptance gates (every commit)

- Golden 17/17 numerical match — actual-causation tests don't touch the IIT-formalism path; golden fixtures should be unchanged. (Renames affect serialized config blobs in fixtures; golden regenerates if the blob changes but the computed phi values must match.)
- Hypothesis fast lane 21 green.
- Fast unit lane (subsystem_surface, formalism_pickle, parallel, scheduler, sampling, install_snapshot, invariants) — green.
- `test/test_actual.py` (currently skipped) runs green after the Transition rewrite commit.
- Paper-fixture tests pin α values from 2019 Albantakis et al. Figs 5, 6, 7B, 7C, 8A.
- `test/test_macro_system.py` stays skipped (deferred macro work).
- Pyright clean on `pyphi/actual.py`, `pyphi/conf/formalism.py`.
- Ruff clean.
- Pre-commit hooks pass on every commit; no `--no-verify`.

## Phasing

Five commits on `feature/p14-actual-resurrection`:

### Commit 1 — Delete concept-style cuts machinery

- Delete: `ConceptStyleSystem`, `concept_cuts`, `directional_sia`, `SystemIrreducibilityAnalysisConceptStyle`, `sia_concept_style` from `pyphi/formalism/iit3/__init__.py`.
- Collapse `iit3.sia()` to a single-branch call to `_sia`.
- Delete `_VALID_SYSTEM_CUTS` and `system_cuts` from `pyphi/conf/formalism.py`.
- Delete `test/test_concept_style_cuts.py` entirely.
- Delete `test_system_cut_styles` from `test/test_big_phi.py`.
- Remove `SYSTEM_CUTS` validation entry from `test/test_config.py`.
- Remove `system_cuts` from `pyphi_config.yml`.

**Acceptance**: golden 17/17 unchanged; hypothesis fast lane unchanged; fast unit lane green; pyright + ruff clean.

### Commit 2 — Config audit and rename

- Restructure `pyphi/conf/formalism.py`: `FormalismConfig` becomes a thin holder of `IITConfig` + `ActualCausationConfig` frozen dataclasses, each with their own `__post_init__` validation.
- Rename and migrate per the full rename map. ~30 call sites across `pyphi/` updated mechanically.
- Migrate `pyphi_config.yml` to nested form.
- Update `test/test_config_layers.py` for nested structure.
- Update fast-lane test_config.py for renamed key validation.
- New AC keys (`mechanism_partition_scheme`, `partitioned_repertoire_scheme`, `background_strategy`, `alpha_aggregation`) added with defaults but not yet wired in (registries created but consumed only after Commit 4).

**Acceptance**: golden 17/17 numerical match (config blobs in serialized fixtures regenerate but computed values unchanged); hypothesis fast lane unchanged; fast unit lane green; pyright + ruff clean.

### Commit 3 — Add `TransitionSystem` and AC registries (no callers)

- Implement `TransitionSystem` class in `pyphi/actual.py`.
- Register `partitioned_repertoire_schemes` (`PRODUCT` default), `background_strategies` (`UNIFORM` default), `alpha_aggregations` (`SUBTRACTIVE` default) in `pyphi/actual.py`.
- Implement `pyphi.actual.partitioned_repertoire` free function.
- Add unit tests for `TransitionSystem`: `test_transition_system_is_frozen`, `test_transition_system_satisfies_protocol`, `test_transition_system_cause_uses_after_state`, `test_transition_system_effect_uses_before_state`, `test_transition_system_external_indices_excludes_cause_indices`, `test_transition_system_apply_cut_returns_new_instance`. Kept under the existing module-level skip in `test_actual.py`; lift skip locally for these new tests via a separate skip-free file or marker.
- Existing `Transition` class untouched.

**Acceptance**: new TransitionSystem tests pass; golden + hypothesis + fast unit lanes unchanged; pyright + ruff clean.

### Commit 4 — Rewrite `Transition`; remove skips; paper-fixture tests

- Replace `Transition` body with the frozen dataclass design above.
- Rewrite `Transition.partitioned_repertoire` to delegate to `pyphi.actual.partitioned_repertoire` (the Q2 fix).
- Remove `pytestmark = pytest.mark.skip(...)` in `test/test_actual.py`.
- Remove `pytest.skip(...)` in `test/conftest.py:381–384` inside the `transition` fixture.
- Add paper-fixture tests against the worked-example α values from 2019 Albantakis et al. Figs 5, 6, 7B, 7C, 8A.

**Acceptance**: full `test_actual.py` (826 lines) runs green; paper-fixture tests pass; golden + hypothesis + fast unit lanes unchanged; pre-commit clean.

### Commit 5 — ROADMAP + changelog

- Mark P14 complete in `ROADMAP.md`. Update P14 entry to reflect actual-only scope; drop the "third PhiFormalism implementation" claim for AC.
- Insert a new project entry for the deferred macro / Marshall-2024-intrinsic-units work, citing the paper.
- New changelog fragment `changelog.d/p14-actual-resurrection.fix.md` covering: actual.Transition resurrection, formalism config audit (nested namespaces + renames), AC's new config knobs, concept-style cuts deletion, paper-fixture acceptance tests.

**Acceptance**: docs reflect new state; suite still green.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Paper-fixture α values don't match the legacy implementation's output (signaling either a legacy bug or an interpretation mismatch) | Honest reckoning, not a blocker. The paper IS the published reference; if the legacy diverges, the paper wins. Investigate any divergence; document in the changelog. |
| `TransitionSystem` TPM marginalization differs subtly from legacy | The 826 lines of `test_actual.py` encode known-good Transition behavior; if marginalization differs, fixtures will fail. Combined with paper-fixture tests, this is a strong oracle. |
| `validate.state_reachable` raised in `__post_init__` breaks legacy callers that relied on suppression | Run the suite under default config; if any caller failed, surface the failure case-by-case. |
| Config rename breaks user-saved `pyphi_config.yml` files in the wild | 2.0 hasn't been released; no user configs are in the wild yet. Internal churn only. |
| Paper-fixture tests are slow (each requires running a full SIA on a 3- or 4-node example) | Mark `@pytest.mark.slow` if needed; included in the slow lane rather than the fast lane. |

## What does NOT happen in P14

- `pyphi/macro.py` resurrection. Deferred to a paper-faithful Marshall 2024 intrinsic-units project (separate brainstorm).
- Promotion of `SystemPublicInterface` to a public top-level name. Stays in `pyphi.protocols`.
- Changes to `pyphi/system.py`, `pyphi/protocols.py`, `pyphi/core/repertoire_algebra.py`. Out of scope.
- Refactoring of `actual` module-level free functions beyond the new `partitioned_repertoire` helper. Bodies unchanged.
- New `actual.Transition` features beyond the legacy surface (e.g., the optional `alpha_aggregation` knob exposes a registry but ships only the paper-faithful `SUBTRACTIVE` default; the `RATIO` alternative can be added in a later commit if a user request surfaces).
- A formal AC theoretical update (intrinsic-differentiation-style, sequence-aware, etc.). The configuration surface is *prepared* for such an update by being decomposed into parameterized choices, but the actual research work is its own project.
