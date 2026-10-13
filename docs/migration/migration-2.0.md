# Migrating to PyPhi 2.0

PyPhi 2.0 is a breaking release. It computes IIT 4.0 (Albantakis et al., 2023;
Mayner, Marshall & Tononi, 2026), in which system integrated information
includes the system's intrinsic information. IIT 3.0, and IIT 4.0 as published
in 2023, remain available for reproducing earlier results. There are no
deprecation shims: code written against pre-2.0
PyPhi must be updated to run.

The PyPhi MCP server carries a condensed, agent-facing copy of this guide as its
`migration` reference topic, so an assistant can port code without this site.

This guide documents the changes a pre-2.0 user hits, organized by topic. Each
topic is tagged with who it affects:

- **[1.x]** — released PyPhi 1.x (the PyPhi-paper / IIT 3.0 era).
- **[4.0-branch]** — the IIT 4.0 feature branch.
- **[both]** — everyone.
- **[rc]** — the 2.0 release candidates (rc1 to rc3). If you are upgrading from
  one of those, read {ref}`upgrading-from-rc`
  first: some calls that ran without error under a release candidate now
  compute different numbers.

## Renames at a glance

| Old | New | Affects |
| --- | --- | --- |
| `pyphi.Network` | `pyphi.Substrate` | [1.x] |
| `pyphi.Subsystem(network, state, nodes)` | `pyphi.System(substrate, state, subset=nodes)` | [1.x] |
| `network.tpm` (an array) | `substrate.tpm` (a `FactoredTPM`; `np.asarray(substrate.tpm)` gives the state-by-node array when every unit is binary) | [1.x] |
| `pyphi.compute.*` | `pyphi.analyze(...)` | [1.x] |
| `subsystem.cause_tpm`, `proper_cause_tpm` | `system.cause_marginal`, `proper_cause_marginal` | [4.0-branch] |
| `subsystem.effect_tpm`, `proper_effect_tpm` | `system.effect_marginal`, `proper_effect_marginal` | [4.0-branch] |
| `subsystem.cut` | `system.partition` | [1.x] |
| `pyphi.jsonify` | `pyphi.serialize` / `pyphi.save` / `pyphi.load` | [both] |
| `pyphi.config.IIT_VERSION` | `pyphi.config.formalism.iit.version` | [both] |
| `pyphi.models.Concept` | `pyphi.models.Distinction` | [1.x] |
| `pyphi.models.CauseEffectStructure`, the concepts | `pyphi.models.Distinctions` | [1.x] |
| `PhiStructure` | `pyphi.models.CauseEffectStructure` | [4.0-branch] |
| `subsystem.null_concept` | `system.null_distinction` | [1.x] |
| `subsystem.apply_cut(cut)`, `transition.apply_cut(cut)` | `system.apply_partition(partition)`, `transition.apply_partition(partition)` | [1.x] |
| `pyphi.actual.events(network, previous_state, current_state, next_state, nodes)`; `indices=` on `true_events` and `extrinsic_events` | `pyphi.actual.events(substrate, previous_state, current_state, next_state, subset=nodes)`; `subset=` on all three | [1.x] |
| `pyphi.convert.b2l`, `l2b`, `l2s`, `b2s`, `s2l`, `s2b` | `be2le`, `le2be`, `le_index2state`, `be_index2state`, `state2le_index`, `state2be_index` | [1.x] |
| `pyphi.convert.b2l_sbs`, `l2b_sbs` | `be2le_state_by_state`, `le2be_state_by_state` | [1.x] |
| `pyphi.convert.to_md`, `to_2d` | `to_multidimensional`, `to_2dimensional` | [1.x] |
| `pyphi.convert.sbn2sbs`, `sbs2sbn` | `state_by_node2state_by_state`, `state_by_state2state_by_node` | [1.x] |
| `pyphi.new_big_phi` | `pyphi.formalism.iit4` (`sia`, `ces`) | [4.0-branch] |
| `pyphi.network_generator` | `pyphi.substrate_generator` | [4.0-branch] |
| `pyphi.metrics` | `pyphi.measures` | [4.0-branch] |
| `ria.purview_state`, `mice.purview_state` (the current state of the purview units) | `ria.current_state`, `mice.current_state`; the state a mechanism specifies is `mice.specified_state.state` | [4.0-branch] |
| `state_specification.intrinsic_information` | `state_specification.intrinsic_specification`. `sia.intrinsic_information` is a different quantity, ii(s), and keeps its name | [4.0-branch] |
| `subsystem.intrinsic_information(direction, mechanism, purview, ...)` | `system.intrinsic_specification(direction, mechanism, purview, ...)` | [4.0-branch] |
| `pyphi.models.SystemIrreducibilityAnalysis` | `pyphi.formalism.iit4.SystemIrreducibilityAnalysis` (IIT 4.0) or `pyphi.models.IIT3SystemIrreducibilityAnalysis` (IIT 3.0) | [1.x] |
| `pyphi.models.Bipartition`, `Tripartition`, `KPartition` | `JointBipartition`, `JointTripartition`, `JointPartition` | [1.x] |
| `pyphi.models.KCut`, `pyphi.models.ActualCut` | `DirectedJointPartition(direction, partition)`, `DirectedBipartition` (applied with `Transition(..., partition=)` or `transition.apply_partition`) | [1.x] |

Most removed names, used as an attribute, raise an `AttributeError` that gives
the replacement: `pyphi.Network` and the other 1.x names in this table,
`pyphi.distance` and `pyphi.utils.eq`, the IIT 4.0 branch's `cause_tpm`,
`effect_tpm` (and their `proper_` forms) and `PhiStructure`, and every renamed
example function, including the release candidates' unprefixed 2014 names such
as `fig4_substrate()`. Importing one, as in `from pyphi import Network`, raises
a plain `ImportError` instead, because Python's import statement drops that
message; look the name up in the table above. A few names give no
replacement and raise a plain `AttributeError` (or a plain `ImportError` when
imported): the `pyphi.tpm` module, the IIT 4.0 branch's `pyphi.new_big_phi`,
`pyphi.network_generator` and `pyphi.metrics`, `purview_state` and
`intrinsic_information` on a result or a system, and most names that existed
only in a release candidate (see the last section).

The name `CauseEffectStructure` now means a different object. In 1.x it was
the list of concepts, which 2.0 calls `Distinctions`; in 2.0 it is the
distinctions together with their relations and the system irreducibility
analysis. An unported import of `CauseEffectStructure` therefore still
succeeds, but gives the newer object.

`cause_marginal` and `effect_marginal` (with the `proper_cause_marginal` /
`proper_effect_marginal` variants) replace the IIT 4.0 branch's `cause_tpm` and
`effect_tpm`; released 1.x had neither. They hold the cause and effect TPMs of
IIT 4.0, 𝒯_c and 𝒯_e (Albantakis et al., 2023, Eqs. 3–4), with the background
causally marginalized conditional on the current state: held at its current
state for effects, and with its past states weighted by their probability
given the current state for causes. `effect_marginal` is a factored transition
matrix and `cause_marginal` a mapping of per-unit cause factors. See
[Substrate and system](../theory/substrate-and-system.md) for what they are.

### Example networks

The example functions follow the same vocabulary: every `*_network` is now
`*_substrate` and every `*_subsystem` is `*_system`. Transitions and bare
matrices (`prevention_transition`, `cond_depend_tpm`, `cond_independ_tpm`,
`differentiation_*_tpm`) keep their names. The networks from the 2014 IIT 3.0
paper (Oizumi et al., 2014) carry the prefix `iit3_2014_`, as the later papers'
networks carry theirs. In the 2.0 release candidates they had no prefix
(`fig4_substrate()`). Every old name in this table raises an `AttributeError`
that gives the new one.

| Old | New |
| --- | --- |
| `examples.basic_network()` | `examples.basic_substrate()` |
| `examples.basic_subsystem()` | `examples.basic_system()` |
| `examples.basic_noisy_selfloop_network()` / `_subsystem()` | `examples.basic_noisy_selfloop_substrate()` / `_system()` |
| `examples.grid3_network()` / `_subsystem()` | `examples.grid3_substrate()` / `_system()` |
| `examples.residue_network()` / `_subsystem()` | `examples.residue_substrate()` / `_system()` |
| `examples.xor_network()` / `_subsystem()` | `examples.xor_substrate()` / `_system()` |
| `examples.rule110_network()` / `_subsystem()` | `examples.rule110_substrate()` / `_system()` |
| `examples.rule154_network()` / `_subsystem()` | `examples.rule154_substrate()` / `_system()` |
| `examples.macro_network()` / `_subsystem()` | `examples.macro_substrate()` / `_system()` |
| `examples.blackbox_network()` | `examples.blackbox_substrate()` |
| `examples.propagation_delay_network()` | `examples.propagation_delay_substrate()` |
| `examples.fig1a()`, `fig3a()`, `fig3b()`, `fig16()` [1.x]; `fig1a_network()` and the like [4.0-branch] | `examples.iit3_2014_fig1a_substrate()`, `iit3_2014_fig3a_substrate()`, `iit3_2014_fig3b_substrate()`, `iit3_2014_fig16_substrate()` |
| `examples.fig4()`, `fig5a()`, `fig5b()` [1.x]; `fig4_network()` / `_subsystem()` [4.0-branch] | `examples.iit3_2014_fig4_substrate()` / `iit3_2014_fig4_system()`, `iit3_2014_fig5a_*`, `iit3_2014_fig5b_*` |
| `examples.fig6()`, `fig8()`, `fig9()`, `fig10()` (the Figure 4 network), `fig14()` (the Figure 1A network) [1.x] | `examples.iit3_2014_fig6_substrate()`, `iit3_2014_fig8_substrate()`, `iit3_2014_fig9_substrate()`, `iit3_2014_fig10_substrate()`, `iit3_2014_fig14_substrate()` |
| `examples.actual_causation()` [1.x]; `actual_causation_network()` [4.0-branch] | `examples.actual_causation_substrate()` |
| `examples.prevention()` [1.x] | `examples.prevention_transition()` |
| `examples.disjunction_conjunction_network()` | `examples.disjunction_conjunction_substrate()` |
| `examples.frog_example()` | `examples.frog_substrate()` and `examples.frog_transition()` |
| `examples.differentiation_micro_1_subsystem()` | `examples.differentiation_micro_1_system()` |

`iit3_2014_fig5b_substrate()` follows Figure 5B of the 2014 paper, in which
unit B is AND(A, C); the 1.x `fig5b()` network computed OR(A, C), so its
results differ.

The IIT 4.0 paper's networks are new in 2.0 (`iit4_2023_fig1a_substrate()`
and the other `iit4_2023_*`, `marshall_2023_*`, and `mayner_2026_*`
functions); the {doc}`examples gallery </reference/examples>` lists every
registered example.

### Configuration options

Every 1.x option and where it went. Names are lowercase and belong to a layer.
Read one with `pyphi.config.<layer>.<option>`. Set one with a top-level write
such as `pyphi.config.precision = 6`, which is routed to its layer, with a
dotted path such as `pyphi.config["iit.mechanism_partition_scheme"] = ...`,
or for a block of code with `pyphi.config.override(...)`. The layer objects
are read-only, so `pyphi.config.numerics.precision = 6` raises an error that
shows these forms. Every write is checked against the other options, and a
combination that matches no version of IIT raises a `ConfigurationError`.
An option whose name only changed case still works under its uppercase 1.x
spelling (`pyphi.config.PRECISION`), with a `FutureWarning` giving the new
name; any other 1.x name raises an error that gives its replacement. Measure
values are the names in {mod}`pyphi.measures`.

| 1.x option | 2.0 option |
| --- | --- |
| `IIT_VERSION` | `formalism.iit.version` (`"IIT_3_0"`, `"IIT_4_0_2023"`, `"IIT_4_0_2026"`); select it with `analyze(..., formalism=)` or a preset, which also set the measures. Setting it alone raises a `ConfigurationError` |
| `MEASURE` [1.x], `REPERTOIRE_DISTANCE` [4.0-branch] | `formalism.iit.mechanism_phi_measure` (mechanism level) and `formalism.iit.system_phi_measure` (system level) |
| `REPERTOIRE_DISTANCE_SPECIFICATION` | `formalism.iit.specification_measure` |
| `REPERTOIRE_DISTANCE_DIFFERENTIATION` | removed: intrinsic differentiation is the surprisal of the specified state and has no measure option |
| `CES_DISTANCE` | `formalism.iit.ces_measure` |
| `ACTUAL_CAUSATION_MEASURE` | `formalism.actual_causation.alpha_measure` |
| `PARTITION_TYPE` | `formalism.iit.mechanism_partition_scheme` for IIT, and `formalism.actual_causation.mechanism_partition_scheme` for actual causation, which reads its own option. A bare `pyphi.config.mechanism_partition_scheme = ...` is ambiguous and raises; qualify it. The values were renamed, see below |
| `SYSTEM_PARTITION_TYPE` | `formalism.iit.system_partition_scheme` |
| `SYSTEM_PARTITION_INCLUDE_COMPLETE` | `formalism.iit.system_partition_include_total` |
| `SYSTEM_CUTS` | removed; the `iit3` preset sets `system_partition_scheme="DIRECTED_BIPARTITION"` (the 3.0-style cut), and concept-style cuts are gone |
| `DISTINCTION_PHI_NORMALIZATION` | `formalism.iit.distinction_phi_normalization` |
| `RELATION_COMPUTATION` | `formalism.iit.relation_computation` |
| `STATE_TIE_RESOLUTION`, `MIP_TIE_RESOLUTION`, `PURVIEW_TIE_RESOLUTION` [4.0-branch] | `formalism.iit.state_tie_resolution`, `mip_tie_resolution`, `purview_tie_resolution` (plus the new `sia_tie_resolution`); see {doc}`/howto/tie-breaking` |
| `PICK_SMALLEST_PURVIEW` [1.x] | `formalism.iit.purview_tie_resolution`; the values are given below |
| `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` | `formalism.iit.assume_partitions_cannot_create_new_concepts` |
| `SHORTCIRCUIT_SIA` | `formalism.iit.shortcircuit_sia` |
| `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | `formalism.iit.single_micro_nodes_with_selfloops_have_phi` |
| `PRECISION` | `numerics.precision` |
| `PARALLEL`, `PARALLEL_WORKERS`, `PARALLEL_BACKEND` | `infrastructure.parallel`, `parallel_workers`, `parallel_backend` |
| `PARALLEL_COMPLEX_EVALUATION`, `PARALLEL_PURVIEW_EVALUATION`, `PARALLEL_MECHANISM_PARTITION_EVALUATION`, `PARALLEL_RELATION_EVALUATION` | the same names, lowercase, under `infrastructure`. These took a boolean in 1.x and now take a mapping; see below |
| `PARALLEL_CUT_EVALUATION` | `infrastructure.parallel_partition_evaluation` |
| `PARALLEL_CONCEPT_EVALUATION` | `infrastructure.parallel_distinction_evaluation`; see {doc}`/howto/parallel` |
| `MAXIMUM_CACHE_MEMORY_PERCENTAGE` | `infrastructure.memory_ceiling_percentage` (and `memory_ceiling_bytes`) |
| `CACHE_REPERTOIRES`, `CACHE_POTENTIAL_PURVIEWS` | the same names under `infrastructure`; see {doc}`/howto/cache` |
| `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` | `infrastructure.clear_system_caches_after_computing_sia` |
| `REDIS_CACHE`, `REDIS_CONFIG` | removed; results persist through the disk result cache, `infrastructure.disk_cache_results` |
| `RAY_CONFIG` | removed |
| `LOG_FILE`, `LOG_FILE_LEVEL`, `LOG_STDOUT_LEVEL` | `pyphi.enable_logging(level, file)` |
| `PROGRESS_BARS` | `infrastructure.progress_bars` |
| `WELCOME_OFF` | `infrastructure.welcome_off` |
| `REPR_VERBOSITY`, `PRINT_FRACTIONS`, `LABEL_SEPARATOR` | the same names under `infrastructure` |
| `VALIDATE_SUBSYSTEM_STATES` | `infrastructure.validate_system_states` |
| `VALIDATE_CONDITIONAL_INDEPENDENCE` | `infrastructure.validate_conditional_independence` |
| `VALIDATE_JSON_VERSION` | removed |

Some option values changed along with the names. Write the new value: a 1.x
value such as `"TRI"` is rejected when it is written, with a
`ConfigurationError` that lists the registered scheme names.

| 1.x setting | 2.0 setting |
| --- | --- |
| `PARTITION_TYPE = "BI"` | `mechanism_partition_scheme = "JOINT_BIPARTITION"` |
| `PARTITION_TYPE = "TRI"` | `mechanism_partition_scheme = "WEDGE_TRIPARTITION"` |
| `PARTITION_TYPE = "ALL"` | `mechanism_partition_scheme = "JOINT_PARTITION_ALL"` |
| `PICK_SMALLEST_PURVIEW = False` (the 1.x default) | `purview_tie_resolution = ["PHI", "PURVIEW_SIZE"]`, which keeps the largest of the tied purviews |
| `PICK_SMALLEST_PURVIEW = True` | `purview_tie_resolution = ["PHI", "NEGATIVE_PURVIEW_SIZE"]`, which keeps the smallest |
| `PARALLEL_COMPLEX_EVALUATION = True` | `parallel_complex_evaluation = {"parallel": True}`; the other keys of the mapping (`chunksize`, `sequential_threshold`, `progress`) keep their defaults |

### The quantities

| 1.x | 2.0 |
| --- | --- |
| `compute.phi(subsystem)`, IIT 3.0's Φ | `analyze(substrate, state, formalism="IIT_3_0").phi`: the same quantity. `.big_phi` is IIT 4.0's structure integrated information and raises under IIT 3.0 |
| `compute.sia(subsystem).phi` | `analyze(...).sia.phi` |
| `compute.ces(subsystem)`, the concepts | `analyze(..., formalism="IIT_3_0").ces.distinctions`. Under every version `.ces` is a `CauseEffectStructure`; under IIT 3.0 its `.relations` are empty |
| `ces.phis`, `ces.mechanisms` | `ces.distinctions.phis`, `ces.distinctions.mechanisms`, tuples |
| `concept.phi`, `concept.mechanism` | unchanged on each distinction |
| `compute.major_complex(network, state)` | `substrate.maximal_complex(state, formalism="IIT_3_0")`: a `Complex` whose `.sia` is the analysis, or a falsy null complex when no candidate is irreducible |
| `compute.complexes(network, state)`: every irreducible subsystem, overlapping ones included | `substrate.irreducible_sias(state, formalism="IIT_3_0")` |
| `compute.condensed(network, state)`: the maximal non-overlapping complexes | `substrate.complexes(state, formalism="IIT_3_0")`, recursive exclusion over every subset, strongest first. The result can be empty, as it is under the default formalism for a deterministic network, so check it before indexing |
| `compute.all_complexes(network, state)`: the analysis of every subsystem, reducible ones included | `substrate.all_sias(state, formalism="IIT_3_0")` |

The 1.x name `complexes` now belongs to a different set: 1.x
`compute.complexes` returned every irreducible subsystem, and 2.0
`substrate.complexes` returns the non-overlapping subset that exclusion
accepts, which 1.x called `condensed`. Renaming one to the other gives a
smaller set with no error.

IIT 3.0 values from PyPhi 1.x agree with 2.0's to about 1e-5. The two differ in
the sixth decimal place on some published examples (for instance a mechanism's
cause information of 0.214284 in 1.x and 0.214286 in 2.0). Compare ported
results against stored 1.x outputs with that tolerance, not with
`pyphi.numerics.eq` at the preset's precision of 6.

### Result and system attributes

Attributes that the 1.x examples use were renamed too. Each old spelling raises
an `AttributeError`, or an `ImportError` when the module is gone.

| 1.x | 2.0 |
| --- | --- |
| `subsystem.concept(mechanism)` | `system.distinction(mechanism)` |
| `subsystem.network` | `system.substrate` |
| `sia.cut`, on IIT 3.0, IIT 4.0 and actual-causation results | `sia.partition` |
| `sia.ces`, `sia.partitioned_ces` | `sia.distinctions`, `sia.partitioned_distinctions` |
| `major_complex.subsystem`, `major_complex.subsystem.nodes` | `complex.sia` and `complex.node_indices` |
| `causal_nexus(...).transition` | `cause_indices` and `effect_indices` on the result |
| `pyphi.models.Cut(from_nodes, to_nodes)` with `Subsystem(..., cut=)` | `pyphi.models.DirectedBipartition(direction, from_nodes, to_nodes)` with `System(..., partition=)` |
| `pyphi.compute.conceptual_info` | `pyphi.formalism.iit3.conceptual_info` |
| `pyphi.distance.hamming_emd` | `pyphi.measures.distribution.hamming_emd` |
| `pyphi.tpm.marginalize_out`, `node.tpm_on` | removed with the `pyphi.tpm` module; a substrate's TPM is a `FactoredTPM` (`pyphi.FactoredTPM`) with one conditional distribution per unit |

## Building and analyzing

**[1.x]** The two core objects are renamed, and the `compute` module is replaced
by a single entry point, `pyphi.analyze`.

Before:

```python
import pyphi

network = pyphi.Network(tpm, cm)
subsystem = pyphi.Subsystem(network, state, nodes)
phi = pyphi.compute.phi(subsystem)
ces = pyphi.compute.ces(subsystem)
```

After:

```python
import pyphi

substrate = pyphi.Substrate(tpm, cm=cm)
analysis = pyphi.analyze(substrate, state)

phi = analysis.phi   # system integrated information, φ_s
ces = analysis.ces   # the Φ-structure (a CauseEffectStructure)
```

`pyphi.analyze` returns an `Analysis` with `.phi`, `.ces`, `.sia` (the system
irreducibility analysis), and `.system` (the analyzed system). It analyzes
the given units — the whole substrate by default, or the `subset` argument —
and does not search for the complex. To find complexes, use
`substrate.complexes()` or `substrate.maximal_complex()`; to analyze a
specific subset, pass `subset=` or construct a `System` directly:
`pyphi.System(substrate, state, subset=(0, 1, 2))`.

## Macro systems

**[1.x]** The 1.x macro module (`MacroSubsystem`, `CoarseGrain`, `Blackbox`,
`MacroNetwork`, `all_coarse_grains`, `all_blackboxes`, `all_macro_systems`,
`emergence`, `phi_by_grain`, `effective_info`) is gone, and an attribute
lookup on `pyphi.macro` for one of them raises an error with no replacement
named. Its successor in `pyphi.macro` is built on the intrinsic units of
Marshall et al. (2026). A `MacroUnit` is defined by its micro constituents, an
update grain, and a mapping from the constituents' joint state history to the
unit's state. `MacroSystem.from_micro` builds the macro cause and effect TPMs
from the micro substrate and that history.

| 1.x | 2.0 |
| --- | --- |
| `MacroSubsystem(network, state, coarse_grain=cg)` | `MacroSystem.from_micro(substrate, units, state)` |
| `CoarseGrain(partition, grouping)` | one `MacroUnit(constituents=group, update_grain=1, mapping=coarse_grain(len(group), on_counts={...}))` per group |
| `Blackbox(partition, output_indices)` with `time_scale=t` | one `MacroUnit(constituents=box, update_grain=t, mapping=blackbox(len(box), t, output_constituents))` per box, where `output_constituents` are positions within the box; pass a micro history of `t` states in place of the single state |
| `all_coarse_grains`, `all_blackboxes`, `all_macro_systems`, `emergence`, `phi_by_grain` | `pyphi.analyze(substrate, state, grains=True)` and {func}`pyphi.macro.complexes`, which search the candidate units and systems and apply exclusion; see {doc}`/howto/grain-search` |

For example, the 1.x emergence example groups the four units of
`macro_network()` into the pairs (0, 1) and (2, 3), each ON when both of its
units are ON. In 2.0:

```python
from pyphi.macro import MacroSystem, MacroUnit, coarse_grain

substrate = pyphi.examples.macro_substrate()
units = [
    MacroUnit(constituents=group, update_grain=1,
              mapping=coarse_grain(2, on_counts={2}))
    for group in [(0, 1), (2, 3)]
]
with pyphi.config.override(**pyphi.iit3):
    macro = MacroSystem.from_micro(substrate, units, (0, 0, 0, 0))
    macro.sia().phi
```

**1.x macro values do not reproduce.** The macro TPM this code builds is the
one 1.x built, but the system integrated information differs: 1.x printed
0.597212 for this example and 2.0 gives 0.86905 under the IIT 3.0 preset. The
micro value, 0.113889, is the same in both. Macro results computed with 1.x
cannot be reproduced in 2.0 and have to be recomputed.

## Choosing a formalism

**[both]** In 1.x a single `IIT_VERSION` config toggle selected the formalism,
defaulting to IIT 3.0. In 2.0 **the default is IIT 4.0** (with the
intrinsic-information requirement), and an earlier version is selected per call:

Before:

```python
pyphi.config.IIT_VERSION = 3.0   # global toggle, default 3.0
```

After:

```python
# per call — the reliable way; sets the compatible measures for you
analysis = pyphi.analyze(substrate, state, formalism="IIT_3_0")

# for a block of code, apply the version's complete preset
with pyphi.config.override(**pyphi.iit3):
    analysis = pyphi.analyze(substrate, state)

pyphi.config.formalism.iit.version   # "IIT_4_0_2026" by default
```

The available formalisms are `"IIT_3_0"`, `"IIT_4_0_2023"`, and `"IIT_4_0_2026"`.
Because the default changed from IIT 3.0 to IIT 4.0, the same substrate and state
give a different result than a 1.x default run unless you request
`formalism="IIT_3_0"`. See
[Reproduce results from earlier versions of IIT](../howto/earlier-versions.md)
for the differences, and {ref}`formalism-selection` for the three equivalent
ways to select one.

## Configuration

**[both]** The configuration file moved from a flat format to a **layered nested**
one. Loading a legacy flat `pyphi_config.yml` is rejected with a rename map
pointing each old key to its new location.

Before (`pyphi_config.yml`):

```yaml
PRECISION: 6
PARALLEL: true
```

After (`pyphi_config.yml`):

```yaml
numerics:
  precision: 6
infrastructure:
  parallel: true
```

The three top-level layers are `formalism` (with the sub-namespaces `iit` and
`actual_causation`), `infrastructure` (parallelism, caching, logging), and
`numerics` (precision). At runtime, read a value from its layer
(`pyphi.config.numerics.precision`); a top-level write such as
`pyphi.config.precision = 6` is routed to the correct layer automatically.
The layer objects themselves are read-only.

## Saving and loading results

**[both]** The custom `pyphi.jsonify` layer (and the per-class `to_json` /
`from_json` hooks) is gone. Results are now saved and loaded with a typed
`msgspec`-based serializer:

Before:

```python
import pyphi.jsonify

data = pyphi.jsonify.jsonify(result)
```

After:

```python
ces = analysis.ces

pyphi.save(ces, "ces.json")      # or ces.save("ces.json")
ces = pyphi.load("ces.json")     # or CauseEffectStructure.load("ces.json")
```

`save` / `load` (and the `.save()` / `.load()` methods) apply to every
serializable result type, including the top-level `Analysis` wrapper — see
{doc}`Save and load results </howto/save-load>`. The format is inferred from
the extension:
`.json`, `.mpk` (msgpack), and a transparent `.gz` layer for any of them. This
is a **format break with no standalone converter**: results saved in the old
`jsonify` format cannot be loaded and must be recomputed. `pyphi.load` tells
you when a file came from 1.x. A saved 1.x network holds its TPM and
connectivity matrix as plain lists, so it can be rebuilt from the parsed JSON
with `pyphi.Substrate(data["tpm"], cm=data["cm"])`.

## Precision and φ comparison

**[both]** The φ, Φ, and α values on results (`.phi`, `.alpha`, and the small-φ
of distinctions) are plain Python floats. A direct `==` or `<` between two of
them is an **exact** floating-point comparison, so two values that differ only
by summation noise below `config.numerics.precision` compare as unequal. To
compare tolerantly at the configured precision, use the scalar predicates in the
new `pyphi.numerics` module:

```python
from pyphi import numerics

numerics.eq(a.phi, b.phi)     # tolerant equality up to config.numerics.precision
numerics.is_zero(a.phi)       # tolerant test against 0
numerics.is_positive(a.phi)
```

The precision-aware comparison helpers moved from `pyphi.utils` to
`pyphi.numerics`: `pyphi.utils.eq`, `is_positive`, and `is_nonpositive` are now
`pyphi.numerics.eq` / `is_positive` / `is_nonpositive`, alongside the new
`is_zero`, `positive_mask`, and `round_to_precision`. Update imports
accordingly.

The `PyPhiFloat` wrapper type is removed. Values that previously carried metadata
alongside a float (repertoire distances) are now `DistanceResult`, which *is* a
float subtype: it compares and arithmetic-combines exactly like the number it
holds, so no unwrapping is needed.

Because tie resolution now clusters candidate values tolerantly, a reported tie
set may include members that earlier versions silently dropped when two
candidates differed only by sub-precision noise. Selection is deterministic and
order-independent.

## Changed defaults

**[both]** The default changed from IIT 3.0 (1.x) to IIT 4.0, with the
intrinsic-information requirement
(`system_phi_measure="INTRINSIC_INFORMATION"`).
This silently changes computed values relative to a 1.x default run, so a
migration that expects IIT 3.0 numbers must request `formalism="IIT_3_0"`
explicitly; request `formalism="IIT_4_0_2023"` for the IIT 4.0 system φ
without that requirement.

A practical consequence: **deterministic networks compute $\varphi_s = 0$ by
default.** The classic examples (`xor`, `basic`, the cellular-automaton
rules) are all deterministic, so analyses ported from 1.x or from the
literature will show 0 where papers print nonzero values. This is the
expected behavior of the intrinsic-information requirement; pin
`formalism="IIT_4_0_2023"` to reproduce numbers published with the IIT 4.0
paper in 2023. See
{doc}`../theory/intrinsic-information`.

(upgrading-from-rc)=

## Upgrading from a 2.0 release candidate

**[rc]** The 2.0 release candidates (rc1 to rc3) were pre-releases, so the
names they introduced were changed without aliases, and most of the old names
fail with an ordinary `AttributeError` or `TypeError` that does not give the
replacement. Files saved by rc1, rc2 or rc3 still load; a sweep table saved by
a release candidate loads in the long format described below. Everything else
that changed between rc3 and 2.0.0 is in the tables here.

### Results that change without an error

Code that ran under a release candidate can run under 2.0.0 and return
different numbers. Recompute results from any of these.

- **`subset=` and `formalism=` were ignored.** In rc1 to rc3,
  `Substrate.sia`, `complexes`, `maximal_complex`, `irreducible_sias` and
  `all_sias`, and `System.sia`, took `**kwargs` and dropped any keyword they
  did not use. A call such as `substrate.sia(state, subset=(0, 1))` analyzed
  the whole substrate, and `substrate.complexes(state, formalism="IIT_4_0_2023")`
  or `system.sia(formalism="IIT_3_0")` computed under the active
  configuration (the 2026 default). In 2.0.0 both keywords are honored, and
  any other unrecognized keyword raises a `TypeError`. For example, on
  `examples.basic_noisy_selfloop_substrate()` in state `(1, 0, 0)`,
  `complexes(state, formalism="IIT_4_0_2023")` returned φₛ of
  0.152003, 0.121863 and 0.014518 in rc3, which are the 2026 values, and
  returns 0.686877, 0.254967 and 0.014518 now. (`Substrate.ces` raised a
  `TypeError` for these keywords in rc3.)
- **Grain search.** The search for intrinsic units and complexes
  (`pyphi.analyze(..., grains=True)`, `pyphi.macro.complexes`) now checks
  whether a macro system's state is reachable on the cause TPM, so candidates
  that are reachable only through the background's past are no longer
  discarded. It compares a candidate unit only with competitors whose units
  update no more slowly than its own (Marshall et al. 2026, Eq. 17; the
  default `max_update_grain=1` is unaffected), and it drops candidates with
  φₛ = 0 before the exclusion cascade. The search can return different
  complexes than rc3 did. It also raises a `ValueError` under
  `background_conditioning="CONDITION_CURRENT_STATE"`: the macro cause TPM is
  defined only with the background causally marginalized, and rc3 built it
  that way whatever the setting, so its values disagreed with
  `Substrate.complexes` (see {doc}`../howto/grain-search`).
- **Setting the version alone.** rc3 accepted
  `pyphi.config["formalism.iit.version"] = "IIT_4_0_2023"` (or
  `config["iit.version"]`) on its own. That changed the version recorded with
  each result but left the measures as they were, so results computed after
  such a write carry a version label that does not match the measures they
  were computed with. Recompute them. 2.0.0 raises a `ConfigurationError`;
  apply the whole preset with `pyphi.config.override(**pyphi.iit4_2023)`, or
  pass `formalism="IIT_4_0_2023"`.
- **Runner-up partition.** `sia.runner_up` is `None` when the partition sweep
  stopped early at a partition with φₛ = 0, because the partitions evaluated
  before the stop need not include the true runner-up; rc3 reported the best
  of them. Under the intrinsic-information requirement, the runner-up keeps
  the values it was ranked on, from before the requirement applies, where rc3
  lowered them by the requirement. `RunnerUp.gap` is new: how far the
  runner-up's ranking value exceeds the MIP's. Files saved by a release
  candidate load with `gap=None`.
- **Presets.** `pyphi.iit3`, `pyphi.iit4_2023`, `pyphi.iit4_2026` and
  `pyphi.conf.presets.by_name` are read-only mappings; copy one with
  `dict(pyphi.iit3)` to change it. Passing a modified copy as `formalism=`
  raises a `ValueError` where rc3 silently ran the unmodified preset. Applying
  a preset, or `formalism=`, keeps the current `relation_computation`
  instead of resetting it, because that option changes how relations are
  computed and not their values.
- **Sweep tables.** `SweepResult.df` has a default `RangeIndex` and the
  columns `substrate`, `formalism`, `subset` and `state` first, however many
  values each axis has. A table over several axes had a `MultiIndex`, so
  `df.loc[(...)]` and `df.xs(...)` fail now, and `df.iloc[:, 0]` is the
  `substrate` column, no longer `phi`. A flat sequence is one subset: rc3 read
  `subsets=("A", "B")` as the two one-unit subsets `("A",)` and `("B",)`, and
  2.0.0 runs the single subset `(0, 1)`. The `subset` column holds sorted unit
  indices whatever spelling was given, and a subset or formalism given more
  than once, under any spelling, runs once. A state given more than once gives
  one row each time, as in rc3, so states drawn with replacement keep their
  repeats.
  `sweep(subsets=None)` means the whole system. With a callable `compute`, a
  scalar result goes in a `value` column; there is no `phi` column of `None`.
  `LandscapeSection.df` has a `theta` column and a default index, where it was
  indexed by θ.
- **`to_pandas()` on one object.** A system irreducibility analysis, a
  distinction or an estimate returns a one-row `DataFrame` instead of a
  `Series` (a `Perturbation`, which had no `to_pandas()`, now has one that
  returns a one-row `DataFrame` too), and the analysis names its units column `subset`
  (sorted indices) instead of `system`. Code that reads `row["phi"]` from the
  old `Series` now gets a one-element `Series`; use `row["phi"].iloc[0]` or
  `pd.concat` the rows.
- **Purview labels.** Purview labels in distinction cards, tables and
  `plot_ces` are written in the state the distinction specifies (uppercase
  for ON, lowercase for OFF), as relation cards already did; they showed the
  current state in rc3.
- **Configuration errors.** Writing an unknown dotted path
  (`pyphi.config["iit.mechanism_phi_measur"] = ...`, `override({...})` or a
  `pyphi_config.yml` entry) raises a `ConfigurationError`, where rc3 raised a
  `KeyError`. More values are rejected when they are written:
  `parallel_workers = 0`, an unknown `parallel_backend`, a
  `memory_ceiling_percentage` outside 0 to 100, a negative
  `memory_ceiling_bytes`, an empty tie-resolution list,
  and a partition scheme, measure or actual-causation version that is not
  registered. `parallel=` must be `True`, `False` or `None` everywhere.
- **MCP tools.** The server's tools reject an argument they do not define,
  where rc3 dropped it, and a JSON boolean in `subset`. `estimate_cost` takes
  `subset`.
- **Candidates for the complex search.** `candidates=` on `Substrate.complexes`,
  `maximal_complex`, `all_sias` and `irreducible_sias` evaluates a system named
  more than once (as `(1, 2)` and `(2, 1)`, or as a subset and a `System`) once,
  at its first occurrence. In rc3 the copies tied and excluded each other, so
  the true complex could be dropped and the substrate reported to have no
  complex, or a weaker overlapping one. A flat sequence such as
  `candidates=(0, 1)` or `candidates=("A", "B")` is one candidate; rc3 read
  `("A", "B")` as the two one-unit candidates. A string raises a `TypeError`,
  as candidates and as a subset (`analyze(..., subset="AB")`), where rc3 split
  it into one-character labels.
- **IIT 3.0 `current_state`.** Under IIT 3.0, `sia.current_state` and the
  `current_state` column of its `to_pandas()` frame are the state of the
  system's units, as under IIT 4.0. In rc3 they were the state of the whole
  substrate, so a table that mixed formalisms mixed the two meanings in one
  column.
- **Reachability checks.** `System.sia()` and `System.ces()` raise
  `StateUnreachableForwardsError` when the system's state cannot be analyzed
  under the formalism in effect for the call, whether chosen with
  `formalism=` or with `pyphi.config.override(...)`, as `pyphi.analyze` does.
  rc3 returned a null result (Φ = 0). The complex functions check a `System`
  given in `candidates=` the same way. A system built with
  `validate_system_states` off is not checked. `System.ces(sia=, distinctions=)`
  and `prepare_ces(sia=)` raise a `ValueError` for results computed on another
  system.
- **Macro unit labels.** A `MacroSystem` built by `MacroSystem.from_micro`
  without `node_labels`, including every complex the grain search returns,
  labels each unit by its micro constituents' labels (`"C"`, `"AB"`) where rc3
  used `n0`, `n1`. Code that looked up a macro unit by its default label needs
  the new label.
- **Mechanism queries.** `System.mic`, `mie`, `find_mice` and `distinction`
  accept unit labels and any order of units, and raise a `ValueError` for a
  unit that is out of range or repeated. `System.find_mip`, `cause_mip`,
  `effect_mip`, `phi_cause_mip` and `phi_effect_mip` accept measure names, raise
  `MeasureNotCompatibleError` for a measure the active version does not define,
  and raise a `TypeError` when `state` is not a specified state.
  `System.sia(partition_scheme=<unknown>)` raises a `ValueError` where rc3
  raised a `KeyError`.
- **`cause_info` and `effect_info`.** `System.cause_info`, `effect_info` and
  `cause_effect_info` with the configured measure are IIT 3.0 quantities.
  Outside IIT 3.0 they raise a `TypeError` that points to `mic`, `mie` and
  `intrinsic_specification`; rc3 failed on every such call with an error about
  dispatching the measure. With an explicit `repertoire_distance=`, given as a
  measure or a registered name such as `"KLD"`, they work under every version.
- **Repeated values and argument types.** `progress=` must be `True`, `False` or
  `None`. The integer fields of `SearchBounds` reject floats and booleans. The
  `pyphi.campaign` planners reject `jobs` below 1, a `units_per_job` that is not
  positive, and an unknown `ordering` (rc3 ignored or crashed on them). A
  `config.iit = replace(..., relation_computation=X)` write applies as given,
  even when the result equals a preset. `create_substrate` and
  `substrate_generator.composite` raise a `TypeError` for an unknown key in a
  spec or an unknown parameter. A string `subset`
  raises a `TypeError`.
- **Tables and cards.** `Distinctions.to_pandas()` and
  `CauseEffectStructure.to_pandas()` have a `RangeIndex` and a `mechanism`
  column instead of an index of mechanisms (`df.set_index("mechanism")` restores
  the old shape). The `specified_state` column of `RepertoireIrreducibilityAnalysis`
  and `MaximallyIrreducibleCause/Effect` frames is a tuple, and a new
  `intrinsic_specification` column holds the intrinsic specification.
  `Complex.to_pandas()` has a `node_indices` column, and `Analysis.to_pandas()`
  starts with `formalism`, `subset`, `state` (the substrate's state, as in a
  sweep's rows) and `current_state` (the subset's state) columns. The
  `partition` column of `LandscapeSection.df` and `OptimizationResult.to_pandas()`
  uses the string of the analysis tables (`2 parts: {AB,C}; cut C→A, C→B`, and
  `None` for no partition); tables saved by a release candidate get the new
  strings when they load. The `Complex` card labels its row "Maximal
  complex" (it read "Is maximal"). Its "Excluded candidates" row no longer
  counts the candidates that were excluded without computing their φₛ; a
  separate "Gated candidates (φₛ not computed)" row counts those, and appears
  only when there are any. `PhiPosterior` compares by value, prints a compact card, and has
  `to_pandas()`.

### Renamed or changed

| rc3 | 2.0.0 |
| --- | --- |
| `System.from_substrate(substrate, state, nodes)` | `System(substrate, state, subset=nodes)` |
| `TransitionSystem.from_substrate(substrate, before_state, after_state, cause_indices, effect_indices, direction, partition)` | `TransitionSystem(...)` with the same arguments |
| `MacroSystem.from_substrate(...)`, which always raised a `TypeError` | `MacroSystem.from_micro(...)` |
| `System(substrate, state, node_indices)`, positional or as `node_indices=`; `Substrate.sia(state, indices)` and `ces`, positional or as `indices=` | `subset=`, keyword-only. The `TypeError` says so |
| `MacroSystem(substrate, state, node_indices, ...)` | `MacroSystem(substrate, state, units, ...)` with `subset=` keyword-only; `MacroSystem.from_micro` is unchanged |
| `pyphi.actual.events(..., nodes)`, and `indices=` on `true_events` and `extrinsic_events` | `subset=` on all three, keyword-only |
| `major_complex=` on `true_events` and `extrinsic_events` | `maximal_complex=` |
| `pyphi.substrate.reachable_systems(substrate, indices, state)` | `reachable_systems(substrate, state, subset=...)` |
| `Transition(..., partition, noise_background)` positional; `PerceptualSystem(substrate, system_indices, sensory_indices)`; `TriggeredTPM(array, sensory_indices, ...)`; `build_triggered_tpm(substrate, sensory_indices, ...)` | `partition=`, `noise_background=`, `system_indices=` and `sensory_indices=` are keyword-only |
| `candidates=` positional on `Substrate.complexes`, `maximal_complex`, `all_sias`, `irreducible_sias`, and on the same functions in `pyphi.substrate` | keyword-only; it also accepts subsets of units, and the module-level functions take `formalism=` |
| `analyze(..., parallel_kwargs={...})`; `parallel_kwargs=` on `Substrate.complexes` and the `pyphi.macro` search functions | `parallel=True`, `False` or `None`, and `progress=`. Finer tuning goes through the per-level options in `config.infrastructure`, with `pyphi.config.override`. `bounds=` and `prune=` on the search functions are keyword-only |
| `sweep(..., seed=)`, `campaign.prepare(..., seed=)`, `campaign.prepare_ces(..., seed=)` | removed, because no random numbers are drawn; record a seed with the `pyphi.provenance` writers (`save_dataframe(..., seed=)`) |
| `optimize(..., direction="minimize")`; `OptimizationResult.direction` | `goal="minimize"`; `OptimizationResult.goal`. Saved results load |
| `optimize(..., objective=)`; `OptimizationResult.objective_name` | `quantity=` (a callable is still accepted; `objective=` raises a `TypeError`); `quantity_name`. Saved results load. `best_objective` and the `objective` column of the trajectory keep their names |
| `pyphi.macro.estimate_search(bounds, n, limit)`; `SearchBounds.estimate(substrate, limit)` | `estimate_search(substrate, *, bounds=None, limit=1_000_000)`; `limit` is keyword-only on both |
| `allow_neg` positional on `Transition.find_causal_link` and `find_mip`; `mechanisms`, `purviews` and `allow_neg` positional on `actual.directed_account` | keyword-only |
| `actual.sia(..., **kwargs)` and the `System` mechanism queries (`find_mip`, `find_mice`, `cause_mip`, ...) with `reduce_func`, `shortcircuit_func`, `map_func` or other parallel-map options | `actual.sia` takes `parallel=` and `progress=`; the `System` queries take `parallel`, `progress`, `chunksize` and `sequential_threshold`. Anything else raises a `TypeError` |
| `SubstratePosterior.sample(...)` with neither or both of `seed` and `rng` raised a `ValueError` | raises a `TypeError` |
| `pyphi.analyze`, `pyphi.sweep` and `pyphi.optimize` as modules (`from pyphi.sweep import SweepResult`) | the modules are `pyphi.analysis`, `pyphi.sweeps` and `pyphi.optimization`; the classes are also `pyphi.Analysis`, `pyphi.SweepResult` and `pyphi.OptimizationResult`. The functions keep their top-level names |
| `StateSpecification.intrinsic_information`, `runner_up_intrinsic_information` | `intrinsic_specification`, `runner_up_intrinsic_specification` |
| `System.intrinsic_information(...)`, `pyphi.core.repertoire_algebra.intrinsic_information`, `pyphi.formalism.iit4.system_intrinsic_information` | `System.intrinsic_specification`, `repertoire_algebra.intrinsic_specification`, `iit4.system_intrinsic_specification` |
| `ria.purview_state`, `mice.purview_state`, and the `purview_state` column of `to_pandas()` | `current_state` |
| `Distinctions(concepts=...)`, `.concepts` | `Distinctions(distinctions=...)`; the object is a sequence of distinctions, so iterate or index it |
| `analysis.ces` under IIT 3.0 was a `ResolvedDistinctions` (`len(analysis.ces)`, `analysis.ces.phis`) | always a `CauseEffectStructure`; the distinctions are `analysis.ces.distinctions` |
| `examples.fig1a_substrate()`, `fig4_substrate()`, `fig4_system()`, and the rest of the unprefixed 2014 examples | `examples.iit3_2014_fig1a_substrate()`, `iit3_2014_fig4_substrate()`, `iit3_2014_fig4_system()`, and so on |
| `pyphi.models.Concept`, `System.null_concept`, and the short `pyphi.convert` names (`b2l`, `sbn2sbs`, ...), which rc3 kept as aliases | removed; use the 2.0 names in the first table of this guide |
| `System.apply_cut`, `TransitionSystem.apply_cut` and `Transition.apply_cut`, the only name rc3 had for the method | `apply_partition`. The `AttributeError` gives the new name |
| `system.distinctions(True)` | `system.distinctions(congruent=True)`; `congruent` is keyword-only, and a positional argument raises a `TypeError` |
| `DecompositionVerdict(constituents, background_apportionment, verdict)` | a required `micro_grain` field (the candidate's update grain τ) comes before `verdict`, and `intrinsic_units(...).verdicts` holds one verdict per decomposition and update grain, where rc3 held one per decomposition |
| `from pyphi import *` bound `TransitionSystem` and the internal modules `cache`, `combinatorics`, `constants`, `data_structures`, `deferred`, `log`, `node`, `protocols`, `registry`, `relabel`, `serializable` and `utils` | the star import binds the public names only; import those directly (`from pyphi.actual import TransitionSystem`, `from pyphi import utils`) |
| `AnalysisEstimate.compute == "full"` | `None`, as the `compute=` argument spells the full analysis |
| `SystemIrreducibilityAnalysis.to_pandas()` column `system` | `subset` |
| `pyphi.models.pandas.record_to_series` | removed; `to_pandas()` always returns a `DataFrame` |
| Matplotlib helpers returned a graph, a figure, axes or a tuple depending on the helper | all return `(fig, ax)`; `plot_repertoires` no longer returns the repertoires and `plot_system` no longer returns a graph |
| `pyphi.sweep(compute=...)` accepted `"sia"`, `"ces"` or a callable | also `"distinctions"` and `None` (the full analysis), as in `analyze` |
| `pyphi.estimate_analysis`, `pyphi.phi_posterior` | take `formalism=`; `phi_posterior` also takes `progress=` |
| `dynamics.simulate(tpm, initial_state, timesteps, clamp, rng, seed)` and `dynamics.mean_dynamics(tpm, repetitions, seed)` with `seed` or `rng` positional; `rng` won when both were given | `seed=` and `rng=` are keyword-only on both, and giving both raises a `TypeError` |

The IIT 3.0 and actual-causation methods that took `**kwargs` and ignored what
they did not use (`Transition.find_*`, `actual.sia`, `actual.account`,
`System.find_mice`, `find_mip`, `mic`, `mie`, `phi`, `all_distinctions`, and
`build_substrate`, `build_tpm`, `weight_axis` with a misspelt unit-function
option) now raise a `TypeError` for an unknown keyword. The unit-function
check is skipped when one of the unit functions is your own and takes
`**kwargs`, since it can read any keyword.
