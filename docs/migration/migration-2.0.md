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

## Renames at a glance

| Old | New | Affects |
| --- | --- | --- |
| `pyphi.Network` | `pyphi.Substrate` | [1.x] |
| `pyphi.Subsystem` | `pyphi.System` | [1.x] |
| `pyphi.compute.*` | `pyphi.analyze(...)` | [1.x] |
| `subsystem.cause_tpm` | `system.cause_marginal` | [both] |
| `subsystem.effect_tpm` | `system.effect_marginal` | [both] |
| `pyphi.jsonify` | `pyphi.serialize` / `pyphi.save` / `pyphi.load` | [both] |
| `pyphi.config.IIT_VERSION` | `pyphi.config.formalism.iit.version` | [both] |
| `pyphi.__version__` | `importlib.metadata.version("pyphi")` | [both] |

`cause_marginal` and `effect_marginal` (with the `proper_cause_marginal` /
`proper_effect_marginal` variants) are the causal marginals of IIT 4.0. The old
`cause_tpm` / `effect_tpm` names were a misnomer — the value was never a
transition probability matrix but a distribution over cause/effect states. See
[Substrate and system](../theory/substrate-and-system.md) for what they are.

### Example networks

The example functions follow the same vocabulary: every `*_network` is now
`*_substrate` and every `*_subsystem` is `*_system`. Transitions and bare
matrices (`prevention_transition`, `cond_depend_tpm`, `cond_independ_tpm`,
`differentiation_*_tpm`) keep their names.

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
| `examples.fig1a_network()`, `fig3a`, `fig3b`, `fig16` | `examples.fig1a_substrate()`, `fig3a_substrate()`, `fig3b_substrate()`, `fig16_substrate()` |
| `examples.fig4_network()` / `_subsystem()`, `fig5a`, `fig5b` | `examples.fig4_substrate()` / `_system()`, `fig5a_*`, `fig5b_*` |
| `examples.actual_causation_network()` | `examples.actual_causation_substrate()` |
| `examples.disjunction_conjunction_network()` | `examples.disjunction_conjunction_substrate()` |
| `examples.frog_example()` | `examples.frog_substrate()` and `examples.frog_transition()` |
| `examples.differentiation_micro_1_subsystem()` | `examples.differentiation_micro_1_system()` |

The IIT 4.0 paper's networks are new in 2.0 (`iit4_2023_fig1a_substrate()`
and the other `iit4_2023_*`, `marshall_2023_*`, and `mayner_2026_*`
functions); the {doc}`examples gallery </reference/examples>` lists every
registered example.

### Configuration options

Every 1.x option and where it went. Names are lowercase and live under a
layer; set one with `pyphi.config.<layer>.<option>` or a top-level write such
as `pyphi.config.precision = 6`, which is routed to its layer. Measure values
are the names in {mod}`pyphi.measures`.

| 1.x option | 2.0 option |
| --- | --- |
| `IIT_VERSION` | `formalism.iit.version` (`"IIT_3_0"`, `"IIT_4_0_2023"`, `"IIT_4_0_2026"`); prefer `analyze(..., formalism=)` or a preset, which also set the measures |
| `MEASURE` (before 1.2), `REPERTOIRE_DISTANCE` | `formalism.iit.mechanism_phi_measure` (mechanism level) and `formalism.iit.system_phi_measure` (system level) |
| `REPERTOIRE_DISTANCE_SPECIFICATION` | `formalism.iit.specification_measure` |
| `REPERTOIRE_DISTANCE_DIFFERENTIATION` | removed: intrinsic differentiation is the surprisal of the specified state and has no measure option |
| `CES_DISTANCE` | `formalism.iit.ces_measure` |
| `ACTUAL_CAUSATION_MEASURE` | `formalism.actual_causation.alpha_measure` |
| `PARTITION_TYPE` | `formalism.iit.mechanism_partition_scheme` |
| `SYSTEM_PARTITION_TYPE` | `formalism.iit.system_partition_scheme` |
| `SYSTEM_PARTITION_INCLUDE_COMPLETE` | `formalism.iit.system_partition_include_total` |
| `SYSTEM_CUTS` | removed; the `iit3` preset sets `system_partition_scheme="DIRECTED_BIPARTITION"` (the 3.0-style cut), and concept-style cuts are gone |
| `DISTINCTION_PHI_NORMALIZATION` | `formalism.iit.distinction_phi_normalization` |
| `RELATION_COMPUTATION` | `formalism.iit.relation_computation` |
| `STATE_TIE_RESOLUTION`, `MIP_TIE_RESOLUTION`, `PURVIEW_TIE_RESOLUTION` | `formalism.iit.state_tie_resolution`, `mip_tie_resolution`, `purview_tie_resolution` (plus the new `sia_tie_resolution`); see {doc}`/howto/tie-breaking` |
| `PICK_SMALLEST_PURVIEW` (before 1.2) | `formalism.iit.purview_tie_resolution` |
| `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` | `formalism.iit.assume_partitions_cannot_create_new_concepts` |
| `SHORTCIRCUIT_SIA` | `formalism.iit.shortcircuit_sia` |
| `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | `formalism.iit.single_micro_nodes_with_selfloops_have_phi` |
| `PRECISION` | `numerics.precision` |
| `PARALLEL`, `PARALLEL_WORKERS`, `PARALLEL_BACKEND` | `infrastructure.parallel`, `parallel_workers`, `parallel_backend` |
| `PARALLEL_COMPLEX_EVALUATION`, `PARALLEL_PURVIEW_EVALUATION`, `PARALLEL_MECHANISM_PARTITION_EVALUATION`, `PARALLEL_RELATION_EVALUATION` | the same names, lowercase, under `infrastructure` |
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

### The quantities

| 1.x | 2.0 |
| --- | --- |
| `compute.phi(subsystem)`, IIT 3.0's Φ | `analyze(substrate, state, formalism="IIT_3_0").phi`: the same quantity. `.big_phi` is IIT 4.0's structure integrated information and raises under IIT 3.0 |
| `compute.sia(subsystem).phi` | `analyze(...).sia.phi` |
| `compute.ces(subsystem)`, the concepts | `analyze(..., formalism="IIT_3_0").ces`, a `ResolvedDistinctions`; the concepts are `.concepts` |
| `concept.phi`, `concept.mechanism` | unchanged on each concept |
| `compute.major_complex(network, state)` | `substrate.complexes(state)`, recursive exclusion over every subset; the first entry is the strongest |

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
`pyphi.System(substrate, state, node_indices=(0, 1, 2))`.

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

# or via configuration
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
`jsonify` format cannot be loaded and must be recomputed.

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
