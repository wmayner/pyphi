# Migrating pre-2.0 PyPhi code to 2.0

PyPhi 2.0 is a breaking release. **There are no deprecation shims:** every
pre-2.0 name below is gone, not aliased, so old code raises `ImportError` or
`AttributeError` until it is updated. When helping someone port their code,
change every occurrence — a partial rename will not run.

(The full guide is at
https://pyphi.readthedocs.io/en/stable/migration/migration-2.0.html, with its
markdown source at `_sources/migration/migration-2.0.md.txt` under the same
site root; this is the condensed agent-facing copy. Keep the two in sync.)

## 1. Renames at a glance

| Old (pre-2.0) | New (2.0) |
| --- | --- |
| `pyphi.Network` | `pyphi.Substrate` |
| `pyphi.Subsystem` | `pyphi.System` |
| `pyphi.network_generator` | `pyphi.substrate_generator` |
| `pyphi.compute.phi(...)` / `pyphi.compute.ces(...)` | `pyphi.analyze(...)` |
| `from pyphi import new_big_phi` | `from pyphi.formalism import iit4` |
| `pyphi.new_big_phi.phi_structure` | `pyphi.formalism.iit4.ces` |
| `pyphi.new_big_phi.sia` | `pyphi.formalism.iit4.sia` |
| `pyphi.metrics` | `pyphi.measures` |
| `pyphi.models.Concept` | `pyphi.models.Distinction` (`Concept` still works as an alias) |
| `subsystem.cause_tpm` / `effect_tpm` | `system.cause_marginal` / `effect_marginal` (plus `proper_*` variants) |
| `pyphi.jsonify` | `pyphi.serialize` / `pyphi.save` / `pyphi.load` |
| `pyphi.utils.eq` / `is_positive` / `is_nonpositive` | `pyphi.numerics.eq` / `is_positive` / `is_nonpositive` |
| `pyphi.config.IIT_VERSION` | `pyphi.config.formalism.iit.version` |
| `pyphi.__version__` | `importlib.metadata.version("pyphi")` |

Two traps in that table:

- **The result-type names were swapped.** The old `CauseEffectStructure` is now
  `pyphi.models.Distinctions`; the old `PhiStructure` is now
  `pyphi.models.CauseEffectStructure`. The same words point at different
  objects, so an unported import can succeed and still be wrong.
- **`phi` and `big_phi` changed kind.** The top-level
  `pyphi.compute.phi(subsystem)` *function* (IIT 3.0's Φ) is gone. `.phi` is
  now the φₛ property of an analysis, and `big_phi` names the **Φ** property
  of an IIT 4.0 cause-effect structure (`ces.big_phi`). Do not translate the
  old function call into a property access blindly; the modern entry point
  is `pyphi.analyze`.

`cause_marginal` / `effect_marginal` are the causal marginals of IIT 4.0. The
old `cause_tpm` / `effect_tpm` names were a misnomer: the value was never a
transition probability matrix but a distribution over cause/effect states.

### Example networks

Every `*_network` example is now `*_substrate` and every `*_subsystem` is
`*_system` (`basic_network()` → `basic_substrate()`, `basic_subsystem()` →
`basic_system()`, and so on for `xor`, `residue`, `grid3`, `rule110`,
`rule154`, `macro`, `fig4`, `fig5a`, `fig5b`); `frog_example()` split into
`frog_substrate()` and `frog_transition()`. Transitions and bare matrices keep
their names. The IIT 4.0 paper's networks are new: `iit4_2023_fig1a_substrate()`
and the other `iit4_2023_*` functions.

### Configuration options

| 1.x option | 2.0 option |
| --- | --- |
| `IIT_VERSION` | `formalism.iit.version`; prefer `analyze(..., formalism=)` or a preset |
| `MEASURE` (before 1.2), `REPERTOIRE_DISTANCE` | `formalism.iit.mechanism_phi_measure` and `formalism.iit.system_phi_measure` |
| `REPERTOIRE_DISTANCE_SPECIFICATION` | `formalism.iit.specification_measure` |
| `REPERTOIRE_DISTANCE_DIFFERENTIATION` | removed: differentiation is the surprisal of the specified state |
| `CES_DISTANCE` | `formalism.iit.ces_measure` |
| `ACTUAL_CAUSATION_MEASURE` | `formalism.actual_causation.alpha_measure` |
| `PARTITION_TYPE` / `SYSTEM_PARTITION_TYPE` | `formalism.iit.mechanism_partition_scheme` / `system_partition_scheme` |
| `SYSTEM_PARTITION_INCLUDE_COMPLETE` | `formalism.iit.system_partition_include_total` |
| `SYSTEM_CUTS` | removed; the `iit3` preset sets `system_partition_scheme="DIRECTED_BIPARTITION"` |
| `PICK_SMALLEST_PURVIEW` (before 1.2), `*_TIE_RESOLUTION` | `formalism.iit.purview_tie_resolution`, `state_tie_resolution`, `mip_tie_resolution`, `sia_tie_resolution` |
| `DISTINCTION_PHI_NORMALIZATION`, `RELATION_COMPUTATION`, `SHORTCIRCUIT_SIA`, `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS`, `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | the same names, lowercase, under `formalism.iit` (`assume_partitions_cannot_create_new_concepts`) |
| `PRECISION` | `numerics.precision` |
| `PARALLEL*`, `CACHE_*`, `PROGRESS_BARS`, `WELCOME_OFF`, `REPR_VERBOSITY`, `PRINT_FRACTIONS`, `LABEL_SEPARATOR`, `VALIDATE_*` | the same names, lowercase, under `infrastructure` (`PARALLEL_CUT_EVALUATION` → `parallel_partition_evaluation`, `PARALLEL_CONCEPT_EVALUATION` → `parallel_distinction_evaluation`, `VALIDATE_SUBSYSTEM_STATES` → `validate_system_states`, `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` → `clear_system_caches_after_computing_sia`) |
| `MAXIMUM_CACHE_MEMORY_PERCENTAGE` | `infrastructure.memory_ceiling_percentage` |
| `REDIS_*`, `RAY_CONFIG`, `VALIDATE_JSON_VERSION` | removed; results persist through `infrastructure.disk_cache_results` |
| `LOG_*` | `pyphi.enable_logging(level, file)` |

### The quantities

| 1.x | 2.0 |
| --- | --- |
| `compute.phi(subsystem)`, IIT 3.0's Φ | `analyze(substrate, state, formalism="IIT_3_0").phi`; `.big_phi` is IIT 4.0's Φ and raises under IIT 3.0 |
| `compute.sia(subsystem).phi` | `analyze(...).sia.phi` |
| `compute.ces(subsystem)` | `analyze(..., formalism="IIT_3_0").ces`, a `ResolvedDistinctions` with the concepts under `.concepts` |
| `compute.major_complex(network, state)` | `substrate.complexes(state)`; the first entry is the strongest |

## 2. Building and analyzing

The two core objects are renamed, and the whole `compute` module is replaced by
one entry point, `pyphi.analyze`.

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
ces = analysis.ces   # the Φ-structure
```

`pyphi.analyze` returns an `Analysis` with `.phi`, `.ces`, `.sia` (the system
irreducibility analysis), and `.system` (the analyzed system). It analyzes
the given units — the whole substrate by default, or the `subset` argument —
and does not search for the complex; use `substrate.complexes()` or
`substrate.maximal_complex()` for that. To analyze a specific subset,
pass `subset=` or construct a `System` directly:
`pyphi.System(substrate, state, node_indices=(0, 1, 2))`. Code that imported the
low-level IIT 4.0 functions from `new_big_phi` now imports `sia` and `ces` from
`pyphi.formalism.iit4`.

## 3. Choosing a formalism

In 1.x a single `IIT_VERSION` toggle selected the formalism, defaulting to IIT
3.0. In 2.0 the formalism is chosen per call, and **the default is now IIT 4.0
(2026)** with the intrinsic information requirement.

Before:

```python
pyphi.config.IIT_VERSION = 3.0   # global toggle, default 3.0
```

After:

```python
# per call — the reliable way; it sets the compatible measures for you
analysis = pyphi.analyze(substrate, state, formalism="IIT_3_0")

# or via configuration
pyphi.config.formalism.iit.version   # "IIT_4_0_2026" by default
```

The available formalisms are `"IIT_3_0"`, `"IIT_4_0_2023"`, and
`"IIT_4_0_2026"`.

## 4. Configuration

The configuration file moved from a flat format to a layered nested one. A
legacy flat `pyphi_config.yml` is **rejected** on load with an error pointing
each old key to its new location.

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

## 5. Saving and loading results

The custom `pyphi.jsonify` layer (and the per-class `to_json` / `from_json`
hooks) is gone. Results are saved and loaded with a typed `msgspec` serializer.

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

The format is inferred from the extension: `.json`, `.mpk` (msgpack), and a
transparent `.gz` layer for either. This is a **format break with no converter**
— data written by the old `jsonify` cannot be loaded and must be recomputed.

## 6. Comparing φ tolerantly

The φ, Φ, and α values on results are plain Python floats, so `==` and `<`
between them are **exact** floating-point comparisons: two values that differ
only by summation noise below `config.numerics.precision` compare as unequal.
Use the scalar predicates in `pyphi.numerics` for tolerant comparison:

```python
from pyphi import numerics

numerics.eq(a.phi, b.phi)     # tolerant equality at config.numerics.precision
numerics.is_zero(a.phi)       # tolerant test against 0
numerics.is_positive(a.phi)
```

The precision-aware helpers `eq`, `is_positive`, and `is_nonpositive` moved from
`pyphi.utils` to `pyphi.numerics`, joined by `is_zero`, `positive_mask`, and
`round_to_precision`.

## 7. Changed default formalism: φ_s = 0 for deterministic systems

Because the default formalism changed from IIT 3.0 to IIT 4.0 (2026), the same
substrate and state give a **different result** than a 1.x default run unless
`formalism="IIT_3_0"` is requested.

Under the 2026 default, **deterministic networks compute φ_s = 0.** The classic examples (`xor`,
`basic`, the cellular-automaton rules) are all deterministic, so analyses ported
from 1.x or from the literature will show 0 where papers print nonzero values.
This follows from the intrinsic-information requirement of IIT 4.0 (2026). To
reproduce old numbers:

- IIT 3.0 numbers → `formalism="IIT_3_0"`
- IIT 4.0 (2023) system φ, without the requirement → `formalism="IIT_4_0_2023"`

See the `gotchas` and `theory` references for what the requirement is.

## 8. Checklist for migrating a file

1. Rename `Network`→`Substrate`, `Subsystem`→`System`, and the other names in
   the table above; fix every `import`.
2. Replace `pyphi.compute.*` calls with `pyphi.analyze(substrate, state)` and
   read `.phi` / `.ces` / `.sia` off the result.
3. Convert any `pyphi_config.yml` and `pyphi.config.*` writes to the layered
   form.
4. Replace `pyphi.jsonify` save/load with `pyphi.save` / `pyphi.load`; recompute
   any data stored in the old format.
5. If the code must reproduce pre-2.0 numbers, pass `formalism="IIT_3_0"` (or
   `"IIT_4_0_2023"`) — otherwise it silently runs under the new default.
6. Flag every place where the result changed value, not just the API.
