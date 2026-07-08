# Migrating to PyPhi 2.0

PyPhi 2.0 is a breaking release. It implements IIT 4.0 (Albantakis et al., 2023)
as its default formalism, while retaining IIT 3.0 through configuration. There
are no deprecation shims: code written against pre-2.0 PyPhi must be updated to
run.

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

`cause_marginal` and `effect_marginal` (with the `proper_cause_marginal` /
`proper_effect_marginal` variants) are the causal marginals of IIT 4.0. The old
`cause_tpm` / `effect_tpm` names were a misnomer — the value was never a
transition probability matrix but a distribution over cause/effect states. See
[Substrate and system](../theory/substrate-and-system.md) for what they are.

## Building and analyzing

**[1.x]** The two core objects are renamed, and the `compute` module is replaced
by a single entry point, `pyphi.analyze`.

Before:

```python
import pyphi

network = pyphi.Network(tpm, cm)
subsystem = pyphi.Subsystem(network, state, nodes)
phi = pyphi.compute.big_phi(subsystem)
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

`pyphi.analyze` returns an `Analysis` carrying `.phi`, `.ces`, `.sia` (the system
irreducibility analysis), and `.system` (the complex). To analyze a specific
subset rather than searching for the complex, construct a `System` directly:
`pyphi.System(substrate, state, node_indices=(0, 1, 2))`.

## Choosing a formalism

**[both]** In 1.x a single `IIT_VERSION` config toggle selected the formalism,
defaulting to IIT 3.0. In 2.0 the formalism is chosen per call, and **the default
is now IIT 4.0 (2023)**:

Before:

```python
pyphi.config.IIT_VERSION = 3.0   # global toggle, default 3.0
```

After:

```python
# per call — the reliable way; sets the compatible measures for you
analysis = pyphi.analyze(substrate, state, formalism="IIT_3_0")

# or via configuration
pyphi.config.formalism.iit.version   # "IIT_4_0_2023" by default
```

The available formalisms are `"IIT_3_0"`, `"IIT_4_0_2023"`, and `"IIT_4_0_2026"`.
Because the default changed from IIT 3.0 to IIT 4.0, the same substrate and state
give a different result than a 1.x default run unless you request
`formalism="IIT_3_0"`. See [formalism versions](../theory/formalism-versions.md)
for the differences.

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

`save` / `load` (and the `.save()` / `.load()` methods) apply to the serializable
result types — cause–effect structures and the analyses that compose them — not
to the top-level `Analysis` wrapper. The format is inferred from the extension:
`.json`, `.mpk` (msgpack), and a transparent `.gz` layer for any of them. This
is a **format break with no standalone converter**: results saved in the old
`jsonify` format cannot be loaded and must be recomputed.

## Changed defaults

**[both]** The default formalism changed from IIT 3.0 (1.x) to IIT 4.0 (2023).
This silently changes computed values relative to a 1.x default run, so a
migration that expects IIT 3.0 numbers must request `formalism="IIT_3_0"`
explicitly.
