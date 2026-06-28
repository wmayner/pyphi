# Migrating from `substrate_modeler` to current PyPhi

This guide ports code written against Bjørn Juel's external
[`substrate_modeler`](https://github.com/bjorneju/substrate_modeler) library to
the current PyPhi API. It is written to be executed top-to-bottom by an agent or
a person; every code block runs as shown.

`substrate_modeler`'s mechanism library now ships natively inside
`pyphi.substrate_generator`, so the recommended path is a **full PyPhi-native
rebuild** with no runtime dependency on the old library. The 16 unit mechanisms,
the 6 composite-combination strategies, and a per-node `create_substrate()`
factory are all built in, and the resulting substrate reproduces the original
library's `dynamic_tpm` **byte-for-byte**.

---

## 1. Orientation: the two things called "Unit"

The single most important fact: **`Unit` means something different in each
library.**

| | `substrate_modeler` | current PyPhi (`pyphi.core.unit.Unit`) |
|---|---|---|
| What it is | A mechanism-carrying object | A trivial identity value type |
| Holds | `index`, `inputs`, `mechanism` (e.g. `"sigmoid"`), `params`, mutable `state`, `input_state` | `index`, `label`, `alphabet_size` — nothing else |
| Computes a TPM? | Yes — the unit owns its update logic | No — units do not carry dynamics |
| Mutable? | Yes (state changes in place) | No (`frozen=True` dataclass) |

So migration **never** maps `Unit → Unit`. Instead, a unit's *behavior* (its
mechanism + inputs + params) becomes one entry in a `node_params` dict consumed
by `pyphi.substrate_generator.create_substrate()`, which builds a stateless
`pyphi.Substrate`.

### The architecture shift

| concept | `substrate_modeler` | current PyPhi |
|---|---|---|
| dynamics container | `Substrate(units, state)` — stateful, owns TPM logic | `Substrate(tpm, cm, node_labels)` — **stateless** value type |
| where state lives | on the substrate and on each unit | on `System`, not on the substrate |
| analyzable object | `pyphi` `Subsystem` (old) | `pyphi.System` |
| per-node construction | build `Unit`/`CompositeUnit` objects, pass a list to `Substrate(...)` | pass a `node_params` dict to `create_substrate(...)` |

The original analyzed the substrate's **`dynamic_tpm`** (present state = past
state). PyPhi has no separate "present" and "past" — a single static
`Substrate` *is* that dynamic TPM, with a unit's dependence on its own current
state expressed as a self-loop. `create_substrate` reproduces the original
`dynamic_tpm` exactly.

Porting a script is therefore two independent jobs:

1. **Construction** — build a PyPhi `Substrate` (§2–§4).
2. **Downstream calls** — update the old-PyPhi calls the script makes on the
   result (§5).

---

## 2. Construction: the canonical rebuild with `create_substrate`

`create_substrate(node_params)` mirrors the original's per-node construction most
directly: each node names a `mechanism`, its `inputs`, and a `params` dict.

```python
import pyphi
from pyphi.substrate_generator import create_substrate

substrate = create_substrate(
    {
        0: {"mechanism": "sigmoid", "inputs": (1, 2),
            "params": {"input_weights": (0.9, 0.5), "determinism": 4.0}},
        1: {"mechanism": "and", "inputs": (0, 2)},
        2: {"mechanism": "xor", "inputs": (0, 1)},
    },
    labels=("A", "B", "C"),
)

system = pyphi.System.from_substrate(substrate, state=(1, 0, 0))
```

The translation from `substrate_modeler` is mechanical:

| `substrate_modeler` | `create_substrate` `node_params` entry |
|---|---|
| `Unit(index=j, inputs=I, mechanism=M, params=P, label=L)` | `j: {"mechanism": M, "inputs": I, "params": P}` (label via the `labels` arg) |
| a unit's `state` / `input_state` | dropped — state is passed to `System`, not the substrate |
| `Substrate([u0, u1, ...])` | `create_substrate({0: ..., 1: ...})` |

What `create_substrate` does for you:

- **Builds the weight matrix** from each node's `inputs` and
  `params["input_weights"]` (convention `W[i, j]` = input from `i` to `j`, the
  same as the original). Inputs without weights become connectivity markers.
- **Inserts a self-loop** for state-dependent mechanisms (see §4) so the
  connectivity matrix is consistent with the unit's dependence on its own state.
- **Binds each node's params independently** (so two nodes can use the same
  mechanism with different params — no shared-parameter limitation).

`node_params` may be an integer-keyed dict (sorted by index) or an ordered list.

---

## 3. Mechanism mapping table

Every `substrate_modeler` mechanism is now a built-in. In `create_substrate`,
mechanism names resolve through `pyphi.substrate_generator.MECHANISMS`, which
preserves the original library's semantics exactly.

| `substrate_modeler` mechanism | `create_substrate` `"mechanism"` | params (defaults) |
|---|---|---|
| `sigmoid` | `"sigmoid"` | `input_weights`, `determinism` (5.0), `threshold` (0.0), `ising` (True), `floor` (0.0), `ceiling` (1.0) |
| `resonator` | `"resonator"` | `determinism`, `threshold`, `weight_scale_mapping` (the matching-paper coupling *g*), `input_weights`, `floor`, `ceiling` |
| `sor` | `"sor"` | `pattern_selection`, `ceiling`, `selectivity` (2.0) |
| `gabor` | `"gabor"` | `preferred_states`, `ceiling`, `floor` |
| `mismatch_corrector` | `"mismatch_corrector"` | `bias` (0.0), `floor`, `ceiling` |
| `mismatch_pattern_detector` | `"mismatch_pattern_detector"` | `pattern_selection`, `selectivity`, `ceiling`, `floor` — see caveat below |
| `copy` | `"copy"` | `floor`, `ceiling` |
| `and` / `or` / `xor` | `"and"` / `"or"` / `"xor"` | `floor`, `ceiling` (2-input truth tables, as in the original) |
| `democracy` / `majority` | `"democracy"` / `"majority"` | `floor`, `ceiling` |
| `weighted_mean` | `"weighted_mean"` | `input_weights`, `floor`, `ceiling` |
| `modulated_sigmoid` | `"modulated_sigmoid"` | `input_weights`, `modulation`, `determinism`, `threshold`, `floor`, `ceiling` |
| `biased_sigmoid` | `"biased_sigmoid"` | `input_weights` (last entry is the bias factor), `determinism`, `threshold` |
| `stabilized_sigmoid` | `"stabilized_sigmoid"` | `input_weights`, `determinism`, `threshold`, `modulation` — see caveat below |

`floor`/`ceiling` are native everywhere — no wrapper needed. `weighted_mean`
takes its weights under `input_weights` (the original used `weights`).
**Spelling:** the original library spells the endorsement mechanism
`"resonnator"`; PyPhi uses the corrected `"resonator"`, so update that name when
porting.

**Two caveats (both faithfully handled, neither used by the matching paper):**

- `mismatch_pattern_detector`: the original library has a bug (a `Nonee` typo)
  that raises on every call; the PyPhi port implements the *documented* intended
  behavior instead.
- `stabilized_sigmoid`: the original swaps its input and modulator axes (a
  Fortran-reshape artifact); the PyPhi port uses the documented convention
  (modulators are exactly `modulation["modulator"]`), so it does not reproduce
  that artifact.

Everything else reproduces the original's `dynamic_tpm` byte-for-byte.

---

## 4. State-dependent mechanisms (the endorsement family)

`resonator`, `mismatch_corrector`, `mismatch_pattern_detector`,
`modulated_sigmoid`, and `stabilized_sigmoid` depend on the unit's **own current
state** (`state[element]`). `resonator` is the matching paper's "endorsement"
mechanism: inputs agreeing with the unit's state are excitatory and amplified,
disagreeing inputs inhibitory. Its default `weight_scale_mapping` is the paper's
coupling factor *g* = `{(0,0): 1.0, (1,0): 0.5, (0,1): 0.75, (1,1): 1.5}`
(keyed by `(unit_state, input_state)`).

`create_substrate` inserts the required self-loop automatically, so you only list
the functional inputs:

```python
substrate = create_substrate(
    {
        0: {"mechanism": "sigmoid", "inputs": (0,),
            "params": {"input_weights": (0.9,), "determinism": 4.0}},
        1: {"mechanism": "resonator", "inputs": (0, 1, 2),
            "params": {"input_weights": (0.5, 0.8, 0.3), "determinism": 4.0,
                       "threshold": 0.0}},  # weight_scale_mapping defaults to g
        2: {"mechanism": "sigmoid", "inputs": (1,),
            "params": {"input_weights": (0.8,), "determinism": 3.0}},
    }
)
```

### Composite units

A `CompositeUnit` becomes a node spec with a `composite` list and a
`mechanism_combination` (one of `selective`, `average`, `maximal`,
`first_necessary`, `integrator`, `serial`, default `selective`):

```python
node = {
    "composite": [
        {"mechanism": "resonator", "inputs": (0, 1, 2),
         "params": {"input_weights": (0.5, 0.8, 0.3), "determinism": 4.0,
                    "threshold": 0.0}},
        {"mechanism": "mismatch_corrector", "inputs": (0,),
         "params": {"bias": 0.5}},
    ],
    "mechanism_combination": "selective",
}
```

`first_necessary` exposes the original's two hardcoded constants as
`combination_params={"steepness": 5.0, "offset": 0.5}` (the defaults reproduce
the original).

---

## 5. Lower-level construction (uniform mechanisms, custom callables)

For a substrate where every node uses the same mechanism, or for a genuinely
custom mechanism, use `build_substrate(unit_functions, weights)` directly. A unit
function has the signature `f(element, weights, state, **kwargs) -> float` and is
called once per from-state.

```python
import numpy as np
from pyphi.substrate_generator import build_substrate

W = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
substrate = build_substrate("sigmoid", W, determinism=4.0)  # uniform
```

A custom mechanism is any callable with that signature; a state-dependent one
reads `state[element]` (and needs a self-loop in `W`):

```python
def my_threshold(element, weights, state, *, theta=1.0, **kwargs):
    drive = float(np.dot(state, weights[:, element]))
    return 1.0 if drive >= theta else 0.0

substrate = build_substrate([my_threshold, "and", "xor"], W, theta=2.0)
```

Note: in `build_substrate`, names resolve through
`pyphi.substrate_generator.UNIT_FUNCTIONS`, where `"and"`/`"or"` keep an older
**weighted-threshold** meaning (input sum `>= num_inputs` / `>= 1`) rather than
the 2-input truth tables. For faithful `substrate_modeler` gate semantics, use
`create_substrate` (which resolves through `MECHANISMS`).

`build_substrate`'s `**kwargs` are shared across all nodes; for per-node params
use `create_substrate`, or pass a list of closures.

---

## 6. Downstream API rename map

`substrate_modeler` returns *old* PyPhi objects, so the analysis code after
construction must be updated too. **The entire `pyphi.compute` module has been
removed** — those operations are now methods on `System` and `Substrate`.

| old `substrate_modeler` / old PyPhi | current PyPhi |
|---|---|
| `pyphi.network.Network(tpm, cm)` | `pyphi.Substrate(tpm, cm, node_labels)` |
| `pyphi.subsystem.Subsystem(net, state, nodes)` | `pyphi.System.from_substrate(sub, state, nodes)` (or `System(sub, state, node_indices=nodes)`) |
| `substrate.get_network(state)` | the `Substrate` is stateless — build it once; state goes to the `System` |
| `substrate.get_subsystem(state, nodes)` | `System.from_substrate(sub, state, nodes)` |
| `substrate.dynamic_tpm` | `sub.tpm` (a `FactoredTPM`; `np.asarray(sub.tpm.to_joint())[..., 1]` for the state-by-node ON-probabilities) |
| `compute.big_phi(subsystem)` | `system.sia().phi` |
| `compute.ces(subsystem)` | `system.ces()` |
| `compute.major_complex(network, state)` | `sub.maximal_complex(state)` |
| (all candidate complexes) | `sub.complexes(state)` / `sub.all_sias(state)` |
| `subsystem.concept(mechanism)` | `system.distinction(mechanism)` |
| `subsystem.cause_repertoire(m, p)` | `system.cause_repertoire(m, p)` (same name) |
| `subsystem.effect_repertoire(m, p)` | `system.effect_repertoire(m, p)` (same name) |

Count distinctions in a cause-effect structure with `len(ces.distinctions)`
(a `CauseEffectStructure` itself has no `len`).

### Matching / perception

If the script used the matching research repo's perception layer, that now lives
in `pyphi.matching` (`PerceptualSystem`, `TriggeredTPM`, `triggering_coefficient`,
`Perception`, `Differentiation`, `MatchingAnalysis`). A natively-built substrate
feeds straight in:

```python
from pyphi.matching import PerceptualSystem

ps = PerceptualSystem(substrate, system_indices=(1, 2), sensory_indices=(0,))
triggered = ps.triggered_states(tau=2, tau_clamp=1)
```

### IIT version

`substrate_modeler` predates IIT 4.0 and targeted 3.0-era PyPhi. Current PyPhi
defaults to IIT 4.0, so Φ values and the available quantities differ. To
reproduce 3.0-era results, switch the formalism:

```python
with pyphi.config.override(version="IIT_3_0"):
    sia = system.sia()
```

---

## 7. Worked end-to-end example

The classic OR / AND / XOR network (the IIT 4.0 paper's basic example).

**Before** — `substrate_modeler` style (illustrative; do not run):

```python
from units import Unit, Substrate
import pyphi

a = Unit(index=0, inputs=(1, 2), mechanism="or",  label="A")  # A <- B, C
b = Unit(index=1, inputs=(2,),   mechanism="and", label="B")  # B <- C
c = Unit(index=2, inputs=(0, 1), mechanism="xor", label="C")  # C <- A, B

substrate = Substrate([a, b, c], state=(1, 0, 0))
subsystem = substrate.get_subsystem(state=(1, 0, 0))
phi = pyphi.compute.big_phi(subsystem)
```

**After** — current PyPhi (runs as shown):

```python
import numpy as np
import pyphi
from pyphi.substrate_generator import build_substrate

# W[i, j] = input from i to j.  A <- B, C ; B <- C ; C <- A, B
W = np.array([
    [0.0, 0.0, 1.0],   # A -> C
    [1.0, 0.0, 1.0],   # B -> A, C
    [1.0, 1.0, 0.0],   # C -> A, B
])

substrate = build_substrate(
    unit_functions=["or", "and", "parity"],   # xor == parity
    weights=W,
    node_labels=("A", "B", "C"),
)

system = pyphi.System.from_substrate(substrate, state=(1, 0, 0))
print("phi:", float(system.sia().phi))         # 0.415037
print("distinctions:", len(system.ces().distinctions))
```

This `substrate` is byte-identical to `pyphi.examples.basic_substrate()` (Φ =
0.415037) — a handy correctness anchor while porting. It uses `build_substrate`
with the **weighted-threshold** gates because this network has a single-input
`B = and(C)`, which a 2-input truth-table gate can't express; the weighted
`"and"`/`"or"` handle any fan-in (§5). For a mechanism-rich or endorsement
substrate, use `create_substrate` with `resonator` / composite specs (§2, §4) —
the resulting `dynamic_tpm` matches the original library exactly.

---

## 8. Porting checklist

1. For each old `Unit` / `CompositeUnit`, record its `mechanism` (or sub-units +
   `mechanism_combination`), `inputs`, `params`, and `label`.
2. Build a `node_params` dict: one entry per node, keyed by index.
3. `substrate = create_substrate(node_params, labels=...)`.
4. `system = pyphi.System.from_substrate(substrate, state, nodes)`.
5. Replace every old `compute.*` / `Subsystem` / `Network` call using §6.
6. If reproducing pre-4.0 numbers, wrap analysis in
   `pyphi.config.override(version="IIT_3_0")`.
7. To check fidelity, compare against the original's `dynamic_tpm`:
   `np.asarray(substrate.tpm.to_joint())[..., 1]`.

---

## Reference: source modules

- `pyphi/substrate_generator/__init__.py` — `create_substrate`, `build_tpm`,
  `build_substrate`, `UNIT_FUNCTIONS`.
- `pyphi/substrate_generator/mechanisms.py` — the 16 unit mechanisms,
  `MECHANISMS`, `STATE_DEPENDENT`, `WEIGHTED`.
- `pyphi/substrate_generator/mechanism_combinations.py` — the 6 combination
  strategies, `MECHANISM_COMBINATIONS`, `composite`.
- `pyphi/substrate_generator/ising.py`, `unit_functions.py`, `weights.py` —
  earlier built-in unit functions and weight-matrix helpers.
- `pyphi/substrate.py` — `Substrate`, `complexes`, `maximal_complex`, `all_sias`.
- `pyphi/system.py` — `System`, `from_substrate`, `sia`, `ces`, `distinction`.
- `pyphi/matching/` — `PerceptualSystem`, `TriggeredTPM`, `MatchingAnalysis`.
- `pyphi/examples.py` — `*_substrate` / `*_system` factories to use as anchors.
