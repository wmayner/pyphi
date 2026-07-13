---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Intrinsic units: analyzing systems at a macro grain

{download}`Download this page as a Jupyter notebook <macro.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/macro.ipynb)

A system's cause-effect power need not be maximal at the grain of its
smallest parts. The intrinsic-units framework of Marshall, Findlay,
Albantakis, and Tononi (2024) extends the IIT analysis to *macro units*:
groups of micro units, possibly evaluated over several micro updates, whose
joint state is read out by an explicit state mapping. PyPhi implements the
full framework in {mod}`pyphi.macro`: the macro TPM construction, the
intrinsic-unit criteria, and a bounded search that answers "which units, and
which grain, are intrinsic for this substrate in this state?"

This tutorial walks through the authors' *minimal* example, then lets the
search rediscover the coarse-graining example from the paper.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

Throughout we use the configuration preset that reproduces the paper's
settings. (The pin also matters for the numbers: several specimens here are
deterministic or near-deterministic, so under the 2026 default's
intrinsic-information requirement they would compute $\varphi_s = 0$ — see
{doc}`../theory/intrinsic-information`.)

```{code-cell} python
import numpy as np

from pyphi import config
from pyphi.conf import presets
```

## A minimal example, at the micro grain

The substrate has two units, `A` and `B`. Each is nearly silent on its own,
weakly noisy, and strongly ON only when both are already ON (the rows are the
little-endian states `00, 10, 01, 11`; the columns give each unit's
probability of being ON at the next update):

```{code-cell} python
from pyphi.substrate import Substrate
from pyphi.system import System

tpm = np.array(
    [
        [0.05, 0.05],
        [0.05, 0.06],
        [0.06, 0.05],
        [0.95, 0.95],
    ]
)
substrate = Substrate(tpm, node_labels=("A", "B"))
state = (0, 0)
```

At the micro grain the system is barely integrated:

```{code-cell} python
with config.override(**presets.iit4_2023):
    micro_phi = System(substrate, state).sia().phi
round(micro_phi, 6)
```

## Defining a macro unit

A macro unit is specified by its direct constituents, an update grain, and a
*mapping*: a truth table over the constituents' joint sequence-states that
says when the macro unit counts as ON. The helper
{func}`pyphi.macro.coarse_grain` builds mappings from ON-counts; here, "the
group is ON exactly when both constituents are ON":

```{code-cell} python
from pyphi.macro import MacroUnit, coarse_grain

coarse_grain(2, on_counts={2})
```

```{code-cell} python
alpha = MacroUnit(
    constituents=(0, 1),
    update_grain=1,
    mapping=coarse_grain(2, on_counts={2}),
)
```

{func}`pyphi.macro.blackbox` builds the other common family, where the macro
state reads out designated *output* constituents at the final update of a
window — for example `blackbox(2, update_grain=1, output_constituents=(0,))`
is the table `(0, 1, 0, 1)`. Update grains above 1 evaluate the unit over a
sliding window of several micro updates; see {mod}`pyphi.macro.units`.

```{code-cell} python
from pyphi.macro import blackbox

blackbox(2, update_grain=1, output_constituents=(0,))
```

## Analyzing the macro system

{meth}`pyphi.macro.MacroSystem.from_micro` builds the macro cause and effect
TPMs by the paper's four-step construction and yields an object the IIT
pipeline consumes exactly like a {class}`~pyphi.system.System`:

```{code-cell} python
from pyphi.macro import MacroSystem

macro = MacroSystem.from_micro(substrate, (alpha,), state)
macro.state
```

```{code-cell} python
with config.override(**presets.iit4_2023):
    macro_phi = macro.sia().phi
round(macro_phi, 6)
```

Macroing raised the system's integrated information by two orders of
magnitude — the framework's central phenomenon.

## Is the macro unit intrinsic?

A grouping does not automatically count as one unit. A candidate is an
*intrinsic unit* only if its constituent system is integrated (Eq. 15) and
strictly more irreducible than every competing system that could be built
within its footprint (Eq. 16). {func}`pyphi.macro.is_intrinsic_unit` returns
a verdict with the evidence:

```{code-cell} python
from pyphi.macro import is_intrinsic_unit

with config.override(**presets.iit4_2023):
    verdict = is_intrinsic_unit(substrate, alpha, state)

verdict.valid, round(verdict.phi, 6), verdict.num_competitors
```

The candidate's two competitors (the single-unit systems over `A` and over
`B`) both have $\varphi_s = 0$, so the pair wins. The verdict depends only on
the candidate's constituents and background apportionment — not on its
mapping or grain — so every variant of the same grouping, whatever its
mapping, shares it. Micro units themselves are exempt (they are the base
case of the recursion), even when their own $\varphi_s$ is zero:

```{code-cell} python
from pyphi.macro import micro_unit

with config.override(**presets.iit4_2023):
    verdict = is_intrinsic_unit(substrate, micro_unit(0), state)

verdict.valid, verdict.phi
```

## Searching across grains

{func}`pyphi.macro.complexes` is the one-call driver: it derives every
intrinsic unit within the search bounds, assembles every admissible system of
them (Eq. 18), evaluates each over the full universe, and condenses the
candidates into the *complexes* by the recursive exclusion cascade — accept
the φₛ-maximal candidate, exclude everything overlapping it, and continue on
the remainder (Eq. 19 applied tier by tier; a candidate excluded by an
accepted complex has no standing to exclude others). The same search is
also available as `pyphi.analyze(substrate, state,
grains=True)`. The winners are
{class}`~pyphi.models.complex.Complex` objects, returned together with the
full evaluation record:

```{code-cell} python
from pyphi.macro import SearchBounds, complexes

with config.override(**presets.iit4_2023):
    result = complexes(substrate, state, SearchBounds(mappings="EXHAUSTIVE"))

len(result.complexes)
```

```{code-cell} python
winner = result.complexes[0]
winner.units
```

The search, given every possible 2-constituent mapping, finds exactly the
both-ON coarse-graining we built by hand. The winner reports its own φₛ and
the record holds every evaluated system:

```{code-cell} python
round(float(winner.phi), 6), len(result.records)
```

Candidate mappings are enumerated up to state-label complementation (a
mapping and its complement describe the same physical unit, with the two
macro state labels swapped), and ties at the configured precision are
respected: overlapping candidates that tie on φₛ escalate to Φ
(Composition), and a clique that still ties fails exclusion — none of its
members is a complex, the clique is reported in `result.ties`, and the
cascade continues past it.

## Rediscovering the paper's coarse-graining example

Example 1 of the paper is a four-unit substrate built from two interacting
pairs. Its micro system has $\varphi_s$ of about $0.02$, but the authors show
that coarse-graining each pair into a both-ON macro unit yields a two-unit
macro system with $\varphi_s$ of about $1.004$. The default search bounds
(one macroing level, update grain 1, the coarse-graining and black-boxing
mapping families) recover that analysis from scratch:

```{code-cell} python
tpm4 = np.array(
    [
        [0.05, 0.05, 0.05, 0.05],
        [0.06, 0.15, 0.05, 0.05],
        [0.15, 0.06, 0.05, 0.05],
        [0.16, 0.16, 0.85, 0.85],
        [0.05, 0.05, 0.06, 0.15],
        [0.06, 0.15, 0.06, 0.15],
        [0.15, 0.06, 0.06, 0.15],
        [0.16, 0.16, 0.86, 0.95],
        [0.05, 0.05, 0.15, 0.06],
        [0.06, 0.15, 0.15, 0.06],
        [0.15, 0.06, 0.15, 0.06],
        [0.16, 0.16, 0.95, 0.86],
        [0.85, 0.85, 0.16, 0.16],
        [0.86, 0.95, 0.16, 0.16],
        [0.95, 0.86, 0.16, 0.16],
        [0.96, 0.96, 0.96, 0.96],
    ]
)
substrate4 = Substrate(tpm4, node_labels=("A", "B", "C", "D"))
```

```{code-cell} python
with config.override(**presets.iit4_2023):
    result = complexes(substrate4, (0, 0, 0, 0))  # a few seconds

len(result.complexes)
```

```{code-cell} python
for unit in result.complexes[0].units:
    print(unit.constituents, unit.mapping)
```

The unique complex is exactly the paper's macro system: both-ON
coarse-grainings over `(A, B)` and `(C, D)`.

## Blackboxing

Coarse-graining and blackboxing are the two mapping families the default
search enumerates, and they read a group's joint state in opposite ways.
Coarse-graining pools the constituents by how many are ON, so every joint
state with the same ON-count collapses to the same macro state. Blackboxing
instead reads out a chosen subset of *output* constituents and discards the
rest. Only blackboxing extends to update grains above one: reading a
designated output at the final micro update of a window is exactly what a unit
spanning several updates does, so the temporal search in the next section is
built on this family.

The two helpers show the difference. Each returns a mapping — a truth
table over the group's joint states — for a two-constituent group:

```{code-cell} python
print("coarse_grain(2, (0, 2)):", coarse_grain(2, (0, 2)))
print("blackbox(2, 1, (0,)):   ", blackbox(2, 1, (0,)))
```

The coarse-graining `(1, 0, 0, 1)` is ON for the joint states `00` and `11`
(ON-counts 0 and 2) and OFF for `10` and `01`; it reports whether the two
constituents agree. The blackboxing `(0, 1, 0, 1)` copies constituent `0` and
ignores constituent `1`: its macro state is ON exactly when `A` is ON,
whatever `B` does.

A blackboxed unit is analyzed by the same pipeline as a coarse-grained one.
Here is the blackboxing of the tutorial's two-unit substrate that reads out
`A`, built into a `MacroUnit` and analyzed exactly as `alpha` was:

```{code-cell} python
boxed = MacroUnit(
    constituents=(0, 1),
    update_grain=1,
    mapping=blackbox(2, 1, (0,)),
)
boxed_macro = MacroSystem.from_micro(substrate, (boxed,), state)
with config.override(**presets.iit4_2023):
    boxed_phi = boxed_macro.sia().phi

boxed_macro.state, round(float(boxed_phi), 6)
```

## Temporal grains

A macro unit may exist over several micro updates as well as over several
micro units. A unit with update grain τ = 2 has a state defined by a mapping
over two-step sequences of its constituents, so searching a substrate for such
units requires a micro history of two states — the two most recent universe
states, oldest first — rather than a single current state.

The following substrate is one where a temporal unit wins. It is a three-unit
deterministic system given as a function table: entry `i` is the index of the
state the system moves to from state `i`, with states written in little-endian
order (`A` is bit 0). A short loop turns the table into an 8 × 3
state-by-node TPM:

```{code-cell} python
# state index -> next-state index; little-endian, A = bit 0
fn_table = [2, 3, 4, 3, 0, 0, 5, 0]

tpm = np.zeros((8, 3))
for state_index, next_index in enumerate(fn_table):
    for bit in range(3):
        tpm[state_index, bit] = (next_index >> bit) & 1

substrate = Substrate(tpm, node_labels=("A", "B", "C"))
history = [(0, 0, 1), (0, 0, 0)]
```

Searching with `max_update_grain=2` lets the driver build units that span two
micro updates. Each complex reports its footprint, its φₛ, and the update
grain of each of its units:

```{code-cell} python
with config.override(**presets.iit4_2023):
    result = pyphi.analyze(
        substrate, history, grains=SearchBounds(max_update_grain=2)
    )

for complex_ in result.complexes:
    grains = [unit.micro_grain for unit in complex_.units]
    print(complex_.node_indices, f"{float(complex_.phi):.4f}", grains)
```

Both complexes are temporal: each is a single update-grain-2 unit — `{A}` with
φₛ ≈ 0.5083 and `{B, C}` with φₛ ≈ 0.4630. The winner is the temporal unit
over `A`:

```{code-cell} python
winner = result.maximal_complex
winner.units[0], round(winner.exclusion_margin, 6)
```

Its `exclusion_margin` is how far it beats the strongest overlapping
alternative. The competition was real: the whole substrate
`{A, B, C}` evaluated at micro time (every unit at update grain 1) is itself
integrated, and it appears in the winner's `excluded` record:

```{code-cell} python
micro_universe = max(
    (
        candidate
        for candidate in winner.excluded
        if candidate.node_indices == (0, 1, 2)
        and all(unit.update_grain == 1 for unit in candidate.units)
    ),
    key=lambda candidate: candidate.phi,
)
print(micro_universe.node_indices, round(micro_universe.phi, 4))
```

The full micro universe reaches φₛ ≈ 0.2075, yet the temporal unit over `A`
alone reaches 0.5083. That unit reads `A` every second step, and that view of
the substrate has more than twice the integrated information of the whole
system read update by update.

Temporal wins are not automatic. On a symmetric substrate — a deterministic
rotation, say — the integration criterion rejects a pair decomposition before
any temporal variant of it is ever built, so no unit longer than one update
survives. Asymmetric substrates are what make temporal units win outright: a
seeded random search over such substrates found temporal complexes in roughly
one run in five. The mixed-grain case occurs too, where a single complex holds
units of the same system at different grains.

## Bounding the search

The space of groupings, mappings, and grains grows combinatorially, so the
search is explicitly bounded by {class}`pyphi.macro.SearchBounds`:

- `max_constituents` — cap on a unit's micro footprint (default 4);
- `max_update_grain` — largest update grain per level (default 1; set 2+ to
  search macroing over updates);
- `max_depth` — macroing levels above micro (default 1; higher levels build
  units out of already-validated meso units);
- `mappings` — `"FAMILIES"` (coarse-grainings and black-boxings, the default)
  or `"EXHAUSTIVE"` (every surjective table, capped by `exhaustive_cap`
  sequence-states);
- `apportionment` / `max_background` — opt-in enumeration of background
  apportionments (Eq. 12/29).

Every $\varphi_s$ evaluation in a driver run is memoized, and
`result.records` exposes all of them, so the derivation is fully inspectable:
{func}`pyphi.macro.intrinsic_units` returns the unit pool with one verdict
per judged decomposition, and {func}`pyphi.macro.valid_systems` the
admissible-system set.

## Parallelism

A search reduces to many independent `MacroSystem.sia()` evaluations, which
can run across processes. Enable it by turning on the global switch and the
search's own per-site option, or pass `parallel_kwargs` to a driver for a
one-off:

```python
with config.override(
    parallel=True,
    parallel_macro_system_evaluation={"parallel": True},
):
    result = complexes(substrate4, (0, 0, 0, 0))
```

Results are identical to a sequential run — same complexes, ties, and records
in the same order. Within a worker the per-evaluation `sia` runs sequentially
(search-level parallelism is one process-pool deep); for a single very large
evaluation, leave the search-level option off and let the partition-level
parallelism inside `sia` do the work instead. The benefit grows with the
number of comparably sized evaluations — large substrates and `EXHAUSTIVE`
sweeps — and is slight for small searches dominated by one expensive system.
