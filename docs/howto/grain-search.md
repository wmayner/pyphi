---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Search across grains

A substrate's cause-effect power need not be maximal at the grain of its
smallest parts. The intrinsic-units search asks which units — at which spatial
grouping and over which temporal window — are the ones that actually exist for
a substrate in a given state. The search is combinatorial: it builds candidate
macro units, assembles them into candidate systems, and compares those systems
in an exclusion cascade. Because the number of candidates grows quickly, the
first step of any real analysis is to pre-flight its cost, not to launch it
blind. For the theory behind macro units and grains, see
{doc}`../theory/macro-units`; for a guided walkthrough, see the
{doc}`intrinsic-units tutorial <../tutorials/macro>`.

```{code-cell} python
import numpy as np

import pyphi
from pyphi import config
from pyphi.conf import presets
from pyphi.macro.search import SearchBounds
from pyphi.substrate import Substrate

pyphi.config.progress_bars = False
```

The demos use a two-unit substrate. Each unit is nearly silent on its own and
strongly ON only when both are already ON. The rows are the little-endian
states `00, 10, 01, 11`; the columns give each unit's probability of being ON
at the next update:

```{code-cell} python
tpm = np.array(
    [
        [0.05, 0.05],
        [0.05, 0.06],
        [0.06, 0.05],
        [0.95, 0.95],
    ]
)
substrate = Substrate(tpm, node_labels=("A", "B"))
```

Every computation below runs under the configuration preset that reproduces
the settings of Marshall et al. (2024). (The pin also matters for the
numbers: several specimens here are deterministic or near-deterministic, and
the 2026 default's intrinsic-information cap would zero them — see
{doc}`../theory/intrinsic-information`.)

## Pre-flight the cost

{meth}`SearchBounds.estimate <pyphi.macro.search.SearchBounds.estimate>`
counts the candidate systems a set of bounds would visit, without constructing
a single macro TPM or computing any φₛ. Run it before the search itself:

```{code-cell} python
bounds = SearchBounds()
estimate = bounds.estimate(substrate)
estimate.distinct_systems_upper_bound
```

The number of distinct candidate systems is the count to compare against the
search's own record of what it evaluated. Three flags qualify it:

- `is_exact` is `True` only when the enumeration is not an upper bound but the
  exact number of candidates. That happens only at `max_depth=0`, where no
  macro grains are built and the count is a plain enumeration. With any macro
  levels the count is a worst case, so `is_exact` is `False`.
- `truncated` is `True` if the counting itself hit its internal `limit` and
  stopped early, in which case the reported counts are lower bounds of the true
  bound.
- `partitions_capped` is `True` if some candidate has more macro units than the
  partition-count cap, so the partition-sweep total covers only the counted
  candidates.

```{code-cell} python
estimate.is_exact, estimate.truncated, estimate.partitions_capped
```

Every count is a worst case that assumes each candidate decomposition passes
the intrinsic-unit criteria. The search can only do less work than the
estimate, never more.

## Run the search

Pass `grains=True` to {func}`pyphi.analyze <pyphi.analyze>` to run the search
with the default bounds:

```{code-cell} python
with config.override(**presets.iit4_2023):
    result = pyphi.analyze(substrate, (0, 0), grains=True)
round(result.maximal_complex.phi, 6)
```

`grains=True` is shorthand for `grains=SearchBounds()`. To control the search,
pass a configured `SearchBounds` instead of `True`. The underlying driver is
{func}`pyphi.macro.complexes <pyphi.macro.search.complexes>`, which
`analyze` calls with the substrate, state, and bounds; call it directly when
you want the driver without the rest of the `analyze` interface.

## Supply a micro history for temporal grains

A macro unit at update grain τ reads its constituents over τ micro updates, so
its state is defined by a window of past states, not a single one. As soon as
the bounds admit any grain above 1, the search needs a micro history rather
than a bare state. Passing a bare state raises:

```{code-cell} python
with config.override(**presets.iit4_2023):
    try:
        pyphi.analyze(substrate, (0, 0), grains=SearchBounds(max_update_grain=2))
    except ValueError as error:
        print(error)
```

Supply the history as a sequence of universe states, oldest first. Its
required length is `max_update_grain ** max_depth` — here `2 ** 1 == 2`:

```{code-cell} python
with config.override(**presets.iit4_2023):
    temporal = pyphi.analyze(
        substrate, [(0, 0), (0, 0)], grains=SearchBounds(max_update_grain=2)
    )
len(temporal.complexes)
```

## Read the result

{func}`pyphi.analyze <pyphi.analyze>` with `grains` returns a
`ComplexesResult`. Its `complexes` are the winners of the exclusion cascade,
each a `Complex` carrying its micro footprint (`node_indices`), its φₛ, and its
macro units. Iterate them to see which units won and at which grain (each
unit's `update_grain` is its temporal window):

```{code-cell} python
for complex_ in result.complexes:
    grains = tuple(unit.update_grain for unit in complex_.units)
    print(complex_.node_indices, round(complex_.phi, 6), grains)
```

Every winner reports its `exclusion_margin` — how far it finished ahead of the
best rival it beat, in φₛ — and whether that margin is small enough to
count as an effective tie at the configured precision:

```{code-cell} python
winner = result.maximal_complex
round(winner.exclusion_margin, 6), winner.effectively_tied
```

Each winner also carries the candidates it excluded, in `excluded`. Here the
winner is a coarse-graining of both units, and its strongest beaten rival is a
*blackboxing* of the same two units:

```{code-cell} python
rival = max(winner.excluded, key=lambda candidate: candidate.phi)
rival.node_indices, round(rival.phi, 6)
```

In a condensation with several complexes, `excluded` can also hold *shadows* —
candidates whose own φₛ is *higher* than the complex they appear under, kept out
not by it but by a different complex that carved their footprint away. There is
a single complex here, so no shadows arise. Exclusion is recursive: an excluded
candidate cannot in turn exclude anything. For how the cascade resolves
overlapping candidates, see the
{doc}`recursive-exclusion tutorial <../tutorials/recursive-exclusion>`.

`records` holds every system the search actually evaluated, so its length is
the realized version of the pre-flight estimate. Here they match: the worst
case of eight candidate systems was reached exactly.

```{code-cell} python
len(result.records), estimate.distinct_systems_upper_bound
```

`ties` holds cliques of overlapping candidates that stayed tied even through Φ
escalation and so failed exclusion — none of them is a complex. It is empty
here:

```{code-cell} python
result.ties
```

## Parallelize

The search evaluates candidate systems independently, so it parallelizes over
them. Forward a `parallel_kwargs` dictionary through `analyze` and it reaches
the candidate sweep:

```python
result = pyphi.analyze(
    substrate,
    (0, 0),
    grains=True,
    parallel_kwargs={"parallel": True, "chunksize": 4},
)
```

For the parallel backend and its options, see {doc}`parallel`.

## Bound the search

`SearchBounds` exposes the knobs that set the size of the search space. Each
tightens one axis of the combinatorial cost:

| Knob | Bounds | Cost it drives |
| --- | --- | --- |
| `max_constituents` | units per candidate macro unit | how large a group may be coarse-grained or blackboxed |
| `max_update_grain` | largest temporal grain τ per level | the temporal window, and the required micro-history length |
| `max_depth` | macro levels above the micro grain | how far the hierarchy is built; `0` disables macroing |
| `mappings` | `"FAMILIES"` or `"EXHAUSTIVE"` | which candidate state mappings are enumerated per unit shape |
| `exhaustive_cap` | sequence-state count allowed under `"EXHAUSTIVE"` | guards the doubly-exponential mapping count |
| `apportionment` / `max_background` | assigning background micro units to derived candidates | extra candidates per derived unit when enumerating |

For how these axes combine into the overall cost, and how to read a pre-flight
estimate against it, see {doc}`../theory/computational-complexity`.
