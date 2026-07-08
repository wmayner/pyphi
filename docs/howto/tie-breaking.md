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

# Control tie-breaking

At several points in an IIT analysis, PyPhi compares $\varphi$ values to pick a
single winner: the candidate state of a mechanism, the minimum-information
partition of a mechanism, the purview that a mechanism specifies, and the system
partition that defines $\varphi_s$. When two or more candidates are equal up to the
configured numerical `precision`, the comparison is a **tie**, and PyPhi must
choose one. This page shows how to control which one it chooses.

Ties are not rare curiosities. Deterministic systems with symmetric inputs and
outputs are especially prone to them, and different tie-resolution rules can
change *which* object is returned — and therefore the reported purviews and, in
some cases, the composition of the cause-effect structure. The scalar $\varphi$
and $\varphi_s$ values are unaffected by tie-breaking (all tied candidates share
the same value by definition); what changes is which representative object
carries that value forward.

For background on the role of ties in the formalism, see Krohn & Ostwald
(2017), Moon (2019), and Hanson & Walker (2021).

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The four tie-resolution options

Tie resolution is governed by four options under
`pyphi.config.formalism.iit`, one for each place a tie can arise:

```{code-cell} python
iit = pyphi.config.formalism.iit
{
    "state_tie_resolution": iit.state_tie_resolution,
    "mip_tie_resolution": iit.mip_tie_resolution,
    "purview_tie_resolution": iit.purview_tie_resolution,
    "sia_tie_resolution": iit.sia_tie_resolution,
}
```

| Option | Resolves ties between | Selects |
| --- | --- | --- |
| `state_tie_resolution` | candidate states of a mechanism | maximum |
| `mip_tie_resolution` | mechanism partitions (the MIP) | minimum |
| `purview_tie_resolution` | purviews a mechanism could specify | maximum |
| `sia_tie_resolution` | system partitions (the system MIP) | minimum |

Each value is a **strategy**: either a single strategy name (a string) or a
list of names forming a *cascade*. A cascade is applied left to right — the
first strategy narrows the field, the next breaks any remaining ties among the
survivors, and so on. The default for `sia_tie_resolution`, for example,
prefers larger normalized $\varphi$, then smaller raw $\varphi$, then a
lexicographic ordering of the partition as a final deterministic tiebreak:

```{code-cell} python
pyphi.config.formalism.iit.sia_tie_resolution
```

## Available strategies

The following strategy names can appear in any of the four options:

| Name | Comparison key |
| --- | --- |
| `PHI` | raw $\varphi$ |
| `NEGATIVE_PHI` | raw $\varphi$, preferring the smaller value |
| `NORMALIZED_PHI` | $\varphi$ normalized by the number of cut connections |
| `NEGATIVE_NORMALIZED_PHI` | normalized $\varphi$, preferring the smaller value |
| `PURVIEW_SIZE` | number of units in the purview (prefer larger) |
| `NEGATIVE_PURVIEW_SIZE` | number of units in the purview (prefer smaller) |
| `PARTITION_LEX` | lexicographic ordering of the partition |
| `NONE` | leave ties unresolved (no preference applied) |

```{code-cell} python
from pyphi.resolve_ties import phi_object_tie_resolution_strategies

sorted(phi_object_tie_resolution_strategies)
```

When a cascade does not fully resolve a tie, PyPhi falls back to a stable choice:
the first candidate encountered in its fixed evaluation order (purviews and
partitions are enumerated in a deterministic, lexicographic-by-index order). The
choice is therefore always reproducible for a given configuration, even when no
strategy strictly prefers one candidate over another.

## A worked example: a purview tie

The basic example system has a genuine tie in the effect purview of one of its
mechanisms. Mechanism `C` (index `2`) specifies its effect over two different
purviews that are exactly tied at $\varphi = 1$: the single unit `(1,)` and the
pair `(0, 1)`. Which one PyPhi reports depends entirely on
`purview_tie_resolution`.

```{code-cell} python
system = pyphi.examples.basic_system()
system.state, list(system.node_labels)
```

Under the default (`"PHI"` alone), the two purviews are tied on $\varphi$ and
nothing further distinguishes them, so PyPhi returns the first in its evaluation
order:

```{code-cell} python
mie = system.mie((2,))
mie.purview, round(mie.phi, 6)
```

Adding `PURVIEW_SIZE` to the cascade breaks the tie in favor of the larger
purview:

```{code-cell} python
with pyphi.config.override(purview_tie_resolution=["PHI", "PURVIEW_SIZE"]):
    mie_large = system.mie((2,))

mie_large.purview, round(mie_large.phi, 6)
```

Using `NEGATIVE_PURVIEW_SIZE` instead prefers the smaller purview:

```{code-cell} python
with pyphi.config.override(purview_tie_resolution=["PHI", "NEGATIVE_PURVIEW_SIZE"]):
    mie_small = system.mie((2,))

mie_small.purview, round(mie_small.phi, 6)
```

The $\varphi$ value is `1.0` in every case; only the selected purview changes.
This is the sense in which tie-breaking can affect a result: it determines the
representative object, not the integrated-information value it carries.

## Changing the rules

Set a tie-resolution option globally like any other configuration value. The
flat write is routed to the `formalism.iit` layer automatically:

```{code-cell} python
pyphi.config.purview_tie_resolution = ["PHI", "PURVIEW_SIZE"]
pyphi.config.formalism.iit.purview_tie_resolution
```

```{code-cell} python
# restore the default
pyphi.config.purview_tie_resolution = "PHI"
pyphi.config.formalism.iit.purview_tie_resolution
```

For a temporary change scoped to a single computation, prefer
`config.override`, as in the example above — it restores the previous value
even if the block raises.

## Interaction with precision and presets

Whether two candidates count as tied is decided up to `numerics.precision`
decimal places. Coarser precision produces more ties; finer precision produces
fewer. The EMD distance measure used by the IIT 3.0 formalism is particularly
prone to $\varphi$ ties across purviews and, because of its numerical optimizer,
runs at a lower default precision than the intrinsic-difference measures.

The built-in presets set tie-resolution options to match their respective
papers. Applying the `iit3` preset, for instance, switches
`purview_tie_resolution` to the two-step cascade `["PHI", "PURVIEW_SIZE"]` used
by PyPhi 1.x, and sets the MIP and system-partition options to break remaining
ties lexicographically:

```{code-cell} python
import warnings

from pyphi import iit3

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with pyphi.config.override(**iit3):
        print("purview:", pyphi.config.formalism.iit.purview_tie_resolution)
        print("mip:    ", pyphi.config.formalism.iit.mip_tie_resolution)
        print("sia:    ", pyphi.config.formalism.iit.sia_tie_resolution)
```

For the full list of configuration options, see the
{doc}`configuration reference </configuration>`; for how to read, set, and load
configuration in general, see {doc}`Configure PyPhi <configure>`.
