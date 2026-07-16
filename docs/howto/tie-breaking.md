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

In theoretical research on IIT, we often use deterministic systems with
symmetric inputs and outputs. These are especially prone to ties, and different
tie-resolution rules can change which object is returned — and therefore the
reported purviews and, in some cases, the composition of the cause-effect
structure. The scalar $\varphi$ and $\varphi_s$ values are unaffected by
tie-breaking (all tied candidates share the same value by definition); what
changes is which representative object is reported.

```{note}
In contrast to the examples often used in the literature, realistic systems are often noisy and lack exact symmetry, so ties are less important in empirical practice.
```

The PyPhi authors have been aware of ties since this project was conceived, but
earlier versions did not implement a principled method for breaking them. For
others' treatment of ties in the formalism, see Krohn & Ostwald (2017), Moon
(2019), and Hanson & Walker (2021).

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

The basic example system has an exact tie in the effect purview of one of its
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
representative object, not its integrated-information value.

## System-state ties: the postulate ladder

The specified cause and effect states of a *system* are selected by maximal
intrinsic information (Eq. 12), and symmetric substrates routinely tie. These
ties are not resolved by a configurable cascade: they follow the postulate
ladder of the S1 tie-resolution supplement (Albantakis et al. 2023, S1),
applied with each formalism's own $\varphi_s$ — under IIT 4.0 (2026), the
value including the intrinsic-information term.

1. **Integration.** Each tied $(\text{cause}, \text{effect})$ reading gets
   its own MIP search, and the readings are compared on $\varphi_s$. A unique
   maximum wins.
2. **Composition.** Readings still tied at $\varphi_s > 0$ are compared on
   the structure integrated information $\Phi$ of the cause–effect structure
   each reading resolves (the mechanism sweep is shared; each reading pays
   congruence resolution plus the closed-form relation sum). The
   $\Phi$-maximal reading wins — "it is the integrated information
   $\varphi_s$ that determines in which cause-effect state the system exists
   the most", and past $\varphi_s$, the structure does.
3. **A $\Phi$ tie** is *extrinsic* when the tied readings' structures are
   intrinsically identical (isomorphic up to a relabeling of the units): the
   system still qualifies, PyPhi reports a canonical representative, and the
   full tied set is surfaced on ``sia.ties``. A $\Phi$ tie between genuinely
   *distinct* structures violates the information postulate: the system does
   not qualify as a complex, and the SIA is null with reason
   ``NONUNIQUE_SYSTEM_STATE``.

When every tied reading has $\varphi_s = 0$ — under the 2026 default this is
every deterministic system — the system is not a complex under any reading,
so nothing remains for $\Phi$ to adjudicate. The ladder stops: no
cause–effect structure is computed, and the reported state is a canonical,
relabeling-invariant representative whose choice is presentational.

A noisy XOR loop shows the Composition step live. At $(0, 0, 0)$ its two
specified cause readings tie at a positive $\varphi_s$, and the
congruent reading supports more relation structure, so $\Phi$ selects it:

```{code-cell} python
import numpy as np

p, n = 0.85, 3
tpm = np.zeros((2**n, n))
for i, s in enumerate(pyphi.utils.all_states((2,) * n)):
    for j in range(n):
        tpm[i, j] = p if (sum(s) - s[j]) % 2 == 1 else 1 - p
noisy_xor = pyphi.Substrate(tpm, cm=np.ones((n, n), dtype=int) - np.eye(n, dtype=int))

sia = pyphi.System(noisy_xor, (0, 0, 0)).sia()
{
    "phi": round(float(sia.phi), 6),
    "chosen cause state": sia.system_state.cause.state,
    "tied readings": len(sia.ties),
}
```

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

## Selection margins: how close was the call?

A tie is the limiting case of a more general question: *by how much* did the
winning candidate win? Every selection that can tie also reports its
**margin** — the gap between the winner and the best competitor, in the units
of the selection's own comparison key. A margin of zero is an exact tie; a
small margin means the analysis is near a boundary where a different
candidate would win, so the reported objects are sensitive to small changes
in the substrate.

At the system level, the SIA reports the margin of the system-partition
selection (in normalized $\varphi$) and of the specified cause and effect
states (in intrinsic information), along with which selections are
effectively tied at the configured `precision`:

```{code-cell} python
sia = pyphi.examples.basic_system().sia()
{
    "partition_margin": round(float(sia.partition_margin), 6),
    "cause_state_margin": round(float(sia.state_margins[pyphi.Direction.CAUSE]), 6),
    "effect_state_margin": round(float(sia.state_margins[pyphi.Direction.EFFECT]), 6),
    "tied_selections": sia.tied_selections,
    "effectively_tied": sia.effectively_tied,
}
```

A symmetric substrate produces an exact tie: the two best system partitions
of the 3-unit grid are symmetry-related and tie exactly in normalized
$\varphi$:

```{code-cell} python
grid_sia = pyphi.examples.grid3_system().sia()
grid_sia.tied_selections, float(grid_sia.partition_margin)
```

Mechanism-level analyses report margins the same way. The purview tie from
the worked example above is a purview margin of exactly zero:

```{code-cell} python
mie = system.mie((2,))
float(mie.purview_margin), mie.effectively_tied
```

Margins also appear in `explain()` findings and, at `FULL` repr verbosity
(`pyphi.config.repr_verbosity = 3`), as rows on the result cards:

```{code-cell} python
[
    (finding.label, finding.value)
    for finding in grid_sia.explain().findings
    if "margin" in finding.kind or finding.kind == "effectively_tied"
]
```

One caveat: when a partition sweep stops early because it detected
reducibility ($\varphi_s = 0$), no exact margin exists and `partition_margin`
is `None`. Setting `shortcircuit_sia=False` evaluates every partition, which
makes margins exact everywhere (at the cost of exhaustive evaluation on
reducible systems) without changing any computed $\varphi$ value. The same
applies at the mechanism level: when `shortcircuit_distinctions` (on by
default) detects that a distinction is reducible, the remaining MICE search is
skipped, so the skipped direction carries no margins or ties and reports the
`OTHER_DIRECTION_REDUCIBLE` reason instead. Set it to `False` to evaluate both
directions in full.

To move from margins in value units to distances in *parameter* units — how
far a connection weight can move before a selection switches — see
{doc}`Explore substrate parameter landscapes <landscape>`.

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

For the full list of configuration options, see the configuration classes in
the {doc}`API reference </reference/index>`; for how to read, set, and load
configuration in general, see {doc}`Configure PyPhi <configure>`.
