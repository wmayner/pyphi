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

# Explore substrate parameter landscapes

Every quantity PyPhi computes is a function of the substrate's parameters:
change a connection weight and $\varphi_s$ changes with it. `pyphi.landscape`
makes that function inspectable along one parameter axis —
`landscape_section` evaluates the system irreducibility analysis over a grid
of parameter values, and `perturb` estimates local derivatives at a single
point. Both track not just $\varphi_s$ but the identity of every discrete
selection behind it, because the landscape's most important features are the
places where a selection *switches*.

```{code-cell} python
import numpy as np

import pyphi

pyphi.config.progress_bars = False
```

## A parameter axis

A parameter axis is any callable mapping a float to a `Substrate`. For
substrates built from a weight matrix, `weight_axis` varies a single
connection. Here we use the three-unit substrate of Figure 1A of the IIT 4.0
paper (Albantakis et al. 2023), taking its A→B coupling — published value
0.7 — as the axis:

```{code-cell} python
from pyphi.substrate_generator import ising

weights = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
axis = pyphi.weight_axis([ising.probability] * 3, weights, (0, 1), temperature=0.25)
```

## A section through the landscape

`landscape_section` analyzes the substrate at every grid point and returns a
tidy table:

```{code-cell} python
section = pyphi.landscape_section(axis, (1, 0, 0), np.linspace(0.40, 0.75, 15))
section.df[["phi", "signed_phi", "normalized_phi", "partition_margin", "regime"]].round(4)
```

The `regime` column groups grid points by their **selection regime**: the
set of points where the minimum information partition and the specified
cause and effect states are all the same. Within a regime, $\varphi_s$ is a
smooth function of the parameter. At a regime boundary a selection switches,
and the reported $\varphi_s$ can jump — here, the MIP switches near
$\theta \approx 0.45$ and $\varphi_s$ drops discontinuously, even though the
*normalized* value is continuous across that switch. `boundaries` brackets
every switch at the grid's resolution:

```{code-cell} python
section.boundaries
```

The raw analysis objects are kept alongside the table (`section.sias`), so
any point can be inspected in full detail.

## Local derivatives and distance to a switch

`perturb` evaluates three analyses (at $\theta - h$, $\theta$, $\theta + h$)
and reports finite-difference derivatives. The default quantity is
`signed_phi` — the value *before* the positive-part clamp — because the
clamped `phi` is exactly flat wherever the raw integration is negative,
while the signed value still carries gradient information:

```{code-cell} python
result = pyphi.perturb(axis, (1, 0, 0), 0.7)
{
    "value": round(result.value, 6),
    "derivative": round(result.derivative, 4),
    "same_regime": result.same_regime,
}
```

`same_regime` confirms all three evaluation points share one selection
regime, so the derivative describes a smooth stretch of the landscape. When
it is `False`, the estimate straddles a switch: trust only the one-sided
`left_derivative` and `right_derivative`, each on its own side.

The margins reported by the analysis (see
{doc}`Control tie-breaking <tie-breaking>`) are in the selections' own value
units. `perturb` converts them into *parameter* units: `switch_distances`
divides each margin by the rate at which it is shrinking, giving a
first-order estimate of how far the parameter can move before that selection
switches.

```{code-cell} python
{name: round(d, 5) for name, d in result.switch_distances.items()}
```

This substrate — the published Figure 1A system, at its published weights —
sits about `0.0017` away from a specified-cause-state switch in the A→B
weight. Crossing it collapses $\varphi_s$ to zero: the published value is
correct and exactly reproducible, and also close to a boundary where the
substrate specifies a different past state. That distance is invisible in
$\varphi_s$ itself and in its derivative; it is what the margins and
`switch_distances` are for. (The estimate is a linearization; here it agrees
with the bisected switch location to about one percent.)

```{code-cell} python
before = pyphi.analyze(axis(0.700), (1, 0, 0), compute="sia")
after = pyphi.analyze(axis(0.704), (1, 0, 0), compute="sia")
(float(before.phi), tuple(before.system_state.cause.state)), (
    float(after.phi),
    tuple(after.system_state.cause.state),
)
```

## Notes

- Setting a weight to exactly 0 removes the connection from the derived
  connectivity matrix — a discrete topology change. A section whose grid
  crosses 0 will show it as its own (typically reducible) regime.
- Each grid point is one full SIA, so section cost scales linearly with the
  grid and exponentially with the number of units; `perturb` costs exactly
  three analyses.
- For sweeping the discrete axes of a *fixed* substrate — states, candidate
  subsystems, formalisms — see
  {doc}`Sweep states and subsystems <sweep>`.
