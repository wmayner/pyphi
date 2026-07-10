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

# Recursive exclusion: how complexes carve a substrate

{download}`Download this page as a Jupyter notebook <recursive-exclusion.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/recursive-exclusion.ipynb)

The exclusion postulate says that candidate systems sharing units cannot
both exist: among overlapping candidates, only one specifies its
cause–effect structure. PyPhi applies the postulate *recursively*
(Marshall, Albantakis, and Tononi, 2023): candidates are walked in
descending order of system integrated information $\varphi_s$, each
accepted complex claims its units, and — crucially — a candidate excluded
by an accepted complex no longer exists, so it cannot exclude anything
else in turn.

This recursion has a consequence that surprises many readers: **a complex
can coexist with an overlapping candidate of higher $\varphi_s$**, as long
as that candidate was itself excluded by some other complex. This tutorial
builds the smallest substrate where that happens, finds its complexes, and
reads the exclusion records and selection margins that document who beat
whom.

```{code-cell} python
import numpy as np

import pyphi
from pyphi.conf import presets

pyphi.config.progress_bars = False
```

## A chain of decaying couplings

Four units A, B, C, D in a chain, with reciprocal coupling strengths that
decay along it: A–B couple strongly (0.6), B–C moderately (0.3), C–D weakly
(0.15). Each unit's probability of turning on is a baseline 0.05, plus 0.05
times its own state, plus the coupled inputs.

```{code-cell} python
n = 4
weights = np.zeros((n, n))
weights[0, 1] = weights[1, 0] = 0.6
weights[1, 2] = weights[2, 1] = 0.3
weights[2, 3] = weights[3, 2] = 0.15
for i in range(n):
    weights[i, i] = 0.05

tpm = np.zeros((2**n, n))
for row in range(2**n):
    state = np.array([(row >> k) & 1 for k in range(n)])
    tpm[row] = 0.05 + weights @ state

substrate = pyphi.Substrate(tpm, node_labels=("A", "B", "C", "D"))
state = (0, 0, 0, 0)
```

By construction the $\varphi_s$ landscape is a chain:
$\{A,B\} > \{B,C\} > \{C,D\}$, with $\{B,C\}$ overlapping both of its
neighbors.

## Finding the complexes

```{code-cell} python
with pyphi.config.override(**presets.iit4_2023):
    found = substrate.complexes(state)

for complex_ in found:
    print(complex_.node_indices, float(complex_.phi))
```

Both $\{A,B\}$ *and* $\{C,D\}$ are complexes. A non-recursive reading of
exclusion would reject $\{C,D\}$: it overlaps $\{B,C\}$, which has higher
$\varphi_s$. But $\{B,C\}$ overlaps $\{A,B\}$, which has higher
$\varphi_s$ still — so $\{B,C\}$ is excluded first, and once excluded it
does not exist and has no standing to exclude $\{C,D\}$. The recursion
carves the substrate from the top down, and $\{C,D\}$ is the maximum among
the candidates that remain.

## Shadows: excluded candidates with higher φₛ

Each complex records the overlapping candidates excluded in its favor:

```{code-cell} python
cd = found[1]
for record in sorted(cd.excluded, key=lambda r: -r.phi):
    marker = "shadow" if record.phi > float(cd.phi) else "beaten"
    print(f"{record.node_indices}  φₛ={record.phi:.4f}  [{marker}]")
```

$\{C,D\}$'s records contain candidates with **higher** $\varphi_s$ than
$\{C,D\}$ itself — $\{B,C\}$ among them. These are *shadows*: candidates
that out-inform the complex but were carved away by a different complex
before this one was accepted. They document the recursion at work; they
were never rivals that $\{C,D\}$ had to beat.

## Selection margins

How decisively did each complex win? The `exclusion_margin` of a
{class}`~pyphi.models.complex.Complex` reports the $\varphi_s$ gap to the
best overlapping rival the complex actually beat — shadows do not enter
the margin:

```{code-cell} python
for complex_ in found:
    print(
        complex_.node_indices,
        f"φₛ={float(complex_.phi):.4f}",
        f"margin={complex_.exclusion_margin:.4f}",
    )
```

$\{A,B\}$ won by a wide margin over the runner-up on its units. $\{C,D\}$
beat only the singletons $\{C\}$ and $\{D\}$, and its margin measures the
gap to the best of them. A margin of zero (equivalently,
`complex_.effectively_tied`) would mean an overlapping rival tied at the
configured precision and the selection was decided beyond $\varphi_s$; see
{doc}`../howto/tie-breaking` for how ties are resolved.

## References

- Marshall W, Albantakis L, Tononi G (2023). System integrated information.
  *Entropy* 25(2):334, Algorithm A1.
- Albantakis L et al. (2023). Integrated information theory (IIT) 4.0.
  *PLoS Computational Biology* 19(10):e1011465 (the exclusion postulate).
