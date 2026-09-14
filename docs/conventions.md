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

(tpm-conventions)=
# Transition probability matrix conventions

A {class}`~pyphi.substrate.Substrate` is built from a transition probability
matrix (TPM) in one of the forms below. Whatever the input form, PyPhi stores
the matrix in **factored form**, one conditional distribution per unit, as a
{class}`~pyphi.core.tpm.factored.FactoredTPM` on `substrate.tpm`; the joint
matrix is available on demand from `substrate.joint_tpm()`. This page defines
the forms, the row order they share, and the connectivity matrix.

```{code-cell} python
import numpy as np
import pyphi

pyphi.config.progress_bars = False
```

(state-by-node-form)=
## State-by-node form

The usual input. Entry $(i, j)$ is the probability that unit $j$ is ON at
$t+1$ given that the substrate is in state $i$ at $t$: one row per state,
one column per unit. For binary units only; multi-valued units use the
factored form below.

```{code-cell} python
tpm = np.array([
    [0.1, 0.2],   # current state (0, 0)
    [0.3, 0.4],   # (1, 0)
    [0.5, 0.6],   # (0, 1)
    [0.7, 0.8],   # (1, 1)
])
substrate = pyphi.Substrate(tpm, node_labels=("A", "B"))
substrate.tpm.to_pandas()
```

The table view is indexed by the current state, so one row can be read
directly:

```{code-cell} python
substrate.tpm.to_pandas().loc[(1, 0)]
```

(multidimensional-state-by-node-form)=
## Multidimensional state-by-node form

The same matrix reshaped to $n + 1$ dimensions: the first $n$ index the
current state of each unit and the last holds the probabilities of each
unit being ON next. A state then indexes the array directly. Accepted as
input, and the shape of the per-unit factors PyPhi stores:

```{code-cell} python
factor_a = substrate.tpm.factor(0)  # unit A's conditional distribution
factor_a.shape, factor_a[(1, 0)]     # P(A = 0), P(A = 1) after state (1, 0)
```

(state-by-state-form)=
## State-by-state form

Entry $(i, j)$ is the probability that the substrate moves from state $i$
to state $j$: one row and one column per state, both in the row order
below. Accepted as input for binary units and converted to the factored
form.

```{warning}
A state-by-state matrix can encode dependencies between units' next states
that no state-by-node or factored matrix can, so the conversion loses
information unless the matrix is conditionally independent, which is the
case PyPhi requires; see {ref}`conditional-independence`. A matrix that is
not is rejected with a
{class}`~pyphi.exceptions.ConditionallyDependentError`.
```

## Factored form and multi-valued units

The stored form can also be given directly: `marginals=`, a sequence of
one array per unit with shape `(*alphabet_sizes, k_i)`, where $k_i$ is the
number of states of unit $i$. This is the only input form for units with
more than two states, together with `alphabet=` (one size for every unit) or
`state_space=` (a tuple of state labels per unit). The example below has a
ternary unit and a binary unit:

```{code-cell} python
alphabet = (3, 2)
rng = np.random.default_rng(0)
marginals = []
for k in alphabet:
    factor = rng.random((*alphabet, k))
    marginals.append(factor / factor.sum(axis=-1, keepdims=True))
ternary = pyphi.Substrate(
    marginals=marginals, state_space=((0, 1, 2), (0, 1)), node_labels=("P", "M")
)
ternary.tpm.state_space
```

`state_space` lists each unit's states; `alphabet=3` would instead give
every unit three states. Multi-valued states are written in result cards
with the state as a subscript, `P₂`.

```{tip}
{mod}`pyphi.convert` converts between the joint forms
({func}`~pyphi.convert.state_by_state2state_by_node`,
{func}`~pyphi.convert.state_by_node2state_by_state`,
{func}`~pyphi.convert.to_multidimensional`,
{func}`~pyphi.convert.to_2dimensional`) and between states and row indices
({func}`~pyphi.convert.state2le_index`,
{func}`~pyphi.convert.le_index2state`).
```

(little-endian-convention)=
## Little-endian convention

Every form above orders states the same way, and a state-by-state matrix
orders its columns the same way too. Of the two possible orders, either the
first unit changes state fastest:

| State at $t$ (A, B) | Pr(A = ON) at $t+1$ | Pr(B = ON) at $t+1$ |
| --- | --- | --- |
| (0, 0) | 0.1 | 0.2 |
| (1, 0) | 0.3 | 0.4 |
| (0, 1) | 0.5 | 0.6 |
| (1, 1) | 0.7 | 0.8 |

or the last unit does:

| State at $t$ (A, B) | Pr(A = ON) at $t+1$ | Pr(B = ON) at $t+1$ |
| --- | --- | --- |
| (0, 0) | 0.1 | 0.2 |
| (0, 1) | 0.5 | 0.6 |
| (1, 0) | 0.3 | 0.4 |
| (1, 1) | 0.7 | 0.8 |

A row index encodes a state in binary, one bit per unit. **PyPhi always uses
the first order: the state of the first unit (the lowest index) varies
fastest**, so the least significant bit gives the state of the lowest-index
unit. This is the little-endian convention of computer memory; the other is
big-endian. With multi-valued units the same rule holds with each unit's
alphabet size in place of 2: the first unit's state still varies fastest.

```{code-cell} python
from pyphi.convert import le_index2state, state2le_index

state2le_index((1, 0)), le_index2state(2, 2)
```

The little-endian mapping is stable under changes in the number of units: the
same bit always corresponds to the same unit index.

```{note}
This applies only where an integer index encodes a state. A state written
as a tuple uses the only sensible convention: the $i$-th element is the
state of the $i$-th unit.
```

(cm-conventions)=
## Connectivity matrix conventions

If $CM$ is a connectivity matrix, $CM_{i,j} = 1$ means there is a directed
connection from unit $i$ to unit $j$, and $CM_{i,j} = 0$ means there is
none. For example, this substrate of four units

```{image} _static/connectivity-matrix-example-network.png
:width: 150px
```

has the connectivity matrix

```{code-cell} python
cm = np.array([
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
])
```

Without a connectivity matrix PyPhi assumes every unit may influence every
other, including itself, which is always correct and only slower. A matrix
that omits a connection the TPM implies is rejected (see
`validate_connectivity` in the {doc}`configuration reference
</reference/configuration>`); declaring an unused connection is allowed.

## Where to go next

- {doc}`/howto/build-substrate` builds a substrate from a matrix in any of
  these forms.
- {doc}`/reference/glossary` defines the terms used here.
