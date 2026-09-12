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

# Build a substrate

A {class}`~pyphi.substrate.Substrate` is a set of units and the probability
of each unit's next state given the current state of all of them. This page
shows the four ways to make one: from a transition probability matrix you
already have, from a weight matrix with logistic units, from a function per
unit, and from recorded transitions. It ends with how to check the result
before analyzing it.

```{code-cell} python
import numpy as np
import pyphi

pyphi.config.progress_bars = False
```

## From a transition probability matrix

The usual input is **state-by-node** form: one row per current state, one
column per unit, each entry the probability that the unit is ON at the next
step. Rows are ordered so that the **first unit changes fastest** (PyPhi's
little-endian convention): for three units the rows are the states
`(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)`.

```{code-cell} python
tpm = np.array([
    [0.1, 0.1, 0.1],   # current state (0, 0, 0)
    [0.1, 0.9, 0.1],   # (1, 0, 0)
    [0.9, 0.1, 0.9],   # (0, 1, 0)
    [0.9, 0.9, 0.1],   # (1, 1, 0)
    [0.1, 0.1, 0.9],   # (0, 0, 1)
    [0.9, 0.1, 0.9],   # (1, 0, 1)
    [0.9, 0.9, 0.9],   # (0, 1, 1)
    [0.1, 0.9, 0.1],   # (1, 1, 1)
])
substrate = pyphi.Substrate(tpm, node_labels=("A", "B", "C"))
substrate
```

A state-by-state matrix (one row and one column per state) is accepted too
and converted; the {ref}`TPM conventions <tpm-conventions>` page describes
every accepted form and the row order in full.

Pass a connectivity matrix (`cm=`; `cm[i, j] = 1` when unit `i` is an input
to unit `j`) only when you know the wiring. Without one PyPhi assumes every
unit may influence every other, which is always correct and only slower. A
wrong connectivity matrix gives a wrong result, not a slow one.

Check one transition you know before trusting anything computed from the
matrix. The card above prints the matrix with its row states, and the
matrix's table view can be indexed by state:

```{code-cell} python
substrate.tpm.to_pandas().loc[(1, 0, 0)]  # the row for current state (1, 0, 0)
```

## From a weight matrix and logistic units

The networks of the IIT 4.0 papers are logistic (sigmoid) units of their
weighted inputs. Give {func}`~pyphi.substrate_generator.build_substrate` a
weight matrix, where `w[i, j]` is the weight from unit `i` to unit `j`, and
the unit function by name. `determinism` is the slope `k` of the logistic
function in Albantakis et al. (2023, Eq. 60) and Marshall et al. (2023,
Eq. 2); inputs enter as ±1.

```{code-cell} python
from pyphi.substrate_generator import build_substrate

w = np.array([
    [0.2, 0.4, 0.1, 0.3],
    [0.3, 0.2, 0.4, 0.1],
    [0.1, 0.3, 0.2, 0.4],
    [0.4, 0.1, 0.3, 0.2],
])
logistic = build_substrate("sigmoid", w, determinism=3.0, node_labels=("A", "B", "C", "D"))
logistic
```

The connectivity matrix is read from the nonzero weights. This is the
construction behind every `iit4_2023_*` and `marshall_2023_*` example in
{mod}`pyphi.examples`; open one of them to see the weights of a published
network.

## From a function per unit

Logic gates and other named mechanisms go through
{func}`~pyphi.substrate_generator.create_substrate`, one specification per
unit: the mechanism's name, its inputs, and any parameters.

```{code-cell} python
from pyphi.substrate_generator import create_substrate

gates = create_substrate(
    [
        {"mechanism": "or", "inputs": (1, 2)},
        {"mechanism": "and", "inputs": (0, 2)},
        {"mechanism": "xor", "inputs": (0, 1)},
    ],
    labels=("A", "B", "C"),
)
gates
```

The mechanism names are the keys of `pyphi.substrate_generator.MECHANISMS`.
A unit you write yourself is a function `f(element, weights, state, **params)`
returning the probability that `element` is ON at the next step; pass it (or
a list mixing functions and names, one per unit) to
{func}`~pyphi.substrate_generator.build_substrate` with a weight matrix.

## From recorded transitions

{func}`pyphi.estimate_substrate` fits a posterior over transition
probabilities to observed `(current, next)` state pairs. IIT's analysis is
defined on an *interventional* matrix, what each unit does when the system
is put into each state, so say which regime produced the data:
`"perturbational"` when states were set, `"observational"` when they were
only recorded.

```{code-cell} python
rng = np.random.default_rng(0)
current = rng.integers(0, 2, size=(200, 3))
# The row of ``tpm`` for each sampled state: little-endian, so the first
# unit is the least significant bit.
rows = np.ravel_multi_index(current.T[::-1], (2, 2, 2))
next_state = (rng.random((200, 3)) < tpm[rows]).astype(int)

posterior = pyphi.estimate_substrate(
    (current, next_state), regime="perturbational", node_labels=("A", "B", "C")
)
posterior.mean_substrate()
```

The posterior's mean substrate is a reference point, not an estimate of
φ: analyzing it mixes what is unknown about the matrix with the substrate's
own indeterminism. Sample substrates from the posterior to carry that
uncertainty through an analysis; see {ref}`the what's-new tour
<whats-new-estimate>`.

## Units with more than two states

Pass `alphabet=` (one size for every unit) or `state_space=` (a tuple of
state labels per unit). The number of rows is then the product of the
alphabet sizes, still with the first unit changing fastest.

## Check the substrate before analyzing it

- `substrate` prints the units, the connectivity, and the matrix with its row
  states; a wrong row order shows up here as a transition you did not
  intend.
- PyPhi rejects a matrix whose units are not conditionally independent
  given the previous state (a hidden common cause); see
  {doc}`../theory/conditional-independence`.
- A state the substrate cannot reach is refused at analysis time
  ({doc}`FAQ <faq>`).
- Before a large run, count the work: {doc}`estimate-cost`.

## Where to go next

- {doc}`Getting started <../getting-started/index>` analyzes a substrate end to end.
- {doc}`Read a result <read-result>` explains what comes back.
- {doc}`../theory/substrate-and-system` gives the theory behind the matrix.
