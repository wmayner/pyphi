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

(conditional-independence)=

# Conditional independence

The whole framework rests on one assumption about the causal model, introduced
on the [substrate page](substrate-and-system.md): the units of a substrate are
**conditionally independent** given the previous state. Each unit's next state
depends only on the previous state of the substrate, not on what the other units
happen to do at the same step:

$$ p(\bar{u} \mid u) = \prod_{i=1}^{n} p(\bar{u}_i \mid u). $$

This is what lets a substrate be described by per-unit transition probabilities
(a state-by-node transition probability matrix) rather than by joint transitions.
It is the second equation of the IIT 4.0 formalism (Albantakis et al., 2023,
Eq. 2), and PyPhi requires it of every substrate.

## PyPhi enforces it

A transition probability matrix that violates conditional independence describes
*instantaneous causality* — units influencing each other within a single step —
which signals a missing exogenous variable. PyPhi rejects such a matrix. Consider
two units that stay put when they agree and flip with probability one-half when
they disagree; their joint (state-by-state) transitions are not conditionally
independent:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

pyphi.examples.cond_depend_tpm()
```

Building a `Substrate` from it raises an error rather than silently accepting an
ill-defined causal model:

```{code-cell} python
try:
    pyphi.Substrate(pyphi.examples.cond_depend_tpm())
except pyphi.exceptions.ConditionallyDependentError as error:
    print(error)
```

## Recovering an independent representation

Every state-by-node matrix corresponds to a unique conditionally independent
state-by-state matrix. Converting a conditionally dependent matrix to state-by-node
form and back reveals the independent representation PyPhi would assume — the two
units become independent, each flipping with probability one-half:

```{code-cell} python
from pyphi import convert

sbn = convert.state_by_state2state_by_node(pyphi.examples.cond_depend_tpm())
convert.state_by_node2state_by_state(sbn)
```

The dependence in the original matrix can always be restored by adding the
missing variable explicitly. `pyphi.examples.cond_independ_tpm()` does this for
the same two units, introducing a third unit whose state decides whether they
flip; the resulting three-unit substrate satisfies conditional independence and
is accepted. Deterministic transitions are always conditionally independent, so
deterministic substrates never run into this constraint.
