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

The assumption follows from what a substrate is taken to be. A substrate is a
complete causal model: it contains every variable that influences its units,
so any dependence between them can be traced to a state inside the model rather
than to something left out (Albantakis et al., 2019). Time in the model is
explicit and discrete, and units act on one another only from one step to the
next, never within a step. Its transition probabilities are defined by
intervention: each state is imposed with the do-operator and the next state
observed, so the units are physical in the sense that each can be observed and
manipulated on its own (Albantakis et al., 2023). Put together, once the whole
previous state is fixed there is nothing left for two units' next states to
share. Any residual correlation between them would have to come from a common
influence within the same step, which a complete model rules out. The joint
transition therefore factors into one term per unit.

This is what lets a substrate be described by per-unit transition probabilities
(a state-by-node transition probability matrix) rather than by joint transitions,
and it is what allows a partition to be applied to individual connections: noising
one unit's inputs changes that unit's factor and leaves the others as they were.
The assumption is the second equation of the IIT 4.0 formalism (Albantakis et
al., 2023, Eq. 2), and PyPhi requires it of every substrate.

## PyPhi enforces it

A transition probability matrix that violates conditional independence describes
*instantaneous causality* — units influencing each other within a single step —
which indicates a missing exogenous variable. PyPhi rejects such a matrix. Consider
two units that stay put when they agree and flip with probability one-half when
they disagree; their joint (state-by-state) transitions are not conditionally
independent:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

pyphi.examples.cond_depend_tpm()
```

Building a {class}`~pyphi.substrate.Substrate` from it raises an error rather than silently accepting an
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
missing variable explicitly. {func}`~pyphi.examples.cond_independ_tpm` does this for
the same two units, introducing a third unit whose state decides whether they
flip; the resulting three-unit substrate satisfies conditional independence and
is accepted. Deterministic transitions are always conditionally independent, so
deterministic substrates never run into this constraint.

## References

- Albantakis L, Marshall W, Hoel E, Tononi G (2019). What caused what? A
  quantitative account of actual causation using dynamical causal networks.
  *Entropy* 21(5): 459.
- Albantakis L, Barbosa L, Findlay G, Grasso M, et al. (2023). Integrated
  information theory (IIT) 4.0. *PLOS Computational Biology* 19(10): e1011465.
