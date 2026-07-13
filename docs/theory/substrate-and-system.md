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

# Substrate and system

Everything IIT computes starts from a **substrate**: a set of units and a
complete description of how they influence one another. This page covers the
substrate and the candidate **system** drawn from it, and the two postulates
they enforce — *existence* and *intrinsicality*.

## The substrate is a causal model

A substrate $U = \{U_1, U_2, \ldots, U_n\}$ is $n$ interacting units with a
finite state space $\Omega_U$. Its cause–effect power is captured entirely by
its **transition probability function** — the probability of each next state
$\bar{u}$ given each current state $u$ (Albantakis et al., 2023, Eq. 1):

$$ \mathcal{T}_U \equiv p(\bar{u} \mid u), \qquad u, \bar{u} \in \Omega_U. $$

Because a substrate's units are assumed conditionally independent given the
previous state, this factorizes over units (Eq. 2):

$$ p(\bar{u} \mid u) = \prod_{i=1}^{n} p(\bar{u}_i \mid u). $$

This factorization is the causal-model assumption the whole framework rests on;
see [conditional independence](conditional-independence.md) for what it means
and how PyPhi enforces it.

In PyPhi a substrate is a `Substrate`. The worked example is the paper's
three-unit logistic network:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The repr shows the transition probability matrix (TPM) $\mathcal{T}_U$ in
state-by-node form: one row per current state, one column per unit, giving the
probability that each unit turns on at the next step. The **connectivity
matrix** records which units influence which — the causal wiring that the TPM
quantifies:

```{code-cell} python
substrate.cm  # cm[i, j] == 1 means unit i is an input to unit j
```

The transition probabilities *are* the substrate's cause–effect power: the units
take and make a difference. This is the **existence** postulate, IIT's operational
starting point — to exist is to have cause–effect power (Albantakis et al., 2023).

## A system is an intrinsic point of view

The units we analyze are usually an open subset $S \subseteq U$ of a larger
substrate. The **intrinsicality** postulate requires that a system's cause–effect
power be assessed *from its own perspective*: the remaining units $W = U \setminus S$
are treated as fixed background conditions that do not themselves count as part
of the system. PyPhi enforces this by **causally marginalizing** the background
units — conditioning on their current state and averaging them out, so they
become causally inert (Albantakis et al., 2023, Fig 1B).

A `System` is a candidate subset of a substrate in a definite state. The worked
example's candidate is the pair $\{A, B\}$, so unit $C$ is background:

```{code-cell} python
system = pyphi.System(substrate, (0, 1, 1), node_indices=(0, 1))
system
```

Here `node_indices` selects the subset $S = \{A, B\}$; the remaining unit $C$
is causally marginalized, so the analysis sees $A$ and $B$ from their own
intrinsic perspective, with $C$'s influence averaged out.

From the system's intrinsic point of view, IIT builds two derived transition
matrices — a **cause TPM** and an **effect TPM** — by marginalizing the
background and applying Bayes' rule. They describe how the system's *own* units
constrain each other's past and future. PyPhi exposes them on the system:

```{code-cell} python
type(system.cause_marginal).__name__, type(system.effect_marginal).__name__
```

These are the objects the next steps operate on. With the substrate fixed as a
causal model and the system fixed as an intrinsic point of view, the following
page asks whether the system is irreducible, and by how much: its
[system integrated information](system-integration.md), $\varphi_s$.
