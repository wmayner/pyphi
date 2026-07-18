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

# Actual causation

{download}`Download this page as a Jupyter notebook <actual-causation.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/actual-causation.ipynb)

This tutorial shows how to use PyPhi to evaluate *actual causation* — what
caused what in a particular observed transition. This is a distinct framework
from the integrated information $\Phi$ of a system in a state; it asks, given
that the system went from one state to the next, which events were the actual
causes and actual effects of which. The formalism is described in

> Albantakis L, Marshall W, Hoel E, Tononi G (2019). What Caused What? A
> Quantitative Account of Actual Causation Using Dynamical Causal Networks.
> *Entropy*, 21 (5), 459. <https://doi.org/10.3390/e21050459>

The tools are in {mod}`pyphi.actual`. Turning off the progress bars keeps the
output readable here; leave them on for real computations.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

from pyphi import actual, Direction
```

Unlike the integrated-information computations elsewhere in the documentation,
actual causation uses its own formalism, configured by default to reproduce the
2019 paper. No special configuration is needed — in particular, the IIT
formalism version (and the 2026 intrinsic-information requirement) does not
affect actual causation.

## The example network

We work through the basic OR-AND network from Figure 1 of the paper. It is a
two-unit substrate: an `OR` gate (unit `0`) and an `AND` gate (unit `1`), each
receiving input from both.

```{code-cell} python
substrate = pyphi.examples.actual_causation_substrate()
substrate
```

The transition probability matrix gives the dynamics — the probability that each
unit turns on given the current state of the system:

```{code-cell} python
substrate.tpm
```

Name the two units for readability:

```{code-cell} python
OR, AND = 0, 1
```

## The transition of interest

We observe the whole substrate at time $t-1$ and again at time $t$, with `OR`
on and `AND` off in both observations:

```{code-cell} python
X = Y = (OR, AND)
X_state = Y_state = (1, 0)
```

The {class}`~pyphi.actual.Transition` object is the core of every actual
causation computation. To build one, pass the substrate, its state at $t-1$ and
at $t$, and the units of interest on the cause side ($t-1$) and the effect side
($t$). The state given must be the state of the entire substrate, not just the
units in the transition. The transition must also satisfy the *realization*
requirement: if the effect-side occurrence has zero probability given the
state at $t-1$, the transition cannot have occurred, and construction raises
{class}`~pyphi.exceptions.TransitionUnreachableError`.

```{code-cell} python
transition = actual.Transition(substrate, X_state, Y_state, X, Y)
transition
```

## Cause and effect repertoires

Cause and effect repertoires can be read off the transition. For example, the
right side of Figure 2B shows how the occurrence $\{OR = 1\}$ at $t-1$
constrains the probability distribution over the purview $\{OR, AND\}$ at $t$:

```{code-cell} python
transition.effect_repertoire((OR,), (OR, AND))
```

Similarly, Figure 2C shows how the occurrence $\{OR, AND = 10\}$ at $t$
constrains the purview $\{OR\}$ at $t-1$:

```{code-cell} python
transition.cause_repertoire((OR, AND), (OR,))
```

```{note}
In all {class}`~pyphi.actual.Transition` methods the constraining occurrence is
passed as the `mechanism` argument and the constrained occurrence is the
`purview` argument, mirroring the terminology used elsewhere in PyPhi.
```

## Cause and effect ratios

The transition also computes cause and effect ratios — the log-ratio by which an
occurrence raises or lowers the probability of a purview state. The effect ratio
of $\{OR = 1\}$ at $t-1$ constraining $\{OR\}$ at $t$ (Figure 3A) is positive:

```{code-cell} python
round(transition.effect_ratio((OR,), (OR,)), 6)
```

The effect ratio of $\{OR = 1\}$ constraining $\{AND\}$ is negative — the
occurrence makes the observed `AND` outcome *less* likely:

```{code-cell} python
round(transition.effect_ratio((OR,), (AND,)), 6)
```

And the cause ratio of $\{OR = 1\}$ at $t$ constraining $\{OR, AND\}$ at $t-1$
(Figure 3B) is:

```{code-cell} python
round(transition.cause_ratio((OR,), (OR, AND)), 6)
```

## Finding the minimum information partition

To find the irreducible cause or effect ratio $\alpha$ of a particular pair of
occurrences (Figure 3C), use {meth}`~pyphi.actual.Transition.find_mip`. Consider
the candidate effect link $\{OR, AND\} \rightarrow \{OR, AND\}$:

```{code-cell} python
link = transition.find_mip(Direction.EFFECT, (OR, AND), (OR, AND))
```

This returns an object describing the minimum information partition. This
particular link is reducible, as its $\alpha$ is zero:

```{code-cell} python
round(link.alpha, 6)
```

The `partition` attribute shows the partition that reduces it:

```{code-cell} python
link.partition
```

The candidate *cause* link $\{OR, AND\} \rightarrow \{OR, AND\}$ (Figure 3D), by
contrast, is irreducible, with positive $\alpha$:

```{code-cell} python
cause_link = transition.find_mip(Direction.CAUSE, (OR, AND), (OR, AND))
round(cause_link.alpha, 6)
```

To find the actual cause or actual effect of an occurrence — the maximally
irreducible link over it — use
{meth}`~pyphi.actual.Transition.find_actual_cause` or
{meth}`~pyphi.actual.Transition.find_actual_effect`:

```{code-cell} python
transition.find_actual_cause((OR, AND))
```

## Accounts

The complete causal account of the transition — every irreducible cause and
effect link — is computed with {func}`~pyphi.actual.account`:

```{code-cell} python
account = actual.account(transition)
account
```

These are the causal links shown in Figure 4. The account behaves like a
sequence of links:

```{code-cell} python
len(account)
```

## Irreducible accounts

Whether the account of the transition is itself irreducible — whether the
transition, taken as a whole, is more than the sum of its parts — is evaluated
with {func}`~pyphi.actual.sia` (system irreducibility analysis):

```{code-cell} python
sia = actual.sia(transition)
round(sia.alpha, 6)
```

As shown in Figure 4, the second-order occurrence $\{OR, AND = 10\}$ is
destroyed by the minimum information partition, so the partitioned account has
only the four first-order links:

```{code-cell} python
sia.partitioned_account
```

The partition that achieves this is available on the result:

```{code-cell} python
sia.partition
```

## The causal nexus

The analysis so far fixed one transition of interest. To search over all
possible transitions between the observed states and keep the irreducible ones,
use {func}`~pyphi.actual.nexus`. Each result is a system irreducibility analysis
whose cause and effect units identify the transition it describes:

```{code-cell} python
labels = substrate.node_labels
for candidate in actual.nexus(substrate, X_state, Y_state):
    cause = ",".join(labels[i] for i in candidate.cause_indices)
    effect = ",".join(labels[i] for i in candidate.effect_indices)
    print(f"[{cause}] -> [{effect}]   big_alpha = {round(candidate.alpha, 6)}")
```

{func}`~pyphi.actual.causal_nexus` returns the single maximally irreducible
transition among these:

```{code-cell} python
cn = actual.causal_nexus(substrate, X_state, Y_state)
round(cn.alpha, 6)
```

```{code-cell} python
cause = ",".join(labels[i] for i in cn.cause_indices)
effect = ",".join(labels[i] for i in cn.effect_indices)
print(f"[{cause}] -> [{effect}]")
```

The single-unit transition $\{OR\} \rightarrow \{OR\}$ is the causal nexus of
this observed state change.
