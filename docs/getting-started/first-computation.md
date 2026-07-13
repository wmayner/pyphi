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

# Your first computation

{download}`Download this page as a Jupyter notebook <first-computation.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/getting-started/first-computation.ipynb)

This page walks through a complete PyPhi computation end to end: build a small
substrate from the IIT 4.0 paper, analyze it in a chosen state, read off the
integrated information $\varphi_s$ and the associated $\Phi$-structure, and
save the result to disk. It should take about ten minutes.

If you have not installed PyPhi yet:

```
pip install pyphi
```

Start by importing the package. Turning off the progress bars keeps the output
readable here; leave them on for real computations, where they are helpful.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## Build a substrate

A {class}`~pyphi.substrate.Substrate` is a set of interacting units defined by a
transition probability matrix and a connectivity matrix. PyPhi ships the
example systems used in the IIT literature; we will use the three-unit network
the IIT 4.0 paper introduces the theory with (Albantakis et al. 2023, Fig 1A) —
three units `A`, `B`, and `C`, each a noisy logistic function of its inputs.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The connectivity matrix shows which units influence which, and the transition
probability matrix gives the probability that each unit turns on given the
current state of the system. Note the probabilities are strictly between 0
and 1: the network is probabilistic, not deterministic.

## Analyze a state

Integrated information is a property of a substrate *in a particular state*. We
use the state analyzed in the paper — `A` off, `B` and `C` on — ordered
`(A, B, C)`, and hand it, together with the substrate, to
{func}`~pyphi.analyze`.

```{code-cell} python
state = (0, 1, 1)
analysis = pyphi.analyze(substrate, state)
analysis
```

That single call runs the whole analysis of the three-unit system: it measures
the system's irreducibility, finds its distinctions, and computes their
relations.

## Read the results

The scalar `analysis.phi` is $\varphi_s$, the system integrated information —
how much the system, as a whole, is irreducible to its parts.

```{code-cell} python
round(analysis.phi, 4)
```

The value is positive: the three units hang together as one system. (It
reproduces the value published in the paper's Fig 1E for this system, 0.13.)

## Find the complexes

Not every subset of units exists as a whole of its own. Subsets *compete*:
among overlapping candidates, only the one with maximal $\varphi_s$ — a
**complex** — exists. {meth}`~pyphi.substrate.Substrate.complexes` runs that
competition over every subset:

```{code-cell} python
for complex_ in substrate.complexes(state):
    print(complex_.node_indices, round(float(complex_.phi), 4))
```

The substrate condenses into two complexes: the single unit `C`, and the pair
`{A, B}` — the complex the paper features in Fig 1E, written "aB". Every other
candidate, including the full three-unit system we just analyzed, is excluded
by one of these two. This is IIT's exclusion postulate in action: $\varphi_s >
0$ makes a candidate *eligible*; being a local maximum among everything it
overlaps makes it a complex.

## The Φ-structure

The richer object is the $\Phi$-structure, available as `analysis.ces` (a
{class}`~pyphi.models.ces.CauseEffectStructure`). It is the collection of
*distinctions* — the irreducible mechanisms the system specifies — together
with the *relations* among them.

```{code-cell} python
ces = analysis.ces
print("distinctions:", len(ces.distinctions))
print("relations:   ", len(ces.relations))
```

Its `big_phi` attribute is the structure integrated information $\Phi$, the
summed $\varphi$ of the distinctions and relations.

```{code-cell} python
round(float(ces.big_phi), 4)
```

## Save the result

Analyses can be expensive, so it is worth saving them. {func}`~pyphi.save`
writes any PyPhi result object to JSON, and {func}`~pyphi.load` reads it back.

```{code-cell} python
pyphi.save(ces, "ces.json")
```

## Where to go next

That is a full PyPhi computation. From here:

- The {doc}`worked example <../tutorials/worked-example>` follows this same
  network through the paper's Figures 1, 2, and 4, reproducing the published
  numbers.
- The theory page {doc}`../theory/overview` explains what these quantities mean
  and how they are defined.
- Deterministic networks have $\varphi_s = 0$ under the default formalism —
  see {doc}`../theory/intrinsic-information` for why, and for how to
  reproduce values published under earlier formulations.
