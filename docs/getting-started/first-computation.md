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
substrate, analyze it in a chosen state, read off the integrated information
$\varphi_s$ and the associated $\Phi$-structure, and save the result to disk.
It should take about ten minutes.

If you have not installed PyPhi yet:

```
pip install pyphi
```

Start by importing the package. Turning off the progress bars keeps the output
clean here; leave them on for real computations, where they are helpful.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## Build a substrate

A {class}`~pyphi.substrate.Substrate` is a set of interacting units defined by a
transition probability matrix and a connectivity matrix. PyPhi ships a few small
example systems; we will use the standard three-unit network that appears
throughout the documentation.

```{code-cell} python
substrate = pyphi.examples.basic_substrate()
substrate
```

The three units are named `A`, `B`, and `C`. The connectivity matrix shows which
units influence which, and the transition probability matrix gives the
probability that each unit turns on given the current state of the system.

## Analyze a state

Integrated information is a property of a substrate *in a particular state*. We
pick a state — one value per unit, ordered `(A, B, C)` — and hand it, together
with the substrate, to {func}`~pyphi.analyze`.

```{code-cell} python
state = (1, 0, 0)
analysis = pyphi.analyze(substrate, state)
analysis
```

That single call runs the whole analysis: it identifies the maximally
irreducible system, its distinctions, and their relations.

## Read the results

The scalar `analysis.phi` is $\varphi_s$, the integrated information of the
system as a whole — how much the system is more than the sum of its parts.

```{code-cell} python
round(analysis.phi, 4)
```

The richer object is the $\Phi$-structure, available as `analysis.ces` (a
{class}`~pyphi.models.ces.CauseEffectStructure`). It is the collection of
*distinctions* — the irreducible mechanisms the system specifies — together with
the *relations* among them.

```{code-cell} python
ces = analysis.ces
print("distinctions:", len(ces.distinctions))
print("relations:   ", len(ces.relations))
```

Its `big_phi` attribute is $\sum \varphi_d$, the total small-$\varphi$ summed
over the distinctions.

```{code-cell} python
ces.big_phi
```

## Save the result

Analyses can be expensive, so it is worth saving them. {func}`~pyphi.save`
writes any PyPhi result object to JSON, and {func}`~pyphi.load` reads it back.

```{code-cell} python
pyphi.save(ces, "ces.json")
```

## Where to go next

That is a full PyPhi computation. From here:

- The {doc}`tutorials <../tutorials/index>` build up larger systems and walk
  through cause-effect structures, macro analysis, and actual causation in
  depth.
- The theory page {doc}`../theory/overview` explains what these quantities mean
  and how they are defined.
