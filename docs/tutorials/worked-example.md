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

# A complete worked example

{download}`Download this page as a Jupyter notebook <worked-example.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/worked-example.ipynb)

This page works through a small system from start to finish: a network of three
XOR nodes, analyzed under IIT 4.0. The system is simple enough to reason about
by hand, but rich enough to illustrate every layer of a PyPhi analysis — the
integrated information $\Phi$ of the whole, the distinctions that compose its
cause-effect structure, and the relations between them.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The XOR substrate

The substrate is three fully connected XOR gates, labeled $A$, $B$, and $C$,
with no self-connections. Each node turns ON in the next state exactly when an
odd number of its two inputs are currently ON.

```{code-cell} python
substrate = pyphi.examples.xor_substrate()
substrate
```

The transition probability matrix and connectivity matrix are shown above. Every
transition is deterministic (each next-state probability is 0 or 1), and the
connectivity matrix confirms that every node feeds the other two but not itself.

We will analyze the system in the state where all three nodes are OFF.

```{code-cell} python
state = (0, 0, 0)
```

## Does the whole exist?

According to IIT, existence is a property of the whole system before it is a
property of the parts. The first question is therefore whether the system as a
whole is integrated — whether it specifies an irreducible cause-effect structure
with $\Phi > 0$.

The {func}`pyphi.analyze` function performs the full analysis: it finds the
system's minimum-information partition, builds the cause-effect structure of
distinctions, and computes the relations among them.

```{code-cell} python
analysis = pyphi.analyze(substrate, state)
analysis
```

The system integrates: its structural integrated information is
$\Phi = 1.5$. The whole exists.

```{code-cell} python
analysis.phi
```

## The cause-effect structure

Having established that the whole exists, we can look at what exists *within* it.
The cause-effect structure is the set of **distinctions** — mechanisms that
specify an irreducible cause and an irreducible effect — together with the
relations among them.

```{code-cell} python
ces = analysis.ces
distinctions = ces.distinctions
len(distinctions)
```

There are four distinctions. Three are the second-order mechanisms $AB$, $AC$,
and $BC$; the fourth is the whole-system mechanism $ABC$.

```{code-cell} python
for d in distinctions:
    print(
        f"{d.mechanism_label:>4}  "
        f"φ_d = {float(d.phi):.3f}  "
        f"cause {d.cause.purview}  effect {d.effect.purview}"
    )
```

The three second-order mechanisms each have $\varphi_d = \tfrac{1}{2}$; the
whole-system mechanism has $\varphi_d = 1$. Notice which mechanisms are *absent*:
none of the first-order mechanisms $A$, $B$, or $C$ appears. We return to why
below.

## One distinction up close

By the symmetry of the network, the three second-order distinctions behave
alike, so it is enough to examine one. Take the distinction specified by
mechanism $AB$.

```{code-cell} python
ab = distinctions[0]
ab.mechanism_label, float(ab.phi)
```

Its cause and effect purviews tell us what part of the system it constrains in
each temporal direction.

```{code-cell} python
ab.cause.purview, ab.effect.purview
```

The **cause purview** is the whole system $ABC$ (indices `(0, 1, 2)`), while the
**effect purview** is just node $C$ (index `(2,)`).

The {meth}`~pyphi.models.distinction.Distinction.explain` method summarizes why,
naming the purview and the partition that the distinction is irreducible over.

```{code-cell} python
print(ab.explain())
```

The interpretation is the following. Knowing that $A$ and $B$ are both currently
OFF constrains the *past*: the previous state of the whole system was either all
OFF or all ON, with equal probability. That is why the cause purview is the whole
of $ABC$. Looking *forward*, the mechanism $AB$ completely fixes the next state
of $C$ — because $C$ is the XOR of $A$ and $B$, and both are OFF, so $C$ will be
OFF — which is why the effect purview is exactly $C$. The mechanism says nothing
about the next state of $A$ or $B$ on its own, since those depend on the value of
$C$, so any effect purview larger than $C$ would be reducible.

By symmetry, $AC$ specifies node $B$ as its effect and $BC$ specifies node $A$;
each of the three second-order distinctions locks the next state of the one node
it excludes.

## Intrinsic versus extrinsic existence

The most instructive feature of this example is what does *not* exist. None of
the individual nodes $A$, $B$, or $C$ forms a distinction. We can confirm this
directly: asking the system for the distinction of a single node returns a null
distinction with $\varphi_d = 0$.

```{code-cell} python
system = analysis.system
for mechanism, label in [((0,), "A"), ((1,), "B"), ((2,), "C")]:
    d = system.distinction(mechanism)
    print(f"{label}: φ_d = {float(d.phi):.3f}")
```

This can be surprising. The XOR gates are physical objects sitting on a table; an
observer can touch each one, manipulate it, and watch its causes and effects. But
that is *extrinsic* existence — existence *for* an external observer. What matters
for IIT is *intrinsic* existence: does the mechanism have an irreducible cause
*and* an irreducible effect *within the system itself*?

A mechanism must have both. To see why $A$ fails, compare its irreducible cause
with its irreducible effect:

```{code-cell} python
A = (0,)
mic = system.mic(A)  # maximally irreducible cause
mie = system.mie(A)  # maximally irreducible effect
print(f"cause  φ = {float(mic.phi):.3f}  over purview {mic.purview}")
print(f"effect φ = {float(mie.phi):.3f}  over purview {mie.purview}")
```

Mechanism $A$ *does* have irreducible cause power ($\varphi = 0.5$ over $BC$),
but its effect power is zero. With no self-loop, $A$ cannot affect itself; and
knowing only the current state of $A$ says nothing about the next state of $B$ or
$C$, because each of those is an XOR that also depends on the third node. Since a
distinction's $\varphi_d$ is the smaller of its cause and effect values, and $A$'s
effect value is zero, $A$ specifies no distinction. Having cause power is not
enough — intrinsic existence requires irreducible power in both directions.

## Relations

Distinctions are not the whole story in IIT 4.0. Distinctions whose purviews
overlap bind together into **relations**, and these contribute to $\Phi$
alongside the distinctions themselves. This system has fifteen of them.

```{code-cell} python
len(ces.relations), ces.sum_phi_relations
```

The total structural integrated information combines both levels: the sum of the
distinction $\varphi_d$ values and the sum of the relation $\varphi_r$ values.

```{code-cell} python
ces.sum_phi_distinctions, ces.sum_phi_relations
```

## Summary

For three XOR gates in the all-OFF state, IIT 4.0 finds:

- an integrated whole, with $\Phi = 1.5$;
- four distinctions — the second-order mechanisms $AB$, $AC$, $BC$, each with
  $\varphi_d = \tfrac{1}{2}$, and the whole-system mechanism $ABC$ with
  $\varphi_d = 1$;
- no first-order distinctions, because each single node has irreducible cause
  power but no effect power, and so does not exist intrinsically;
- fifteen relations binding the overlapping distinctions together.
