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

# Computing a cause-effect structure

{download}`Download this page as a Jupyter notebook <cause-effect-structure.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/cause-effect-structure.ipynb)

A cause-effect structure (CES) is what IIT 4.0 unfolds from a system in a
state: the set of irreducible **distinctions** and the **relations** between
them. This page is a hands-on tour of those objects — how to reach them from a
single analysis, what each one carries, and how they add up to $\Phi$. For the
theory behind why these are the right objects, see
{doc}`../theory/distinctions-and-relations`.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## One analysis, everything downstream

Everything on this page comes from a single call. We analyze the three-node
`basic_substrate` in state $(1, 1, 0)$; the result exposes the cause-effect
structure at `.ces`.

```{code-cell} python
analysis = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0))
ces = analysis.ces
ces
```

## Distinctions

Each distinction is a mechanism (a subset of nodes) whose cause-effect
repertoire is irreducible. Iterate `ces.distinctions` to see them. Every
distinction carries its `.mechanism`, its integrated information `.phi`
($\varphi_d$), and the `.cause_purview` and `.effect_purview` it specifies.

```{code-cell} python
for distinction in ces.distinctions:
    print(
        f"mechanism={distinction.mechanism}  "
        f"φ_d={distinction.phi:.4f}  "
        f"cause={distinction.cause_purview}  "
        f"effect={distinction.effect_purview}"
    )
```

Node indices label the mechanism and purviews. A single distinction prints a
labelled summary of both sides:

```{code-cell} python
ces.distinctions[1]
```

### The cause repertoire

A distinction's `.cause_repertoire` is the probability the mechanism assigns to
its specified cause state. `basic_substrate` is deterministic, so each
distinction pins its cause down completely and the value is $1$:

```{code-cell} python
distinction = ces.distinctions[1]
print("mechanism:", distinction.mechanism)
print("cause purview:", distinction.cause_purview)
print("cause repertoire:", distinction.cause_repertoire)
```

The full distribution over the purview's states lives on the system object. Here
is the cause repertoire that mechanism `A` specifies over purview `(B, C)` — a
genuine distribution over the four states of two binary nodes, laid out on a
$2 \times 2$ grid indexed by (B, C):

```{code-cell} python
analysis.system.cause_repertoire((0,), (1, 2)).squeeze().round(3)
```

## Relations

A relation binds distinctions that specify overlapping purviews; its `.phi`
($\varphi_r$) is the integrated information over the shared `.purview`.

```{code-cell} python
for relation in ces.relations:
    print(f"φ_r={relation.phi:.4f}  purview={set(relation.purview)}")
```

## The whole structure as a table

`ces.to_pandas()` collapses the distinctions into a data frame — handy for
sorting, filtering, or exporting.

```{code-cell} python
ces.to_pandas()
```

## Putting it together: $\Phi$

The structure's $\Phi$ is the sum of every distinction's $\varphi_d$ and every
relation's $\varphi_r$:

$$\Phi = \sum_d \varphi_d + \sum_r \varphi_r$$

```{code-cell} python
sum_distinctions = sum(d.phi for d in ces.distinctions)
sum_relations = sum(r.phi for r in ces.relations)

print(f"Σ φ_d = {sum_distinctions:.6f}")
print(f"Σ φ_r = {sum_relations:.6f}")
print(f"total = {sum_distinctions + sum_relations:.6f}")
print(f"ces.big_phi = {ces.big_phi:.6f}")
```

The two agree: `ces.big_phi` is exactly that sum, the integrated information of
the whole cause-effect structure.

This is a different quantity from the $\Phi$ shown at the top of the CES repr
above. That value is the system integrated information $\Phi_s$ (equal to
`analysis.phi`), computed as the loss across the system's minimum-information
partition. The sum here, $\sum_d \varphi_d + \sum_r \varphi_r$, is the total
small-phi of the structure's constituents. Both are reported, and they are not
the same number.

## Where to go next

- {doc}`../theory/distinctions-and-relations` — why distinctions and relations
  are the constituents of a cause-effect structure.
- {doc}`../theory/phi-structure` — how $\Phi$ over the structure relates to the
  system's integrated information.
