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

# Distinctions and relations

The system integrated information $\varphi_s$ says *that* a complex exists and
how irreducible it is. The **composition** postulate says *what* it is: subsets
of its units specify their own cause–effect states, and those overlap. Unfolding
this internal structure produces the complex's **distinctions** and **relations**
— the content of the $\Phi$-structure (Albantakis et al., 2023).

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.basic_substrate()
analysis = pyphi.analyze(substrate, (1, 1, 0))
ces = analysis.ces
```

## Distinctions

A subset of the complex's units — a **mechanism** — specifies a cause and an
effect state over subsets of units called **purviews**. A mechanism together
with the cause and effect state it specifies is a **distinction**. Each
distinction is itself required to satisfy the postulates: its cause and effect
purviews are the ones over which its cause–effect power is maximally irreducible,
measured by the distinction's integrated information $\varphi_d$ (Albantakis et
al., 2023).

Only distinctions whose states are *congruent* with the complex's own
cause–effect state belong to it — the specificity required by the information
postulate. For the worked example the complex specifies three distinctions:

```{code-cell} python
[(d.mechanism, d.cause_purview, d.effect_purview, round(float(d.phi), 3))
 for d in ces.distinctions]
```

Each row is a distinction: its mechanism, the cause purview it constrains, the
effect purview it constrains, and its $\varphi_d$. A single distinction carries
the full detail — the mechanism, the two purviews, and the cause and effect
repertoires (the probability distributions over purview states it specifies):

```{code-cell} python
ces.distinctions[1]  # the mechanism-(2,) distinction, φ_d = 0.5
```

The distinctions' $\varphi_d$ values sum to the structure's distinction total:

```{code-cell} python
round(float(ces.sum_phi_distinctions), 3)
```

## Relations

Distinctions do not sit in isolation. When the cause or effect purviews of two
or more distinctions overlap *congruently* — covering the same units in the same
state — that overlap is itself an irreducible fact about the complex, a
**relation**. A relation binds the distinctions it relates over a shared purview
and, like everything else, has an integrated information $\varphi_r$ measuring
its irreducibility (Albantakis et al., 2023).

The worked example has two relations:

```{code-cell} python
[(round(float(r.phi), 3), r.purview) for r in ces.relations]
```

Each is a congruent overlap over a purview (here, over one unit and over a pair),
with its $\varphi_r$. Their values sum to the structure's relation total:

```{code-cell} python
round(float(ces.sum_phi_relations), 3)
```

Together, the three distinctions and the two relations *are* the $\Phi$-structure
of this complex. The [next page](phi-structure.md) assembles them, defines the
structure integrated information $\Phi$ that measures the whole, and gives the
complete map from the paper's symbols to PyPhi's types.
