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

This page follows the worked example of the IIT 4.0 paper (Albantakis et al.
2023) from start to finish, reproducing its published numbers under PyPhi's
default formalism: **Figure 1** — is a set of units a complex, and how
irreducible is it ($\varphi_s$)? **Figure 2** — what distinctions compose its
cause-effect structure? **Figure 4** — how do those distinctions bind into
relations? One small network carries all three:
{func}`pyphi.examples.iit4_2023_fig1a_substrate`.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The substrate: three logistic units

Fig 1A defines three units $A$, $B$, $C$. Each unit's probability of turning
ON is a logistic function (slope $k = 4$) of its weighted inputs, with the
inputs read as $\pm 1$ (paper Eq. 60). $A$ and $B$ excite each other strongly
($\pm 0.7$), $C$ inhibits $B$ ($-0.8$), and each unit weakly affects itself.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
substrate
```

The paper analyzes the state written "aBC": $A$ off, $B$ and $C$ on.
(Lowercase marks an OFF unit.)

```{code-cell} python
state = (0, 1, 1)
```

## Figure 1: integration and exclusion

Fig 1E asks which candidate systems exist. We compute $\varphi_s$ for the
three candidates the paper compares — the single unit $\{A\}$, the pair
$\{A, B\}$ ("aB"), and the whole substrate:

```{code-cell} python
for subset in [(0,), (0, 1), (0, 1, 2)]:
    analysis = pyphi.analyze(substrate, state, subset=subset)
    print(subset, round(analysis.phi, 4))
```

These reproduce the paper's Fig 1E values: $0.04$, $0.17$, and $0.13$. The
pair aB beats both its subset and its superset — and in fact every candidate
that overlaps it — so **aB is a complex**. PyPhi's exhaustive competition
confirms it, and finds one other, non-overlapping complex:

```{code-cell} python
for complex_ in substrate.complexes(state):
    print(complex_.node_indices, round(float(complex_.phi), 4))
```

The single unit $\{C\}$ has the globally maximal $\varphi_s$ here; since it
does not overlap aB, both exist. The paper presents aB as *a* complex —
maximal among the candidates that share its units — which is exactly what
PyPhi finds.

Where does aB's $\varphi_s$ come from? Fig 1D splits it by temporal
direction. Integration is measured separately over the system's causes and
its effects, and the system is only as integrated as its weaker direction:

```{code-cell} python
aB = pyphi.analyze(substrate, state, subset=(0, 1))
print("φ_c =", round(float(aB.sia.cause.phi), 4))
print("φ_e =", round(float(aB.sia.effect.phi), 4))
print("φ_s =", round(aB.phi, 4))
```

$\varphi_c = 0.24$ and $\varphi_e = 0.17$, the paper's published split; the
effect side is the weaker one, so $\varphi_s = \varphi_e$. The partition
responsible — the minimum information partition — is on the analysis as
`aB.sia.partition`.

## Figure 2: the distinctions

With the complex fixed, the composition postulate unfolds what exists *within*
it. Every subset of aB's units — every **mechanism** — is tested for an
irreducible cause and effect. The irreducible ones are the complex's
**distinctions**, and they live on the cause-effect structure:

```{code-cell} python
ces = aB.ces
for d in ces.distinctions:
    print(
        f"{d.mechanism_label:>3}  φ_d = {float(d.phi):.4f}  "
        f"cause {d.cause_purview}  effect {d.effect_purview}"
    )
```

Three distinctions, matching Fig 2: the first-order mechanisms $a$
($\varphi_d = 0.33$) and $B$ ($0.32$), and the second-order mechanism $aB$
($0.07$). Each specifies the *purviews* shown — the units its cause and effect
power is about. A single distinction prints its full detail, including the
repertoires (the probability distributions it specifies over its purviews):

```{code-cell} python
ces.distinctions[1]
```

The whole set collapses to a table with `to_pandas`, handy for sorting,
filtering, or exporting:

```{code-cell} python
ces.to_pandas()
```

## Figure 4: the relations

Distinctions whose purviews overlap congruently — same units, same specified
state — bind together into **relations**. Fig 4 works out the relation between
the distinctions $a$ and $aB$, which overlap over unit $b$:

```{code-cell} python
relation = next(
    r for r in ces.relations
    if {tuple(m) for m in r.mechanisms} == {(0,), (0, 1)}
)
print("φ_r =", round(float(relation.phi), 4))
print("faces:", relation.num_faces)
```

$\varphi_r = 0.036$ with all $9$ faces, the paper's Fig 4 relation (quoted
there as $0.035$, from the rounded $\varphi_d(aB) = 0.07$ divided over the
two-unit purview union). The structure has seven relations in all — including
*self-relations*, where a single distinction's own cause and effect purviews
overlap:

```{code-cell} python
for r in ces.relations:
    mechs = [tuple(m) for m in r.mechanisms]
    print(f"φ_r = {float(r.phi):.4f}  mechanisms {mechs}")
```

## The Φ-structure, summed

Distinctions and relations together are the complex's $\Phi$-structure, and
their summed $\varphi$ is the **structure integrated information** $\Phi$:

```{code-cell} python
print("Σ φ_d =", round(float(ces.sum_phi_distinctions), 4))
print("Σ φ_r =", round(float(ces.sum_phi_relations), 4))
print("Φ     =", round(float(ces.big_phi), 4))
```

Note that $\Phi$ (the structure's total, $1.56$ here) is a different quantity
from $\varphi_s$ (the system's irreducibility over its minimum partition,
$0.17$ here). Both are reported on the analysis: `aB.phi` is $\varphi_s$;
`aB.ces.big_phi` is $\Phi$.

## Summary

For the paper's Fig 1A network in state aBC, PyPhi reproduces, under the
default formalism:

- $\varphi_s = 0.04 / 0.17 / 0.13$ for $\{A\}$ / aB / aBC (Fig 1E), with
  $\varphi_c = 0.24$, $\varphi_e = 0.17$ for aB (Fig 1D);
- aB as a complex, alongside the non-overlapping complex $\{C\}$;
- aB's three distinctions with $\varphi_d = 0.33, 0.32, 0.07$ and their
  purviews (Fig 2);
- the relation $r(\{a, aB\})$ with $\varphi_r = 0.035$ and 9 faces (Fig 4);
- the summed structure, $\Phi = 1.56$.

## Where to go next

- {doc}`../theory/index` — the same pipeline, quantity by quantity, with the
  paper-to-code map.
- {doc}`../theory/intrinsic-information` — why a *deterministic* network
  computes $\varphi_s = 0$ under this formalism.
- {doc}`iit-4.0-demo` — the paper's own supplementary notebook, going deeper
  into the algorithm.
