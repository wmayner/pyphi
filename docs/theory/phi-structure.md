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

# The Φ-structure

The distinctions and relations of a complex, taken together, are its
**$\Phi$-structure** — the cause–effect structure that, in IIT, corresponds to
the quality of an experience. Its total irreducibility is the **structure
integrated information** $\Phi$ ("big phi"), the sum of the integrated
information of every distinction and relation it contains (Albantakis et al.,
2023):

$$ \Phi = \sum_{d} \varphi_d \;+\; \sum_{r} \varphi_r. $$

In PyPhi the $\Phi$-structure is a `CauseEffectStructure`, returned as
`analysis.ces`:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

analysis = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0))
ces = analysis.ces

(round(float(ces.sum_phi_distinctions), 3),
 round(float(ces.sum_phi_relations), 3),
 round(float(ces.big_phi), 3))
```

The three distinction $\varphi_d$ values sum to $1.330$, the two relation
$\varphi_r$ values to $0.527$, and their total is $\Phi \approx 1.857$. The whole
structure can be viewed as a table:

```{code-cell} python
ces.to_pandas()
```

This is the complete answer to the two questions from the [overview](overview.md):
the complex exists with system integrated information $\varphi_s \approx 0.208$,
and the experience it specifies has the $\Phi$-structure above, of quantity
$\Phi \approx 1.857$.

## The paper-to-code map

Every named quantity of IIT 4.0 corresponds to a type or attribute in PyPhi. The
table maps the symbols of Albantakis et al. (2023) to where they live in the
code, using the worked example (`analysis = pyphi.analyze(substrate, (1, 1, 0))`).

| Symbol | Quantity | In PyPhi |
| --- | --- | --- |
| $U$ | substrate (units and their interactions) | `Substrate`, e.g. `pyphi.examples.basic_substrate()` |
| $\mathcal{T}_U$ | transition probability matrix (Eq. 1) | `substrate.tpm`, `substrate.factored_tpm` |
| $S$ | candidate system in a state | `System`; analyzed via `pyphi.analyze` |
| $\mathit{ii}$ | intrinsic information (maximal cause–effect state) | selected inside `analyze`; carried on `analysis.sia` |
| $\varphi_c,\ \varphi_e$ | cause- and effect-side integrated information | `analysis.sia.cause.phi`, `analysis.sia.effect.phi` |
| MIP | minimum information partition | `analysis.sia.partition` |
| $\varphi_s$ | system integrated information, $\min(\varphi_c,\varphi_e)$ | `analysis.phi` (`analysis.sia.phi`) |
| $\varphi_s^{\ast}$ | maximal system integrated information (the complex's) | the $\varphi_s$ of the complex `analyze` returns |
| complex | maximal substrate | `analysis.system` |
| mechanism | subset specifying a distinction | `distinction.mechanism` |
| purview | subset a mechanism constrains | `distinction.cause_purview`, `distinction.effect_purview` |
| distinction | mechanism with its cause and effect state | `Distinction`; `analysis.ces.distinctions` |
| $\varphi_d$ | distinction integrated information | `distinction.phi` |
| relation | congruent overlap of distinctions | `Relation`; `analysis.ces.relations` |
| $\varphi_r$ | relation integrated information | `relation.phi` |
| $\Phi$-structure | the distinctions and relations together | `CauseEffectStructure`; `analysis.ces` |
| $\Phi$ | structure integrated information | `analysis.ces.big_phi` |

This mapping is what makes the theory computable: each line of the IIT 4.0
formalism has a home in the code, and the worked example on the preceding pages
walks the whole table from top to bottom.
