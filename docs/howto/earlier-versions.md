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

# Reproduce results from earlier versions of IIT

PyPhi computes IIT as formulated by Albantakis et al. (2023) and Mayner,
Marshall & Tononi (2026). Much of the published literature was computed under
earlier versions of the theory: IIT 4.0 as first published in 2023, and
IIT 3.0 (Oizumi, Albantakis & Tononi, 2014). PyPhi implements both, so those
results can be reproduced. This page describes how the versions differ, how to
compute under an earlier one, and how to compare the numbers.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## How the versions differ

The same substrate and state give a different system integrated information
under each version, because each version defines that quantity differently.
On the three-XOR network:

```{code-cell} python
substrate = pyphi.examples.xor_substrate()

{version: round(float(pyphi.analyze(substrate, (0, 0, 0), formalism=version).phi), 4)
 for version in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026")}
```

`"IIT_4_0_2026"` is what PyPhi computes when no version is given.

### IIT 4.0 as published in 2023

Albantakis et al. (2023) define system integrated information as the smaller
of the cause-side and effect-side integration,
$\varphi_s = \min(\varphi_c, \varphi_e)$. IIT's current definition also
includes the system's intrinsic information,
$\varphi_s = \min(\varphi_c, \varphi_e, \mathit{ii}(s))$ (Mayner, Marshall &
Tononi, 2026; see {doc}`../theory/intrinsic-information`). Everything else is
the same: the partition search, the specified-state search, the distinctions,
the relations, and $\Phi$.

The two therefore give different values only where $\mathit{ii}(s)$ is the
smallest of the three terms. That happens in deterministic systems, where
$\mathit{ii}(s) = 0$ but the 2023 value is positive, and in probabilistic
systems whose intrinsic information is below their integration. The XOR
network above is deterministic. Most of the classic worked examples in the
literature are deterministic too, and their published nonzero $\varphi_s$
values are 2023 values.

The two-unit system aB of the IIT 4.0 paper's Fig. 1A network is a
probabilistic example. Its cause-side and effect-side integration are
$\varphi_c \approx 0.245$ and $\varphi_e \approx 0.172$, as in the paper's
Fig. 1D, so the 2023 value is $\varphi_s = \varphi_e \approx 0.172$, the value
printed in Fig. 1E. Its intrinsic information is smaller still, so PyPhi
reports $\varphi_s \approx 0.04$:

```{code-cell} python
fig1a = pyphi.examples.iit4_2023_fig1a_substrate()
{version: round(float(pyphi.analyze(fig1a, (0, 1, 1), subset=(0, 1),
                                    formalism=version, compute="sia").phi), 4)
 for version in ("IIT_4_0_2023", "IIT_4_0_2026")}
```

The quantity the 2023 paper calls "intrinsic information" is the one PyPhi
calls intrinsic specification. A result computed under the 2023 version uses
the paper's name for it on its analysis card.

### IIT 3.0

IIT 3.0 has the same overall shape as IIT 4.0: mechanisms specify causes and
effects, and a system's irreducibility is measured over its minimum
partition. It differs in what it builds and how it measures it:

- A mechanism specifies a **concept**: its maximally irreducible cause and
  effect repertoires, with a small-$\varphi$ value. The concepts of a system
  form its **cause–effect structure** (a *constellation* in concept space).
- The system's **big-$\Phi$** measures how irreducible the whole
  constellation is under the minimum partition, using the earth mover's
  distance between constellations.
- There are **no relations**. Relations between overlapping purviews were
  introduced in IIT 4.0, as was the intrinsic-difference measure that
  replaced the earth mover's distance.
- Units outside the system (its background) are held at their current state
  for causes as well as effects. This is PyPhi 1.x's convention, so results
  computed with PyPhi 1.x reproduce. Oizumi et al. (2014) held the background
  at its actual past state for causes (IIT 4.0, S2 Text). IIT 4.0 instead
  causally marginalizes the background's past conditional on its current
  state (Albantakis et al., 2023, Eqs. 3–4). The difference matters only for
  systems smaller than the whole substrate.

```{code-cell} python
analysis = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), formalism="IIT_3_0")
analysis.phi
```

The result's `analysis.sia` is an `IIT3SystemIrreducibilityAnalysis`, and its
cause–effect structure is a set of concepts rather than the distinctions and
relations of a $\Phi$-structure. For the full IIT 3.0 formalism, see Oizumi,
Albantakis & Tononi (2014).

(formalism-selection)=
## Compute under an earlier version

Each version is defined by a complete set of settings under
`pyphi.config.formalism.iit`: the version itself, the distance measures, the
partition schemes, the tie-resolution rules, and, for IIT 3.0, the background
convention and `precision`. PyPhi packages each set as a preset. There are
three ways to apply one, and they give the same results:

- The `formalism=` argument of {func}`pyphi.analyze`, for a single call:
  `pyphi.analyze(substrate, state, formalism="IIT_4_0_2023")`.
- A preset in a `pyphi.config.override` block, for a block of code:
  `with pyphi.config.override(**pyphi.iit4_2023): ...`. The presets are
  `pyphi.iit4_2023` and `pyphi.iit3`, also available from
  {mod}`pyphi.conf.presets`.
- Replacing the formalism layer for a whole session:
  `pyphi.config.formalism = dataclasses.replace(pyphi.config.formalism, **pyphi.iit4_2023)`.

```{code-cell} python
import warnings

# Applying a preset emits advisory warnings listing the options it changes;
# they are silenced here to keep the output readable.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with pyphi.config.override(**pyphi.iit3):
        print("version:  ", pyphi.config.formalism.iit.version)
        print("precision:", pyphi.config.precision)
```

Apply the whole preset. Setting `formalism.iit.version` on its own raises a
`ConfigurationError`, because the other settings would stay at their current
values and the result would match no published version. The same holds in a
`pyphi_config.yml` file: to select a version there, write out every field the
preset sets. The repository ships
[`pyphi_config_3.0.yml`](https://github.com/wmayner/pyphi/blob/develop/pyphi_config_3.0.yml)
as a complete example for IIT 3.0.

A result computed under an earlier version records it: `analysis.formalism`
gives the version, and the analysis card shows it in a **Formalism** row.

## Compare the numbers

Compare $\varphi_s$ values only when they were computed under the same
version. Compare them with a tolerance: `pyphi.numerics.eq(a, b)` respects
the configured precision, and `==` does not. When a number does not match a
paper, first check which version the paper used, then check the substrate,
the state, and the candidate system.

{func}`pyphi.sweep` can compute the same cells under several versions in one
call through its `formalisms=` argument, which indexes the resulting table by
version; see {doc}`sweep`.

## References

- Albantakis L, Barbosa L, Findlay G, Grasso M, et al. (2023). Integrated
  information theory (IIT) 4.0: Formulating the properties of phenomenal
  existence in physical terms. *PLOS Computational Biology* 19(10): e1011465.
- Mayner WGP, Marshall W, Tononi G. (2026). Intrinsic cause–effect power: the
  tradeoff between differentiation and specification. *Entropy* 28(4): 410.
- Oizumi M, Albantakis L, Tononi G. (2014). From the phenomenology to the
  mechanisms of consciousness: Integrated Information Theory 3.0. *PLOS
  Computational Biology* 10(5): e1003588.
