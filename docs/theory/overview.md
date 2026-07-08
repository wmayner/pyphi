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

# What IIT 4.0 computes

Integrated Information Theory (IIT) starts from the properties of experience and
formulates them as requirements on the *cause–effect power* of a physical
substrate. PyPhi is a computational implementation of the resulting formalism,
IIT 4.0 (Albantakis et al., 2023). This section explains what PyPhi computes and
why, mapping each quantity of the theory to the type or function that computes
it, and following one small example the whole way through.

For a broad orientation to the theory itself, see the [IIT wiki](https://iit.wiki);
the authoritative source for the formalism is Albantakis et al. (2023). This
section is self-contained — it does not assume you have read either — but it
cites the paper for the full derivations it does not repeat.

## What PyPhi computes

Given a **substrate** (a set of interacting units, defined by how they influence
one another) and a **state** of that substrate, PyPhi answers two questions:

1. **Is a set of units a subject of experience, and how much?** IIT identifies
   the units that form a *complex* — a set whose cause–effect power is maximally
   irreducible — and measures its irreducibility as the **system integrated
   information**, $\varphi_s$.
2. **What is the structure of that experience?** From the complex, PyPhi unfolds
   the **$\Phi$-structure**: the **distinctions** the units specify and the
   **relations** among them. The $\Phi$-structure's total irreducibility is the
   **structure integrated information**, $\Phi$ ("big phi").

The rest of this section builds these two answers up in the order PyPhi computes
them.

## The pipeline

PyPhi's objects mirror the theory's layering:

$$ \textsf{Substrate} \;\rightarrow\; \textsf{System} \;\rightarrow\; \textsf{formalism} \;\rightarrow\; \Phi\textsf{-structure} $$

- A `Substrate` is the causal model: the units and their transition
  probabilities.
- A `System` is a candidate subset of the substrate's units, in a state.
- A **formalism** (IIT 4.0, IIT 3.0, actual causation) is the set of rules for
  turning a system into results; which one applies is a matter of configuration.
- The **$\Phi$-structure** is the result: the distinctions and relations the
  complex specifies, with their integrated-information values.

## The postulates

IIT derives its formalism from six *postulates of physical existence*, each the
physical counterpart of a property of experience (Albantakis et al., 2023):

- **Existence** — the units must have cause–effect power: they take and make a
  difference.
- **Intrinsicality** — that power must be *intrinsic*: the units must take and
  make a difference *within themselves*, from their own perspective.
- **Information** — the power must be *specific*: the system in its state selects
  a particular cause–effect state, the one with maximal *intrinsic information*
  ($\mathit{ii}$).
- **Integration** — the power must be *unitary*: the cause–effect state must be
  irreducible to independent parts, measured by *integrated information* over the
  system's minimum partition ($\varphi_s$).
- **Exclusion** — the power must be *definite*: exactly one set of units, the one
  with maximal integrated information, is the complex.
- **Composition** — the power must be *structured*: subsets of the units
  (mechanisms) specify cause–effect states over subsets of units (purviews) —
  the **distinctions** — which overlap in **relations**, together forming the
  $\Phi$-structure.

Each page that follows takes one step of the pipeline and names the postulate it
enforces.

## The worked example

One substrate carries this section: the three-unit example system from the IIT
4.0 paper, available as `pyphi.examples.basic_substrate()`.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.basic_substrate()
substrate
```

Analyzing it in the state $(1, 1, 0)$ runs the whole pipeline:

```{code-cell} python
analysis = pyphi.analyze(substrate, (1, 1, 0))
analysis.phi  # the system integrated information, φ_s
```

This value, $\varphi_s$, answers the first question: the units form a complex,
and $\varphi_s \approx 0.208$ measures how irreducible it is. The second
question — the structure — is answered by the $\Phi$-structure it specifies:

```{code-cell} python
(len(analysis.ces.distinctions), len(analysis.ces.relations), analysis.ces.big_phi)
```

Three distinctions, two relations, and a structure integrated information
$\Phi \approx 1.857$. The pages ahead unpack each of these: the [substrate and
system](substrate-and-system.md) it starts from, the [system integrated
information](system-integration.md) $\varphi_s$ that finds the complex, the
[distinctions and relations](distinctions-and-relations.md) that compose the
structure, and [the $\Phi$-structure](phi-structure.md) itself with the full map
from paper symbols to PyPhi types.
