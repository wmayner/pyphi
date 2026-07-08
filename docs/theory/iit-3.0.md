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

# IIT 3.0

Before IIT 4.0, the theory's formalism was IIT 3.0 (Oizumi, Albantakis & Tononi,
2014). PyPhi still implements it, both for reproducing earlier results and for
comparison. This page sketches how it differs from 4.0; the earlier pages of
this section describe 4.0, the default.

## What IIT 3.0 computes

IIT 3.0 shares the overall shape of the analysis — mechanisms specifying causes
and effects, a system whose irreducibility is measured over its minimum
partition — but differs in what it builds and how it measures it:

- A mechanism specifies a **concept**: its maximally irreducible cause and effect
  repertoires, with a small-$\varphi$ value. The concepts of a system form its
  **cause–effect structure** (a *constellation* in concept space).
- The system's **big-$\Phi$** measures how irreducible the whole constellation is
  under the minimum partition, using the earth-mover's distance between
  constellations.
- There are **no relations**: the explicit relations between overlapping
  purviews are an IIT 4.0 addition, as is the intrinsic-difference measure that
  4.0 uses in place of the earth-mover's distance.

Because the quantities are defined differently, the same substrate and state give
a different value than under 4.0:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.basic_substrate()
analysis = pyphi.analyze(substrate, (1, 1, 0), formalism="IIT_3_0")
analysis.phi
```

The result is an IIT 3.0 system irreducibility analysis
(`analysis.sia` is an `IIT3SystemIrreducibilityAnalysis`), and its cause–effect
structure is a set of concepts rather than the distinctions and relations of a
4.0 $\Phi$-structure.

## When to use it

Use IIT 3.0 to reproduce or compare against results computed under the earlier
formalism. For new work, IIT 4.0 is the current theory and PyPhi's default; see
[formalism versions](formalism-versions.md) for selecting between them and for
the further refinement in IIT 4.0 (2026). For the full IIT 3.0 formalism, see
Oizumi, Albantakis & Tononi (2014).
