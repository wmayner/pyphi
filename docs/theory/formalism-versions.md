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

# Formalism versions

The preceding pages describe IIT 4.0 in its 2026 refinement — PyPhi's default
formalism. PyPhi also implements the 2023 formulation and IIT 3.0, and a
separate analysis of actual causation. A **formalism** is the set of rules
that turns a system into results; which one applies is a matter of
configuration.

The simplest way to select a formalism is the `formalism` argument to
`pyphi.analyze`, which sets the compatible measures for you. The same
substrate and state yield a different system integrated information under
each formalism, because each defines that quantity differently — here on the
three-XOR network:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.xor_substrate()

{version: round(float(pyphi.analyze(substrate, (0, 0, 0), formalism=version).phi), 4)
 for version in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026")}
```

## IIT 4.0 (2026)

`"IIT_4_0_2026"` — the default. It refines the account of system integrated
information to require both that a system *specify* a cause–effect state and
that it provide itself with a *repertoire of alternatives* — intrinsic
**differentiation**. The system's intrinsic information enters the minimum
that defines $\varphi_s$ (Mayner, Marshall, Tononi 2026). The XOR network
above shows the consequence: it is deterministic, so it furnishes no
alternatives, and its $\varphi_s$ is $0$, where the 2023 value is $1.5$. See
{doc}`intrinsic-information` for details.

## IIT 4.0 (2023)

`"IIT_4_0_2023"` — the formulation of Albantakis et al. (2023), identical to
the default except that it has no intrinsic-information requirement:
$\varphi_s = \min(\varphi_c, \varphi_e)$. Distinctions, relations, and $\Phi$
are the same under both. Choose it to reproduce numbers published against the 2023
paper — most classic worked examples in the literature are deterministic, and
their published nonzero $\varphi_s$ values are 2023 quantities.

## IIT 3.0 (2014)

`"IIT_3_0"` is the earlier formalism (Oizumi et al., 2014). It computes a
cause–effect structure of *concepts* rather than distinctions and relations,
and its integrated information is defined differently — hence the third value
above. It remains available for reproducing older results and for comparison;
see the [IIT 3.0 overview](iit-3.0.md).

## Actual causation

Actual causation answers a different question — not *how integrated is this
system*, but *which past events actually caused a given present event, and
which effects will it actually cause* (Albantakis et al., 2019). It operates
on a `Transition` (a substrate observed across two time steps) rather than a
`System`, and is provided by `pyphi.actual`. It is its own formalism, unaffected by
the IIT versions above (in particular, the 2026 intrinsic-information
requirement does not apply to it), and is documented with the tutorials.

## What `formalism=` sets

`formalism=` sets `pyphi.config.formalism.iit.version` together with the
distance measures each version requires. You can also set the version through
configuration directly, but then you must make the measures compatible with
it yourself; using the `formalism` argument, or applying a whole preset from
`pyphi.conf.presets` (`iit3`, `iit4_2023`, `iit4_2026`) with
`config.override`, avoids that pitfall. The three IIT versions correspond to
the namespaces `pyphi.iit3`, `pyphi.iit4_2023`, and `pyphi.iit4_2026`.
