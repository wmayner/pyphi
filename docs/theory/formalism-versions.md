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

The preceding pages describe IIT 4.0 (2023), PyPhi's default formalism. PyPhi
also implements earlier and refined formalisms, and a separate analysis of
actual causation. A **formalism** is the set of rules that turns a system into
results; which one applies is a matter of configuration.

The cleanest way to select a formalism is the `formalism` argument to
`pyphi.analyze`, which sets the compatible measures for you:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.basic_substrate()

{version: round(float(pyphi.analyze(substrate, (1, 1, 0), formalism=version).phi), 4)
 for version in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026")}
```

The same substrate and state yield a different system integrated information
under each formalism, because each defines that quantity differently.

## IIT 4.0 (2023)

`"IIT_4_0_2023"` — the default, and the formalism this section describes
(Albantakis et al., 2023). System integrated information is
$\varphi_s = \min(\varphi_c, \varphi_e)$, and the cause–effect structure is
unfolded into distinctions and relations.

## IIT 4.0 (2026)

`"IIT_4_0_2026"` refines the account of system integrated information to require
not only that a system *specify* a cause–effect state but also that it provide
itself with a *repertoire of alternatives* — intrinsic **differentiation** —
formalizing a tradeoff between differentiation and specification (Mayner et al.,
2026). A system that cannot furnish enough alternatives has no intrinsic
cause–effect power under this formalism. For the worked example that requirement
is decisive: its $\varphi_s$ falls to $0$, because the substrate does not provide
sufficient intrinsic differentiation.

## IIT 3.0 (2014)

`"IIT_3_0"` is the earlier formalism (Oizumi et al., 2014). It computes a
cause–effect structure of *concepts* rather than distinctions and relations, and
its integrated information is defined differently — hence the different value
above ($\varphi \approx 0.188$). It remains available for reproducing older
results and for comparison; see the [IIT 3.0 overview](iit-3.0.md).

## Actual causation

Actual causation answers a different question — not *how integrated is this
system*, but *which past events actually caused a given present event, and which
effects will it actually cause* (Albantakis et al., 2019). It operates on a
`Transition` (a substrate observed across two time steps) rather than a `System`,
and lives in `pyphi.actual`. It is its own formalism, distinct from the IIT
versions above, and is documented with the how-to guides.

## Under the hood

`formalism=` sets `pyphi.config.formalism.iit.version` together with the
distance measures each version requires. You can also set the version through
configuration directly, but the measures must be made compatible with it; the
`formalism` argument is the reliable path. The three IIT versions correspond to
the namespaces `pyphi.iit3`, `pyphi.iit4_2023`, and `pyphi.iit4_2026`.
