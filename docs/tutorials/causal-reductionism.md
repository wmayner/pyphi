---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Causal reductionism and the frog

{download}`Download this page as a Jupyter notebook <causal-reductionism.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/tutorials/causal-reductionism.ipynb)

*Causal reductionism* is the assumption that once every elementary unit's cause
has been accounted for, there is nothing causal left to explain — a whole is,
causally, nothing more than its parts. This tutorial reproduces the central
demonstration of

> Grasso M, Albantakis L, Lang JP, Tononi G (2021). Causal reductionism and
> causal structures. *Nature Neuroscience*, 24, 1348–1355.
> <https://doi.org/10.1038/s41593-021-00911-8>

that the assumption is false: some *composite* (higher-order) mechanisms have
causes of their own, irreducible to the causes of their parts, and an
actual-causation analysis makes those causes explicit. It uses PyPhi's
actual-causation tools; for an introduction to those tools on a smaller
network, see {doc}`actual-causation`.

```{code-cell} python
import pyphi
from dataclasses import replace
from pyphi import actual

pyphi.config.progress_bars = False
```

## The frogs

The paper studies three species of simulated "frog" organisms that catch bugs.
Each frog has **sensors** (on its retina), **central neurons** (its brain), and
**motor** units (that drive a jump). The species differ only in how the central
neurons are wired:

- **F3** — three sensors (`SL`, `SC`, `SR`), three central neurons
  (`CL`, `CC`, `CR`), two motors (`ML`, `MR`). `CC` is a *super-bug detector*:
  a single composite unit that fires when the left and right sensors are on
  together.
- **F2** — the super-bug detector `CC` has been removed, but the remaining
  central neurons `CL` and `CR` still act together on the motors.
- **F1** — a pair of "half-frogs," each reduced to two sensors, one central
  neuron, and one motor. This is the most reductionist wiring.

Each frog is available as a substrate:

```{code-cell} python
substrate = pyphi.examples.frog_substrate("F3")
substrate
```

## Setting the formalism

The paper's analysis is IIT 3.0 actual causation. We build that configuration
from the built-in `iit3` preset and layer on the specific choices the paper
uses: `WEDGE_TRIPARTITION` mechanism partitions, the absolute-intrinsic-
difference (`AID`) measure, and the `WPMI` α-measure for actual causation. State
validation is turned off because the frogs' observed states need not be
reachable from every other state.

```{code-cell} python
frog_formalism = dict(
    iit=replace(
        pyphi.iit3["iit"],
        mechanism_partition_scheme="WEDGE_TRIPARTITION",
        mechanism_phi_measure="AID",
    ),
    validate_system_states=False,
    alpha_measure="WPMI",
)
```

Constructing a substrate or transition does not depend on this configuration;
only *computing the account* does. So we apply `frog_formalism` in a
`config.override` block around each account computation.

## F3: the account of a frog's behavior

`frog_transition` gives the state transition the paper analyzes — the frog
observed catching a bug, described by its state just before and just after,
together with the sensor units taken as potential causes and the motor and
central units taken as potential effects.

```{code-cell} python
with pyphi.config.override(**frog_formalism):
    account = actual.account(pyphi.examples.frog_transition("F3"))

account
```

The account is the frog's full causal structure: every irreducible causal link
between an occurrence at one time and an occurrence at the other, each with its
irreducibility $\alpha$. It behaves like a sequence of links.

```{code-cell} python
len(account)
```

### The causes reductionism misses

A *first-order* account would list only the causes of individual units. The
interesting links are the **composite** ones, whose cause or effect spans more
than one unit:

```{code-cell} python
composite = [link for link in account if len(link.purview) >= 2]
len(composite)
```

Among them is the cause of the super-bug detector `CC`. Its cause is not any
single sensor but the *joint* occurrence of all three sensors — a composite
cause that no first-order analysis can represent:

```{code-cell} python
labels = substrate.node_labels
for link in account:
    mechanism = ",".join(labels[i] for i in link.mechanism)
    purview = ",".join(labels[i] for i in link.purview)
    if mechanism == "CC" or purview == "SL,SC,SR":
        print(f"{link.direction!s:>6}  {mechanism:>8}  purview [{purview}]  α = {float(link.alpha):.3f}")
```

The `explain` method spells a single link out in words:

```{code-cell} python
cc_cause = next(
    link for link in account
    if ",".join(labels[i] for i in link.mechanism) == "CC"
    and link.direction is pyphi.Direction.CAUSE
)
print(cc_cause.explain())
```

The whole account is also available as a data frame, for sorting or export:

```{code-cell} python
account.to_pandas()
```

## Comparing the three species

The paper's argument is comparative: as the wiring is simplified, the composite
causal structure thins out, but it never disappears entirely — even the reduced
frogs contain composite causes that reductionism cannot account for. We count
the composite links in each species' account.

```{code-cell} python
for species in ("F3", "F2", "F1"):
    with pyphi.config.override(**frog_formalism):
        account = actual.account(pyphi.examples.frog_transition(species))
    composite = sum(1 for link in account if len(link.purview) >= 2)
    print(f"{species}:  {len(account):>3} links,  {composite:>2} composite")
```

`F3`, with its dedicated super-bug detector, has the richest composite
structure; `F1`, the reductionist baseline, the least. But the count is never
zero: composite causes are a feature of the wiring, not an artifact of one
elaborate design.

## Takeaway

Fixing the cause of every individual unit does *not* fix the causes of the
composite mechanisms those units form. The super-bug detector's cause is the
three sensors acting together, and no enumeration of single-unit causes
contains it. Actual causation makes such composite causes explicit, which is
why — contrary to the reductionist intuition — a system can be, causally, more
than the sum of its parts.

## Where to go next

- {doc}`actual-causation` — the actual-causation tools on a smaller network.
- {doc}`../howto/configure` — reading, setting, and scoping configuration,
  including the formalism presets used here.
