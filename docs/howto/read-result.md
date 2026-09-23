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

# Read a result

{func}`pyphi.analyze` returns an {class}`~pyphi.analyze.Analysis`. This page
walks through what it prints and what each row means, then through the
system irreducibility analysis underneath it and how to interpret an unexpected value.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
analysis = pyphi.analyze(substrate, (0, 1, 1), subset=(0, 1))
analysis
```

## The analysis card

- **Φ**: the structure integrated information, the sum of φ over every
  distinction and every relation. It measures how much structure the system
  specifies. It is `analysis.big_phi`.
- **$\varphi_s$**: the system integrated information: whether the system exists as
  one whole, and how irreducibly. It is `analysis.phi`. Zero means
  *reducible*, not "no structure": a system can have $\varphi_s = 0$ and a nonzero Φ.
- **Distinctions, $\Sigma\varphi_d$**: how many mechanisms specify an irreducible
  cause–effect state, and their total φ.
- **Relations, $\Sigma\varphi_r$**: how many congruent overlaps bind those distinctions,
  and their total φ. Under the default analytical backend these are computed
  in closed form; the individual relations are not enumerated, so they
  cannot be listed one by one. {doc}`query-relations` shows what can be asked
  of them and how to enumerate when you must.
- **Formalism**: appears only on a result computed under an earlier version
  of IIT, and gives that version (see {doc}`earlier-versions`). It is
  `analysis.formalism`.
- **The distinction table**: one row per distinction: its mechanism, $\varphi_d$,
  and its cause and effect purviews, each written in the *state* the
  distinction specifies. An uppercase letter is a unit ON, lowercase is OFF,
  and a unit with more than two states carries its state as a subscript
  (`A₂`). A cause purview `a` means the mechanism specifies unit A being OFF
  in the past.
- **System**: the units and current state analyzed, the cause and effect
  states the system specifies, the minimum partition (the cut
  that makes the least difference: the system's weakest link), the system's
  intrinsic information ii(s), and, when the intrinsic-information
  requirement set $\varphi_s$, which term and direction did so. The next section
  explains these.

## The system irreducibility analysis

```{code-cell} python
sia = analysis.sia
sia
```

- **Normalized $\varphi_s$**: $\varphi_s$ divided by the partition's normalization; the
  minimum partition is chosen on this value.
- **Specified state** (cause and effect): the past and future states the
  system specifies with maximal intrinsic information.
- **Intrinsic specification**: how selectively and informatively the
  specified state is picked out.
- **Intrinsic differentiation**: the surprisal of the specified state: how
  much of a repertoire of alternatives the system provides itself. Zero for
  a deterministic transition.
- **MIP**: the partition and, in the grid, the connections it severs; "Tied
  MIPs" counts partitions tied with it.

$\varphi_s$ is the smallest of three terms: the cause-side integration
$\varphi_c$, the effect-side integration $\varphi_e$, and the intrinsic
information ii(s), itself the smaller of specification and
differentiation over both directions (see
{doc}`../theory/intrinsic-information`). `sia.explain()` says which term
won:

```{code-cell} python
for finding in sia.explain().findings:
    print(finding.kind, "=", finding.value)
```

Here the effect-side differentiation is the smallest term, so it is $\varphi_s$.

## When φₛ is zero

There are two different reasons, and the findings above tell them apart.

**Ordinary reducibility.** One side's integration is already zero: some
partition of the system makes no difference to its cause or effect
repertoire. Then `binding_direction` gives that side and there is no
`requirement_binding` finding.

**The intrinsic-information requirement.** Both $\varphi_c$ and $\varphi_e$ are positive,
but the system provides itself no repertoire of alternatives (a
deterministic transition has zero differentiation), so ii(s) is zero and
with it $\varphi_s$. Then a `requirement_binding` finding gives the term
(`differentiation` or `specification`) and the direction.

```{code-cell} python
basic = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), compute="sia")
print(float(basic.cause.phi), float(basic.effect.phi), basic.intrinsic_information)
[f.value for f in basic.explain().findings if f.kind == "requirement_binding"]
```

## Where to go next

- {doc}`../theory/overview` for what these quantities are in the theory.
- {doc}`tie-breaking` for the selection margins and what an "effectively
  tied" result means.
- {doc}`sweep` to compute the same quantities over every state at once.
