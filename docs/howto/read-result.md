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
system irreducibility analysis underneath it and the questions to ask when
a value surprises you.

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
  specifies. It is `analysis.big_phi`. Under IIT 3.0 there are no relations
  and this row is absent; that formalism's Φ is the next row.
- **φ_s**: the system integrated information: whether the system exists as
  one whole, and how irreducibly. It is `analysis.phi`. Zero means
  *reducible*, not "no structure": a system can have φ_s = 0 and a nonzero Φ.
- **Distinctions, Σφ_d**: how many mechanisms specify an irreducible
  cause–effect state, and their total φ.
- **Relations, Σφ_r**: how many congruent overlaps bind those distinctions,
  and their total φ. Under the default analytical backend these are computed
  in closed form; the individual relations are not enumerated, so they
  cannot be listed one by one. {doc}`query-relations` shows what can be asked
  of them and how to enumerate when you must.
- **Formalism**: which version of the theory produced the numbers above.
  Every other row depends on it; a φ value reported without it cannot be
  compared with anything. It is `analysis.formalism`.
- **The distinction table**: one row per distinction: its mechanism, φ_d,
  and its cause and effect purviews, each written in the *state* the
  distinction specifies. An uppercase letter is a unit ON, lowercase is OFF,
  and a unit with more than two states carries its state as a subscript
  (`A₂`). A cause purview `a` means the mechanism specifies unit A being OFF
  in the past.
- **System: MIP, ii(s), Requirement binds**: the minimum information
  partition (the cut that makes the least difference: the system's weakest
  link), the system's intrinsic information, and, when the
  intrinsic-information requirement set φ_s, which term and direction did
  so. The next section explains these.

## The system irreducibility analysis

```{code-cell} python
sia = analysis.sia
sia
```

- **Normalized φ_s**: φ_s divided by the partition's normalization; the
  minimum information partition is chosen on this value.
- **Specified state** (cause and effect): the past and future states the
  system specifies with maximal intrinsic information.
- **Intrinsic specification** (labelled **Intrinsic information** under the
  2023 formalism, which is that paper's name for the same quantity): how
  selectively and informatively the specified state is picked out.
- **Intrinsic differentiation**: the surprisal of the specified state: how
  much of a repertoire of alternatives the system provides itself. Zero for
  a deterministic transition.
- **MIP**: the partition and, in the grid, the connections it severs; "Tied
  MIPs" counts partitions tied with it.

Under the default formalism, IIT 4.0 (2026), φ_s is the smallest of three
terms: the cause-side integration φ_c, the effect-side integration φ_e, and
the intrinsic information ii(s), itself the smaller of specification and
differentiation over both directions (see
{doc}`../theory/intrinsic-information`). `sia.explain()` says which term
won:

```{code-cell} python
for finding in sia.explain().findings:
    print(finding.kind, "=", finding.value)
```

Here the effect-side differentiation is the smallest term, so it is φ_s.
Under the 2023 formalism, which does not apply the requirement, the same
pair has φ_s = min(φ_c, φ_e) = 0.17, the value the paper prints.

## When φ_s is zero

There are two different reasons, and the findings above tell them apart.

**Ordinary reducibility.** One side's integration is already zero: some
partition of the system makes no difference to its cause or effect
repertoire. Then `binding_direction` names that side and there is no
`requirement_binding` finding. This happens under every formalism.

**The intrinsic-information requirement.** Both φ_c and φ_e are positive,
but the system provides itself no repertoire of alternatives (a
deterministic transition has zero differentiation), so ii(s) is zero and
with it φ_s. Then a `requirement_binding` finding gives the term
(`differentiation` or `specification`) and the direction. This happens only
under IIT 4.0 (2026); the same system under `formalism="IIT_4_0_2023"` keeps
its `min(φ_c, φ_e)`.

```{code-cell} python
basic = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), compute="sia")
print(float(basic.cause.phi), float(basic.effect.phi), basic.intrinsic_information)
[f.value for f in basic.explain().findings if f.kind == "requirement_binding"]
```

## Reading the numbers across formalisms

The same substrate and state give different φ_s under each formalism,
because each defines it differently; see {doc}`../theory/formalism-versions`.
Compare values only within one formalism, and compare them tolerantly:
`pyphi.numerics.eq(a, b)` respects the configured precision where `==` does
not.

## Where to go next

- {doc}`../theory/overview` for what these quantities are in the theory.
- {doc}`tie-breaking` for the selection margins and what an "effectively
  tied" result means.
- {doc}`sweep` to compute the same quantities over every state at once.
