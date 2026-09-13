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

# The intrinsic-information requirement

The intrinsicality postulate requires that a system's cause–effect power be
assessed from the system's own perspective. Mayner, Marshall, and Tononi
(2026) formulate this as two complementary requirements. To have cause–effect
power intrinsically, a system must provide itself with a repertoire of
alternative cause–effect states, its intrinsic **differentiation**, and it
must specify one of those alternatives, its intrinsic **specification**. The
two trade off: a system that offers many alternatives specifies each of them
weakly, and a system that specifies one state sharply offers few alternatives.
Both are measured by the system's **intrinsic information**
$\mathit{ii}(s)$, which enters the minimum that defines system integrated
information:

$$ \varphi_s = \min\{\varphi_c,\ \varphi_e,\ \mathit{ii}(s)\}. $$

This is Eq. 23 of "Intrinsic Cause–Effect Power: The Tradeoff Between
Differentiation and Specification" (*Entropy* 28, 410), the formulation
PyPhi computes by default. $\mathit{ii}(s)$ is the minimum, across the cause
and effect directions, of each direction's intrinsic information, itself the
minimum of that direction's differentiation $i^{c/e}_{\mathrm{diff}}(s)$ and
specification $i^{c/e}_{\mathrm{spec}}(s)$ (Section 2.3, preceding Eq. 13):

$$
\mathit{ii}(s) = \min\{\mathit{ii}_c(s),\ \mathit{ii}_e(s)\}, \qquad
\mathit{ii}_{c/e}(s) = \min\{i^{c/e}_{\mathrm{diff}}(s),\ i^{c/e}_{\mathrm{spec}}(s)\}.
$$

$\varphi_c$ and $\varphi_e$ are the cause- and effect-side integrated
information of {doc}`system-integration`. This page explains the
differentiation requirement, follows the measure through small examples, and
describes the scope of the requirement.

## Differentiation and determinism

The requirement is easiest to see in the paper's opening example (Section 2):
a single unit implementing deterministic COPY logic. From the outside, an
experimenter can set the unit to each of its states in turn, observe that it
copies them, and conclude that it has cause–effect power. From the unit's own
perspective, its current state admits exactly one past state and one future
state; no alternatives are available to it, so there is no difference for it
to make to itself. Intrinsic differentiation quantifies the availability of
such alternatives: like entropy, it is zero for a perfectly deterministic
system and increases with decreasing determinism (Section 2.2).

Specification behaves in the opposite way. As the paper puts it: "Purely
deterministic systems provide no genuine alternatives, and thus their
intrinsic differentiation is zero, while purely random systems specify no
state, leaving intrinsic specification at zero" (Section 4). A deterministic
system therefore has $\varphi_s = 0$, a maximally noisy system likewise has
$\varphi_s = 0$, and positive intrinsic information requires a balance of the
two.

The three-XOR network is the deterministic case:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

xor = pyphi.examples.xor_substrate()
analysis = pyphi.analyze(xor, (0, 0, 0))
analysis.phi
```

The analysis records where the zero comes from. Both directions are
integrated and specify a state, but the effect side has zero differentiation:

```{code-cell} python
sia = analysis.sia
print("φ_c =", float(sia.cause.phi), "  φ_e =", float(sia.effect.phi))
print("differentiation:",
      {str(d): float(v) for d, v in sia.intrinsic_differentiation.items()})
```

The deterministic transition offers no alternative effect, so the effect-side
differentiation is $0$. The cause side records the network's two-fold
predecessor degeneracy: each state is reachable from exactly two prior states,
so $-\log_2 \tfrac{1}{2} = 1$. The cause side is evaluated on the Bayesian
posterior over prior states (Eqs. 6 and 11), so it measures predecessor
degeneracy and can be positive even for deterministic dynamics; the effect
side alone brings the minimum, and with it $\varphi_s$, to $0$.

## Two ways to reach zero

A system's $\varphi_s$ is zero either because one side's integration is zero
(some partition makes no difference) or because the requirement binds: both
$\varphi_c$ and $\varphi_e$ are positive and $\mathit{ii}(s)$ is zero.
`explain()` distinguishes them: the second case carries a
`requirement_binding` finding, the first does not.

```{code-cell} python
basic = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 1, 0), compute="sia")
(float(basic.cause.phi), float(basic.effect.phi), basic.intrinsic_information,
 [f.value for f in basic.explain().findings if f.kind == "requirement_binding"])
```

The XOR network above is the same case. A network whose cause side is
reducible outright shows $\varphi_c = 0$ and no such finding; see
{doc}`Read a result <../howto/read-result>`.

## Reading the two terms

Both terms are available on the result. On the system irreducibility
analysis, `intrinsic_specification` gives, per direction, the selectivity
times informativeness of the specified state (Eqs. 7 and 9);
`intrinsic_differentiation` gives that state's surprisal (Eqs. 4 and 6);
`intrinsic_information` is their joint minimum (Eq. 13); and
`integrated_fraction` is $\varphi_s / \mathit{ii}(s)$. The two-unit system aB
of the Fig 1A network (Albantakis et al., 2023) has all four:

```{code-cell} python
fig1a = pyphi.examples.iit4_2023_fig1a_substrate()
sia = pyphi.analyze(fig1a, (0, 1, 1), subset=(0, 1)).sia
{str(d): (sia.intrinsic_specification[d], float(sia.intrinsic_differentiation[d]))
 for d in sia.intrinsic_specification}
```

```{code-cell} python
sia.intrinsic_information, sia.integrated_fraction
```

When the requirement sets $\varphi_s$, `explain()` reports which direction
and which term did so:

```{code-cell} python
[f for f in sia.explain().findings if f.kind == "requirement_binding"]
```

## Indeterminism and grain

Any indeterminism provides a repertoire of alternatives, so probabilistic
systems have $\varphi_s > 0$ whenever they are integrated. The requirement can
still set the value: for the aB system above, $\mathit{ii}(s)$ is smaller
than $\min(\varphi_c, \varphi_e)$, so $\varphi_s$ is about $0.04$. Slight
noise suffices for existence; the three-unit noisy grid computes a small but
positive value:

```{code-cell} python
pyphi.analyze(pyphi.examples.grid3_substrate(), (0, 0, 0)).phi
```

Differentiation is distinct from indeterminism in the micro dynamics. It is
a requirement on the availability of alternative cause–effect states, and
alternatives can arise from the system's description and grain as well as
from noise: at a macro grain, many micro configurations may realize the same
macro state, and that degeneracy can give a macro unit a repertoire of
alternatives even when the underlying micro dynamics are nearly
deterministic (Sections 2.2 and 4; see {doc}`macro-units`).

## Scope of the requirement

The requirement applies to the system-level quantity $\varphi_s$. The
distinctions, the relations, and their summed structure integrated
information $\Phi$ are defined at the level of mechanisms and are computed
the same way whether or not the system's $\varphi_s$ is positive. The XOR
network's cause-effect structure:

```{code-cell} python
ces = analysis.ces
(len(ces.distinctions), ces.relations.num_relations(), float(ces.big_phi))
```

A system with $\varphi_s = 0$ is not a complex, so this structure is not
specified by any existing whole; it remains available for analysis and
comparison.

The minimum information partition is selected on the normalized integrated
information without the intrinsic-information term, and $\mathit{ii}(s)$
enters the minimum at the selected partition. Specified-state ties are
compared on $\varphi_s$ including the term, so a deterministic system's tied
readings compare equal at zero and the reported state is a canonical
representative, while readings tied at positive $\varphi_s$ escalate to
$\Phi$ (see {doc}`Control tie-breaking <../howto/tie-breaking>`).

## The 2023 formulation

Albantakis et al. (2023) define system integrated information as
$\varphi_s = \min\{\varphi_c, \varphi_e\}$, without the intrinsic-information
term; that paper's "intrinsic information" is the quantity called intrinsic
specification here, and the analysis card uses each formulation's own name
for it. The two formulations share the partition search, the specified-state
search, and every mechanism-level quantity, so they differ only where
$\mathit{ii}(s)$ is the smallest of the three terms: deterministic systems,
which the 2023 formulation assigns a positive value, and probabilistic systems
whose intrinsic information is below their integration. The three-XOR network
under the 2023 formulation:

```{code-cell} python
pyphi.analyze(xor, (0, 0, 0), formalism="IIT_4_0_2023").phi
```

Values published under the 2023 formulation, including those of the
deterministic examples in the IIT literature, reproduce under
`formalism="IIT_4_0_2023"` or `pyphi.config.override(**pyphi.iit4_2023)`;
{doc}`formalism-versions` describes selecting a formulation.
