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

PyPhi's default formalism, IIT 4.0 (2026), refines the account of system
integrated information given on the preceding pages (Mayner, Marshall, and
Tononi, 2026). The motivation for the refinement is conceptual: the
intrinsicality postulate requires that a system's cause–effect power be
assessed from the system's own perspective, and to have cause–effect power
intrinsically a system must satisfy two complementary requirements. It must
provide itself with a repertoire of alternative cause–effect states —
intrinsic **differentiation** — and it must specify one of those alternatives
— intrinsic **specification**. The 2023 formulation quantified the second
requirement; the 2026 refinement makes the first explicit in the measure. The
two requirements trade off, and both are assessed by the system's **intrinsic
information** $\mathit{ii}(s)$, which enters the minimum that defines the
system integrated information:

$$ \varphi_s = \min\{\varphi_c,\ \varphi_e,\ \mathit{ii}(s)\}. $$

This is Eq. 23 of Mayner, Marshall, and Tononi (2026), "Intrinsic
Cause–Effect Power: The Tradeoff Between Differentiation and Specification"
(*Entropy* 28, 410). The paper defines $\mathit{ii}(s)$ as the minimum,
across the cause and effect directions, of each direction's intrinsic
information, which is itself the minimum of that direction's intrinsic
differentiation $i^{c/e}_{\mathrm{diff}}(s)$ and intrinsic specification
$i^{c/e}_{\mathrm{spec}}(s)$ (Section 2.3, immediately preceding Eq. 13):

$$
\mathit{ii}(s) = \min\{\mathit{ii}_c(s),\ \mathit{ii}_e(s)\}, \qquad
\mathit{ii}_{c/e}(s) = \min\{i^{c/e}_{\mathrm{diff}}(s),\ i^{c/e}_{\mathrm{spec}}(s)\}.
$$

This page explains the differentiation requirement, follows the measure
through a small example, and describes what the requirement does and does
not change.

## Differentiation and determinism

The requirement is easiest to see in the paper's opening example (Section 2):
a single unit implementing deterministic COPY logic. From the outside, an
experimenter can set the unit to each of its states in turn, observe that it
copies them, and conclude that it has cause–effect power. From the unit's own
perspective, however, its current state admits exactly one past state and one
future state; no alternatives are available to it, so there is no difference
for it to make to itself. Intrinsic differentiation quantifies the
availability of such alternatives: like entropy, it is zero for a perfectly
deterministic system and increases with decreasing determinism (Section 2.2).

Specification behaves in the opposite way. As the paper puts it: "Purely
deterministic systems provide no genuine alternatives, and thus their
intrinsic differentiation is zero, while purely random systems specify no
state, leaving intrinsic specification at zero" (Section 4). Under the 2026
formalism a deterministic system therefore has $\varphi_s = 0$, a maximally
noisy system likewise has $\varphi_s = 0$, and positive intrinsic information
requires a balance of the two.

The three-XOR network illustrates the deterministic case. Under the 2023
formalism, which does not include the intrinsic-information requirement, it
is highly integrated:

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

xor = pyphi.examples.xor_substrate()
pyphi.analyze(xor, (0, 0, 0), formalism="IIT_4_0_2023").phi
```

Under the default it computes zero:

```{code-cell} python
analysis = pyphi.analyze(xor, (0, 0, 0))
analysis.phi
```

The analysis records exactly where the zero comes from. Both directions have
substantial integration and a fully specified state, but zero
differentiation:

```{code-cell} python
sia = analysis.sia
print("φ_c =", float(sia.cause.phi), "  φ_e =", float(sia.effect.phi))
print("differentiation:",
      {str(d): float(v) for d, v in sia.intrinsic_differentiation.items()})
```

$\varphi_c = 1.5$ and $\varphi_e = 3.0$ are the same values as in the 2023
analysis. The effect-side differentiation is $0$ — the deterministic
transition offers no alternative effect — while the cause side records the
network's two-fold predecessor degeneracy (each state is reachable from
exactly two prior states, so $-\log_2 \tfrac{1}{2} = 1$). (The cause side is
evaluated on the Bayesian posterior over prior states — Eqs. 6 and 11 — so it
measures predecessor degeneracy and can stay positive even for deterministic
dynamics; the effect side alone suffices to bring the minimum to zero.) The
minimum over directions — and with it $\varphi_s$ — is $0$.

## Reproducing published values

Many example systems in the IIT literature are deterministic — the XOR
network above, the OR/COPY/XOR gates of `basic`, the cellular-automaton
rules — and their published nonzero $\varphi_s$ values are quantities of the
2023 formulation. Under the 2026 default these systems compute $\varphi_s =
0$. To reproduce a published value, compute it under the formalism it was
published with, as in the first cell above or with
`pyphi.config.override(**pyphi.conf.presets.iit4_2023)`; see
{doc}`formalism-versions`.

Any indeterminism provides a repertoire of alternatives. The paper's own
worked example — the {doc}`Fig 1A logistic network <../tutorials/worked-example>`
used throughout this section — is probabilistic, which is why its published
values are identical under 2023 and 2026. Even slight noise suffices: the
three-unit noisy grid computes a small but positive value under the default.

```{code-cell} python
pyphi.analyze(pyphi.examples.grid3_substrate(), (0, 0, 0)).phi
```

Differentiation is distinct from indeterminism in the micro dynamics,
however. It is a requirement on the availability of alternative cause–effect
states, and alternatives can arise from the system's description and grain as
well as from noise: at a macro grain, many micro configurations may realize
the same macro state, and that degeneracy can give a macro unit a repertoire
of alternatives even when the underlying micro dynamics are nearly
deterministic (Sections 2.2 and 4; see {doc}`macro-units`).

## What the requirement does and does not change

The requirement applies to the *system-level* quantity only. Mechanism-level
quantities — distinctions, relations, and their summed structure $\Phi$ — are
computed exactly as in 2023, so the XOR network's cause-effect structure is
as rich as ever:

```{code-cell} python
ces = analysis.ces
(len(ces.distinctions), ces.relations.num_relations(), float(ces.big_phi))
```

What changes is the system's claim to existence: with $\varphi_s = 0$, the
candidate is not a complex, so under the 2026 formalism this structure is not
specified by any existing whole. The structure remains available for analysis
and comparison; the theory's verdict on the deterministic system itself is
$\varphi_s = 0$.

The minimum information partition is also unaffected: the MIP is selected on
the normalized integrated information without the intrinsic-information term,
exactly as in 2023, and $\mathit{ii}(s)$ enters the minimum only at the
selected partition. Specified-*state* ties, however, are compared on
$\varphi_s$ as each formalism defines it — under 2026, the value including
the intrinsic-information term — so a deterministic system's tied readings
all compare equal at zero and the reported state is a canonical
representative, while readings tied at positive $\varphi_s$ escalate to the
structure integrated information $\Phi$ (see
{doc}`Control tie-breaking <../howto/tie-breaking>`, "System-state ties").

For choosing between formalism versions — and reproducing published 2023 or
IIT 3.0 numbers — see {doc}`formalism-versions`.
