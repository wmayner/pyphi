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

# The intrinsic-information cap

PyPhi's default formalism, IIT 4.0 (2026), refines the system integrated
information of the preceding pages in one way: beyond *specifying* an
irreducible cause–effect state, a system must provide itself with a
*repertoire of alternatives* — intrinsic **differentiation** (Mayner,
Marshall, and Tononi, 2026). The two requirements trade off, and the system
integrated information is capped by their minimum, the system's **intrinsic
information** $\mathit{ii}(s)$:

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

This page shows the cap in action, and its most consequential theorem:
**a deterministic system has $\varphi_s = 0$.**

## Determinism means zero differentiation

A deterministic system in a state pins its cause and effect down completely —
maximal specification. But differentiation asks the opposite question: what
repertoire of alternatives does the system furnish itself? A deterministic
transition offers exactly one effect, so the effect-side intrinsic
differentiation is zero, the cap binds at zero, and $\varphi_s = 0$ — however
tightly the units are wired together. (The cause side is evaluated on the
Bayesian posterior over prior states — Eqs. 6 and 11 — so it measures
*predecessor degeneracy* and can stay positive even for deterministic
dynamics; the effect side alone suffices to force the minimum to zero.) The paper states the property directly: "Purely deterministic
systems provide no genuine alternatives, and thus their intrinsic
differentiation is zero, while purely random systems specify no state,
leaving intrinsic specification at zero" (Section 4).

The classic three-XOR network makes this concrete. Under the *uncapped* 2023
formalism it is the textbook integrated system:

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

The analysis records exactly where the zero comes from. Both directions carry
substantial integration and a fully specified state — and zero
differentiation:

```{code-cell} python
sia = analysis.sia
print("φ_c =", float(sia.cause.phi), "  φ_e =", float(sia.effect.phi))
print("differentiation:",
      {str(d): float(v) for d, v in sia.intrinsic_differentiation.items()})
```

$\varphi_c = 1.5$ and $\varphi_e = 3.0$ survive from the 2023 analysis. The
effect-side differentiation is $0$ — the deterministic transition offers no
alternative effect — while the cause side records the network's two-fold
predecessor degeneracy (each state is reachable from exactly two prior
states, so $-\log_2 \tfrac{1}{2} = 1$). The minimum over directions — and
with it $\varphi_s$ — is $0$.

## This is a theorem, not a bug

Almost the entire classic IIT teaching repertoire is deterministic — the XOR
network above, the `basic` OR/COPY/XOR gates, the cellular-automaton rules —
so under the default formalism *all of them* compute $\varphi_s = 0$. If you
port an analysis from the literature or from an earlier PyPhi and see zero
where a paper printed $1.5$: nothing is broken. The system is deterministic,
the default formalism is the 2026 refinement, and the published number is the
uncapped 2023 quantity — still available by pinning that formalism, as in the
first cell above or with `pyphi.config.override(**pyphi.conf.presets.iit4_2023)`.

Any indeterminism restores a repertoire of alternatives. The paper's own
worked example — the {doc}`Fig 1A logistic network <../tutorials/worked-example>`
threaded through this section — is probabilistic, which is why its published
values are identical under 2023 and 2026. Even slight noise suffices: the
three-unit noisy grid computes a small but positive value under the default.

```{code-cell} python
pyphi.analyze(pyphi.examples.grid3_substrate(), (0, 0, 0)).phi
```

## What the cap does and does not change

The cap applies to the *system-level* quantity only. Mechanism-level
quantities — distinctions, relations, and their summed structure $\Phi$ — are
computed exactly as in 2023, so the XOR network's cause-effect structure is
as rich as ever:

```{code-cell} python
ces = analysis.ces
(len(ces.distinctions), len(ces.relations), float(ces.big_phi))
```

What changes is the system's claim to existence: with $\varphi_s = 0$, the
candidate is not a complex, so under the 2026 formalism this structure is not
specified by any existing whole. The structure remains available for analysis
and comparison; the theory's verdict on the deterministic system itself is
$\varphi_s = 0$.

The minimum information partition is also unaffected: the MIP is selected on
the *uncapped* normalized integrated information, exactly as in 2023, and the
cap is applied once to the selected partition's value. Margins and
tie-breaking therefore behave identically across the two formalisms (see
{doc}`Control tie-breaking <../howto/tie-breaking>`).

For choosing between formalism versions — and reproducing published 2023 or
IIT 3.0 numbers — see {doc}`formalism-versions`.
