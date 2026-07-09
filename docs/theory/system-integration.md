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

# System integrated information

Given a substrate and a candidate system, IIT asks whether that system exists as
*one* thing — irreducibly, from its own intrinsic perspective — and how much.
The answer is the **system integrated information** $\varphi_s$. This page covers
the three postulates that produce it: *information*, *integration*, and
*exclusion*.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False

substrate = pyphi.examples.basic_substrate()
analysis = pyphi.analyze(substrate, (1, 1, 0))
```

## Information: the maximal cause–effect state

By the **information** postulate, a system must be *specific*: in its current
state it selects a particular cause–effect state — the past it constrains and the
future it points to. Among the candidates, IIT selects the one with maximal
**intrinsic information** $\mathit{ii}$, which is the product of two competing
quantities (Albantakis et al., 2023, Fig 1C): *informativeness*, how much the
state deviates from chance, and *selectivity*, how concentrated that cause–effect
power is on a specific state. Their product is what the system's cause–effect
power is *about*.

## Integration: irreducibility over the minimum partition

By the **integration** postulate, having cause–effect power and selecting a state
is not enough: the system must specify its cause–effect state *irreducibly*,
as one set of units rather than as independent parts. Irreducibility is tested by
**partitioning** the system and measuring how much the partition reduces the
intrinsic information. The reduction is evaluated over the partition that makes
the *least* difference — the **minimum information partition** (MIP) — so that the
result reflects the system's weakest link (Albantakis et al., 2023).

Integrated information is computed separately on the cause side and the effect
side, $\varphi_c$ and $\varphi_e$, and the system integrated information is the
smaller of the two — a system is only as integrated as its weaker direction:

$$ \varphi_s = \min(\varphi_c, \varphi_e). $$

```{code-cell} python
(analysis.sia.cause.phi, analysis.sia.effect.phi, analysis.phi)
```

For the worked example the cause direction is the binding one:
$\varphi_c \approx 0.208$ is smaller than $\varphi_e \approx 0.415$, so
$\varphi_s = \varphi_c \approx 0.208$. The partition responsible — the MIP — is
carried on the analysis:

```{code-cell} python
analysis.sia.partition
```

`analysis.phi` is $\varphi_s$; `analysis.sia` is the full **system irreducibility
analysis** that produced it, including the MIP, the cause and effect sides, and a
size-normalized value (`analysis.sia.normalized_phi`) used when comparing systems
of different sizes.

## Selection margins

The definitions above make two kinds of discrete choice. The specified
cause and effect states are chosen by *maximizing* intrinsic information
over candidate states — the principle of maximal existence (Albantakis et
al., 2023, Eq. 12). The MIP is chosen by *minimizing* integrated
information over partitions; the minimization compares partitions on a
**normalized** value (each partition's $\varphi_s$ divided by the maximum
number of connections it could sever), while the reported $\varphi_s$ is
the **unnormalized** value at the winning partition (Albantakis et al.,
2023, Eqs. 22–23). Normalization ensures the comparison is fair across
partitions of different sizes; the reported quantity, once the MIP is
identified, is an absolute one.

Each such choice has a **margin**: the gap between the winner and the best
competitor, in the units of the comparison. PyPhi reports these on the
analysis — `partition_margin` (in normalized $\varphi$), the per-direction
specified-state margins (in intrinsic information), and `tied_selections`
naming any selection whose margin is within the configured numerical
precision of zero.

```{code-cell} python
(
    float(analysis.sia.partition_margin),
    {str(d): float(m) for d, m in analysis.sia.state_margins.items()},
    analysis.sia.tied_selections,
)
```

Margins are the theory-native form of sensitivity analysis. Because the
selections implement the principles of maximal and minimal existence, a
substrate near a selection boundary is near a point where *what exists* —
which state it specifies, where its weakest link lies — changes discretely,
even though the substrate's own parameters vary continuously. A small
margin flags exactly that proximity, which no derivative of $\varphi_s$
reveals. Exact zeros are ties, resolved by explicit, configurable rules;
see {doc}`Control tie-breaking </howto/tie-breaking>` for reading and
acting on margins, and
{doc}`Explore substrate parameter landscapes </howto/landscape>` for
converting them into distances in parameter space.

## Exclusion: the complex

Many overlapping subsets of a substrate may each have positive $\varphi_s$. The
**exclusion** postulate requires a *definite* subject: exactly one set of units
counts. IIT resolves this by keeping the set whose integrated information over
itself is maximal, $\varphi_s^{\ast}$ — the **maximal substrate**, or **complex**
(Albantakis et al., 2023). `pyphi.analyze` performs this search and reports the
complex it finds:

```{code-cell} python
analysis.system.node_indices  # the units of the complex
```

Here the complex is the whole three-unit substrate. With the complex fixed, the
final postulate — *composition* — unfolds its internal structure: the
[distinctions and relations](distinctions-and-relations.md) it specifies.
