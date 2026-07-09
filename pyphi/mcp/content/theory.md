# IIT theory: what the numbers mean

This is a working understanding of Integrated Information Theory sufficient to
use PyPhi correctly and explain its results. For the equations, see
`get_iit_reference("equations")`; for the subtleties, `get_iit_reference("gotchas")`.
The authoritative source is Albantakis et al. (2023), *Integrated information
theory (IIT) 4.0*, PLoS Computational Biology 19(10): e1011465.

## The starting point: cause–effect power

IIT takes physical existence to mean having **cause–effect power** — the
ability to take and make a difference. A **substrate** is a set of interacting
units whose cause–effect power is captured entirely by its **transition
probability matrix** (TPM): the probability of each next state given each
current state. PyPhi requires the units to be **conditionally independent**
given the previous state, which is what lets the substrate be described by
per-unit transition probabilities rather than joint ones.

## The postulates

IIT derives its formalism from postulates, each the physical counterpart of a
property of experience:

- **Existence** — the units must have cause–effect power.
- **Intrinsicality** — that power must be assessed from the system's own
  perspective. Units outside the system are treated as fixed background and are
  *causally marginalized* so they do not contribute.
- **Information** — the system in its state must select a specific cause–effect
  state, the one with maximal **intrinsic information** (ii).
- **Integration** — the system must specify that state *irreducibly*, as one
  whole. Irreducibility is measured over the **minimum information partition**
  (MIP) — the partition that makes the least difference.
- **Exclusion** — exactly one set of units counts: the one with maximal φₛ,
  called the **complex**.
- **Composition** — subsets of the units (**mechanisms**) specify their own
  cause–effect states over subsets of units (**purviews**); these are the
  **distinctions**, and their congruent overlaps are the **relations**.

## φₛ versus Φ — two different quantities

This distinction is the single most important thing to keep straight.

**φₛ (system integrated information)** answers *does this system exist as one
integrated whole?* It is computed on the cause side and the effect side
separately, and φₛ is the smaller of the two — a system is only as integrated
as its weaker direction. It is evaluated over the *normalized* minimum
information partition, and it is **not** compositional. The set of units that
maximizes φₛ over itself is the complex. φₛ = 0 means the system is reducible.

**Φ (structure integrated information)** answers *how much structure does the
complex specify?* Once the complex is fixed, its Φ-structure is unfolded, and Φ
is the plain **sum** of the integrated information of every distinction and
every relation. There is no partition and no normalization in Φ; it is a sum
over content.

So: φₛ decides existence-as-one; Φ measures quantity-of-content. A system can
have modest φₛ but large Φ, or vice versa. In PyPhi, `analyze(...).phi` and
`.sia.phi` are φₛ; `.ces.big_phi` is Φ.

## Distinctions and relations

A **distinction** is a mechanism together with the specific cause state and
effect state it specifies, over its maximally irreducible cause and effect
purviews, carrying an integrated information value φ_d. Only distinctions whose
states are *congruent* with the complex's own cause–effect state belong to it.

A **relation** is a congruent overlap among the purviews of two or more
distinctions — the same units in the same states — with its own integrated
information φ_r. Relations are how the distinctions are bound together. The
number of *possible* relations grows doubly-exponentially in the number of
units, which is why a full Φ-structure is expensive and large; PyPhi can
compute relation totals analytically without enumerating them.

## The pipeline in PyPhi

```
Substrate  →  System (a candidate subset in a state)  →  formalism  →  Φ-structure
```

`pyphi.analyze(substrate, state)` runs the whole thing: it finds the complex,
computes φₛ, and unfolds the Φ-structure. The MCP `analyze` tool wraps this.

## Formalism versions

- **IIT 4.0 (2023)** — the default. φₛ = min(φ_c, φ_e); the cause–effect
  structure has distinctions and relations; the distance measure is the
  **intrinsic difference**.
- **IIT 4.0 (2026)** — refines φₛ to require **intrinsic differentiation**: the
  system must also provide itself with a repertoire of alternatives. A fully
  deterministic system provides none, so its φₛ falls to 0 under this version.
- **IIT 3.0 (2014)** — the earlier formalism. It computes *concepts* (not
  distinctions and relations — 3.0 has **no relations**), uses the **earth
  mover's distance** rather than the intrinsic difference, and defines its
  quantities differently, so the same substrate gives different numbers.

Select a version with the `formalism` argument to `analyze`. **Actual
causation** is a separate formalism answering a different question — see the
gotchas reference.
