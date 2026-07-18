# IIT and PyPhi gotchas

The subtleties that most often lead to wrong results or wrong interpretations.

## 1. States are little-endian

PyPhi orders states so the **first node is the least-significant bit** — the
opposite of ordinary positional notation. In a 3-node system, state `(0, 0, 1)`
(only the third node on) is row index 4 of a state-by-node TPM, and `(1, 0, 0)`
(only the first node on) is row index 1. "The first node varies fastest." When
flattening a repertoire array to one dimension, use Fortran (column-major)
order. This is the most common source of indexing mistakes. Always pass and
read states as tuples in node order, e.g. `(1, 1, 0)`, and let PyPhi handle the
indexing.

## 2. Φ = 0 means reducible, not "no structure"

A zero value of φₛ or Φ means the system is **reducible** — some partition makes
no difference, so it does not exist as one integrated whole. It does not mean
the system is empty or uninteresting. Feed-forward systems, for instance, have
Φ = 0 by construction.

## 3. φₛ and Φ are different quantities

`analyze(...).phi` (and `.sia.phi`) is **φₛ**, system integrated information:
whether the system exists as one whole, computed as the minimum over the cause
and effect sides, over the normalized minimum information partition.
`.ces.big_phi` is **Φ**, structure integrated information: the plain sum of φ
over all distinctions and relations. φₛ decides existence; Φ measures the
quantity of structure. Do not report one as if it were the other.

## 4. Ties are common in small toy networks

Ties for maximal φ arise from **symmetries in the transition probability
matrix**, and they are frequent in small, deterministic toy models — exactly
the networks a newcomer reaches for first. IIT resolves them by descending the
postulates (maximize φₛ, then Φ, then congruence with the system state, then
the number of relations, which favors larger purviews). When a tie cannot be
resolved, the cause–effect structure is genuinely non-unique. If a small
example gives a surprising or unstable result, suspect a tie. In realistic
systems with graded connection strengths and some indeterminism, ties are less
likely.

## 5. The formalism versions differ, numerically and conceptually

Selected with the `formalism` argument to `analyze`:

- **Measure.** IIT 3.0 uses the **earth mover's distance** (a sum that weights
  distant states more); IIT 4.0 uses the **intrinsic difference** (a max over a
  single state). The same substrate gives different φ under each.
- **Relations.** IIT 3.0 has **no relations** — they are a 4.0 addition. A 3.0
  result has concepts, not distinctions and relations.
- **Background conditioning.** IIT 4.0 causally marginalizes background units
  (`background_conditioning="CAUSAL_MARGINALIZATION"`); the 3.0 preset conditions
  on fixed states (`"CONDITION_CURRENT_STATE"`). This only affects proper-subset
  systems (a system smaller than the whole substrate).
- **The 2026 differentiation cap.** IIT 4.0 (2026) requires the system to
  provide itself a repertoire of alternatives (intrinsic differentiation). A
  fully deterministic system provides none, so its φₛ is 0 under 2026 even when
  it is positive under 2023.

## 6. Multi-valued (k-ary) units

Units may have more than two states (set with `alphabet`). Then the number of
states is the **product** of the per-node alphabet sizes, not 2ⁿ. Two
consequences worth stating: **more states does not mean more Φ** — adding
causally irrelevant states can lower it — and **binarizing a multi-valued model
generally does not preserve its causal structure**, so a Boolean approximation
of a k-ary network can have a very different Φ. The earth mover's distance is
unavailable for k-ary systems; use the intrinsic-difference family.

## 7. A wrong connectivity matrix gives a wrong Φ

The connectivity matrix is an optimization: absent edges let PyPhi rule out
influence and skip work. But an *incorrect* connectivity matrix produces an
incorrect result, not merely a slower one. When unsure of the wiring, omit the
connectivity matrix entirely — PyPhi then assumes full connectivity, which is
always correct (just slower).

## 8. The TPM must be interventional

IIT's analysis is defined on a transition probability matrix obtained by
**intervention** (perturbing the units into every state and recording the
outcome — the do-operator), and the units must be conditionally independent
given the previous state. A TPM fit from observed time-series data (e.g. an EEG
recording) without checking these properties is not a technically valid input,
though users may nonetheless want to conduct analyses on such data. If the user
appears unaware of this caveat, alert them.

## 9. Cost grows very fast

The computation is exponential in the number of units (roughly O(n·5³ⁿ)), and
the number of *possible relations* grows doubly-exponentially (2^(2^(N−1))−1).
The practical ceiling for an exact analysis of the full cause-effect structure
is about 10–12 units, depending on the substrate topology and the machine specs,
and a full cause–effect structure is far more expensive than system integrated
information alone. A concrete Φ-structure can be megabytes. For relation totals,
the analytical backend (the default) does not enumerate relations; for
individual relations, cap the degree. The `analyze` tool refuses large full/CES
requests unless `confirm_large=True`.

## 10. Actual causation answers a different question

IIT proper asks about a system's **potential** cause–effect power over all
possible states (φₛ, Φ, complexes). **Actual causation** asks about a single
realized transition — *what actually caused what, this time* (token causation).
It operates on a `Transition`, measures link strength as α (alpha) in bits
(`alpha_measure="PMI"`), and evaluates causes and effects independently. Do not
mix its quantities with φₛ/Φ; they answer different questions. The transition
must be realizable: constructing a `Transition` whose effect occurrence has
zero probability — or calling `causal_nexus()` and related entry points with a
state pair the TPM says cannot occur — raises `TransitionUnreachableError`.
