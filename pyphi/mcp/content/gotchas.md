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

## 2. φₛ = 0 means the system does not exist as one whole, not "no structure"

A zero value of φₛ means the system does not exist as one integrated whole.
There are two ways to reach it. Either the system is **reducible**: some
partition makes no difference. Or the system provides itself no repertoire of
alternatives (ii(s) = 0), which sets φₛ to 0 even when φ_c and φ_e are both
positive; the `requirement_binding` key of the `analyze` summary says when
this is the case. Neither means the system is empty
or uninteresting. Feed-forward systems, for instance, have φₛ = 0 by
construction: the way the minimum partition is defined (Albantakis et al. 2023,
Eq. 23) ensures φₛ = 0 for any system that is not strongly connected.

## 3. φₛ and Φ are different quantities

`analyze(...).phi` (and `.sia.phi`) is **φₛ**, system integrated information:
whether the system exists as one whole, computed as the minimum of the cause
side, the effect side (both over the normalized minimum partition), and the
system's intrinsic information ii(s).
`analyze(...).big_phi` (and `.ces.big_phi`) is **Φ**, structure integrated
information: the plain sum of φ over all distinctions and relations. φₛ decides
existence; Φ measures the quantity of structure. Do not report one as if it were
the other. The result card prints both, labelled `φ_s` and `Φ`; the `analyze`
tool's summary returns them as `system_phi` and `big_phi`.

## 4. Ties are common in small toy networks

Ties for maximal φ arise from **symmetries in the transition probability
matrix**, and they are frequent in small, deterministic toy models — exactly
the networks a newcomer reaches for first. IIT resolves them by appealing to
the subsequent postulates: tied system states are compared on φₛ and then on
the structure integrated information Φ; a distinction's tied readings are
filtered to those congruent with the system's cause–effect state, and any
remaining choice is the combination of readings that maximizes Φ. If candidate
systems or system states still tie in Φ, the system does not qualify as a
complex, unless the tied cause–effect structures are identical from the
intrinsic perspective (Albantakis et al. 2023, S1 Text). PyPhi's tie-resolution
settings pick one reading so that the computation can proceed. If a small
example gives a surprising or unstable result, suspect a tie. In realistic
systems with graded connection strengths and some indeterminism, ties are less
likely.

## 5. Results from earlier versions of IIT are not comparable

PyPhi can also compute earlier versions of IIT, to reproduce published results.
Their φ values are defined differently and cannot be compared with values
computed under IIT; read `get_iit_reference("configuration")` before running or
interpreting one.

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

## 9. A system must be in a state it could have reached

IIT evaluates a system that *is* in a state, and a state with
no possible predecessor has no cause. PyPhi refuses such a state: `analyze` raises
`StateUnreachableForwardsError`. The substrate's state must have a possible
predecessor (its marginal probability must be positive). Because IIT causally
marginalizes the background (Albantakis et al. 2023, Eq. 4), that is
sufficient.

This is common in small deterministic toy models. In the 3-node XOR network
every unit is the XOR of the other two, so every state the network can produce
has even parity. The four odd-parity states — `(1,0,0)`, `(0,1,0)`, `(0,0,1)`,
`(1,1,1)` — have no preimage and each raises. They are states the units can be
*put into* by an outside intervention, but not states the system can have
arrived at on its own, so IIT assigns them no Φ-structure rather than assigning
one that would rest on a past that could not have happened. This is a different
error from actual causation's `TransitionUnreachableError` (§11), which is about
a *pair* of states.

## 10. Cost grows very fast

The computation is exponential in the number of units (roughly O(n⁵·3ⁿ)), and
the number of *possible relations* grows doubly-exponentially (2^(2^N−1)−1).
The practical ceiling for an exact analysis of the full cause-effect structure
is about 10–12 units, depending on the substrate topology and the machine specs,
and a full cause–effect structure is far more expensive than system integrated
information alone. A concrete Φ-structure can be megabytes. For relation totals,
the analytical backend (the default) does not enumerate relations; for
individual relations, cap the degree. The `analyze` tool refuses large full/CES
requests unless `confirm_large=True`.

## 11. Actual causation answers a different question

IIT proper asks about a system's **potential** cause–effect power over all
possible states (φₛ, Φ, complexes). **Actual causation** asks about a single
realized transition — *what actually caused what, this time* (token causation).
It operates on a `Transition`, measures link strength as α (alpha) in bits
(`alpha_measure="PMI"`), and evaluates causes and effects independently. Do not
mix its quantities with φₛ/Φ; they answer different questions. The transition
must be realizable: constructing a `Transition` whose effect occurrence has
zero probability — or calling `causal_nexus()` and related entry points with a
state pair the TPM says cannot occur — raises `TransitionUnreachableError`.
