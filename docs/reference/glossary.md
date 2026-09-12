# Glossary

Terms as PyPhi and the IIT 4.0 papers use them. Each entry points to the
page that treats it.

```{glossary}
substrate
  A set of units and the probability of each unit's next state given the
  current state of all of them: a causal model. {class}`~pyphi.substrate.Substrate`;
  {doc}`../theory/substrate-and-system`; {doc}`../howto/build-substrate`.

system
  A candidate subset of a substrate's units in a definite state, analyzed
  from its own perspective; the rest of the substrate is its
  {term}`background`. {class}`~pyphi.system.System`;
  {doc}`../theory/substrate-and-system`.

background
  The units outside a system. Under IIT 4.0 they are causally marginalized
  (averaged out conditional on the current state); under the IIT 3.0 preset
  they are fixed at their current state. The `background_conditioning`
  option selects the rule; {doc}`../theory/substrate-and-system`.

transition probability matrix (TPM)
  The substrate's probabilities of next states given current states, in
  state-by-node form: one row per current state (first unit changing
  fastest), one column per unit. {doc}`../howto/build-substrate`;
  {ref}`tpm-conventions`.

little-endian
  PyPhi's state order: the first unit is the least significant bit, so the
  state `(1, 0, 0)` is row 1 of the matrix and `(0, 0, 1)` is row 4.
  {ref}`tpm-conventions`.

repertoire
  A probability distribution over the states of a purview: the cause
  repertoire over past states, the effect repertoire over future states,
  as constrained by a mechanism in its state.
  {doc}`../theory/distinctions-and-relations`.

mechanism
  A subset of the system's units, in their current state, considered for
  the cause–effect state it specifies.
  {doc}`../theory/distinctions-and-relations`.

purview
  The subset of units over which a mechanism's cause or effect is assessed.
  In result cards a purview is written in the state it is specified in:
  uppercase ON, lowercase OFF. {doc}`../howto/read-result`.

specified state
  The purview (or system) state that maximizes intrinsic information: the
  cause and effect the mechanism (or system) selects.
  {doc}`../theory/system-integration`.

intrinsic information
  For a mechanism or a system: how selectively and informatively it picks
  out its specified state. Under IIT 4.0 (2026) the *system's* intrinsic
  information ii(s) is the smaller of its {term}`intrinsic specification`
  and {term}`intrinsic differentiation` over both directions.
  {doc}`../theory/intrinsic-information`.

intrinsic specification
  The selectivity times informativeness of the specified state (Mayner et
  al. 2026, Eqs. 7 and 9); Albantakis et al. (2023) call the same quantity
  intrinsic information, and the card follows the formalism's own name.
  {doc}`../theory/intrinsic-information`.

intrinsic differentiation
  The surprisal of the specified state: how much of a repertoire of
  alternatives the system provides itself. Zero for a deterministic
  transition. {doc}`../theory/intrinsic-information`.

intrinsic-information requirement
  Under IIT 4.0 (2026), φₛ = min{φ_c, φ_e, ii(s)}: the system's integrated
  information cannot exceed its intrinsic information (Mayner et al. 2026,
  Eq. 23). {doc}`../theory/intrinsic-information`.

partition
  A cut of a system or mechanism into parts whose connections are severed,
  to test irreducibility. {doc}`../theory/system-integration`.

minimum information partition (MIP)
  The partition that makes the least difference, judged on normalized φ:
  the system's weakest link. Reported on every system irreducibility
  analysis. {doc}`../theory/system-integration`.

integrated information (φ)
  How much a partition changes what is specified. φ_d for a distinction,
  φ_r for a relation, φₛ for a system. {doc}`../theory/overview`.

system integrated information (φₛ)
  The irreducibility of a system's specified cause–effect state over its
  minimum information partition; whether the system exists as one whole.
  `analysis.phi`. {doc}`../theory/system-integration`.

structure integrated information (Φ)
  The sum of φ over every distinction and relation of a Φ-structure; how
  much structure the system specifies. `analysis.big_phi`; not defined
  under IIT 3.0, whose Φ is the system-level value.
  {doc}`../theory/phi-structure`.

distinction
  A mechanism together with the cause and effect states it irreducibly
  specifies over its maximally irreducible purviews, with its φ_d.
  IIT 3.0's name is *concept*. {doc}`../theory/distinctions-and-relations`.

relation
  A congruent overlap among the purviews of two or more distinctions (the
  same units specified in the same states), with its φ_r.
  {doc}`../theory/distinctions-and-relations`; {doc}`../howto/query-relations`.

Φ-structure (cause–effect structure)
  The distinctions a complex specifies and the relations among them.
  {class}`~pyphi.models.ces.CauseEffectStructure`; {doc}`../theory/phi-structure`.

complex
  A set of units whose φₛ is maximal among all sets overlapping it; the
  exclusion postulate's answer to which units exist as one whole.
  {meth}`~pyphi.substrate.Substrate.complexes`; {doc}`../tutorials/recursive-exclusion`.

maximally irreducible cause and effect (MICE)
  For a mechanism: the cause purview and effect purview with the highest
  φ, the search a distinction's computation performs.
  {doc}`../theory/distinctions-and-relations`.

selection margin
  How far the winning partition (or state) was from its nearest competitor;
  zero means an effective tie. {doc}`../howto/tie-breaking`.

formalism
  The set of rules that turns a system into results: IIT 4.0 (2026), IIT
  4.0 (2023), IIT 3.0, or actual causation. {doc}`../theory/formalism-versions`.

actual causation
  The analysis of what caused what in one observed transition, measured in
  α (bits); a separate formalism from φ and Φ.
  {doc}`../tutorials/actual-causation`.

macro unit
  A unit defined over several micro units by coarse-graining or
  blackboxing, admitted when it is maximally irreducible within.
  {doc}`../theory/macro-units`.
```
