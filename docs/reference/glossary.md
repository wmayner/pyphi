# Glossary

Terms as PyPhi and the IIT 4.0 papers use them. Each entry points to the
page that treats it. Where the [IIT wiki](https://www.iit.wiki/glossary) uses
a different word for the same thing, the entry says so.

```{glossary}
substrate
  A set of units and the probability of each unit's next state given the
  current state of all of them: a causal model. The wiki calls the full set
  of units under analysis the *network* and uses *substrate* for any
  candidate subset of it. {class}`~pyphi.substrate.Substrate`;
  {doc}`../theory/substrate-and-system`; {doc}`../howto/build-substrate`.

system
  A candidate subset of a substrate's units in a definite state, analyzed
  from its own perspective; the rest of the substrate supplies its
  {term}`background conditions`. {class}`~pyphi.system.System`;
  {doc}`../theory/substrate-and-system`.

background conditions
background
  The current state of the units outside a system. Under IIT 4.0 those
  units are causally marginalized (averaged out conditional on their
  current state); under the IIT 3.0 preset they are fixed at their current
  state. The `background_conditioning` option selects the rule;
  {doc}`../theory/substrate-and-system`.

transition probability matrix (TPM)
  The substrate's probabilities of next states given current states, in
  state-by-node form: one row per current state (first unit changing
  fastest), one column per unit. {doc}`../howto/build-substrate`;
  {ref}`tpm-conventions`.

little-endian
  PyPhi's state order: the first unit is the least significant bit, so the
  state `(1, 0, 0)` is row 1 of the matrix and `(0, 0, 1)` is row 4.
  {ref}`tpm-conventions`.

unit
  A constituent of a substrate at the grain it is observed and manipulated
  in. PyPhi units may have any finite number of states; the wiki's units
  are binary. {doc}`../conventions`.

repertoire
  A probability distribution over the states of a purview, as constrained
  by a mechanism or a system in its current state: the cause repertoire
  over past states, the effect repertoire over future states. Also the set
  of those states itself, as in a system's repertoire of alternatives.
  {doc}`../theory/distinctions-and-relations`.

mechanism
  A subset of the system's units, in their current state, considered for
  the cause–effect state it specifies. The wiki calls this a *candidate
  mechanism* and reserves *mechanism* for one that specifies a
  {term}`distinction`. {doc}`../theory/distinctions-and-relations`.

purview
  The subset of units over which a mechanism's cause or effect is assessed;
  for a relation, the units of its congruent overlap. In result cards a
  purview is written in the state it is specified in: uppercase ON,
  lowercase OFF. {doc}`../howto/read-result`.

order
  The number of units in a mechanism or purview: a first-order mechanism
  is a single unit. {doc}`../theory/distinctions-and-relations`.

cause–effect state
specified state
  The cause state and effect state that maximize intrinsic information:
  what a mechanism selects over a purview, or a system selects over itself.
  Only distinctions whose specified states are {term}`congruent` with the
  system's cause–effect state belong to its Φ-structure.
  {doc}`../theory/system-integration`;
  {doc}`../theory/distinctions-and-relations`.

congruent
  Two specified states agree on every unit they share. A distinction must
  be congruent with the system's cause–effect state, and a relation exists
  only where purviews overlap congruently.
  {doc}`../theory/distinctions-and-relations`.

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
  A set of cuts that severs the connections between parts of a system,
  between a mechanism and its purview, or among the distinctions of a
  relation, to test irreducibility. {doc}`../theory/system-integration`.

minimum partition (MIP)
  The partition that makes the least difference, judged on normalized φ:
  the system's weakest link. Earlier papers call it the minimum information
  partition. Reported on every system irreducibility analysis.
  {doc}`../theory/system-integration`.

normalized φ
  A partition's φ divided by the maximum number of connections it could
  sever. The minimum partition is chosen on this value; the reported φ is
  the unnormalized value at the winning partition.
  {doc}`../theory/system-integration`.

partition scheme
  The family of partitions a search ranges over: one setting for system
  partitions and one for mechanism partitions, fixed by the formalism
  preset. {doc}`../theory/computational-complexity`.

integrated information (φ)
  How much a partition changes what is specified. φ_d for a distinction,
  φ_r for a relation, φₛ for a system. {doc}`../theory/overview`.

system integrated information (φₛ)
  The irreducibility of a system's specified cause–effect state over its
  minimum partition; whether the system exists as one whole.
  `analysis.phi`. {doc}`../theory/system-integration`.

system irreducibility analysis (SIA)
  The record of one system's integration test: its specified cause–effect
  state, intrinsic information, minimum partition, φ_c, φ_e, and φₛ.
  `analysis.sia`; {doc}`../howto/read-result`.

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
  A congruent overlap among the purviews of one or more distinctions (the
  same units specified in the same states), with its φ_r. A single
  distinction whose cause and effect purviews overlap congruently has a
  self-relation. {doc}`../theory/distinctions-and-relations`;
  {doc}`../howto/query-relations`.

relation face
  The congruent overlap of a subset of a relation's purviews; a component
  of the relation. {doc}`../howto/query-relations`.

degree
  Of a relation, the number of distinctions it binds; a self-relation has
  degree 1. Of a relation face, the number of purviews that overlap.
  {doc}`../howto/query-relations`.

Φ-structure
cause–effect structure (CES)
  The distinctions a complex specifies and the relations among them.
  IIT 3.0's name is *conceptual structure*.
  {class}`~pyphi.models.ces.CauseEffectStructure`; {doc}`../theory/phi-structure`.

unfolding
  Computing the Φ-structure a complex specifies: its distinctions and the
  relations among them. {doc}`../theory/phi-structure`.

complex
  A set of units whose φₛ is maximal among all sets overlapping it; the
  exclusion postulate's answer to which units exist as one whole.
  {meth}`~pyphi.substrate.Substrate.complexes`; {doc}`../tutorials/recursive-exclusion`.

condensation
  The recursive exclusion cascade that splits a substrate into disjoint
  complexes: take the system with maximal φₛ, set its units aside, and
  repeat on the rest. The wiki calls the first the major complex and the
  others minor complexes. {mod}`pyphi.condensation`;
  {doc}`../tutorials/recursive-exclusion`.

maximally irreducible cause and effect (MICE)
  For a mechanism: the cause purview and effect purview with the highest
  φ, the search a distinction's computation performs.
  {doc}`../theory/distinctions-and-relations`.

selection margin
  How far the winning partition (or state) was from its nearest competitor;
  zero means an effective tie. {doc}`../howto/tie-breaking`.

tie
  Two candidates (states, purviews, partitions, or systems) whose values
  agree at the configured precision. Ties are resolved by the papers'
  cascades and reported with a zero margin. {doc}`../howto/tie-breaking`.

formalism
  The set of rules that turns a system into results: IIT 4.0 (2026), IIT
  4.0 (2023), IIT 3.0, or actual causation. {doc}`../theory/formalism-versions`.

preset
  A complete formalism configuration: `iit3`, `iit4_2023`, or `iit4_2026`
  in `pyphi.conf.presets`, applied with `config.override(**preset)`;
  `analyze(..., formalism="IIT_4_0_2026")` selects one for a single run.
  {doc}`../theory/formalism-versions`.

actual causation
  The analysis of what caused what in one observed transition, measured in
  α (bits); a separate formalism from φ and Φ.
  {doc}`../tutorials/actual-causation`.

grain
  The scale at which units are defined: the constituent grain (which
  smaller units make up a unit) and the update grain (how many of their
  updates one step of the unit spans). The exclusion postulate picks the
  grain that maximizes φₛ. {doc}`../theory/macro-units`.

macro unit
  A unit defined over several micro units, by coarse-graining at update
  grain 1 or by blackboxing over a window of updates, admitted when it is
  maximally irreducible within. {doc}`../theory/macro-units`.
```
