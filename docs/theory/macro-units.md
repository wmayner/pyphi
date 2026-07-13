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

# Macro units and grains

A substrate's cause–effect power need not be maximal at the grain of its
smallest parts. Grouping micro units into coarser **macro units**, and reading
their state over a window of several micro updates, can raise a system's
integrated information $\varphi_s$ by orders of magnitude. IIT takes this
seriously: the units that actually exist for a substrate are the ones — at
whichever spatial grouping and temporal window — that *maximize* $\varphi_s$,
its **intrinsic units** (Marshall et al., 2024). This page covers what a macro
unit is, the criteria a candidate unit must satisfy, and how candidates at
different grains compete in a single exclusion cascade. It maps each notion onto
the types in {mod}`pyphi.macro`; for a worked walkthrough see the
{doc}`intrinsic-units tutorial <../tutorials/macro>`, and for running and
bounding the search see {doc}`Search across grains <../howto/grain-search>`.

## Units at a grain

A macro unit is a coarser unit built from finer ones. Formally it is a tuple of
its direct constituents, an update grain, and a mapping — the core of the unit
tuple of Eq. 11 (Marshall et al., 2024): the constituents $V$ are the finer
units it is composed of; the update
grain $\tau'$ is the number of micro updates over which those constituents are
read; and the mapping $g'$ is a truth table that assigns a binary macro state to
each joint sequence-state of the constituents over the window (Eq. 13),

$$ g' : \Omega^{\tau'}_{V} \to \{0, 1\}. $$

At update grain 1 the mapping reads a single joint state of the constituents. At
grain $\tau' > 1$ it reads a *sequence* of $\tau'$ successive constituent states, so
the macro state depends on the trajectory, not only on the final micro state.

Two mapping families cover the common cases. A **coarse-graining** groups the
constituents' joint states by their ON-count: the macro unit is ON exactly for a
chosen set of counts, at update grain 1. A **blackboxing** reads out a
designated subset of *output* constituents at the final update of the window,
and admits any update grain. These are the two families the default search
enumerates; the full space is every non-constant mapping. The number of possible
mappings for a unit with $|V|$ constituents at update grain $\tau'$, counted up
to complementation of the two macro state labels, is

$$ 2^{\,2^{\tau'|V|} - 1} - 1 $$

(Marshall et al., 2024, given after Eq. 13). It grows doubly exponentially in
$\tau'|V|$, which is why the search enumerates the two mapping families by
default and bounds the exhaustive alternative.

Macro units stack into a hierarchy. A unit's constituents may themselves be
macro units (**meso** units), so that its mapping composes with theirs down to
the micro units at the bottom (Marshall et al., 2024, Eq. 14; Fig.
3E). Building a unit directly on micro constituents leaves more mappings to
choose from than building it on meso constituents, whose mappings are already
fixed; which construction wins is decided, like everything else, by whichever
maximizes $\varphi_s$. Because each level contributes its own update grain, the
grains multiply down the hierarchy: a search that allows update grains up to
`max_update_grain` over `max_depth` levels reads each unit over a window of up to
`max_update_grain ** max_depth` micro updates, which is exactly the length of
micro history such a search requires.

In PyPhi a macro unit is a {class}`~pyphi.macro.MacroUnit` with attributes
`constituents`, `update_grain`, and `mapping`. The two families are built by
{func}`~pyphi.macro.coarse_grain` (from ON-count classes) and
{func}`~pyphi.macro.blackbox` (from output constituents, at any grain);
{func}`~pyphi.macro.micro_unit` is the trivial identity unit over a single micro
index, the base of the hierarchy.

## The intrinsic-unit criteria

A grouping does not become a unit by fiat. To exist as *one* unit, a candidate's
constituent system — the constituents evaluated over the full universe, with
everything else held as background — must satisfy the same postulates a complex
does. Two criteria capture this (Marshall et al., 2024, Eqs. 15–16). First, the
constituent system must be **integrated**: its own system integrated information
is positive,

$$ \varphi_s(v^{J}) > 0 \qquad \text{(Eq. 15).} $$

Second, it must be **maximally irreducible within** its footprint: no competing
system that could be built from the same micro units and background may match or
beat it,

$$ \varphi_s(v^{J}) > \varphi_s(v') \quad \text{for every competitor } v'
\qquad \text{(Eq. 16).} $$

Both criteria are properties of the pair (constituents, background) alone. A
candidate's mapping and update grain do not enter either inequality, so all
variants of one decomposition — whatever their mappings and update grains —
share a single verdict. The grain is chosen later, by the exclusion
competition, not here.

{func}`~pyphi.macro.judge_candidate` applies both inequalities to a candidate's
$\varphi_s$ and its evaluated competitor set and returns a
{class}`~pyphi.macro.UnitVerdict`. The verdict's `reason` is a
{class}`~pyphi.macro.Reason`: `VALID` when both criteria hold, `NOT_INTEGRATED`
when the constituent system has $\varphi_s = 0$ and so fails the integration
criterion of Eq. 15, and `NOT_MAXIMAL` or `TIED` when a competitor beats or ties
it under Eq. 16. Micro units are exempt — they are the base case of the
recursion, and count as units even when their own $\varphi_s$ is zero.

## Exclusion across grains

Many candidate units, at many grains, can each be integrated. They cannot all
exist over the same substrate: the **exclusion** postulate requires a definite
set of units (Albantakis et al., 2023). IIT resolves the competition by keeping,
among overlapping candidates, only the one whose system integrated information is
maximal, and this applies across grains — a micro candidate system and a macro
candidate system over the same micro units are rivals, not separate answers
(Marshall et al., 2024, Eq. 19).

PyPhi runs this competition as one cascade over **micro footprints**, so
candidates at every grain compete on the same basis. Each candidate's footprint
is the set of micro units it ultimately covers; the cascade walks candidates in
descending $\varphi_s$, accepts the maximal one, drops every remaining candidate
whose footprint overlaps it, and continues on what is left. The cascade is
**recursive**: a candidate that has been excluded by an accepted complex has no
standing to exclude anything else in turn. One consequence is counterintuitive:
a complex can coexist with an overlapping candidate of *higher*
$\varphi_s$, a **shadow**, provided that shadow was itself excluded by some other
complex. Ties within a tier escalate to the composition measure Φ, and a tier
that still ties fails exclusion outright: none of its members becomes a complex,
and their units remain available to lower-$\varphi_s$ candidates further down.

This cascade is the same {mod}`pyphi.condensation` machinery the micro complex
search uses; the macro search simply feeds it candidate systems at every grain.
It returns a {class}`~pyphi.macro.ComplexesResult` whose winners are
{class}`~pyphi.models.complex.Complex` objects. Each winner reports an
{attr}`~pyphi.models.complex.Complex.exclusion_margin` — the $\varphi_s$ gap to
the best overlapping rival it beat — and records the candidates it excluded,
shadows included; shadows do not enter the margin. A margin of zero means a
rival tied at the configured precision, so the selection was decided by
criteria beyond $\varphi_s$. For
how the recursion resolves overlapping candidates step by step, see the
{doc}`recursive-exclusion tutorial <../tutorials/recursive-exclusion>`; for
reading margins and controlling how ties are broken, see
{doc}`Control tie-breaking <../howto/tie-breaking>`.

## From theory to the library

| Notion | Type or function |
| --- | --- |
| A macro unit: constituents, update grain, mapping (Eqs. 11, 13) | {class}`~pyphi.macro.MacroUnit` |
| Coarse-graining and blackboxing mapping families | {func}`~pyphi.macro.coarse_grain`, {func}`~pyphi.macro.blackbox` |
| The trivial micro unit | {func}`~pyphi.macro.micro_unit` |
| A system of macro units, evaluated by the IIT pipeline | {class}`~pyphi.macro.MacroSystem` |
| The intrinsic-unit criteria (Eqs. 15–16) | {func}`~pyphi.macro.judge_candidate`, {class}`~pyphi.macro.UnitVerdict`, {class}`~pyphi.macro.Reason` |
| The recursive exclusion cascade (Eq. 19) | {mod}`pyphi.condensation` |
| The bounded search across grains | {func}`pyphi.macro.complexes`, `pyphi.analyze(substrate, state, grains=...)` |
| The search bounds and their cost estimate | {class}`~pyphi.macro.SearchBounds`, {class}`~pyphi.macro.SearchEstimate` |
| The search result: complexes, ties, and evaluation records | {class}`~pyphi.macro.ComplexesResult` |

## References

- Marshall W, Findlay G, Albantakis L, Tononi G (2024). Intrinsic units:
  identifying a system's causal grain. *bioRxiv* 2024.04.12.589163.
  <https://doi.org/10.1101/2024.04.12.589163>
- Albantakis L, Barbosa L, Findlay G, Grasso M, et al. (2023). Integrated
  information theory (IIT) 4.0. *PLOS Computational Biology* 19(10): e1011465.
