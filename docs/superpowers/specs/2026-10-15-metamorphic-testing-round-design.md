# PyPhi 2.0.0 metamorphic and differential testing round — design

Status: APPROVED (2026-10-16).

## Purpose

Four review rounds since 2.0.0rc3 found 86, 50, 45 and 5 problems. The first
three read the whole surface or the code changed since rc3; the fourth read
only the third fix pass and reported only problems that would block the
release. The trend says the code that recently changed is now in good shape.

The fourth round's most serious finding came from somewhere else. It was in
code no fix pass had touched: `System.cause_mip` and five related methods
gave a wrong φ when a purview was given out of order, a bug that probably
dates from 1.x. No review had fed those methods that input, and no test
checked that the order of units does not matter. Reading code finds bugs of
that kind only by luck. A check that states the property and tries it on
many inputs finds them directly.

This round therefore does not read code. Agents write and run checks of two
kinds, across the public surface:

- **Metamorphic relations.** Transform the input in a way that must leave
  the result unchanged, or change it in a known way, and compare. Example:
  permuting a substrate's units must permute every part of the result to
  match.
- **Differential pairs.** Compute the same quantity by two routes that must
  agree, and compare. Example: a sweep row against `analyze` on the same
  input.

A survey of the test suite (summarized under each area below) found that
most of these relations are untested or tested on one or two hand-built
fixtures. Two findings from the survey shaped the design: whole-result
invariance under unit permutation is tested only on a 2-unit pair and on
grid3, and most "parallel equals sequential" tests never dispatch to workers,
because small systems fall below the parallel thresholds.

## What the round produces

1. **Findings** that meet the release bar (below), each with a seed and a
   minimal reproduction, verified by rerunning the reproduction.
2. **A record of coverage**: for every relation in the catalogue, how many
   non-trivial cases it held on, or why it was not run.
3. **Checks worth keeping**, written so they can become tests in the suite
   with little change (see "Turning checks into tests").

## The release bar

Only these count as findings, at medium or high severity:

- a wrong computed value or a wrong selection (partition, state, purview,
  complex);
- a crash on valid input, or valid input rejected;
- invalid input accepted and giving a wrong result;
- state that leaks between calls, so a result depends on what ran before;
- data lost or changed in a save/load round trip;
- two public routes to the same quantity that disagree.

Polish, wording, error-message quality, missing features and performance are
out of scope, unless a slowdown makes a documented example impractical.
Polish noticed along the way goes on a list for 2.0.1, not in the findings.

## How results are compared

These rules prevent the two failure modes that would waste the round: false
alarms from legitimate tie-breaking or floating-point noise, and vacuous
passes where both sides are trivially zero.

**Values.** Two φ, α or Φ values agree when
`abs(a - b) <= 1e-10 * max(1, abs(a), abs(b))`. That is far looser than
summation-order noise (about 1e-15) and far tighter than any real error seen
so far. A difference between 1e-13 and 1e-10 is recorded in the coverage
record as a note, not reported as a finding. Save/load round trips are the
exception: they must be exact (`==` on the loaded object, and bit-equal
arrays).

**Selections and ties.** Where a result selects among candidates whose
values tie, a transformation may legitimately change which one is selected.
Published Φ values already depend on enumeration order where ties exist.
A selection mismatch is a finding only if the result reports no tie at that
level, and the margin to the runner-up is at least 1e-10. PyPhi exposes ties
and margins on every result:

| Result | Tie and margin attributes |
| --- | --- |
| IIT 4.0 SIA | `ties`, `tied_selections`, `effectively_tied`, `partition_margin` (None if the partition sweep short-circuited), `state_margins`, `system_state.{cause,effect}.ties` |
| IIT 3.0 SIA | `ties` |
| RIA | `state_ties`, `partition_ties`, `state_margin`, `partition_margin`, `effectively_tied` |
| MICE | `purview_ties`, `state_ties`, `partition_ties`, `purview_margin`, `state_margin`, `effectively_tied` |
| Distinction | `effectively_tied` |
| Distinctions | `UnresolvedDistinctions` when the specified state is tied |
| Complex | `exclusion_margin`, `effectively_tied` |
| Grain search | `ComplexesResult.ties` |
| Actual causation | `AcSIA.ties`, `partition_ties`, `purview_ties` |

When a selection is tied, compare what does not depend on the tie-break:
the selected value, and the set of tied candidates mapped through the
transformation. Φ depends on the tie-break when the cause or effect state is
tied, so Φ is compared only when the system state is untied.

**Non-trivial cases.** A check counts toward coverage only on cases where
the compared quantities are not all zero. Each check reports how many of its
cases were non-trivial; a relation that held only on trivial cases is
reported as untested.

**Independent routes.** A check must not test PyPhi against itself through
a shared helper. The permuted substrate is built with numpy from the
original arrays, not with a PyPhi permutation helper. The brute-force complex
search is written in the check, not taken from `pyphi.condensation`. Where
PyPhi has its own helper for the same job (`FactoredTPM.permute_nodes`,
`CES.relabel`, `automorphism.structure_signature`), comparing it against the
independent route is a separate differential check. If the two disagree, the
check's own construction is suspected first.

## Fixtures

The suite's random generators produce 2–3 binary units, and none produces a
random k-ary or labelled substrate whose TPM agrees with its connectivity
matrix. PyPhi rejects a TPM that depends on an edge the connectivity matrix
omits, so a generator has to build each unit's TPM column from a table keyed
by the states of that unit's inputs.

Most random substrates are trivial: in a sample of 40, about one in eight
gave φ_s > 0 under the default formalism, at both 3 and 4 units. Screening
for non-trivial cases is therefore part of building fixtures, and it is
done once, before the agents start, not by each agent.

**The fixture bank** (built in the session scratchpad as `fixtures.py` plus
a `bank.json`):

- `random_substrate(seed, n, *, p_edge=0.6, alphabet=None, labels=False,
  extreme=0.0)`: a substrate whose TPM agrees with its connectivity matrix.
  `alphabet` gives per-unit state counts for k-ary substrates. `labels`
  attaches single-letter labels. `extreme` is the fraction of table entries
  drawn near 0 or 1, which raises the share of non-trivial systems. It uses
  `np.random.default_rng(seed)` only.
- A screened list of (generator parameters, state) entries recording, per
  formalism (2026, 2023, IIT 3.0), φ_s, Φ, whether any selection is tied, and
  the analysis time. Targets: at least 20 non-trivial binary entries at 3
  units and 15 at 4 units per formalism, at least 8 k-ary entries, at least
  5 entries with a tied selection (for the tie rules), and at least 5
  subset systems with background units.
- The named examples the paper reproductions use: `iit4_2023_fig1a`,
  `basic`, `xor`, `grid3`, `rule110`, a k-ary example (the Gómez p53
  substrate), and the hand-built symmetric fixtures from
  `test/example_substrates.py` (copied, since the agents run from an
  installed wheel without the test directory).

**Size budget.** Measured on this machine, a full analysis of a random
substrate takes about 0.1 s at 3 units, 0.25 s at 4 units and 13 s at
5 units under the default formalism. Under IIT 3.0 it takes 0.4 s, 3 s and
50 s. Checks use 3 and 4 units. Five units are used only for a few cases
per relation, where a relation could plausibly break only at that size
(for example, set partitions with three parts).

## The catalogue

The relations are split among three agents by area of the public surface.
Each entry gives the relation, the entry points, and what the survey found
already tested. Priority 1 relations come first within each area; an agent
finishes priority 1 before starting priority 2.

Throughout, "permute" means: choose a permutation p of the units, build the
permuted substrate (TPM, connectivity matrix, alphabets and labels), map the
state and every unit set (subset, mechanism, purview, candidate) through p,
and map the result back before comparing. States are little-endian: the
first unit is the least-significant bit of a TPM row index.

### Area A — single-system analysis

Entry points: `pyphi.analyze`, `System` and its query methods, `Substrate`
methods, relations and their query surface, `explain()`, `to_pandas()`.

Priority 1:

- **A1. Unit permutation of a whole analysis.** Under 2026, 2023 and IIT 3.0,
  on binary and k-ary substrates, whole and subset systems: φ_s, the MIP
  (mapped), the specified cause and effect states (mapped), every
  distinction (mechanism, purviews, specified states, φ), every relation
  (relata and φ), and Φ. Tested today only on grid3 (2026) and a 2-unit pair.
- **A2. Unit permutation of the System query methods.** Repertoires (as
  arrays with permuted axes), `find_mip`, `mic`, `mie`, `phi`, `distinction`,
  `intrinsic_specification`. Not tested today.
- **A3. Input spellings.** Subset order, labels against indices, numpy
  integers and arrays against tuples, for `state`, `subset`, mechanisms,
  purviews and candidates, on `analyze`, `System`, `Substrate.complexes` and
  `Transition`. Results must be equal. Tested today only for mechanism and
  purview order and some subset spellings.
- **A4. Repeated and interleaved calls.** Run analysis X, then unrelated
  analyses Y and Z (other substrates, other formalisms on the same object),
  then X again: the two X results must be equal. Also cache settings
  (`cache_repertoires`, `cache_potential_purviews`) on against off. Tested
  today only as same-call repetition on one system.
- **A5. Formalism selection.** `formalism=` given as a name, as a version
  string and as a preset, against `config.override(**preset)`, on analyze,
  ces, distinctions and relations. Results must be equal. Tested today only
  for IIT 3.0 complexes.

Priority 2:

- **A6. Compute modes.** `compute="sia"`, `"ces"`, `"distinctions"` and the
  full analysis must agree on what they share; `System.ces(sia=...,
  distinctions=...)` with its own precomputed parts must equal
  `System.ces()` in full (distinctions and relations, not only Φ). Tested
  today on φ only, and on one substrate for the precomputed case.
- **A7. Structure identities.** Φ = Σφ over distinctions + Σφ over relations
  (Albantakis et al. 2023, Eq. 59) on every computed structure; analytical
  against concrete relations under 2026 and on k-ary substrates;
  `to_pandas()` values equal to the attributes they come from. Eq. 59 is
  asserted today only on sweep rows and in the bounds tests.
- **A8. Background units.** Under IIT 3.0 (`CONDITION_CURRENT_STATE`), the
  background is held at its current state for causes and effects alike, so
  a subset System's analysis must equal two independent constructions: the
  same subset of `Substrate.inactivate(background_state)`, and a substrate
  of the subset's units alone whose TPM is built with numpy by conditioning
  the original on the background state. Not tested today. (Under IIT 4.0
  the background's past states are weighted by their posterior, so no such
  simple equivalence holds.)

### Area B — many-system surfaces and persistence

Entry points: `sweep`, `landscape_section`, `perturb`, `optimize`,
`phi_posterior`, `Substrate.complexes`, the campaign API, the parallel
backends, `pyphi.save`/`pyphi.load`, the MCP server's tools,
`estimate_analysis`.

Priority 1:

- **B1. Complexes against brute force.** `Substrate.complexes` (and
  `all_sias`, the maximal complex) against a search written in the check:
  analyze every subset, then apply exclusion greedily by φ_s with overlap
  removal. Under 2026 and IIT 3.0. Tested today only for structural
  invariants and one value pin.
- **B2. Unit permutation of complexes and sweeps.** Complexes and sweep
  tables must permute to match. Not tested today.
- **B3. Parallel against sequential, with dispatch forced.** Thresholds set
  so work really goes to workers (loky and thread backends), for the 2026
  CES with relations, `Substrate.complexes`, k-ary substrates, and actual
  causation. Equal results required. Tested today with forced dispatch only
  for the IIT 4.0 (2023) and IIT 3.0 SIA.
- **B4. Save/load round trips of computed results.** CES with concrete
  relations on non-trivial substrates, k-ary results, IIT 3.0 CES, sweep,
  optimization and landscape results, complexes. Then use a loaded result as
  input (`System.ces(sia=loaded_sia)`, recomputation from a loaded
  substrate) and compare with the original. Tested today mostly on hand-built
  objects and on basic.

Priority 2:

- **B5. Many-system rows against direct analysis.** Sweep rows over
  `subsets="all"`, under IIT 3.0 and on k-ary substrates; landscape rows
  under 2023 and IIT 3.0; optimization trajectory rows; each row against
  `analyze` on the same substrate, state and subset. Tested today for sweeps
  of basic and fig1a under IIT 4.0 and landscape rows under 2026.
- **B6. Campaigns against direct analysis.** `prepare` / `run` / `collect`
  for SIA and CES, sharded and unsharded, under 2026 and IIT 3.0, on
  substrates beyond basic, comparing every relation (not only Σφ_r) and the
  specified states. Tested today on basic for the sharded CES path.
- **B7. MCP tools against the API.** `analyze`, `inspect`, `estimate_cost`
  and `describe_substrate` called in-process against `pyphi.analyze` and
  `estimate_analysis` on the same input, including labels and subsets.
  Tested today only against hard-coded constants.
- **B8. Estimates against the work done.** `estimate_analysis` counts
  (partitions, purview evaluations) against instrumented counts during real
  runs. The estimate is a guard, not a result, so a discrepancy is a finding
  only if it makes the MCP server refuse an analysis that finishes in
  seconds, or admit one that runs far past its stated limit.

### Area C — macro, actual causation and substrate construction

Entry points: `pyphi.macro` (`MacroSystem`, `MacroUnit`, `complexes`,
`analyze(grains=...)`), `Transition` and `pyphi.actual`,
`pyphi.substrate_generator`, `Substrate` constructors.

Priority 1:

- **C1. Identity macro units against the micro system.** A `MacroSystem`
  whose units are single micro units must equal the micro System in φ_s,
  MIP, specified states, distinctions (with purviews and states) and
  relations, under 2026. Tested today on φ and distinction φ only, under
  2023.
- **C2. Unit permutation of the grain search and macro systems.** Permute
  the micro units; the grain search's complexes, records, excluded lists and
  margins must permute to match, under both prune modes. Not tested today.
- **C3. Unit permutation of actual causation.** Permute the units of a
  `Transition`; α for every link, the account's links (mapped), and the AC
  system analysis must match. Not tested today.
- **C4. Certified against full prune on 4-unit substrates.** The grain
  search with `prune="certified"` against `prune="off"`: equal complexes,
  records (where evaluated), excluded lists and margins. Tested today on
  3-unit random substrates with `max_constituents=2`, and on two fixed
  4-unit substrates.

Priority 2:

- **C5. Construction routes.** The same substrate built as a joint TPM, as
  a state-by-node TPM, as a `FactoredTPM` (including a k-ary encoding with
  every alphabet of size 2), and through `substrate_generator.create_substrate`
  from equivalent specs, must give equal substrates and equal analyses.
- **C6. Actual causation round trips and parallelism.** Save, load and
  recompute an account; parallel against sequential for `account` and the
  AC system analysis.
- **C7. Grain search input spellings.** Labels against indices, numpy
  histories against tuples, for `macro.complexes` and `analyze(grains=...)`.

## Known non-invariances (not findings)

Each agent's brief lists these so they are not reported:

- Tie-broken selections that change under a transformation where the result
  reports a tie or a margin below 1e-10 (see "How results are compared").
- `CES.relabel` and the other result relabeling helpers raise for IIT 3.0
  SIAs and structure views, and drop tie back-references. That is a
  documented limit of the helper; compare against recomputation instead.
- Label strings in tables and cards change when labels change.
- `partition_margin` is None when the partition sweep short-circuits; use
  `shortcircuit_sia=False` where an exact margin is needed.
- The grain search and macro construction raise under IIT 3.0 and under
  `CONDITION_CURRENT_STATE`.
- `System.ces(sia=...)` does not detect an analysis computed with the
  background units in another state (documented).
- Single results name their state column `current_state`; sweeps name it
  `state`.
- Performance, unless it makes a documented example impractical.

## How the round runs

**Before launching:** build a wheel from current `main` into the
scratchpad, install it with all extras into a shared venv, and build and
screen the fixture bank. Record the commit the wheel was built from. The
agents test the code as it will ship, and read the repository only to
understand it.

**Three Opus agents** (areas A, B and C), each with:

- the shared rules: repository read-only, scratch directory of its own,
  the wheel's venv, `PYPHI_WELCOME_OFF=1` and `PYPHI_AGENT_NOTE_OFF=1`;
- the comparison rules, the fixture bank and the known non-invariances;
- its area of the catalogue, in priority order;
- a time box of about 45 minutes, a stopping rule (priority 1 complete,
  then priority 2 as time allows), and explicit permission to report that a
  relation could not be checked, or that a relation in the catalogue is
  wrong.

Each agent writes its checks as pytest files in its scratch directory, one
file per relation, using the fixture bank, so a failing check is its own
reproduction and a passing one can move into the suite.

**Output schema per agent:** findings (id, relation, title, category,
severity, the failing check's file and test name, seed and fixture, minimal
reproduction as runnable code with its output, why it matters, suggested
fix); and a coverage record per relation (cases run, non-trivial cases,
result, notes on differences between 1e-13 and 1e-10, runtime, reason if not
run).

**One Sonnet verifier per agent** reruns each reproduction verbatim in a
fresh venv from the same wheel and marks it confirmed, partly confirmed
(with a correction), refuted or unverifiable, and whether it meets the bar.

**After the round:**

1. I deduplicate the findings and bring you a short triage list with a
   recommendation for each.
2. Fixes follow the pattern of the fourth round: a regression test that
   fails on the old code, the full fast suite with doctests, the slow suite
   in two halves, and one narrow adversarial check of the fix diff before
   committing.
3. Checks are moved into the suite (next section).

## Turning checks into tests

A check moves into the suite if it found a bug, or if it covers a relation
the survey found untested and runs fast enough.

- A shared generator for random substrates whose TPM agrees with the
  connectivity matrix (binary and k-ary, optionally labelled) goes into
  `test/hypothesis_utils.py`, which today lacks one.
- Metamorphic checks go into `test/integration/test_metamorphic.py`, and
  differential checks into the existing file for their entry point
  (`test/test_sweep.py`, `test/campaign/…`, `test/parallel/…`,
  `test/macro/…`).
- Fast-lane checks use a fixed list of screened seeds, pinned formalisms
  (`IIT_3_CONFIG` / `IIT_4_CONFIG` or presets), and at most a few seconds
  each. Wider seed ranges and 5-unit cases go in the slow lane under
  `@pytest.mark.slow`.
- Each test asserts its own non-triviality (for example, that φ_s > 0 on
  its fixture), so a later change cannot turn it vacuous.

## Cost

About the same as the fourth round's reviewers plus verifiers: three Opus
agents at 45 minutes and three Sonnet verifiers, roughly 0.8 to 1 million
subagent tokens, plus building the fixture bank (in this session, a few
minutes of computation). The fix pass and the test conversion come after,
sized by what the round finds.

## Non-goals

- Reading code for style, naming or documentation quality.
- Performance work.
- New features, including a substrate-level permutation helper. The checks
  build permuted substrates themselves.
- Re-verifying paper reproductions and goldens, which the suite already
  pins.
- `pyphi.matching`, `pyphi.dynamics` and the visualization modules. Their
  correctness does not affect computed φ values.

## Success criteria

- Every priority 1 relation ran on at least 10 non-trivial cases (at least
  3 for relations that need 5-unit or IIT 3.0 4-unit analyses), or the
  report says why it could not run.
- Every finding has a seed, a fixture and a reproduction that reruns.
- No relation is reported as holding on trivial cases only.
- The checks kept as tests pass in the suite's fast lane in well under a
  minute in total.
