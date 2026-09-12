# Documentation user-experience review — findings and backlog

**Date:** 2026-09-11
**Design:** `2026-09-11-docs-ux-review-design.md`
**Method:** a structural pass over the docs site, the README, and the MCP
reference content, plus four cold walkthroughs (one fresh-context sonnet agent
per reader, allowed only those surfaces plus Python's `help()`). The walkthrough
logs are the evidence cited below as R1–R4:

| Reader | Tasks | Outcome |
|---|---|---|
| R1 New IIT researcher | own 4-unit logistic network; reproduce φₛ = 0.17; why deterministic → 0 | (a) done with workaround, 9 pages + `help()`; (b) done, 2 pages; (c) done, 2 pages |
| R2 Computational scientist, no IIT | own TPM → first number and meaning; every state + plot; laptop sizing | (a) done with workaround, 7 pages + `help()` + grep; (b) done, 1 page; (c) done, ~20 min |
| R3 PyPhi 1.x migrator | port `Network`/`Subsystem`/`compute` script; reproduce Φ = 2.3125; why default differs | all done, 3 pages; (b) and (c) needed facts assembled from three pages each |
| R4 MCP / agent user | which formalism produced 0.0405; 8-unit feasibility; interpret a card | (a) **failed** (no subset support); (b) done; (c) done, two rows only via raw JSON |

Nothing was changed in the documentation during this review.

## 1. Findings

Severity: **blocks** (a task failed or needed a workaround outside the docs),
**costs time** (the answer exists but had to be assembled or guessed),
**cosmetic**.

### Blocks

**F1. No page shows how to build a substrate from your own TPM, weights, or unit
functions.** Readers: R1, R2, R4. Construction from a numpy TPM appears only
inside other tutorials (`recursive-exclusion`, `macro`, `tie-breaking`); from a
weight matrix with logistic units only in the `substrate_modeler` migration
page; TPM formats only in `conventions.rst` under Reference. R1 exhausted every
section index, then grepped, then used `help()`; R2 the same; the README
advertises "Substrate generation" as a headline feature. The MCP content has
`building-iit-systems`; the site has no equivalent. (Structural pass G1; R1 (a);
R2 (a).)

**F2. The MCP `analyze` tool cannot analyze a proper subset, and silently drops
an unknown `subset` argument.** Reader: R4. The `theory` topic says
`pyphi.analyze` takes "the whole substrate by default, or a `subset`"; the
tool's parameters are handle, state, formalism, compute, detail, confirm_large,
parallel, workers. Passing `subset=["A","B"]` returned the full-system result,
byte-identical, with no warning. R4's task (a) failed on this. (Verified against
`pyphi/mcp/server.py`.)

**F3. A result does not say which formalism produced it, and no page says where
to look.** Readers: R1, R2, R4. `Analysis` exposes no formalism attribute and
the printed card has no formalism row; the value is at
`analysis.sia.config.formalism.iit.version`, which R2 found by trial. The one
place it surfaced automatically was the `formalism` column of a sweep table.
(Verified: the SIA carries a `ConfigSnapshot`.)

### Costs time

**F4. Reading a result card is under-documented.** R4 could not explain the
letter-case convention of purview labels (upper = specified ON, lower = OFF)
from any reference topic and reverse-engineered it from `inspect` JSON; the
MIP is in the JSON summary but not in the card the tool returns; the value
ii(s) and which term set φₛ are absent from the summary; individual relations
cannot be listed under the analytical backend and the reference does not say
so. The site has no "reading a result" page at all; interpretation guidance
lives only in the MCP `interpreting-iit-results` topic. (R4 (c); G4.)

**F5. φₛ = 0 has two causes and the docs teach only one.** R2's own network gave
φₛ = 0 with a non-deterministic TPM; `intrinsic-information.md` explains the
requirement's zero (the XOR case) but not ordinary reducibility (φ_c = 0). R2
resolved it only through `sia.explain()`, noticing the empty
`requirement_binding` list. R4 hit the mirror case: cause_phi and effect_phi
both positive with φₛ = 0, where the MCP `theory` topic's "φₛ is the smaller
of the two" precedes its 2026 caveat by several paragraphs. (R2 (a); R4 (c).)

**F6. The migration guide stops at the core API.** R3's very first error
(`examples.basic_network`) has no documented answer (`basic_substrate` was
found by chance in `theory/iit-3.0.md`); 1.x `MEASURE` and `PARTITION_TYPE`
appear nowhere; the correspondence "1.x big Φ = 2.0 `analysis.phi` under
`formalism="IIT_3_0"`, not `.big_phi`" is stated on no single page (R3
assembled it from an error message and two pages); under IIT 3.0 both
`System.ces()` and `analyze().ces` return `UnresolvedDistinctions` with
`.concepts`, a name that appears in no page. (R3 (a)–(c); verified.)

**F7. Sizing and cost guidance oversells the pre-flight and undersells its
caveats.** `computational-complexity.md` calls `estimate_analysis` free; R2
measured 30–50 s at 10–20 units, with `capped=True` at the default limit for
both `full` and `sia`, and found no example of raising `limit`. The page's
enumerator demo stops at n = 7 without saying why; R2 called
`directed_set_partitions(range(10))` directly and had to kill it. On the MCP
side, `estimate_cost(compute="sia")` returns no seconds, and the fact that
`estimated_cpu_seconds` excludes the system-partition axis lives only in the
tool docstring. The practical answer ("about 10–12 units for φₛ, 6–8 for the
structure; estimate first") is buried in a 600-line theory page. (R2 (c); R4
(b); G5.)

**F8. `estimate_analysis(substrate, state)` silently binds the state to
`subset`.** R1 got a plausible wrong estimate ("1 units, full") with no error.
(Verified: `subset` is the second positional parameter.)

**F9. The unit functions' `determinism` parameter is never connected to the
papers' slope k.** R1 verified `determinism == k` by rebuilding the Fig 1A
network and comparing TPMs by eye. (R1 (a).)

**F10. Terms used before definition, and no glossary.** "MICE" appears in the
`analyze` docstring and the complexity page's cost table unexpanded (R2);
φ_c and φ_e are used on `intrinsic-information.md` without a definition or link
(R2). The site has no glossary; the only `{abbr}` is "SIA" on the sweep page.
(G2.)

**F11. The Reference section is two stubs.** `reference/api.md` is an
autosummary directive (R2 opened it first when looking for the `Substrate`
constructor and found nothing); `configure.md`'s option table lists seven
options and defers to the API classes; there is no configuration-options page
and no task index. (G7; R2 (a).)

**F12. No FAQ or troubleshooting page.** Every confusion in the four logs is FAQ
material: why is φₛ zero; why did my 1.x numbers change; a state that raises
`StateUnreachableForwardsError`; a TPM rejected for conditional dependence;
`capped=True`; ties in small networks. The MCP `gotchas` topic covers most of
these for agents; the site covers none in one place. (G3.)

### Cosmetic

**F13. Navigation.** Pages with no onward link: `tutorials/actual-causation`,
`howto/export`, `howto/save-load`, `conventions.rst`,
`migration/from-substrate-modeler`. "What's new" is in the hidden toctree but
not on the index grid. The README's Documentation section links only the two
RTD roots, not getting-started, and its release note links the GitHub markdown
rather than the rendered page. (G8, G10.)

**F14. Terminology leftover.** `pyphi/mcp/content/gotchas.md` §5 is headed "The
2026 differentiation cap"; R4 adopted "cap" throughout its report from it.
(G9.)

**F15. Three equivalent ways to select a formalism** (`formalism=`,
`config.override(**preset)`, `config.iit = presets.x["iit"]`) are introduced
on different pages without cross-reference. (R3.)

**F16. No examples gallery** for the 58 registered examples (which paper and
figure, size, formalism they reproduce), and no contributing page on the site.
(G6, G11.)

## 2. Coverage matrix

Covered ✓, partial ◐, missing ✗. The page named is where the need is met.

| Need | R1 IIT researcher | R2 no-IIT scientist | R3 1.x migrator | R4 MCP user |
|---|---|---|---|---|
| Install | ✓ getting-started | ✓ | ✓ | ✓ mcp-server |
| First number end to end | ✓ getting-started | ✓ getting-started | ✓ migration-2.0 | ✓ primer |
| Build my own substrate | ✗ (F1) | ✗ (F1) | ◐ from-substrate-modeler | ◐ building-iit-systems (subset impossible, F2) |
| φₛ versus Φ | ✓ getting-started, theory/overview | ✓ | ◐ 1.x Φ ↔ φₛ not stated (F6) | ✓ theory, gotchas |
| Choose a formalism | ✓ formalism-versions | ✓ | ✓ migration-2.0 | ✓ |
| Know which formalism produced a result | ✗ (F3) | ✗ (F3) | ◐ | ◐ `summary.formalism` only |
| Interpret a result / card | ◐ getting-started, intrinsic-information | ◐ (F5) | ◐ | ◐ interpreting topic (F4) |
| Reproduce a published value | ✓ worked-example | — | ✓ migration-2.0 | ◐ (F2) |
| Why φₛ = 0 | ✓ intrinsic-information (requirement case) | ◐ (F5) | ✓ migration-2.0 | ◐ (F5) |
| Size and cost before running | ◐ computational-complexity (F7) | ◐ (F7) | — | ◐ performance (F7) |
| Sweep states, plot | ✓ howto/sweep | ✓ howto/sweep | — | — |
| Port 1.x code | — | — | ◐ (F6) | ✓ migration topic |
| Troubleshoot an error | ✗ (F12) | ✗ (F12) | ✗ | ◐ gotchas |
| Look up a term | ✗ (F10) | ✗ (F10) | — | ◐ theory |
| Look up an option | ◐ configure (F11) | ◐ | ◐ | ✓ configuration |
| Find an example network | ◐ autosummary only (F16) | ◐ | — | ✓ list_examples |

## 3. Backlog

Ranked by impact (readers affected × severity) against effort (S: under a day;
M: one to three days; L: more). Items marked **code** change the library or the
MCP server, not only prose.

| # | Item | Findings | Impact | Effort |
|---|---|---|---|---|
| B1 | **New how-to: Build a substrate.** From a state-by-node TPM; from a weight matrix with logistic units (`build_substrate`, `ising.probability`, and the slope k, written as `determinism`); from per-unit functions (`create_substrate`); from data (`estimate_substrate`); then validate (`describe`, a known transition, conditional independence) and link the TPM conventions. Link it from getting-started's "Where to go next" and from theory/substrate-and-system. | F1, F9 | high | M |
| B2 | **New how-to: Read a result.** The analysis card and the SIA card row by row; φₛ versus Φ; the MIP; the letter-case convention; margins and ties; the two causes of φₛ = 0 and the `requirement_binding` finding; where the formalism is recorded. Mirror the additions into the MCP `interpreting-iit-results` topic (case convention, ii(s), the relations-listing limit). | F4, F5, F3 | high | M |
| B3 | **Results state their formalism** (code): `Analysis.formalism` and a card row; one sentence in getting-started at the first `analyze()` call. | F3 | high | S |
| B4 | **MCP `analyze` gains `subset`; unknown arguments are rejected; the card and summary carry the MIP, ii(s), and the binding term** (code). Note the gap in `primer` until landed. | F2, F4 | high | M |
| B5 | **Migration guide additions:** `pyphi.examples` renames; a 1.x → 2.0 table for `MEASURE`, `PARTITION_TYPE` and the other config flags (or an explicit "not preserved"); a quantities table (1.x Φ ↔ `analysis.phi` under `IIT_3_0`, not `.big_phi`); the IIT 3.0 result type and `.concepts`. Consolidate the three ways to select a formalism with cross-references. | F6, F15 | medium | S |
| B6 | **New page: Glossary.** One entry per term the theory pages define (substrate, system, background, repertoire, purview, mechanism, MIP, distinction, relation, complex, specified state, intrinsic information / specification / differentiation, margin, MICE, φ / φₛ / Φ, α), each linking to its theory page. Expand acronyms at first use on existing pages. | F10 | medium | M |
| B7 | **New page: FAQ and troubleshooting.** Why is φₛ zero; why did my numbers change from 1.x; `StateUnreachableForwardsError`; conditional-independence rejection; `capped=True`; ties; memory. Draw on the MCP `gotchas` topic. | F12 | medium | M |
| B8 | **Sizing: a short how-to "Estimate the cost before you run"** (ceilings table, `estimate_analysis` with `limit` and `capped`, what to reduce), with `computational-complexity.md` kept as the deep page; correct its "free" wording and add the warning at the enumerator demo. MCP `performance`: say `estimated_cpu_seconds` excludes the system-partition axis, and give `estimate_cost(compute="sia")` a projection or explain why none (code). | F7 | medium | S–M |
| B9 | **`intrinsic-information.md`:** state the two causes of φₛ = 0 up front with the one-line check; define or link φ_c and φ_e. MCP `theory`: caveat the "smaller of the two" sentence. | F5, F10 | medium | S |
| B10 | **`estimate_analysis`: make `subset` keyword-only** (code). | F8 | medium | S |
| B11 | **Configuration reference page** listing every option by layer with its default and one line of meaning (generated from the config dataclasses if practical). | F11 | medium | M |
| B12 | **Navigation fixes:** onward links on the five dead-end pages; a "What's new" card on the index; README links to getting-started and to the rendered what's-new; an "I want to…" task index on the how-to landing page. | F13 | low–medium | S |
| B13 | **Examples gallery page** generated from the registry: name, size, alphabet, paper and figure, formalism. | F16 | low–medium | M |
| B14 | **Terminology:** fix `gotchas.md` §5 ("differentiation cap"); grep for "cap" in MCP content once more. | F14 | low | S |
| B15 | **Rename or alias the IIT 3.0 result type** so `.ces` under IIT 3.0 is not `UnresolvedDistinctions` (code). | F6 | low | S–M |
| B16 | **Contributing page** on the site (from the README section). | F16 | low | S |

Suggested order: B3, B10, B14 (small, immediate); B1, B2 (the two new
how-tos); B4 (MCP); B5, B9, B8; B6, B7; B11, B12, B13; B15, B16.

## 4. What is working

Worth keeping as is: the getting-started walkthrough (every reader reached a
first number from it); the theory section's single running example and the
formalism-versions page (R1 (b) and (c) took two pages each); the migration
guide's before/after examples and its explicit statement about deterministic
networks (R3 (b) reproduced 2.3125 on the first try); `howto/sweep` (R2 (b) took
one page); the MCP `theory`/`gotchas`/`interpreting` topics (R4 explained
every headline row from them). The docs' numbers agreed with live runs at
every check (getting-started, mcp-server's example, the worked example).

## 5. Out-of-scope observations (for a later pass)

- API reference and docstrings: `reference/api.md` is an autosummary directive
  that renders only in the built site; `sigmoid`'s `determinism` has a
  one-line docstring; `analyze`'s docstring uses "MICE".
- `pyphi.actual` beyond the tutorial has no how-to; nobody in this review
  needed one.
