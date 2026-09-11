# Documentation user-experience review — design

**Date:** 2026-09-11
**Status:** approved in discussion; awaiting sign-off on this document

## Goal

Find what is missing, unclear, or hard to reach in PyPhi's documentation for
four kinds of reader, and produce a ranked backlog of additions and changes.
This pass writes nothing into the documentation itself.

## Readers

1. **New IIT researcher.** Knows the theory from the papers, has never used
   PyPhi. Wants a number for their own network and to understand what it means.
2. **Computational scientist without IIT background.** Comfortable with Python
   and numpy; arrives from a citation or a talk. Needs the concepts before the
   API.
3. **PyPhi 1.x user migrating.** Has code against `Network`/`Subsystem` and IIT
   3.0. Needs to port it and understand why numbers differ.
4. **AI agent / MCP user.** Reads the MCP reference content and shipped skills,
   not the docs site. Needs self-contained, findable reference material.

## Surfaces in scope

- The docs site: `docs/` (getting started, what's new, tutorials, how-to
  guides, theory, reference, migration, conventions).
- `README.md` (also the PyPI page).
- The MCP reference content and skills: `pyphi/mcp/content/*.md`,
  `pyphi/mcp/skills/*/SKILL.md`.

Out of scope: the API reference and docstrings (a separate review), the
changelog.

## Method

### Part 1 — structural pass (no subagents)

Every page in scope is read against one checklist:

- Which of the four readers needs this page, and for what.
- Does the page say what to read next, and is it reachable from the index and
  its section landing page.
- Do all code blocks execute (executed pages) or match the current API (static
  pages).
- Does the terminology match the papers and the rest of the site
  (in particular: φₛ versus Φ; "intrinsic-information requirement"; 2023 versus
  2026 names for the specification term).
- Is the page the right length and level for its section (tutorial versus
  how-to versus theory versus reference).

Output: a reader-by-need coverage matrix, with a cell for every combination of
reader and need, marked covered / partial / missing, and the page that covers
it.

Candidate gaps to verify, not assume: glossary; FAQ or troubleshooting page
(validation errors, cost refusals, memory); a decision guide for choosing the
formalism and settings; result interpretation on the site (today only in the
MCP content); a short sizing guide ("how big a system can I run"); an examples
gallery for the registered examples; a real reference section; an "I want
to…" task index; a contributing guide.

### Part 2 — cold walkthroughs (four subagents)

One fresh-context subagent per reader, run concurrently (at most five
subagents at any time), model **sonnet**: a less capable reader is the stricter
test of the documentation. Each agent:

- may read only the surfaces in scope, plus run Python to test snippets;
- must not read the library source or tests;
- logs, for each task: where it looked first, every dead end, every term it
  could not resolve from the docs, every snippet that failed, and the number of
  pages read before its first result.

Tasks:

| Reader | Tasks |
|---|---|
| New IIT researcher | (a) φₛ and the Φ-structure of a four-unit logistic network from a weight matrix, in a given state; (b) reproduce one number from Albantakis et al. (2023); (c) explain why a deterministic network gets φₛ = 0. |
| Computational scientist | (a) from a numpy TPM to a first number and a plain reading of φₛ versus Φ; (b) analyze every state and plot the result; (c) decide how large a system a laptop can handle. |
| 1.x migrator | (a) port a snippet built on `Network`, `Subsystem`, `compute.sia`, `compute.ces`; (b) reproduce an IIT 3.0 result from 1.x under 2.0; (c) explain why the default numbers differ from 1.x. |
| MCP user | using the MCP content and skills only: (a) determine which formalism produced a given number; (b) judge whether an eight-unit analysis is feasible; (c) interpret a result card. |

Each agent returns a structured report: per task, the outcome (done / done
with workaround / failed), the log above, and a short list of what would have
helped.

### Part 3 — synthesis

One document, `docs/superpowers/specs/2026-09-11-docs-ux-review.md`, with:

- **Findings**, each tied to a reader, a page (or "absent"), and a severity
  (blocks the task / costs time / cosmetic), with the evidence (rubric cell or
  walkthrough log line).
- **Backlog**, ranked by impact (how many readers, how severe) against effort,
  with the proposed new pages or sections listed with a one-line scope each.
- **Out-of-scope observations** (API reference, docstrings) noted for a later
  pass.

Implementation of the backlog is a separate plan, written after the backlog is
approved.

## Constraints

- No documentation is written or changed in this pass.
- Subagents: at most five concurrent; sonnet for the walkthroughs.
- Every finding cites its evidence; no finding rests on a guess about what a
  reader would do.
