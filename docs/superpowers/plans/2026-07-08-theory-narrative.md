# Theory Narrative (IIT 4.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the eight-page IIT 4.0 theory section defined in
`docs/superpowers/specs/2026-07-08-theory-narrative-design.md` — a
self-contained, computation-driven, postulate-grounded narrative, grounded in
one build-time-executed worked example, with every equation paper-verified and
every code claim implementation-verified.

**Architecture:** Eight MyST Markdown pages under `docs/theory/`, wired into the
Theory section's toctree. Pages 1–5 carry a single worked example
(`pyphi.examples.basic_substrate()` at state `(1, 1, 0)`) whose code cells
execute at build time via myst-nb. Each page is authored by one agent, then
checked by an independent verification agent against the paper PDFs and the
code before it is accepted — mirroring the docstring sweep's two-stage process.

**Tech Stack:** MyST Markdown, myst-nb (executed cells), Sphinx `-W` build, the
paper PDFs in `papers/`, `graphify-out/bridge-edges.json`.

## Global Constraints

- **The worked example is `pyphi.examples.basic_substrate()` at state
  `(1, 1, 0)`.** Verified facts about it (agents must reproduce, not invent):
  Φ_s = 0.208; the cause–effect structure has **3 distinctions** and **2
  relations**. `pyphi.analyze(substrate, (1, 1, 0))` returns an `Analysis` with
  `.phi`, `.ces`, `.sia`, `.system`.
- **Executable pages.** Concept-bearing pages use MyST `{code-cell}` blocks that
  execute at build time. Set `pyphi.config.progress_bars = False` in the first
  cell of each executable page so output is clean. Every page must build under
  `uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs
  docs/_build/html` (warnings are errors).
- **Accuracy is paramount.** Every equation/section/figure citation is verified
  against the actual paper PDF in `papers/` — never from memory. Every code
  claim (type, function, output value, symbol→code mapping) is verified against
  the implementation. Where the 2023 paper and the 2026 cap differ, cite both.
- **Self-contained.** The section orients the reader itself; it points to the
  IIT wiki (<https://iit.wiki>) as an additional resource but never depends on
  it. Full derivations are left to the papers, which are cited throughout.
- **Primary sources:** `papers/2023__albantakis-et-al__iit-4.0.pdf` (+ its S1–S4
  supplements), `papers/2026__mayner-et-al__intrinsic-cause-effect-power.pdf`
  (the 2026 ii-cap), `papers/2014__oizumi-et-al__iit-3.0.pdf`,
  `papers/2019__albantakis-et-al__what-caused-what.pdf`.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
  Never `--no-verify`; never `git add -A` (stage only the page(s) named); no
  planning-artifact markers in committed content.

## Verified API surface (for the worked example)

Agents ground examples in these confirmed calls; verify any others before use:

- `a = pyphi.analyze(sub, (1, 1, 0))` → `Analysis`: `a.phi` (0.208), `a.ces`
  (`CauseEffectStructure`), `a.sia` (system irreducibility analysis),
  `a.system`.
- `a.sia`: `.phi`, `.partition`, `.cause`, `.effect`, `.normalized_phi`,
  `.node_indices`, `.node_labels`.
- `a.ces`: `.distinctions` (3), `.relations` (2, a `ConcreteRelations`),
  `.big_phi`, `.sum_phi_distinctions`, `.sum_phi_relations`.
- `d = a.ces.distinctions[i]` → `Distinction`: `.mechanism`, `.phi`, `.cause`,
  `.effect`, `.cause_purview`, `.effect_purview`, `.cause_repertoire`,
  `.effect_repertoire`, `.mice`.
- Formalism versions: `pyphi.config.formalism.iit.version` (`"IIT_4_0_2023"`);
  the namespaces `pyphi.iit3`, `pyphi.iit4_2023`, `pyphi.iit4_2026` exist.

## File structure

```
docs/theory/index.md              modified: Theory landing + toctree of the 8 pages
docs/theory/overview.md           new: page 1
docs/theory/substrate-and-system.md   new: page 2
docs/theory/system-integration.md     new: page 3 (Φ_s)
docs/theory/distinctions-and-relations.md  new: page 4
docs/theory/phi-structure.md      new: page 5 (+ paper-to-code map)
docs/theory/formalism-versions.md new: page 6
docs/theory/conditional-independence.md  new: page 7 (ported)
docs/theory/iit-3.0.md            new: page 8
docs/examples/conditional_independence.rst   deleted after porting to page 7
ROADMAP.md                        modified: dashboard row (final task)
```

---

## Shared asset: the page-authoring prompt

Each page task dispatches this prompt, filled with the page's `{TITLE}`,
`{FILE}`, `{SCOPE}`, `{SOURCES}`, and `{EXAMPLE}`.

```
You are writing one page of PyPhi's IIT 4.0 theory documentation.

Page: {TITLE}
File to create: {FILE}   (MyST Markdown)
Repo: /Users/will/projects/pyphi

## What this page covers
{SCOPE}

## Sources you MUST consult and cite accurately
{SOURCES}
- Read the relevant sections of the cited paper PDFs (use the Read tool with a
  page range) and the relevant pyphi source. Never cite an equation, section,
  or figure number you have not located in the actual PDF. Never state a code
  behavior you have not confirmed in the source.
- graphify-out/bridge-edges.json maps code files to paper concepts — use it to
  find where a concept is implemented.

## Voice and depth
- Self-contained and computation-driven: explain the concept for a reader doing
  IIT but not steeped in the 4.0 details, then map it to the PyPhi type/function
  and show it running. Do not re-derive the paper; cite it for depth.
- Plain, precise prose (the project's house style: no compressed shorthand, no
  quoted-phrase adjectives). Present tense, impersonal.
- Name the IIT postulate(s) the step embodies where the scope calls for it.

## The worked example (for executable pages)
{EXAMPLE}
- Use MyST executable cells: fenced ```{code-cell} python blocks. The FIRST
  code cell of the page must be:
      import pyphi
      pyphi.config.progress_bars = False
- Show the actual call and let its real output render. Do not paste output you
  typed by hand — myst-nb executes the cell and captures it.
- Use Unicode symbols (Φ, φ, φ_s) in prose, not LaTeX, except for genuine
  multi-part formulae which use $...$ (dollarmath is enabled).

## Build check
From the repo root, confirm the page builds and its cells execute:
  uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
The build must exit 0. If a cell errors, fix the code (it is wrong), not by
removing the check. Confirm your page's expected values (e.g. Φ_s = 0.208)
appear in docs/_build/html/theory/<page>.html.

## Report
Write to {REPORT_FILE}: every equation/section/figure citation you added with
the exact PDF locus and page where you confirmed it; every code call you show
and its confirmed output; anything you could not confirm and how you handled
it. Do NOT run git. Return: status, build result, citation count, report path.
```

## Shared asset: the verify prompt

```
You are the ACCURACY GATE for one page of PyPhi's IIT 4.0 theory docs.
Page file: {FILE}   Author report: {REPORT_FILE}   Repo: /Users/will/projects/pyphi

Verify, adversarially:
1. CITATIONS: for every equation/section/figure/theorem the page cites, open the
   cited paper PDF in papers/ and confirm it exists and says what the page
   claims. A locus you cannot find is a finding.
2. CODE CLAIMS: for every type, function, attribute, or output value the page
   states, confirm it against the pyphi source and by running the shown call if
   needed (env -u VIRTUAL_ENV uv run python). A value that does not reproduce is
   a finding.
3. EXECUTED OUTPUT: confirm the page builds under -W with cells executed
   (env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -W
   --keep-going -b html docs docs/_build/html) and that the prose matches the
   rendered cell output (e.g. if the prose says Φ_s = 0.208, the executed cell
   shows 0.208).
4. SELF-CONTAINMENT & VOICE: the page is understandable without the wiki; no
   development-process narrative; plain precise prose; postulates named where
   the scope requires.
Read-only; do not modify the tree or run git. Report findings
(Critical/Important/Minor, file:line, fix) and a verdict: Approved | Needs fixes.
```

## Page task table

Each row (Tasks 2–9) is: dispatch the authoring prompt with the row's fields →
build → dispatch the verify prompt → fix findings → accept → commit the page.
All authoring and verify agents run on **Opus** (deep theory + paper reading;
verification is the accuracy backstop). "Postulates" names the ones the page
grounds. `{EXAMPLE}` is the worked example slice the page shows.

| # | Page / file | Scope & postulates | Sources | Example slice |
|---|-------------|--------------------|---------|---------------|
| 2 | Overview — `overview.md` | The `Substrate → System → formalism → Φ-structure` layering (the architecture guide); the six postulates (existence, intrinsicality, information, integration, exclusion, composition) named and one-lined; introduce the worked example and what the section will compute. | Albantakis 2023 (postulates section + Fig. 1); the pyphi top-level API (`analyze`, `Substrate`, `System`). | Introduce `sub = pyphi.examples.basic_substrate()`, show its repr; state that at `(1, 1, 0)` PyPhi computes Φ_s and a Φ-structure of 3 distinctions and 2 relations, to be unpacked over the next pages. |
| 3 | Substrate and system — `substrate-and-system.md` | The causal model (the TPM); a candidate `System` in a state; **existence** and **intrinsicality** postulates; causal marginalization. | Albantakis 2023 (existence/intrinsicality; the cause TPM / causal marginalization equation — verify the number); `pyphi/substrate.py`, `pyphi/system.py`, `pyphi/core/tpm/`. | Build the substrate, construct `System(sub, (1,1,0), node_indices=(0,1,2))`, show the TPM/connectivity and the system repr. |
| 4 | System integrated information — `system-integration.md` | **Integration** and **exclusion** at the system level: system partitions, the minimum-information partition, Φ_s, and finding the complex. | Albantakis 2023 (system integrated information; system φ / MIP equations — verify numbers); `pyphi/analyze.py`, the SIA type, `pyphi/formalism/iit4/`. | `a = pyphi.analyze(sub, (1,1,0))`; show `a.phi` (Φ_s = 0.208), `a.sia.partition` (the MIP), `a.sia.normalized_phi`. |
| 5 | Distinctions and relations — `distinctions-and-relations.md` | **Composition** at the mechanism level: cause/effect repertoires, intrinsic information, the maximally irreducible cause–effect distinction (φ_d); then relations between distinction purviews (φ_r). | Albantakis 2023 (distinctions, MICE, intrinsic information — verify equation numbers, esp. the integrated-information/‖·‖+ equations; relations section + supplement S3); `pyphi/models/` (distinction, mice, relations), `pyphi/relations.py`, `pyphi/measures/`. | From `a.ces`: `.distinctions` (3), one distinction's `.mechanism`, `.phi`, `.cause_purview`/`.effect_purview`, a `.cause_repertoire`; then `.relations` (2), one relation's members. |
| 6 | The Φ-structure and paper-to-code map — `phi-structure.md` | Assemble distinctions + relations into the Φ-structure; the capstone reference table: every named (Greek-letter) quantity in the paper → its runtime type/function. | Albantakis 2023 (Φ-structure definition; the full symbol set); `graphify-out/bridge-edges.json`; the pyphi types for each quantity. | `a.ces` as the Φ-structure: `.big_phi`, `.sum_phi_distinctions`, `.sum_phi_relations`; `a.ces.to_pandas()` to show the structure as a table. |
| 7 | Formalism versions — `formalism-versions.md` | IIT 4.0 (2023) vs the 2026 intrinsic-information cap vs IIT 3.0 vs actual causation; how `config` selects among them. | Albantakis 2023; `2026__mayner-et-al__intrinsic-cause-effect-power.pdf` (the cap — verify the capping equation); Oizumi 2014 (3.0); Albantakis 2019 (AC); `pyphi/conf/`, `pyphi/formalism/`. | Show `pyphi.config.formalism.iit.version`, and switching formalism via `pyphi.config.override(...)` or the `formalism=` argument to `analyze`; name the `pyphi.iit3` / `iit4_2023` / `iit4_2026` namespaces. |
| 8 | Conditional independence — `conditional-independence.md` | Port `docs/examples/conditional_independence.rst`: the causal-model assumption (units are conditionally independent given the previous state) the framework rests on. | The existing `.rst`; `pyphi/tpm` / `validate` for how it is enforced. | Port the existing example, converting `.rst` to MyST and its examples to executed cells; verify the doctests still hold as executed cells. |
| 9 | IIT 3.0 overview — `iit-3.0.md` | Brief overview of the IIT 3.0 formalism and how it differs from 4.0; when to use it in PyPhi. | `papers/2014__oizumi-et-al__iit-3.0.pdf` (cite for depth); `pyphi/formalism/iit3/`. | Show computing under IIT 3.0 via the `iit3` formalism on the same substrate, for contrast (a short executed cell). |

For each of Tasks 2–9:

- [ ] **Step 1:** Dispatch the authoring prompt (Opus) filled with the row's fields and a per-page report path. Await status + build result.
- [ ] **Step 2:** Build the page yourself to confirm (`uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`, exit 0).
- [ ] **Step 3:** Dispatch the verify prompt (Opus). If Needs fixes, dispatch a fix agent with the findings, rebuild, re-verify until Approved.
- [ ] **Step 4:** Commit the page (stage only the page file; trailer required). For Task 8, also `git rm docs/examples/conditional_independence.rst` in the same commit and update any toctree/xref that pointed at it.

---

## Task 1: Theory section skeleton and toctree

**Files:** `docs/theory/index.md` (modify), the 8 page files (create as stubs).

- [ ] **Step 1: Create the eight stub pages.** Each is a MyST file with just a
  title and a one-line "under construction" note, e.g. `docs/theory/overview.md`:

  ```markdown
  # What IIT 4.0 computes

  ```

  Create all eight with their H1 titles: overview.md ("What IIT 4.0 computes"),
  substrate-and-system.md ("Substrate and system"), system-integration.md
  ("System integrated information"), distinctions-and-relations.md
  ("Distinctions and relations"), phi-structure.md ("The Φ-structure"),
  formalism-versions.md ("Formalism versions"), conditional-independence.md
  ("Conditional independence"), iit-3.0.md ("IIT 3.0").

- [ ] **Step 2: Wire the toctree.** Replace `docs/theory/index.md` with:

  ````markdown
  # Theory

  How IIT 4.0's quantities map onto PyPhi's types and functions, grounded in a
  single worked example. For a broad orientation to the theory itself, see also
  the [IIT wiki](https://iit.wiki); the authoritative source is
  [Albantakis et al. (2023)](https://doi.org/10.1371/journal.pcbi.1011465).

  ```{toctree}
  :maxdepth: 1

  overview
  substrate-and-system
  system-integration
  distinctions-and-relations
  phi-structure
  formalism-versions
  conditional-independence
  iit-3.0
  ```
  ````

- [ ] **Step 3: Build.** `uv run --all-extras --group docs sphinx-build -W
  --keep-going -b html docs docs/_build/html` → exit 0 (stub pages, no
  warnings; the old `../examples/conditional_independence` reference is gone
  from this toctree — Task 8 deletes the source).

- [ ] **Step 4: Commit.**

  ```bash
  git add docs/theory/
  git commit -m "Scaffold the IIT 4.0 theory section"
  ```

---

## Task 10: Whole-section gate and ROADMAP

- [ ] **Step 1: Clean-slate build.** `rm -rf docs/_build docs/reference/_autosummary`
  then the `-W` build → exit 0, every theory page's cells executed.
- [ ] **Step 2: Consistency check.** Grep the built theory pages for the
  worked-example values and confirm they agree across pages (e.g. Φ_s = 0.208
  appears the same wherever cited): `grep -rl "0.208" docs/_build/html/theory/`.
- [ ] **Step 3: No stale cross-references.** Confirm nothing references the
  deleted `examples/conditional_independence`:
  `grep -rn "conditional_independence" docs --include="*.md" --include="*.rst" | grep -v theory/conditional-independence`.
- [ ] **Step 4: ROADMAP.** Update the Documentation-overhaul dashboard row and
  the "Theory" bullet in the overhaul section to mark the theory narrative
  landed; note remaining pieces are tutorials/how-tos and the migration guide.
- [ ] **Step 5: Commit** the ROADMAP change.

---

## Self-review checklist

- Spec pages (8) → Tasks 2–9 one-to-one. ✓
- Spec "architecture guide" → folded into Overview (Task 2). ✓
- Spec "paper-to-code map" → Task 6 (phi-structure.md). ✓
- Spec accuracy (paper-verified equations, code-verified claims, two-stage
  verify) → authoring + verify prompts + per-page Steps 1–3. ✓
- Spec executable one-example → Global Constraints + the `{EXAMPLE}` column
  (worked example `(1, 1, 0)`, verified 3 distinctions / 2 relations / Φ_s
  0.208). ✓
- Spec self-contained + wiki pointer → authoring prompt voice + Task 1 toctree
  intro. ✓
- Spec success criteria → Task 10 gate. ✓
- Placeholder scan: the per-page equation numbers are intentionally left for
  the authoring agents to VERIFY against the PDFs (citing a number here I have
  not confirmed would violate the accuracy rule); the scope names the concept
  and the agent confirms the locus. This is a deliberate accuracy safeguard,
  not a placeholder gap.
