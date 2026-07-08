# Docstring Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite every docstring under `pyphi/**` to the standard defined in
`docs/superpowers/specs/2026-07-08-docstring-sweep-design.md` — final-state,
plain, precise, impersonal, with literature citations — verified for accuracy
against the code and the papers.

**Architecture:** The work runs in an isolated worktree on a dedicated branch.
Modules are grouped into ~20 subpackage-sized batches, each of disjoint files.
Each batch passes through two independent agents: a **rewrite** agent (follows
the rewrite prompt below) and an **accuracy-verification** agent (follows the
verify prompt below). Findings from verification go to a fix pass before the
batch is accepted. Whole-sweep mechanical gates (docs build, pytest doctests,
substitution grep) run before the branch merges to `main`.

**Tech Stack:** Python 3.13+, Sphinx/autodoc (renders these docstrings),
pytest `--doctest-modules`, the papers in `papers/`, the curated
`graphify-out/bridge-edges.json`.

## Global Constraints

- **Prose only.** No code changes of any kind — not signatures, not type
  annotations, not logic. A discovered bug is reported, never fixed here.
- **Doctests are inviolable.** Lines beginning `>>>` / `...` and their expected
  output are reproduced byte-for-byte. If one looks wrong, report it.
- **Accuracy is paramount.** Over-read the implementation before rewriting;
  the code is the source of truth; never assume; confirm every citation
  against the actual paper.
- **The full standard** is in the spec's "The standard" section and is
  reproduced in the rewrite prompt below — that prompt is the operational
  contract.
- Never `--no-verify`; never `git add -A` (stage only the batch's files);
  commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- No planning-artifact markers (P-/N-/B-numbers, "Wave", task/batch numbers)
  in docstrings.

---

## Task 1: Create the sweep worktree

**Files:** none in `pyphi/` — environment setup only.

- [ ] **Step 1: Create the branch and worktree**

```bash
cd /Users/will/projects/pyphi
git worktree add -b docstring-sweep .claude/worktrees/docstring-sweep main
```

- [ ] **Step 2: Provision the worktree environment**

The worktree needs its own environment; `.envrc` points `VIRTUAL_ENV` at the
main `.venv`, so unset it for the install (per the project's worktree note):

```bash
cd /Users/will/projects/pyphi/.claude/worktrees/docstring-sweep
env -u VIRTUAL_ENV uv sync --all-extras --group docs
```

- [ ] **Step 3: Baseline the gates**

Confirm the starting point is green so later failures are attributable:

```bash
env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs docs/_build/html   # exit 0
env -u VIRTUAL_ENV uv run --all-extras pytest -q -x --doctest-modules pyphi 2>&1 | tail -3                        # doctests green
```

All batch work happens in this worktree. The controller records this worktree
path and uses it for every subsequent dispatch.

---

## Shared asset: the rewrite prompt

Every batch's rewrite agent is dispatched with this prompt, filling
`{FILES}`, `{PAPERS}`, and `{WORKTREE}`. It is the operational form of the
spec's standard.

```
You are rewriting docstrings for accuracy and clarity in these files:
{FILES}

Work in the worktree: {WORKTREE} (all paths relative to it). You may ONLY edit
docstrings in the listed files. Do not change code — not signatures, not type
annotations, not logic. Do not edit any file not listed.

## What you are producing

Every module, class, method, and function docstring in the listed files,
rewritten to a single standard. Most existing docstrings are decent; this is
targeted correction and polish, not blind rewriting. Leave a docstring
unchanged if it already meets the standard — but read the code first to know
that it does.

## Accuracy is paramount — the overriding rule

- Before you touch a docstring, READ THE FULL IMPLEMENTATION it describes: the
  whole function or class body, not the signature or the old docstring. If
  what the code does is unclear, read its callers, its callees, and the tests
  that exercise it. Over-read. Spend tokens on exploration, not economy.
- The CODE is the source of truth, not the existing docstring — which may be
  stale or wrong. Verify every claim against the implementation. Where the old
  docstring and the code disagree, the code wins, and you note the
  disagreement in your report.
- Preserve technical precision exactly: parameter names, types, return types,
  units, ranges, defaults, raised exceptions, mathematical definitions. Never
  round a precise statement into a vague one for readability.
- A plausible-but-wrong docstring is the worst possible outcome — worse than
  leaving one untouched. If you cannot confirm what something does, say so in
  your report and leave it.

## Voice: final-state

- Present tense; describe what the thing IS and DOES. Never what it was, what
  it replaces, what it used to be called, or how it came to exist. Remove
  "formerly", "renamed", "previously", "legacy", "new", and similar.
- Impersonal. No first person, no reference to the development process, no
  context that belongs to a plan or conversation. A fresh reader who never
  sees the git history must find nothing that assumes history.

## Substantive insight is welcome; process narrative is not

The rule is not "no why" — it is what the why is about.
- KEEP and ADD subject-matter insight that helps the reader understand the
  concept, the algorithm, or correct usage: a mathematical fact, a non-obvious
  property, a complexity bound, a numerical-stability caveat, a subtle usage
  requirement, the reason a result takes a particular form. This is the
  NumPy-docstring standard. Bring such insight IN even when it is not evident
  from the code and currently lives only in a test, a spec, or a cited paper —
  if it is part of understanding the code, its home is the docstring.
- REMOVE development-process narrative: how the code came to be, what it
  replaced, alternatives framed as choices "we" made, plan/conversation
  contamination.
- Test for any sentence: would it still make sense, and be worth reading, to
  someone who uses PyPhi for years and never sees its git history? If yes, it
  can stay. If it only makes sense as a footnote to how the code was built,
  cut it.
- Anything you ADD is governed by accuracy exactly as rewrites are: it must be
  verifiable against the code or a cited source. Never import an unverified
  claim from an outside document — if a spec/roadmap statement cannot be
  confirmed against the implementation or a paper, report it, do not add it.

## Prose: plain and clear

- Prefer plain language where it is equally exact; do not use an obscure term
  where a common one is just as precise. But never sacrifice precision for
  simplicity — use the correct technical term when it is correct, and
  define/link it.
- No compressed shorthand: no quoted phrases used as adjectives, no long
  hyphen-joined phrases used as nouns, no stacked modifiers that need
  unpacking. Write the sentence out.
- Use NumPy docstring style (napoleon), the scientific-Python standard — NOT
  Google style. A real one-line summary, then prose, then the sections that
  apply: Parameters, Returns, Yields, Raises, and where they add value Notes
  (the home for mathematical insight and non-obvious properties) and References
  (numbered literature citations). Convert any Google-style
  (Args:/Returns:/Raises:) docstring you touch to NumPy sections as part of the
  rewrite. Example:

      Parameters
      ----------
      system : System
          The candidate system in a definite state.

      Returns
      -------
      float
          The value Φ ≥ 0; zero iff the system is reducible.

      Notes
      -----
      Φ is the minimum over system partitions of the integration
      measure [1]_, Eq. 8.

      References
      ----------
      .. [1] Albantakis et al. (2023). Integrated information theory
             (IIT) 4.0. PLoS Comput Biol 19(10): e1011465.

## Literature references

Cite the literature where a docstring documents something that implements or
derives from a specific result. Relevant papers for THIS batch:
{PAPERS}

- Use `graphify-out/bridge-edges.json` (read it directly; it is committed JSON)
  to find which papers/concepts a file realizes. Heed the `confidence` field:
  `INFERRED` edges are hypotheses to confirm, not facts.
- Use the actual PDF in `papers/` for the specific equation/section/figure
  number — reading the referenced paper is part of over-reading here.
- Accuracy governs citations strictly: a cited number must be verified in the
  actual paper. Never cite from memory, from a concept's name, or from an
  unconfirmed INFERRED edge. If you cannot confirm the precise locus, cite at
  the level you can (the paper, or a named section) or omit it — a wrong
  citation is worse than none.
- Short form: "Albantakis et al. (2023), Eq. 12" / "Oizumi et al. (2014), §II".

## Preserved verbatim

Doctests (`>>>`/`...` and their output), mathematical notation, existing
working cross-reference roles (:class:/:func:/:meth:/:mod:), and signatures.

## Symbols

Replace `|big_phi|`-style RST substitutions with literal Unicode, using
super/subscript characters where they exist (they read well in `help()` AND
render fine on the web): |big_phi|→Φ, |small_phi|→φ, |small_phi_s|→φₛ,
|small_phi_max|→φᵐᵃˣ, |big_alpha|→𝒜, |alpha|→α, etc. After your edits, no `|…|`
substitution remains in the listed files. Use the `:math:` role only for
genuine multi-part formulae (fractions, sums, products) that Unicode cannot
express cleanly — MathJax typesets those on the web.

## Process

1. Read each listed file in full, and the implementations behind its
   docstrings, before editing.
2. Consult the bridge edges and the relevant papers for citations.
3. Edit docstrings only. Run `env -u VIRTUAL_ENV uv run --all-extras pytest
   --doctest-modules {FILES}` to confirm doctests still pass.
4. Commit the batch (stage only the listed files; trailer required).
5. Write your report to {REPORT_FILE}: files touched; per-file summary of what
   changed and why; every citation you added with the exact paper locus and
   how you confirmed it; every disagreement between old docstring and code;
   every code-level bug or wrong annotation you found (reported, NOT fixed);
   any docstring you could not confirm and left as-is.

Report back (under 15 lines): status, commit SHA, doctest result, count of
citations added, count of bugs/disagreements reported, report file path.
```

## Shared asset: the verify prompt

Every batch's verification agent is dispatched with this prompt after the
rewrite commits.

```
You are verifying a docstring-rewrite batch for ACCURACY. Files:
{FILES}
Diff to review: {DIFF_FILE}   Rewrite agent's report: {REPORT_FILE}
Worktree: {WORKTREE}

Your job is to catch docstrings that are now wrong, citations that don't
check out, doctests that changed, and any code that was altered. You are the
accuracy gate; the rewrite agent's report is unverified claims.

## Checks (report findings with file:line)

1. ACCURACY: For each changed docstring, read the CURRENT CODE behind it (not
   the old docstring, not the report) and confirm the new prose is true of the
   implementation. Flag anything the code does not support. This is the
   priority.
2. CITATIONS: For every literature citation the diff adds or changes, confirm
   the cited equation/section/figure exists in the cited paper (in `papers/`)
   and matches the claim. A number you cannot locate in the paper is a finding.
3. DOCTESTS: Confirm every `>>>`/`...` line and its output is byte-identical to
   the pre-diff version (the diff shows this — docstring doctest lines must not
   appear as changed content unless only reflowed prose around them moved).
4. NO CODE CHANGED: Confirm the diff touches only docstring content — no
   signature, annotation, or logic line changed.
5. SUBSTITUTIONS: Confirm no `|…|` substitution markup remains in the files.
6. VOICE/PROSE: Confirm no development-process narrative, first person, or
   process contamination entered; confirm no compressed-shorthand constructions
   (quoted-phrase adjectives, hyphen-chain nouns). Subject-matter insight is
   allowed and good — do not flag it.
7. FORMAT: Confirm docstrings use NumPy style (Parameters/Returns/Raises with
   underlined section headers; Notes/References where used), not Google style
   (Args:/Returns:). A docstring left in Google style is a finding. Citations
   use the References section with numbered `[1]_` entries, or an inline
   short-form ("Albantakis et al. (2023), Eq. 8").

Read-only: do not modify the tree. Inspect code beyond the diff freely to
confirm accuracy (that is the point). Report each finding as
Critical/Important/Minor with file:line, what's wrong, and the fix. End with a
verdict: Approved | Needs fixes.
```

## Batch table

Each row is one task: dispatch the rewrite prompt with these files and papers,
commit, dispatch the verify prompt, fix findings, accept. Model tier is a
floor; dense theory batches use the more capable tier. "Papers" names the
primary reference(s); the agent still consults the bridge edges.

| # | Batch | Files (under `pyphi/`) | Papers | Tier |
|---|-------|------------------------|--------|------|
| 2 | root-core-types | substrate.py, system.py, node.py, direction.py, types.py, protocols.py | Albantakis 2023 (IIT 4.0) + S4 algorithm | high |
| 3 | root-orchestration | analyze.py, sweep.py, resolve_ties.py, relations.py, partition.py, compositional_state.py, distribution.py, dynamics.py, timescale.py | IIT 4.0 + S1 (ties) + S3 (relations); Mayner 2022 counting-relations | high |
| 4 | root-actual | actual.py | Albantakis 2019 (what-caused-what) | high |
| 5 | root-utils | convert.py, utils.py, combinatorics.py, connectivity.py, automorphism.py, relabel.py, graph.py | — (mostly formalism-agnostic); IIT 4.0 for TPM conventions | standard |
| 6 | root-infra | __init__.py, examples.py, estimate.py, labels.py, validate.py, provenance.py, registry.py, exceptions.py, constants.py, warnings.py, log.py, serializable.py | IIT 4.0 (example networks cite paper figures) | standard |
| 7 | cache | cache/*.py | — | standard |
| 8 | conf | conf/*.py | IIT 4.0 (measure/formalism options) | standard |
| 9 | core | core/*.py, core/tpm/*.py | IIT 4.0 + S4; Barbosa 2020 (intrinsic information) for repertoire algebra | high |
| 10 | data-display | data_structures/*.py, deferred/*.py, display/*.py, display/render/*.py | — | standard |
| 11 | formalism-base | formalism/{__init__,base,queries}.py, formalism/actual_causation/*.py, formalism/iit3/*.py | Oizumi 2014 (IIT 3.0); Albantakis 2019 (AC) | high |
| 12 | formalism-iit4 | formalism/iit4/{__init__,bounds,formalism}.py | Albantakis 2023 + S4; Zaeemzadeh 2024 (upper bounds) for bounds.py | high |
| 13 | macro | macro/*.py | Marshall 2024 (intrinsic units); Hoel/macro; Marshall 2023 (system integration) | high |
| 14 | matching | matching/*.py | Mayner 2024 (intrinsic meaning, perception, matching) | high |
| 15 | measures | measures/*.py | Barbosa 2020 (intrinsic information); Barbosa 2021 (mechanism integrated information); IIT 4.0 | high |
| 16 | models-a | models/{sia,ces,complex,distinction,distinctions,mice,ria}.py | Albantakis 2023 (IIT 4.0) | high |
| 17 | models-b | models/{actual_causation,partitions,state_specification,explanation,diff,cmp,fmt,pandas,protocols,__init__}.py | IIT 4.0; Albantakis 2019 (AC models) | standard |
| 18 | parallel | parallel/*.py, parallel/backends/*.py | — | standard |
| 19 | serialize | serialize/*.py | — | standard |
| 20 | substrate_generator | substrate_generator/*.py | Gomez 2020 (multi-valued elements) for TPM building; Ising refs | standard |
| 21 | visualize | visualize/*.py, visualize/projection/*.py, visualize/render/*.py | IIT 4.0 (what is being visualized) | standard |

For each batch task (2–21):

- [ ] **Step 1:** Dispatch the rewrite prompt (filled with the row's files, papers, worktree path, and a per-batch report-file path). Await DONE + commit SHA.
- [ ] **Step 2:** Generate the diff (`git -C {WORKTREE} show <SHA>` or a review-package) and dispatch the verify prompt.
- [ ] **Step 3:** If verify returns Needs fixes, dispatch a fix agent with the findings (prose-only, re-run the batch's doctests), then re-verify. Repeat until Approved.
- [ ] **Step 4:** Record the batch's added-citation count and any reported bugs/disagreements in the sweep findings log (`docstring-sweep-findings.md` in the worktree root — these are the code-bug follow-ups and citation records).

---

## Task 22: Whole-sweep gates and merge

**Files:** verification only, then the merge.

- [ ] **Step 1: Substitution grep is clean**

```bash
cd /Users/will/projects/pyphi/.claude/worktrees/docstring-sweep
grep -rn '|[A-Za-z]' pyphi --include="*.py" | grep -v "|=\||\s*$" || echo "no substitutions remain"
```
Expected: no `|big_phi|`-style markup remains. (Bitwise-or operators in code are fine; the grep is scoped to docstring-style `|word|` — inspect any hit.)

- [ ] **Step 2: Docs build green with warnings-as-errors**

```bash
env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: exit 0 (no docstring introduced a rendering warning).

- [ ] **Step 3: Full suite green (the doctest gate)**

```bash
(env -u VIRTUAL_ENV uv run --all-extras pytest -q; echo "RC=$?") > /tmp/docstring-sweep-suite.log 2>&1 &
```
Judge by the `RC=` line. Expected: RC=0 — no prose edit disturbed executable doctest content.

- [ ] **Step 4: Review the findings log**

Read `docstring-sweep-findings.md`. Every reported code-level bug, wrong
annotation, and old-docstring/code disagreement is a follow-up item — surface
them to the user; they are NOT fixed in this sweep. Confirm no finding was a
code change smuggled into the sweep.

- [ ] **Step 5: Disable Google-style parsing (consistency guard)**

The sweep standardizes on NumPy style. napoleon renders NumPy by default with
no config change needed, but after the sweep, set `napoleon_google_docstring =
False` in `docs/conf.py` so any future Google-style docstring is caught rather
than silently rendered. Rebuild `-W` to confirm nothing regressed. (Commit this
one-line conf change with the merge.)

- [ ] **Step 6: Merge to main**

After the user reviews the branch, merge it in one step (ask before any push):

```bash
cd /Users/will/projects/pyphi
git checkout main && git pull --ff-only 2>/dev/null; git merge --no-ff docstring-sweep
```
Then remove the worktree per finishing-a-development-branch, and update the
ROADMAP row and the umbrella spec's P15/docs-overhaul status.

---

## Self-review checklist

- Spec "The standard" → the rewrite prompt reproduces all four groups
  (accuracy, voice, insight-vs-narrative, prose) plus citations and preserved-
  verbatim. ✓
- Spec "Verification" (two independent stages, citation check, doctest/no-code
  checks) → the verify prompt + per-batch Steps 2–3. ✓
- Spec "Decomposition" (subpackage batches, big ones split) → batch table
  (models→a/b, iit4 its own, root→5 batches). ✓
- Spec "Isolation" (worktree) → Task 1 + Task 22 merge. ✓
- Spec "Success criteria" (build, pytest, grep, bug-findings recorded) →
  Task 22. ✓
- Every file from the inventory appears in exactly one batch row. (Verify by
  listing before executing — a missed file is an unswept docstring.)
