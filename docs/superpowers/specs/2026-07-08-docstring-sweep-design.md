# Docstring sweep (final-state voice) — design

The second sub-project of the documentation overhaul
(`docs/superpowers/specs/2026-07-07-documentation-overhaul-design.md` §5).
Rewrites every docstring under `pyphi/**` to a single, consistent standard:
final-state, plain, precise, and impersonal. This is the text `autodoc`
renders for the API reference and the text maintainers read at the source, so
it is content work, not cleanup.

## Scope

Every module under `pyphi/**` — all 163 files, public and private. Private
modules are included: their docstrings do not render on the site, but they are
what maintainers read, and a uniform standard across the whole package is the
goal. The units of work are module docstrings, class docstrings, method and
function docstrings, and any attribute/parameter prose within them.

Explicitly out of scope: code changes of any kind. This sweep edits docstring
prose only. If an agent finds a bug, a wrong type annotation, or a docstring
that is accurate only because the code is wrong, it reports the finding — it
does not fix the code.

## The standard

Every rewritten docstring must satisfy all of the following. These are the
rules that go, verbatim, into both the rewrite prompt and the verification
prompt.

### Accuracy (paramount)

- **Never assume.** Before rewriting a docstring, read the full implementation
  it describes — the whole function or class body, not the signature or the
  old docstring. If what the code does is unclear, read its callers, its
  callees, and the tests that exercise it. Over-read: spend tokens on
  exploration rather than economy. A docstring that is plausible but wrong is
  the worst outcome of this sweep, worse than one left unchanged.
- **The code is the source of truth, not the old docstring.** The existing
  docstring may be stale or wrong. Verify every claim against the
  implementation. When the old docstring and the code disagree, the code
  wins, and the disagreement is reported.
- **Preserve technical precision.** Parameter names, types, return types,
  units, ranges, defaults, raised exceptions, and mathematical definitions
  must be exactly right. Do not round off a precise statement into a vague
  one for the sake of readability.

### Voice: final-state

- Describe what the thing **is** and **does**, in the present tense. Never
  what it was, what it replaces, what it used to be called, or how it came to
  exist. No "formerly", "renamed from", "previously", "legacy", "new".
- **Substantive insight is welcome; process narrative is not.** The line is
  not "no *why*" — it is what the *why* is about. Explanation that helps the
  reader understand the concept, the algorithm, or correct usage belongs in
  the docstring, in the spirit of NumPy's reference docstrings: a mathematical
  fact, a non-obvious property, a complexity bound, a numerical-stability
  caveat, a subtle usage requirement, the reason a result takes a particular
  form. Include such insight even when it is not evident from reading the code
  and is currently recorded only in a spec, a test, the roadmap, or a cited
  paper — if it is genuinely part of understanding the code, its home is the
  docstring, and moving it there adds value. What must go is narrative about
  the **development process**: how the code came to be, what it used to be,
  what it replaced, alternatives framed as choices "we" made, or anything that
  reads as contamination from a plan or a conversation. The test: would this
  sentence still make sense, and still be worth reading, to someone who uses
  PyPhi for years and never sees its git history? If yes, it can stay; if it
  only makes sense as a footnote to how the code was built, it goes.
- **Additions are governed by accuracy exactly as rewrites are.** Insight
  brought into a docstring must be verifiable against the code or the cited
  source. Never import an unverified claim from an outside document; if a spec
  or roadmap statement cannot be confirmed against the implementation or a
  cited paper, it is reported, not added.
- Impersonal. No first person, no reference to the process that produced the
  code, no context that belongs to a conversation or a plan rather than to
  the software. A fresh reader with no history must find nothing that assumes
  history.

### Literature references

Where a docstring documents something that implements or derives from a
specific result in the IIT literature, cite it precisely — the equation,
section, definition, theorem, or figure number, with the paper in short form.
The original docstrings rarely cited the literature; adding these pointers is
part of the sweep's value, matching the citation practice used in this
project's specs.

Sources, in order of use:

- **`graphify-out/bridge-edges.json`** (committed, directly readable JSON — no
  tooling required): 238 curated `implements`/`cites` edges mapping each code
  file to the paper concepts it realizes. Use it to find *which* papers and
  concepts a module relates to. Heed the `confidence` field: `EXTRACTED` edges
  are grounded in the source; `INFERRED` edges are hypotheses to confirm
  against the paper, not facts to cite blind.
- **The papers themselves**, in `papers/` (readable as PDFs), for the
  *specific* locus — the bridge edge names the concept, but the equation or
  section number comes from the paper. Primary references:
  `2023__albantakis-et-al__iit-4.0.pdf` and its supplements S1–S4 (ties, 3.0
  comparison, analytical relations, algorithm) for IIT 4.0;
  `2014__oizumi-et-al__iit-3.0.pdf` for IIT 3.0;
  `2019__albantakis-et-al__what-caused-what.pdf` for actual causation;
  `2023__marshall-et-al__system-integrated-information.pdf`,
  `2020__barbosa-et-al__intrinsic-information.pdf`, and
  `2024__zaeemzadeh-tononi__upper-bounds.pdf` for the modules that implement
  them. Reading the referenced paper is part of over-reading for any module
  that implements a paper result.

Accuracy governs citations as strictly as prose:

- A cited equation, section, or figure number must be verified against the
  actual paper. Never cite a number from memory, infer it from a concept's
  name, or take it from an `INFERRED` bridge edge without confirming it.
- If the precise locus cannot be confirmed, cite at the level you can confirm
  (the paper, or a named section) or omit the pointer. A wrong citation is
  worse than none; the verification stage treats an unconfirmable citation as
  a finding.
- Use a consistent short form: "Albantakis et al. (2023), Eq. 12" /
  "Oizumi et al. (2014), §II".

### Prose: plain and clear

- Prefer plain language to jargon where plain language is exact. Do not use an
  obscure technical term where a common one is equally precise. But do not
  sacrifice precision for simplicity: use the exact technical term when it is
  the correct one, and define or link it.
- No compressed or shorthand constructions: avoid quoted phrases used as
  adjectives, long hyphen-joined phrases used as nouns, and stacked modifiers
  that require unpacking. Write the sentence out. (This mirrors the project's
  house prose style.)
### Format: NumPy style

Docstrings use **NumPy style** (rendered by napoleon), the scientific-Python
standard. A real one-line summary, then prose, then the sections that apply:
`Parameters`, `Returns`, `Yields`, `Raises`, and — where they add value —
`Notes` (the home for mathematical insight and non-obvious properties) and
`References` (numbered literature citations). This is a change from the older
Google style; the sweep converts each docstring's section syntax as it
rewrites. Keep the summary line a real summary.

Citations use the `References` section with numbered entries and `[1]_`
in-text references, the conventional scientific form:

```
Notes
-----
Φ is the minimum over system partitions of the integration measure
[1]_, Eq. 8.

References
----------
.. [1] Albantakis et al. (2023). Integrated information theory (IIT) 4.0.
       PLoS Comput Biol 19(10): e1011465.
```

Because each documented object renders on its own page (the recursive
autosummary layout), citation labels are page-local and do not collide across
modules. If the `-W` build ever surfaces a duplicate-citation warning, fall
back to an inline short-form citation ("Albantakis et al. (2023), Eq. 8") for
that docstring.

### Preserved verbatim (never altered by this sweep)

- **Doctests.** Any line beginning with `>>>` or `...` and its expected
  output is executable and tested. Reproduce it exactly, including
  whitespace. If a doctest looks wrong, report it — do not edit it.
- **Mathematical notation and equation/paper citations.** Preserve references
  to IIT-paper equations, theorems, and figures exactly.
- **Cross-reference roles** (`:class:`, `:func:`, `:meth:`, `:mod:`) that
  already resolve. Rewrites may add correct roles but must not break existing
  ones.
- **Signatures.** This sweep does not touch code, including parameter names
  and type annotations.

### Symbols: Unicode for the common case, `:math:` for real formulae

A docstring is a single source string, and no terminal renders LaTeX, so there
is no automatic way to get typeset math on the web and readable symbols under
`help()` at once. The strategy is to pick, per expression, the representation
that is acceptable in both, and fall back to web-only `:math:` solely where
Unicode cannot express the thing.

- **Simple symbols and moderate super/subscripts → Unicode.** Replace the
  `|big_phi|`-style RST substitutions (defined in the `rst_prolog` block of
  `docs/conf.py`) with literal Unicode: `|big_phi|` → `Φ`, `|small_phi|` → `φ`,
  `|small_phi_s|` → `φₛ`, `|small_phi_max|` → `φᵐᵃˣ`, `|big_alpha|` → `𝒜`,
  `|alpha|` → `α`, and so on for the full table. Use Unicode super/subscript
  characters (`φₛ`, `Φᵐᵃˣ`) where they exist — these read well under `help()`
  and render acceptably on the web, widening the range that is good in both
  contexts. A lone Greek letter looks essentially the same as its MathJax
  rendering.
- **Genuine multi-part formulae → `:math:`.** Fractions, sums, products, and
  aligned expressions (e.g. `\frac{AB}{DE} \times \frac{\varnothing}{C}`) use
  the `:math:` role: MathJax typesets them on the web, and the raw LaTeX in the
  terminal is the accepted cost — these are rare and remain parseable.

After the sweep, no `|…|` substitution markup remains in `pyphi/**`, and the
`rst_prolog` block can be deleted (the last content plan does so).

## Verification

Accuracy is enforced structurally, not trusted. Each batch of modules passes
through two stages with independent agents:

1. **Rewrite.** One agent rewrites the docstrings in a batch of modules,
   following the standard above and the over-reading protocol. It commits its
   batch.
2. **Accuracy verification.** A second, independent agent reviews the batch's
   diff. It reads the **current code** behind each changed docstring (not the
   old docstring) and confirms the new prose is accurate against the
   implementation. It confirms every literature citation the rewrite added or
   changed against the actual paper (an equation or section number that cannot
   be located in the cited paper is a finding). It separately confirms:
   doctests are byte-identical to before, no substitution markup remains, no
   code changed, and the voice/prose rules hold. It reports any docstring
   whose new text it cannot confirm from the code, any unlocatable citation,
   and any code-level bug the rewrite surfaced. Findings go back to a fix pass
   before the batch is accepted.

Mechanical gates run on the whole sweep, not per batch:

- `uv run --all-extras --group docs sphinx-build -W --keep-going -b html docs
  docs/_build/html` stays green (no docstring introduced a rendering warning).
- `uv run --all-extras pytest` stays green — this is the doctest gate (the
  14 doctest-bearing modules run under pytest's `--doctest-modules`), and it
  confirms no prose edit disturbed executable content.
- `grep -rn '|[A-Za-z]' pyphi --include="*.py"` finds no surviving
  substitution markup in docstrings.

## Decomposition

The work divides by subpackage into batches sized so one agent can hold a
batch's code in context and over-read it properly. The natural units (from the
package layout):

- The `pyphi/` root (35 modules, ~13k LOC) splits into thematic batches: the
  core value types (`substrate`, `system`, `node`, …), the computation entry
  points (`analyze`, `sweep`, …), and the utilities (`utils`, `convert`,
  `connectivity`, `combinatorics`, …).
- `pyphi/models` (17 modules) splits into two batches.
- Each remaining subpackage is one batch, with the largest
  (`formalism/iit4`, `measures`, `macro`, `serialize`, `conf`, `core/tpm`,
  `visualize/render`) split by file where a single batch would exceed a
  comfortable reading load.

Batches are independent — each touches a disjoint set of files — so the
rewrite stage parallelizes. The plan enumerates the exact batches and the
files in each.

## Isolation

The sweep touches ~163 files while other sessions commit to `pyphi/` on
`main`. To avoid the interleaving races already seen in this project, the
sweep runs on a dedicated branch in a worktree under `.claude/worktrees/`,
merged to `main` in one reviewed step when complete. (Worktree setup follows
the project's venv note: install into the worktree's own environment. The plan
covers the exact commands.) The alternative — running on `main` with
per-batch commits — is workable but exposes every batch to the same
cross-session races; the worktree is the safer default for a pass this large.

## Success criteria

- Every module under `pyphi/**` has been through the rewrite-and-verify
  process; no docstring retains migration-journey phrasing, development-process
  narrative, first-person or process contamination, or substitution markup —
  while substantive subject-matter insight is preserved and, where valuable,
  surfaced into the docstring from adjacent sources.
- Each batch's accuracy verification passed with no unresolved findings.
- The `-W` docs build is green; the full pytest suite (doctests included) is
  green; no substitution markup remains in `pyphi/**`.
- Any code-level bugs the sweep surfaced are recorded as separate findings for
  follow-up, not silently fixed inside the sweep.
