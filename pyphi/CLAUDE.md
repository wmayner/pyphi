# Docstring style (enforced)

All `pyphi/**` docstrings follow one standard, enforced by the docs build
(`napoleon_google_docstring = False`, so a Google-style section fails the
`-W` build). New and modified docstrings must match it:

- **NumPy style, not Google.** Underlined sections — `Parameters`,
  `Returns`, `Yields`, `Raises`, and, where they add value, `Notes` and
  `References` (never `Args:`/`Returns:` colon headers). A param is
  `name : type` with the description indented beneath.
- **Final-state, impersonal voice.** Describe what the thing *is* and
  *does*, in the present tense. Never what it was, replaced, or how it was
  built; no first person, no development-process or planning narrative. The
  test: would the sentence still make sense to someone who uses PyPhi for
  years and never sees its git history?
- **Substantive insight is welcome.** A mathematical fact, a complexity
  bound, a numerical-stability caveat, or a subtle usage requirement belongs
  in `Notes` (NumPy-docstring spirit). The rule bans *process* narrative,
  not *subject-matter* explanation. Anything asserted must be verifiable
  against the code or a cited source.
- **Plain, precise prose.** Prefer plain words where they are equally exact;
  never sacrifice precision for simplicity. Avoid compressed shorthand
  (quoted-phrase adjectives, hyphen-chain nouns, stacked modifiers).
- **Symbols: Unicode, not RST substitutions.** Write `Φ`, `φ`, `φₛ`, `𝒜`,
  `α`, `×`, `−` directly (they read correctly under `help()`); use Unicode
  sub/superscripts where they exist. These are whitelisted in ruff's
  `allowed-confusables`. Reserve the `:math:` role for genuine multi-part
  formulae (fractions, sums) that Unicode cannot express. Do **not**
  reintroduce `|big_phi|`-style substitution markup — there is no
  `rst_prolog` defining it.
- **Cite the literature** where a docstring documents a paper result — the
  exact equation/section/figure number, verified against the actual paper
  (`papers/`) — in a `References` section (`[1]_` entries) or inline
  short-form ("Albantakis et al. (2023), Eq. 33"). Never cite a number from
  memory. `graphify-out/bridge-edges.json` maps code files to paper
  concepts.
- **Doctests are executable and tested** (`--doctest-modules`); keep `>>>`
  lines and their output correct.
