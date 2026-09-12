# Documentation site restyle: design

**Date:** 2026-09-12
**Status:** approved in review (decisions recorded from the design proposal artifact)

## Goal

Give the documentation site a considered visual identity without changing
any page's content: a palette anchored on the result cards' cause and effect
colours, one type superfamily, matched code-block styles for both themes,
result cards that follow the site, a landing page that leads with the cards,
styled tables and admonitions, and the theme's release chrome. Alongside it,
make the IIT 4.0 demo notebook page show the notebook itself, with guards
against its stored outputs going stale.

## Scope

- `docs/_static/custom.css` (currently 7 lines) carries the whole visual
  system as CSS variables and component rules.
- `docs/conf.py` gains theme options (announcement, version switcher,
  Pygments styles, sidebar footer template) and registers the Selenized
  Pygments styles from `docs/_ext/selenized.py`.
- `docs/index.md` is reordered (tagline, cards with glyphs, condensed
  citation strip).
- `pyphi/display`: one change, a single label-column width per card for
  key/value sections, in both the HTML and the ASCII backends; table headers
  in the HTML backend stay in the sans face.
- `docs/tutorials/iit-4.0-demo.md` becomes the notebook itself, rendered
  from stored outputs; two tests guard the outputs.

Out of scope: theme replacement, layout templating beyond the sidebar
footer, any page prose.

## Decisions

### 1. Palette: Okabe–Ito blue (option A)

The cards colour cause `#D55C00` and effect `#009E73` (Okabe–Ito). The site's
primary is the same palette's blue, so the three read as one set; cause and
effect stay reserved for their meaning and are never used for chrome.

| Token | Light | Dark |
| --- | --- | --- |
| ground | `#fbfaf7` | `#15171a` |
| ground, raised (cards, table stripes, code) | `#f2f0ea` | `#1d2024` |
| text | `#1d1f22` | `#e8e6e1` |
| muted text | `#5b6068` | `#a2a7ae` |
| rules and borders | `#d9d6ce` | `#33373d` |
| primary (`--pst-color-primary`) | `#0072B2` | `#5db4dd` |
| links (`--pst-color-link`) | `#0b6e9e` | `#5db4dd` |
| primary tint (admonition note, badges) | `#e4f0f7` | `#14323f` |
| cause (unchanged) | `#D55C00` | `#D55C00` |
| effect (unchanged) | `#009E73` | `#009E73` |

Every token is defined on `html[data-theme="light"]` and
`html[data-theme="dark"]` (the theme stamps one of the two); components use
tokens only.

### 2. Typography: IBM Plex (option A)

- Body: IBM Plex Sans, 16 px, line height 1.6.
- Headings: IBM Plex Serif 600; h1 2.0 rem, h2 1.5 rem, h3 1.2 rem, h4 1.0 rem;
  `text-wrap: balance`.
- Code and every number in a card: IBM Plex Mono, 0.9 em of the surrounding
  text.
- Uppercase labels (card section labels, table headers): 0.04 em letter
  spacing.
- Fonts load from Google Fonts with `display=swap` and system fallbacks
  (`system-ui`, `Georgia`, `ui-monospace`).

### 3. Code blocks: Selenized light and Selenized dark

`docs/_ext/selenized.py` defines two Pygments `Style` classes from the
published Selenized palettes (light ground `#fbf3db`, dark ground `#103c48`)
and sets an explicit colour for plain `Name` tokens so a block never depends
on the page's text colour. `conf.py` adds `docs/_ext` to `sys.path`,
registers both under `pygments.styles._STYLE_NAME_TO_MODULE_MAP` as
`selenized-light` and `selenized-dark`, and sets the theme's
`pygments_light_style` / `pygments_dark_style` to them. Blocks get 6 px
corners and a 1 px border in the rules colour; the copy button stays.

### 4. Result cards

The cards style themselves through `--pc-*` variables. The stylesheet maps
them to the site tokens (ground, raised ground for header and section rules,
muted text, primary tint and primary for the badge), sets the body face and
the mono face, and matches the 6 px corner radius. Cause and effect tones are
untouched.

Label alignment: labels stay left-aligned, and a card's key/value sections
share one label-column width, set by the widest label in the card, so values
line up down the whole card. This is a change in `pyphi/display`: the ASCII
backend pads every key/value section to the card-wide label width, and the
HTML backend emits `--pc-kcol` on the card with that width in `ch` and the
key/value grid uses it. Card goldens are regenerated for the new padding.
Table headers in the HTML backend use the sans face; cells use mono.

### 5. Landing page

Order: announcement bar (theme), logo, one-line tagline ("Integrated
information, computed." plus one sentence), the six section cards with an
Octicon glyph each (`rocket`, `book`, `tools`, `beaker`, `list-unordered`,
`arrow-switch`), then a compact "Cite" strip with the three references on
one line each, then the hidden toctree. The seventh card (what's new) moves
into the announcement bar and stays in the hidden toctree.

### 6. Tables

Small uppercase headers in the sans, a 2 px rule under the header, hairline
row rules, alternate-row tint in the raised ground, `tabular-nums`,
`max-width: 100%` with horizontal scroll in a wrapper, and no stretching of
narrow tables (`width: auto`).

### 7. Admonitions

Flat tint with a 4 px left rule and no icon box. Note, seealso, and
important take the primary tint; warning and danger take a 12% cause tint
with a vermilion rule; tip and hint take a 12% effect tint with a green rule.

### 8. Chrome

- Announcement: "PyPhi 2.0 is released: what's new, and the migration guide
  for 1.x users." with both links; removed after the release settles.
- Version switcher: `docs/_static/switcher.json` listing `stable`
  (preferred), `latest`, and `2.0.0`; `version_match` derived from the
  installed package version at build time; `navbar_end` = version switcher,
  theme switcher, icon links.
- Sidebar footer: a `_templates/sidebar-cite.html` template appended to the
  primary sidebar with the software citation and the GitHub link.
- Dark mode: every token above has a dark value; nothing is defined only
  under one theme.

### 9. Demo notebook page

`docs/examples/IIT_4.0_demo.ipynb` is executed once and committed with its
outputs; the tutorial page becomes the notebook (moved to
`docs/tutorials/iit-4.0-demo.ipynb`, listed in the tutorials toctree, and in
`nb_execution_excludepatterns` so the build renders stored outputs). The
summary text becomes the notebook's first Markdown cell; the download link
reads "Download the notebook". `just notebook-outputs` re-executes and
rewrites it.

Drift guards:

- The execution recipe stores a SHA-256 of the code cells' sources in
  `metadata.pyphi.code_hash`. A fast test recomputes the hash from the file
  and fails when the code changed without regeneration.
- A slow-lane test re-executes the notebook with nbclient and compares each
  code cell's text outputs (`stream` and `execute_result` text, with
  warnings and timing lines stripped) to the stored ones, failing on any
  difference. This replaces the release gate's manual re-execution.

## Verification

- `just docs` ends in "build succeeded" (strict). Both themes are viewed on:
  index, getting-started, read-result (cards), estimate-cost (tables and a
  code block), theory/intrinsic-information (math and admonitions), the
  configuration reference (a wide table), the demo notebook.
- `uv run pytest` green, with the card goldens regenerated for the label
  width and the two notebook tests added; `uv run pytest -m slow --slow`
  green for the notebook re-execution.
- No horizontal page scroll at 400 px width; the table wrapper scrolls
  instead.
