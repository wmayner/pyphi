# B21 — Unified Object Display / Rendering Model

**Status:** Draft for review
**Date:** 2026-06-15
**Roadmap item:** B21 (Wave 2, pre-freeze, surface-affecting)

---

## 1. Motivation

PyPhi's object formatting has accumulated into spaghetti. `pyphi/models/fmt.py`
is 1169 lines of ~50 flat `fmt_*` helpers plus Unicode box-drawing and the
`align_columns` workhorse. The consequences, verified against the code:

- **Inconsistent reprs.** Some types route through `fmt.py`; `Relation` /
  `RelationFace` / `AcSIA` hand-roll their own boxed reprs bypassing it; some
  `__str__` return `repr(self)` (RIA, MICE, Complex, Substrate — a circular
  delegation); some types define no `__str__` (`CauseEffectStructure`,
  `RelationFace`, `Relations`, state specs) and `System` defines no `__repr__`.
- **Thin, partial HTML.** Only 4 classes implement `_repr_html_`
  (`IIT3SystemIrreducibilityAnalysis`, `CauseEffectStructure`,
  `AcSystemIrreducibilityAnalysis`, IIT 4.0 `SystemIrreducibilityAnalysis`),
  all via a bare unstyled `html_columns` table. The other ~16 user-facing
  result types have no notebook rendering at all. Partitions/cuts have none.
- **`repr_verbosity` read in scattered places** (`fmt.make_repr`,
  `fmt_distinction`, `fmt_ria`).
- **Poor default output.** Captured live (IIT 4.0 SIA on `basic_system()`):
  - raw float precision — `φ_s: 0.41503749927884376`
  - raw numpy matrix dumped for the partition — `[[0 0 0] [1 0 0] [1 0 0]]`
  - inconsistent label/value alignment
  - `CauseEffectStructure` **embeds the entire SIA box inside its own box**
    (recursive heavy nesting).

This is a pre-freeze (Wave 2) item: it defines the public repr / HTML surface
that the P15 freeze locks. It is foundational for B8 (`result.explain()`) and
B15 (`result.diff()`), which should plug into this renderer rather than grow
their own. It shares one labeled-field extraction with the landed P14d
`to_pandas` work (P14d owns the pandas surface; this owns display).

## 2. Goals & non-goals

**Goals**

1. One declarative description per result type (`_describe()`), rendered by
   pluggable backends — the single source of truth for *what* to show.
2. Full display coverage: every user-facing result type gets consistent
   `__repr__` / `__str__` / `_repr_html_` (advances ship-criterion #1).
3. A redesigned, consistent **terminal** appearance (boxed-card visual
   language) and a genuinely **styled HTML** appearance.
4. `repr_verbosity` read in exactly one place.
5. Golden snapshot coverage of rendered output so future drift is caught
   (currently absent).

**Non-goals**

- `pyphi/visualize/` (matplotlib / plotly *figures*) — separate heavy-optional
  concern, untouched.
- The `rich` backend — the backend seam is designed for it, but it is **not
  built** in this work (YAGNI; a self-contained later drop-in).
- `to_pandas` surface — landed under P14d; this reuses its `NodeLabels`
  utilities but does not change it.
- Bayesian-network / CPD semantics — that is N11.

## 3. Architecture

New top-level package **`pyphi/display/`** (top-level, not under `models/`,
because `System` / `Substrate` / `Node` are not models):

```
pyphi/display/
├── __init__.py          # public: Displayable, describe vocabulary, render entry
├── description.py       # the declarative description vocabulary (frozen dataclasses)
├── mixin.py             # Displayable mixin (the only repr_verbosity read site)
├── numbers.py           # display number formatting (rounding)
└── render/
    ├── __init__.py      # backend selection
    ├── ascii.py         # ASCII/Unicode backend (absorbs fmt.py primitives)
    └── html.py          # styled HTML backend
    # render/rich.py     # DEFERRED — seam only
```

### 3.1 The description vocabulary (`description.py`)

A small set of frozen dataclasses describing *what* to show, independent of any
backend. This generalizes the existing `_repr_columns()` pattern (which already
returns `list[tuple[label, value]]` on 12 classes).

```python
@dataclass(frozen=True)
class Row:
    """One aligned key/value line, with optional trailing extra fields."""
    label: str
    value: Any
    extra: tuple[tuple[str, Any], ...] = ()   # e.g. ("II_c", 3.0), ("int.diff", 0.0)

@dataclass(frozen=True)
class Section:
    """A named group rendered with a rule divider."""
    label: str | None
    rows: tuple[Row, ...] = ()
    body: tuple["Component", ...] = ()        # nested components (Table, Inline, Nested)

@dataclass(frozen=True)
class Table:
    """Tabular list — distinctions, relations, account links."""
    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

@dataclass(frozen=True)
class Inline:
    """Pre-formatted fragment owned by the type (partition diagram, repertoire)."""
    text: str                                  # ASCII form
    html: str | None = None                    # optional HTML override

@dataclass(frozen=True)
class Nested:
    """A child result rendered COMPACTLY (one-line summary), never a recursive box."""
    description: "Description"

Component = Row | Section | Table | Inline | Nested

@dataclass(frozen=True)
class Description:
    title: str
    subtitle: str | None = None                # e.g. "φ_s 0.415037"
    sections: tuple[Section, ...] = ()
    compact: str | None = None                 # the repr_verbosity=LOW one-liner
```

Each result type implements one method:

```python
def _describe(self, verbosity: int) -> Description: ...
```

This **replaces** the per-class `__repr__` / `__str__` / `_repr_columns` and
the high-level `fmt.fmt_*(obj)` composers.

### 3.2 Backends (`render/`)

- **`ascii.py`** — `render(description, verbosity) -> str`. Absorbs the good
  `fmt.py` primitives as private internals: `box`, `header`, `align_columns`,
  `side_by_side`, `indent`, decimal alignment, Unicode constants. Implements the
  boxed-card layout (§4).
- **`html.py`** — `render(description, verbosity) -> str`. Emits a styled card
  with **scoped CSS classes** (a `pyphi-` prefix + a once-injected `<style>`
  block) so notebook themes don't clash. Card → panel; `Section` → labeled row
  group; `Row` → grid row; `Table` → `<table>`; `Nested` → compact inline.
- **`render/__init__.py`** — selects the backend (`ascii` for `__str__`/`__repr__`,
  `html` for `_repr_html_`). The `rich` seam: a backend registry keyed by name,
  so a future `render/rich.py` registers without touching call sites.

### 3.3 The `Displayable` mixin (`mixin.py`)

Supplies all dunders, each dispatching to `_describe()` + the active backend.
**The only place `repr_verbosity` is read.**

```python
class Displayable:
    def _describe(self, verbosity: int) -> Description: ...   # subclass hook

    def __repr__(self) -> str:
        v = _current_verbosity()
        d = self._describe(v)
        if v == LOW and d.compact is not None:
            return d.compact
        return ascii_render(d, v)

    __str__ = __repr__                          # one path; kills the str→repr circularity
    def _repr_html_(self) -> str: return html_render(self._describe(_current_verbosity()), ...)
    def _repr_mimebundle_(self, **_): return {"text/plain": str(self), "text/html": self._repr_html_()}
```

## 4. Terminal visual language (boxed card)

Rounded-corner card, grouped sections with rule dividers, rounded numbers,
aligned columns. **Nesting never recurses into full boxes** — a parent renders
children as compact one-line rows.

IIT 4.0 SIA (redesigned):

```
╭─ SystemIrreducibilityAnalysis ─────────╮
│ System   A,B,C   state (1,0,0)         │
│ φ_s      0.415037   norm 0.207519      │
├─ Cause ────────────────────────────────┤
│ (1,1,0)   II_c 3.0   int.diff 0.0      │
├─ Effect ───────────────────────────────┤
│ (0,0,1)   II_e 3.0   int.diff 0.0      │
├─ MIP ──────────────────────────────────┤
│ {A,BC}    tied 0                        │
╰────────────────────────────────────────╯
```

CauseEffectStructure (children compact — the key fix vs. today's embedded box):

```
╭─ CauseEffectStructure ─────────────────╮
│ Φ 0.415037   distinctions 2            │
│ Σφ_d 1.0     relations 0   Σφ_r 0      │
├─ Distinctions ─────────────────────────┤
│ A    φ_d 0.5   cause {A,B} effect {C}  │
│ BC   φ_d 0.5   cause {B}   effect {A}  │
╰────────────────────────────────────────╯
```

Account (richest type — validates the `Table` vocabulary):

```
╭─ Account ──────────────────────────────╮
│ links 5   Σα 1.245                     │
├─ Causal links ─────────────────────────┤
│ dir    purview   mechanism   α         │
│ CAUSE  OR        {AND}       0.415      │
│ EFFECT AND        {OR}       0.415      │
│ …                                       │
╰────────────────────────────────────────╯
```

## 5. Cross-cutting output rules

- **Rounded numbers** by default to **6 significant figures** (`display/numbers.py`),
  exact value always reachable via the attribute. `0.41503749927884376` →
  `0.415037`.
- **`repr_verbosity`**: `LOW` → the compact `ClassName(attr=…)` one-liner
  (`Description.compact`); `MEDIUM` / `HIGH` → the card. (No new verbosity level;
  full float precision is not a repr concern — the attribute holds it.)
- **Partitions** render in readable cut notation (`{A,BC}` with the existing
  `fmt_partition` arrow form); the raw numpy matrix is dropped from the default
  view.
- **Consistent alignment** via the single `align_columns` path.

## 6. Coverage

Every user-facing result type implements `_describe` + the `Displayable` mixin:

- 3.0 SIA, 4.0 SIA (2023 / 2026)
- `CauseEffectStructure`, `PhiFold`
- `Distinction`, `Distinctions`, `ResolvedDistinctions`
- `RepertoireIrreducibilityAnalysis`, `MICE` variants
- `Relation`, `RelationFace`, `Relations`
- partitions & cuts (all `_PartitionBase` subclasses)
- `Complex`, `ExcludedCandidate`
- `System`, `Substrate`, `Node`
- `AcRIA`, `CausalLink`, `Account`, `AcSIA`
- repertoires, `StateSpecification` / `SystemStateSpecification` / `UnitState`

This fixes every gap from the audit (`CES.__str__`, `System.__repr__`,
undisplayed `PhiFold` / `Relation`, partitions' missing `_repr_html_`). The
ad-hoc reprs and the `DirectedBipartition` special-casing all route through the
model.

`fmt.py` is removed: its low-level primitives move into `render/ascii.py`; its
high-level `fmt_*(obj)` composers are deleted (no back-compat shim — 2.0 is
breaking). Callers update to the mixin.

## 7. Testing

- **Golden snapshot tests** (`test/test_display.py`, NEW) — pinned ASCII output
  per result type for a small example network, captured *once* after the look is
  settled. The regression net that currently does not exist.
- **HTML structural tests** — extend `test/test_result_protocols.py` from 4
  types to all: assert the panel container, section labels, and key field labels
  are present.
- **`Displayable` invariant** — a parametrized test over every registered result
  type asserting it has `_describe` returning a valid `Description` and all three
  dunders resolve without error.
- **Deliberate exact-string updates** — the small set of layout-sensitive
  assertions: `test_models.py` partition/tripartition strings (lines 727, 746),
  `test_system.py:163` `System(B, C)`, `test_models.py:367` `NullCut((2, 3))`.
  These change with the redesign and are re-reviewed as intended surface.
- **Doctest sweep** — verify with `uv run pytest` (no path argument) so the
  `pyphi/` source-module doctests run. (`docs/*.rst` use reST substitutions, not
  rendered repr blocks, so they are unaffected; verify by reading.)

## 8. Delivery staging (for the implementation plan)

1. Build `description.py` + `ascii.py` + `html.py` + `mixin.py` + `numbers.py`
   and the golden harness; migrate **one type end-to-end** (IIT 4.0 SIA) to
   prove the vocabulary and lock the look.
2. Migrate remaining types in batches (SIAs/CES → distinctions/relations → AC →
   substrate/system/node → partitions/repertoires/state specs).
3. Delete `fmt.py` high-level composers; update all callers.
4. Extend tests; capture goldens; update the exact-string assertions.
5. Full `uv run pytest` (no path) green; changelog fragment.

## 9. Risks & mitigations

- **Load-bearing strings** — quantified and small (§1); enumerated in §7.
- **Vocabulary expressiveness** — validated up front against the richest types
  (Account, Relations) via the §4 mockups before the vocab is committed.
- **HTML theme clash** — scoped `pyphi-` CSS classes, style injected once.
- **Architectural-refactor breakage** — failing tests under the unified contract
  may reveal latent repr inconsistencies; diagnose before reverting (do not
  abandon at first breakage).

## 10. Open questions / contingencies

None blocking. The `rich` backend is explicitly deferred. B8 / B15 will consume
this model when they land.
