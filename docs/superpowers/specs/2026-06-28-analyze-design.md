# N7 `pyphi.analyze()` — design

## Goal

One high-level, package-level entry point — `pyphi.analyze(substrate, state, …)` —
that runs the IIT analysis of a single system and returns a self-displaying
`Analysis` bundle, with inline formalism switching and a cheap scalar-Φ escape
hatch.

## Motivation

A user with a `Substrate` and a `state` already has `Substrate.sia(state, indices)`
and `Substrate.ces(state, indices)`. What is missing, and what N7 supplies:

1. **Inline formalism switching.** Today you switch formalism with
   `with config.override(**presets.iit4_2023): substrate.sia(state)`. Neither
   `Substrate.sia` nor `.ces` takes a formalism argument.
2. **Package-level discoverability.** `pyphi.analyze(...)` at the root, rather
   than requiring the caller to know the `Substrate` method exists.
3. **A uniform bundle across formalisms.** Under IIT 4.0 the `CauseEffectStructure`
   already embeds `.sia`; under IIT 3.0 `ces()` returns bare `Distinctions` with
   no SIA. `analyze` hands back one object that exposes `.sia` / `.ces` / `.phi`
   identically regardless of formalism, so the caller never has to know about
   that difference.

This is a thin ergonomic wrapper. It reuses the existing compute path
(`System.from_substrate(...).ces()` / `.sia()`), the formalism presets, and the
`Displayable` repr machinery. It introduces no new computation.

## Public API

New module `pyphi/analyze.py`, mirroring the structure of `pyphi/sweep.py`.

```python
def analyze(
    substrate,
    state,
    *,
    subset=None,        # node indices; None = the full substrate
    formalism=None,     # None = the active config formalism; else a version name:
                        #   "IIT_3_0" / "IIT_4_0_2023" / "IIT_4_0_2026"
    compute=None,       # None = Analysis bundle (default);
                        #   "sia" / "ces" = the raw result object;
                        #   callable = callable(system)
) -> Analysis | Any
```

- `compute=None` (default) returns an `Analysis` bundle.
- `compute="sia"` returns the raw `SystemIrreducibilityAnalysis` — the cheap
  path; it never builds the CES, so "just give me Φ" does not pay full-CES cost.
- `compute="ces"` returns the raw cause-effect structure
  (`CauseEffectStructure` under 4.0, `Distinctions` under 3.0).
- `compute=<callable>` returns `callable(system)`, where `system` is the
  constructed `System` — the same escape hatch `sweep` offers.

`formalism` accepts a single version-name string (analyze handles one system,
so it is singular, unlike `sweep`'s `formalisms` list). `None` leaves the
active config untouched; a name applies that preset for the duration of the
call only and restores the prior config afterward.

## The `Analysis` bundle

A frozen dataclass that subclasses `pyphi.display.Displayable` (so its repr,
str, and HTML come from the shared card machinery — not a hand-rolled repr):

```python
@dataclass(frozen=True)
class Analysis(Displayable):
    system: System   # the analyzed System
    sia: Any         # SystemIrreducibilityAnalysis
    ces: Any         # CauseEffectStructure (4.0) or Distinctions (3.0)

    @property
    def phi(self) -> float:
        return float(self.sia.phi)

    def to_pandas(self) -> pd.DataFrame: ...
    def _describe(self, verbosity: int) -> Description: ...
```

### Display (`_describe`)

`Analysis` renders the same flat rich card as the other result types. The
`CauseEffectStructure` card already produces exactly the target layout — a
summary section (Φ / Distinctions / Σφ_d / Relations / Σφ_r), the distinctions
table, the relations table, and (at `FULL` verbosity) the SIA's sections folded
in **flat** (no nested boxes). `Analysis._describe` reuses that card rather than
re-implementing it:

```python
def _describe(self, verbosity):
    desc = self.ces._describe(verbosity)        # full flat card for the CES
    sections = list(desc.sections)
    if getattr(self.ces, "sia", None) is None:  # IIT 3.0: bare Distinctions, no SIA
        sections.extend(self.sia._describe(min(verbosity, FULL)).sections)
    return Description(
        title="Analysis",
        sections=tuple(sections),
        compact=f"Analysis(Φ={format_value(self.phi)})",
    )
```

- Under **IIT 4.0**: the card is the CES's own rich flat card (which already
  embeds the SIA), titled `Analysis`.
- Under **IIT 3.0**: where `ces` is bare `Distinctions` with no embedded SIA,
  the separately-computed SIA's sections are appended flat so the card still
  leads with Φ — uniform across formalisms.

The mixin handles `repr_verbosity` (a compact one-liner at `LOW`, the full card
at `HIGH`, SIA grids at `FULL`) and gives `_repr_html_` / `_repr_mimebundle_`
for free.

### Tabular export (`to_pandas`)

A one-row `DataFrame` — the scalar/count summary, distinct from the card:

| column | value |
| --- | --- |
| `phi` | `float(self.sia.phi)` |
| `normalized_phi` | `float(getattr(self.sia, "normalized_phi", nan))` |
| `n_distinctions` | number of distinctions in the CES |
| `sum_phi_r` | `relations.sum_phi()` if the CES has relations, else `nan` |

## Data flow

The `System` is built once; the SIA is extracted formalism-agnostically so 4.0
and 3.0 look the same to the caller:

```python
result = None
with (config.override(**presets.by_name[formalism]) if formalism else nullcontext()):
    indices = substrate.node_indices if subset is None else subset
    system = System.from_substrate(substrate, state, indices)
    if callable(compute):
        result = compute(system)
    elif compute == "sia":
        result = system.sia()
    elif compute == "ces":
        result = system.ces()
    else:                                       # default: the bundle
        ces = system.ces()
        sia = getattr(ces, "sia", None)         # 4.0: embedded; 3.0: None
        if sia is None:
            sia = system.sia()                  # 3.0: compute the SIA separately
        result = Analysis(system, sia, ces)
return result
```

`result` is initialized before the `with` so the CLI pyright run does not flag
the `with config.override(...): result = …; return result` shape as "possibly
unbound" (the `__exit__ -> bool` analysis trap).

## Shared formalism-name lookup

Both `analyze` and `sweep` need to map a version-name string to its preset dict.
Today `sweep` carries a private `_PRESETS` for this. Add a public mapping to
`pyphi/conf/presets.py`:

```python
by_name: dict[str, dict[str, Any]] = {
    "IIT_3_0": iit3,
    "IIT_4_0_2023": iit4_2023,
    "IIT_4_0_2026": iit4_2026,
}
```

`analyze` uses `presets.by_name`; `sweep`'s `_PRESETS` is replaced by it (a
one-line swap) so the mapping lives in one place. An unknown formalism name
raises `KeyError` from this dict — `analyze` wraps it in a `ValueError` naming
the valid options.

## Deliberate omissions (kept out on purpose)

- **No `seed` / provenance stamping.** Provenance is N8's concern; a caller who
  wants it calls `.with_provenance()` on the returned object. `analyze` is about
  ergonomically getting the analysis object, not stamping it.
- **No `formalism=` kwarg on `Substrate.sia` / `.ces`.** The free function
  covers inline formalism switching; adding the kwarg to the methods as well
  would be a second, redundant surface.
- **No new computation, no bundle-only mode.** The default returns the bundle;
  the `compute="sia"` escape hatch exists precisely so the bundle never forces
  full-CES cost on a caller who only wants Φ.

## Wiring

- Export `analyze` and `Analysis` from `pyphi/__init__.py` and add both to
  `__all__`.
- Add `presets.by_name`; swap `sweep`'s `_PRESETS` to it.
- Changelog fragment (`changelog.d/analyze.feature.md`).
- Flip the N7 row in the ROADMAP Status Dashboard to ✅ landed.

## Testing (`test/test_analyze.py`)

1. **Parity.** `analyze(basic_substrate(), basic_state()).phi` equals
   `basic_substrate().sia(basic_state()).phi`.
2. **Bundle type.** Default return is an `Analysis`; `.sia` and `.ces` are the
   expected types; under 4.0 `analysis.sia` is the CES's embedded SIA (phi-equal).
3. **`compute` modes.** `compute="sia"` returns the raw SIA equal to
   `system.sia()` (not wrapped); `compute="ces"` returns the raw CES;
   `compute=<callable>` returns `callable(system)`.
4. **Inline formalism.** `analyze(..., formalism="IIT_3_0")` computes under 3.0
   and leaves the global config formalism unchanged afterward.
5. **3.0 bundle.** Under `formalism="IIT_3_0"`, `.ces` is `Distinctions`, `.sia`
   is a separately-computed SIA, and `.phi` works.
6. **Subset.** `analyze(..., subset=(0, 1))` analyzes the two-node subsystem.
7. **`to_pandas`.** Returns a one-row `DataFrame` with the expected columns.
8. **Repr.** The card renders without error and stays concise at `LOW`
   verbosity (the compact `Analysis(Φ=…)` form); the full card at `HIGH`
   contains the distinctions section.
