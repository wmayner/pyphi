# Dotted Config Access — Design Spec

**Status:** Approved (design); pending implementation plan.
**Date:** 2026-05-28.

## Motivation

PyPhi's config has three top-level layers; `formalism` contains two
sub-namespaces:

```
config
├── numerics             → config.numerics.precision
├── infrastructure       → config.infrastructure.parallel
└── formalism
    ├── iit               → config.formalism.iit.version
    └── actual_causation  → config.formalism.actual_causation.version
```

A field name that exists in **both** `iit` and `actual_causation` (currently
`mechanism_partition_scheme`) is a *colliding field*: flat access by bare name
is ambiguous, so `config.mechanism_partition_scheme` and
`config.override(mechanism_partition_scheme=...)` raise `ConfigurationError`,
directing the user to a qualified form.

This becomes pressing for the **AC formalism object** work, which wants to add
`version` to `ActualCausationConfig` to mirror `IITConfig.version`. That makes
`version` a colliding field — and `version` is set by flat
`config.override(version="IIT_4_0_2023", ...)` in `test/conftest.py` (module
load) and `test/test_system_cause_effect_info.py`. With `version` colliding,
those flat overrides would raise.

The qualified escape hatches for colliding fields are uneven today:

- **Read:** `config.formalism.iit.version` (attribute chain) and
  `config["formalism.iit.version"]` (subscript) both work.
- **Write (single):** `config["formalism.iit.version"] = x` (subscript) and
  `config.iit = replace(config.formalism.iit, version=x)` (wholesale) both work.
- **Scoped override (`with` / `@`):** `config.override(...)` routes every kwarg
  through `setattr`, so a colliding name has **no** qualified form — you must
  pass a whole sub-config object (`config.override(iit=IITConfig(...))`), which
  is verbose for changing one field in a test scope.

This spec closes that gap with ergonomic **dotted access**: `override` accepts
dotted-path keys, and the dotted grammar gains a sub-namespace shorthand
(`iit.version` ≡ `formalism.iit.version`). It is a precursor to the AC
formalism work: once it lands, `version` can be a clean colliding field with no
rename and no routing-precedence hack.

## Goals

- `config.override(...)` accepts dotted-path keys (full path and sub-namespace
  shorthand), routed to the owning sub-namespace, with correct scoped restore.
- Dotted subscript gains the sub-namespace shorthand (`config["iit.version"]`)
  in addition to the existing full path (`config["formalism.iit.version"]`).
- Colliding-field error messages point at the dotted `override` / subscript
  forms.
- Migrate the flat-`version` override sites to dotted form so the subsequent AC
  `version` field collides cleanly with nothing to fix.

## Non-goals

- No change to the flat-colliding-write behavior: bare colliding names still
  **raise** (dotted access is the explicit disambiguator, not a silent
  reroute).
- No change to `FIELD_TO_LAYER` or `colliding_formalism_fields` semantics, and
  no routing-precedence rule between `iit` and `actual_causation`.
- No new attribute-style dotted access — `config.formalism.iit.version`
  (attribute chain) and `config.iit = ...` (wholesale) already work and are
  untouched.
- Does not add the AC `version` field — that lands in the AC formalism work.

## Current architecture (reference)

`pyphi/conf/_global.py`:

- `__getitem__(path)` (≈ line 220) splits on `.`; a single part routes via
  `FIELD_TO_LAYER`; multiple parts walk attributes from `self`.
- `__setitem__(path, value)` (≈ line 249) requires ≥2 parts, the first being a
  top-level layer name (`_LAYER_NAMES`), then rebuilds the nested layer via
  `_rebuild_nested`.
- `__contains__` delegates to `__getitem__`.
- `override(**kwargs)` (≈ line 176) returns `_OverrideContext`.
- `_OverrideContext.__enter__` (≈ line 451) snapshots, then `setattr(config,
  name, value)` per kwarg; `__exit__` restores the wholesale snapshot.
- `__setattr__` (≈ line 354) and `__getattr__` (≈ line 340) raise on bare
  colliding names with a hint message.

`pyphi/conf/_field_routing.py` builds `FIELD_TO_LAYER` and exposes
`colliding_formalism_fields()` (the IIT ∩ AC field-name set), which are
**unchanged** by this work.

## Design

### 1. Sub-namespace roots in dotted-path resolution

Add a single normalization step in the path parsing shared by `__getitem__`,
`__setitem__`, and `__contains__`: if the first path segment is a formalism
sub-namespace name (`"iit"` or `"actual_causation"`), prepend `"formalism"`.
So `iit.version` → `formalism.iit.version` before the existing resolution
runs. Full paths are unaffected. The set of sub-namespace names derives from
`fields(FormalismConfig)` (the names of the `iit` / `actual_causation` fields),
not a hardcoded literal, so it stays correct if the formalism layer changes.

`__setitem__` currently rejects a first segment that isn't in `_LAYER_NAMES`;
after normalization, `iit.version` becomes `formalism.iit.version`, whose first
segment **is** a layer name, so the existing `_rebuild_nested` path applies
without further change.

### 2. `override` accepts dotted keys

`override(self, _paths: Mapping[str, Any] | None = None, /, **kwargs) ->
_OverrideContext`. It merges `_paths` (if given) with `kwargs` into one dict
(a positional-only first parameter avoids clobbering a flat field literally
named `_paths`, which cannot exist). `_OverrideContext` routes each entry on
apply:

```python
for name, value in self._new_values.items():
    if "." in name:
        self._config[name] = value   # __setitem__ (dotted, shorthand-aware)
    else:
        setattr(self._config, name, value)   # flat (today's path)
```

`__exit__` restore is unchanged (wholesale snapshot; colliding fields already
round-trip). Supported call forms:

- `config.override(precision=6)` — flat kwargs (unchanged).
- `config.override({"iit.version": "IIT_3_0"})` — positional dotted dict.
- `config.override(**{"iit.version": "IIT_3_0"})` — dotted via kwargs (Python
  permits dotted string keys through `**kwargs`).
- `config.override({"iit.version": "IIT_3_0"}, mechanism_phi_measure="EMD")` —
  dotted + flat mixed.

### 3. Error messages mention the dotted forms

Extend the colliding-field messages so they advertise the dotted forms in
addition to the existing wholesale hint:

- `__setattr__` `ConfigurationError`: add `config["formalism.iit.<name>"] =
  ...` and `config.override({"iit.<name>": ...})`.
- `__getattr__` `AttributeError`: already suggests
  `config.formalism.iit.<name>`; add the subscript form `config["iit.<name>"]`.

### 4. Migration of flat-`version` override sites

These currently pass `version` as a flat kwarg (works today because `version`
is IIT-unique). Migrate to dotted so they remain correct once AC adds a
colliding `version`:

- `test/conftest.py` `IIT_4_CONFIG`: `config.override(version="IIT_4_0_2023",
  mechanism_phi_measure=..., system_phi_measure=..., system_partition_scheme=...)`
  → `config.override({"iit.version": "IIT_4_0_2023"}, mechanism_phi_measure=...,
  system_phi_measure=..., system_partition_scheme=...)` (only `version` moves to
  dotted; the IIT-unique fields stay flat).
- `test/test_system_cause_effect_info.py` (3 decorators):
  `@config.override(version="IIT_3_0", mechanism_phi_measure="EMD")` →
  `@config.override({"iit.version": "IIT_3_0"}, mechanism_phi_measure="EMD")`.

## Testing strategy

New tests in `test/test_config_layers.py` (or a focused new module):

- `config.override({"iit.version": "IIT_3_0"})` inside a `with` sets
  `config.formalism.iit.version` and restores the prior value on exit.
- Sub-namespace shorthand and full path are equivalent for override and
  subscript (`config["iit.version"]` == `config["formalism.iit.version"]`,
  read and write).
- Mixed dotted + flat override applies both and restores both.
- `config.override(**{"iit.version": ...})` (kwargs form) works.
- Colliding-field flat write still raises, and the message names the dotted
  forms.
- `actual_causation` shorthand resolves (`config["actual_causation.alpha_measure"]`
  and `config["actual_causation"...]` shorthand).

Existing `test/test_config_layers.py` invariants (colliding fields excluded
from `FIELD_TO_LAYER`, `as_kwargs` excludes colliders) stay green — this work
does not touch routing semantics.

## Correctness contract

Additive ergonomics: existing flat/subscript/attribute access is unchanged; the
only behavioral additions are (a) override routing dotted keys and (b) the
sub-namespace shorthand in dotted paths. The flat-colliding-raise behavior is
preserved. Full `uv run pytest` (no path) stays green.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Sub-namespace shorthand collides with a future top-level layer name. | Normalization only fires when the first segment is a known formalism sub-namespace name (derived from `fields(FormalismConfig)`); top-level layer names take precedence and are disjoint from sub-namespace names. |
| `override` positional param shadows a real flat field. | The first parameter is positional-only (`/`) and named `_paths`; flat fields are passed as keywords, so there is no collision. |
| Migrating conftest's module-load override breaks collection. | The dotted form is exercised by the new tests first; the migration is a behavior-preserving rewrite of an already-working override (verified by the full suite). |

## Open questions

None blocking.
