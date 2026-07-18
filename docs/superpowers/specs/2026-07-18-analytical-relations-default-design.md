# ANALYTICAL Relation-Computation Default — Design

**Date:** 2026-07-18
**Status:** Approved

## Goal

Make `ANALYTICAL` the default `relation_computation` (currently `CONCRETE`).
The analytical backend summarizes the relation set in closed form (Σφ_r,
degree spectrum, moments, top-k, sampling) without enumerating it, and is
value-identical to the concrete backend everywhere both run — a forced-analytical
full-suite run (2026-07-11) found no numerical mismatch, only
iteration/subscripting `TypeError`s (~102 test sites: visualize ~55,
relation/CES ~28, matching ~17). Flipping the default removes the
exponential relation enumeration from the default `phi_structure` path.

## Decisions

1. **Default flip:** `relation_computation: str = "ANALYTICAL"` in
   `pyphi/conf/formalism.py` (currently line 108).
2. **Visualization default:** `max_relations=None` keeps meaning "all
   relations" on an enumerable backend and comes to mean "the strongest 1000
   by φ_r" on the analytical backend. Tunable if plotly struggles at 1000.
3. **Matching:** `Perception` materializes analytical relations internally on
   first use (cached), so matching works under any ambient default with no
   user-facing change.
4. **Test policy:** concrete-needing tests get explicit `CONCRETE` pins;
   new targeted tests cover the shipping default path.

## Design

### 1. The flip

Change the `IITConfig.relation_computation` field default to `"ANALYTICAL"`.

The presets (`pyphi/conf/presets.py`) continue to **not** set this field.
`relation_computation` is a computation-strategy knob, not a
formalism-defining one (the backends agree numerically), so it is outside the
"presets pin measure fields explicitly" doctrine. Consequences:

- `iit4_2023` / `iit4_2026` presets — and therefore `pyphi.analyze` /
  `pyphi.sweep` and the test-suite `IIT_4_CONFIG` pins — inherit the new
  backend. Tests that need enumeration pin it back individually (§4).
- The IIT 3.0 path never reads the option (guarded out at the call
  boundary), so the `iit3` preset is unaffected.

### 2. Visualization default

`project_ces` (`pyphi/visualize/projection/__init__.py`) currently raises
`ValueError` when `max_relations is None` and `ces.relations` is not
enumerable. Replace that branch with a default:

- A module constant `DEFAULT_MAX_ANALYTICAL_RELATIONS = 1000` in
  `pyphi/visualize/projection/__init__.py`.
- `max_relations=None` → all relations when `ces.relations` is enumerable
  (unchanged); `strongest(k=DEFAULT_MAX_ANALYTICAL_RELATIONS)` when it is
  not.
- An explicit `max_relations=N` behaves as today on both backends.

The entry points that forward `max_relations` (`plot_ces`,
`highlight_phi_fold` in `pyphi/visualize/__init__.py`, and `project_ces`
itself) get docstring updates: `None` means every relation on the concrete
backend and the strongest 1000 on the analytical backend.

### 3. Matching

Matching's per-relation quantities require the enumerated relation set;
the analytical summary cannot supply them (the perception weight carries a
mean-of-triggering factor with no closed form). Two sites enumerate
relations:

- `Perception.richness` (`pyphi/matching/perception.py`, Eq. 13) iterates
  `self.ces.relations`.
- `_component_perceptions` (`pyphi/matching/differentiation.py`) iterates
  `perception.ces.relations` and currently raises `TypeError` on
  `AnalyticalRelations`, directing users to `analytical_differentiation`.

`Perception` gains a cached private property:

```python
@cached_property
def _relations(self) -> ConcreteRelations:
    relations = self.ces.relations
    if isinstance(relations, AnalyticalRelations):
        return relations.materialize()
    return relations
```

Both sites iterate `perception._relations` instead of `ces.relations`; the
`TypeError` branch in `_component_perceptions` is removed. Cost is identical
to having computed the structure with the concrete backend, paid once per
`Perception` and only when a per-relation quantity is actually used. The
`Perception` class docstring documents that analytical relation summaries
are materialized on first use. `Differentiation.analytical_differentiation`
is untouched — it remains the closed-form path for D, reading only
distinctions.

### 4. Tests

**Pins.** Test sites that iterate/index relations get an explicit
`CONCRETE` pin: a module-scoped autouse override fixture where a whole file
tests concrete-backend behavior, an inline `config.override` otherwise.
Expected triage of the known ~102 breakages: visualize (~55) pass via §2,
matching (~17) via §3, the remainder (~30, relation/CES/serialization) get
pins — or a one-line adaptation onto the analytical query surface where the
assertion is trivially re-expressible.

**New default-path tests:**

- `test_default_relation_computation_is_analytical` — asserts the shipping
  default, deliberately unpinned, alongside
  `test_default_formalism_is_iit4_2026` in
  `test/conf/test_config_layers.py`.
- Under the default config, `phi_structure(...)` yields a CES whose
  `.relations` is `AnalyticalRelations`.
- The viz default-k path renders: `project_ces` on an analytical-backed CES
  with `max_relations=None` succeeds and carries at most 1000 relation
  edges.
- Analytical/concrete `sum_phi` parity on a small structure (guards the
  value-identity the flip rests on).
- `Perception.richness` and `Differentiation.perceptual_differentiation`
  agree between a concrete-backed and an analytical-backed structure for
  the same system (exercises the materialize-on-first-use path).

**Forced-CONCRETE spot check** (verification, not a committed test): run the
pinned modules once with the flip in place to confirm they still exercise
the concrete backend.

### 5. Docs and changelog

- Changelog fragment `changelog.d/analytical-relations-default.change.md`:
  the new default, what changes for users (`ces.relations` is a closed-form
  summary by default; iteration/indexing raise with guidance), and the
  migration paths — `.strongest(k)`, `.materialize()`, or
  `relation_computation: CONCRETE` under the `formalism` key in
  `pyphi_config.yml`.
- Docs sweep for pages that iterate `ces.relations` or state the `CONCRETE`
  default: tutorials, `docs/howto/visualize.md`, configuration docs, and the
  MCP content surfaces (`pyphi/mcp/server.py` docstrings,
  `pyphi/mcp/content/`).
- `docs/whats-new-in-2.0.md` is excluded: the file is held by a concurrent
  session. Any needed edit there joins the existing deferred site.

### 6. Verification

Worktree flow. Completion gate: pathless `uv run pytest` green in the
worktree and on main after merge; slow lane (`uv run pytest -m slow
--slow`) green; the forced-CONCRETE spot check above.

## Out of scope

- Tuning the viz default below/above 1000 (adjust later if plotly
  struggles).
- An analytical form for perception richness or D_p (open research; the
  mean-of-triggering factor breaks the min-algebra).
- Edits to `docs/whats-new-in-2.0.md` (deferred until the concurrent
  session settles).
- Degree caps / parallel controls for the analytical backend (it rejects
  kwargs by design).
