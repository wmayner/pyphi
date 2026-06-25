# Import-surface cleanup + dead-dependency drops — design

Part of the surface-freeze bundle: the last surface-affecting changes before
the public `pyphi.X` namespace stops changing. This sub-project settles the
shape of the top-level namespace and removes three unused or replaceable
dependencies.

## Goal

Replace the eager recursive submodule walk in `pyphi/__init__.py` with explicit
registry population plus lazy submodule access, guarded by a registry-contents
test; and drop the `tblib`, `ordered-set`, and `toolz` dependencies.

## Background: the current import machinery

`pyphi/__init__.py` lifts a handful of core names (`Substrate`, `System`,
`Direction`, the TPM types, `config`, the three IIT config namespaces,
`Transition`/`TransitionSystem`) and then calls `_import_submodules(__name__)`,
which uses `pkgutil.walk_packages` to recursively import **every** submodule of
the package. A hand-maintained `_skip_import = ["visualize",
"_factored_backends_xarray"]` list excludes modules that need optional
dependencies. `__all__` is then built dynamically from the names the walk found.

The walk exists for two reasons:

1. **Registry population.** Several registries are populated by decorator side
   effects (`@partition_types.register("BIPARTITION")`, etc.). A registry stays
   empty until the module holding its registrants is imported, so the walk
   imports everything to guarantee they all run.
2. **Submodule attribute convenience.** After a bare `import pyphi`, code can
   reach `pyphi.examples`, `pyphi.compute`, and so on without a separate import,
   because the walk already imported them.

### Costs of the walk

- `import pyphi` eagerly imports the entire package, so import is slow and a
  syntax or import error in any one peripheral module breaks `import pyphi`
  wholesale.
- The `_skip_import` list must be maintained by hand as optional-dependency
  modules are added.
- The eager walk undermines the lazy-import discipline used elsewhere for
  optional dependencies.

## The registry audit (the load-bearing risk, resolved)

Removing the walk is only safe if every registrant is still reached by an
explicit import; otherwise a measure, partition scheme, or strategy silently
disappears from its registry. The audit found a reassuring invariant:

**Every decorator registration lives in the single module that defines its
registry.** All 53 `@<registry>.register(...)` sites map one-to-one to the file
that creates the registry instance:

| Registry instance(s) | Defining module |
| --- | --- |
| `distribution_measures`, `stateful_distribution_measures`, `state_aware_measures`, `composite_measures`, `actual_causation_measures` | `pyphi/measures/distribution.py` |
| `measures` (CES) | `pyphi/measures/ces.py` |
| `partition_types`, `system_partition_types` | `pyphi/partition.py` |
| `phi_object_tie_resolution_strategies` | `pyphi/resolve_ties.py` |
| `relation_computations` | `pyphi/relations.py` |
| `distinction_phi_normalizations` | `pyphi/models/state_specification.py` |
| `partitioned_repertoire_schemes`, `background_strategies`, `alpha_aggregations` | `pyphi/formalism/actual_causation/compute.py` |

No registrant is scattered into a module reachable only by the filesystem walk.
So "make every registrant reachable from an explicit import" reduces to importing
those registrant modules.

The formalism registries (`FORMALISM_REGISTRY`,
`ACTUAL_CAUSATION_FORMALISM_REGISTRY`) are populated by explicit
`.register(...)` calls in `pyphi/formalism/__init__.py` and
`pyphi/formalism/actual_causation/__init__.py`, not decorators, so importing
`pyphi.formalism` populates them.

## The submodule-access surface (the freeze decision, settled by usage)

Whether to preserve `pyphi.<submodule>` attribute access is the one genuine
public-surface decision. The usage data settles it: across `pyphi/`, `test/`,
and `docs/`, attribute access is heavy — `pyphi.visualize` (155 references),
`pyphi.examples` (93), `pyphi.relations` (34), `pyphi.compute` (30),
`pyphi.partition` (27), `pyphi.utils` (26), `pyphi.actual` (21),
`pyphi.jsonify` (13), `pyphi.convert` (12), and more. Narrowing the surface
would break several hundred call sites for no benefit the freeze needs.
Decision: **preserve the surface, but make it lazy.**

## Design

### Part A — retire the eager walk

In `pyphi/__init__.py`:

1. **Explicit registry population.** Replace the `_import_submodules` call with a
   small, commented block that imports the registrant modules for their
   registration side effects:
   - `pyphi.measures.distribution`, `pyphi.measures.ces`
   - `pyphi.partition`
   - `pyphi.resolve_ties`
   - `pyphi.relations`
   - `pyphi.models.state_specification`
   - `pyphi.formalism` (transitively registers the three IIT formalisms and,
     via `pyphi.formalism.actual_causation`, the AC formalism and the AC
     scheme/strategy/aggregation registries)

   The block is auditable — a reader sees exactly which modules are imported and
   why. Third-party plugins are unaffected: they register when the user imports
   them.

   *Planning-phase check:* confirm that importing `pyphi.formalism` actually
   executes `pyphi/formalism/actual_causation/compute.py` (the AC
   `partitioned_repertoire_schemes` / `background_strategies` /
   `alpha_aggregations` registrants). If it does not, add an explicit import of
   that module. The registry-contents test (Part B) catches this regardless.

2. **Lazy submodule access (PEP 562).** Add a module-level `__getattr__` to
   `pyphi/__init__.py` that imports a submodule on first attribute access and
   caches it on the module object. `pyphi.examples`, `pyphi.compute`,
   `pyphi.visualize`, etc. keep working, but `import pyphi` no longer imports
   the whole tree. Benefits: faster import; `import pyphi` no longer fails when
   a peripheral module is broken; the lazy-import discipline for optional
   dependencies is no longer undermined.

3. **Delete `_skip_import`.** With lazy access, an optional-dependency module
   such as `pyphi.visualize` is imported only when accessed and raises
   `ImportError` if its dependency is absent — the correct, informative behavior,
   and an improvement on today's `AttributeError`.

4. **Static `__all__`.** Replace the walk-derived `__all__` with an explicit
   curated list: the lifted top-level names plus the public submodules intended
   to be reachable as attributes.

### Part B — registry-contents test (the safety net)

Add a test asserting that each registry's set of keys equals a pinned expected
set. This is what makes removing the walk safe: a dropped registrant fails the
test loudly instead of vanishing silently, and it guards every future import
refactor. The expected key sets are written explicitly in the test (the
canonical list of what each registry must contain), derived from the current
registry contents at authoring time and reviewed as intended.

### Part C — drop dead/replaceable dependencies (B17)

Three independent tasks, each landable as its own commit:

1. **`tblib`** — declared in `pyproject.toml` (line 41) with zero references in
   `pyphi/`. Remove the declaration.

2. **`ordered-set` → `dict.fromkeys`-backed wrapper.** `pyphi/data_structures/`
   re-exports `OrderedSet` and defines `HashableOrderedSet(OrderedSet[Any])`
   (the subclass exists specifically to make instances hashable and
   pickle-compatible). Replace the dependency with an internal insertion-ordered
   set backed by a `dict`, preserving the surface actually used across the
   codebase: `add`, `append`, `index`, `intersection`, `union`, `update`, `pop`,
   iteration, length, membership, equality, hashing, and **pickle round-trip**.
   *Planning-phase task:* enumerate the exact `OrderedSet` method surface in use
   (the codebase-wide grep over-counts because `add`/`append`/`pop` collide with
   plain-list calls) before fixing the wrapper's API. Drop the `ordered-set`
   dependency.

3. **`toolz` → stdlib / `more_itertools`.** Nine imports across `partition.py`,
   `utils.py`, `models/fmt.py`, `models/distinctions.py`, `models/ria.py`,
   `models/mice.py`, `substrate_generator/unit_functions.py`,
   `substrate_generator/ising.py`:
   - `concat` → `itertools.chain.from_iterable`
   - `unique` → `more_itertools.unique_everseen` (`more_itertools` is already a
     dependency)
   - `curry` (6 decorator sites: `utils.py` ×2, `unit_functions.py` ×3,
     `ising.py` ×1) → decided **per-site in the planning phase**: a tiny local
     `curry` helper or a `functools.partial` refactor, whichever is the lightest
     correct replacement for each site, with the rationale recorded in the plan.

   Drop the `toolz` dependency.

## Out of scope

- `jsonify` → `msgspec` serialization (the next P15 sub-project).
- The docstring sweep, test reorg, `to_pandas` extension, PR triage, docs
  rebuild, and changelog condense (later P15 sub-projects).

## Verification

- The new registry-contents test passes (and would fail if a registrant were
  dropped).
- `uv run pytest` with **no path argument**, so the `pyphi/` doctest sweep runs
  and catches any broken `pyphi.X` doctest.
- An import-time before/after measurement confirms `import pyphi` no longer
  eagerly imports the whole tree (e.g. `pyphi.visualize` is absent from
  `sys.modules` until accessed).
- A test that `import pyphi` succeeds even when a peripheral submodule raises on
  import (the robustness win).
- `SKIP=pyright uv run pre-commit run --all-files` and `uv run pyright pyphi`
  clean.
- The install closure no longer lists `tblib`, `ordered-set`, or `toolz`.
- CI green on the PR into `main`.

## Risks

- **Missed registrant.** Mitigated by the registry-contents test, which is
  authored before the walk is removed so a regression is caught immediately.
- **`ordered-set` API gap.** Mitigated by enumerating the in-use method surface
  before reimplementing, and by the existing test suite exercising the data
  structures (including pickle paths).
- **`toolz.curry` semantics.** `toolz.curry` supports partial application across
  multiple calls; the per-site replacement must match each function's actual
  call pattern, verified by the existing tests that exercise those functions.
