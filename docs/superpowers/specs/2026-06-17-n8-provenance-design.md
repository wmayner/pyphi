# N8 — Provenance stamp on results

**Date:** 2026-06-17
**Roadmap item:** N8 (Wave 2, pre-freeze surface-affecting)
**Status:** design, pending approval

## Goal

Every top-level result object should carry a self-contained record of *how, when, and
by what code* it was computed, so a saved result can be audited and reproduced without
external context. This extends the existing per-result `ConfigSnapshot` (which records
*what settings* produced a result) with a sibling record of runtime and environment
metadata. It supports reproducible re-runs and pairs with the planned disk-backed
result cache (N4).

It is a public-surface addition, so it must land before the P15 freeze.

## Scope

The provenance stamp attaches to exactly the four result types that already carry a
`ConfigSnapshot`:

- the IIT 4.0 `SystemIrreducibilityAnalysis` (`pyphi/formalism/iit4/__init__.py`)
- the IIT 3.0 `SystemIrreducibilityAnalysis` (`pyphi/models/sia.py`)
- the `CauseEffectStructure` (`pyphi/models/ces.py`)
- the `AcSIA` (`pyphi/models/actual_causation.py`)

Mechanism-level results (`RepertoireIrreducibilityAnalysis`, `MICE`, `Distinction`) and
`Account` carry neither a config snapshot nor a provenance stamp, consistent with the
existing design.

## The `Provenance` value type

A new frozen dataclass in a standalone module `pyphi/provenance.py` (runtime/environment
metadata is a distinct concern from config, so it sits beside `conf/` rather than inside
it):

```python
@dataclass(frozen=True)
class Provenance:
    pyphi_version: str          # importlib.metadata.version("pyphi")
    git_sha: str | None         # None for wheel / PyPI installs
    git_dirty: bool | None      # None when git_sha is None
    timestamp: str              # ISO-8601 UTC, captured at construction
    python_version: str         # platform.python_version()
    numpy_version: str          # numpy.__version__
    scipy_version: str          # scipy.__version__
    platform: str               # f"{system}/{machine}", e.g. "Darwin/arm64"
    wall_time: float | None = None   # seconds; filled by the entry-point timer
    seed: int | None = None          # only when an RNG was actually consumed
```

### Capture

A classmethod `Provenance.capture(*, wall_time=None, seed=None) -> Provenance` populates
all fields:

- **version** via `importlib.metadata.version("pyphi")`.
- **git** via a module-level `@functools.cache`'d helper that runs `git rev-parse HEAD`
  and `git status --porcelain` once per process, rooted at the package directory.
  Returns `(None, None)` when git is unavailable or the package is not in a working
  tree (e.g. a wheel install). Any non-empty `status --porcelain` output sets
  `git_dirty=True`.
- **timestamp** via `datetime.now(UTC).isoformat()`.
- **python_version / numpy_version / scipy_version / platform** via `platform` /
  `numpy.__version__` / `scipy.__version__`.

The git subprocess is the only non-trivial cost and is cached after the first call, so
auto-stamping every result is cheap.

### Seed semantics

Core SIA/CES/Φ computation is deterministic and never consumes an RNG, so those results
record `seed=None`. The `seed` field is populated only for results produced by a code
path that actually drew from an RNG (matching, sampling, simulation). Threading a seed
into those paths is out of scope for this item — it lands when a stochastic result type
needs it — but the field exists so it can be filled honestly when that happens.

## Attachment

Add `provenance: Provenance | None = None` as a **sibling field** to each of the four
result classes (alongside the existing `config` field). It is auto-stamped in
`__post_init__` when `None`, mirroring the existing config-snapshot pattern:

```python
def __post_init__(self):
    ...
    if self.provenance is None:
        self.provenance = Provenance.capture()
```

`result.config` (the `ConfigSnapshot`) is unchanged, so B8 (`.explain()`) and B15
(`.diff()`) keep reading `result.config` with zero churn.

## Wall-time instrumentation

The compute functions (`sia()`, `ces()`) have many internal `return` paths (null SIAs,
degenerate cases, the main result), so timing is done at the single public dispatch
chokepoint rather than at each return: the entry points in `pyphi/formalism/queries.py`
(and the IIT 4.0 / 3.0 `sia` / `ces` functions they dispatch to). The wrapper records
elapsed wall-clock seconds around the compute and stamps the returned result via
`dataclasses.replace(result.provenance, wall_time=elapsed)`.

A result constructed directly (not through an entry point) keeps `wall_time=None` — the
rest of its provenance is still auto-stamped.

## Display: a new verbosity level

Provenance is metadata about the computation, not part of the result's mathematical
content, so it gets its own verbosity tier above the current top level rather than being
folded into `FULL`. The display levels (`pyphi/display/mixin.py`) become:

| Level | Value | Shows |
|---|---|---|
| `LOW` | 0 | one-line compact form |
| `MEDIUM` | 1 | card minus expensive embedded grids |
| `HIGH` | 2 (default) | the full standard card |
| `FULL` | 3 | HIGH plus exhaustive mathematical content (cut grids, repertoires) |
| `PROVENANCE` | 4 | FULL plus the provenance section |

Changes required:
- add `PROVENANCE = 4` to `pyphi/display/mixin.py` and export it from
  `pyphi/display/__init__.py`;
- extend `_VALID_REPR_VERBOSITY` (`pyphi/conf/infrastructure.py:21`) from `{0,1,2,3}`
  to `{0,1,2,3,4}`;
- each of the four result types' `_describe()` appends a provenance `Section` only when
  `verbosity >= PROVENANCE`;
- `FULL`'s docstring is updated from "everything" to "all mathematical content" so the
  new top tier is accurate.

The default (`HIGH`) repr is unaffected; provenance appears only when explicitly
requested.

## Non-pollution (safety property)

Provenance contains fields that vary run-to-run (timestamp, wall_time) or machine-to-machine
(git_sha, git_dirty, platform, library versions), so it must not leak into value
comparisons or stored fixtures. This holds by construction:

- **Equality** — result equality is φ-based (`cmp.OrderableByPhi`), not field-wise, so
  two results of the same computation with different timestamps remain equal.
- **Goldens** — the golden harness (`test/golden/`) extracts named fields (`sia.phi`,
  etc.); it does not serialize the whole result, so provenance never enters a fixture.
- **B15 `diff()`** — `ResultDiff` compares φ / structure / `config` only; provenance is
  explicitly excluded.
- **jsonify** — `Provenance` is registered so results round-trip; a round-trip preserves
  provenance but, because equality is φ-based, round-trip equality still holds.

## Testing

- `Provenance.capture()`: all fields populated with the expected types; `pyphi_version`
  matches `importlib.metadata.version("pyphi")`.
- git helper fallback: when run outside a working tree (mocked subprocess failure),
  returns `(None, None)` and `git_dirty` is `None`.
- **Coverage invariant**: every result type that carries a `ConfigSnapshot` also carries
  a `Provenance` (parametrized over the result types, like the B8/B15/B21 coverage
  tests).
- jsonify round-trip on each of the four result types preserves provenance.
- wall-time: a result obtained through the public `sia()` / `ces()` entry points has
  `provenance.wall_time is not None` and `>= 0`; a directly-constructed result has
  `wall_time is None`.
- **Non-pollution**: two runs of the same computation are `==` despite different
  timestamps, and `a.diff(b)` reports no φ / config change.
- display: at `PROVENANCE` (4) the card includes a provenance section; at `FULL` (3) and
  below it does not. An invalid `repr_verbosity=5` is rejected by the config validator.

## Out of scope

- Threading seeds through the stochastic code paths (matching/sampling/simulation) — the
  `seed` field exists but is populated only when a stochastic result type lands.
- Hostname capture (dropped: not needed for reproducibility, bakes machine identity into
  every serialized result; could be a later opt-in).
- The disk-backed result cache (N4), which will consume this provenance.

## Verification

Run `uv run pytest` with no path argument at least once (so the `pyphi/` doctest sweep
runs), since this touches result `__post_init__`, display, and config validation.
