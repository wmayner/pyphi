# Configuring PyPhi

All of PyPhi's behavior is controlled by one object, `pyphi.config`. Through
this server's tools you rarely touch it — `analyze` takes a `formalism`
argument and pins the whole preset for you. Reach for `pyphi.config` when you
write PyPhi code in a shell, and read this first.

## The three layers

Options are grouped into three namespaces by what they affect:

- **`formalism`** — the theory: which IIT version, distance measures, partition
  schemes, tie-resolution rules. This layer determines *what* is computed.
  It has sub-namespaces `formalism.iit` and `formalism.actual_causation`.
- **`infrastructure`** — how the computation runs: parallelism, caching,
  progress bars, verbosity. Changing this never changes a result, only how fast
  you get it and what you see.
- **`numerics`** — floating-point behavior, principally `precision`, the number
  of decimal places used when comparing φ values.

## Reading and setting

Read through the full path or the flat shortcut, which routes to whichever layer
owns the option:

```python
pyphi.config.numerics.precision   # full path
pyphi.config.precision            # flat shortcut, same value
```

Writes use the flat form and are routed automatically. A plain assignment is
global and persists until changed:

```python
pyphi.config.precision = 6
```

For a scoped change that is restored on exit (even if the block raises), use
`override` as a context manager. This is the safe way to run one computation
under non-default settings; it nests and accepts any number of options:

```python
with pyphi.config.override(precision=10):
    ...
```

## Presets

A preset is a bundle of options reproducing a specific paper's settings. Three
are provided as dictionaries you unpack into `override`:

- `iit3` — IIT 3.0 (Oizumi et al. 2014)
- `iit4_2023` — IIT 4.0 (Albantakis et al. 2023), without the
  intrinsic-information requirement
- `iit4_2026` — IIT 4.0 with the requirement (Mayner, Marshall, Tononi 2026),
  the default

```python
from pyphi import iit3
with pyphi.config.override(**iit3):
    ...
```

Applying a preset switches the version, the measures, the partition schemes, and
`precision` together. Selecting a formalism by name — either `analyze(...,
formalism="iit3")` through the tools, or a preset in a script — is safer than
setting `formalism.iit.version` alone, which leaves the measures on the previous
formalism.

## Configuration file

If `pyphi_config.yml` exists in the directory where Python starts, PyPhi reads
it at import time. It uses the nested format, one top-level key per layer:

```yaml
formalism:
  iit:
    version: IIT_4_0_2023
infrastructure:
  parallel: false
numerics:
  precision: 13
```

The file is consulted only from the working directory, only at import time.

## Options worth knowing

| Option | Layer | Effect |
| --- | --- | --- |
| `precision` | numerics | Decimal places used to compare φ values; lower is coarser and faster. |
| `parallel` | infrastructure | Master gate for parallelism — necessary but not sufficient; each level has its own switch. See `get_iit_reference("parallelization")`. |
| `progress_bars` | infrastructure | Show progress bars during long computations. |
| `cache_repertoires` | infrastructure | Memoize repertoire computations (on by default). |
| `disk_cache_results` | infrastructure | Persist whole results to disk (off by default). |
| `version` | formalism.iit | Which IIT version to use. |
| `ces_measure` | formalism.iit | Distance measure for cause-effect structures. |
| `shortcircuit_distinctions` | formalism.iit | Skip the remaining MICE search when a distinction is already known reducible (on by default; `False` gives exhaustive sweeps with exact margins). |

Caching and running large analyses without losing work have their own topic:
read `get_iit_reference("performance")` before starting anything expensive.

The complete option reference — every option in every layer, with its default —
is appended below, generated from the config classes so it always matches the
installed version.
