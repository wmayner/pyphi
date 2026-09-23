# Configuring PyPhi

All of PyPhi's behavior is controlled by one object, `pyphi.config`. Through
this server's tools you rarely touch it. Reach for `pyphi.config` when you
write PyPhi code in a shell, and read this first.

## The three layers

Options are grouped into three namespaces by what they affect:

- **`formalism`** — the theory: the version of IIT and the measures, partition
  schemes, and tie-resolution rules that define it. This layer determines
  *what* is computed. It has sub-namespaces `formalism.iit` and
  `formalism.actual_causation`.
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

## Earlier versions of IIT

PyPhi computes IIT (Albantakis et al. 2023; Mayner, Marshall & Tononi 2026)
by default. It can also compute two earlier versions, to reproduce results
published under them. Each is a complete earlier statement of the theory, and
its φ values cannot be compared with values computed under IIT.

- **`IIT_4_0_2023`** (Albantakis et al. 2023, without the intrinsic-information
  requirement of Mayner et al. 2026): φₛ = min(φ_c, φ_e), so a fully
  deterministic system can have positive φₛ. Distinctions, relations, and the
  intrinsic difference are the same as in IIT.
- **`IIT_3_0`** (Oizumi, Albantakis & Tononi 2014, PLoS Comput Biol 10(5):
  e1003588):
  - It computes *concepts*. There are no relations, and so no structure
    integrated information: its Φ is the system-level value on `.phi`, and
    `.big_phi` raises. `plot_ces` cannot draw its structure. The `analyze`
    tool's summary has no `big_phi`, no `intrinsic_information`, and no
    `congruence` key, because IIT 3.0 has no congruence filter.
  - It measures differences with the **earth mover's distance** (a sum that
    weights distant states more) rather than the intrinsic difference (a max
    over a single state). Eq. 3 is cause–effect information
    `cei = min(ci, ei)`, Eq. 8 small phi `φ = min(φ_cause, φ_effect)` over
    the MIP, and Eq. 11 big Φ, the earth mover's distance between the whole
    constellation and its unidirectionally partitioned version.
  - Its background convention is PyPhi 1.x's: the background is held at its
    current state for causes as well as effects
    (`background_conditioning="CONDITION_CURRENT_STATE"`). Oizumi et al. (2014)
    held it at its actual past state for causes (IIT 4.0, S2 Text). IIT instead
    causally marginalizes the background on the cause side, weighting past
    background states by their probability given the current state
    (`"CAUSAL_MARGINALIZATION"`, Albantakis et al. 2023, Eqs. 3–4). This only
    affects systems smaller than the whole substrate. Under the IIT 3.0
    convention, a system's state must also be producible with the background
    held at its current state, so states that are reachable under IIT can
    raise `StateUnreachableForwardsError`.

Select a version by name: `analyze(..., formalism="IIT_3_0")` (or
`"IIT_4_0_2023"`) through the tools. In a script, apply the complete preset,
which sets the version, the measures, the partition schemes, and `precision`
together:

```python
from pyphi import iit3
with pyphi.config.override(**iit3):
    ...
```

The presets are `pyphi.iit3`, `pyphi.iit4_2023`, and `pyphi.iit4_2026` (IIT,
the default). Setting `formalism.iit.version` on its own raises a
`ConfigurationError`: the measures would stay where they were, and the result
would be a mixture matching no paper.

## Configuration file

If `pyphi_config.yml` exists in the directory where Python starts, PyPhi reads
it at import time. It uses the nested format, one top-level key per layer:

```yaml
formalism:
  iit:
    shortcircuit_distinctions: false
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
| `version` | formalism.iit | The version of IIT; set it through a preset (see below). |
| `ces_measure` | formalism.iit | Distance measure for cause-effect structures. |
| `shortcircuit_distinctions` | formalism.iit | Skip the remaining MICE search when a distinction is already known reducible (on by default; `False` gives exhaustive sweeps with exact margins). |

Caching and running large analyses without losing work have their own topic:
read `get_iit_reference("performance")` before starting anything expensive.

The complete option reference — every option in every layer, with its default —
is appended below, generated from the config classes so it always matches the
installed version.
