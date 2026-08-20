---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Configure PyPhi

PyPhi's behavior is controlled by a single configuration object,
`pyphi.config`. This page shows how to read options, change them (globally
or temporarily), load a configuration file, and switch between the built-in
formalism presets.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The three layers

Configuration is split into three namespaces, grouped by what the options
affect:

- **`formalism`** — the theory itself: which IIT version, which distance
  measures, which partition schemes, tie-resolution rules. This layer
  determines *what* is computed.
- **`infrastructure`** — how the computation is carried out: parallelization,
  caching, progress bars, and output verbosity. Changing this layer never
  changes a result, only how fast you get it and what you see.
- **`numerics`** — floating-point behavior, principally `precision`, the
  number of decimal places used when comparing $\varphi$ values.

The `formalism` layer has sub-namespaces of its own: `formalism.iit` for
integrated-information options and `formalism.actual_causation` for actual
causation.

## Reading options

Every option can be read through its full layered path:

```{code-cell} python
pyphi.config.numerics.precision
```

```{code-cell} python
pyphi.config.formalism.iit.version
```

```{code-cell} python
pyphi.config.infrastructure.parallel
```

For convenience, a read on `config` directly is routed to whichever layer
owns the option, so the layer name is optional:

```{code-cell} python
pyphi.config.precision
```

## Setting options

Writes use the flat form. The write is routed to the correct layer
automatically:

```{code-cell} python
pyphi.config.precision = 6
pyphi.config.numerics.precision  # the write was routed to the numerics layer
```

```{code-cell} python
# restore the default
pyphi.config.precision = 13
pyphi.config.precision
```

A change made this way is global and persists until you change it again.
Use `precision` to trade speed for accuracy: lower precision makes $\varphi$
comparisons coarser and computations faster; higher precision is stricter.

## Temporary changes with `override`

To change one or more options for the duration of a block and have them
automatically restored afterward, use `config.override` as a context
manager:

```{code-cell} python
with pyphi.config.override(precision=10):
    print("inside: ", pyphi.config.precision)

print("outside:", pyphi.config.precision)
```

This is the safest way to run a computation under non-default settings: the
previous configuration is restored even if the block raises. It nests, and
it accepts any number of options at once.

```{warning}
Overrides apply to the whole process, not just the current thread. There is
one configuration object per Python process, so while an override is active,
every thread reads the overridden values — a computation running concurrently
in another thread will silently use them. To run computations under different
configurations at the same time, use separate processes. PyPhi's own
process-based parallel backends are unaffected: each worker process receives
its own copy of the configuration.
```

## Presets

A preset is a bundle of options that reproduces the settings of a specific
IIT paper. Three are provided:

- `iit3` — IIT 3.0 (Oizumi et al. 2014)
- `iit4_2023` — IIT 4.0 (Albantakis et al. 2023), without the
  intrinsic-information requirement
- `iit4_2026` — IIT 4.0 with the intrinsic-information requirement (Mayner,
  Marshall, Tononi 2026), the default formalism

Each preset is a dictionary, so you apply it by unpacking it into
`override`. Applying `iit3`, for example, switches the version, the distance
measures, the partition schemes, and `precision` all at once:

```{code-cell} python
import warnings

from pyphi import iit3

# Switching the formalism emits advisory warnings that several config options
# are changing; they are silenced here to keep the output readable.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with pyphi.config.override(**iit3):
        print("version:  ", pyphi.config.formalism.iit.version)
        print("precision:", pyphi.config.precision)

# back to the default formalism outside the block
pyphi.config.formalism.iit.version
```

You can override individual options on top of a preset by passing them as
extra keyword arguments to the same `override` call; later keywords win.

## Loading a configuration file

If a file named `pyphi_config.yml` exists in the directory where you start
Python, PyPhi reads it automatically at import time. It uses the nested
format, with one top-level key per layer:

```yaml
formalism:
  iit:
    version: IIT_4_0_2023
    ces_measure: SUM_SMALL_PHI
infrastructure:
  parallel: false
  progress_bars: true
numerics:
  precision: 13
```

The file is only consulted from the working directory at import time. If you
change directories before importing PyPhi, or run from a directory without
the file, the built-in defaults apply. To change options after import, use
the assignment and `override` forms shown above.

## A few options worth knowing

| Option | Layer | Effect |
| --- | --- | --- |
| `precision` | numerics | Decimal places used to compare $\varphi$ values. |
| `parallel` | infrastructure | Global switch for parallel computation. |
| `cache_repertoires` | infrastructure | Cache repertoire computations. |
| `progress_bars` | infrastructure | Show progress bars during long computations. |
| `repr_verbosity` | infrastructure | Detail level of `repr()` output for result objects. |
| `version` | formalism.iit | Which IIT version to use. |
| `ces_measure` | formalism.iit | Distance measure for cause-effect structures. |

For the complete list of options and their meanings, see the configuration
classes (`IITConfig`, `InfrastructureConfig`, `NumericsConfig`) in the
{doc}`API reference </reference/index>`.
