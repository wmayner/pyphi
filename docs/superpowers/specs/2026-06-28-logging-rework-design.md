# Logging rework — Design

**Status:** proposed
**Date:** 2026-06-28

## Goal

PyPhi currently behaves like an application rather than a library: on config
load it configures the **root** logger and, because `log_file_level` defaults
to `"INFO"`, attaches a `FileHandler` that writes a growing `pyphi.log` into
whatever directory the process runs in. This rework makes PyPhi silent by
default — a `NullHandler` on the `pyphi` logger, nothing on the root logger,
no file — and provides one function, `pyphi.enable_logging()`, to opt in.

## Background

The current mechanism:

- `pyphi/conf/infrastructure.py` defines three config fields: `log_file`
  (`"pyphi.log"`), `log_file_level` (`"INFO"`), `log_stdout_level`
  (`"WARNING"`), each with a validator.
- `pyphi/conf/_callbacks.py:configure_logging` runs `logging.config.dictConfig`
  on the **root** logger, attaching a `FileHandler` (when `log_file_level` is
  set) and a `TqdmHandler` on stdout (when `log_stdout_level` is set).
- `pyphi/conf/_global.py` calls `configure_logging` once in `__init__` and
  again from `_fire_field_callback` whenever any of the three log fields
  change (gated by the `_LOG_FIELDS` frozenset).
- Six modules emit through `logging.getLogger(__name__)` under the `pyphi.*`
  namespace. `pyphi/log.py` provides `TqdmHandler`, a `StreamHandler` that
  writes via `tqdm.write` so log lines do not corrupt progress bars.

The defect is twofold: configuring the root logger is poor library citizenship
(it changes logging for the whole host application), and the default
`FileHandler` writes an unbounded file nobody asked for.

The Python `warnings` system (`PyPhiWarning`, and
`warn_distinction_phi_normalization_change`) is independent of logging and is
**not** touched by this rework.

## Approach

The standard-library remedy for a library that should not configure logging on
its consumers' behalf is to attach a single `NullHandler` to the library's own
logger and let the application decide on real handlers. Adopting it lets the
entire config-driven logging machinery be deleted rather than merely retuned.

This is **Option 1 (function-only)**: remove the three config fields and the
callback machinery, and add one opt-in function. The alternative of keeping the
config fields (defaulted to silent) was rejected because it would retain the
`configure_logging`/`dictConfig` machinery and leave logging configurable two
ways; logging is operational, not result-affecting, so it does not belong in
the `ConfigSnapshot` that every result carries for reproducibility.

## New behavior

### Default (import time)

In `pyphi/__init__.py`, at import:

```python
import logging

logging.getLogger("pyphi").addHandler(logging.NullHandler())
```

PyPhi is now silent: no root-logger configuration, no console output, no file.
The six existing `getLogger(__name__)` call sites continue to emit under the
`pyphi.*` namespace and are absorbed by the `NullHandler` until a user opts in.

### Opt-in — `pyphi.enable_logging`

Defined in `pyphi/log.py` (alongside `TqdmHandler`) and re-exported from
`pyphi/__init__.py`:

```python
def enable_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    """Route PyPhi's logs to the console or a file.

    With no arguments, INFO-and-above messages from the ``pyphi`` logger are
    written to stderr through a progress-bar-safe handler. Pass ``file`` to
    write to a file instead. Calling this again replaces the handler a previous
    call installed (handlers do not stack).
    """
```

Behavior:

- Removes any handler a prior `enable_logging()` call installed (tracked so
  repeated calls replace rather than accumulate); the `NullHandler` stays.
- `file is None` → attach a `TqdmHandler()` (stderr) at `level`.
- `file` given → attach a `logging.FileHandler(file)` at `level`.
- Both handlers use the existing standard formatter
  (`"%(asctime)s [%(name)s] %(levelname)s %(processName)s: %(message)s"`).
- Sets the `pyphi` logger's level to `level` and `propagate = False`, so
  PyPhi's records do not also flow into a host application's root handlers.

`level` accepts the standard level names (`"DEBUG"`, `"INFO"`, `"WARNING"`,
`"ERROR"`, `"CRITICAL"`). An unknown name raises `ValueError` (from
`logging.getLevelName` validation in the implementation).

There is no `disable_logging()` in v1 (YAGNI); it can be added if a real need
to toggle logging off mid-session appears.

## Deletions

| File | Remove |
|---|---|
| `pyphi/conf/infrastructure.py` | the `log_file`, `log_file_level`, `log_stdout_level` fields; their three validators; the `_format_log_levels` helper and `_VALID_LOG_LEVELS` constant and the `Path` import **if** they become unused (confirm with ruff) |
| `pyphi/conf/_callbacks.py` | `configure_logging` in full; keep `warn_distinction_phi_normalization_change` and the `_loaded` machinery |
| `pyphi/conf/_global.py` | the `_LOG_FIELDS` frozenset; the `configure_logging` call in `__init__`; the `_LOG_FIELDS` branch in `_fire_field_callback` (keep the `distinction_phi_normalization` branch) |

The `FIELD_TO_LAYER` routing map, `ConfigSnapshot`, `as_kwargs`, and `diff` all
derive from the dataclass fields via `dataclasses.fields`, so removing the
three fields removes them from the config system and the reproducibility
snapshot with no further edits.

## Touch-ups outside source

- `test/conf/test_config_layers.py` — drop the two assertions on
  `log_file_level` / `log_stdout_level` (lines 154–155).
- `pyphi_config_3.0.yml` — remove the three `log_*` lines (57–59) from the
  sample config.
- `AGENTS.md` — replace the three `log_file` / `log_file_level` /
  `log_stdout_level` bullets (474–476) with a short note on
  `pyphi.enable_logging()` and silent-by-default. (Staged with this work by
  explicit consent.)
- `changelog.d/p10-config-split.refactor.md` — **left unchanged**; it is an
  accurate historical record of the p10 config rename. A fresh changelog
  fragment documents this removal.

## Testing

A new `test/test_logging.py`:

1. **Silent by default:** the `pyphi` logger has a `NullHandler` and emits
   nothing to console or file in the default state (assert via a `caplog`/
   capture that no records reach a real handler).
2. **`enable_logging()` attaches one handler** at INFO; a log call at INFO is
   emitted.
3. **Replacement, not stacking:** a second `enable_logging(file=tmp)` leaves a
   single PyPhi-installed handler (plus the `NullHandler`), and writes to the
   file.
4. **Config fields are gone:** setting `config.log_file_level = "INFO"` raises
   the standard unknown-option `ConfigurationError`.

Full verification: `uv run --all-extras pytest` with no path argument stays
green (doctests + Hypothesis), since nothing result-affecting changes.

## Files

- `pyphi/__init__.py` — NullHandler at import; export `enable_logging`.
- `pyphi/log.py` — add `enable_logging`; keep `TqdmHandler`.
- `pyphi/conf/infrastructure.py` — remove the three fields + validators.
- `pyphi/conf/_callbacks.py` — remove `configure_logging`.
- `pyphi/conf/_global.py` — remove `_LOG_FIELDS` + the two call sites.
- `test/conf/test_config_layers.py` — drop the two field assertions.
- `test/test_logging.py` — new tests above.
- `pyphi_config_3.0.yml` — drop the three log lines.
- `AGENTS.md` — update the logging bullets.
- `changelog.d/logging-rework.change.md` — new fragment.

## Scope

**In scope:** silent-by-default logging, the `enable_logging` opt-in, and
removal of the config-driven logging machinery.

**Out of scope:** the `warnings` / `PyPhiWarning` system (independent of
logging); structured/JSON logging; per-module level control; a
`disable_logging()` counterpart.
