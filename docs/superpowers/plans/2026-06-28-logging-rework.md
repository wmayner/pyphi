# Logging rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PyPhi silent by default (a `NullHandler` on the `pyphi` logger, no root-logger config, no auto-written `pyphi.log`), opt-in via one `pyphi.enable_logging()` function, and delete the config-driven logging machinery.

**Architecture:** Add `enable_logging(level, file)` to `pyphi/log.py` and attach a `NullHandler` to the `pyphi` logger at import in `pyphi/__init__.py`. Then remove the three `log_*` config fields, their validators, the `configure_logging` callback, and its two call sites. The config snapshot/routing derive from dataclass fields, so the fields drop out everywhere automatically.

**Tech Stack:** Python 3.13+, stdlib `logging` (`NullHandler`, `FileHandler`, `StreamHandler`), the existing `pyphi.log.TqdmHandler`, the layered `pyphi.conf` system.

## Global Constraints

- Python 3.13+ only; no back-compat shims.
- Silent by default: after `import pyphi`, the `pyphi` logger has a `NullHandler` and nothing is written to console or file until `enable_logging()` is called.
- Breaking change (intended, fine for 2.0): `pyphi_config.yml` can no longer configure logging; the three `log_*` config fields are removed.
- The `warnings` / `PyPhiWarning` system is untouched (independent of logging).
- No planning-artifact markers (no "N…", "Wave", P-numbers) in `pyphi/` source, docstrings, or `changelog.d/`. Spec/plan files may reference them.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Never `--no-verify`; never `git add -A`. Stage only the named files. `AGENTS.md` **is** staged in this work (explicit user consent for the logging doc update). The pre-commit ruff hook may reformat staged files and abort ("Restored changes from patch") — re-`git add` the same files and re-commit.
- Full verification = `uv run --all-extras pytest` with no path argument (doctests + slow Hypothesis run only this way). Fast lane: `uv run pytest test/test_logging.py test/conf/ -q -p no:cacheprovider`.

---

### Task 1: `enable_logging` + import-time `NullHandler`

**Files:**
- Modify: `pyphi/log.py` (add `enable_logging` + a module-level format constant)
- Modify: `pyphi/__init__.py` (NullHandler at import; export `enable_logging`; add to `__all__`)
- Test: `test/test_logging.py` (new)

**Interfaces:**
- Consumes: `pyphi.log.TqdmHandler` (existing), stdlib `logging`.
- Produces: `pyphi.log.enable_logging(level: str = "INFO", file: str | Path | None = None) -> None`; `pyphi.enable_logging` (re-export). A call removes any non-`NullHandler` handler on the `pyphi` logger first, so repeat calls replace rather than stack — no module-level state.

- [ ] **Step 1: Write the failing test**

Create `test/test_logging.py`:

```python
"""Silent-by-default logging and the enable_logging opt-in."""

from __future__ import annotations

import logging

import pytest

from pyphi import enable_logging
from pyphi import log as plog


def _pyphi_logger() -> logging.Logger:
    return logging.getLogger("pyphi")


def _installed_handlers() -> list[logging.Handler]:
    """PyPhi-logger handlers that are not the import-time NullHandler."""
    return [
        h
        for h in _pyphi_logger().handlers
        if not isinstance(h, logging.NullHandler)
    ]


@pytest.fixture
def restore_pyphi_logging():
    """Snapshot and restore the pyphi logger so a test cannot leak handlers."""
    logger = _pyphi_logger()
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_null_handler_present_by_default():
    assert any(
        isinstance(h, logging.NullHandler) for h in _pyphi_logger().handlers
    )


def test_enable_logging_attaches_one_handler_at_info(restore_pyphi_logging):
    enable_logging()
    installed = _installed_handlers()
    assert len(installed) == 1
    assert isinstance(installed[0], plog.TqdmHandler)
    assert _pyphi_logger().level == logging.INFO
    assert _pyphi_logger().propagate is False


def test_enable_logging_replaces_rather_than_stacks(tmp_path, restore_pyphi_logging):
    enable_logging()
    log_path = tmp_path / "run.log"
    enable_logging(level="DEBUG", file=str(log_path))
    installed = _installed_handlers()
    assert len(installed) == 1
    assert isinstance(installed[0], logging.FileHandler)
    _pyphi_logger().debug("hello-from-test")
    assert "hello-from-test" in log_path.read_text()


def test_enable_logging_rejects_unknown_level(restore_pyphi_logging):
    with pytest.raises(ValueError):
        enable_logging(level="NOPE")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/test_logging.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'enable_logging' from 'pyphi'`.

- [ ] **Step 3: Add `enable_logging` to `pyphi/log.py`**

Replace the contents of `pyphi/log.py` with (the `TqdmHandler` body is unchanged; only the imports, a format constant, the tracker, and `enable_logging` are added):

```python
# log.py
"""Utilities for logging and progress bars."""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(processName)s: %(message)s"


class TqdmHandler(logging.StreamHandler):
    """Logging handler that writes through ``tqdm`` in order to not break
    progress bars.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream, end=self.terminator)
            self.flush()
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)


def enable_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    """Route PyPhi's logs to the console or a file.

    With no arguments, ``INFO``-and-above messages from the ``pyphi`` logger
    are written to stderr through a progress-bar-safe handler. Pass ``file`` to
    write to that path instead. Calling this again replaces the handler a
    previous call installed (handlers do not stack).

    Args:
        level: A standard level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``). An unknown name raises ``ValueError``.
        file: A path to log to. If ``None``, logs go to stderr.
    """
    logger = logging.getLogger("pyphi")
    for installed in list(logger.handlers):
        if not isinstance(installed, logging.NullHandler):
            logger.removeHandler(installed)
    handler: logging.Handler = (
        TqdmHandler() if file is None else logging.FileHandler(str(file))
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
```

Note: `logging.Handler.setLevel` / `Logger.setLevel` accept a level **name**; an unknown name raises `ValueError`, satisfying the unknown-level test without extra validation. Clearing non-`NullHandler` handlers first (rather than tracking the last-installed one in module state) keeps repeat calls from stacking with no global.

- [ ] **Step 4: Wire `pyphi/__init__.py`**

Add `import logging` to the stdlib import block (near `import os`, `pyphi/__init__.py:68-71`):

```python
import importlib
import logging
import os
import pkgutil
from types import ModuleType
```

Add the import-time NullHandler immediately after that import block (before the registry-population imports, around line 72) — this must run on `import pyphi`:

```python
# Silent by default: a library attaches only a NullHandler to its own logger
# and leaves real handlers to the application (or pyphi.enable_logging).
logging.getLogger("pyphi").addHandler(logging.NullHandler())
```

Add the re-export beside the other top-level lifts (next to `from .system import System`, line 102):

```python
from .log import enable_logging as enable_logging
```

Add `"enable_logging"` to the `__all__` list (`pyphi/__init__.py:124`), keeping it alphabetical-ish among the lowercase entries (after `"config"`):

```python
    "config",
    "enable_logging",
    "iit3",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_logging.py -q -p no:cacheprovider`
Expected: PASS (4 passed).

- [ ] **Step 6: Lint**

Run: `uv run ruff check pyphi/log.py pyphi/__init__.py test/test_logging.py && uv run pyright pyphi/log.py pyphi/__init__.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add pyphi/log.py pyphi/__init__.py test/test_logging.py
git commit -m "$(cat <<'EOF'
Add pyphi.enable_logging and a default NullHandler

PyPhi now attaches a NullHandler to the `pyphi` logger at import, so it is
silent unless asked. enable_logging(level, file) attaches a single handler
(TqdmHandler to stderr, or a FileHandler) to the pyphi logger, replacing any
handler a prior call installed and setting propagate=False so records do not
also flow into a host application's root handlers.

<trailer>
EOF
)"
```

---

### Task 2: Remove the config-driven logging machinery

**Files:**
- Modify: `pyphi/conf/infrastructure.py` (remove fields + validators + now-dead helpers/import)
- Modify: `pyphi/conf/_callbacks.py` (remove `configure_logging` + now-dead imports)
- Modify: `pyphi/conf/_global.py` (remove `_LOG_FIELDS`, the import, and the two call sites)
- Modify: `test/conf/test_config_layers.py` (drop the two field assertions)
- Test: `test/test_logging.py` (add silent-by-default + removed-field tests)

**Interfaces:**
- Consumes: `pyphi.conf._field_routing.ConfigurationError` (raised on an unknown config option).
- Produces: no new public surface; `config.log_file` / `log_file_level` / `log_stdout_level` no longer exist.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_logging.py`:

```python
def test_setting_removed_log_field_raises(restore_pyphi_logging):
    from pyphi.conf import config
    from pyphi.conf._field_routing import ConfigurationError

    with pytest.raises(ConfigurationError):
        config.log_file_level = "INFO"


def test_import_writes_no_log_file(tmp_path):
    import subprocess
    import sys

    env = {"PYPHI_WELCOME_OFF": "1", "PATH": __import__("os").environ["PATH"]}
    subprocess.run(
        [sys.executable, "-c", "import pyphi"],
        cwd=tmp_path,
        check=True,
        env=env,
    )
    assert not (tmp_path / "pyphi.log").exists()
```

- [ ] **Step 2: Run to verify the removed-field test fails**

Run: `uv run pytest test/test_logging.py -q -p no:cacheprovider -k "removed_log_field or writes_no_log_file"`
Expected: `test_setting_removed_log_field_raises` FAILS (the field still exists, so the assignment succeeds); `test_import_writes_no_log_file` FAILS (the default `FileHandler` still writes `pyphi.log`).

- [ ] **Step 3: Remove the config fields and validators in `pyphi/conf/infrastructure.py`**

Delete the three field declarations (`pyphi/conf/infrastructure.py:90-92`):

```python
    log_file: str | Path = "pyphi.log"
    log_file_level: str | None = "INFO"
    log_stdout_level: str | None = "WARNING"
```

Delete the three validator blocks in `__post_init__` (`infrastructure.py:158-170`):

```python
        if not isinstance(self.log_file, (str, Path)):
            raise ValueError(
                f"log_file must be str or Path; got {type(self.log_file).__name__}"
            )
        if self.log_file_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_file_level={self.log_file_level!r} not in {_format_log_levels()}"
            )
        if self.log_stdout_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_stdout_level={self.log_stdout_level!r} not in "
                f"{_format_log_levels()}"
            )
```

Delete the now-dead `_VALID_LOG_LEVELS` constant (`infrastructure.py:18-23`) and `_format_log_levels` helper (`infrastructure.py:24-25`), and remove the now-unused `from pathlib import Path` import (`infrastructure.py:13`). (Run ruff after to confirm nothing else uses them.)

- [ ] **Step 4: Remove `configure_logging` in `pyphi/conf/_callbacks.py`**

Delete the entire `configure_logging` function (`pyphi/conf/_callbacks.py:41-79`). Keep `warn_distinction_phi_normalization_change`, the `_LoadedFlag`/`_loaded`/`mark_loaded`/`is_loaded` machinery. Remove the now-unused imports `import logging`, `import logging.config`, and `from pathlib import Path` (the remaining code uses only `warn`, `PyPhiWarning`, and `Any`). Update the module docstring's first paragraph, which currently describes logging reconfiguration:

Replace the opening of the module docstring (`_callbacks.py:1-10`) with:

```python
"""Field-change callbacks for the layered config.

A warning is emitted if ``distinction_phi_normalization`` is changed after
initial load (since it would invalidate cached MICE on existing systems).

The ``_loaded`` flag suppresses warnings during default-state setup; it
flips to ``True`` after any ``pyphi_config.yml`` auto-load completes (or
immediately after construction if no user config is present).
"""
```

- [ ] **Step 5: Remove the call sites in `pyphi/conf/_global.py`**

Remove the import (`pyphi/conf/_global.py:40`):

```python
from pyphi.conf._callbacks import configure_logging
```

Remove the `_LOG_FIELDS` frozenset (`_global.py:58`):

```python
_LOG_FIELDS = frozenset({"log_file", "log_file_level", "log_stdout_level"})
```

Remove the `configure_logging` call in `__init__` (`_global.py:152-153`), leaving the rest of `__init__`:

```python
        object.__setattr__(self, "_numerics", NumericsConfig())
        infra = self._infrastructure
        configure_logging(infra.log_file, infra.log_file_level, infra.log_stdout_level)
```

becomes:

```python
        object.__setattr__(self, "_numerics", NumericsConfig())
```

(The `infra = self._infrastructure` line existed only to feed `configure_logging`; remove it too.)

Replace the `_LOG_FIELDS` branch in `_fire_field_callback` (`_global.py:461-468`):

```python
    def _fire_field_callback(self, field_name: str, old: Any, new: Any) -> None:
        if field_name in _LOG_FIELDS:
            infra = self._infrastructure
            configure_logging(
                infra.log_file, infra.log_file_level, infra.log_stdout_level
            )
        elif field_name == "distinction_phi_normalization":
            warn_distinction_phi_normalization_change(old, new)
```

with:

```python
    def _fire_field_callback(self, field_name: str, old: Any, new: Any) -> None:
        if field_name == "distinction_phi_normalization":
            warn_distinction_phi_normalization_change(old, new)
```

- [ ] **Step 6: Drop the two assertions in `test/conf/test_config_layers.py`**

Remove lines 154-155:

```python
        assert cfg.log_file_level == "INFO"
        assert cfg.log_stdout_level == "WARNING"
```

If removing them empties the assertion body of that test or leaves an unused `cfg`, read the surrounding function (`sed -n '140,160p' test/conf/test_config_layers.py`) and keep it coherent (the test asserts default infrastructure values; leave its other assertions intact).

- [ ] **Step 7: Run the new tests + the config layer tests**

Run: `uv run pytest test/test_logging.py test/conf/test_config_layers.py -q -p no:cacheprovider`
Expected: PASS (6 in `test_logging.py` + the config-layer file green).

- [ ] **Step 8: Lint + full suite**

Run: `uv run ruff check pyphi/conf/infrastructure.py pyphi/conf/_callbacks.py pyphi/conf/_global.py test/test_logging.py test/conf/test_config_layers.py && uv run pyright pyphi/conf/infrastructure.py pyphi/conf/_callbacks.py pyphi/conf/_global.py`
Run (complete check, no path argument): `uv run --all-extras pytest`
Expected: ruff/pyright clean; suite green.

- [ ] **Step 9: Commit**

```bash
git add pyphi/conf/infrastructure.py pyphi/conf/_callbacks.py pyphi/conf/_global.py test/conf/test_config_layers.py test/test_logging.py
git commit -m "$(cat <<'EOF'
Remove the config-driven logging machinery

Delete the log_file / log_file_level / log_stdout_level config fields, their
validators, the configure_logging callback that configured the root logger,
and its two call sites. PyPhi no longer writes pyphi.log or touches the root
logger; logging is controlled solely by pyphi.enable_logging. The config
snapshot and routing derive from dataclass fields, so the fields drop out of
the reproducibility snapshot automatically.

<trailer>
EOF
)"
```

---

### Task 3: Docs, sample config, changelog

**Files:**
- Modify: `AGENTS.md` (replace the three logging-option bullets)
- Modify: `pyphi_config_3.0.yml` (drop the three `log_*` lines)
- Create: `changelog.d/logging-rework.change.md`

- [ ] **Step 1: Update `AGENTS.md`**

Read the block first: `grep -n "log_file\|log_stdout_level\|Debugging & Output" AGENTS.md`. Replace the three bullets (currently `AGENTS.md:474-476`):

```
- **`log_file`**: Log file path (default: ``"pyphi.log"``)
- **`log_file_level`**: File logging level (default: ``"INFO"``)
- **`log_stdout_level`**: Console logging level (default: ``"WARNING"``)
```

with:

```
- **Logging**: PyPhi is silent by default (a ``NullHandler`` on the ``pyphi``
  logger; the root logger is untouched and no log file is written). Opt in with
  ``pyphi.enable_logging(level="INFO", file=None)`` — console (progress-bar
  safe) when ``file`` is omitted, or a file path when given.
```

- [ ] **Step 2: Update `pyphi_config_3.0.yml`**

Remove the three `log_*` lines (`pyphi_config_3.0.yml:57-59`):

```yaml
  log_stdout_level: WARNING
  log_file_level: INFO
  log_file: pyphi.log
```

If that empties an `infrastructure:` block or leaves a dangling mapping, read the file (`sed -n '50,62p' pyphi_config_3.0.yml`) and keep it valid YAML (remove a now-empty parent key only if it has no other children).

- [ ] **Step 3: Add the changelog fragment**

Create `changelog.d/logging-rework.change.md`:

```markdown
PyPhi is now silent by default: it attaches a `NullHandler` to the `pyphi` logger and no longer configures the root logger or writes a `pyphi.log` file. Enable logging explicitly with `pyphi.enable_logging(level="INFO", file=None)`. The `log_file`, `log_file_level`, and `log_stdout_level` configuration options have been removed; logging is no longer set through `pyphi_config.yml`.
```

- [ ] **Step 4: Verify + full suite once more**

Run: `uv run --all-extras pytest`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md pyphi_config_3.0.yml changelog.d/logging-rework.change.md
git commit -m "$(cat <<'EOF'
Document silent-by-default logging and enable_logging

<trailer>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Import-time `NullHandler`, silent by default → Task 1 (NullHandler) + Task 2 (removing the root-logger config that wrote the file); `test_import_writes_no_log_file` proves it. ✓
- `enable_logging(level="INFO", file=None)`, TqdmHandler/FileHandler, replace-not-stack, `propagate=False`, unknown level → `ValueError` → Task 1 + its four tests. ✓
- Remove three config fields + validators + `_format_log_levels`/`_VALID_LOG_LEVELS`/`Path` → Task 2 Step 3. ✓
- Remove `configure_logging` (keep the normalization warning) → Task 2 Step 4. ✓
- Remove `_LOG_FIELDS` + the two `_global.py` call sites → Task 2 Step 5. ✓
- Snapshot/routing auto-derive (no extra edits) → relied on, no task needed (verified: `FIELD_TO_LAYER` is built from `dataclasses.fields`). ✓
- Touch-ups: `test_config_layers.py` assertions (Task 2 Step 6), `pyphi_config_3.0.yml` (Task 3 Step 2), `AGENTS.md` (Task 3 Step 1), changelog (Task 3 Step 3). `changelog.d/p10-config-split.refactor.md` left unchanged (spec says so). ✓
- `warnings`/`PyPhiWarning` untouched → no task touches it. ✓
- No `disable_logging()` → not added. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The two "if removing empties X, read the surrounding lines and keep it coherent" notes (Task 2 Step 6, Task 3 Step 2) are conditional cleanups with the exact inspection command given, not placeholders.

**Type consistency:** `enable_logging(level: str = "INFO", file: str | Path | None = None) -> None` matches between the implementation (Task 1 Step 3) and the tests (Task 1 Step 1). The tests import `pyphi.log as plog` only for `plog.TqdmHandler` (no module-level handler state remains). `ConfigurationError` is imported from `pyphi.conf._field_routing` in both the raise site (existing code) and the test (Task 2 Step 1). The `restore_pyphi_logging` fixture name matches across all tests that use it.
