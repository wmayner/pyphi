# Serialization loading surface + lazy registration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `pyphi.serialize` an ergonomic public surface — path-based `save`/`load`, top-level `pyphi.save`/`pyphi.load`, and `.save()`/`.load()` methods on user-facing result types — and defer type registration to first use so `pyphi.serialize` is safe to import anywhere.

**Architecture:** Registration in `serialize/convert.py` moves from import-time `_register_*()` calls to a single idempotent `_ensure_registered()` triggered by `to_schema`/`from_schema`. The file/path API consolidates onto `save`/`load` (path or file object, format inferred from extension) alongside the bytes `dumps`/`loads`. A standalone `pyphi/serializable.py` defines a `Serializable` mixin whose `save`/`load` **delegate** to `pyphi.serialize` (deferred import); it is mixed into the user-facing result types. `substrate.from_json` is removed.

**Tech Stack:** Python 3.12+, `msgspec`, pytest, ruff + pyright.

## Global Constraints

- **Continue on branch `serialize-msgspec`** (this is part of the serialization rewrite, unmerged). **Ask before `git push`.**
- **Delegation-only rule (load-bearing):** `Serializable.save`/`load` may only delegate to `pyphi.serialize`. No field mapping, reconstruction, or format branching on the domain classes — all serialization logic stays in `serialize/convert.py`.
- **Stage only the files named in each task**; never `git add -A`. Leave `.claude/`, `graphify-out/`, untracked scratch alone.
- **Pre-commit = ruff + pyright; never `--no-verify`.** Ruff reformats and aborts when it changes a file — re-`git add` and re-commit. One import per line.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Full verification = `uv run --all-extras pytest` with NO path argument** (runs the `pyphi/` doctest sweep).

---

## Task 1: Lazy registration in `serialize/convert.py`

**Files:**
- Modify: `pyphi/serialize/convert.py` (the bottom-of-module `_register_*()` block + `to_schema`/`from_schema`)
- Test: `test/test_serialize_lazy_registration.py`

**Interfaces:**
- Produces: `convert._ensure_registered() -> None` (idempotent); `to_schema`/`from_schema` call it on entry.

- [ ] **Step 1: Write the failing test**

`test/test_serialize_lazy_registration.py`:
```python
import subprocess
import sys


def test_importing_serialize_does_not_import_domain_modules():
    # Importing the serializer must not pull in the domain tree; registration
    # is deferred to first use.
    code = (
        "import sys; import pyphi.serialize; "
        "assert 'pyphi.models.sia' not in sys.modules, "
        "'serialize import eagerly imported pyphi.models.sia'; "
        "print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_round_trip_still_works_after_lazy_registration():
    from pyphi import examples, serialize

    sia = examples.basic_system().sia()
    assert serialize.loads(serialize.dumps(sia)) == sia
```

- [ ] **Step 2: Run — expect FAIL** (the first test fails: `pyphi.models.sia` is imported eagerly today)

Run: `uv run --all-extras pytest test/test_serialize_lazy_registration.py -q`

- [ ] **Step 3: Implement lazy registration**

In `pyphi/serialize/convert.py`, replace the bottom block of 35 `_register_*()` calls with a single guard, and call it from the two entry points. The `to_schema`/`from_schema` bodies become:
```python
def to_schema(obj: Any) -> Any:
    _ensure_registered()
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
    _ensure_registered()
    decode = _DECODERS.get(type(struct))
    if decode is None:
        raise TypeError(f"No deserializer registered for {type(struct).__name__}")
    return decode(struct)
```
Replace the trailing `_register_direction()` … `_register_complex()` lines with:
```python
_REGISTERED = False


def _ensure_registered() -> None:
    """Populate the encoder/decoder registries on first use.

    Registration imports the domain modules; deferring it to first
    ``to_schema``/``from_schema`` keeps ``import pyphi.serialize`` free of
    domain imports (and free of import cycles).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    _register_direction()
    _register_pyphi_float()
    # ... every existing _register_* call, in the same order ...
    _register_complex()
```
(Move the existing call list verbatim into the function body; set `_REGISTERED = True` *before* the calls so re-entry during nested encoding short-circuits.)

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serialize_lazy_registration.py -q`

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/convert.py test/test_serialize_lazy_registration.py
git commit  # "Defer serialize type registration to first use"
```

---

## Task 2: Path-based `save`/`load` + format inference

**Files:**
- Modify: `pyphi/serialize/__init__.py` (replace `dump`/`load`; add `save`/`load`/`_infer_format`)
- Test: `test/test_serialize_io.py`

**Interfaces:**
- Produces:
  - `save(obj, target, *, format=None) -> None` — `target` is `str | os.PathLike | <binary file object>`
  - `load(target, *, format=None) -> Any` — same `target`
  - `dumps`/`loads` unchanged.

- [ ] **Step 1: Write the failing test**

`test/test_serialize_io.py`:
```python
import io

import pytest

from pyphi import examples, serialize

FORMATS = ["json", "msgpack"]


@pytest.fixture
def sia():
    return examples.basic_system().sia()


@pytest.mark.parametrize("ext,fmt", [(".json", "json"), (".msgpack", "msgpack"),
                                     (".mpk", "msgpack")])
def test_save_load_roundtrip_by_path(tmp_path, sia, ext, fmt):
    path = tmp_path / f"result{ext}"
    serialize.save(sia, path)            # format inferred from extension
    assert serialize.load(path) == sia
    # the file really is in the inferred format
    assert serialize.loads(path.read_bytes(), format=fmt) == sia


def test_save_load_roundtrip_by_file_object(sia):
    buf = io.BytesIO()
    serialize.save(sia, buf)             # file object defaults to JSON
    buf.seek(0)
    assert serialize.load(buf) == sia


def test_explicit_format_overrides_extension(tmp_path, sia):
    path = tmp_path / "result.json"      # .json suffix ...
    serialize.save(sia, path, format="msgpack")   # ... but written as msgpack
    assert serialize.load(path, format="msgpack") == sia


def test_unknown_extension_defaults_to_json(tmp_path, sia):
    path = tmp_path / "result.dat"
    serialize.save(sia, path)
    assert serialize.loads(path.read_bytes(), format="json") == sia
```

- [ ] **Step 2: Run — expect FAIL** (`serialize.save` does not exist)

Run: `uv run --all-extras pytest test/test_serialize_io.py -q`

- [ ] **Step 3: Implement**

In `pyphi/serialize/__init__.py`, add imports `import os` and `from pathlib import Path` at the top, **remove** the existing `dump` and `load` functions, and add:
```python
_SUFFIX_FORMATS = {".json": "json", ".msgpack": "msgpack", ".mpk": "msgpack"}


def _infer_format(target: Any, format: str | None) -> str:
    if format is not None:
        return format
    if isinstance(target, (str, os.PathLike)):
        return _SUFFIX_FORMATS.get(Path(target).suffix.lower(), "json")
    return "json"


def save(obj: Any, target: Any, *, format: str | None = None) -> None:
    """Serialize ``obj`` to ``target`` (a path or an open binary file object).

    The wire format is inferred from a path's extension (``.json`` →
    ``"json"``; ``.msgpack`` / ``.mpk`` → ``"msgpack"``; otherwise ``"json"``)
    unless ``format`` is given.
    """
    data = dumps(obj, format=_infer_format(target, format))
    if isinstance(target, (str, os.PathLike)):
        with open(target, "wb") as f:
            f.write(data)
    else:
        target.write(data)


def load(target: Any, *, format: str | None = None) -> Any:
    """Deserialize from ``target`` (a path or an open binary file object)."""
    fmt = _infer_format(target, format)
    if isinstance(target, (str, os.PathLike)):
        with open(target, "rb") as f:
            data = f.read()
    else:
        data = target.read()
    return loads(data, format=fmt)
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serialize_io.py -q`

- [ ] **Step 5: Update the conftest/test call-sites that used the old `load(fp)`**

The previous `load(fp)` signature is subsumed by the new `load(target)` (it accepts file objects), so `test/conftest.py`, `test/test_relations.py`, and `test/test_iit4.py` (`serialize.load(f)`) keep working unchanged. Confirm:
Run: `uv run --all-extras pytest test/test_relations.py test/test_iit4.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/serialize/__init__.py test/test_serialize_io.py
git commit  # "Add path-based serialize.save/load with format inference"
```

---

## Task 3: `Serializable` mixin + apply to `Substrate`

**Files:**
- Create: `pyphi/serializable.py`
- Modify: `pyphi/substrate.py` (add the mixin to `Substrate`'s bases)
- Test: `test/test_serializable_mixin.py`

**Interfaces:**
- Produces: `pyphi.serializable.Serializable` with `save(self, target, *, format=None)` and classmethod `load(cls, target, *, format=None)`.

- [ ] **Step 1: Write the failing test**

`test/test_serializable_mixin.py`:
```python
import pytest

from pyphi import examples
from pyphi.substrate import Substrate


def test_instance_save_and_classmethod_load(tmp_path):
    sub = examples.basic_substrate()
    path = tmp_path / "sub.json"
    sub.save(path)
    assert Substrate.load(path) == sub


def test_load_typechecks(tmp_path):
    # A file holding a different type must not load as a Substrate.
    from pyphi import serialize

    sia = examples.basic_system().sia()
    path = tmp_path / "sia.json"
    serialize.save(sia, path)
    with pytest.raises(TypeError):
        Substrate.load(path)
```

- [ ] **Step 2: Run — expect FAIL** (`Substrate` has no `save`)

Run: `uv run --all-extras pytest test/test_serializable_mixin.py -q`

- [ ] **Step 3: Create the mixin**

`pyphi/serializable.py`:
```python
"""The ``Serializable`` mixin.

Adds ``save``/``load`` convenience to user-facing result types. These methods
are **thin delegations** to :mod:`pyphi.serialize` — all serialization logic
lives there, never on the domain classes. The ``serialize`` import is deferred
to call time so this module stays free of heavy imports and import cycles.
"""

from __future__ import annotations

from typing import Any


class Serializable:
    """Adds ``save``/``load`` that delegate to :mod:`pyphi.serialize`."""

    def save(self, target: Any, *, format: str | None = None) -> None:
        """Serialize this object to ``target`` (a path or binary file object)."""
        from pyphi import serialize

        serialize.save(self, target, format=format)

    @classmethod
    def load(cls, target: Any, *, format: str | None = None) -> Any:
        """Load an instance of this type from ``target``.

        Raises ``TypeError`` if the file holds a different type.
        """
        from pyphi import serialize

        obj = serialize.load(target, format=format)
        if not isinstance(obj, cls):
            raise TypeError(
                f"{target!r} contains a {type(obj).__name__}, not a {cls.__name__}"
            )
        return obj
```

- [ ] **Step 4: Add the mixin to `Substrate`**

In `pyphi/substrate.py`: add `from .serializable import Serializable` with the other intra-package imports, and change `class Substrate(Displayable):` to `class Substrate(Displayable, Serializable):`.

- [ ] **Step 5: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serializable_mixin.py -q`

- [ ] **Step 6: Commit**

```bash
git add pyphi/serializable.py pyphi/substrate.py test/test_serializable_mixin.py
git commit  # "Add Serializable mixin (save/load delegating to serialize); apply to Substrate"
```

---

## Task 4: Apply the mixin to the remaining user-facing result types

**Files (each: add `Serializable` to the class bases + the import):**
- Modify: `pyphi/system.py` (`System`)
- Modify: `pyphi/actual.py` (`Transition`)
- Modify: `pyphi/models/sia.py` (`IIT3SystemIrreducibilityAnalysis`)
- Modify: `pyphi/formalism/iit4/__init__.py` (`SystemIrreducibilityAnalysis` — `Null…` inherits)
- Modify: `pyphi/models/ces.py` (`CauseEffectStructure` — `Null…` inherits)
- Modify: `pyphi/models/complex.py` (`Complex`)
- Modify: `pyphi/models/actual_causation.py` (`Account` — `DirectedAccount` inherits; `AcSystemIrreducibilityAnalysis`)
- Modify: `pyphi/models/distinctions.py` (`Distinctions` — `Resolved`/`Unresolved` inherit)
- Modify: `pyphi/relations.py` (`Relations` — `ConcreteRelations`/`NullRelations`/`AnalyticalRelations` inherit)
- Test: `test/test_serializable_mixin.py` (extend)

**Interfaces:**
- Consumes: `Serializable` (Task 3).

- [ ] **Step 1: Write the failing parametrized test** (append to `test/test_serializable_mixin.py`)

(`test/test_serializable_mixin.py` already imports `pytest` at the top from
Task 3; do not re-import it.)
```python
def _all_result_objects():
    from pyphi import actual, examples
    from pyphi.actual import Transition
    from pyphi.models.complex import Complex

    system = examples.basic_system()
    sia = system.sia()
    ces = system.ces()
    ac_sub = examples.actual_causation_substrate()
    transition = Transition(ac_sub, before_state=(1, 1), after_state=(1, 1),
                            cause_indices=(0, 1), effect_indices=(0, 1))
    return {
        "system": system,
        "transition": transition,
        "sia": sia,
        "ces": ces,
        "account": actual.account(transition),
        "acsia": actual.sia(transition),
        "complex": Complex(sia=sia, substrate=system.substrate,
                           is_maximal=True, excluded=()),
        "distinctions": ces.distinctions,
        "relations": ces.relations,
    }


@pytest.mark.parametrize(
    "name",
    ["system", "transition", "sia", "ces", "account", "acsia", "complex",
     "distinctions", "relations"],
)
def test_every_user_facing_type_round_trips_via_method(tmp_path, name):
    obj = _all_result_objects()[name]
    path = tmp_path / f"{name}.json"
    obj.save(path)
    restored = type(obj).load(path)
    assert restored == obj
```

- [ ] **Step 2: Run — expect FAIL** (most types have no `.save`)

Run: `uv run --all-extras pytest test/test_serializable_mixin.py -q`

- [ ] **Step 3: Add the mixin to each class**

For each file above, add the import (`from pyphi.serializable import Serializable`, matching the file's import style) and append `Serializable` to the target class's base list. Examples:
- `pyphi/system.py`: `class System(Displayable, Serializable):` (it is an `@dataclass`; a method-only mixin adds no fields).
- `pyphi/models/sia.py`: `class IIT3SystemIrreducibilityAnalysis(HasProvenance, Displayable, cmp.OrderableByPhi, Serializable):`
- `pyphi/formalism/iit4/__init__.py`: add `Serializable` to `SystemIrreducibilityAnalysis`'s bases (Null subclass inherits).
- `pyphi/models/ces.py`: add to `CauseEffectStructure` (Null subclass inherits).
- `pyphi/models/complex.py`: add to `Complex`.
- `pyphi/models/actual_causation.py`: add to `Account` (DirectedAccount inherits) and `AcSystemIrreducibilityAnalysis`.
- `pyphi/models/distinctions.py`: add to `Distinctions` (subclasses inherit).
- `pyphi/relations.py`: add to `Relations` (subclasses inherit).
- `pyphi/actual.py`: add to `Transition`.

For frozenset-based types reached via a base (`Relations`/`Distinctions`), adding the mixin to the base is sufficient. Verify imports don't cycle: `serializable.py` imports nothing heavy, so these module-level imports are safe.

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serializable_mixin.py -q`

- [ ] **Step 5: Commit**

```bash
git add pyphi/system.py pyphi/actual.py pyphi/models/sia.py pyphi/formalism/iit4/__init__.py pyphi/models/ces.py pyphi/models/complex.py pyphi/models/actual_causation.py pyphi/models/distinctions.py pyphi/relations.py test/test_serializable_mixin.py
git commit  # "Apply Serializable to the user-facing result types"
```

---

## Task 5: Top-level `pyphi.save` / `pyphi.load`

**Files:**
- Modify: `pyphi/__init__.py` (lift `save`/`load`; add to `__all__`)
- Test: `test/test_serialize_io.py` (extend)

- [ ] **Step 1: Write the failing test** (append to `test/test_serialize_io.py`)

```python
def test_top_level_save_load(tmp_path, sia):
    import pyphi

    path = tmp_path / "r.json"
    pyphi.save(sia, path)
    assert pyphi.load(path) == sia
```

- [ ] **Step 2: Run — expect FAIL** (`pyphi.save` undefined)

Run: `uv run --all-extras pytest test/test_serialize_io.py::test_top_level_save_load -q`

- [ ] **Step 3: Lift the names**

In `pyphi/__init__.py`, after the existing value imports (near `from .system import System`), add:
```python
from .serialize import load
from .serialize import save
```
and add `"load"` and `"save"` to the static `__all__` list (in sorted position).

- [ ] **Step 4: Run — expect PASS, and confirm `import pyphi` is clean**

Run: `uv run --all-extras pytest test/test_serialize_io.py -q`
Run: `uv run python -c "import pyphi; print(pyphi.save, pyphi.load)"`
Expected: both PASS (no import error; lazy registration keeps the `serialize` import light).

- [ ] **Step 5: Commit**

```bash
git add pyphi/__init__.py test/test_serialize_io.py
git commit  # "Expose top-level pyphi.save / pyphi.load"
```

---

## Task 6: Remove `substrate.from_json`

**Files:**
- Modify: `pyphi/substrate.py` (delete the module-level `from_json(filename)` function)

- [ ] **Step 1: Confirm no callers**

Run: `grep -rn "substrate.from_json\|from pyphi.substrate import from_json" --include="*.py" .`
Expected: empty (verified during planning).

- [ ] **Step 2: Delete the function**

Remove the entire module-level `def from_json(filename: str) -> Substrate:` block (including its deferred `from . import serialize`) from `pyphi/substrate.py`. Users now use `pyphi.load(path)` / `Substrate.load(path)`.

- [ ] **Step 3: Verify**

Run: `uv run python -c "from pyphi import examples, serialize; import tempfile,os; p=tempfile.mktemp(suffix='.json'); examples.basic_substrate().save(p); print(type(__import__('pyphi').load(p)).__name__); os.unlink(p)"`
Expected: prints `Substrate`.

- [ ] **Step 4: Commit**

```bash
git add pyphi/substrate.py
git commit  # "Remove redundant substrate.from_json (use serialize.load / Substrate.load)"
```

---

## Task 7: Demo notebook, changelog, ROADMAP, full verification

**Files:**
- Modify: `serialize_demo.ipynb` (`serialize.dump` → `serialize.save`)
- Create: `changelog.d/serialize-loading-surface.feature.md`
- Modify: `ROADMAP.md` (serialization-loading-surface note)

- [ ] **Step 1: Update the demo notebook**

In `serialize_demo.ipynb`, replace the two `serialize.dump(ces, f, format=...)` calls with `serialize.save(ces, "ces.json")` / `serialize.save(ces, "ces.msgpack")` (path-based, format inferred), and update the surrounding prose to the new API. Re-extract and run the code cells to confirm they execute (as in the original notebook verification).

- [ ] **Step 2: Changelog**

```bash
echo 'Added ergonomic loading helpers to `pyphi.serialize`: path-based `save`/`load` (format inferred from extension), top-level `pyphi.save`/`pyphi.load`, and `.save()`/`.load()` methods on result objects (`result.save("r.json")`, `CauseEffectStructure.load("r.json")`). Removed the redundant `pyphi.substrate.from_json` loader.' > changelog.d/serialize-loading-surface.feature.md
```

- [ ] **Step 3: Full verification**

Run: `uv run --all-extras pytest`
Expected: PASS (doctest sweep included).
Run: `SKIP=pyright uv run pre-commit run --all-files && uv run pyright pyphi`
Expected: clean.

- [ ] **Step 4: Update ROADMAP**

In the Wave-5 jsonify→msgspec detail, note the loading surface landed (path-based `save`/`load`, top-level re-exports, `Serializable` mixin, `substrate.from_json` removed, lazy registration). Keep the `validate_json_version` follow-up note.

- [ ] **Step 5: Commit and finish the branch**

```bash
git add serialize_demo.ipynb changelog.d/serialize-loading-surface.feature.md ROADMAP.md
git commit  # "Record serialization loading surface in changelog and ROADMAP"
```
Then use superpowers:finishing-a-development-branch. **Ask before pushing.**

---

## Notes for the implementer

- **Lazy registration ordering:** set `_REGISTERED = True` before the `_register_*()` calls so nested encoding (an encoder calling `to_schema`) re-enters `_ensure_registered` harmlessly.
- **`save` replaces `dump`:** the old `dump(obj, fp)` / `load(fp)` pair collapses into `save`/`load`, which accept a path *or* a file object. The only `serialize.dump` call-site is the demo notebook (Task 7).
- **Mixin is delegation-only.** If you find yourself adding serialization logic to a domain class, stop — it belongs in `serialize/convert.py`.
- **Verification of record** is `uv run --all-extras pytest` with no path argument (the doctest sweep), Task 7 Step 3.
