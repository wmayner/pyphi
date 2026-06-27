# Disk-backed result cache (N4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist top-level SIA/CES results to disk, keyed on the P9.5 content fingerprint + result-affecting config + a code-version component, so re-running the same computation in a later session loads instead of recomputes.

**Architecture:** A small content-addressed file store (`pyphi/cache/disk.py`) holds `pyphi.serialize`-encoded results, one file per key. A pure key builder folds `System._fingerprint`, a curated config digest, the result kind, and a git-sha/version code stamp into one hex key, returning `None` (do-not-cache) on a dirty tree. `System.sia()` / `System.ces()` call a thin `maybe_disk_cached` wrapper that is off by default, bypassed when result-affecting kwargs are passed, and bypassed on a dirty tree.

**Tech Stack:** Python 3.13+, `pyphi.serialize` (msgpack+gzip), `hashlib.blake2b`, `pyphi.provenance` git/version logic, the existing `pyphi.cache` registry.

## Global Constraints

- Python 3.13+ only; no back-compat shims.
- Opt-in: `config.infrastructure.disk_cache_results` defaults to **False**. With it off, no directory/file is created and the compute path is byte-identical to today.
- The disk path is correctness-critical: a hit must equal recomputation. The key must separate any difference that changes the result (math identity, result-affecting config, result kind, code version); a dirty git tree disables the cache entirely.
- joblib is **not** removed (loky is the process backend); the Hamming `joblib_memory` cache is untouched.
- No planning-artifact markers (no "N4", "Wave", P-numbers) in `pyphi/` source, docstrings, or `changelog.d/`. Spec/plan files may reference them.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Never `--no-verify`; never `git add -A`; never stage `AGENTS.md`. The pre-commit ruff hook may reformat staged files and abort ("Restored changes from patch") — re-`git add` the same files and re-commit.
- Full verification = `uv run --all-extras pytest` with no path argument (must stay green; doctests + slow Hypothesis run only this way).

---

### Task 1: `DiskCache` store + record codec

**Files:**
- Create: `pyphi/cache/disk.py`
- Modify: `pyphi/cache/__init__.py` (export `DiskCache`)
- Test: `test/cache/test_disk_cache.py`

**Interfaces:**
- Consumes: `pyphi.constants.DISK_CACHE_LOCATION` (`Path("__pyphi_cache__")`), the `pyphi.cache.registry.register` + `pyphi.cache.cache_utils._CacheInfo` surface, `pyphi.serialize.dumps`/`loads`.
- Produces: `DiskCache(name: str, subdir: str)` with `get(key: str) -> bytes | None`, `put(key: str, data: bytes) -> None`, `clear() -> None`, `size: int` property, `info() -> _CacheInfo`, `name: str`; module-level `_decode_or_none(data: bytes) -> Any | None` (returns `None` on any deserialize error). Encoding at the call site is plain `serialize.dumps(obj, format="msgpack")` — no in-file version tag (the cache key's code-version component already guards staleness).

- [ ] **Step 1: Write the failing test**

Create `test/cache/test_disk_cache.py`:

```python
"""Disk-backed result store: round-trip, atomic writes, corruption tolerance."""

from __future__ import annotations

from pyphi.cache.disk import DiskCache
from pyphi.cache.disk import _decode_or_none


def test_put_get_round_trip(tmp_path, monkeypatch):
    import pyphi.constants as constants

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    cache = DiskCache("test.disk", "results_t1")
    assert cache.get("abc") is None  # miss before put
    cache.put("abc", b"hello")
    assert cache.get("abc") == b"hello"
    assert cache.size == 1
    cache.clear()
    assert cache.get("abc") is None
    assert cache.size == 0


def test_decode_or_none_tolerates_corruption():
    assert _decode_or_none(b"not a valid record") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/cache/test_disk_cache.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.cache.disk'`.

- [ ] **Step 3: Implement `pyphi/cache/disk.py`**

```python
"""Disk-backed content-addressed store for top-level results.

Persists serialized results to one file per key under
``DISK_CACHE_LOCATION``. Keys are opaque hex strings built by the key
module functions; values are ``serialize``-encoded results. A truncated or
unreadable file decodes to ``None`` (a silent miss), never an exception
reaching the caller; staleness across code or config changes is handled by
the cache key (its code-version component), not an in-file tag.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyphi import constants
from pyphi import serialize
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.registry import register as _register_policy


def _decode_or_none(data: bytes) -> Any | None:
    """Deserialize a stored result; ``None`` on any error (a cache miss).

    Staleness across code or config changes is handled entirely by the cache
    key (it folds in a code-version component), so there is no in-file version
    tag; this only tolerates a corrupt/truncated file.
    """
    try:
        return serialize.loads(data, format="msgpack")
    except Exception:  # noqa: BLE001 - any decode failure is a cache miss
        return None


class DiskCache:
    """A content-addressed file store satisfying the CachePolicy surface."""

    def __init__(self, name: str, subdir: str) -> None:
        self.name = name
        self._subdir = subdir
        self.hits = 0
        self.misses = 0
        _register_policy(self)

    @property
    def _dir(self) -> Path:
        return constants.DISK_CACHE_LOCATION / self._subdir

    def get(self, key: str) -> bytes | None:
        try:
            data = (self._dir / key).read_bytes()
        except OSError:
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / f".{key}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, self._dir / key)

    def clear(self) -> None:
        if self._dir.exists():
            for path in self._dir.iterdir():
                if path.is_file():
                    path.unlink()

    @property
    def size(self) -> int:
        if not self._dir.exists():
            return 0
        return sum(1 for path in self._dir.iterdir() if path.is_file())

    def info(self) -> _CacheInfo:
        return _CacheInfo(self.hits, self.misses, self.size)
```

- [ ] **Step 4: Export from the cache package**

In `pyphi/cache/__init__.py`, with the other bottom-of-file re-exports (next to `from .content import ContentCache as ContentCache`):

```python
from .disk import DiskCache as DiskCache  # noqa: E402
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/cache/test_disk_cache.py -q -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint**

Run: `uv run ruff check pyphi/cache/disk.py test/cache/test_disk_cache.py && uv run pyright pyphi/cache/disk.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add pyphi/cache/disk.py pyphi/cache/__init__.py test/cache/test_disk_cache.py
git commit -m "$(cat <<'EOF'
Add DiskCache content-addressed file store for results

One serialized result per key under __pyphi_cache__, written atomically
(temp + os.replace). A truncated or unreadable file decodes to None (a silent
miss), never an exception; staleness across code/config changes is handled by
the cache key, not an in-file tag. Satisfies the CachePolicy surface
(info/clear) and registers with the cache registry.

<trailer>
EOF
)"
```

---

### Task 2: Cache-key builder

**Files:**
- Modify: `pyphi/cache/disk.py` (add the key functions)
- Test: `test/cache/test_disk_cache_key.py`

**Interfaces:**
- Consumes: `System._fingerprint` (bytes), a `ConfigSnapshot` (with `.formalism.iit` fields `version`, `mechanism_phi_measure`, `system_phi_measure`, `specification_measure`, `ces_measure`, `mechanism_partition_scheme`, `system_partition_scheme`; `.numerics.precision`), `pyphi.provenance._git_info() -> tuple[str | None, bool | None]`, `importlib.metadata.version`.
- Produces: `result_cache_key(system, kind: str, snapshot) -> str | None` (hex, or `None` on a dirty tree); helpers `_config_digest(snapshot) -> bytes` and `_code_version() -> str`.

The key separates everything that changes a result: the system's math identity (`_fingerprint`), the **result kind** (`"sia"` vs `"ces"` — same system+config, different computation), the result-affecting config, and the code version.

- [ ] **Step 1: Write the failing test**

Create `test/cache/test_disk_cache_key.py`:

```python
"""Cache-key builder: separates what changes a result, reuses what doesn't."""

from __future__ import annotations

import pyphi.cache.disk as disk
from pyphi import examples
from pyphi.conf import _global
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.substrate import Substrate


def _snap():
    return _global.snapshot()


def test_key_is_hex_str_and_deterministic():
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        k1 = disk.result_cache_key(s, "sia", _snap())
        k2 = disk.result_cache_key(s, "sia", _snap())
    assert isinstance(k1, str) and k1 == k2
    int(k1, 16)  # hex


def test_kind_separates_sia_from_ces():
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        assert disk.result_cache_key(s, "sia", _snap()) != disk.result_cache_key(
            s, "ces", _snap()
        )


def test_config_separates_formalism_versions():
    s = examples.basic_system()
    with config.override(**presets.iit4_2023):
        k_2023 = disk.result_cache_key(s, "sia", _snap())
    with config.override(**presets.iit4_2026):
        k_2026 = disk.result_cache_key(s, "sia", _snap())
    assert k_2023 != k_2026


def test_relabeled_equivalent_system_shares_key():
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        relabeled = Substrate.from_factored(
            s.substrate.factored_tpm,
            cm=s.substrate.cm,
            node_labels=("X", "Y", "Z"),
        )
        from pyphi import System

        s2 = System(relabeled, s.state)
        assert disk.result_cache_key(s, "sia", _snap()) == disk.result_cache_key(
            s2, "sia", _snap()
        )


def test_dirty_tree_returns_none(monkeypatch):
    monkeypatch.setattr(disk, "_git_info", lambda: ("abc123", True))
    with config.override(**presets.iit4_2023):
        s = examples.basic_system()
        assert disk.result_cache_key(s, "sia", _snap()) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/cache/test_disk_cache_key.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'pyphi.cache.disk' has no attribute 'result_cache_key'`.

- [ ] **Step 3: Implement the key functions in `pyphi/cache/disk.py`**

Add the imports at the top of `pyphi/cache/disk.py`:

```python
import hashlib
import importlib.metadata

from pyphi.provenance import _git_info
```

Add the functions:

```python
def _config_digest(snapshot: Any) -> bytes:
    """Digest only the configuration fields that change a result value."""
    iit = snapshot.formalism.iit
    fields = (
        iit.version,
        iit.mechanism_phi_measure,
        iit.system_phi_measure,
        iit.specification_measure,
        iit.ces_measure,
        iit.mechanism_partition_scheme,
        iit.system_partition_scheme,
        snapshot.numerics.precision,
    )
    return repr(fields).encode()


def _code_version() -> str:
    """The running pyphi code's identity: git sha in a checkout, else version."""
    sha, _dirty = _git_info()
    if sha is not None:
        return f"git:{sha}"
    return f"v:{importlib.metadata.version('pyphi')}"


def result_cache_key(system: Any, kind: str, snapshot: Any) -> str | None:
    """Hex cache key, or ``None`` (do not cache) when the git tree is dirty."""
    _sha, dirty = _git_info()
    if dirty:
        return None
    h = hashlib.blake2b(digest_size=32)
    h.update(system._fingerprint)
    h.update(kind.encode())
    h.update(_config_digest(snapshot))
    h.update(_code_version().encode())
    return h.hexdigest()
```

Note: `result_cache_key` calls `_git_info()` for the dirty check, and `_code_version()` calls it again for the sha; the redundant call is negligible (git is shelled once per, and the result feeds a per-System cached property's consumer, not a hot loop). Keep it simple rather than threading the tuple through.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/cache/test_disk_cache_key.py -q -p no:cacheprovider`
Expected: PASS (5 passed). If `Substrate.from_factored` rejects the relabel, match the construction used in `test/test_fingerprint.py::test_relabeling_collides_and_agrees`.

- [ ] **Step 5: Lint**

Run: `uv run ruff check pyphi/cache/disk.py test/cache/test_disk_cache_key.py && uv run pyright pyphi/cache/disk.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add pyphi/cache/disk.py test/cache/test_disk_cache_key.py
git commit -m "$(cat <<'EOF'
Add result-cache key builder (fingerprint + config + kind + code version)

Keys separate everything that changes a result: the P9.5 system fingerprint,
the result kind (sia vs ces), a digest of result-affecting config, and a
code-version stamp (git sha in a checkout, else the released version). A dirty
git tree returns None so the cache disables itself when the sha no longer
identifies the running code. Relabeled-equivalent systems share a key.

<trailer>
EOF
)"
```

---

### Task 3: Config switch + wire the cache into `System.sia` / `System.ces`

**Files:**
- Modify: `pyphi/conf/infrastructure.py` (add `disk_cache_results`)
- Modify: `pyphi/cache/disk.py` (add `maybe_disk_cached` + the module-level store)
- Modify: `pyphi/system.py` (`sia`, `ces` call the wrapper)
- Test: `test/cache/test_disk_cache_integration.py`

**Interfaces:**
- Consumes: `DiskCache`, `_decode_or_none`, `serialize.dumps`, `result_cache_key` from Task 1–2 (all in `pyphi/cache/disk.py`, which already imports `serialize`); `config.infrastructure.disk_cache_results`; `pyphi.conf._global.snapshot()`.
- Produces: `maybe_disk_cached(system, kind, user_kwargs, compute) -> Any` and module-level `_RESULT_DISK_CACHE = DiskCache("disk.results", "results")`.

`maybe_disk_cached` bypasses (just calls `compute()`) when the cache is off, when the caller passed result-affecting `user_kwargs` (which the key does not capture), or when the tree is dirty.

- [ ] **Step 1: Write the failing integration test**

Create `test/cache/test_disk_cache_integration.py`:

```python
"""End-to-end: disk hits equal recomputation; opt-in; bypasses."""

from __future__ import annotations

import pyphi.cache.disk as disk
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets


def _fresh_cache(tmp_path, monkeypatch):
    import pyphi.constants as constants

    monkeypatch.setattr(constants, "DISK_CACHE_LOCATION", tmp_path)
    disk._RESULT_DISK_CACHE.hits = 0
    disk._RESULT_DISK_CACHE.misses = 0


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023):
        examples.basic_system().sia()
    assert not any(tmp_path.rglob("*")), "cache off must create no files"


def test_sia_disk_hit_equals_recompute(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().sia()
        warm = examples.basic_system().sia()  # second call: disk hit
    assert warm == cold
    assert disk._RESULT_DISK_CACHE.hits >= 1


def test_ces_disk_hit_equals_recompute(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().ces()
        warm = examples.basic_system().ces()
    assert warm == cold


def test_kwargs_bypass_the_cache(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    from pyphi.measures.distribution import resolve_system_measure

    with config.override(**presets.iit4_2023, disk_cache_results=True):
        # passing an explicit measure kwarg must bypass (key can't capture it)
        examples.basic_system().sia(
            system_measure=resolve_system_measure(
                config.formalism.iit.system_phi_measure
            )
        )
    assert not any(tmp_path.rglob("*")), "explicit kwargs must not be cached"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/cache/test_disk_cache_integration.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'pyphi.cache.disk' has no attribute '_RESULT_DISK_CACHE'`.

- [ ] **Step 3: Add the config field**

In `pyphi/conf/infrastructure.py`, beside the other cache toggles (`cache_repertoires`, `cache_potential_purviews` around line 85):

```python
    disk_cache_results: bool = False
```

- [ ] **Step 4: Add `maybe_disk_cached` + the store to `pyphi/cache/disk.py`**

```python
_RESULT_DISK_CACHE = DiskCache("disk.results", "results")


def maybe_disk_cached(system: Any, kind: str, user_kwargs: dict, compute: Any) -> Any:
    """Return a disk-cached result for ``compute()`` when it is safe to.

    Bypasses (just calls ``compute()``) when the cache is disabled, when the
    caller passed result-affecting kwargs the key cannot capture, or when the
    git tree is dirty (``result_cache_key`` returns ``None``).
    """
    from pyphi.conf import _global
    from pyphi.conf import config

    if user_kwargs or not config.infrastructure.disk_cache_results:
        return compute()
    key = result_cache_key(system, kind, _global.snapshot())
    if key is None:
        return compute()
    hit = _RESULT_DISK_CACHE.get(key)
    if hit is not None:
        result = _decode_or_none(hit)
        if result is not None:
            return result
    result = compute()
    _RESULT_DISK_CACHE.put(key, serialize.dumps(result, format="msgpack"))
    return result
```

- [ ] **Step 5: Wire `System.sia` and `System.ces`**

In `pyphi/system.py`, wrap each method body so the existing computation becomes the `compute` thunk. For `sia` (around line 672), keep the body but route the final return through the wrapper:

```python
    def sia(self, **kwargs: Any) -> Any:
        """Return the system irreducibility analysis under the active formalism.
        ...(docstring unchanged)...
        """
        from pyphi.cache.disk import maybe_disk_cached

        def _compute() -> Any:
            from pyphi.conf import config as _config
            from pyphi.formalism import sia as _sia
            from pyphi.measures.distribution import resolve_mechanism_measure
            from pyphi.measures.distribution import resolve_system_measure

            call_kwargs = dict(kwargs)
            if _config.formalism.iit.version != "IIT_3_0":
                call_kwargs.setdefault(
                    "system_measure",
                    resolve_system_measure(_config.formalism.iit.system_phi_measure),
                )
                call_kwargs.setdefault(
                    "specification_measure",
                    resolve_mechanism_measure(
                        _config.formalism.iit.specification_measure
                    ),
                )
            return _sia(self, **call_kwargs)

        return maybe_disk_cached(self, "sia", kwargs, _compute)
```

Apply the same shape to `ces` (around line 696): move its existing body into a nested `_compute()` (keeping the IIT 3.0 / 4.0 branch and the `setdefault`s, using a local `call_kwargs = dict(kwargs)` instead of mutating `kwargs`), then `return maybe_disk_cached(self, "ces", kwargs, _compute)`.

Note: `kwargs` passed to `maybe_disk_cached` is the caller's original (empty in the cached path); the resolved measures live in `call_kwargs` inside `_compute`, and they are config-derived, so the config digest already covers them.

- [ ] **Step 6: Run the integration tests to verify they pass**

Run: `uv run pytest test/cache/test_disk_cache_integration.py -q -p no:cacheprovider`
Expected: PASS (4 passed).

- [ ] **Step 7: Lint + full suite**

Run: `uv run ruff check pyphi/cache/disk.py pyphi/system.py pyphi/conf/infrastructure.py && uv run pyright pyphi/cache/disk.py pyphi/system.py`
Run (complete check, no path argument): `uv run --all-extras pytest`
Expected: ruff/pyright clean; the suite stays green (default-off changes no result), plus the new tests.

- [ ] **Step 8: Commit**

```bash
git add pyphi/cache/disk.py pyphi/system.py pyphi/conf/infrastructure.py test/cache/test_disk_cache_integration.py
git commit -m "$(cat <<'EOF'
Persist SIA/CES results to the disk cache (opt-in)

System.sia/ces route through maybe_disk_cached: when disk_cache_results is on
(default off), the result is loaded from / stored to the DiskCache keyed on the
system fingerprint + config + kind + code version. Bypassed when the caller
passes result-affecting kwargs (the key cannot capture them) or the tree is
dirty. A disk hit equals recomputation.

<trailer>
EOF
)"
```

---

### Task 4: Changelog + roadmap

**Files:**
- Create: `changelog.d/disk-result-cache.feature.md`
- Modify: `ROADMAP.md` (flip the N4 dashboard row to landed)

- [ ] **Step 1: Add the changelog fragment**

Create `changelog.d/disk-result-cache.feature.md` (no roadmap markers):

```markdown
Added an opt-in disk-backed cache for top-level results (`config.disk_cache_results`, default off). When enabled, `System.sia()` and `System.ces()` persist results under `__pyphi_cache__/results/` keyed on the system's mathematical identity, the result-affecting configuration, and the running pyphi version, so repeated computations (notebook re-runs, paper reproductions) load instead of recompute. The cache disables itself on a dirty git working tree.
```

- [ ] **Step 2: Flip the ROADMAP N4 row**

Re-read first: `grep -n "N4 disk-backed result cache" ROADMAP.md`. Change the dashboard row's status from `⬜ open` to `✅ landed` and tighten the one-liner to past tense, e.g.:

```
| N4 disk-backed result cache | ✅ landed | 6 | Opt-in disk cache (`disk_cache_results`, default off) persisting SIA/CES under `__pyphi_cache__/results/`, keyed on the P9.5 fingerprint + result-affecting config + code version; bypassed on a dirty tree. Hit equals recompute. |
```

- [ ] **Step 3: Verify + full suite once more**

Run: `uv run --all-extras pytest`
Expected: green (2790-baseline + the new disk-cache tests).

- [ ] **Step 4: Commit**

```bash
git add changelog.d/disk-result-cache.feature.md ROADMAP.md
git commit -m "$(cat <<'EOF'
Record the disk result cache as landed

<trailer>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Cache key (`_fingerprint` + config digest + code-version, dirty guard) → Task 2; the spec's "result kind" gap is closed by adding `kind` to the key (Task 2) since SIA and CES share a system+config. ✓
- `DiskCache` store, atomic write, corruption→miss (`_decode_or_none`) → Task 1. ✓
- Config digest = result-affecting subset only → Task 2 `_config_digest`. ✓
- Hook at the SIA/CES chokepoint, opt-in, dirty bypass → Task 3 (`System.sia`/`ces` + `maybe_disk_cached`). The spec said `queries.sia`/`queries.ces`; there is no `queries.ces`, so both hook at the `System` methods — the same single user-facing chokepoint, intent preserved. ✓
- Config field `disk_cache_results=False`, `__pyphi_cache__/results/`, registry `clear` → Task 1 (registration) + Task 3 (field). ✓
- Exact-parity (SIA+CES), key sensitivity, dirty bypass, corruption miss, opt-in-off → Tasks 1–3 tests. ✓
- Out of scope (AC account, eviction/LRU, mechanism-level) → not implemented. ✓
- Roadmap bookkeeping → Task 4. ✓

**Refinement vs spec:** the key gains a `kind` component (necessary: SIA and CES of one system+config must not collide), and explicit result-affecting kwargs bypass the cache (the key can't capture them) — both tighten correctness without changing the design.

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The only verify-against-live note is "match `test_fingerprint`'s relabel construction if `from_factored` rejects it" (Task 2 Step 4), which is a fallback instruction, not a placeholder.

**Type consistency:** `result_cache_key(system, kind, snapshot) -> str | None`, `_decode_or_none(data) -> Any | None`, inline `serialize.dumps(result, format="msgpack")` at the put site, `maybe_disk_cached(system, kind, user_kwargs, compute)`, `_RESULT_DISK_CACHE`, and `DiskCache(name, subdir)` are used consistently across Tasks 1–3. `DiskCache.info()` returns `_CacheInfo(hits, misses, size)` matching the `CachePolicy` protocol.
