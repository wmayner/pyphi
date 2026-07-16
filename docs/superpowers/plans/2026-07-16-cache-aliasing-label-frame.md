# Cache/Aliasing Safety and the Serialization Label Frame — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cached and hashed arrays become read-only (kernel repertoires, `max_entropy_distribution`, FactoredTPM factors), the disk result cache serves every hit in the requester's node labels via a document-level label frame, `cache_repertoires` actually gates the kernel cache, and the dead per-instance cache machinery is deleted.

**Architecture:** Freeze-at-the-source with the existing `utils.np_immutable` for every cached/hashed ndarray; a `contextvars`-based label frame in `pyphi/serialize/convert.py` so each serialized document carries one label table (claimed by the first labeled object, inherited on decode, overridable at `loads()`); the disk cache stamps hits with the requesting system's labels; `_memoize` gains `store=config.infrastructure.cache_repertoires`, mirroring the `cache_potential_purviews` gate in `substrate.py`.

**Tech Stack:** Python 3.13, numpy, msgspec, contextvars, pytest. All commands run through `uv run`.

**Spec:** `docs/superpowers/specs/2026-07-16-cache-aliasing-label-frame-design.md`

## Global Constraints

- Work in the worktree `.claude/worktrees/cache-aliasing-safety` (branch `cache-aliasing-safety`, based on `43bf02d5`). All paths below are relative to the worktree root.
- Every commit message ends with the two trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe`
- **Never `git commit --no-verify`.** If a commit's output shows only hook lines and `git status` still shows `MM` files, the pre-commit formatter modified files and the commit did NOT land: re-stage and re-commit.
- Test-suite imports from the test-suite root use `from test.conftest import ...` (ruff TID252 bans `from ..conftest import`). Use `next(iter(x))`, never `list(x)[0]` (RUF015).
- Never pipe a test run through `tail`/`head`/`grep`; redirect to a file (`> log 2>&1`) and read the summary line from the file.
- Docstrings: NumPy style, final-state impersonal voice, no migration narrative, no planning-artifact references (no wave/finding/review mentions in code, docstrings, or changelog).
- `FORMAT_VERSION` stays 1. All schema/envelope changes must be additive (new fields optional with defaults).
- Pyright diagnostics from the IDE inside `.claude/worktrees/` claiming worktree/main module mismatches are nested-repo artifacts; the pre-commit pyright hook is the authority.

---

### Task 0: Worktree environment setup

**Files:** none (environment only)

- [ ] **Step 1: Create the venv and install dev extras**

```bash
cd /Users/will/projects/pyphi/.claude/worktrees/cache-aliasing-safety
uv venv
WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"
env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray]" pot
```

- [ ] **Step 2: Precompile `pot` once** (a fresh venv's uncompiled `ot` package raises SyntaxWarning, which pytest's warning filters escalate into a collection-killing error)

```bash
uv run python -W ignore -c "import ot"
```

- [ ] **Step 3: Sanity-check collection**

```bash
uv run pytest test/cache -q > /tmp/wt_sanity.log 2>&1
```

Read `/tmp/wt_sanity.log`; expect all tests passing.

---

### Task 1: Kernel cache and `max_entropy_distribution` return read-only arrays

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py:61-68` (the `_memoize` wrapper)
- Modify: `pyphi/distribution.py:261-262` (the `max_entropy_distribution` return)
- Test: `test/core/test_core_repertoire_algebra.py` (append)

**Interfaces:**
- Consumes: `pyphi.utils.np_immutable(a) -> np.ndarray` (freezes in place, returns `a`; already imported in `repertoire_algebra.py` as `_utils`).
- Produces: every array returned by the seven `@_memoize` functions and by `max_entropy_distribution` has `flags.writeable == False`. Task 2 edits the same wrapper again (adds `store=`).

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_core_repertoire_algebra.py` (add any missing imports at the top of the file: `import numpy as np`, `import pytest`, `from pyphi import examples`):

```python
def test_cached_repertoire_is_read_only():
    """A caller must not be able to poison the kernel cache in place."""
    system = examples.basic_system()
    r = system.cause_repertoire((0,), (1,))
    with pytest.raises(ValueError, match="read-only"):
        r[...] = 99.0
    again = examples.basic_system().cause_repertoire((0,), (1,))
    assert np.array_equal(again, r)


def test_effect_repertoire_is_read_only():
    system = examples.basic_system()
    r = system.effect_repertoire((0,), (1,))
    with pytest.raises(ValueError, match="read-only"):
        r[...] = 99.0


def test_max_entropy_distribution_is_read_only():
    from pyphi.distribution import max_entropy_distribution

    d = max_entropy_distribution((0, 1, 2), (1,))
    with pytest.raises(ValueError, match="read-only"):
        d[...] = 99.0
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest test/core/test_core_repertoire_algebra.py -k "read_only" -v > /tmp/t1_red.log 2>&1
```

Read the log. Expected: 3 FAILED — `DID NOT RAISE <class 'ValueError'>`.

- [ ] **Step 3: Implement.** In `pyphi/core/repertoire_algebra.py`, change the wrapper's return (line 66):

```python
        return cache.get_or_compute(
            fp, key_args, lambda: _utils.np_immutable(fn(cs, *args))
        )
```

Append to the `_memoize` docstring's first paragraph block (after the sentence about memory limits): `Returned arrays are read-only; callers that need a mutable copy must copy explicitly.`

In `pyphi/distribution.py`, change the `max_entropy_distribution` return (line 262):

```python
    return np_immutable(distribution / distribution.size)
```

Add the import if absent (check the file's existing imports first): `from pyphi.utils import np_immutable`. Add a sentence to its `Returns` docstring entry: `The array is cached and read-only.`

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest test/core/test_core_repertoire_algebra.py -k "read_only" -v > /tmp/t1_green.log 2>&1
```

Read the log. Expected: 3 passed.

- [ ] **Step 5: Run the fast lane to surface latent in-place mutators.** Freezing may break internal code that mutates a cached array. Run:

```bash
uv run pytest test/ -m "not slow" -q > /tmp/t1_fast.log 2>&1
```

Read the summary line and any failures. For each failure whose traceback shows an in-place operation (`+=`, `*=`, `/=`, `[...] =`, `np.___(..., out=...)`) on an array that came from a memoized function or `max_entropy_distribution`: insert an explicit `.copy()` at that mutation site (e.g. `rep = rep.copy()` immediately before the in-place op) — mutating a cached operand was a latent bug at that site. Re-run the fast lane until green. If a failure is NOT an in-place-mutation symptom, stop and investigate before changing anything. If any `.copy()` landed on a repertoire hot path (inside `pyphi/core/repertoire_algebra.py` or a per-partition loop), sanity-check wall time with `just bench` before committing.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Return read-only arrays from the repertoire caches

Kernel-memoized repertoires and max_entropy_distribution are cached and
shared across equivalent systems; freezing them makes caller mutation
raise instead of silently corrupting later computations.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

Verify with `git status` (clean) and `git log --oneline -1`.

---

### Task 2: Wire `cache_repertoires` into `_memoize`; delete the dead cache machinery

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (wrapper + module import)
- Modify: `pyphi/cache/__init__.py` (delete `DictCache`, `validate_parent_cache`, `method`; docstrings)
- Modify: `pyphi/cache/policy.py:35` (docstring mention of `DictCache`)
- Modify: `test/cache/test_cache.py` (rewrite the option test; delete dead-path tests)
- Modify: `test/cache/test_cache_registry.py` (delete the two `DictCache` registration tests)

**Interfaces:**
- Consumes: Task 1's wrapper (this task adds `store=` to the same `get_or_compute` call); `ContentCache.get_or_compute(..., store: bool = True)` (existing); `pyphi.core.repertoire_algebra._kernel_caches: dict[str, ContentCache]` and `clear_caches()` (existing).
- Produces: `_memoize` reads `config.infrastructure.cache_repertoires` per call; `pyphi.cache` no longer exposes `DictCache`, `method`, or `validate_parent_cache`.

- [ ] **Step 1: Write the failing test.** In `test/cache/test_cache.py`, replace `test_cache_repertoires_config_option` (lines 80-97) and the `factory()` helper above it (lines 54-77) with:

```python
def test_cache_repertoires_config_option():
    """The option gates whether the kernel cache stores repertoires."""
    from pyphi import examples
    from pyphi.core import repertoire_algebra

    repertoire_algebra.clear_caches()
    try:
        with config.override(cache_repertoires=False):
            examples.basic_system().cause_repertoire((0,), (1,))
            sizes = {n: c.size for n, c in repertoire_algebra._kernel_caches.items()}
            assert all(size == 0 for size in sizes.values()), sizes
        with config.override(cache_repertoires=True):
            examples.basic_system().cause_repertoire((0,), (1,))
            assert any(c.size > 0 for c in repertoire_algebra._kernel_caches.values())
    finally:
        repertoire_algebra.clear_caches()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest test/cache/test_cache.py::test_cache_repertoires_config_option -v > /tmp/t2_red.log 2>&1
```

Read the log. Expected: FAILED on the `all(size == 0 ...)` assertion (the kernel currently stores regardless of the option).

- [ ] **Step 3: Implement the gate.** In `pyphi/core/repertoire_algebra.py`, add a module-level import (alphabetical placement with the other `pyphi` imports; `pyphi/core/tpm/factored.py` already imports config at module level, so this is safe):

```python
from pyphi.conf import config
```

Change the wrapper (final form, including Task 1's freeze):

```python
    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        fp = cs._fingerprint
        cache.observe(cs, fp)
        key_args = (cs._resolved_background_conditioning(), *args)
        return cache.get_or_compute(
            fp,
            key_args,
            lambda: _utils.np_immutable(fn(cs, *args)),
            store=config.infrastructure.cache_repertoires,
        )
```

Extend the `_memoize` docstring sentence about memory limits to also name the option, e.g.: `Stops inserting new entries when ``cache_repertoires`` is false or when ``cache_utils.memory_full()`` reports process memory above ``maximum_cache_memory_percentage`` — already-computed values are still returned, just not cached.`

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest test/cache/test_cache.py::test_cache_repertoires_config_option -v > /tmp/t2_green.log 2>&1
```

Read the log. Expected: PASS.

- [ ] **Step 5: Delete the dead machinery.** In `pyphi/cache/__init__.py`:

1. Delete the `DictCache` class (lines 161-239), `validate_parent_cache` (lines 242-246), and `method` (lines 249-292) in their entirety.
2. Delete `from functools import wraps` (only `method` used it; `update_wrapper` stays — `cache()` uses it).
3. In the module docstring, rewrite the sentence `The ``cache`` decorator and ``DictCache`` below are oriented to process-isolated parallelism (each worker process owns its caches) and are not shared across threads by the current schedulers; their counters carry the same best-effort caveat under free-threading.` to drop `DictCache`: `The ``cache`` decorator below is oriented to process-isolated parallelism (each worker process owns its caches) and is not shared across threads by the current schedulers; its counters carry the same best-effort caveat under free-threading.`

In `pyphi/cache/policy.py`, update the `_DictCacheAdapter` docstring line `Used by the module-level ``@cache(...)`` decorator and by ``DictCache``` to name live users only: `Used by the module-level ``@cache(...)`` decorator and by ``ContentCache``.`

In `test/cache/test_cache.py`, delete: `test_cache` (the `DictCache` unit test), the module-level `SomeObject` class and `test_cache_decorator`, and `test_cache_key_generation`. Remove `from pyphi import cache` if nothing left uses it (the purview-cache tests use `config` and `Direction` only — check before removing).

In `test/cache/test_cache_registry.py`, delete `test_dict_cache_with_name_registers` and `test_dict_cache_without_name_does_not_register` plus their section banner comment (`DictCache opt-in registry registration`).

- [ ] **Step 6: Run the cache suite and a pyright spot-check**

```bash
uv run pytest test/cache -q > /tmp/t2_cache.log 2>&1
uv run pyright pyphi/cache > /tmp/t2_pyright.log 2>&1
```

Read both logs. Expected: cache suite green; pyright 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Honor cache_repertoires in the kernel cache; drop dead cache machinery

The kernel memoizer now passes the option as ContentCache's store flag,
matching the cache_potential_purviews gate. DictCache, cache.method, and
validate_parent_cache had no production users since the content-cache
migration; the option's only reader was the dead decorator.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 3: FactoredTPM owns its storage

**Files:**
- Modify: `pyphi/core/tpm/factored.py:87-101` (`__init__`), plus a new module-level helper
- Test: Create `test/core/test_factored_tpm_ownership.py`

**Interfaces:**
- Consumes: `FactoredTPM.__init__(factors, state_space=None, backend=None, node_labels=None)`; `Substrate(marginals=[...])`; `Substrate._fingerprint`.
- Produces: `_own_factor(f) -> NDArray[np.float64]` (module-private in `factored.py`); every stored factor is read-only and shares no writable memory with caller input.

- [ ] **Step 1: Write the failing tests** — create `test/core/test_factored_tpm_ownership.py`:

```python
"""FactoredTPM storage ownership: stored factors are read-only and immune
to post-construction mutation of the caller's arrays."""

import numpy as np
import pytest

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.substrate import Substrate


def _uniform_factors():
    """Two binary nodes, each factor shape (2, 2, 2), uniform outputs."""
    return np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)


def test_writable_input_is_copied_and_frozen():
    f0, f1 = _uniform_factors()
    tpm = FactoredTPM([f0, f1])
    before = tpm.factor(0).copy()
    h = hash(tpm)
    f0[...] = 0.9  # caller mutates their own array after construction
    assert np.array_equal(tpm.factor(0), before)
    assert hash(tpm) == h
    assert not tpm.factor(0).flags.writeable
    assert f0.flags.writeable, "the caller's own array must stay writable"


def test_non_float64_input_is_frozen_and_detached():
    f0 = np.zeros((2, 2, 2), dtype=int)
    f0[..., 0] = 1
    f1 = np.zeros((2, 2, 2), dtype=int)
    f1[..., 1] = 1
    tpm = FactoredTPM([f0, f1])
    f0[...] = 7
    assert not tpm.factor(0).flags.writeable
    assert tpm.factor(0).max() <= 1.0


def test_read_only_input_is_stored_without_copy():
    f0, f1 = _uniform_factors()
    f0.flags.writeable = False
    f1.flags.writeable = False
    tpm = FactoredTPM([f0, f1])
    assert tpm.factor(0) is f0


def test_xarray_backend_factors_are_read_only():
    pytest.importorskip("xarray")
    f0, f1 = _uniform_factors()
    tpm = FactoredTPM([f0, f1], backend="xarray")
    assert not tpm.factor(0).flags.writeable
    f0[...] = 0.9
    assert float(tpm.factor(0).max()) == 0.5


def test_substrate_fingerprint_immune_to_caller_mutation():
    f0, f1 = _uniform_factors()
    sub = Substrate(marginals=[f0, f1])
    fp = sub._fingerprint
    f0[...] = 0.9
    pristine = Substrate(marginals=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)])
    assert sub._fingerprint == fp == pristine._fingerprint
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest test/core/test_factored_tpm_ownership.py -v > /tmp/t3_red.log 2>&1
```

Read the log. Expected: `test_read_only_input_is_stored_without_copy` may already pass (asarray keeps the object); the other four FAIL (mutation leaks through / factors writable).

- [ ] **Step 3: Implement.** In `pyphi/core/tpm/factored.py`, add above the `FactoredTPM` class:

```python
def _own_factor(f: ArrayLike) -> NDArray[np.float64]:
    """Return a read-only float64 array of ``f`` sharing no writable memory
    with the caller's input.

    A writable input array is copied before freezing, so the caller's own
    array is never modified; a read-only input is stored as-is; a fresh
    dtype conversion is frozen in place without a further copy.
    """
    a = np.asarray(f, dtype=np.float64)
    if a.flags.writeable:
        if isinstance(f, np.ndarray) and np.may_share_memory(a, f):
            a = a.copy()
        a.flags.writeable = False
    return a
```

In `__init__` (line 94), change:

```python
        factor_arrays = tuple(np.asarray(f, dtype=np.float64) for f in factors)
```

to:

```python
        factor_arrays = tuple(_own_factor(f) for f in factors)
```

Add to the `FactoredTPM` class docstring: `Stored factors are read-only and independent of the arrays passed in: a hashed value type's contents cannot change after construction.`

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest test/core/test_factored_tpm_ownership.py -v > /tmp/t3_green.log 2>&1
```

Read the log. Expected: 5 passed.

- [ ] **Step 5: Run the fast lane** (internal code may mutate factor arrays it obtained from a FactoredTPM):

```bash
uv run pytest test/ -m "not slow" -q > /tmp/t3_fast.log 2>&1
```

Read the summary. Fix any in-place-mutation failures with a `.copy()` at the mutation site, same rule as Task 1 Step 5. Re-run until green.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Make FactoredTPM own its factor storage

Factors are copied when storing would alias the caller's writable array,
and frozen. A hashed, fingerprinted value type's bytes can no longer be
changed after construction, which also protects the cached substrate
fingerprint.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 4: The document-level label frame

**Files:**
- Modify: `pyphi/serialize/convert.py` (contextvars, `_enc_labels`/`_dec_labels`, `encode_document`/`decode_document`, 18 site swaps)
- Modify: `pyphi/serialize/__init__.py` (`_Document`, `dumps`, `loads`, module docstring)
- Test: Create `test/serialize/test_serialize_label_frame.py`

**Interfaces:**
- Consumes: `convert.to_schema` / `convert.from_schema`; `schema.NodeLabelsSchema`; `NodeLabels(labels, node_indices)` from `pyphi.labels`; the RIA tie API (`ria._partition_ties = (a, b)`).
- Produces:
  - `convert.encode_document(obj) -> tuple[payload, frame]` where `frame` is `schema.NodeLabelsSchema | None`.
  - `convert.decode_document(payload, frame, node_labels=None) -> Any` (`node_labels`: domain `NodeLabels` override).
  - `serialize.loads(data, *, format="json", node_labels=None)` — Task 5 relies on this exact signature.
  - `_Document` gains `node_labels: schema.NodeLabelsSchema | None = None`.

- [ ] **Step 1: Write the failing tests** — create `test/serialize/test_serialize_label_frame.py`:

```python
"""Document-level node-labels frame: one label table per document, inherited
on decode, with per-object overrides for heterogeneous documents."""

import msgspec
import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.labels import NodeLabels
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.serialize import convert

FORMATS = ["json", "msgpack"]

LABELS = NodeLabels(("A", "B"), (0, 1))
OTHER = NodeLabels(("X", "Y"), (0, 1))


def make_labeled_ria(node_labels=LABELS, phi=0.3):
    return RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=Direction.CAUSE,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
        node_labels=node_labels,
    )


def test_document_claims_one_frame():
    data = serialize.dumps(make_labeled_ria())
    doc = msgspec.json.decode(data)
    assert doc["node_labels"] is not None
    assert doc["payload"]["node_labels"] is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_labels_round_trip_via_frame(fmt):
    obj = make_labeled_ria()
    restored = serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)
    assert restored == obj
    assert tuple(restored.node_labels) == tuple(LABELS)


def test_unlabeled_object_round_trips_without_frame():
    obj = make_labeled_ria(node_labels=None)
    data = serialize.dumps(obj)
    doc = msgspec.json.decode(data)
    assert doc["node_labels"] is None
    restored = serialize.loads(data)
    assert restored.node_labels is None


def test_old_format_per_object_labels_still_load():
    # Documents written before the frame carried labels on every struct and
    # no envelope frame; encoding outside a document context reproduces that
    # layout exactly.
    payload = convert.to_schema(make_labeled_ria())
    assert payload.node_labels is not None  # inline, old-style
    doc = serialize._Document(format_version=1, payload=payload)
    restored = serialize.loads(msgspec.json.encode(doc))
    assert tuple(restored.node_labels) == tuple(LABELS)


def test_heterogeneous_labels_survive_as_overrides():
    a = make_labeled_ria(node_labels=LABELS)
    b = make_labeled_ria(node_labels=OTHER)
    tied = (a, b)
    a._partition_ties = tied
    b._partition_ties = tied
    restored = serialize.loads(serialize.dumps(a))
    assert tuple(restored.node_labels) == tuple(LABELS)
    peer = next(t for t in restored._partition_ties if t is not restored)
    assert tuple(peer.node_labels) == tuple(OTHER)


def test_loads_node_labels_override_wins():
    data = serialize.dumps(make_labeled_ria())
    restored = serialize.loads(data, node_labels=OTHER)
    assert tuple(restored.node_labels) == ("X", "Y")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest test/serialize/test_serialize_label_frame.py -v > /tmp/t4_red.log 2>&1
```

Read the log. Expected: `test_document_claims_one_frame` FAILS with `KeyError: 'node_labels'` (no envelope field yet); `test_loads_node_labels_override_wins` FAILS with `TypeError` (unexpected keyword); others fail or pass incidentally.

- [ ] **Step 3: Implement the frame in `pyphi/serialize/convert.py`.** Add `import contextvars` to the imports. Below `_enc_optional`/`_dec_optional` (around line 50), add:

```python
# Document label frame. dumps()/loads() establish these contexts; encoders
# and decoders resolve per-object labels against them. Outside a document
# context (a direct to_schema/from_schema call), labels stay per-object.
_ENC_FRAME: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_ENC_FRAME", default=None
)
_DEC_FRAME: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_DEC_FRAME", default=None
)


def _enc_labels(labels: Any) -> Any:
    """Encode a ``node_labels`` attribute against the document frame.

    The first labeled object claims the frame and writes ``None`` into its
    own struct; labels equal to the frame also write ``None``; labels that
    differ are written per-object.
    """
    if labels is None:
        return None
    encoded = to_schema(labels)
    holder = _ENC_FRAME.get()
    if holder is None:
        return encoded
    if holder[0] is None:
        holder[0] = encoded
        return None
    if encoded == holder[0]:
        return None
    return encoded


def _dec_labels(stored: Any) -> Any:
    """Resolve labels: the object's own stored labels, else the frame."""
    if stored is not None:
        return from_schema(stored)
    return _DEC_FRAME.get()


def encode_document(obj: Any) -> tuple[Any, Any]:
    """Encode ``obj`` to a payload struct plus the claimed label frame."""
    holder: list = [None]
    token = _ENC_FRAME.set(holder)
    try:
        payload = to_schema(obj)
    finally:
        _ENC_FRAME.reset(token)
    return payload, holder[0]


def decode_document(payload: Any, frame: Any, node_labels: Any = None) -> Any:
    """Decode ``payload`` under a document label frame.

    ``frame`` is the document's stored ``NodeLabelsSchema`` (or ``None``);
    ``node_labels`` is a caller-supplied domain ``NodeLabels`` that
    replaces it.
    """
    resolved = node_labels
    if resolved is None and frame is not None:
        resolved = from_schema(frame)
    token = _DEC_FRAME.set(resolved)
    try:
        return from_schema(payload)
    finally:
        _DEC_FRAME.reset(token)
```

Then swap the label sites. Every encoder occurrence of `_enc_optional(<x>.node_labels)` becomes `_enc_labels(<x>.node_labels)`, and every decoder occurrence of `_dec_optional(<s>.node_labels)` becomes `_dec_labels(<s>.node_labels)`, in these nine registration functions (current line numbers): `_register_edge_cut` (226/231), `_register_complete_edge_cut` (240/243), `_register_directed_set_partition` (254/260), the RIA pair (296/322), the IIT 3.0 SIA pair (542/561), the IIT 4.0 SIA pair (664/690), `_register_substrate` (860/871), the AC RIA pair (935/952), the AC SIA pair (1036/1059). Do NOT touch the `_opt_tuple(p.node_labels)` sites at 1131/1140 — that is FactoredTPM's plain-string-tuple labels, a different type outside the frame.

- [ ] **Step 4: Implement the envelope in `pyphi/serialize/__init__.py`.**

```python
class _Document(msgspec.Struct, frozen=True):
    format_version: int
    # A φ value serialized on its own is a native float; every other domain
    # object serializes to a tagged Struct in ``schema.Schema``.
    payload: schema.Schema | float
    # The document's node-labels frame: claimed once by the first labeled
    # payload object, inherited on decode by every object that carries none.
    node_labels: schema.NodeLabelsSchema | None = None


def dumps(obj: Any, *, format: str = "json") -> bytes:
    payload, frame = convert.encode_document(obj)
    doc = _Document(format_version=FORMAT_VERSION, payload=payload, node_labels=frame)
    return _encoder(format)(doc)


def loads(data: bytes, *, format: str = "json", node_labels: Any = None) -> Any:
    """Deserialize a document produced by :func:`dumps`.

    Parameters
    ----------
    data : bytes
        The serialized document.
    format : {"json", "msgpack"}, optional
        Wire format. Defaults to ``"json"``.
    node_labels : NodeLabels, optional
        Replacement label frame. If given, it is used in place of the
        document's stored frame; objects carrying their own per-object
        labels keep them.
    """
    doc = _decode(data, format)
    if doc.format_version > FORMAT_VERSION:
        raise ValueError(
            f"cannot load format_version {doc.format_version}: this version of "
            f"PyPhi reads format_version {FORMAT_VERSION} or lower"
        )
    return convert.decode_document(doc.payload, doc.node_labels, node_labels=node_labels)
```

Extend the module docstring sentence `The document carries a single top-level ``format_version``.` to: `The document carries a single top-level ``format_version`` and a node-labels frame written once per document.`

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest test/serialize/test_serialize_label_frame.py -v > /tmp/t4_green.log 2>&1
```

Read the log. Expected: 7 passed (6 test functions, one parametrized ×2).

- [ ] **Step 6: Run the whole serialize suite** (the existing round-trip tests are the regression net for the site swaps):

```bash
uv run pytest test/serialize test/cache -q > /tmp/t4_suite.log 2>&1
```

Read the summary. Expected: all green. A label-equality failure here means a decoder site was missed in Step 3 — the object decoded with `None` labels; find it by the failing family and swap its `_dec_optional` site.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Serialize node labels once per document via an envelope frame

The first labeled object claims the document frame; equal labels in
nested objects are omitted and inherited on decode; differing labels are
written per-object. loads() accepts a replacement frame. Old documents
with per-object labels decode unchanged; the format version is unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 5: Disk-cache hits carry the requester's labels

**Files:**
- Modify: `pyphi/cache/disk.py:30-40` (`_decode_or_none`), `:143-147` (`maybe_disk_cached` hit path)
- Test: `test/cache/test_disk_cache_integration.py` (append)

**Interfaces:**
- Consumes: `serialize.loads(data, format="msgpack", node_labels=...)` from Task 4; `System.node_labels` (a domain `NodeLabels`); `Substrate.from_factored(factored, cm=..., node_labels=...)`; the test module's existing `_fresh_cache(tmp_path, monkeypatch)` helper and `examples.basic_system()/basic_substrate()/basic_state()`.
- Produces: every disk hit is decoded with the requesting system's labels.

- [ ] **Step 1: Write the failing tests** — append to `test/cache/test_disk_cache_integration.py` (the file already imports `examples`, `disk`, `config`, `presets`):

```python
def test_sia_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    """The key is label-free by design, so a mathematically identical but
    differently-labeled system hits; the result must carry the requester's
    labels, not the computing system's."""
    from pyphi import System
    from pyphi.substrate import Substrate

    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        cold = examples.basic_system().sia()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).sia()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    assert warm.phi == cold.phi
    assert tuple(warm.node_labels) == ("X", "Y", "Z")
    assert warm.cause is None or tuple(warm.cause.node_labels) == ("X", "Y", "Z")


def test_ces_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    from pyphi import System
    from pyphi.substrate import Substrate

    _fresh_cache(tmp_path, monkeypatch)
    with config.override(**presets.iit4_2023, disk_cache_results=True):
        examples.basic_system().ces()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).ces()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    d = next(iter(warm.distinctions))
    assert tuple(d.cause.node_labels) == ("X", "Y", "Z")


def test_iit3_sia_hit_carries_the_requesters_labels(tmp_path, monkeypatch):
    from pyphi import System
    from pyphi.substrate import Substrate
    from test.conftest import IIT_3_CONFIG

    _fresh_cache(tmp_path, monkeypatch)
    with IIT_3_CONFIG, config.override(disk_cache_results=True):
        cold = examples.basic_system().sia()
        sub = examples.basic_substrate()
        twin_sub = Substrate.from_factored(
            sub.factored_tpm, cm=sub.cm, node_labels=("X", "Y", "Z")
        )
        warm = System(twin_sub, examples.basic_state()).sia()
    assert disk._RESULT_DISK_CACHE.hits >= 1
    assert warm.phi == cold.phi
    assert tuple(warm.node_labels) == ("X", "Y", "Z")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest test/cache/test_disk_cache_integration.py -k "requesters_labels" -v > /tmp/t5_red.log 2>&1
```

Read the log. Expected: 3 FAILED on the label assertions (hits currently return the computing system's labels).

- [ ] **Step 3: Implement.** In `pyphi/cache/disk.py`:

```python
def _decode_or_none(data: bytes, node_labels: Any = None) -> Any | None:
    """Deserialize a stored result; ``None`` on any error (a cache miss).

    Staleness across code or config changes is handled entirely by the cache
    key (it folds in a code-version component), so there is no in-file version
    tag; this only tolerates a corrupt/truncated file. ``node_labels``
    replaces the stored label frame, so a hit is decoded in the requesting
    system's labels.
    """
    try:
        return serialize.loads(data, format="msgpack", node_labels=node_labels)
    except Exception:  # any decode failure is a cache miss, not an error
        return None
```

And in `maybe_disk_cached`, change the hit path:

```python
    hit = _RESULT_DISK_CACHE.get(key)
    if hit is not None:
        result = _decode_or_none(hit, node_labels=system.node_labels)
        if result is not None:
            return result
```

Also update the `maybe_disk_cached` docstring to note: `A hit is decoded with the requesting system's node labels (the key is label-free, so an equivalent system with different labels may have produced the entry).`

- [ ] **Step 4: Run the tests**

```bash
uv run pytest test/cache/test_disk_cache_integration.py test/cache/test_disk_cache.py -v > /tmp/t5_green.log 2>&1
```

Read the log. Expected: all pass, including the pre-existing integration tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Decode disk-cache hits with the requesting system's labels

The result-cache key is label-free by design, so an entry may have been
computed by an equivalent system labeled differently; stamping the
requester's label frame at decode time makes every hit come back in the
caller's own labels.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

### Task 6: Changelog, surface checks, full-suite gate

**Files:**
- Create: `changelog.d/cache-aliasing-safety.fix.md`
- Create: `changelog.d/serialize-label-frame.change.md`
- Create: `changelog.d/remove-dead-cache-machinery.refactor.md`

- [ ] **Step 1: Write the changelog fragments**

`changelog.d/cache-aliasing-safety.fix.md`:

```markdown
Fixed cache and aliasing safety holes: arrays returned by the kernel
repertoire cache and `max_entropy_distribution()` are now read-only, so
caller mutation raises instead of silently corrupting every later
computation on equivalent systems; `FactoredTPM` copies and freezes its
factors at construction, so mutating the arrays passed in can no longer
change a hashed value type or stale the substrate fingerprint; disk
result-cache hits are decoded with the requesting system's node labels
instead of the computing system's; and `cache_repertoires = false` now
actually disables kernel repertoire caching.
```

`changelog.d/serialize-label-frame.change.md`:

```markdown
Serialized documents now carry node labels once, on the document envelope,
instead of duplicating them into every nested object; nested objects
inherit the frame on decode, and objects whose labels differ from the
frame keep per-object labels. `pyphi.serialize.loads()` accepts a
`node_labels` argument that replaces the stored frame on load. Documents
written by earlier versions load unchanged.
```

`changelog.d/remove-dead-cache-machinery.refactor.md`:

```markdown
Removed `pyphi.cache.DictCache`, `pyphi.cache.method`, and
`pyphi.cache.validate_parent_cache`, which had no users since repertoire
memoization moved to the content-addressed kernel cache.
```

- [ ] **Step 2: MCP/docs surface check.** Run:

```bash
grep -rn "DictCache\|cache\.method\|cache_repertoires" pyphi/mcp/content/ docs/*.md docs/**/*.rst 2>/dev/null | grep -v superpowers
```

The `configuration.md` line for `cache_repertoires` ("Memoize repertoire computations") is already accurate once the option works — leave it. If any other hit describes the deleted API or per-object label serialization, update that text to the new behavior; otherwise nothing to do.

- [ ] **Step 3: Full pathless suite in the worktree**

```bash
uv run pytest -q > /tmp/t6_full.log 2>&1
```

Read the summary line from the log (never trust the exit code). Expected: 0 failures. Doctests in `pyphi/` run under this invocation — if a doctest shows a repertoire being mutated or prints label-frame-affected output, fix the doctest text to match the new behavior.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "Add changelog fragments for cache safety and the label frame

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
```

---

## Completion / merge checklist (finishing-a-development-branch)

1. Full pathless `uv run pytest` green in the worktree (Task 6 Step 3).
2. Check the main-tree tip before merging — concurrent sessions may have advanced it; rebase or re-verify if it moved past `43bf02d5`.
3. Merge `--no-ff` into `main` with a summary message; run the full pathless suite in the main tree; read the summary line.
4. Remove the two stale generated stubs in the main tree if present (they are untracked build artifacts and would break a `-W` docs build against the deleted API): `docs/reference/_autosummary/pyphi.cache.DictCache.rst`, `docs/reference/_autosummary/pyphi.cache.method.rst`.
5. `git worktree remove` + `git branch -d` (from the main tree, not inside the worktree).
6. Update the review status block in `REVIEW-2026-07-13.md` and the session memory (both live only in the main tree). Do not touch the concurrent session's files (`docs/whats-new-in-2.0.md`, `experiments/`, benchmark JSONs).
