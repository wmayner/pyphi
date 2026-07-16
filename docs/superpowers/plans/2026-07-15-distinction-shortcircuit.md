# Distinction-Level Reducibility Shortcircuit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip the remaining MICE search when a distinction is already known to be reducible (empty effect candidate-purview set, or a cause MICE with φ = 0), behind a new `formalism.iit.shortcircuit_distinctions` config option (default `True`).

**Architecture:** The cross-direction logic lives in `pyphi/formalism/queries.py::distinction()`, which grows purview kwargs so IIT 3.0's `concept()` can delegate to it (one implementation, two entry points). The skipped direction is a null MICE carrying a new `NullResultReason.OTHER_DIRECTION_REDUCIBLE`. φ = 0 distinctions are filtered out of every CES before ties/congruence/relations run, so surviving structures are bit-identical with the flag on or off.

**Tech Stack:** Python 3.13, pytest, `uv run` for everything.

**Spec:** `docs/superpowers/specs/2026-07-15-distinction-shortcircuit-design.md`

## Global Constraints

- Run all Python via `uv run` (e.g. `uv run pytest`).
- Any test that asserts a φ value pins its formalism with the preset context managers `IIT_3_CONFIG` / `IIT_4_CONFIG` from `test/conftest.py` — never the ambient default.
- When running tests whose result matters, redirect to a file (`uv run pytest ... > log 2>&1`) and read the summary line; never pipe through `tail`/`head`.
- Stage only the files you changed; other Claude instances may have unrelated working-dir changes. Never use `git checkout`/`git reset` on files you didn't touch.
- Never bypass pre-commit hooks (`--no-verify` is forbidden). If a commit doesn't land, read the hook output.
- Docstrings: NumPy style, final-state impersonal voice, Unicode symbols (φ not `:math:`), no development-process narrative.
- Commit messages describe what changed and why — no conversation narrative. End with:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01VqFMyh2PwtYvdWsSK2V6FY
  ```

---

### Task 1: Config option `shortcircuit_distinctions`

**Files:**
- Modify: `pyphi/conf/formalism.py` (docstring section near line 68–79; field near line 94; `__post_init__` bool list near line 108)
- Modify: `pyphi/conf/CLAUDE.md` (Computational Behavior section)
- Modify: `test/cache/test_disk_cache_key.py:92-101` (parametrize list)
- Test: `test/formalism/test_distinction_shortcircuit.py` (new file)

**Interfaces:**
- Produces: `config.formalism.iit.shortcircuit_distinctions: bool = True`, readable everywhere via `pyphi.conf.config`. Tasks 3–5 read it.

- [ ] **Step 1: Write the failing tests**

Create `test/formalism/test_distinction_shortcircuit.py`:

```python
"""Tests for distinction-level reducibility short-circuiting
(``formalism.iit.shortcircuit_distinctions``)."""

import pytest

import pyphi
from pyphi.conf.formalism import IITConfig
from test.conftest import IIT_4_CONFIG


@pytest.fixture(autouse=True)
def _pin_formalism():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield


def test_shortcircuit_distinctions_default_true():
    assert IITConfig().shortcircuit_distinctions is True


def test_shortcircuit_distinctions_must_be_bool():
    with pytest.raises(ValueError, match="shortcircuit_distinctions"):
        IITConfig(shortcircuit_distinctions="yes")


def test_presets_carry_shortcircuit_distinctions():
    from pyphi.conf import presets

    for preset in (presets.iit3, presets.iit4_2023, presets.iit4_2026):
        assert preset["iit"].shortcircuit_distinctions is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t1.log 2>&1` then read `/tmp/t1.log`.
Expected: FAIL / ERROR with `AttributeError` or `TypeError` — `shortcircuit_distinctions` unknown.

- [ ] **Step 3: Implement the config field**

In `pyphi/conf/formalism.py`, insert a docstring section directly after the existing "Reducibility short-circuiting (``shortcircuit_sia``)" section (which ends "...Does not gate IIT 3.0's early-exit logic." at line 79):

```
    Distinction short-circuiting (``shortcircuit_distinctions``)
        When ``True`` (default), evaluating a distinction stops early on
        detected reducibility: if the effect direction has no candidate
        purviews, neither MICE search runs, and if the cause MICE comes
        out with φ = 0, the effect search is skipped. A skipped
        direction is a null MICE carrying the
        ``OTHER_DIRECTION_REDUCIBLE`` reason; its φ reads 0 as a
        placeholder (that direction's own maximal φ is unknown), and its
        selection margins and ties are absent. The distinction's φ —
        the minimum across directions — is unaffected, so cause-effect
        structures are identical either way; only the contents of
        zero-φ distinctions differ. When ``False``, both directions are
        always evaluated in full, with exact margins and complete ties.
        Applies to every formalism, including IIT 3.0 concepts.
```

Add the field after `shortcircuit_sia: bool = True` (line 94):

```python
    shortcircuit_distinctions: bool = True
```

Add `"shortcircuit_distinctions",` to the bool-validation tuple in `__post_init__` (after `"shortcircuit_sia",` at line 111).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t1.log 2>&1` then read `/tmp/t1.log`.
Expected: 3 passed.

- [ ] **Step 5: Update the option references**

In `pyphi/conf/CLAUDE.md`, under "## Computational Behavior (`config.formalism.iit`)", after the `shortcircuit_sia` line, add:

```markdown
- **`shortcircuit_distinctions`**: Skip the remaining MICE search when a
  distinction is already known reducible (default: true)
```

In `test/cache/test_disk_cache_key.py`, add to the parametrize list after `("shortcircuit_sia", False),`:

```python
        ("shortcircuit_distinctions", False),
```

(The digest-completeness test in that file iterates dataclass fields, so it covers the new field automatically.)

- [ ] **Step 6: Run the cache-key tests**

Run: `uv run pytest test/cache/test_disk_cache_key.py -v > /tmp/t1b.log 2>&1` then read `/tmp/t1b.log`.
Expected: all pass, including the new parametrized case.

- [ ] **Step 7: Commit**

```bash
git add pyphi/conf/formalism.py pyphi/conf/CLAUDE.md test/cache/test_disk_cache_key.py test/formalism/test_distinction_shortcircuit.py
git commit -m "Add shortcircuit_distinctions config option"
```

---

### Task 2: `NullResultReason.OTHER_DIRECTION_REDUCIBLE`

**Files:**
- Modify: `pyphi/models/explanation.py:41-62`
- Test: `test/models/test_explanation.py`

**Interfaces:**
- Produces: `NullResultReason.OTHER_DIRECTION_REDUCIBLE` with `.level == "mechanism"`. Task 3 attaches it to skipped MICEs.

- [ ] **Step 1: Write the failing test**

Add to `test/models/test_explanation.py` (match the file's existing import style):

```python
def test_other_direction_reducible_is_mechanism_level():
    from pyphi.models.explanation import NullResultReason

    assert NullResultReason.OTHER_DIRECTION_REDUCIBLE.level == "mechanism"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/models/test_explanation.py -v > /tmp/t2.log 2>&1` then read `/tmp/t2.log`.
Expected: FAIL with `AttributeError: OTHER_DIRECTION_REDUCIBLE`.

- [ ] **Step 3: Implement**

In `pyphi/models/explanation.py`, add after `REDUCIBLE_OVER_PARTITION = auto()` (line 46):

```python
    OTHER_DIRECTION_REDUCIBLE = auto()
```

Add `NullResultReason.OTHER_DIRECTION_REDUCIBLE,` to `_MECHANISM_REASONS` (after `REDUCIBLE_OVER_PARTITION` at line 61).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/models/test_explanation.py -v > /tmp/t2.log 2>&1` then read `/tmp/t2.log`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/explanation.py test/models/test_explanation.py
git commit -m "Add OTHER_DIRECTION_REDUCIBLE null-result reason"
```

---

### Task 3: Shortcircuit logic in `queries.distinction()`

**Files:**
- Modify: `pyphi/formalism/queries.py:375-385` (the `distinction` function) and its imports
- Test: `test/formalism/test_distinction_shortcircuit.py`

**Interfaces:**
- Consumes: `config.formalism.iit.shortcircuit_distinctions` (Task 1), `NullResultReason.OTHER_DIRECTION_REDUCIBLE` (Task 2).
- Produces: `distinction(cs, mechanism, purviews=None, cause_purviews=None, effect_purviews=None, **kwargs) -> Concept`. Task 4's `concept()` delegates to this exact signature.

- [ ] **Step 1: Write the failing tests**

Add these imports to `test/formalism/test_distinction_shortcircuit.py` (merged into the existing import block in the standard order — ruff's isort rule enforces it):

```python
import numpy as np

from pyphi import examples
from pyphi import numerics
from pyphi.direction import Direction
from pyphi.formalism import queries
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import _null_ria
from pyphi.models.explanation import NullResultReason
from pyphi.substrate import Substrate
from pyphi.system import System
```

Then add:

```python
@pytest.fixture
def sink_system():
    """A → B → C chain; C has no outputs, so a mechanism containing only C
    has an empty candidate effect-purview set. A takes no inputs and is
    always 0; B copies A; C copies B."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    # fmt: on
    substrate = Substrate(tpm, cm=cm, node_labels=("A", "B", "C"))
    return System(substrate, (0, 0, 0))


def _recording_find_mice(monkeypatch):
    """Wrap queries.find_mice to record the directions it is called with."""
    calls = []
    real = queries.find_mice

    def recording(cs, direction, mechanism, **kwargs):
        calls.append(direction)
        return real(cs, direction, mechanism, **kwargs)

    monkeypatch.setattr(queries, "find_mice", recording)
    return calls


def test_cause_search_skipped_when_effect_trivially_reducible(
    sink_system, monkeypatch
):
    calls = _recording_find_mice(monkeypatch)
    d = queries.distinction(sink_system, (2,))
    assert Direction.CAUSE not in calls
    assert not numerics.is_positive(d.phi)
    assert tuple(d.effect.reasons) == (NullResultReason.NO_PURVIEWS,)
    assert tuple(d.cause.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_effect_search_skipped_when_cause_phi_zero(monkeypatch):
    calls = []
    real = queries.find_mice

    def zero_cause(cs, direction, mechanism, **kwargs):
        calls.append(direction)
        if direction == Direction.CAUSE:
            return MaximallyIrreducibleCause(
                _null_ria(
                    Direction.CAUSE,
                    mechanism,
                    (0,),
                    reasons=(NullResultReason.REDUCIBLE_OVER_PARTITION,),
                )
            )
        return real(cs, direction, mechanism, **kwargs)

    monkeypatch.setattr(queries, "find_mice", zero_cause)
    system = examples.basic_system()
    d = queries.distinction(system, (0,))
    assert calls == [Direction.CAUSE]
    assert not numerics.is_positive(d.phi)
    assert tuple(d.effect.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_flag_off_evaluates_both_directions(sink_system, monkeypatch):
    calls = _recording_find_mice(monkeypatch)
    with pyphi.config.override(shortcircuit_distinctions=False):
        d = queries.distinction(sink_system, (2,))
    assert Direction.CAUSE in calls
    assert Direction.EFFECT in calls
    assert NullResultReason.OTHER_DIRECTION_REDUCIBLE not in (d.cause.reasons or ())
    assert tuple(d.effect.reasons) == (NullResultReason.NO_PURVIEWS,)


def test_ces_identical_with_and_without_shortcircuit():
    system = examples.basic_system()
    with pyphi.config.override(shortcircuit_distinctions=True):
        on = list(system.all_distinctions())
    with pyphi.config.override(shortcircuit_distinctions=False):
        off = list(system.all_distinctions())
    assert on == off
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t3.log 2>&1` then read `/tmp/t3.log`.
Expected: the two skip tests FAIL (both directions currently always evaluated: `calls` contains `Direction.CAUSE`/both, and the skipped-reason assertions fail). `test_flag_off_evaluates_both_directions` and `test_ces_identical_with_and_without_shortcircuit` may already pass.

- [ ] **Step 3: Implement**

In `pyphi/formalism/queries.py`, add to the imports (after `from pyphi import conf as _conf` block, matching alphabetical order):

```python
from pyphi import numerics
```

Replace the `distinction` function (lines 375–385) with:

```python
def distinction(
    cs: System,
    mechanism: tuple[int, ...],
    purviews: Any | None = None,
    cause_purviews: Any | None = None,
    effect_purviews: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return the distinction specified by a mechanism.

    A distinction pairs the mechanism's maximally irreducible cause (its
    MIC) with its maximally irreducible effect (its MIE). The empty
    mechanism specifies the null distinction.

    Parameters
    ----------
    cs : System
        The system the mechanism belongs to.
    mechanism : tuple[int]
        The mechanism for which to determine the distinction.
    purviews : tuple[tuple[int]], optional
        A list of purviews to consider in both directions.
    cause_purviews : tuple[tuple[int]], optional
        A list of cause purviews to consider, overriding ``purviews``.
    effect_purviews : tuple[tuple[int]], optional
        A list of effect purviews to consider, overriding ``purviews``.

    Returns
    -------
    Concept
        The distinction specified by the mechanism.

    Notes
    -----
    When ``config.formalism.iit.shortcircuit_distinctions`` is set,
    evaluation stops as soon as the distinction is known to be reducible:
    if the effect direction has no candidate purviews, or once the cause
    MICE comes out with φ = 0, the remaining search is skipped and the
    unevaluated direction is a null MICE carrying the
    :attr:`~pyphi.models.explanation.NullResultReason.OTHER_DIRECTION_REDUCIBLE`
    reason. The distinction's φ (the minimum across directions) is
    unaffected.
    """
    if not mechanism:
        return _ra.null_concept(cs)
    cause_purviews = cause_purviews if cause_purviews is not None else purviews
    effect_purviews = effect_purviews if effect_purviews is not None else purviews
    shortcircuit = config.formalism.iit.shortcircuit_distinctions  # pyright: ignore[reportAttributeAccessIssue]

    if shortcircuit and not _ra.potential_purviews(
        cs, Direction.EFFECT, mechanism, effect_purviews
    ):
        # The effect side is trivially reducible, so the distinction's φ is 0
        # no matter the cause; the cause search is skipped.
        effect = find_mice(
            cs, Direction.EFFECT, mechanism, purviews=effect_purviews, **kwargs
        )
        cause = MaximallyIrreducibleCause(
            _null_ria(
                Direction.CAUSE,
                mechanism,
                (),
                reasons=(NullResultReason.OTHER_DIRECTION_REDUCIBLE,),
            )
        )
        return Concept(mechanism=mechanism, cause=cause, effect=effect)

    cause = find_mice(cs, Direction.CAUSE, mechanism, purviews=cause_purviews, **kwargs)
    if shortcircuit and not numerics.is_positive(float(cause.phi)):
        effect = MaximallyIrreducibleEffect(
            _null_ria(
                Direction.EFFECT,
                mechanism,
                (),
                reasons=(NullResultReason.OTHER_DIRECTION_REDUCIBLE,),
            )
        )
        return Concept(mechanism=mechanism, cause=cause, effect=effect)

    effect = find_mice(cs, Direction.EFFECT, mechanism, purviews=effect_purviews, **kwargs)
    return Concept(mechanism=mechanism, cause=cause, effect=effect)
```

Note the guard is `numerics.is_positive(float(cause.phi))`, not `not cause`: the MICE classes define no `__bool__`, so object truthiness is always `True`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t3.log 2>&1` then read `/tmp/t3.log`.
Expected: all pass.

- [ ] **Step 5: Run the neighboring formalism tests for regressions**

Run: `uv run pytest test/formalism/ -x -q > /tmp/t3b.log 2>&1` then read `/tmp/t3b.log`.
Expected: all pass (no φ value changes anywhere; only zero-φ distinction contents changed).

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/queries.py test/formalism/test_distinction_shortcircuit.py
git commit -m "Shortcircuit distinction evaluation on detected reducibility"
```

---

### Task 4: IIT 3.0 `concept()` delegates; confirm the partitioned-constellation path

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py:54-99` (the `concept` function)
- Test: `test/formalism/test_distinction_shortcircuit.py`

**Interfaces:**
- Consumes: `distinction(cs, mechanism, purviews=None, cause_purviews=None, effect_purviews=None, **kwargs)` from Task 3, and the `sink_system` fixture plus `_recording_find_mice(monkeypatch)` helper already defined in `test/formalism/test_distinction_shortcircuit.py` by Task 3.
- Produces: `concept()` with unchanged public signature and docstring.

- [ ] **Step 1: Write the failing test**

Add to `test/formalism/test_distinction_shortcircuit.py`, with `from test.conftest import IIT_3_CONFIG` merged into the import block:

```python
def test_iit3_concept_shortcircuits(sink_system, monkeypatch):
    calls = _recording_find_mice(monkeypatch)
    from pyphi.formalism import iit3

    with IIT_3_CONFIG, pyphi.config.override(progress_bars=False):
        c = iit3.concept(sink_system, (2,))
    assert Direction.CAUSE not in calls
    assert tuple(c.cause.reasons) == (NullResultReason.OTHER_DIRECTION_REDUCIBLE,)


def test_iit3_sia_unchanged_by_shortcircuit():
    """Confirmation experiment for the spec's verification point: the IIT 3.0
    partitioned-constellation path consumes nothing from skipped MICEs, so the
    SIA is identical with the flag on and off."""
    with IIT_3_CONFIG, pyphi.config.override(progress_bars=False):
        system = examples.basic_system()
        with pyphi.config.override(shortcircuit_distinctions=True):
            sia_on = system.sia()
        with pyphi.config.override(shortcircuit_distinctions=False):
            sia_off = system.sia()
    assert numerics.eq(sia_on.phi, sia_off.phi)
    assert sia_on.partition == sia_off.partition
```

- [ ] **Step 2: Run tests to verify the concept test fails**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t4.log 2>&1` then read `/tmp/t4.log`.
Expected: `test_iit3_concept_shortcircuits` FAILS (`concept()` still calls `mic()` unconditionally, so `Direction.CAUSE` is in `calls`). `test_iit3_sia_unchanged_by_shortcircuit` should already pass — it confirms current behavior; keep it either way.

- [ ] **Step 3: Implement the delegation**

In `pyphi/formalism/iit3/__init__.py`, replace the body of `concept()` (keep the signature and full docstring exactly as they are, lines 54–85) with:

```python
    from pyphi.formalism.queries import distinction

    return distinction(
        system,
        mechanism,
        purviews=purviews,
        cause_purviews=cause_purviews,
        effect_purviews=effect_purviews,
        **kwargs,
    )
```

Remove the now-unused function-local imports (`_ra`, `mic`, `mie`) — `distinction()` handles the empty-mechanism null concept itself. Check whether `Concept` and other names remain used elsewhere in the module before touching the module-level imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t4.log 2>&1` then read `/tmp/t4.log`.
Expected: all pass.

- [ ] **Step 5: Run the IIT 3.0 test files for regressions**

Run: `uv run pytest test/formalism/test_formalism.py test/formalism/test_iit3_divergence_audit.py -q > /tmp/t4b.log 2>&1` then read `/tmp/t4b.log`.
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit3/__init__.py test/formalism/test_distinction_shortcircuit.py
git commit -m "Route IIT 3.0 concept() through the shortcircuiting distinction path"
```

---

### Task 5: Early return in `queries.phi()`

**Files:**
- Modify: `pyphi/formalism/queries.py:248-257` (the `phi` function)
- Test: `test/formalism/test_distinction_shortcircuit.py`

**Interfaces:**
- Consumes: nothing new. Unconditional (no config gate): `phi()` returns a bare float with no ties or margins to truncate, and φ values are non-negative.
- Produces: `phi(cs, mechanism, purview, **kwargs) -> float`, signature unchanged.

- [ ] **Step 1: Write the failing test**

Add to `test/formalism/test_distinction_shortcircuit.py`:

```python
def test_phi_skips_effect_mip_when_cause_phi_zero(sink_system, monkeypatch):
    calls = []
    real = queries.phi_effect_mip

    def recording(cs, mechanism, purview, **kwargs):
        calls.append((mechanism, purview))
        return real(cs, mechanism, purview, **kwargs)

    monkeypatch.setattr(queries, "phi_effect_mip", recording)
    # Purview (0,) is not a potential cause purview of mechanism (2,)
    # (A receives no edge from C), so the cause MIP is null with φ = 0.
    result = queries.phi(sink_system, (2,), (0,))
    assert result == 0
    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py::test_phi_skips_effect_mip_when_cause_phi_zero -v > /tmp/t5.log 2>&1` then read `/tmp/t5.log`.
Expected: FAIL — `calls` is non-empty (`min()` currently evaluates both arguments).

- [ ] **Step 3: Implement**

Replace `phi()` in `pyphi/formalism/queries.py` (lines 248–257) with:

```python
def phi(
    cs: System,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> float:
    """Return the φ of a mechanism over a purview: the minimum of the cause
    and effect MIP φ values. The effect MIP is not evaluated when the cause
    MIP's φ is already 0, since φ values are non-negative."""
    cause_phi = phi_cause_mip(cs, mechanism, purview, **kwargs)
    if not numerics.is_positive(float(cause_phi)):
        return cause_phi
    return min(cause_phi, phi_effect_mip(cs, mechanism, purview, **kwargs))
```

- [ ] **Step 4: Run the test file to verify everything passes**

Run: `uv run pytest test/formalism/test_distinction_shortcircuit.py -v > /tmp/t5.log 2>&1` then read `/tmp/t5.log`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/queries.py test/formalism/test_distinction_shortcircuit.py
git commit -m "Skip the effect MIP in phi() when the cause MIP is reducible"
```

---

### Task 6: Changelog, MCP surfaces, full-suite verification

**Files:**
- Create: `changelog.d/shortcircuit-distinctions.config.md`
- Modify: `pyphi/mcp/content/configuration.md` (and `pyphi/mcp/content/performance.md` if it discusses short-circuiting)

**Interfaces:**
- Consumes: everything from Tasks 1–5. No code interfaces produced.

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/shortcircuit-distinctions.config.md`:

```markdown
Added `formalism.iit.shortcircuit_distinctions` (default `True`): distinction
evaluation now skips the remaining MICE search when the distinction is already
known to be reducible — an empty candidate effect-purview set, or a cause MICE
with φ = 0. The skipped direction is a null MICE carrying the new
`OTHER_DIRECTION_REDUCIBLE` reason; set the option to `False` to always
evaluate both directions in full (exact selection margins and complete ties).
```

- [ ] **Step 2: Surface the option in the MCP content**

Read `pyphi/mcp/content/configuration.md` and `pyphi/mcp/content/performance.md`. In `configuration.md`, add `shortcircuit_distinctions` wherever the `formalism.iit` options are listed, using the surrounding entries' format, with a one-sentence description matching the changelog fragment. If `performance.md` has a section on reducibility short-circuiting or cost control, mention the option there in one sentence; if not, leave it alone.

- [ ] **Step 3: Full-suite verification (no path argument — doctest sweep included)**

Run: `uv run pytest -q -m "not slow" > /tmp/full.log 2>&1` then read the summary at the end of `/tmp/full.log`.
Expected: all pass. Then kick off the slow lane in the background per the project convention (`uv run pytest -m slow --slow -q > /tmp/slow.log 2>&1` with `run_in_background=true`) and read its summary before declaring the project done.

- [ ] **Step 4: Commit**

```bash
git add changelog.d/shortcircuit-distinctions.config.md pyphi/mcp/content/configuration.md
git commit -m "Document the shortcircuit_distinctions option"
```

(Include `pyphi/mcp/content/performance.md` in the `git add` if it was changed.)
