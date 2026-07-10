# `analyze()` Grain Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `grains=` keyword on `pyphi.analyze.analyze` that dispatches to the bounded grain search (`pyphi.macro.complexes`) and returns its `ComplexesResult`.

**Architecture:** Pure dispatch: validation up front (mutual exclusions, `grains` normalization), then `pyphi.macro.complexes(substrate, state, bounds, parallel_kwargs=...)` inside the same formalism-override context the single-system path uses. Deferred `pyphi.macro` import so the single-system path pays nothing. No new types.

**Tech Stack:** Python 3.13, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-analyze-grain-axis-design.md`

## Global Constraints

- `uv run` for all python. Never `--no-verify`. Stage only files the task touches. Pre-commit hooks must pass; if a hook modifies files, re-stage and re-commit.
- Docstrings NumPy-style, final-state impersonal voice, Unicode symbols (Φ, φₛ). No planning artifacts in code or the changelog fragment.
- `grains=False` raises `ValueError` (checked via `grains is True`, never `isinstance(grains, bool)` truthiness); `parallel_kwargs is not None` without `grains` raises (an empty dict raises too).
- Changelog fragment per user-facing change.
- Full verification at the end: `uv run pytest` with NO path argument.

---

### Task 1: The `grains=` dispatch

**Files:**
- Modify: `pyphi/analyze.py` (module docstring, `analyze` signature + docstring + body)
- Test: `test/test_analyze.py` (append)
- Create: `changelog.d/analyze-grains.feature.md`

**Interfaces:**
- Consumes (existing): `pyphi.macro.search.complexes(substrate, micro_history, bounds, parallel_kwargs=None) -> ComplexesResult`; `pyphi.macro.search.SearchBounds`; `_normalized_history` inside the driver already accepts a bare state when `bounds.max_micro_grain == 1` and raises a descriptive `ValueError` otherwise; the driver raises `ValueError` mentioning `IIT_3_0` under IIT 3.0.
- Produces: `analyze(substrate, state, *, subset=None, formalism=None, compute=None, grains=None, parallel_kwargs=None) -> Analysis | Any` — returns `ComplexesResult` when `grains` is set.

- [ ] **Step 1: Write the failing tests** — append to `test/test_analyze.py`:

```python
def test_analyze_grains_matches_macro_complexes():
    from pyphi.macro.search import ComplexesResult
    from pyphi.macro.search import SearchBounds
    from pyphi.macro.search import complexes
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with config.override(**presets.iit4_2023):
        via_analyze = analyze(substrate, (0, 0), grains=True)
        direct = complexes(substrate, (0, 0), SearchBounds())
    assert isinstance(via_analyze, ComplexesResult)
    assert via_analyze.maximal_complex.units == direct.maximal_complex.units
    assert float(via_analyze.maximal_complex.phi) == pytest.approx(
        float(direct.maximal_complex.phi), abs=1e-13
    )
    assert len(via_analyze.records) == len(direct.records)


def test_analyze_grains_accepts_bounds_instance():
    from pyphi.macro.search import SearchBounds
    from pyphi.models.complex import Complex
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with config.override(**presets.iit4_2023):
        result = analyze(substrate, (0, 0), grains=SearchBounds(max_depth=0))
    assert len(result.complexes) == 1
    assert all(isinstance(c, Complex) for c in result.complexes)


def test_analyze_grains_iit3_raises():
    from test.macro.test_macro_criteria import min_substrate

    substrate = min_substrate()
    with pytest.raises(ValueError, match="IIT_3_0"):
        analyze(substrate, (0, 0), formalism="IIT_3_0", grains=True)


def test_analyze_grains_mutual_exclusions():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="subset"):
        analyze(substrate, state, grains=True, subset=(0,))
    with pytest.raises(ValueError, match="compute"):
        analyze(substrate, state, grains=True, compute="sia")
    with pytest.raises(ValueError, match="parallel_kwargs"):
        analyze(substrate, state, parallel_kwargs={})


def test_analyze_grains_rejects_non_bounds():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with pytest.raises(ValueError, match="grains"):
        analyze(substrate, state, grains=0.5)
    # False is a confusion signal, not "no search".
    with pytest.raises(ValueError, match="grains"):
        analyze(substrate, state, grains=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_analyze.py -v -k grains`
Expected: FAIL with `TypeError: analyze() got an unexpected keyword argument 'grains'`.

- [ ] **Step 3: Implement** — in `pyphi/analyze.py`:

Widen the module docstring by appending one sentence to its paragraph:

```
``analyze`` takes a substrate and a state, builds the candidate system, runs
the analysis under the active (or a named) formalism, and returns an
:class:`Analysis` — a small bundle exposing the system irreducibility analysis,
the cause-effect structure, and the scalar Φ uniformly across formalisms. A
``compute`` argument selects a cheaper or custom result instead of the bundle;
a ``grains`` argument runs the bounded intrinsic-unit search over the whole
substrate instead, returning its complexes.
```

Replace the `analyze` signature and add the validation/dispatch (the body
below shows the complete function; the single-system branch is unchanged):

```python
def analyze(
    substrate: Any,
    state: tuple[int, ...],
    *,
    subset: Any = None,
    formalism: str | None = None,
    compute: Any = None,
    grains: Any = None,
    parallel_kwargs: dict | None = None,
) -> Analysis | Any:
    """Analyze one candidate system over ``substrate`` in ``state``, or run
    a grain search over the whole substrate.

    Parameters
    ----------
    substrate
        The substrate to analyze.
    state : tuple[int, ...]
        The state of the substrate's nodes. When ``grains`` admits update
        grains above 1, a sequence of micro states (oldest first) of the
        required length instead — see
        :func:`pyphi.macro.complexes`.
    subset : optional
        Node indices of the candidate system; ``None`` uses the whole
        substrate. Incompatible with ``grains``.
    formalism : str or None, optional
        ``None`` uses the active config formalism; a version name
        (``"IIT_3_0"`` / ``"IIT_4_0_2023"`` / ``"IIT_4_0_2026"``) applies that
        formalism for this call only.
    compute : optional
        ``None`` returns an :class:`Analysis` bundle; ``"sia"`` or ``"ces"``
        returns the raw result object; a callable returns ``compute(system)``.
        Incompatible with ``grains``.
    grains : optional
        ``None`` analyzes the single candidate system. ``True`` runs the
        bounded intrinsic-unit search with default
        :class:`~pyphi.macro.SearchBounds`; a
        :class:`~pyphi.macro.SearchBounds` instance runs it with those
        bounds. The search returns its
        :class:`~pyphi.macro.ComplexesResult`.
    parallel_kwargs : dict or None, optional
        Forwarded to :func:`pyphi.macro.complexes`; meaningful only with
        ``grains``.

    Returns
    -------
    Analysis
        The full analysis bundle; the raw result object when ``compute``
        selects one; or the grain search's
        :class:`~pyphi.macro.ComplexesResult` when ``grains`` is set.

    Raises
    ------
    ValueError
        If ``formalism`` is not a known version name; if ``compute`` is not
        ``"sia"``, ``"ces"``, a callable, or ``None``; if ``grains`` is not
        ``True``, a :class:`~pyphi.macro.SearchBounds`, or ``None``; if
        ``grains`` is combined with ``subset`` or ``compute``; or if
        ``parallel_kwargs`` is given without ``grains``.
    """
    if formalism is not None and formalism not in presets.by_name:
        valid = ", ".join(sorted(presets.by_name))
        raise ValueError(f"unknown formalism {formalism!r}; expected one of: {valid}")

    bounds: Any = None
    if grains is None:
        if parallel_kwargs is not None:
            raise ValueError(
                "parallel_kwargs applies only to a grain search; pass grains="
            )
    else:
        from pyphi.macro.search import SearchBounds

        if subset is not None:
            raise ValueError(
                "grains cannot be combined with subset: the grain search "
                "assembles candidate systems over the whole universe"
            )
        if compute is not None:
            raise ValueError(
                "grains cannot be combined with compute: the grain search "
                "returns its ComplexesResult"
            )
        if grains is True:
            bounds = SearchBounds()
        elif isinstance(grains, SearchBounds):
            bounds = grains
        else:
            raise ValueError(
                f"grains must be True or a SearchBounds instance; got {grains!r}"
            )

```

The validation block above ends where the existing `ctx = (...)` lines
begin — keep those and the rest of the function as they are, except for
the `with ctx:` body, which becomes (the single-system branch is the
current code verbatim, one indent deeper under `else:`):

```python
    result: Any = None
    with ctx:
        if bounds is not None:
            from pyphi.macro.search import complexes as grain_complexes

            result = grain_complexes(
                substrate, state, bounds, parallel_kwargs=parallel_kwargs
            )
        else:
            indices = substrate.node_indices if subset is None else subset
            system = System.from_substrate(substrate, state, indices)
            if callable(compute):
                result = compute(system)
            elif compute == "sia":
                result = system.sia()
            elif compute == "ces":
                result = system.ces()
            elif compute is not None:
                raise ValueError(
                    f"unknown compute: {compute!r}; expected 'sia', 'ces', a "
                    "callable, or None for the full Analysis bundle"
                )
            else:
                ces = system.ces()
                sia = getattr(ces, "sia", None)
                if sia is None:
                    sia = system.sia()
                result = Analysis(system=system, sia=sia, ces=ces)
    return result
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/test_analyze.py -v`
Expected: all PASS (15 existing + 5 new).

- [ ] **Step 5: Changelog fragment**

```bash
echo '`analyze()` gained a `grains` argument: `analyze(substrate, state, grains=True)` (or `grains=SearchBounds(...)`) runs the bounded intrinsic-unit grain search over the whole substrate and returns its `ComplexesResult`, making the grain search reachable from the front door.' > changelog.d/analyze-grains.feature.md
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/analyze.py test/test_analyze.py changelog.d/analyze-grains.feature.md
git commit -m "Add a grains axis to analyze()

analyze(substrate, state, grains=True) dispatches to the bounded
intrinsic-unit search and returns its ComplexesResult, under the same
formalism handling as the single-system path. grains is mutually
exclusive with subset and compute; parallel_kwargs forwards to the
search driver."
```

### Task 2: Full verification

- [ ] **Step 1: Run the full suite with the doctest sweep**

Run: `uv run pytest` (NO path argument)
Expected: all green (recent baseline on merged main: 3164 passed, 284 skipped).

- [ ] **Step 2: If anything fails,** fix within Task 1 and re-run; do not proceed with failures.
