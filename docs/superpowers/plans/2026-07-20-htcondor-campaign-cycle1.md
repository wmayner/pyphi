# HTCondor Campaign Surface (Cycle 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute PyPhi sweeps across an HTCondor pool as self-contained
tasks-as-data condor jobs: `sweep()` gains a substrates axis, and a new
`pyphi.campaign` package prepares a campaign directory, runs task files via
`python -m pyphi.campaign run`, and collects the exact `SweepResult` a local
sweep would return.

**Architecture:** Scheduler-neutral core with a condor emitter (spec
`docs/superpowers/specs/2026-07-20-htcondor-campaign-design.md`). Tasks are
`pyphi.serialize` documents; done = output file exists and loads; no condor
invocation from Python, no long-lived coordinator. Packing uses the existing
`cost_balanced_partition` fed by `estimate_analysis` weights.

**Tech Stack:** Python 3.13, msgspec (via `pyphi.serialize`), pandas,
existing `pyphi.sweep` / `pyphi.cost` / `pyphi.parallel.chunking` machinery.
No new dependencies.

## Global Constraints

- Commit messages end with both trailers, each on its own line:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV`.
- Never `git commit --no-verify`. After EVERY commit run
  `git log --oneline -1` — the ruff-format hook aborts commits silently.
- All Python invocations via `uv run` (e.g. `uv run pytest`).
- Never pipe pytest through `tail`/`head`; redirect to a log file
  (`> /tmp/log 2>&1`) and read the summary line. Never end verification
  commands with `; echo`.
- Docstrings: NumPy style, final-state impersonal voice, Unicode symbols
  (`φ`, `Φ`). No planning-artifact references (spec/plan/roadmap item names)
  in code, docstrings, comments, or changelog fragments.
- Tests that assert φ values pin their formalism explicitly. Campaign/sweep
  tests pass `formalisms=["IIT_4_0_2026"]` explicitly (the sweep machinery
  installs the preset per cell, which is a complete pin).
- Do NOT touch files owned by concurrent sessions: `docs/whats-new-in-2.0.md`,
  `REVIEW-2026-07-13.md`, `TRIAGE-WAVE5.md`, `color-theory/`,
  `experiments/`, benchmark result JSONs. Stage only files this plan names.
- Config-override merging: snapshot overrides and a formalism preset share
  keys — always merge into ONE dict first
  (`{**wire_overrides, **presets.by_name[f], "parallel": False, ...}`),
  never pass two `**` expansions to `config.override`.
- `prepare()` never writes into an existing directory. The runner writes
  outputs atomically (temp name in the same directory + `os.replace`).
- Substrate labels must match `[A-Za-z0-9_-]+` (they become filenames).
- Warnings use `pyphi.warnings.PyPhiWarning`.

## File Map

| File | Responsibility |
|---|---|
| `pyphi/sweep.py` (modify) | substrates axis; shared cell enumeration + `_run_cell`/`_extract_row`/`_build_df` reused by campaign |
| `pyphi/campaign/__init__.py` (create) | `prepare`, `status`, `collect`, `CampaignStatus`, `CampaignTask`, `CellOutput`, `CampaignTaskOutput`; condor emitter |
| `pyphi/campaign/runner.py` (create) | `run_task()` — execute one task file |
| `pyphi/campaign/__main__.py` (create) | `python -m pyphi.campaign run` CLI |
| `pyphi/serialize/schema.py` + `convert.py` (modify) | `campaign_task` / `campaign_task_output` schemas; 4-tuple `skipped` |
| `pyphi/mcp/server.py`, `content.py`, `content/campaigns.md` | MCP tools + reference page |
| `test/test_sweep.py`, `test/serialize/test_serialize_results.py` (modify) | substrates-axis + schema tests |
| `test/campaign/` (create) | campaign unit + end-to-end tests |
| `docs/howto/campaigns.md` (create), `index.md`, `chtc.md`, `parallel.md` | docs |
| `changelog.d/` | two feature fragments |

**Plan-level resolution of a spec ambiguity:** the spec says both "single
substrate → constant column, as with the other axes" and "single-substrate
results byte-identical to current behavior". These conflict: the constant
`substrate` column is new. Parity with the other axes wins — a single
substrate yields a constant `substrate` column (label `0`), and existing
tests are updated accordingly.

**No `pyphi/__init__.py` change is needed** for the spec's lazy-import
deliverable: the PEP-562 `__getattr__` builds `_SUBMODULE_NAMES` from
`pkgutil.iter_modules`, so the new `pyphi/campaign/` package is
automatically importable as `pyphi.campaign` (verified against
`pyphi/__init__.py`).

---

### Task 1: Substrates axis on `sweep()`

**Files:**
- Modify: `pyphi/sweep.py`
- Modify: `pyphi/serialize/schema.py` (SweepResultSchema.skipped),
  `pyphi/serialize/convert.py` (`_register_sweep_result`)
- Test: `test/test_sweep.py`, `test/serialize/test_serialize_results.py`
- Create: `changelog.d/sweep-substrates.feature.md`

**Interfaces:**
- Consumes: existing `pyphi.sweep` internals; `Substrate` from
  `pyphi.substrate`.
- Produces (campaign tasks rely on these exact names):
  - `sweep(substrates, *, states, subsets="full", formalisms=None,
    compute="sia", parallel=None, progress=None, seed=None) -> SweepResult`
    where `substrates` is a `Substrate`, a sequence, or a
    `Mapping[str, Substrate]`.
  - `_normalize_substrates(substrates) -> list[tuple[str | int, Substrate]]`
  - `_enumerate_cells(labeled, states, subsets, formalisms_)
    -> list[tuple[label, formalism, subset, state]]`
  - `_run_cell(cell, *, substrates, compute, skip)` with
    `cell = (label, subset, state)` and `substrates` a dict label→Substrate.
  - `_build_df(keys, rows, enumerated)` — 4-tuple keys; an axis becomes an
    index level iff it takes >1 distinct value among `enumerated` cells.
  - `SweepResult.skipped` entries are `(label, formalism, subset, state)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_sweep.py`:

```python
class TestSubstratesAxis:
    def test_single_substrate_constant_column(self):
        substrate = examples.basic_substrate()
        result = sweep(
            substrate,
            states=[(1, 0, 0)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert list(result.df["substrate"]) == [0]

    def test_dict_labels_become_index_level(self):
        subs = {"basic": examples.basic_substrate(), "xor": examples.xor_substrate()}
        result = sweep(
            subs,
            states=[(1, 0, 0)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert result.df.index.name == "substrate"
        assert set(result.df.index) == {"basic", "xor"}

    def test_sequence_labels_are_positions(self):
        subs = [examples.basic_substrate(), examples.xor_substrate()]
        result = sweep(
            subs,
            states=[(1, 0, 0)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert set(result.df.index) == {0, 1}

    def test_all_states_enumerated_per_substrate(self):
        # Substrates of different sizes coexist under states="all".
        subs = {
            "small": examples.basic_substrate(),
            "fig4": examples.fig4_substrate(),
        }
        result = sweep(
            subs,
            states="all",
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        computed_plus_skipped = len(result.df) + len(result.skipped)
        n_small = len(examples.basic_substrate())
        n_fig4 = len(examples.fig4_substrate())
        assert computed_plus_skipped == 2**n_small + 2**n_fig4

    def test_skipped_entries_are_4_tuples(self):
        result = sweep(
            examples.basic_substrate(),
            states="all",
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        for entry in result.skipped:
            label, formalism, subset, state = entry
            assert label == 0
            assert formalism == "IIT_4_0_2026"
```

(If `test/test_sweep.py` does not already import `examples` and `sweep`,
add `from pyphi import examples` and `from pyphi.sweep import sweep` at the
top. If `fig4_substrate` does not exist under that name, check
`uv run python -c "import pyphi; print([n for n in dir(pyphi.examples) if 'fig4' in n])"`
and use the substrate accessor listed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_sweep.py::TestSubstratesAxis -x -q > /tmp/t1a.log 2>&1`; read the log.
Expected: FAIL (`KeyError: 'substrate'` or TypeError from passing a dict).

- [ ] **Step 3: Implement the substrates axis**

In `pyphi/sweep.py`:

1. Add imports:

```python
from collections.abc import Mapping

from pyphi.substrate import Substrate
```

2. Replace the module docstring's first paragraph to mention the substrates
axis:

```python
"""Cartesian batch driver: run an IIT computation across substrates, states,
subsystems, and formalisms, and collect the results into one tidy DataFrame.

``sweep`` takes one or more substrates and up to three further axes (states,
candidate subsets, formalisms), runs the chosen computation on the cartesian
product, and returns a :class:`SweepResult` holding a long-format DataFrame
and the aligned raw result objects. Each result carries its own configuration
snapshot, so a row is independently reproducible.
"""
```

3. Update the `SweepResult` docstring `skipped` sentence to:
`` ``skipped`` lists the ``(substrate, formalism, subset, state)`` cells
dropped because their state is dynamically unreachable (only when an axis is
enumerated via ``"all"``; explicit cells fail loud instead). ``

4. Add normalization and enumeration helpers after `_normalize_formalisms`:

```python
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _normalize_substrates(substrates: Any) -> list[tuple[Any, Any]]:
    """Normalize the substrates argument to labeled ``(label, substrate)`` pairs.

    A mapping supplies its own labels; a bare substrate gets label ``0``; any
    other iterable is labeled by position.
    """
    if isinstance(substrates, Mapping):
        for label in substrates:
            if not (isinstance(label, str) and _LABEL_RE.match(label)):
                raise ValueError(
                    f"substrate label {label!r} must match [A-Za-z0-9_-]+ "
                    "(labels are used in filenames)"
                )
        return list(substrates.items())
    if isinstance(substrates, Substrate):
        return [(0, substrates)]
    return list(enumerate(substrates))


def _enumerate_cells(
    labeled: list[tuple[Any, Any]],
    states: Any,
    subsets: Any,
    formalisms_: list[str],
) -> list[tuple[Any, str, tuple, tuple]]:
    """Enumerate ``(label, formalism, subset, state)`` cells in canonical order.

    Explicit ``states``/``subsets`` apply to every substrate; ``"all"`` is
    enumerated per substrate, so substrates of different sizes coexist.
    """
    cells = []
    for formalism in formalisms_:
        for label, substrate in labeled:
            for subset, state in product(
                _normalize_subsets(substrate, subsets),
                _normalize_states(substrate, states),
            ):
                cells.append((label, formalism, subset, state))
    return cells
```

Add `import re` to the imports.

5. Replace `_run_cell` (cells now carry the substrate label; the substrate
dict makes the callable picklable once for all cells):

```python
def _run_cell(
    cell: tuple[Any, Any, Any], *, substrates: dict, compute: Any, skip: bool
) -> Any:
    """Build the system for one (label, subset, state) cell and run its computation.

    Module-level and config-free so it is picklable for the process backend;
    the active formalism is installed in the worker via the propagated config
    snapshot, not set here. When ``skip`` is true, an unreachable (uncomputable)
    state yields a :class:`_Skipped` sentinel instead of raising.
    """
    label, subset, state = cell
    try:
        system = System(substrates[label], state, node_indices=subset)
        return _dispatch_compute(system, compute)
    except _UNREACHABLE:
        if skip:
            return _Skipped(cell)
        raise
```

6. In `_run_cells_sequential` and `_run_cells_parallel`, change the
`substrate` parameter to `substrates: dict` and thread it through:
sequential body becomes
`_run_cell(c, substrates=substrates, compute=compute, skip=skip)`;
parallel binding becomes
`cell_fn = partial(_run_cell, substrates=substrates, compute=compute, skip=skip)`.

7. Replace `_build_df`:

```python
_AXIS_NAMES = ("substrate", "formalism", "subset", "state")


def _build_df(
    keys: list[tuple[Any, str, tuple, tuple]],
    rows: list[dict[str, Any]],
    enumerated: list[tuple[Any, str, tuple, tuple]],
) -> pd.DataFrame:
    """Build the tidy table; an axis is an index level iff it varies.

    ``enumerated`` is the full cell enumeration (before unreachable-state
    skips), so whether an axis varies is a property of what was asked for,
    not of which cells happened to compute.
    """
    df = pd.DataFrame(rows)
    levels: dict[str, list[Any]] = {}
    for pos, name in enumerate(_AXIS_NAMES):
        distinct = {cell[pos] for cell in enumerated}
        if len(distinct) > 1:
            levels[name] = [k[pos] for k in keys]
        else:
            df[name] = [next(iter(distinct))] * len(df)
    if len(levels) == 1:
        name, values = next(iter(levels.items()))
        # tupleize_cols=False keeps tuple state/subset values as scalar index
        # entries instead of expanding them into a MultiIndex.
        df.index = pd.Index(values, name=name, tupleize_cols=False)
    elif len(levels) > 1:
        df.index = pd.MultiIndex.from_arrays(
            list(levels.values()), names=list(levels.keys())
        )
    return df
```

8. Replace `sweep()`'s body (signature: first parameter renamed to
`substrates`; docstring's first parameter updated to describe the three
accepted forms):

```python
def sweep(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    compute: Any = "sia",
    parallel: bool | None = None,
    progress: bool | None = None,
    seed: int | None = None,
) -> SweepResult:
    """Run a computation across the cartesian product of axes into a tidy table.

    Parameters
    ----------
    substrates
        A single substrate, a sequence of substrates (labeled by position),
        or a mapping of label to substrate.
    states
        A state tuple, an iterable of states, or ``"all"``. Explicit states
        apply to every substrate; ``"all"`` enumerates per substrate.
    subsets
        ``"full"`` (whole system), ``"all"`` (non-empty powerset), or an
        iterable of node-index tuples. Explicit subsets apply to every
        substrate; ``"full"`` and ``"all"`` are resolved per substrate.
    formalisms
        ``None`` (the active formalism) or an iterable of version names
        (``"IIT_3_0"``, ``"IIT_4_0_2023"``, ``"IIT_4_0_2026"``).
    compute
        ``"sia"`` (default), ``"ces"``, or a callable taking a ``System``.
    parallel : bool or None, optional
        ``None`` follows ``config.infrastructure.parallel``; ``True`` or
        ``False`` forces.
    progress : bool or None, optional
        ``None`` follows config; ``True`` or ``False`` forces.
    seed : int or None, optional
        Stamped into each result's provenance (a bookkeeping label).

    Returns
    -------
    SweepResult
        The tidy long-format table, the aligned raw result objects, and the
        list of cells skipped because their state is dynamically unreachable.

    Notes
    -----
    Cells are skipped (rather than raising) only when an axis is enumerated via
    ``"all"``; when every axis is given explicitly, an uncomputable cell raises.
    """
    skip_uncomputable = states == "all" or subsets == "all"
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    formalisms_ = _normalize_formalisms(formalisms)
    enumerated = _enumerate_cells(labeled, states, subsets, formalisms_)
    use_parallel = config.infrastructure.parallel if parallel is None else parallel

    keys: list[tuple[Any, str, tuple, tuple]] = []
    raw: list[Any] = []
    skipped: list[tuple[Any, str, tuple, tuple]] = []
    for formalism in formalisms_:
        cells = [
            (label, subset, state)
            for label, f, subset, state in enumerated
            if f == formalism
        ]
        if use_parallel:
            results = _run_cells_parallel(
                substrate_map, formalism, cells, compute, skip_uncomputable, progress
            )
        else:
            results = _run_cells_sequential(
                substrate_map, formalism, cells, compute, skip_uncomputable, progress
            )
        for (label, subset, state), result in zip(cells, results, strict=True):
            if isinstance(result, _Skipped):
                skipped.append((label, formalism, subset, state))
            else:
                keys.append((label, formalism, subset, state))
                raw.append(result)

    if seed is not None:
        for result in raw:
            with_provenance = getattr(result, "with_provenance", None)
            if with_provenance is not None:
                with_provenance(seed=seed)

    rows = [_extract_row(result, compute) for result in raw]
    df = _build_df(keys, rows, enumerated)
    return SweepResult(df=df, results=raw, skipped=skipped)
```

- [ ] **Step 4: Update the serialization schema**

In `pyphi/serialize/schema.py`, replace `SweepResultSchema`:

```python
class SweepResultSchema(msgspec.Struct, frozen=True, tag="sweep_result"):
    df: DataFrameSchema
    results: tuple["Schema | float", ...]
    skipped: tuple[
        tuple["str | int", str, tuple[int, ...], tuple[int, ...]], ...
    ]
```

In `pyphi/serialize/convert.py` `_register_sweep_result`, update both
directions to 4-tuples:

```python
        skipped=tuple(
            (label, formalism, tuple(subset), tuple(state))
            for label, formalism, subset, state in r.skipped
        ),
```

and in `_decode_sweep_result`:

```python
            skipped=[
                (label, formalism, tuple(subset), tuple(state))
                for label, formalism, subset, state in s.skipped
            ],
```

- [ ] **Step 5: Run the new tests and repair existing sweep/serialize tests**

Run: `uv run pytest test/test_sweep.py test/serialize/test_serialize_results.py -q > /tmp/t1b.log 2>&1`; read the log.

The new `TestSubstratesAxis` tests must pass. Existing tests may fail for
exactly two legitimate reasons: (1) the new constant `substrate` column in
DataFrames, (2) `skipped` entries now being 4-tuples. Update those
assertions in place. Any other failure means the implementation is wrong —
fix the implementation, not the test.

- [ ] **Step 6: Changelog fragment**

```bash
printf '%s\n' '`pyphi.sweep` accepts multiple substrates: a sequence or a `{label: substrate}` mapping adds a `substrate` axis to the sweep, enumerated per substrate under `states="all"`/`subsets="all"`.' > changelog.d/sweep-substrates.feature.md
```

- [ ] **Step 7: Commit**

```bash
git add pyphi/sweep.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/test_sweep.py test/serialize/test_serialize_results.py changelog.d/sweep-substrates.feature.md
git commit -m "Add substrates axis to sweep()"   # include the two standard trailers
git log --oneline -1
```

---

### Task 2: Campaign task/output types and serialization

**Files:**
- Create: `pyphi/campaign/__init__.py` (types only in this task)
- Modify: `pyphi/serialize/schema.py`, `pyphi/serialize/convert.py`
- Test: `test/campaign/__init__.py` (empty), `test/campaign/test_types.py`

**Interfaces:**
- Consumes: `to_schema`/`from_schema`/`_ENCODERS`/`_DECODERS` machinery in
  `pyphi/serialize/convert.py` (follow `_register_sweep_result` as the
  model, including where `_register_*` functions are invoked near line
  1380).
- Produces (used verbatim by Tasks 3–5):

```python
@dataclass(frozen=True)
class CampaignTask:
    task_id: int
    kind: str                       # "sweep_cells"
    compute: str | None             # "sia" | "ces"; None when compute_ref is set
    compute_ref: str | None         # "module:qualname" for callable computes
    config_overrides: dict[str, Any]
    cells: tuple[tuple[Any, str, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool

@dataclass(frozen=True)
class CellOutput:
    status: str                     # "ok" | "skipped" | "error"
    result: Any | None
    traceback: str | None

@dataclass(frozen=True)
class CampaignTaskOutput:
    task_id: int
    pyphi_version: str
    entries: tuple[CellOutput, ...]
```

- [ ] **Step 1: Write the failing test**

`test/campaign/test_types.py`:

```python
from pyphi import examples
from pyphi.campaign import CampaignTask
from pyphi.campaign import CampaignTaskOutput
from pyphi.campaign import CellOutput
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.serialize import load
from pyphi.serialize import save
from pyphi.system import System


def test_campaign_task_roundtrip(tmp_path):
    task = CampaignTask(
        task_id=3,
        kind="sweep_cells",
        compute="sia",
        compute_ref=None,
        config_overrides={"precision": 13},
        cells=(("basic", "IIT_4_0_2026", (0, 1, 2), (1, 0, 0)),),
        skip_uncomputable=True,
    )
    path = tmp_path / "task-0003.json.gz"
    save(task, path)
    loaded = load(path)
    assert loaded == task


def test_campaign_task_output_roundtrip_with_embedded_result(tmp_path):
    with config.override(
        **presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
    ):
        sia = System(examples.basic_substrate(), (1, 0, 0)).sia()
    out = CampaignTaskOutput(
        task_id=3,
        pyphi_version="test",
        entries=(
            CellOutput(status="ok", result=sia, traceback=None),
            CellOutput(status="skipped", result=None, traceback=None),
            CellOutput(status="error", result=None, traceback="Traceback: boom"),
        ),
    )
    path = tmp_path / "task-0003.json.gz"
    save(out, path)
    loaded = load(path)
    assert loaded.task_id == 3
    assert [e.status for e in loaded.entries] == ["ok", "skipped", "error"]
    assert float(loaded.entries[0].result.phi) == float(sia.phi)
    assert loaded.entries[2].traceback == "Traceback: boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/ -x -q > /tmp/t2a.log 2>&1`; read the log.
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.campaign'`.
(Create `test/campaign/__init__.py` as an empty file first, matching the
other `test/` subpackages.)

- [ ] **Step 3: Create the types**

`pyphi/campaign/__init__.py`:

```python
"""Distribute PyPhi computations across an HTCondor pool as batch campaigns.

A campaign is a self-contained directory of serialized task files that
independent condor jobs execute via ``python -m pyphi.campaign run``; results
are collected from per-task output files. :func:`prepare` writes the
directory, the user submits the generated submit file with ``condor_submit``,
and :func:`status` / :func:`collect` operate purely on the directory's
files — a task is done exactly when its output file exists and loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CampaignTask",
    "CampaignTaskOutput",
    "CellOutput",
]


@dataclass(frozen=True)
class CampaignTask:
    """One condor job's worth of work: the cells it owns and their context.

    ``config_overrides`` is the preparing session's configuration in override
    form (JSON-compatible); the runner installs it beneath each cell's
    formalism preset. ``compute_ref`` is a ``"module:qualname"`` reference
    used when the computation is a callable rather than ``"sia"``/``"ces"``.
    """

    task_id: int
    kind: str
    compute: str | None
    compute_ref: str | None
    config_overrides: dict[str, Any]
    cells: tuple[tuple[Any, str, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool


@dataclass(frozen=True)
class CellOutput:
    """One cell's outcome: ``ok`` (with the result), ``skipped``, or ``error``."""

    status: str
    result: Any | None
    traceback: str | None


@dataclass(frozen=True)
class CampaignTaskOutput:
    """A task's per-cell outcomes, aligned 1:1 with the task's cells."""

    task_id: int
    pyphi_version: str
    entries: tuple[CellOutput, ...]
```

- [ ] **Step 4: Register the schemas**

In `pyphi/serialize/schema.py`, after `SweepResultSchema`:

```python
class CampaignTaskSchema(msgspec.Struct, frozen=True, tag="campaign_task"):
    task_id: int
    kind: str
    compute: "str | None"
    compute_ref: "str | None"
    config_overrides: dict[str, Any]
    cells: tuple[
        tuple["str | int", str, tuple[int, ...], tuple[int, ...]], ...
    ]
    skip_uncomputable: bool


class CellOutputSchema(msgspec.Struct, frozen=True, tag="campaign_cell_output"):
    status: str
    result: "Schema | None"
    traceback: "str | None"


class CampaignTaskOutputSchema(
    msgspec.Struct, frozen=True, tag="campaign_task_output"
):
    task_id: int
    pyphi_version: str
    entries: tuple[CellOutputSchema, ...]
```

(If `Any` is not imported in `schema.py`, check its existing imports —
`config_overrides` values are plain builtins; use the same typing idiom the
file already uses for `dict[str, Any]` fields such as
`OptimizationResultSchema.config_snapshot`.)

In `pyphi/serialize/convert.py`, add alongside `_register_sweep_result`:

```python
def _register_campaign() -> None:
    from pyphi.campaign import CampaignTask
    from pyphi.campaign import CampaignTaskOutput
    from pyphi.campaign import CellOutput

    _ENCODERS[CampaignTask] = lambda t: schema.CampaignTaskSchema(
        task_id=t.task_id,
        kind=t.kind,
        compute=t.compute,
        compute_ref=t.compute_ref,
        config_overrides=dict(t.config_overrides),
        cells=tuple(
            (label, formalism, tuple(subset), tuple(state))
            for label, formalism, subset, state in t.cells
        ),
        skip_uncomputable=t.skip_uncomputable,
    )

    def _decode_campaign_task(s: schema.CampaignTaskSchema) -> Any:
        return CampaignTask(
            task_id=s.task_id,
            kind=s.kind,
            compute=s.compute,
            compute_ref=s.compute_ref,
            config_overrides=dict(s.config_overrides),
            cells=tuple(
                (label, formalism, tuple(subset), tuple(state))
                for label, formalism, subset, state in s.cells
            ),
            skip_uncomputable=s.skip_uncomputable,
        )

    _DECODERS[schema.CampaignTaskSchema] = _decode_campaign_task

    _ENCODERS[CellOutput] = lambda e: schema.CellOutputSchema(
        status=e.status,
        result=None if e.result is None else to_schema(e.result),
        traceback=e.traceback,
    )

    def _decode_cell_output(s: schema.CellOutputSchema) -> Any:
        return CellOutput(
            status=s.status,
            result=None if s.result is None else from_schema(s.result),
            traceback=s.traceback,
        )

    _DECODERS[schema.CellOutputSchema] = _decode_cell_output

    _ENCODERS[CampaignTaskOutput] = lambda o: schema.CampaignTaskOutputSchema(
        task_id=o.task_id,
        pyphi_version=o.pyphi_version,
        entries=tuple(to_schema(e) for e in o.entries),
    )

    def _decode_campaign_task_output(s: schema.CampaignTaskOutputSchema) -> Any:
        return CampaignTaskOutput(
            task_id=s.task_id,
            pyphi_version=s.pyphi_version,
            entries=tuple(from_schema(e) for e in s.entries),
        )

    _DECODERS[schema.CampaignTaskOutputSchema] = _decode_campaign_task_output
```

Call `_register_campaign()` next to the `_register_sweep_result()` call
(same registration block, near line 1380). If the `Schema` union in
`schema.py` is an explicit union of schema types (check how
`SweepResultSchema.results` resolves `"Schema | float"`), add the three new
schema classes to that union the same way the sweep/optimization schemas
were added.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ test/serialize/ -q > /tmp/t2b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/campaign/__init__.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/campaign/
git commit -m "Add campaign task and output types with serialization"
git log --oneline -1
```

---

### Task 3: `prepare()` — packing, admission control, directory + condor emitter

**Files:**
- Modify: `pyphi/campaign/__init__.py`
- Test: `test/campaign/test_prepare.py`

**Interfaces:**
- Consumes: Task 1's `_normalize_substrates`, `_enumerate_cells` (from
  `pyphi.sweep`); Task 2's `CampaignTask`;
  `cost_balanced_partition(weights, k)` from `pyphi.parallel.chunking`;
  `estimate_analysis` from `pyphi.cost`; `presets.by_name`;
  `pyphi.serialize.save`; `Displayable`/`Description`/`Row`/`Section` from
  `pyphi.display` (follow `AnalysisEstimate._describe` in `pyphi/cost.py`
  as the display model).
- Produces:

```python
@dataclass(frozen=True)
class CampaignStatus:  # Displayable
    directory: str
    n_tasks: int
    n_cells: int
    done: tuple[int, ...]
    failed: tuple[int, ...]
    pending: tuple[int, ...]
    total_units: float

def prepare(substrates, *, states, subsets="full", formalisms=None,
            compute="sia", directory, jobs=None, units_per_job=None,
            infeasible_threshold=1e9, strict=False,
            container_image="pyphi.sif", request_memory="4GB",
            request_disk="4GB", seed=None) -> CampaignStatus
```

Directory layout written by `prepare` (Tasks 4–5 read it):
`manifest.json`, `substrates/substrate-<label>.json.gz`,
`tasks/task-<id:04d>.json.gz`, empty `outputs/` and `logs/`,
`remaining.txt` (all task ids, one per line), `run_task.sh`, `pyphi.sub`.

Manifest keys (plain JSON): `kind` ("sweep_cells"), `pyphi_version`,
`created` (UTC ISO), `seed`, `compute` (str) / `compute_ref` (str|null),
`axes` (`{"states": ..., "subsets": ..., "formalisms": [...]}` as given,
tuples as lists), `substrate_labels`, `cells` (list of
`[label, formalism, subset, state]`), `weights` (per cell, floats),
`capped` (per cell, bools), `tasks` (list of lists of cell indices,
each inner list ascending), `skip_uncomputable`, `infeasible_threshold`,
`packing` (`{"jobs": ..., "units_per_job": ...}`).

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_prepare.py`:

```python
import json

import pytest

from pyphi import examples
from pyphi.campaign import prepare
from pyphi.serialize import load
from pyphi.warnings import PyPhiWarning

AXES = dict(states="all", subsets="full", formalisms=["IIT_4_0_2026"])


def test_prepare_writes_campaign_directory(tmp_path):
    directory = tmp_path / "camp"
    cs = prepare(
        examples.basic_substrate(), **AXES, directory=directory, units_per_job=50.0
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kind"] == "sweep_cells"
    assert len(manifest["cells"]) == 8  # 2**3 states x 1 subset x 1 formalism
    assert len(manifest["weights"]) == 8
    assert sorted(i for task in manifest["tasks"] for i in task) == list(range(8))
    assert (directory / "substrates" / "substrate-0.json.gz").exists()
    task = load(directory / "tasks" / "task-0000.json.gz")
    assert task.kind == "sweep_cells"
    assert task.skip_uncomputable is True
    assert (directory / "outputs").is_dir()
    assert (directory / "run_task.sh").stat().st_mode & 0o111
    submit = (directory / "pyphi.sub").read_text()
    assert "queue task_id from remaining.txt" in submit
    assert "container_image" in submit
    assert "pyphi.sif" in submit
    remaining = (directory / "remaining.txt").read_text().split()
    assert remaining == [str(t) for t in range(cs.n_tasks)]
    assert cs.pending == tuple(range(cs.n_tasks))
    assert cs.done == ()


def test_prepare_refuses_existing_directory(tmp_path):
    directory = tmp_path / "camp"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        prepare(examples.basic_substrate(), **AXES, directory=directory)


def test_default_packing_is_one_cell_per_task(tmp_path):
    prepare(examples.basic_substrate(), **AXES, directory=tmp_path / "c")
    manifest = json.loads((tmp_path / "c" / "manifest.json").read_text())
    assert all(len(task) == 1 for task in manifest["tasks"])


def test_jobs_and_units_per_job_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="jobs.*units_per_job|units_per_job.*jobs"):
        prepare(
            examples.basic_substrate(),
            **AXES,
            directory=tmp_path / "c",
            jobs=2,
            units_per_job=10.0,
        )


def test_jobs_packing_is_cost_balanced_and_deterministic(tmp_path):
    prepare(examples.basic_substrate(), **AXES, directory=tmp_path / "a", jobs=3)
    prepare(examples.basic_substrate(), **AXES, directory=tmp_path / "b", jobs=3)
    ma = json.loads((tmp_path / "a" / "manifest.json").read_text())
    mb = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert ma["tasks"] == mb["tasks"]
    assert len(ma["tasks"]) == 3


def test_admission_control_warns_and_strict_raises(tmp_path):
    with pytest.warns(PyPhiWarning, match="exceeds"):
        prepare(
            examples.basic_substrate(),
            **AXES,
            directory=tmp_path / "warn",
            infeasible_threshold=1.0,
        )
    with pytest.raises(ValueError, match="exceeds"):
        prepare(
            examples.basic_substrate(),
            **AXES,
            directory=tmp_path / "strict",
            infeasible_threshold=1.0,
            strict=True,
        )


def _double_phi(system):
    return 2.0


def test_callable_compute_recorded_by_reference(tmp_path):
    prepare(
        examples.basic_substrate(),
        states=[(1, 0, 0)],
        formalisms=["IIT_4_0_2026"],
        compute=_double_phi,
        directory=tmp_path / "c",
    )
    task = load(tmp_path / "c" / "tasks" / "task-0000.json.gz")
    assert task.compute is None
    assert task.compute_ref == "test.campaign.test_prepare:_double_phi"


def test_lambda_compute_rejected(tmp_path):
    with pytest.raises(ValueError, match="importable"):
        prepare(
            examples.basic_substrate(),
            states=[(1, 0, 0)],
            formalisms=["IIT_4_0_2026"],
            compute=lambda s: 0.0,
            directory=tmp_path / "c",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_prepare.py -x -q > /tmp/t3a.log 2>&1`; read the log.
Expected: FAIL with `ImportError: cannot import name 'prepare'`.

- [ ] **Step 3: Implement `prepare()`**

Add to `pyphi/campaign/__init__.py` (extend `__all__` with
`"CampaignStatus"`, `"prepare"`):

```python
import json
import math
import stat
import warnings
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path

import msgspec

import pyphi
from pyphi import serialize
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.cost import estimate_analysis
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.sweep import _enumerate_cells
from pyphi.sweep import _normalize_formalisms
from pyphi.sweep import _normalize_substrates
from pyphi.warnings import PyPhiWarning


@dataclass(frozen=True)
class CampaignStatus(Displayable):
    """A campaign's task ledger: which tasks are done, failed, or pending."""

    directory: str
    n_tasks: int
    n_cells: int
    done: tuple[int, ...]
    failed: tuple[int, ...]
    pending: tuple[int, ...]
    total_units: float

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        rows = [
            Row("Directory", self.directory),
            Row("Tasks", str(self.n_tasks)),
            Row("Cells", str(self.n_cells)),
            Row("Done", str(len(self.done))),
            Row("Failed", str(len(self.failed))),
            Row("Pending", str(len(self.pending))),
            Row("Total work units", f"{self.total_units:.3g}"),
        ]
        return Description(
            title="CampaignStatus",
            subtitle=f"{len(self.done)}/{self.n_tasks} tasks done",
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"CampaignStatus(done={len(self.done)}/{self.n_tasks}, "
                f"failed={len(self.failed)})"
            ),
        )


def _wire_overrides() -> dict[str, Any]:
    """The active configuration as JSON-compatible override kwargs."""

    def enc_hook(x: Any) -> Any:
        return dict(x) if isinstance(x, Mapping) else str(x)

    overrides = config.snapshot().as_overrides()
    return json.loads(
        json.dumps(msgspec.to_builtins(overrides, enc_hook=enc_hook))
    )


def _resolve_compute_ref(ref: str) -> Any:
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _compute_spec(compute: Any) -> tuple[str | None, str | None]:
    """Split a compute argument into (name, importable reference)."""
    if isinstance(compute, str):
        if compute not in ("sia", "ces"):
            raise ValueError(
                f"unknown compute: {compute!r}; expected 'sia', 'ces', or a callable"
            )
        return compute, None
    ref = f"{compute.__module__}:{compute.__qualname__}"
    try:
        resolved = _resolve_compute_ref(ref)
    except (ImportError, AttributeError):
        resolved = None
    if resolved is not compute:
        raise ValueError(
            f"compute callable {compute!r} is not importable as {ref!r}; "
            "campaign computations must be module-level functions "
            "(lambdas and local functions cannot ship to jobs)"
        )
    return None, ref


def _cell_weights(
    cells: list, substrate_map: dict, compute_name: str | None
) -> tuple[list[float], list[bool]]:
    """Per-cell work estimates; state-independent, so memoized per
    (label, formalism, subset)."""
    memo: dict[tuple, tuple[float, bool]] = {}
    weights, capped = [], []
    for label, formalism, subset, _state in cells:
        key = (label, formalism, subset)
        if key not in memo:
            if compute_name is None:
                memo[key] = (1.0, False)
            else:
                with config.override(
                    **presets.by_name[formalism], progress_bars=False
                ):
                    est = estimate_analysis(
                        substrate_map[label], subset=subset, compute=compute_name
                    )
                axes = (
                    est.system_partitions,
                    est.mechanisms,
                    est.purview_evaluations,
                    est.mechanism_partition_sweeps,
                )
                memo[key] = (
                    float(sum(a for a in axes if a is not None)),
                    est.capped,
                )
        w, c = memo[key]
        weights.append(w)
        capped.append(c)
    return weights, capped


def _pack(
    weights: list[float], jobs: int | None, units_per_job: float | None
) -> list[list[int]]:
    if jobs is not None and units_per_job is not None:
        raise ValueError("pass either jobs or units_per_job, not both")
    if jobs is None and units_per_job is None:
        return [[i] for i in range(len(weights))]
    if jobs is None:
        jobs = max(1, math.ceil(sum(weights) / units_per_job))
    bins = cost_balanced_partition(weights, jobs)
    return [sorted(b) for b in bins]


_SUBMIT_TEMPLATE = """\
universe            = container
container_image     = {container_image}
executable          = run_task.sh
arguments           = $(task_id)
transfer_input_files = tasks/task-$(task_id).json.gz, substrates/
transfer_output_remaps = "task-$(task_id).json.gz = outputs/task-$(task_id).json.gz"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus        = 1
request_memory      = {request_memory}
request_disk        = {request_disk}
log                 = logs/task-$(task_id).log
output              = logs/task-$(task_id).out
error               = logs/task-$(task_id).err
queue task_id from remaining.txt
"""

_RUN_TASK_SH = """\
#!/bin/bash
set -e
exec python -m pyphi.campaign run tasks/task-$1.json.gz --substrates substrates --outputs .
"""


def prepare(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    compute: Any = "sia",
    directory: Any,
    jobs: int | None = None,
    units_per_job: float | None = None,
    infeasible_threshold: float = 1e9,
    strict: bool = False,
    container_image: str = "pyphi.sif",
    request_memory: str = "4GB",
    request_disk: str = "4GB",
    seed: int | None = None,
) -> CampaignStatus:
    """Materialize a sweep as a self-contained HTCondor campaign directory.

    Enumerates exactly the cells :func:`pyphi.sweep.sweep` would run over the
    same axes, estimates each cell's workload with
    :func:`pyphi.cost.estimate_analysis` under its formalism preset, packs
    cells into cost-balanced tasks, and writes the campaign directory: one
    serialized task file per condor job, each substrate serialized once, the
    generated submit file, and a manifest recording every estimate and
    packing decision. Submit with ``condor_submit pyphi.sub`` from the
    campaign directory; monitor and collect with :func:`status` and
    :func:`collect`.

    Parameters
    ----------
    substrates
        As in :func:`pyphi.sweep.sweep`: one substrate, a sequence, or a
        ``{label: substrate}`` mapping.
    states, subsets, formalisms, compute
        As in :func:`pyphi.sweep.sweep`. A callable ``compute`` must be an
        importable module-level function.
    directory
        Target directory; created, and must not already exist.
    jobs : int, optional
        Pack cells into exactly this many cost-balanced tasks.
    units_per_job : float, optional
        Target work units per task; the task count is
        ``ceil(total / units_per_job)``. Mutually exclusive with ``jobs``;
        with neither, each cell is its own task.
    infeasible_threshold : float, optional
        A single cell whose estimate exceeds this triggers a warning naming
        the cell (or an error with ``strict``). The default marks cells
        that cannot finish in a 72-hour slot unless per-unit cost is well
        below a millisecond.
    strict : bool, optional
        Escalate admission-control warnings to errors.
    container_image, request_memory, request_disk : str, optional
        Substituted into the generated submit file.
    seed : int, optional
        Recorded in the manifest; stamped into result provenance by
        :func:`collect`.

    Returns
    -------
    CampaignStatus
        The freshly prepared ledger (all tasks pending).
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"campaign directory {directory} already exists; "
            "campaign directories are never overwritten"
        )
    compute_name, compute_ref = _compute_spec(compute)
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    formalisms_ = _normalize_formalisms(formalisms)
    cells = _enumerate_cells(labeled, states, subsets, formalisms_)
    if not cells:
        raise ValueError("the given axes enumerate no cells")
    skip_uncomputable = states == "all" or subsets == "all"

    weights, capped = _cell_weights(cells, substrate_map, compute_name)
    for cell, weight in zip(cells, weights, strict=True):
        if weight > infeasible_threshold:
            message = (
                f"cell {cell!r} estimate {weight:.3g} exceeds "
                f"infeasible_threshold {infeasible_threshold:.3g}; consider "
                "narrowing the axes or raising the threshold"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, PyPhiWarning, stacklevel=2)
    tasks = _pack(weights, jobs, units_per_job)

    directory.mkdir(parents=True)
    (directory / "outputs").mkdir()
    (directory / "logs").mkdir()
    substrates_dir = directory / "substrates"
    substrates_dir.mkdir()
    for label, substrate in labeled:
        serialize.save(substrate, substrates_dir / f"substrate-{label}.json.gz")

    tasks_dir = directory / "tasks"
    tasks_dir.mkdir()
    overrides = _wire_overrides()
    for task_id, cell_indices in enumerate(tasks):
        task = CampaignTask(
            task_id=task_id,
            kind="sweep_cells",
            compute=compute_name,
            compute_ref=compute_ref,
            config_overrides=overrides,
            cells=tuple(cells[i] for i in cell_indices),
            skip_uncomputable=skip_uncomputable,
        )
        serialize.save(task, tasks_dir / f"task-{task_id:04d}.json.gz")

    manifest = {
        "kind": "sweep_cells",
        "pyphi_version": pyphi.__version__,
        "created": datetime.now(UTC).isoformat(),
        "seed": seed,
        "compute": compute_name,
        "compute_ref": compute_ref,
        "axes": {
            "states": _axis_as_json(states),
            "subsets": _axis_as_json(subsets),
            "formalisms": list(formalisms_),
        },
        "substrate_labels": [label for label, _ in labeled],
        "cells": [
            [label, formalism, list(subset), list(state)]
            for label, formalism, subset, state in cells
        ],
        "weights": weights,
        "capped": capped,
        "tasks": tasks,
        "skip_uncomputable": skip_uncomputable,
        "infeasible_threshold": infeasible_threshold,
        "packing": {"jobs": jobs, "units_per_job": units_per_job},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "remaining.txt").write_text(
        "".join(f"{task_id}\n" for task_id in range(len(tasks)))
    )
    run_task = directory / "run_task.sh"
    run_task.write_text(_RUN_TASK_SH)
    run_task.chmod(run_task.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (directory / "pyphi.sub").write_text(
        _SUBMIT_TEMPLATE.format(
            container_image=container_image,
            request_memory=request_memory,
            request_disk=request_disk,
        )
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(tasks),
        n_cells=len(cells),
        done=(),
        failed=(),
        pending=tuple(range(len(tasks))),
        total_units=float(sum(weights)),
    )
```

Also define the helper the manifest uses, next to `_wire_overrides`:

```python
def _axis_as_json(axis: Any) -> Any:
    """An axis argument in JSON form: a mode string, or a list of lists."""
    if isinstance(axis, str):
        return axis
    if isinstance(axis, tuple) and all(isinstance(x, int) for x in axis):
        return [list(axis)]
    return [list(x) for x in axis]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -q > /tmp/t3b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/test_prepare.py
git commit -m "Add campaign prepare() with cost-balanced packing and condor emitter"
git log --oneline -1
```

---

### Task 4: Runner and `python -m pyphi.campaign`

**Files:**
- Create: `pyphi/campaign/runner.py`, `pyphi/campaign/__main__.py`
- Test: `test/campaign/test_runner.py`

**Interfaces:**
- Consumes: `CampaignTask`/`CellOutput`/`CampaignTaskOutput` (Task 2);
  `_run_cell`, `_Skipped` from `pyphi.sweep` (Task 1);
  `_resolve_compute_ref` (Task 3); `presets.by_name`;
  `pyphi.serialize.load`/`save`.
- Produces:
  - `run_task(task_path, substrates_dir, outputs_dir) -> int` in
    `pyphi.campaign.runner` (0 = all cells ok/skipped; 1 = any error cell).
  - CLI: `python -m pyphi.campaign run TASK_FILE [--substrates DIR]
    [--outputs DIR]` (defaults `substrates`, `.`), exit code from
    `run_task`.
  - Output file `task-<id:04d>.json.gz` in `outputs_dir`, written
    atomically; an existing output is renamed to
    `task-<id:04d>.attempt-<n>.json.gz` first.

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_runner.py`:

```python
import subprocess
import sys

import pytest

from pyphi import examples
from pyphi.campaign import prepare
from pyphi.campaign.runner import run_task
from pyphi.serialize import load

AXES = dict(states="all", subsets="full", formalisms=["IIT_4_0_2026"])


@pytest.fixture()
def campaign_dir(tmp_path):
    directory = tmp_path / "camp"
    prepare(examples.basic_substrate(), **AXES, directory=directory, jobs=2)
    return directory


def test_run_task_writes_output(campaign_dir):
    rc = run_task(
        campaign_dir / "tasks" / "task-0000.json.gz",
        substrates_dir=campaign_dir / "substrates",
        outputs_dir=campaign_dir / "outputs",
    )
    assert rc == 0
    out = load(campaign_dir / "outputs" / "task-0000.json.gz")
    task = load(campaign_dir / "tasks" / "task-0000.json.gz")
    assert out.task_id == 0
    assert len(out.entries) == len(task.cells)
    assert all(e.status in ("ok", "skipped") for e in out.entries)
    assert any(e.status == "ok" for e in out.entries)


def test_runner_cli_via_subprocess(campaign_dir):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyphi.campaign",
            "run",
            str(campaign_dir / "tasks" / "task-0001.json.gz"),
            "--substrates",
            str(campaign_dir / "substrates"),
            "--outputs",
            str(campaign_dir / "outputs"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (campaign_dir / "outputs" / "task-0001.json.gz").exists()


def _exploding(system):
    raise RuntimeError("deliberate test failure")


def test_error_cell_recorded_and_exit_nonzero(tmp_path):
    directory = tmp_path / "camp"
    prepare(
        examples.basic_substrate(),
        states=[(1, 0, 0), (0, 0, 0)],
        formalisms=["IIT_4_0_2026"],
        compute=_exploding,
        directory=directory,
        jobs=1,
    )
    rc = run_task(
        directory / "tasks" / "task-0000.json.gz",
        substrates_dir=directory / "substrates",
        outputs_dir=directory / "outputs",
    )
    assert rc == 1
    out = load(directory / "outputs" / "task-0000.json.gz")
    assert [e.status for e in out.entries] == ["error", "error"]
    assert "deliberate test failure" in out.entries[0].traceback


def test_rerun_renames_previous_attempt(campaign_dir):
    task_path = campaign_dir / "tasks" / "task-0000.json.gz"
    kwargs = dict(
        substrates_dir=campaign_dir / "substrates",
        outputs_dir=campaign_dir / "outputs",
    )
    run_task(task_path, **kwargs)
    run_task(task_path, **kwargs)
    assert (campaign_dir / "outputs" / "task-0000.json.gz").exists()
    assert (campaign_dir / "outputs" / "task-0000.attempt-1.json.gz").exists()


def test_config_overrides_recorded_in_task(tmp_path):
    # Prepare under a modified precision; the task file must carry it.
    import pyphi

    directory = tmp_path / "camp2"
    with pyphi.config.override(precision=7):
        prepare(examples.basic_substrate(), **AXES, directory=directory, jobs=1)
    task = load(directory / "tasks" / "task-0000.json.gz")
    assert task.config_overrides["precision"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_runner.py -x -q > /tmp/t4a.log 2>&1`; read the log.
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.campaign.runner'`.

- [ ] **Step 3: Implement the runner**

`pyphi/campaign/runner.py`:

```python
"""Execute one campaign task file and write its output document.

The runner is a fixed entry point (``python -m pyphi.campaign run``) that
behaves identically inside the campaign's container, in a local shell, and
under test: it loads the task, loads the substrates it references, installs
the shipped configuration beneath each cell's formalism preset, runs the
cells in order, and atomically writes one output document holding a per-cell
outcome. The process exit code is nonzero when any cell errored, so
scheduler logs reflect failures, but the output document is written in every
case.
"""

from __future__ import annotations

import os
import traceback as _traceback
from pathlib import Path
from typing import Any

import pyphi
from pyphi import serialize
from pyphi.campaign import CampaignTask
from pyphi.campaign import CampaignTaskOutput
from pyphi.campaign import CellOutput
from pyphi.campaign import _resolve_compute_ref
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.sweep import _run_cell
from pyphi.sweep import _Skipped

__all__ = ["run_task"]


def _load_substrates(task: CampaignTask, substrates_dir: Path) -> dict:
    labels = {cell[0] for cell in task.cells}
    return {
        label: serialize.load(substrates_dir / f"substrate-{label}.json.gz")
        for label in labels
    }


def _write_output(output: CampaignTaskOutput, outputs_dir: Path) -> None:
    final = outputs_dir / f"task-{output.task_id:04d}.json.gz"
    if final.exists():
        n = 1
        while (
            attempt := final.with_name(
                f"task-{output.task_id:04d}.attempt-{n}.json.gz"
            )
        ).exists():
            n += 1
        final.rename(attempt)
    # Temp name keeps the .json.gz suffixes so format inference is unchanged;
    # os.replace makes the final path appear atomically.
    tmp = final.with_name(f".tmp-{final.name}")
    serialize.save(output, tmp)
    os.replace(tmp, final)


def run_task(
    task_path: Any,
    substrates_dir: Any = "substrates",
    outputs_dir: Any = ".",
) -> int:
    """Run one task file; return 0 if every cell is ok or skipped, else 1.

    Parameters
    ----------
    task_path
        Path to a serialized campaign task.
    substrates_dir
        Directory holding the campaign's serialized substrates.
    outputs_dir
        Directory to write ``task-<id>.json.gz`` into (atomically; a
        pre-existing output is preserved under an ``attempt-<n>`` name).
    """
    task = serialize.load(task_path)
    substrates = _load_substrates(task, Path(substrates_dir))
    compute = (
        task.compute if task.compute is not None
        else _resolve_compute_ref(task.compute_ref)
    )
    entries: list[CellOutput] = []
    failed = False
    for label, formalism, subset, state in task.cells:
        overrides = {
            **task.config_overrides,
            **presets.by_name[formalism],
            "parallel": False,
            "progress_bars": False,
        }
        try:
            with config.override(**overrides):
                result = _run_cell(
                    (label, subset, state),
                    substrates=substrates,
                    compute=compute,
                    skip=task.skip_uncomputable,
                )
            if isinstance(result, _Skipped):
                entries.append(
                    CellOutput(status="skipped", result=None, traceback=None)
                )
            else:
                entries.append(CellOutput(status="ok", result=result, traceback=None))
        except Exception:  # noqa: BLE001 — every cell failure becomes data
            entries.append(
                CellOutput(
                    status="error", result=None, traceback=_traceback.format_exc()
                )
            )
            failed = True
    output = CampaignTaskOutput(
        task_id=task.task_id,
        pyphi_version=pyphi.__version__,
        entries=tuple(entries),
    )
    _write_output(output, Path(outputs_dir))
    return 1 if failed else 0
```

`pyphi/campaign/__main__.py`:

```python
"""Command-line entry point: ``python -m pyphi.campaign run TASK_FILE``."""

from __future__ import annotations

import argparse
import sys

from pyphi.campaign.runner import run_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyphi.campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute one campaign task file")
    run_parser.add_argument("task_file")
    run_parser.add_argument("--substrates", default="substrates")
    run_parser.add_argument("--outputs", default=".")
    args = parser.parse_args(argv)
    return run_task(args.task_file, args.substrates, args.outputs)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -q > /tmp/t4b.log 2>&1`; read the log.
Expected: PASS. The subprocess test exercises the real CLI.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/runner.py pyphi/campaign/__main__.py test/campaign/test_runner.py
git commit -m "Add campaign runner and python -m pyphi.campaign CLI"
git log --oneline -1
```

---

### Task 5: `status()` and `collect()` — including the headline invariant

**Files:**
- Modify: `pyphi/campaign/__init__.py`
- Test: `test/campaign/test_collect.py`

**Interfaces:**
- Consumes: manifest layout (Task 3), output documents (Task 4),
  `_extract_row`/`_build_df` from `pyphi.sweep` (Task 1), `SweepResult`.
- Produces:
  - `status(directory) -> CampaignStatus` — classifies every task, rewrites
    `remaining.txt` with pending+failed ids (ascending).
  - `collect(directory, partial=False) -> SweepResult` — deterministic cell
    order; raises `RuntimeError` on incomplete campaigns unless `partial`.

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_collect.py`:

```python
import subprocess
import sys

import pandas as pd
import pytest

from pyphi import examples
from pyphi.campaign import collect
from pyphi.campaign import prepare
from pyphi.campaign import status
from pyphi.sweep import sweep
from pyphi.warnings import PyPhiWarning

AXES = dict(states="all", subsets="full", formalisms=["IIT_4_0_2026"], compute="sia")


def _run_all_tasks(directory):
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyphi.campaign",
                "run",
                str(task_file),
                "--substrates",
                str(directory / "substrates"),
                "--outputs",
                str(directory / "outputs"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


@pytest.fixture(scope="module")
def executed_campaign(tmp_path_factory):
    directory = tmp_path_factory.mktemp("campaign") / "camp"
    substrates = {"basic": examples.basic_substrate(), "xor": examples.xor_substrate()}
    prepare(substrates, **AXES, directory=directory, units_per_job=100.0, seed=7)
    _run_all_tasks(directory)
    return directory, substrates


def test_campaign_equals_local_sweep(executed_campaign):
    directory, substrates = executed_campaign
    local = sweep(substrates, **AXES, parallel=False, progress=False, seed=7)
    result = collect(directory)
    pd.testing.assert_frame_equal(result.df, local.df)
    assert [float(r.phi) for r in result.results] == [
        float(r.phi) for r in local.results
    ]
    assert result.skipped == local.skipped


def test_status_after_execution(executed_campaign):
    directory, _ = executed_campaign
    st = status(directory)
    assert st.failed == ()
    assert st.pending == ()
    assert len(st.done) == st.n_tasks
    assert (directory / "remaining.txt").read_text() == ""


def test_missing_output_is_pending_and_resubmittable(tmp_path):
    directory = tmp_path / "camp"
    prepare(examples.basic_substrate(), **AXES, directory=directory, jobs=2)
    _run_all_tasks(directory)
    (directory / "outputs" / "task-0001.json.gz").unlink()
    st = status(directory)
    assert st.pending == (1,)
    assert (directory / "remaining.txt").read_text() == "1\n"
    with pytest.raises(RuntimeError, match="incomplete"):
        collect(directory)
    with pytest.warns(PyPhiWarning):
        partial = collect(directory, partial=True)
    assert len(partial.df) + len(partial.skipped) <= 8
    assert len(partial.df) >= 1


def _exploding(system):
    raise RuntimeError("boom")


def test_failed_task_listed_for_resubmission(tmp_path):
    directory = tmp_path / "camp"
    prepare(
        examples.basic_substrate(),
        states=[(1, 0, 0)],
        formalisms=["IIT_4_0_2026"],
        compute=_exploding,
        directory=directory,
    )
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyphi.campaign",
                "run",
                str(task_file),
                "--substrates",
                str(directory / "substrates"),
                "--outputs",
                str(directory / "outputs"),
            ],
            capture_output=True,
        )
    st = status(directory)
    assert st.failed == (0,)
    assert (directory / "remaining.txt").read_text() == "0\n"
```

Remove the unused `task0_cells` line if ruff flags it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_collect.py -x -q > /tmp/t5a.log 2>&1`; read the log.
Expected: FAIL with `ImportError: cannot import name 'collect'`.

- [ ] **Step 3: Implement `status()` and `collect()`**

Add to `pyphi/campaign/__init__.py` (extend `__all__` with `"status"`,
`"collect"`):

```python
def _load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _manifest_cells(manifest: dict) -> list[tuple[Any, str, tuple, tuple]]:
    return [
        (label, formalism, tuple(subset), tuple(state))
        for label, formalism, subset, state in manifest["cells"]
    ]


def status(directory: Any) -> CampaignStatus:
    """Classify every task from the output files and refresh ``remaining.txt``.

    A task is done exactly when its output file exists, loads, and every
    entry is ``ok`` or ``skipped``; it is failed when the output loads with
    an ``error`` entry or does not load; otherwise it is pending. Pending
    and failed task ids are rewritten to ``remaining.txt``, so resubmission
    is running ``condor_submit pyphi.sub`` again.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    done, failed, pending = [], [], []
    for task_id in range(len(manifest["tasks"])):
        path = directory / "outputs" / f"task-{task_id:04d}.json.gz"
        if not path.exists():
            pending.append(task_id)
            continue
        try:
            output = serialize.load(path)
        except Exception:  # noqa: BLE001 — an unloadable output is a failed task
            failed.append(task_id)
            continue
        if any(entry.status == "error" for entry in output.entries):
            failed.append(task_id)
        else:
            done.append(task_id)
    (directory / "remaining.txt").write_text(
        "".join(f"{task_id}\n" for task_id in sorted(pending + failed))
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(manifest["tasks"]),
        n_cells=len(manifest["cells"]),
        done=tuple(done),
        failed=tuple(failed),
        pending=tuple(pending),
        total_units=float(sum(manifest["weights"])),
    )


def collect(directory: Any, partial: bool = False) -> SweepResult:
    """Reassemble the campaign's outputs into the local-sweep result.

    Cells are restored to their preparation order, so the result is
    identical to what :func:`pyphi.sweep.sweep` returns over the same axes.
    With missing or failed tasks the default is to raise with a per-task
    summary; ``partial=True`` instead warns and returns the result built
    from the completed tasks.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    st = status(directory)
    incomplete = sorted(set(st.failed) | set(st.pending))
    if incomplete:
        summary = (
            f"{len(incomplete)} of {st.n_tasks} tasks incomplete "
            f"(failed: {list(st.failed)}, pending: {list(st.pending)}); "
            "resubmit with condor_submit pyphi.sub"
        )
        if not partial:
            raise RuntimeError(summary)
        warnings.warn(summary, PyPhiWarning, stacklevel=2)
    incomplete_set = set(incomplete)

    cells = _manifest_cells(manifest)
    compute = manifest["compute"] if manifest["compute"] is not None else "callable"
    by_index: dict[int, tuple[str, Any]] = {}
    for task_id, cell_indices in enumerate(manifest["tasks"]):
        if task_id in incomplete_set:
            continue
        output = serialize.load(directory / "outputs" / f"task-{task_id:04d}.json.gz")
        for cell_index, entry in zip(cell_indices, output.entries, strict=True):
            by_index[cell_index] = (entry.status, entry.result)

    keys, raw, skipped = [], [], []
    for cell_index in sorted(by_index):
        cell = cells[cell_index]
        entry_status, result = by_index[cell_index]
        if entry_status == "skipped":
            skipped.append(cell)
        else:
            keys.append(cell)
            raw.append(result)

    if manifest["seed"] is not None:
        for result in raw:
            with_provenance = getattr(result, "with_provenance", None)
            if with_provenance is not None:
                with_provenance(seed=manifest["seed"])

    rows = [_extract_row(result, compute) for result in raw]
    df = _build_df(keys, rows, cells)
    return SweepResult(df=df, results=raw, skipped=skipped)
```

Add the imports this needs at the top of the file:
`from pyphi.sweep import SweepResult`, `from pyphi.sweep import _build_df`,
`from pyphi.sweep import _extract_row`.

- [ ] **Step 4: Run the full campaign suite**

Run: `uv run pytest test/campaign/ test/test_sweep.py -q > /tmp/t5b.log 2>&1`; read the log.
Expected: PASS, including `test_campaign_equals_local_sweep`.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/test_collect.py
git commit -m "Add campaign status() and collect() with exact sweep reconstruction"
git log --oneline -1
```

---

### Task 6: MCP tools and content page

**Files:**
- Modify: `pyphi/mcp/server.py`, `pyphi/mcp/content.py` (TOPICS),
  `pyphi/mcp/content/parallelization.md` (pointer line)
- Create: `pyphi/mcp/content/campaigns.md`
- Test: `test/mcp/test_server.py`

**Interfaces:**
- Consumes: `pyphi.campaign.prepare/status/collect` (Tasks 3+5); the
  server's `_get_substrate`, `_register_result`, `_result_summary` helpers
  and `@mcp.tool()` decorator; `dataclasses.asdict`.
- Produces: MCP tools `prepare_campaign`, `campaign_status`,
  `collect_campaign`; TOPICS entry `"campaigns"`.

- [ ] **Step 1: Write the failing tests**

Add to `test/mcp/test_server.py`, following the file's existing style for
invoking tools (inspect how existing tests call tools — direct function
calls on the imported module — and mirror it):

```python
class TestCampaignTools:
    def test_prepare_status_collect_roundtrip(self, tmp_path):
        handle = server.load_example("basic")["handle"]
        directory = tmp_path / "camp"
        prepared = server.prepare_campaign(
            handles=[handle],
            states="all",
            formalisms=["IIT_4_0_2026"],
            directory=str(directory),
            jobs=2,
        )
        assert prepared["status"]["n_tasks"] == 2
        assert "card" in prepared

        # Execute the tasks locally (the runner, not condor).
        from pyphi.campaign.runner import run_task

        for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
            assert (
                run_task(
                    task_file,
                    substrates_dir=directory / "substrates",
                    outputs_dir=directory / "outputs",
                )
                == 0
            )

        st = server.campaign_status(directory=str(directory))
        # dataclasses.asdict preserves tuples in-process; check emptiness,
        # not list equality.
        assert not st["status"]["failed"]
        assert not st["status"]["pending"]

        collected = server.collect_campaign(directory=str(directory))
        assert "result_ref" in collected
        assert collected["rows"] >= 1
```

(Adjust the `server.` access path to match how the existing tests import
and call tools — e.g. if tools are accessed via `server.<name>.fn` under
FastMCP, follow that pattern exactly.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/mcp/test_server.py -k Campaign -x -q > /tmp/t6a.log 2>&1`; read the log.
Expected: FAIL with `AttributeError` (no `prepare_campaign`).

- [ ] **Step 3: Implement the tools**

In `pyphi/mcp/server.py`, after `estimate_cost`:

```python
@mcp.tool()
def prepare_campaign(
    handles: list[str],
    directory: str,
    states: Any = "all",
    subsets: Any = "full",
    formalisms: list[str] | None = None,
    compute: str = "sia",
    jobs: int | None = None,
    units_per_job: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Materialize a sweep as an HTCondor campaign directory.

    Enumerates the sweep cells over the given substrate handles, packs them
    into cost-balanced condor tasks, and writes a self-contained campaign
    directory (task files, substrates, submit file). The user submits it
    with ``condor_submit pyphi.sub``; monitor with ``campaign_status`` and
    reassemble results with ``collect_campaign``. See the ``campaigns``
    reference topic for the workflow.

    Parameters
    ----------
    handles : list of str
        Substrate handles from ``load_example`` or ``build_substrate``; the
        handle strings become the substrate labels in the result.
    directory : str
        Target campaign directory; must not already exist.
    states, subsets, formalisms, compute
        Sweep axes, as in the library's sweep: explicit lists, or ``"all"``
        / ``"full"``.
    jobs : int, optional
        Pack into exactly this many cost-balanced tasks.
    units_per_job : float, optional
        Target work units per task (mutually exclusive with ``jobs``).
    seed : int, optional
        Recorded in the manifest and stamped into provenance at collection.

    Returns
    -------
    dict
        A ``card`` (human-readable summary) and a ``status`` mapping with
        the task ledger.
    """
    from pyphi import campaign

    substrates = {handle: _get_substrate(handle) for handle in handles}
    states_ = states if isinstance(states, str) else [tuple(s) for s in states]
    subsets_ = subsets if isinstance(subsets, str) else [tuple(s) for s in subsets]
    result = campaign.prepare(
        substrates,
        states=states_,
        subsets=subsets_,
        formalisms=formalisms,
        compute=compute,
        directory=directory,
        jobs=jobs,
        units_per_job=units_per_job,
        seed=seed,
    )
    return {"card": str(result), "status": asdict(result)}


@mcp.tool()
def campaign_status(directory: str) -> dict[str, Any]:
    """Report a campaign's task ledger and refresh its resubmission list.

    Classifies every task purely from the campaign directory's output
    files — done, failed, or pending — and rewrites ``remaining.txt`` so
    that resubmitting is running ``condor_submit pyphi.sub`` again.

    Parameters
    ----------
    directory : str
        A campaign directory written by ``prepare_campaign``.

    Returns
    -------
    dict
        A ``card`` and a ``status`` mapping with the task ledger.
    """
    from pyphi import campaign

    result = campaign.status(directory)
    return {"card": str(result), "status": asdict(result)}


@mcp.tool()
def collect_campaign(directory: str, partial: bool = False) -> dict[str, Any]:
    """Reassemble a campaign's outputs into a sweep result.

    Returns the identical result a local sweep over the same axes would
    produce, registered as a result handle for ``inspect``.

    Parameters
    ----------
    directory : str
        A campaign directory whose tasks have been executed.
    partial : bool
        Return the result built from completed tasks even when some are
        missing or failed (default: raise with a per-task summary).

    Returns
    -------
    dict
        The ``result_ref`` handle, the number of collected ``rows``, and
        the number of ``skipped`` cells.
    """
    from pyphi import campaign

    result = campaign.collect(directory, partial=partial)
    ref = _register_result(result)
    return {
        "result_ref": ref,
        "rows": int(len(result.df)),
        "skipped": len(result.skipped),
    }
```

(`asdict` is already imported in `server.py`; `Any` likewise — verify and
add imports only if missing.)

- [ ] **Step 4: Write the content page and register it**

`pyphi/mcp/content/campaigns.md` — write ~60–100 lines covering: what a
campaign is (tasks-as-data, no coordinator, done = output exists and
loads); the workflow (`prepare_campaign` → copy directory to the access
point → `condor_submit pyphi.sub` → `campaign_status` → resubmit →
`collect_campaign`); when to use a campaign versus the Dask backend (many
independent cells and no live worker pool ↔ interactive distribution with
an active client); packing (`jobs` / `units_per_job`, admission-control
warnings); and the failure model (per-cell error entries, task-granularity
resubmission). Plain prose, present tense, no planning-artifact references.

In `pyphi/mcp/content.py`, add to `TOPICS` after `"performance"`:

```python
    "campaigns": (
        "campaigns.md",
        "Distributing sweeps across an HTCondor cluster: prepare, submit, "
        "monitor, and collect batch campaigns.",
    ),
```

In `pyphi/mcp/content/parallelization.md`, add one sentence in the section
that discusses clusters, pointing to the `campaigns` topic for HTCondor
batch campaigns.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/mcp/ -q > /tmp/t6b.log 2>&1`; read the log.
Expected: PASS (existing MCP tests must stay green — some assert on the
full tool list; update those lists if they enumerate tools).

- [ ] **Step 6: Commit**

```bash
git add pyphi/mcp/server.py pyphi/mcp/content.py pyphi/mcp/content/campaigns.md pyphi/mcp/content/parallelization.md test/mcp/test_server.py
git commit -m "Add campaign MCP tools and reference content"
git log --oneline -1
```

---

### Task 7: Documentation, changelog, ROADMAP, full-suite gate

**Files:**
- Create: `docs/howto/campaigns.md`, `changelog.d/campaign.feature.md`
- Modify: `docs/howto/index.md` (toctree), `docs/howto/chtc.md`,
  `docs/howto/parallel.md`, `ROADMAP.md` (P11 row)

**Interfaces:**
- Consumes: the complete library surface from Tasks 1–6.

- [ ] **Step 1: Write `docs/howto/campaigns.md`**

Structure (write real content under each heading, in the how-to style of
`docs/howto/chtc.md` — imperative, concrete commands, no first person):

1. `# Run a sweep as an HTCondor campaign` — what a campaign is; the
   lifecycle diagram in words (prepare → submit → status → collect).
2. `## Prepare the campaign` — a worked `prepare()` example with two
   substrates, `units_per_job`, the returned `CampaignStatus`, and what the
   directory contains.
3. `## Submit` — copy the directory to the access point,
   `condor_submit pyphi.sub`; note the container image requirement and
   point to the container-build section of the CHTC guide.
4. `## Monitor and resubmit` — `status()`; done/failed/pending semantics;
   resubmission is `condor_submit pyphi.sub` again.
5. `## Collect` — `collect()` returning the exact local-sweep result;
   `partial=True`; seed stamping.
6. `## Running tasks without condor` — `python -m pyphi.campaign run` for
   local execution and testing.
7. `## Packing and admission control` — `jobs`/`units_per_job`, what work
   units are (counts, not seconds), the `infeasible_threshold` warning.

- [ ] **Step 2: Wire the docs**

- `docs/howto/index.md`: add `campaigns` to the toctree after `chtc`.
- `docs/howto/chtc.md`: replace Pattern A's body with a short paragraph
  pointing to the campaigns how-to as the supported workflow, keeping a
  condensed version of the manual per-job recipe as a "rolling your own"
  note. Leave Patterns B/C untouched.
- `docs/howto/parallel.md`: in the "Running on a cluster" section, add one
  sentence pointing to the campaigns page for HTCondor batch campaigns.

Verify: `just docs > /tmp/docs.log 2>&1`; read the end of the log for
"build succeeded" (warnings about the pre-existing `whats-new-in-2.0.md`
orphan are a known issue owned by another session — do not touch that
file).

- [ ] **Step 3: Changelog fragment**

```bash
printf '%s\n' 'Added `pyphi.campaign`: distribute sweeps across an HTCondor pool as self-contained batch campaigns — `prepare()` writes a campaign directory with cost-balanced task packing and a generated submit file, `python -m pyphi.campaign run` executes one task, and `status()`/`collect()` reassemble the exact local-sweep result from per-task output files. Includes `prepare_campaign`/`campaign_status`/`collect_campaign` MCP tools and a campaigns how-to.' > changelog.d/campaign.feature.md
```

- [ ] **Step 4: Update the ROADMAP P11 row**

In `ROADMAP.md`, edit the `P11 cluster backends` row: keep 🟡 partial;
replace the "Remaining:" clause so it records that the campaign
infrastructure + sweep-cell task type landed (substrates axis on `sweep()`,
`pyphi.campaign` prepare/status/collect + runner, cost-balanced packing
with admission control, MCP tools, campaigns how-to) and that the remaining
work is the CES-sharding cycle per
`docs/superpowers/specs/2026-07-20-ces-sharding-design.md`.

- [ ] **Step 5: Full-suite verification gate**

Run the complete suite with no path argument (collects doctests in
`pyphi/`):

```bash
uv run pytest -q > /tmp/full.log 2>&1
```

Read the summary line of `/tmp/full.log`. Expected: 0 failures (skips are
fine). If a doctest in a module this plan touched fails, fix the docstring.

- [ ] **Step 6: Commit**

```bash
git add docs/howto/campaigns.md docs/howto/index.md docs/howto/chtc.md docs/howto/parallel.md changelog.d/campaign.feature.md ROADMAP.md
git commit -m "Document HTCondor campaigns and update roadmap"
git log --oneline -1
```
