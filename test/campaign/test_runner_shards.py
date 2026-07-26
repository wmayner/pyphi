import json
from dataclasses import replace

import pytest

from pyphi import examples
from pyphi.campaign import prepare_ces
from pyphi.campaign.runner import _shard_config
from pyphi.campaign.runner import run_task
from pyphi.serialize import load

BASIC_STATE = (1, 0, 0)


def _run_all(directory):
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        rc = run_task(
            task_file,
            substrates_dir=directory / "substrates",
            outputs_dir=directory / "outputs",
        )
        assert rc == 0


def test_shard_outputs_align_with_items(tmp_path):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
    )
    _run_all(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    saw_stride_aux = False
    for row in manifest["tasks"]:
        task = load(directory / "tasks" / f"task-{row['task_id']:04d}.json.gz")
        out = load(directory / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        if row["kind"] == "sia_shard":
            assert len(out.entries) == 1
            assert out.entries[0].aux is not None
            assert "tie_indices" in out.entries[0].aux
        elif task.spec.payload_kind == "mechanisms":
            assert len(out.entries) == len(task.spec.mechanisms)
        elif task.spec.payload_kind == "purview_range":
            assert len(out.entries) == len(task.spec.purviews)
        elif task.spec.payload_kind == "partition_stride":
            assert len(out.entries) == 1
            aux = out.entries[0].aux
            assert aux is not None and "tie_indices" in aux
            saw_stride_aux = True
    assert saw_stride_aux


def test_shard_execution_bounds_caches_by_its_memory_request(tmp_path):
    """Every CES shard runs with a cache ceiling inside its own request."""
    from pyphi.cost import shard_cache_budget_bytes

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
    )
    seen = 0
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        task = load(task_file)
        if task.kind != "ces_shard":
            continue
        budget = _shard_config(task)["memory_ceiling_bytes"]
        assert budget == shard_cache_budget_bytes(task.spec.memory_bytes)
        assert 0 < budget < task.spec.memory_bytes
        # A ceiling configured at preparation time wins over the derived one.
        pinned = replace(
            task,
            config_overrides={
                **task.config_overrides,
                "memory_ceiling_bytes": 7,
            },
        )
        assert _shard_config(pinned)["memory_ceiling_bytes"] == 7
        seen += 1
    assert seen


def test_granted_memory_sets_the_cache_ceiling(tmp_path, monkeypatch):
    """The request the scheduler was asked for beats the planned one.

    With no cgroup limit to read — a pool that grants memory without
    confining the job to it — the exported request is the only signal that
    a resubmission was given more room than planning predicted.
    """
    from pyphi.cache import cache_utils
    from pyphi.cost import shard_cache_budget_bytes

    monkeypatch.setattr(cache_utils, "_cgroup_memory_limit", lambda: None)

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
        request_memory="1GB",
    )
    task = next(
        t
        for t in map(load, sorted((directory / "tasks").glob("task-*.json.gz")))
        if t.kind == "ces_shard"
    )
    monkeypatch.setenv("PYPHI_SHARD_MEMORY", "8GB")
    assert _shard_config(task)["memory_ceiling_bytes"] == shard_cache_budget_bytes(
        8 * 1024**3
    )
    # Anything unparseable falls back to the planned request rather than
    # leaving the caches unbounded.
    monkeypatch.setenv("PYPHI_SHARD_MEMORY", "lots")
    assert _shard_config(task)["memory_ceiling_bytes"] == shard_cache_budget_bytes(
        task.spec.memory_bytes
    )


def test_shard_results_are_unchanged_by_a_binding_cache_ceiling(tmp_path):
    """A ceiling that evicts throughout a shard's run does not alter results."""
    from pyphi.cache import cache_utils
    from pyphi.core import repertoire_algebra

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
    )
    _run_all(directory)
    unbounded = {
        f.name: load(f) for f in sorted((directory / "outputs").glob("task-*.json.gz"))
    }

    bounded_dir = tmp_path / "camp_bounded"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=bounded_dir,
        units_per_job=5.0,
    )
    repertoire_algebra.clear_caches()
    real_memory_full = cache_utils.memory_full
    cache_utils.memory_full = lambda: True
    try:
        _run_all(bounded_dir)
    finally:
        cache_utils.memory_full = real_memory_full
        repertoire_algebra.clear_caches()

    for name, expected in unbounded.items():
        got = load(bounded_dir / "outputs" / name)
        assert [e.status for e in got.entries] == [e.status for e in expected.entries]
        assert [str(e.result) for e in got.entries] == [
            str(e.result) for e in expected.entries
        ]


def test_bottleneck_ordering_gives_same_results(tmp_path):
    a = tmp_path / "plain"
    b = tmp_path / "ordered"
    for directory, ordering in ((a, None), (b, "bottleneck_first")):
        prepare_ces(
            examples.basic_substrate(),
            states=BASIC_STATE,
            formalisms="IIT_4_0_2026",
            directory=directory,
            units_per_job=5.0,
            ordering=ordering,
        )
        _run_all(directory)
    manifest = json.loads((a / "manifest.json").read_text())
    for row in manifest["tasks"]:
        oa = load(a / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        ob = load(b / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        for ea, eb in zip(oa.entries, ob.entries, strict=True):
            if ea.result is not None and hasattr(ea.result, "phi"):
                assert float(ea.result.phi) == float(eb.result.phi)


def test_order_cap_agrees_between_planning_and_execution(tmp_path):
    """Every collected distinction's purviews obey the per-order cap."""
    from pyphi.campaign import collect
    from pyphi.campaign.scope import CESScope

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        scope=CESScope(max_purview_order_by_mechanism_order=((1, 1), (2, 2))),
        directory=directory,
        units_per_job=1e9,
    )
    _run_all(directory)
    result = collect(directory)
    caps = {1: 1, 2: 2}
    for d in result.distinctions:
        cap = caps.get(len(d.mechanism))
        if cap is not None:
            assert len(d.cause.purview) <= cap
            assert len(d.effect.purview) <= cap


def test_shard_output_records_what_the_shard_cost(tmp_path):
    """Every task's output carries the observed cost of running it."""
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    _run_all(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    seen_kinds = set()
    for row in manifest["tasks"]:
        out = load(directory / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        metrics = out.metrics
        assert metrics is not None
        assert metrics["cpu_s"] >= 0
        assert metrics["wall_s"] >= metrics["cpu_s"] * 0.5
        assert metrics["cache_hits"] >= 0
        assert metrics["cache_misses"] > 0
        assert metrics["cache_evictions"] == 0
        if row["kind"] == "ces_shard":
            # The planned charge travels with the observed cost, so a
            # campaign's outputs alone calibrate units against runtime.
            assert metrics["units"] == row["units"]
            assert metrics["memory_bytes"] == row["memory_bytes"]
            seen_kinds.add(metrics["payload_kind"])
    assert "mechanisms" in seen_kinds


def _ring_substrate(units, radius=2):
    """Periodic Ising chain — mechanisms local enough to have small purviews."""
    import numpy as np

    from pyphi.substrate_generator import build_substrate
    from pyphi.substrate_generator import ising

    weights = np.zeros((units, units))
    for i in range(units):
        for d in range(-radius, radius + 1):
            if d:
                weights[i, (i + d) % units] = 1.0
    return build_substrate([ising.probability] * units, weights, temperature=0.25)


@pytest.mark.slow
def test_planned_units_predict_runtime_across_shard_forms(tmp_path):
    """Shards packed to one budget take comparable time per unit.

    A unit has to mean the same amount of work whichever ladder rung produced
    the shard carrying it, or ``units_per_job`` cannot bound runtime. The
    fixture exercises whole-mechanism packing and purview-range splitting
    together, which is where the two forms' cost per unit can diverge.
    """
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope
    from pyphi.cost import runtime_seconds

    directory = tmp_path / "camp"
    units = 6
    prepare_ces(
        _ring_substrate(units),
        states=(0,) * units,
        formalisms="IIT_4_0_2026",
        scope=CESScope(
            mechanisms=AxisScope(max_order=4),
            cause_purviews=AxisScope(max_order=3),
            effect_purviews=AxisScope(max_order=3),
        ),
        directory=directory,
        units_per_job=4000.0,
    )
    _run_all(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    observed = {}
    for row in manifest["tasks"]:
        if row["kind"] != "ces_shard":
            continue
        metrics = load(
            directory / "outputs" / f"task-{row['task_id']:04d}.json.gz"
        ).metrics
        observed.setdefault(metrics["payload_kind"], []).append(
            metrics["cpu_s"] / metrics["units"]
        )
    assert {"mechanisms", "purview_range"} <= set(observed), sorted(observed)
    per_form = {kind: sorted(rates)[len(rates) // 2] for kind, rates in observed.items()}
    slowest, fastest = max(per_form.values()), min(per_form.values())
    assert slowest / fastest < 4.0, per_form
    # And the calibration is in the right decade for this hardware, so a
    # runtime target translates into a usable budget.
    assert 0.05 < runtime_seconds(1.0) / slowest < 20.0, per_form
