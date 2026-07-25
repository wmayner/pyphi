import json
from dataclasses import replace

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
        budget = _shard_config(task)["maximum_cache_memory_bytes"]
        assert budget == shard_cache_budget_bytes(task.spec.memory_bytes)
        assert 0 < budget < task.spec.memory_bytes
        # A ceiling configured at preparation time wins over the derived one.
        pinned = replace(
            task,
            config_overrides={
                **task.config_overrides,
                "maximum_cache_memory_bytes": 7,
            },
        )
        assert _shard_config(pinned)["maximum_cache_memory_bytes"] == 7
        seen += 1
    assert seen


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
