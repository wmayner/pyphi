import json

from pyphi import examples
from pyphi.campaign import prepare_ces
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
