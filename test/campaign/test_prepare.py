import json
import os

import pytest

from pyphi import examples
from pyphi.campaign import prepare
from pyphi.serialize import load
from pyphi.warnings import PyPhiWarning

AXES = {"states": "all", "subsets": "full", "formalisms": ["IIT_4_0_2026"]}


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
    assert (directory / "substrates" / "substrate-0.msgpack.gz").exists()
    task = load(directory / "tasks" / "task-0000.json.gz")
    assert task.kind == "sweep_cells"
    assert task.skip_uncomputable is True
    assert (directory / "outputs").is_dir()
    if os.name == "posix":  # Windows has no executable bit to set
        assert (directory / "run_task.sh").stat().st_mode & 0o111
    submit = (directory / "pyphi.sub").read_text()
    assert "queue task_id, memory from remaining.txt" in submit
    assert "container_image" in submit
    assert "pyphi.sif" in submit
    remaining = (directory / "remaining.txt").read_text().splitlines()
    assert [line.split(",")[0] for line in remaining] == [
        str(t) for t in range(cs.n_tasks)
    ]
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
    with pytest.raises(ValueError, match=r"jobs.*units_per_job|units_per_job.*jobs"):
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
            compute=lambda _s: 0.0,
            directory=tmp_path / "c",
        )


def test_sweep_scaffold_writes_uniform_memory_column(tmp_path):
    directory = tmp_path / "camp"
    prepare(
        examples.basic_substrate(),
        **AXES,
        directory=directory,
        request_memory="2GB",
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["request_memory"] == "2GB"
    lines = (directory / "remaining.txt").read_text().splitlines()
    assert lines
    for line in lines:
        assert line.split(",")[1].strip() == "2GB"
