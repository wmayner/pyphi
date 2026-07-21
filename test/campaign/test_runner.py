import subprocess
import sys

import pytest

from pyphi import examples
from pyphi.campaign import prepare
from pyphi.campaign.runner import run_task
from pyphi.serialize import load

AXES = {"states": "all", "subsets": "full", "formalisms": ["IIT_4_0_2026"]}


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
        check=False,
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
    kwargs = {
        "substrates_dir": campaign_dir / "substrates",
        "outputs_dir": campaign_dir / "outputs",
    }
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
