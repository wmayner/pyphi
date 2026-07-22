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

AXES = {
    "states": "all",
    "subsets": "full",
    "formalisms": ["IIT_4_0_2026"],
    "compute": "sia",
}


def _run_all_tasks(directory, check=True):
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
            check=False,
        )
        if check:
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
    assert (directory / "remaining.txt").read_text() == "1, 4GB\n"
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
    _run_all_tasks(directory, check=False)
    st = status(directory)
    assert st.failed == (0,)
    assert (directory / "remaining.txt").read_text() == "0, 4GB\n"
