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


def test_campaign_default_formalism_honors_ambient_customizations(tmp_path):
    """prepare(formalisms=None) must run cells under the preparing session's
    configuration (which travels in each task's config_overrides), not reset
    it with the version preset — mirroring sweep(formalisms=None).

    The fixture has power: under IIT_3_0 the customized ces_measure changes
    Phi from 2.3125 (preset) to 1.0833."""
    from dataclasses import replace

    from pyphi.conf import config
    from pyphi.conf import presets
    from pyphi.system import System

    directory = tmp_path / "camp"
    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    preset = presets.by_name["IIT_3_0"]
    custom_iit = replace(preset["iit"], ces_measure="SUM_SMALL_PHI")
    with config.override(
        **{k: v for k, v in preset.items() if k != "iit"},
        iit=custom_iit,
        parallel=False,
        progress_bars=False,
    ):
        direct = float(System(substrate, state).sia().phi)
        with config.override(iit=preset["iit"]):
            preset_value = float(System(substrate, state).sia().phi)
        assert direct != preset_value  # the customization has an effect
        prepare(
            substrate,
            states=[state],
            subsets="full",
            formalisms=None,
            compute="sia",
            directory=directory,
            units_per_job=100.0,
        )
        _run_all_tasks(directory)
        result = collect(directory)
    assert float(result.df["phi"].iloc[0]) == direct
    # The table reports the version the campaign was prepared under.
    assert result.df["formalism"].iloc[0] == "IIT_3_0"
