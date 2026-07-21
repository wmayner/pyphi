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
