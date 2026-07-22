import json

import pytest

from pyphi import examples
from pyphi.campaign import prepare_ces
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.serialize import load

BASIC_STATE = (1, 0, 0)


def test_prepare_ces_writes_shard_campaign(tmp_path):
    directory = tmp_path / "camp"
    status = prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kind"] == "ces"
    assert manifest["sia_mode"] == "shards"
    assert manifest["groups"][0]["formalism"] == "IIT_4_0_2026"
    assert (directory / "scope.json.gz").exists()
    assert status.n_tasks == len(manifest["tasks"])
    kinds = {t["kind"] for t in manifest["tasks"]}
    assert kinds == {"ces_shard", "sia_shard"}
    task0 = load(directory / "tasks" / "task-0000.json.gz")
    assert task0.kind in ("ces_shard", "sia_shard")
    assert (directory / "pyphi.sub").exists()


def test_precomputed_sia_skips_sia_shards(tmp_path):
    import pyphi
    from pyphi.conf import presets

    substrate = examples.basic_substrate()
    with pyphi.config.override(
        **presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
    ):
        sia = pyphi.System(substrate, BASIC_STATE).sia()
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
        sia=sia,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["sia_mode"] == "precomputed"
    assert {t["kind"] for t in manifest["tasks"]} == {"ces_shard"}
    assert (directory / "sia.json.gz").exists()


def test_scope_lands_in_manifest_and_tasks(tmp_path):
    directory = tmp_path / "camp"
    scope = CESScope(mechanisms=AxisScope(containing=(0,)))
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        scope=scope,
        directory=directory,
        units_per_job=1e9,
    )
    saved_scope = load(directory / "scope.json.gz")
    assert saved_scope.mechanisms.containing == (0,)
    manifest = json.loads((directory / "manifest.json").read_text())
    mechs = [
        tuple(int(x) for x in key.split(","))
        for key in manifest["groups"][0]["mechanism_workloads"]
    ]
    assert all(0 in m for m in mechs)


def test_empty_scope_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="zero mechanisms"):
        prepare_ces(
            examples.basic_substrate(),
            states=BASIC_STATE,
            formalisms="IIT_4_0_2026",
            scope=CESScope(mechanisms=AxisScope(explicit=())),
            directory=tmp_path / "camp",
            units_per_job=1.0,
        )


def test_limit_threads_through_prepare_ces(tmp_path):
    with pytest.raises(ValueError, match="narrow the scope or raise the limit"):
        prepare_ces(
            examples.basic_substrate(),
            states=BASIC_STATE,
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            limit=1,
        )
    # The failed call must not leave a campaign directory behind.
    assert not (tmp_path / "camp").exists()


def test_scaffold_requests_memory_per_task(tmp_path):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    sub = (directory / "pyphi.sub").read_text()
    assert "request_memory      = $(memory)" in sub
    assert "queue task_id, memory from remaining.txt" in sub
    manifest = json.loads((directory / "manifest.json").read_text())
    lines = (directory / "remaining.txt").read_text().splitlines()
    assert len(lines) == len(manifest["tasks"])
    for line, row in zip(lines, manifest["tasks"], strict=True):
        task_id, memory = (part.strip() for part in line.split(","))
        assert int(task_id) == row["task_id"]
        assert memory == f"{row['memory_bytes'] // 1024**2}MB"
    # default 4GB floor: no task requests less
    assert all(row["memory_bytes"] >= 4 * 1024**3 for row in manifest["tasks"])


def test_status_rewrite_preserves_memory_column(tmp_path):
    from pyphi.campaign import status

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    before = (directory / "remaining.txt").read_text()
    (directory / "remaining.txt").write_text("")  # clobber
    status(directory)  # all tasks pending: full rewrite
    assert (directory / "remaining.txt").read_text() == before


def test_multi_state_campaign_replicates_shards_per_state(tmp_path):
    directory = tmp_path / "camp"
    states = [(1, 0, 0), (1, 1, 0)]
    prepare_ces(
        examples.basic_substrate(),
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["cells"]) == 2
    assert len(manifest["groups"]) == 1  # one (label, formalism, subset) group
    by_cell: dict[int, int] = {}
    for row in manifest["tasks"]:
        by_cell[row["cell"]] = by_cell.get(row["cell"], 0) + 1
    # identical plan per state: same task count for each cell
    assert by_cell[0] == by_cell[1]
    task0 = load(directory / "tasks" / "task-0000.json.gz")
    assert tuple(task0.state) in {tuple(s) for s in states}


def test_multi_cell_rejects_precomputed_sia(tmp_path):
    import pyphi
    from pyphi.conf import presets as _presets

    substrate = examples.basic_substrate()
    with pyphi.config.override(
        **_presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
    ):
        sia = pyphi.System(substrate, BASIC_STATE).sia()
    with pytest.raises(ValueError, match="single-cell"):
        prepare_ces(
            substrate,
            states=[(1, 0, 0), (1, 1, 0)],
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=1e9,
            sia=sia,
        )


def test_formalisms_accepts_bare_string(tmp_path):
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=tmp_path / "camp",
        units_per_job=1e9,
    )
    manifest = json.loads((tmp_path / "camp" / "manifest.json").read_text())
    assert manifest["cells"][0][1] == "IIT_4_0_2026"


def test_states_all_skips_unreachable_cells(tmp_path):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states="all",
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    # basic substrate: 2 of 8 states are dynamically unreachable
    assert len(manifest["cells"]) == 6
    assert len(manifest["skipped_cells"]) == 2


def test_explicit_unreachable_state_fails_loud(tmp_path):
    from pyphi.exceptions import StateUnreachableError

    with pytest.raises(StateUnreachableError):
        prepare_ces(
            examples.basic_substrate(),
            states=(0, 1, 0),
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=1e9,
        )
    assert not (tmp_path / "camp").exists()
