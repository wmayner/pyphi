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


def _resolution_state_for(substrate, state, formalism="IIT_4_0_2026"):
    import pyphi
    from pyphi.conf import presets as _presets
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(
        **_presets.by_name[formalism], parallel=False, progress_bars=False
    ):
        system = pyphi.System(substrate, state)
        return system_intrinsic_information(
            system,
            specification_measure=resolve_mechanism_measure(
                pyphi.config.formalism.iit.specification_measure
            ),
        )


def test_multi_state_resolution_state_mapping_suppresses_sia_shards(tmp_path):
    substrate = examples.basic_substrate()
    states = [(1, 0, 0), (1, 1, 0)]
    resolution = {tuple(s): _resolution_state_for(substrate, s) for s in states}
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
        resolution_state=resolution,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["sia_mode"] == "none"
    assert {t["kind"] for t in manifest["tasks"]} == {"ces_shard"}
    assert (directory / "resolution_state-0000.json.gz").exists()
    assert (directory / "resolution_state-0001.json.gz").exists()


def test_resolution_state_callable_form(tmp_path):
    substrate = examples.basic_substrate()
    states = [(1, 0, 0), (1, 1, 0)]
    specs = {tuple(s): _resolution_state_for(substrate, s) for s in states}
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
        resolution_state=lambda cell: specs[tuple(cell[3])],
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["sia_mode"] == "none"
    assert {t["kind"] for t in manifest["tasks"]} == {"ces_shard"}


def test_resolution_state_type_validated_at_prepare(tmp_path):
    with pytest.raises(TypeError, match="system_intrinsic_information"):
        prepare_ces(
            examples.basic_substrate(),
            states=BASIC_STATE,
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            resolution_state=BASIC_STATE,  # a plain state tuple, not a specification
        )


def test_resolution_state_missing_cell_raises(tmp_path):
    substrate = examples.basic_substrate()
    resolution = {(1, 0, 0): _resolution_state_for(substrate, (1, 0, 0))}
    with pytest.raises(ValueError, match="no entry"):
        prepare_ces(
            substrate,
            states=[(1, 0, 0), (1, 1, 0)],
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            resolution_state=resolution,
        )


def test_state_keyed_mapping_requires_singleton_axes(tmp_path):
    sub_a = examples.basic_substrate()
    resolution = {(1, 0, 0): _resolution_state_for(sub_a, (1, 0, 0))}
    with pytest.raises(ValueError, match="full cell"):
        prepare_ces(
            {"a": sub_a, "b": examples.basic_substrate()},
            states=(1, 0, 0),
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            resolution_state=resolution,
        )


def _expand_condor_macros(pattern: str, task_id: str) -> str:
    """Emulate HTCondor's expansion of the macros the submit template uses.

    ``$INT(task_id,%0Nd)`` -> the task id zero-padded to N digits;
    ``$(task_id)`` -> the raw id. Enough to reconstruct the filename the
    scheduler would transfer for a given ``remaining.txt`` row.
    """
    import re

    pattern = re.sub(
        r"\$INT\(task_id,\s*%0(\d+)d\)",
        lambda m: str(int(task_id)).zfill(int(m.group(1))),
        pattern,
    )
    return pattern.replace("$(task_id)", task_id)


def test_submit_filenames_match_padded_task_files(tmp_path):
    """Every input file the submit template references for a task must be the
    zero-padded name that ``prepare`` actually wrote (regression: a bare
    ``$(task_id)`` expanded to ``task-0.json.gz`` and held every job)."""
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    import re

    submit = (directory / "pyphi.sub").read_text()
    # The value can hold commas inside ``$INT(task_id,%04d)``, so match the
    # ``tasks/...json.gz`` token directly instead of splitting on commas.
    input_pattern = next(
        re.search(r"tasks/\S*\.json\.gz", line).group(0)
        for line in submit.splitlines()
        if line.strip().startswith("transfer_input_files")
    )
    remap_lhs = next(
        line.split("=", 1)[1].split("=", 1)[0].strip().strip('"')
        for line in submit.splitlines()
        if line.strip().startswith("transfer_output_remaps")
    )
    task_ids = [
        line.split(",", 1)[0].strip()
        for line in (directory / "remaining.txt").read_text().splitlines()
        if line.strip()
    ]
    assert task_ids
    for task_id in task_ids:
        # The file the scheduler transfers in must be the one on disk.
        transferred = _expand_condor_macros(input_pattern, task_id)
        assert (directory / transferred).exists(), (
            f"submit references {transferred} but it does not exist"
        )
        # The remap's source name must match what the runner writes back
        # (``task-{task_id:04d}.json.gz``), or output transfer silently fails.
        assert _expand_condor_macros(remap_lhs, task_id) == (
            f"task-{int(task_id):04d}.json.gz"
        )


def test_submit_preserves_relative_paths(tmp_path):
    """run_task.sh invokes `tasks/task-$1.json.gz`, so the submit file must keep
    the tasks/ layout on the execute node; without preserve_relative_paths
    HTCondor flattens the input to the scratch root and the runner can't find
    it (FileNotFoundError, no output)."""
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    assert "preserve_relative_paths = true" in (directory / "pyphi.sub").read_text()
    assert "tasks/task-" in (directory / "run_task.sh").read_text()
