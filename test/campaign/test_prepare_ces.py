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
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kind"] == "ces"
    assert manifest["sia_mode"] == "shards"
    assert manifest["formalism"] == "IIT_4_0_2026"
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
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
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
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        scope=scope,
        directory=directory,
        units_per_job=1e9,
    )
    saved_scope = load(directory / "scope.json.gz")
    assert saved_scope.mechanisms.containing == (0,)
    manifest = json.loads((directory / "manifest.json").read_text())
    mechs = [
        tuple(int(x) for x in key.split(",")) for key in manifest["mechanism_workloads"]
    ]
    assert all(0 in m for m in mechs)


def test_empty_scope_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="zero mechanisms"):
        prepare_ces(
            examples.basic_substrate(),
            state=BASIC_STATE,
            formalism="IIT_4_0_2026",
            scope=CESScope(mechanisms=AxisScope(explicit=())),
            directory=tmp_path / "camp",
            units_per_job=1.0,
        )


def test_limit_threads_through_prepare_ces(tmp_path):
    with pytest.raises(ValueError, match="narrow the scope or raise the limit"):
        prepare_ces(
            examples.basic_substrate(),
            state=BASIC_STATE,
            formalism="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            limit=1,
        )
    # The failed call must not leave a campaign directory behind.
    assert not (tmp_path / "camp").exists()
