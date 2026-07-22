import json

import pytest

from pyphi import examples
from pyphi.campaign import collect
from pyphi.campaign import prepare_ces
from pyphi.campaign import scope_report
from pyphi.campaign.runner import run_task
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.system import System
from pyphi.warnings import PyPhiWarning

BASIC_STATE = (1, 0, 0)
PIN = {"parallel": False, "progress_bars": False}


def _resolution_state(system):
    from pyphi.conf import config as _config
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure

    return system_intrinsic_information(
        system,
        specification_measure=resolve_mechanism_measure(
            _config.formalism.iit.specification_measure
        ),
    )


def _run_all(directory):
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        assert (
            run_task(
                task_file,
                substrates_dir=directory / "substrates",
                outputs_dir=directory / "outputs",
            )
            == 0
        )


def _campaign(tmp_path, formalism, **kwargs):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms=formalism,
        directory=directory,
        units_per_job=5.0,  # tiny: forces all three ladder rungs
        **kwargs,
    )
    _run_all(directory)
    return directory


@pytest.mark.parametrize("formalism", ["IIT_4_0_2026", "IIT_4_0_2023"])
def test_sharded_equals_unsharded(tmp_path, formalism):
    directory = _campaign(tmp_path, formalism)
    result = collect(directory)
    with config.override(**presets.by_name[formalism], **PIN):
        reference = System(examples.basic_substrate(), BASIC_STATE).ces()
    assert float(result.sia.phi) == float(reference.sia.phi)
    assert len(result.distinctions) == len(reference.distinctions)
    got = sorted(
        (d.mechanism, d.cause.purview, d.effect.purview, float(d.phi))
        for d in result.distinctions
    )
    want = sorted(
        (d.mechanism, d.cause.purview, d.effect.purview, float(d.phi))
        for d in reference.distinctions
    )
    assert got == want
    assert float(result.relations.sum_phi()) == float(reference.relations.sum_phi())


def test_scope_report_written_and_certified(tmp_path):
    scope = CESScope(mechanisms=AxisScope(containing=(0,)))
    directory = _campaign(tmp_path, "IIT_4_0_2026", scope=scope)
    result = collect(directory)
    report = scope_report(directory)
    assert report.mechanisms_possible == 7
    assert report.mechanisms_admitted == 4
    assert report.sum_phi_r_lower == float(result.relations.sum_phi())
    assert report.sum_phi_r_upper is None or (
        report.sum_phi_r_upper >= report.sum_phi_r_lower
    )
    assert (directory / "scope_report.json").exists()


def test_precomputed_sia_mode(tmp_path):
    import pyphi

    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        sia = pyphi.System(examples.basic_substrate(), BASIC_STATE).sia()
    directory = _campaign(tmp_path, "IIT_4_0_2026", sia=sia)
    result = collect(directory)
    assert float(result.sia.phi) == float(sia.phi)


def test_no_sia_mode_carries_no_phi_s(tmp_path):
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

    directory = tmp_path / "camp"
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = System(examples.basic_substrate(), BASIC_STATE)
        state = _resolution_state(system)
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
        resolution_state=state,
    )
    _run_all(directory)
    result = collect(directory)
    assert isinstance(result.sia, NullSystemIrreducibilityAnalysis)
    assert len(result.distinctions) >= 1


def test_version_guard_refuses_mismatched_outputs(tmp_path):
    directory = _campaign(tmp_path, "IIT_4_0_2026")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["pyphi_version"] = "0.0.0"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="prepared under"):
        collect(directory)


def test_partial_collect_reports_missing_groups(tmp_path):
    directory = _campaign(tmp_path, "IIT_4_0_2026")
    manifest = json.loads((directory / "manifest.json").read_text())
    victim = next(
        row["task_id"] for row in manifest["tasks"] if row["kind"] == "ces_shard"
    )
    (directory / "outputs" / f"task-{victim:04d}.json.gz").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        collect(directory)
    with pytest.warns(PyPhiWarning):
        partial = collect(directory, partial=True)
    report = scope_report(directory)
    assert report.missing_groups
    assert len(partial.distinctions) >= 0


def test_multi_state_collect_returns_sweep_result(tmp_path):
    import pyphi
    from pyphi.sweep import SweepResult

    substrate = examples.basic_substrate()
    states = [(1, 0, 0), (1, 1, 0)]
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    _run_all(directory)
    result = collect(directory)
    assert isinstance(result, SweepResult)
    assert len(result.results) == 2
    # each cell's structure equals the local computation
    for state, structure in zip(states, result.results, strict=True):
        with pyphi.config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
            local = pyphi.System(substrate, state).ces()
        assert sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in structure.distinctions
        ) == sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in local.distinctions
        )
    reports = scope_report(directory)
    assert set(reports) == {
        (0, "IIT_4_0_2026", (0, 1, 2), tuple(state)) for state in states
    }


def test_weakly_connected_subset_collects(tmp_path):
    """A subset without strong connectivity short-circuits its SIA; shards
    carry that result instead of failing."""
    import pyphi

    substrate = examples.basic_substrate()
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=BASIC_STATE,
        subsets=[(0, 1)],
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    _run_all(directory)
    result = collect(directory)
    with pyphi.config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        local = pyphi.System(substrate, BASIC_STATE, node_indices=(0, 1)).ces()
    assert float(result.sia.phi) == float(local.sia.phi)
    assert sorted(
        (tuple(d.mechanism), round(float(d.phi), 10)) for d in result.distinctions
    ) == sorted(
        (tuple(d.mechanism), round(float(d.phi), 10)) for d in local.distinctions
    )


def test_multi_subset_sweep_collects_each_cell(tmp_path):
    import pyphi
    from pyphi.sweep import SweepResult

    substrate = examples.basic_substrate()
    subsets = [(0, 1, 2), (0, 1)]
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=BASIC_STATE,
        subsets=subsets,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    _run_all(directory)
    result = collect(directory)
    assert isinstance(result, SweepResult)
    for subset, structure in zip(subsets, result.results, strict=True):
        with pyphi.config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
            local = pyphi.System(substrate, BASIC_STATE, node_indices=subset).ces()
        assert sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in structure.distinctions
        ) == sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in local.distinctions
        )


def test_two_substrate_sweep_collects_per_label(tmp_path):
    from pyphi.sweep import SweepResult

    directory = tmp_path / "camp"
    prepare_ces(
        {"a": examples.basic_substrate(), "b": examples.basic_substrate()},
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    assert (directory / "substrates" / "substrate-a.msgpack.gz").exists()
    assert (directory / "substrates" / "substrate-b.msgpack.gz").exists()
    _run_all(directory)
    result = collect(directory)
    assert isinstance(result, SweepResult)
    assert len(result.results) == 2
