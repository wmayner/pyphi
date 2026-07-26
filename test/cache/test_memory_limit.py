"""Detecting the memory a confined process is actually allowed."""

import psutil
import pytest

from pyphi import config
from pyphi.cache import cache_utils

GIB = 1024**3


@pytest.fixture
def cgroup(tmp_path):
    """Build a cgroup hierarchy under ``tmp_path`` and read a limit from it."""

    def build(self_cgroup_text, files):
        root = tmp_path / "cgroup"
        for relative, contents in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        proc = tmp_path / "self_cgroup"
        proc.write_text(self_cgroup_text)
        return cache_utils._cgroup_memory_limit(cgroup_root=root, self_cgroup=proc)

    return build


def test_v2_limit_read_from_the_processes_own_group(cgroup):
    assert (
        cgroup(
            "0::/system.slice/condor.service/job_1234\n",
            {"system.slice/condor.service/job_1234/memory.max": str(4 * GIB)},
        )
        == 4 * GIB
    )


def test_v2_unlimited_group_reports_no_limit(cgroup):
    assert (
        cgroup(
            "0::/user.slice\n",
            {"user.slice/memory.max": "max\n"},
        )
        is None
    )


def test_v1_limit_read_from_the_memory_controller(cgroup):
    assert (
        cgroup(
            "12:memory:/slurm/uid_1000/job_77\n11:cpu:/slurm\n",
            {"memory/slurm/uid_1000/job_77/memory.limit_in_bytes": str(2 * GIB)},
        )
        == 2 * GIB
    )


def test_v1_unlimited_sentinel_reports_no_limit(cgroup):
    """cgroup v1 spells "unlimited" as a number near the word size."""
    assert (
        cgroup(
            "12:memory:/\n",
            {"memory/memory.limit_in_bytes": "9223372036854771712\n"},
        )
        is None
    )


def test_namespaced_container_limit_found_at_the_root(cgroup):
    """Inside a container's cgroup namespace the limit sits at the root."""
    assert cgroup("0::/\n", {"memory.max": str(512 * 1024**2)}) == 512 * 1024**2


def test_missing_hierarchy_reports_no_limit(cgroup):
    assert cgroup("0::/nonexistent\n", {}) is None


def test_unreadable_self_cgroup_reports_no_limit(tmp_path):
    assert (
        cache_utils._cgroup_memory_limit(
            cgroup_root=tmp_path / "absent", self_cgroup=tmp_path / "absent"
        )
        is None
    )


@pytest.fixture
def detected_limit(monkeypatch):
    """Report a given cgroup allowance, clearing the memoized lookup around it."""

    def set_to(value):
        monkeypatch.setattr(cache_utils, "_cgroup_memory_limit", lambda: value)
        cache_utils.memory_limit_bytes.cache_clear()

    yield set_to
    cache_utils.memory_limit_bytes.cache_clear()


def test_memory_limit_falls_back_to_physical_memory(detected_limit):
    detected_limit(None)
    assert cache_utils.memory_limit_bytes() == psutil.virtual_memory().total


def test_memory_limit_uses_the_allowance_when_confined(detected_limit):
    detected_limit(4 * GIB)
    assert cache_utils.memory_limit_bytes() == 4 * GIB


def test_percentage_bounds_against_the_allowance_not_the_machine(detected_limit):
    """The percentage must bound a confined process, which is its purpose."""
    import os

    resident = cache_utils._process_handle(os.getpid()).memory_info().rss
    unbounded = {
        "memory_ceiling_bytes": None,
        "memory_ceiling_percentage": 100,
    }

    # An allowance below current usage: the percentage trips even though the
    # machine has plenty of memory left.
    detected_limit(resident // 2)
    with config.override(**unbounded):
        assert cache_utils.memory_full()

    # Free of any allowance, the same percentage of the machine leaves room.
    detected_limit(None)
    with config.override(**unbounded):
        assert not cache_utils.memory_full()


def test_shard_ceiling_prefers_the_actual_allocation(monkeypatch, tmp_path):
    """Raising a job's request must raise the enforced cache ceiling with it."""
    from dataclasses import replace

    from pyphi import examples
    from pyphi.campaign import prepare_ces
    from pyphi.campaign.runner import _shard_config
    from pyphi.cost import shard_cache_budget_bytes
    from pyphi.serialize import load

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        states=(1, 0, 0),
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
    )
    task = next(
        t
        for t in (load(f) for f in sorted((directory / "tasks").glob("task-*.json.gz")))
        if t.kind == "ces_shard"
    )

    # Undetectable allocation: the planned request stands in.
    monkeypatch.setattr(cache_utils, "_cgroup_memory_limit", lambda: None)
    planned = _shard_config(task)["memory_ceiling_bytes"]
    assert planned == shard_cache_budget_bytes(task.spec.memory_bytes)

    # A job actually granted more than planning predicted gets the extra room.
    monkeypatch.setattr(cache_utils, "_cgroup_memory_limit", lambda: 16 * GIB)
    assert _shard_config(task)["memory_ceiling_bytes"] == (
        shard_cache_budget_bytes(16 * GIB)
    )

    # A ceiling pinned at preparation time still wins.
    pinned = replace(
        task,
        config_overrides={**task.config_overrides, "memory_ceiling_bytes": 7},
    )
    assert _shard_config(pinned)["memory_ceiling_bytes"] == 7
