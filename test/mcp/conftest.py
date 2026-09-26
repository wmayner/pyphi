"""Shared fixtures for the MCP tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Point ``Path.home()`` at an empty directory for every test here.

    ``pyphi-mcp install`` writes agent skills under the home directory, and
    resolves it through ``Path.home()`` whenever the caller does not name one.
    A test that drives the command line would otherwise install into whichever
    agents the developer running the suite happens to have.
    """
    directory = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: directory))
    return directory


@pytest.fixture(autouse=True)
def no_agent_commands(monkeypatch):
    """Keep ``pyphi-mcp install`` from running a real ``claude`` or ``codex``.

    The plugin step runs each agent's own plugin commands, which would change
    the configuration of whichever agents the developer has installed.
    """
    from pyphi.mcp import agents

    def refuse(command):
        pytest.fail(f"a test ran a real agent command: {' '.join(command)}")

    monkeypatch.setattr(agents, "_execute", refuse)
