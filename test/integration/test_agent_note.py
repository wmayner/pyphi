"""The note PyPhi prints when an AI coding agent imports it."""

import os
import subprocess
import sys


def _import_pyphi(**env_overrides):
    """Import pyphi in a subprocess and return its (stdout, stderr)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CLAUDECODE", "PYPHI_AGENT", "PYPHI_AGENT_NOTE_OFF"}
    }
    env["PYPHI_WELCOME_OFF"] = "1"
    # On Windows the child's stderr defaults to the console codepage with
    # errors="backslashreplace", turning φ_s into the literal φ_s;
    # force UTF-8 so the note's characters survive the round trip.
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", "import pyphi"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout, result.stderr


def test_note_is_printed_under_an_agent():
    _, stderr = _import_pyphi(CLAUDECODE="1")
    assert "notes for AI assistants" in stderr
    # The two facts the note exists to carry.
    assert "φ_s" in stderr and "Φ" in stderr
    assert "little-endian" in stderr


def test_any_harness_can_opt_in():
    _, stderr = _import_pyphi(PYPHI_AGENT="1")
    assert "notes for AI assistants" in stderr


def test_note_never_reaches_stdout():
    """stdout is the MCP server's JSON-RPC channel; writing there corrupts it."""
    stdout, _ = _import_pyphi(CLAUDECODE="1")
    assert stdout == ""


def test_note_is_absent_without_an_agent():
    _, stderr = _import_pyphi()
    assert "notes for AI assistants" not in stderr


def test_note_suppressed_by_its_own_env_var():
    _, stderr = _import_pyphi(CLAUDECODE="1", PYPHI_AGENT_NOTE_OFF="1")
    assert "notes for AI assistants" not in stderr


def test_welcome_off_does_not_suppress_the_note():
    """The two messages have different audiences and independent switches."""
    _, stderr = _import_pyphi(CLAUDECODE="1", PYPHI_WELCOME_OFF="1")
    assert "notes for AI assistants" in stderr


def test_note_points_at_the_mcp_tools_without_discouraging_scripts():
    """Exploration and interpretation through the tools, durable work in scripts.

    A researcher writing a reproducible analysis is right to script it — the
    server holds results only in memory — so the note must not read as a
    blanket instruction to avoid scripting.
    """
    _, stderr = _import_pyphi(CLAUDECODE="1")
    assert "for exploration and" in stderr
    assert "reproducible belongs" in stderr


def test_welcome_banner_does_not_reach_stdout():
    """The MCP server speaks JSON-RPC over stdout; a banner there corrupts it."""
    env = {k: v for k, v in os.environ.items() if k != "PYPHI_WELCOME_OFF"}
    result = subprocess.run(
        [sys.executable, "-c", "import pyphi"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert "Welcome to PyPhi!" in result.stderr
