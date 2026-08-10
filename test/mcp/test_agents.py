"""Agent detection and skill delivery for ``pyphi-mcp install``.

These exercise :mod:`pyphi.mcp.agents`, which does not import the optional
``mcp`` dependency, so they run on a base install.
"""

import pytest

from pyphi.mcp import agents as mod


class TestDetection:
    def test_nothing_detected_in_an_empty_home(self, tmp_path):
        assert mod.detect(tmp_path) == []

    def test_detects_only_agents_whose_probe_exists(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".cursor").mkdir()
        found = {target.name for target in mod.detect(tmp_path)}
        assert found == {"claude-code", "cursor"}

    def test_target_path_is_the_skills_directory(self, tmp_path):
        (tmp_path / ".codex").mkdir()
        (target,) = mod.detect(tmp_path)
        assert target.path == tmp_path / ".codex" / "skills"
        assert target.display == "Codex"

    def test_a_probe_that_is_a_file_is_not_an_agent(self, tmp_path):
        (tmp_path / ".claude").write_text("not a directory")
        assert mod.detect(tmp_path) == []

    def test_detection_is_ordered_by_the_table(self, tmp_path):
        for probe in (".cursor", ".claude", ".codex"):
            (tmp_path / probe).mkdir()
        assert [t.name for t in mod.detect(tmp_path)] == [
            "claude-code",
            "codex",
            "cursor",
        ]


class TestExplicitTargets:
    def test_named_agent_is_returned_undetected(self, tmp_path):
        (target,) = mod.chosen(["codex"], [], home=tmp_path)
        assert target.path == tmp_path / ".codex" / "skills"

    def test_unknown_agent_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown agent"):
            mod.chosen(["nope"], [], home=tmp_path)

    def test_an_explicit_path_is_used_verbatim(self, tmp_path):
        elsewhere = tmp_path / "somewhere" / "skills"
        (target,) = mod.chosen([], [elsewhere], home=tmp_path)
        assert target.path == elsewhere
        assert target.name == str(elsewhere)

    def test_resolve_prefers_explicit_over_detection(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        targets = mod.resolve(["codex"], [], home=tmp_path)
        assert [t.name for t in targets] == ["codex"]

    def test_resolve_detects_when_nothing_is_explicit(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        assert [t.name for t in mod.resolve([], [], home=tmp_path)] == ["claude-code"]
