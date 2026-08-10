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


class TestDelivery:
    def test_ships_both_skills(self):
        assert sorted(mod.skill_names()) == ["iit", "pyphi"]

    def test_writes_every_skill(self, tmp_path):
        mod.deliver(mod.Target("x", "X", tmp_path))
        for name in mod.skill_names():
            assert (tmp_path / name / "SKILL.md").is_file()

    def test_stamps_a_sentinel_holding_the_version(self, tmp_path):
        mod.deliver(mod.Target("x", "X", tmp_path))
        stamp = (tmp_path / "iit" / mod.SENTINEL).read_text().strip()
        assert stamp

    def test_fills_references_from_the_content_topics(self, tmp_path):
        from pyphi.mcp import content

        mod.deliver(mod.Target("x", "X", tmp_path))
        references = tmp_path / "pyphi" / "references"
        written = {path.stem for path in references.glob("*.md")}
        assert written == set(content.topics())

    def test_the_configuration_reference_keeps_its_generated_half(self, tmp_path):
        mod.deliver(mod.Target("x", "X", tmp_path))
        text = (tmp_path / "pyphi" / "references" / "configuration.md").read_text()
        assert "Complete option reference" in text

    def test_the_gate_skill_has_no_references(self, tmp_path):
        mod.deliver(mod.Target("x", "X", tmp_path))
        assert not (tmp_path / "iit" / "references").exists()

    def test_delivering_twice_refreshes_rather_than_failing(self, tmp_path):
        target = mod.Target("x", "X", tmp_path)
        mod.deliver(target)
        (tmp_path / "iit" / "SKILL.md").write_text("stale")
        mod.deliver(target)
        assert (tmp_path / "iit" / "SKILL.md").read_text() != "stale"


class TestRemoval:
    def test_removes_what_deliver_wrote(self, tmp_path):
        target = mod.Target("x", "X", tmp_path)
        mod.deliver(target)
        assert sorted(mod.remove(target)) == ["iit", "pyphi"]
        assert not (tmp_path / "iit").exists()
        assert not (tmp_path / "pyphi").exists()

    def test_leaves_a_hand_written_skill_of_the_same_name(self, tmp_path):
        mine = tmp_path / "iit"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("mine")
        assert mod.remove(mod.Target("x", "X", tmp_path)) == []
        assert (mine / "SKILL.md").read_text() == "mine"

    def test_is_safe_where_nothing_was_installed(self, tmp_path):
        assert mod.remove(mod.Target("x", "X", tmp_path / "absent")) == []

    def test_leaves_unrelated_skills_alone(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text("other")
        mod.deliver(mod.Target("x", "X", tmp_path))
        mod.remove(mod.Target("x", "X", tmp_path))
        assert (other / "SKILL.md").read_text() == "other"
