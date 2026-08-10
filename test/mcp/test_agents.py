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

    def test_a_dropped_reference_does_not_survive_a_reinstall(self, tmp_path):
        target = mod.Target("x", "X", tmp_path)
        mod.deliver(target)
        stale = tmp_path / "pyphi" / "references" / "renamed-away.md"
        stale.write_text("from an older PyPhi")
        mod.deliver(target)
        assert not stale.exists()

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


class TestFlow:
    def _home(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        return tmp_path

    def test_no_agents_means_no_report_and_no_writes(self, tmp_path):
        assert mod.install_step(skills=True, names=[], paths=[], home=tmp_path) == []

    def test_declining_writes_nothing(self, tmp_path):
        home = self._home(tmp_path)
        assert mod.install_step(skills=False, names=[], paths=[], home=home) == []
        assert not (home / ".claude" / "skills").exists()

    def test_accepting_writes_without_prompting(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        monkeypatch.setattr(mod, "confirm", lambda _question: pytest.fail("prompted"))
        actions = mod.install_step(skills=True, names=[], paths=[], home=home)
        assert (home / ".claude" / "skills" / "iit" / "SKILL.md").is_file()
        assert any("Claude Code" in line or "skills" in line for line in actions)

    def test_non_interactive_skips_and_says_how_to_do_it_later(
        self, tmp_path, monkeypatch
    ):
        home = self._home(tmp_path)
        monkeypatch.setattr(mod, "interactive", lambda: False)
        (action,) = mod.install_step(skills=None, names=[], paths=[], home=home)
        assert "--skills" in action
        assert not (home / ".claude" / "skills").exists()

    def test_interactive_yes_installs(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        monkeypatch.setattr(mod, "interactive", lambda: True)
        monkeypatch.setattr(mod, "confirm", lambda _question: True)
        mod.install_step(skills=None, names=[], paths=[], home=home)
        assert (home / ".claude" / "skills" / "pyphi" / "SKILL.md").is_file()

    def test_interactive_no_writes_nothing(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        monkeypatch.setattr(mod, "interactive", lambda: True)
        monkeypatch.setattr(mod, "confirm", lambda _question: False)
        assert mod.install_step(skills=None, names=[], paths=[], home=home) == []

    def test_the_prompt_names_every_detected_agent(self, tmp_path, monkeypatch):
        home = tmp_path
        (home / ".claude").mkdir()
        (home / ".codex").mkdir()
        monkeypatch.setattr(mod, "interactive", lambda: True)
        asked = []
        monkeypatch.setattr(mod, "confirm", asked.append)
        mod.install_step(skills=None, names=[], paths=[], home=home)
        assert "Claude Code, Codex" in asked[0]

    def test_one_failing_agent_does_not_stop_the_others(self, tmp_path, monkeypatch):
        home = tmp_path
        (home / ".claude").mkdir()
        (home / ".codex").mkdir()
        real = mod.deliver

        def failing(target):
            if target.name == "claude-code":
                raise OSError("permission denied")
            real(target)

        monkeypatch.setattr(mod, "deliver", failing)
        actions = mod.install_step(skills=True, names=[], paths=[], home=home)
        assert (home / ".codex" / "skills" / "iit").is_dir()
        assert any("could not" in line for line in actions)

    def test_the_report_gives_full_paths(self, tmp_path):
        home = self._home(tmp_path)
        actions = mod.install_step(skills=True, names=[], paths=[], home=home)
        assert str(home / ".claude" / "skills") in "\n".join(actions)

    def test_removal_reports_what_it_removed(self, tmp_path):
        home = self._home(tmp_path)
        mod.install_step(skills=True, names=[], paths=[], home=home)
        actions = mod.remove_step(names=[], paths=[], home=home)
        assert any("iit" in line for line in actions)

    def test_describe_writes_nothing(self, tmp_path):
        home = self._home(tmp_path)
        lines = mod.describe(names=[], paths=[], home=home)
        assert not (home / ".claude" / "skills").exists()
        assert any("iit" in line for line in lines)


class TestConfirm:
    @pytest.mark.parametrize("answer", ["", "y", "Y", "yes", " YES "])
    def test_accepting_answers(self, answer, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt: answer)
        assert mod.confirm("Install?")

    @pytest.mark.parametrize("answer", ["n", "no", "nope"])
    def test_declining_answers(self, answer, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt: answer)
        assert not mod.confirm("Install?")

    def test_end_of_input_declines(self, monkeypatch):
        def raise_eof(_prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert not mod.confirm("Install?")


class TestInteractive:
    def test_ci_is_not_interactive(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert not mod.interactive()


class TestShippedSkills:
    def _front_matter(self, name):
        from importlib import resources

        text = (resources.files("pyphi.mcp") / "skills" / name / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert text.startswith("---\n")
        return text.split("---\n", 2)[1], text

    def test_front_matter_names_match_their_directories(self):
        for name in mod.skill_names():
            front, _ = self._front_matter(name)
            assert f"name: {name}\n" in front

    def test_every_skill_has_a_description(self):
        for name in mod.skill_names():
            front, _ = self._front_matter(name)
            assert "description:" in front

    def test_the_gate_says_not_to_answer_from_recollection(self):
        _, text = self._front_matter("iit")
        assert "recollection" in text

    def test_the_library_skill_warns_about_the_swapped_names(self):
        _, text = self._front_matter("pyphi")
        assert "CauseEffectStructure" in text
        assert "Distinctions" in text

    def test_referenced_topics_named_in_the_body_exist(self):
        from pyphi.mcp import content

        _, text = self._front_matter("pyphi")
        for topic in ("migration", "building-systems", "reproducible-work"):
            assert f"references/{topic}.md" in text
            assert topic in content.topics()


@pytest.mark.slow
def test_the_skills_reach_a_built_wheel(tmp_path):
    """A wheel built from this tree carries the skill files.

    Guards the Hatchling configuration: ``pyphi/mcp/skills/**`` ships only
    because non-Python files under a packaged directory are included by
    default, which a future build change could silently undo.
    """
    import subprocess
    import zipfile

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (wheel,) = tmp_path.glob("*.whl")
    shipped = {
        name
        for name in zipfile.ZipFile(wheel).namelist()
        if name.startswith("pyphi/mcp/skills/")
    }
    assert "pyphi/mcp/skills/iit/SKILL.md" in shipped
    assert "pyphi/mcp/skills/pyphi/SKILL.md" in shipped
