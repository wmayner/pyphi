"""``pyphi-mcp install`` and ``uninstall``.

These exercise :mod:`pyphi.mcp.install`, which does not import the optional
``mcp`` dependency, so they run on a base install.
"""

import json
import sys

import pytest

from pyphi.mcp import install as mod


def _install(tmp_path, **kwargs):
    return mod.install(tmp_path, **kwargs)


class TestRegistration:
    def test_default_launches_the_environment_that_installed_it(self):
        assert mod.registration() == {
            "command": sys.executable,
            "args": ["-m", "pyphi.mcp"],
        }

    def test_a_specification_launches_through_uvx(self):
        assert mod.registration("pyphi[mcp]") == {
            "command": "uvx",
            "args": ["--from", "pyphi[mcp]", "pyphi-mcp"],
        }

    def test_written_into_an_empty_directory(self, tmp_path):
        _install(tmp_path)
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["pyphi"] == mod.registration()

    def test_merged_without_disturbing_other_servers(self, tmp_path):
        path = tmp_path / ".mcp.json"
        path.write_text(
            json.dumps({"mcpServers": {"other": {"command": "foo"}}, "keep": 1})
        )
        _install(tmp_path)
        config = json.loads(path.read_text())
        assert config["mcpServers"]["other"] == {"command": "foo"}
        assert config["keep"] == 1
        assert "pyphi" in config["mcpServers"]

    def test_identical_entry_is_not_rewritten(self, tmp_path):
        _install(tmp_path)
        assert not mod.write_registration(tmp_path / ".mcp.json", mod.registration())

    def test_conflicting_entry_is_refused_without_force(self, tmp_path):
        _install(tmp_path)
        with pytest.raises(FileExistsError, match="--force"):
            _install(tmp_path, spec="something-else")

    def test_force_replaces_a_conflicting_entry(self, tmp_path):
        _install(tmp_path)
        _install(tmp_path, spec="something-else", force=True)
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["pyphi"]["args"][1] == "something-else"

    def test_user_scope_and_desktop_resolve_outside_the_project(self, tmp_path):
        project = mod.config_path(tmp_path, "project", "claude-code")
        user = mod.config_path(tmp_path, "user", "claude-code")
        desktop = mod.config_path(tmp_path, "project", "claude-desktop")
        assert project == tmp_path / ".mcp.json"
        assert user.name == ".claude.json" and tmp_path not in user.parents
        assert desktop.name == "claude_desktop_config.json"

    def test_unknown_scope_or_client_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="scope"):
            mod.config_path(tmp_path, "global", "claude-code")
        with pytest.raises(ValueError, match="client"):
            mod.config_path(tmp_path, "project", "emacs")


class TestInstructionBlock:
    def test_written_where_the_file_is_absent(self, tmp_path):
        _install(tmp_path)
        text = (tmp_path / mod.INSTRUCTIONS_FILE).read_text()
        assert mod.BLOCK_BEGIN in text and mod.BLOCK_END in text
        assert "φₛ and Φ are different quantities" in text

    @pytest.mark.parametrize("existing", ["", "\n\n", "# Mine\n\nBuild with make.\n"])
    def test_appended_without_losing_what_was_there(self, tmp_path, existing):
        path = tmp_path / mod.INSTRUCTIONS_FILE
        path.write_text(existing)
        _install(tmp_path)
        text = path.read_text()
        assert mod.BLOCK_BEGIN in text
        if existing.strip():
            assert "Build with make." in text

    def test_reinstalling_replaces_the_block_and_preserves_the_rest(self, tmp_path):
        path = tmp_path / mod.INSTRUCTIONS_FILE
        _install(tmp_path)
        path.write_text(f"before\n\n{path.read_text()}\nafter\n")
        _install(tmp_path)
        text = path.read_text()
        assert text.count(mod.BLOCK_BEGIN) == 1
        assert text.startswith("before")
        assert text.rstrip().endswith("after")


class TestClaudeImport:
    """Claude Code reads CLAUDE.md and not AGENTS.md, so the block is bridged.

    See https://code.claude.com/docs/en/memory.
    """

    def test_import_written_where_claude_md_is_absent(self, tmp_path):
        _install(tmp_path)
        assert (tmp_path / mod.CLAUDE_FILE).read_text().strip() == mod.CLAUDE_IMPORT

    def test_import_prepended_to_an_existing_file(self, tmp_path):
        path = tmp_path / mod.CLAUDE_FILE
        path.write_text("## Mine\n\nUse plan mode.\n")
        _install(tmp_path)
        text = path.read_text()
        assert text.startswith(mod.CLAUDE_IMPORT)
        assert "Use plan mode." in text

    def test_import_not_duplicated(self, tmp_path):
        _install(tmp_path)
        _install(tmp_path)
        text = (tmp_path / mod.CLAUDE_FILE).read_text()
        assert text.count(mod.CLAUDE_IMPORT) == 1

    def test_a_symlink_is_left_alone(self, tmp_path):
        (tmp_path / mod.INSTRUCTIONS_FILE).write_text("# Mine\n")
        (tmp_path / mod.CLAUDE_FILE).symlink_to(mod.INSTRUCTIONS_FILE)
        _install(tmp_path)
        claude = tmp_path / mod.CLAUDE_FILE
        assert claude.is_symlink()
        # The block reaches Claude Code through the link, so no import is added.
        assert mod.CLAUDE_IMPORT not in claude.read_text()


class TestUninstall:
    def test_removes_everything_install_created(self, tmp_path):
        _install(tmp_path)
        mod.uninstall(tmp_path)
        assert not (tmp_path / ".mcp.json").exists()
        assert not (tmp_path / mod.INSTRUCTIONS_FILE).exists()
        assert not (tmp_path / mod.CLAUDE_FILE).exists()

    def test_leaves_surrounding_content_byte_identical(self, tmp_path):
        agents = tmp_path / mod.INSTRUCTIONS_FILE
        claude = tmp_path / mod.CLAUDE_FILE
        agents.write_text("# Mine\n\nBuild with make.\n")
        claude.write_text("## Mine\n\nUse plan mode.\n")
        _install(tmp_path)
        mod.uninstall(tmp_path)
        assert agents.read_text() == "# Mine\n\nBuild with make.\n"
        assert claude.read_text() == "## Mine\n\nUse plan mode.\n"

    def test_leaves_other_servers_registered(self, tmp_path):
        path = tmp_path / ".mcp.json"
        path.write_text(json.dumps({"mcpServers": {"other": {"command": "foo"}}}))
        _install(tmp_path)
        mod.uninstall(tmp_path)
        assert json.loads(path.read_text())["mcpServers"] == {
            "other": {"command": "foo"}
        }

    def test_is_safe_to_run_when_nothing_is_installed(self, tmp_path):
        assert mod.uninstall(tmp_path) == ["nothing to remove"]


class TestCommandLine:
    def test_print_writes_nothing(self, tmp_path, capsys):
        args = mod.build_parser().parse_args(
            ["install", "--print", "--directory", str(tmp_path)]
        )
        assert mod.run(args) == 0
        assert list(tmp_path.iterdir()) == []
        assert mod.BLOCK_BEGIN in capsys.readouterr().out

    def test_conflict_exits_nonzero(self, tmp_path, capsys):
        parser = mod.build_parser()
        mod.run(parser.parse_args(["install", "--directory", str(tmp_path)]))
        conflicting = parser.parse_args(
            ["install", "--directory", str(tmp_path), "--from", "other"]
        )
        assert mod.run(conflicting) == 1
        assert "--force" in capsys.readouterr().out

    def test_no_subcommand_means_run_the_server(self):
        assert mod.build_parser().parse_args([]).command is None

    def test_scope_defaults_to_project(self):
        args = mod.build_parser().parse_args(["install"])
        assert args.scope == "project"
