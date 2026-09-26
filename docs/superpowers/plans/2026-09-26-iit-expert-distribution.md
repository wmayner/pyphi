# IIT Expert Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish IIT Expert as one plugin for Claude Code, Codex and Cursor, have `pyphi-mcp install` offer it in place of PyPhi's retired `iit` skill, and present both tools in PyPhi's documentation.

**Architecture:** Three repositories. `~/projects/iit-expert-plugin` becomes the public plugin (`wmayner/iit-expert-plugin`) holding one shared skill plus the connector, with manifests for each agent. `~/projects/iit-reference` stays the skill's source and gains a sync script and a rewritten install page. PyPhi's `pyphi/mcp/agents.py` drops the `iit` skill, stops writing a second copy of the `pyphi` skill for Cursor, and gains a step that runs each agent's own plugin commands.

**Tech Stack:** Python 3.13 stdlib (`subprocess`, `shutil`, `json`), pytest, Claude Code 2.1.283 and Codex 0.156.1 plugin CLIs, Sphinx/MyST docs, Cloudflare Worker deploy (`npm run deploy`).

**Spec:** `docs/superpowers/specs/2026-09-26-iit-expert-distribution-design.md`

## Facts established while planning (2026-09-26)

These were verified against the installed CLIs with isolated config directories, and they change two details of the spec:

- A draft plugin with `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` and `.mcp.json` (`"type": "http"`) passes `claude plugin validate`, installs with `claude plugin marketplace add <dir>` + `claude plugin install iit-expert@iit-expert`, and `claude plugin details` lists the skill and the MCP server. Rerunning both commands exits 0 ("already installed").
- **Codex reads the Claude Code files directly.** With only `.claude-plugin/` and `.mcp.json` present, `codex plugin marketplace add <dir>` + `codex plugin add iit-expert@iit-expert` installs the plugin, copies `skills/`, and `codex mcp list` shows `iit-expert https://mcp.learniit.org enabled`. Rerunning exits 0. **So the spec's `.agents/plugins/marketplace.json` is dropped.**
- **Codex 0.156 still reads `$CODEX_HOME/skills` (default `~/.codex/skills`)**, found in the binary's own skill-installer text. PyPhi's Codex path stays as it is.
- The Agent Plugins `plugin.json` + `mcp.json` (`"type": "streamable-http"`) stay in the layout for Cursor, which cannot be tested here: the local `cursor-agent` dates from 2025-08 and has no plugin support, and the Cursor app is not installed. Cursor is checked by hand in Task 2.
- Cursor's documentation lists `~/.claude/skills` and `~/.codex/skills` among the folders it reads. That could not be tested locally, so the Cursor deduplication in Task 5 relies on the documented behavior.
- Marketplace name and plugin name are both `iit-expert`, so the plugin id in every command is `iit-expert@iit-expert`.
- Removal commands: `claude plugin uninstall iit-expert@iit-expert`, `codex plugin remove iit-expert@iit-expert`.

## Global Constraints

- Plugin repository: `wmayner/iit-expert-plugin`, public. Plugin id `iit-expert@iit-expert`. Connector URL `https://mcp.learniit.org`. Install page `https://learniit.org/install`.
- Manifest license `CC-BY-4.0`; author "Center for Sleep and Consciousness, University of Wisconsin–Madison"; homepage `https://learniit.org`; first version `0.1.0`.
- Every public description of IIT Expert says it is a work in progress, once, plainly.
- PyPhi supports Python 3.13+ only; use `uv run` for every Python command in the PyPhi repo.
- Public prose (docs pages, README, changelog, install page, plugin README) is written with the `writing-naturally` skill.
- No planning labels ("Task 3", "per the plan") in code, comments, docstrings or changelog.
- Commits stage only the files the task touched; other sessions commit to the PyPhi checkout concurrently. Never `--no-verify`.
- Pushing, creating the GitHub repository, and deploying `learniit.org` each need the user's explicit go-ahead at the time.
- Test verdicts come from reading the pytest summary line in a log file, never from an exit code through a pipe.

## Review Focus

1. **A developer running the PyPhi test suite with `claude` or `codex` on PATH** expects no real plugin command to run. Pinned by the autouse fixture in Task 6 Step 1 and its own test.
2. **A second `pyphi-mcp install`** (the documented way to refresh skills) expects the plugin step to report success again, not an error. Both CLIs exit 0 on rerun (verified above); Task 9 Step 3 checks it end to end.
3. **A user who upgrades PyPhi with the old `iit` skill installed** expects the next `install` or `uninstall` to remove it, while a hand-written `iit` skill survives. Tests in Task 4.
4. **A user who passes `--agent cursor` explicitly while Claude Code is also detected** expects the skill written where they asked. Test in Task 5.
5. **A plugin command that hangs** (network stall while cloning the marketplace) expects `install` to give up after the timeout and print the commands. Test in Task 6.

---

### Task 1: Build the plugin repository

**Files (all under `~/projects/iit-expert-plugin/`):**
- Modify: `.claude-plugin/plugin.json`
- Create: `.claude-plugin/marketplace.json`, `.mcp.json`, `plugin.json`, `mcp.json`, `.cursor-plugin/marketplace.json`, `scripts/check_manifests.py`, `.github/workflows/check.yml`
- Replace: `skills/iit-expert/SKILL.md` (copied from `~/projects/iit-reference/surfaces/skill/SKILL.md`)
- Modify: `README.md`

**Interfaces:**
- Produces: plugin id `iit-expert@iit-expert`; `scripts/check_manifests.py` exits 0 when consistent, 1 with one line per problem otherwise. Task 3's sync script writes `version` into `plugin.json` and `.claude-plugin/plugin.json`.

- [ ] **Step 1: Write the manifests**

`.claude-plugin/plugin.json`:
```json
{
  "name": "iit-expert",
  "version": "0.1.0",
  "description": "Work in progress. Answers questions about Integrated Information Theory from its primary literature, through the IIT Expert skill and the learniit.org connector.",
  "author": {
    "name": "Center for Sleep and Consciousness, University of Wisconsin–Madison"
  },
  "homepage": "https://learniit.org",
  "repository": "https://github.com/wmayner/iit-expert-plugin",
  "license": "CC-BY-4.0"
}
```

`.claude-plugin/marketplace.json`:
```json
{
  "name": "iit-expert",
  "description": "The IIT Expert plugin: a skill and a connector for Integrated Information Theory's primary literature.",
  "owner": { "name": "Will Mayner" },
  "plugins": [
    {
      "name": "iit-expert",
      "source": "./",
      "description": "Work in progress. The IIT Expert skill and the learniit.org connector."
    }
  ]
}
```

`.mcp.json` (Claude Code; Codex reads it too):
```json
{
  "mcpServers": {
    "iit-expert": { "type": "http", "url": "https://mcp.learniit.org" }
  }
}
```

`plugin.json` (Agent Plugins, for Cursor):
```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
  "name": "iit-expert",
  "version": "0.1.0",
  "description": "Work in progress. Answers questions about Integrated Information Theory from its primary literature, through the IIT Expert skill and the learniit.org connector.",
  "author": {
    "name": "Center for Sleep and Consciousness, University of Wisconsin–Madison"
  },
  "homepage": "https://learniit.org",
  "repository": "https://github.com/wmayner/iit-expert-plugin",
  "license": "CC-BY-4.0"
}
```

`mcp.json`:
```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
  "mcpServers": {
    "iit-expert": { "type": "streamable-http", "url": "https://mcp.learniit.org" }
  }
}
```

`.cursor-plugin/marketplace.json`:
```json
{
  "name": "iit-expert",
  "owner": { "name": "Will Mayner" },
  "plugins": [{ "name": "iit-expert", "source": "./" }]
}
```

- [ ] **Step 2: Copy the current skill**

```bash
cp ~/projects/iit-reference/surfaces/skill/SKILL.md ~/projects/iit-expert-plugin/skills/iit-expert/SKILL.md
```

- [ ] **Step 3: Write the consistency check**

`scripts/check_manifests.py`:
```python
"""Fail if the plugin's manifests disagree with each other.

Claude Code and Codex read ``.claude-plugin/``; Cursor reads the Agent Plugins
files at the root. The fields every agent shows a user must match, and every MCP
file must name the same connector.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ("name", "version", "description", "homepage", "repository", "license", "author")
MANIFESTS = ("plugin.json", ".claude-plugin/plugin.json")
MCP_FILES = (".mcp.json", "mcp.json")
MARKETPLACES = (".claude-plugin/marketplace.json", ".cursor-plugin/marketplace.json")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def problems() -> list[str]:
    found = []
    reference, *others = (load(path) for path in MANIFESTS)
    for path, manifest in zip(MANIFESTS[1:], others):
        for field in SHARED:
            if manifest.get(field) != reference.get(field):
                found.append(f"{path}: {field} differs from {MANIFESTS[0]}")
    urls = {
        path: {server["url"] for server in load(path)["mcpServers"].values()}
        for path in MCP_FILES
    }
    if len({frozenset(value) for value in urls.values()}) != 1:
        found.append(f"MCP files name different servers: {urls}")
    for path in MARKETPLACES:
        names = [entry["name"] for entry in load(path)["plugins"]]
        if names != [reference["name"]]:
            found.append(f"{path}: lists {names}, expected [{reference['name']!r}]")
    if not (ROOT / "skills" / reference["name"] / "SKILL.md").is_file():
        found.append(f"skills/{reference['name']}/SKILL.md is missing")
    return found


if __name__ == "__main__":
    found = problems()
    for line in found:
        print(line)
    sys.exit(1 if found else 0)
```

`.github/workflows/check.yml`:
```yaml
name: check
on: [push, pull_request]
jobs:
  manifests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: python scripts/check_manifests.py
```

- [ ] **Step 4: Verify the check passes, and fails on a real disagreement**

```bash
cd ~/projects/iit-expert-plugin && python3 scripts/check_manifests.py; echo "clean: $?"
sed -i '' 's/"version": "0.1.0"/"version": "0.1.9"/' plugin.json
python3 scripts/check_manifests.py; echo "broken: $?"
sed -i '' 's/"version": "0.1.9"/"version": "0.1.0"/' plugin.json
python3 scripts/check_manifests.py; echo "restored: $?"
```
Expected: `clean: 0`; then `.claude-plugin/plugin.json: version differs from plugin.json` and `broken: 1`; then `restored: 0`.

- [ ] **Step 5: Rewrite the README**

Apply the `writing-naturally` skill. Content: one paragraph on what IIT Expert is (answers from IIT's primary literature, skill plus connector, the skill makes the assistant consult the connector before answering); a "Work in progress" paragraph (the corpus and glossary are still being vetted; answers may be incomplete; report problems as GitHub issues); an install section with exactly these commands:

````markdown
**Claude Code**

```
claude plugin marketplace add wmayner/iit-expert-plugin
claude plugin install iit-expert@iit-expert
```

**Codex**

```
codex plugin marketplace add wmayner/iit-expert-plugin
codex plugin add iit-expert@iit-expert
```

**Cursor:** open Customize → From GitHub Repository and enter `wmayner/iit-expert-plugin`.

**claude.ai and Claude Desktop** cannot install plugins from this repository; follow <https://learniit.org/install>.
````

then one line pointing to PyPhi's MCP server (`https://pyphi.readthedocs.io/en/latest/howto/ai-assistants.html`) for computing IIT quantities; then the licence line kept from the current README. Remove every `reference.iit.wiki` and `mcp.iit.wiki` mention.

- [ ] **Step 6: Validate and install locally in isolated Claude Code and Codex homes**

```bash
P=~/projects/iit-expert-plugin; S=$(mktemp -d); C=$(which -a claude | grep -v alias | head -1)
$C plugin validate $P
CLAUDE_CONFIG_DIR=$S/claude $C plugin marketplace add $P && CLAUDE_CONFIG_DIR=$S/claude $C plugin install iit-expert@iit-expert
CLAUDE_CONFIG_DIR=$S/claude $C plugin details iit-expert@iit-expert
mkdir -p $S/codex $S/home
CODEX_HOME=$S/codex HOME=$S/home codex plugin marketplace add $P && CODEX_HOME=$S/codex HOME=$S/home codex plugin add iit-expert@iit-expert
CODEX_HOME=$S/codex HOME=$S/home codex mcp list
```
Expected: `Validation passed` (a warning is acceptable, an error is not); `details` shows `Skills (1) iit-expert` and `MCP servers (1) iit-expert`; `codex mcp list` shows `iit-expert  https://mcp.learniit.org  …  enabled`.

- [ ] **Step 7: Commit**

```bash
cd ~/projects/iit-expert-plugin && git add -A && git commit -m "Publish the IIT Expert skill and connector as a cross-agent plugin

One skill directory and one connector serve Claude Code and Codex through
.claude-plugin/ and .mcp.json, and Cursor through the Agent Plugins files.
A CI check keeps the manifests consistent."
```
(End the message with the attribution lines from the session's system reminder.)

---

### Task 2: Create the GitHub repository and verify from it

**Needs the user's go-ahead before Step 1.**

- [ ] **Step 1: Create and push** (after the user approves)

```bash
cd ~/projects/iit-expert-plugin && gh repo create wmayner/iit-expert-plugin --public --source . --push --description "IIT Expert: a skill and connector for Integrated Information Theory's primary literature (work in progress)"
```

- [ ] **Step 2: Verify the install from GitHub** in fresh isolated homes, repeating Task 1 Step 6 with `wmayner/iit-expert-plugin` in place of `$P`. Expected: the same results. Confirm the `check` workflow passed with `gh run list -R wmayner/iit-expert-plugin -L 1`.

- [ ] **Step 3: Ask the user to check Cursor by hand**: Customize → From GitHub Repository → `wmayner/iit-expert-plugin`, then confirm the `iit-expert` skill and connector appear. If Cursor rejects the plugin, fix the Agent Plugins files, rerun Task 1 Steps 4 and 6, commit, and ask the user to push.

---

### Task 3: Sync script, install page and labels in `iit-reference`

**Files (under `~/projects/iit-reference/`):**
- Create: `build/sync_plugin.py`, `tests/test_sync_plugin.py`
- Modify: `build/build.py:66-96` (`_INSTALL`), `build/build.py:49-59` (`_HOME`)
- Modify: `tests/test_render.py:412-437`

**Interfaces:**
- Produces: `uv run python -m build.sync_plugin <plugin-checkout> --version X.Y.Z` copies the skill and writes `version` into both plugin manifests.

- [ ] **Step 1: Write the failing sync test**

`tests/test_sync_plugin.py`:
```python
import json

import pytest

from build.sync_plugin import sync


def _plugin(tmp_path):
    (tmp_path / ".claude-plugin").mkdir(parents=True)
    for path in ("plugin.json", ".claude-plugin/plugin.json"):
        (tmp_path / path).write_text(json.dumps({"name": "iit-expert", "version": "0.1.0"}))
    return tmp_path


def test_copies_the_skill_and_sets_the_version(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: iit-expert\n---\nnew\n")
    plugin = _plugin(tmp_path / "plugin")
    sync(source, plugin, "0.2.0")
    assert (plugin / "skills" / "iit-expert" / "SKILL.md").read_text().endswith("new\n")
    for path in ("plugin.json", ".claude-plugin/plugin.json"):
        assert json.loads((plugin / path).read_text())["version"] == "0.2.0"


def test_refuses_a_directory_that_is_not_the_plugin(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "SKILL.md").write_text("x")
    with pytest.raises(FileNotFoundError):
        sync(source, tmp_path / "elsewhere", "0.2.0")
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd ~/projects/iit-reference && uv run pytest tests/test_sync_plugin.py -q > /tmp/sync.log 2>&1; tail -3 /tmp/sync.log`
Expected: collection error, `No module named 'build.sync_plugin'`.

- [ ] **Step 3: Write the script**

`build/sync_plugin.py`:
```python
"""Copy the skill into a checkout of the IIT Expert plugin and set its version.

The skill's source is ``surfaces/skill/``; the plugin repository and the
download on the site are both generated from it. Claude Code offers an update
only when the manifest version changes, so every sync names a new one.

Usage: ``uv run python -m build.sync_plugin <plugin-checkout> --version X.Y.Z``
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ("plugin.json", ".claude-plugin/plugin.json")


def sync(source: Path, plugin: Path, version: str) -> None:
    for relative in MANIFESTS:
        if not (plugin / relative).is_file():
            raise FileNotFoundError(f"{plugin / relative} not found; is this the plugin checkout?")
    destination = plugin / "skills" / "iit-expert"
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)
    for relative in MANIFESTS:
        path = plugin / relative
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["version"] = version
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plugin", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    sync(ROOT / "surfaces" / "skill", args.plugin, args.version)
    print(f"synced the skill into {args.plugin} at version {args.version}")
```

- [ ] **Step 4: Run it to see it pass**

Run: `uv run pytest tests/test_sync_plugin.py -q > /tmp/sync.log 2>&1; tail -3 /tmp/sync.log`
Expected: `2 passed`.

- [ ] **Step 5: Update the install-page test first**

In `tests/test_render.py`, `test_the_skill_is_bundled_and_the_install_page_is_written`, after the existing `href="/iit-expert-skill.zip"` assertion, replace `assert "Add custom connector" in page` with:
```python
    assert "claude plugin marketplace add wmayner/iit-expert-plugin" in page
    assert "codex plugin add iit-expert@iit-expert" in page
    assert "From GitHub Repository" in page
    assert "Add custom connector" in page
    assert "work in progress" in page.lower()
    assert "cursor://anysphere.cursor-deeplink/mcp/install" in page
```
and after the `home = …` line add `assert "work in progress" in home.lower()`.

Run: `uv run pytest tests/test_render.py -q -k install_page > /tmp/r.log 2>&1; tail -3 /tmp/r.log` — Expected: 1 failed.

- [ ] **Step 6: Rewrite `_INSTALL` and `_HOME`**

Apply the `writing-naturally` skill to the prose. Compute the Cursor link once in `build.py` beside `MCP_URL`:
```python
_CURSOR_LINK = (
    "cursor://anysphere.cursor-deeplink/mcp/install?name=iit-expert&config="
    + base64.b64encode(json.dumps({"url": MCP_URL}).encode()).decode()
)
```
(add `import base64` and `import json` if absent) and pass `cursor=_CURSOR_LINK` to the `_INSTALL.format(...)` call in `_write_skill`. Structure of the new `_INSTALL` (keep `{mcp}`, `{zip}`, add `{cursor}`):

1. Opening paragraph: the connector gives the assistant the sources, the skill makes it read them before answering; install both. One sentence: IIT Expert is a work in progress, and the corpus is still being checked.
2. `## Claude Code, Codex and Cursor`: one plugin installs both. The Claude Code, Codex and Cursor commands exactly as in Task 1 Step 5.
3. `## claude.ai and Claude Desktop`: the existing three connector steps (keep "Add custom connector" and `{mcp}`), then the skill upload via `[the skill]({zip})` and **Settings → Skills → Add**.
4. `## The connector alone`: `{mcp}` for any MCP client; for Cursor, `[add it to Cursor]({cursor})`.
5. `## Computing with PyPhi`: PyPhi has its own MCP server that computes IIT quantities; link `https://pyphi.readthedocs.io/en/latest/howto/ai-assistants.html`.

In `_HOME`, add after the first paragraph: `IIT Expert is a work in progress: the corpus and glossary are still being checked against the sources.`

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q > /tmp/r.log 2>&1; tail -3 /tmp/r.log` — Expected: all pass.

- [ ] **Step 8: Sync the plugin and check it**

```bash
cd ~/projects/iit-reference && uv run python -m build.sync_plugin ~/projects/iit-expert-plugin --version 0.1.0
cd ~/projects/iit-expert-plugin && python3 scripts/check_manifests.py && git status --short
```
Expected: check exits 0; `git status` is empty (Task 1 already copied the same skill) or shows only `SKILL.md` if the source changed since, in which case commit it in the plugin repo.

- [ ] **Step 9: Commit in `iit-reference`**

```bash
cd ~/projects/iit-reference && git add build/sync_plugin.py tests/test_sync_plugin.py build/build.py tests/test_render.py && git commit -m "Lead the install page with the cross-agent plugin

Adds a sync script that copies the skill into the plugin checkout and sets
its version, rewrites the install page around the plugin commands for
Claude Code, Codex and Cursor, and labels the site a work in progress."
```

- [ ] **Step 10: Deploy** (after the user approves): `npm run deploy`, then `curl -s https://learniit.org/install.md | grep -c "iit-expert@iit-expert"` — Expected: at least `1`.

---

### Task 4: Retire PyPhi's `iit` skill

**Files:**
- Delete: `pyphi/mcp/skills/iit/`
- Modify: `pyphi/mcp/agents.py` (constants after `REFERENCED`; `deliver`; `remove`)
- Test: `test/mcp/test_agents.py`

**Interfaces:**
- Produces: `agents.RETIRED: frozenset[str]`; `agents._remove_marked(path: Path, names: Iterable[str]) -> list[str]`.

- [ ] **Step 1: Update the existing tests and add the retirement tests**

In `test/mcp/test_agents.py`:
- `test_ships_both_skills` → rename `test_ships_the_library_skill`, assert `mod.skill_names() == ["pyphi"]`.
- `test_stamps_a_sentinel_holding_the_version`: use `"pyphi"` instead of `"iit"`.
- Delete `test_the_gate_skill_has_no_references` and `test_the_gate_says_not_to_answer_from_recollection`.
- `test_delivering_twice_refreshes_rather_than_failing`, `test_removes_what_deliver_wrote`, `test_leaves_a_hand_written_skill_of_the_same_name`: use `"pyphi"`; in the removal test assert `== ["pyphi"]`.
- In `TestFlow`: every `"iit"` becomes `"pyphi"`.
- In `test_the_skills_reach_a_built_wheel`: replace the `iit` assertion with `assert not any(name.startswith("pyphi/mcp/skills/iit/") for name in shipped)`.

Add to `TestRemoval`:
```python
    def _retired(self, tmp_path, marked=True):
        old = tmp_path / "iit"
        old.mkdir()
        (old / "SKILL.md").write_text("old", encoding="utf-8")
        if marked:
            (old / mod.SENTINEL).write_text("2.0.0rc1\n", encoding="utf-8")
        return old

    def test_delivery_removes_a_retired_skill(self, tmp_path):
        old = self._retired(tmp_path)
        mod.deliver(mod.Target("t", "t", tmp_path))
        assert not old.exists()

    def test_removal_removes_a_retired_skill(self, tmp_path):
        old = self._retired(tmp_path)
        assert "iit" in mod.remove(mod.Target("t", "t", tmp_path))
        assert not old.exists()

    def test_a_hand_written_skill_with_a_retired_name_survives(self, tmp_path):
        old = self._retired(tmp_path, marked=False)
        target = mod.Target("t", "t", tmp_path)
        mod.deliver(target)
        mod.remove(target)
        assert (old / "SKILL.md").read_text(encoding="utf-8") == "old"
```

- [ ] **Step 2: Run to see the failures**

Run: `uv run pytest test/mcp/test_agents.py -q > /tmp/agents.log 2>&1; tail -5 /tmp/agents.log`
Expected: failures in `test_ships_the_library_skill` and the three retirement tests.

- [ ] **Step 3: Implement**

```bash
git rm -r pyphi/mcp/skills/iit
```

In `pyphi/mcp/agents.py`, replace the `REFERENCED` comment's second sentence (it describes the gate skill) so the block reads:
```python
#: Skills whose ``references/`` is filled from the reference topics at install
#: time.
REFERENCED: frozenset[str] = frozenset({"pyphi"})

#: Skills earlier versions of PyPhi installed and this one no longer ships.
#: Installing and uninstalling both delete a copy PyPhi wrote.
RETIRED: frozenset[str] = frozenset({"iit"})
```
Add `from collections.abc import Iterable` to the imports. Add above `deliver`:
```python
def _remove_marked(path: Path, names: Iterable[str]) -> list[str]:
    """Delete each named skill under ``path`` that holds the sentinel file."""
    removed = []
    for name in names:
        destination = path / name
        if (destination / SENTINEL).is_file():
            shutil.rmtree(destination)
            removed.append(name)
    return removed
```
At the top of `deliver`'s body add `_remove_marked(target.path, sorted(RETIRED))`. Replace the loop in `remove` with:
```python
    return _remove_marked(target.path, [*skill_names(), *sorted(RETIRED)])
```
(keeping the `if not target.path.is_dir(): return []` guard above it).

- [ ] **Step 4: Run to see them pass**

Run: `uv run pytest test/mcp -q > /tmp/mcp.log 2>&1; tail -3 /tmp/mcp.log`
Expected: all pass (the wheel test is `slow` and deselected unless `--slow`).

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/agents.py test/mcp/test_agents.py pyphi/mcp/skills
git commit -m "Retire the iit skill from pyphi-mcp install

The iit-expert skill from the IIT Expert plugin covers the theory, and
two skills triggering on the same questions gave assistants overlapping
instructions. Installing or uninstalling now deletes a copy an earlier
PyPhi wrote, and leaves a hand-written skill of the same name alone."
```

---

### Task 5: Stop writing a second copy of the `pyphi` skill for Cursor

**Files:**
- Modify: `pyphi/mcp/agents.py` (new constant, new `_split_cursor`, `install_step`, `describe`)
- Test: `test/mcp/test_agents.py`

**Interfaces:**
- Consumes: `Target`, `resolve`, `remove` (existing).
- Produces: `agents.CURSOR_READS: frozenset[str]`; `agents._split_cursor(targets: list[Target], explicit: bool) -> tuple[list[Target], Target | None]`.

- [ ] **Step 1: Write the failing tests** (new class in `test/mcp/test_agents.py`)

```python
class TestCursorDeduplication:
    def test_cursor_is_skipped_when_it_already_reads_another_target(self, tmp_path):
        for probe in (".claude", ".cursor"):
            (tmp_path / probe).mkdir()
        actions = mod.install_step(skills=True, names=[], paths=[], home=tmp_path)
        assert (tmp_path / ".claude" / "skills" / "pyphi" / "SKILL.md").is_file()
        assert not (tmp_path / ".cursor" / "skills" / "pyphi").exists()
        assert any("Cursor" in line for line in actions)

    def test_an_old_cursor_copy_is_removed(self, tmp_path):
        for probe in (".claude", ".cursor"):
            (tmp_path / probe).mkdir()
        mod.deliver(mod.Target("cursor", "Cursor", tmp_path / ".cursor" / "skills"))
        mod.install_step(skills=True, names=[], paths=[], home=tmp_path)
        assert not (tmp_path / ".cursor" / "skills" / "pyphi").exists()

    def test_cursor_alone_gets_the_skill(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        mod.install_step(skills=True, names=[], paths=[], home=tmp_path)
        assert (tmp_path / ".cursor" / "skills" / "pyphi" / "SKILL.md").is_file()

    def test_naming_cursor_explicitly_writes_there(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        mod.install_step(skills=True, names=["claude-code", "cursor"], paths=[], home=tmp_path)
        assert (tmp_path / ".cursor" / "skills" / "pyphi" / "SKILL.md").is_file()

    def test_describe_matches_what_install_writes(self, tmp_path):
        for probe in (".codex", ".cursor"):
            (tmp_path / probe).mkdir()
        lines = mod.describe(names=[], paths=[], home=tmp_path)
        assert not any(str(tmp_path / ".cursor" / "skills") + ":" in line for line in lines)
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest test/mcp/test_agents.py -q -k Cursor > /tmp/c.log 2>&1; tail -3 /tmp/c.log` — Expected: 3 failed (`skipped`, `old copy`, `describe`), 2 passed.

- [ ] **Step 3: Implement**

In `pyphi/mcp/agents.py`, after `RETIRED`:
```python
#: Agents whose skills directory Cursor also reads, so writing to Cursor's own
#: directory as well would load every skill twice there.
CURSOR_READS: frozenset[str] = frozenset({"claude-code", "codex"})
```
Add after `resolve`:
```python
def _split_cursor(
    targets: list[Target], explicit: bool
) -> tuple[list[Target], Target | None]:
    """Separate a detected Cursor target that another target already covers.

    Returns the targets to write and the Cursor target left out, if any. A
    target the user named is always written.
    """
    names = {target.name for target in targets}
    if explicit or "cursor" not in names or not names & CURSOR_READS:
        return targets, None
    cursor = next(target for target in targets if target.name == "cursor")
    return [target for target in targets if target is not cursor], cursor
```
In `install_step`, replace `targets = resolve(names, paths, home=home)` with:
```python
    targets, covered = _split_cursor(
        resolve(names, paths, home=home), explicit=bool(names or paths)
    )
```
and, just before `return actions`, add:
```python
    if covered is not None:
        remove(covered)
        actions.append(
            f"left {covered.display} out: it reads the skills written for "
            "the other agents"
        )
```
In `describe`, replace `for target in resolve(names, paths, home=home)` with `for target in _split_cursor(resolve(names, paths, home=home), explicit=bool(names or paths))[0]`.

- [ ] **Step 4: Run to see them pass**

Run: `uv run pytest test/mcp -q > /tmp/mcp.log 2>&1; tail -3 /tmp/mcp.log` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/agents.py test/mcp/test_agents.py
git commit -m "Write the pyphi skill once for Cursor

Cursor reads the Claude Code and Codex skills directories as well as its
own, so installing into all three loaded the skill twice there. A detected
Cursor target is now left out when either of the others is written, and a
copy an earlier install left in it is removed."
```

---

### Task 6: Offer the IIT Expert plugin from `pyphi-mcp install`

**Files:**
- Modify: `pyphi/mcp/agents.py` (imports, constants, `_execute`, `plugin_step`, `describe_plugin`, `plugin_removal_hint`)
- Modify: `pyphi/mcp/install.py` (`build_parser`, `run`)
- Modify: `test/mcp/conftest.py`
- Test: `test/mcp/test_agents.py`, `test/mcp/test_install.py`

**Interfaces:**
- Consumes: `resolve`, `interactive`, `confirm`, `AGENTS`, `Target`.
- Produces:
  - `agents.PLUGIN: str = "iit-expert@iit-expert"`, `agents.PLUGIN_REPOSITORY: str = "wmayner/iit-expert-plugin"`, `agents.INSTALL_PAGE: str = "https://learniit.org/install"`, `agents.PLUGIN_TIMEOUT: int = 120`
  - `agents.PLUGIN_COMMANDS: dict[str, tuple[tuple[str, ...], ...]]`, `agents.PLUGIN_REMOVAL: dict[str, tuple[str, ...]]`
  - `agents._execute(command: tuple[str, ...]) -> str | None` (None on success, otherwise why it failed)
  - `agents.plugin_step(*, plugin: bool | None, names: list[str], paths: list[Path], home: Path | None = None) -> list[str]`
  - `agents.describe_plugin(*, names, paths, home=None) -> list[str]`
  - `agents.plugin_removal_hint(*, names, paths, home=None) -> list[str]`
  - CLI flags `--iit-expert` / `--no-iit-expert` → `args.iit_expert: bool | None`

- [ ] **Step 1: Guard the suite against real agent commands**

Append to `test/mcp/conftest.py`:
```python
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
```

- [ ] **Step 2: Write the failing tests**

At the top of `test/mcp/test_agents.py`, after `from pyphi.mcp import agents as mod`, add (the name binds the real function before the autouse fixture replaces the attribute):
```python
import subprocess
import sys

from pyphi.mcp.agents import _execute as real_execute
```
Add:
```python
class TestExecute:
    def test_a_missing_executable_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))
        assert "not on PATH" in real_execute(("claude", "plugin", "list"))

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
    def test_a_failing_command_is_reported_with_its_status(self, monkeypatch, tmp_path):
        script = tmp_path / "claude"
        script.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        error = real_execute(("claude", "plugin", "list"))
        assert "status 3" in error and "boom" in error

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script")
    def test_a_succeeding_command_returns_none(self, monkeypatch, tmp_path):
        script = tmp_path / "claude"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert real_execute(("claude", "plugin", "list")) is None

    def test_a_hung_command_times_out(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: f"/bin/{name}")

        def hang(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], mod.PLUGIN_TIMEOUT)

        monkeypatch.setattr(mod.subprocess, "run", hang)
        assert "timed out" in real_execute(("claude", "plugin", "list"))


class TestPluginStep:
    def _home(self, tmp_path, *probes):
        for probe in probes:
            (tmp_path / probe).mkdir()
        return tmp_path

    def _record(self, monkeypatch, fail_on=None):
        ran = []

        def execute(command):
            ran.append(command)
            return "exited with status 1" if command == fail_on else None

        monkeypatch.setattr(mod, "_execute", execute)
        return ran

    def test_the_guard_refuses_real_commands(self):
        with pytest.raises(pytest.fail.Exception):
            mod._execute(("claude", "plugin", "list"))

    def test_no_agents_means_nothing(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        assert mod.plugin_step(plugin=True, names=[], paths=[], home=tmp_path) == []
        assert ran == []

    def test_declining_runs_nothing(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        home = self._home(tmp_path, ".claude")
        assert mod.plugin_step(plugin=False, names=[], paths=[], home=home) == []
        assert ran == []

    def test_non_interactive_prints_how_instead_of_running(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        monkeypatch.setattr(mod, "interactive", lambda: False)
        home = self._home(tmp_path, ".claude")
        (line,) = mod.plugin_step(plugin=None, names=[], paths=[], home=home)
        assert "--iit-expert" in line and mod.INSTALL_PAGE in line
        assert ran == []

    def test_the_prompt_names_every_detected_agent(self, tmp_path, monkeypatch):
        self._record(monkeypatch)
        monkeypatch.setattr(mod, "interactive", lambda: True)
        asked = []
        monkeypatch.setattr(mod, "confirm", lambda question: asked.append(question) or False)
        home = self._home(tmp_path, ".claude", ".codex")
        mod.plugin_step(plugin=None, names=[], paths=[], home=home)
        assert "IIT Expert" in asked[0] and "Claude Code, Codex" in asked[0]

    def test_accepting_runs_each_agents_commands_in_order(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        home = self._home(tmp_path, ".claude", ".codex")
        actions = mod.plugin_step(plugin=True, names=[], paths=[], home=home)
        assert ran == [
            *mod.PLUGIN_COMMANDS["claude-code"],
            *mod.PLUGIN_COMMANDS["codex"],
        ]
        assert ran[0] == ("claude", "plugin", "marketplace", "add", mod.PLUGIN_REPOSITORY)
        assert sum("installed the IIT Expert plugin" in line for line in actions) == 2

    def test_a_failure_stops_that_agent_and_prints_its_commands(self, tmp_path, monkeypatch):
        first = mod.PLUGIN_COMMANDS["claude-code"][0]
        ran = self._record(monkeypatch, fail_on=first)
        home = self._home(tmp_path, ".claude", ".codex")
        actions = mod.plugin_step(plugin=True, names=[], paths=[], home=home)
        assert mod.PLUGIN_COMMANDS["claude-code"][1] not in ran
        assert mod.PLUGIN_COMMANDS["codex"][1] in ran
        failure = next(line for line in actions if "could not" in line)
        assert "claude plugin install iit-expert@iit-expert" in failure

    def test_cursor_gets_instructions_not_commands(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        home = self._home(tmp_path, ".cursor")
        (line,) = mod.plugin_step(plugin=True, names=[], paths=[], home=home)
        assert "From GitHub Repository" in line and mod.PLUGIN_REPOSITORY in line
        assert ran == []

    def test_an_explicit_skills_directory_is_not_an_agent(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        assert mod.plugin_step(plugin=True, names=[], paths=[tmp_path], home=tmp_path) == []
        assert ran == []

    def test_describe_lists_the_commands_and_runs_nothing(self, tmp_path, monkeypatch):
        ran = self._record(monkeypatch)
        home = self._home(tmp_path, ".codex")
        text = "\n".join(mod.describe_plugin(names=[], paths=[], home=home))
        assert "codex plugin add iit-expert@iit-expert" in text
        assert ran == []

    def test_uninstall_says_how_to_remove_the_plugin(self, tmp_path):
        home = self._home(tmp_path, ".claude")
        (line,) = mod.plugin_removal_hint(names=[], paths=[], home=home)
        assert "claude plugin uninstall iit-expert@iit-expert" in line
```
Add to `TestCommandLine` in `test/mcp/test_install.py`:
```python
    def test_iit_expert_defaults_to_asking(self):
        assert mod.build_parser().parse_args(["install"]).iit_expert is None

    def test_no_iit_expert_is_false(self):
        args = mod.build_parser().parse_args(["install", "--no-iit-expert"])
        assert args.iit_expert is False

    def test_iit_expert_is_true(self):
        assert mod.build_parser().parse_args(["install", "--iit-expert"]).iit_expert is True

    def test_print_shows_the_plugin_commands(self, tmp_path, capsys, isolated_home):
        (isolated_home / ".codex").mkdir()
        args = mod.build_parser().parse_args(
            ["install", "--print", "--directory", str(tmp_path)]
        )
        assert mod.run(args) == 0
        assert "codex plugin add iit-expert@iit-expert" in capsys.readouterr().out
```

- [ ] **Step 3: Run to see them fail**

Run: `uv run pytest test/mcp -q > /tmp/mcp.log 2>&1; tail -5 /tmp/mcp.log`
Expected: failures and errors naming `_execute`, `plugin_step`, `PLUGIN_COMMANDS`, `iit_expert`.

- [ ] **Step 4: Implement in `agents.py`**

Add `import subprocess` to the imports. Update the module docstring's first paragraph to end: "…and the IIT Expert plugin, which covers the theory, is installed through each agent's own plugin commands." Add after `CURSOR_READS`:
```python
#: The IIT Expert plugin: the ``iit-expert`` skill and the connector to the
#: IIT literature at ``mcp.learniit.org``, published for Claude Code, Codex and
#: Cursor from one repository.
PLUGIN = "iit-expert@iit-expert"
PLUGIN_REPOSITORY = "wmayner/iit-expert-plugin"
INSTALL_PAGE = "https://learniit.org/install"

#: Seconds each plugin command may take; adding the marketplace clones it.
PLUGIN_TIMEOUT = 120

#: Agent name mapped to the commands that install the plugin, run in order.
#: Both agents treat a repeat as success, so running ``install`` again is safe.
PLUGIN_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "claude-code": (
        ("claude", "plugin", "marketplace", "add", PLUGIN_REPOSITORY),
        ("claude", "plugin", "install", PLUGIN),
    ),
    "codex": (
        ("codex", "plugin", "marketplace", "add", PLUGIN_REPOSITORY),
        ("codex", "plugin", "add", PLUGIN),
    ),
}

#: Agent name mapped to the command that removes the plugin.
PLUGIN_REMOVAL: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude", "plugin", "uninstall", PLUGIN),
    "codex": ("codex", "plugin", "remove", PLUGIN),
}

#: Cursor installs plugins from its settings rather than a command line.
CURSOR_STEPS = (
    f"in Cursor, open Customize → From GitHub Repository and enter "
    f"{PLUGIN_REPOSITORY}"
)
```
Add after `confirm`:
```python
def _shown(commands: tuple[tuple[str, ...], ...]) -> str:
    return "; ".join(" ".join(command) for command in commands)


def _execute(command: tuple[str, ...]) -> str | None:
    """Run one plugin command.

    Returns
    -------
    str or None
        None if the command succeeded, otherwise why it did not.
    """
    shown = " ".join(command)
    executable = shutil.which(command[0])
    if executable is None:
        return f"`{command[0]}` is not on PATH"
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=PLUGIN_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"`{shown}` timed out after {PLUGIN_TIMEOUT} s"
    except OSError as error:
        return f"`{shown}` could not start: {error}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = f": {detail[-1]}" if detail else ""
        return f"`{shown}` exited with status {result.returncode}{last}"
    return None


def _agents(names: list[str], paths: list[Path], home: Path | None) -> list[Target]:
    """The known agents among the targets; an explicit directory is not one."""
    return [target for target in resolve(names, paths, home=home) if target.name in AGENTS]


def plugin_step(
    *,
    plugin: bool | None,
    names: list[str],
    paths: list[Path],
    home: Path | None = None,
) -> list[str]:
    """Offer the IIT Expert plugin and install it through each agent's commands.

    Parameters
    ----------
    plugin : bool or None
        True installs without asking, False skips, None asks where a person is
        there to answer and skips otherwise.
    names : list of str
        Agents named explicitly, whether or not they were detected.
    paths : list of Path
        Skills directories given explicitly; these belong to no known agent
        and are ignored here.
    home : Path, optional
        The directory agents are resolved under. If None, the user's home
        directory.

    Returns
    -------
    list of str
        One line per agent. A command that is missing or fails is reported
        with the commands to run by hand; it never fails the install.
    """
    targets = _agents(names, paths, home)
    if not targets or plugin is False:
        return []
    if plugin is None:
        if not interactive():
            return [
                "skipped the IIT Expert plugin; run `pyphi-mcp install "
                f"--iit-expert` to add it, or see {INSTALL_PAGE}"
            ]
        displayed = ", ".join(target.display for target in targets)
        if not confirm(
            f"Install the IIT Expert plugin (skill + connector) for {displayed}?"
        ):
            return []
    actions = []
    for target in targets:
        commands = PLUGIN_COMMANDS.get(target.name)
        if commands is None:
            actions.append(f"to add IIT Expert: {CURSOR_STEPS}")
            continue
        for command in commands:
            error = _execute(command)
            if error is not None:
                actions.append(
                    f"could not install the IIT Expert plugin in "
                    f"{target.display} ({error}); run: {_shown(commands)}"
                )
                break
        else:
            actions.append(f"installed the IIT Expert plugin in {target.display}")
    return actions


def describe_plugin(
    *, names: list[str], paths: list[Path], home: Path | None = None
) -> list[str]:
    """Return what :func:`plugin_step` would run, running nothing."""
    return [
        f"IIT Expert plugin for {target.display}: "
        + (_shown(PLUGIN_COMMANDS[target.name]) if target.name in PLUGIN_COMMANDS else CURSOR_STEPS)
        for target in _agents(names, paths, home)
    ]


def plugin_removal_hint(
    *, names: list[str], paths: list[Path], home: Path | None = None
) -> list[str]:
    """Say how to remove the plugin, which PyPhi installs but does not own."""
    return [
        f"the IIT Expert plugin stays installed in {target.display}; "
        f"remove it with: {' '.join(PLUGIN_REMOVAL[target.name])}"
        for target in _agents(names, paths, home)
        if target.name in PLUGIN_REMOVAL
    ]
```
Run `uv run ruff format pyphi/mcp/agents.py` afterwards (the `describe_plugin` line will be rewrapped).

- [ ] **Step 5: Implement in `install.py`**

In `build_parser`, after the `skills` group:
```python
    iit_expert = install_parser.add_mutually_exclusive_group()
    iit_expert.add_argument(
        "--iit-expert",
        dest="iit_expert",
        action="store_true",
        default=None,
        help="install the IIT Expert plugin (skill + connector) without asking",
    )
    iit_expert.add_argument(
        "--no-iit-expert",
        dest="iit_expert",
        action="store_false",
        help="do not install the IIT Expert plugin",
    )
```
In `run`, in the `print_only` branch after the `agents.describe` loop:
```python
                for line in agents.describe_plugin(
                    names=args.agent, paths=args.agent_path
                ):
                    print(f"\n{line}")
```
after `actions += agents.install_step(...)`:
```python
            actions += agents.plugin_step(
                plugin=args.iit_expert, names=args.agent, paths=args.agent_path
            )
```
and in the uninstall branch, after `actions = actions or ["nothing to remove"]`:
```python
            actions += agents.plugin_removal_hint(
                names=args.agent, paths=args.agent_path
            )
```

- [ ] **Step 6: Run to see them pass**

Run: `uv run pytest test/mcp -q > /tmp/mcp.log 2>&1; tail -3 /tmp/mcp.log` — Expected: all pass. Then `uv run pyright pyphi/mcp/agents.py pyphi/mcp/install.py` — Expected: 0 errors.

- [ ] **Step 7: Verify the guard can fail**

Temporarily delete the `no_agent_commands` fixture from `conftest.py` and run `uv run pytest test/mcp/test_agents.py -q -k guard_refuses > /tmp/g.log 2>&1; tail -3 /tmp/g.log`. Expected: 1 failed (the real `_execute` returns instead of raising). Restore the fixture and rerun: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add pyphi/mcp/agents.py pyphi/mcp/install.py test/mcp/conftest.py test/mcp/test_agents.py test/mcp/test_install.py
git commit -m "Offer the IIT Expert plugin from pyphi-mcp install

For each detected agent with a plugin command line (Claude Code, Codex),
install runs that agent's own commands to add the plugin, which brings
the iit-expert skill and the learniit.org connector and lets the agent
handle updates. Cursor gets the steps to follow. A missing or failing
command prints the commands to run and never fails the install;
--iit-expert and --no-iit-expert answer the prompt without a terminal."
```

---

### Task 7: Point the `pyphi` skill and the server primer at IIT Expert

**Files:**
- Modify: `pyphi/mcp/skills/pyphi/SKILL.md:14-15`
- Modify: `pyphi/mcp/content/primer.md:3-8`
- Test: `test/mcp/test_agents.py` (`TestShippedSkills`)

- [ ] **Step 1: Write the failing test**

```python
    def test_the_library_skill_points_to_iit_expert_for_the_theory(self):
        _, text = self._front_matter("pyphi")
        assert "`iit-expert`" in text
        assert "`iit` skill" not in text

    def test_the_primer_names_the_iit_expert_connector(self):
        from pyphi.mcp import content

        assert "mcp.learniit.org" in content.load("primer")
```
Run: `uv run pytest test/mcp/test_agents.py -q -k "iit_expert or connector" > /tmp/t.log 2>&1; tail -3 /tmp/t.log` — Expected: 2 failed.

- [ ] **Step 2: Edit the skill**

Replace lines 14–15 of `pyphi/mcp/skills/pyphi/SKILL.md`:
```markdown
PyPhi computes Integrated Information Theory quantities. For what the theory
says, use the `iit-expert` skill from the IIT Expert plugin
(`pyphi-mcp install` offers it; see https://learniit.org/install); this one is
about the software.
```

- [ ] **Step 3: Edit the primer**

Append to the first paragraph of `pyphi/mcp/content/primer.md` (after "`pyphi://theory/*` resources."):
```markdown
For what the theory says beyond interpreting a result, prefer the IIT Expert
connector (`mcp.learniit.org`) where it is connected: it serves IIT's primary
literature, with a locator for every claim.
```

- [ ] **Step 4: Run to see them pass, then the MCP suite**

Run: `uv run pytest test/mcp -q > /tmp/mcp.log 2>&1; tail -3 /tmp/mcp.log` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/skills/pyphi/SKILL.md pyphi/mcp/content/primer.md test/mcp/test_agents.py
git commit -m "Point the pyphi skill and the server primer at IIT Expert

The pyphi skill named the retired iit skill for questions about the
theory; it now names iit-expert. The server's startup instructions tell
an assistant to prefer the IIT Expert connector for the theory where it
is connected."
```

---

### Task 8: PyPhi documentation, changelog and roadmap

**Files:**
- Create: `docs/howto/ai-assistants.md`, `changelog.d/iit-expert-plugin.change.md`, `changelog.d/ai-assistants-page.doc.md`
- Modify: `docs/howto/index.md`, `docs/index.md:65-74`, `docs/howto/mcp-server.md` (top; "Skills for your coding agent" section, lines 59–90), `README.md:99-105`, `docs/whats-new-in-2.0.md:459-469`, `ROADMAP.md` (Status Dashboard)

- [ ] **Step 1: Load the `writing-naturally` skill** before writing any of the prose below.

- [ ] **Step 2: Write `docs/howto/ai-assistants.md`**

```markdown
# Use PyPhi with an AI assistant

Two tools make an AI assistant useful for work on IIT, and they are meant to be
used together:

- **IIT Expert** answers questions about the theory from its primary literature
  — the IIT wiki, the papers, a glossary of the postulates and measures — and
  cites where each claim comes from. It is hosted, so there is nothing to
  install beyond the plugin or connector.
- **The PyPhi MCP server** computes: it builds substrates, estimates the cost
  of an analysis, runs it, and plots the result. It runs locally, in the Python
  environment PyPhi is installed in.

IIT Expert is a work in progress. Its corpus and glossary are still being
checked against the sources.

## IIT Expert

IIT Expert has two parts: a connector, which gives the assistant the sources,
and a skill, which makes it read them before it answers. Install both; the
plugin does that in one step.

**Claude Code**

    claude plugin marketplace add wmayner/iit-expert-plugin
    claude plugin install iit-expert@iit-expert

**Codex**

    codex plugin marketplace add wmayner/iit-expert-plugin
    codex plugin add iit-expert@iit-expert

**Cursor:** open Customize → From GitHub Repository and enter
`wmayner/iit-expert-plugin`.

**claude.ai and Claude Desktop** add the connector and the skill separately;
<https://learniit.org/install> has the steps.

`pyphi-mcp install` offers to run the Claude Code and Codex commands for you.
The install page at <https://learniit.org/install> is kept current as IIT
Expert changes.

## The PyPhi MCP server

In a uv project:

    uv add "pyphi[mcp]"
    uv run pyphi-mcp install

{doc}`mcp-server` covers what `install` writes, other clients, and what the
server exposes.
```
(Use fenced code blocks in the real file; the indented blocks here avoid nesting fences in this plan.) Rewrite the prose with `writing-naturally`; keep every command exactly.

- [ ] **Step 3: Link the page**

`docs/howto/index.md`: change the bullet to `- work with an AI assistant: {doc}`ai-assistants`, {doc}`mcp-server``, and add `ai-assistants` above `mcp-server` in the "Start" toctree.

`docs/index.md`: replace the single `AI agents` grid (lines 65–74) with:
```markdown
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book` Learn IIT with an AI assistant
:link: howto/ai-assistants
:link-type: doc
IIT Expert answers questions about the theory from its primary literature, with
citations. A work in progress.
:::

:::{grid-item-card} {octicon}`dependabot` Compute with an AI assistant
:link: howto/mcp-server
:link-type: doc
PyPhi's MCP server lets an assistant build substrates, size runs, and analyze
them. This site is also readable by AI agents.
:::
::::
```

- [ ] **Step 4: Update `docs/howto/mcp-server.md`**

After the opening two paragraphs, add a `{tip}` admonition: for questions about the theory itself, pair the server with IIT Expert — {doc}`ai-assistants`.

Replace the "Skills for your coding agent" section's first paragraph and prompt example with: `install` offers the `pyphi` skill (the 2.0 API, state ordering, cost estimation, reproducible scripts) and then the IIT Expert plugin, which brings the `iit-expert` skill and the connector. Show both prompts:
```
Install the PyPhi skills for Claude Code, Codex? [Y/n]
Install the IIT Expert plugin (skill + connector) for Claude Code, Codex? [Y/n]
```
Then: `--iit-expert` / `--no-iit-expert` answer the second question as `--skills` / `--no-skills` answer the first; for Claude Code and Codex, `install` runs the agent's own plugin commands, and if one is missing or fails it prints them instead; for Cursor it prints the steps. Replace the sentence claiming it writes into `~/.cursor`: Cursor also reads the Claude Code and Codex skills folders, so `install` writes to Cursor's own folder only when neither of those agents is present. State that `uninstall` does not remove the plugin and prints the command that does. Keep the existing paragraphs about `--agent`, `--agent-path`, home-directory placement and refreshing.

- [ ] **Step 5: README and what's new**

`README.md`, "For AI assistants": before the MCP paragraph, add a paragraph naming IIT Expert (work in progress, answers from the primary literature with citations), the Claude Code plugin commands, and a link to `https://pyphi.readthedocs.io/en/latest/howto/ai-assistants.html` for Codex, Cursor and claude.ai.

`docs/whats-new-in-2.0.md`, "An interface for AI assistants": append one paragraph: `pyphi-mcp install` offers the IIT Expert plugin for questions about the theory; see [Use PyPhi with an AI assistant](howto/ai-assistants.md).

- [ ] **Step 6: Changelog fragments and roadmap**

`changelog.d/iit-expert-plugin.change.md`:
```markdown
`pyphi-mcp install` no longer installs the `iit` skill, and removes a copy an earlier version wrote. It offers the IIT Expert plugin instead, which brings the `iit-expert` skill and the connector to IIT's primary literature: for Claude Code and Codex it runs the agent's own plugin commands, and for Cursor it prints the steps. `--iit-expert` and `--no-iit-expert` answer the prompt without a terminal. The `pyphi` skill is no longer written to Cursor's skills folder when Claude Code or Codex is also present, since Cursor reads theirs.
```
`changelog.d/ai-assistants-page.doc.md`:
```markdown
Added [Use PyPhi with an AI assistant](howto/ai-assistants.md), which introduces IIT Expert for questions about the theory alongside the PyPhi MCP server for computation.
```
`ROADMAP.md`: add a dashboard row after "Single-formalism presentation": `| IIT Expert distribution | ✅ landed | 1 | IIT Expert (learniit.org) published as one plugin for Claude Code, Codex and Cursor (wmayner/iit-expert-plugin); pyphi-mcp install retires the iit skill and offers the plugin through each agent's plugin commands; Cursor no longer gets a duplicate pyphi skill; docs hub howto/ai-assistants.md. Spec/plan: docs/superpowers/{specs,plans}/2026-09-26-iit-expert-distribution* |` (match the column count of neighboring rows).

- [ ] **Step 7: Build the docs**

Run: `just docs > /tmp/docs.log 2>&1; tail -20 /tmp/docs.log`
Expected: build succeeds with no warnings about `ai-assistants`. Open `docs/_build/html/index.html` and `docs/_build/html/howto/ai-assistants.html` in a browser and check the two cards sit side by side on a wide window and stack on a narrow one.

- [ ] **Step 8: Commit**

```bash
git add docs/howto/ai-assistants.md docs/howto/index.md docs/index.md docs/howto/mcp-server.md README.md docs/whats-new-in-2.0.md changelog.d/iit-expert-plugin.change.md changelog.d/ai-assistants-page.doc.md ROADMAP.md
git commit -m "Document IIT Expert alongside the PyPhi MCP server

A new how-to introduces the two tools, one for the theory and one for
computation, with install steps for each agent; the landing page gets a
card for each; the MCP server page, README and what's new describe the
plugin offer in pyphi-mcp install."
```
(`ROADMAP.md` is staged whole; if another session has unrelated edits in it, stage only this row's hunk with `git add -p`.)

---

### Task 9: Final verification

- [ ] **Step 1: Full suite, including doctests**

Run: `uv run pytest > /tmp/full.log 2>&1; tail -5 /tmp/full.log`
Expected: summary line with 0 failed. The `test/parallel/test_dask_backend*` errors are known and unrelated if they appear; anything else is a failure to fix.

- [ ] **Step 2: Slow lane for the wheel test**

Run: `uv run pytest test/mcp -m slow --slow > /tmp/slow.log 2>&1; tail -3 /tmp/slow.log` — Expected: passes (the wheel carries `pyphi/mcp/skills/pyphi/` and no `iit/`).

- [ ] **Step 3: End-to-end install in a scratch home** (after Task 2 has pushed the plugin)

```bash
S=$(mktemp -d); mkdir -p $S/home/.claude $S/home/.codex $S/proj
cd $S/proj && uv init -q && uv add -q --editable ~/projects/pyphi --extra mcp
HOME=$S/home CLAUDE_CONFIG_DIR=$S/home/.claude CODEX_HOME=$S/home/.codex uv run pyphi-mcp install --skills --iit-expert
HOME=$S/home CLAUDE_CONFIG_DIR=$S/home/.claude CODEX_HOME=$S/home/.codex uv run pyphi-mcp install --skills --iit-expert
```
Expected, both runs: lines `installed the pyphi skills in …/.claude/skills`, `installed the IIT Expert plugin in Claude Code`, `installed the IIT Expert plugin in Codex`. Then `CODEX_HOME=$S/home/.codex codex mcp list` shows `iit-expert`.

- [ ] **Step 4: Report** the results, including the Cursor check from Task 2 Step 3, to the user.
