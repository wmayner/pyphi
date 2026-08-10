# Agent skills from `pyphi-mcp install` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pyphi-mcp install` offers to install two agent skills — `iit` and `pyphi` — into every AI coding agent it detects, so that a model is interrupted before it answers about IIT from recollection or writes PyPhi code from memory of PyPhi 1.x.

**Architecture:** The skills ship inside the wheel under `pyphi/mcp/skills/`. A new module, `pyphi/mcp/agents.py`, knows where agents keep their skills, copies the shipped directories there, and removes them again. `pyphi/mcp/install.py` calls into it from `run()`, so `install()` and `uninstall()` keep their present signatures and the MCP registration path is untouched. The `pyphi` skill's `references/` is filled at install time from `pyphi.mcp.content`, so the reference documents have one source rather than two copies.

**Tech Stack:** Python 3.13+, `argparse`, `shutil`, `importlib.resources`, `importlib.metadata`, pytest. No new dependencies.

## Global Constraints

- Python 3.13+ only. No backward-compatibility shims.
- No new runtime dependencies.
- Docstrings follow `pyphi/CLAUDE.md`: NumPy style, underlined sections, final-state impersonal voice, no process narrative, Unicode symbols rather than RST substitutions.
- Docstrings in `pyphi/**` are executed by `--doctest-modules`. Any `>>>` line must be correct, or must not be written.
- `pyphi/mcp/install.py` must not import the optional `mcp` dependency. `pyphi/mcp/agents.py` inherits this.
- Skills are always written under `$HOME`, whatever `--scope` says. `--scope` governs only the MCP registration.
- The sentinel filename is exactly `.pyphi-skill`.
- The three agent probe directories are exactly `~/.claude`, `~/.codex`, `~/.cursor`; their skill directories are those paths plus `/skills`.
- Verification runs `uv run pytest` with **no path argument** at least once, because bare paths override `testpaths` and skip the doctest sweep.
- Never pipe a test run through `tail`/`head`/`grep`. Redirect to a file and read the summary line.
- Changelog fragments go in `changelog.d/<name>.<type>.md`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `pyphi/mcp/content/reproducible-work.md` | New reference topic: seeding, provenance writers, no-clobber, raw data. |
| `pyphi/mcp/content.py` | Modified: one new `TOPICS` entry. |
| `pyphi/mcp/agents.py` | New: the agent table, detection, and skill delivery and removal. |
| `pyphi/mcp/skills/iit/SKILL.md` | New: the gate skill. |
| `pyphi/mcp/skills/pyphi/SKILL.md` | New: the library skill. |
| `pyphi/mcp/install.py` | Modified: CLI flags and two calls in `run()`. |
| `test/mcp/test_agents.py` | New: detection, delivery, removal, the flow. |
| `test/mcp/test_install.py` | Modified: CLI flag tests. |
| `test/mcp/test_content.py` | Modified or created: the new topic loads. |
| `docs/howto/mcp-server.md` | Modified: a section on what the skills step writes. |
| `changelog.d/mcp-install-skills.feature.md` | New. |

`pyphi/mcp/skills/` is package data, not a package: it needs no `__init__.py`. A wheel built from this tree already carries every non-Python file under `pyphi/`, verified against `pyphi/mcp/content/*.md`, so no `pyproject.toml` change is required. Task 4 adds a test that guards that.

The module is named `agents.py` rather than `skills.py` deliberately: a module `pyphi/mcp/skills.py` next to a data directory `pyphi/mcp/skills/` makes the import resolution depend on a subtlety of the path finder. Avoid the collision.

---

## Task 1: The reproducible-work reference topic

**Files:**
- Create: `pyphi/mcp/content/reproducible-work.md`
- Modify: `pyphi/mcp/content.py` (the `TOPICS` dict)
- Test: `test/mcp/test_content.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the topic key `"reproducible-work"` in `pyphi.mcp.content.TOPICS`, loadable through `content.load("reproducible-work")`. Task 3 writes every topic into the `pyphi` skill's `references/`; Task 4's `SKILL.md` names `references/reproducible-work.md`.

- [ ] **Step 1: Write the failing test**

Append to `test/mcp/test_content.py` (create the file with the import header if it does not exist):

```python
from pyphi.mcp import content


def test_reproducible_work_is_a_topic():
    assert "reproducible-work" in content.topics()


def test_reproducible_work_loads():
    text = content.load("reproducible-work")
    assert text.startswith("# ")
    assert "save_json" in text
    assert "default_rng" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/mcp/test_content.py -v`
Expected: FAIL — `assert 'reproducible-work' in {...}`

- [ ] **Step 3: Write the reference document**

Create `pyphi/mcp/content/reproducible-work.md`:

```markdown
# Reproducible work

The MCP server holds results in memory only, and none of its tools writes to
disk. Anything that has to survive the session, be rerun, or be cited belongs
in a script. These are the conventions that make such a script reproducible.

## Seeding

Any computation that draws randomness takes a `seed` argument and uses an
isolated generator built from it:

```python
rng = np.random.default_rng(seed)
```

Never call `np.random.seed()` or `random.seed()`. Those mutate global state,
which makes the function non-reentrant and silently couples callers that were
meant to be independent.

Save the seed alongside the output, not just to the log. A seed that exists
only in a terminal scrollback has not been recorded.

## Saving results

`pyphi.provenance` has three writers:

```python
from pyphi.provenance import save_json, save_npz, save_dataframe
```

Each one puts the run's parameters in the filename, refuses to overwrite an
existing file, and embeds a provenance record — the PyPhi version, the
formalism, the parameters, and the time — inside the file it writes.

`pyphi.provenance.read_metadata(path)` returns that record without loading the
payload, so a directory of results can be inventoried cheaply.

`pyphi.save` and `pyphi.load` serialize PyPhi objects themselves — an
`Analysis`, a `CauseEffectStructure` — where you want the object back rather
than a summary.

## Never overwrite

Encode the parameters that distinguish a run in its filename: the seed, the
number of trials, the substrate, the formalism. Where a file of that name
already exists, write `_v2`, `_v3`, and so on. Replacing a result is the
user's decision, made by deleting the old file first, and not something a
script should do on its own.

## Save the raw values, not only the summary

Where a script computes a summary — a mean, a correlation, a rate — the
per-trial or per-element values behind it go to disk too, in the same NPZ or
JSON. Without them a reviewer cannot recompute the summary a different way,
and re-running the experiment is the only way to answer a follow-up question.

If a script computes a correlation between two quantities, the paired
observations are part of the output.

## Pin the formalism

A φ value means nothing without the formalism that produced it. Pin it
explicitly in the script rather than inheriting the ambient default, and record
which one in the output:

```python
with pyphi.config.override(**pyphi.iit4_2026):
    analysis = pyphi.analyze(substrate, state)
```

## Long runs

`pyphi.cost.estimate_analysis` is free; call it before committing to a run.
`get_iit_reference("performance")` covers the disk cache and checkpointing, and
`get_iit_reference("campaigns")` covers distributing a sweep across a cluster.
```

- [ ] **Step 4: Register the topic**

In `pyphi/mcp/content.py`, add to the `TOPICS` dict after the `"visualization"` entry:

```python
    "reproducible-work": (
        "reproducible-work.md",
        "Writing a PyPhi script whose results can be rerun and cited: "
        "seeding, the provenance writers, filename and no-clobber "
        "conventions, and saving raw values alongside summaries.",
    ),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/mcp/test_content.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyphi/mcp/content/reproducible-work.md pyphi/mcp/content.py test/mcp/test_content.py
git commit -m "Add a reproducible-work reference topic

Seeding with isolated generators, the provenance writers, the no-clobber
filename convention, and saving per-trial values alongside summaries. The
server had no topic covering the work it cannot do itself."
```

---

## Task 2: Agent detection

**Files:**
- Create: `pyphi/mcp/agents.py`
- Test: `test/mcp/test_agents.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AGENTS: dict[str, tuple[str, str]]` — agent name to (probe directory relative to home, display name).
  - `Target` — a frozen dataclass with fields `name: str`, `display: str`, `path: Path`.
  - `detect(home: Path | None = None) -> list[Target]`
  - `chosen(names: list[str], paths: list[Path], home: Path | None = None) -> list[Target]`
  - `resolve(names, paths, home=None) -> list[Target]` — `chosen` where either argument is non-empty, `detect` otherwise.

- [ ] **Step 1: Write the failing test**

Create `test/mcp/test_agents.py`:

```python
"""Agent detection and skill delivery for ``pyphi-mcp install``."""

from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/mcp/test_agents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.mcp.agents'`

- [ ] **Step 3: Write the module**

Create `pyphi/mcp/agents.py`:

```python
"""Delivery of the PyPhi agent skills to the coding agents installed on a
machine.

A skill is matched against the task before the model acts, which is what the
Model Context Protocol registration and the project instruction block cannot
do: both are consulted only once something has already decided to consult
them. The skills shipped under ``pyphi/mcp/skills/`` are copied from the wheel
into each agent's own skills directory.

Nothing here imports the optional ``mcp`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Agent name mapped to the directory probed under the home directory and the
#: name shown to the user. An agent keeps its skills in ``<probe>/skills``.
AGENTS: dict[str, tuple[str, str]] = {
    "claude-code": (".claude", "Claude Code"),
    "codex": (".codex", "Codex"),
    "cursor": (".cursor", "Cursor"),
}


@dataclass(frozen=True)
class Target:
    """A skills directory to write to.

    Attributes
    ----------
    name : str
        The agent's name, or the directory itself where one was given
        explicitly.
    display : str
        The name shown to the user.
    path : Path
        The directory the skills are written into.
    """

    name: str
    display: str
    path: Path


def detect(home: Path | None = None) -> list[Target]:
    """Return a target for every agent whose probe directory exists.

    Parameters
    ----------
    home : Path, optional
        The directory to probe under. If None, the user's home directory.

    Returns
    -------
    list of Target
        In the order of :data:`AGENTS`, so a report reads the same way twice.
    """
    root = Path.home() if home is None else Path(home)
    return [
        Target(name, display, root / probe / "skills")
        for name, (probe, display) in AGENTS.items()
        if (root / probe).is_dir()
    ]


def chosen(
    names: list[str], paths: list[Path], home: Path | None = None
) -> list[Target]:
    """Return targets named explicitly rather than found by probing.

    Parameters
    ----------
    names : list of str
        Keys of :data:`AGENTS`, used whether or not the agent was detected.
    paths : list of Path
        Skills directories belonging to agents not in :data:`AGENTS`.
    home : Path, optional
        The directory ``names`` are resolved under. If None, the user's home
        directory.

    Raises
    ------
    ValueError
        If a name is not a key of :data:`AGENTS`.
    """
    root = Path.home() if home is None else Path(home)
    targets = []
    for name in names:
        try:
            probe, display = AGENTS[name]
        except KeyError:
            known = ", ".join(AGENTS)
            raise ValueError(f"unknown agent {name!r}; known agents: {known}") from None
        targets.append(Target(name, display, root / probe / "skills"))
    targets.extend(Target(str(path), str(path), Path(path)) for path in paths)
    return targets


def resolve(
    names: list[str], paths: list[Path], home: Path | None = None
) -> list[Target]:
    """Return the explicit targets where any were given, the detected ones
    otherwise."""
    if names or paths:
        return chosen(names, paths, home=home)
    return detect(home=home)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/mcp/test_agents.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/agents.py test/mcp/test_agents.py
git commit -m "Detect the coding agents installed on a machine

A table of three probe directories under the home directory, with explicit
targets for agents that are not detected or not in the table."
```

---

## Task 3: Skill delivery and removal

**Files:**
- Modify: `pyphi/mcp/agents.py`
- Test: `test/mcp/test_agents.py`

**Interfaces:**
- Consumes: `Target`, `resolve` from Task 2; `pyphi.mcp.content.topics` and `content.load` from Task 1.
- Produces:
  - `SENTINEL: str` — `".pyphi-skill"`.
  - `REFERENCED: frozenset[str]` — skill names whose `references/` is filled.
  - `skill_names() -> list[str]`
  - `deliver(target: Target) -> None` — writes every skill into one target; raises `OSError` on failure.
  - `remove(target: Target) -> list[str]` — removes managed skills from one target, returning the names removed.

- [ ] **Step 1: Write the failing test**

Append to `test/mcp/test_agents.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/mcp/test_agents.py -v -k "Delivery or Removal"`
Expected: FAIL with `AttributeError: module 'pyphi.mcp.agents' has no attribute 'skill_names'`

- [ ] **Step 3: Extend the module**

Add the imports at the top of `pyphi/mcp/agents.py`:

```python
import shutil
from importlib import metadata
from importlib import resources
from importlib.resources.abc import Traversable
```

and after the `AGENTS` table:

```python
#: Written inside every installed skill directory, holding the PyPhi version
#: that wrote it. Removal touches only directories containing this file, so a
#: hand-written skill that shares a name is left alone.
SENTINEL = ".pyphi-skill"

#: Skills whose ``references/`` is filled from the reference topics at install
#: time. The gate skill carries none: its purpose is to send the reader to the
#: reference rather than to ship a copy of it.
REFERENCED: frozenset[str] = frozenset({"pyphi"})
```

and these functions at the end:

```python
def _source() -> Traversable:
    return resources.files(__package__) / "skills"


def skill_names() -> list[str]:
    """Return the names of the skills shipped with this PyPhi."""
    return sorted(entry.name for entry in _source().iterdir() if entry.is_dir())


def _version() -> str:
    try:
        return metadata.version("pyphi")
    except metadata.PackageNotFoundError:
        return "unknown"


def deliver(target: Target) -> None:
    """Write every shipped skill into ``target``, replacing earlier copies.

    The ``references/`` directory of each skill in :data:`REFERENCED` is filled
    from :mod:`pyphi.mcp.content`, so the reference documents have one source
    rather than a copy per surface.

    Raises
    ------
    OSError
        If the target directory cannot be written.
    """
    # Imported here so that detection costs nothing on a machine with no
    # agents installed: loading the reference documents imports pyphi itself.
    from pyphi.mcp import content

    for name in skill_names():
        destination = target.path / name
        with resources.as_file(_source() / name) as source:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        (destination / SENTINEL).write_text(_version() + "\n", encoding="utf-8")
        if name in REFERENCED:
            references = destination / "references"
            references.mkdir(exist_ok=True)
            for topic in content.topics():
                (references / f"{topic}.md").write_text(
                    content.load(topic), encoding="utf-8"
                )


def remove(target: Target) -> list[str]:
    """Delete the skills written by :func:`deliver` from ``target``.

    Only directories holding a :data:`SENTINEL` file are removed, so a
    hand-written skill that shares a name survives.

    Returns
    -------
    list of str
        The names removed.
    """
    if not target.path.is_dir():
        return []
    removed = []
    for name in skill_names():
        destination = target.path / name
        if (destination / SENTINEL).is_file():
            shutil.rmtree(destination)
            removed.append(name)
    return removed
```

- [ ] **Step 4: Run tests to verify they pass**

The delivery tests need Task 4's skill files to exist. Create the two directories with placeholder bodies now so this task is testable on its own; Task 4 replaces the bodies:

```bash
mkdir -p pyphi/mcp/skills/iit pyphi/mcp/skills/pyphi
printf -- '---\nname: iit\ndescription: placeholder\n---\n' > pyphi/mcp/skills/iit/SKILL.md
printf -- '---\nname: pyphi\ndescription: placeholder\n---\n' > pyphi/mcp/skills/pyphi/SKILL.md
```

Run: `uv run pytest test/mcp/test_agents.py -v`
Expected: PASS, 21 tests

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/agents.py pyphi/mcp/skills test/mcp/test_agents.py
git commit -m "Copy the shipped skills into an agent's skills directory

Each installed directory carries a sentinel holding the PyPhi version that
wrote it, so removal never deletes a hand-written skill that shares a name.
The library skill's references are filled from pyphi.mcp.content, which keeps
one source for the reference documents."
```

---

## Task 4: The two skills

**Files:**
- Modify: `pyphi/mcp/skills/iit/SKILL.md`
- Modify: `pyphi/mcp/skills/pyphi/SKILL.md`
- Test: `test/mcp/test_agents.py`

**Interfaces:**
- Consumes: `skill_names` from Task 3.
- Produces: the two shipped skill bodies. No Python interface.

- [ ] **Step 1: Write the failing test**

Append to `test/mcp/test_agents.py`:

```python
class TestShippedSkills:
    def _front_matter(self, name):
        from importlib import resources

        text = (
            resources.files("pyphi.mcp") / "skills" / name / "SKILL.md"
        ).read_text(encoding="utf-8")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/mcp/test_agents.py -v -k ShippedSkills`
Expected: FAIL — the placeholder front matter has `description: placeholder` and no body

- [ ] **Step 3: Write the gate skill**

Replace `pyphi/mcp/skills/iit/SKILL.md`:

```markdown
---
name: iit
description: >-
  Use before explaining, discussing, evaluating, or implementing anything
  involving Integrated Information Theory — its axioms, postulates, φ, Φ,
  distinctions, relations, complexes, or whether some system is conscious on
  IIT's account — and before interpreting any number PyPhi produced. Read the
  sources named here before answering. Most published summaries of IIT
  misstate it, so answering from recollection reproduces those errors.
---

# IIT: read before answering

Your recollection of IIT was built out of secondary sources, and most published
summaries of the theory misstate it — often they invert it. You will not notice
this from the inside, because a wrong account of IIT is as fluent as a right
one. Read a source before answering, including for questions that seem too
small to need one.

## Where the theory is

In order of preference:

1. **`mcp.learniit.org`**, where the connector is available. Every claim there
   carries a locator — paper, section, equation — so an answer can be checked.
   Where it is not available and the person is working on IIT seriously, tell
   them they can add it under Connectors in their client's settings; it needs
   no account.
2. **`get_iit_reference("theory")`, `("equations")` and `("gotchas")`**, where
   the PyPhi MCP server is connected. Calling `get_iit_reference` with no
   argument lists every topic.
3. **`python -c "from pyphi.mcp import content; print(content.load('theory'))"`**
   otherwise. This needs only PyPhi installed — no server, no network — and
   offers the same topics.
4. **The papers**, in a PyPhi checkout under `papers/`, named by year and
   author.

## Which formulation is current

- **IIT 4.0** (Albantakis et al. 2023) is the canonical mathematical statement.
- **Tononi & Boly 2025**, *Integrated Information Theory: A Consciousness-First
  Approach to What Exists*, is the current non-mathematical account. Prefer it
  for framing, motivation, and the ontology.
- **IIT 3.0** (Oizumi, Albantakis & Tononi 2014) is superseded. Cite it for
  history, or to say what changed.

Say which formulation you are answering from wherever it could matter. The
theory has moved before and will move again.

## Expound, do not survey

State what IIT holds: "IIT holds that…", "the exclusion postulate requires…".
This is exposition of a body of theory, the way a textbook explains general
relativity — not a report on opinion. Do not balance IIT against other theories
unless asked to compare, and do not hedge with what critics say. Whether IIT is
right is a separate question from what it claims, and only the second is being
asked here.

Answer a standard objection the way IIT answers it. Those replies are in the
sources.

Never invent. Where the sources do not settle something, say that IIT has not
addressed it, or that you are extrapolating, and say which.

## Errors that recur

- **φ, φₛ and Φ are three quantities.** φ is a distinction's integrated
  information, φₛ the system's, Φ the Φ-structure's.
- **Integrated information is not Shannon information about the system.**
- **Behaviour does not settle Φ.** Two systems can behave identically and
  differ in Φ.
- **IIT is not a functionalist or computational account.** It is a claim about
  physical cause-effect power.

## Where a number would settle it

Compute it rather than asserting it — see the `pyphi` skill. Keep the system
small; the computation is superexponential in the number of units.
```

- [ ] **Step 4: Write the library skill**

Replace `pyphi/mcp/skills/pyphi/SKILL.md`:

````markdown
---
name: pyphi
description: >-
  Use when writing, reading, or running any code that imports pyphi — building
  a Substrate or a TPM, calling analyze(), choosing a formalism, estimating
  cost before an expensive run, or saving results reproducibly. Covers the
  PyPhi 2.0 API and the MCP server's tools. PyPhi 2.0 renamed most of the
  pre-2.0 surface with no aliases, so code written from memory of older
  versions will not run.
---

# PyPhi

PyPhi computes Integrated Information Theory quantities. For what the theory
says, use the `iit` skill; this one is about the software.

## Use the server for exploration

Where the PyPhi MCP server is connected, drive it through its tools rather than
writing a script. They report which formalism produced each number, refuse runs
too large to finish, and keep φₛ and Φ apart. Where it is not connected,
`pyphi-mcp install` registers it.

The server holds results in memory only, and none of its tools writes to disk.
Anything that has to be reproducible belongs in a script — see Reproducible
work below.

## The 2.0 API

PyPhi 2.0 renamed most of the pre-2.0 surface and shipped no aliases, so code
written from memory of PyPhi 1.x raises `ImportError`. The full table is
`references/migration.md`. The essentials:

```python
import pyphi

substrate = pyphi.Substrate(tpm, cm=cm, node_labels=labels)
analysis = pyphi.analyze(substrate, state)

analysis.phi           # φₛ, system integrated information
analysis.ces           # the Φ-structure
analysis.ces.big_phi   # Φ, structure integrated information
```

`pyphi.Network` is now `pyphi.Substrate`, `pyphi.Subsystem` is now
`pyphi.System`, and the whole `pyphi.compute` module is replaced by
`pyphi.analyze`.

**One rename is silent.** The old `CauseEffectStructure` is now
`pyphi.models.Distinctions`, and the old `PhiStructure` is now
`pyphi.models.CauseEffectStructure`. The same words point at different objects,
so unported code can import successfully and mean something else. Check every
occurrence of either name.

Compare φ values with `pyphi.numerics.eq`, not `==`.

## A φ value means nothing without its formalism

φ is defined relative to a formalism, and PyPhi ships three presets:
`pyphi.iit3`, `pyphi.iit4_2023`, `pyphi.iit4_2026`. Pin one rather than
relying on the ambient default, and say which one whenever you report a number.

```python
with pyphi.config.override(**pyphi.iit4_2026):
    analysis = pyphi.analyze(substrate, state)
```

## States are little-endian

The first node is the least significant bit, so `(1, 0, 0)` is the first node
on and the rest off. Reversing this produces a well-formed, wrong answer.

## Estimate before running

Analyses are superexponential in substrate size. `pyphi.cost.estimate_analysis`
is free — call it before any run over more than a handful of units.

```python
print(pyphi.cost.estimate_analysis(substrate))
```

## Building a substrate

`references/building-systems.md` has the procedure. The trap worth stating
here: a transition probability matrix built with its axes in the wrong order is
still well-formed, and a symmetric test network will not catch it. After
building one, check that a known state's transition comes back the way you
expect before trusting anything computed from it.

## Reproducible work

`references/reproducible-work.md` has the detail. In short: seed a generator
instance with `np.random.default_rng(seed)` rather than seeding the global one;
save with `pyphi.provenance.save_json`, `save_npz` or `save_dataframe`, which
put the parameters in the filename, refuse to overwrite, and embed the
provenance; and save per-trial values alongside any summary computed from them.

## References

`references/` holds one file per reference topic. Read the one the task calls
for rather than all of them.
````

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/mcp/test_agents.py -v > /tmp/agents.log 2>&1; tail -1 /tmp/agents.log`
Expected: PASS (the wheel test is marked slow and does not run here)

Then the slow lane once: `uv run pytest test/mcp/test_agents.py -m slow --slow -v > /tmp/agents-slow.log 2>&1`
Expected: PASS — read the summary line in the file, not the exit code of a pipeline.

- [ ] **Step 6: Commit**

```bash
git add pyphi/mcp/skills test/mcp/test_agents.py
git commit -m "Write the iit and pyphi skills

The gate interrupts before a model answers about IIT from recollection and
routes it to learniit.org, get_iit_reference, or the bundled content. The
library skill covers the 2.0 API, the formalism requirement, endianness, cost,
and the conventions for reproducible scripts."
```

---

## Task 5: Wire the step into the command

**Files:**
- Modify: `pyphi/mcp/agents.py` (the flow)
- Modify: `pyphi/mcp/install.py` (flags, `run()`)
- Test: `test/mcp/test_agents.py`, `test/mcp/test_install.py`

**Interfaces:**
- Consumes: `resolve`, `deliver`, `remove` from Tasks 2 and 3.
- Produces:
  - `interactive() -> bool`
  - `confirm(question: str) -> bool`
  - `install_step(*, skills: bool | None, names: list[str], paths: list[Path], home: Path | None = None) -> list[str]`
  - `remove_step(*, names, paths, home=None) -> list[str]`
  - `describe(*, names, paths, home=None) -> list[str]`
  - New `install` flags: `--skills`, `--no-skills`, `--agent` (repeatable), `--agent-path` (repeatable). `uninstall` gains `--agent` and `--agent-path`.

- [ ] **Step 1: Write the failing test**

Append to `test/mcp/test_agents.py`:

```python
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
        monkeypatch.setattr(mod, "confirm", lambda question: pytest.fail("prompted"))
        actions = mod.install_step(skills=True, names=[], paths=[], home=home)
        assert (home / ".claude" / "skills" / "iit" / "SKILL.md").is_file()
        assert any("Claude Code" in line for line in actions)

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
        monkeypatch.setattr(mod, "confirm", lambda question: True)
        mod.install_step(skills=None, names=[], paths=[], home=home)
        assert (home / ".claude" / "skills" / "pyphi" / "SKILL.md").is_file()

    def test_interactive_no_writes_nothing(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        monkeypatch.setattr(mod, "interactive", lambda: True)
        monkeypatch.setattr(mod, "confirm", lambda question: False)
        assert mod.install_step(skills=None, names=[], paths=[], home=home) == []

    def test_the_prompt_names_every_detected_agent(self, tmp_path, monkeypatch):
        home = tmp_path
        (home / ".claude").mkdir()
        (home / ".codex").mkdir()
        monkeypatch.setattr(mod, "interactive", lambda: True)
        asked = []
        monkeypatch.setattr(mod, "confirm", lambda question: asked.append(question))
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
        monkeypatch.setattr("builtins.input", lambda prompt: answer)
        assert mod.confirm("Install?")

    @pytest.mark.parametrize("answer", ["n", "no", "nope"])
    def test_declining_answers(self, answer, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: answer)
        assert not mod.confirm("Install?")

    def test_end_of_input_declines(self, monkeypatch):
        def raise_eof(prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert not mod.confirm("Install?")


class TestInteractive:
    def test_ci_is_not_interactive(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert not mod.interactive()
```

And append to `test/mcp/test_install.py`, inside `TestCommandLine`:

```python
    def test_skills_defaults_to_asking(self):
        assert mod.build_parser().parse_args(["install"]).skills is None

    def test_no_skills_is_false(self):
        assert mod.build_parser().parse_args(["install", "--no-skills"]).skills is False

    def test_skills_is_true(self):
        assert mod.build_parser().parse_args(["install", "--skills"]).skills is True

    def test_agents_are_repeatable(self):
        args = mod.build_parser().parse_args(
            ["install", "--agent", "codex", "--agent", "cursor"]
        )
        assert args.agent == ["codex", "cursor"]

    def test_uninstall_takes_agent_flags(self):
        args = mod.build_parser().parse_args(["uninstall", "--agent", "codex"])
        assert args.agent == ["codex"]

    def test_install_does_not_write_skills_without_agents(self, tmp_path):
        args = mod.build_parser().parse_args(
            ["install", "--skills", "--directory", str(tmp_path)]
        )
        assert mod.run(args) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/mcp/test_agents.py test/mcp/test_install.py -v -k "Flow or Confirm or Interactive or skills or agent"`
Expected: FAIL with `AttributeError: module 'pyphi.mcp.agents' has no attribute 'install_step'`

- [ ] **Step 3: Add the flow to `agents.py`**

Add `import os` and `import sys` to the imports, then append:

```python
def interactive() -> bool:
    """Whether a question can be put to a person.

    False on a pipe, under a scheduler, and wherever ``CI`` is set, so an
    unattended run never blocks waiting for an answer.
    """
    return sys.stdin.isatty() and sys.stdout.isatty() and not os.environ.get("CI")


def confirm(question: str) -> bool:
    """Ask ``question`` and return whether the answer was yes.

    An empty answer accepts. End of input declines, so a closed pipe does not
    raise.
    """
    try:
        answer = input(f"{question} [Y/n] ").strip().lower()
    except EOFError:
        return False
    return answer in ("", "y", "yes")


def install_step(
    *,
    skills: bool | None,
    names: list[str],
    paths: list[Path],
    home: Path | None = None,
) -> list[str]:
    """Offer the skills, install them where accepted, and report what happened.

    Parameters
    ----------
    skills : bool or None
        True installs without asking, False skips, None asks where a person is
        there to answer and skips otherwise.
    names : list of str
        Agents named explicitly, whether or not they were detected.
    paths : list of Path
        Skills directories given explicitly.
    home : Path, optional
        The directory agents are resolved under. If None, the user's home
        directory.

    Returns
    -------
    list of str
        One line per action taken, empty where nothing was written.
    """
    targets = resolve(names, paths, home=home)
    if not targets or skills is False:
        return []
    if skills is None:
        if not interactive():
            return [
                "skipped the skills; run `pyphi-mcp install --skills` to add them"
            ]
        displayed = ", ".join(target.display for target in targets)
        if not confirm(f"Install the PyPhi skills for {displayed}?"):
            return []
    actions = []
    for target in targets:
        try:
            deliver(target)
        except OSError as error:
            actions.append(f"could not write skills to {target.path}: {error}")
        else:
            installed = ", ".join(skill_names())
            actions.append(f"installed the {installed} skills in {target.path}")
    return actions


def remove_step(
    *, names: list[str], paths: list[Path], home: Path | None = None
) -> list[str]:
    """Delete the installed skills and report what was removed."""
    actions = []
    for target in resolve(names, paths, home=home):
        removed = remove(target)
        if removed:
            actions.append(f"removed the {', '.join(removed)} skills from {target.path}")
    return actions


def describe(
    *, names: list[str], paths: list[Path], home: Path | None = None
) -> list[str]:
    """Return what :func:`install_step` would write, writing nothing."""
    installed = ", ".join(skill_names())
    return [
        f"{target.path}: the {installed} skills"
        for target in resolve(names, paths, home=home)
    ]
```

- [ ] **Step 4: Wire it into `install.py`**

Add the import near the top of `pyphi/mcp/install.py`:

```python
from pyphi.mcp import agents
```

In `_add_common`, add the agent flags so both subcommands take them:

```python
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "install skills for this agent whether or not it was detected "
            f"({', '.join(agents.AGENTS)}); repeatable"
        ),
    )
    parser.add_argument(
        "--agent-path",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="a skills directory to write to; repeatable",
    )
```

In `build_parser`, add to `install_parser` only:

```python
    skills = install_parser.add_mutually_exclusive_group()
    skills.add_argument(
        "--skills",
        dest="skills",
        action="store_true",
        default=None,
        help="install the agent skills without asking",
    )
    skills.add_argument(
        "--no-skills",
        dest="skills",
        action="store_false",
        help="do not install the agent skills",
    )
```

In `run`, extend the `--print` branch:

```python
            if args.print_only:
                config = {"mcpServers": {"pyphi": registration(args.spec)}}
                print(f"{config_path(args.directory, args.scope, args.client)}:")
                print(json.dumps(config, indent=2))
                print(f"\n{args.directory / INSTRUCTIONS_FILE}:")
                print(block())
                for line in agents.describe(
                    names=args.agent, paths=args.agent_path
                ):
                    print(f"\n{line}")
                return 0
```

and after the `install(...)` call succeeds, add the step:

```python
            actions += agents.install_step(
                skills=args.skills, names=args.agent, paths=args.agent_path
            )
```

In the `uninstall` branch:

```python
        actions = uninstall(args.directory, scope=args.scope, client=args.client)
        actions += agents.remove_step(names=args.agent, paths=args.agent_path)
        actions = actions or ["nothing to remove"]
```

`uninstall()` currently returns `["nothing to remove"]` itself, which would make
a run that removed only skills report both that line and the removal. Change its
last line from `return actions or ["nothing to remove"]` to `return actions`, and
let `run` supply the fallback as above.

**That changes an existing test.** `test/mcp/test_install.py::TestUninstall::test_is_safe_to_run_when_nothing_is_installed`
asserts the old return value. Update it to assert on the command instead:

```python
    def test_is_safe_to_run_when_nothing_is_installed(self, tmp_path, capsys):
        assert mod.uninstall(tmp_path) == []
        args = mod.build_parser().parse_args(["uninstall", "--directory", str(tmp_path)])
        assert mod.run(args) == 0
        assert "nothing to remove" in capsys.readouterr().out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/mcp/ -v > /tmp/mcp.log 2>&1; tail -1 /tmp/mcp.log`
Expected: PASS

- [ ] **Step 6: Run the whole suite with no path argument**

Run: `uv run pytest -m "not slow" > /tmp/full.log 2>&1; tail -3 /tmp/full.log`
Expected: PASS. Read the summary line in the file — the exit code of a pipeline is not pytest's.

- [ ] **Step 7: Commit**

```bash
git add pyphi/mcp/agents.py pyphi/mcp/install.py test/mcp/
git commit -m "Offer the agent skills from pyphi-mcp install

The step runs after the registration and the instruction block, so an
interrupted prompt still leaves a working install. --skills and --no-skills
answer it without a terminal, and --agent and --agent-path reach agents that
were not detected. Skills are always written under the home directory, whatever
--scope says."
```

---

## Task 6: Documentation and changelog

**Files:**
- Modify: `docs/howto/mcp-server.md`
- Create: `changelog.d/mcp-install-skills.feature.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: the finished command from Task 5.
- Produces: nothing in code.

- [ ] **Step 1: Add the documentation section**

In `docs/howto/mcp-server.md`, after the "What `install` writes" section, add:

```markdown
### Skills for your coding agent

`install` also offers to write two skills into every AI coding agent it finds:
`iit`, which stops an assistant answering about the theory from recollection
and points it at the reference, and `pyphi`, which covers the 2.0 API, the
formalism requirement, state ordering, cost estimation, and the conventions for
a reproducible script.

It probes `~/.claude`, `~/.codex` and `~/.cursor`, and asks before writing:

```
Install the PyPhi skills for Claude Code, Codex? [Y/n]
```

`--skills` and `--no-skills` answer it without a terminal, which is what a
script or a continuous-integration run needs; with neither flag and no
terminal, nothing is written. `--agent NAME` reaches an agent that was not
detected, and `--agent-path DIR` reaches one PyPhi does not know about. Both
are repeatable.

Skills are written under your home directory whatever `--scope` says, because
they are about IIT and PyPhi rather than about one project. The report prints
the full path of each one. `pyphi-mcp uninstall` removes them again; it deletes
only directories PyPhi wrote, so a skill of your own that shares a name is left
alone.

Running `install` again refreshes the skills, which is how you update them
after upgrading PyPhi.
```

- [ ] **Step 2: Verify the docs build**

Run: `just docs 2>&1 | tail -20`
Expected: no warnings about `docs/howto/mcp-server.md`

- [ ] **Step 3: Write the changelog fragment**

```bash
cat > changelog.d/mcp-install-skills.feature.md <<'EOF'
`pyphi-mcp install` offers to install two agent skills, `iit` and `pyphi`, into
Claude Code, Codex and Cursor. `--skills` and `--no-skills` answer the prompt
without a terminal; `--agent` and `--agent-path` reach agents that were not
detected. `pyphi-mcp uninstall` removes them.
EOF
```

- [ ] **Step 4: Update the roadmap**

`ROADMAP.md` already has an `| MCP server | ✅ landed | — | …` row that records
each addition as a bold dated sentence appended to the same cell. Do not add a
new row. Append to the end of that cell, before the closing `|`:

```
**Agent skills (2026-08-09):** `pyphi-mcp install` offers to install two skills into every coding agent it detects (`~/.claude`, `~/.codex`, `~/.cursor`, plus `--agent` and `--agent-path` for the rest) — `iit`, which stops an assistant answering about the theory from recollection and routes it to `mcp.learniit.org`, `get_iit_reference`, or the bundled content, and `pyphi`, which covers the 2.0 API, the formalism requirement, state ordering, cost estimation, and reproducible scripts. The skills ship inside the wheel under `pyphi/mcp/skills/`, so they never drift from the installed version; the library skill's `references/` is filled from `pyphi/mcp/content/`, making skills a fourth surface over the same reference documents rather than a copy. `--skills`/`--no-skills` answer the prompt without a terminal, installs are stamped with a `.pyphi-skill` sentinel so `uninstall` never deletes a hand-written skill of the same name, and a new `reproducible-work` reference topic covers seeding, the provenance writers, and the no-clobber convention. Spec/plan: `docs/superpowers/{specs,plans}/2026-08-09-mcp-install-skills*`.
```

- [ ] **Step 5: Run the full suite once with no path argument**

Run: `uv run pytest > /tmp/final.log 2>&1; tail -5 /tmp/final.log`
Expected: PASS, including the doctest sweep over `pyphi/`.

- [ ] **Step 6: Commit**

```bash
git add docs/howto/mcp-server.md changelog.d/mcp-install-skills.feature.md ROADMAP.md
git commit -m "Document the skills step

What install writes, which agents it probes, how to answer without a terminal,
and why the skills are written outside the project directory."
```

---

## Self-review notes

Checked against the spec:

- Detection table, the two escape-hatch flags, and always-under-home: Tasks 2 and 5.
- Prompt, the two flags, the CI and non-terminal skip, per-agent failure tolerance: Task 5.
- Copy rather than symlink, sentinel with version, refresh on re-install: Task 3.
- `references/` from `pyphi/mcp/content/`, and the new reproducible-work topic: Tasks 1 and 3.
- Both skill bodies with the exact front-matter descriptions from the spec: Task 4.
- Wheel packaging guard: Task 4.
- `--print` writing nothing: Task 5.
- Every spec test bullet has a test in Tasks 2 through 5.

One deviation from the spec, recorded deliberately: the spec described the
skills step as part of `install()`. The plan puts the flow in `agents.py` and
calls it from `run()`, leaving `install()` and `uninstall()` with their present
signatures. This keeps the orchestration testable without threading five new
parameters through a function whose only caller is the command line.

One cost worth knowing: filling `references/` calls `content.load`, and the
`configuration` topic's loader imports `pyphi.conf`, which imports PyPhi. The
skills step therefore pays a full PyPhi import. The import is deferred into
`deliver`, so detection and a declined prompt cost nothing. Writing the raw
file instead would drop the generated option reference from that topic and
leave a silently incomplete document, which is worse.
```
