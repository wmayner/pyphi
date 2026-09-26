"""Delivery of the PyPhi agent skills to the coding agents installed on a
machine.

A skill is matched against the task before the model acts. The skills shipped under
``pyphi/mcp/skills/`` are copied out of the package into each agent's own skills
directory. The IIT Expert plugin, which covers the theory, is installed through
each agent's own plugin commands.

Nothing here imports the optional ``mcp`` dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from pyphi.mcp import content

#: Agent name mapped to the directory probed under the home directory and the
#: name shown to the user. An agent keeps its skills in ``<probe>/skills``.
AGENTS: dict[str, tuple[str, str]] = {
    "claude-code": (".claude", "Claude Code"),
    "codex": (".codex", "Codex"),
    "cursor": (".cursor", "Cursor"),
}

#: Written inside every installed skill directory, holding the PyPhi version
#: that wrote it. Removal touches only directories containing this file, so a
#: hand-written skill that shares a name is left alone.
SENTINEL = ".pyphi-skill"

#: Skills whose ``references/`` is filled from the reference topics at install
#: time.
REFERENCED: frozenset[str] = frozenset({"pyphi"})

#: Skills earlier versions of PyPhi installed and this one no longer ships.
#: Installing and uninstalling both delete a copy PyPhi wrote.
RETIRED: frozenset[str] = frozenset({"iit"})

#: Agents whose skills directory Cursor also reads, so writing to Cursor's own
#: directory as well would load every skill twice there.
CURSOR_READS: frozenset[str] = frozenset({"claude-code", "codex"})

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
    f"in Cursor, open Customize → From GitHub Repository and enter {PLUGIN_REPOSITORY}"
)


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

    Returns
    -------
    list of Target
        The named agents first, then the explicit directories.

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


def _remove_marked(path: Path, names: Iterable[str]) -> list[str]:
    """Delete each named skill under ``path`` that holds the sentinel file."""
    removed = []
    for name in names:
        destination = path / name
        if (destination / SENTINEL).is_file():
            shutil.rmtree(destination)
            removed.append(name)
    return removed


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
    _remove_marked(target.path, sorted(RETIRED))
    for name in skill_names():
        destination = target.path / name
        with resources.as_file(_source() / name) as source:
            shutil.copytree(source, destination, dirs_exist_ok=True)
        (destination / SENTINEL).write_text(_version() + "\n", encoding="utf-8")
        if name in REFERENCED:
            references = destination / "references"
            # Rebuilt rather than merged, so a topic that was renamed or
            # dropped does not leave its old document behind on re-install.
            shutil.rmtree(references, ignore_errors=True)
            references.mkdir()
            for topic in content.topics():
                (references / f"{topic}.md").write_text(
                    content.load(topic), encoding="utf-8"
                )


def remove(target: Target) -> list[str]:
    """Delete the skills written by :func:`deliver` from ``target``, and any
    :data:`RETIRED` skill an earlier PyPhi wrote there.

    Only directories holding a :data:`SENTINEL` file are removed, so a
    hand-written skill that shares a name survives.

    Returns
    -------
    list of str
        The names removed.
    """
    if not target.path.is_dir():
        return []
    return _remove_marked(target.path, [*skill_names(), *sorted(RETIRED)])


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
    targets, covered = _split_cursor(
        resolve(names, paths, home=home), explicit=bool(names or paths)
    )
    if not targets or skills is False:
        return []
    if skills is None:
        if not interactive():
            return ["skipped the skills; run `pyphi-mcp install --skills` to add them"]
        displayed = ", ".join(target.display for target in targets)
        if not confirm(f"Install the PyPhi skills for {displayed}?"):
            return []
    installed = ", ".join(skill_names())
    actions = []
    for target in targets:
        try:
            deliver(target)
        except OSError as error:
            actions.append(f"could not write skills to {target.path}: {error}")
        else:
            actions.append(f"installed the {installed} skills in {target.path}")
    if covered is not None:
        remove(covered)
        actions.append(
            f"left {covered.display} out: it reads the skills written for "
            "the other agents"
        )
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
        for target in _split_cursor(
            resolve(names, paths, home=home), explicit=bool(names or paths)
        )[0]
    ]


def _agents(names: list[str], paths: list[Path], home: Path | None) -> list[Target]:
    """The known agents among the targets; an explicit directory is not one."""
    return [
        target for target in resolve(names, paths, home=home) if target.name in AGENTS
    ]


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
    lines = []
    for target in _agents(names, paths, home):
        commands = PLUGIN_COMMANDS.get(target.name)
        how = _shown(commands) if commands is not None else CURSOR_STEPS
        lines.append(f"IIT Expert plugin for {target.display}: {how}")
    return lines


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
