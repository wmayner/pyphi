"""Delivery of the PyPhi agent skills to the coding agents installed on a
machine.

A skill is matched against the task before the model acts, which is what the
Model Context Protocol registration and the project instruction block cannot
do: both are consulted only once something has decided to consult them. The
skills shipped under ``pyphi/mcp/skills/`` are copied out of the package into
each agent's own skills directory.

Nothing here imports the optional ``mcp`` dependency.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import metadata
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

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
#: time. The gate skill carries none: it sends the reader to the reference
#: rather than shipping a copy of it.
REFERENCED: frozenset[str] = frozenset({"pyphi"})


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
    # Imported here so that detection and a declined prompt cost nothing:
    # loading the reference documents imports PyPhi itself.
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
