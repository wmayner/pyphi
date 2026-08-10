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
