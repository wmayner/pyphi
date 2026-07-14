"""Access to the bundled IIT reference documents.

The Markdown files under ``pyphi/mcp/content/`` are the single source of the
server's teaching material. They are surfaced three ways — as the server's
always-loaded primer, as ``pyphi://theory/*`` resources, and through the
``get_iit_reference`` tool — so an agent understands the theory before working
with it regardless of which Model Context Protocol features its client
supports.
"""

from __future__ import annotations

from importlib import resources

#: Reference topics mapped to their Markdown file and a one-line description.
TOPICS: dict[str, tuple[str, str]] = {
    "primer": (
        "primer.md",
        "Orientation: core IIT vocabulary and how to drive the server's tools.",
    ),
    "theory": (
        "theory.md",
        "The postulates, the φₛ-versus-Φ distinction, distinctions and "
        "relations, and the formalism versions.",
    ),
    "equations": (
        "equations.md",
        "The key IIT equations with citations verified against the papers.",
    ),
    "gotchas": (
        "gotchas.md",
        "Subtleties that trip up newcomers: state ordering, φ=0, version "
        "differences, k-ary units, and more.",
    ),
    "interpreting-results": (
        "interpreting-iit-results.md",
        "How to read and narrate an analysis result in plain language.",
    ),
    "building-systems": (
        "building-iit-systems.md",
        "How to turn a description of some units into a valid transition "
        "probability matrix.",
    ),
    "migration": (
        "migration.md",
        "Migrating pre-2.0 PyPhi code to 2.0: the renames, config, and "
        "default-formalism changes, with before/after snippets.",
    ),
}


def topics() -> dict[str, str]:
    """Return the available reference topics and their descriptions.

    Returns
    -------
    dict[str, str]
        A mapping from topic name to a one-line description.
    """
    return {name: description for name, (_, description) in TOPICS.items()}


def load(topic: str) -> str:
    """Return the Markdown text of a reference topic.

    Parameters
    ----------
    topic : str
        One of the keys of :data:`TOPICS`.

    Returns
    -------
    str
        The document's Markdown text.

    Raises
    ------
    KeyError
        If ``topic`` is not a known topic.
    """
    try:
        filename, _ = TOPICS[topic]
    except KeyError:
        available = ", ".join(sorted(TOPICS))
        raise KeyError(
            f"Unknown reference topic {topic!r}. Available topics: {available}."
        ) from None
    return (resources.files(__package__) / "content" / filename).read_text(
        encoding="utf-8"
    )
