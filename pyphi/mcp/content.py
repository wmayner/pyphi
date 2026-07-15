"""Access to the bundled IIT reference documents.

The Markdown files under ``pyphi/mcp/content/`` are the single source of the
server's teaching material. They are surfaced three ways — as the server's
always-loaded primer, as ``pyphi://theory/*`` resources, and through the
``get_iit_reference`` tool — so an agent understands the theory before working
with it regardless of which Model Context Protocol features its client
supports.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import MISSING
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
    "configuration": (
        "configuration.md",
        "The layered config object, override, presets, pyphi_config.yml, and "
        "the options worth knowing.",
    ),
    "performance": (
        "performance.md",
        "Running expensive analyses: the cost ceiling, the in-memory and "
        "opt-in disk caches, and checkpointing so a long run survives a crash.",
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

    The ``configuration`` topic has the complete option reference appended,
    generated live from the config dataclasses by :func:`config_reference`, so
    it never drifts from the code.

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
    text = (resources.files(__package__) / "content" / filename).read_text(
        encoding="utf-8"
    )
    if topic == "configuration":
        text += "\n" + config_reference()
    return text


def config_reference() -> str:
    """Return a Markdown reference of every ``pyphi.config`` option.

    Generated from the config dataclasses, so the option names, defaults, and
    layer structure always match the installed code. Each namespace's class
    docstring supplies the prose descriptions; the per-option list gives every
    field with its default value.

    Returns
    -------
    str
        Markdown listing all options under ``numerics``, ``infrastructure``,
        ``formalism.iit``, and ``formalism.actual_causation``.
    """
    # Imported here to avoid a circular import at module load and to keep the
    # config the single source of truth for the option set.
    from pyphi.conf.formalism import ActualCausationConfig
    from pyphi.conf.formalism import IITConfig
    from pyphi.conf.infrastructure import InfrastructureConfig
    from pyphi.conf.numerics import NumericsConfig

    namespaces = [
        ("numerics", NumericsConfig),
        ("infrastructure", InfrastructureConfig),
        ("formalism.iit", IITConfig),
        ("formalism.actual_causation", ActualCausationConfig),
    ]
    lines = ["## Complete option reference", ""]
    for path, cls in namespaces:
        lines.append(f"### `pyphi.config.{path}`")
        lines.append("")
        doc = inspect.getdoc(cls)
        if doc:
            lines.append(doc)
            lines.append("")
        for field in dataclasses.fields(cls):
            if field.default is not MISSING:
                default = field.default
            elif field.default_factory is not MISSING:
                default = field.default_factory()
            else:
                default = None
            lines.append(f"- `{field.name}` (default `{default!r}`)")
        lines.append("")
    return "\n".join(lines)
