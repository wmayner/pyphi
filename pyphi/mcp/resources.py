"""IIT reference material, exposed as resources and a tool.

The same Markdown documents are surfaced two ways so an agent can reach them
regardless of client support: as ``pyphi://theory/<topic>`` resources (the
canonical Model Context Protocol mechanism for background documents) and
through the ``get_iit_reference`` tool (callable in every client, for those
that do not surface resources to the model).
"""

from __future__ import annotations

from typing import Any

from . import content


def register(mcp: Any) -> None:
    """Register the theory resources and the ``get_iit_reference`` tool.

    Parameters
    ----------
    mcp : FastMCP
        The server application to register on.
    """

    def _make_reader(topic: str):
        def read() -> str:
            return content.load(topic)

        return read

    for topic in content.TOPICS:
        mcp.resource(
            f"pyphi://theory/{topic}",
            name=f"IIT reference: {topic}",
            description=content.TOPICS[topic][1],
            mime_type="text/markdown",
        )(_make_reader(topic))

    @mcp.tool()
    def get_iit_reference(topic: str = "") -> str:
        """Read the grounded IIT reference the server ships with.

        Read the relevant topic *before* building substrates or interpreting Φ,
        so the analysis rests on the actual theory rather than guesswork.

        Parameters
        ----------
        topic : str
            One of the available topics. Leave empty to list them.

        Returns
        -------
        str
            The requested document, or the list of topics when none is given.
        """
        if not topic:
            lines = ["Available IIT reference topics:", ""]
            lines += [f"- {name}: {desc}" for name, desc in content.topics().items()]
            lines += ["", "Call get_iit_reference(topic) to read one."]
            return "\n".join(lines)
        return content.load(topic)
