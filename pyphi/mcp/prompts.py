"""Reusable prompts for common IIT workflows.

Prompts are user-invokable templates that a client surfaces (for example, in a
slash-command menu). They stitch the server's tools and reference material into
the two tasks newcomers most often want help with: making sense of a result,
and turning a description of some units into a valid transition probability
matrix.
"""

from __future__ import annotations

from typing import Any


def register(mcp: Any) -> None:
    """Register the workflow prompts.

    Parameters
    ----------
    mcp : FastMCP
        The server application to register on.
    """

    @mcp.prompt()
    def explain_result(result_ref: str) -> str:
        """Narrate an analysis result in plain language.

        Parameters
        ----------
        result_ref : str
            A ``result_ref`` returned by the ``analyze`` tool.
        """
        return (
            f"Explain the IIT analysis result {result_ref} in plain language "
            "for someone new to the theory.\n\n"
            "First read the 'interpreting-results' and 'gotchas' reference "
            "topics with get_iit_reference. Then use inspect() on the result "
            "to read its parts. In your explanation, cover:\n"
            "- What Φ is here, and remember that Φ=0 means the system is "
            "reducible, not that it lacks structure.\n"
            "- The distinction between φₛ (whether the system exists as one "
            "integrated whole) and Φ (how much structure it specifies).\n"
            "- What the distinctions and relations are, concretely.\n"
            "- Any caveats: ties from a symmetric transition probability "
            "matrix, or the formalism version's effect on the numbers.\n"
            "Avoid jargon where a plain word is just as precise."
        )

    @mcp.prompt()
    def build_system_walkthrough(description: str) -> str:
        """Turn a description of some units into a valid substrate.

        Parameters
        ----------
        description : str
            A natural-language description of the units and how they influence
            one another (e.g. "three units A, B, C where A fires if B and C "
            "were both on last step").
        """
        return (
            "Help me build a PyPhi substrate for this system:\n\n"
            f"{description}\n\n"
            "First read the 'building-systems' and 'gotchas' reference topics "
            "with get_iit_reference, paying attention to the little-endian "
            "state ordering (the first node is the least-significant bit) and "
            "the requirement that the transition probability matrix be "
            "interventional and conditionally independent. Then construct the "
            "state-by-node transition probability matrix row by row, explain "
            "each row, call build_substrate to create it, and confirm the "
            "result with describe_substrate."
        )
