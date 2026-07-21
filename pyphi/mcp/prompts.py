"""Reusable prompts for common IIT workflows.

Prompts are user-invokable templates that a client surfaces (for example, in a
slash-command menu). They stitch the server's tools and reference material into
guided workflows: making sense of a result, porting pre-2.0 code, turning a
description of some units into a valid transition probability matrix, and
planning a cluster campaign for a large system.
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
            "- What are φₛ and Φ here; remember that φₛ=0 means the system is "
            "reducible.\n"
            "- What the distinctions and relations are, concretely.\n"
            "- Any caveats, such as ties from a symmetric transition probability "
            "matrix.\n"
            "Avoid jargon where a plain word is just as precise."
        )

    @mcp.prompt()
    def migrate_code(code: str) -> str:
        """Rewrite pre-2.0 PyPhi code for PyPhi 2.0.

        Parameters
        ----------
        code : str
            The pre-2.0 PyPhi code to port (e.g. code using ``pyphi.Network``,
            ``pyphi.Subsystem``, ``pyphi.compute.*``, or a flat config).
        """
        return (
            "Rewrite this PyPhi code for version 2.0:\n\n"
            f"{code}\n\n"
            "First read the 'migration' and 'gotchas' reference topics with "
            "get_iit_reference. There are no deprecation shims, so every "
            "pre-2.0 name must be changed. Apply the renames, replace "
            "pyphi.compute.* with pyphi.analyze, update any config and jsonify "
            "usage, and preserve the original behavior — if the code relied on "
            'the old IIT 3.0 default, pass formalism="IIT_3_0" so the numbers '
            "still match. Show the rewritten code and note any change that "
            "alters computed values (especially that deterministic systems give "
            "φ_s = 0 under the new 2026 default)."
        )

    @mcp.prompt()
    def campaign_walkthrough(description: str = "") -> str:
        """Plan a cluster campaign for a large system, step by step.

        Parameters
        ----------
        description : str, optional
            Anything already known about the system and the desired
            results (size, dynamics, which quantities matter, cluster
            access). Leave empty to start from scratch.
        """
        context = f"What I know so far:\n\n{description}\n\n" if description else ""
        return (
            "Help me set up an HTCondor campaign (for example on UW-Madison's "
            f"CHTC) to analyze a large system with PyPhi.\n\n{context}"
            "First read the 'campaigns' and 'performance' reference topics "
            "with get_iit_reference. Then walk me through the setup "
            "interactively — one question at a time, each step's outcome "
            "reviewed with me before the next:\n\n"
            "1. The system. Get the substrate into the server: an example via "
            "load_example, or build mine with build_substrate (use the "
            "build_system_walkthrough approach if I only have a description). "
            "Confirm it with describe_substrate. Establish the state to "
            "analyze and what result I actually need: the cause-effect "
            "structure of one system (a CES campaign), or many independent "
            "runs across states/substrates/formalisms (a sweep campaign).\n"
            "2. Honest feasibility. Price the full, unscoped workload with "
            "estimate_cost. Work units are enumeration counts, not seconds; "
            "as an anchor, a single 72-hour condor slot covers very roughly "
            "10^8 to 10^10 units depending on per-unit cost, so counts far "
            "beyond that per cell mean the full computation is out of reach "
            "no matter how many jobs we use. Tell me plainly if it is.\n"
            "3. Scope (CES campaigns). If the full surface is infeasible, "
            "elicit a feasible one: which mechanisms matter to me (an "
            "explicit list, an order bound, units that must be involved), "
            "and any purview constraints. Re-run estimate_cost with the "
            "scope until the total is tractable, showing me the numbers at "
            "each step. Be clear about what exclusion means: within the "
            "scope every value is exact, and the excluded remainder is "
            "covered by certified bounds in the scope report — a scope "
            "narrows the computation, it never approximates it.\n"
            "4. The SIA. Decide how the system irreducibility analysis is "
            "handled: sharded in the campaign (the default), supplied "
            "precomputed (sia_ref, if I already have one), or skipped — in "
            "which case congruence resolves against the intrinsic-"
            "information state and the result carries no Φₛ.\n"
            "5. Budget and packing. Choose units_per_job with me: enough "
            "jobs to use the pool, each job minutes-to-hours of work, and "
            "well under the 72-hour slot. Review the planned task ledger "
            "and any admission-control warnings from prepare_ces_campaign "
            "(or prepare_campaign for sweeps) before treating the campaign "
            "as ready. Set a seed.\n"
            "6. Hand-off. Give me the exact cluster steps: build the "
            "container image, copy the campaign directory to the access "
            "point, condor_submit pyphi.sub; then campaign_status to "
            "monitor (resubmission is just condor_submit again), and "
            "collect_campaign when done — including how to read the scope "
            "report's Σφ_r lower bound and measured upper bounds.\n\n"
            "Never prepare into an existing directory, and confirm my "
            "choices at each numbered step before acting on them."
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
            "result with describe_substrate. Check with the user that the "
            "substrate you built matches their intent."
        )
