"""Model Context Protocol server exposing PyPhi to AI agents.

The server lets an agent build substrates, run IIT analyses, inspect the
resulting cause-effect structures, and read a grounded reference on the theory,
all through the tools, resources, and prompts of the Model Context Protocol.

It runs locally against the PyPhi installed in the current environment. Install
its dependency with ``pip install pyphi[mcp]`` and start it with the
``pyphi-mcp`` console script (or ``python -m pyphi.mcp``).

The server implementation lives in :mod:`pyphi.mcp.server`, which requires the
optional ``mcp`` dependency; importing this package does not.
"""

import sys


def main(argv: list[str] | None = None) -> int:
    """The ``pyphi-mcp`` entry point.

    Runs :func:`pyphi.mcp.server.main`, or says how to install the optional
    ``mcp`` dependency if it is missing.
    """
    try:
        from .server import main as run
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "mcp":
            raise
        sys.stderr.write(
            "pyphi-mcp needs the optional `mcp` dependency; install it with "
            '`pip install "pyphi[mcp]"`.\n'
        )
        return 1
    return run(argv)
