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
