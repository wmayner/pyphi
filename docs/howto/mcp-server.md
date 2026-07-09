# Use PyPhi from an AI assistant (MCP server)

PyPhi ships a [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server that lets an AI assistant build substrates, run IIT analyses, inspect the
results, visualize them, and read a grounded reference on the theory — all by
talking to it in plain language. It runs locally against the PyPhi in your own
environment; no computation is sent anywhere.

It is useful two ways: as a calculator and interpreter for researchers who would
rather describe a system than write the API calls, and as a tutor for newcomers
exploring what Φ, distinctions, and relations actually mean.

## Install

```bash
pip install "pyphi[mcp]"
```

The IIT 3.0 formalism needs the earth-mover-distance backend, and plotting needs
the visualization stack, so install those extras too if you want them:

```bash
pip install "pyphi[mcp,emd,visualize]"
```

## Connect it to a client

The server speaks MCP over stdio and is started by the `pyphi-mcp` console
script (equivalently, `python -m pyphi.mcp`). Clients launch it for you; you
rarely run it by hand.

### Claude Desktop

Add it to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "pyphi-mcp"
    }
  }
}
```

If `pyphi-mcp` is not on the client's `PATH`, use the absolute path to the
executable in your environment (the one `which pyphi-mcp` prints inside your
virtual environment).

### Claude Code

Add an `.mcp.json` to your project (or run `claude mcp add`):

```json
{
  "mcpServers": {
    "pyphi": {
      "command": "pyphi-mcp"
    }
  }
}
```

## A first conversation

Once connected, you talk to the assistant, not to the tools. Here is a typical
session and what happens under the hood.

**You:** *What example networks does PyPhi have, and what is the "basic" one?*

The assistant calls `list_examples` and `load_example("basic")`. It gets back a
handle for the substrate plus its shape: three nodes A, B, C — a logic-gate
network where A is an OR, B a COPY, and C an XOR of the others.

**You:** *Compute Φ for it in the state where A and B are on and C is off.*

The assistant calls `analyze` with the state `(1, 1, 0)` and reports:

- **φₛ = 0.208** — the system integrated information, which says the network
  exists as one integrated whole (a positive value; zero would mean it is
  reducible).
- **Φ = 1.857** — the structure integrated information, summed over the
  network's **3 distinctions** and **2 relations**.

Because φₛ and Φ are different quantities — one is about existing as a whole, the
other about how much structure is specified — a good assistant keeps them
distinct rather than reporting "the phi value." The bundled reference is what
teaches it to do that (see below).

**You:** *Which mechanism contributes the most, and why?*

The assistant calls `inspect` on the result to read a specific distinction in
full — its mechanism, its cause and effect purviews, and its φ value — without
dragging the entire (potentially very large) structure back into the
conversation.

**You:** *How would this change under IIT 3.0, or the 2026 formalism?*

The assistant re-runs `analyze` with a different `formalism`. The same network
gives **0.208** under IIT 4.0 (2023), **0.0** under the 2026 formalism (which
requires a system to provide itself a repertoire of alternatives — a fully
deterministic network provides none), and **0.188** under IIT 3.0 (which uses a
different distance measure and has no relations). Same system, three theories,
correctly different answers.

**You:** *Show me the cause-effect structure.*

The assistant calls `plot`. The Φ-structure is an interactive 3-D figure, so the
tool returns a path to a self-contained HTML file to open in a browser — it is
not shown inline, because a static snapshot of something meant to be rotated and
hovered would be misleading. The connectivity graph, transition probability
matrix, and repertoire plots are static, so those *are* returned as an inline
image.

## What it exposes

**Tools**

- `list_examples` / `load_example` — the standard networks from the IIT
  literature.
- `build_substrate` — build a substrate from a transition probability matrix.
- `describe_substrate` — inspect a substrate's nodes, connectivity, and state
  convention.
- `analyze` — compute system integrated information φₛ and the Φ-structure of a
  substrate in a state, under any of the three formalism versions.
- `inspect` — drill into one part of a result (a distinction, a repertoire).
- `plot` — render PyPhi's built-in visualizations (needs the `visualize`
  extra): the cause-effect structure (`kind="ces"`), the cause and effect
  repertoires (`"repertoires"`), the causal connectivity graph
  (`"connectivity"`), or the transition probability matrix (`"tpm"`).
- `get_iit_reference` — read the bundled, citation-checked reference on the
  theory, the equations, and the common pitfalls.

**Resources** — the same reference documents, at `pyphi://theory/{topic}`.

**Prompts** — `explain_result` (narrate an analysis in plain language) and
`build_system_walkthrough` (turn a description of some units into a valid
transition probability matrix).

## Good to know

- **The assistant is taught the theory.** A citation-checked reference (the
  postulates, the key equations, and the subtleties that most often mislead)
  ships with the server as its startup instructions, as `pyphi://theory/*`
  resources, and through `get_iit_reference`. This is what keeps it from, for
  example, reading Φ = 0 as "no structure" (it means *reducible*) or confusing
  φₛ with Φ. If you are new to IIT, asking the assistant to read
  `get_iit_reference("theory")` and `get_iit_reference("gotchas")` first is
  worthwhile.
- **State ordering is little-endian.** The first node is the least-significant
  bit — the opposite of ordinary positional notation. States are given as tuples
  in node order, e.g. `(1, 1, 0)`. The assistant knows this, but it is the most
  common source of confusion when reading raw indices.
- **Big analyses are guarded.** Φ is combinatorially expensive (the practical
  ceiling for an exact analysis is about 10–12 units), so `analyze` refuses a
  full analysis of a large substrate unless you confirm it. A full Φ-structure
  can serialize to megabytes, so `analyze` returns a compact summary by default
  and `inspect` reads a specific part in full on request.
- **It runs entirely locally.** The server executes in your environment against
  your installed PyPhi; nothing is uploaded.
