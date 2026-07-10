# PyPhi MCP server — primer

This server computes Integrated Information Theory (IIT) quantities with PyPhi
and helps explain them. Read the reference before interpreting results:
`get_iit_reference("theory")`, `get_iit_reference("gotchas")`, and
`get_iit_reference("equations")`. The same documents are also available as
`pyphi://theory/*` resources.

## What IIT computes

Given a **substrate** (a set of interacting units, defined by a transition
probability matrix) and a **state** of that substrate, IIT answers two
questions:

1. **Does a set of units exist as one integrated whole, and how much?** This is
   the **system integrated information**, written φₛ ("small phi, system"). The
   set of units with maximal φₛ is a **complex**.
2. **What structure does it specify?** Unfolding the complex gives its
   **Φ-structure**: the **distinctions** its subsets specify and the
   **relations** among them. The total is the **structure integrated
   information**, Φ ("big phi").

φₛ and Φ are different quantities — do not conflate them. See the theory
reference.

## The tools

- `list_examples()` / `load_example(name)` — the standard networks from the IIT
  literature. Start here.
- `build_substrate(tpm, cm?, node_labels?, alphabet?)` — build your own from a
  transition probability matrix.
- `describe_substrate(handle)` — inspect a substrate.
- `analyze(handle, state, formalism?, compute?, detail?, confirm_large?)` — the
  workhorse. Returns a readable card, scalar values, and a `result_ref`.
- `inspect(result_ref, path)` — drill into one part of a result.
- `plot(target, kind)` — render PyPhi's built-in visualizations (needs
  `pip install pyphi[visualize]`): `kind="ces"` the Φ-structure,
  `"repertoires"` the cause/effect repertoires, `"connectivity"` the causal
  graph, `"tpm"` the transition probability matrix.
- `get_iit_reference(topic)` — the grounded theory reference.

## Two things to keep straight from the start

- **States are little-endian.** The *first* node is the least-significant bit,
  the opposite of ordinary positional notation. A state is given as a tuple in
  node order, e.g. `(1, 1, 0)`.
- **Φ = 0 means the system is reducible**, not that it has no structure. A
  positive value is what indicates irreducible integrated information.

## Prompts

- `explain_result` — narrate an analysis result in plain language.
- `build_system_walkthrough` — turn a description of some units into a valid
  transition probability matrix.
