# PyPhi MCP server — primer

This server computes Integrated Information Theory (IIT) quantities with PyPhi
and helps explain them. The gotchas — the mistakes that produce wrong results —
follow this primer, already loaded. For anything beyond them, read
`get_iit_reference("theory")` and `get_iit_reference("equations")` before
interpreting results; the same documents are also available as
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
- `analyze(handle, state, formalism?, compute?, detail?, confirm_large?,
  parallel?, workers?)` — the workhorse. Returns a readable card, scalar
  values, and a `result_ref`. `parallel=true` runs that call on multiple
  cores; see `get_iit_reference("parallelization")`.
- `configure_parallel(enable?, levels?, workers?, reset?)` — read or
  persistently set the server's parallelization configuration.
- `inspect(result_ref, path)` — drill into one part of a result.
- `plot(target, kind)` — render PyPhi's built-in visualizations (needs
  `pip install pyphi[visualize]`): `kind="ces"` the Φ-structure,
  `"repertoires"` the cause/effect repertoires, `"connectivity"` the causal
  graph, `"tpm"` the transition probability matrix. For a particular view of the
  Φ-structure, an analytically-computed structure, or plotting one you built
  yourself, read `get_iit_reference("visualization")`.
- `get_iit_reference(topic)` — the grounded theory reference. Porting code
  written against an older PyPhi? Read `get_iit_reference("migration")` first:
  2.0 renamed the core objects and has no compatibility shims.

These tools cover the common path. For anything beyond them — sweeps over many
systems, macro (coarse-graining), actual causation, relation queries, saving and
loading results — drive PyPhi through its Python API. Consult the documentation
at https://pyphi.readthedocs.io, or the installed package's own docstrings
(`help(pyphi.analyze)`, `dir(pyphi)`), rather than guessing method names.

When you run PyPhi in a shell rather than through these tools, use the project's
virtual environment — `uv run python` if it is a uv project, otherwise the
environment's own `python` — never the bare system `python`. Running PyPhi in a
shell does not exempt you from this reference: the gotchas below govern a
result however it was computed, and for anything they do not cover, read
`get_iit_reference("theory")` rather than answering from memory. IIT's
quantities are easy to name plausibly and get wrong, and a wrong claim written
into a project's notes or scripts propagates into every later answer.

## Configuring and running PyPhi

- **Configuration** is one object, `pyphi.config`, in three layers: `formalism`
  (what is computed), `infrastructure` (parallelism, caching, progress), and
  `numerics` (precision). Through these tools, select a formalism by passing
  `formalism=` to `analyze`; read `get_iit_reference("configuration")` before
  changing options in a script.
- **To use multiple cores**, read `get_iit_reference("parallelization")` first
  — the global `parallel` flag alone does nothing; each level of the
  computation must also be enabled, and the tools' `parallel` controls handle
  this for you.
- **Expensive runs can hang the machine, and PyPhi has no built-in
  checkpointing** — a run that hangs or is killed loses everything in progress.
  Exact Φ for all distinctions is practical to about 10–12 units depending on
  the sparsity of the substrate topology. Before starting any analysis large
  enough to plausibly thrash, consider enabling the opt-in disk result cache
  (`pyphi.config.disk_cache_results = True`) so completed results survive, and
  for a sweep over many systems save progress as it goes so a restart resumes
  instead of recomputing (but confirm with the user first, in case they don't
  want persistent consumption of disk space). See
  `get_iit_reference("performance")`.

## Prompts

- `explain_result` — narrate an analysis result in plain language.
- `build_system_walkthrough` — turn a description of some units into a valid
  transition probability matrix.
- `migrate_code` — rewrite pre-2.0 PyPhi code for 2.0.
