# Reproducible work

The MCP server holds results in memory only, and none of its tools writes to
disk. Anything that has to survive the session, be rerun, or be cited belongs
in a script. These are the conventions that make such a script reproducible.

## Seeding

Any computation that draws randomness takes a `seed` argument and uses an
isolated generator built from it:

```python
rng = np.random.default_rng(seed)
```

Never call `np.random.seed()` or `random.seed()`. Those mutate global state,
which makes the function non-reentrant and silently couples callers that were
meant to be independent.

Save the seed alongside the output, not just to the log. A seed that exists
only in a terminal scrollback has not been recorded.

## Saving results

`pyphi.provenance` has three writers — `save_json`, `save_npz` and
`save_dataframe` — which take the same arguments and differ only in what they
write:

```python
from pyphi.provenance import save_json

path = save_json(
    results,
    directory="results",
    name="phi-sweep",
    params={"seed": seed, "n": n_trials, "formalism": "iit4_2026"},
)
```

Each writer does three things a bare `open()` does not:

- **The parameters go into the filename.** `params` becomes one
  `{key}{value}` segment per entry, so `phi-sweep_seed42_n60_formalismiit4_2026.json`
  says what it holds without being opened.
- **An existing file is never overwritten.** A colliding name gets a `_v2`,
  then `_v3`, and so on. Replacing a result stays the user's decision, made by
  deleting the old file first.
- **The file describes itself.** A JSON file holds
  `{"provenance": …, "params": …, "data": …}`; an NPZ holds the same two
  records under the reserved `_provenance` and `_params` keys. The provenance
  record carries the PyPhi version, the git revision and whether the tree was
  dirty, the timestamp, the Python, numpy and scipy versions, the platform, and
  the seed — taken from `seed=` or from `params["seed"]`.

`pyphi.provenance.read_metadata(path)` returns those two records without
loading the payload, so a directory of results can be inventoried cheaply.

`pyphi.save` and `pyphi.load` serialize PyPhi objects themselves — an
`Analysis`, a `CauseEffectStructure` — where you want the object back rather
than a summary.

## Save the raw values, not only the summary

Where a script computes a summary — a mean, a correlation, a rate — the
per-trial or per-element values behind it go to disk too, in the same NPZ or
JSON. Without them a reviewer cannot recompute the summary a different way, and
re-running the experiment becomes the only way to answer a follow-up question.

If a script computes a correlation between two quantities, the paired
observations are part of the output.

## Pin the formalism

A φ value means nothing without the formalism that produced it. Pin it
explicitly in the script rather than inheriting the ambient default, and record
which one in the output:

```python
with pyphi.config.override(**pyphi.iit4_2026):
    analysis = pyphi.analyze(substrate, state)
```

## Long runs

`pyphi.cost.estimate_analysis` is free; call it before committing to a run. The
`performance` topic covers the disk cache and checkpointing, and the
`campaigns` topic covers distributing a sweep across a cluster.
