---
name: pyphi
description: >-
  Use when writing, reading, or running any code that imports pyphi — building
  a Substrate or a TPM, calling analyze(), estimating cost before an
  expensive run, or saving results reproducibly. Covers the
  PyPhi 2.0 API and the MCP server's tools. PyPhi 2.0 renamed most of the
  pre-2.0 surface with no aliases, so code written from memory of older
  versions will not run.
---

# PyPhi

PyPhi computes Integrated Information Theory quantities. For what the theory
says, use the `iit` skill; this one is about the software.

## Use the server for exploration

Where the PyPhi MCP server is connected, drive it through its tools rather than
writing a script. They refuse runs too large to finish and keep φₛ and Φ
apart. Where it is not connected,
`pyphi-mcp install` registers it.

The server holds results in memory only, and none of its tools writes to disk.
Anything that has to be reproducible belongs in a script — see Reproducible
work below.

## The 2.0 API

PyPhi 2.0 renamed most of the pre-2.0 surface and shipped no aliases, so code
written from memory of PyPhi 1.x raises `ImportError`. The full table is
`references/migration.md`. The essentials:

```python
import pyphi

substrate = pyphi.Substrate(tpm, cm=cm, node_labels=labels)
analysis = pyphi.analyze(substrate, state)

analysis.phi           # φₛ, system integrated information
analysis.ces           # the Φ-structure
analysis.ces.big_phi   # Φ, structure integrated information
```

`pyphi.Network` is now `pyphi.Substrate`, `pyphi.Subsystem` is now
`pyphi.System`, and the whole `pyphi.compute` module is replaced by
`pyphi.analyze`.

**One rename is silent.** The old `CauseEffectStructure` is now
`pyphi.models.Distinctions`, and the old `PhiStructure` is now
`pyphi.models.CauseEffectStructure`. The same words point at different objects,
so unported code can import successfully and mean something else. Check every
occurrence of either name.

Compare φ values with `pyphi.numerics.eq`, not `==`.

## Pin the formalism in scripts

PyPhi computes IIT by default. A script should still pin it with the complete
preset, so that a `pyphi_config.yml` in the working directory or a later
change to PyPhi's defaults cannot change its numbers:

```python
with pyphi.config.override(**pyphi.iit4_2026):
    analysis = pyphi.analyze(substrate, state)
```

When you report a number, call its theory IIT. Name a version only when the
number was computed under an earlier one, which PyPhi supports for reproducing
published results; `references/configuration.md` covers them.

## States are little-endian

The first node is the least significant bit, so `(1, 0, 0)` is the first node
on and the rest off. Reversing this produces a well-formed, wrong answer.

## Estimate before running

Analyses are superexponential in substrate size. `pyphi.cost.estimate_analysis`
is free — call it before any run over more than a handful of units.

```python
print(pyphi.cost.estimate_analysis(substrate))
```

## Building a substrate

`references/building-systems.md` has the procedure. The trap worth stating
here: a transition probability matrix built with its axes in the wrong order is
still well-formed, and a symmetric test network will not catch it. After
building one, check that a known state's transition comes back the way you
expect before trusting anything computed from it.

## Reproducible work

`references/reproducible-work.md` has the detail. In short: seed a generator
instance with `np.random.default_rng(seed)` rather than seeding the global one;
save with `pyphi.provenance.save_json`, `save_npz` or `save_dataframe`, which
put the parameters in the filename, refuse to overwrite, and embed the
provenance; and save per-trial values alongside any summary computed from them.

## References

`references/` holds one file per reference topic. Read the one the task calls
for rather than all of them. For anything they do not cover, the documentation
site is readable directly: `references/documentation.md` gives its `llms.txt`,
per-page markdown sources, and API inventory.
