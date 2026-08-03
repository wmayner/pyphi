---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Save and load results

PyPhi results are expensive to compute, so you will usually want to store them
and read them back later. Two functions do this: `pyphi.save` writes a result
to a file, and `pyphi.load` reads it back into a live object. The same
operations are also available as `.save()` and `.load()` methods on each
serializable result type.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

First, compute something worth saving. Here `pyphi.analyze` returns an
`Analysis`, from which we take the cause-effect structure and the
system-irreducibility analysis.

```{code-cell} python
from pyphi import examples

system = examples.iit4_2023_fig1a_system()
analysis = pyphi.analyze(system.substrate, system.state)

ces = analysis.ces  # cause-effect structure (distinctions + relations)
sia = analysis.sia  # system-irreducibility analysis (Φ_s)

print(f"Φ_s = {float(sia.phi):.4f}   |   {len(ces.distinctions)} distinctions")
```

## Save and load

`pyphi.save` takes the object and a path; `pyphi.load` takes the path and
reconstructs the object. Equality is value-based, so a round-trip compares
equal to the original.

```{code-cell} python
from pathlib import Path
from tempfile import mkdtemp

out = Path(mkdtemp())

pyphi.save(ces, out / "ces.json")
restored = pyphi.load(out / "ces.json")

print("type preserved:", type(restored).__name__)
print("round-trips   :", restored == ces)
```

Every serializable type also has these operations as methods. `obj.save(path)`
is the same as `pyphi.save(obj, path)`, and the classmethod `Type.load(path)`
is the same as `pyphi.load(path)`:

```{code-cell} python
ces.save(out / "ces2.json")
CauseEffectStructure = type(ces)
CauseEffectStructure.load(out / "ces2.json") == ces
```

## File formats

The wire format is inferred from the file extension:

- `.json` — a human-readable JSON document.
- `.mpk` — [msgpack](https://msgpack.org/), a compact binary encoding of the
  same document. Prefer it for large structures.
- a trailing `.gz` on either (`.json.gz`, `.mpk.gz`) is transparently
  gzip-compressed on save and decompressed on load.

```{code-cell} python
for name in ["ces.json", "ces.mpk", "ces.json.gz"]:
    pyphi.save(ces, out / name)
    size = (out / name).stat().st_size
    print(f"{name:<12} {size:>7,} bytes")
```

All three round-trip to the same object:

```{code-cell} python
all(pyphi.load(out / name) == ces for name in ["ces.json", "ces.mpk", "ces.json.gz"])
```

To choose the format explicitly instead of inferring it from the extension,
pass `format="json"` or `format="msgpack"` to either function.

## What is serializable

Most result types serialize: `Substrate`, `System`, cause-effect structures,
system-irreducibility analyses, and actual-causation results all round-trip.

```{code-cell} python
for obj in [system.substrate, system, sia, ces]:
    pyphi.save(obj, out / "obj.mpk")
    round_tripped = pyphi.load(out / "obj.mpk") == obj
    print(f"{type(obj).__name__:<30} {round_tripped}")
```

Batch-run results round-trip too: a `pyphi.sweep` table with its raw
results, and an `optimize` outcome including the winning substrate and its
analysis. Their DataFrames are embedded in the document as
[parquet](https://parquet.apache.org/), so dtypes and NaN values survive
exactly.

```{code-cell} python
import pandas as pd

result = pyphi.sweep(system.substrate, states=[system.state], progress=False)
pyphi.save(result, out / "sweep.json")
loaded = pyphi.load(out / "sweep.json")
pd.testing.assert_frame_equal(loaded.df, result.df)
loaded.df
```

The `Analysis` returned by `pyphi.analyze` saves as a whole, carrying its
system, system-irreducibility analysis, and cause-effect structure together, so
you do not have to save the components separately:

```{code-cell} python
pyphi.save(analysis, out / "analysis.json")
reloaded = pyphi.load(out / "analysis.json")

print(f"φ_s = {reloaded.phi:.4f}   |   Φ = {reloaded.big_phi:.4f}")
```

## Experiment provenance writers

Experiment scripts need two things beyond `pyphi.save`: output files that
never overwrite earlier runs, and a record of how each file was produced.
The writers in `pyphi.provenance` provide both. Parameters are encoded
into the filename, a repeated save lands in a `_v2` file instead of
clobbering the first, and every file embeds a full provenance record —
pyphi version, git commit, timestamp, and seed.

```{code-cell} python
from pyphi import provenance

path = provenance.save_json(
    {"phi": 0.133873},
    out,
    "sweep_study",
    params={"seed": 42, "trials": 60},
)
path.name
```

```{code-cell} python
provenance.save_json(
    {"phi": 0.5}, out, "sweep_study", params={"seed": 42, "trials": 60}
).name
```

`save_npz` does the same for arrays of raw per-trial data, and
`save_dataframe` writes a DataFrame as parquet — the format used for
DataFrame outputs throughout PyPhi — with the metadata embedded in the
parquet schema. `read_metadata` recovers the provenance and parameters
from any of the three formats:

```{code-cell} python
metadata = provenance.read_metadata(path)
{key: metadata["provenance"][key] for key in ("seed", "pyphi_version")}
```

## Compatibility note

This serializer is a deliberate format break from the `jsonify` layer used in
PyPhi 1.x. Files written by the old layer cannot be read by `pyphi.load`, and
files written here are not readable by the old `pyphi.jsonify` machinery.
Cause-effect structures are also stored more compactly now: each distinction is
written once in a table, and relations reference their members by index rather
than embedding a full copy of every distinction they contain.
