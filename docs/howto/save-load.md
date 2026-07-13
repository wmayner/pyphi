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

Every serializable type also carries the operations as methods. `obj.save(path)`
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

The `Analysis` wrapper returned by `pyphi.analyze` is *not* itself
serializable — it is a convenience container. Save its components (`.ces`,
`.sia`) instead:

```{code-cell} python
try:
    pyphi.save(analysis, out / "analysis.json")
except TypeError as error:
    print(error)
```

## Compatibility note

This serializer is a deliberate format break from the `jsonify` layer used in
PyPhi 1.x. Files written by the old layer cannot be read by `pyphi.load`, and
files written here are not readable by the old `pyphi.jsonify` machinery.
Cause-effect structures are also stored more compactly now: each distinction is
written once in a table, and relations reference their members by index rather
than embedding a full copy of every distinction they contain.
