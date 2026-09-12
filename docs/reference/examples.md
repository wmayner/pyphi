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

# Example networks

Every registered example, generated from {mod}`pyphi.examples` at build
time. Each loads with `pyphi.examples.<name>()`; the first line of its
docstring gives its source. A system example is the corresponding substrate
in the paper's state.

```{code-cell} python
:tags: [hide-input]
import inspect

import pandas as pd

import pyphi
from pyphi.examples import EXAMPLES

pyphi.config.progress_bars = False


def first_line(func):
    return (inspect.getdoc(func) or "").split("\n")[0]


rows = []
for category in ("substrate", "system", "transition", "tpm"):
    for stem, func in sorted(EXAMPLES[category].items()):
        units = ""
        if category == "substrate":
            units = func().size
        source = first_line(func)
        if not source and stem in EXAMPLES["substrate"]:
            source = first_line(EXAMPLES["substrate"][stem])
        rows.append(
            {
                "name": f"{stem}_{category}",
                "category": category,
                "units": units,
                "source": source,
            }
        )
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.DataFrame(rows).set_index("name")
```
