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

{mod}`pyphi.examples` collects the substrates, systems, TPMs and transitions
used throughout the IIT literature and this documentation, so that a worked
example is one function call away. Every example is registered in
`pyphi.examples.EXAMPLES`, a mapping from category (`substrate`, `system`,
`tpm`, `transition`) to the functions that build them; `xor_substrate`, for
instance, is `EXAMPLES["substrate"]["xor"]`. The cards and connectivity graphs
below are drawn from the objects themselves when this page is built. The cards leave out the transition probability
matrices; load an example to see its TPM.

```{code-cell} python
:tags: [hide-input]
import inspect

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import pyphi
from pyphi.examples import EXAMPLES
from pyphi.visualize import plot_system

pyphi.config.progress_bars = False
# The cards below omit each substrate's TPM grid; the object carries it.
pyphi.config.repr_verbosity = 1


def summary(func):
    """The first paragraph of a function's docstring."""
    doc = inspect.getdoc(func) or ""
    return " ".join(doc.split("\n\n")[0].split())
```

## Substrates

```{code-cell} python
:tags: [hide-input]
for name, func in EXAMPLES["substrate"].items():
    display(Markdown(f"### {name}\n\n{summary(func)}"))
    try:
        substrate = func()
        display(substrate)
        system = pyphi.System(
            substrate,
            state=(0,) * substrate.size,
            node_indices=tuple(range(substrate.size)),
        )
        plt.figure(figsize=(4, 4))
        plot_system(system)
        plt.show()
        plt.close()
    except Exception as exc:  # noqa: BLE001
        display(Markdown(f"*Could not render:* `{type(exc).__name__}: {exc}`"))
```

## Systems, TPMs and transitions

A system example is its substrate in the state used by the source it comes
from. TPM examples return a bare transition probability matrix, and transition
examples a {class}`~pyphi.actual.Transition` for actual causation.

```{code-cell} python
:tags: [hide-input]
for category in ("system", "tpm", "transition"):
    items = "\n".join(
        f"- `{name}_{category}` — {summary(func)}"
        for name, func in EXAMPLES[category].items()
    )
    display(Markdown(f"### {category.capitalize()}s\n\n{items}"))
```
