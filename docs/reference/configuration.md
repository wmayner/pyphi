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

# Configuration options

Every option, its layer, and its default, read from the configuration
classes at build time. The three layers and how to set an option are
described in {doc}`../howto/configure`; each class's documentation below
explains its options in prose.

```{code-cell} python
:tags: [hide-input]
import dataclasses

import pandas as pd

from pyphi.conf.formalism import ActualCausationConfig, IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig

rows = []
for layer, cls in (
    ("formalism.iit", IITConfig),
    ("formalism.actual_causation", ActualCausationConfig),
    ("infrastructure", InfrastructureConfig),
    ("numerics", NumericsConfig),
):
    instance = cls()
    for field in dataclasses.fields(cls):
        rows.append(
            {
                "option": field.name,
                "layer": layer,
                "default": repr(getattr(instance, field.name)),
            }
        )
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.DataFrame(rows).set_index("option")
```

## Formalism: IIT

```{eval-rst}
.. autoclass:: pyphi.conf.formalism.IITConfig
   :noindex:
```

## Formalism: actual causation

```{eval-rst}
.. autoclass:: pyphi.conf.formalism.ActualCausationConfig
   :noindex:
```

## Infrastructure

```{eval-rst}
.. autoclass:: pyphi.conf.infrastructure.InfrastructureConfig
   :noindex:
```

## Numerics

```{eval-rst}
.. autoclass:: pyphi.conf.numerics.NumericsConfig
   :noindex:
```
