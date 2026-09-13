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

Every option with its default and what it does, read from the configuration
classes at build time so the page cannot drift from the code. The three
layers and how to set an option are described in {doc}`../howto/configure`;
the presets `pyphi.iit3`, `pyphi.iit4_2023`, and `pyphi.iit4_2026` set the
formalism options together. The classes themselves are
{class}`~pyphi.conf.formalism.IITConfig`,
{class}`~pyphi.conf.formalism.ActualCausationConfig`,
{class}`~pyphi.conf.infrastructure.InfrastructureConfig`, and
{class}`~pyphi.conf.numerics.NumericsConfig`.

```{code-cell} python
:tags: [hide-input]
import ast
import dataclasses
import html
import inspect
import re

from IPython.display import HTML

from pyphi.conf.formalism import ActualCausationConfig, IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


def attribute_docs(cls):
    """The docstring written under each field of a dataclass, by field name."""
    tree = ast.parse(inspect.getsource(cls))
    body = tree.body[0].body
    docs = {}
    for stmt, nxt in zip(body, body[1:]):
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[stmt.target.id] = nxt.value.value
    return docs


def describe(doc):
    """The docstring as HTML: roles become plain names, literals become code."""
    text = " ".join(doc.split())
    text = re.sub(r":\w+:`~?([^`<]+?)(?: <[^>]*>)?`", r"\1", text)
    text = html.escape(text, quote=False)
    return re.sub(r"``([^`]+)``", r"<code>\1</code>", text)


LAYERS = (
    ("Formalism: IIT", "formalism.iit", IITConfig),
    ("Formalism: actual causation", "formalism.actual_causation", ActualCausationConfig),
    ("Infrastructure", "infrastructure", InfrastructureConfig),
    ("Numerics", "numerics", NumericsConfig),
)
parts = ['<div class="pp-options">']
for title, layer, cls in LAYERS:
    docs = attribute_docs(cls)
    instance = cls()
    parts.append(f"<h2>{title} <small>({layer})</small></h2><dl>")
    for field in dataclasses.fields(cls):
        default = html.escape(repr(getattr(instance, field.name)))
        parts.append(
            f'<dt id="opt-{field.name}"><code>{field.name}</code>'
            f' <span class="pp-default">= {default}</span></dt>'
            f"<dd>{describe(docs.get(field.name, ''))}</dd>"
        )
    parts.append("</dl>")
parts.append("</div>")
HTML("".join(parts))
```
