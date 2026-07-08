---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Your first computation

{download}`Download this page as a Jupyter notebook <first-computation.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/getting-started/first-computation.ipynb)

This page verifies your installation by constructing the three-node example
system used throughout the documentation.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
system = pyphi.examples.basic_system()
system
```

A full walkthrough — from a transition probability matrix to a φ-structure —
is being written; see the tutorials section in the meantime.
