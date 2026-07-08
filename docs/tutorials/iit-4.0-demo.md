---
jupytext:
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

# The IIT 4.0 demo notebook

{download}`Download this page as a Jupyter notebook <../examples/IIT_4.0_demo.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/examples/IIT_4.0_demo.ipynb)

This notebook is the supplement to the IIT 4.0 paper
([Albantakis et al. 2023](https://doi.org/10.1371/journal.pcbi.1011465)). It
walks through a full IIT 4.0 analysis in PyPhi and reproduces the paper's
numbers on the paper's own example systems.

Part 1 works through a System Irreducibility Analysis one IIT postulate at a
time — intrinsicality, information, integration ($φ_s$), exclusion (finding the
first complex), and composition (unfolding the Φ-structure of distinctions and
relations) — on the deterministic three-node A/B/C system from Figure 8C.

Part 2 reproduces Figures 1, 2, and 4 on the nondeterministic Ising network
used in the main text, going deeper into the algorithmic computation of
complexes, distinctions, and relations.

The notebook is a standalone download-and-run artifact rather than an executed
documentation page. For the theory behind each step, see the
{doc}`../theory/index`; for a shorter hands-on tour of the cause-effect
structure, see {doc}`cause-effect-structure`.
