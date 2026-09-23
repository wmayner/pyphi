# PyPhi

```{raw} html
<p class="pp-tagline">The toolbox for Integrated Information Theory.</p>
```

PyPhi 2.0 is out as a release candidate. Install it with `--pre`; without
the flag, pip installs the 1.2 release instead.

```bash
pip install --pre pyphi
```

```python
import pyphi

# The IIT 4.0 paper's Fig. 1A network, analyzing units A and B
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
analysis = pyphi.analyze(substrate, state=(0, 1, 1), subset=(0, 1))

analysis.phi      # φₛ ≈ 0.04, system integrated information
analysis.big_phi  # Φ ≈ 1.56, structure integrated information
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/index
:link-type: doc
Install PyPhi and compute your first φ.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: tutorials/index
:link-type: doc
Learn the library through worked, executable examples.
:::

:::{grid-item-card} {octicon}`tools` How-to guides
:link: howto/index
:link-type: doc
Build a substrate, read a result, size a run, configure, parallelize, export.
:::

:::{grid-item-card} {octicon}`beaker` Theory
:link: theory/index
:link-type: doc
How IIT's mathematics maps onto PyPhi's types and functions.
:::

:::{grid-item-card} {octicon}`list-unordered` Reference
:link: reference/index
:link-type: doc
The API reference, the glossary, configuration options, and conventions.
:::

:::{grid-item-card} {octicon}`arrow-switch` Migration
:link: migration/index
:link-type: doc
Moving to PyPhi 2.0 from earlier versions and related tools.
:::
::::

::::{grid} 1
:gutter: 3

:::{grid-item-card} {octicon}`dependabot` AI assistants
:link: howto/mcp-server
:link-type: doc
PyPhi ships an MCP server that lets an assistant build substrates, size runs,
and analyze them. The site is also readable without it: `llms.txt` and
`llms-full.txt` at the site root, the MyST source of every page under
`_sources/`, and `objects.inv` for resolving API names.
:::
::::

---

If you use this software in your research, please cite the software paper:

> Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018).
> PyPhi: A toolbox for integrated information theory.
> *PLOS Computational Biology* 14(7): e1006343.
> <https://doi.org/10.1371/journal.pcbi.1006343>

The theory it implements, IIT 4.0, is described in:

> Albantakis L, Barbosa L, Findlay G, Grasso M, … Tononi G. (2023).
> Integrated information theory (IIT) 4.0: formulating the properties of
> phenomenal existence in physical terms.
> *PLoS Computational Biology* 19(10): e1011465.
> <https://doi.org/10.1371/journal.pcbi.1011465>
>
> Mayner WGP, Marshall W, Tononi G. (2026).
> Intrinsic cause–effect power: the tradeoff between differentiation and
> specification.
> *Entropy* 28(4): 410.
> <https://doi.org/10.3390/e28040410>

BibTeX entries are on the {doc}`citing` page.
To report issues, use the [issue tracker](https://github.com/wmayner/pyphi/issues).
For general discussion, join the [pyphi-users group](https://groups.google.com/forum/#!forum/pyphi-users).

## Everything in the docs

```{toctree}
:maxdepth: 2

getting-started/index
tutorials/index
howto/index
theory/index
reference/index
migration/index
```
