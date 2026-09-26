<p>
  <a href="https://pyphi.readthedocs.io/">
    <img alt="PyPhi logo" src="https://github.com/wmayner/pyphi/raw/main/docs/_static/pyphi-logo-text-776x196.png" height="90px" width="380px" style="max-width:100%">
  </a>
</p>

[![PyPI version](https://img.shields.io/pypi/v/pyphi?style=flat-square)](https://pypi.org/project/pyphi/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyphi?style=flat-square)](https://pypi.org/project/pyphi/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://img.shields.io/github/actions/workflow/status/wmayner/pyphi/test.yml?branch=main&style=flat-square&label=tests)](https://github.com/wmayner/pyphi/actions/workflows/test.yml)
[![Build](https://img.shields.io/github/actions/workflow/status/wmayner/pyphi/build.yml?branch=main&style=flat-square&label=build)](https://github.com/wmayner/pyphi/actions/workflows/build.yml)
[![Coverage](https://img.shields.io/codecov/c/github/wmayner/pyphi/main?style=flat-square)](https://codecov.io/gh/wmayner/pyphi)
[![Documentation](https://img.shields.io/readthedocs/pyphi/stable?style=flat-square)](https://pyphi.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.1371%2Fjournal.pcbi.1006343-blue?style=flat-square)](https://doi.org/10.1371/journal.pcbi.1006343)

PyPhi is a Python platform for research in [Integrated Information Theory
(IIT)](https://www.iit.wiki). Its core task is computing
**Φ** (integrated information) and the **cause–effect structure** a system
specifies; around that it provides a broad toolkit for the analyses IIT
research needs (see [Beyond Φ](#beyond-φ)).

Given a **substrate** — a network of interacting units defined by its
transition probabilities — and a **state**, PyPhi computes:

- **φ_s**, the system integrated information — whether a set of units exists
  as one integrated whole — by finding the partition that makes the least
  difference;
- the **cause–effect structure** (**Φ-structure**): the **distinctions**
  (irreducible mechanisms) a system specifies and the **relations** that bind
  them, whose total is **Φ**, the structure integrated information.

It implements **IIT 4.0** ([Albantakis et al., 2023](https://doi.org/10.1371/journal.pcbi.1011465);
[Mayner, Marshall & Tononi, 2026](https://doi.org/10.3390/e28040410)).

## Example

```python
import pyphi

# The example system from the IIT 4.0 paper (Fig. 1A).
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)

# Analyze the candidate system {A, B} in that state.
analysis = pyphi.analyze(substrate, state, subset=(0, 1))

print(analysis.phi)      # system integrated information, φ_s ≈ 0.04
print(analysis.big_phi)  # structure integrated information, Φ ≈ 1.56
```

The result carries the full Φ-structure — its distinctions, relations, and the
minimum-information partition. See the
[documentation](https://pyphi.readthedocs.io/) for a complete walkthrough.

## Beyond Φ

Around the core Φ and cause–effect-structure computations, PyPhi is a toolkit
for IIT research:

- **Actual causation** — which specific past events actually caused a given
  present event, and which effects it will actually cause (Albantakis et al.,
  2019).
- **Matching and perception** — quantify how well a system's cause–effect
  structure matches the causal structure of its environment, the basis of
  perception and intrinsic meaning in IIT (Mayner et al., 2024).
- **Macro and micro scales** — coarse-grain or black-box a substrate to
  analyze integrated information at different spatial and temporal scales.
- **Analytical bounds** — bound Φ and its components from above without the
  full combinatorial computation (Zaeemzadeh & Tononi, 2024).
- **Parameter sweeps** — evaluate many substrates, states, or configurations in
  one call, with optional parallelism.
- **Estimating substrates from data** — infer a substrate, with epistemic
  uncertainty, from observed state transitions.
- **Simulating dynamics** — settle a substrate to its most probable next states
  or sample stochastic trajectories.
- **Substrate generation** — build substrates from a library of unit
  mechanisms, weight matrices, or Ising models.
- **Visualization** — plot connectivity, repertoires, and Φ-structures
  (requires the `visualize` extra).
- **Export and interop** — export results to pandas DataFrames or xarray, and
  substrates to networkx, GraphML, or a two-timeslice dynamic Bayesian network.
- **Saving and loading** — persist any result to disk (JSON, transparently
  gzipped) and reload it later.

> **Release status.** The current release on PyPI is the **2.0** line, which
> implements IIT 4.0. Upgrading from 1.x involves breaking changes; see
> [What's new in 2.0](https://pyphi.readthedocs.io/en/stable/whats-new-in-2.0.html).
> The **1.x** line (IIT 3.0) remains available: install `"pyphi<2"`.

## Documentation

- [Getting started](https://pyphi.readthedocs.io/en/stable/getting-started/index.html):
  install and a first computation in ten minutes
- [Documentation for the latest stable release](https://pyphi.readthedocs.io/en/stable/)
- [Documentation for the latest development version](https://pyphi.readthedocs.io/en/latest/)
- Documentation for any object is also available in the interpreter with the
  `help` function.

### For AI assistants

[IIT Expert](https://learniit.org), a work in progress, helps an assistant
answer questions about the theory from IIT's primary literature, citing where
each claim comes from. In Claude Code:

```
claude plugin marketplace add wmayner/iit-expert-plugin
claude plugin install iit-expert@iit-expert
```

[Use PyPhi with an AI assistant](https://pyphi.readthedocs.io/en/latest/howto/ai-assistants.html)
covers Codex, Cursor, claude.ai and Claude Desktop.

PyPhi ships an [MCP server](https://pyphi.readthedocs.io/en/stable/howto/mcp-server.html)
(`pip install "pyphi[mcp]"`, then `pyphi-mcp install`) that gives an assistant
tools for building substrates, estimating cost, and running analyses, along with
the theory reference it needs to interpret results. The documentation site is
also readable without it:

- <https://pyphi.readthedocs.io/en/stable/llms.txt> lists every page with a
  one-line summary; `llms-full.txt` at the same location is the whole site as
  one markdown file.
- Every page's MyST source is served under `_sources/`, for example
  <https://pyphi.readthedocs.io/en/stable/_sources/theory/conditional-independence.md.txt>.
- <https://pyphi.readthedocs.io/en/stable/objects.inv> is the intersphinx
  inventory that maps API names to their pages.

## Installation

PyPhi requires **Python 3.13+**.

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install the current release. Version 2.0 is out as a release candidate, so
`--pre` is needed until the final release; without it, pip installs 1.2
instead.

```bash
uv pip install --pre pyphi
```

To install the latest development version from GitHub instead:

```bash
uv pip install "git+https://github.com/wmayner/pyphi@main"
```

Optional features are available as extras: `visualize` (plotting), `caching`
(Redis-backed caches), `emd` (earth-mover's-distance measures), `xarray`
(labeled array export), `cluster` (Dask-based cluster execution), and `mcp`
(the MCP server for AI assistants). Install one or more with, e.g.:

```bash
uv pip install --pre "pyphi[visualize,emd]"
```

### Using pip

```bash
python -m pip install --pre pyphi                                 # 2.0 release candidate
python -m pip install "git+https://github.com/wmayner/pyphi@main" # development version
```

## Contributing

To help develop PyPhi, fork the project on GitHub, clone your fork, and install
the runtime extras plus the development tooling with uv:

```bash
git clone https://github.com/YOUR_USERNAME/pyphi.git
cd pyphi
uv sync --all-extras --group dev
```

Common development tasks are defined in the `justfile` (install
[just](https://github.com/casey/just)):

```bash
just test    # run the test suite
just bench   # run the performance benchmarks
just docs    # build the HTML documentation
```

The [contributing page](https://pyphi.readthedocs.io/en/stable/contributing.html)
covers the test suite, changelog fragments, and the documentation build.

## User group

For discussion about the software or integrated information theory in general,
join the [pyphi-users
group](https://groups.google.com/forum/#!forum/pyphi-users).

For bug reports and feature requests, use the [issues
page](https://github.com/wmayner/pyphi/issues).

## Citation

If you use this software in your research, please cite the papers:

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018).
[PyPhi: A toolbox for integrated information
theory](https://doi.org/10.1371/journal.pcbi.1006343). PLOS Computational
Biology 14(7): e1006343.

```
@article{mayner2018pyphi,
  title={PyPhi: A toolbox for integrated information theory},
  author={Mayner, William GP and Marshall, William and Albantakis, Larissa and Findlay, Graham and Marchman, Robert and Tononi, Giulio},
  journal={PLoS Computational Biology},
  volume={14},
  number={7},
  pages={e1006343},
  year={2018},
  publisher={Public Library of Science},
  doi={10.1371/journal.pcbi.1006343},
  url={https://doi.org/10.1371/journal.pcbi.1006343}
}
```

For the theory PyPhi 2.0 implements, cite the IIT 4.0 papers:

Albantakis L, Barbosa L, Findlay G, Grasso M, Haun AM, Marshall W, Mayner WGP,
Zaeemzadeh A, Boly M, Juel BE, Sasai S, Fujii K, David I, Hendren J, Lang JP,
Tononi G. (2023). [Integrated information theory (IIT) 4.0: Formulating the
properties of phenomenal existence in physical
terms](https://doi.org/10.1371/journal.pcbi.1011465). PLOS Computational
Biology 19(10): e1011465.

```
@article{albantakis2023iit4,
  title={Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms},
  author={Albantakis, Larissa and Barbosa, Leonardo and Findlay, Graham and Grasso, Matteo and Haun, Andrew M and Marshall, William and Mayner, William GP and Zaeemzadeh, Alireza and Boly, Melanie and Juel, Bj{\o}rn E and Sasai, Shuntaro and Fujii, Keiko and David, Isaac and Hendren, Jeremiah and Lang, Jonathan P and Tononi, Giulio},
  journal={PLoS Computational Biology},
  volume={19},
  number={10},
  pages={e1011465},
  year={2023},
  publisher={Public Library of Science},
  doi={10.1371/journal.pcbi.1011465},
  url={https://doi.org/10.1371/journal.pcbi.1011465}
}
```

Mayner WGP, Marshall W, Tononi G. (2026). [Intrinsic cause–effect power: the
tradeoff between differentiation and
specification](https://doi.org/10.3390/e28040410). Entropy 28(4): 410.

```
@article{mayner2026intrinsic,
  title={Intrinsic cause--effect power: the tradeoff between differentiation and specification},
  author={Mayner, William GP and Marshall, William and Tononi, Giulio},
  journal={Entropy},
  volume={28},
  number={4},
  pages={410},
  year={2026},
  publisher={MDPI},
  doi={10.3390/e28040410},
  url={https://doi.org/10.3390/e28040410}
}
```

For results computed under the earlier IIT 3.0, cite:

Oizumi M, Albantakis L, Tononi G. (2014). [From the Phenomenology to the
Mechanisms of Consciousness: Integrated Information Theory
3.0](https://doi.org/10.1371/journal.pcbi.1003588). PLOS Computational Biology
10(5): e1003588.

```
@article{oizumi2014iit3,
  title={From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0},
  author={Oizumi, Masafumi and Albantakis, Larissa and Tononi, Giulio},
  journal={PLoS Computational Biology},
  volume={10},
  number={5},
  pages={e1003588},
  year={2014},
  publisher={Public Library of Science},
  doi={10.1371/journal.pcbi.1003588},
  url={https://doi.org/10.1371/journal.pcbi.1003588}
}
```

## Acknowledgements

The initial version of this project was inspired by a [previous
project](https://github.com/albantakis/iit) written in MATLAB by L. Albantakis,
M. Oizumi, A. Hashmi, A. Nere, U. Olcese, P. Rana, and B. Shababo.

## Correspondence

Correspondence regarding the PyPhi software should be directed to Will Mayner,
at [<mayner@wisc.edu>](mailto:mayner@wisc.edu).
