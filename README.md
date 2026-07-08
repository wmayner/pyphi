<p>
  <a href="http://pyphi.readthedocs.io/">
    <img alt="PyPhi logo" src="https://github.com/wmayner/pyphi/raw/main/docs/_static/pyphi-logo-text-776x196.png" height="90px" width="380px" style="max-width:100%">
  </a>
</p>

[![Tests](https://img.shields.io/github/actions/workflow/status/wmayner/pyphi/test.yml?branch=main&style=flat-square&label=tests)](https://github.com/wmayner/pyphi/actions/workflows/test.yml)
[![Build](https://img.shields.io/github/actions/workflow/status/wmayner/pyphi/build.yml?branch=main&style=flat-square&label=build)](https://github.com/wmayner/pyphi/actions/workflows/build.yml)
[![Codecov](https://img.shields.io/codecov/c/github/wmayner/pyphi/main?style=flat-square)](https://codecov.io/gh/wmayner/pyphi)
[![Documentation](https://img.shields.io/readthedocs/pyphi/stable?style=flat-square)](https://pyphi.readthedocs.io/)
[![PyPI version](https://img.shields.io/pypi/v/pyphi?style=flat-square)](https://pypi.org/project/pyphi/)
[![Python 3.13+](https://img.shields.io/pypi/pyversions/pyphi?style=flat-square)](https://pypi.org/project/pyphi/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)

PyPhi is a Python library for computing integrated information (Φ) and the
objects and quantities of [Integrated Information Theory
(IIT)](https://doi.org/10.1371/journal.pcbi.1011465). Given a **substrate** — a
network of interacting units defined by its transition probabilities — and a
**state**, PyPhi computes:

- **Φ**, the integrated information of a system, by finding the partition that
  makes the least difference;
- the **cause–effect structure** (**Φ-structure**): the **distinctions**
  (irreducible mechanisms) a system specifies and the **relations** that bind
  them;
- **actual causation**: which past events actually caused a given present
  event, and which effects it will actually cause.

It implements the current formalism, **IIT 4.0** (Albantakis et al., 2023), and
retains the earlier **IIT 3.0** formalism, selectable by configuration.

> **Release status.** The version on the `main` branch is the in-development
> **2.0** line, which implements IIT 4.0. The current release on PyPI is the
> **1.x** line (IIT 3.0). To use the IIT 4.0 code today, install from GitHub
> (see [Installation](#installation)).

## Example

```python
import pyphi

# A simple 3-node substrate (the example system from the IIT 4.0 paper).
substrate = pyphi.examples.basic_substrate()
state = (1, 0, 0)

# Analyze the substrate in that state under IIT 4.0.
analysis = pyphi.analyze(substrate, state)

print(analysis.phi)  # the system's integrated information, Φ
```

The result carries the full Φ-structure — its distinctions, relations, and the
partition that minimizes Φ. See the
[documentation](http://pyphi.readthedocs.io/) and the
[IIT 4.0 demo notebook](https://github.com/wmayner/pyphi/blob/main/docs/examples/IIT_4.0_demo.ipynb)
for a complete walkthrough.

## Documentation

- [Documentation for the latest stable release](http://pyphi.readthedocs.io/en/stable/)
- [Documentation for the latest development version](http://pyphi.readthedocs.io/en/latest/)
- Documentation for any object is also available in the interpreter with the
  `help` function.

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

Install the current PyPI release (1.x, IIT 3.0):

```bash
uv pip install pyphi
```

Install the in-development 2.0 line (IIT 4.0) from GitHub:

```bash
uv pip install "git+https://github.com/wmayner/pyphi@main"
```

Optional features are available as extras: `visualize` (plotting), `caching`
(Redis-backed caches), `emd` (earth-mover's-distance measures), and `xarray`
(labeled array export). Install one or more with, e.g.:

```bash
uv pip install "pyphi[visualize,emd]"
```

### Using pip

```bash
pip install pyphi                     # current release (1.x)
pip install "git+https://github.com/wmayner/pyphi@main"   # 2.0 (IIT 4.0)
```

### Detailed installation guide for macOS

[See here](https://github.com/wmayner/pyphi/blob/main/INSTALLATION.rst).

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

## User group

For discussion about the software or integrated information theory in general,
join the [pyphi-users
group](https://groups.google.com/forum/#!forum/pyphi-users).

For bug reports and feature requests, use the [issues
page](https://github.com/wmayner/pyphi/issues).

## Credit

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

For the theory PyPhi 2.0 implements, cite the IIT 4.0 paper:

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

For the IIT 3.0 formalism, cite:

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

This project is inspired by a [previous
project](https://github.com/albantakis/iit) written in MATLAB by L. Albantakis,
M. Oizumi, A. Hashmi, A. Nere, U. Olcese, P. Rana, and B. Shababo.

Correspondence regarding the PyPhi software should be directed to Will Mayner,
at [<mayner@wisc.edu>](mailto:mayner@wisc.edu).
