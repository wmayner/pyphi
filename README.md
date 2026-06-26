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

PyPhi is a Python library for computing integrated information (𝚽), and the
associated quantities and objects.

**If you use this code, please cite the paper:**

---

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018)
[PyPhi: A toolbox for integrated information
theory](https://doi.org/10.1371/journal.pcbi.1006343). PLOS Computational
Biology 14(7): e1006343. <https://doi.org/10.1371/journal.pcbi.1006343>

---

An [illustrated tutorial on how Φ is calculated](https://doi.org/10.1371/journal.pcbi.1006343.s001) is available as a supplement to the paper.


## Usage, Examples, and API documentation

- [Documentation for the latest stable
  release](http://pyphi.readthedocs.io/en/stable/)
- [Documentation for the latest (potentially unstable) development
  version](http://pyphi.readthedocs.io/en/latest/).
- Documentation is also available within the Python interpreter with the `help`
  function.


## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it with:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install PyPhi:

```bash
uv pip install pyphi
```

### Using pip

Set up a Python 3.13+ virtual environment and install with:

```bash
pip install pyphi
```

To install the latest development version, which is a work in progress and may
have bugs, run:

```bash
pip install "git+https://github.com/wmayner/pyphi@main#egg=pyphi"
```

### Legacy: Conda (Deprecated)

**Note:** The conda package is deprecated. Please use uv or pip instead.

If you encounter issues on Windows with older systems, you can use the [Anaconda
Python](https://www.anaconda.com/what-is-anaconda/) distribution and
[install PyPhi with conda](https://anaconda.org/wmayner/pyphi):

```bash
conda install -c wmayner pyphi
```

### Detailed installation guide for Mac OS X

[See here](https://github.com/wmayner/pyphi/blob/main/INSTALLATION.rst).


## User group

For discussion about the software or integrated information theory in general,
you can join the [pyphi-users
group](https://groups.google.com/forum/#!forum/pyphi-users).

For technical issues with PyPhi or feature requests, please use the [issues
page](https://github.com/wmayner/pyphi/issues).


## Contributing

To help develop PyPhi, fork the project on GitHub and install with uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone your fork
git clone https://github.com/YOUR_USERNAME/pyphi.git
cd pyphi

# Create virtual environment and install with dev dependencies
uv venv
uv pip install -e ".[dev,parallel,visualize,graphs,emd,caching]"
```

The `Makefile` defines some tasks to help with development:

```bash
make test
```

runs the unit tests every time you change the source code.

```bash
make benchmark
```

runs performance benchmarks.

```bash
make docs
```

builds the HTML documentation.

### Developing on Linux

With uv, all dependencies including compiled packages are installed automatically
from pre-built wheels. If you need system headers for development:

```bash
sudo apt-get install python3-dev
```

### Developing on Windows

All dependencies now have pre-built wheels for Windows, so installation should work
seamlessly with uv:

```bash
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install PyPhi for development
uv venv
uv pip install -e ".[dev]"
```

If you encounter issues, the legacy conda approach is still available (see Installation section above).

## Credit

### Please cite these papers if you use this code:

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018)
[PyPhi: A toolbox for integrated information
theory](https://doi.org/10.1371/journal.pcbi.1006343). PLOS Computational
Biology 14(7): e1006343. <https://doi.org/10.1371/journal.pcbi.1006343>

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

Albantakis L, Oizumi M, Tononi G (2014). [From the Phenomenology to the
Mechanisms of Consciousness: Integrated Information Theory
3.0](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1003588).
PLoS Comput Biol 10(5): e1003588. doi: 10.1371/journal.pcbi.1003588.

```
@article{iit3,
    title={From the Phenomenology to the Mechanisms of Consciousness:
    author={Albantakis, Larissa AND Oizumi, Masafumi AND Tononi, Giulio},
    Integrated Information Theory 3.0},
    journal={PLoS Comput Biol},
    publisher={Public Library of Science},
    year={2014},
    month={05},
    volume={10},
    pages={e1003588},
    number={5},
    doi={10.1371/journal.pcbi.1003588},
    url={http://dx.doi.org/10.1371%2Fjournal.pcbi.1003588}
}
```

This project is inspired by a [previous
project](https://github.com/albantakis/iit) written in Matlab by L. Albantakis,
M. Oizumi, A. Hashmi, A. Nere, U. Olcese, P. Rana, and B. Shababo.

Correspondence regarding this code and the PyPhi paper should be directed to
Will Mayner, at [<mayner@wisc.edu>](mailto:mayner@wisc.edu). Correspondence
regarding the Matlab code and the IIT 3.0 paper should be directed to Larissa
Albantakis, PhD, at [<albantakis@wisc.edu>](mailto:albantakis@wisc.edu).
