<p>
  <a href="http://pyphi.readthedocs.io/">
    <img alt="PyPhi logo" src="https://github.com/wmayner/pyphi/raw/develop/docs/_static/pyphi-logo-text-760x180.png" height="90px" width="380px" style="max-width:100%">
  </a>
</p>

[![Documentation badge](https://readthedocs.org/projects/pyphi/badge/?style=flat-square&maxAge=600)](https://pyphi.readthedocs.io/)
[![Travis build badge](https://img.shields.io/travis/wmayner/pyphi.svg?style=flat-square&maxAge=600)](https://travis-ci.org/wmayner/pyphi)
[![Coveralls.io badge](https://img.shields.io/coveralls/wmayner/pyphi/develop.svg?style=flat-square&maxAge=600)](https://coveralls.io/github/wmayner/pyphi?branch=develop)
[![License badge](https://img.shields.io/github/license/wmayner/pyphi.svg?style=flat-square&maxAge=86400)](https://github.com/wmayner/pyphi/blob/master/LICENSE.md)
[![Python versions badge](https://img.shields.io/pypi/pyversions/pyphi.svg?style=flat-square&maxAge=86400)](https://wiki.python.org/moin/Python2orPython3)

PyPhi is a Python library for computing integrated information (𝚽), and the
associated quantities and objects.

**If you use this code, please cite the manuscript:**

---

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G (2017).
[PyPhi: A toolbox for integrated
information](https://arxiv.org/abs/1712.09644). arXiv:1712.09644 \[q-bio.NC\].

---

The manuscript is available at <https://arxiv.org/abs/1712.09644>.


## Usage, Examples, and API documentation

- [Documentation for the latest stable
  release](http://pyphi.readthedocs.io/en/stable/)
- [Documentation for the latest (potentially unstable) development
  version](http://pyphi.readthedocs.io/en/latest/).
- Documentation is also available within the Python interpreter with the `help`
  function.


## Installation

Set up a Python 3 virtual environment and install with

```bash
pip install pyphi
```

To install the latest development version, which is a work in progress and may
have bugs, run:

```bash
pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"
```

**Note:** this software is only supported on Linux and macOS. However, if you
use Windows, you can run it by using the [Anaconda
Python](https://www.anaconda.com/what-is-anaconda/) distribution and
[installing PyPhi with conda](https://anaconda.org/wmayner/pyphi):

```bash
conda install -c wmayner pyphi
```

### Detailed installation guide for Mac OS X

[See here](https://github.com/wmayner/pyphi/blob/develop/INSTALLATION.rst).


## User group

For discussion about the software or integrated information theory in general,
you can join the [pyphi-users
group](https://groups.google.com/forum/#!forum/pyphi-users).

For technical issues with PyPhi or feature requests, please use the [issues
page](https://github.com/wmayner/pyphi/issues).


## Contributing

To help develop PyPhi, fork the project on GitHub and install the requirements
with

```bash
pip install -r requirements.txt
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

Make sure you install the C headers for Python 3, SciPy, and NumPy
before installing the requirements:

```bash
sudo apt-get install python3-dev python3-scipy python3-numpy
```


## Credit

### Please cite these papers if you use this code:

Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G (2017).
[PyPhi: A toolbox for integrated
information](https://arxiv.org/abs/1712.09644). arXiv:1712.09644 \[q-bio.NC\].

```
@article{mayner2017pyphi,
  title={PyPhi: A toolbox for integrated information},
  author={Mayner, William, Gerald Paul AND Marshall, William AND 
          Albantakis, Larissa AND Findlay, Graham AND 
          Marchman, Robert AND Tononi, Giulio},
  journal={arXiv:1712.09644 [q-bio.NC]},
  year={2017},
  month={12},
  url={https://arxiv.org/abs/1712.09644}
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
M. Oizumi, A. Hashmi, A. Nere, U. Olces, P. Rana, and B. Shababo.

Correspondence regarding this code and the PyPhi paper should be directed to
Will Mayner, at [<mayner@wisc.edu>](mailto:mayner@wisc.edu). Correspondence
regarding the Matlab code and the IIT 3.0 paper should be directed to Larissa
Albantakis, PhD, at [<albantakis@wisc.edu>](mailto:albantakis@wisc.edu).
