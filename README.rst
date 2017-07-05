.. image:: http://wmayner.github.io/pyphi/_static/pyphi-icon-and-text-380x90.png
    :target: http://pyphi.readthedocs.io/en/latest/
    :alt: PyPhi logo

|

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.55692.svg
    :target: http://dx.doi.org/10.5281/zenodo.55692
    :alt: Zenodo DOI badge

.. image:: https://readthedocs.org/projects/pyphi/badge/?version=latest
    :target: https://pyphi.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation badge

.. image:: https://img.shields.io/travis/wmayner/pyphi.svg?maxAge=601
    :target: https://travis-ci.org/wmayner/pyphi
    :alt: Travis build badge

.. image:: https://img.shields.io/coveralls/wmayner/pyphi/master.svg?maxAge=600
    :target: https://coveralls.io/github/wmayner/pyphi
    :alt: Coveralls.io badge

.. image:: http://img.shields.io/badge/Python%203%20-compatible-brightgreen.svg
    :target: https://wiki.python.org/moin/Python2orPython3
    :alt: Python 3 compatible

|

PyPhi is a Python library for computing integrated information (|phi|), and the
associated quantities and objects.

**If you use this code, please cite it, as well as the** `IIT 3.0 paper
<http://dx.doi.org/10.1371/journal.pcbi.1003588>`_.

To cite the code, use the Zenodo DOI for the verison you used. The latest one
is `10.5281/zenodo.55692 <http://dx.doi.org/10.5281/zenodo.55692>`_.
For example::

    Mayner, William GP et al. (2016). PyPhi: 0.8.1. Zenodo. 10.5281/zenodo.595866

Or in BibTeX::

    @misc{pyphi,
      author = {Mayner, William Gerald Paul and
                Marshall, William and
                Marchman, Bo},
      title  = {PyPhi: 0.8.1},
      month  = Feb,
      year   = 2016,
      doi    = {10.5281/zenodo.55692},
      url    = {http://dx.doi.org/10.5281/zenodo.55692}
    }

(Just make sure to use the version number, DOI, and URL for the version you
actually used.)


Usage, Examples, and API documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `documentation for the latest stable release
<http://pyphi.readthedocs.io/en/stable/>`_, or the `documentation for the
latest (potentially unstable) development version
<http://pyphi.readthedocs.io/en/latest/>`_.

The documentation is also available within the Python interpreter with the
``help`` function.


Installation
~~~~~~~~~~~~

Set up a Python 3 virtual environment and install with

.. code:: bash

    pip install pyphi

To install the latest development version, which is a work in progress and may
have bugs, run:

.. code:: bash

    pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"

**Note:** this software has only been tested on the Mac OS X and Linux
operating systems. Windows is not supported, though it might work with minor
modifications. If you do get it to work, a writeup of the steps would be much
appreciated!


Detailed installation guide for Mac OS X
````````````````````````````````````````

`See here <https://github.com/wmayner/pyphi/blob/develop/INSTALLATION.md>`_.


Discussion
~~~~~~~~~~

For technical issues with PyPhi or feature requests, please use the `issues
page <https://github.com/wmayner/pyphi/issues>`_.

For discussion about the software or integrated information theory in general,
you can join the `PyPhi users group
<https://groups.google.com/forum/#!forum/pyphi-users>`_.


Contributing
~~~~~~~~~~~~

To help develop PyPhi, fork the project on GitHub and install the requirements
with ``pip install -r requirements.txt``.


Development workflow
````````````````````

The ``Makefile`` defines some tasks to help with development:

.. code:: bash

    make test

runs the unit tests every time you change the source code.

.. code:: bash

    make benchmark

runs performance benchmarks.

.. code:: bash

    make docs

builds the HTML documentation.

``Gruntfile.js`` defines similar tasks. To get grunt, first install
`Node.js <http://nodejs.org/>`_. Then, within the ``pyphi`` directory, run
``npm install`` to install the local npm dependencies, then run
``sudo npm install -g grunt grunt-cli`` to install the ``grunt`` command to your
system. You should now be able to run tasks with ``grunt``.


Developing on Linux
```````````````````

Make sure you install the Python 3 C headers before installing the
requirements:

.. code:: bash

    sudo apt-get install python3-dev python3-scipy python3-numpy


Credits
~~~~~~~

This code is based on a `previous project <https://github.com/albantakis/iit>`_
written in Matlab by L. Albantakis, M. Oizumi, A. Hashmi, A. Nere, U. Olces, P.
Rana, and B. Shababo.

Correspondence regarding the Matlab code and the IIT 3.0 paper (below) should
be directed to Larissa Albantakis, PhD, at `albantakis@wisc.edu
<mailto:albantakis@wisc.edu>`_.

Please cite this paper if you use this code:
````````````````````````````````````````````

Albantakis L, Oizumi M, Tononi G (2014) `From the Phenomenology to the
Mechanisms of Consciousness: Integrated Information Theory 3.0
<http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1003588>`_.
PLoS Comput Biol 10(5): e1003588. doi: 10.1371/journal.pcbi.1003588


.. code:: latex

    @article{iit3,
        author = {Albantakis, , Larissa AND Oizumi, , Masafumi AND Tononi, ,
            Giulio},
        journal = {PLoS Comput Biol},
        publisher = {Public Library of Science},
        title = {From the Phenomenology to the Mechanisms of Consciousness:
            Integrated Information Theory 3.0},
        year = {2014},
        month = {05},
        volume = {10},
        url = {http://dx.doi.org/10.1371%2Fjournal.pcbi.1003588},
        pages = {e1003588},
        number = {5},
        doi = {10.1371/journal.pcbi.1003588}
    }


.. |phi| unicode:: U+1D6BD .. mathematical bold capital phi
.. |small_phi| unicode:: U+1D6D7 .. mathematical bold phi
