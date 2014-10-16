.. Zenodo DOI badge
.. image:: https://zenodo.org/badge/4651/wmayner/pyphi.png 
    :target: http://dx.doi.org/10.5281/zenodo.12194
.. Travis build badge
.. image:: http://img.shields.io/travis/wmayner/pyphi/develop.svg
    :target: https://travis-ci.org/wmayner/pyphi
.. Coveralls.io badge
.. image:: http://img.shields.io/coveralls/wmayner/pyphi/develop.svg
    :target: https://coveralls.io/r/wmayner/pyphi?branch=develop

*************************
PyPhi: |phi| for Python 3
*************************

PyPhi is a Python 3 library for computing integrated information (|phi|), and
the associated quantities and objects.

If you use this code, please cite both this repository (DOI
`10.5281/zenodo.12194 <http://dx.doi.org/10.5281/zenodo.12194>`_) and the IIT
3.0 paper (DOI `10.1371/journal.pcbi.1003588
<http://dx.doi.org/10.1371/journal.pcbi.1003588>`_).


Usage, Examples, and API documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `documentation for the latest release
<https://pythonhosted.org/pyphi>`_, or the `documentation for the latest
development version <https://wmayner.github.io/pyphi>`_.

The documentation is also available within the Python interpreter with the
``help`` function.


Installation
~~~~~~~~~~~~

Set up a Python 3 virtual environment and install using ``pip install pyphi``.


Detailed guide for those unfamiliar with Python
```````````````````````````````````````````````

This is a Python 3 project, so in order to use it you must install `Python
3 <https://www.python.org/downloads/>`_.

Once you've installed Python 3, it is highly recommended to set up a **virtual
environment** in which to install PyPhi. Virtual environments allow different
projects to isolate their dependencies from one another, so that they don't
interact in unexpected ways. They also protect your system's version of Python
from unwanted changes. Please see `this guide
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ for more
information.

To do this, you must install ``virtualenv`` and ``virtualenvwrapper``, a `tool
for manipulating virtual environments
<http://virtualenvwrapper.readthedocs.org/en/latest/>`_. Both of those tools
are available on `PyPI <https://pypi.python.org/pypi>`_, the Python package
index, and can be installed with ``pip``, the command-line utility for
installing and managing Python packages (``pip`` is installed automatically
with Python):

.. code:: bash

    pip install virtualenv virtualenvwrapper

Then use ``virtualenvwrapper`` to create a Python 3 virtual environment, like
so:

.. code:: bash

    mkvirtualenv -p `which python3` <name_of_your_project>

The ``-p `which python3``` option ensures that when the virtual environment is
activated, the commands ``python`` and ``pip`` will refer to their Python 3
counterparts.

The virtual environment should have been activated automatically after creating
it. It can be manually activated with ``workon <name_of_your_project>``, and
deactivated with ``deactivate``.

Remember to activate the virtual environment *every time* you begin working on
your project. Also, note that the currently active virtual environment is *not*
associated with any particular folder; it is associated with a terminal shell.

Finally, you can install PyPhi into your new virtual environment:

.. code:: bash

    pip install pyphi

To install the latest development version (which is a work in progress and may
have bugs):

.. code:: bash

    pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"


Optional: caching with a database
`````````````````````````````````

PyPhi stores the results of |Phi| calculations as they're computed in order to
avoid expensive re-computation. These results can be stored locally on the
filesystem (the default setting), or in a full-fledged database. 

Using the default caching system is easier and works out of the box, but using
a database is more robust.

To use the database-backed caching system, you must install `MongoDB
<http://www.mongodb.org/>`_. Please see their `installation guide
<http://docs.mongodb.org/manual/installation/>`_ for instructions.

Once you have MongoDB installed, use ``mongod`` to start the MongoDB server.
Make sure the ``mongod`` configuration matches the PyPhi's database
configuration settings in ``pyphi_config.yml`` (see the `configuration section
<https://pythonhosted.org/pyphi/index.html#configuration>`_ of PyPhi's
documentation).

You can also check out MongoDB's `Getting Started guide
<http://docs.mongodb.org/manual/tutorial/getting-started/>`_ or the full
`manual <http://docs.mongodb.org/manual/>`_.


Contributing
~~~~~~~~~~~~

To help develop PyPhi, fork the project on GitHub and install the requirements
with ``pip install -r requirements.txt``.

Development workflow
````````````````````

``Gruntfile.js`` defines some tasks to help with development. These are run
with `Grunt.js <http:gruntjs.com>`_.

To get ``grunt``, first install `Node.js <http://nodejs.org/>`_. Then, within
the ``pyphi`` directory, run ``npm install`` to install the local ``npm``
dependencies, then run ``sudo npm install -g grunt grunt-cli`` to install the
``grunt`` command to your system. Now you should be able to run tasks with
``grunt``, *e.g.*

.. code:: bash

    grunt test

which will run the unit tests every time you change the source code. Similarly,

.. code:: bash

    grunt docs

will rebuild the HTML documentation on every change.

At some point I'll try to use a Makefile instead, since many more people have
access to ``make``.

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
