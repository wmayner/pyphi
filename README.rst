.. image:: http://wmayner.github.io/pyphi/_static/pyphi-icon-and-text-380x90.png
    :target: http://wmayner.github.io/pyphi/
    :alt: PyPhi logo

|

.. image:: https://zenodo.org/badge/4651/wmayner/pyphi.svg
    :target: https://zenodo.org/badge/latestdoi/4651/wmayner/pyphi
    :alt: Zenodo DOI badge

.. image:: https://travis-ci.org/wmayner/pyphi.svg?branch=master
    :target: https://travis-ci.org/wmayner/pyphi
    :alt: Travis build badge

.. image:: https://coveralls.io/repos/wmayner/pyphi/badge.svg?branch=master
    :target: https://coveralls.io/github/wmayner/pyphi
    :alt: Coveralls.io badge

PyPhi is a Python library for computing integrated information (|phi|), and the
associated quantities and objects.

If you use this code, please cite both this repository and the `IIT 3.0 paper
<http://dx.doi.org/10.1371/journal.pcbi.1003588>`_.

To cite this code, use the Zenodo DOI for the verison you used. The latest one
is `available here <https://zenodo.org/badge/latestdoi/4651/wmayner/pyphi>`_.
For example::

    Will Mayner et al.. (2015). pyphi: 0.7.0. Zenodo. 10.5281/zenodo.17498

Or in BibTeX::

    @misc{will_mayner_2015_17498,
      author       = {Will Mayner and
                      William Marshall},
      title        = {pyphi: 0.7.0},
      month        = May,
      year         = 2015,
      doi          = {10.5281/zenodo.17498},
      url          = {http://dx.doi.org/10.5281/zenodo.17498}
    }

(Just make sure to use the version number and DOI for the version you actually
used.)


Usage, Examples, and API documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `documentation for the latest release
<https://pythonhosted.org/pyphi>`_, or the `documentation for the latest
development version <https://wmayner.github.io/pyphi>`_.

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
operating systems. Windows is not supported, though it might work on with minor
modifications. If you do get it to work, a writeup of the steps would be much
appreciated!


Detailed installation guide for Mac OS X
````````````````````````````````````````

`See here <https://github.com/wmayner/pyphi/blob/develop/INSTALLATION.md>`_.


Optional: caching with MongoDb
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


Optional: caching with Redis
`````````````````````````````

PyPhi can also use Redis as a fast in-memory global LRU cache to store Mice
objects, reducing the memory load on PyPhi processes.

`Install Redis <http://redis.io/download>`_. The `redis.conf` file provided
with PyPhi includes the minimum settings needed to run Redis as an LRU cache:

.. code:: bash

    redis-server /path/to/pyphi/redis.conf

Once the server is running you can enable Redis caching by setting
``REDIS_CACHE: true`` in your ``pyphi_config.yml``.

**Note:** PyPhi currently flushes the connected Redis database at the start of
every execution. If you are running Redis for another application be sure PyPhi
connects to its own Redis server.


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
