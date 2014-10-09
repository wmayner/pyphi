.. Zenodo DOI sticker:
.. image:: https://zenodo.org/badge/4651/wmayner/cyphi.png 
    :target: http://dx.doi.org/10.5281/zenodo.11998
.. Travis build sticker:
.. image:: https://travis-ci.org/wmayner/cyphi.svg
    :target: https://travis-ci.org/wmayner/cyphi

***********************
CyPhi: |phi| for Python
***********************

CyPhi is a Python library for computing integrated information (|phi|), and
the associated quantities and objects.

If you use this code, please cite both this repository (see DOI badge above)
and the IIT 3.0 paper (see last section below).


Usage, Examples, and API documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `documentation for the latest release
<https://pythonhosted.org/cyphi>`_, or the `documentation for the latest
development version <https://wmayner.github.io/cyphi>`_.

The documentation is also available within the Python interpreter with the
``help`` function.


Installation
~~~~~~~~~~~~

To install the latest release:

.. code:: bash

    pip install cyphi

To install the latest development version:

.. code:: bash

    pip install "git+https://github.com/wmayner/cyphi@develop#egg=cyphi"


Result caching and MongoDB
``````````````````````````
CyPhi stores the results of |Phi| calculations as they're computed in order to
avoid expensive re-computation. These results can be stored locally on the
filesystem (the default setting), or in a full-fledged database. Using the
default caching system is easier and works out of the box, but using a database
is more robust.

To use the database-backed caching system, you must install `MongoDB
<http://www.mongodb.org/>`_. Please see their `installation guide
<http://docs.mongodb.org/manual/installation/>`_ for instructions.

Once you have MongoDB installed, use ``mongod`` to start the MongoDB server.
Make sure the ``mongod`` configuration matches the CyPhi's database
configuration settings in ``cyphi_config.yml`` (see the `configuration section
<https://pythonhosted.org/cyphi/index.html#configuration>`_ of CyPhi's
documentation).

You can also check out MongoDB's `Getting Started guide
<http://docs.mongodb.org/manual/tutorial/getting-started/>`_ or the full
`manual <http://docs.mongodb.org/manual/>`_.


Contributing
~~~~~~~~~~~~

To help develop CyPhi, fork the project on GitHub and install the requirements
with ``pip install -r requirements.txt``.

Development workflow
````````````````````

``Gruntfile.js`` defines some tasks to help with development. These are run
with `Grunt.js <http:gruntjs.com>`_.

To get ``grunt``, first install `Node.js <http://nodejs.org/>`_. Then, within
the ``cyphi`` directory, run ``npm install`` to install the local ``npm``
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
written in Matlab by L. Albantakis, A. Hashmi, A. Nere, U. Olces, P. Rana, and
B. Shababo.

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
