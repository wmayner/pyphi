***********************
CyPhi: |phi| for Python
***********************

CyPhi is a Python library for computing integrated information (|phi|), and
the associated quantities and objects.


Installation
~~~~~~~~~~~~

To install the latest release:

.. code:: bash

    pip install cyphi

To install the latest development version:

.. code:: bash

    pip install "git+https://github.com/wmayner/cyphi@develop#egg=pyemd"


Usage
~~~~~

.. code:: bash

    >>> import cyphi

TODO


Check out the `API <https://readthedocs.org/projects/cyphi>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Limitations and Caveats
~~~~~~~~~~~~~~~~~~~~~~~

TODO


Contributing
~~~~~~~~~~~~

To help develop CyPhi, fork the project on GitHub and install the requirements
with ``pip install -r requirements.txt``.

Development workflow
````````````````````

``Gruntfile.js`` defines some tasks to help with development. These are run
with `Grunt.js <http:gruntjs.com>`_.

To get ``grunt``, first install `Node.js <http://nodejs.org/>`_. Then, within the ``cyphi``
directory, run ``npm install``. Now you should be able to run tasks with
``grunt``, *e.g.*

.. code:: bash

    grunt tests

which will run the unit tests every time you change the source code. Similarly,

.. code:: bash

    grunt docs

will rebuild the HTML documentation on every change.

At some point I'll try to use a Makefile instead, since many more people have
access to ``make``.


Credits
~~~~~~~

This code is based on a `previous project <https://github.com/albantakis/iit>`_
written in Matlab by B. Shababo, A. Nere, A. Hashmi, U. Olcese, P. Rana, and L.
Albantakis.

Please cite these papers if you use this code:
``````````````````````````````````````````````

TODO

.. code:: latex

    @INPROCEEDINGS{citationname,
      title={},
      author={},
      booktitle={},
      pages={},
      year={},
      month={},
      publisher={}
    }


.. |phi| unicode:: U+1D6BD .. mathematical bold capital phi
.. |small_phi| unicode:: U+1D6D7 .. mathematical bold phi
