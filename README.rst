.. image:: https://zenodo.org/badge/4651/wmayner/pyphi.png
    :target: http://dx.doi.org/10.5281/zenodo.12194
    :alt: Zenodo DOI

.. image:: https://travis-ci.org/wmayner/pyphi.svg?branch=develop
    :target: https://travis-ci.org/wmayner/pyphi
    :alt: Travis build

.. image:: https://coveralls.io/repos/wmayner/pyphi/badge.png?branch=develop
    :target: https://coveralls.io/r/wmayner/pyphi?branch=develop
    :alt: Coveralls.io

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

Set up a Python 3 virtual environment and install with

.. code:: bash

    pip install pyphi

To install the latest development version, which is a work in progress and may
have bugs, instead run:

.. code:: bash

    pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"

**Note:** this software has only been tested on the Mac OS X and Linux
operating systems. Windows is not supported, though it might work on with minor
modifications. If you do get it to work, a writeup of the steps would be much
appreciated!


Detailed installation guide for Mac OS X
````````````````````````````````````````

This is a step-by-step guide intended for those unfamiliar with Python or the
command-line (*a.k.a.* the “shell”).

A shell can be opened by opening a new tab in the Terminal app (located in
Utilities). Text that is ``formatted like code`` is meant to be copied and
pasted into the terminal (hit the Enter key to run the command).

The fist step is to install the versions of Python that we need. The most
convenient way of doing this is to use the OS X package manager `Homebrew
<http://brew.sh/>`_. Install Homebrew by running this command:

.. code:: bash

   ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Now you should have access to the ``brew`` command. First, we need to install
Python 2 and 3. Using these so-called “brewed” Python versions, rather than the
version of Python that comes with your computer, will protect your computer's
Python version from unwanted changes that could interfere with other
applications.

.. code:: bash

   brew install python python3

Then we need to ensure that the terminal “knows about” the newly-installed
Python versions:

.. code:: bash

    brew link --overwrite python
    brew link --overwrite python3

Now that we're using our shiny new Python versions, it is highly recommended to
set up a **virtual environment** in which to install PyPhi. Virtual
environments allow different projects to isolate their dependencies from one
another, so that they don't interact in unexpected ways. . Please see `this
guide <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ for more
information.

To do this, you must install ``virtualenv`` and ``virtualenvwrapper``, a `tool
for manipulating virtual environments
<http://virtualenvwrapper.readthedocs.org/en/latest/>`_. Both of those tools
are available on `PyPI <https://pypi.python.org/pypi>`_, the Python package
index, and can be installed with ``pip``, the command-line utility for
installing and managing Python packages (``pip`` was installed automatically
with the brewed Python):

.. code:: bash

    pip install virtualenvwrapper

Now we need to edit your shell startup file. This is a file that runs automatically every time you open a new shell (a new window or tab in the Terminal app). This file should be in your home directory, though it will be invisible in the Finder because the filename is preceded by a period. On most Macs it is called ``.bash_profile``. You can open this in a text editor by running this command:

.. code:: bash

    open -a TextEdit ~/.bash_profile

If this doesn't work because the file doesn't exist, then run ``touch
~/.bash_profile`` first.

Now, you'll add three lines to the shell startup file. These lines will set the
location where the virtual environments will live, the location of your
development project directories, and the location of the script installed with
this package, respectively. **Note:** The location of the script can be found
by running ``which virtualenvwrapper.sh``.

The filepath after the equals sign on second line will different for everyone,
but here is an example:

.. code:: bash

    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/dev
    source /usr/local/bin/virtualenvwrapper.sh

After editing the startup file and saving it, open a new terminal shell by
opening a new tab or window (or just reload the startup file by running
``source ~/.bash_profile``).

Now that ``virtualenvwrapper`` is fully installed, use it to create a Python 3
virtual environment, like so:

.. code:: bash

    mkvirtualenv -p `which python3` <name_of_your_project>

The ``-p `which python3``` option ensures that when the virtual environment is
activated, the commands ``python`` and ``pip`` will refer to their Python 3
counterparts.

The virtual environment should have been activated automatically after creating
it. It can be manually activated with ``workon <name_of_your_project>``, and
deactivated with ``deactivate``.

**Important:** Remember to activate the virtual environment *every time* you
begin working on your project. Also, note that the currently active virtual
environment is *not* associated with any particular folder; it is associated
with a terminal shell.

Finally, you can install PyPhi into your new virtual environment:

.. code:: bash

    pip install pyphi

Congratulations, you've just installed PyPhi!

To play around with the software, ensure that you've activated the virtual
environment with ``workon <name_of_your_project>``. Then run ``python`` to
start a Python 3 interpreter. Then, in the interpreter's command-line (which is
preceded by the ``>>>`` prompt), run 

.. code:: python

    import pyphi

Please see the documentation for some `examples
<http://pythonhosted.org/pyphi/#usage-and-examples>`_ and information on how to
`configure <http://pythonhosted.org/pyphi/#configuration-optional>`_ it.


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
