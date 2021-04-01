.. _macos-installation:

Detailed installation guide for macOS
=====================================

This is a step-by-step guide intended for those unfamiliar with Python
or the command-line (*a.k.a.* the “shell”).

A shell can be opened by opening a new tab in the Terminal app (located in
Utilities). Text that is ``formatted like code`` is meant to be copied and
pasted into the terminal (hit the Enter key to run the command).

The fist step is to install the versions of Python that we need. The most
convenient way of doing this is to use the `Miniconda distribution of Python
<https://docs.conda.io/en/latest/miniconda.html>`__. Install Miniconda by
downloading and running the installer script:

.. code:: bash

    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh

Then run it with:

.. code:: bash

    sh Miniconda3-latest-MacOSX-x86_64.sh

Once you have installed Miniconda, **close and re-open your Terminal window**
and confirm that your ``python`` command points to the Minconda-installed
version of Python, rather than your computers's default Python, by running
``which python``. This should print something like
``/Users/<your_username>/minconda3/bin/python``.

Using the Miniconda Python rather than the version of Python that comes with
your computer will protect your computer's Python version from unwanted
changes that could interfere with other applications.

Now we want to use the ``conda`` command-line tool (installed with Miniconda)
to create an isolated Python `environment
<https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_
within which to install PyPhi. Environments allow different projects to
isolate their dependencies from one another, so that they don't interact in
unexpected ways.

.. code:: bash

    conda create --name <name_of_your_project>

Once we've created the environment, we need to "activate" it so that when we
run Python or install Python packages, we're doing those things inside the
isolated environment. To activate the environment we just created, run
``conda activate <name_of_your_project>`` (and to deactivate it, run ``conda
deactivate``, or start a new Terminal session).

.. important::

    Remember to activate your project's environment **every time you begin
    working on your project**. Also, note that the currently active virtual
    environment is *not* associated with any particular folder; it is
    associated with a Terminal session. In other words, each time you open a
    new Terminal tab or Terminal window, you need to run ``conda activate
    <name_of_your_project``. When the environment is active, your
    command-line prompt should show the name of the environment.

The first thing we need to do inside the new environment is install Python:

... code:: bash

    conda install python

Now we're ready to install PyPhi. To do this, we'll use ``pip``, the Python
package manager:

.. code:: bash

    pip install pyphi

Congratulations, we've just installed PyPhi!

To play around with the software, let's install `IPython
<https://ipython.readthedocs.io/en/stable/#>`__. IPython provides an enhanced
Python `REPL
<https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`__ that
has tab-completion, syntax highlighting, and many other nice features.

.. code:: bash

    pip install ipython

Now we can run ``ipython`` to start an IPython session. In the Python
command-line that appears (it's preceded by the ``>>>`` prompt), run

.. code:: python

    import pyphi

Now you've imported PyPhi and can start using it!

Next, please see the documentation for some `examples
<https://pyphi.readthedocs.io/page/examples/>`__ of what PyPhi can do and
information on how to `configure
<https://pyphi.readthedocs.io/page/configuration.html>`__ it.
