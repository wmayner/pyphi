.. _macos-installation:

Detailed installation guide for macOS
=====================================

This is a step-by-step guide intended for those unfamiliar with Python
or the command-line (*a.k.a.* the "shell").

A shell can be opened by opening a new tab in the Terminal app (located in
Utilities). Text that is ``formatted like code`` is meant to be copied and
pasted into the terminal (hit the Enter key to run the command).

Installing uv
-------------

The first step is to install `uv <https://github.com/astral-sh/uv>`__, a fast
Python package manager that will handle Python installation and package
management for us.

Install uv by running this command in your terminal:

.. code:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

After installation completes, **close and re-open your Terminal window** for
the changes to take effect.

Installing PyPhi
----------------

Now we can use uv to create a Python environment and install PyPhi. Navigate
to the directory where you want to work on your project:

.. code:: bash

    cd ~/your/project/directory

Create a new Python environment for your project:

.. code:: bash

    uv venv

This creates a virtual environment in a ``.venv`` directory. Virtual
environments allow different projects to isolate their dependencies from one
another, so that they don't interact in unexpected ways.

Activate the environment:

.. code:: bash

    source .venv/bin/activate

.. important::

    Remember to activate your project's environment **every time you begin
    working on your project**. When the environment is active, your
    command-line prompt should show ``(.venv)`` at the beginning.

Now install PyPhi:

.. code:: bash

    uv pip install pyphi

Congratulations, we've just installed PyPhi!

To play around with the software, let's install `IPython
<https://ipython.readthedocs.io/en/stable/#>`__. IPython provides an enhanced
Python `REPL
<https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`__ that
has tab-completion, syntax highlighting, and many other nice features.

.. code:: bash

    uv pip install ipython

Now we can run ``ipython`` (or ``uv run ipython``) to start an IPython session.
In the Python command-line that appears (it's preceded by the ``>>>`` prompt),
run:

.. code:: python

    import pyphi

Now you've imported PyPhi and can start using it!

Next, please see the documentation for some `examples
<https://pyphi.readthedocs.io/page/examples/>`__ of what PyPhi can do and
information on how to `configure
<https://pyphi.readthedocs.io/page/configuration.html>`__ it.

Legacy Installation with Conda
-------------------------------

.. note::

    The conda-based installation method is deprecated. We recommend using uv
    as described above for a faster and more reliable experience.

If you prefer to use conda, you can follow the old installation method:

Install Miniconda by downloading and running the installer script:

.. code:: bash

    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
    sh Miniconda3-latest-MacOSX-x86_64.sh

Create and activate a conda environment:

.. code:: bash

    conda create --name <name_of_your_project> python=3.12
    conda activate <name_of_your_project>

Install PyPhi:

.. code:: bash

    pip install pyphi
