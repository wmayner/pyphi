Installation
~~~~~~~~~~~~

Using uv (Recommended)
======================

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package manager. Install it with:

.. code-block:: bash

    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Then install PyPhi:

.. code-block:: bash

    uv pip install pyphi

To install the latest development version:

.. code-block:: bash

    uv pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"

Using pip
=========

To install the latest stable release with pip:

.. code-block:: bash

    pip install pyphi

To install the latest development version:

.. code-block:: bash

    pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"

.. tip::
    For detailed instructions on how to install PyPhi on macOS, see the
    :ref:`macos-installation`.

Legacy: Conda (Deprecated)
==========================

.. warning::
    The conda package is deprecated and may not receive updates. Please use uv or pip instead.

**Windows users:** If you encounter issues with uv or pip on older Windows systems,
you can use the `Anaconda Python <https://www.anaconda.com/what-is-anaconda/>`_
distribution and `install PyPhi with conda <https://anaconda.org/wmayner/pyphi>`_:

.. code-block:: bash

    conda install -c wmayner pyphi
