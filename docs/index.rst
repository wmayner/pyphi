PyPhi
=====

PyPhi is a Python library for computing integrated information.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.

For general discussion, you are welcome to join the `pyphi-users group
<https://groups.google.com/forum/#!forum/pyphi-users>`_.

.. important::
    Each version of PyPhi has its own documentation—make sure you're looking
    at the documentation for the version you're using. You can switch
    documentation versions in the bottom-left corner.

    The ``stable`` version of the documentation corresponds to the most recent
    stable release of PyPhi; this is the version you have if you installed
    PyPhi with ``pip install pyphi``. The ``latest`` version corresponds to the
    most recent unreleased development version (which may have bugs). 

Installation
~~~~~~~~~~~~

To install the latest stable release, run

.. code-block:: bash

    pip install pyphi

To install the latest development version, which is a work in progress and may
have bugs, run

.. code-block:: bash

    pip install "git+https://github.com/wmayner/pyphi@develop#egg=pyphi"

**For detailed instructions on how to install PyPhi on macOS, see the**
`installation guide
<https://github.com/wmayner/pyphi/blob/develop/INSTALLATION.md>`_.

.. note::
    PyPhi is only supported on Linux and macOS operating systems; Windows is
    not supported.

.. toctree::
    :caption: Usage and Examples
    :glob:
    :maxdepth: 1

    examples/index
    examples/*

.. toctree::
    :caption: Conventions
    :glob:
    :maxdepth: 1

    conventions

.. toctree::
    :caption: Configuration
    :glob:
    :maxdepth: 1

    configuration

.. toctree::
    :caption: API Reference
    :glob:
    :maxdepth: 1

    api/*
