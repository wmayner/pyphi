PyPhi
=====

PyPhi is a Python library for computing integrated information.

See the documentation for :mod:`pyphi.examples` for information on how to use
it.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.


Getting started
~~~~~~~~~~~~~~~

The :class:`pyphi.network` object is the main object on which computations are
performed. It represents the network of interest.

The :class:`pyphi.subsystem` object is the secondary object; it represents a
subsystem of a network. |big_phi| is defined on subsystems.

The :mod:`pyphi.compute` module is the main entry-point for the library. It
contains methods for calculating concepts, constellations, complexes, etc.

Examples
~~~~~~~~

The best way to familiarize yourself with the software is to go through the
examples below, following along in a REPL.

.. toctree::
    :maxdepth: 2

    examples/index


Configuration
~~~~~~~~~~~~~

PyPhi can be configured in various important ways; see below for details.

.. toctree::
    :maxdepth: 2

    configuration


API Reference
~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    api/index
