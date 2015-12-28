PyPhi
=====

PyPhi is a Python library for computing integrated information.

To report issues, please use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.


Getting started
~~~~~~~~~~~~~~~

The :class:`~pyphi.network.Network` object is the main object on which
computations are performed. It represents the network of interest.

The :class:`~pyphi.subsystem.Subsystem` object is the secondary object; it
represents a subsystem of a network. |big_phi| is defined on subsystems.

The :mod:`~pyphi.compute` module is the main entry-point for the library. It
contains methods for calculating concepts, constellations, complexes, etc.

The best way to familiarize yourself with the software is to go through the
examples. All the examples dicussed are available in the :mod:`~pyphi.examples`
module, so you can follow along in a REPL. The relevant functions are listed at
the beginning of each example.

.. toctree::
    :glob:
    :maxdepth: 2

    examples/index


Configuration
~~~~~~~~~~~~~

PyPhi can be configured in various important ways; see the :mod:`~pyphi.config`
module for details.

.. toctree::
    :glob:
    :maxdepth: 2

    configuration


Conventions
~~~~~~~~~~~

PyPhi uses some conventions for TPM and connectivity matrix formats. These are
important to keep in mind when setting up networks.

.. toctree::
    :glob:
    :maxdepth: 2

    conventions
