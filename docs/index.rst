PyPhi
=====

PyPhi is a Python library for computing integrated information.

The latest formalism of Integrated Information Theory (IIT 4.0) is outlined in this paper:

    | Albantakis L, Barbosa L, Findlay G, Grasso M, ... Tononi G. (2023)
    | Integrated information theory (IIT) 4.0: formulating the properties of phenomenal existence in physical terms. 
    | *PLoS Computational Biology* 19(10): e1011465.
    | https://doi.org/10.1371/journal.pcbi.1011465

If you use this software in your research, please cite the paper:

    | Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018)
    | PyPhi: A toolbox for integrated information theory.
    | *PLOS Computational Biology* 14(7): e1006343.
    | https://doi.org/10.1371/journal.pcbi.1006343

A `jupyter notebook
<https://colab.research.google.com/github/wmayner/pyphi/blob/feature/iit-4.0/docs/examples/IIT_4.0_demo.ipynb>`_  illustrating how to use PyPhi is available as a
supplement to the `IIT 4.0 paper
<https://doi.org/10.1371/journal.pcbi.1006343.s001>`_.

To report issues, use the issue tracker on the `GitHub repository
<https://github.com/wmayner/pyphi>`_. Bug reports and pull requests are
welcome.

For general discussion, you are welcome to join the `pyphi-users group
<https://groups.google.com/forum/#!forum/pyphi-users>`_.

.. _installation:

.. include:: installation.rst

.. toctree::
    :caption: Usage and Examples
    :glob:
    :maxdepth: 1

    installation.rst
    examples/index
    examples/2014paper
    examples/conditional_independence
    examples/xor
    examples/emergence
    examples/actual_causation
    examples/residue
    examples/magic_cut
    macos_installation

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
    caching

.. toctree::
    :caption: API Reference
    :glob:
    :maxdepth: 1

    api/*
