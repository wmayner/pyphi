Tie-breaking
===========

At several points during the IIT analysis, ties may arise when integrated
information values of objects (partitions, purviews, subsystems) are compared.
See the following papers:

    | Krohn S, Ostwald D.
    | Computing integrated information.
    | *Neuroscience of Consciousness*. 2017; 2017(nix017).
    | https://doi.org/10.1093/nc/nix017

    | Moon K.
    | Exclusion and Underdetermined Qualia.
    | *Entropy*. 2019; 21(4):405.
    | https://doi.org/10.3390/e21040405

    | Hanson JR, Walker SI.
    | On the Non-uniqueness Problem in Integrated Information Theory.
    | *bioRxiv*. 2021;2021.04.07.438793.
    | https://doi.org/10.1101/2021.04.07.438793

This documentation is solely meant to describe how the software deals with
ties. Future work is required to resolve the role of ties in the IIT formalism.

.. warning::

    Currently, only one of the tied objects is returned (selected as described
    below).

    An exception is the actual causation computation, which returns all tied
    actual causes and effects.

In general, when searching for a minimal or maximal value, the first one
encountered is selected as the output and used in further analysis (see
additional selection criteria below).

This arbitrary selection process was chosen for computational simplicity. It
also ensures that the same purviews are selected in the intact and cut
subsystem, if the purviews are not affected by the system cut. This is important
because causal distinctions unaffected by the system cut should not contribute
to |big_phi|.

As a consequence, the object selected may depend on the speed at which it was
computed when performing parallel computations. This applies to tied system cuts
if :attr:`~pyphi.conf.PyphiConfig.PARALLEL_CUT_EVALUATION` is set to ``True``,
and to subsystems if :attr:`~pyphi.conf.PyphiConfig.PARALLEL_COMPLEX_EVALUATION`
is set to ``True`` and subsystems of the same size are tied for
:func:`pyphi.compute.network.major_complex` (note that
:func:`pyphi.compute.network.complexes` returns the list of all complexes within
a system). However, ties in partitions and subsystems do not have further
effects on other quantities in the computation.

Mechanism purviews are always evaluated in the same order. In principle, ties in
the |small_phi| values across multiple possible purivews propagate to the |sia|.
Under most choices of settings, including the default IIT 3.0 configuration, the
|big_phi| value of a subsystem depends on the particular purviews of its
mechanisms, not just their |small_phi| value (except if
:attr:`~pyphi.conf.PyphiConfig.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE` is set
to ``True``.

.. warning::

    In case of purview ties, the cause-effect structure and |big_phi|
    value for a given subsystem are not necessarily unique.

Our expectation is that the resulting value is representative. The variance of
possible |big_phi| values will depend on the specific system under consideration
and the particular config settings for performing the |sia|. Deterministic
systems with symmetric inputs and outputs are generally more prone to ties than
probabilistic systems. The EMD measure used in the default IIT 3.0 settings
is also particularly prone to ties in |small_phi| across purviews. Moreover,
due to the numerical optimization algorithm of the EMD measure, the |PRECISION|
has to be set to a relatively low value (default is 6 decimal places; see
|PRECISION|). Other difference measures allow for significantly higher
precision.

Selection criteria
~~~~~~~~~~~~~~~~~~

|MIC| and |MIE| objects can be compared using the built-in Python comparison
operators (``<``, ``>``, etc.) First, |small_phi| values are compared. If
these are equal up to |PRECISION|, the size of the purview is compared. If
|PICK_SMALLEST_PURVIEW| is set to ``True``, the partition with the smallest
purview is returned; otherwise the largest purview is returned. By default, this
is set to ``False`` (which of these should be used depends on the choice of
difference measure :attr:`~pyphi.conf.PyphiConfig.MEASURE` and
:attr:`~pyphi.conf.PyphiConfig.PARTITION_TYPE` for |small_phi| computations; the
default configuration settings correspond to the IIT 3.0 formalism).

|SIA| objects are compared this way as well. First, |big_phi| values are
compared. If these are equal up to |PRECISION|, then the largest subsystem is
returned. The :func:`pyphi.compute.complexes` function computes all complexes
with |big_phi| > 0 and can be used to inspect ties in |big_phi| across
subsystems.

All remaining ties between objects of the same size are resolved in an arbitrary
but stable manner by chosing the first one encountered. That is, if there is no
unique largest (or smallest, depending on configuration) purview with maximal
|small_phi|, the returned purview is the first one as ordered by
:func:`pyphi.subsystem.Subsystem.potential_purviews` (lexicographical by node index).
Similarly, if there is no unique largest subsystem with maximal |big_phi|, then
the returned subsystem is the first one as ordered by
:func:`pyphi.compute.possible_complexes` (also lexicographical by node
index).

Below we list all instances in which ties may occur.

Comparing mechanism partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first instance of ties can arise when finding the partition with the
minimum |small_phi| value (MIP) for a given purview.  After computing each
partition's |small_phi| value, it is compared to the previously minimal
|small_phi|. If less, the MIP will be updated. Therefore, the first of all
minimal |small_phi| found will be selected.

In ``subsystem.py``, `line 610 <https://github.com/wmayner/pyphi/blob/develop/pyphi/subsystem.py#L610>`_:

.. code:: python

    if phi < mip.phi:
        mip = _mip(phi, partition, partitioned_repertoire)

This is performed for both causes and effects.

Comparing purviews
~~~~~~~~~~~~~~~~~~

After computing the minimum information partition, we take the ``max()`` across
all potential purviews. In the case of a tie, Python's builtin ``max()``
function returns the first maximal element.

In ``subsystem.py``, `line 703 <https://github.com/wmayner/pyphi/blob/develop/pyphi/subsystem.py#L703>`_:

.. code:: python

    if not purviews:
        max_mip = _null_ria(direction, mechanism, ())
    else:
        max_mip = max(
            self.find_mip(direction, mechanism, purview) for purview in purviews
        )

This is performed for both causes and effects.


Comparing system partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we find the system partition with minimal |big_phi|.

Assuming we don't short-circuit (*i.e.,* find a |SIA| with |big_phi| = 0), each
new SIA is compared with the previous minimum. Again the returned minimum is the
first one, as ordered by :func:`pyphi.compute.subsystem.sia_bipartitions`.

In ``compute/subsystem.py``, `line 191 <https://github.com/wmayner/pyphi/blob/develop/pyphi/compute/subsystem.py#L191>`_:

.. code:: python

    def process_result(self, new_sia, min_sia):
        """Check if the new SIA has smaller |big_phi| than the standing
        result.
        """

        if new_sia.phi == 0:
            self.done = True  # Short-circuit
            return new_sia

        elif abs(new_sia.phi) < abs(min_sia.phi):
            return new_sia

        return min_sia

Comparing candidate systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, a search is performed for the candidate system with maximal |big_phi|.
We compare the candidate systems with the builtin ``max()``, returning the first
one, as ordered by :func:`pyphi.compute.networks.possible_complexes`.

In ``compute/network.py``, `line 149 <https://github.com/wmayner/pyphi/blob/develop/pyphi/compute/network.py#L149>`_:

.. code:: python

    result = complexes(network, state)
    if result:
        result = max(result)

This is then the major complex of the network.
