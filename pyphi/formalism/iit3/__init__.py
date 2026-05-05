"""IIT 3.0 formalism (Oizumi et al. 2014).

Distribution-distance-based phi computation. Partition scheme: bipartitions
(``BI``). Compatible metrics: ``EMD``, ``L1``, ``KLD``, ``ENTROPY_DIFFERENCE``,
``PSQ2``, ``MP2Q``, ``ABSOLUTE_INTRINSIC_DIFFERENCE``,
``INTRINSIC_DIFFERENCE``.

The :class:`IIT3Formalism` instance delegates to ``pyphi.compute.subsystem``
for SIA computation; the implementation files remain there for now.
"""

from .formalism import IIT3Formalism

__all__ = ["IIT3Formalism"]
