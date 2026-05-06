"""CausalModel value type — substrate + TPM. The zeroth postulate of IIT 4.0.

Zero computation. No caches. Operations on a CausalModel are free
functions in :mod:`pyphi.core.tpm.marginalization` or
:mod:`pyphi.core.repertoire_algebra`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .substrate import Substrate
from .tpm.base import TPM
from .tpm.explicit import ExplicitTPM
from .unit import Unit

if TYPE_CHECKING:
    from pyphi.network import Network


@dataclass(frozen=True, eq=False)
class CausalModel:
    """An immutable :class:`Substrate` paired with a :class:`TPM`."""

    substrate: Substrate
    tpm: TPM

    @classmethod
    def from_network(cls, network: Network) -> CausalModel:
        """Build a CausalModel from a legacy Network.

        Migration helper. Stays through P7+P7b+P8; deleted before 2.0
        ships if all callers go direct.
        """
        units = tuple(
            Unit(index=i, label=str(network.node_labels[i])) for i in range(network.size)
        )
        substrate = Substrate(units=units, connectivity_matrix=network.cm)
        tpm = ExplicitTPM(network.tpm)
        return cls(substrate=substrate, tpm=tpm)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalModel):
            return NotImplemented
        if self.substrate != other.substrate:
            return False
        a = self.tpm.to_array()
        b = other.tpm.to_array()
        return a.shape == b.shape and bool((a == b).all())

    def __hash__(self) -> int:
        arr = self.tpm.to_array()
        return hash((self.substrate, arr.tobytes(), arr.shape))
