"""Intrinsic-units macro framework (Marshall et al. 2024).

Macro units are defined by sliding-window state mappings over their
micro constituents; macro cause and effect TPMs are built by the
four-step construction (Eqs. 26-40) and analyzed by the IIT 4.0
pipeline exactly as micro systems are.
"""

from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit

__all__ = ["MacroUnit", "blackbox", "coarse_grain", "macro_tpms", "micro_unit"]
