"""Intrinsic-units macro framework (Marshall et al. 2024).

Macro units are defined by sliding-window state mappings over their
micro constituents; macro cause and effect TPMs are built by the
four-step construction (Eqs. 26-40) and analyzed by the IIT 4.0
pipeline exactly as micro systems are.
"""

from pyphi.macro.units import MacroUnit

__all__ = ["MacroUnit"]
