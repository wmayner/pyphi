"""Display-time formatting of numeric values."""

from numbers import Integral
from numbers import Real

SIG_FIGS = 6


def format_value(value, sig_figs: int = SIG_FIGS) -> str:
    """Format a value for display.

    Real (non-integer) numbers are rounded to ``sig_figs`` significant figures
    and always keep a decimal point (so continuous quantities read
    consistently). Integers — counts, indices — render without one. Everything
    else is rendered with ``str``. The exact numeric value remains available on
    the source object's attribute.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        formatted = f"{float(value):.{sig_figs}g}"
        # Keep a decimal point on whole-valued floats (3.0 -> "3.0"); leave
        # exponential/inf/nan untouched.
        if not any(c in formatted for c in ".en"):
            formatted += ".0"
        return formatted
    return str(value)
