"""Display-time formatting of numeric values."""

from numbers import Real

SIG_FIGS = 6


def format_value(value, sig_figs: int = SIG_FIGS) -> str:
    """Format a value for display.

    Real numbers are rounded to ``sig_figs`` significant figures; everything
    else is rendered with ``str``. The exact numeric value remains available on
    the source object's attribute.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.{sig_figs}g}"
    return str(value)
