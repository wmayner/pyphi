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


def format_column(values, sig_figs: int = SIG_FIGS) -> list[str]:
    """Format one table column, aligning numbers on the decimal point.

    If every value is a real number, each formatted string is padded with
    spaces so that the decimal points (or the ends of integers) share a
    character position and all strings have equal length; in a monospace font
    the column then reads as decimal-aligned under any text alignment. A column
    containing any non-numeric value is formatted cell by cell, unpadded.
    """
    strings = [format_value(v, sig_figs) for v in values]
    if not all(isinstance(v, Real) and not isinstance(v, bool) for v in values):
        return strings
    parts = [s.partition(".") for s in strings]
    int_w = max((len(whole) for whole, _, _ in parts), default=0)
    frac_w = max((len(dot + frac) for _, dot, frac in parts), default=0)
    return [
        whole.rjust(int_w) + (dot + frac).ljust(frac_w) for whole, dot, frac in parts
    ]
