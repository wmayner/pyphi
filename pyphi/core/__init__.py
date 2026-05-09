"""Lower-level value types and primitives shared across PyPhi.

User-facing classes :class:`pyphi.Substrate` and :class:`pyphi.System` live at
the top level. This subpackage holds the kernel pieces composed below them:
the :class:`Unit` type, repertoire algebra primitives, and TPM utilities.
"""

from .unit import Unit as Unit
