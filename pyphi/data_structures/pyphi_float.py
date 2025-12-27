# data_structures/pyphi_float.py

from typing import Any

from ..conf import config
from ..utils import eq


# TODO(4.0) use throughout
class PyPhiFloat(float):
    """A floating-point value that's compared using config.PRECISION."""

    # NOTE: Cannot use functools.total_ordering because it doesn't re-implement
    # existing comparison methods

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return False
        return eq(self, float(other))

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return True
        return not eq(self, float(other))

    def __lt__(self, other: float) -> bool:
        return super().__lt__(other) and not eq(self, other)

    def __gt__(self, other: float) -> bool:
        return super().__gt__(other) and not eq(self, other)

    def __le__(self, other: float) -> bool:
        return super().__le__(other) or eq(self, other)

    def __ge__(self, other: float) -> bool:
        return super().__ge__(other) or eq(self, other)

    def __hash__(self) -> int:
        return hash(round(self, config.PRECISION))

    def to_json(self) -> dict[str, float]:
        return {"value": float(self)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PyPhiFloat":
        return cls(data["value"])
