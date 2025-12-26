# data_structures/pyphi_float.py

from ..conf import config
from ..utils import eq


# TODO(4.0) use throughout
class PyPhiFloat(float):
    """A floating-point value that's compared using config.PRECISION."""

    # NOTE: Cannot use functools.total_ordering because it doesn't re-implement
    # existing comparison methods

    def __eq__(self, other):
        return eq(self, other)

    def __ne__(self, other):
        return not eq(self, other)

    def __lt__(self, other):
        return super().__lt__(other) and not eq(self, other)

    def __gt__(self, other):
        return super().__gt__(other) and not eq(self, other)

    def __le__(self, other):
        return super().__le__(other) or eq(self, other)

    def __ge__(self, other):
        return super().__ge__(other) or eq(self, other)

    def __hash__(self):
        return hash(round(self, config.PRECISION))

    def to_json(self):
        return {"value": float(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["value"])
