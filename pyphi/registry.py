# registry.py
"""Provides a ``Registry`` class for storing user-provided functions."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any
from typing import TypeVar

T = TypeVar("T")


class Registry[T](Mapping[str, Callable[..., Any]]):
    """Generic registry for user-supplied functions.

    See ``pyphi.subsystem.PartitionRegistry`` and
    ``pyphi.distance.MeasureRegistry`` for concrete usage examples.
    """

    desc = ""

    def __init__(self) -> None:
        self.store: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for registering a function with PyPhi.

        Args:
            name (string): The name of the function
        """

        def register_func(func: Callable[..., T]) -> Callable[..., T]:
            self.store[name] = func
            return func

        return register_func

    def all(self) -> list[str]:
        """Return a list of all registered functions"""
        return list(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, name: str) -> Callable[..., T]:
        try:
            return self.store[name]
        except KeyError:
            raise KeyError(
                f'"{name}" not found. Try using one of the installed {self.desc} {self.all()} or '
                "register your own."
            )
