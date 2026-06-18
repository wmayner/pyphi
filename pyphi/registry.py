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

    See ``pyphi.system.PartitionRegistry`` and
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
        except KeyError as err:
            raise KeyError(
                f'"{name}" not found. Try using one of the installed '
                f"{self.desc} {self.all()} or register your own."
            ) from err


class InstanceRegistry[T](Mapping[str, T]):
    """Generic registry for named *instances* (not functions).

    The sibling :class:`Registry` stores callables and is parameterized by
    their return type; this stores objects directly, so ``self[name]`` returns
    a ``T`` rather than a callable producing one. Subclasses register concrete
    instances and typically override :meth:`register` to validate them against
    a Protocol. See :class:`pyphi.formalism.base.FormalismRegistry`.
    """

    desc = ""

    def __init__(self) -> None:
        self.store: dict[str, T] = {}

    def register(self, name: str, instance: T) -> T:
        """Register ``instance`` under ``name`` and return it."""
        self.store[name] = instance
        return instance

    def all(self) -> list[str]:
        """Return a list of all registered names."""
        return list(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, name: str) -> T:
        try:
            return self.store[name]
        except KeyError as err:
            raise KeyError(
                f'"{name}" not found. Try using one of the installed '
                f"{self.desc} {self.all()} or register your own."
            ) from err
