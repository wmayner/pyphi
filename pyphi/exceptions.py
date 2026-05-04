# pyright: strict
# exceptions.py
"""PyPhi exceptions."""


class MissingOptionalDependenciesError(ModuleNotFoundError):
    """The user needs to install extra dependencies."""

    MSG: str = (
        "Please re-install PyPhi with `pyphi[{dependencies}]` to use this feature."
    )


class StateUnreachableError(ValueError):
    """Base class for state unreachability errors."""

    def __init__(self, state: tuple[int, ...], message: str | None = None) -> None:
        self.state = state
        if message is None:
            message = f"The state {state} cannot be reached."
        super().__init__(message)


class StateUnreachableForwardsError(StateUnreachableError):
    """The current state cannot be reached from any previous state.

    This error is raised when the forward/effect TPM validation fails,
    meaning no previous state can transition to the current state.
    """

    def __init__(self, state: tuple[int, ...]) -> None:
        message = (
            f"The state {state} cannot be reached from any previous state "
            "(forward TPM check)."
        )
        super().__init__(state, message)


class StateUnreachableBackwardsError(StateUnreachableError):
    """The current state has zero probability when computing the backward TPM.

    This error is raised when the normalization factor for the backward TPM
    computation is zero, indicating the state is unreachable in the reverse direction.
    """

    def __init__(self, state: tuple[int, ...]) -> None:
        message = (
            f"The state {state} has zero probability when computing the backward TPM."
        )
        super().__init__(state, message)


class ConditionallyDependentError(ValueError):
    """The TPM is conditionally dependent."""


class JSONVersionError(ValueError):
    """JSON was serialized with a different version of PyPhi."""


class WrongDirectionError(ValueError):
    """The wrong direction was provided."""
