# exceptions.py
"""PyPhi exceptions."""


class MissingOptionalDependenciesError(ModuleNotFoundError):
    """The user needs to install extra dependencies."""

    MSG = "Please re-install PyPhi with `pyphi[{dependencies}]` to use this feature."


class StateUnreachableError(ValueError):
    """The current state cannot be reached from any previous state."""

    def __init__(self, state):
        self.state = state
        msg = "The state {} cannot be reached in the given TPM."
        super().__init__(msg.format(state))


class ConditionallyDependentError(ValueError):
    """The TPM is conditionally dependent."""


class JSONVersionError(ValueError):
    """JSON was serialized with a different version of PyPhi."""


class WrongDirectionError(ValueError):
    """The wrong direction was provided."""
