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
            "(forward TPM check). IIT evaluates a system that arrived at its "
            "state through its own cause-effect power, so a state with no "
            "possible predecessor has no defined analysis."
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


class TransitionUnreachableError(StateUnreachableError):
    """The transition has zero probability under the substrate dynamics.

    Raised when a state pair violates the Realization principle of
    Albantakis et al. (2019): a transition is defined only when
    p(after state | before state) > 0.
    """

    def __init__(
        self,
        before_state: tuple[int, ...],
        after_state: tuple[int, ...],
        message: str | None = None,
    ) -> None:
        self.before_state = before_state
        self.after_state = after_state
        if message is None:
            message = (
                f"The transition {before_state} -> {after_state} has zero "
                "probability under the substrate dynamics."
            )
        super().__init__(after_state, message)


class IntractableCauseInversionError(ValueError):
    """The cause inversion cannot proceed within the intermediate-size cap.

    Raised when every remaining step of the sum-product contraction would
    materialize an intermediate array larger than the cap — the substrate's
    coupling is too dense for the reduced inversion at this size.
    """


class ConditionallyDependentError(ValueError):
    """The TPM is conditionally dependent."""


class WrongDirectionError(ValueError):
    """The wrong direction was provided."""


class InvalidTPM(ValueError):
    """A TPM violates a structural or probability axiom."""


class NonConvergenceError(ValueError):
    """A deterministic trajectory entered a limit cycle instead of a fixed point."""
