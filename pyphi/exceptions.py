#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exceptions.py

"""PyPhi exceptions."""

import warnings


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


def warn_about_tie_serialization(
    name, serialize=False, deserialize=False, stacklevel=3
):
    # XOR; exactly one of serialize or deserialize must be True
    if not serialize ^ deserialize:
        raise ValueError("Exactly one of ``serialize``, ``deserialize`` must be True")
    if serialize:
        msg = (
            "Serializing ties to JSON in {name} is not currently "
            "supported; tie information will be lost."
        )
    if deserialize:
        msg = (
            "Deserializing ties in {name} from JSON is not currently "
            "supported; tie information was lost during serialization."
        )
    warnings.warn(msg.format(name=name), UserWarning, stacklevel=stacklevel)
