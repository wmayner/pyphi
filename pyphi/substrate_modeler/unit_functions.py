from typing import Any, TYPE_CHECKING

import numpy as np
from itertools import product

from .. import utils
from .utils import map_to_floor_and_ceil

if TYPE_CHECKING:
    from .unit import Unit

# UNIT FUNCTIONS


def composite(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
) -> np.ndarray:
    """Placeholder mechanism for CompositeUnit.

    This should never be called directly; CompositeUnit overrides compute_tpm.
    """
    raise NotImplementedError(
        "composite mechanism should not be called directly; "
        "CompositeUnit overrides compute_tpm"
    )


def sigmoid(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: tuple[float, ...] = (),
    determinism: float = 5.0,
    threshold: float = 0.0,
    ising: bool = True,
) -> np.ndarray:
    # validate all kwargs
    validate_kwargs(locals())

    n_nodes = len(input_weights)

    def LogFunc(s, input_weights, determinism, threshold):
        if ising:
            s = tuple(x * 2 - 1 for x in s)
        total_input = sum(s * np.array([input_weights[n] for n in range(n_nodes)]))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return map_to_floor_and_ceil(y, floor, ceiling)

    tpm = np.array(
        [
            [
                LogFunc(
                    s,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for s in utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def sor_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: tuple[tuple[int, ...], ...] = (),
    selectivity: float = 2.0,
) -> np.ndarray:
    # Ensure states are tuples
    patterns = [tuple(p) for p in pattern_selection]

    # Ensure ceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        raise ValueError(
            "Selectivity for SOR gates must be bigger than 1, "
            f"got {selectivity} for unit {unit.label}."
        )

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * len(input_state)) * floor

    # if the input state matches a pattern in the pattern selection, activation probability
    # given that state is ceiling; otherwise, it is increased by a fraction of the difference
    # between floor and ceiling (given by selectivity)
    pattern = floor + (ceiling - floor) / selectivity
    for s in patterns:
        tpm[s] = pattern
    if input_state in patterns:
        tpm[input_state] = ceiling
    else:
        tpm[input_state] = 0.0

    return tpm


def resonnator(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: tuple[float, ...] = (),
    determinism: float = 5.0,
    threshold: float = 0.0,
    weight_scale_mapping: dict,
) -> np.ndarray:
    # Ensure ceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    n_nodes = len(input_weights)

    # alter weight to make it push unit towards its state, weighted using weight_scale_mapping
    w = [
        (
            input_weights[i] * weight_scale_mapping[(state, inp_s)]
            if inp_s == state
            else -input_weights[i] * weight_scale_mapping[(state, inp_s)]
        )
        for i, inp_s in enumerate(input_state)
    ]

    def resonnatorFunc(s, input_weights, determinism, threshold, unit_state):
        # make state interpreted as ising
        s = [2 * x - 1 for x in s]

        total_input = np.sum(s * np.array(w))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return y

    # producing transition probability matrix
    tpm = np.array(
        [
            [resonnatorFunc(s, input_weights, determinism, threshold, state)]
            for s in utils.all_states(n_nodes)
        ]
    )

    # make it between floor and ceiling
    tpm = map_to_floor_and_ceil(tpm, floor, ceiling)

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def mismatch_pattern_detector(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: tuple[tuple[int, ...], ...] = (),
    selectivity: float = 1.0,
) -> np.ndarray:
    # This mechanism is selective to certain inputs (i.e. they turn it ON with P=ceiling,
    # while the remaining possible input patterns turn it OFF with P=floor). However, its
    # selectivity (probability of turning on) depends on the state of the unit: if the unit
    # is already in the state that matches the pattern, then the effect of the inputs is
    # reduced by the selectivity factor.

    # Ensure states are tuples
    patterns = [tuple(p) for p in pattern_selection]

    # Ensure ceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    if floor is None:
        floor = 1.0 - ceiling

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        raise ValueError(
            "Selectivity for mismatch_pattern_detector must be bigger than 1, "
            f"got {selectivity} for unit {unit.label}."
        )

    # Check if the unit is ON
    if state == 1:
        # since it is ON, it will only respond strongly if a non-pattern is on its inputs
        P_pattern = 0.5 + (ceiling - 0.5) / selectivity
        P_no_pattern = floor
    else:
        # since it is OFF, it will only respond strongly if a pattern is on its inputs
        P_pattern = ceiling
        P_no_pattern = 0.5 - (0.5 - floor) / selectivity

    N = len(input_state)
    tpm = np.ones([2] * N)

    for s in utils.all_states(N):
        if s in patterns:
            tpm[s] = P_pattern
        else:
            tpm[s] = P_no_pattern

    return tpm


def copy_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    tpm = np.ones([2]) * floor
    tpm[1] = ceiling
    return tpm


def and_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    tpm = np.ones((2, 2)) * floor
    tpm[(1, 1)] = ceiling
    return tpm


def or_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    tpm = np.ones((2, 2)) * ceiling
    tpm[(0, 0)] = floor
    return tpm


def xor_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    tpm = np.ones((2, 2)) * floor
    tpm[(0, 1)] = ceiling
    tpm[(1, 0)] = ceiling
    return tpm


def weighted_mean(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    weights: list[float] | None = None,
) -> np.ndarray:
    if weights is None:
        weights = []
    weights = [w / np.sum(weights) for w in weights]
    N = len(weights)

    tpm = np.ones((2,) * N)
    for s in utils.all_states(N):
        wm = (
            sum((1 + w * (x * 2 - 1)) / 2 for w, x in zip(weights, s)) / N
        )
        tpm[s] = wm * (ceiling - floor) + floor

    return tpm


def democracy(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for s in utils.all_states(N):
        avg_vote = np.mean(s)
        tpm[s] = avg_vote * (ceiling - floor) + floor

    return tpm


def majority(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for s in utils.all_states(N):
        avg_vote = round(np.mean(s))
        tpm[s] = avg_vote * (ceiling - floor) + floor

    return tpm


def mismatch_corrector(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    if len(unit.inputs) > 1:
        raise ValueError(
            f"mismatch_corrector requires exactly one input, "
            f"but unit {unit.label} has {len(unit.inputs)}."
        )

    if bias > 1:
        raise ValueError(
            f"bias must be <= 1 for unit {unit.label}, got {bias}."
        )

    # check whether state of unit matches its input, and create tpm accordingly
    if state == input_state[0]:
        tpm = np.ones([2]) * 0.5 - (state * 2 - 1) * bias * 0.5
    else:
        tpm = np.array([floor, ceiling])

    return tpm


def modulated_sigmoid(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    input_weights: list[float],
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
    determinism: float = 2.0,
    threshold: float = 0.0,
) -> np.ndarray:
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float}
    # modulation will update the sigmoid function depending on whether the unit is ON or OFF

    def LogFunc(
        inp_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(inp_state * np.array([weight for weight in weights]))
        # count how many of the modulators are ON
        mods_on = sum(modulation_state)

        # modulate threshold based on unit state and the state of the modulators
        new_threshold = threshold + unit_state * mods_on * modulation["threshold"]

        # modulate determinism based on unit state and the state of the modulators
        new_determinism = determinism + unit_state * mods_on * modulation["determinism"]

        y = ceiling * (
            floor
            + (1 - floor)
            / (1 + np.e ** (-new_determinism * (total_input - new_threshold)))
        )
        return y

    # first inputs will be interpreted as true inputs, while the last will be modulator inputs
    n_mods = len(modulation["modulator"])
    n_inputs = len(input_weights)
    n_nodes = n_inputs + n_mods

    unit_state = state * 2 - 1  # making unit state "ising" rather than binary, to make modulation symmetric

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    inp_state,
                    modulation_state,
                    unit_state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for modulation_state, inp_state in product(
                utils.all_states(n_mods), utils.all_states(n_inputs)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def stabilized_sigmoid(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    input_weights: list[float],
    determinism: float,
    threshold: float,
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float, 'selectivity': float}
    # modulation will update the sigmoid function depending on whether the unit is ON or OFF

    def LogFunc(
        inp_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(inp_state * np.array([weight for weight in weights]))

        # The modulation should work in such a way as to always "stabilize" the current
        # state of the unit, but it should be stronger, the more modulation is "active"

        # count how many of the modulators are ON
        mods_on = sum(modulation_state)
        if mods_on == 0:
            mods_on = 1 / modulation["selectivity"]

        ising_state = (
            unit_state * 2 - 1
        )  # making unit state "ising" rather than binary, to make modulation symmetric

        # modulate threshold based on unit state and the state of the modulators
        new_threshold = threshold - ising_state * mods_on * modulation["threshold"]

        # modulate determinism based on unit state and the state of the modulators
        new_determinism = (
            determinism
            if (mods_on == 0 or modulation["determinism"] == 0)
            else (
                determinism * float(mods_on * modulation["determinism"]) ** ising_state
            )
        )

        y = ceiling * (
            floor
            + (1 - floor)
            / (1 + np.e ** (-new_determinism * (total_input - new_threshold)))
        )
        return y

    # first inputs will be interpreted as true inputs, while the last will be modulator inputs
    n_mods = len(modulation["modulator"])
    n_inputs = len(input_weights)
    n_nodes = n_inputs + n_mods

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    inp_state,
                    modulation_state,
                    state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for inp_state, modulation_state in product(
                utils.all_states(n_inputs), utils.all_states(n_mods)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def biased_sigmoid(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: tuple[float, ...] = (),
    determinism: float = 2.0,
    threshold: float = 0.0,
) -> np.ndarray:
    # A sigmoid unit that is biased in its activation by the last unit in the inputs.
    # The bias consists in a rescaling of the activation probability to make it more
    # in line with the biasing unit. The biasing unit is assumed to be the last one
    # of the inputs.

    def LogFunc(total_input, determinism, threshold):
        y = ceiling * (
            floor
            + (1 - floor) / (1 + np.e ** (-determinism * (total_input - threshold)))
        )
        return y

    n_nodes = len(input_weights)

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                (
                    LogFunc(
                        sum(
                            s[:-1]
                            * np.array([weight for weight in input_weights[:-1]])
                        ),
                        determinism,
                        threshold,
                    )
                    / input_weights[-1]
                    if s[-1] == 0
                    else 1
                    - (
                        1
                        - LogFunc(
                            sum(
                                s[:-1]
                                * np.array([weight for weight in input_weights[:-1]])
                            ),
                            determinism,
                            threshold,
                        )
                    )
                    / input_weights[-1]
                )
            ]
            for s in utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def gabor_gate(
    unit: "Unit",
    state: int,
    input_state: tuple[int, ...],
    *,
    floor: float = 0.0,
    ceiling: float = 1.0,
    preferred_states: tuple[tuple[int, ...], ...] = (),
) -> np.ndarray:
    # Ensure states are tuples
    prefs = [tuple(p) for p in preferred_states]
    anti_states = [tuple(int(1 - x) for x in p) for p in prefs]

    # Ensure ceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # if the unit is ON, its tpm should indicate that the past state was likely one of
    # its preferred_states; if OFF, one of its anti_states

    tpm = np.ones([2] * len(input_state)) * 0.5
    for s in utils.all_states(len(input_state)):
        if s in prefs:
            tpm[s] = ceiling
        elif s in anti_states:
            tpm[s] = floor

    return tpm


### UNITS
UNIT_FUNCTIONS = {
    "composite": composite,
    "gabor": gabor_gate,
    "sigmoid": sigmoid,
    "resonnator": resonnator,
    "sor": sor_gate,
    "mismatch_pattern_detector": mismatch_pattern_detector,
    "copy": copy_gate,
    "and": and_gate,
    "or": or_gate,
    "xor": xor_gate,
    "weighted_mean": weighted_mean,
    "democracy": democracy,
    "majority": majority,
    "mismatch_corrector": mismatch_corrector,
    "modulated_sigmoid": modulated_sigmoid,
    "stabilized_sigmoid": stabilized_sigmoid,
    "biased_sigmoid": biased_sigmoid,
}


# VALIDATION FUNCTIONS


# SEE VALIDATION FUNCTIONS BELOW
def validate_kwargs(kwargs):
    """Validates keyword arguments using respective validation functions."""
    for arg_name, arg_value in kwargs.items():
        # Get validation function for this argument, if provided
        validation_func = VALIDATION_FUNCTIONS.get(arg_name)

        if validation_func:
            # Validate argument using its validation function
            if not validation_func(kwargs["unit"], arg_value):
                raise ValueError(f"Invalid value for argument {arg_name}: {arg_value}")


def validate_unit(unit: "Unit", value: Any) -> bool:
    """Validates that value is an object of the class "Unit", and has the properties inputs, state and input_state."""
    if not hasattr(value, "inputs"):
        raise ValueError(f"value must have an 'inputs' attribute")
        return False
    return True


def validate_floor(unit: "Unit", value: Any) -> bool:
    """Validates that value is a float between 0 and 1, inclusive."""
    valid = True
    if not isinstance(value, float):
        raise ValueError(f"floor must be a float, but got {type(value)}")
        valid = False
    if not 0 <= value <= 1:
        if value < 0:
            value = 0.0
            print("Warning: floor value was less than 0, it has been set to 0")
            valid = False
        else:
            value = 1.0
            print("Warning: floor value was greater than 1, it has been set to 1")
            valid = False
    return valid


def validate_ceiling(unit: "Unit", value: Any) -> bool:
    """Validates that value is a float between 0 and 1, inclusive."""
    valid = True
    if not isinstance(value, float):
        raise ValueError(f"ceiling must be a float, but got {type(value)}")
        valid = False
    if not 0 <= value <= 1:
        if value < 0:
            value = 0.0
            print("Warning: ceiling value was less than 0, it has been set to 0")
            valid = False
        else:
            value = 1.0
            print("Warning: ceiling value was greater than 1, it has been set to 1")
            valid = False
    return valid


def validate_input_weights(unit: "Unit", value: Any) -> bool:
    """Validates that value is a tuple of floats between 0 and 1, inclusive, and its length is the same as unit.inputs."""
    valid = True
    if not isinstance(value, tuple):
        raise ValueError(f"input_weights must be a tuple, but got {type(value)}")
        valid = False
    if len(value) != len(unit.inputs):
        value = (1.0,) * len(unit.inputs)
        print(
            f"Warning: input_weights must be a tuple of length {len(unit.inputs)}, it has been set to {(1,) * len(unit.inputs)}"
        )
        valid = False
    for v in value:
        if not isinstance(v, float):
            raise ValueError(
                f"all values in the input_weights tuple must be floats, but got {type(v)}"
            )
            valid = False
    return valid


VALIDATION_FUNCTIONS = {
    "unit": validate_unit,
    "floor": validate_floor,
    "ceiling": validate_ceiling,
    "input_weights": validate_input_weights,
    # Add more validation functions for other arguments as needed
}
