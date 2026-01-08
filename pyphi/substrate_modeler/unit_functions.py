import pyphi
import numpy as np
from itertools import product
from typing import Tuple, List, Any, Iterable

from .utils import map_to_floor_and_ceil

# UNIT FUNCTIONS


def composite(
    unit: "Unit",
):
    return True


def sigmoid(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism: float = 5.0,
    threshold: float = 0.0,
    ising: bool = True,
):
    # validate all kwargs
    validate_kwargs(locals())

    def LogFunc(state, input_weights, determinism, threshold):

        if ising:
            state = tuple([s * 2 - 1 for s in state])
        total_input = sum(state * np.array([input_weights[n] for n in range(n_nodes)]))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return map_to_floor_and_ceil(y, floor, ceiling)

    n_nodes = len(input_weights)

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def sor_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: Tuple[Tuple[int]] = None,
    selectivity: float = 2.0,
):

    # get state of

    # Ensure states are tuples
    pattern_selection = list(map(tuple, pattern_selection))

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        print(
            "Selectivity for SOR gates must be bigger than 1, adjusting to inverse of value given."
        )
        selectivity = 1 / selectivity

    # Ensure unit has input state
    if unit.input_state is None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * floor

    # if the input state matches a pattern in the patternselction, activation probability given that state is ceiling, otherwise, it is increased by a fraction of the difference between floor and ceiling (given by selectivity)
    pattern = floor + (ceiling - floor) / selectivity
    for state in pattern_selection:
        tpm[state] = pattern
    if unit.input_state in pattern_selection:
        tpm[unit.input_state] = ceiling
    else:
        tpm[unit.input_state] = 0.0

    return tpm


def resonnator(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism=None,
    threshold=None,
    weight_scale_mapping=None,
):

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure unit has input state
    if unit.input_state is None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.mechanism, unit.label
            )
        )
        unit.input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.state((0,))

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * floor

    # if the input state matches a pattern in the patternselction, activation probability given that state is ceiling, otherwise, it is increased by a fraction of the difference between floor and ceiling (given by selectivity)

    n_nodes = len(input_weights)
    unit_state = unit.state[0]

    # alter weight to make it push unit towards its state, weighted using weight_scale_mapping
    w = [
        (
            input_weights[i] * weight_scale_mapping[(unit_state, input_state)]
            if input_state == unit_state
            else -input_weights[i] * weight_scale_mapping[(unit_state, input_state)]
        )
        for i, input_state in enumerate(unit.input_state)
    ]

    def resonnatorFunc(state, input_weights, determinism, threshold, unit_state):
        # make state interpreted as ising
        state = [2 * s - 1 for s in state]

        total_input = np.sum(state * np.array(w))
        y = 1 / (1 + np.e ** (-determinism * (total_input - threshold)))
        return y

    # producing transition probability matrix
    tpm = np.array(
        [
            [resonnatorFunc(state, input_weights, determinism, threshold, unit_state)]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    # make it between floor and ceiling
    tpm = map_to_floor_and_ceil(tpm, floor, ceiling)

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def mismatch_pattern_detector(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    pattern_selection: Tuple[Tuple[int]] = None,
    selectivity: float = 1.0,
):
    # This mechanism is selective to certain inputs (i.e. they turn it ON with P=ceiling, while the remaining possible input patterns turn it OFF with P=floor). However, it's selectivity (probability of turning on) depends the state of the unit: if the unit is already in the state that matches the pattern, then the effect of the inputs is reduced by the selectivity factor. That is, if the unit is ON, and one of its patterns are on its inputs, then the probability that *this mechanism* will turn keep it ON in the next step i P=0.5+(ceiling-0.5)/selectivity.
    # The mechanism is supposed to mimic short-term plasticity mechanisms (or other short term adaptive changes in the function of cells) that make them strongly responsive to mismatching/unpredicted inputs, but weakly coupled to inputs that provide no new information (inputs that match the predicted state of things).

    # Ensure states are tuples
    pattern_selection = list(map(tuple, pattern_selection))

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    if floor is None:
        floor = 1.0 - ceiling

    # Ensure selectivity is larger than 1
    if not selectivity > 1:
        print(
            "Selectivity for SOR gates must be bigger than 1, adjusting to inverse of value given."
        )
        selectivity = 1 / selectivity

    # Ensure unit has input state
    if unit.input_state is None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # Check if the unit is ON
    if unit.state == (1,):
        # since it is ON, it will only respond strongly if a non-pattern is on its inputs
        P_pattern = 0.5 + (ceiling - 0.5) / selectivity
        P_no_pattern = floor
    else:
        # since it is OFF, it will only respond strongly if a pattern is on its inputs
        P_pattern = ceiling
        P_no_pattern = 0.5 - (0.5 - floor) / selectivity

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    N = len(unit.input_state)
    tpm = np.ones([2] * N)

    for state in pyphi.utils.all_states(N):
        if state in pattern_selection:
            tpm[state] = P_pattern
        else:
            tpm[state] = P_no_pattern

    return tpm


def copy_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones([2]) * floor
    tpm[1] = ceiling
    return tpm


def and_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * floor
    tpm[(1, 1)] = ceiling
    return tpm


def or_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * ceiling
    tpm[(0, 0)] = floor
    return tpm


def xor_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    tpm = np.ones((2, 2)) * floor
    tpm[(0, 1)] = ceiling
    tpm[(1, 0)] = ceiling
    return tpm


def weighted_mean(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    weights: List[float] = [],
):

    weights = [w / np.sum(weights) for w in weights]
    N = len(weights)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        weighted_mean = (
            sum([(1 + w * (s * 2 - 1)) / 2 for w, s in zip(weights, state)]) / N
        )
        tpm[state] = weighted_mean * (ceiling - floor) + floor

    return tpm


def democracy(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):

    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        avg_vote = np.mean(state)
        tpm[state] = avg_vote * (ceiling - floor) + floor

    return tpm


def majority(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
):

    N = len(unit.inputs)

    tpm = np.ones((2,) * N)
    for state in pyphi.utils.all_states(N):
        avg_vote = round(np.mean(state))
        tpm[state] = avg_vote * (ceiling - floor) + floor

    return tpm


def mismatch_corrector(
    unit: "Unit", floor: float = 0.0, ceiling: float = 1.0, bias: float = 0.0
):

    # Ensure unit has input state
    if unit.input_state is None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # Ensure unit has state
    if bias > 1:
        print("bias must be below 1, setting to 1.".format(unit.label))
        bias = 1

    # Ensure that there is only one unit
    if len(unit.inputs) > 1:
        print(
            "Unit {} has too many inputs for mechanism of typ {}. Using only first input.".format(
                unit.label
            )
        )
        unit.set_inputs(tuple(unit.inputs[[0]]))

    # check whether state of unit matches its input, and create tpm accordingly
    if unit.state == unit.input_state:
        tpm = np.ones([2]) * 0.5 - (unit.state[0] * 2 - 1) * bias * 0.5
    else:
        tpm = np.array([floor, ceiling])

    return tpm


def modulated_sigmoid(
    unit: "Unit",
    input_weights: List[float],
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
    determinism: float = 2.0,
    threshold: float = 0.0,
):
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float}
    # modulation will update the sigmoid function as indicated by the functions, and depending on whether the unit is ON or OFF

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    def LogFunc(
        input_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(input_state * np.array([weight for weight in weights]))
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

    unit_state = (
        unit.state[0] * 2 - 1
    )  # making unit state "ising" rather than binary, to make modulation symmetric

    # producing transition probability matrix
    tpm = np.array(
        [
            [
                LogFunc(
                    input_state,
                    modulation_state,
                    unit_state,
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for modulation_state, input_state in product(
                pyphi.utils.all_states(n_mods), pyphi.utils.all_states(n_inputs)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def stabilized_sigmoid(
    unit: "Unit",
    input_weights: list,
    determinism: float,
    threshold: float,
    modulation: dict,
    floor: float = 0.0,
    ceiling: float = 1.0,
):
    # modulation must be a dict like {'modulator': tuple(index), 'threshold': float, 'determinism': float}
    # modulation will update the sigmoid function as indicated by the functions, and depending on whether the unit is ON or OFF

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    def LogFunc(
        input_state, modulation_state, unit_state, weights, determinism, threshold
    ):
        total_input = sum(input_state * np.array([weight for weight in weights]))

        # The modulation should work in such a way as to always "stabilize" the current state of the unit, but it should be stronger, the more modulation is "active"

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
                    input_state,  # tuple(s * 2 - 1 for s in input_state),
                    modulation_state,
                    unit.state[0],
                    input_weights,
                    determinism,
                    threshold,
                )
            ]
            for input_state, modulation_state in product(
                pyphi.utils.all_states(n_inputs), pyphi.utils.all_states(n_mods)
            )
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def biased_sigmoid(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    input_weights: Tuple[float] = None,
    determinism: float = 2.0,
    threshold: float = 0.0,
):
    # A sigmoid unit that is biased in its activation by the last unit in the inputs.
    # The bias consists in a rescaling of the activation probability to make it more in line with the biasing unit. The biasing unit is assumed to be the last one of the inputs.
    # For example, if the biased unit is OFF, the sigmoid activation probability is divided by the factor given in the last value of input_weights. If the unit is ON, 1 - the activation probability is divided by the factor (in essence reducing the probability that it will NOT activate).

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
                            state[:-1]
                            * np.array([weight for weight in input_weights[:-1]])
                        ),
                        determinism,
                        threshold,
                    )
                    / input_weights[-1]
                    if state[-1] == 0
                    else 1
                    - (
                        1
                        - LogFunc(
                            sum(
                                state[:-1]
                                * np.array([weight for weight in input_weights[:-1]])
                            ),
                            determinism,
                            threshold,
                        )
                    )
                    / input_weights[-1]
                )
            ]
            for state in pyphi.utils.all_states(n_nodes)
        ]
    )

    return tpm.reshape([2] * n_nodes + [1], order="F").astype(float)


def gabor_gate(
    unit: "Unit",
    floor: float = 0.0,
    ceiling: float = 1.0,
    preferred_states: Tuple[Tuple[int]] = None,
):

    # Ensure states are tuples
    preferred_states = list(map(tuple, preferred_states))
    anti_states = [tuple([int(1 - s) for s in state]) for state in preferred_states]

    # Ensure nceiling is not more than 1
    if ceiling > 1:
        ceiling = 1.0

    # Ensure unit has input state
    if unit.input_state is None:
        print(
            "input state not given for {} unit {}. Setting to all off.".format(
                unit.params["mechanism"], unit.label
            )
        )
        unit.set_input_state((0,) * len(unit.inputs))

    # Ensure unit has state
    if unit.state is None:
        print("State not given unit {}. Setting to off.".format(unit.label))
        unit.set_state((0,))

    # if the unit is ON, its tpm should indicate that the past state was likely one of its preferred_states
    # if the unit is OFF, its tpm should indicate that the past state was likely one of its anti_states

    # the tpm is uniform for all states except the input state (i.e. short term plasticity)
    tpm = np.ones([2] * (len(unit.input_state))) * 0.5
    for input_state in pyphi.utils.all_states(len(unit.inputs)):
        if input_state in preferred_states:
            tpm[input_state] = ceiling
        elif input_state in anti_states:
            tpm[input_state] = floor
    """
    if unit.state[0]:
        for input_state in pyphi.utils.all_states(len(unit.inputs)):
            if input_state in preferred_states:
                tpm[input_state] = ceiling
            elif input_state in anti_states:
                tpm[input_state] = floor
    else:
        for input_state in pyphi.utils.all_states(len(unit.inputs)):
            if input_state in preferred_states:
                tpm[input_state] = floor
            elif input_state in anti_states:
                tpm[input_state] = ceiling
                """
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
    valid = True
    if not hasattr(value, "inputs"):
        raise ValueError(f"value must have an 'inputs' attribute")
        valid = False
    if not hasattr(value, "state"):
        value.state = (0,)
        print("Warning: state property was not present, it has been set to (0,)")
        valid = False
    if not hasattr(value, "input_state"):
        value.input_state = (0,) * len(value.inputs)
        print(
            "Warning: input_state property was not present, it has been set to (0,) * len(inputs)"
        )
        valid = False
    return valid


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
    if not isinstance(value, Iterable):
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
