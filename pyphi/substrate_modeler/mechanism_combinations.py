import numpy as np

from .utils import reshape_to_md


def selective(expanded_tpms):

    def get_selective(P):
        Q = np.array([np.abs(p - 0.5) for p in P])
        return P[np.argmax(Q)]

    return reshape_to_md(
        np.array(
            [
                [get_selective(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


def average(expanded_tpms):

    return reshape_to_md(
        np.array(
            [
                [np.mean(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


def maximal(expanded_tpms):

    return reshape_to_md(
        np.array(
            [
                [np.max(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


def first_necessary(expanded_tpms):

    def first_necessary(ap):
        # non-primary units boost activation probability as a function of the primary unit's activation probability
        primary = ap[0]

        if primary > 0.5:
            non_primary = np.prod([1 - p for p in ap[1:]])
            max_boost = 1 - primary
            boost = max_boost / (1 + np.e ** (-5 * (1 - non_primary - 0.5)))
            return primary + boost
        else:
            return primary

    return reshape_to_md(
        np.array(
            [
                [first_necessary(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


def integrator(expanded_tpms):

    def get_cumulated_probability(activation_probabilities):
        cumsum = np.sum(activation_probabilities)
        if cumsum > 1.0:
            return 1.0
        elif cumsum < 0.0:
            return 0.0
        else:
            return cumsum

    return reshape_to_md(
        np.array(
            [
                [get_cumulated_probability(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


def serial(expanded_tpms):

    def serial_func(P):
        remainder = 1
        for p in P:
            remainder -= p * remainder
        return 1 - remainder

    return reshape_to_md(
        np.array(
            [
                [serial_func(activation_probabilities)]
                for activation_probabilities in expanded_tpms
            ]
        )
    )


MECHANISM_COMBINATIONS = {
    "selective": selective,
    "average": average,
    "maximal": maximal,
    "first_necessary": first_necessary,
    "integrator": integrator,
    "serial": serial,
}
