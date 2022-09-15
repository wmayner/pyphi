# -*- coding: utf-8 -*-
# network_generator/ising.py

import numpy as np

from ..utils import all_states
from . import utils


def sigmoid(E, T):
    return 1 / (1 + np.exp(-E / T))


def inverse_sigmoid(p, sum_w, field):
    return np.log(p / (1 - p)) / (sum_w - field)


def energy(element, weights, state, field):
    """Return the energy associated with the given spin."""
    return utils.input_weight(element, weights, state) + field


def probability(
    element,
    weights,
    state,
    temperature=1.0,
    field=0.0,
    constant_log_odds=False,
):
    """Return the probability that the given spin flips."""
    if temperature == 0:
        raise NotImplementedError("temperature is 0: need to decide correct behavior")

    if constant_log_odds:
        total_input_weight = weights[:, element].sum()
        if total_input_weight != 0:
            # Scale temperature by total input weight
            # This has the effect of ensuring that the ratio of log-odds ON to OFF, given
            # all inputs to a node are ON, is constant regardless of total weight
            temperature = temperature * total_input_weight

    state = utils.binary2spin(state)
    E = energy(element, weights, state, field)
    return sigmoid(E, temperature)


###############################################################################
# Plotting
###############################################################################

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sb

    def plot_sigmoid(x, temperature):
        y = sigmoid(x, temperature)
        ax = sb.lineplot(x=x, y=y, linewidth=3)
        ax.set_title(f"T = {temperature}")
        ax.vlines(x=0, ymin=0, ymax=1, color="grey", linewidth=1)
        return ax

    def plot_inputs(data, x, y, label, ax=None, sep=0.015):
        ax = sb.scatterplot(data=data, x=x, y=y, ax=ax, s=100, color="red", alpha=0.25)
        seen = dict()
        for _, row in data.iterrows():
            if row[x] in seen:
                seen[row[x]] += sep
            else:
                seen[row[x]] = sep
            plt.text(x=row[x], y=row[y] + seen[row[x]], s=row[label])
        return ax

    def plot(weights, temperature, field, N=None):
        if N is None:
            N = weights.shape[0]
        else:
            weights = weights[:N, :N]

        spin = 0
        energies = []
        probabilities = []
        states = list(all_states(N))
        for state in states:
            spin_state = binary2spin(state)
            # Compute probability that i'th spin is "on" in the next micro-timestep
            E = energy(spin, weights, spin_state, field)
            energies.append(E)
            probabilities.append(sigmoid(E, temperature))

        data = pd.DataFrame(
            {
                "energy": energies,
                "probability": probabilities,
                "state": ["".join(map(str, state)) for state in states],
            }
        )

        limit = np.max(np.abs(data["energy"]))
        x = np.linspace(-limit, limit, num=200)

        fig = plt.figure(figsize=(15, 6))
        ax = plot_sigmoid(x, temperature=temperature)
        ax = plot_inputs(
            data=data, x="energy", y="probability", label="state", ax=ax, sep=0.05
        )

        return fig

except (ImportError, DeprecationWarning):
    pass
