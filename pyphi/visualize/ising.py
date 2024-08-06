# visualize/__init__.py
"""Visualize the Ising model."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from ..utils import all_states
from ..network_generator import utils
from ..network_generator.ising import energy


def plot_sigmoid(x, temperature=1.0, field=0.0):
    y = utils.sigmoid(x, temperature=temperature, field=field)
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


def plot(weights, temperature, field, N=None, spin=0):
    if N is None:
        N = weights.shape[0]
    else:
        weights = weights[:N, :N]

    energies = []
    probabilities = []
    states = list(all_states(N))
    for state in states:
        spin_state = utils.binary2spin(state)
        # Compute probability that i'th spin is "on" in the next micro-timestep
        E = energy(spin, weights, spin_state)
        energies.append(E)
        probabilities.append(utils.sigmoid(E, temperature=temperature, field=field))

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
    ax = plot_sigmoid(x, temperature=temperature, field=field)
    ax = plot_inputs(
        data=data, x="energy", y="probability", label="state", ax=ax, sep=0.05
    )

    return fig
