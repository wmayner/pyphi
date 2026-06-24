# visualize/ising.py
"""Visualize the Ising model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from pyphi.substrate_generator import utils
from pyphi.substrate_generator.ising import energy
from pyphi.utils import all_states


def plot_sigmoid(x, temperature=1.0, field=0.0):
    y = utils.sigmoid(x, temperature=temperature, field=field)
    ax = sb.lineplot(x=x, y=y, linewidth=3)
    ax.set_title(f"T = {temperature}")
    ax.vlines(x=0, ymin=0, ymax=1, color="grey", linewidth=1)
    return ax


def plot_inputs(data, x, y, label, ax=None, sep=0.015):
    ax = sb.scatterplot(data=data, x=x, y=y, ax=ax, s=100, color="red", alpha=0.25)
    seen = {}
    for _, row in data.iterrows():
        if row[x] in seen:
            seen[row[x]] += sep
        else:
            seen[row[x]] = sep
        plt.text(x=row[x], y=row[y] + seen[row[x]], s=row[label])
    return ax


def _state_energies(weights, temperature, field, N=None, spin=0):
    """Energy and activation probability of one spin across all states."""
    if N is None:
        N = weights.shape[0]
    else:
        weights = weights[:N, :N]
    rows = []
    for state in all_states(N):
        spin_state = utils.binary2spin(state)
        # Probability that the spin is "on" in the next micro-timestep.
        e = energy(spin, weights, spin_state)
        rows.append(
            {
                "energy": e,
                "probability": utils.sigmoid(e, temperature=temperature, field=field),
                "state": "".join(map(str, state)),
            }
        )
    return pd.DataFrame(rows)


def plot(weights, temperature, field, N=None, spin=0):
    data = _state_energies(weights, temperature, field, N=N, spin=spin)
    limit = np.max(np.abs(data["energy"]))
    x = np.linspace(-limit, limit, num=200)
    fig = plt.figure(figsize=(15, 6))
    ax = plot_sigmoid(x, temperature=temperature, field=field)
    ax = plot_inputs(
        data=data, x="energy", y="probability", label="state", ax=ax, sep=0.05
    )
    return fig
