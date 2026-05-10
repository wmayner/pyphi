# visualize/distribution.py
"""Visualize distributions."""

import string
from math import log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from pyphi import config
from pyphi import distribution
from pyphi import utils
from pyphi.direction import Direction


def all_states_str(*args, **kwargs):
    """Return all states as bit strings."""
    for state in utils.all_states(*args, **kwargs):
        yield "".join(map(str, state))


def _plot_distribution_bar(
    data,
    ax,
    label,
    show_label=True,
    label_font="monospace",
    label_color="black",
    **kwargs,
):
    sb.barplot(data=data, x="state", y="probability", ax=ax, **kwargs)

    # Set xtick labels rotation and alignment using correct matplotlib API
    xtick_pad = 6
    xtick_length = 6
    ax.tick_params(axis="x", pad=xtick_pad, length=xtick_length)
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
        fontname=label_font,
    )

    # Add state label
    if show_label:
        ax.annotate(
            str(label) if label is not None else "",
            xy=(-0.5, 0),
            xycoords="data",
            xytext=(0, -(xtick_pad + xtick_length)),
            textcoords="offset points",
            annotation_clip=False,
            rotation=90,
            ha="right",
            va="top",
            fontname=label_font,
            color=label_color,
        )
    return ax


def _plot_distribution_line(data, ax, **kwargs):
    sb.lineplot(data=data, x="state", y="probability", ax=ax, **kwargs)
    return ax


def plot_distribution(
    *distributions,
    states=None,
    label=None,
    figsize=(9, 3),
    fig=None,
    ax=None,
    lineplot_threshold=64,
    title=None,
    y_label="Pr(state)",
    validate=True,
    labels=None,
    **kwargs,
):
    """Plot a distribution over states.

    Arguments:
        d (array_like): The distribution. If no states are provided, must
            have length equal to a power of 2. Multidimensional distributions
            are flattened with ``pyphi.distribution.flatten()``.

    Keyword Arguments:
        states (Iterable | None): The states corresponding to the
            probabilities in the distribution; if ``None``, infers states from
            the length of the distribution and assumes little-endian ordering.
        **kwargs: Passed to ``sb.barplot()``.
    """
    if validate and not all(np.allclose(d.sum(), 1, rtol=1e-4) for d in distributions):
        raise ValueError("a distribution does not sum to 1!")

    defaults = {}
    # Overrride defaults with keyword arguments
    kwargs = {**defaults, **kwargs}

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    distributions = [pd.Series(distribution.flatten(d)) for d in distributions]
    d = distributions[0]

    if validate and not all(
        (distributions[0].index == d.index).all() for d in distributions
    ):
        raise ValueError("distribution indices do not match")

    N = log2(np.prod(d.shape))
    if states is None:
        if N.is_integer() and len(d) <= lineplot_threshold:
            N = int(N)
            states = list(all_states_str(N))
            if label is None:
                label = string.ascii_uppercase[:N]
        else:
            states = np.arange(len(d))

    if labels is None:
        labels = list(map(str, range(len(distributions))))

    data = pd.concat(
        [
            pd.DataFrame({"probability": d, "state": states, "hue": [label] * len(d)})
            for d, label in zip(distributions, labels, strict=False)
        ]
    ).reset_index(drop=True)

    if len(d) > lineplot_threshold:
        ax = _plot_distribution_line(data, ax, hue="hue", **kwargs)
    else:
        ax = _plot_distribution_bar(data, ax, label, hue="hue", **kwargs)

    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_xlabel("state", labelpad=12)
    ax.legend(bbox_to_anchor=(1.1, 1.05))

    return fig, ax


def plot_repertoires(system, sia, **kwargs):
    if config.formalism.iit.repertoire_measure not in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
        raise NotImplementedError(
            "Only repertoire_measure = "
            "GENERALIZED_INTRINSIC_DIFFERENCE or INTRINSIC_INFORMATION is supported"
        )
    cut_system = system.apply_cut(sia.partition)

    labels = ["unpartitioned", "partitioned"]
    systems = dict(zip(labels, [system, cut_system], strict=False))
    repertoires = {
        direction: {
            label: s.forward_repertoire(direction, s.node_indices, s.node_indices)
            for label, s in systems.items()
        }
        for direction in Direction.both()
    }

    fig = plt.figure(figsize=(12, 9))
    axes = fig.subplots(2, 1)
    for ax, direction in zip(axes, Direction.both(), strict=False):
        plot_distribution(
            repertoires[direction][labels[0]],
            repertoires[direction][labels[1]],
            validate=False,
            title=str(direction),
            labels=labels,
            ax=ax,
            **kwargs,
        )
    fig.tight_layout(h_pad=0.5)
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.1, 1.1))
    return fig, axes, repertoires
