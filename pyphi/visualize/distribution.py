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


def _distribution_frame(
    distributions, states=None, labels=None, lineplot_threshold=64, validate=True
):
    """Tidy frame of probabilities by state and series, plus the default
    state-space label (unit names) when bit-string states are inferred."""
    if validate and not all(np.allclose(d.sum(), 1, rtol=1e-4) for d in distributions):
        raise ValueError("a distribution does not sum to 1!")
    series = [pd.Series(distribution.flatten(d)) for d in distributions]
    first = series[0]
    if validate and not all((first.index == s.index).all() for s in series):
        raise ValueError("distribution indices do not match")
    n = log2(np.prod(first.shape))
    default_label = None
    if states is None:
        if n.is_integer() and len(first) <= lineplot_threshold:
            states = list(all_states_str(int(n)))
            default_label = string.ascii_uppercase[: int(n)]
        else:
            states = np.arange(len(first))
    if labels is None:
        labels = list(map(str, range(len(series))))
    frame = pd.concat(
        [
            pd.DataFrame({"probability": s, "state": states, "hue": [lab] * len(s)})
            for s, lab in zip(series, labels, strict=False)
        ]
    ).reset_index(drop=True)
    return frame, default_label


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
    data, default_label = _distribution_frame(
        distributions,
        states=states,
        labels=labels,
        lineplot_threshold=lineplot_threshold,
        validate=validate,
    )
    if label is None:
        label = default_label

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    n_points = len(data) // len(distributions)
    if n_points > lineplot_threshold:
        ax = _plot_distribution_line(data, ax, hue="hue", **kwargs)
    else:
        ax = _plot_distribution_bar(data, ax, label, hue="hue", **kwargs)

    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_xlabel("state", labelpad=12)
    ax.legend(bbox_to_anchor=(1.1, 1.05))

    return fig, ax


def _repertoire_comparison(system, sia):
    """Forward repertoires of the system and its partitioned counterpart,
    keyed by direction, then by "unpartitioned"/"partitioned"."""
    if config.formalism.iit.mechanism_phi_measure not in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
        raise NotImplementedError(
            "Only mechanism_phi_measure = "
            "GENERALIZED_INTRINSIC_DIFFERENCE or INTRINSIC_INFORMATION is supported"
        )
    systems = {
        "unpartitioned": system,
        "partitioned": system.apply_cut(sia.partition),
    }
    return {
        direction: {
            label: s.forward_repertoire(direction, s.node_indices, s.node_indices)
            for label, s in systems.items()
        }
        for direction in Direction.both()
    }


def plot_repertoires(system, sia, **kwargs):
    repertoires = _repertoire_comparison(system, sia)
    labels = ["unpartitioned", "partitioned"]
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
