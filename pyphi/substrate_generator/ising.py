# substrate_generator/ising.py
"""Utilities for implementing the Ising model."""

from . import utils


def energy(element, weights, state):
    """Return the local field acting on the given spin.

    This is the weighted sum of the spin's inputs, ``Σ_i w_iⱼ sᵢ`` for
    element ``j``; larger values bias the spin toward the ON (+1) state.
    """
    return utils.total_weighted_input(element, weights, state)


def probability(
    element,
    weights,
    state,
    temperature=1.0,
    field=0.0,
    constant_log_odds=False,
    **kwargs,
):
    """Return the probability that the given spin is ON (+1) at the next step.

    The binary ``state`` is first mapped to spins (0 → -1) and the local field
    ``E`` is passed through a logistic function, giving the Glauber-style
    activation probability ``σ((E - field) / temperature)``.

    Parameters
    ----------
    element : int
        Index of the spin whose activation probability is computed.
    weights : numpy.ndarray
        Connection weight matrix; ``weights[i, j]`` couples input ``i`` to
        element ``j``.
    state : Sequence[int]
        Binary state of the substrate (entries in ``{0, 1}``).
    temperature : float, optional
        Logistic temperature. Higher values flatten the response toward 0.5.
        Must be nonzero.
    field : float, optional
        External field subtracted from the energy before the logistic.
    constant_log_odds : bool, optional
        When ``True``, the temperature is scaled by the total input weight to
        ``element`` (``Σ_i weights[i, element]``), so that the log-odds ratio of
        ON to OFF when every input is ON does not depend on the total weight.

    Returns
    -------
    float
        The activation probability in [0, 1].

    Raises
    ------
    NotImplementedError
        If ``temperature`` is 0.
    """
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
    E = energy(element, weights, state)
    return utils.sigmoid(E, temperature=temperature, field=field)
