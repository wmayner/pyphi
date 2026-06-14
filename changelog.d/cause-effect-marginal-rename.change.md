Renamed the misleading ``System`` properties ``cause_tpm`` → ``cause_marginal``
and ``effect_tpm`` → ``effect_marginal`` (and the ``proper_*`` variants:
``proper_cause_tpm`` → ``proper_cause_marginal``, ``proper_effect_tpm`` →
``proper_effect_marginal``), along with the ``marginalization.py`` module
functions and internal helpers. ``cause_tpm`` returned a *posterior over past
states* (the Bayesian inversion of IIT 4.0 Eq. 4), not a transition probability
matrix; the new names reflect that these are *causal marginals* (Eq. 3/4), not
TPMs. The rename also covers ``TransitionSystem`` (which delegates the System
public surface). No change to any computed value.
