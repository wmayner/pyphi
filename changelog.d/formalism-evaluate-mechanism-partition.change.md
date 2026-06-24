**Breaking (2.0):** Mechanism-partition integration moves from
``Subsystem.evaluate_partition`` into the active formalism's
``evaluate_mechanism_partition`` method. Each formalism now owns its
integration shape (which repertoires to feed the metric, what the metric
expects to receive). ``Subsystem.evaluate_partition`` becomes a thin
dispatcher routing to ``config.FORMALISM.evaluate_mechanism_partition``.

The change enforces ``compatible_metrics``: each formalism declares the
set of distance metrics whose calling shape it accepts. Configurations
like ``FORMALISM="IIT_4_0_2023"`` with ``REPERTOIRE_DISTANCE="EMD"`` —
which the legacy code accepted by silently swapping algorithms based on
metric name — now raise ``MetricNotCompatibleError`` at the dispatch
boundary. EMD on full repertoires computes a distribution distance
(IIT 3.0's mathematical object), not state-aware integrated information
(IIT 4.0's). The combination is mathematically inconsistent; rejecting
it at the formalism boundary surfaces it.

If you previously combined ``REPERTOIRE_DISTANCE = "EMD"`` (or any
distribution metric) with the default ``FORMALISM = "IIT_4_0_2023"``,
switch ``FORMALISM = "IIT_3_0"`` to keep the same numerical behavior.

A related test value (``noisy_selfloop_single`` SIA phi under EMD)
changed from 0.36 to 0.6868 because the IIT 3.0 SIA path now executes
correctly (the legacy hybrid IIT 4.0+EMD path silently took a different
algorithm).
