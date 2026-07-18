`pyphi_config.yml` loading now rejects fields nested under the wrong layer
section (previously a misfiled field was silently routed to its owning layer),
and rejects unrecognized keys under `formalism`.
