`pyphi.campaign.prepare_ces()` accepts a `workloads` mapping, planning
shards against caller-supplied per-mechanism costs instead of the analytic
counting walk. Useful when measured runtimes describe a workload better than
the model does.
