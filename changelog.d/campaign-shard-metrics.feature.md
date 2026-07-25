Campaign task outputs now record what the task cost to run.
`CampaignTaskOutput.metrics` carries wall and CPU seconds, cache
hit/miss/eviction counts, and the shard's planned units, payload kind, and
memory request — enough to recalibrate `pyphi.cost` against a campaign's own
observed runtimes.
