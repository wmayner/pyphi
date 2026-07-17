The thread backend now honors the short-circuit predicate on its
below-threshold sequential path and collects short-circuited results in
submission order, so truncated sweeps match sequential evaluation (and the
process backend) instead of varying with thread scheduling.
