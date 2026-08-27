`pointwise_mutual_information_vector()` now returns 0 where the log-ratio is undefined (`p = 0` or `q = 0`), as documented, instead of substituting the maximum finite float for infinite ratios.
