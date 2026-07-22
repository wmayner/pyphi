`Substrate.potential_purviews` accepts `max_order`, bounding the purview
enumeration itself instead of enumerating the full powerset and filtering
afterward (the cache key includes the bound, so bounded and unbounded
results never alias). Scoped campaign planning, cost estimation, shard
execution, and collection derive the bound from the scope via
`AxisScope.order_bound()`, and explicit purview lists passed to
mechanism-level queries bound the enumeration to the largest given
purview. On large substrates this removes the dominant planning cost:
at 21 units a scoped walk enumerates hundreds of candidates instead of
~2 million per mechanism.
