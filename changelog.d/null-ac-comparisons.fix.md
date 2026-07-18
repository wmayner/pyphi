Null actual-causation results support `==`, `!=`, ordering, and `to_pandas()`
(previously these raised `TypeError`), and `find_actual_cause`/
`find_actual_effect` return a null `CausalLink` (reason
`NO_POSITIVE_ALPHA`) instead of a bare empty list when no purview yields
positive α.
