`state_by_state2state_by_node`, `to_multidimensional`, and
`be2le_state_by_state` now raise a clear `ValueError` for TPMs whose state
count is not a power of two, instead of silently truncating the node count
and producing wrong results.
