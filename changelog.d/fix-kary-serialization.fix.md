Fixed serialization of multi-valued (k-ary) substrates: `pyphi.save` and
`serialize.dumps` crashed on any substrate with a non-binary unit. The
substrate encoding now stores the per-node conditional factors and state
space, which round-trips any alphabet sizes.
