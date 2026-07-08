``Substrate.num_states`` now returns the product of the per-node alphabet
sizes rather than ``2**size``, so it is correct for substrates with
non-binary (k-ary) units. Previously a substrate with any non-binary
alphabet reported a wrong state count.
