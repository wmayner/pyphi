Tied distinction readings now resolve by the S1 rule itself rather than its
purview-size proxy: among the congruent readings of each distinction, the
combination that maximizes the structure integrated information Φ is selected
(computed jointly across distinctions via the analytical Σφ_r, since a
reading's relation support depends on the other distinctions' readings), with
residual Φ-ties closed deterministically. On the example fixtures the proxy
and the exact rule agree, so no golden values change; on generic substrates
the proxy frequently understated Φ (145 of the random 3-node cases swept,
by up to ~13%). Beyond 4096 tied combinations a greedy per-distinction pass
approximates the joint maximum with a warning.

Two published 2023-paper figure reproductions change under the exact rule:
Fig 6D's Φ becomes 12395 (published: 11452) and Fig 7B's relation count and Φ
become 13498 and 19.32 (published: 13111 and 18.55) — φ_s and the distinction
counts still match the figures exactly. The published values embed the old
enumeration-order tie resolution, which is relabeling-dependent and
sub-maximal under the S1 supplement's own rule.
