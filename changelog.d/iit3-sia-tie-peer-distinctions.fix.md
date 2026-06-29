The IIT 3.0 system irreducibility analysis now attaches the unpartitioned
cause-effect structure to every phi-tied partition in its tie set, not only
to the winning partition. The distinctions are a property of the system and
state rather than of any partition, so all tied alternatives share them.
This keeps tied alternatives comparable under ``__eq__`` (which compares
distinctions), so a tie-tolerant comparison matches whichever co-optimal
partition the lexicographic tiebreaker selects.
