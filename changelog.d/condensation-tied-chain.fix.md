Fixed the exclusion cascade dropping φₛ-tied candidates that overlapped only
*excluded* rivals: within a tied tier, the cascade accepted a single Φ-maximal
winner per overlap-connected component and discarded every other member — on
chain topologies, a candidate disjoint from the winner was silently neither
accepted nor recorded as failed. Tied candidates are now resolved recursively
per Marshall et al. (2023, Algorithm A1) and the IIT 4.0 S1 tie supplement:
tied candidates with no overlap conflict are complexes; overlap conflicts
escalate to Φ; and a Φ tie fails exclusion only among candidates that overlap
*each other* (a Φ tie between disjoint candidates accepts both).
