`TransitionSystem.save()` previously delegated to the underlying `System`, so
loading the file silently returned a `System` and lost the transition data
(before/after states, cause/effect sets, direction). `TransitionSystem` now
has its own serialization schema and round-trips faithfully.
