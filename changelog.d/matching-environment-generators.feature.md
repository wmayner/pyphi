Added environment generators for matching (`pyphi.matching.environment`):
`segment`, `point`, and `noise` build world distributions over a sensory
interface; `superpose` (independent OR) and `mixture` (weighted choice) compose
them; `sample` draws example stimuli with a seeded, isolated RNG. These let
`MatchingAnalysis` run without a hand-coded world distribution and reproduce the
manuscript's environments (E1/E2/E1b/E3).
