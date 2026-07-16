Serialization now round-trips every computed field the result objects carry.
Previously dropped silently: RIA `signed_phi` (negative preventative-cause
values were re-clamped to 0 on reload), `selectivity`, and `reasons`; MICE
purview ties (a never-computed tie state was also rewritten as "no ties");
IIT 3.0 and IIT 4.0 SIA `runner_up`; IIT 3.0 SIA `reasons`, `config`, and
`provenance`; CES `config` and `provenance`; actual-causation `node_labels`,
`reasons`, and alpha-tie sets; and `Transition.noise_background` (a noised
transition reloaded as frozen, changing α). Loaded IIT 3.0, AC, and CES
results no longer claim the loader's config and load time as their
provenance. Null actual-causation results (empty accounts) now serialize
instead of raising `TypeError`, `Transition` equality and hashing now include
`noise_background`, and `pyphi.load`/`pyphi.serialize.loads` reject files
written by a newer serialization format instead of silently dropping their
fields.
