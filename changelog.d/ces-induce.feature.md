Added `CauseEffectStructure.induce(distinctions)`, returning an
`InducedSubstructure` view: the selected distinctions plus exactly the
relations contained among them. Views expose `relation_closed`, and
`project_ces`/`plot_ces` now accept any relation-closed object (still
rejecting `PhiFold`).
