`pyphi.sweep` SIA rows now carry the selection margins: `partition_margin`,
`cause_state_margin`, `effect_state_margin`, and `effectively_tied` columns,
so near-tied cells can be flagged directly in the sweep DataFrame. Cells
computed under formalisms without margin reporting (IIT 3.0) carry `None`.
