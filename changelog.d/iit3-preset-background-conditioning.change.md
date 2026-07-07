`presets.iit3` now sets `background_conditioning="CONDITION_CURRENT_STATE"`,
so IIT 3.0 analyses of systems that are proper subsets of their substrate
reproduce published PyPhi 1.x results (background units fixed at their
observed state on the cause side, rather than causally marginalized per
IIT 4.0 Eq. 4). Full-substrate analyses are unaffected — the conventions
coincide when there is no background.
