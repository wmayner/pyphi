Scaffolded the N1 paper-reproduction acceptance suite
(`test/test_paper_reproduction.py`): published worked examples pinned against
the papers themselves and wired in as a CI gate, distinct from the
PyPhi-self-referential golden regression suite. Covers **IIT 4.0 (2023)
Albantakis et al. Figs 1, 2, 4 & 6C** (the Fig 1A logistic substrate
reconstructed from its causal-model diagram, its distinctions and a relation,
and the Fig 6C copy-ring) , **IIT 3.0 (2014) Oizumi et al. Fig 12** — the
A=OR/B=AND/C=XOR worked example reproducing `Φ = 1.92` (23/12) and its
six-concept constellation — and **Actual Causation (2019) Albantakis et al.
Fig 6** — the OR-AND causal account with `α = log₂(4/3) = 0.415` and
`log₂(9/8) = 0.170` bits — and **Gómez et al. (2020) Fig 3A** — the original
multi-valued p53-Mdm2 regulatory network (ternary p53 + binary nuclear/
cytoplasmic Mdm2; new `pyphi.examples.gomez_p53_mdm2_substrate`) reproducing
`Φ = 0.44` over 3 mechanisms under the paper's `AID` config (the suite's first
k>2 reproduction; also added as a k>2 IIT-3.0 golden). Each pinned value
reproduces its published figure to the paper's stated precision.
