Scaffolded the N1 paper-reproduction acceptance suite
(`test/test_paper_reproduction.py`): published worked examples pinned against
the papers themselves and wired in as a CI gate, distinct from the
PyPhi-self-referential golden regression suite. First entry reproduces **IIT
4.0 (2023) Albantakis et al. Figure 1** — the Fig 1A logistic substrate is
reconstructed from its causal-model diagram and its Fig 1E system integrated
information reproduces the paper to two decimals (`φ_s(a)=0.04`,
`φ_s(aB)=0.17`, `φ_s(aBC)=0.13`), with `aB` confirmed a complex.
