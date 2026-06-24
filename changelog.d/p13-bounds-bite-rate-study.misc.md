Ran the P13 SP2 bite-rate study (``benchmarks/bounds_bite_study.py``) to decide
whether the certified Zaeemzadeh (2024) bounds could prune the Φ search. They
do **not** bite in their certified domain — a 0% useful prune rate across the
binary goldens (n ≤ 4): the Theorem-1 purview cap ``|M||Z|`` (1–9) dwarfs
realized small-φ (< 1), and the ``n(n-1)`` system bound (valid only for n ≥ 2)
exceeds the observed maximum φ_s. So the search-integration pruning is not
built; the bounds module and its B1 runtime assertions are the final P13
deliverable. The study script is committed for reproducibility (e.g. to
re-measure at larger n).
