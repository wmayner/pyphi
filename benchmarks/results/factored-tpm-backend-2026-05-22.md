# FactoredTPM storage-backend benchmark

- Seed: 2026
- Warmup trials: 5
- Measure trials: 50
- xarray available: True
- **Decision: `ndarray`**

Max xarray:ndarray ratio across (op, size): `3.518` (rule: xarray default iff <= 2.0).

## Binary networks

| n | op | backend | median (s) | p95 (s) |
|---|---|---|---|---|
| 3 | from_joint | ndarray | 4.022950e-05 | 4.296684e-05 |
| 3 | to_joint | ndarray | 5.290996e-06 | 5.541995e-06 |
| 3 | condition | ndarray | 3.575000e-05 | 3.889115e-05 |
| 3 | factor_access | ndarray | 8.349889e-08 | 1.250010e-07 |
| 3 | from_joint | xarray | 3.866600e-05 | 4.094170e-05 |
| 3 | to_joint | xarray | 6.333008e-06 | 6.731095e-06 |
| 3 | condition | xarray | 5.893700e-05 | 6.280609e-05 |
| 3 | factor_access | xarray | 2.500019e-07 | 3.145513e-07 |
| 5 | from_joint | ndarray | 6.372950e-05 | 6.758765e-05 |
| 5 | to_joint | ndarray | 9.958501e-06 | 1.243955e-05 |
| 5 | condition | ndarray | 6.479150e-05 | 6.804391e-05 |
| 5 | factor_access | ndarray | 8.399365e-08 | 1.065462e-07 |
| 5 | from_joint | xarray | 7.160450e-05 | 8.019375e-05 |
| 5 | to_joint | xarray | 1.150000e-05 | 1.483780e-05 |
| 5 | condition | xarray | 1.150000e-04 | 1.267706e-04 |
| 5 | factor_access | xarray | 2.920133e-07 | 3.750029e-07 |
| 8 | from_joint | ndarray | 1.485000e-04 | 1.555768e-04 |
| 8 | to_joint | ndarray | 2.943751e-05 | 3.100885e-05 |
| 8 | condition | ndarray | 1.316250e-04 | 1.455913e-04 |
| 8 | factor_access | ndarray | 1.249937e-07 | 1.250010e-07 |
| 8 | from_joint | xarray | 1.452915e-04 | 1.695037e-04 |
| 8 | to_joint | xarray | 2.983350e-05 | 3.773489e-05 |
| 8 | condition | xarray | 2.130830e-04 | 2.290062e-04 |
| 8 | factor_access | xarray | 3.330060e-07 | 4.391004e-07 |
| 10 | from_joint | ndarray | 2.705830e-04 | 3.037561e-04 |
| 10 | to_joint | ndarray | 6.756251e-05 | 7.564170e-05 |
| 10 | condition | ndarray | 1.655835e-04 | 1.800641e-04 |
| 10 | factor_access | ndarray | 8.300412e-08 | 1.250010e-07 |
| 10 | from_joint | xarray | 2.349590e-04 | 2.447104e-04 |
| 10 | to_joint | xarray | 7.137500e-05 | 8.279170e-05 |
| 10 | condition | xarray | 2.825835e-04 | 2.979503e-04 |
| 10 | factor_access | xarray | 2.920060e-07 | 4.159956e-07 |

## k=3 preview

| n | k | op | backend | median (s) | p95 (s) |
|---|---|---|---|---|---|
| 4 | 3 | to_joint | ndarray | 9.937503e-06 | 1.275000e-05 |
| 4 | 3 | condition | ndarray | 5.087500e-05 | 5.823355e-05 |
| 4 | 3 | to_joint | xarray | 1.035450e-05 | 1.206410e-05 |
| 4 | 3 | condition | xarray | 8.708300e-05 | 9.813315e-05 |
