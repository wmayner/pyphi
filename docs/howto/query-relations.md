---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Query relational structure

In IIT 4.0, a cause-effect structure is a set of **distinctions** together with the
**relations** among them. A relation is any set of two or more distinctions whose
purviews overlap in a congruent way, and its integrated information φ_r contributes
to the system's Φ. Relations are the combinatorial bottleneck: a structure with
`|D|` distinctions can have up to `2^{|D|} − 1` relations, and `|D|` itself can reach
`2^n − 1` for `n` units. The IIT 4.0 paper's Figure 6D — 27 distinctions — has
**1,537,080 relations**, and its Φ is dominated by the relational part.

But the relation set is fully determined by the distinctions: each distinction
contributes a set of state-tagged purview units (its "atoms") and a φ density, and
every question about the relations is a function of those. PyPhi 2.0 exposes that
fact as a query interface. A relation set answers structural questions — sums,
counts, spectra, histograms, the strongest few, unbiased samples — **without ever
enumerating the relations**, in closed form and in milliseconds, at any scale.

This guide walks through the interface on a small network where we can also
enumerate the relations directly and check that the two agree. For the theory
of distinctions and relations, see {doc}`../theory/distinctions-and-relations`.

```{code-cell} python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pyphi
from pyphi import examples
from pyphi.conf import config
from pyphi.formalism import iit4
from pyphi.measures.distribution import (
    resolve_mechanism_measure,
    resolve_system_measure,
)
from pyphi.relations import AnalyticalRelations, ConcreteRelations

sns.set_theme(style="whitegrid", context="notebook")
config.progress_bars = False  # keep the output readable


# Compute a cause-effect structure (distinctions and relations) for a system,
# using the measures selected by the active configuration.
def compute_ces(system):
    return iit4.ces(
        system,
        system_measure=resolve_system_measure(config.formalism.iit.system_phi_measure),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    )
```

## 1. A cause-effect structure and its relations

We use `grid3`, a three-unit network small enough to enumerate every relation
explicitly, yet rich enough to have relations of several degrees.

```{code-cell} python
ces = compute_ces(examples.grid3_system())
ces
```

## 2. Two backends, one set of answers

A relation set is computed by one of two backends, selected with
`config.formalism.iit.relation_computation`:

- **`CONCRETE`** builds every relation as an explicit object. Exact and fully
  general, but the output grows exponentially with the number of distinctions.
- **`ANALYTICAL`** stores only the distinction set and answers queries in closed
  form. It never materializes a relation.

Both answer the same questions. Here we build each backend from the same
distinctions and confirm the three summary quantities agree.

```{code-cell} python
concrete = ConcreteRelations(ces.relations)          # the enumerated relations
analytical = AnalyticalRelations(ces.distinctions)   # the closed-form view

pd.DataFrame(
    {
        "concrete (enumerated)": [
            concrete.num_relations(),
            concrete.sum_phi(),
            concrete.apportioned_sum_phi(),
        ],
        "analytical (closed form)": [
            analytical.num_relations(),
            analytical.sum_phi(),
            analytical.apportioned_sum_phi(),
        ],
    },
    index=["num_relations", "Σφ_r", "Σφ_r / |r|"],
)
```

Every query below works on both backends. On `ConcreteRelations` it iterates the
enumerated relations; on `AnalyticalRelations` it is a closed-form expression over
the distinctions. We run the analytical backend from here on and check it against
the enumeration where it is cheap to do so.

## 3. Closed-form structural queries

### Degree spectrum

The **degree** of a relation is its number of relata. `degree_spectrum()` returns,
for each degree, how many relations have it and how much φ_r they contribute —
exactly, without listing a single relation. (Degree 1 is the self-relations: a distinction's
cause overlapping its own effect.)

```{code-cell} python
spectrum = analytical.degree_spectrum()

spectrum_df = pd.DataFrame(
    [
        {"degree": degree, "count": count, "sum_phi_r": sum_phi}
        for degree, (count, sum_phi) in spectrum.items()
    ]
)
spectrum_df
```

```{code-cell} python
long = spectrum_df.melt(
    id_vars="degree", value_vars=["count", "sum_phi_r"], var_name="quantity"
)
grid = sns.catplot(
    data=long,
    x="degree",
    y="value",
    col="quantity",
    kind="bar",
    height=3.5,
    aspect=1.1,
    sharey=False,
    color="#4C72B0",
)
grid.set_titles("{col_name}")
grid.figure.suptitle("grid3 relation degree spectrum (exact)", y=1.05)
plt.show()
```

### Moments, extremes, and faces

The φ_r distribution's mean and standard deviation, the single strongest relation's
φ_r, and the total number of relation *faces* are all closed-form. The `basic`
statistics come from the count and the first two moments; `max_phi` scans only pairs
and self-relations (the maximum is provably attained at degree ≤ 2); `num_faces`
counts faces by Möbius inversion over the individual causes and effects.

```{code-cell} python
mean, std = analytical.phi_mean_std()
print(f"φ_r mean ± std:  {mean:.4f} ± {std:.4f}")
print(f"max φ_r:         {analytical.max_phi():.4f}")
print(f"Σφ_r² (2nd mom): {analytical.sum_phi_moment(2):.4f}")
print(f"total faces:     {analytical.num_faces()}")

# these match the enumeration exactly
assert analytical.max_phi() == max(float(r.phi) for r in concrete)
assert analytical.num_faces() == concrete.num_faces()
print("\n(analytical == concrete enumeration)")
```

### The exact φ_r histogram

`phi_histogram()` returns the exact distribution of φ_r values over all relations,
grouped at the configured numerical precision. On a million-relation structure this
is the same closed-form computation and just as fast.

```{code-cell} python
hist = analytical.phi_histogram()
hist_df = (
    pd.DataFrame({"phi_r": list(hist.keys()), "count": list(hist.values())})
    .sort_values("phi_r")
    .reset_index(drop=True)
)
hist_df["φ_r"] = hist_df["phi_r"].map(lambda v: f"{v:.3f}")

ax = sns.barplot(data=hist_df, x="φ_r", y="count", color="#55A868")
ax.set_title("grid3 exact φ_r histogram")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print(f"distinct φ_r values: {len(hist)}")
print(f"counts sum to num_relations: {sum(hist.values())} == {analytical.num_relations()}")
```

### The binding matrix

`binding_matrix()` is the relational skeleton of the structure: entry `(a, b)` is
the total strength of the relations that jointly bind the state-tagged units `a` and
`b`. It is one closed-form expression per pair of atoms — the natural input for
asking which units are bound to which, and how strongly.

```{code-cell} python
bm = analytical.binding_matrix()
bm.index = [str(a) for a in bm.index]
bm.columns = [str(a) for a in bm.columns]

ax = sns.heatmap(bm, annot=True, fmt=".3f", cmap="rocket_r", square=True, cbar_kws={"label": "binding strength"})
ax.set_title("grid3 atom-pair binding matrix")
plt.tight_layout()
plt.show()
```

## 4. The strongest relations, lazily

`strongest(k)` yields relations one at a time in exact descending φ_r order, and
stops after `k`. Because φ_r never increases when a relatum is added, a best-first
search produces the global order without materializing the rest — so the top few of
a million-relation structure cost a handful of operations, not a million.

```{code-cell} python
top = list(analytical.strongest(k=5))

pd.DataFrame(
    [
        {
            "relata": ", ".join(r.labeled_mechanisms),
            "degree": len(r),
            "phi_r": float(r.phi),
        }
        for r in top
    ]
)
```

## 5. Sampling for questions without a closed form

For arbitrary per-relation quantities — a predicate on degree, on which units a
relation touches, on any property you can write as a function — `sample(n, seed=...)`
draws relations with probability proportional to their overlap and returns an
unbiased estimate with a standard error. The draw is reproducible from the seed.

Here we estimate the number of degree-2 (non-self) relations and watch the estimate
converge to the exact value as the sample grows.

```{code-cell} python
exact_degree2 = sum(1 for r in concrete if not r.is_self_relation and len(r) == 2)

rows = []
for n in [100, 300, 1000, 3000, 10000, 30000]:
    sample = analytical.sample(n, seed=0)
    estimate, stderr = sample.estimate(lambda r: 1.0 if len(r) == 2 else 0.0)
    rows.append({"n": n, "estimate": estimate, "stderr": stderr})
conv = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(conv["n"], conv["estimate"], yerr=conv["stderr"], marker="o", capsize=4, label="estimate ± SE")
ax.axhline(exact_degree2, color="#C44E52", ls="--", label=f"exact = {exact_degree2}")
ax.set_xscale("log")
ax.set_xlabel("sample size n")
ax.set_ylabel("estimated # degree-2 relations")
ax.set_title("Horvitz–Thompson estimate converges to the exact count")
ax.legend()
plt.tight_layout()
plt.show()

conv
```

## 6. Distinction importance and folds

`distinction_importance()` ranks each distinction by its additive contribution to Φ
— its own φ plus its share of every relation it participates in. These contributions
tile Φ exactly: they sum to `big_phi`.

```{code-cell} python
importance = ces.distinction_importance()

imp_df = pd.DataFrame(
    [
        {"mechanism": str(tuple(d.mechanism)), "contribution": contribution}
        for d, contribution in importance
    ]
)

ax = sns.barplot(data=imp_df, x="mechanism", y="contribution", color="#4C72B0")
ax.set_title("grid3 distinction importance (contributions tile Φ)")
plt.tight_layout()
plt.show()

print(f"Σ contributions = {imp_df['contribution'].sum():.4f}  (Φ = {ces.big_phi:.4f})")
```

A **fold** is the slice of the structure seeded by a set of distinctions: those
distinctions plus every relation incident to them. Every query above works on a
fold, restricted to the incident relations. Here we fold on the single most
important distinction and confirm the fold's relation count and Σφ_r match the
relations that actually touch it.

```{code-cell} python
top_distinction, _ = importance[0]
fold = ces.fold([top_distinction])

seed = {top_distinction}
incident = ConcreteRelations(r for r in concrete if not seed.isdisjoint(r))

pd.DataFrame(
    {
        "fold (closed form)": [fold.relations.num_relations(), fold.relations.sum_phi()],
        "incident (enumerated)": [incident.num_relations(), incident.sum_phi()],
    },
    index=["num_relations", "Σφ_r"],
)
```

## 7. Why this scales

Everything above ran on a 3-unit network so we could enumerate the relations and
check the closed forms against them. The point of the analytical backend is what
happens when you cannot enumerate.

The cost of every closed-form query scales with the number of distinctions and the
number of atoms (at most `2n` for `n` binary units) — **not** with the number of
relations. So the same calls that took microseconds here take milliseconds on the
IIT 4.0 paper's Figure 6D structure, where the relations number in the millions:

| | grid3 (this guide) | Figure 6D |
|---|---|---|
| distinctions | 7 | 27 |
| relations | 39 | 1,537,080 |
| enumerate every relation | instant | ≈ 50 s, ≈ 1.4 GiB |
| `num_relations()`, `sum_phi()`, `degree_spectrum()`, … | instant | ≈ milliseconds |

Past roughly 35 distinctions the enumeration does not fit in memory at all, while
the analytical queries are unaffected. `num_relations()` already reflects this: it
returns the exact count — 1,537,080 for Figure 6D — as a closed-form expression,
having built no relations. When you do need explicit relation objects,
`materialize(max_degree=..., min_phi=...)` builds a bounded subset of them, and
`strongest(k)` streams the important ones in order.

To use the analytical backend on your own structures, set

```python
config.formalism.iit.relation_computation = "ANALYTICAL"
```

and every relation set a computation returns will answer these queries in closed
form.
