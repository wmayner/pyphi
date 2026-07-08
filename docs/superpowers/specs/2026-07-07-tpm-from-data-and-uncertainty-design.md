# Building substrates from data, and carrying input uncertainty through Φ

**Status:** exploration / design sketch, not a commitment. Written for review.
**Date:** 2026-07-07

This document works out what PyPhi would have to become to (1) build a substrate
from observed data rather than a hand-specified transition probability matrix
(TPM), and (2) report Φ and the cause–effect structure in a way that reflects
uncertainty in that input. It is grounded in the IIT 4.0 paper (Albantakis et
al. 2023), the intrinsic-information paper (Barbosa et al. 2020), and the actual
PyPhi code as it stands on this branch. Every quantitative claim below was run
against the library; the demonstration script and its saved outputs are
described in the appendix.

Throughout I mark what **already exists in PyPhi**, what is **established method
elsewhere**, and what is **my own proposal**, because the three should not be
confused.

---

## 1. The short version

Two claims organize everything that follows.

**Claim 1 — the TPM is an interventional object, and that sharply limits what
"estimating it from data" can mean.** In IIT the TPM is not the distribution of
transitions the system happens to produce. It is `p(u̅ | do(u))`: the
probability of each next state when the current state `u` is *set by
intervention*, evaluated over a *uniform* perturbation of all states, with the
units *conditionally independent* given the previous state (Albantakis et al.
2023, Eqs. 1–2, p. 8). Passive recordings give you none of these three things
for free. So "fit a TPM from a multivariate time series" is not a
data-cleaning task; it is a causal-inference task with strong, often untestable,
assumptions — and for a large class of empirical data the object IIT needs
simply cannot be recovered. Section 3 makes this precise and demonstrates the
failure.

**Claim 2 — "uncertainty in Φ" is not error bars on a number.** The naive
reading — put a posterior on the TPM, push it through, report Φ ± something — is
half wrong, and the wrong half is the important half. Φ is assembled almost
entirely out of *selections*: the argmax that picks the maximal cause–effect
state (Eq. 12), the argmin that picks the minimum-information partition, the
argmax over candidate substrates that picks the complex. These are not smooth
functions of the TPM. Under a posterior over TPMs, the *identities* of the
selected objects — which units form the complex, which purview a mechanism
specifies, whether a distinction exists at all — become discrete random
variables. The posterior over the scalar Φ is generically a **mixture of a point
mass at zero** (the samples where the system is reducible, i.e. *does not exist*
as an integrated entity) **and a continuous density** (the samples where it
does). Averaging across that mixture produces a number that describes no
possible system. The honest output is a distribution over *structures*, with an
existence probability attached to every object, not a scalar with a spread.
Section 5 demonstrates all of this on a three-unit example.

The rest of the document develops these, proposes representations and an
interface, is honest about tractability, and ends with the smallest thing worth
building.

---

## 2. What PyPhi is today (grounded)

The facts in this section are from the code on this branch; file:line citations
let you check them.

**The TPM is stored as a product of per-unit conditionals, and that product form
is IIT's conditional-independence assumption made concrete.** A `Substrate`
holds a `FactoredTPM` (`pyphi/substrate.py:178`), which represents the joint
`P(s_{t+1} | s_t)` as `N` independent factors, one per unit, factor `i` being
`P(s_{i,t+1} | s_t)` (`pyphi/core/tpm/factored.py:1-12`). This is exactly Eq. 2
of the paper. There is no way to store a conditionally *dependent* joint in a
Substrate: the joint is only ever materialized *from* the factors
(`factored.py:262`), and the one conversion that could introduce dependence,
state-by-state to state-by-node, explicitly warns that dependence is silently
dropped (`pyphi/convert.py:222-229`). Row-stochasticity is checked per factor at
a tolerance of `10^-precision` (`factored.py:505,525`); an estimated factor that
does not sum to one is rejected as an invalid TPM.

**Connectivity is inferred structurally from the point TPM.**
`FactoredTPM.infer_cm()` (`factored.py:314`) marks an edge `a → b` when unit
`b`'s factor is not constant along input axis `a`, tested by exact inequality up
to `10^-precision` (`factored.py:302-312`). This is a clean, marginalization-free
dependence oracle on an *exact* TPM. Section 4 shows it does the wrong thing on
an *estimated* one.

**Results already carry a non-float Φ.** The scalar Φ on a
`SystemIrreducibilityAnalysis` is typed `float | DistanceResult`
(`pyphi/formalism/iit4/__init__.py:167`) and is stored as a `PyPhiFloat` — a
`float` subclass whose comparisons respect `config.numerics.precision`
(`pyphi/data_structures/pyphi_float.py`). `DistanceResult`
(`pyphi/measures/distribution.py:53`) is a `PyPhiFloat` subclass that carries an
arbitrary `aux` dictionary, and that dictionary already round-trips through
serialization (`pyphi/serialize/schema.py:25-27`, `convert.py:62-67`). Every
φ-bearing field in every schema is typed to accept it. This is a genuine
extension seam and Section 6 uses it — but also warns where leaning on it would
smuggle in exactly the scalar-averaging error of Claim 2.

**Provenance travels with results but has no structured slot for an
estimator.** `Provenance` (`pyphi/provenance.py:59`) records versions, git SHA,
timestamp, wall time, seed, and a free-form `note`. It deliberately never
affects equality or hashing. Recording *how a TPM was estimated* today means a
string in `note`; a first-class slot would be one added field.

**There is tooling to go from parameters to a TPM, and none to go from data to a
TPM.** `substrate_generator.build_tpm` (`pyphi/substrate_generator/__init__.py:58`)
fills a TPM by evaluating a unit function (logistic, Ising, sixteen logic gates)
over every state; `dynamics.simulate` (`pyphi/dynamics.py:40`) runs a TPM
forward to a stochastic trajectory. Both are the *forward* direction. Grepping
the repo for model fitting (`fit`, `glm`, `logistic`, `maxent`, `estimate.*tpm`)
returns nothing: **no data-to-TPM path exists.** `sweep`
(`pyphi/sweep.py:237`) ranges over states, subsets, and formalisms of a *single
fixed substrate*; it has no axis for an ensemble of TPMs.

So the gap is specific: forward models and connectivity inference exist;
estimation, an uncertainty layer, and an ensemble sweep do not.

---

## 3. Where does the TPM come from, and what can data legitimately give you?

### 3.1 The three properties, and why data resists each

The paper is unusually explicit about the TPM's semantics, and each property is
an obstacle to estimation.

1. **Interventional, not observational.** `p(u̅ | u) = p(u̅ | do(u))`
   (Eq. 1 and surrounding text, p. 8): the conditional is defined under the
   do-operator. Recovering an interventional conditional from passive
   observation is the central problem of causal inference; it is possible only
   under assumptions such as no unobserved common cause of `u` and `u̅` and
   correct specification of the variable set (Pearl's back-door conditions).
   These are assumptions about the world, not about the data, and they are
   usually untestable from the data alone. The paper itself notes that the
   intrinsic cause–effect state "does not necessarily correspond to the actual
   cause and effect states ... in the dynamical evolution of the system, which
   typically also depends on extrinsic influences" (p. 15). The dynamics you
   observe and the interventional object IIT wants are different things.

2. **Uniform over all states, not weighted by what actually happens.** Intrinsic
   information imposes a uniform prior over the state space (Eqs. 6, 8, and the
   emphasis on p. 15 that "all probabilities must be derived from the system's
   interventional transition probability function, while imposing a uniform
   prior distribution over all possible system states"; and p. 7, "the uniform
   distribution ... rather than ... an observed probability distribution").
   Every row of the TPM matters equally, including rows for states the system
   essentially never visits on its own. Observational data is distributed over
   the attractor, which is a vanishing fraction of the `k^n` states for any
   interesting system. The data is densest exactly where IIT weights it least.

3. **Conditionally independent per unit.** Eq. 2. This one is a *gift* for
   estimation, addressed in 3.3.

### 3.2 Two data regimes — both are real user needs

Both regimes occur in practice, and the design must serve both rather than bless
one and refuse the other.

**Regime P — perturbational data.** You can set the system's state (optogenetic
or electrical stimulation, or a controllable artificial substrate) and record
the next state. This is `do(u)` directly. If you can drive the system across the
whole state space, estimation is a clean counting problem and the three
obstacles evaporate. **This is the common case for theoretical work at feasible
substrate sizes** — the user controls the substrate and can perturb it into every
state — and it is exactly where the whole enterprise is legitimate.

**Regime O — observational data.** You have a passively recorded time series and
get transitions `u_t → u_{t+1}` only for the states the system chose to occupy.
To treat these as interventional you must assume the recorded dynamics are the
causal dynamics — no unobserved driver, the recorded units are the causal units
at the causal grain, and the process is stationary. When these hold, Regime O
collapses to Regime P restricted to the visited states, and the restriction is
the problem (Section 3.4). **Users do attempt this** — extracting a TPM from an
empirical time series — despite the caveats, and they will continue to whether or
not the library helps them. That is the argument for supporting it: an
unsupported user writes the naive counting loop by hand, gets a fully-populated
TPM with most rows silently resting on the prior, and reports a single Φ with no
signal that the data never constrained it. A supported Regime O path is worth
building **precisely so it can be the honest one** — it must surface which rows
the data left unconstrained and decline to present a lone Φ as if the whole TPM
were identified (Section 8). The library cannot verify the causal-sufficiency
assumptions; it can refuse to hide their consequences.

### 3.3 The factorization is the natural estimand

Because the TPM factorizes (Eq. 2, and PyPhi stores exactly this), the estimation
target decomposes into `N` independent per-unit problems: for each unit `i`,
estimate `P(s_{i,t+1} | s_t)`. This is one supervised model per unit predicting
that unit's next value from the full current state. Concretely:

- **Nonparametric / counting** (my proposal, and the honest default): for each
  unit and each current state, count the next-value frequencies. This makes no
  functional assumption but needs every state visited many times — feasible only
  in Regime P for small systems.
- **Parametric per-unit models** (established method, e.g. logistic regression
  or a generalized linear model per unit; a maximum-entropy / Ising fit): borrow
  statistical strength across states so unvisited states get a model-based
  prediction. PyPhi already contains the *forward* logistic and Ising unit
  functions (`substrate_generator/mechanisms.py`, `ising.py`); fitting their
  parameters to data is new code but small.

Either way, the per-unit factorization means conditional independence is enforced
*by construction* — you fit one unit at a time — which is convenient but also a
commitment: if the real units share instantaneous noise (a common input that
correlates their transitions within a step), the factorized fit cannot represent
it, and PyPhi has nowhere to put it even if you could (Section 2). This is a
place where the model and the data can genuinely disagree and the disagreement
is invisible.

### 3.4 The observational wall, demonstrated

I took `basic_substrate` (three deterministic units — OR, AND, XOR — a real
example network in the library, true Φ = 0.415037 in state `(1,0,0)`) and
estimated its TPM two ways.

**Perturbational (Regime P):** draw current states uniformly, sample next states
from the true dynamics, count with a uniform (Laplace) prior on each row. Φ of
the re-estimated substrate as total sample size grows:

| total samples | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| estimated Φ | 0 | 0 | 0 | 0.114 | 0.108 | 0.239 | 0.276 | 0.354 | 0.382 |

It converges to the true 0.415. (The early collapse to zero is itself
instructive and is discussed in 3.5.)

**Observational (Regime O):** run the dynamics free of intervention for 2000
steps and estimate from the trajectory. The system falls onto a short orbit and
**visits only 3 of its 8 states.** Five of the eight TPM rows are never observed
and sit at the prior. Estimated Φ came out ≈ 0.4154 — close to the truth — but
this is an accident of the prior, not identification of the system. To show that
directly: I built a second substrate identical to the true one on the three
visited rows and arbitrary on the five unvisited rows. It produces **the same
observational data** — the free-running orbit never touches the altered rows —
yet its Φ is **0.327**, not 0.415.

> Observational data cannot distinguish two substrates that differ only on
> unvisited states, and those substrates can have materially different Φ. For a
> system on a low-dimensional attractor — which is most systems worth studying —
> the interventional TPM that IIT requires is not identified by passive
> recording, at any sample size. This is not estimator variance that shrinks
> with more data; it is non-identifiability that does not.

This is, I think, the most important negative result in the document, and it is
not a PyPhi limitation — it is a property of the theory's object meeting the
structure of real data. The correct response is not a cleverer estimator. It is
to report that the rows are unconstrained and to refuse to collapse that into a
single Φ.

### 3.5 A second, subtler failure: the deterministic boundary

In the perturbational table above, Φ is zero until about sixteen *total* samples.
The reason is worth pinning down precisely, because it turns out to be two
separate effects that are easy to conflate, and I did conflate them in an earlier
draft until the experiment below separated them.

Indeterminism *reduces* intrinsic information — the selectivity term shrinks when
a state leads to many states (paper p. 14, "intrinsic information is reduced by
indeterminism"). Two things can inject spurious indeterminism into an estimate of
a deterministic system:

1. **Unvisited rows (a coverage effect).** In the perturbational table, `N` is a
   *total* budget spread over eight states, so at small `N` most rows are never
   sampled and sit at exactly `P(on) = 0.5` — maximal indeterminism. This is what
   drove Φ to exactly zero there, and it is the same non-identifiability as
   Section 3.4, not a property of the prior.
2. **Smoothing on visited rows (a bias effect).** Even a fully-sampled row is
   pulled off the boundary by the prior: the posterior-mean estimate is
   `(k_on + a) / (N + 2a)` under a `Beta(a, a)` prior, so a visited all-on row
   reads `(N + a)/(N + 2a) < 1`. This biases Φ downward but does not zero it.

**I ran the confirmation experiment to separate these and to choose the prior.**
Sampling `N` sequences *per state* (full coverage, so effect 1 is excluded),
paired across four priors (the same drawn counts feed all four), 24 seeds, on two
deterministic example networks. Mean absolute error of the estimated Φ:

`basic_substrate` (true Φ = 0.415):

| samples/state | Laplace `a=1` | Jeffreys `a=½` | `a=0.1` | `a=0.05` |
|---|---|---|---|---|
| 1 | 0.364 | 0.316 | 0.147 | 0.087 |
| 4 | 0.247 | 0.170 | 0.048 | 0.025 |
| 16 | 0.104 | 0.059 | 0.013 | 0.007 |

`xor_substrate` (true Φ = 1.5) shows the same ordering, more extreme (at 1
sample/state: Laplace 1.14, Jeffreys 0.84, `a=0.1` 0.22, `a=0.05` 0.11).

Two clean results. First, **with full coverage no prior ever produced a false
Φ = 0** (the "fraction of runs with Φ̂ = 0" was 0.000 in every cell): the
zero-collapse is a coverage phenomenon, not a smoothing one. Second, **the prior
ordering is unambiguous and monotone** — Laplace is the worst choice at every
sample size, Jeffreys roughly halves its error, and concentrating further roughly
halves it again. The `ε`-boundary sensitivity check is consistent (pushing the
true probabilities off the boundary by ε gives Φ = 0.374 at ε = 0.02 and 0.413 at
ε = 0.001).

**Recommendation (now backed by the experiment):** default to **Jeffreys
`Beta(½, ½)`, not Laplace.** It halves Laplace's boundary bias, is the standard
noninformative prior so it needs no bespoke justification, and its posterior is
appropriately U-shaped near the boundary (expressing "probably near 0 or 1," not
false confidence). A more concentrated prior (`a ≈ 0.1`) lowers point-estimate
bias further and is reasonable when the substrate is believed near-deterministic,
but I would not make something as aggressive as `a = 0.05` the default, since for
a genuinely *stochastic* unit with an interior true probability a very small `a`
would over-concentrate the posterior at the boundaries. The one firm conclusion:
**Laplace should not be the default** — the smoothing it is prized for is actively
harmful for recovering Φ near deterministic dynamics.

---

## 4. Aleatoric versus epistemic — the distinction that has to come first

"The system is stochastic, so what is Φ?" and "the TPM is estimated from finite
data, so what is Φ?" are different questions, and the word "uncertainty" hides
the difference.

- **Aleatoric** — the substrate is genuinely indeterministic. Its TPM has
  entries strictly between 0 and 1. **This is already first-class in IIT and in
  PyPhi.** It is not uncertainty *about* Φ; it is part of the substrate, and the
  formalism propagates it exactly — indeterminism enters the selectivity term
  and lowers intrinsic information, as above. A probabilistic TPM produces one
  well-defined Φ. Nothing to add.

- **Epistemic** — you do not know which TPM is the true one, because you
  estimated it. This is uncertainty over the *choice of substrate*, and PyPhi
  has no representation for it. This is the thing to build.

Conflating them is the first error, and it is easy to make: a stochastic TPM and
a posterior-mean TPM look like the same object (both have interior entries) but
mean opposite things. A stochastic TPM with `P(on) = 0.5` says "this unit really
does flip a fair coin." A posterior-mean TPM with `P(on) = 0.5` says "I have no
idea what this unit does." IIT reads both as maximal indeterminism and quietly
suppresses Φ — which is correct for the first and a silent lie for the second.
Any uncertainty layer must keep the two apart: the epistemic object is a
*distribution over* (possibly stochastic) TPMs, and it must never be flattened
to its mean before Φ is computed, because Φ of the mean is not the mean of Φ, and
worse, near the selection boundaries it is not even close.

---

## 5. Propagating epistemic uncertainty, and why the result is a distribution over structures

### 5.1 The representation: a posterior over factored TPMs

**My proposal.** Represent the uncertain substrate as a posterior over the
per-unit factors. Under counting with a Dirichlet–multinomial model, each row of
each factor `P(s_{i,t+1} | s_t = u)` has an independent Dirichlet posterior
(Beta, for binary units) — conjugate, cheap, and respecting the factorization
exactly. The object is `N` arrays of Dirichlet parameters, the same shape as the
factors themselves plus a count axis. For parametric per-unit fits (GLM/Ising) it
is instead a posterior over the weight vectors, sampled by the usual Laplace
approximation or MCMC. Either way the interface primitive is the same: a callable
that draws a full TPM sample. Everything downstream is Monte Carlo over that
draw. (An ensemble of bootstrap refits is the frequentist analogue and slots into
the identical machinery.)

### 5.2 The propagation, demonstrated

I put a `Beta(1 + on, 1 + off)` posterior on every cell of `grid3_substrate` (a
three-unit *stochastic* example network in the library; true Φ = 0.02467 in
state `(0,0,0)`, deliberately near the reducibility boundary), from just **five
perturbational samples per state**, and drew 300 TPMs, computing Φ for each.

- posterior mean Φ = **0.0077**
- posterior median Φ = **0.000**
- 95% credible interval = **[0.000, 0.074]**
- **P(Φ > 0) = 0.20**

Read that again: with this much data the posterior says there is only a **20%
chance the system is integrated at all.** Eighty percent of the posterior TPMs
are *reducible* — Φ is exactly zero, the system does not exist as one entity. The
posterior over Φ is a spike at zero holding 80% of the mass plus a
right-skewed continuous density holding the rest. *Conditional* on Φ > 0, the
median is 0.0268 and the interval is [0.0016, 0.133], which does bracket the
truth. But the unconditional **mean of 0.0077 describes no system** — it is an
average of "does not exist" and "exists with Φ around 0.03," and it equals
neither.

> The correct report is two objects, not one number: **P(the entity exists) ≈
> 0.20**, and **the distribution of Φ given that it exists** (median 0.027,
> interval [0.002, 0.13]). A scalar with error bars would either report 0.008 ±
> something — a value the system never takes — or silently drop the 80% of mass
> at zero.

### 5.3 The selections make *identity* uncertain too

The point mass at zero is one face of a general fact: the objects IIT selects are
argmax/argmin outputs, and finite-data noise makes the argmax flip.

The sharpest instance is in the same demonstration. In `grid3` at `(0,0,0)` the
maximal substrate (the complex) is chosen by maximizing φ over candidate unit
sets, and at the *true* TPM two candidates are **exactly tied**, bit for bit:
units `{0}` and `{2}` both give φ = 0.7518570038729475 (the network is
symmetric). A tie is a measure-zero event that the resolving-ties supplement (S1)
handles by convention. But under a posterior the tie is broken *randomly by the
data*: across the 300 samples the selected complex was

- `{0}` in 62% of samples,
- `{2}` in 31%,
- `{1}` in 7%.

So "which units constitute the system" is itself a categorical random variable.
You cannot average Φ across these samples, because 0.0077 would be averaging Φ
values that belong to *different substrates of consciousness*. This is the
general phenomenon: near any selection boundary — a near-tie between two maximal
cause–effect states, two candidate MIPs, two purviews for a mechanism, or the
zero-crossing where a distinction blinks in and out — input uncertainty turns a
deterministic choice into a distribution over discrete outcomes. Ties are not a
corner case here; finite data puts you *near* ties with positive probability
everywhere.

The honest generalization of "uncertainty in Φ" to the whole structure is
therefore a set of **existence probabilities and conditional distributions over
identities**:

- For the complex: a categorical distribution over which unit set is maximal.
- For each candidate distinction (mechanism × purview): the probability it
  exists in the structure, and, conditional on existing, the distribution of its
  φ and of its specified purview and state.
- For each candidate relation: the probability it exists, conditional on its
  distinctions existing.
- For Φ itself: a mixture (mass at zero for reducible samples) plus a density.

This is a distribution over Φ-structures, not a Φ-structure with numeric error
bars. I do not have a clean, canonical way to *summarize* such a distribution
(Section 7 is honest about that), but I am confident the raw object is the right
one and that any scalar summary computed before this decomposition is
misleading.

### 5.4 Connectivity inference breaks on estimated TPMs — a concrete, verified defect

`infer_cm` decides an edge exists when a factor varies along an input axis by
more than `10^-precision` (Section 2). On an *estimated* TPM every factor varies
along every axis by sampling noise, so the tolerance test always fires. I
confirmed it: `grid3`'s true connectivity has two absent edges
(`cm[0,2] = cm[2,0] = 0`), but under the five-samples-per-state posterior,
`infer_cm` reported **every** edge present with probability 1.0, including the
two that are truly absent. The existing structural connectivity oracle silently
returns "fully connected" for any continuously-estimated substrate.

The fix is conceptual, not a tolerance tweak: under uncertainty, "is there an
edge `a → b`?" is a hypothesis test — is unit `b`'s dependence on `a`
distinguishable from zero given the data? — with an answer that is a probability,
not a bit. **My proposal:** an `edge_probability` matrix (the fraction of
posterior samples in which the dependence exceeds a threshold, or a proper Bayes
factor per edge), replacing the exact-equality test when the substrate is
uncertain. This is cheap and reuses the existing `infer_cm` per sample.

---

## 6. Result types, interface, and where display and serialization break

**The cheap mechanical path exists, and is a trap if used naively.** Because Φ
is already `float | DistanceResult` and `DistanceResult` carries a serializable
`aux` dict (Section 2), you *can* return a Φ that is a float (some point
estimate) with the posterior samples stashed in `aux`. It would flow through
construction, `to_pandas`, display, and serialization with almost no changes.
The trap is that this presents a single float as "the answer" and buries the
distribution as metadata — precisely the scalar-averaging error of Section 5. A
`DistanceResult` whose float value is the posterior mean would advertise 0.0077
for the `grid3` example, a value describing no system.

**My proposal** is to keep the seam but make the primary object honestly
plural:

- An `UncertainPhi` type carrying the sample vector (or the mixture: mass at
  zero plus the positive samples), exposing `p_positive`, quantiles, and a
  conditional distribution — and *refusing* to coerce to a bare float without an
  explicit choice of summary. Where PyPhi's internals demand a `float`
  (`_pandas_record` does `float(self.phi)` at `iit4/__init__.py:314`; the
  comparison sites route through `utils.eq`/`is_positive`), the coercion should
  raise unless a summary policy is set, rather than silently pick the mean. The
  comparison operators (`==`, `is_positive`, ordering) are the single lowest-touch
  place to define what uncertain Φ means relationally — override them on the type
  and most call sites inherit the semantics — but "what does `Φ_A > Φ_B` mean for
  two distributions" is a real modeling choice (stochastic dominance? median?
  posterior probability of exceedance?), not a default to reach past.

- A structural posterior object holding the existence probabilities and identity
  distributions of Section 5.3. This has no analogue in the current code; the
  closest existing container, `SweepResult` (`pyphi/sweep.py:40`), is a flat
  product table of independent deterministic results, not an ensemble over one
  system, so it is a starting shape rather than a fit.

- **Provenance for the estimator** (`pyphi/provenance.py`): add one structured
  field recording the data regime (perturbational/observational), the estimator
  (counts/GLM/Ising), the prior, the per-state sample counts, and the seed. The
  per-state counts matter because they are exactly what tells a downstream reader
  which rows are unconstrained (Section 3.4). This is the metadata that makes a
  Φ-under-uncertainty result interpretable, and the seam for it already exists.

**What breaks in display and serialization** (verified against the code): the
`float(self.phi)` coercions in every `_pandas_record` (`iit4/__init__.py:314`,
`models/sia.py:125`, `sweep.py:117`) would either raise on a non-coercible
distribution or silently flatten a float-subclass one; `format_value`
(`display/numbers.py:22`) collapses any `Real` to `%.6g`, so a distribution that
subclasses float renders as its scalar and the spread vanishes; the diff and
runner-up-gap logic (`iit4/__init__.py:417,423`) does raw float subtraction and
assumes a total order. None of these are hard to fix, but each is a place where
the current code assumes Φ is exactly one number, and each would give a
confidently wrong display if handed a distribution through the float subclass
seam. The serialization layer is the most ready: `aux: dict` already round-trips
arbitrary JSON-able payloads, so samples-as-a-list serialize today; a first-class
distribution schema would be a new `msgspec.Struct` added to the `PhiSchema`
union (`serialize/schema.py:32`), with numpy arrays handled via the existing
bytes treatment used for repertoires.

---

## 7. Tractability, honestly

Φ is already combinatorially expensive. Monte Carlo multiplies it by the number
of posterior samples, and the structural questions of Section 5.3 multiply it
again by the number of candidate objects. The `grid3` demonstration — three
units, 300 samples, plus a complex search per sample — ran in well under a minute,
but it is the smallest interesting case. This does not scale naively, and I want
to be clear about which economies are real and which are illusions.

- **Local linearization (sensitivity) works only in the smooth interior.** Where
  no selection is near a boundary, Φ is a smooth function of the TPM and you
  could propagate a covariance with a single gradient (a delta-method error bar),
  cheaply. **But this is exactly wrong near the boundaries that matter** — a tie
  or a zero-crossing — where Φ is non-differentiable and the whole point is that
  the argmax flips. A cheap sensitivity estimate would be most confident exactly
  where it is most wrong. So linearization is usable only with a guard that
  detects proximity to a selection boundary and falls back to sampling there.
  This is a real design idea (my proposal) but the guard is the hard part.

- **Importance / nested sampling** to resolve `P(Φ > 0)` when it is small: the
  reducibility boundary is a rare-event probability, and plain Monte Carlo needs
  many samples to pin a small `p`. Established rare-event methods apply, but they
  need a way to smoothly interpolate toward the boundary, which the argmax
  structure resists.

- **Screening**: propagate uncertainty only to the decisions that flip. Most of
  the posterior mass usually agrees on most selections; you only need many
  samples where the selection is contested. Identifying the contested selections
  cheaply (a first-order check on the relevant intrinsic-information margins) and
  sampling densely only there is, I think, the most promising route, and it is
  unbuilt.

- **The structural-summary problem has no clean answer.** Even with samples in
  hand, summarizing a distribution over Φ-structures runs into label-switching
  (which distinction in sample A corresponds to which in sample B, when the
  purviews differ?) and the absence of a canonical metric on structures. PyPhi
  has cause–effect-structure distances (`pyphi/metrics/ces.py`) that could seed a
  matching, but "the posterior-mean Φ-structure" is not a well-defined object and
  I would not promise one. Reporting per-object existence probabilities sidesteps
  the matching problem for the objects but not for relations between them.

My honest read: the scalar posterior (`P(Φ>0)` plus the conditional
distribution) and the per-object existence probabilities are tractable for the
small systems PyPhi actually runs, and worth building. A faithful posterior over
the *full* Φ-structure, summarized, is a research problem, not an engineering
task, and should be named as one.

---

## 8. A minimal first build, and what to leave out

In the spirit of not building speculative machinery: the smallest thing that
delivers the honest part of this and nothing more.

Both data regimes are in scope from the start (Section 3.2). The perturbational
path is the clean common case; the observational path exists so that the users
who attempt it anyway get the honest version instead of a hand-rolled counting
loop that hides its own gaps.

**Build:**

1. `estimate_substrate(data, *, regime, prior, model="counts")` → returns a
   posterior object (the Dirichlet parameters per factor). `regime` is required
   and explicit — the caller asserts perturbational vs observational, because the
   library cannot tell from the data and the interpretation differs entirely.
   Counts model only to start; it is the honest default and needs no functional
   assumption. Default prior **Jeffreys `Beta(½,½)`**, not Laplace — settled by
   the Section 3.5 experiment. Records the regime, per-state sample counts, prior,
   and seed in provenance.
2. A `sample()` method on that object yielding a `Substrate`, so the entire
   existing compute stack is reused unchanged.
3. A thin driver — `phi_posterior(uncertain_substrate, state, n_samples)` — that
   samples, computes `sia`, and returns `{p_positive, conditional_quantiles,
   samples}` plus the complex-identity categorical. This is ~50 lines wrapping
   the existing API; it is deliberately *not* a new axis on `sweep`.
4. A **coverage report** as a first-class output of `estimate_substrate`, not a
   downstream afterthought: which rows the data pinned and which rest on the
   prior, with per-row effective sample size. In the perturbational regime this
   is a reassuring check that the perturbation covered the state space; in the
   observational regime it is the headline (Section 3.4), and `phi_posterior`
   should refuse to report a bare scalar Φ when coverage is partial — returning
   the posterior and the uncovered-row list instead, so a lone number never
   stands in for an unidentified TPM. This single output is what separates the
   supported observational path from the naive loop it replaces.

**Explicitly do not build (yet):**

- A first-class distribution-valued `Φ` threaded through every result type,
  display path, and schema. The `aux`-dict seam serializes samples today; a new
  schema type is premature until the summary semantics are decided.
- A general "distribution over Φ-structures" object with matching and mean
  structures. Named as a research problem above; do not ship a summary that
  pretends the matching problem is solved.
- GLM/Ising fitting. Add it when a dataset needs to generalize to unvisited
  states *and* the user accepts the functional assumption — not before. Counts
  first.
- An ensemble axis on `sweep`. The outer loop is three lines; the abstraction
  earns its place only if several call sites want it.

---

## 9. What remains unresolved

- **Observational non-identifiability (Section 3.4) has no fix inside PyPhi.**
  The best PyPhi can do is refuse to hide it: report unconstrained rows and the
  resulting spread, and decline to emit a single Φ. Whether a given dataset is in
  Regime P or Regime O is a claim about the world that the library cannot check;
  it can only record which the user asserted.
- **The prior near the deterministic boundary is not neutral** (Section 3.5).
  This one is now settled by experiment: Jeffreys `Beta(½,½)` is the recommended
  default and Laplace should not be. What remains open is the right default for
  *stochastic* substrates with interior true probabilities, where the
  bias/variance tradeoff of the prior runs the other way; the experiment covered
  only deterministic systems.
- **Comparison semantics for uncertain Φ** (`Φ_A > Φ_B`, `is_positive`) are a
  modeling choice, not a default. The library currently assumes a total order on
  a scalar; every relational use site inherits that assumption.
- **Summarizing a distribution over structures** (Section 7) is open, including
  the label-switching / matching problem across samples.
- **Within-step dependence** between units (Section 3.3) cannot be represented at
  all: the factored form is conditional independence, and if the data violate it
  the violation is silently absorbed into the per-unit fits with no diagnostic.

None of these is a reason not to build the minimal, honest version in Section 8.
They are reasons not to build the tidy version that pretends they are solved.

---

## Appendix — demonstration script and reproducibility

All numbers above come from `demo.py` (kept with this exploration, not part of
the library), run under `env -u VIRTUAL_ENV uv run python` against this branch.
Every randomized routine takes an explicit `seed` and uses an isolated
`np.random.default_rng`; aggregates are saved to JSON and the underlying
per-trial arrays to NPZ alongside them, so the analyses can be redone without
recomputation.

- `demo_A(seed=1)` — Section 3.4/3.5. Interventional convergence table,
  observational state-coverage (3/8 states), and the non-identifiability pair
  (Φ = 0.415 vs 0.327 on identical observational data). Raw:
  `demoA_raw_seed1.npz`, `demoA_seed1.json`.
- `demo_BC(seed=2, n_per_state=5, M=300)` — Section 5.2/5.3. Beta posterior over
  `grid3`, posterior over Φ (`P(Φ>0)=0.20`, mean 0.0077, conditional median
  0.027), and the complex-identity categorical (62/31/7) breaking the exact tie
  φ = 0.7518570038729475. Raw: `demoBC_raw_seed2.npz`, `demoBC_seed2.json`.
- `prior_experiment.py` — Section 3.5. Paired comparison of four priors
  (Laplace, Jeffreys, `Beta(0.1)`, `Beta(0.05)`) on `basic_substrate` and
  `xor_substrate`, 24 seeds, sample sizes 1–32 per state; reports mean absolute
  error and false-zero rate of estimated Φ. Aggregates: `prior_experiment.json`;
  per-seed raw Φ estimates: `prior_experiment_raw.json`.
- Connectivity saturation (Section 5.4) and the ε-boundary sensitivity
  (Section 3.5) are short inline checks reusing the same helpers
  (`ground_truth_pon`, `substrate_from_pon`).

The helper `substrate_from_pon` — build a `Substrate` from an estimated
state-by-node `P(on)` matrix via `convert.to_multidimensional` — is the one
primitive the library lacks today and that an estimation layer would provide.
The roundtrip was validated: feeding the exact ground-truth probabilities back
through it recovers Φ = 0.415037 on `basic_substrate`, matching the direct
computation to all printed digits.
