# Building a substrate

How to turn a description of some units into a valid PyPhi substrate.

## The transition probability matrix

A substrate is defined by its **transition probability matrix** (TPM). The
usual input is **state-by-node** form: one row per current system state, one
column per node, each entry giving the probability that node turns **on** at the
next step given that current state.

For `n` binary nodes there are 2ⁿ rows. States are ordered **little-endian**:
the first node is the least-significant bit. So for 3 nodes the rows are, in
order, the states `(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1),
(0,1,1), (1,1,1)` — the *first* coordinate flips fastest. Getting this order wrong is
the most common mistake; write the rows out explicitly and check a couple.

## Requirements PyPhi enforces

- **Interventional.** The TPM must describe what happens when the units are
  *perturbed* into each state (the do-operator), not correlations observed in a
  time series. Build it from each unit's input–output function.
- **Conditional independence.** Each unit's next state must depend only on the
  previous state of the system, not on what the other units do at the same
  step. PyPhi rejects a TPM that violates this (it signals a hidden common
  cause). If your units are genuinely coupled within a step, introduce the
  shared variable explicitly as another unit.

## Worked pattern: logic gates

Suppose three binary units A, B, C where each turns on according to a gate over
the others' previous states — say A = OR(B, C), B = COPY(C), C = XOR(A, B). For
each of the 8 current states, compute each gate's output; that gives a
deterministic row (probabilities 0 or 1). Deterministic TPMs always satisfy
conditional independence.

Then build it:

```python
build_substrate(tpm=[[...], ...], node_labels=["A", "B", "C"])
```

Pass a **connectivity matrix** only if you are sure of the wiring
(`cm[i][j] = 1` means node i is an input to node j) — a wrong one gives a wrong
Φ. Omitting it is always correct, just slower.

## Multi-valued units

For units with more than two states, pass `alphabet=[k_A, k_B, ...]`. Remember
that more states does not necessarily raise Φ, and that a binary approximation
of a multi-valued system generally does not preserve its causal structure.

## Check your work

After building, call `describe_substrate(handle)` to confirm the node count,
labels, and connectivity, then `analyze(handle, state)` on a state of interest.
If a small deterministic system gives a surprising result, suspect a tie from a
symmetry in the TPM (see the gotchas reference).

## Or start from an example

`list_examples()` shows the standard networks from the literature;
`load_example(name)` loads one. Building on a known example is often faster than
authoring a TPM from scratch.
