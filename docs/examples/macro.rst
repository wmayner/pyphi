Intrinsic units: analyzing systems at a macro grain
===================================================

A system's cause-effect power need not be maximal at the grain of its
smallest parts. The intrinsic-units framework of Marshall, Findlay,
Albantakis, and Tononi (2024) extends the IIT analysis to *macro
units*: groups of micro units, possibly evaluated over several micro
updates, whose joint state is read out by an explicit state mapping.
PyPhi implements the full framework in :mod:`pyphi.macro`: the macro
TPM construction, the intrinsic-unit criteria, and a bounded search
that answers "which units, and which grain, are intrinsic for this
substrate in this state?"

This tutorial walks through the authors' *minimal* example, then lets
the search rediscover the coarse-graining example from the paper.

Throughout we use the configuration preset that reproduces the paper's
settings:

    >>> import pyphi
    >>> from pyphi import config
    >>> from pyphi.conf import presets
    >>> import numpy as np

A minimal example, at the micro grain
-------------------------------------

The substrate has two units, ``A`` and ``B``. Each is nearly silent on
its own, weakly noisy, and strongly ON only when both are already ON
(the rows are the little-endian states ``00, 10, 01, 11``; the columns
give each unit's probability of being ON at the next update):

    >>> from pyphi.substrate import Substrate
    >>> from pyphi.system import System
    >>> tpm = np.array(
    ...     [
    ...         [0.05, 0.05],
    ...         [0.05, 0.06],
    ...         [0.06, 0.05],
    ...         [0.95, 0.95],
    ...     ]
    ... )
    >>> substrate = Substrate(tpm, node_labels=("A", "B"))
    >>> state = (0, 0)

At the micro grain the system is barely integrated:

    >>> with config.override(**presets.iit4_2023):
    ...     print(System(substrate, state).sia().phi)
    0.005106576483955726

Defining a macro unit
---------------------

A macro unit is specified by its direct constituents, an update grain,
and a *mapping*: a truth table over the constituents' joint
sequence-states that says when the macro unit counts as ON. The helper
:func:`pyphi.macro.coarse_grain` builds mappings from ON-counts; here,
"the group is ON exactly when both constituents are ON":

    >>> from pyphi.macro import MacroUnit, coarse_grain
    >>> coarse_grain(2, on_counts={2})
    (0, 0, 0, 1)
    >>> alpha = MacroUnit(
    ...     constituents=(0, 1),
    ...     update_grain=1,
    ...     mapping=coarse_grain(2, on_counts={2}),
    ... )

(:func:`pyphi.macro.blackbox` builds the other common family, where
the macro state reads out designated *output* constituents at the
final update of a window — e.g. ``blackbox(2, update_grain=1,
output_constituents=(0,))`` is the table ``(0, 1, 0, 1)``. Update
grains above 1 evaluate the unit over a sliding window of several
micro updates; see :mod:`pyphi.macro.units`.)

Analyzing the macro system
--------------------------

:meth:`pyphi.macro.MacroSystem.from_micro` builds the macro cause and
effect TPMs by the paper's four-step construction and yields an object
the IIT pipeline consumes exactly like a :class:`~pyphi.system.System`:

    >>> from pyphi.macro import MacroSystem
    >>> macro = MacroSystem.from_micro(substrate, (alpha,), state)
    >>> macro.state
    (0,)
    >>> with config.override(**presets.iit4_2023):
    ...     print(macro.sia().phi)
    0.7883339770634884

Macroing raised the system's integrated information by two orders of
magnitude — the framework's central phenomenon.

Is the macro unit intrinsic?
----------------------------

Existing as one unit must be earned. A candidate is an *intrinsic
unit* only if its constituent system is integrated (Eq. 15) and
strictly more irreducible than every competing system that could be
built within its footprint (Eq. 16). :func:`pyphi.macro.is_intrinsic_unit`
returns a verdict carrying the evidence:

    >>> from pyphi.macro import is_intrinsic_unit
    >>> with config.override(**presets.iit4_2023):
    ...     verdict = is_intrinsic_unit(substrate, alpha, state)
    >>> verdict.valid
    True
    >>> verdict.phi
    0.005106576483955726
    >>> verdict.num_competitors
    2

The candidate's two competitors (the single-unit systems over ``A``
and over ``B``) both have ``phi_s = 0``, so the pair wins. Note the
verdict depends only on the candidate's constituents and background
apportionment — not on its mapping or grain — so every mapped variant
of the same grouping shares it. Micro units themselves are exempt
(they are the ground the recursion builds on), even when their own
``phi_s`` is zero:

    >>> from pyphi.macro import micro_unit
    >>> with config.override(**presets.iit4_2023):
    ...     verdict = is_intrinsic_unit(substrate, micro_unit(0), state)
    >>> verdict.valid, verdict.phi
    (True, 0.0)

Searching across grains
-----------------------

:func:`pyphi.macro.complexes` is the one-call driver: it derives every
intrinsic unit within the search bounds, assembles every admissible
system of them (Eq. 18), evaluates each over the full universe, and
returns the *complexes* — the systems that strictly beat every
overlapping rival (Eq. 19) — together with the full evaluation record:

    >>> from pyphi.macro import SearchBounds, complexes
    >>> with config.override(**presets.iit4_2023):
    ...     result = complexes(
    ...         substrate, state, SearchBounds(mappings="EXHAUSTIVE")
    ...     )
    >>> len(result.complexes)
    1
    >>> winner = result.complexes[0]
    >>> winner.units
    (MacroUnit(constituents=(0, 1), update_grain=1, mapping=(0, 0, 0, 1), background_apportionment=()),)

The search, given every possible 2-constituent mapping, finds exactly
the both-ON coarse-graining we built by hand. The record holds every
evaluated system and its ``phi_s``:

    >>> phis = {r.system: r.phi for r in result.records}
    >>> print(phis[winner])
    0.7883339770634884
    >>> len(result.records)
    10

Candidate mappings are enumerated up to state-label complementation (a
mapping and its complement describe the same physical unit, with the
two macro state labels swapped), and ties at the configured precision
are respected: systems that tie their strongest overlapping rival are
not complexes, and ``result.ties`` reports the tied pairs.

Rediscovering the paper's coarse-graining example
-------------------------------------------------

Example 1 of the paper is a four-unit substrate built from two
interacting pairs. Its micro system has ``phi_s`` of about ``0.02``,
but the authors show that coarse-graining each pair into a both-ON
macro unit yields a two-unit macro system with ``phi_s`` of about
``1.004``. The default search bounds (one macroing level, update grain
1, the coarse-graining and black-boxing mapping families) recover that
analysis from scratch:

    >>> tpm4 = np.array(
    ...     [
    ...         [0.05, 0.05, 0.05, 0.05],
    ...         [0.06, 0.15, 0.05, 0.05],
    ...         [0.15, 0.06, 0.05, 0.05],
    ...         [0.16, 0.16, 0.85, 0.85],
    ...         [0.05, 0.05, 0.06, 0.15],
    ...         [0.06, 0.15, 0.06, 0.15],
    ...         [0.15, 0.06, 0.06, 0.15],
    ...         [0.16, 0.16, 0.86, 0.95],
    ...         [0.05, 0.05, 0.15, 0.06],
    ...         [0.06, 0.15, 0.15, 0.06],
    ...         [0.15, 0.06, 0.15, 0.06],
    ...         [0.16, 0.16, 0.95, 0.86],
    ...         [0.85, 0.85, 0.16, 0.16],
    ...         [0.86, 0.95, 0.16, 0.16],
    ...         [0.95, 0.86, 0.16, 0.16],
    ...         [0.96, 0.96, 0.96, 0.96],
    ...     ]
    ... )
    >>> substrate4 = Substrate(tpm4, node_labels=("A", "B", "C", "D"))
    >>> with config.override(**presets.iit4_2023):
    ...     result = complexes(substrate4, (0, 0, 0, 0))  # ~a few seconds
    >>> len(result.complexes)
    1
    >>> for unit in result.complexes[0].units:
    ...     print(unit.constituents, unit.mapping)
    (0, 1) (0, 0, 0, 1)
    (2, 3) (0, 0, 0, 1)

The unique complex is exactly the paper's macro system: both-ON
coarse-grainings over ``(A, B)`` and ``(C, D)``.

Bounding the search
-------------------

The space of groupings, mappings, and grains grows combinatorially, so
the search is explicitly bounded by :class:`pyphi.macro.SearchBounds`:

- ``max_constituents`` — cap on a unit's micro footprint (default 4);
- ``max_update_grain`` — largest update grain per level (default 1;
  set 2+ to search macroing over updates);
- ``max_depth`` — macroing levels above micro (default 1; higher
  levels build units out of already-validated meso units);
- ``mappings`` — ``"FAMILIES"`` (coarse-grainings and black-boxings,
  the default) or ``"EXHAUSTIVE"`` (every surjective table, capped by
  ``exhaustive_cap`` sequence-states);
- ``apportionment`` / ``max_background`` — opt-in enumeration of
  background apportionments (Eq. 12/29).

Every ``phi_s`` evaluation in a driver run is memoized, and
``result.records`` exposes all of them, so the derivation is fully
inspectable: :func:`pyphi.macro.intrinsic_units` returns the unit pool
with one verdict per judged decomposition, and
:func:`pyphi.macro.valid_systems` the admissible-system set.
