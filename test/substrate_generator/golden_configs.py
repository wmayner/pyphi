"""Shared substrate specifications for the substrate_generator golden tests.

``GOLDEN_CONFIGS`` maps a config id to ``create_substrate`` ``node_params``. The
same specs drive both the golden-fixture generator
(``scripts/gen_substrate_generator_goldens.py``, which builds the equivalent
substrate with the original ``substrate_modeler`` and records its ``dynamic_tpm``)
and ``test_mechanisms.py`` (which rebuilds via :func:`create_substrate` and
asserts equality). Keeping the specs here means the two never drift.

Excluded from the bundle goldens (asserted directly in the tests instead):

- ``mismatch_pattern_detector``: the original is broken (a ``Nonee`` typo that
  raises on every call), so there is no reference output.
- ``stabilized_sigmoid``: the original swaps the input/modulator axes (a
  Fortran-reshape artifact); this port uses the documented convention instead.
"""

WEIGHT_SCALE_MAPPING = {
    (0, 0): 1.0,
    (1, 0): 0.5,
    (0, 1): 0.75,
    (1, 1): 1.5,
}

GOLDEN_CONFIGS = {
    # Logic gates (2-input truth tables) + single-input copy.
    "logic": {
        0: {"mechanism": "copy", "inputs": (1,)},
        1: {"mechanism": "and", "inputs": (2, 3)},
        2: {"mechanism": "or", "inputs": (1, 3)},
        3: {"mechanism": "xor", "inputs": (1, 2)},
    },
    # Voting / averaging mechanisms.
    "voting": {
        0: {"mechanism": "democracy", "inputs": (1, 2, 3)},
        1: {"mechanism": "majority", "inputs": (0, 2, 3)},
        2: {
            "mechanism": "weighted_mean",
            "inputs": (0, 1, 3),
            "params": {"input_weights": (0.2, 0.3, 0.5)},
        },
        3: {"mechanism": "copy", "inputs": (0,)},
    },
    # Sigmoid (with self-input) and the gabor detector.
    "sigmoid_gabor": {
        0: {
            "mechanism": "sigmoid",
            "inputs": (0, 1),
            "params": {
                "input_weights": (0.9, 0.5),
                "determinism": 4.0,
                "threshold": 0.1,
            },
        },
        1: {
            "mechanism": "gabor",
            "inputs": (0, 2),
            "params": {"preferred_states": [(1, 0)]},
        },
        2: {
            "mechanism": "sigmoid",
            "inputs": (1,),
            "params": {"input_weights": (0.8,), "determinism": 3.0},
        },
    },
    # The endorsement family: state-dependent resonator + composites
    # (selective and serial). This is the matching-paper substrate structure.
    "endorsement": {
        0: {
            "mechanism": "sigmoid",
            "inputs": (0,),
            "params": {"input_weights": (0.9,), "determinism": 4.0, "threshold": 0.0},
        },
        1: {
            "composite": [
                {
                    "mechanism": "resonator",
                    "inputs": (0, 1, 2),
                    "params": {
                        "input_weights": (0.5, 0.8, 0.3),
                        "determinism": 4.0,
                        "threshold": 0.0,
                        "weight_scale_mapping": WEIGHT_SCALE_MAPPING,
                    },
                },
                {
                    "mechanism": "mismatch_corrector",
                    "inputs": (0,),
                    "params": {"bias": 0.5},
                },
            ],
            "mechanism_combination": "selective",
        },
        2: {
            "composite": [
                {
                    "mechanism": "sor",
                    "inputs": (0, 1),
                    "params": {"pattern_selection": [(1, 0)], "selectivity": 2.0},
                },
                {
                    "mechanism": "resonator",
                    "inputs": (2,),
                    "params": {
                        "input_weights": (0.7,),
                        "determinism": 3.0,
                        "threshold": 0.0,
                        "weight_scale_mapping": WEIGHT_SCALE_MAPPING,
                    },
                },
            ],
            "mechanism_combination": "serial",
        },
    },
    # Modulated and biased sigmoids.
    "modulation": {
        0: {
            "mechanism": "modulated_sigmoid",
            "inputs": (1, 2),
            "params": {
                "input_weights": [1.0],
                "modulation": {"modulator": (2,), "threshold": 0.3, "determinism": 1.0},
                "determinism": 2.0,
                "threshold": 0.0,
            },
        },
        1: {
            "mechanism": "biased_sigmoid",
            "inputs": (0, 2),
            "params": {
                "input_weights": (1.0, 2.0),
                "determinism": 2.0,
                "threshold": 0.0,
            },
        },
        2: {"mechanism": "copy", "inputs": (0,)},
    },
}

# Combination strategies exercised by the composite nodes above.
COMBINATIONS_USED = ("selective", "serial")
