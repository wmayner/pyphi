"""Generate golden ``dynamic_tpm`` fixtures for the substrate_generator tests.

For each spec in ``test/substrate_generator/golden_configs.py`` this builds the
equivalent substrate with Bjørn Juel's original ``substrate_modeler`` and records
its ``dynamic_tpm``. The committed fixture lets the tests assert that
:func:`pyphi.substrate_generator.create_substrate` reproduces the original
byte-for-byte without depending on the original library at test time.

Run manually to (re)generate, pointing at a checkout of the original library::

    uv run python scripts/gen_substrate_generator_goldens.py \
        --substrate-modeler /path/to/substrate_modeler

The original imports several legacy/optional modules only used for plotting and
old PyPhi shims; this script stubs them so it runs under current PyPhi.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "test"))
from substrate_generator.golden_configs import GOLDEN_CONFIGS  # noqa: E402

FIXTURE = REPO / "test" / "data" / "substrate_generator" / "dynamic_tpms.npz"


class _ExplicitTPM:
    """Minimal stand-in for the removed ``pyphi.tpm.ExplicitTPM``.

    Wraps an array and returns (at least 1-D) arrays on indexing, which is all
    the original ``substrate_modeler`` relies on (it does ``tpm[state][0]``).
    """

    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def __getitem__(self, index):
        return np.atleast_1d(self.array[index])

    def __array__(self, dtype=None):
        return self.array if dtype is None else self.array.astype(dtype)


def _install_shims():
    import pyphi

    tpm_mod = types.ModuleType("pyphi.tpm")
    setattr(tpm_mod, "ExplicitTPM", _ExplicitTPM)  # noqa: B010
    sys.modules["pyphi.tpm"] = tpm_mod
    for name, attr in [("pyphi.subsystem", "Subsystem"), ("pyphi.network", "Network")]:
        mod = types.ModuleType(name)
        setattr(mod, attr, object)
        sys.modules[name] = mod
        setattr(pyphi, name.split(".", 1)[1], mod)
    for name in ("matplotlib", "matplotlib.pyplot", "networkx"):
        sys.modules.setdefault(name, types.ModuleType(name))


def _bundle_params(mechanism, params):
    """Adapt create_substrate params to the original library's parameter names.

    The original ``weighted_mean`` takes its input weights under ``weights``,
    whereas every other weighted mechanism (and this port) uses ``input_weights``.
    """
    params = dict(params)
    if mechanism == "weighted_mean" and "input_weights" in params:
        params["weights"] = params.pop("input_weights")
    return params


def _bundle_unit(un, index, mechanism, inputs, params):
    return un.Unit(
        index=index,
        inputs=tuple(inputs),
        mechanism=mechanism,
        params=_bundle_params(mechanism, params),
        state=0,
        input_state=(0,) * len(inputs),
    )


def _build_unit(un, index, spec):
    if "composite" in spec:
        sub_units = [
            _bundle_unit(
                un, index, sub["mechanism"], sub["inputs"], sub.get("params", {})
            )
            for sub in spec["composite"]
        ]
        return un.CompositeUnit(
            index=index,
            units=sub_units,
            state=0,
            mechanism_combination=spec.get("mechanism_combination", "selective"),
        )
    return _bundle_unit(
        un, index, spec["mechanism"], spec["inputs"], spec.get("params", {})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--substrate-modeler",
        required=True,
        help="Path to a checkout of the original substrate_modeler library "
        "(the directory containing the substrate_modeler package).",
    )
    args = parser.parse_args()

    _install_shims()
    sys.path.insert(0, args.substrate_modeler)
    import substrate_modeler.substrate as subs
    import substrate_modeler.unit as un

    fixtures = {}
    for config_id, node_params in GOLDEN_CONFIGS.items():
        specs = [node_params[k] for k in sorted(node_params)]
        units = [_build_unit(un, j, spec) for j, spec in enumerate(specs)]
        dynamic_tpm = np.asarray(subs.Substrate(units).dynamic_tpm, dtype=float)
        fixtures[config_id] = dynamic_tpm
        print(f"{config_id:16s} shape {dynamic_tpm.shape}")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **fixtures)
    print(f"\nwrote {FIXTURE}")


if __name__ == "__main__":
    main()
