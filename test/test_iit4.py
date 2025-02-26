from pathlib import Path

import pytest

from pyphi import jsonify, new_big_phi
from pyphi.examples import EXAMPLES

EXAMPLE_NAMES = ["basic", "basic_noisy_selfloop", "fig4", "grid3", "xor"]

DATA_PATH = Path("test/data/phi_structure")


@pytest.mark.parametrize("example_name", EXAMPLE_NAMES)
def test(example_name):
    subsystem = EXAMPLES["subsystem"][example_name]()
    actual = new_big_phi.phi_structure(subsystem)
    expected = load_expected(example_name)
    assert actual == expected


def load_expected(example_name):
    with open(DATA_PATH / f"{example_name}.json") as f:
        return jsonify.load(f)
