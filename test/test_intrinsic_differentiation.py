from pyphi import Direction
from pyphi import config
from pyphi.examples import EXAMPLES


def test_intrinsic_information():
    with config.override(REPERTOIRE_DISTANCE_SPECIFICATION="INTRINSIC_SPECIFICATION"):
        subsystem = EXAMPLES["subsystem"]["differentiation_micro_1"]()
        mechanism = (0, 1)
        result = subsystem.intrinsic_information(Direction.CAUSE, mechanism, mechanism)
        assert result.state == (1, 1)
        assert result.intrinsic_information == 1.8857840667050536
