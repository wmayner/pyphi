from pyphi import Direction
from pyphi import config
from pyphi.examples import EXAMPLES
from pyphi.measures.distribution import resolve_mechanism_measure


def test_intrinsic_information():
    with config.override(specification_measure="INTRINSIC_SPECIFICATION"):
        system = EXAMPLES["system"]["differentiation_micro_1"]()
        mechanism = (0, 1)
        result = system.intrinsic_information(
            Direction.CAUSE,
            mechanism,
            mechanism,
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        assert result.state == (1, 1)
        assert result.intrinsic_information == 1.8857840667050536
