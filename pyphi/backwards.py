from tqdm.auto import tqdm

from . import Direction, config, models, new_big_phi, utils
from .conf import fallback


def find_all_mice(direction, subsystem, mechanisms=None, progress=None, **kwargs):
    if mechanisms is None:
        mechanisms = utils.powerset(subsystem.node_indices, nonempty=True)
        total = 2 ** len(subsystem.node_indices) - 1
    if fallback(progress, config.PROGRESS_BARS):
        try:
            total = len(mechanisms)
        except TypeError:
            pass
        mechanisms = tqdm(mechanisms, total=total)

    return [
        subsystem.find_mice(direction, mechanism, **kwargs) for mechanism in mechanisms
    ]


def combine_ces(ces_cause, ces_effect):
    distinctions = filter(
        None,
        (
            models.mechanism.Concept(
                mechanism=distinction_cause.mechanism,
                cause=distinction_cause,
                effect=distinction_effect,
            )
            for distinction_cause, distinction_effect in zip(
                ces_cause, ces_effect, strict=True
            )
        ),
    )
    return models.subsystem.CauseEffectStructure(distinctions)


def compute_combined_ces(subsystem_cause, subsystem_effect, **kwargs):
    subsystems = {
        Direction.CAUSE: subsystem_cause,
        Direction.EFFECT: subsystem_effect,
    }
    distinctions = {
        direction: find_all_mice(direction, subsystems[direction], **kwargs)
        for direction in Direction.both()
    }
    return combine_ces(
        ces_cause=distinctions[Direction.CAUSE],
        ces_effect=distinctions[Direction.EFFECT],
    )


def sia(subsystem_cause, subsystem_effect, **kwargs):
    return new_big_phi.sia(subsystem_effect, subsystem_cause=subsystem_cause, **kwargs)
