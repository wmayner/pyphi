import pyphi
from tqdm.auto import tqdm


def find_all_mice(direction, subsystem, mechanisms=None, progress=None, **kwargs):
    if mechanisms is None:
        mechanisms = pyphi.utils.powerset(subsystem.node_indices, nonempty=True)
        total = 2 ** len(subsystem.node_indices) - 1
    if pyphi.conf.fallback(progress, pyphi.config.PROGRESS_BARS):
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
            pyphi.models.mechanism.Concept(
                mechanism=distinction_cause.mechanism,
                cause=distinction_cause,
                effect=distinction_effect,
            )
            for distinction_cause, distinction_effect in zip(
                ces_cause, ces_effect, strict=True
            )
        ),
    )
    return pyphi.models.subsystem.CauseEffectStructure(distinctions)


def compute_combined_ces(subsystem_cause, subsystem_effect, **kwargs):
    subsystems = {
        pyphi.Direction.CAUSE: subsystem_cause,
        pyphi.Direction.EFFECT: subsystem_effect,
    }
    distinctions = {
        direction: find_all_mice(direction, subsystems[direction], **kwargs)
        for direction in pyphi.Direction.both()
    }
    return combine_ces(
        ces_cause=distinctions[pyphi.Direction.CAUSE],
        ces_effect=distinctions[pyphi.Direction.EFFECT],
    )


def combine_system_states(state_cause, state_effect):
    return pyphi.models.subsystem.SystemStateSpecification(
        cause=state_cause.cause,
        effect=state_effect.effect,
    )


def combine_sia(sia_cause, sia_effect):
    minimal_side = sorted(
        [sia_cause, sia_effect], key=pyphi.new_big_phi.sia_minimization_key
    )[0]
    return pyphi.new_big_phi.SystemIrreducibilityAnalysis(
        phi=min(sia_cause.phi, sia_effect.phi),
        partition=minimal_side.partition,
        normalized_phi=minimal_side.normalized_phi,
        cause=sia_cause.cause,
        effect=sia_effect.effect,
        system_state=combine_system_states(
            state_cause=sia_cause.system_state,
            state_effect=sia_effect.system_state,
        ),
        current_state=minimal_side.current_state,
        node_indices=minimal_side.node_indices,
        node_labels=minimal_side.node_labels,
        reasons=minimal_side.reasons,
    )


def compute_combined_sia(subsystem_cause, subsystem_effect, **kwargs):
    subsystems = {
        pyphi.Direction.CAUSE: subsystem_cause,
        pyphi.Direction.EFFECT: subsystem_effect,
    }
    sias = {
        direction: pyphi.new_big_phi.sia(
            subsystems[direction], directions=[direction], **kwargs
        )
        for direction in pyphi.Direction.both()
    }
    return combine_sia(
        sia_cause=sias[pyphi.Direction.CAUSE], sia_effect=sias[pyphi.Direction.EFFECT]
    )


def all_combined_complexes(network, state, **kwargs):
    for subset in pyphi.utils.powerset(
        network.node_indices, nonempty=True, reverse=True
    ):
        yield compute_combined_sia(
            subsystem_cause=pyphi.Subsystem(network, state, subset, backward_tpm=True),
            subsystem_effect=pyphi.Subsystem(network, state, subset),
            **kwargs
        )
