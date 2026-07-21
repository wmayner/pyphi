from pyphi import examples
from pyphi.campaign.merge import build_distinction
from pyphi.campaign.merge import merge_purview_rias
from pyphi.campaign.merge import merge_sia_strides
from pyphi.campaign.merge import merge_stride_rias
from pyphi.campaign.shards import enumerate_partition_stride
from pyphi.campaign.shards import enumerate_system_partition_stride
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.formalism.queries import find_mice
from pyphi.formalism.queries import find_mip
from pyphi.system import System

PIN = {"parallel": False, "progress_bars": False, "shortcircuit_sia": False}


def _system():
    return System(examples.basic_substrate(), (1, 0, 0))


def test_stride_merge_equals_full_find_mip():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism, purview = (0, 1), (0, 2)
        direction = Direction.EFFECT
        full = find_mip(system, direction, mechanism, purview)
        k = 3
        entries = []
        for i in range(k):
            parts, indices = enumerate_partition_stride(
                mechanism, purview, system.node_labels, i, k
            )
            ria = find_mip(system, direction, mechanism, purview, partitions=parts)
            local = {str(p): g for p, g in zip(parts, indices, strict=True)}
            tie_indices = {}
            for pin in ria._state_ties or (ria,):
                pin_ties = pin._partition_ties or (pin,)
                tie_indices[repr(pin.specified_state.state)] = [
                    local[str(t.partition)] for t in pin_ties
                ]
            entries.append((ria, {"tie_indices": tie_indices}))
        merged = merge_stride_rias(entries)
    assert float(merged.phi) == float(full.phi)
    assert str(merged.partition) == str(full.partition)
    assert repr(merged.specified_state.state) == repr(full.specified_state.state)


def test_purview_merge_equals_full_find_mice():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism, direction = (0, 1), Direction.EFFECT
        purviews = system.potential_purviews(direction, mechanism)
        full = find_mice(system, direction, mechanism)
        rias = [find_mip(system, direction, mechanism, p) for p in purviews]
        merged = merge_purview_rias(direction, rias, list(purviews))
    assert float(merged.phi) == float(full.phi)
    assert merged.purview == full.purview
    assert merged.purview_margin == full.purview_margin


def test_distinction_assembly_matches_direct():
    from pyphi.formalism.queries import distinction as _distinction

    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism = (0, 1)
        direct = _distinction(system, mechanism)
        mic = find_mice(system, Direction.CAUSE, mechanism)
        mie = find_mice(system, Direction.EFFECT, mechanism)
        built = build_distinction(mechanism, mic, mie)
    assert float(built.phi) == float(direct.phi)
    assert built.mechanism == direct.mechanism


def test_sia_stride_merge_equals_full():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        full = system.sia()
        scheme = config.formalism.iit.system_partition_scheme
        entries = []
        k = 2
        for i in range(k):
            parts, indices = enumerate_system_partition_stride(system, scheme, i, k)
            sia = system.sia(partitions=parts)
            local = {str(p): g for p, g in zip(parts, indices, strict=True)}
            ties = getattr(sia, "ties", None) or (sia,)
            entries.append(
                (sia, {"tie_indices": [local[str(t.partition)] for t in ties]})
            )
        merged = merge_sia_strides(entries)
    assert float(merged.phi) == float(full.phi)
    assert str(merged.partition) == str(full.partition)
