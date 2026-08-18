import pytest

from pyphi import examples
from pyphi.campaign.merge import build_distinction
from pyphi.campaign.merge import merge_purview_rias
from pyphi.campaign.merge import merge_sia_strides
from pyphi.campaign.merge import merge_stride_rias
from pyphi.campaign.runner import partition_stride_entries
from pyphi.campaign.runner import sia_stride_entries
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


def _stride_entries(system, direction, mechanism, purview, k):
    """Build the per-stride payloads exactly as the shard runner does."""
    scheme = config.formalism.iit.mechanism_partition_scheme
    entries = []
    for i in range(k):
        parts, indices = enumerate_partition_stride(
            mechanism, purview, system.node_labels, i, k
        )
        entries.extend(
            (cell.result, cell.aux)
            for cell in partition_stride_entries(
                system, direction, mechanism, purview, parts, indices, scheme
            )
        )
    return entries


# The rule110 combinations are the confirmed divergence cases: pins that
# lose their own stride's local selection hold the globally minimizing
# partitions, so a merge that only sees stride-local winners inflates φ
# (reporting a reducible distinction as real) or flips the specified state.
_STRIDE_CASES = [
    ("basic", (1, 0, 0), (0, 1), Direction.EFFECT, (0, 2), 3),
    ("rule110", (0, 0, 0), (0, 1), Direction.CAUSE, (0, 2), 3),
    ("rule110", (0, 0, 0), (0, 1), Direction.EFFECT, (0, 1, 2), 3),
    ("rule110", (0, 0, 0), (0, 2), Direction.CAUSE, (1, 2), 3),
    ("rule110", (0, 0, 0), (1, 2), Direction.CAUSE, (0, 1), 3),
    ("rule110", (0, 0, 0), (0, 1), Direction.CAUSE, (0, 2), 2),
]


@pytest.mark.parametrize(
    ("example", "state", "mechanism", "direction", "purview", "k"), _STRIDE_CASES
)
def test_stride_merge_equals_full_find_mip(
    example, state, mechanism, direction, purview, k
):
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        substrate = getattr(examples, f"{example}_substrate")()
        system = System(substrate, state)
        full = find_mip(system, direction, mechanism, purview)
        entries = _stride_entries(system, direction, mechanism, purview, k)
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


_SIA_CASES = [
    ("basic", (1, 0, 0), 2),
    ("xor", (0, 0, 0), 2),
    ("xor", (0, 0, 0), 3),
    ("rule110", (0, 0, 0), 2),
    ("rule110", (0, 0, 0), 3),
    ("fig5a", (0, 0, 0), 3),
]


@pytest.mark.parametrize(("example", "state", "k"), _SIA_CASES)
def test_sia_stride_merge_equals_full(example, state, k):
    """The merged SIA must match the unsharded sia() exactly: φ_s, the MIP,
    and — because congruence resolution consumes it — the specified system
    state. The xor and rule110 cases have (cause, effect) state ties whose
    per-pair minima span strides, the class of case a stride-local state
    cascade gets wrong."""
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        substrate = getattr(examples, f"{example}_substrate")()
        system = System(substrate, state)
        full = system.sia()
        scheme = config.formalism.iit.system_partition_scheme
        entries = []
        for i in range(k):
            parts, indices = enumerate_system_partition_stride(system, scheme, i, k)
            entries.extend(
                (cell.result, cell.aux)
                for cell in sia_stride_entries(system, parts, indices, scheme)
            )
        merged = merge_sia_strides(entries, system=system)
    assert float(merged.phi) == float(full.phi)
    assert str(merged.partition) == str(full.partition)
    assert str(merged.system_state) == str(full.system_state)
