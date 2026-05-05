"""Surface-drift test: ``Subsystem``'s public attributes must match the
declaration in :mod:`pyphi.protocols`.

The test introspects ``Subsystem`` to discover its public attributes (public
methods, properties, and instance attributes set in ``__init__``), then
asserts the discovered set matches ``PUBLIC_SUBSYSTEM_ATTRS``. Adding,
renaming, or removing a public attribute on ``Subsystem`` requires updating
the Protocol declaration in the same change — making both ends of the
public contract visible to reviewers and to type-checkers.

Internal-only attributes (those starting with ``_``) are not part of the
checked surface.
"""

from __future__ import annotations

import inspect
import re

import pytest

from pyphi.protocols import PUBLIC_SUBSYSTEM_ATTRS
from pyphi.subsystem import Subsystem


def _public_class_members(cls: type) -> set[str]:
    """Return public methods and properties defined on a class (and bases)."""
    return {name for name in dir(cls) if not name.startswith("_")}


def _instance_attrs_from_init(cls: type) -> set[str]:
    """Return ``self.X = ...`` assignment targets in ``__init__``.

    Restricted to public names. Picks up attributes that aren't visible on
    the class itself but are set during construction.
    """
    src = inspect.getsource(cls.__init__)
    return {m for m in re.findall(r"self\.(\w+)\s*=", src) if not m.startswith("_")}


def _discovered_public_surface() -> set[str]:
    return _public_class_members(Subsystem) | _instance_attrs_from_init(Subsystem)


def test_subsystem_public_surface_matches_protocol():
    """Subsystem's discovered public surface must equal PUBLIC_SUBSYSTEM_ATTRS.

    If this fails, you have either:

    1. **Added** a public attribute to ``Subsystem`` — also add it to
       :data:`pyphi.protocols.PUBLIC_SUBSYSTEM_ATTRS` and to
       :class:`SubsystemPublicInterface`. Adding a public attribute is a
       change to the cross-module contract.

    2. **Removed** a public attribute from ``Subsystem`` — also remove it
       from :data:`pyphi.protocols.PUBLIC_SUBSYSTEM_ATTRS` and the
       Protocol. Removing one is a breaking change for external callers.

    3. **Renamed** a public attribute — combine (1) and (2).

    Drift detection is intentional: the cost of keeping the Protocol in
    lockstep with the class is part of the cost of changing the public
    surface. Run the test, see the diff, decide if the change is wanted,
    update both ends.
    """
    discovered = _discovered_public_surface()
    declared = PUBLIC_SUBSYSTEM_ATTRS

    added = discovered - declared
    removed = declared - discovered

    if added or removed:
        pytest.fail(
            "Subsystem public surface drifted from "
            "pyphi.protocols.PUBLIC_SUBSYSTEM_ATTRS:\n"
            + (
                f"  ADDED (on class but not in Protocol): {sorted(added)}\n"
                if added
                else ""
            )
            + (
                f"  REMOVED (in Protocol but not on class): {sorted(removed)}\n"
                if removed
                else ""
            )
            + (
                "Update pyphi/protocols.py: PUBLIC_SUBSYSTEM_ATTRS and "
                "SubsystemPublicInterface."
            )
        )


def test_subsystem_public_attrs_set_matches_protocol_annotations():
    """The frozenset and the Protocol class must declare the same names.

    Defends against drift between the two declarations within
    ``pyphi/protocols.py`` itself.
    """
    from pyphi.protocols import SubsystemPublicInterface

    annotated = set(SubsystemPublicInterface.__annotations__)

    only_in_set = PUBLIC_SUBSYSTEM_ATTRS - annotated
    only_in_protocol = annotated - PUBLIC_SUBSYSTEM_ATTRS

    if only_in_set or only_in_protocol:
        pytest.fail(
            "PUBLIC_SUBSYSTEM_ATTRS and SubsystemPublicInterface annotations "
            "are out of sync:\n"
            + (f"  Only in set: {sorted(only_in_set)}\n" if only_in_set else "")
            + (
                f"  Only in Protocol: {sorted(only_in_protocol)}\n"
                if only_in_protocol
                else ""
            )
        )
