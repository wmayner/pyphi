# data_structures/deepchainmap.py

# From https://github.com/neutrinoceros/deep_chainmap; MIT License
# Copied here for easier maintainability

from typing import Any, Mapping
from collections import ChainMap


def _depth_first_update(target: dict, source: Mapping) -> None:
    for key, val in source.items():
        if not isinstance(val, Mapping):
            target[key] = val
            continue

        if key not in target:
            target[key] = {}
        _depth_first_update(target[key], val)


class DeepChainMap(ChainMap):
    """A recursive variant of ChainMap."""

    def __getitem__(self, key):
        submaps = [mapping for mapping in self.maps if key in mapping]
        if not submaps:
            return self.__missing__(key)
        if isinstance(submaps[0][key], Mapping):
            return self.SUBMAPPING_TYPE(*(submap[key] for submap in submaps))
        return super().__getitem__(key)

    def to_dict(self) -> dict:
        d = {}
        for mapping in reversed(self.maps):
            _depth_first_update(d, mapping)
        return d

    SUBMAPPING_TYPE = None


DeepChainMap.SUBMAPPING_TYPE = DeepChainMap


class AttrDeepChainMap(DeepChainMap):
    """A recursive variant of ChainMap that allows object-like key access with
    dot notation."""

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            if name in self:
                return self[name]
            raise e

    def __setattr__(self, name: str, value: Any) -> None:
        # Special-case `maps` since this is used internally by ChainMap
        if name == "maps" or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self.maps[0][name] = value


AttrDeepChainMap.SUBMAPPING_TYPE = AttrDeepChainMap
