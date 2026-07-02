# This files is part of the pyTTN package.
# (C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Dict, Union, Optional, Iterator, List
from pyttn.ttnpp import primitive_mode_data, mode_data, system_modes

CompositeInput = Union[primitive_mode_data, Dict[str, primitive_mode_data]]

class SystemInfo:
    """A container for defining composite and primitive mode structure.

    This class stores a mapping from composite mode labels to collections of
    primitive modes. It supports flexible input formats and performs automatic
    normalisation into a consistent internal representation.

    Composite modes may be defined incrementally using dictionary-style access.
    """

    def __init__(self, initial : Optional[Dict[str, CompositeInput]] = None):
        """Initialise the labelled system.

        :param initial: Optional mapping of composite labels to mode definitions
        :type initial: dict[str, CompositeInput], optional
        """

        self._data: Dict[str, Dict[str, primitive_mode_data]] = {}
        if initial is not None:
            for k, v in initial.items():
                self[k] = v

    
    def __setitem__(self, key: str, value: CompositeInput) -> None:
        """Define or update a composite mode.

        :param key: Composite label
        :type key: str
        :param value: Definition of composite mode
        :type value: CompositeInput
        """

        # single primitive
        if isinstance(value, primitive_mode_data):
            self._data[key] = {f"{key}": value}

        # explicit dict
        elif isinstance(value, dict):
            normalised = {}
            for p_label, p_mode in value.items():
                if not isinstance(p_mode, primitive_mode_data):
                    raise ValueError(f"Invalid primitive for '{key}:{p_label}'")
                normalised[p_label] = p_mode
            self._data[key] = normalised
            return
        else:
            raise ValueError(f"Unsupported type for '{key}': {type(value)}")

    def __getitem__(self, key: str) -> Dict[str, primitive_mode_data]:
        """Return primitive definitions for a composite mode.

        :param key: Composite label
        :type key: str
        :return: Mapping of primitive labels to primitive_mode_data
        :rtype: dict[str, primitive_mode_data]
        """
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        """Remove a composite mode."""
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a composite mode exists."""
        return key in self._data
     
    def __len__(self) -> int:
        """Return number of composite modes."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over composite labels."""
        return iter(self._data)

    def keys(self):
        """Return composite labels."""
        return self._data.keys()
    
    def value(self):
        """Return composite mode data."""
        return self._data.values()

    def items(self):
        """Return composite to primitive mappings."""
        return self._data.items()
    

    def clear(self) -> None:
        """Remove all composite modes."""
        self._data.clear()

    def as_dict(self) -> Dict[str, Dict[str, primitive_mode_data]]:
        """Return a copy of the internal representation.

        The returned structure is:

            {composite_label:
                {primitive_label: primitive_mode_data}}

        :return: Normalised mapping
        :rtype: dict
        """
        return {k: dict(v) for k, v in self._data.items()}

    def composite_labels(self) -> List[str]:
        """Return list of composite labels."""
        return list(self._data.keys())

    def primitive_labels(self, composite: str) -> List[str]:
        """Return primitive labels for a composite mode.

        :param composite: Composite label
        :type composite: str
        :return: List of primitive labels
        :rtype: list[str]
        """
        return list(self._data[composite].keys())

    def nprimitive(self) -> int:
        """Return total number of primitive modes."""
        return sum(len(v) for v in self._data.values())
    
    def __repr__(self) -> str:
        return f"labelled_system({self._data})"

    def build_flattened_modes(self, primitive_ordering : List[str]) -> system_modes:
        """Construct a system_modes object from a given primitive ordering.

        :param primitive_ordering: Ordered list of primitive labels
        :type primitive_ordering: list[str]

        :return: system_modes object
        :rtype: system_modes
        """
        modes = []
        for prim_label in primitive_ordering:
            found = False
            for comp_dict in self._data.values():
                if prim_label in comp_dict:
                    modes.append(mode_data(comp_dict[prim_label]))
                    found = True
                    break
            if not found:
                raise ValueError(f"Primitive label '{prim_label}' not found in any composite mode")
        
        return system_modes(modes)

    def build_system_modes(self, ordering):
        """Construct a system_modes object from a given composite ordering.

        :param ordering: Ordered list of composite labels
        :type ordering: list[str]

        :return: Dictionary containing system_modes and mappings
        :rtype: dict
        """
        if not isinstance(ordering, list):
            raise ValueError("Ordering must be a list of composite labels")
        
        missing = [lx for lx in ordering if lx not in self._data]
        if missing:
            raise ValueError(f"Missing definitions for labels: {missing}")

        extra = [lx for lx in self._data if lx not in ordering]
        if extra:
            raise ValueError(f"Unused labels in system: {extra}")

        modes = []
        primitive_labels = []
        primitive_label_to_index = {}
        label_to_prim_indices = {}

        prim_counter = 0

        for comp_label in ordering:
            prim_dict = self._data[comp_label]

            prims = []
            indices = []

            for prim_label, prim_mode in prim_dict.items():
                if prim_label in primitive_label_to_index:
                    raise ValueError(f"Duplicate primitive label '{prim_label}' detected")
                primitive_labels.append(prim_label)
                primitive_label_to_index[prim_label] = prim_counter

                prims.append(prim_mode)
                indices.append(prim_counter)

                prim_counter += 1

            label_to_prim_indices[comp_label] = indices

            if len(prims) == 1:
                modes.append(mode_data(prims[0]))
            else:
                modes.append(mode_data(prims))

        sys = system_modes(modes)
            
        return {
            "system_modes": sys,
            "primitive_labels": primitive_labels,
            "primitive_label_to_index": primitive_label_to_index,
            "labels_to_prim_indices": label_to_prim_indices
        }
    
    def group_modes(self, groups: Dict[str, List[str]]) -> "SystemInfo":
        """
        Construct a new SystemInfo by regrouping primitive modes.

        :param groups: Mapping from new composite labels to lists of primitive labels
        :type groups: dict[str, list[str]]

        :return: New SystemInfo with regrouped composite modes
        :rtype: SystemInfo
        """

        primitive_lookup: Dict[str, primitive_mode_data] = {}

        for comp_dict in self._data.values():
            for p_label, p_mode in comp_dict.items():
                if p_label in primitive_lookup:
                    raise ValueError(f"Duplicate primitive label '{p_label}' detected")
                primitive_lookup[p_label] = p_mode

        all_primitives = set(primitive_lookup.keys())

        used = set()

        for comp_label, prims in groups.items():
            if not isinstance(prims, list):
                raise ValueError(f"Grouping for '{comp_label}' must be a list")

            for p in prims:
                if p not in primitive_lookup:
                    raise ValueError(f"Unknown primitive '{p}' in group '{comp_label}'")

                if p in used:
                    raise ValueError(f"Primitive '{p}' appears in multiple groups")

                used.add(p)

        unused = all_primitives - used
        new_sys = SystemInfo()

        # Add grouped composites
        for comp_label, prims in groups.items():
            new_sys[comp_label] = {
                p: primitive_lookup[p]
                for p in prims
            }

        # Add ungrouped primitives as their own composites
        for p in unused:
            new_sys[p] = primitive_lookup[p]

        return new_sys


def primitive_label(i : int, N : int, prefix="p") -> str:
    """
    Generate a zero-padded primitive label for index i in a system of size N.

    Example:
        i=3, N=10 → "p3"
        i=3, N=100 → "p03"
    """
    width = len(str(N - 1))
    return f"{prefix}{i:0{width}d}"

def primitive_labels(N: int, prefix="p") -> list[str]:
    """
    Generate zero-padded primitive labels.

    Example:
        N=4 → ["p0","p1","p2","p3"]
        N=12 → ["p00","p01",...,"p11"]
    """
    width = len(str(N - 1))
    return [f"{prefix}{i:0{width}d}" for i in range(N)]


def group_consecutive_labels(labels: list[str], K: int, prefix="G"):
    """
    Group labels into consecutive chunks of size K.
    """
    groups = {}
    for idx, i in enumerate(range(0, len(labels), K)):
        groups[f"{prefix}{idx}"] = labels[i:i+K]
    return groups
