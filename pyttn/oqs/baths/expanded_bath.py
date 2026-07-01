# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Callable, Optional, Union
import numpy as np
import abc

from pyttn.utils.truncate import DepthTruncation, TruncationBase
from pyttn.utils.mode_combination import ModeCombination
from pyttn import (
    system_modes,
    boson_mode,
    fermion_mode,
    ntreeBuilder,
    ntreeNode,
    OPBase,
    sSOP,
    SOP,
)

class ExpandedBath(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def system_information(self, representation: str):
        """Return system_modes for this bath."""
        pass

    @abc.abstractmethod
    def is_fermionic(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def representation(self) -> str:
        """Return 'hilbert' or 'liouville'."""
        pass

    def add_to_tree(        
        self,
        node: ntreeNode,
        degree: int,
        chi: Union[int, list[int], Callable[[int], int]],
        lhd: Optional[Union[int, list[int], Callable[[int], int]]] = None,
    ) -> list[list[int]]:
        """Default: reuse existing implementation."""
        return super().add_bath_tree(node, degree, chi, lhd)

    @abc.abstractmethod
    def add_to_dynamics(
        self,
        H,
        coupling_ops,
        representation: str,
        **kwargs
    ):
        """Attach bath contribution to Hamiltonian or generator."""
        pass

    def initial_state(self, kind="vacuum"):
        """unified initial state hook."""
        raise NotImplementedError()
