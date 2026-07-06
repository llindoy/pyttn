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

from typing import Optional, Union, List
from pyttn.ttns.sop import SystemInfo, lCSOP
from .baths.bath import Bath, BathSpec

from itertools import combinations
import numpy as np


class OQSModel:
    """A container describing an open quantum system.

    This class stores the system Hamiltonian, system degrees of freedom,
    and a collection of baths with their couplings to the system. It provides 
    a declarative description of the model which can later be used to construct
    everything needed for setting up various open quantum system simulations

    All operators stored in this object are assumed to act only on the
    system degrees of freedom. Bath modes and extended representations
    (e.g. HEOM or pseudomodes) are introduced during the build stage.
    """

    def __init__(self, *,
                 system_info : Optional[SystemInfo] = None, 
                 system_hamiltonian : Optional[lCSOP] = None,
                 baths: Optional[list[BathSpec]] = None): 
        """Initialise an open quantum system model.

        :param system_info: System degrees of freedom
        :type system_info: SystemInfo, optional
        :param system_hamiltonian: Labelled operator describing the system Hamiltonian
        :type system_hamiltonian: lSOP, optional
        :param baths: Optional list of bath specifications
        :type baths: list[BathSpec], optional


        """
        self._system_info = system_info
        self._system_hamiltonian = system_hamiltonian
        self._baths = list(baths) if baths is not None else []

    def set_system_info(self, sysinf : SystemInfo) -> "OQSModel":
        """Set the system degrees of freedom.

        :param sysinf: System information describing the system modes
        :type sysinf: SystemInfo
        :return: The updated model instance
        :rtype: OQSModel
        """

        self._system_info = sysinf
        return self

    @property
    def system_info(self) -> SystemInfo:
        """Return the system degrees of freedom.

        :return: System information object
        :rtype: SystemInfo
        """

        return self._system_info

    @system_info.setter
    def system_info(self, value : SystemInfo):
        """Set the system degrees of freedom.

        :param value: System information describing the system modes
        :type value: SystemInfo
        """

        self._system_info = value

    def set_system_hamiltonian(self, H : lCSOP) -> "OQSModel":
        """Set the system Hamiltonian.

        :param H: Labelled operator acting on system degrees of freedom
        :type H: lSOP
        :return: The updated model instance
        :rtype: OQSModel
        """

        self._system_hamiltonian = H
        return self

    @property
    def system_hamiltonian(self) -> lCSOP:
        """Return the system Hamiltonian.

        :return: Labelled operator describing the system Hamiltonian
        :rtype: lSOP
        """

        return self._system_hamiltonian

    @system_hamiltonian.setter
    def system_hamiltonian(self, H : lCSOP):
        """Set the system Hamiltonian.

        :param H: Labelled operator acting on system degrees of freedom
        :type H: lSOP
        """

        self._system_hamiltonian = H

    def add_bath(self, bath: Bath, coupling_ops: Union[lCSOP, list[lCSOP]], params: Optional[dict] = None, tag: Optional[str] = None,) -> "OQSModel":
        """Add a bath coupled to the system.

        This function associates a bath object (describing the environment)
        with one or more system operators defining how the system couples to it.

        Supports both:

        - Single-channel baths (one coupling operator)
        - Multi-channel correlated baths (list of coupling operators)

        :param bath: Bath object describing the environment
        :type bath: Bath
        :param coupling_ops: System operator(s) describing the coupling to the bath
        :type coupling_ops: lSOP or list[lSOP]
        :param params: Optional method specific parameters for representing the bath
        :type params: dict, optional
        :param tag: Optional label identifying the bath
        :type tag: str, optional
        :return: The updated model instance
        :rtype: OQSModel
        """
        # Normalise to list for consistent internal handling
        if not isinstance(coupling_ops, (list, tuple)):
            coupling_ops = [coupling_ops]

        spec = BathSpec(
            bath=bath,
            coupling_ops=list(coupling_ops),
            params=params or {},
            tag=tag,
        )

        self._baths.append(spec)
        return self

    @property
    def baths(self) -> List[BathSpec]:
        """Return the list of bath specifications associated with the model.

        :return: List of bath specifications
        :rtype: list[BathSpec]
        """
        return self._baths
