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
from pyttn.ttns.sop import SystemInfo, lSOP, lCSOP
from .baths.bath import Bath, BathSpec

from enum import Enum
class Representation(Enum):    
    HILBERT = "hilbert"
    LIOUVILLE = "liouville"

class OQSModel:
    """A container describing an open quantum system.

    This class stores the system Generator, system degrees of freedom,
    and a collection of baths with their couplings to the system. It provides 
    a declarative description of the model which can later be used to construct
    everything needed for setting up various open quantum system simulations

    All operators stored in this object are assumed to act only on the
    system degrees of freedom. Bath modes and extended representations
    (e.g. HEOM or pseudomodes) are introduced during the build stage.
    """

    def __init__(self, *, system_info : Optional[SystemInfo] = None,  system_generator : Optional[Union[lSOP, lCSOP]] = None, baths: Optional[list[BathSpec]] = None, representation : Representation = Representation.HILBERT): 
        """Initialise an open quantum system model.

        :param system_info: System degrees of freedom
        :type system_info: SystemInfo, optional
        :param system_generator: Labelled operator describing the system Generator
        :type system_generator: lSOP or lCSOP, optional
        :param baths: Optional list of bath specifications
        :type baths: list[BathSpec], optional
        :param representation: Whether or not this is storing a Hilbert or Liouville space representation of the generator
        :type representation: Representation

        Notes
        -----                    
        If the toplogy object has been specified it must specify the full tree connectivity for the system degrees of freedom.
        Optionally it can also include specification of where in the tree bath degrees of freedom are too be attached.  
        When including bath connectivity it is necessary to include additional leaf nodes in the tree structure that have the
        same label as the tag used when defining the BathSpec representation of the Bath in order to ensure that Baths can be 
        appropriately identified.
        """
        self._system_info = system_info
        if isinstance(system_generator, lCSOP):
            self._system_generator = system_generator
        else:
            self._system_generator = system_generator.to_lCSOP()

        self._baths = list(baths) if baths is not None else []
        self._representation = representation

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

    def set_system_generator(self, H : Union[lSOP, lCSOP]) -> "OQSModel":
        """Set the system Generator.

        :param H: Labelled operator acting on system degrees of freedom
        :type H: Union[lSOP, lCSOP]
        :return: The updated model instance
        :rtype: OQSModel
        """

        if isinstance(H, lCSOP):
            self._system_generator = H
        else:
            self._system_generator = H.to_lCSOP()        
        return self

    @property
    def system_generator(self) -> lCSOP:
        """Return the system Generator.

        :return: Labelled operator describing the system Generator
        :rtype: lCSOP
        """

        return self._system_generator

    @system_generator.setter
    def system_generator(self, H : Union[lSOP, lCSOP]):
        """Set the system Generator.

        :param H: Labelled operator acting on system degrees of freedom
        :type H: Union[lSOP, lCSOP]
        """
        if isinstance(H, lCSOP):
            self._system_generator = H
        else:
            self._system_generator = H.to_lCSOP()

    @property
    def representation(self) -> Representation:
        """Return the representation in which the system generator is defined.

        :return: Representation identifier
        :rtype: Representation
        """
        return self._representation

    @representation.setter
    def representation(self, value: Representation):
        """Set the representation in which the system generator is defined.

        :param value: Representation identifier
        :type value: Representation
        """
        self._representation = value

    def hilbert_space(self) -> bool:
        """Return whether this is a Hilbert-space model.

        :return: True if the model is Hilbert-space based
        :rtype: bool
        """
        return self._representation is Representation.HILBERT


    def liouville_space(self) -> bool:
        """Return whether this is a Liouville-space model.

        :return: True if the model is Liouville-space based
        :rtype: bool
        """
        return self._representation is Representation.LIOUVILLE

    def validate(self):
        """
        Validate the consistency of the OQS model

        :raises ValueError: If any inconsistency is detected.
        """

        if(self._system_info is None):
            raise ValueError("No SystemInfo has been defined")
        
        if self._system_generator is None:
            raise ValueError("No system Generator has been defined")
        system_labels = set(self._system_info.primitive_labels())
        hsites = self._system_generator.sites()
        invalid = hsites - system_labels

        if invalid:
            raise ValueError("System Hamiltonian contains sites " f"not present in SystemInfo: {sorted(invalid)}")

        bath_tags = set()
        for spec in self._baths:
            if spec.tag in bath_tags:
                raise ValueError( f"Duplicate bath tag '{spec.tag}'.")
            bath_tags.add(spec.tag)

            for op in spec.coupling_ops:
                invalid = op.sites() - system_labels
                if invalid:
                    raise ValueError(f"Bath '{spec.tag}' contains " f"coupling operators acting on " f"unknown sites: {sorted(invalid)}")

                if len(op.sites()) == 0:
                    raise ValueError(f"Bath '{spec.tag}' contains an empty coupling operator.")


    def add_bath(self, bath: Bath, coupling_ops: Union[lCSOP, list[lCSOP]], tag : str, params: Optional[dict] = None) -> "OQSModel":
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
        :param tag: Label identifying the bath
        :type tag: str
        :param params: Optional method specific parameters for representing the bath
        :type params: dict, optional
        :return: The updated model instance
        :rtype: OQSModel
        """
        spec = BathSpec( bath, coupling_ops, tag, params=params or {})

        self._baths.append(spec)
        return self

    @property
    def baths(self) -> List[BathSpec]:
        """Return the list of bath specifications associated with the model.

        :return: List of bath specifications
        :rtype: list[BathSpec]
        """
        return self._baths
    
    def bath(self, tag : str) -> BathSpec:
        """Return a bath specification by tag
        
        :param tag: Bath tag
        :type tag: str
        :return: Bath specification
        :rtype: BathSpec
        """

        for bath in self._baths:
            if bath.tag == tag:
                return bath
    
        raise KeyError(f"Unknown bath '{tag}'.")


