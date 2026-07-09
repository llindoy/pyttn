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

import abc
from pyttn.ttns.sop import lSOP, lCSOP

class Bath(metaclass=abc.ABCMeta):
    """An abstract base class for representing a bath object"""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def expfit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def discretise(self, *args, **kwargs):
        pass


class BathSpec:
    """Container describing a system-bath coupling specification.

    This class associates a bath object describing the physical properties of an environment with one or more system operators that define how the system couples to that bath. It also stores any additional parameters required to  construct a numerical model (e.g. HEOM, discretisation, pseudomode mapping).

    The BathSpec serves as an intermediate object linking:
      - bath physics (correlation functions, spectral density)
      - system operators (defined via lCSOP)

    It supports both single-channel (uncorrelated) and multi-channel (correlated) baths. In the multi-channel case, the number of system operators should match the number of bath coupling channels.

    :param bath: Bath object describing the environment (e.g. BosonicBath, FermionicBath)
    :type bath: Bath
    :param coupling_ops: System operator(s) describing the coupling to the bath. Can be a single lCSOP or a list of lCSOP objects for multi-channel (correlated) baths.
    :type coupling_ops: lCSOP or list[lCSOP]
    :param tag: User-defined label identifying this bath
    :type tag: str
    :param params: Additional method-specific parameters used when constructing the numerical representation of the bath
    :type params: dict, optional

    Notes
    -----
    - The bath object stores only the *intrinsic properties* of the environment (e.g. spectral density, correlation functions).
    - The system coupling operators are stored separately in this class to maintain a clean separation between system and environment.
    - In the case of a correlated bath with multiple channels, the ordering of `coupling_ops` should correspond to the channel ordering used in  the bath's correlation matrix or spectral density.
    - The `params` field is used by the simulation builder layer and is not interpreted directly by this class.
    - The `tag` field has is used for constructing topology trees.

    Examples
    --------

    Single-channel bath:

    >>> bath = BosonicBath(Jw, beta=1.0)
    >>> spec = BathSpec(bath, Hcoupling, "single channel")

    Multi-channel correlated bath:

    >>> bath = CorrelatedBosonicBath(Jij)
    >>> spec = BathSpec(bath, [A1, A2], "multi channel")

    Using optional parameters:

    >>> spec = BathSpec(bath, A, "phonon_bath", params={"K": 8})
    """

    def __init__( self, bath, coupling_ops, tag: str, params=None):
        if not isinstance(coupling_ops, (list, tuple)):
            coupling_ops = [coupling_ops]
        self.coupling_ops = []
        for op in coupling_ops:
            if isinstance(op, lCSOP):
                self.coupling_ops.append(op)
            elif isinstance(op, lSOP):
                self.coupling_ops.append(op.to_lCSOP())
            else:
                raise TypeError("coupling_ops must contain lSOP or lCSOP objects")

        self.bath = bath
        self.params = params or {}
        self.tag = tag

    def nchannels(self) -> int:    
        """Return the number of coupling channels for this bath specification.

        This corresponds to the number of system operators used to couple to
        the bath. For single-channel (uncorrelated) baths this will be 1, while
        for correlated baths it will match the number of coupling operators
        associated with the bath channels.

        :return: Number of coupling channels
        :rtype: int
        """
        return len(self.coupling_ops)

    @property
    def label(self):
        return self.tag

    def __repr__(self) -> str:
        """Return a string representation of the bath specification.

        The representation includes the bath tag (if provided), and the number 
        of coupling channels.

        :return: String representation of the bath specification
        :rtype: str
        """
        return ( f"BathSpec(tag={self.tag}, " f"nchannels={len(self.coupling_ops)})")
