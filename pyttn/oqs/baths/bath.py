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

    This class associates a bath object describing the physical properties of
    an environment with one or more system operators that define how the system
    couples to that bath. It also stores any additional parameters required to 
    construct a numerical model (e.g. HEOM, discretisation, pseudomode mapping).

    The BathSpec serves as an intermediate object linking:
      - bath physics (correlation functions, spectral density)
      - system operators (defined via lSOP)

    It supports both single-channel (uncorrelated) and multi-channel
    (correlated) baths. In the multi-channel case, the number of system
    operators should match the number of bath coupling channels.

    :param bath: Bath object describing the environment (e.g. BosonicBath, FermionicBath)
    :type bath: Bath
    :param coupling_ops: System operator(s) describing the coupling to the bath.
                         Can be a single lSOP or a list of lSOP objects for
                         multi-channel (correlated) baths.
    :type coupling_ops: lSOP or list[lSOP]
    :param params: Additional method-specific parameters used when constructing
                   the numerical representation of the bath
    :type params: dict, optional
    :param tag: Optional user-defined label identifying this bath
    :type tag: str, optional

    Notes
    -----
    - The bath object stores only the *intrinsic properties* of the environment
      (e.g. spectral density, correlation functions).
    - The system coupling operators are stored separately in this class to
      maintain a clean separation between system and environment.
    - In the case of a correlated bath with multiple channels, the ordering
      of `coupling_ops` should correspond to the channel ordering used in
      the bath's correlation matrix or spectral density.
    - The `params` field is used by the simulation builder layer and is not
      interpreted directly by this class.
    - The `tag` field has no physical meaning and is intended for debugging,
      logging, and output labelling.

    Examples
    --------

    Single-channel bath:

    >>> bath = BosonicBath(Jw, beta=1.0)
    >>> spec = BathSpec(bath, coupling_ops=Hcoupling)

    Multi-channel correlated bath:

    >>> bath = CorrelatedBosonicBath(Jij)
    >>> spec = BathSpec(
    ...     bath,
    ...     coupling_ops=[A1, A2],
    ... )

    Using optional parameters and tag:

    >>> spec = BathSpec(
    ...     bath,
    ...     coupling_ops=A,
    ...     params={"K": 8},
    ...     tag="phonon_bath"
    ... )
    """

    def __init__(
        self,
        bath,
        coupling_ops,
        params=None,
        tag=None,
    ):
        if not isinstance(coupling_ops, (list, tuple)):
            coupling_ops = [coupling_ops]

        self.bath = bath
        self.coupling_ops = list(coupling_ops)
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

    def __repr__(self) -> str:
        """Return a string representation of the bath specification.

        The representation includes the bath tag (if provided), and the number 
        of coupling channels.

        :return: String representation of the bath specification
        :rtype: str
        """

        return (
            f"BathSpec(tag={self.tag}, "
            f"nchannels={len(self.coupling_ops)})"
        )
