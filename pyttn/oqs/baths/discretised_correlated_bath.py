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

from typing import Optional, Union

import numpy as np

from pyttn import (
    SOP,
    OPBase,
    boson_mode,
    sSOP,
    system_modes,
)
from pyttn.utils.mode_combination import ModeCombination
from pyttn.utils.truncate import DepthTruncation, TruncationBase

from ..unitary import add_correlated_bosonic_bath_hamiltonian
from .discretised_bath import DiscreteBath


class DiscreteCorrelatedOQSBath(DiscreteBath):
    """The base class for handling a bath representing a Discrete bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k \\boldsymbol{g}_{k}^\\dagger \\boldsymbol{g}_{k} \\exp(-1.0j w_k t)

    :param gk: The matrix valued coefficient in the exponential decomposition
    :type gk: np.ndarray
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param fermionic: Whether or not the bath is a fermionic bath (default False)
    :type fermionic: bool, optional
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(self, gk : np.ndarray, wk: np.ndarray, fermionic: bool=False, tol: float=1e-12) -> None:
        if gk.shape[0] != len(wk) or  gk.shape[1] != gk.shape[2]:
            raise RuntimeError("Invalid bath decomposition")

        self._gk = np.array(gk)
        self._wk = np.array(wk)
        self._composite_modes = []

        self._fermion = fermionic
        self._mode_dims = []
        self._sysinf = None

    def is_fermionic(self) -> bool:
        """Returns whether or not the bath is fermionic
        :rtype: bool
        """
        return self._fermion

    def Ct(self, t : Union[float, np.ndarray]) ->  np.ndarray:
        """Returns the value of the non-interacting bath correlation function evaluated at the time points t,
        defined by:

        .. math::
            C(t) = \\sum_k \\boldsymbol{g}_{k}^\\dagger \\boldsymbol{g}_{k} \\exp(-1.0j w_k t)

        :param t: time
        :type t: float | np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        return_scalar = False
        if isinstance(t, (float, int)):
            t = np.array([t])
            return_scalar = True

        ret = np.zeros((self._gk.shape[1], self._gk.shape[2]), dtype=np.complex128)
        for k in range(len(self._wk)):
            ret += np.outer(self._gk[k, :, :].conj().T @ self._gk[k, :, :], np.exp(-1.0j * self._wk[k] * t)).reshape(ret.shape)

        if return_scalar:
            return ret[:,:,0]
        else:
            return ret

    @property
    def primitive_mode_dims(self):
        """An array containing the dimensionality of each of the modes"""
        return self._mode_dims

    @property
    def gk(self):
        """An array containing the bath decomposition coefficients"""
        return self._gk

    @property
    def wk(self):
        """An array containing the bath decomposition decay rates"""
        return self._wk


class DiscreteCorrelatedBosonicBath(DiscreteCorrelatedOQSBath):
    """A class for handling a bosonic bath representing a Discrete discrete bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k \\boldsymbol{g}_{k}^\\dagger \\boldsymbol{g}_{k} \\exp(-1.0j w_k t)

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarray
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(self, gk : np.ndarray, wk : np.ndarray, tol: float=1e-12):
        DiscreteCorrelatedOQSBath.__init__(self, gk, wk, fermionic=False, tol=tol)
        self._N = gk.shape[0]
        self._gk_trunc = gk
        self._wk_trunc = wk
        self.truncate_modes()

    def truncate_modes(self, truncation:Optional[TruncationBase]=None):
        """Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(8))
        :type truncation: TruncationBase, optional

        """
        if truncation is None:
            truncation = DepthTruncation(8)
        self._mode_dims = truncation(self._gk_trunc, self._wk_trunc, False)

    def system_information(self, mode_comb : ModeCombination=None, force_evaluate=False):
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :param mode_comb: A mode combination object to apply to the system information class.  (Default: None)
        :type mode_comb: ModeCombination, optional
        :param force_evaluate: Forces evaluation of the system_modes object regardless of whether or not one has already been formed. (Default: False)
        :type force_evaluation: bool, optional

        :return: Bath system information
        :rtype: system_modes
        """

        if self._sysinf is None or force_evaluate:
            if not len(self._mode_dims) == len(self._wk):
                raise RuntimeError(
                    "Failed to compute system information object.  The bath object has not not been truncated."
                )

            self._sysinf = system_modes(len(self._mode_dims))
            for ind in range(len(self._mode_dims)):
                self._sysinf[ind] = boson_mode(self._mode_dims[ind])

            if mode_comb is not None:
                self._sysinf = mode_comb(self._sysinf)
        return self._sysinf

    def __str__(self):
        return (
            "bosonic bath: \n "
            + "\n \alpha "
            + str(self._gk)
            + "\n \nu "
            + str(self._wk)
            + "\n modes "
            + str(self._mode_dims)
            + "\n composite "
            + str(self._composite_modes)
        )


    def add_system_bath_hamiltonian(
        self,
        H: Union[sSOP, SOP],
        Sp: list[OPBase],
        Sm: Optional[list[OPBase]] = None,
        geom: str = "star",
        binds: Optional[list[int]] = None,
        bskip: Optional[int] = 1,
    ) -> Union[sSOP, SOP]:
        """Attach the bath and system bath coupling Hamiltonians associated with this bath object to an existing SOP Hamiltonian

        :param H: The total Hamiltonian
        :type H: sSOP | SOP
        :param Sp: An operator that couples to the bath annihilation operator terms
        :type Sp: list[OPBase]
        :param Sm: An operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(:math:`a^\\dagger` + a) (Default: None)
        :type Sm: Optional[list[OPBase]]
        :param geom: The geometry of the bath to use
        :type geom: {"star", "chain", "ipchain"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds: Optional[list[int]]
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: Optional[int]

        :return: The total Hamiltonian now including the system bath terms
        :rtype: sSOP | SOP
        """

        H, freq = add_correlated_bosonic_bath_hamiltonian(H, Sp, self._gk, self._wk, Sm=Sm, binds=binds, geom=geom, bskip=bskip, return_frequencies=True,)
        self._wk_trunc = freq
        return H

    def system_bath_hamiltonian(
        self,
        Sp: list[OPBase],
        Sm: Optional[list[OPBase]] = None,
        geom: str = "star",
        binds: Optional[list[int]] = None,
        bskip: Optional[int] = 1,
    ) -> sSOP:
        """Construct a sSOP containing the system bath Hamiltonian of the object.

        :param H: The total Hamiltonian
        :type H: sSOP | SOP
        :param Sp: An operator that couples to the bath annihilation operator terms
        :type Sp: OPBase
        :param Sm: An operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(:math:`a^\\dagger` + a) (Default: None)
        :type Sm: Optional[OPBase]
        :param geom: The geometry of the bath to use
        :type geom: {"star", "chain", "ipchain"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds:  Optional[list[int]]
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: Optional[int]

        :return: The total Hamiltonian now including the system bath terms
        :rtype: sSOP
        """
        H = sSOP()

        H, freq = add_correlated_bosonic_bath_hamiltonian(H, Sp, self._gk, self._wk, Sm=Sm, binds=binds, geom=geom, bskip=bskip, return_frequencies=True,)
        self._wk_trunc = freq
        return H
