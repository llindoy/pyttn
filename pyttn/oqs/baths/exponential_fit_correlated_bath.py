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
    boson_mode,
    system_modes,
)
from pyttn.utils.mode_combination import ModeCombination
from pyttn.utils.truncate import DepthTruncation, TruncationBase

from .exponential_fit_bath import ExpFitBath


class ExpFitCorrelatedOQSBath(ExpFitBath):
    """The base class for handling a bath representing an exponential fit to a matrix valued bath 
    correlation function of the form

    .. math::
        C(t) = \\sum_k d_{k} \\exp(-z_k t)

    :param dk: The coefficient in the exponential decomposition
    :type dk: np.ndarray
    :param zk: The decay rates in the exponential decomposition
    :type zk: np.ndarray
    :param fermionic: Whether or not the bath is a fermionic bath (default False)
    :type fermionic: bool, optional
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(
        self,
        dk: np.ndarray,
        zk: np.ndarray,
        fermionic: bool = False,
        combine_real: bool = False,
        tol: float = 1e-12,
    ) -> None:
        if dk.shape[0] != len(zk) or  dk.shape[1] != dk.shape[2]:
            raise RuntimeError("Invalid bath decomposition")
        self._ck = dk
        self._wk = zk
        self._real_mode = []
        self._composite_modes = []

        ndk = 0

        for i in range(dk.shape[0]):
            if combine_real and np.abs(np.imag(zk[i])) < tol:
                ndk+=1

            # otherwise we add on separate modes for forward and backward paths
            else:
                ndk += 2

        self._dk = np.zeros((ndk, dk.shape[1], dk.shape[2]), dtype=np.complex128)
        self._zk = np.zeros((ndk, dk.shape[1]), dtype=np.complex128)

        #to do need to actually implement this functionality
        counter = 0
        #now we iterate over all of the terms and correctly add the dk and zk terms
        for i in range(dk.shape[0]):
            if combine_real and np.abs(np.imag(zk[i])) < tol:
                self._zk[counter, :] = zk[i]
                self._dk[counter, :, :] = np.sqrt(dk[i])

                # flag that this is a real valued mode
                self._real_mode.append(True)

                # set up the information that will be used for additional mode combination
                self._composite_modes.append([counter])
                counter = counter + 1


            # otherwise we add on separate modes for forward and backward paths
            else:
                # handle the forward path object
                self._zk[counter, :] = zk[i]
                self._dk[counter, :, :] = np.sqrt(dk[i])

                # handle the backward path object
                self._zk[counter+1, :] = np.conj(zk[i])
                self._dk[counter+1, :, :] = np.sqrt(np.conj(dk[i]))

                # now set up the information that will be used for attempting additional mode combination
                self._real_mode.append(False)
                self._real_mode.append(False)
                self._composite_modes.append([counter, counter + 1])
                counter = counter + 2

        self._fermion = fermionic
        self._mode_dims = []
        self._sysinf = None

    def is_fermionic(self) -> bool:
        """Returns whether or not the bath is fermionic
        :rtype: bool
        """
        return self._fermion

    def Ct(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Returns the matrix valued non-interacting bath correlation function evaluated at the time points t,
        defined by:

        .. math::
            C(t) = \\sum_k d_{k} \\exp(-z_k t)

        :param t: time
        :type t: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray

        """
        return_scalar = False
        if isinstance(t, (float, int)):
            t = np.array([t])
            return_scalar = True

        ret = np.zeros((self._ck.shape[1], self._ck.shape[2], *t.shape), dtype=np.complex128)
        for k in range(self._ck.shape[0]):
            ret += np.outer(self._ck[k, :, :], np.exp(-self._wk[k] * t)).reshape(ret.shape)

        if return_scalar:
            return ret[0]
        else:
            return ret

    def _get_composite_params(
        self,
    ) -> tuple[list[list[np.complex128]], list[list[np.complex128]]]:
        zks = [[self._zk[x] for x in cmode] for cmode in self._composite_modes]
        dks = [[self._dk[:, :, x] for x in cmode] for cmode in self._composite_modes]
        return dks, zks

    @property
    def mode_dims(self) -> list[int]:
        """An array containing the dimensionality of each of the modes"""
        return self._mode_dims

    @property
    def dk(self) -> np.ndarray:
        """An array containing the bath decomposition coefficients"""
        return self._dk

    @property
    def zk(self) -> np.ndarray:
        """An array containing the bath decomposition decay rates"""
        return self._zk


class ExpFitCorrelatedBosonicBath(ExpFitCorrelatedOQSBath):
    """A class for handling a bosonic bath representing an exponential fit to a bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k d_{k} \\exp(-z_k t)

    :param dk: The coefficient in the exponential decomposition
    :type dk: np.ndarray
    :param zk: The decay rates in the exponential decomposition
    :type zk: np.ndarray
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(
        self,
        dk: np.ndarray,
        zk: np.ndarray,
        combine_real: bool = False,
        tol: float = 1e-12,
    ) -> None:
        ExpFitCorrelatedOQSBath.__init__(
            self, dk, zk, fermionic=False, combine_real=combine_real, tol=tol
        )
        self.truncate_modes()

    def truncate_modes(self, truncation: Optional[TruncationBase] = None) -> None:
        print(self._dk, self._zk)
        """Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(8))
        :type truncation: TruncationBase, optional

        """
        if truncation is None:
            truncation = DepthTruncation(8)
        self._mode_dims = truncation(self._dk, self._zk, False)

    def system_information(
        self, mode_comb: Optional[ModeCombination] = None, force_evaluate: bool = False
    ) -> system_modes:
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :param mode_comb: A mode combination object to apply to the system information class.  (Default: None)
        :type mode_comb: ModeCombination, optional
        :param force_evaluate: Forces evaluation of the system_modes object regardless of whether or not one has already been formed. (Default: False)
        :type force_evaluation: bool, optional

        :return: Bath system information
        :rtype: system_modes
        """

        if self._sysinf is None or force_evaluate:
            if not len(self._mode_dims) == len(self._zk):
                raise RuntimeError(
                    "Failed to compute system information object.  The bath object has not not been truncated."
                )

            self._sysinf = system_modes(len(self._composite_modes))
            for ind, cmode in enumerate(self._composite_modes):
                self._sysinf[ind] = [boson_mode(self._mode_dims[x]) for x in cmode]

            if mode_comb is not None:
                self._sysinf = mode_comb(self._sysinf)
        return self._sysinf

    def __str__(self) -> str:
        return (
            "bosonic bath: \n "
            + "\n \alpha "
            + str(self._ck)
            + "\n \nu "
            + str(self._wk)
            + "\n modes "
            + str(self._mode_dims)
            + "\n composite "
            + str(self._composite_modes)
        )

    #def add_system_bath_generator(
    #    self,
    #    H: sSOP | SOP,
    #    Sp: OPBase,
    #    Sm: Optional[OPBase] = None,
    #    method: str = "heom",
    #    binds: Optional[list[int]] = None,
    #    bskip: Optional[int] = 2,
    #) -> sSOP | SOP:
    #    """Attach the bath and system bath coupling Generators associated with this bath object to an existing SOP Generator

    #    :param H: The total Generator
    #    :type H: sSOP | SOP
    #    :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
    #    :type Sp: OPBase
    #    :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\\dagger + a) (Default: None)
    #    :type Sm: OPBase, optional
    #    :param method: The method used to represent the bath.
    #    :type method: {"heom", "pseudomode"}
    #    :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
    #    :type binds: list[int], optional
    #    :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
    #    :type bskip: int, optional

    #    :return: The total Generator now including the system bath terms
    #    :rtype: sSOP | SOP
    #    """

    #    dks, zks = super()._get_composite_params()

    #    H = add_bosonic_bath_generator(
    #        H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip, method=method
    #    )
    #    return H

    #def system_bath_generator(
    #    self,
    #    Sp: OPBase,
    #    Sm: Optional[OPBase] = None,
    #    method: str = "heom",
    #    binds: Optional[list[int]] = None,
    #    bskip: Optional[int] = 2,
    #    dtype: Union[np.float64, np.complex128, float, complex] = np.complex128,
    #) -> sSOP:
    #    """Construct a sSOP containing the system bath Generator of the object.

    #    :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
    #    :type Sp: OPBase
    #    :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\\dagger + a) (Default: None)
    #    :type Sm: OPBase, optional
    #    :param method: The method used to represent the bath.
    #    :type method: {"heom", "pseudomode"}
    #    :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
    #    :type binds: list, optional
    #    :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
    #    :type bskip: int, optional

    #    :return: The total Generator now including the system bath terms
    #    :rtype: sSOP
    #    """

    #    dks, zks = super()._get_composite_params()

    #    H = sSOP(dtype=dtype)

    #    H = add_bosonic_bath_generator(
    #        H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip, method=method
    #    )
    #    return H


