# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np

from pyttn.utils.truncate import DepthTruncation
from pyttn import system_modes, boson_mode, fermion_mode, ntreeBuilder, sSOP


class ExpFitOQSBath:
    r"""The base class for handling a bath representing an exponential fit to a bath correlation function
    of the form

    .. math::
        C(t) = \sum_k d_{k} \exp(-z_k t)

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

    def __init__(self, dk, zk, fermionic=False, combine_real=False, tol=1e-12):
        if len(dk) != len(zk):
            raise RuntimeError("Invalid bath decomposition")

        self._ck = dk
        self._wk = zk
        self._real_mode = []
        self._composite_modes = []

        _dk = []
        _zk = []

        counter = 0
        for i in range(len(dk)):
            if combine_real and np.abs(np.imag(zk[i])) < tol:
                _zk.append(zk[i])  # set the mode frequency
                _dk.append(np.sqrt(dk[i]))  # set the mode coupling constant
                # flag that this is a real valued mode
                self._real_mode.append(True)

                # set up the information that will be used for additional mode combination.
                self._composite_modes.append([counter])
                counter = counter + 1

            # otherwise we add on separate modes for forward and backward paths
            else:
                # handle the forward path object
                _zk.append(zk[i])
                _dk.append(np.sqrt(dk[i]))
                self._real_mode.append(False)

                # handle the backward path object
                _zk.append(np.conj(zk[i]))
                _dk.append(np.sqrt(np.conj(dk[i])))
                self._real_mode.append(False)

                # now set up the information that will be used for attempting additional mode combination
                self._composite_modes.append([counter, counter + 1])
                counter = counter + 2

        self._dk = np.array(_dk, dtype=np.complex128)
        self._zk = np.array(_zk, dtype=np.complex128)

        self._fermion = fermionic
        self._mode_dims = []
        self._sysinf = None

    def is_fermionic(self):
        r"""Returns whether or not the bath is fermionic
        :rtype: bool
        """
        return self._fermion

    def Ct(self, t):
        r"""Returns the value of the non-interacting bath correlation function evaluated at the time points t,
        defined by:

        .. math::
            C(t) = \sum_k d_{k} \exp(-z_k t)

        :param t: time
        :type t: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray

        """
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k] * np.exp(-self._wk[k] * t)
        return ret

    def add_bath_tree(self, node, degree, chi, lhd=None):
        r"""Append a tree as a child of node that represents the modes represented by this expfit bath object.

        :param node: The node where the subtree should be added
        :type node: ntreeNode
        :param degree: The degree of the tree.  If degree = 1 this appends an MPS subtree otherwise it adds a balanced degree-ary tree.
        :type degree: int
        :param chi: The bond dimension to insert throughout the tree.  This can accept all types supported by the ntreeBuilder objects
        :type chi: int, list[int], (callable(int))
        :param lhd: The dimension of local Hilbert space transformation nodes.  This can accept all types supported by the ntreeBuilder objects. (Default: None)
        :type lhd: int, list[int], (callable(int)), optional

        :return: The indices of leaf nodes added to the tree
        :rtype: list[list[int]]
        """

        nelem = node.size()

        lmode_dims = self._sysinf.mode_dimensions()

        if len(lmode_dims) == 0:
            return []

        nindex = node.index()

        if degree == 1:
            if isinstance(lhd, list) or lhd is not None:
                ntreeBuilder.mps_subtree(node, lmode_dims, chi, lhd)
            else:
                ntreeBuilder.mps_subtree(node, lmode_dims, chi)
        elif degree > 1:
            if isinstance(lhd, list) or lhd is not None:
                ntreeBuilder.mlmctdh_subtree(node, lmode_dims, degree, chi, lhd)
            else:
                ntreeBuilder.mlmctdh_subtree(node, lmode_dims, degree, chi)
        else:
            raise RuntimeError("Cannot add tree with Degree < 1.")

        linds = node[nelem].leaf_indices()
        indices = [nindex + li for li in linds]
        return indices

    def identity_product_state(self, method="heom"):
        r"""Set up arrays necessary to construct the identity state used for trace evaluation with na
        expfit bath object.
        For method = "heom", this corresponds to the vacuum state of the  bath
        For method = "pseudomode", this corresponds to a flattened identity operator

        :param method: The method used to represent the bath.
        :type method: {"heom", "pseudomode"}

        :return: A list of numpy arrays that are used to set the identity operator for the bath modes.
        :rtype: list[np.ndarray]
        """
        lmode_dims = self._sysinf.mode_dimensions()

        res = []
        if method == "heom":
            for dim in lmode_dims:
                state_vec = np.zeros(dim, dtype=np.complex128)
                state_vec[0] = 1
                res.append(state_vec)
        elif method == "pseudomode":
            for dim in lmode_dims:
                sqdim = int(np.sqrt(dim))
                if not (sqdim * sqdim == dim):
                    raise RuntimeError(
                        "Failed to set up bath identity product state.  Invalid mode dimensions (must be a square number)."
                    )
                state_vec = np.identity(sqdim, dtype=np.complex128).flatten()
                res.append(state_vec)
        else:
            raise RuntimeError(
                "Failed to set up bath identity product state.  Unknown method argument."
            )
        return res

    def _get_composite_params(self):
        zks = []
        dks = []
        for ind, cmode in enumerate(self._composite_modes):
            zks.append([self._zk[x] for x in cmode])
            dks.append([self._dk[x] for x in cmode])
        return dks, zks

    @property
    def mode_dims(self):
        r"""An array containing the dimensionality of each of the modes"""
        return self._mode_dims

    @property
    def dk(self):
        r"""An array containing the bath decomposition coefficients"""
        return self._dk

    @property
    def zk(self):
        r"""An array containing the bath decomposition decay rates"""
        return self._zk


class ExpFitBosonicBath(ExpFitOQSBath):
    r"""A class for handling a bosonic bath representing an exponential fit to a bath correlation function
    of the form

    .. math::
        C(t) = \sum_k d_{k} \exp(-z_k t)

    :param dk: The coefficient in the exponential decomposition
    :type dk: np.ndarray
    :param zk: The decay rates in the exponential decomposition
    :type zk: np.ndarray
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(
            self, dk, zk, fermionic=False, combine_real=combine_real, tol=tol
        )
        self.truncate_modes()

    def truncate_modes(self, truncation=DepthTruncation(8)):
        r"""Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(8))
        :type truncation: TruncationBase, optional

        """
        self._mode_dims = truncation(self._dk, self._zk, False)

    def system_information(self, mode_comb=None, force_evaluate=False):
        r"""Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

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

    def __str__(self):
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

    def add_system_bath_generator(
        self, H, Sp, Sm=None, method="heom", binds=None, bskip=2
    ):
        r"""Attach the bath and system bath coupling Generators associated with this bath object to an existing SOP Generator

        :param H: The total Generator
        :type H: SOP
        :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
        :type Sp: list[sOP or sPOP or sNBO or sSOP]
        :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\dagger + a) (Default: None)
        :type Sm: list[sOP or sPOP or sNBO or sSOP, optional]
        :param method: The method used to represent the bath.
        :type method: {"heom", "pseudomode"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds: list, optional
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: int, optional

        :return: The total Generator now including the system bath terms
        :rtype: type(H)
        """

        dks, zks = super()._get_composite_params()

        from .heom import add_bosonic_bath_generator

        H = add_bosonic_bath_generator(
            H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip, method=method
        )
        return H

    def system_bath_generator(self, Sp, Sm=None, method="heom", binds=None, bskip=2):
        r"""Construct a sSOP containing the system bath Generator of the object.

        :param H: The total Generator
        :type H: SOP
        :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
        :type Sp: list[sOP or sPOP or sNBO or sSOP]
        :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\dagger + a) (Default: None)
        :type Sm: list[sOP or sPOP or sNBO or sSOP, optional]
        :param method: The method used to represent the bath.
        :type method: {"heom", "pseudomode"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds: list, optional
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: int, optional

        :return: The total Generator now including the system bath terms
        :rtype: sSOP
        """

        dks, zks = super()._get_composite_params()

        H = sSOP()
        from .heom import add_bosonic_bath_generator

        H = add_bosonic_bath_generator(
            H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip, method=method
        )
        return H


# This is currently completely incorrect.  We need to set this up to split this into filled and unfilled baths
class ExpFitFermionicBath(ExpFitOQSBath):
    r"""A class for handling a fermionic bath representing an exponential fit to a bath correlation function
    of the form

    .. math::
        C(t) = \sum_k d_{k} \exp(-z_k t)

    :param dk: The coefficient in the exponential decomposition
    :type dk: np.ndarray
    :param zk: The decay rates in the exponential decomposition
    :type zk: np.ndarray
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional

    """

    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(
            self, dk, zk, fermionic=True, combine_real=combine_real, tol=tol
        )
        self.truncate_modes()

    def truncate_modes(self, truncation=DepthTruncation(2)):
        r"""Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(2))
        :type truncation: TruncationBase, optional

        """
        self._mode_dims = truncation(self._dk, self._zk, True)

    def __str__(self):
        return (
            "fermionic bath: "
            + "\n \alpha "
            + str(self._ck)
            + "\n \nu "
            + str(self._wk)
            + "\n modes "
            + str(self._mode_dims)
            + "\n composite "
            + str(self._composite_modes)
        )

    def system_information(self, mode_comb=None, force_evaluate=False):
        r"""Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :param mode_comb: A mode combination object to apply to the system information class.  (Default: None)
        :type mode_comb: ModeCombination, optional
        :param force_evaluate: Forces evaluation of the system_modes object regardless of whether or not one has already been formed. (Default: False)
        :type force_evaluation: bool, optional

        :return: Bath system information
        :rtype: system_modes
        """

        if self._sysinf is None or force_evaluate:
            self._sysinf = system_modes(len(self._composite_modes))
            for ind, cmode in enumerate(self._composite_modes):
                self._sysinf[ind] = [fermion_mode() for x in cmode]

            if mode_comb is not None:
                self._sysinf = mode_comb(self._sysinf)
        return self._sysinf

    def add_system_bath_generator(self, H, Sp, Sm, method="heom", binds=None, bskip=2):
        r"""Attach the bath and system bath coupling Generators associated with this bath object to an existing SOP Generator

        :param H: The total Generator
        :type H: SOP
        :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
        :type Sp: list[sOP or sPOP or sNBO or sSOP]
        :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\dagger + a) (Default: None)
        :type Sm: list[sOP or sPOP or sNBO or sSOP, optional]
        :param method: The method used to represent the bath.
        :type method: {"heom", "pseudomode"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds: list, optional
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: int, optional

        :return: The total Generator now including the system bath terms
        :rtype: type(H)
        """

        dks, zks = super()._get_composite_params()

        from .heom import add_fermionic_bath_generator

        H = add_fermionic_bath_generator(
            H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip, method=method
        )
        return H

    def system_bath_generator(self, Sp, Sm, method="heom", binds=None, bskip=2):
        r"""Construct a sSOP containing the system bath Generator of the object.

        :param H: The total Generator
        :type H: SOP
        :param Sp: A list containing the left and right acting operators that couples to the bath annihilation operator terms
        :type Sp: list[sOP or sPOP or sNBO or sSOP]
        :param Sm: A list containing the left and right operator that couples to the bath creation operator terms.  If set to None then, we consider coupling of the form Sp(a^\dagger + a) (Default: None)
        :type Sm: list[sOP or sPOP or sNBO or sSOP, optional]
        :param method: The method used to represent the bath.
        :type method: {"heom", "pseudomode"}
        :param binds: A list containing the indices of the bath modes. If this is set to None, the bath modes will be placed in a contiguous block starting at index bskip (Default: None)
        :type binds: list, optional
        :param bskip: The index to start the contiguous block of bath indices.  This object is ignored if the binds parameter is specified. (Default: 1)
        :type bskip: int, optional

        :return: The total Generator now including the system bath terms
        :rtype: sSOP
        """
        H = sSOP()
        from .heom import add_fermionic_bath_generator

        H = add_fermionic_bath_generator(
            H, Sp, self._gk, self._wk, Sm=Sm, binds=binds, bskip=bskip, method=method
        )
        return H
