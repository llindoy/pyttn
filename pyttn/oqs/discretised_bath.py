import numpy as np
from .fermionic_bath import *
from .bosonic_bath import *

from pyttn.utils.truncate import *
from pyttn import system_modes, boson_mode, fermion_mode


class discretisedOQSBath:
    """The base class for handling a bath representing a discretised bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k g_{k}^2 \exp(-1.0j w_k t)

    :param gk: The coefficient in the exponential decomposition
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

    def __init__(self, gk, wk, fermionic=False, tol=1e-12):
        if (len(dk) != len(zk)):
            raise RuntimeError("Invalid bath decomposition")

        self._gk = np.array(gk)
        self._wk = np.array(wk)
        self._composite_modes = []

        self._fermion = fermionic
        self._mode_dims = []

    def is_fermionic(self):
        """Returns whether or not the bath is fermionic
        :rtype: bool 
        """
        return self._fermion
    
    def Ct(self, t):
        """Returns the value of the non-interacting bath correlation function evaluated at the time points t, 
        defined by:

        .. math::
            C(t) = \\sum_k g_{k}^2 \exp(-1.0j*w_k t)

        :param t: time
        :type t: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += np.abs(self._gk[k])**2*np.exp(-1.0j*self._wk[k]*t)
        return ret

    @property
    def mode_dims(self):
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


class discretisedBosonicBath(discretisedOQSBath):
    """A class for handling a bosonic bath representing a discretised bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k g_{k}^2 \exp(-1.0j w_k t)

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarray
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional
    """

    def __init__(self, gk, wk, tol=1e-12):
        discretisedOQSBath.__init__(self, gk, wk, fermionic=False,
                                tol=tol)
        self.truncate_modes()

    def truncate_modes(self, truncation=DepthTruncation(8)):
        """Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(8))
        :type truncation: TruncationBase, optional

        """
        self._mode_dims = truncation(self._gk, self._wk, False)

    def system_information(self):
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :return: Bath system information
        :rtype: system_modes
        """
        if not len(self._mode_dims) == len(self._wk):
            raise RuntimeError(
                "Failed to compute system information object.  The bath object has not not been truncated.")

        ret = system_modes(len(self._mode_dims))
        for ind in rang(len(self._mode_dims)):
            ret[ind] = boson_mode(self._mode_dims[ind])
        return ret

    def __str__(self):
        return 'bosonic bath: \n ' + '\n \\alpha ' + str(self._gk) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

    # def add_bath_hamiltonian(self, H, Spl, Spr, Sml = None, Smr = None, binds=None):
    #    if binds  == None:
    #        binds = [x+1 for i in range(len(self._dk))]
    #
    #    H +=

class discretisedFermionicBath(discretisedOQSBath):
    """A class for handling a fermionic bath representing an exponential fit to a bath correlation function
    of the form

    .. math::
        C(t) = \\sum_k g_{k}^2 \exp(-1.0j w_k t)

    :param gk: The coefficient in the exponential decomposition
    :type gk: np.ndarray
    :param wk: The decay rates in the exponential decomposition
    :type wk: np.ndarray
    :param combine_real: Whether or not to combine real frequency modes (default False)
    :type combine_real: bool, optional
    :param tol: The tolerance used to determine if a mode is a real frequency mode (default 1e-12)
    :type tol: float, optional

    """

    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        discretisedOQSBath.__init__(self, dk, zk, fermionic=True,
                               combine_real=combine_real, tol=tol)
        self.truncate_modes()

    def truncate_modes(self, truncation=DepthTruncation(2)):
        """Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(2))
        :type truncation: TruncationBase, optional

        """
        self._mode_dims = truncation(self._dk, self._zk, True)

    def __str__(self):
        return 'fermionic bath: ' + '\n \\alpha ' + str(self._gk) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

    def system_information(self):
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :return: Bath system information
        :rtype: system_modes

        """
        ret = system_modes(len(self._mode_dims))
        for ind in rang(len(self._mode_dims)):
            ret[ind] = fermion_mode()
        return ret
