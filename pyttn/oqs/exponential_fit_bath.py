import numpy as np
from .fermionic_bath import *
from .bosonic_bath import *

from pyttn.utils.truncate import *
from pyttn import system_modes, boson_mode, fermion_mode


class ExpFitOQSBath:
    """The base class for handling a bath representing an exponential fit to a bath correlation function
    of the form

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

    def __init__(self, dk, zk, fermionic=False, combine_real=False, tol=1e-12):
        if (len(dk) != len(zk)):
            raise RuntimeError("Invalid bath decomposition")

        self._ck = dk
        self._wk = zk
        self._real_mode = []
        self._composite_modes = []

        _dk = []
        _zk = []

        counter = 0
        for i in range(len(dk)):
            if (combine_real and np.abs(np.imag(zk[i])) < tol):
                _zk.append(zk[i])  # set the mode frequency
                _dk.append(dk[i])  # set the mode coupling constant
                # flag that this is a real valued mode
                self._real_mode.append(True)

                # set up the information that will be used for additional mode combination.
                self._composite_modes.append([counter])
                counter = counter+1

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
                self._composite_modes.append([counter, counter+1])
                counter = counter+2

        self._dk = np.array(_dk, dtype=np.complex128)
        self._zk = np.array(_zk, dtype=np.complex128)

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
            C(t) = \\sum_k d_{k} \\exp(-z_k t)

        :param t: time
        :type t: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray

        """
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k]*np.exp(-self._wk[k]*t)
        return ret

    @property
    def mode_dims(self):
        """An array containing the dimensionality of each of the modes"""
        return self._mode_dims

    @property
    def dk(self):
        """An array containing the bath decomposition coefficients"""
        return self._dk

    @property
    def zk(self):
        """An array containing the bath decomposition decay rates"""
        return self._zk


class ExpFitBosonicBath(ExpFitOQSBath):
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

    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=False,
                               combine_real=combine_real, tol=tol)
        self.truncate_modes()

    def truncate_modes(self, truncation=DepthTruncation(8)):
        """Determines the local Hilbert space dimension (stored in mode_dims) of each of the bosonic bath modes
        using the truncation rule defined in the truncation object.

        :param truncation: The truncation rule used to determine the potentially frequency and coupling strength dependent local Hilbert space dimension for each mode in the bath. (Default DepthTruncation(8))
        :type truncation: TruncationBase, optional

        """
        self._mode_dims = truncation(self._dk, self._zk, False)

    def system_information(self):
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :return: Bath system information
        :rtype: system_modes
        """
        if not len(self._mode_dims) == len(self._zk):
            raise RuntimeError(
                "Failed to compute system information object.  The bath object has not not been truncated.")

        ret = system_modes(len(self._composite_modes))
        for ind, cmode in enumerate(self._composite_modes):
            ret[ind] = [boson_mode(self._mode_dims[x]) for x in cmode]
        return ret

    def __str__(self):
        return 'bosonic bath: \n ' + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

    # def add_bath_hamiltonian(self, H, Spl, Spr, Sml = None, Smr = None, binds=None):
    #    if binds  == None:
    #        binds = [x+1 for i in range(len(self._dk))]
    #
    #    H +=


# This is currently completely incorrect.  We need to set this up to split this into filled and unfilled baths
class ExpFitFermionicBath(ExpFitOQSBath):
    """A class for handling a fermionic bath representing an exponential fit to a bath correlation function
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

    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=True,
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
        return 'fermionic bath: ' + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

    def system_information(self):
        """Constructs and returns a system_modes object suitable for handling the bath degrees of freedom described by this object.

        :return: Bath system information
        :rtype: system_modes

        """
        ret = system_modes(len(self._composite_modes))
        for ind, cmode in enumerate(self._composite_modes):
            ret[ind] = [fermion_mode() for x in cmode]
        return ret
