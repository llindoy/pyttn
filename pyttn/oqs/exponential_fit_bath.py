import numpy as np
from .fermionic_bath import *
from .bosonic_bath import *

from pyttn.utils.truncate import *
from pyttn import system_modes, boson_mode, fermion_mode

class ExpFitOQSBath:
    def __init__(self, dk, zk, fermionic=False, combine_real=False, tol=1e-12):
        if(len(dk) != len(zk)):
            raise RuntimeError("Invalid bath decomposition")

        self._ck = dk
        self._wk = zk
        self._real_mode = []
        self._composite_modes = []

        _dk = []
        _zk = []

        counter = 0
        for i in range(len(dk)):
            if(combine_real and np.abs(np.imag(zk[i])) < tol):
                _zk.append(zk[i])           #set the mode frequency 
                _dk.append(dk[i])            #set the mode coupling constant
                self._real_mode.append(True)      #flag that this is a real valued mode

                #set up the information that will be used for additional mode combination.
                self._composite_modes.append([counter])
                counter = counter+1

            #otherwise we add on separate modes for forward and backward paths
            else:
                #handle the forward path object
                _zk.append(zk[i])
                _dk.append(np.sqrt(dk[i]))
                self._real_mode.append(False)

                #handle the backward path object
                _zk.append(np.conj(zk[i]))
                _dk.append(np.sqrt(np.conj(dk[i])))
                self._real_mode.append(False)

                #now set up the information that will be used for attempting additional mode combination
                self._composite_modes.append([counter, counter+1])
                counter = counter+2

        self._dk = np.array(_dk, dtype=np.complex128)
        self._zk = np.array(_zk, dtype=np.complex128)

        self._fermion = fermionic
        self._mode_dims = []

    def is_fermionic(self):
        return self._fermion

    @property
    def mode_dims(self):
        return self._mode_dims

    @property
    def dk(self):
        return self._dk

    @property
    def zk(self):
        return self._zk

class ExpFitBosonicBath(ExpFitOQSBath):
    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=False, combine_real=combine_real, tol=tol)
        self.truncate_modes()

    def truncate_modes(self, truncation = DepthTruncation(8)):
        self._mode_dims = truncation(self._dk, self._zk, False)

    def Ct(self, t):
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k]*np.exp(-self._wk[k]*t)
        return ret

    def system_information(self):
        ret = system_modes(len(self._composite_modes))
        for ind, cmode in enumerate(self._composite_modes):
            ret[ind] = [boson_mode(self._mode_dims[x]) for x in cmode]
        return ret


    #def add_bath_hamiltonian(self, H, Spl, Spr, Sml = None, Smr = None, binds=None):
    #    if binds  == None:
    #        binds = [x+1 for i in range(len(self._dk))]
    #
    #    H += 



    def __str__(self):
        return 'bosonic bath: \n ' + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)




#This is currently completely incorrect.  We need to set this up to split this into filled and unfilled baths
class ExpFitFermionicBath(ExpFitOQSBath):
    def __init__(self, dk, zk, combine_real=False, tol=1e-12):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=True, combine_real=combine_real, tol=tol)
        self.truncate_modes()

    def Ct(self, t):
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k]*np.exp(-self._wk[k]*t)
        return ret

    def truncate_modes(self, truncation = DepthTruncation(2)):
        self._mode_dims = truncation(self._dk, self._zk, True)

    def __str__(self):
        return 'fermionic bath: ' + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

    def system_information(self):
        ret = system_modes(len(self._composite_modes))
        for ind, cmode in enumrate(self._composite_modes):
            ret[ind] = [fermion_mode() for x in cmode]
        return ret




class FermionicBathFit:
    def __init__(self, bath, expbath, t):
        self._t = t
        self._dk = expbath._ck
        self._zk = expbath._zk
        self._Ctp = bath.Ct(t, sigma='+')
        self._Ctpfit = expbath.Ct(t)
        self._Ctm = bath.Ct(t, sigma='-')
        self._Ctmfit = expbath.Ct(t)

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            plt.semilogy(self._t, np.abs(self._Ctp-self._Ctpfit), label=r'$\Delta C^{+}(t)$')
            plt.semilogy(self._t, np.abs(self._Ctm-self._Ctmfit), label=r'$\Delta C^{-}(t)$')
        except:
            return



