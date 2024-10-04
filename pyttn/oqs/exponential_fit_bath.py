import numpy as np
from .fermionic_bath import *
from .bosonic_bath import *

from pyttn.utils.truncate import *
from pyttn.utils.mode_combination import ModeCombination

class ExpFitOQSBath:
    def __init__(self, dk, zk, fermionic=False):
        if(len(dk) != len(zk)):
            raise RuntimeError("Invalid bath decomposition")

        self._ck = dk
        self._wk = zk

        self._dk = np.zeros(2*len(dk), dtype=np.complex128)
        self._zk = np.zeros(2*len(zk), dtype=np.complex128)

        for i in range(len(dk)):
            self._dk[2*i] = np.sqrt(dk[i])
            self._dk[2*i+1] = np.sqrt(np.conj(dk[i]))
            self._zk[2*i] = -1.0j*zk[i]
            self._zk[2*i+1] = -1.0j*np.conj(zk[i])

        self._fermion = fermionic
        self._mode_dims = []
        self._composite_modes = []

    def mode_combination(self, mode_combination = ModeCombination(2, 1)):
        self._composite_modes = mode_combination(self._mode_dims, [i for i in range(len(self._mode_dims))], 2)

    def is_fermionic(self):
        return self._fermion


class ExpFitBosonicBath(ExpFitOQSBath):
    def __init__(self, Sl, Sr, dk, zk):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=False)
        self._Sl = Sl
        self._Sr = Sr

        self.truncate_modes()

    def truncate_modes(self, truncation = DepthTruncation(8)):
        self._mode_dims = truncation(self._dk, self._zk, False)
        self._composite_modes = [[i] for i in range(len(self._mode_dims))]

    def Ct(self, t):
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k]*np.exp(-self._wk[k]*t)
        return ret

    def __str__(self):
        return 'bosonic bath: \n Sl = ' + str(self._Sl) + '\n Sr = ' + str(self._Sr) + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)



class ExpFitFermionicBath(ExpFitOQSBath):
    def __init__(self, Spl, Spr, Sml, Smr, dk, zk):
        ExpFitOQSBath.__init__(self, dk, zk, fermionic=True)

        self._Spl = Spl
        self._Spr = Spr

        self._Sml = Sml
        self._Smr = Smr

        self.truncate_modes()

    def Ct(self, t):
        ret = np.zeros(t.shape, dtype=np.complex128)
        for k in range(len(self._ck)):
            ret += self._ck[k]*np.exp(-self._wk[k]*t)
        return ret

    def truncate_modes(self, truncation = DepthTruncation(2)):
        self._mode_dims = truncation(self._dk, self._zk, True)
        self._composite_modes = [[i] for i in range(len(self._mode_dims))]

    def __str__(self):
        return 'fermionic bath: \n S^{+}l = ' + str(self._Spl) + '\n S^{+}r = ' + str(self._Spr)  + '\n S^{-}l = ' + str(self._Sml) + '\n S^{-}r = ' + str(self._Smr) + '\n \\alpha ' + str(self._ck) + '\n \\nu ' + str(self._wk) + '\n modes ' + str(self._mode_dims) + '\n composite ' + str(self._composite_modes)

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

class BosonicBathFit:
    def __init__(self, bath, expbath, t):
        self._t = t
        self._dk = expbath._ck
        self._zk = expbath._zk
        self._Ct = bath.Ct(t)
        self._Ctfit = expbath.Ct(t)

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            #plt.plot(self._t, (self._Ct), label=r'$\Delta C(t)$')
            plt.plot(self._t, (self._Ctfit), label=r'$\Delta C(t)$')
        except:
            return

    #def save(self, filename, tag="bath_fitting", ftype='h5', append=False):
    #    #if ftype == 'h5':
    #    #    import h5py
    #    #    file = h5py.File(
    #    #elif ftype == 'json':
    #    #    import json


    #    #else:
    #    #    raise RuntimeError("Failed to save bosonic bath fit.  Invalid file type"):

def bath_fitting_quality(bath, expbath, t):
    if isinstance(bath, FermionicBath) and isinstance(expbath, ExpFitFermionicBath):
        return FermionicBathFit(bath, expbath, t)
    elif isinstance(bath, BosonicBath) and isinstance(expbath, ExpFitBosonicBath):
        return BosonicBathFit(bath, expbath, t)
    else:
        raise RuntimeError("Invalid bath types")


