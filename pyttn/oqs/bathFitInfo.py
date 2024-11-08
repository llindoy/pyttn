import numpy as np
from .fermionic_bath import *
from .bosonic_bath import *

from pyttn.utils.truncate import *
from pyttn import system_modes, boson_mode, fermion_mode

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


