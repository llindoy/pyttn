import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn import oqs
from numba import jit

import matplotlib.pyplot as plt

def sbm_dynamics():
    dt = 0.05
    nstep = 300
    t = np.arange(nstep+1)*dt

    alpha = 1
    wc = 5
    s=1

    eps = 0
    delta = 1
    beta = None

    sysinf = system_modes(1)
    sysinf[0] = spin_mode(2)

    #and add on the system parts
    Hsys = eps*sOP("sz", 0) + delta *sOP("sz", 0)
    S = sOP("sz", 0)

    #setup the function for evaluatS, ing the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc))*(1.5+np.cos(w)+0.5*np.cos(2*w))/2)*np.where(w > 0, 1.0, -1.0)#np.where(np.abs(w) < 1, 1, 0)#

    bath = oqs.BosonicBath(J, S, beta=beta)
    HEOMengine = oqs.OQSEngine(sysinf, Hsys, [oqs.BosonicBath(J, S, beta=beta) for i in range(4)])
    print(HEOMengine.system_liouvillian())
    HEOMengine.fit_baths([oqs.ESPRITDecomposition(8, tmax=dt*nstep, Nt=nstep), oqs.ESPRITDecomposition(16, tmax=dt*nstep, Nt=nstep), oqs.AAADecomposition(wmin=-100, wmax=100), oqs.AAADecomposition(wmin=-100, wmax=100, Naaa=2000, tol=1e-7)])
    HEOMengine.truncate_baths([oqs.EnergyTruncation(30, 16, 3), oqs.EnergyTruncation(10, 8, 3), oqs.EnergyTruncation(50, 130, 3), oqs.EnergyTruncation(35, 20, 3)])
    HEOMengine.bath_mode_combination(oqs.ModeCombination(6, 2000))
    #HEOMengine.bath_mode_combination({"nmode": 6, "nhilb": 2000})

    dt = 0.005
    nstep = 3000
    t = np.arange(nstep+1)*dt

    print(HEOMengine)

    data = HEOMengine.bath_fitting_info(t)
    plt.plot(t, bath.Ct(t), 'k-', linewidth=3)
    for d in data:
        d.plot()
    #data.plot()
    plt.show()

sbm_dynamics()
