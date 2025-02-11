import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs, utils
from numba import jit


def fit_bath(Nb, alpha, wc, s, dt, beta = None, nstep = 1, method="orthopol", tol=1e-13, Nw=10):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    if method == "orthopol":
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))

        fitbath = oqs.DiscreteBosonicBath(g, w)

    elif method == "density":
        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        g,w = bath.discretise(oqs.DensityDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))

        fitbath = oqs.DiscreteBosonicBath(g, w)

    elif method == "ESPRIT":
        dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=Nb, tmax=nstep*dt, Nt = nstep))

        fitbath = oqs.ExpFitBosonicBath(dk, zk)

    elif method == "AAA":
        dk, zk = bath.expfit(oqs.AAADecomposition(tol=tol, K=Nb, wmin=-Nw*wc, wmax=Nw*wc))
        print(dk, zk)

        fitbath = oqs.ExpFitBosonicBath(dk, zk)

    import matplotlib.pyplot as plt
    Cte = bath.Ct(t)
    Ctfit = fitbath.Ct(t)
    plt.plot(t, np.real(bath.Ct(t)), 'k-')
    plt.plot(t, np.imag(bath.Ct(t)), 'k--')
    plt.plot(t, np.real(fitbath.Ct(t)), 'r-')
    plt.plot(t, np.imag(fitbath.Ct(t)), 'r--')
    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--N', type=int, default=16)

    parser.add_argument('--beta', type = float, default=None)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--method', type=str, default='orthopol')

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    fit_bath(args.N, args.alpha, args.wc, args.s, args.dt, beta=args.beta, nstep=nstep, method=args.method)

