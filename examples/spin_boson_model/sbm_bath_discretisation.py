import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *

from numba import jit
from pyttn import oqs

import matplotlib.pyplot as plt


def discretise_bath(Nb, alpha, wc, s, beta = 1, Nw = 10, moment_scaling=None, atol=0, rtol=1e-10, method='orthopol'):

    wmax = Nw*wc
    g, w = oqs.discretise_bosonic(J,  Nb, wmax, method=method, beta=beta, atol=atol, rtol=rtol)
    #and compute the polaron transformed renormalisation of the truncated high frequency modes
    renorm = np.exp(-2.0/np.pi*scipy.integrate.quad(lambda x : J(x)/(x*x), wmax, np.inf)[0])

    return np.array(g), np.array(w), renorm

#compute the bath correlation function from the discretised frequencies
@jit(nopython=True)
def Ct(t, w, g):
    ct = np.zeros(t.shape, dtype=np.complex128)
    g2 = np.abs(g)**2
    for wi in range(w.shape[0]):
        ct += np.exp(-1.0j*w[wi]*t)*g2[wi]

    return ct


def sbm_dynamics(Nb, alpha, wc, s, dt, nstep = 1, Nw = 15):
    t = np.arange(nstep+1)*dt

    beta = 1

    #the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    Sz = None
    bath = oqs.bosonic_bath(J, Sz, beta=beta)
    ct = bath.Ct(t, Nw*wc)

    plt.plot(t, np.real(ct))
    plt.plot(t, np.imag(ct))

    g,w = bath.discretise(Nb, Nw*wc, method='orthopol')
    ct = Ct(t, w, g)
    plt.plot(t, np.real(ct))
    plt.plot(t, np.imag(ct))

    g,w = bath.discretise(Nb*20, Nw*wc, method='density')
    ct = Ct(t, w, g)
    plt.plot(t, np.real(ct))
    plt.plot(t, np.imag(ct))

    plt.show()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')
    parser.add_argument('alpha', type = float)
    parser.add_argument('--N', type = int, default=128)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--wb', type = float, default=10)
    args = parser.parse_args()

    nstep = int(30.0/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.dt, nstep = nstep, Nw=args.wb)
