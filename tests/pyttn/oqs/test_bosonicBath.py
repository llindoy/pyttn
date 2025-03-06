import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import sys
import copy
import h5py


from pyttn import oqs, utils
from numba import jit

import pytest

@pytest.mark.parametrize("beta, expected_error", [ (None, 0.0008), (1, 0.0008)])
def test_orthopol(beta, expected_error):
    dt = 0.005
    tmax=10
    alpha=0.1
    wc=5
    s=1
    Nb=200
    Nw=10

    nstep = int(tmax/dt)+1
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))
    fitbath = oqs.DiscreteBosonicBath(g, w)
    res = np.abs(bath.Ct(t)-fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error

@pytest.mark.parametrize("beta, expected_error", [ (None, 0.03) ])
def test_density(beta, expected_error):
    dt = 0.005
    tmax=10
    alpha=0.1
    wc=5
    s=1
    Nb=2000
    Nw=10

    nstep = int(tmax/dt)+1
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    g,w = bath.discretise(oqs.DensityDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))
    fitbath = oqs.DiscreteBosonicBath(g, w)
    res = np.abs(bath.Ct(t)-fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


@pytest.mark.parametrize("beta, expected_error", [ (None, 0.0015), (1, 0.0007)])
def test_AAA(beta, expected_error):
    dt = 0.005
    tmax=10
    alpha=0.1
    wc=5
    s=1
    Nb=30
    Nw=10
    tol=1e-13

    nstep = int(tmax/dt)+1
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    dk, zk = bath.expfit(oqs.AAADecomposition(tol=tol, K=Nb, wmin=-Nw*wc, wmax=Nw*wc))
    fitbath = oqs.ExpFitBosonicBath(dk, zk)
    res = np.abs(bath.Ct(t)-fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error

@pytest.mark.parametrize("beta, expected_error", [ (None, 2e-6), (1, 3e-6)])
def test_ESPRIT(beta, expected_error):
    dt = 0.005
    tmax=10
    alpha=0.1
    wc=5
    s=1
    Nb=10
    Nw=10

    nstep = int(tmax/dt)+1
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=Nb, tmax=nstep*dt, Nt = nstep))
    fitbath = oqs.ExpFitBosonicBath(dk, zk)
    res = np.abs(bath.Ct(t)-fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


