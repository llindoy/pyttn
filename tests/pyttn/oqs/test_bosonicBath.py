import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np



from pyttn import oqs
from numba import jit

import pytest


@pytest.mark.parametrize("beta, expected_error", [(None, 0.0008), (1, 0.0008)])
def test_orthopol(beta, expected_error):
    dt = 0.005
    tmax = 10
    alpha = 0.1
    wc = 5
    s = 1
    Nb = 200
    Nw = 10

    nstep = int(tmax / dt) + 1
    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(
            np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))
        ) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    g, w = bath.discretise(
        oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc)
    )
    fitbath = oqs.DiscreteBosonicBath(g, w)
    res = np.abs(bath.Ct(t) - fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


@pytest.mark.parametrize("beta, expected_error", [(None, 0.03)])
def test_density(beta, expected_error):
    dt = 0.005
    tmax = 10
    alpha = 0.1
    wc = 5
    s = 1
    Nb = 2000
    Nw = 10

    nstep = int(tmax / dt) + 1
    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(
            np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))
        ) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    g, w = bath.discretise(
        oqs.DensityDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc)
    )
    fitbath = oqs.DiscreteBosonicBath(g, w)
    res = np.abs(bath.Ct(t) - fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


@pytest.mark.parametrize("beta, expected_error", [(None, 0.0015), (1, 0.0007)])
def test_AAA(beta, expected_error):
    dt = 0.005
    tmax = 10
    alpha = 0.1
    wc = 5
    s = 1
    Nb = 30
    Nw = 10
    tol = 1e-13

    nstep = int(tmax / dt) + 1
    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(
            np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))
        ) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    dk, zk = bath.expfit(
        oqs.AAADecomposition(tol=tol, K=Nb, wmin=-Nw * wc, wmax=Nw * wc)
    )
    fitbath = oqs.ExpFitBosonicBath(dk, zk)
    res = np.abs(bath.Ct(t) - fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


@pytest.mark.parametrize("beta, expected_error", [(None, 2e-6), (1, 3e-6)])
def test_ESPRIT(beta, expected_error):
    dt = 0.01
    tmax = 5
    alpha = 0.1
    wc = 5
    s = 1
    Nb = 10

    nstep = int(tmax / dt) + 1
    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(
            np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))
        ) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=Nb, tmax=nstep * dt, Nt=nstep))
    fitbath = oqs.ExpFitBosonicBath(dk, zk)
    res = np.abs(bath.Ct(t) - fitbath.Ct(t))
    maxerr = np.max(res)
    assert maxerr < expected_error


def test_reorganisation_energy_ohmic():
    # for J(w) = (pi/2)*alpha*wc*(w/wc)*exp(-w/wc), the reorganisation energy
    # lambda = (1/pi) int_0^inf J(w)/w dw has the closed form alpha*wc/2.
    alpha = 1.25
    wc = 5.0

    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * (w / wc) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    bath = oqs.BosonicBath(J, beta=None)
    lam = bath.reorganisation_energy()
    assert np.abs(lam - alpha * wc / 2) < 1e-8


def test_reorganisation_energy_truncated_matches_bounds():
    alpha = 1.0
    wc = 5.0

    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * (w / wc) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    bath = oqs.BosonicBath(J, beta=None)
    lam_full = bath.reorganisation_energy()
    lam_truncated = bath.reorganisation_energy(wmax=8 * wc)
    # truncating the integration range should give a slightly smaller, but close, value
    assert lam_truncated < lam_full
    assert np.abs(lam_truncated - lam_full) < 0.01 * lam_full


def test_reorganisation_energy_debye_shortcut():
    debye = oqs.DebyeSpectralDensity(Lambda=3.7, wc=5.0)
    bath = oqs.BosonicBath(debye, beta=None)
    assert bath.reorganisation_energy() == pytest.approx(3.7)
