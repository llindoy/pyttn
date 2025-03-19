# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from pyttn import oqs
from numba import jit
import matplotlib.pyplot as plt


def fit_bath(Nb, alpha, wc, s, dt, beta=None, nstep=1, method="orthopol", tol=1e-13, Nw=10):
    r"""A function for fitting and plotting the fit quality of a bath spectral density using the methods supported in pyTTN
    Here we consider a spectral density of the form
    .. math::
        J(\omega) = \frac{\pi}{2} \frac{\alpha}{\omega_c^{s-1}} \exp\left(-\frac{\omega}{\omega_c}\right)


    :param Nb: The number of bath modes to use in the discretisation
    :type Nb: int
    :param alpha: The Kondo parameter associated with the bath
    :type alpha: float
    :param wc: The bath cutoff frequency
    :type wc: float
    :param s: The exponent in the bath spectral density
    :type s: float
    :param dt: The timestep used for integration
    :type dt: float
    :param beta: The inverse temperature to use in the simulation.  For beta=None we perform a zero temperature simulation (default: None)
    :type beta: float, optional
    :param nstep: The number of timesteps to perform (default: 1)
    :type nstep: int, optional
    :param method: The method used to fit the bath correlation function.   (default: "orthopol")
    :type method: str, optional
    :param tol: An optional tolerance parameter for fitting.  (default: 1e-13)
    :type tol: float, optional
    :param Nw: A factor (that multiplies wc) to define a hard frequency limit for the discretisation approach.  (default: 10)
    :type Nw: float, optional

    """

    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    if method == "orthopol":
        # discretise the bath correleation function using the orthonormal polynomial based cutoff
        g, w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc))
        fitbath = oqs.DiscreteBosonicBath(g, w)

    elif method == "density":
        # discretise the bath correleation function using the orthonormal polynomial based cutoff
        g, w = bath.discretise(oqs.DensityDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc))
        fitbath = oqs.DiscreteBosonicBath(g, w)

    elif method == "ESPRIT":
        dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=Nb, tmax=nstep * dt, Nt=nstep))
        fitbath = oqs.ExpFitBosonicBath(dk, zk)

    elif method == "AAA":
        dk, zk = bath.expfit(oqs.AAADecomposition(tol=tol, K=Nb, wmin=-Nw * wc, wmax=Nw * wc))
        fitbath = oqs.ExpFitBosonicBath(dk, zk)

    plt.plot(t, np.abs(bath.Ct(t) - fitbath.Ct(t)), "k-")
    plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    # exponential bath cutoff parameters
    parser.add_argument("alpha", type=float)
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)

    # number of bath modes
    parser.add_argument("--N", type=int, default=16)

    parser.add_argument("--beta", type=float, default=None)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=10)

    # output file name
    parser.add_argument("--method", type=str, default="orthopol")

    args = parser.parse_args()

    nstep = int(args.tmax / args.dt) + 1
    fit_bath(
        args.N,
        args.alpha,
        args.wc,
        args.s,
        args.dt,
        beta=args.beta,
        nstep=nstep,
        method=args.method,
    )
