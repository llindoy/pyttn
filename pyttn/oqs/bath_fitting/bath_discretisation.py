# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Callable, Optional
from pyttn.ttnpp.utils import orthopol_discretisation, density_discretisation
import numpy as np


class BathDiscretisation:
    r"""Base class for bath discretisations

    :param Nb: The number of discrete points to find
    :type Nb: int
    :param wmin: The minimum allowed frequency.
    :type wmin: float
    :param wmax: The maximum allowed frequency.
    :type wmax: float
    """

    def __init__(self, Nb: int, wmin: float, wmax: float):
        self.Nb = Nb
        self.wmin = wmin
        self.wmax = wmax


class DensityDiscretisation(BathDiscretisation):
    r"""A class wrapping the density based discretisation approach.  Selects frequencies from a density of frequencies :math:`\rho(\omega)`
    according to the expression

    .. math::
        \int_{\omega_{\mathrm{min}}}^{\omega_k} \rho(\omega)\mathrm{d} \omega = k,

    with coupling constants then determined by

    .. math::
        g_k^2 = \frac{1}{\pi} \frac{S(\omega_k)}{\rho(\omega_k)}.

    Constructor arguments

    :param Nb: The number of discrete points to find
    :type Nb: int
    :param wmin: The minimum allowed frequency.
    :type wmin: float
    :param wmax: The maximum allowed frequency.
    :type wmax: float
    :param rho: The density of frequencies to use in the discretisation process. (Default: None)
    :type rho: callable or None, optional
    :param atol: Absolute tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 0)
    :type atol: float, optional
    :param rtol: Relative tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 1e-10)
    :type rtol: float, optional
    :param nquad: The degree of the gauss-Legendre polynomial based quadrature scheme used in the evaluation of integrals.  (Default: 100)
    :type nquad: int, optional
    :param wtol: Frequency tolerance used in Newton's method step for finding frequencies.  (Default: 1e-10)
    :type wtol: float, optional
    :param ftol: Error tolerance used in Newton's method step for finding frequencies.  (Default: 1e-10)
    :type ftol: float, optional
    :param niters: The maximum number of Newton's method iterations used to find frequencies.  (Default: 100)
    :type niters: int, optional
    :param wcut: A lower bound on the frequency to cutoff the 1/w.  (Default: 1e-8)
    :type wcut: float, optional

    Callable arguments:

    :param S: The spectral density function to be decomposed.
    :type S: callable
    :returns:
        - g (np.ndarray) - The coupling constants of the discrete bath modes
        - w (np.ndarray) - The frequencies of the discrete bath modes
    """

    def __init__(
        self,
        Nb: int,
        wmin: float,
        wmax: float,
        rho: Optional[
            Callable[[np.ndarray | float], np.ndarray | float | np.complex128]
        ] = None,
        atol: float = 0,
        rtol: float = 1e-10,
        nquad: int = 100,
        wtol: float = 1e-10,
        ftol: float = 1e-10,
        niters: float = 100,
        wcut: float = 1e-8,
    ):
        BathDiscretisation.__init__(self, Nb, wmin, wmax)
        self.rho = rho
        self.wcut = wcut
        self.atol = atol
        self.rtol = rtol
        self.nquad = nquad
        self.wtol = wtol
        self.ftol = ftol
        self.niters = niters

    # discretise a bosonic bath using the density algorithm.  Here we allow for specification of the density of frequencies (rho), if this isn't specified we use a density of frequencies
    # that is given by S(w)/w for np.abs(w) > 1e-12 and S(w)/1e-12 for np.abs(w) < 1e-12.
    # this currently doesn't seem to work
    def __call__(
        self, S: Callable[[np.ndarray | float], np.ndarray | float | np.complex128]
    ) -> tuple[np.ndarray, np.ndarray]:
        g = None
        w = None

        if self.rho is None:
            # @jit(nopython=True)
            def rhofunc(w):
                return S(w) / np.where(np.abs(w) < self.wcut, self.wcut, np.abs(w))

            g, w = density_discretisation.discretise(
                S,
                rhofunc,
                self.wmin,
                self.wmax,
                self.Nb,
                atol=self.atol,
                rtol=self.rtol,
                nquad=self.nquad,
                wtol=self.wtol,
                ftol=self.ftol,
                niters=self.niters,
            )

        else:
            g, w = density_discretisation.discretise(
                S,
                self.rho,
                self.wmin,
                self.wmax,
                self.Nb,
                atol=self.atol,
                rtol=self.rtol,
                nquad=self.nquad,
                wtol=self.wtol,
                ftol=self.ftol,
                niters=self.niters,
            )

        return np.array(g), np.array(w)


class OrthopolDiscretisation:
    r"""A class wrapping the orthonormal polynomial based discretisation scheme.  This scheme constructs a set of orthonormal polynomials
    satisfying the orthogonality constraint

    .. math::
        \int_{\omega_{\mathrm{min}}}^{\omega_\mathrm{max}} S(\omega) \pi_i(\omega) \pi_j(\omega) \mathrm{d}\omega = \delta_{ij}.

    The coupling constants and frequencies are then obtained from the weights and nodes, respectively, of the Gaussian quadrature rule associated
    with these orthonormal polynomials.

    Constructor arguments

    :param Nb: The number of discrete points to find
    :type Nb: int
    :param wmin: The minimum allowed frequency.
    :type wmin: float
    :param wmax: The maximum allowed frequency.
    :type wmax: float
    :param moment_scaling: A parameter to scale the frequency coordinate by in order to avoid over/underflows. (Default: None)
    :type moment_scaling: float or None, optional
    :param atol: Absolute tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 0)
    :type atol: float, optional
    :param rtol: Relative tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 1e-10)
    :type rtol: float, optional
    :param minbound: A parameter for early bailout of the moment scaling step due to underflows. (Default: 1e-25)
    :type minbound: float, optional
    :param maxbound: A parameter for early bailout of the moment scaling step due to overflows. (Default: 1e-25)
    :type maxbound: float, optional
    :param moment_scaling_steps: The number of iterations of moment scaling to perform. (Default: 4)
    :type moment_scaling_steps: int, optional
    :param nquad: The degree of the gauss-Legendre polynomial based quadrature scheme used in the evaluation of integrals.  (Default: 100)
    :type nquad: int, optional

    Callable arguments:

    :param S: The spectral density function to be decomposed.
    :type S: callable
    :returns:
        - g (np.ndarray) - The coupling constants of the discrete bath modes
        - w (np.ndarray) - The frequencies of the discrete bath m
    """

    def __init__(
        self,
        Nb: int,
        wmin: float,
        wmax: float,
        moment_scaling: Optional[float] = None,
        atol: float = 0,
        rtol: float = 1e-10,
        minbound: float = 1e-25,
        maxbound: float = 1e25,
        moment_scaling_steps: float = 4,
        nquad: int = 100,
    ):
        BathDiscretisation.__init__(self, Nb, wmin, wmax)
        self.moment_scaling = moment_scaling
        self.atol = atol
        self.rtol = rtol
        self.minbound = minbound
        self.maxbound = maxbound
        self.moment_scaling_steps = moment_scaling_steps
        self.nquad = nquad

    def find_moment_scaling_factor(
        S: Callable[[np.ndarray | float], np.ndarray | float | np.complex128],
        wmin: float,
        wmax: float,
        Nb: int,
        atol: float = 0,
        rtol: float = 1e-10,
        minbound: float = 1e-30,
        maxbound: float = 1e30,
        Nsteps: int = 5,
        nquad: int = 100,
    ) -> float:
        r"""Finds a constant to scale the frequency axis by in order to ensure well defined scaling of the modified moments as we go to very high moments.
        here this is done by computing the modified moments up to varying orders (with early termination if the values leave some min and max bounds)
        and fitting the resultant decay or growth to an exponential function.  Based on this fitting we extract a constant that aims to minimise the
        decay or growth, and repeats this process with growing orders (up to the maximum order we need for the discretisation) until it reaches a
        point that the moment decay does not leave the early termination bounds.  This may not always be optimal if the Nsteps variable is too large
        and in this case either try reducing Nsteps.

        :param S: The spectral density function to be decomposed.
        :type S: callable
        :param wmin: The minimum allowed frequency.
        :type wmin: float
        :param wmax: The maximum allowed frequency.
        :type wmax: float
        :param Nb: The number of discrete points to find
        :type Nb: int
        :param atol: Absolute tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 0)
        :type atol: float, optional
        :param rtol: Relative tolerance to use in the evaluation of integrals within the discretisation process.  (Default: 1e-10)
        :type rtol: float, optional
        :param minbound: A parameter for early bailout of the moment scaling step due to underflows. (Default: 1e-25)
        :type minbound: float, optional
        :param maxbound: A parameter for early bailout of the moment scaling step due to overflows. (Default: 1e-25)
        :type maxbound: float, optional
        :param Nsteps: The number of iterations of moment scaling to perform. (Default: 4)
        :type Nsteps: int, optional
        :param nquad: The degree of the gauss-Legendre polynomial based quadrature scheme used in the evaluation of integrals.  (Default: 100)
        :type nquad: int, optional

        """
        Nbs = (np.arange(Nsteps) + 1) * (Nb // (Nsteps + 1))
        ms = 1
        for Nbi in Nbs:
            Nbi = max(2, Nbi)
            moments = np.array(
                orthopol_discretisation.moments(
                    S,
                    wmin,
                    wmax,
                    Nbi,
                    moment_scaling=ms,
                    atol=atol,
                    rtol=rtol,
                    minbound=minbound,
                    maxbound=maxbound,
                    nquad=nquad,
                )
            )
            nm = np.polyfit(np.arange(moments.shape[0]), np.log(np.abs(moments)), 1)[0]
            ms = ms * np.exp(-nm)
            # if this hasn't exited early then we return at this point
            if moments.shape[0] == Nbi * 2:
                return ms
        return ms

    def __call__(
        self, S: Callable[[np.ndarray | float], np.ndarray | float | np.complex128]
    ) -> tuple[np.ndarray, np.ndarray]:
        g = None
        w = None

        # if the moment scaling parameter is not zero set its value
        if self.moment_scaling is None:
            self.moment_scaling = OrthopolDiscretisation.find_moment_scaling_factor(
                S,
                self.wmin,
                self.wmax,
                self.Nb,
                atol=self.atol,
                rtol=self.rtol,
                minbound=self.minbound,
                maxbound=self.maxbound,
                Nsteps=self.moment_scaling_steps,
                nquad=self.nquad,
            )

        g, w = orthopol_discretisation.discretise(
            S,
            self.wmin,
            self.wmax,
            self.Nb,
            moment_scaling=self.moment_scaling,
            atol=self.atol,
            rtol=self.rtol,
            nquad=self.nquad,
        )
        return np.array(g), np.array(w)
