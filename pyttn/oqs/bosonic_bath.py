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

import numpy as np
import scipy as sp


class BosonicBath:
    r"""A class for managing a continuous bosonic gaussian bath.  This provides
    functions for computing non-interacting bath correlation functions, as well as
    decomposing the correlation function into a linear combination of
    complex valued exponentials (expfit) or oscillator terms (discretise)

    :param Jw: The bath spectral function defining the non-interacting correlation function
    :param S: The system operator
    :type S: SOP, optional
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional
    :param wmax: the maximum frequency bound, defaults to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, defaults to None
    :type wmin: float, optional
    """

    def __init__(self, Jw, S=None, beta=None, wmax=np.inf, wmin=None):
        self.Jw = Jw
        self.S = S
        self.beta = beta
        if wmin is None:
            wmin = self.find_wmin(wmax)

        self.wmin = wmin
        self.wmax = wmax

    def find_wmin(self, wmax, npoints=1000):
        r"""Computes an estimate of the minimum frequency used for discretising a bath.
        Here this is done by taking the maximum frequency and finding the largest value in
        the range from [-wmax, 0] that has the same spectral weight as the upper bound.

        :param wmax: the maximum frequency bound, defaults to self.wmax
        :type wmax: float, optional
        :return: the maximum and minimum frequency bounds
        :rtype: float, float
        """
        if self.beta is None:
            return 0
        else:
            if wmax == np.inf:
                return -np.inf
            else:
                Swmax = self.Sw(wmax)
                wrange = np.linspace(-wmax, 0, npoints, endpoint=False)
                Swmin = self.Sw(wrange)
                return wrange[np.argmax(Swmin > Swmax) - 1]

    def estimate_bounds(self, wmax=None):
        r"""Returns estimates for the upper and lower bounds of the spectral density to be used for the
        discretisation function

        :param wmax: the maximum frequency bound, defaults to self.wmax
        :type wmax: float, optional
        :return: the maximum and minimum frequency bounds
        :rtype: float, float
        """
        if wmax is None:
            wmax = self.wmax
        wmax = np.abs(wmax)
        wmin = self.find_wmin(wmax)

        return wmin, wmax

    # improve this code to make it all work a bit better
    def Ct(self, t, epsabs=1.49e-12, epsrel=1.49e-12, limit=2000, epsomega=1e-6):
        r"""Returns the value of the non-interacting bath correlation function evaluated at the time points t,
        defined by:

        .. math::
            C(t) = \frac{1}{\pi}\int_{\omega_{\mathrm{min}}}^{\omega_{\mathrm{max}}} J(\omega) f_B(\beta\omega) \exp(- i \omega t)

        :param t: time
        :type t: np.ndarray
        :param epsabs: absolute error tolerance.  (Default: 1.49e-12)
        :type epsabs: float or int, optional
        :param epsrel: relative error tolerance.  (Default: 1.49e-12)
        :type epsrel: float or int, optional
        :param limit: Upper bound on the number of subintervals used in the integration scheme used to evaluate the correlation function.  (Default: 2000)
        :type limit: float or int, optional
        :param epsomega: A bound used to split the integral to avoid singularities at zero that may occur due to the bose function.  (Default: 1e-6)
        :type epsomega: float or int, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        wmin = self.wmin
        wmax = self.wmax

        if self.beta is None and wmin < 0:
            wmin = 0
        Ctr = np.zeros(t.shape, dtype=np.complex128)
        Cti = np.zeros(t.shape, dtype=np.complex128)

        wc = epsomega
        # if wmin > wc we don't span zero
        if wmin >= wc:
            # if wmax is infinite then we do a standard integral
            if wmax == np.inf:
                Ctr = sp.integrate.quad_vec(
                    lambda x: self.Sw(x) * np.cos(x * t),
                    wmin,
                    wmax,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
                Cti = sp.integrate.quad_vec(
                    lambda x: self.Sw(x) * np.sin(x * t),
                    wmin,
                    wmax,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
            # otherwise we can make use of the weight function
            else:
                for ti in range(t.shape[0]):
                    Ctr[ti] = sp.integrate.quad(
                        lambda x: self.Sw(x),
                        wmin,
                        wmax,
                        weight="cos",
                        wvar=t[ti],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti[ti] = sp.integrate.quad(
                        lambda x: self.Sw(x),
                        wmin,
                        wmax,
                        weight="sin",
                        wvar=t[ti],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
        # if wmin < wc then we need to split the integral
        else:
            # if wmin is negative we will split it into either two or three regions
            # depending on the value of wmin and -wc.
            # Handle the region [wc, wmax]
            if wmax == np.inf:
                Ctr += sp.integrate.quad_vec(
                    lambda x: self.Sw(x) * np.cos(x * t),
                    wc,
                    wmax,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
                Cti += sp.integrate.quad_vec(
                    lambda x: self.Sw(x) * np.sin(x * t),
                    wc,
                    wmax,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
            # otherwise we can make use of the weight function
            else:
                for ti in range(t.shape[0]):
                    Ctr[ti] += sp.integrate.quad(
                        lambda x: self.Sw(x),
                        wc,
                        wmax,
                        weight="cos",
                        wvar=t[ti],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti[ti] += sp.integrate.quad(
                        lambda x: self.Sw(x),
                        wc,
                        wmax,
                        weight="sin",
                        wvar=t[ti],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    print(ti, Ctr[ti])

            # Handle the two regions [wmin, -wc], (-wc, wc)
            if wmin < -wc:
                if wmin == -np.inf:
                    Ctr += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.cos(x * t),
                        wmin,
                        -wc,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.sin(x * t),
                        wmin,
                        -wc,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                # otherwise we can make use of the weight function
                else:
                    for ti in range(t.shape[0]):
                        Ctr[ti] += sp.integrate.quad(
                            lambda x: self.Sw(x),
                            wmin,
                            -wc,
                            weight="cos",
                            wvar=t[ti],
                            epsabs=epsabs,
                            epsrel=epsrel,
                            limit=limit,
                        )[0]
                        Cti[ti] += sp.integrate.quad(
                            lambda x: self.Sw(x),
                            wmin,
                            -wc,
                            weight="sin",
                            wvar=t[ti],
                            epsabs=epsabs,
                            epsrel=epsrel,
                            limit=limit,
                        )[0]

                    Ctr += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.cos(x * t),
                        -wc,
                        wc,
                        points=[0],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.sin(x * t),
                        -wc,
                        wc,
                        points=[0],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]

            # Handle the regions (wmin, wc)
            else:
                if wmin <= 0:
                    Ctr += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.cos(x * t),
                        wmin,
                        wc,
                        points=[0],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.sin(x * t),
                        wmin,
                        wc,
                        points=[0],
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                else:
                    Ctr += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.cos(x * t),
                        wmin,
                        wc,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]
                    Cti += sp.integrate.quad_vec(
                        lambda x: self.Sw(x) * np.sin(x * t),
                        wmin,
                        wc,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )[0]

        return (Ctr - 1.0j * Cti) / np.pi

    def Ctexp(t, dk, zk):
        r"""Returns the value of the non-interacting bath correlation function evaluated
        at the time points t using the results of discretisation or expfit:

        :param t: time
        :type t: np.ndarray
        :param dk: the weights of each term in the fit
        :type dk: np.ndarray
        :param zk: the (complex) frequencies of each term in the fit
        :type zk: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        ret = np.zeros(t.shape, dtype=np.complex128)
        for i in range(len(dk)):
            ret += dk[i] * np.exp(-1.0j * zk[i] * t)
        return ret

    def Sw(self, w):
        r"""Returns the non-interacting bath spectral function at w

        :param w: frequency
        :type w: np.ndarray
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        if self.beta is None:
            return self.Jw(np.abs(w)) * np.where(w > 0, 1.0, 0.0)
        else:
            return (
                np.where(w > 0, self.Jw(w), -self.Jw(-w))
                * 0.5
                * (1.0 + 1.0 / np.tanh(self.beta * w / 2.0))
            )

    def discretise(self, discretisation_engine):
        r"""Returns the coupling constants and frequencies associated with a discretised representation of the bath

        :param discretisation_engine: An object defining how to discretise a continuous bath
        :type discretisation_engine: np.ndarray
        :return: Discrete system bath coupling constants :math:`g_k`and bath frequencies :math:`\omega_k`
        :rtype: np.ndarray, np.ndarray
        """

        if discretisation_engine.wmin is None:
            if self.wmin is None:
                discretisation_engine.wmin = -self.wmax
            else:
                discretisation_engine.wmin = self.wmin
        if discretisation_engine.wmax is None:
            discretisation_engine.wmin = 2 * self.wmax
        return discretisation_engine(self.Sw)

    def expfit(self, fitting_engine):
        r"""Returns the coefficients and decay rates associated with a sum-of-exponential decomposition of the bath correlation function

        :param fitting_engine: An object defining how to decompose a correlation function for a continuous bath into a sum-of-exponential decomposition
        :type fitting_engine: np.ndarray
        :return: Discrete system bath coupling constants :math:`g_k` and bath frequencies :math:`\omega_k`
        :rtype: np.ndarray, np.ndarray
        """

        from .bath_fitting import AAADecomposition, ESPRITDecomposition

        dk = None
        zk = None
        if isinstance(fitting_engine, AAADecomposition):
            if fitting_engine.wmax is None:
                fitting_engine.wmin = 2 * self.wmax
            if fitting_engine.wmin is None:
                if self.wmin is None or np.abs(self.wmin) < 1e-12:
                    fitting_engine.wmin = -2 * self.wmax
                else:
                    fitting_engine.wmin = 2 * self.wmin
            dk, zk, _ = fitting_engine(self.Sw)

        elif isinstance(fitting_engine, ESPRITDecomposition):
            dk, zk, _ = fitting_engine(lambda x: self.Ct(x))
        return dk, zk
