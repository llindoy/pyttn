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

from typing import Callable, Optional, Union

import numpy as np
import scipy as sp

from pyttn.ttns import OPBase

from ..bath_fitting import (
    BathDiscretisation,
    CtExpFitDecomposition,
    ExpFitDecomposition,
    SwExpFitDecomposition,
)
from .bath import Bath


class FermionicBath(Bath):
    """A class for managing a continuous fermionic gaussian bath.  This provides
    functions for computing non-interacting bath correlation functions, as well as
    decomposing the correlation function into a linear combination of
    complex valued exponentials (expfit) or oscillator terms (discretise)

    :param Jw: The bath spectral function defining the non-interacting correlation function
    :type Jw: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]] 
    :param Sp: The system raising operators
    :type Sp: Optional[OPBase]
    :param Sm: The system raising operators
    :type Sm: Optional[OPBase]    
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional
    :param wmax: the maximum frequency bound, default to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, default to np.inf
    :type wmin: float, optional
    :param wtol: a value for determining wmin based on wmax.  See fermionic.bath.estimate_bounds, default to None
    :type wtol: float, optional
    """

    def __init__(
        self,
        Jw: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]],
        Sp: Optional[OPBase] = None,
        Sm: Optional[OPBase] = None,
        beta: Optional[float] = None,
        wmax: float = np.inf,
        wmin: Optional[float] = None,
        wtol: Optional[float] = None,
    ):
        self.Jw = Jw
        self.Sp = Sp
        self.Sm = Sm
        self.beta = beta
        self.wmin = wmin
        self.wmax = wmax
        self.wtol = wtol

    def Ct(
        self,
        t: np.ndarray,
        Ef: float = 0,
        sigma: str = "+",
        epsabs: float = 1.49e-12,
        epsrel: float = 1.49e-12,
        limit: int = 2000,
    ) -> np.ndarray:
        """Returns the value of the non-interacting bath correlation function evaluated
        at the time points t:

        .. math::
            C^{\\sigma}(t) = \\frac{1}{pi}\\int_{wmin}^{wmax} J(\\omega) f_F(\\sigma\\beta(\\omega - Ef)) exp(\\sigma i \\omega t)

        :param t: time
        :type t: np.ndarray
        :param Ef: The fermi energy, default to 0
        :type Ef: float, optional
        :param sigma: Whether to compute greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        wmin, wmax = self.estimate_bounds(Ef=Ef, sigma=sigma)
        Ct = np.zeros(t.shape, dtype=np.complex128)

        coeff = 1
        if sigma == "-":
            coeff = -1

        if wmax == np.inf or wmin == -np.inf:
            ctr = sp.integrate.quad_vec(
                lambda x: self.Sw(x, Ef=Ef, sigma=sigma) * np.cos(x * t),
                wmin,
                wmax,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )[0]
            cti = sp.integrate.quad_vec(
                lambda x: self.Sw(x, Ef=Ef, sigma=sigma) * np.sin(x * t),
                wmin,
                wmax,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )[0]
            Ct = ctr + coeff * 1.0j * cti
        else:
            for ti in range(t.shape[0]):
                ctr = sp.integrate.quad(
                    lambda x: self.Sw(x, Ef=Ef, sigma=sigma),
                    wmin,
                    wmax,
                    weight="cos",
                    wvar=t[ti],
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
                cti = sp.integrate.quad(
                    lambda x: self.Sw(x, Ef=Ef, sigma=sigma),
                    wmin,
                    wmax,
                    weight="sin",
                    wvar=t[ti],
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )[0]
                Ct[ti] = ctr + coeff * 1.0j * cti
        return Ct / np.pi

    def Ctexp(
        t: np.ndarray, dk: np.ndarray, zk: np.ndarray, sigma: str = "+"
    ) -> np.ndarray:
        """Returns the value of the non-interacting bath correlation function evaluated
        at the time points t using the results of discretisation or expfit:

        :param t: time
        :type t: np.ndarray
        :param dk: the weights of each term in the fit
        :type dk: np.ndarray
        :param zk: the (complex) frequencies of each term in the fit
        :type zk: np.ndarray
        :param sigma: Whether to compute greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        coeff = 1
        if sigma == "-":
            coeff = -1

        ret = np.zeros(t.shape, dtype=np.complex128)
        for i in range(len(dk)):
            ret += dk[i] * np.exp(coeff * 1.0j * zk[i] * t)
        return ret

    def fermi_distrib(self, w: np.ndarray, Ef: float) -> np.ndarray:
        """Returns the value fermi function at w and fermi energy Ef:

        :param w: frequency
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        if self.beta is None:
            return np.where(w <= Ef, 1.0, 0.0)
        else:
            if isinstance(w, np.ndarray):
                res = 0.0 * w
                res[w < Ef] = 1 / (1 + np.exp(self.beta * (w[w < Ef] - Ef)))
                res[w >= Ef] = np.exp(-self.beta * (w[w >= Ef] - Ef)) / (
                    1 + np.exp(-self.beta * (w[w >= Ef] - Ef))
                )
            else:
                if w < Ef:
                    return 1 / (1 + np.exp(self.beta * (w - Ef)))
                else:
                    return np.exp(-self.beta * (w - Ef)) / (
                        1 + np.exp(-self.beta * (w - Ef))
                    )

            return res

    def Sw(self, w: np.ndarray, Ef: float = 0, sigma: str = "+") -> np.ndarray:
        """Returns the non-interacting bath spectral function at w and fermi energy Ef

        .. math::
            S^{\\sigma}(\\omega) = J(\\omega) f_{f}(\\sigma \\omega; \\beta)


        :param w: frequency
        :type w: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        if sigma == "+":
            return self.Jw(w) * self.fermi_distrib(w, Ef)
        else:
            return self.Jw(w) * (1 - self.fermi_distrib(w, Ef))

    def estimate_bounds(
        self, wmax: Optional[float] = None, Ef: float = 0, sigma: str = "+"
    ) -> tuple[float, float]:
        """Returns estimates for the upper and lower bounds of the spectral density to be used for the
        discretisation function

        :param wmax: the maximum frequency bound, defaults to self.wmax
        :type wmax: float, optional
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: the maximum and minimum frequency bounds
        :rtype: float, float
        """
        if wmax is None:
            wmax = self.wmax
        wmin = self.wmin
        wtol = self.wtol
        wmax = np.abs(wmax)
        if self.beta is None:
            if sigma == "+":
                wmin = -wmax
                wmax = Ef
            else:
                wmin = Ef
        else:
            if wtol is None:
                wmin = -wmax
            else:
                if sigma == "+":
                    wmin = -wmax
                    wmax = min(wmax, (1.0 / self.beta * np.log(1 / wtol - 1)) + Ef)
                else:
                    wmin = min(wmax, (Ef - 1.0 / self.beta * np.log(1 / wtol - 1)))

        return wmin, wmax

    def discretise(
        self, discretisation_engine: BathDiscretisation, Ef: float = 0, sigma: str = "+"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the coupling constants and frequencies associated with a discretised representation of the bath

        :param discretisation_engine: An object defining how to discretise a continuous bath
        :type discretisation_engine: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: Discrete system bath coupling constants :math:`g_k`and bath frequencies :math:`\\omega_k`
        :rtype: np.ndarray, np.ndarray
        """

        wmin, wmax = self.estimate_bounds(Ef=Ef, sigma=sigma)
        if discretisation_engine.wmin is None:
            discretisation_engine.wmin = wmin
        if discretisation_engine.wmax is None:
            discretisation_engine.wmin = wmax
        return discretisation_engine(lambda x: self.Sw(x, Ef=Ef, sigma=sigma))

    def expfit(
        self, fitting_engine: ExpFitDecomposition, Ef: float = 0, sigma: str = "+"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the coefficients and decay rates associated with a sum-of-exponential decomposition of the bath correlation function

        :param fitting_engine: An object defining how to decompose a correlation function for a continuous bath into a sum-of-exponential decomposition
        :type fitting_engine: np.ndarray
        :param Ef: Fermi Energy
        :type Ef: float
        :param sigma: Whether to compute the spectral function associated with the greater (+) or lesser (-) Green's Function, default to +
        :type sigma: str, optional
        :return: Discrete system bath coupling constants :math:`g_k`and bath frequencies :math:`\\omega_k`
        :rtype: np.ndarray, np.ndarray
        """
        dk = None
        zk = None
        if isinstance(fitting_engine, SwExpFitDecomposition):
            wmin, wmax = self.estimate_bounds(Ef=Ef, sigma=sigma)
            wav = (wmax - wmin) / 2
            if fitting_engine.wmin is None:
                fitting_engine.wmin = wav - 2 * (wav - wmin)
            if fitting_engine.wmax is None:
                fitting_engine.wmin = wav + 2 * (wmax - wav)
            dk, zk, _ = fitting_engine(lambda x: self.Sw(x, Ef=Ef, sigma=sigma))

        elif isinstance(fitting_engine, CtExpFitDecomposition):
            dk, zk, _ = fitting_engine(lambda x: self.Ct(x, Ef=Ef, sigma=sigma))
        else:
            raise RuntimeError(
                "Failed to fit fermionic bath. Invalid fitting engine object."
            )
        return dk, zk
