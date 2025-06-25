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

import numpy as np
from typing import Callable, Optional, Union
from .bath import Bath
from ..bath_fitting import (
    ExpFitDecomposition,
    CtExpFitDecomposition,
    SwExpFitDecomposition,
    BathDiscretisation,
)

from pyttn.ttns import OP_type
from ..spectral_density import CorrelatedSpectralDensity
from .bosonic_bath import (
    evaluate_bosonic_bath_correlation_function,
    evaluate_bosonic_spectral_function,
)


class CorrelatedBosonicBath(Bath):
    r"""A class for managing a continuous correlated bosonic gaussian bath.  This provides
    functions for computing non-interacting bath correlation functions, as well as
    decomposing the correlation function into a linear combination of
    complex valued exponentials (expfit) or oscillator terms (discretise)

    :param Jw: The matrix valued bath spectral function defining the non-interacting correlation function
    :type Jw: CorrelatedSpectralDensity,
    :param S: The system operator
    :type S: list[OP_type], optional
    :param beta: The inverse temperature of the bath, defaults to None
    :type beta: float, optional
    :param wmax: the maximum frequency bound, defaults to np.inf
    :type wmax: float, optional
    :param wmin: the minimum frequency bound, defaults to None
    :type wmin: float, optional
    :param scalar_func: The function used to extract a scalar valued function from the matrix valued bath functions, defaults to "trace"
    :type scalar_func: Optional["trace" | "sum" | Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
    """

    def __init__(
        self,
        Jw: CorrelatedSpectralDensity,
        S: Optional[list[OP_type]] = None,
        beta: Optional[float] = None,
        wmax: float = np.inf,
        wmin: Optional[float] = None,
        scalar_func: Optional[
            Union[str, Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
        ] = "trace",
    ):
        self.Jw = Jw
        self.S = S
        self.beta = beta
        if wmin is None:
            wmin = self.find_wmin(wmax)

        self.wmin = wmin
        self.wmax = wmax
        self.scalarFunc = scalar_func

    def find_wmin(self, wmax: float, npoints: int = 1000) -> float:
        """Computes an estimate of the minimum frequency used for discretising a bath.
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
                Swmax = self.scalarSw(wmax, "trace")
                wrange = np.linspace(-wmax, 0, npoints, endpoint=False)
                Swmin = self.scalarSw(wrange, "trace")
                return wrange[np.argmax(Swmin > Swmax) - 1]

    def estimate_bounds(self, wmax: Optional[float] = None) -> tuple[float, float]:
        """Returns estimates for the upper and lower bounds of the spectral density to be used for the
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

    def Ct(
        self,
        t: Union[np.ndarray, float],
        epsabs: float = 1.49e-12,
        epsrel: float = 1.49e-12,
        limit: int = 2000,
        epsomega: float = 1e-6,
    ) -> np.ndarray:
        r"""Returns the value of the non-interacting bath correlation function evaluated at the time points t,
        defined by:

        .. math::
            \boldsymbol{C}(t) = \frac{1}{\pi}\int_{\omega_{\mathrm{min}}}^{\omega_{\mathrm{max}}} \boldsymbol{J}(\omega) f_B(\beta\omega) \exp(- i \omega t)

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

        return_scalar = False
        if isinstance(t, (float, int)):
            t = np.array([t])
            return_scalar = True

        Ctv = np.zeros((*self.Jw.shape, *t.shape), dtype=np.complex128)

        for i in self.Jw.nonzero_elements():
            Ctv[i[0], i[1]] = evaluate_bosonic_bath_correlation_function(
                lambda x: self.Sw_elem(x, *i),
                wmin,
                wmax,
                t,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
                epsomega=epsomega,
            )

        if return_scalar:
            return Ctv[:, :, 0]
        else:
            return Ctv

    def scalarCt(
        self,
        t: Union[np.ndarray, float],
        scalar_func: Optional[
            Union[str, Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
        ] = None,
        epsabs: float = 1.49e-12,
        epsrel: float = 1.49e-12,
        limit: int = 2000,
        epsomega: float = 1e-6,
    ) -> Union[np.ndarray, complex]:
        r"""Returns the bath correlation fucntion evaluated for some scalar argument
        defined by:

        .. math::
            \boldsymbol{C}(t) = \frac{1}{\pi}\int_{\omega_{\mathrm{min}}}^{\omega_{\mathrm{max}}} \boldsymbol{J}(\omega) f_B(\beta\omega) \exp(- i \omega t)

        :param t: time
        :type t: np.ndarray
        :param scalar_func: The function used to extract the scalar function from the bath correlation function, defaults to None
        :type scalar_func: Optional[str | Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
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

        return_scalar = False
        if isinstance(t, (float, int)):
            t = np.array([t])
            return_scalar = True

        Ctv = evaluate_bosonic_bath_correlation_function(
            lambda x: self.scalarSw(x, scalar_func=scalar_func),
            wmin,
            wmax,
            t,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            epsomega=epsomega,
        )

        if return_scalar:
            return Ctv[0]
        else:
            return Ctv

    def Ctexp(
        t: Union[np.ndarray, float], dk: np.ndarray, zk: np.ndarray
    ) -> Union[np.ndarray, complex]:
        """Returns the value of the non-interacting bath correlation function evaluated
        at the time points t using the results of discretisation or expfit:

        :param t: time
        :type t: Union[np.ndarray, float]
        :param dk: the weights of each term in the fit
        :type dk: np.ndarray
        :param zk: the (complex) frequencies of each term in the fit
        :type zk: np.ndarray
        :return: The bath correlation function
        :rtype: Union[np.ndarray, complex]
        """
        return_scalar = False
        if isinstance(t, (float, int)):
            t = np.array([t])
            return_scalar = True

        ret = np.zeros((dk.shape[0], dk.shape[1], *t.shape), dtype=np.complex128)
        for n in range(dk.shape[0]):
            for m in range(dk.shape[1]):
                for i in range(dk.shape[2]):
                    ret[n, m] += dk[n, m, i] * np.exp(-1.0j * zk[i] * t)

        if return_scalar:
            return ret[0]
        else:
            return ret

    def scalarSw(
        self,
        w: Union[np.ndarray, float],
        scalar_func: Optional[
            Union[str, Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
        ] = None,
    ) -> Union[np.ndarray, float]:
        """Returns the non-interacting bath spectral function at w associated with site indices i and j

        :param w: frequency
        :type w: Union[np.ndarray, float]
        :param scalar_func: The function used to extract the scalar function from the bath correlation function, defaults to None
        :type scalar_func: Optional[str | Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
        :return: The scalar bath correlation function
        :rtype: Union[np.ndarray, float]
        """
        if scalar_func is None:
            scalar_func = self.scalarFunc
        if scalar_func is None:
            raise RuntimeError(
                "Failed to compute scalar Sw.  No user defined scalar function variable."
            )
        if isinstance(scalar_func, str):
            if scalar_func == "trace":
                return evaluate_bosonic_spectral_function(
                    lambda x: self.Jw.trace(x), w, beta=self.beta
                )
            elif scalar_func == "sum":
                return evaluate_bosonic_spectral_function(
                    lambda x: self.Jw.sum(x), w, beta=self.beta
                )
            else:
                raise RuntimeError(
                    "Invalid scalar function variables.  String value not recongised."
                )
        elif callable(scalar_func):
            return evaluate_bosonic_spectral_function(
                lambda x: scalar_func(x), w, beta=self.beta
            )
        else:
            raise TypeError(
                "Invalid scalar function variables.  Variable type not supported."
            )

    def Sw_elem(self, w: Union[np.ndarray, float], i: int, j: int) -> Union[np.ndarray, float]:
        """Returns the non-interacting bath spectral function at w associated with site indices i and j

        :param w: frequency
        :type w: Union[np.ndarray, float]
        :param i: The row of the spectral density matrix to consider
        :type i: int
        :param j: The column of the spectral density matrix to consider
        :type j: int
        :return: The bath correlation function
        :rtype: Union[np.ndarray, float]
        """
        if (i, j) not in self.Jw.nonzero_elements():
            return 0.0 * w
        return evaluate_bosonic_spectral_function(self.Jw[i, j], w, beta=self.beta)

    def Sw(self, w: Union[np.ndarray, float]) -> np.ndarray:
        """Returns the non-interacting bath spectral function matrix at w

        :param w: frequency
        :type w: Union[np.ndarray, float]
        :return: The bath correlation function
        :rtype: np.ndarray
        """
        return_scalar = False
        if isinstance(w, (float, int)):
            w = np.array([w])
            return_scalar = True

        Swv = np.zeros((*self.Jw.shape, *w.shape), dtype=np.complex128)

        for i in self.Jw.nonzero_elements():
            Swv[i[0], i[1]] = self.Sw_elem(w, *i)

        if return_scalar:
            return Swv[:, :, 0]
        else:
            return Swv

    def discretise(
        self,
        discretisation_engine: BathDiscretisation,
        scalar_func: Optional[
            Union[str, Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]]
        ] = None,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the coupling constants and frequencies associated with a discretised representation of the bath

        :param discretisation_engine: An object defining how to discretise a continuous bath
        :type discretisation_engine: np.ndarray
        :param scalar_func: An optional scalar valued function used for determining the density of frequencies for discretisation, defaults to None
        :type scalar_func: Optional[str | Callable[ [Union[np.ndarray, float]], Union[np.ndarray, float] ]]
        :param **kwargs: Additional dictionary arguments that are currently not used by this function
        
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
        return discretisation_engine.fit_correlated(
            self.Sw, lambda x: self.scalarSw(x, scalar_func=scalar_func), self.Jw.shape[0]
        )

    def expfit(
        self, 
        fitting_engine: ExpFitDecomposition,
        scalar_func: Optional[
            Union[str, Callable[[Union[np.ndarray, float], float], Union[np.ndarray, complex]]]
        ] = None,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:        
        """Returns the coefficients and decay rates associated with a sum-of-exponential decomposition of the bath correlation function

        :param fitting_engine: An object defining how to decompose a correlation function for a continuous bath into a sum-of-exponential decomposition
        :type fitting_engine: np.ndarray
        :param scalar_func: An optional scalar valued function used for determining the density of frequencies, defaults to None
        :type scalar_func: Optional[str | Callable[ [Union[np.ndarray, float]], float |Union[np.ndarray, complex]]]
        :param **kwargs: Additional dictionary arguments used in the evaluation of the bath correlation if using a CtExpFitDecomposition

        :return: Discrete system bath coupling constants :math:`g_k` and bath frequencies :math:`\omega_k`
        :rtype: np.ndarray, np.ndarray
        """
        
        dk = None
        zk = None
        if isinstance(fitting_engine, SwExpFitDecomposition):
           if fitting_engine.wmax is None:
               fitting_engine.wmin = 2 * self.wmax
           if fitting_engine.wmin is None:
               if self.wmin is None or np.abs(self.wmin) < 1e-12:
                   fitting_engine.wmin = -2 * self.wmax
               else:
                   fitting_engine.wmin = 2 * self.wmin
           dk, zk, _ = fitting_engine.fit_correlated(self.Sw, lambda x: self.scalarSw(x, scalar_func=scalar_func), self.Jw.shape[0])

        elif isinstance(fitting_engine, CtExpFitDecomposition):
            dk, zk, _ = fitting_engine.fit_correlated(lambda x: self.Ct(x, **kwargs), lambda x: self.scalarCt(x, scalar_func=scalar_func, **kwargs), self.Jw.shape[0])
        else:
            raise RuntimeError(
                "Failed to fit fermionic bath. Invalid fitting engine object."
            )

        return dk, zk
