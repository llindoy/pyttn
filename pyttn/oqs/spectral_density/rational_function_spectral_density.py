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


import abc
import copy
from typing import Union

import numpy as np


class RationalFunctionSpectralDensity(metaclass=abc.ABCMeta):
    """An abstact base class for handling Rational Function spectral densities"""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, w):
        pass

    @abc.abstractmethod
    def PolesResiduesOrder(self) -> tuple[list, list, list]:
        return [], [], []


class SumSpectralDensity(RationalFunctionSpectralDensity):
    """A class for managing a sum of rational function spectral densities
    """
    def __init__(self):
        self._terms = []

    def __iadd__(self, t : RationalFunctionSpectralDensity):
        """In place addition to add terms to a SumSpectralDensity object"""
        if isinstance(t, SumSpectralDensity):
            self._terms = self._terms + t._terms
        else:
            self._terms.append(t)
        return self

    def __add__(self, t : RationalFunctionSpectralDensity):
        ret = SumSpectralDensity()
        ret._terms = copy.deepcopy(self._terms)
        ret += t
        return ret

    def __call__(
        self, w: Union[np.ndarray, float, complex, np.complex128]
    ) -> Union[np.ndarray, float, complex, np.complex128]:
        """Evaluate the Debye spectral density at the specified frequency value

        :param w: The frequency at which the spectral density is to be evaluated at
        :type w: Union[np.ndarray, float, complex, np.complex128]
        """
        v = 0.0*w
        for t in self._terms:
            v += t(w)
        return v

    def PolesResiduesOrder(self) -> tuple[list, list, list]:
        """Evaluate and return the poles, residues, and order of the poles of the Brownian oscillator spectral density in the Lower Half plane.
        These values depend on the values of Omega and gamma

        :returns: The poles and residues of the Brownian spectral density
        :rtype: tuple[list, list, list]
        """
        p = []
        r = []
        o = []
        for t in self._terms:
            lp, lr, lo = t.PolesResiduesOrder()
            p = p + lp
            r = r + lr
            o = o + lo
        return p, r, o


class DebyeSpectralDensity(RationalFunctionSpectralDensity):
    """A class for managing a Debye spectral density, that is
        .. math::
            J(\\omega) = \\frac{\\Lambda}{2}\\frac{\\omega_c\\omega}{\\omega^2+\\omega_c^2}

    :param Lambda: The bath reorganisation energy
    :type Lambda: float
    :param wc: The bath cutoff frequency
    :type wc: float
    """

    def __init__(self, Lambda: float, wc: float) -> None:
        self._lambda = Lambda
        self._wc = wc

    def __call__(
        self, w: Union[np.ndarray, float, complex, np.complex128]
    ) -> Union[np.ndarray, float, complex, np.complex128]:
        """Evaluate the Debye spectral density at the specified frequency value

        :param w: The frequency at which the spectral density is to be evaluated at
        :type w: Union[np.ndarray, float, complex, np.complex128]
        """
        wc = self._wc
        la = self._lambda
        return la / 2.0 * wc * w / (w * w + wc * wc)

    def __add__(self, t : RationalFunctionSpectralDensity):
        ret = SumSpectralDensity()
        ret._terms.append(copy.deepcopy(self))
        ret += t
        return ret

    @property
    def Lambda(self) -> float:
        return self._lambda

    @Lambda.setter
    def Lambda(self, value):
        self._lambda = value

    @property
    def wc(self) -> float:
        return self._wc

    @wc.setter
    def wc(self, value):
        self._wc = value

    def PolesResiduesOrder(self) -> tuple[list, list, list]:
        """Evaluate and return the poles, residues, and order of the poles of the Debye spectral density in the Lower Half plane

        :returns: The poles and residues of the Debye spectral density
        :rtype: tuple[list, list, list]
        """
        return [-1.0j * self._wc], [self._lambda * self._wc/4], [1]


class BrownianOscillatorSpectralDensity(RationalFunctionSpectralDensity):
    """A class for managing a Debye spectral density, that is
        .. math::
            J(\\omega) = \\frac{\\Lambda}{2}\\frac{\\gamma\\Omega^2\\omega}{(\\omega^2-\\Omega^2)^2+\\gamma^2\\omega^2}

    :param Lambda: The bath reorganisation energy
    :type Lambda: float
    :param Omega: The reaction coordinate frequencys
    :type Omega: float
    :param gamma: The reaction coordinate friction
    :type gamma: float
    """

    def __init__(self, Lambda: float, Omega: float, gamma: float) -> None:
        self._lambda = Lambda
        self._Omega = Omega
        self._gamma = gamma

    @property
    def Lambda(self) -> float:
        return self._lambda

    @Lambda.setter
    def Lambda(self, value):
        self._lambda = value

    @property
    def Omega(self) -> float:
        return self._Omega

    @Omega.setter
    def Omega(self, value):
        self._Omega = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def __add__(self, t : RationalFunctionSpectralDensity):
        ret = SumSpectralDensity()
        ret._terms.append(copy.deepcopy(self))
        ret += t
        return ret

    def __call__(
        self, w: Union[np.ndarray, float, complex, np.complex128]
    ) -> Union[np.ndarray, float, complex, np.complex128]:
        """Evaluate the Debye spectral density at the specified frequency value

        :param w: The frequency at which the spectral density is to be evaluated at
        :type w: Union[np.ndarray, float, complex, np.complex128]
        """
        Om = self._Omega
        la = self._lambda
        g = self._gamma
        return la / 2.0 * g * Om**2 * w / ((w**2 - Om**2) ** 2 + g**2 * w**2)

    def PolesResiduesOrder(self) -> tuple[list, list, list]:
        """Evaluate and return the poles, residues, and order of the poles of the Brownian oscillator spectral density in the Lower Half plane.
        These values depend on the values of Omega and gamma

        :returns: The poles and residues of the Brownian spectral density
        :rtype: tuple[list, list, list]
        """
        Om = self._Omega
        la = self._lambda
        g = self._gamma

        kappa = np.sqrt(complex((g / 2.0) ** 2 - Om**2))

        if 2 * Om == g:
            raise RuntimeError(
                "Critically Damped Brownian Oscillator Spectral Density currently not supported"
            )

        return (
            [-1.0j * (g / 2 + kappa), -1.0j * (g / 2 - kappa)],
            [-la / 8 * Om**2 / kappa, la / 8 * Om**2 / kappa],
            [1, 1],
        )



