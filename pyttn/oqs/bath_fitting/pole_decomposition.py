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

try:
    from scipy.integrate import nsum
except ImportError:
    def nsum(F: callable, a, b, maxterms = 1048576):
        if a > b:
            t = a
            a = b
            b = t
        if b == np.inf:
            b = maxterms
        if a == -np.inf:
            a = -maxterms
        return np.sum(F(np.arange(a, b)))

from ..spectral_density.rational_function_spectral_density import (
    RationalFunctionSpectralDensity,
)
from .expfit import PoleDecomposition


class MatsubaraDecomposition(PoleDecomposition):
    def __init__(self, K):
        self._K = K

    def bosonic(self, J: RationalFunctionSpectralDensity, beta: float) -> tuple[np.ndarray, np.ndarray, complex]:
        if not isinstance(J, RationalFunctionSpectralDensity):
            raise RuntimeError("Pole decompositions require Rational Function spectral density")
        p, r, o = J.PolesResiduesOrder()

        for order in o:
            if order != 1:
                raise RuntimeError("Failed to perform Matsubara decomposition.  Currently only simple poles are supported.")
            
        K = self._K
        dk = np.zeros((len(p)+K), dtype=np.complex128)
        zk = np.zeros((len(p)+K), dtype=np.complex128)


        for i in range(len(p)):
            zk[i] = 1.0j*p[i]
            dk[i] = -2*np.pi*1.0j*r[i] * 0.5 * (1.0 + 1.0 / np.tanh(beta * p[i] / 2.0))

        #the matsubara frequencies
        for k in range(K):
            nuk = 2*np.pi*(k+1)/beta
            zk[k+len(p)] = nuk
            dk[k+len(p)] = -2.0/beta * J(-1.0j*nuk)

        #compute the truncation correction
        deltakr = nsum(lambda k : -np.real(J(-2.0j*np.pi*(k+1)/beta)/(np.pi*(k+1))), K, np.inf)
        deltaki = nsum(lambda k : -np.imag(J(-2.0j*np.pi*(k+1)/beta)/(np.pi*(k+1))), K, np.inf)

        return dk/np.pi, zk, (deltakr.sum+1.0j*deltaki.sum)/np.pi


    def fermionic(self, J: RationalFunctionSpectralDensity, beta: float) -> tuple[np.ndarray, np.ndarray, complex]:
        pass