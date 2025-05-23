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
import scipy

# Here we have not made use of any of the knowledge of the form of the A matrix
# This will be left as future updates.


def ESPRIT(Ct: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Implementation of the Estimation of Signal Parameters via Rotational Invariant Techniques (ESPRIT)
    method for decomposing a signal into the form

    .. math:
        C(t) \approx \sum_{k=1}^K w_k \exp(-\nu t)

    A. Paulraj, R. Roy, and T. Kailath, Proceedings of the IEEE 74, 1044 (1986).

    :param Ct: An array containing the values to be fit
    :type Ct: np.ndarray
    :param K: The set of support points used for interpolating the function
    :type K: np.ndarray

    :returns:
        - w(np.ndarray) - The weights in the decomposition
        - nu(np.ndarray) - The exponents in the decomposition
        - Ctfit(np.ndarray) - The function fit at the same points as the original Ct
    """

    # extract the frequencies in the signal using ESPRIT
    expnu = ESPRIT_frequencies(Ct, K)

    # construct the time dependence of each of the frequency signals
    ENU, Kn = np.meshgrid(expnu, np.arange(Ct.shape[0]))
    A = np.power(ENU, Kn)

    # extract the coupling constants by performing a laest squares fit
    weights = np.linalg.lstsq(A, Ct, rcond=None)[0]

    # now return the weights, exponents and fit correlation function
    return weights, -np.log(expnu), A @ weights


# Here we have not made use of any of the knowledge of the form of the Y matrix.
# This will seriously limit the number of points in Ct that can be efficiently
# fit using this approach.


def ESPRIT_frequencies(Ct: np.ndarray, K: np.ndarray) -> np.ndarray:
    r"""Extract the frequencies to be used in the ESPRIT algorithm

    :param Ct: An array containing the values to be fit
    :type Ct: np.ndarray
    :param K: The set of support points used for interpolating the function
    :type K: np.ndarray

    :returns:
        - expnu(np.ndarray) - The exponential of the frequencies to be used in the ESPRIT decomposition
    """
    ndata = Ct.shape[0]
    T = (ndata + 1) // 2
    # Form a hankel matrix from the vector of Cts.  This gives us
    # T signal points measured on an array of T detectors
    Y = scipy.linalg.hankel(Ct[:T], Ct[T - 1 : -1])

    # extract the signal subspace from the detector measurements
    U, _, _ = np.linalg.svd(Y)
    Us = U[:, :K]

    # divide into two virtual sub arrays
    S1 = Us[: T - 1, :]
    S2 = Us[1:, :]

    # and use these sub arrays to extract the frequencies
    P = np.linalg.lstsq(S1, S2, rcond=None)[0]
    ls, _ = np.linalg.eig(P)
    return ls[np.argsort(np.abs(ls))[::-1]]
