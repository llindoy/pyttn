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

from typing import Callable, Union

import numpy as np


class CorrelatedSpectralDensity:
    """A class for representing a correlated spectral density matrix.  This class uses a sparse representation
    of the spectral density matrix setting all unset variables to 0

    :param N: The number of correlated degrees of freedom associated with the spectral density
    :type N: int
    """

    def __init__(self, N: int):
        self._N = N
        self._spec_dense = {}
        self._zero_func = lambda x: 0 * x

    @property
    def shape(self) -> list[int]:
        """Return the shape of the Correlated Spectral Density matrix
        :returns: Shape of correlated spectral density
        :rtype: list[int]
        """
        return [self._N, self._N]

    def __setitem__(
        self,
        key: tuple[int, int],
        value: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]],
    ) -> None:
        """Set the element of the correlated spectral density matrix

        :param key: The index of the spectral density matrix to be set
        :type key: tuple[int, int]
        :param value: The value to be placed at location key
        :type value: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]
        """
        i, j = key
        if i >= self._N or j >= self._N or i < 0 or j < 0:
            raise IndexError("Failed to set item.  Index out of bounds.")
        self._spec_dense[key] = value

    def __getitem__(
        self, key: tuple[int, int]
    ) -> Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]:
        """Get the element of the correlated spectral density matrix.  If no index has been set this
        returns the zero function.

        :param key: The index of the spectral density matrix to be set
        :type key: tuple[int, int]
        :returns: The value to be placed at location key
        :rtype: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]]
        """
        i, j = key
        if i >= self._N or j >= self._N or i < 0 or j < 0:
            raise IndexError("Failed to set item.  Index out of bounds.")

        if key in self._spec_dense:
            return self._spec_dense[key]
        else:
            return self._zero_func

    def nonzero_elements(self) -> list[tuple[int, int]]:
        return list(self._spec_dense.keys())

    def __call__(self, w: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate the value of the spectral density matrix.

        :param w: The frequency to evaluate the spectral density matrix at.  Depend
        :type w: Union[np.ndarray, float]
        :returns: The value of the spectral density matrix at the specified frequencies.  If the frequencies are a float this returns a matrix.
                  If the frequencies are a numpy array this returns a rank 3 tensor
        :rtype: np.ndarray
        """

        res = None
        if isinstance(w, (float, int)):
            res = np.zeros((self._N, self._N), dtype=float)
            for k, v in self._spec_dense.items():
                res[k] = v(w)
        elif isinstance(w, np.ndarray):
            res = np.zeros((self._N, self._N, *w.shape))
            for k, v in self._spec_dense.items():
                res[k[0], k[1], :] = v(w)
        else:
            raise TypeError("Invalid type for correlated_spectral_density eval.")
        return res

    def trace(self, w: Union[np.ndarray, float]) -> Union[np.ndarray, float, complex, np.complex128]:
        """Evaluate the trace of the spectral density matrix.

        :param w: The frequency to evaluate the spectral density matrix at.  Depend
        :type w: Union[np.ndarray, float]
        :returns: The trace of the spectral density matrix evaluated at the point(s) w
        :rtype: np.ndarray
        """
        res = None
        if isinstance(w, (float, int)):
            res = 0
            for i in range(self._N):
                if (i, i) in self._spec_dense:
                    res += self._spec_dense[(i, i)](w)
        elif isinstance(w, np.ndarray):
            res = np.zeros(w.shape)
            for i in range(self._N):
                if (i, i) in self._spec_dense:
                    res = res + self._spec_dense[(i, i)](w)
        else:
            raise TypeError("Invalid type for correlated_spectral_density eval.")
        return res

    def sum(self, w: Union[np.ndarray, float]) -> Union[np.ndarray, float, complex, np.complex128]:
        """Evaluate the sum of all elements of the spectral density matrix.

        :param w: The frequency to evaluate the spectral density matrix at.  Depend
        :type w: Union[np.ndarray, float]
        :returns: The sum of all elements of the spectral density matrix evaluated at the point(s) w
        :rtype: np.ndarray
        """

        res = None
        if isinstance(w, (float, int)):
            res = 0
            for _, v in self._spec_dense.items():
                res += v(w)
        elif isinstance(w, np.ndarray):
            res = np.zeros(w.shape)
            for _, v in self._spec_dense.items():
                res = res + v(w)
        else:
            raise TypeError("Invalid type for correlated_spectral_density eval.")
        return res
