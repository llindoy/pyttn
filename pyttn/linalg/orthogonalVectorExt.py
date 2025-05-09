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
from pyttn.linalg.tensorExt import Matrix, Vector
from pyttn.linalg.randomEngineExt import RandomEngine

# import the blas backend
import pyttn.ttnpp.linalg as la

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    __cuda_import = True
except ImportError:
    __cuda_import = False


class orthogonal_vector:
    @staticmethod
    def pad_random(a: Matrix, i: int, rng: RandomEngine) -> None:
        """Pad the columns of the matrix starting at row index i with random vectors that are set to be orthogonal to all
        current vectors using Gram schmidt

        :param a: A matrix that will have all rows including the ith padded with random orthogonal vectors
        :type a: Matrix
        :param i: The starting index of the vector to be padded with random variables
        :type i: int
        :param rng: The random number object used for generating the random vectors
        :type rng: RandomEngine
        """

        if rng.backend() != a.backend():
            raise RuntimeError(
                "Failed to pad random vectors on matrix.  The matrix and random engine object do not have the same backend."
            )

        if rng.backend() == "blas":
            if a.complex_dtype():
                la.orthogonal_vector_complex.pad_random(a, i, rng)
            else:
                la.orthogonal_vector_real.pad_random(a, i, rng)

        elif rng.backend() == "cuda" and __cuda_import:
            if a.complex_dtype():
                cula.orthogonal_vector_complex.pad_random(a, i, rng)
            else:
                cula.orthogonal_vector_real.pad_random(a, i, rng)

        else:
            raise RuntimeError("Invalid backend detected")

    @staticmethod
    def generate(a: Matrix, rng: RandomEngine) -> Vector:
        """Generate a new matrix with random orthogonal rows that are also orthogonal to the rows of matrix a

        :param a: A matrix that will have all rows including the ith padded with random orthogonal vectors
        :type a: Matrix
        :param rng: The random number object used for generating the random vectors
        :type rng: RandomEngine
        :return: A newly generated matrix with random orthogonal rows that are also orthogonal to all rows of matrix a
        :rtype: Vector

        """
        if rng.backend() != a.backend():
            raise RuntimeError(
                "Failed to pad random vectors on matrix.  The matrix and random engine object do not have the same backend."
            )

        if rng.backend() == "blas":
            if a.complex_dtype():
                return la.orthogonal_vector_complex.generate(a, rng)
            else:
                return la.orthogonal_vector_real.generate(a, rng)

        elif rng.backend() == "cuda" and __cuda_import:
            if a.complex_dtype():
                return cula.orthogonal_vector_complex.generate(a, rng)
            else:
                return cula.orthogonal_vector_real.generate(a, rng)

        else:
            raise RuntimeError("Invalid backend detected")
        
    @staticmethod
    def fill_random(a: Matrix, rng: RandomEngine) -> None:
        """Fill a matrix with random orthogonal vectors

        :param a: The matrix to fill
        :type a: Matrix
        :param rng: The random number object used for generating the random vectors
        :type rng: RandomEngine
        """

        return orthogonal_vector.pad_random(a, 0, rng)
