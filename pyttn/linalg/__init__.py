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

from .orthogonalVectorExt import OrthogonalVector, orthogonal_vector
from .randomEngineExt import RandomEngine, random_engine
from .sparseMatrixExt import CSRMatrix, DiagonalMatrix, SparseMatrix, csr_matrix, diagonal_matrix
from .tensorExt import (
        Matrix,
        Tensor,
        Tensor3,
        Tensor4,
        Vector,
        available_backends,
        matrix,
        tensor,
        tensor_3,
        tensor_4,
        vector,
)

__all__ = [
        "vector",
        "matrix",
        "tensor_3",
        "tensor_4",
        "tensor",
        "csr_matrix",
        "diagonal_matrix",
        "available_backends",
        "Vector",
        "Matrix",
        "Tensor3",
        "Tensor4",
        "Tensor",
        "CSRMatrix",
        "SparseMatrix",
        "DiagonalMatrix",
        "OrthogonalVector",
        "orthogonal_vector",
        "random_engine",
        "RandomEngine",
        ]
