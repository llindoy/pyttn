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


from abc import ABCMeta, abstractmethod

import pyttn.ttnpp.ops as ops
from pyttn.linalg import Diagonal_Matrix, Matrix, Sparse_Matrix

if hasattr(ops, "identity_real"):
    _real_ops=True
else:
    _real_ops = False

try:
    import pyttn.ttnpp.cuda.ops as cuops

    _cuda_import = True
except ImportError:
    _cuda_import = False
import numpy as np
import scipy as sp


def _identity_blas(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return ops.identity_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return ops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")


def _identity_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return cuops.identity_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return cuops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")


def _identity(*args, dtype=np.complex128, backend="blas"):
    if backend == "blas":
        return _identity_blas(*args, dtype=dtype)
    elif _cuda_import and backend == "cuda":
        return _identity_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.identity")


def _matrix_blas(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return ops.matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return ops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")


def _matrix_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return cuops.matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return cuops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")


def _matrix(*args, dtype=np.complex128, backend="blas"):
    if backend == "blas":
        return _matrix_blas(*args, dtype=dtype)
    elif _cuda_import and backend == "cuda":
        return _matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.matrix")


def _sparse_matrix_default_blas(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return ops.sparse_matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return ops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")


def _sparse_matrix_blas(*args, dtype=np.complex128):
    if len(args) == 1:
        if isinstance(args[0], sp.sparse.csr_matrix) or isinstance(
            args[0], sp.sparse.coo_matrix
        ):
            m2 = None
            if isinstance(args[0], sp.sparse.csr_matrix):
                m2 = args[0]
            else:
                m2 = args[0].tocsr()

            if m2.dtype == np.complex128 or m2.dtype is complex or dtype is complex or dtype == np.complex128:
                return ops.sparse_matrix_complex(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
            else:
                return ops.sparse_matrix_real(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
        else:
            _sparse_matrix_default_blas(*args, dtype=dtype)
    else:
        _sparse_matrix_default_blas(*args, dtype=dtype)


def _sparse_matrix_default_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return cuops.sparse_matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return cuops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")


def _sparse_matrix_cuda(*args, dtype=np.complex128):
    if len(args) == 1:
        if isinstance(args[0], sp.sparse.csr_matrix) or isinstance(
            args[0], sp.sparse.coo_matrix
        ):
            m2 = None
            if isinstance(args[0], sp.sparse.csr_matrix):
                m2 = args[0]
            else:
                m2 = args[0].tocsr()

            if m2.dtype == np.complex128 or m2.dtype is complex or dtype is complex or dtype == np.complex128:
                return cuops.sparse_matrix_complex(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
            else:
                return cuops.sparse_matrix_real(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
        else:
            _sparse_matrix_default_cuda(*args, dtype=dtype)
    else:
        _sparse_matrix_default_cuda(*args, dtype=dtype)


def _sparse_matrix(*args, dtype=np.complex128, backend="blas"):
    if backend == "blas":
        return _sparse_matrix_blas(*args, dtype=dtype)
    elif _cuda_import and backend == "cuda":
        return _sparse_matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.sparse_matrix")


def _diagonal_matrix_blas(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return ops.diagonal_matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return ops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")


def _diagonal_matrix_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128 or dtype is complex:
        return cuops.diagonal_matrix_complex(*args)
    elif (dtype == np.float64 or dtype is float) and _real_ops:
        return cuops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")


def _diagonal_matrix(*args, dtype=np.complex128, backend="blas"):
    if backend == "blas":
        return _diagonal_matrix_blas(*args, dtype=dtype)
    elif _cuda_import and backend == "cuda":
        return _diagonal_matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.diagonal_matrix")


class siteOp(metaclass=ABCMeta):
    def __new__(cls, *args, type=None, dtype=np.complex128, backend="blas"):
        if type == "identity":
            return _identity(*args, dtype=dtype, backend=backend)
        elif type == "matrix":
            return _matrix(*args, dtype=dtype, backend=backend)
        elif type == "sparse_matrix" or type == "sparse matrix":
            return _sparse_matrix(*args, dtype=dtype, backend=backend)
        elif type == "diagonal_matrix" or type == "diagonal matrix":
            return _diagonal_matrix(*args, dtype=dtype, backend=backend)

    @abstractmethod
    def size(self) -> int:
        """Return the dimension of the space the siteOp acts on

        :return: The local Hilbert space dimension
        :rtype: int
        """
        pass

    @abstractmethod
    def is_identity(self) -> bool:
        """Returns whether or not the present operator is an identity operator

        :return: Whether the operator is an identity
        :rtype: bool
        """
        pass

    @abstractmethod
    def is_resizable(self) -> bool:
        """Returns whether or not the operator can be resized.

        :return: Whether the operator can be resized
        :rtype: bool
        """
        pass

    @abstractmethod
    def resize(self, n: int):
        """Resize the operator changing its local Hilbert space dimension

        :param n: The new size of the siteOp
        :type n: int
        """
        pass

    @abstractmethod
    def clone(self) -> "siteOp":
        """Returns a copy of the present operator

        :return: A copy of the operator
        :rtype: siteOp
        """
        pass

    @abstractmethod
    def transpose(self) -> "siteOp":
        """Returns the transpose present operator

        :return: The transpose of the operator
        :rtype: siteOp
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the siteOp is storing a complex valued dtype

        :return: whether or not the siteOp is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the siteOp object

        :return: The string representation of the siteOp
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the siteOp

        :return: The backend type of the object
        :rtype: str
        """
        pass


class identity(siteOp):
    """A class for handling identity valued site operators"""

    def __new__(cls, *args, dtype=np.complex128, backend="blas"):
        """Factory function for constructing an identity matrix site operator

        :type *args: Variable length list of arguments. Allowed options are
            - empty - Default construct identity operator
            - size (int) - Construct an identity operator of a specified size
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the identity operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: identity operator
        :rtype: ops.identity_complex or ops.identity_real
        """
        return _identity(*args, dtype=dtype, backend=backend)


identity.register(ops.identity_complex)

if _real_ops:
    identity.register(ops.identity_real)

if _cuda_import:
    identity.register(cuops.identity_complex)
    if _real_ops:
        identity.register(cuops.identity_real)


class matrix(siteOp):
    """A class for handling matrix valued site operators"""

    def __new__(cls, *args, dtype=np.complex128, backend="blas"):
        """Factory function for constructing a matrix site operator

        :type *args: Variable length list of arguments. For details see the dense matrix constructors.
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the matrix operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: matrix operator
        :rtype: ops.matrix_complex or ops.matrix_real
        """
        return _matrix(*args, dtype=dtype, backend=backend)

    def matrix(self) -> Matrix:
        """Return the Matrix object defining this operator

        :return: The matrix representation of this operator
        :rtype: Matrix
        """
        pass


matrix.register(ops.matrix_complex)
if _real_ops:
    matrix.register(ops.matrix_real)

if _cuda_import:
    matrix.register(cuops.matrix_complex)

    if _real_ops:
        matrix.register(cuops.matrix_real)


class sparse_matrix(siteOp):
    """A class for handling sparse matrix valued site operators"""

    def __new__(cls, *args, dtype=np.complex128, backend="blas"):
        """Factory function for constructing a sparse matrix site operator

        :type *args: Variable length list of arguments. For details see the sparse matrix constructors
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the sparse_matrix operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: sparse operator
        :rtype: ops.sparse_complex or ops.sparse_real
        """
        return _sparse_matrix(*args, dtype=dtype, backend=backend)

    def matrix(self) -> Sparse_Matrix:
        """Return the Sparse_Matrix object defining this operator

        :return: The matrix representation of this operator
        :rtype: Sparse_Matrix
        """
        pass


sparse_matrix.register(ops.sparse_matrix_complex)

if _real_ops:
    sparse_matrix.register(ops.sparse_matrix_real)

if _cuda_import:
    sparse_matrix.register(cuops.sparse_matrix_complex)

    if _real_ops:
        sparse_matrix.register(cuops.sparse_matrix_real)


class diagonal_matrix(siteOp):
    """A class for handling diagonal matrix valued site operators"""

    def __new__(cls, *args, dtype=np.complex128, backend="blas"):
        """Factory function for constructing an diagonal matrix site operator

        :type *args: Variable length list of arguments. For details see the diagonal matrix constructors
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the diagonal_matrix operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: diagonal operator
        :rtype: ops.diagonal_complex or ops.diagonal_real
        """
        return _diagonal_matrix(*args, dtype=dtype, backend=backend)

    def matrix(self) -> Diagonal_Matrix:
        """Return the Diagonal_Matrix object defining this operator

        :return: The matrix representation of this operator
        :rtype: Diagonal_Matrix
        """
        pass


diagonal_matrix.register(ops.diagonal_matrix_complex)

if _real_ops:
    diagonal_matrix.register(ops.diagonal_matrix_real)

if _cuda_import:
    diagonal_matrix.register(cuops.diagonal_matrix_complex)

    if _real_ops:
        diagonal_matrix.register(cuops.diagonal_matrix_real)
