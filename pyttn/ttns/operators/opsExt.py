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

import pyttn.ttnpp.ops as ops

try:
    import pyttn.ttnpp.cuda.ops as cuops

    __cuda_import = True
except ImportError:
    __cuda_import = False
    cuops = None
import numpy as np
import scipy as sp


def __identity_blas(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return ops.identity_complex(*args)
    elif dtype == np.float64:
        return ops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")


def __identity_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return cuops.identity_complex(*args)
    elif dtype == np.float64:
        return cuops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")


def identity(*args, dtype=np.complex128, backend="blas"):
    r"""Factory function for constructing an identity matrix site operator

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
    if backend == "blas":
        return __identity_blas(*args, dtype=dtype)
    elif __cuda_import and backend == "cuda":
        return __identity_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.identity")


def __matrix_blas(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return ops.matrix_complex(*args)
    elif dtype == np.float64:
        return ops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")


def __matrix_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return cuops.matrix_complex(*args)
    elif dtype == np.float64:
        return cuops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")


def matrix(*args, dtype=np.complex128, backend="blas"):
    r"""Factory function for constructing a matrix site operator

    :type *args: Variable length list of arguments. For details see the dense matrix constructors.
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the matrix operator  (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: matrix operator
    :rtype: ops.matrix_complex or ops.matrix_real
    """
    if backend == "blas":
        return __matrix_blas(*args, dtype=dtype)
    elif __cuda_import and backend == "cuda":
        return __matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.matrix")


def __sparse_matrix_default_blas(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return ops.sparse_matrix_complex(*args)
    elif dtype == np.float64:
        return ops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")


def __sparse_matrix_blas(*args, dtype=np.complex128):
    if len(args) == 1:
        if isinstance(args[0], sp.sparse.csr_matrix) or isinstance(
            args[0], sp.sparse.coo_matrix
        ):
            m2 = None
            if isinstance(args[0], sp.sparse.csr_matrix):
                m2 = args[0]
            else:
                m2 = args[0].tocsr()

            if m2.dtype == np.complex128 or dtype == np.complex128:
                return ops.sparse_matrix_complex(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
            else:
                return ops.sparse_matrix_real(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
        else:
            __sparse_matrix_default_blas(*args, dtype=dtype)
    else:
        __sparse_matrix_default_blas(*args, dtype=dtype)


def __sparse_matrix_default_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return cuops.sparse_matrix_complex(*args)
    elif dtype == np.float64:
        return cuops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")


def __sparse_matrix_cuda(*args, dtype=np.complex128):
    if len(args) == 1:
        if isinstance(args[0], sp.sparse.csr_matrix) or isinstance(
            args[0], sp.sparse.coo_matrix
        ):
            m2 = None
            if isinstance(args[0], sp.sparse.csr_matrix):
                m2 = args[0]
            else:
                m2 = args[0].tocsr()

            if m2.dtype == np.complex128 or dtype == np.complex128:
                return cuops.sparse_matrix_complex(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
            else:
                return cuops.sparse_matrix_real(
                    m2.data, m2.indices, m2.indptr, ncols=m2.shape[1]
                )
        else:
            __sparse_matrix_default_cuda(*args, dtype=dtype)
    else:
        __sparse_matrix_default_cuda(*args, dtype=dtype)


def sparse_matrix(*args, dtype=np.complex128, backend="blas"):
    r"""Factory function for constructing a sparse matrix site operator

    :type *args: Variable length list of arguments. For details see the sparse matrix constructors
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the sparse_matrix operator  (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: sparse operator
    :rtype: ops.sparse_complex or ops.sparse_real
    """
    if backend == "blas":
        return __sparse_matrix_blas(*args, dtype=dtype)
    elif __cuda_import and backend == "cuda":
        return __sparse_matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.sparse_matrix")


def __diagonal_matrix_blas(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return ops.diagonal_matrix_complex(*args)
    elif dtype == np.float64:
        return ops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")


def __diagonal_matrix_cuda(*args, dtype=np.complex128):
    if dtype == np.complex128:
        return cuops.diagonal_matrix_complex(*args)
    elif dtype == np.float64:
        return cuops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")


def diagonal_matrix(*args, dtype=np.complex128, backend="blas"):
    r"""Factory function for constructing an diagonal matrix site operator

    :type *args: Variable length list of arguments. For details see the diagonal matrix constructors
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the diagonal_matrix operator  (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: diagonal operator
    :rtype: ops.diagonal_complex or ops.diagonal_real
    """
    if backend == "blas":
        return __diagonal_matrix_blas(*args, dtype=dtype)
    elif __cuda_import and backend == "cuda":
        return __diagonal_matrix_cuda(*args, dtype=dtype)
    else:
        raise RuntimeError("Invalid backend type for ops.diagonal_matrix")


__site_op_dict__ = {
    "identity": identity,
    "matrix": matrix,
    "sparse_matrix": sparse_matrix,
    "diagonal_matrix": diagonal_matrix,
}
