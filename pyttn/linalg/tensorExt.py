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

# import the blas backend
import pyttn.ttnpp.linalg as la

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    __cuda_import = True
except ImportError:
    __cuda_import = False

import numpy as np


def __vector_blas(v):
    if v.dtype == np.float64:
        return la.vector_real(v)
    elif v.dtype == int:
        return la.vector_real(np.array(v, dtype=np.float64))
    elif v.dtype == np.complex128:
        return la.vector_complex(v)
    else:
        raise RuntimeError("Invalid dtype for linalg.vector")


def __vector_cuda(v):
    if v.dtype == np.float64:
        return cula.vector_real(v)
    elif v.dtype == int:
        return cula.vector_real(np.array(v, dtype=np.float64))
    elif v.dtype == np.complex128:
        return cula.vector_complex(v)
    else:
        raise RuntimeError("Invalid dtype for linalg.vector")


def vector(v, backend="blas"):
    r"""
    A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
    used by the C++ layer of pyTTN.

    :param M: The numpy array
    :type M: np.ndarray
    :param backend: The backend to use for calculation. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 1> object
    :rtype: vector_real, vector_complex
    """

    if backend == "blas":
        return __vector_blas(v)
    elif __cuda_import and backend == "cuda":
        return __vector_cuda(v)
    else:
        raise RuntimeError("Invalid backend type for linalg.vector")


def __matrix_blas(M):
    if M.dtype == np.float64:
        return la.matrix_real(M)
    elif M.dtype == int:
        return la.matrix_real(np.array(M, dtype=np.float64))
    elif M.dtype == np.complex128:
        return la.matrix_complex(M)
    else:
        raise RuntimeError("Invalid dtype for linalg.matrix")


def __matrix_cuda(M):
    if M.dtype == np.float64:
        return cula.matrix_real(M)
    elif M.dtype == int:
        return cula.matrix_real(np.array(M, dtype=np.float64))
    elif M.dtype == np.complex128:
        return cula.matrix_complex(M)
    else:
        raise RuntimeError("Invalid dtype for linalg.matrix")


def matrix(M, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,2> type
    used by the C++ layer of pyTTN.

    :param M: The numpy matrix
    :type M: np.ndarray
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: matrix_real, matrix_complex
    """

    if backend == "blas":
        return __matrix_blas(M)
    elif __cuda_import and backend == "cuda":
        return __matrix_cuda(M)
    else:
        raise RuntimeError("Invalid backend type for linalg.matrix")


def __tensor_3_blas(T):
    if T.dtype == np.float64:
        return la.tensor_3_real(T)
    elif T.dtype == int:
        return la.tensor_3_real(np.array(T, dtype=np.float64))
    elif T.dtype == np.complex128:
        return la.tensor_3_complex(T)
    else:
        raise RuntimeError("Invalid dtype for linalg.tensor_3")


def __tensor_3_cuda(T):
    if T.dtype == np.float64:
        return cula.tensor_3_real(T)
    elif T.dtype == int:
        return cula.tensor_3_real(np.array(T, dtype=np.float64))
    elif T.dtype == np.complex128:
        return cula.tensor_3_complex(T)
    else:
        raise RuntimeError("Invalid dtype for linalg.tensor_3")


def tensor_3(T, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,3> type
    used by the C++ layer of pyTTN.

    :param T: The numpy tensor
    :type T: np.ndarray
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: tensor_3_real, tensor_3_complex
    """

    if backend == "blas":
        return __tensor_3_blas(T)
    elif __cuda_import and backend == "cuda":
        return __tensor_3_cuda(T)
    else:
        raise RuntimeError("Invalid backend type for linalg.tensor_3")


def __tensor_4_blas(T):
    if T.dtype == np.float64:
        return la.tensor_4_real(T)
    elif T.dtype == int:
        return la.tensor_4_real(np.array(T, dtype=np.float64))
    elif T.dtype == np.complex128:
        return la.tensor_4_complex(T)
    else:
        raise RuntimeError("Invalid dtype for linalg.tensor_4")


def __tensor_4_cuda(T):
    if T.dtype == np.float64:
        return cula.tensor_4_real(T)
    elif T.dtype == int:
        return cula.tensor_4_real(np.array(T, dtype=np.float64))
    elif T.dtype == np.complex128:
        return cula.tensor_4_complex(T)
    else:
        raise RuntimeError("Invalid dtype for linalg.tensor_4")


def tensor_4(T, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,4> type
    used by the C++ layer of pyTTN.

    :param T: The numpy tensor
    :type T: np.ndarray
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: tensor_4_real, tensor_4_complex
    """

    if backend == "blas":
        return __tensor_4_blas(T)
    elif __cuda_import and backend == "cuda":
        return __tensor_4_cuda(T)
    else:
        raise RuntimeError("Invalid backend type for linalg.tensor_4")


def tensor(T, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,D> type
    for D<=4 used by the C++ layer of pyTTN.

    :param M: The numpy tensor
    :type M: np.ndarray
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: str, optional

    :returns: A pybind11 wrapped linalg::tensor<T, D> object
    :rtype: pybind11 wrapped linalg::tensor<T,D> object
    """
    if T.ndim == 1:
        return vector(T, backend=backend)
    elif T.ndim == 2:
        return matrix(T, backend=backend)
    elif T.ndim == 3:
        return tensor_3(T, backend=backend)
    elif T.ndim == 4:
        return tensor_4(T, backend=backend)
    else:
        raise RuntimeError("Incompatible matrix dimensions")
