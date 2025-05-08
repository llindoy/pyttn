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

# import the blas backend
import pyttn.ttnpp.linalg as la

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    __cuda_import = True
except ImportError:
    __cuda_import = False

import numpy as np

def available_backends():
    if __cuda_import:
            return ["blas", "cuda"]
    else:
        return ["blas"]

def __is_vector(O):
    is_bla_vector = isinstance(O, (la.vector_real, la.vector_complex))
    if __cuda_import:
        is_vector = is_bla_vector or isinstance(O, (cula.vector_real, cula.vector_complex))
    else:
        is_vector = is_bla_vector
    return is_vector


def __is_matrix(O):
    is_bla_matrix = isinstance(O, (la.matrix_real, la.matrix_complex))
    if __cuda_import:
        is_matrix = is_bla_matrix or isinstance(
            O, (cula.matrix_real, cula.matrix_complex)
        )
    else:
        is_matrix = is_bla_matrix
    return is_matrix


def __is_tensor_3(O):
    is_bla_tensor_3 = isinstance(O, (la.tensor_3_real, la.tensor_3_complex))
    if __cuda_import:
        is_tensor_3 = is_bla_tensor_3 or isinstance(O, (cula.tensor_3_real, cula.tensor_3_complex))
    else:
        is_tensor_3 = is_bla_tensor_3
    return is_tensor_3


def __is_tensor_4(O):
    is_bla_tensor_4 = isinstance(O, (la.tensor_4_real, la.tensor_4_complex))
    if __cuda_import:
        is_tensor_4 = is_bla_tensor_4 or isinstance(O, (cula.tensor_4_real, cula.tensor_4_complex))
    else:
        is_tensor_4 = is_bla_tensor_4
    return is_tensor_4


def __is_tensor(O):
    return __is_vector(O) or __is_matrix(O) or __is_tensor_3(O) or __is_tensor_4(O)


def __build_vector(O, mod, dtype):
    if dtype == np.float64:
        return mod.vector_real(O)
    elif dtype == np.complex128:
        return mod.vector_complex(O)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_matrix(O, mod, dtype):
    if dtype == np.float64:
        return mod.matrix_real(O)
    elif dtype == np.complex128:
        return mod.matrix_complex(O)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_tensor_3(O, mod, dtype):
    if dtype == np.float64:
        return mod.tensor_3_real(O)
    elif dtype == np.complex128:
        return mod.tensor_3_complex(O)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_tensor_4(O, mod, dtype):
    if dtype == np.float64:
        return mod.tensor_4_real(O)
    elif dtype == np.complex128:
        return mod.tensor_4_complex(O)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __setup_numpy(v):
    if v.dtype == int:
        return np.array(v, dtype=np.float64)
    else:
        return v


def __get_dtype_numpy(v, dtype=None):
    if dtype is None:
        return v.dtype
    return dtype


def __get_dtype_la(v, dtype=None):
    if dtype is None:
        if v.complex_dtype:
            return np.complex128
        else:
            return np.float64
    return dtype


def vector(v, dtype=None, backend="blas"):
    r"""
    A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
    used by the C++ layer of pyTTN.

    :param M: The numpy array
    :type M: np.ndarray
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 1> object
    :rtype: vector_real, vector_complex
    """

    if backend == "blas":
        if isinstance(v, np.ndarray):
            v = __setup_numpy(v)
            dtype = __get_dtype_numpy(v, dtype)
            return __build_vector(v, la, dtype)
        elif __is_vector(v):
            dtype = __get_dtype_la(v, dtype)
            return __build_vector(v, la, dtype)
        else:
            raise RuntimeError("Invalid type for vector")
    elif __cuda_import and backend == "cuda":
        if isinstance(v, np.ndarray):
            v = __setup_numpy(v)
            dtype = __get_dtype_numpy(v, dtype)
            return __build_vector(v, cula, dtype)
        elif __is_vector(v):
            dtype = __get_dtype_la(v, dtype)
            return __build_vector(v, cula, dtype)
        else:
            raise RuntimeError("Invalid type for vector")

    else:
        raise RuntimeError("Invalid backend type for linalg.vector")


def matrix(M, dtype=None, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,2> type
    used by the C++ layer of pyTTN.

    :param M: The numpy matrix
    :type M: np.ndarray
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: matrix_real, matrix_complex
    """

    if backend == "blas":
        if isinstance(M, np.ndarray):
            M = __setup_numpy(M)
            dtype = __get_dtype_numpy(M, dtype)
            return __build_matrix(M, la, dtype)
        elif __is_matrix(M):
            dtype = __get_dtype_la(M, dtype)
            return __build_matrix(M, la, dtype)
        else:
            raise RuntimeError("Invalid type for matrix")
    elif __cuda_import and backend == "cuda":
        if isinstance(M, np.ndarray):
            M = __setup_numpy(M)
            dtype = __get_dtype_numpy(M, dtype)
            return __build_matrix(M, cula, dtype)
        elif __is_matrix(M):
            dtype = __get_dtype_la(M, dtype)
            return __build_matrix(M, cula, dtype)
        else:
            raise RuntimeError("Invalid type for matrix")

    else:
        raise RuntimeError("Invalid backend type for linalg.matrix")


def tensor_3(T, dtype=None, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,3> type
    used by the C++ layer of pyTTN.

    :param T: The numpy tensor
    :type T: np.ndarray
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: tensor_3_real, tensor_3_complex
    """

    if backend == "blas":
        if isinstance(T, np.ndarray):
            T = __setup_numpy(T)
            dtype = __get_dtype_numpy(T, dtype)
            return __build_tensor_3(T, la, dtype)
        elif __is_tensor_3(T):
            dtype = __get_dtype_la(T, dtype)
            return __build_tensor_3(T, la, dtype)
        else:
            raise RuntimeError("Invalid type for tensor_3")
    elif __cuda_import and backend == "cuda":
        if isinstance(T, np.ndarray):
            T = __setup_numpy(T)
            dtype = __get_dtype_numpy(T, dtype)
            return __build_tensor_3(T, cula, dtype)
        elif __is_tensor_3(T):
            dtype = __get_dtype_la(T, dtype)
            return __build_tensor_3(T, cula, dtype)
        else:
            raise RuntimeError("Invalid type for tensor_3")

    else:
        raise RuntimeError("Invalid backend type for linalg.tensor_3")


def tensor_4(T, dtype=None, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,4> type
    used by the C++ layer of pyTTN.

    :param T: The numpy tensor
    :type T: np.ndarray
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: tensor_4_real, tensor_4_complex
    """

    if backend == "blas":
        if isinstance(T, np.ndarray):
            T = __setup_numpy(T)
            dtype = __get_dtype_numpy(T, dtype)
            return __build_tensor_4(T, la, dtype)
        elif __is_tensor_4(T):
            dtype = __get_dtype_la(T, dtype)
            return __build_tensor_4(T, la, dtype)
        else:
            raise RuntimeError("Invalid type for tensor_4")
    elif __cuda_import and backend == "cuda":
        if isinstance(T, np.ndarray):
            T = __setup_numpy(T)
            dtype = __get_dtype_numpy(T, dtype)
            return __build_tensor_4(T, cula, dtype)
        elif __is_tensor_4(T):
            dtype = __get_dtype_la(T, dtype)
            return __build_tensor_4(T, cula, dtype)
        else:
            raise RuntimeError("Invalid type for tensor_4")

    else:
        raise RuntimeError("Invalid backend type for linalg.tensor_4")


def tensor(T, dtype=None, backend="blas"):
    r"""
    A function for converting from a numpy array to a C++ linalg::tensor<T,D> type
    for D<=4 used by the C++ layer of pyTTN.

    :param T: The numpy tensor
    :type T: np.ndarray
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: str, optional

    :returns: A pybind11 wrapped linalg::tensor<T, D> object
    :rtype: pybind11 wrapped linalg::tensor<T,D> object
    """
    if isinstance(T, np.ndarray):
        if T.ndim == 1:
            return vector(T, dtype=dtype, backend=backend)
        elif T.ndim == 2:
            return matrix(T, dtype=dtype, backend=backend)
        elif T.ndim == 3:
            return tensor_3(T, dtype=dtype, backend=backend)
        elif T.ndim == 4:
            return tensor_4(T, dtype=dtype, backend=backend)
    elif __is_tensor(T):
        if T.ndim() == 1:
            return vector(T, dtype=dtype, backend=backend)
        elif T.ndim() == 2:
            return matrix(T, dtype=dtype, backend=backend)
        elif T.ndim() == 3:
            return tensor_3(T, dtype=dtype, backend=backend)
        elif T.ndim() == 4:
            return tensor_4(T, dtype=dtype, backend=backend)
    else:
        raise RuntimeError("Incompatible matrix dimensions")
