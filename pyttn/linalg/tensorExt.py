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
from typing import Union, Optional, TypeAlias

# import the blas backend
import pyttn.ttnpp.linalg as la


# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    Vector: TypeAlias = (
        la.vector_real | la.vector_complex | cula.vector_real | cula.vector_complex
    )
    Matrix: TypeAlias = (
        la.matrix_real | la.matrix_complex | cula.matrix_real | cula.matrix_complex
    )
    Tensor3: TypeAlias = (
        la.tensor_3_real
        | la.tensor_3_complex
        | cula.tensor_3_real
        | cula.tensor_3_complex
    )
    Tensor4: TypeAlias = (
        la.tensor_4_real
        | la.tensor_4_complex
        | cula.tensor_4_real
        | cula.tensor_4_complex
    )

    __cuda_import = True
except ImportError:
    Vector: TypeAlias = la.vector_real | la.vector_complex
    Matrix: TypeAlias = la.matrix_real | la.matrix_complex
    Tensor3: TypeAlias = la.tensor_3_real | la.tensor_3_complex
    Tensor4: TypeAlias = la.tensor_4_real | la.tensor_4_complex

    __cuda_import = False


def available_backends():
    if __cuda_import:
        return ["blas", "cuda"]
    else:
        return ["blas"]


def __is_vector(Op):
    is_bla_vector = isinstance(Op, (la.vector_real, la.vector_complex))
    if __cuda_import:
        is_vector = is_bla_vector or isinstance(
            Op, (cula.vector_real, cula.vector_complex)
        )
    else:
        is_vector = is_bla_vector
    return is_vector


def __is_matrix(Op):
    is_bla_matrix = isinstance(Op, (la.matrix_real, la.matrix_complex))
    if __cuda_import:
        is_matrix = is_bla_matrix or isinstance(
            Op, (cula.matrix_real, cula.matrix_complex)
        )
    else:
        is_matrix = is_bla_matrix
    return is_matrix


def __is_tensor_3(Op):
    is_bla_tensor_3 = isinstance(Op, (la.tensor_3_real, la.tensor_3_complex))
    if __cuda_import:
        is_tensor_3 = is_bla_tensor_3 or isinstance(
            Op, (cula.tensor_3_real, cula.tensor_3_complex)
        )
    else:
        is_tensor_3 = is_bla_tensor_3
    return is_tensor_3


def __is_tensor_4(Op):
    is_bla_tensor_4 = isinstance(Op, (la.tensor_4_real, la.tensor_4_complex))
    if __cuda_import:
        is_tensor_4 = is_bla_tensor_4 or isinstance(
            Op, (cula.tensor_4_real, cula.tensor_4_complex)
        )
    else:
        is_tensor_4 = is_bla_tensor_4
    return is_tensor_4


def __is_tensor(Op):
    return __is_vector(Op) or __is_matrix(Op) or __is_tensor_3(Op) or __is_tensor_4(Op)


def __build_vector(Op, mod, dtype):
    if dtype == np.float64 or dtype is float:
        return mod.vector_real(Op)
    elif dtype == np.complex128 or dtype is complex:
        return mod.vector_complex(Op)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_matrix(Op, mod, dtype):
    if dtype == np.float64 or dtype is float:
        return mod.matrix_real(Op)
    elif dtype == np.complex128 or dtype is complex:
        return mod.matrix_complex(Op)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_tensor_3(Op, mod, dtype):
    if dtype == np.float64 or dtype is float:
        return mod.tensor_3_real(Op)
    elif dtype == np.complex128 or dtype is complex:
        return mod.tensor_3_complex(Op)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj")


def __build_tensor_4(Op, mod, dtype):
    if dtype == np.float64 or dtype is float:
        return mod.tensor_4_real(Op)
    elif dtype == np.complex128 or dtype is complex:
        return mod.tensor_4_complex(Op)
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


def vector(
    v: np.ndarray | Vector,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Vector:
    """
    A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
    used by the C++ layer of pyTTN.

    :param M: The input vector type
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 1> object
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


def matrix(
    M: np.ndarray | Matrix,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Matrix:
    """
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


def tensor_3(
    T: np.ndarray | Tensor3,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
):
    """
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


def tensor_4(
    T: np.ndarray | Tensor4,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
):
    """
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


def tensor(
    T: np.ndarray | Vector | Matrix | Tensor3 | Tensor4,
    dtype=None,
    backend: str = "blas",
):
    """
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
