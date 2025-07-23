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
from typing import Optional, Union

import numpy as np

# import the blas backend
import pyttn.ttnpp.linalg as la
from pyttn.ttnpp import convert_to_dense

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    _cuda_import = True
except ImportError:
    _cuda_import = False


def available_backends():
    if _cuda_import:
        return ["blas", "cuda"]
    else:
        return ["blas"]


def _is_vector(Op):
    is_bla_vector = isinstance(Op, (la.vector_real, la.vector_complex))
    if _cuda_import:
        is_vector = is_bla_vector or isinstance(
            Op, (cula.vector_real, cula.vector_complex)
        )
    else:
        is_vector = is_bla_vector
    return is_vector


def _is_matrix(Op):
    is_bla_matrix = isinstance(Op, (la.matrix_real, la.matrix_complex))
    if _cuda_import:
        is_matrix = is_bla_matrix or isinstance(
            Op, (cula.matrix_real, cula.matrix_complex)
        )
    else:
        is_matrix = is_bla_matrix
    return is_matrix


def _is_tensor_3(Op):
    is_bla_tensor_3 = isinstance(Op, (la.tensor_3_real, la.tensor_3_complex))
    if _cuda_import:
        is_tensor_3 = is_bla_tensor_3 or isinstance(
            Op, (cula.tensor_3_real, cula.tensor_3_complex)
        )
    else:
        is_tensor_3 = is_bla_tensor_3
    return is_tensor_3


def _is_tensor_4(Op):
    is_bla_tensor_4 = isinstance(Op, (la.tensor_4_real, la.tensor_4_complex))
    if _cuda_import:
        is_tensor_4 = is_bla_tensor_4 or isinstance(
            Op, (cula.tensor_4_real, cula.tensor_4_complex)
        )
    else:
        is_tensor_4 = is_bla_tensor_4
    return is_tensor_4


def _is_tensor(Op):
    return _is_vector(Op) or _is_matrix(Op) or _is_tensor_3(Op) or _is_tensor_4(Op)


def _build_vector(mod, dtype, *args):
    if dtype == np.float64 or dtype is float:
        return mod.vector_real(*args)
    elif dtype == np.complex128 or dtype is complex:
        return mod.vector_complex(*args)
    elif dtype is None:
        return mod.vector_complex(*args)    
    else:
        raise RuntimeError("Invalid dtype for tensor build obj"+str(dtype))


def _build_matrix(mod, dtype, *args):
    if dtype == np.float64 or dtype is float:
        return mod.matrix_real(*args)
    elif dtype == np.complex128 or dtype is complex:
        return mod.matrix_complex(*args)
    elif dtype is None:
        return mod.matrix_complex(*args)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj"+str(dtype))


def _build_tensor_3(mod, dtype, *args):
    if dtype == np.float64 or dtype is float:
        return mod.tensor_3_real(*args)
    elif dtype == np.complex128 or dtype is complex:
        return mod.tensor_3_complex(*args)
    elif dtype is None:
        return mod.tensor_3_complex(*args)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj"+str(dtype))


def _build_tensor_4(mod, dtype, *args):
    if dtype == np.float64 or dtype is float:
        return mod.tensor_4_real(*args)
    elif dtype == np.complex128 or dtype is complex:
        return mod.tensor_4_complex(*args)
    elif dtype is None:
        return mod.tensor_4_complex(*args)
    else:
        raise RuntimeError("Invalid dtype for tensor build obj"+str(dtype))


def _setup_numpy(v):
    if v.dtype == int:
        return np.array(v, dtype=np.float64)
    else:
        return v


def _get_dtype_numpy(v, dtype=None):
    if dtype is None:
        return v.dtype
    return dtype


def _get_dtype_la(v, dtype=None):
    if dtype is None:
        if v.complex_dtype:
            return np.complex128
        else:
            return np.float64
    return dtype


class Tensor(metaclass=ABCMeta):
    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the Tensor is storing a complex valued dtype
        
        :return: whether or not the Tensor is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the Tensor object 

        :return: The string representation of the Tensor
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the Tensor

        :return: The backend type of the object
        :rtype: str
        """
        pass

    @abstractmethod
    def ndim(self) -> int:
        """Returns the dimensionality of the tensor

        :return: The dimensionality of the tensor
        :rtype: int
        """
        pass

    @abstractmethod
    def shape(self, i) -> int:
        """Returns the shape along dimension i

        :param i: The index to consider
        :type i: int
        :return: The shape along dimension i
        :rtype: int
        """
        pass

    @abstractmethod
    def transpose(self, inds: list[int]) -> 'Tensor':
        """Return the transposed tensor with new ordered inds[0], inds[1], ..., inds[ndim]

        :param inds: The new order of indices
        :type inds: List[int]
        :return: The transposed tensor
        :rtype: Tensor
        """
        pass
class Vector(Tensor):
    """Wrapper of the linalg::tensor<T, 1> object used internally by pyTTN to represent vectors"""

    def __new__(
        cls,
        *args, 
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
        backend: str = "blas",
    ) -> "Vector":
        """
        A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
        used by the C++ layer of pyTTN.

        :param *args: A variable list for specifying the Matrix object.  Valid options are

            - Default construct Vector
            - M (Union[np.ndarray,:class:`Vector`]) - Construct a vector from a numpy array or Vector
        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::tensor<T, 1> object
        :rtype: Vector
        """
        if len(args) == 0:
            if backend == "blas":
                return _build_vector(la, dtype)
            elif _cuda_import and backend == "cuda":
                return _build_vector(cula, dtype)
            else:
                raise RuntimeError("Invalid backend type for linalg.vector")
        elif(len(args) == 1):
            v = args[0]
            if backend == "blas":
                if isinstance(v, np.ndarray):
                    v = _setup_numpy(v)
                    dtype = _get_dtype_numpy(v, dtype)
                    return _build_vector(la, dtype, v)
                elif _is_vector(v):
                    dtype = _get_dtype_la(v, dtype)
                    return _build_vector(la, dtype, v)
                else:
                    raise RuntimeError("Invalid type for vector")
            elif _cuda_import and backend == "cuda":
                if isinstance(v, np.ndarray):
                    v = _setup_numpy(v)
                    dtype = _get_dtype_numpy(v, dtype)
                    return _build_vector(cula, dtype, v)
                elif _is_vector(v):
                    dtype = _get_dtype_la(v, dtype)
                    return _build_vector(cula, dtype, v)
                else:
                    raise RuntimeError("Invalid type for vector")

            else:
                raise RuntimeError("Invalid backend type for linalg.vector")
        else:
            raise RuntimeError("Invalid arguments for Vector")


class Matrix(Tensor):
    """Wrapper of the linalg::tensor<T, 2> object used internally by pyTTN to represent Matrices"""
    def __new__(
        cls,
        *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
        backend: str = "blas",
    ) -> "Matrix":
        """
        A function for constructing a Matrix object.  This function can accept as input a numpy array, Matrix object or an 
        OP_type object and specification of the system_modes object.

        :param *args: A variable list for specifying the Matrix object.  Valid options are

            - Default construct Matrix
            - M (Union[np.ndarray,:class:`Matrix`]) - Construct a matrix from a numpy array or Matrix
            - Op (:class:`OP_type`), sysinf (:class:`system_modes`) - Construct a matrix from a string operator type and the system_modes info
        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::tensor<T, 2> object
        :rtype: Matrix
        """
        if len(args) == 0:
            if backend == "blas":
                return _build_matrix(la, dtype)
            elif _cuda_import and backend == "cuda":
                return _build_matrix(cula, dtype)
        elif len(args) == 1:
            M = args[0]
            if backend == "blas":
                if isinstance(M, np.ndarray):
                    M = _setup_numpy(M)
                    dtype = _get_dtype_numpy(M, dtype)
                    return _build_matrix(la, dtype, M)
                elif _is_matrix(M):
                    dtype = _get_dtype_la(M, dtype)
                    return _build_matrix(la, dtype, M)
                else:
                    raise RuntimeError("Invalid type for Matrix")
            elif _cuda_import and backend == "cuda":
                if isinstance(M, np.ndarray):
                    M = _setup_numpy(M)
                    dtype = _get_dtype_numpy(M, dtype)
                    return _build_matrix(cula, dtype, M)
                elif _is_matrix(M):
                    dtype = _get_dtype_la(M, dtype)
                    return _build_matrix(cula, dtype, M)
                else:
                    raise RuntimeError("Invalid type for Matrix")

            else:
                raise RuntimeError("Invalid backend type for Matrix")
        elif len(args) == 2:
            return convert_to_dense(args[0], args[1])
        else:
            raise RuntimeError("Invalid arguments for Matrix constructor")
        
    def transpose(self, inds: Optional[list[int]]) -> 'Matrix':
        """Either return the matrix transpose or the transposed tensor with new ordered inds[0], inds[1]

        :param inds: The new order of indices
        :type inds: List[int], optional
        :return: The matrix transpose
        :rtype: Matrix
        """
        pass

    def set_subblock(self, v: np.ndarray) -> None:
        """Set a subblock of the Tensor to the value stored in the buffer.

        :param v: The buffer to set the
        :type v: np.ndarray
        """
        pass

class Tensor3(Tensor): 
    def __new__(
        cls,
        *args,
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
        backend: str = "blas",
    ) -> "Tensor3":
        """
        A function for converting from a numpy array to a C++ linalg::tensor<T,3> type
        used by the C++ layer of pyTTN.

        :param *args: A variable list for specifying the Tensor3 object.  Valid options are

            - Default construct Tensor4
            - M (Union[np.ndarray,:class:`Tensor3`]) - Construct a matrix from a numpy array or Tensor3        
        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::tensor<T, 3> object
        :rtype: Tensor3
        """
        if(len(args))==0:
            if backend == "blas":
                return _build_tensor_3(la, dtype)
            elif _cuda_import and backend == "cuda":
                return _build_tensor_3(cula, dtype)
        elif len(args) == 1:
            T = args[0]
            if backend == "blas":
                if isinstance(T, np.ndarray):
                    T = _setup_numpy(T)
                    dtype = _get_dtype_numpy(T, dtype)
                    return _build_tensor_3(la, dtype, T)
                elif _is_tensor_3(T):
                    dtype = _get_dtype_la(T, dtype)
                    return _build_tensor_3(la, dtype, T)
                else:
                    raise RuntimeError("Invalid type for Tensor3")
            elif _cuda_import and backend == "cuda":
                if isinstance(T, np.ndarray):
                    T = _setup_numpy(T)
                    dtype = _get_dtype_numpy(T, dtype)
                    return _build_tensor_3(cula, dtype, T)
                elif _is_tensor_3(T):
                    dtype = _get_dtype_la(T, dtype)
                    return _build_tensor_3(cula, dtype, T)
                else:
                    raise RuntimeError("Invalid type for Tensor3")

            else:
                raise RuntimeError("Invalid backend type for Tensor3")
        else:
            raise RuntimeError("Invalid arguments for Tensor3 constructor")

class Tensor4(Tensor):
    def __new__(
        cls,
        *args, 
        dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
        backend: str = "blas",
    ) -> "Tensor4":
        """
        A function for converting from a numpy array to a C++ linalg::tensor<T,4> type
        used by the C++ layer of pyTTN.

        :param *args: A variable list for specifying the Tensor4 object.  Valid options are

            - Default construct Tensor4
            - M (Union[np.ndarray,:class:`Tensor4`]) - Construct a matrix from a numpy array or Tensor4        
        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::tensor<T, 4> object
        :rtype: Tensor4
        """

        if(len(args))==0:
            if backend == "blas":
                return _build_tensor_4(la, dtype)
            elif _cuda_import and backend == "cuda":
                return _build_tensor_4(cula, dtype)
        elif len(args) == 1:
            T = args[0]
            if backend == "blas":
                if isinstance(T, np.ndarray):
                    T = _setup_numpy(T)
                    dtype = _get_dtype_numpy(T, dtype)
                    return _build_tensor_4(la, dtype, T)
                elif _is_tensor_4(T):
                    dtype = _get_dtype_la(T, dtype)
                    return _build_tensor_4(la, dtype, T)
                else:
                    raise RuntimeError("Invalid type for Tensor4")
            elif _cuda_import and backend == "cuda":
                if isinstance(T, np.ndarray):
                    T = _setup_numpy(T)
                    dtype = _get_dtype_numpy(T, dtype)
                    return _build_tensor_4(cula, dtype, T)
                elif _is_tensor_4(T):
                    dtype = _get_dtype_la(T, dtype)
                    return _build_tensor_4(cula, dtype, T)
                else:
                    raise RuntimeError("Invalid type for Tensor4")

            else:
                raise RuntimeError("Invalid backend type for Tensor4")
        else:
            raise RuntimeError("Invalid arguments for Tensor4 constructor")

Vector.register(la.vector_real)
Vector.register(la.vector_complex)

Matrix.register(la.matrix_real)
Matrix.register(la.matrix_complex)

Tensor3.register(la.tensor_3_real)
Tensor3.register(la.tensor_3_complex)

Tensor4.register(la.tensor_4_real)
Tensor4.register(la.tensor_4_complex)

if _cuda_import:
    Vector.register(cula.vector_real)
    Vector.register(cula.vector_complex)

    Matrix.register(cula.matrix_real)
    Matrix.register(cula.matrix_complex)

    Tensor3.register(cula.tensor_3_real)
    Tensor3.register(cula.tensor_3_complex)

    Tensor4.register(cula.tensor_4_real)
    Tensor4.register(cula.tensor_4_complex)


def vector(
    v: Union[np.ndarray, Vector],
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Vector:
    """
    A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
    used by the C++ layer of pyTTN.

    :param v: The input vector type
    :type v: Union[np.ndarray,Vector]
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 1> object
    :rtype: Vector
    """

    return Vector(v, dtype=dtype, backend=backend)


def matrix(
    M: Union[np.ndarray, Matrix],
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Matrix:
    """
    A function for converting from a numpy array to a C++ linalg::tensor<T,2> type
    used by the C++ layer of pyTTN.

    :param M: The input Matrix
    :type M: Union[np.ndarray,Matrix]
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional
    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: Matrix
    """
    return Matrix(M, dtype=dtype, backend=backend)


def tensor_3(
    T: Union[np.ndarray, Tensor3],
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Tensor3:
    """
    A function for converting from a numpy array to a C++ linalg::tensor<T,3> type
    used by the C++ layer of pyTTN.

    :param T: The tensor
    :type T: Union[np.ndarray,Tensor3]
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 3> object
    :rtype: Tensor3
    """

    return Tensor3(T, dtype=dtype, backend=backend)


def tensor_4(
    T: Union[np.ndarray, Tensor4],
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = None,
    backend: str = "blas",
) -> Tensor4:
    """
    A function for converting from a numpy array to a C++ linalg::tensor<T,4> type
    used by the C++ layer of pyTTN.

    :param T: The tensor
    :type T: Union[np.ndarray,Tensor4]
    :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
    :type dtype: {None, np.float64, np.complex128}, optional
    :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
    :type backend: {"blas", "cuda"}, optional

    :returns: A pybind11 wrapped linalg::tensor<T, 4> object
    :rtype: Tensor4
    """

    return Tensor4(T, dtype=dtype, backend=backend)


def tensor(
    T: Union[np.ndarray, Vector, Matrix, Tensor3, Tensor4],
    dtype=None,
    backend: str = "blas",
) -> Union[Vector, Matrix, Tensor3, Tensor4]:
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
    :rtype: Vector | Matrix | Tensor3 | Tensor4
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
    elif _is_tensor(T):
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
