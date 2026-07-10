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
import numpy as np
from typing import Union
from pyttn.linalg.tensorExt import Matrix, Tensor4

from pyttn.ttnpp import Op_complex

try:
    from pyttn.ttnpp import Op_real

    _real_ttn_import = True

except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import Op_complex as Op_complex_cuda

    _cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import Op_real as Op_real_cuda

except ImportError:
    _cuda_import = False


def _Op_blas(*args, dtype=np.complex128):
    if not _real_ttn_import:
        if dtype is np.complex128 or dtype is complex:
            return Op_complex(*args)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a Op.")
    else:
        if dtype is np.complex128 or dtype is complex:
            return Op_complex(*args)
        elif dtype is np.float64 or dtype is float:
            return Op_real(*args)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a Op.")


def _Op_cuda(*args, dtype=np.complex128):
    if not _real_ttn_import:
        if dtype is np.complex128 or dtype is complex:
            return Op_complex_cuda(*args)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a Op.")
    else:
        if dtype is np.complex128 or dtype is complex:
            return Op_complex_cuda(*args)
        elif dtype is np.float64 or dtype is float:
            return Op_real_cuda(*args)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a Op.")
        
class Op(metaclass=ABCMeta):
    """A class for handling matrix valued operators that act on an arbitrary set of modes"""

    def __new__(cls, *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128, backend : str="blas") -> "Op":
        """A function for constructing a new instance of the Op type based on dtype and backend.

        :param `*args`: Variable length list of arguments. This function can handle the following lists of arguments

            - Default construct Op object
            - op (:class:`Op`) - Copy construct Op object   
            - mat (Union[:class:`linalg.Matrix`, np.ndarray]), inds (list[int]), dims (list[int]) - Construct a operator object with matrix representation mat acting on modes inds with dimensions dims

        :param dtype: The dtype used for the Op object, defaults to np.complex128
        :type dtype: Union[float, complex, np.float64, np.complex128]
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: str, optional

        :returns: A pybind11 wrapped Op<T> object
        :rtype: Op
        """

        if backend == "blas":
            return _Op_blas(*args, dtype=dtype)
        elif backend == "cuda" and _cuda_import:
            return _Op_cuda(*args, dtype=dtype)
        else:
            raise RuntimeError("Invalid backend for Op class")
                        
    @abstractmethod
    def assign(op : "Op") -> None:
        """Assign the value of this operator object given another object

        :param op: The operator object to copy to this object
        :type op: Op
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        "Clear and deallocate all internal buffers of the Op object"
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        :return: A string represent of the Op object
        :rtype: str
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the object stores a complex valued dtype

        :return: dtype
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the NumPy dtype of the underlying operator representation.

        This corresponds to the scalar type used internally, e.g.
        ``np.float64`` or ``np.complex128``.

        :return: The dtype of the operator
        :rtype: numpy.dtype
        """
        pass
    
    @abstractmethod
    def backend(self) -> str:
        """Returns a string labelling the backend of the Op object

        :return: backend label
        :rtype: str
        """
        pass

    @property
    @abstractmethod
    def operator(self) -> Matrix:
        """The matrix stored in the Op object

        :return: The matrix stored in the Op object
        :rtype: Matrix
        """
        pass

    @property
    @abstractmethod
    def indices(self) -> list[int]:
        """The list of indices that the Op object acts on

        :return: The list of indices that the Op object acts on
        :rtype: list[int]
        """
        pass

    @property
    @abstractmethod
    def dims(self) -> list[int]:
        """The dimensionality of the indices that the Op object acts on

        :return: A list containing the dimensionality of the indices that the Op object acts on
        :rtype: list[int]
        """
        pass


    @abstractmethod
    def set_operator(self, mat: Union[Matrix, np.ndarray]) -> None:
        """Set the value of the operator from a Matrix or numpy array.  This function checks if the size is correct and throws an exception if not.

        :param mat: The new value of the matrix
        :type mat: Union[Matrix, np.ndarray]
        """

    @abstractmethod
    def nmodes(self) -> int:
        """Returns the number of modes that the operator acts on

        :return: The number of modes that the operator acts on
        :rtype: int
        """
        pass

    @abstractmethod
    def ndim(self) -> int:
        """Returns the number of modes that the operator acts on

        :return: The number of modes that the operator acts on
        :rtype: int
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Returns the dimensionality of the space the object acts on

        :return: The dimensionality of the space the object acts on
        :rtype: int
        """
        pass

    @abstractmethod
    def as_mpo(self, nbmax : int = - 1, tol : float = -1.0) -> list[Tensor4]:
        """Convert the operator object into an MPO object acting on the indices the node acts on

        :param nbmax: The maximum bond dimension to use in the svd decomposition.  If the value is negative this parameter is ignored, defaults to -1
        :type nbmax: int, optional
        :param tol: The truncation tolerance for the svd decomposition.  If the value is negative this parameter is ignored, defaults to -1.0
        :type tol: float, optional
        :return: A list of tensors containing the MPO decomposition of the tensor
        :rtype: list[Tensor4]
        """
        pass
 