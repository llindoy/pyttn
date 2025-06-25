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

from typing import Union, Optional
from abc import ABCMeta
import numpy as np
from scipy.sparse import csr_matrix as spcsr

from .tensorExt import Vector, Matrix

# import the blas backend
import pyttn.ttnpp.linalg as la

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    _cuda_import = True
except ImportError:
    cula = None
    _cuda_import = False


def __is_csr_la(Op):
    is_bla_csr = isinstance(Op, (la.csr_matrix_real, la.csr_matrix_complex))
    if _cuda_import:
        is_csr = is_bla_csr or isinstance(
            Op, (cula.csr_matrix_real, cula.csr_matrix_complex)
        )
    else:
        is_csr = is_bla_csr
    return is_csr


def _csr_matrix(mod, *args, dtype=None, **kwargs):
    if args:
        if len(args) == 1:
            # if we are working with a csr matrix - then copy construct it
            if __is_csr_la(args[0]):
                if dtype is None:
                    if args[0].complex_dtype():
                        dtype = np.complex128
                    else:
                        dtype = np.float64

                if dtype == np.float64 or dtype is float:
                    return mod.csr_matrix_real(args[0])
                elif dtype == np.complex128 or dtype is complex:
                    return mod.csr_matrix_complex(args[0])
                else:
                    raise RuntimeError("Invalid dtype for csr matrix")
            # if we are working with a scipy csr matrix we build the csr matrix object using its data, indices and indptr arrays
            elif isinstance(args[0], spcsr):
                if dtype is None:
                    dtype = args[0].dtype

                if dtype is int:
                    dtype = np.float64

                if dtype == np.float64 or dtype is float:
                    return mod.csr_matrix_real(
                        np.array(args[0].data, dtype=dtype),
                        args[0].indices,
                        args[0].indptr,
                        ncols=args[0].shape[1],
                    )
                elif dtype == np.complex128 or dtype is complex:
                    return mod.csr_matrix_complex(
                        np.array(args[0].data, dtype=dtype),
                        args[0].indices,
                        args[0].indptr,
                        ncols=args[0].shape[1],
                    )
                else:
                    raise RuntimeError("Invalid dtype for csr matrix")
            # if this is a list of tuples it is a coo format array and we can build it
            elif isinstance(args[0], list):
                if dtype is None:
                    dtype = type(args[0][0][0])

                if dtype is int:
                    dtype = np.float64

                inputs = []
                for t in args[0]:
                    if not isinstance(t, tuple):
                        raise RuntimeError(
                            "Invalid type for csr matrix coo constructor"
                        )
                    if not (len(t) == 3):
                        raise RuntimeError(
                            "Invalid type for csr matrix coo constructor"
                        )
                    inputs.append((t[0], t[1], dtype(t[2])))

                if dtype == np.float64 or dtype is float:
                    return mod.csr_matrix_real(inputs, **kwargs)
                elif dtype == np.complex128 or dtype is complex:
                    return mod.csr_matrix_complex(inputs, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype")
            else:
                raise RuntimeError("Invalid argument list option")
        # if there are three arguments we have data, indices, indptr and we can build
        elif len(args) == 3:
            if (
                isinstance(args[0], (list, np.ndarray))
                and isinstance(args[1], (list, np.ndarray))
                and isinstance(args[2], (list, np.ndarray))
            ):
                if dtype is None:
                    if isinstance(args[0], np.ndarray):
                        dtype = args[0].dtype
                    elif len(args[0]) > 0:
                        dtype = type(args[0][0])
                    else:
                        raise RuntimeError(
                            "Failed to extract dtype from variable array."
                        )
                if dtype is int:
                    dtype = np.float64

                if dtype == np.float64 or dtype is float:
                    return mod.csr_matrix_real(
                        np.array(args[0], dtype=dtype),
                        np.array(args[1], dtype=int),
                        np.array(args[2], dtype=int),
                        **kwargs,
                    )
                elif dtype == np.complex128 or dtype is complex:
                    return mod.csr_matrix_complex(
                        np.array(args[0], dtype=dtype),
                        np.array(args[1], dtype=int),
                        np.array(args[2], dtype=int),
                        **kwargs,
                    )
                else:
                    raise RuntimeError("Invalid dtype")
            else:
                raise RuntimeError("Failed to construct csr matrix")
        else:
            raise RuntimeError("Invalid arguments")
    else:
        raise RuntimeError("Default constructor not supported for csr_matrix")


class CSR_Matrix(metaclass=ABCMeta):
    def __new__(
        cls,
        *args,
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ) -> 'CSR_Matrix':
        """
        Construct a C++ linalg::csr_matrix<T> type that can be used by the C++ layer of pyTTN.

        :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

            - csr matrix (CSR_Matrix) - Copy construct csr matrix object
            - csr_matrix (scipy.sparse.csr_matrix) - construct csr matrix from scipy csr matrix
            - values (list[dtype]), indices (list[int]), rowptr (list[int]), ncols (int, optional) - Construct TTN object from slice of multiset ttn
            - coo array list[tuple(int, int, dtype)], nrows (int, optional), ncols (int, optional)

        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::csr_matrix<T> object
        :rtype: CSR_Matrix
        """

        if backend == "blas":
            return _csr_matrix(la, *args, dtype=dtype, **kwargs)
        elif _cuda_import and backend == "cuda":
            return _csr_matrix(cula, *args, dtype=dtype, **kwargs)
        else:
            raise RuntimeError("Invalid backend type for linalg.csr_matrix")

    
    def complex_dtype(self) -> bool:
        """Returns whether or not the CSR_Matrix is storing a complex valued dtype

        :return: whether or not the CSR_Matrix is storing a complex valued dtype
        :rtype: bool
        """
        pass

    
    def __matmul__(self, b: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        """CSR matrix - dense matrix multiplications

        :param b: The dense matrix that the CSR_Matrix acts on
        :type b: Matrix
        :return: The result of the CSR_Matrix-Matrix multiplication
        :rtype: Matrix
        """
        pass

    def __str__(self) -> str:
        """Return the string representation of the CSR_Matrix object

        :return: The string representation of the CSR_Matrix
        :rtype: str
        """
        pass

    def backend(self) -> str:
        """Returns the backend type of the CSR_Matrix

        :return: The backend type of the object
        :rtype: str
        """
        pass


class Diagonal_Matrix(metaclass=ABCMeta):
    def __new__(
        cls,
        buffer: np.ndarray,
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ):
        """
        Construct a C++ linalg::csr_matrix<T> type that can be used by the C++ layer of pyTTN.

        :param buffer: The numpy array defining the diagonals of the matrix
        :type buffer: np.ndarray
        :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
        :type dtype: {None, np.float64, np.complex128}, optional
        :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
        :type backend: {"blas", "cuda"}, optional

        :returns: A pybind11 wrapped linalg::diagonal_matrix<T> object
        :rtype: Diagonal_Matrix
        """
        raise RuntimeError("Diagonal Matrix interface not yet implemented.")


CSR_Matrix.register(la.csr_matrix_real)
CSR_Matrix.register(la.csr_matrix_complex)

Diagonal_Matrix.register(la.diagonal_matrix_real)
Diagonal_Matrix.register(la.diagonal_matrix_complex)


if _cuda_import:
    CSR_Matrix.register(cula.csr_matrix_real)
    CSR_Matrix.register(cula.csr_matrix_complex)

    Diagonal_Matrix.register(cula.diagonal_matrix_real)
    Diagonal_Matrix.register(cula.diagonal_matrix_complex)

Sparse_Matrix = CSR_Matrix

def csr_matrix(
    *args,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
    backend: str = "blas",
    **kwargs,
) -> CSR_Matrix:
    """
    A function for converting from a numpy array to a C++ linalg::csr_matrix<T> type
     used by the C++ layer of pyTTN.

     :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

         - csr matrix (csr_dtype) - Copy construct csr matrix object
         - csr_matrix (scipy.sparse.csr_matrix) - construct csr matrix from scipy csr matrix
         - values (list[dtype]), indices (list[int]), rowptr (list[int]), ncols (int, optional) - Construct TTN object from slice of multiset ttn
         - coo array list[tuple(int, int, dtype)], nrows (int, optional), ncols (int, optional)

     :param dtype: The dtype to use for the site operator.  If this is None this function attempts to infer the dtype from v (Default: None)
     :type dtype: {None, np.float64, np.complex128}, optional
     :param backend: The backend to use for calculation. Either blas or cuda. (Default: "blas")
     :type backend: {"blas", "cuda"}, optional

     :returns: A pybind11 wrapped linalg::csr_matrix<T> object
     :rtype: CSR_Matrix
    """
    return CSR_Matrix(*args, dtype=dtype, backend=backend, **kwargs)
