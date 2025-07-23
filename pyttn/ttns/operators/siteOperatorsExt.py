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

from pyttn.linalg import Matrix, Vector
from pyttn.ttnpp import system_modes
from pyttn.ttnpp.ops import site_operator_complex

from ..sop.sSOPExt import sOP
from .opsExt import siteOp

try:
    from pyttn.ttnpp.ops import site_operator_real

    _real_ttn_import = True

except ImportError:
    _real_ttn_import = False


# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda.ops import site_operator_complex as site_operator_complex_cuda

    _cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda.ops import site_operator_real as site_operator_real_cuda

except ImportError:
    _cuda_import = False


def _site_operator_blas(*args, mode=None, optype=None, dtype=np.complex128, **kwargs):
    ret = None
    if optype is None:
        if args and len(args) == 1:
            if args[0].complex_dtype() or not _real_ttn_import:
                ret = site_operator_complex(args[0])
            else:
                ret = site_operator_real(args[0])
        elif (args and len(args) <= 3) or (not args):
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                ret = site_operator_complex(*args, **kwargs)
            else:
                ret = site_operator_real(*args, **kwargs)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object invalid arguments."
            )
    else:
        try:
            M = siteOp(*args, type=optype, dtype=dtype, backend="blas", **kwargs)
            if M.complex_dtype() or not _real_ttn_import:
                ret = site_operator_complex(M)
            else:
                ret = site_operator_real(M)
        except RuntimeError:
            message = "Failed to construct site_operator object.  optype not recognized."
            raise RuntimeError(message) from None
    if mode is not None:
        ret.mode = mode
    return ret


def _site_operator_cuda(*args, mode=None, optype=None, dtype=np.complex128, **kwargs):
    ret = None
    if optype is None:
        if args and len(args) == 1:
            if args[0].complex_dtype() or not _real_ttn_import:
                ret = site_operator_complex_cuda(args[0])
            else:
                ret = site_operator_real_cuda(args[0])
        elif args and len(args) <= 3:
            if dtype == np.complex128 or dtype is complex or not _real_ttn_import:
                ret = site_operator_complex_cuda(*args, **kwargs)
            else:
                ret = site_operator_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object invalid arguments."
            )
    else:
        try:
            M = siteOp(*args, type=optype, dtype=dtype, backend="cuda", **kwargs)
            if M.complex_dtype() or not _real_ttn_import:
                ret = site_operator_complex_cuda(M)
            else:
                ret = site_operator_real_cuda(M)
        except RuntimeError:
            message = "Failed to construct site_operator object.  optype not recognized."
            raise RuntimeError(message) from None
    if mode is not None:
        ret.mode = mode
    return ret

class site_operator(metaclass=ABCMeta):
    """A class for handling a single site operator acting on"""

    def __new__(
        cls,
        *args,
        mode: Optional[int] = None,
        optype: Optional[str] = None,
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas",
        **kwargs,
    ) -> "site_operator":
        """Factory function for constructing a one site operator.

        :param *args: Variable length list of arguments. There are several valid options for the *args parameters.  If the optype variable is None the allowed options are

            - Default construct the site operator object
            - site_op (:class:`ops.siteOp`) - Construct a new site_operator object from a siteOp object
            - site_op (:class:`site_operator`) - Construct a new site_operator object from the existing object
            - site_op (:class:`site_operator`) - Construct a new site_operator object from the existing object
            - op (:class:`sOP`), sysinf (:class:`system_modes`) - Construct a new site_operator from the string operator and system information
            - op (:class:`sOP`), sysinf (:class:`system_modes`), opdict (operator_dictionary_real or operator_dictionary_complex) -  Construct a new site_operator from the string operator, system information and used defined operator dictionary.

            Otherwise, if the optype variable has been set then the valid arguments are determined by the specified optype see opsExt.py for details.

        :param mode: The mode the site operator is acting on. (Default: None)
        :type mode: int or None, optional
        :param optype: The type of the operator to be constructed. (Default: None)
        :type optype: {'identity', 'matrix', 'sparse_matrix', 'diagonal_matrix'} or None, optional
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the product operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :param **kwargs: Additional keyword arguments. To construct the site_operator object.  Valid options:
            
            - use_sparse (bool)
        """
        if backend == "blas":
            return _site_operator_blas(
                *args, mode=mode, optype=optype, dtype=dtype, **kwargs
            )
        elif _cuda_import and backend == "cuda":
            return _site_operator_cuda(
                *args, mode=mode, optype=optype, dtype=dtype, **kwargs
            )
        else:
            raise RuntimeError("Invalid backend type for site_operator")

    @abstractmethod
    def initialise(self, op: sOP, sysinf: system_modes, *args, use_sparse: bool = True):
        """Initialise the site_operator object given a sOP and system_modes information

        :param op: The string representation of the mode opertor
        :type op: sOP
        :param sysinf: The information about the system degrees of freedom
        :type sysinf: system_modes
        :param *args: A variable length array of arguments: Valid options are:

            - If no *args are provided construct using standard dictionaries
            - opdict (:class:`operator_dictionary`): Construct site_operator using a user created operator dictionary.
       
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the site_operator is storing a complex valued dtype

        :return: whether or not the site_operator is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear and deallocate all internal buffers of the site_operator"""
        pass

    @abstractmethod
    def transpose(self) -> "site_operator":
        """Returns the transpose of the operator

        :return: The transpose of the operator
        :rtype: site_operator
        """
        pass

    @abstractmethod
    def todense(self, *args) -> Matrix:
        """ Return the dens Matrix representation of this site operator

        :param *args: A variable length list of arguments. 
            - Either empty
            - Or containing a list[int] defining the dimension of each mode the operator acts on.
        """
        pass

    @abstractmethod
    def assign(self, o: Union["site_operator", siteOp]):
        """Assign the value of the site operator from another or a siteOp

        :param o: The site operator to copy into this one
        :type o: Union[site_operator, siteOp]
        """
        pass

    @abstractmethod
    def bind(self, v : siteOp):
        """Assign the value stored in the site operator to a siteOp

        :param v: The new siteOp
        :type v: siteOp
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the site_operator object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the site_operator object"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the local Hilbert space dimension associated with this site operator

        :return: Local Hilbert space dimension
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


    @property
    @abstractmethod
    def mode(self) -> int:
        """The mode the system acts on"""
        pass

    @abstractmethod
    def resize(self, size: int):
        """Resize the site_operator object so that it can describe a system with size modes

        :param size: The number of modes
        :type size: int
        """
        pass

    @abstractmethod
    def apply(self, a: Union[Vector, Matrix], b: Union[Vector, Matrix]):
        """Compute the action of the operator on the object a and store the result in b
            op*a = b
        :param a: The object the operator should be applied on
        :type a: Union[Vector, Matrix]
        :param b: The result of the action of the operator on a (op@a)
        :type b: Union[Vector, Matrix]
        """

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the site_operator object

        :return: The string representation of the site_operator
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the site_operator

        :return: The backend type of the object
        :rtype: str
        """
        pass

site_operator.register(site_operator_complex)
if _real_ttn_import:
    site_operator.register(site_operator_real)

if _cuda_import:
    site_operator.register(site_operator_complex_cuda)
    if _real_ttn_import:
        site_operator.register(site_operator_real_cuda)

site_operator_type = site_operator
